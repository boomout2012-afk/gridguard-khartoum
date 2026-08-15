from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from gridguard.controllers import DecisionContext, FixedRotationController, PPOController
from gridguard.domain import ControllerName, Scenario, ShortageLevel
from gridguard.policy import BUNDLED_ARTIFACT_SHA256, load_ppo_policy
from gridguard.scenarios import SYSTEM_PEAK_MW, generate_scenarios
from gridguard.simulation import build_observation, dispatch_service_tiers, simulate


def test_observation_is_the_exact_32_value_causal_layout() -> None:
    demand = torch.arange(1, 36, dtype=torch.float32).reshape(1, 7, 5)
    supply = torch.tensor([350.0])
    cumulative_demand = torch.arange(10, 80, 10, dtype=torch.float32).reshape(1, 7)
    cumulative_served = cumulative_demand * 0.5
    previous_tier = torch.tensor([[0, 1, 2, 0, 1, 2, 0]], dtype=torch.long)

    observation = build_observation(
        hour=6,
        demand=demand,
        supply=supply,
        cumulative_demand=cumulative_demand,
        cumulative_served=cumulative_served,
        previous_tier=previous_tier,
    )

    assert observation.shape == (1, 32)
    torch.testing.assert_close(
        observation[0, :4],
        torch.tensor([1.0, 0.0, 350.0 / demand.sum(), 350.0 / SYSTEM_PEAK_MW]),
        rtol=0,
        atol=1e-6,
    )
    locality_weights = demand.new_tensor((1.00, 0.95, 0.85, 0.80, 0.70, 0.75, 0.90))
    locality_peaks = SYSTEM_PEAK_MW * locality_weights / locality_weights.sum()
    expected_localities = torch.stack(
        (
            demand.sum(dim=-1)[0] / locality_peaks,
            demand[..., :3].sum(dim=-1)[0] / demand.sum(dim=-1)[0],
            torch.full((7,), 0.5),
            previous_tier[0].to(torch.float32) / 2.0,
        ),
        dim=-1,
    ).flatten()
    torch.testing.assert_close(observation[0, 4:], expected_localities, rtol=0, atol=1e-6)


def test_dispatch_uses_one_global_proportional_scale_and_respects_tiers() -> None:
    demand = torch.arange(1, 36, dtype=torch.float32).reshape(1, 7, 5)
    supply = torch.tensor([100.0])
    tiers = torch.tensor([[0, 1, 2, 0, 1, 2, 0]], dtype=torch.long)

    requested, served = dispatch_service_tiers(demand=demand, supply=supply, tiers=tiers)

    assert torch.count_nonzero(requested[:, (0, 3, 6)]) == 0
    assert torch.count_nonzero(requested[:, (1, 4), 3:]) == 0
    torch.testing.assert_close(requested[:, (1, 4), :3], demand[:, (1, 4), :3])
    torch.testing.assert_close(requested[:, (2, 5)], demand[:, (2, 5)])
    expected_scale = supply / requested.sum(dim=(-1, -2))
    torch.testing.assert_close(served, requested * expected_scale[:, None, None])
    torch.testing.assert_close(served.sum(dim=(-1, -2)), supply, rtol=0, atol=1e-5)


def test_dispatch_does_not_invent_service_when_nothing_is_requested() -> None:
    demand = torch.ones((3, 7, 5), dtype=torch.float32)
    supply = torch.full((3,), 100.0)
    tiers = torch.zeros((3, 7), dtype=torch.long)

    requested, served = dispatch_service_tiers(demand=demand, supply=supply, tiers=tiers)

    torch.testing.assert_close(requested, torch.zeros_like(demand), rtol=0, atol=0)
    torch.testing.assert_close(served, torch.zeros_like(demand), rtol=0, atol=0)


@pytest.mark.parametrize("invalid_tier", [-1, 3])
def test_dispatch_rejects_tiers_outside_the_public_service_contract(
    invalid_tier: int,
) -> None:
    demand = torch.ones((1, 7, 5), dtype=torch.float32)
    supply = torch.ones(1, dtype=torch.float32)
    tiers = torch.zeros((1, 7), dtype=torch.long)
    tiers[0, 0] = invalid_tier

    with pytest.raises(ValueError, match="invalid service tier"):
        dispatch_service_tiers(demand=demand, supply=supply, tiers=tiers)


@dataclass(frozen=True)
class _InvalidShapeController:
    name: ControllerName = ControllerName.PPO

    def act(self, context: DecisionContext) -> torch.Tensor:
        return torch.zeros(
            (context.observation.shape[0], 6),
            dtype=torch.long,
            device=context.observation.device,
        )


@dataclass(frozen=True)
class _InvalidTierController:
    name: ControllerName = ControllerName.PPO

    def act(self, context: DecisionContext) -> torch.Tensor:
        return torch.full(
            (context.observation.shape[0], 7),
            3,
            dtype=torch.long,
            device=context.observation.device,
        )


def test_simulation_rejects_controller_actions_with_the_wrong_shape() -> None:
    scenario = generate_scenarios(
        2,
        seed=3,
        shortage_level=ShortageLevel.MILD,
        device="cpu",
        demand_noise=0.0,
        supply_noise=0.0,
    )

    with pytest.raises(ValueError, match=r"controller action must have shape \[batch, 7\]"):
        simulate(scenario, _InvalidShapeController())


def test_simulation_rejects_controller_actions_with_invalid_tiers_immediately() -> None:
    scenario = generate_scenarios(
        2,
        seed=3,
        shortage_level=ShortageLevel.MILD,
        device="cpu",
        demand_noise=0.0,
        supply_noise=0.0,
    )

    with pytest.raises(ValueError, match="invalid service tier"):
        simulate(scenario, _InvalidTierController())


def test_simulation_has_no_per_hour_tensor_scalar_transfers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = generate_scenarios(
        2,
        seed=3,
        shortage_level=ShortageLevel.MILD,
        device="cpu",
        demand_noise=0.0,
        supply_noise=0.0,
    )
    original_item = torch.Tensor.item
    item_calls = 0

    def counted_item(tensor: torch.Tensor):
        nonlocal item_calls
        item_calls += 1
        return original_item(tensor)

    monkeypatch.setattr(torch.Tensor, "item", counted_item)

    result = simulate(scenario, FixedRotationController())

    assert result.service_tiers.shape == (2, 24, 7)
    assert item_calls == 0


def test_simulation_reward_uses_only_the_locked_five_components() -> None:
    demand = torch.ones((1, 24, 7, 5), dtype=torch.float32)
    supply = torch.full((1, 24), 35.0)
    scenario = Scenario(
        demand=demand,
        supply=supply,
        shortage_level=ShortageLevel.MILD,
        seed=0,
    )

    result = simulate(scenario, FixedRotationController())

    expected_first_step = (5.0 * (9.0 / 21.0) + 2.0 * (15.0 / 35.0) + 2.0 * (3.0 / 7.0)) / 9.0
    assert result.reward[0, 0].item() == pytest.approx(expected_first_step, abs=1e-6)
    assert result.reward.shape == (1, 24)
    assert result.controller_name is ControllerName.FIXED_ROTATION


@pytest.mark.gpu
def test_cuda_end_to_end_simulation_preserves_device_and_power_balance() -> None:
    assert torch.cuda.is_available(), "the presentation CUDA runtime is required"
    scenario = generate_scenarios(
        8,
        seed=20260815,
        shortage_level=ShortageLevel.SEVERE,
        device="cuda:0",
    )

    result = simulate(scenario, FixedRotationController())

    assert all(
        tensor.device == torch.device("cuda:0")
        for tensor in (
            result.scenario.demand,
            result.scenario.supply,
            result.service_tiers,
            result.requested,
            result.served,
            result.reward,
        )
    )
    assert torch.all(result.served <= result.scenario.demand + 1e-3)
    assert torch.all(result.served.sum(dim=(-1, -2)) <= result.scenario.supply + 1e-3)
    assert math.isfinite(result.reward.sum().item())


@pytest.mark.gpu
def test_cuda_matched_pair_stays_within_presentation_latency_budget() -> None:
    assert torch.cuda.is_available(), "the presentation CUDA runtime is required"
    scenario = generate_scenarios(
        32,
        seed=20260815,
        shortage_level=ShortageLevel.MODERATE,
        device="cuda:0",
    )
    actor = load_ppo_policy(
        Path("data/model/manifest.json"),
        expected_artifact_sha256=BUNDLED_ARTIFACT_SHA256,
    )
    controllers = (FixedRotationController(), PPOController(actor))
    for controller in controllers:
        simulate(scenario, controller)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for controller in controllers:
        simulate(scenario, controller)
    end.record()
    end.synchronize()

    assert start.elapsed_time(end) < 300.0

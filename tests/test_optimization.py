from __future__ import annotations

import math

import pytest
import torch

from gridguard.controllers import DecisionContext
from gridguard.domain import ControllerName, ShortageLevel
from gridguard.optimization import OptimizationController
from gridguard.scenarios import generate_scenarios
from gridguard.simulation import build_observation, dispatch_service_tiers, simulate


def _context(*, device: str = "cpu", batch_size: int = 2, hour: int = 8) -> DecisionContext:
    generator = torch.Generator(device=device).manual_seed(1887)
    demand = 1 + 9 * torch.rand(
        (batch_size, 7, 5), generator=generator, dtype=torch.float64, device=device
    )
    supply = demand.sum(dim=(-1, -2)) * 0.63
    cumulative_demand = 10 + 40 * torch.rand(
        (batch_size, 7), generator=generator, dtype=demand.dtype, device=device
    )
    cumulative_served = cumulative_demand * torch.rand(
        (batch_size, 7), generator=generator, dtype=demand.dtype, device=device
    )
    previous_tier = torch.randint(0, 3, (batch_size, 7), generator=generator, device=device)
    if hour == 0:
        cumulative_demand.zero_()
        cumulative_served.zero_()
        previous_tier.zero_()
    observation = build_observation(
        hour=hour,
        demand=demand,
        supply=supply,
        cumulative_demand=cumulative_demand,
        cumulative_served=cumulative_served,
        previous_tier=previous_tier,
    )
    return DecisionContext(
        hour=hour,
        observation=observation,
        current_demand_mw=demand,
        current_supply_mw=supply,
        cumulative_demand_mwh=cumulative_demand,
        cumulative_served_mwh=cumulative_served,
        previous_service_tiers=previous_tier,
    )


def _independent_reward(context: DecisionContext, batch: int, tiers: list[int]) -> float:
    """Scalar definition, independent of the production reward and dispatch helpers."""
    demand = context.current_demand_mw[batch].tolist()
    supply = context.current_supply_mw[batch].item()
    past_demand = context.cumulative_demand_mwh[batch].tolist()
    past_served = context.cumulative_served_mwh[batch].tolist()
    previous = context.previous_service_tiers[batch].tolist()
    requested = [
        [
            value if tier == 2 or (tier == 1 and sector < 3) else 0.0
            for sector, value in enumerate(locality)
        ]
        for tier, locality in zip(tiers, demand, strict=True)
    ]
    total_demand = sum(map(sum, demand))
    total_requested = sum(map(sum, requested))
    scale = min(1.0, supply / max(total_requested, 1e-8))
    served = [[value * scale for value in row] for row in requested]
    critical = sum(sum(row[:3]) for row in served) / max(sum(sum(row[:3]) for row in demand), 1e-8)
    utilization = sum(map(sum, served)) / max(min(supply, total_demand), 1e-8)
    ratios = [
        (history_served + sum(now_served)) / max(history_demand + sum(now_demand), 1e-8)
        for history_served, history_demand, now_served, now_demand in zip(
            past_served, past_demand, served, demand, strict=True
        )
    ]
    fairness = sum(ratios) ** 2 / (7 * sum(value**2 for value in ratios) + 1e-8)
    switching = (
        0
        if context.hour == 0
        else sum(now != before for now, before in zip(tiers, previous, strict=True)) / 7
    )
    oversubscription = max(total_requested - supply, 0) / max(total_demand, 1e-8)
    return (
        5 * critical + 2 * utilization + 2 * fairness - 0.25 * switching - 3 * oversubscription
    ) / 9


@pytest.mark.parametrize("hour", [0, 8, 23])
def test_optimization_matches_independent_exhaustive_one_step_oracle(hour: int) -> None:
    context = _context(hour=hour)
    controller = OptimizationController(candidate_chunk_size=113, scenario_chunk_size=1)
    originals = [
        tensor.clone()
        for tensor in (
            context.current_demand_mw,
            context.current_supply_mw,
            context.cumulative_demand_mwh,
            context.cumulative_served_mwh,
            context.previous_service_tiers,
        )
    ]

    actual = controller.act(context)

    assert controller.name is ControllerName.OPTIMIZATION
    assert actual.shape == (2, 7)
    assert actual.dtype == torch.long
    assert actual.device == context.observation.device
    for batch in range(2):
        candidates = [
            [(encoded // 3**locality) % 3 for locality in range(7)] for encoded in range(3**7)
        ]
        rewards = [_independent_reward(context, batch, action) for action in candidates]
        best = max(range(len(rewards)), key=rewards.__getitem__)
        assert actual[batch].tolist() == candidates[best]
        assert math.isclose(
            _independent_reward(context, batch, actual[batch].tolist()),
            max(rewards),
            rel_tol=0,
            abs_tol=1e-12,
        )
    for before, after in zip(
        originals,
        (
            context.current_demand_mw,
            context.current_supply_mw,
            context.cumulative_demand_mwh,
            context.cumulative_served_mwh,
            context.previous_service_tiers,
        ),
        strict=True,
    ):
        torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_candidate_and_scenario_chunk_sizes_preserve_selected_actions() -> None:
    context = _context(batch_size=3)
    all_at_once = OptimizationController(candidate_chunk_size=2187).act(context)
    chunked = OptimizationController(candidate_chunk_size=97, scenario_chunk_size=1).act(context)
    torch.testing.assert_close(chunked, all_at_once, rtol=0, atol=0)


def test_zero_demand_ties_choose_lowest_base_three_action_without_inventing_service() -> None:
    context = _context(batch_size=1, hour=0)
    context.current_demand_mw.zero_()
    context.current_supply_mw.zero_()

    action = OptimizationController(candidate_chunk_size=31).act(context)
    requested, served = dispatch_service_tiers(
        demand=context.current_demand_mw, supply=context.current_supply_mw, tiers=action
    )

    assert action.tolist() == [[0] * 7]
    assert torch.count_nonzero(requested) == torch.count_nonzero(served) == 0


def test_optimization_can_select_critical_only_for_every_locality() -> None:
    context = _context(batch_size=1, hour=0)
    context.current_demand_mw[..., :3] = 1
    context.current_demand_mw[..., 3:] = 100
    context.current_supply_mw.fill_(21)

    action = OptimizationController().act(context)
    requested, served = dispatch_service_tiers(
        demand=context.current_demand_mw, supply=context.current_supply_mw, tiers=action
    )

    assert action.tolist() == [[1] * 7]
    torch.testing.assert_close(served[..., :3], torch.ones((1, 7, 3), dtype=served.dtype))
    assert torch.count_nonzero(served[..., 3:]) == 0
    assert requested.sum() == served.sum() == context.current_supply_mw.sum() == 21


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("current_demand_mw", float("nan")),
        ("current_supply_mw", float("inf")),
        ("current_supply_mw", -1),
        ("cumulative_demand_mwh", -1),
        ("cumulative_served_mwh", 1e6),
        ("previous_service_tiers", 3),
    ],
)
def test_optimization_revalidates_mutable_dispatch_state(field: str, value: float) -> None:
    context = _context(batch_size=1)
    getattr(context, field).flatten()[0] = value

    with pytest.raises(ValueError, match="finite non-negative power"):
        OptimizationController().act(context)


def test_optimization_rejects_missing_exact_dispatch_state() -> None:
    with pytest.raises(ValueError, match="causal dispatch state"):
        OptimizationController().act(DecisionContext(hour=1, observation=torch.zeros((1, 32))))


@pytest.mark.parametrize("field", ["candidate_chunk_size", "scenario_chunk_size"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_optimization_rejects_invalid_chunk_configuration(field: str, value: object) -> None:
    with pytest.raises((ValueError, TypeError), match=field):
        OptimizationController(**{field: value})


def test_optimization_rejects_candidate_chunk_larger_than_action_space() -> None:
    with pytest.raises(ValueError, match="candidate_chunk_size must not exceed 2187"):
        OptimizationController(candidate_chunk_size=2188)


@pytest.mark.gpu
def test_optimization_runs_real_cuda_for_a_full_matched_day_without_host_array_transfer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert torch.cuda.is_available(), "the presentation CUDA runtime is required"
    scenario = generate_scenarios(
        2, seed=50002, shortage_level=ShortageLevel.MODERATE, device="cuda:0"
    )
    controller = OptimizationController(candidate_chunk_size=256, scenario_chunk_size=1)

    def reject_host_array(*_args, **_kwargs):
        raise AssertionError("Optimization must not copy hot-path tensors to the host")

    original_act = controller.act

    def checked_act(context):
        with monkeypatch.context() as guard:
            guard.setattr(torch.Tensor, "cpu", reject_host_array)
            guard.setattr(torch.Tensor, "numpy", reject_host_array)
            guard.setattr(torch.Tensor, "tolist", reject_host_array)
            result = original_act(context)
        assert result.device == torch.device("cuda:0")
        return result

    class CheckedController:
        name = ControllerName.OPTIMIZATION
        act = staticmethod(checked_act)

    result = simulate(scenario, CheckedController())

    assert result.controller_name is ControllerName.OPTIMIZATION
    assert result.scenario is scenario
    assert result.service_tiers.device == result.served.device == torch.device("cuda:0")
    assert torch.isfinite(result.reward).all()
    assert torch.all(result.served.sum(dim=(-1, -2)) <= scenario.supply + 0.001)

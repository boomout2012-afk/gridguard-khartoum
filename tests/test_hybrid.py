from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from gridguard.constrained import (
    ConstrainedOptimizationController,
    evaluate_constrained_candidates,
)
from gridguard.controllers import DecisionContext
from gridguard.hybrid import FeasiblePPOController, PPOHybridController
from gridguard.policy import ACTION_COUNT, BUNDLED_ARTIFACT_SHA256, load_ppo_policy
from gridguard.simulation import build_observation, dispatch_service_tiers


class StaticPolicy(nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("logits", logits)
        self.eval()

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.logits.expand(observation.shape[0], -1)


def context(*, supply: float = 35, device: str = "cpu", hour: int = 0) -> DecisionContext:
    demand = torch.ones((2, 7, 5), device=device)
    supply_tensor = torch.full((2,), float(supply), device=device)
    cumulative_demand = torch.full((2, 7), float(hour * 5), device=device)
    cumulative_served = cumulative_demand * 0.5
    previous_tier = torch.ones((2, 7), dtype=torch.long, device=device)
    observation = build_observation(
        hour=hour,
        demand=demand,
        supply=supply_tensor,
        cumulative_demand=cumulative_demand,
        cumulative_served=cumulative_served,
        previous_tier=previous_tier,
    )
    return DecisionContext(
        hour=hour,
        observation=observation,
        current_demand_mw=demand,
        current_supply_mw=supply_tensor,
        cumulative_demand_mwh=cumulative_demand,
        cumulative_served_mwh=cumulative_served,
        previous_service_tiers=previous_tier,
    )


def policy_preferring(action_id: int, *, device: str = "cpu") -> StaticPolicy:
    logits = torch.zeros((1, ACTION_COUNT), device=device)
    logits[0, action_id] = 10
    return StaticPolicy(logits)


@pytest.mark.parametrize("supply", [0, 14, 21, 25, 35])
def test_zero_tolerance_exactly_matches_constrained_optimizer(supply: float) -> None:
    decision = context(supply=supply)
    policy = policy_preferring(2186)
    actual = PPOHybridController(policy, reward_tolerance=0).act(decision)
    expected = ConstrainedOptimizationController().act(decision)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_zero_tolerance_keeps_greedy_action_id_tie_rule() -> None:
    decision = context(supply=0)
    decision.current_demand_mw.zero_()
    policy = policy_preferring(2186)
    assert PPOHybridController(policy, reward_tolerance=0).act(decision).tolist() == [[1] * 7] * 2


def test_learned_preference_changes_action_only_within_declared_reward_threshold() -> None:
    decision = context()
    policy = policy_preferring(1093)  # All Critical Only, little-endian base three.
    strict = PPOHybridController(policy, reward_tolerance=0.01).act(decision)
    relaxed = PPOHybridController(policy, reward_tolerance=0.1).act(decision)

    assert strict.tolist() == [[2] * 7] * 2
    assert relaxed.tolist() == [[1] * 7] * 2
    candidates = evaluate_constrained_candidates(decision)
    for tolerance, actions in [(0.01, strict), (0.1, relaxed)]:
        selected = (actions[:, None, :] == candidates.tiers[None, :, :]).all(dim=-1)
        score = candidates.rewards[selected]
        assert torch.all(score >= candidates.rewards.max(dim=1).values - tolerance)


def test_feasible_ppo_is_an_explicit_ablation_without_reward_cutoff() -> None:
    decision = context()
    policy = policy_preferring(1093)
    assert FeasiblePPOController(policy).act(decision).tolist() == [[1] * 7] * 2


@pytest.mark.parametrize("controller_type", [FeasiblePPOController, PPOHybridController])
def test_infeasible_high_logit_is_masked_and_critical_shortfall_falls_back(controller_type) -> None:
    policy = policy_preferring(2186)
    controller = controller_type(policy)
    feasible_decision = context(supply=25)
    action = controller.act(feasible_decision)
    requested, served = dispatch_service_tiers(
        demand=feasible_decision.current_demand_mw,
        supply=feasible_decision.current_supply_mw,
        tiers=action,
    )
    assert torch.all(requested.sum(dim=(-1, -2)) <= 25)
    assert torch.all(served[..., :3] == 1)
    assert torch.all(action >= 1)

    shortfall_decision = context(supply=14)
    fallback = controller.act(shortfall_decision)
    assert fallback.tolist() == [[1] * 7] * 2
    _, fallback_served = dispatch_service_tiers(
        demand=shortfall_decision.current_demand_mw,
        supply=shortfall_decision.current_supply_mw,
        tiers=fallback,
    )
    torch.testing.assert_close(fallback_served.sum(dim=(-1, -2)), torch.full((2,), 14.0))


def test_preference_ties_use_canonical_reward_then_smallest_action_id() -> None:
    policy = StaticPolicy(torch.zeros((1, ACTION_COUNT)))
    controller = FeasiblePPOController(policy)
    assert controller.act(context()).tolist() == [[2] * 7] * 2
    constrained = context(supply=25)
    torch.testing.assert_close(
        controller.act(constrained), ConstrainedOptimizationController().act(constrained)
    )


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf"), True, "0.1"])
def test_rejects_invalid_tolerance(value: object) -> None:
    with pytest.raises((ValueError, TypeError), match="reward_tolerance"):
        PPOHybridController(policy_preferring(1093), reward_tolerance=value)


@pytest.mark.parametrize("controller_type", [FeasiblePPOController, PPOHybridController])
def test_rejects_training_or_trainable_policy_without_mutating_it(controller_type) -> None:
    policy = nn.Linear(32, ACTION_COUNT)
    policy.eval()
    with pytest.raises(ValueError, match="frozen"):
        controller_type(policy)
    assert all(parameter.requires_grad for parameter in policy.parameters())
    policy.requires_grad_(False)
    policy.train()
    with pytest.raises(ValueError, match="eval"):
        controller_type(policy)
    assert policy.training


@pytest.mark.parametrize("kind", ["shape", "integer", "nonfinite", "dtype"])
def test_rejects_invalid_logits(kind: str) -> None:
    logits = torch.zeros((1, ACTION_COUNT))
    if kind == "shape":
        logits = logits[:, :-1]
    elif kind == "integer":
        logits = logits.long()
    elif kind == "nonfinite":
        logits[0, 0] = torch.nan
    else:
        logits = logits.double()
    with pytest.raises((ValueError, TypeError), match="logits|dtype"):
        FeasiblePPOController(StaticPolicy(logits)).act(context())


def test_rejects_mismatched_policy_device_before_inference() -> None:
    policy = policy_preferring(1093).to("meta")
    with pytest.raises(ValueError, match="device"):
        FeasiblePPOController(policy).act(context())


def test_revalidates_mutated_observation_and_policy_modes() -> None:
    policy = policy_preferring(1093)
    controller = PPOHybridController(policy, reward_tolerance=0)
    decision = context()
    decision.observation[0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        controller.act(decision)
    policy.train()
    with pytest.raises(ValueError, match="eval"):
        controller.act(context())


def test_rejects_missing_dispatch_state() -> None:
    decision = replace(
        context(),
        current_demand_mw=None,
        current_supply_mw=None,
        cumulative_demand_mwh=None,
        cumulative_served_mwh=None,
        previous_service_tiers=None,
    )
    with pytest.raises(ValueError, match="dispatch state"):
        FeasiblePPOController(policy_preferring(1093)).act(decision)


@pytest.mark.gpu
def test_real_frozen_ppo_is_deterministic_unchanged_and_stays_on_cuda(monkeypatch) -> None:
    assert torch.cuda.is_available(), "the presentation CUDA runtime is required"
    actor = load_ppo_policy(
        Path(__file__).resolve().parents[1] / "data/model/manifest.json",
        expected_artifact_sha256=BUNDLED_ARTIFACT_SHA256,
    )
    before = {name: tensor.clone() for name, tensor in actor.state_dict().items()}
    decision = context(supply=25, device="cuda:0", hour=4)

    def reject_host_array(*_args, **_kwargs):
        raise AssertionError("hybrid hot-path arrays must stay on CUDA")

    for controller in [FeasiblePPOController(actor), PPOHybridController(actor, 0.01)]:
        with monkeypatch.context() as guard:
            for name in ("cpu", "numpy", "tolist"):
                guard.setattr(torch.Tensor, name, reject_host_array)
            first = controller.act(decision)
            second = controller.act(decision)
        assert first.device == torch.device("cuda:0")
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert not actor.training
    assert not any(parameter.requires_grad for parameter in actor.parameters())
    for name, tensor in actor.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)

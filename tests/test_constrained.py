from __future__ import annotations

import itertools
import math

import pytest
import torch

from gridguard.constrained import (
    ConstrainedOptimizationController,
    FairnessController,
    eligible_candidates,
    evaluate_constrained_candidates,
    select_candidates,
)
from gridguard.controllers import DecisionContext
from gridguard.domain import ControllerName, ShortageLevel
from gridguard.scenarios import generate_scenarios
from gridguard.simulation import build_observation, dispatch_service_tiers, simulate


def _context(*, batch: int = 2, hour: int = 8, device: str = "cpu") -> DecisionContext:
    generator = torch.Generator(device=device).manual_seed(87031)
    demand = 1 + torch.rand((batch, 7, 5), generator=generator, dtype=torch.float64, device=device)
    critical = demand[..., :3].sum(dim=(-1, -2))
    supply = critical + demand[..., 3:].sum(dim=(-1, -2)) * 0.37
    past_demand = torch.full((batch, 7), 100.0, dtype=demand.dtype, device=device)
    past_served = past_demand * torch.rand((batch, 7), generator=generator, device=device)
    previous = torch.randint(1, 3, (batch, 7), generator=generator, device=device)
    if hour == 0:
        past_demand.zero_()
        past_served.zero_()
        previous.zero_()
    return DecisionContext(
        hour=hour,
        observation=build_observation(
            hour=hour,
            demand=demand,
            supply=supply,
            cumulative_demand=past_demand,
            cumulative_served=past_served,
            previous_tier=previous,
        ),
        current_demand_mw=demand,
        current_supply_mw=supply,
        cumulative_demand_mwh=past_demand,
        cumulative_served_mwh=past_served,
        previous_service_tiers=previous,
    )


def _scalar_candidates(context: DecisionContext, row: int) -> list[tuple]:
    """Independent scalar dispatch and reward oracle, including ordered action IDs."""
    demand = context.current_demand_mw[row].tolist()
    supply = context.current_supply_mw[row].item()
    past_demand = context.cumulative_demand_mwh[row].tolist()
    past_served = context.cumulative_served_mwh[row].tolist()
    previous = context.previous_service_tiers[row].tolist()
    values = []
    for tiers in itertools.product((1, 2), repeat=7):
        requested = [
            sum(locality if tier == 2 else locality[:3])
            for locality, tier in zip(demand, tiers, strict=True)
        ]
        requested_total = sum(requested)
        scale = min(1.0, supply / max(requested_total, 1e-8))
        served = [value * scale for value in requested]
        ratios = [
            (history_served + now_served) / max(history_demand + sum(now_demand), 1e-8)
            for history_served, now_served, history_demand, now_demand in zip(
                past_served, served, past_demand, demand, strict=True
            )
        ]
        critical = sum(sum(locality[:3]) for locality in demand)
        critical_ratio = critical * scale / max(critical, 1e-8)
        utilization = sum(served) / max(min(supply, sum(map(sum, demand))), 1e-8)
        fairness = sum(ratios) ** 2 / (7 * sum(value**2 for value in ratios) + 1e-8)
        switching = (
            0
            if context.hour == 0
            else sum(a != b for a, b in zip(tiers, previous, strict=True)) / 7
        )
        excess = max(requested_total - supply, 0) / max(sum(map(sum, demand)), 1e-8)
        reward = (
            5 * critical_ratio + 2 * utilization + 2 * fairness - 0.25 * switching - 3 * excess
        ) / 9
        action_id = sum(tier * 3**locality for locality, tier in enumerate(tiers))
        values.append((action_id, list(tiers), requested_total <= supply, reward, min(ratios)))
    return sorted(values)


@pytest.mark.parametrize("hour", [0, 8, 23])
def test_constrained_optimizer_matches_independent_scalar_oracle(hour: int) -> None:
    context = _context(hour=hour)
    original = tuple(
        value.clone()
        for value in (
            context.current_demand_mw,
            context.current_supply_mw,
            context.cumulative_demand_mwh,
            context.cumulative_served_mwh,
            context.previous_service_tiers,
        )
    )
    candidates = evaluate_constrained_candidates(context)
    actual = ConstrainedOptimizationController().act(context)
    assert candidates.tiers.shape == (128, 7)
    assert candidates.rewards.shape == (2, 128)
    assert candidates.service_ratios.shape == (2, 128, 7)
    assert candidates.action_ids.unique().numel() == 128
    assert bool((candidates.action_ids[1:] > candidates.action_ids[:-1]).all())
    assert ConstrainedOptimizationController.name is ControllerName.OPTIMIZATION
    for row in range(2):
        oracle = _scalar_candidates(context, row)
        best = max((value for value in oracle if value[2]), key=lambda value: value[3])
        assert actual[row].tolist() == best[1]
        assert candidates.action_ids.tolist() == [value[0] for value in oracle]
        assert candidates.feasible[row].tolist() == [value[2] for value in oracle]
        torch.testing.assert_close(
            candidates.rewards[row],
            torch.tensor([value[3] for value in oracle], dtype=context.observation.dtype),
            atol=1e-12,
            rtol=0,
        )
    for before, after in zip(
        original,
        (
            context.current_demand_mw,
            context.current_supply_mw,
            context.cumulative_demand_mwh,
            context.cumulative_served_mwh,
            context.previous_service_tiers,
        ),
        strict=True,
    ):
        torch.testing.assert_close(before, after, atol=0, rtol=0)


@pytest.mark.parametrize("tolerance", [0.0, 0.005, 0.02, 1.0])
def test_fairness_rule_matches_independent_bounded_regret_oracle(tolerance: float) -> None:
    context = _context(batch=3)
    actual = FairnessController(reward_tolerance=tolerance).act(context)
    for row in range(3):
        feasible = [value for value in _scalar_candidates(context, row) if value[2]]
        best_reward = max(value[3] for value in feasible)
        eligible = [value for value in feasible if value[3] >= best_reward - tolerance]
        best = max(eligible, key=lambda value: (value[4], value[3]))
        assert actual[row].tolist() == best[1]
        assert best_reward - best[3] <= tolerance + 1e-12


def test_fairness_rule_serves_locality_with_largest_service_deficit() -> None:
    context = _context(batch=1)
    context.current_demand_mw.fill_(1)
    context.current_supply_mw.fill_(23)
    context.cumulative_demand_mwh.fill_(100)
    context.cumulative_served_mwh.fill_(90)
    context.cumulative_served_mwh[0, 5] = 10
    context.previous_service_tiers.fill_(1)
    action = FairnessController(reward_tolerance=1).act(context)
    assert action.tolist() == [[1, 1, 1, 1, 1, 2, 1]]


@pytest.mark.parametrize("controller", [ConstrainedOptimizationController(), FairnessController()])
def test_critical_shortfall_is_explicit_and_falls_back_to_proportional_critical_only(
    controller,
) -> None:
    context = _context(batch=2)
    context.current_demand_mw.fill_(1)
    context.current_supply_mw[:] = torch.tensor([10.5, 21.0], dtype=torch.float64)
    candidates = evaluate_constrained_candidates(context)
    assert candidates.critical_shortfall.tolist() == [True, False]
    assert candidates.feasible[0].sum() == 0
    assert candidates.feasible[1].sum() == 1
    eligible = eligible_candidates(candidates, reward_tolerance=None)
    assert eligible[0].sum() == eligible[1].sum() == 1
    assert bool(eligible[:, 0].all())
    action = controller.act(context)
    assert action.tolist() == [[1] * 7, [1] * 7]
    requested, served = dispatch_service_tiers(
        demand=context.current_demand_mw, supply=context.current_supply_mw, tiers=action
    )
    torch.testing.assert_close(served[0, :, :3], torch.full((7, 3), 0.5, dtype=torch.float64))
    assert requested[0].sum() > context.current_supply_mw[0]
    assert served[0].sum() == context.current_supply_mw[0]
    assert torch.count_nonzero(served[..., 3:]) == 0


def test_zero_demand_ties_select_lowest_base_three_allowed_action() -> None:
    context = _context(batch=1, hour=0)
    context.current_demand_mw.zero_()
    context.current_supply_mw.zero_()
    candidates = evaluate_constrained_candidates(context)
    assert candidates.action_ids[0] == sum(3**locality for locality in range(7))
    for controller in (ConstrainedOptimizationController(), FairnessController()):
        assert controller.act(context).tolist() == [[1] * 7]


def test_selection_uses_reward_before_action_id_to_break_preference_ties() -> None:
    candidates = evaluate_constrained_candidates(_context())
    equal_preference = torch.zeros_like(candidates.rewards)
    actual = select_candidates(candidates, preference=equal_preference, reward_tolerance=None)
    expected = select_candidates(candidates)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_preferences_never_override_physical_feasibility_or_reward_bound() -> None:
    candidates = evaluate_constrained_candidates(_context())
    preference = torch.arange(128, dtype=candidates.rewards.dtype).expand(2, -1)
    actions = select_candidates(candidates, preference=preference, reward_tolerance=0.005)
    eligible = eligible_candidates(candidates, reward_tolerance=0.005)
    for row in range(2):
        selected = (candidates.tiers == actions[row]).all(dim=1).nonzero().item()
        assert candidates.feasible[row, selected]
        assert eligible[row, selected]
        assert selected == eligible[row].nonzero()[-1].item()


def test_zero_fairness_tolerance_exactly_recovers_constrained_greedy() -> None:
    context = _context(batch=5)
    torch.testing.assert_close(
        FairnessController(reward_tolerance=0).act(context),
        ConstrainedOptimizationController().act(context),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("kind", ["shape", "dtype", "nan"])
def test_selection_rejects_invalid_preference(kind: str) -> None:
    candidates = evaluate_constrained_candidates(_context())
    preference = torch.zeros_like(candidates.rewards)
    if kind == "shape":
        preference = preference[:, :1]
    elif kind == "dtype":
        preference = preference.to(torch.long)
    else:
        preference[0, 0] = math.nan
    with pytest.raises((TypeError, ValueError), match="preference"):
        select_candidates(candidates, preference=preference)


@pytest.mark.parametrize("value", [-0.01, float("nan"), float("inf"), True, "0.1"])
def test_fairness_rejects_invalid_reward_tolerance(value) -> None:
    with pytest.raises((TypeError, ValueError), match="reward_tolerance"):
        FairnessController(reward_tolerance=value)


def test_candidate_evaluation_requires_and_revalidates_exact_causal_state() -> None:
    with pytest.raises(ValueError, match="causal dispatch state"):
        evaluate_constrained_candidates(DecisionContext(hour=0, observation=torch.zeros((1, 32))))
    context = _context()
    context.current_demand_mw[0, 0, 0] = math.nan
    with pytest.raises(ValueError, match="finite non-negative power"):
        evaluate_constrained_candidates(context)


def test_candidate_evaluation_revalidates_mutated_observation() -> None:
    context = _context()
    context.observation[0, 0] = math.nan
    with pytest.raises(ValueError, match="finite"):
        evaluate_constrained_candidates(context)


@pytest.mark.gpu
@pytest.mark.parametrize("controller", [ConstrainedOptimizationController(), FairnessController()])
def test_constrained_methods_keep_cuda_hot_path_on_device(
    monkeypatch: pytest.MonkeyPatch, controller
) -> None:
    assert torch.cuda.is_available(), "the presentation CUDA runtime is required"
    scenario = generate_scenarios(
        2, seed=50011, shortage_level=ShortageLevel.SEVERE, device="cuda:0"
    )

    def reject_host_array(*_args, **_kwargs):
        raise AssertionError("constrained controller must retain hot-path tensors on CUDA")

    def checked_act(context):
        with monkeypatch.context() as guard:
            for name in ("cpu", "numpy", "tolist"):
                guard.setattr(torch.Tensor, name, reject_host_array)
            result = controller.act(context)
        assert result.device == torch.device("cuda:0")
        return result

    class CheckedController:
        name = ControllerName.OPTIMIZATION
        act = staticmethod(checked_act)

    result = simulate(scenario, CheckedController())
    torch.testing.assert_close(result.served[..., :3], scenario.demand[..., :3])
    assert bool((result.requested.sum(dim=(-1, -2)) <= scenario.supply).all())

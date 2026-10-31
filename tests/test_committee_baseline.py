from __future__ import annotations

import pytest
import torch

from gridguard.committee_baseline import (
    REWARD_COMPONENT_NAMES,
    CriticalAwareFixedRotationController,
    audit_diagnostics,
    critical_supply_stress,
)
from gridguard.controllers import FixedRotationController
from gridguard.domain import Scenario, ShortageLevel
from gridguard.duration_metrics import compute_duration_metrics
from gridguard.simulation import simulate
from tests.test_constrained import _context


@pytest.mark.parametrize("hour", range(24))
def test_critical_rotation_matches_independent_greedy_priority(hour: int) -> None:
    context = _context(batch=3, hour=hour)
    actual = CriticalAwareFixedRotationController().act(context)
    for row in range(3):
        demand = context.current_demand_mw[row].tolist()
        remaining = context.current_supply_mw[row].item() - sum(sum(d[:3]) for d in demand)
        expected = [1] * 7
        start = (hour // 3 * 3) % 7
        for offset in range(7):
            locality = (start + offset) % 7
            discretionary = sum(demand[locality][3:])
            if discretionary <= remaining:
                expected[locality] = 2
                remaining -= discretionary
        assert actual[row].tolist() == expected


def test_critical_rotation_ignores_reward_and_service_history(monkeypatch) -> None:
    import gridguard.constrained as constrained
    import gridguard.domain as domain

    def forbidden(*args, **kwargs):
        raise AssertionError("rotation must not calculate or optimize reward")

    monkeypatch.setattr(constrained, "evaluate_constrained_candidates", forbidden)
    monkeypatch.setattr(domain, "canonical_step_reward", forbidden)
    context = _context()
    before = CriticalAwareFixedRotationController().act(context)
    context.cumulative_served_mwh.zero_()
    context.previous_service_tiers.zero_()
    after = CriticalAwareFixedRotationController().act(context)
    assert torch.equal(before, after)


def test_critical_rotation_uses_capacity_beyond_original_three_locality_cap() -> None:
    context = _context(batch=1)
    context.current_supply_mw.copy_(context.current_demand_mw.sum((-1, -2)))
    assert CriticalAwareFixedRotationController().act(context).tolist() == [[2] * 7]


def test_critical_rotation_marks_infeasible_floor_only_through_explicit_fallback() -> None:
    context = _context(batch=2)
    context.current_supply_mw.copy_(context.current_demand_mw[..., :3].sum((-1, -2)) * 0.75)
    assert CriticalAwareFixedRotationController().act(context).tolist() == [[1] * 7] * 2


@pytest.mark.gpu
@pytest.mark.parametrize("fraction", [0.75, 0.25])
def test_extreme_shortage_counts_partial_critical_interruptions(fraction: float) -> None:
    demand = torch.ones((2, 24, 7, 5), device="cuda:0", dtype=torch.float32)
    original = Scenario(
        demand, torch.full((2, 24), 30.0, device="cuda:0"), ShortageLevel.MODERATE, 7
    )
    stress = critical_supply_stress(original, fraction)
    result = simulate(stress, CriticalAwareFixedRotationController())
    metrics = compute_duration_metrics(result)
    diagnostics = audit_diagnostics(result)
    assert torch.equal(original.supply, torch.full_like(original.supply, 30))
    assert metrics.critical_interruption_locality_hours.tolist() == [168.0, 168.0]
    assert metrics.blackout_locality_hours.tolist() == [0.0, 0.0]
    assert diagnostics["critical_supply_shortfall_hours"].tolist() == [24.0, 24.0]
    assert diagnostics["all_critical_fallback_hours"].tolist() == [24.0, 24.0]
    assert diagnostics["complete_critical_outage_locality_hours"].tolist() == [0.0, 0.0]
    torch.testing.assert_close(
        diagnostics["unmet_critical_mwh"],
        torch.full((2,), 24 * 21 * (1 - fraction), device="cuda:0"),
    )


@pytest.mark.gpu
@pytest.mark.parametrize(
    "controller", [FixedRotationController(), CriticalAwareFixedRotationController()]
)
def test_reward_decomposition_reconstructs_locked_trace(controller) -> None:
    demand = torch.ones((2, 24, 7, 5), device="cuda:0", dtype=torch.float32)
    scenario = Scenario(
        demand, torch.full((2, 24), 30.0, device="cuda:0"), ShortageLevel.MODERATE, 7
    )
    result = simulate(scenario, controller)
    components = audit_diagnostics(result)
    actual = sum(components[name] for name in REWARD_COMPONENT_NAMES)
    torch.testing.assert_close(actual, result.reward.sum(1), atol=5e-5, rtol=0)


@pytest.mark.parametrize("fraction", [True, -0.1, 0, 1, float("nan")])
def test_stress_requires_a_finite_fraction_strictly_between_zero_and_one(fraction) -> None:
    demand = torch.ones((1, 24, 7, 5))
    scenario = Scenario(demand, torch.full((1, 24), 30.0), ShortageLevel.MODERATE, 7)
    with pytest.raises((ValueError, TypeError)):
        critical_supply_stress(scenario, fraction)


@pytest.mark.gpu
def test_audit_summary_pairs_the_five_runs_with_each_correct_baseline() -> None:
    from tools.evaluate_committee_baseline import METRICS, summary_tensors

    cube = torch.arange(9, dtype=torch.float64, device="cuda:0")[:, None, None, None, None]
    cube = cube.expand(9, 3, 10, 32, len(METRICS)).contiguous()
    summary = summary_tensors(cube, cube[:, :2])
    assert summary["procedure_means"].shape == (1, len(METRICS))
    torch.testing.assert_close(
        summary["procedure_means"], torch.full_like(summary["procedure_means"], 6)
    )
    for index in range(4):
        torch.testing.assert_close(
            summary["paired"][index, 0], torch.full_like(summary["paired"][index, 0], 6 - index)
        )
    torch.testing.assert_close(summary["mechanical"], torch.ones_like(summary["mechanical"]))


def test_audit_execution_freeze_still_binds_original_sources_and_all_five_models() -> None:
    from tools.evaluate_committee_baseline import load_freeze

    frozen, training = load_freeze()
    assert len(frozen["models"]) == 5
    assert [run["seed"] for run in training["runs"]] == [1729, 2718, 3141, 5772, 8111]
    assert frozen["recipe"]["post_hoc"] is True


@pytest.mark.gpu
def test_published_audit_reconstructs_all_paired_intervals() -> None:
    from tools.evaluate_committee_baseline import OUTPUT, verify

    assert verify(OUTPUT) == {
        "passed": True,
        "schema": "gridguard.committee_baseline_audit.1",
        "nominal_conditions": 960,
        "nominal_trajectories": 8640,
        "stress_trajectories": 5760,
    }


@pytest.mark.parametrize("field", ["nominal_matched_conditions", "controller_names", "models"])
def test_audit_rejects_changed_counts_labels_and_model_contract(tmp_path, field) -> None:
    import hashlib
    import json

    from tools.evaluate_committee_baseline import OUTPUT, verify

    payload = json.loads(OUTPUT.read_text(encoding="utf-8"))
    payload[field] = 17 if field == "nominal_matched_conditions" else []
    encoded = json.dumps(payload).encode()
    path = tmp_path / "audit.json"
    path.write_bytes(encoded)
    with pytest.raises(ValueError, match="frozen contract|condition counts"):
        verify(path, hashlib.sha256(encoded).hexdigest())


@pytest.mark.gpu
@pytest.mark.parametrize("target", ["mean", "interval", "partial_interruption"])
def test_audit_rejects_changed_conclusions_even_when_outer_hash_is_updated(
    tmp_path, target
) -> None:
    import hashlib
    import json

    from tools.evaluate_committee_baseline import METRICS, OUTPUT, verify

    payload = json.loads(OUTPUT.read_text(encoding="utf-8"))
    if target == "mean":
        payload["procedure_means"]["PPO"]["served_mwh"] += 1
    elif target == "interval":
        payload["paired_inference"]["PPO - Fairness Rule"]["served_mwh"]["lower"] += 1
    else:
        payload["stress_metric_cube"][1][0][0][0][
            METRICS.index("critical_interruption_locality_hours")
        ] = 0
    encoded = json.dumps(payload).encode()
    path = tmp_path / "audit.json"
    path.write_bytes(encoded)
    with pytest.raises(ValueError):
        verify(path, hashlib.sha256(encoded).hexdigest())

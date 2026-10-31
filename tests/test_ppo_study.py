from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from gridguard.ppo_study import (
    acceptance_decision,
    checkpoint_eligibility,
    conditional_paired_bootstrap,
    hierarchical_paired_bootstrap,
    select_checkpoint,
    verify_checkpoint_selection,
)
from gridguard.statistics import BootstrapError


def _summary() -> dict:
    row = {"episode_reward": 23.5, "fairness": 0.96, "utilization": 0.97}
    return {
        "overall": row.copy(),
        "by_shortage": {level: row.copy() for level in ("mild", "moderate", "severe")},
        "maxima": {
            "critical_outage_locality_hours": 0.0,
            "critical_interruption_locality_hours": 0.0,
            "power_balance_violation_rate": 0.0,
        },
    }


def _intervals() -> dict:
    return {
        "episode_reward": {"mean": 0.03, "lower": 0.02, "upper": 0.04},
        "fairness": {"mean": 0.0, "lower": -0.002, "upper": 0.002},
        "utilization": {"mean": 0.0, "lower": -0.002, "upper": 0.002},
    }


def _shortage_deltas() -> dict:
    return {
        level: {"episode_reward": 0.0, "fairness": -0.002, "utilization": -0.002}
        for level in ("mild", "moderate", "severe")
    }


def test_checkpoint_accepts_equal_teacher_and_keeps_earliest_tie() -> None:
    reference = _summary()
    records = [
        {"iteration": 50, "summary": deepcopy(reference)},
        {"iteration": 0, "summary": deepcopy(reference)},
    ]
    before = deepcopy(records)
    selected = select_checkpoint(records, reference)
    assert selected["iteration"] == 0
    assert selected["eligible"] is True
    assert selected["fallback"] is False
    assert records == before


@pytest.mark.parametrize(
    ("section", "metric", "change"),
    [
        ("overall", "episode_reward", -0.000001),
        ("overall", "fairness", -0.003),
        ("overall", "utilization", -0.003),
        ("severe", "episode_reward", -0.000001),
        ("mild", "fairness", -0.003),
        ("moderate", "utilization", -0.003),
        ("maxima", "critical_outage_locality_hours", 0.000001),
        ("maxima", "critical_interruption_locality_hours", 0.000001),
        ("maxima", "power_balance_violation_rate", 0.000001),
    ],
)
def test_checkpoint_rejects_each_guardrail(section: str, metric: str, change: float) -> None:
    reference = _summary()
    candidate = deepcopy(reference)
    row = candidate.get(section, candidate["by_shortage"].get(section))
    row[metric] += change
    decision = checkpoint_eligibility(candidate, reference)
    assert decision["eligible"] is False
    assert decision["reasons"]


def test_checkpoint_prefers_eligible_to_higher_reward_and_flags_fallback() -> None:
    reference = _summary()
    unsafe = deepcopy(reference)
    unsafe["overall"]["episode_reward"] += 1
    unsafe["maxima"]["critical_interruption_locality_hours"] = 1
    records = [
        {"iteration": 50, "summary": unsafe},
        {"iteration": 100, "summary": deepcopy(reference)},
    ]
    selected = select_checkpoint(records, reference)
    assert selected["iteration"] == 100
    fallback = select_checkpoint(records[:1], reference)
    assert fallback["iteration"] == 50
    assert fallback["eligible"] is False
    assert fallback["fallback"] is True


def test_selection_verifier_rejects_changed_claims_and_duplicate_iterations() -> None:
    reference = _summary()
    records = [{"iteration": 0, "summary": deepcopy(reference)}]
    selected = select_checkpoint(records, reference)
    verify_checkpoint_selection(records, reference, selected)
    selected["eligible"] = False
    with pytest.raises(ValueError, match="selection"):
        verify_checkpoint_selection(records, reference, selected)
    with pytest.raises(ValueError, match="unique"):
        select_checkpoint(records + records, reference)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, "23.5"])
def test_selection_rejects_invalid_numbers(bad: object) -> None:
    candidate = _summary()
    candidate["overall"]["episode_reward"] = bad
    with pytest.raises(ValueError, match="finite"):
        checkpoint_eligibility(candidate, _summary())


def test_acceptance_observes_strict_reward_boundary_and_explicit_safety() -> None:
    intervals = _intervals()
    decision = acceptance_decision(intervals, _shortage_deltas(), all_safe=True, all_eligible=True)
    assert decision["passed"] is True
    intervals["episode_reward"]["lower"] = 0.01
    assert not acceptance_decision(intervals, _shortage_deltas(), all_safe=True, all_eligible=True)[
        "passed"
    ]
    for safe, eligible in ((False, True), (True, False)):
        assert not acceptance_decision(
            _intervals(), _shortage_deltas(), all_safe=safe, all_eligible=eligible
        )["passed"]


def test_acceptance_rejects_missing_severity_nonfinite_and_reversed_intervals() -> None:
    incomplete = _shortage_deltas()
    del incomplete["severe"]
    with pytest.raises(ValueError, match="shortage"):
        acceptance_decision(_intervals(), incomplete, all_safe=True, all_eligible=True)
    intervals = _intervals()
    intervals["fairness"]["mean"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        acceptance_decision(intervals, _shortage_deltas(), all_safe=True, all_eligible=True)
    intervals = _intervals()
    intervals["fairness"]["lower"] = 2
    with pytest.raises(ValueError, match="reversed"):
        acceptance_decision(intervals, _shortage_deltas(), all_safe=True, all_eligible=True)


def test_bootstrap_rejects_cpu_malformed_and_non_tensor_inputs() -> None:
    with pytest.raises(TypeError, match="torch.Tensor"):
        hierarchical_paired_bootstrap([1])  # type: ignore[arg-type]
    with pytest.raises(BootstrapError, match="shape"):
        hierarchical_paired_bootstrap(torch.zeros(5, 3))
    with pytest.raises(BootstrapError, match="CUDA"):
        hierarchical_paired_bootstrap(torch.zeros(5, 3, 2, 4, 1))


@pytest.mark.gpu
def test_identical_runs_keep_shared_scenario_uncertainty() -> None:
    pattern = torch.linspace(-4, 4, 32, device="cuda:0", dtype=torch.float64)
    deltas = pattern[None, None, None, :, None].expand(5, 3, 2, 32, 1).clone()
    aggregate = hierarchical_paired_bootstrap(deltas, n_resamples=1000, seed=19)
    conditional = conditional_paired_bootstrap(deltas, n_resamples=1000, seed=19)
    assert aggregate.shape == (3, 1)
    assert conditional.shape == (5, 3, 1)
    assert aggregate.device == torch.device("cuda:0")
    assert aggregate.dtype == torch.float64
    assert float(aggregate[1, 0]) < 0 < float(aggregate[2, 0])
    for run in conditional:
        torch.testing.assert_close(aggregate, run, atol=1e-15, rtol=0)


@pytest.mark.gpu
def test_run_variation_is_not_treated_as_fixed_and_conditional_intervals_are_fixed() -> None:
    pattern = torch.arange(5, device="cuda:0", dtype=torch.float64)
    deltas = pattern[:, None, None, None, None].expand(5, 3, 2, 8, 1).clone()
    aggregate = hierarchical_paired_bootstrap(deltas, n_resamples=1000, seed=21)
    conditional = conditional_paired_bootstrap(deltas, n_resamples=1000, seed=21)
    assert float(aggregate[1, 0]) < 2 < float(aggregate[2, 0])
    torch.testing.assert_close(conditional, pattern[:, None, None].expand(5, 3, 1))


@pytest.mark.gpu
def test_bootstrap_retains_coupled_severities_and_scenario_seed_strata() -> None:
    pattern = torch.arange(16, device="cuda:0", dtype=torch.float64)
    deltas = torch.zeros((5, 3, 2, 16, 1), device="cuda:0", dtype=torch.float64)
    deltas[:, 0, :, :, 0] = pattern
    deltas[:, 1, :, :, 0] = -pattern
    deltas[:, :, 0, :, :] += 2
    deltas[:, :, 1, :, :] -= 2
    aggregate = hierarchical_paired_bootstrap(deltas, n_resamples=1000, seed=42)
    torch.testing.assert_close(aggregate, torch.zeros_like(aggregate), atol=1e-15, rtol=0)


@pytest.mark.gpu
def test_bootstrap_is_deterministic_rejects_nonfinite_and_bad_settings() -> None:
    generator = torch.Generator(device="cuda:0").manual_seed(32)
    deltas = torch.randn((5, 3, 2, 8, 2), generator=generator, device="cuda:0")
    first = hierarchical_paired_bootstrap(deltas, n_resamples=1000, seed=11)
    second = hierarchical_paired_bootstrap(deltas, n_resamples=1000, seed=11)
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    for settings in ({"n_resamples": 0}, {"n_resamples": True}, {"seed": -1}):
        with pytest.raises(BootstrapError):
            hierarchical_paired_bootstrap(deltas, **settings)
    deltas[1, 1, 1, 1, 0] = torch.inf
    with pytest.raises(BootstrapError, match="finite"):
        hierarchical_paired_bootstrap(deltas)

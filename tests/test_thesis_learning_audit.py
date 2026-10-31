"""Independent checks for the additive fresh-input learning-stage audit."""

from types import SimpleNamespace

import pytest
import torch

from gridguard.allocation_study import METRIC_NAMES
from gridguard.constrained import ConstrainedOptimizationController
from gridguard.controllers import FixedRotationController
from gridguard.duration_metrics import compute_duration_metrics
from gridguard.metrics import compute_episode_metrics
from gridguard.scenarios import generate_scenario
from gridguard.simulation import simulate
from gridguard.thesis_learning_audit import (
    assert_disjoint_seeds,
    collect_seed_metadata,
    contrast_cubes,
    independent_trace_metrics,
    verify_trace_metrics,
)


def test_seed_inventory_reads_nested_seeds_and_expands_declared_ranges():
    payload = {
        "evaluation": {"seeds": [15, 16]},
        "teacher": {"episode_seed_start": 100, "episode_seed_end_inclusive": 103},
        "metrics": [132201.0, 132202.5],
        "runs": [{"seed": 42}],
    }
    values, formulas = collect_seed_metadata(payload)
    assert values == {15, 16, 42, 100, 101, 102, 103}
    assert not formulas


def test_seed_inventory_expands_training_formula_without_executing_code():
    values, formulas = collect_seed_metadata(
        {
            "training_seeds": [1729, 2718],
            "iterations": 3,
            "training_scenario_seed": "4000000 + run_index * 10000 + iteration_index",
            "bc_scenario_seed": "4500000 + run_index * 10000",
        }
    )
    assert set(range(4000000, 4000003)) <= values
    assert set(range(4010000, 4010003)) <= values
    assert {4500000, 4510000} <= values
    assert len(formulas) == 2


def test_new_seed_overlap_fails_before_generation():
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint_seeds((132201, 132202), {132201})
    assert_disjoint_seeds((132201, 132202), {122201, 122202})


def test_contrast_preserves_own_run_pairing():
    cube = torch.zeros(14, 3, 1, 2, 1, device="cuda:0", dtype=torch.float64)
    for run in range(5):
        cube[4 + run] = run * 10
        cube[9 + run] = run * 10 + run + 1
    contrasts = contrast_cubes(cube)
    torch.testing.assert_close(
        contrasts[0][:, 0, 0, 0, 0], torch.arange(1, 6, device="cuda:0", dtype=torch.float64)
    )
    assert len(contrasts) == 5


@pytest.mark.parametrize(
    "controller", [FixedRotationController(), ConstrainedOptimizationController()]
)
def test_independent_metrics_reconstruct_trace_and_endpoints(controller):
    scenario = generate_scenario(seed=17, shortage_level="severe", device="cuda:0")
    result = simulate(scenario, controller)
    canonical = compute_episode_metrics(result).as_dict()
    canonical.update(compute_duration_metrics(result).as_dict())
    reconstructed, endpoints = independent_trace_metrics(result)
    assert set(METRIC_NAMES) <= reconstructed.keys()
    verify_trace_metrics(canonical, reconstructed)
    assert endpoints.shape == (1, 7)
    torch.testing.assert_close(
        endpoints, result.served.double().sum((1, 3)) / scenario.demand.double().sum((1, 3))
    )


def test_trace_verification_rejects_changed_metric():
    actual = {"fairness": torch.tensor([0.9], device="cuda:0")}
    altered = {"fairness": torch.tensor([0.91], device="cuda:0")}
    with pytest.raises(ValueError, match="fairness"):
        verify_trace_metrics(actual, altered)


def test_independent_reconstruction_counts_partial_critical_deprivation():
    demand = torch.ones(1, 24, 7, 5, device="cuda:0")
    requested = demand.clone()
    requested[..., 3:] = 0
    trace = SimpleNamespace(
        scenario=SimpleNamespace(demand=demand, supply=torch.full((1, 24), 15.75, device="cuda:0")),
        served=requested * 0.75,
        requested=requested,
        service_tiers=torch.ones(1, 24, 7, device="cuda:0", dtype=torch.long),
    )
    metrics, endpoints = independent_trace_metrics(trace)
    assert metrics["critical_interruption_locality_hours"].item() == 168
    assert metrics["blackout_locality_hours"].item() == 0
    assert metrics["critical_service_ratio"].item() == 0.75
    assert metrics["served_mwh"].item() == 378
    torch.testing.assert_close(endpoints, torch.full_like(endpoints, 0.45))


def test_scenario_hash_overlap_is_rejected():
    from tools.evaluate_thesis_learning_audit import assert_fresh_hashes

    with pytest.raises(ValueError, match="previous"):
        assert_fresh_hashes([[["a" * 64]]], {"a" * 64})
    with pytest.raises(ValueError, match="duplicate"):
        assert_fresh_hashes([[["b" * 64, "b" * 64]]], set())
    assert_fresh_hashes([[["b" * 64]]], {"a" * 64})


def test_contract_rejects_relabelled_or_recounted_result():
    from tools.evaluate_thesis_learning_audit import contract_fields, validate_contract

    frozen = {"source_sha256": {"file": "hash"}, "models": {"model": "identity"}}
    payload = contract_fields(frozen, "freezehash")
    validate_contract(payload, frozen, "freezehash")
    payload["matched_scenario_count"] = 959
    with pytest.raises(ValueError, match="contract"):
        validate_contract(payload, frozen, "freezehash")

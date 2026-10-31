from copy import deepcopy
from pathlib import Path

import pytest

from gridguard.proposal_evidence import (
    load_proposal_supplement,
    validate_proposal_supplement,
    verify_proposal_implementation_sources,
    verify_proposal_inference_on_cuda,
)

ROOT = Path(__file__).resolve().parents[1]


def test_extended_evidence_has_all_methods_and_matched_counts():
    payload = load_proposal_supplement()
    assert payload["controller_names"] == ["Fixed Rotation", "PPO", "Optimization"]
    assert payload["protocol"]["controller_evaluations"] == 2880
    assert payload["forecast"]["training_trace_count"] == 768
    assert set(payload["forecast"]["methods"]) == {
        "ridge_ar3",
        "persistence_1h",
        "trailing_mean_3h",
    }


@pytest.mark.parametrize("mutation", ["mean", "counts", "duration", "forecast", "labels"])
def test_extended_evidence_rejects_semantic_mutations(mutation):
    payload = deepcopy(load_proposal_supplement())
    if mutation == "mean":
        payload["controller_means"]["PPO"]["fairness"] += 0.01
    elif mutation == "counts":
        payload["protocol"]["matched_scenario_conditions"] = 959
    elif mutation == "duration":
        payload["locality_duration_cube"][0][0][0][0][0][0] = 25
    elif mutation == "forecast":
        payload["forecast"]["methods"]["ridge_ar3"]["system_total"]["rmse_mw"] += 1
    else:
        payload["controller_names"][0] = "Other"
    with pytest.raises(ValueError):
        validate_proposal_supplement(payload)


def test_extended_evidence_rejects_modified_file(tmp_path):
    path = tmp_path / "tampered.json"
    path.write_bytes((ROOT / "data/evidence/proposal-supplement.json").read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA-256"):
        load_proposal_supplement(path)


def test_current_experiment_source_bytes_match_recorded_provenance():
    verify_proposal_implementation_sources(ROOT)


@pytest.mark.gpu
def test_all_paired_intervals_reconstruct_on_cuda():
    verify_proposal_inference_on_cuda(load_proposal_supplement())


@pytest.mark.gpu
def test_cuda_reconstruction_rejects_invented_confidence_interval():
    payload = deepcopy(load_proposal_supplement())
    payload["paired_inference"]["PPO - Optimization"]["fairness"]["ci95_upper"] += 0.01
    with pytest.raises(ValueError, match="intervals"):
        verify_proposal_inference_on_cuda(payload)

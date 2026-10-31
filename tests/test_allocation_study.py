from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

import gridguard.allocation_study as study
from gridguard.allocation_study import (
    METRIC_NAMES,
    REWARD_TOLERANCES,
    VALIDATION_SEEDS,
    assess_hybrid_comparison,
    select_validation_settings,
)


@pytest.mark.gpu
def test_validation_rejects_reward_gain_that_breaks_service_guardrails() -> None:
    cube = torch.zeros((15, 3, len(VALIDATION_SEEDS), 32, len(METRIC_NAMES)), device="cuda:0")
    reward = METRIC_NAMES.index("episode_reward")
    fairness = METRIC_NAMES.index("fairness")
    utilization = METRIC_NAMES.index("utilization")
    critical = METRIC_NAMES.index("critical_outage_locality_hours")
    cube[..., reward] = 23
    cube[..., fairness] = 0.95
    cube[..., utilization] = 0.98
    # Accept the modest reward improvement, reject a larger gain that loses supply use.
    cube[2, ..., reward] = 23.1
    cube[3, ..., reward] = 23.3
    cube[3, ..., utilization] = 0.97
    # A single interrupted condition must disqualify this setting despite its good mean.
    cube[4, ..., reward] = 23.5
    cube[4, 0, 0, 0, critical] = 1
    # Hybrid ties retain the smallest tolerance; zero is a valid fallback.
    choices = select_validation_settings(cube)
    assert choices.tolist() == [1, 0]
    assert REWARD_TOLERANCES[choices[0].item()] == 0.001


def _comparison() -> tuple[dict, dict]:
    intervals = {
        "episode_reward": {"mean_delta": 0.03, "ci95_lower": 0.02, "ci95_upper": 0.04},
        "fairness": {"mean_delta": 0.003, "ci95_lower": 0.001, "ci95_upper": 0.005},
        "utilization": {"mean_delta": 0.0, "ci95_lower": -0.001, "ci95_upper": 0.001},
    }
    by_shortage = {
        level: {"episode_reward": 0.03, "fairness": 0.003, "utilization": 0.0}
        for level in ("mild", "moderate", "severe")
    }
    return intervals, by_shortage


def test_hybrid_success_requires_practical_reward_gain_and_service_guards() -> None:
    intervals, shortages = _comparison()
    assert assess_hybrid_comparison(intervals, shortages, critical_safe=True)["passes"]
    weak = copy.deepcopy(intervals)
    weak["episode_reward"]["ci95_lower"] = 0.005
    assert not assess_hybrid_comparison(weak, shortages, critical_safe=True)["passes"]
    assert not assess_hybrid_comparison(intervals, shortages, critical_safe=False)["passes"]


def test_aggregate_gain_cannot_hide_severe_utilization_loss() -> None:
    intervals, shortages = _comparison()
    shortages["severe"]["utilization"] = -0.003
    result = assess_hybrid_comparison(intervals, shortages, critical_safe=True)
    assert not result["passes"]
    assert not result["gates"]["severity_service_guardrails"]


def test_claim_rejects_missing_shortage_and_nonfinite_threshold_values() -> None:
    intervals, shortages = _comparison()
    shortages.pop("severe")
    with pytest.raises(ValueError, match="shortage"):
        assess_hybrid_comparison(intervals, shortages, critical_safe=True)
    intervals, shortages = _comparison()
    intervals["episode_reward"]["ci95_upper"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        assess_hybrid_comparison(intervals, shortages, critical_safe=True)


@pytest.mark.parametrize("tiny_positive", [False, True])
def test_final_service_ratio_uses_actual_positive_demand_and_zero_demand_convention(
    tiny_positive: bool,
) -> None:
    from gridguard.controllers import FixedRotationController
    from gridguard.domain import Scenario, ShortageLevel
    from gridguard.simulation import simulate

    demand = torch.ones((1, 24, 7, 5), dtype=torch.float64)
    demand[:, :, 0] = 1e-12 if tiny_positive else 0
    scenario = Scenario(demand, 0.8 * demand.sum(dim=(-1, -2)), ShortageLevel.MILD, 0)
    result = simulate(scenario, FixedRotationController())
    ratios = study.final_service_ratios(result)
    expected = torch.tensor(
        [[0.5 if tiny_positive else 1.0, 0.5, 0.5, 0.375, 0.375, 0.375, 0.375]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(ratios, expected)
    assert ratios.device == result.served.device


def _study_payload(phase: str, root: Path) -> dict:
    names = study.VALIDATION_NAMES if phase == "validation" else study.CONTROLLER_NAMES
    seeds = study.VALIDATION_SEEDS if phase == "validation" else study.TEST_SEEDS
    shape = (len(names), 3, len(seeds), study.BATCH_SIZE)
    cube = np.zeros((*shape, len(study.METRIC_NAMES)), dtype=np.float64)
    values = {
        "episode_reward": 23.0,
        "demand_mwh": 100.0,
        "available_mwh": 80.0,
        "served_mwh": 60.0,
        "curtailed_mwh": 40.0,
        "critical_service_ratio": 1.0,
        "utilization": 0.75,
        "fairness": 0.9,
        "switching_rate": 0.1,
    }
    for name, value in values.items():
        cube[..., study.METRIC_NAMES.index(name)] = value
    means = cube.mean(axis=(1, 2, 3))
    shortages = cube.mean(axis=(2, 3))
    offset = 0 if phase == "validation" else 10000
    scenario_hashes = np.array(
        [f"{offset + i + 1:064x}" for i in range(np.prod(shape[1:]))]
    ).reshape(shape[1:])
    payload = {
        "schema": "gridguard.allocation_study.1",
        "phase": phase,
        "protocol": study.PROTOCOL,
        "source_sha256": study.source_hashes(root),
        "controller_names": list(names),
        "seeds": list(seeds),
        "metric_names": list(study.METRIC_NAMES),
        "episode_metric_cube": cube.tolist(),
        "controller_means": {
            name: dict(zip(study.METRIC_NAMES, means[ci].tolist(), strict=True))
            for ci, name in enumerate(names)
        },
        "shortage_means": {
            level: {
                name: dict(zip(study.METRIC_NAMES, shortages[ci, si].tolist(), strict=True))
                for ci, name in enumerate(names)
            }
            for si, level in enumerate(study.SHORTAGES)
        },
        "locality_duration_names": list(study.LOCALITY_METRIC_NAMES),
        "locality_duration_cube": np.zeros((*shape, 7, 5)).tolist(),
        "final_locality_service_ratio_cube": np.full((*shape, 7), 0.6).tolist(),
        "mean_final_minimum_locality_service_ratio": dict.fromkeys(names, 0.6),
        "critical_supply_shortfall_hours": np.zeros(shape[1:]).tolist(),
        "scenario_sha256_cube": scenario_hashes.tolist(),
        "action_sha256_cube": np.full(shape, "a" * 64).tolist(),
        "selected_tolerances": {"Fairness Rule": 0.0, "PPO Hybrid": 0.0},
    }
    if phase == "validation":
        payload["validation_eligible"] = dict.fromkeys(names, True)
        payload["selection_fallback_used"] = {"Fairness Rule": False, "PPO Hybrid": False}
    else:
        payload["paired_inference"] = {
            f"{names[a]} - {names[b]}": {
                name: {"mean_delta": 0.0, "ci95_lower": 0.0, "ci95_upper": 0.0}
                for name in study.METRIC_NAMES
            }
            for a, b in study.PAIR_INDICES
        }
        comparisons = {}
        deltas = {
            level: {"episode_reward": 0.0, "fairness": 0.0, "utilization": 0.0}
            for level in study.SHORTAGES
        }
        for name in ("Constrained Optimization", "Fairness Rule"):
            key = f"PPO Hybrid - {name}"
            comparisons[key] = assess_hybrid_comparison(
                payload["paired_inference"][key], deltas, critical_safe=True
            )
        payload["claims"] = {
            "comparisons": comparisons,
            "hybrid_reward_improvement_with_service_guardrails": False,
        }
    return payload


def _write_payload(payload: dict, path: Path) -> str:
    raw = study.canonical_bytes(payload)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def study_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "science.py").write_text("# frozen science\n", encoding="utf-8")
    monkeypatch.setattr(study, "SOURCE_PATHS", ("science.py",))
    return tmp_path


def _load_written(payload: dict, root: Path) -> dict:
    path = root / f"{payload['phase']}.json"
    digest = _write_payload(payload, path)
    return study.load_study_artifact(
        path, expected_sha256=digest, phase=payload["phase"], root=root
    )


def _bind_validation(payload: dict, root: Path) -> None:
    validation = _study_payload("validation", root)
    path = root / "selection.json"
    payload["selection_path"] = path.name
    payload["selection_sha256"] = _write_payload(validation, path)


def test_complete_evidence_passes_and_final_selection_is_bound(study_root: Path) -> None:
    assert _load_written(_study_payload("validation", study_root), study_root)
    payload = _study_payload("test", study_root)
    _bind_validation(payload, study_root)
    assert _load_written(payload, study_root)
    payload["selected_tolerances"]["PPO Hybrid"] = 0.001
    with pytest.raises(ValueError, match="selection|selected"):
        _load_written(payload, study_root)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("invalid_energy", "energy"),
        ("invalid_ratio", "ratio"),
        ("duration_total", "duration"),
        ("hash_shape", "hash"),
        ("bad_hash", "hash"),
        ("final_ratio", "ratio"),
        ("shortfall", "shortfall"),
        ("eligibility", "eligib"),
        ("fallback", "fallback"),
    ],
)
def test_rehashed_invalid_evidence_is_rejected(study_root: Path, mutation: str, match: str) -> None:
    payload = _study_payload("validation", study_root)
    if mutation == "invalid_energy":
        payload["episode_metric_cube"][0][0][0][0][study.METRIC_NAMES.index("curtailed_mwh")] = 2
    elif mutation == "invalid_ratio":
        payload["episode_metric_cube"][0][0][0][0][study.METRIC_NAMES.index("fairness")] = 1.2
    elif mutation == "duration_total":
        payload["locality_duration_cube"][0][0][0][0][0][0] = 1
    elif mutation == "hash_shape":
        payload["scenario_sha256_cube"].pop()
    elif mutation == "bad_hash":
        payload["action_sha256_cube"][0][0][0][0] = "not-a-sha"
    elif mutation == "final_ratio":
        payload["final_locality_service_ratio_cube"][0][0][0][0][0] = -0.1
    elif mutation == "shortfall":
        payload["critical_supply_shortfall_hours"][0][0][0] = 0.5
    elif mutation == "eligibility":
        payload["validation_eligible"][study.VALIDATION_NAMES[0]] = False
    elif mutation == "fallback":
        payload["selection_fallback_used"]["Fairness Rule"] = True
    with pytest.raises(ValueError, match=match):
        _load_written(payload, study_root)


def test_final_claim_must_reconstruct_and_validation_path_stays_in_root(study_root: Path) -> None:
    payload = _study_payload("test", study_root)
    _bind_validation(payload, study_root)
    payload["claims"]["hybrid_reward_improvement_with_service_guardrails"] = True
    with pytest.raises(ValueError, match="claim"):
        _load_written(payload, study_root)
    payload["claims"]["hybrid_reward_improvement_with_service_guardrails"] = False
    payload["selection_path"] = "../selection.json"
    with pytest.raises(ValueError, match="selection.*path|path.*root"):
        _load_written(payload, study_root)


def test_study_rejects_changed_sources_and_artifact_bytes(study_root: Path) -> None:
    payload = _study_payload("validation", study_root)
    path = study_root / "validation.json"
    digest = _write_payload(payload, path)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA-256"):
        study.load_study_artifact(path, expected_sha256=digest, phase="validation", root=study_root)
    _write_payload(payload, path)
    (study_root / "science.py").write_text("# changed source\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source"):
        study.load_study_artifact(path, expected_sha256=digest, phase="validation", root=study_root)


def test_zero_tolerance_action_identity_is_checked(study_root: Path) -> None:
    payload = _study_payload("validation", study_root)
    payload["action_sha256_cube"][1][0][0][0] = "b" * 64
    with pytest.raises(ValueError, match="zero tolerance"):
        _load_written(payload, study_root)


@pytest.mark.gpu
def test_cuda_interval_reconstruction_rejects_forged_endpoints(study_root: Path) -> None:
    payload = _study_payload("test", study_root)
    # Constant identical rows have a degenerate zero interval under every paired resample.
    study.verify_gpu_intervals(payload)
    payload["paired_inference"]["PPO Hybrid - Fairness Rule"]["fairness"]["ci95_upper"] = 0.2
    with pytest.raises(ValueError, match="GPU bootstrap interval"):
        study.verify_gpu_intervals(payload)

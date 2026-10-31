from __future__ import annotations

import hashlib
from dataclasses import asdict

import numpy as np
import pytest
import torch

from gridguard.allocation_study import METRIC_NAMES, SHORTAGES
from gridguard.focused_training import CONFIG, TRAINING_SEEDS, select_checkpoint, write_json
from gridguard.ppo_training import PPOTrainingConfig
from tools import evaluate_focused_ppo as evaluation


def _summary(*, fixed: bool = False) -> dict:
    metrics = dict.fromkeys(METRIC_NAMES, 0.0)
    metrics.update(
        episode_reward=14 if fixed else 23.5,
        fairness=0.87 if fixed else 0.96,
        utilization=0.71 if fixed else 0.974,
    )
    if fixed:
        metrics.update(critical_outage_locality_hours=102, critical_interruption_locality_hours=102)
    return {
        "overall": metrics.copy(),
        "by_shortage": {level: metrics.copy() for level in SHORTAGES},
        "maxima": metrics.copy(),
    }


@pytest.fixture
def training_record(tmp_path, monkeypatch):
    bound = {"source.py": "a" * 64}
    monkeypatch.setattr(evaluation, "source_hashes", lambda _: bound.copy())
    reference = _summary(fixed=True)
    runs = []
    for seed in TRAINING_SEEDS:
        checkpoints = [
            {"iteration": iteration, "summary": _summary()} for iteration in range(0, 3001, 100)
        ]
        checkpoints[-1]["summary"]["overall"]["episode_reward"] += 0.1
        run = {
            "seed": seed,
            "checkpoints": checkpoints,
            "selected": select_checkpoint(checkpoints, reference),
            "seconds": 600.0,
            "transitions": 18_432_000,
        }
        for field in ("model", "iteration600_model", "behavior_cloning_model"):
            path = tmp_path / f"{seed}-{field}.npz"
            path.write_bytes(f"numeric model fixture {seed} {field}".encode())
            run[field] = {
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        runs.append(run)
    record = {
        "schema": "gridguard.focused_ppo_training.1",
        "config": CONFIG,
        "ppo_config": asdict(PPOTrainingConfig()),
        "source_sha256": bound,
        "runs": runs,
        "reference_validation": reference,
        "deployment_seed": min(TRAINING_SEEDS),
    }
    freeze_path = tmp_path / "protocol-freeze.json"
    freeze_hash = write_json(
        freeze_path,
        {
            "schema": "gridguard.focused_ppo_freeze.1",
            "frozen_at_utc": "2026-10-24T12:00:00+00:00",
            **{key: record[key] for key in ("config", "ppo_config", "source_sha256")},
        },
    )
    record["protocol_freeze"] = {"path": freeze_path.name, "sha256": freeze_hash}
    return record, tmp_path


def _publish(record, root):
    path = root / "training.json"
    return path, write_json(path, record)


def test_training_contract_accepts_unprotected_fixed_reference(training_record) -> None:
    record, root = training_record
    path, digest = _publish(record, root)
    assert evaluation.load_training_record(path, digest, root=root) == record


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda r: r["runs"].pop(), "all five"),
        (lambda r: r["runs"][0].update(transitions=3_686_400), "transition budget"),
        (lambda r: r["runs"][0]["checkpoints"].pop(), "checkpoint sequence"),
        (lambda r: r["runs"][0]["selected"].update(iteration=600), "selection"),
        (lambda r: r.update(source_sha256={}), "source hashes"),
        (lambda r: r.update(config={}), "configuration"),
        (lambda r: r.pop("protocol_freeze"), "pretraining protocol freeze"),
        (lambda r: r["runs"][0].pop("iteration600_model"), "model provenance"),
        (lambda r: r["runs"][0]["model"].update(path="../outside.npz"), "escapes"),
    ],
)
def test_training_contract_tampering_fails_closed(training_record, mutation, message) -> None:
    record, root = training_record
    mutation(record)
    path, digest = _publish(record, root)
    with pytest.raises(ValueError, match=message):
        evaluation.load_training_record(path, digest, root=root)


def test_changed_reference_model_bytes_fail_before_test_creation(training_record) -> None:
    record, root = training_record
    path, digest = _publish(record, root)
    (root / record["runs"][0]["iteration600_model"]["path"]).write_bytes(b"altered")
    with pytest.raises(ValueError, match="model SHA-256"):
        evaluation.load_training_record(path, digest, root=root)


def _intervals() -> dict:
    return {
        "critical_interruption_locality_hours": {"mean": -100.0, "lower": -102.0, "upper": -98.0},
        "episode_reward": {"mean": 0.03, "lower": 0.01, "upper": 0.05},
        "fairness": {"mean": 0.09, "lower": 0.08, "upper": 0.10},
        "utilization": {"mean": 0.26, "lower": 0.25, "upper": 0.27},
    }


def _shortages() -> dict:
    return {level: {key: row["mean"] for key, row in _intervals().items()} for level in SHORTAGES}


def test_primary_claim_uses_service_outcomes_not_reward() -> None:
    intervals = _intervals()
    intervals["episode_reward"] = {"mean": -10.0, "lower": -11.0, "upper": -9.0}
    decision = evaluation.assess_fixed_comparison(
        intervals, _shortages(), all_safe=True, all_eligible=True
    )
    assert decision["passed"]


@pytest.mark.parametrize(
    ("metric", "endpoint"),
    [
        ("critical_interruption_locality_hours", "upper"),
        ("fairness", "lower"),
        ("utilization", "lower"),
    ],
)
def test_primary_claim_observes_strict_interval_boundaries(metric, endpoint) -> None:
    intervals = _intervals()
    intervals[metric][endpoint] = 0
    assert not evaluation.assess_fixed_comparison(
        intervals, _shortages(), all_safe=True, all_eligible=True
    )["passed"]


def test_primary_claim_cannot_hide_harm_in_one_shortage() -> None:
    rows = _shortages()
    rows["severe"]["fairness"] = -0.001
    assert not evaluation.assess_fixed_comparison(
        _intervals(), rows, all_safe=True, all_eligible=True
    )["passed"]
    for safe, eligible in ((False, True), (True, False)):
        assert not evaluation.assess_fixed_comparison(
            _intervals(), _shortages(), all_safe=safe, all_eligible=eligible
        )["passed"]


def test_more_training_requires_later_selection_and_preserves_service_margin() -> None:
    kwargs = {"all_safe": True, "all_eligible": True, "all_selected_after_600": True}
    assert evaluation.assess_budget_comparison(_intervals(), _shortages(), **kwargs)["passed"]
    kwargs["all_selected_after_600"] = False
    assert not evaluation.assess_budget_comparison(_intervals(), _shortages(), **kwargs)["passed"]
    kwargs["all_selected_after_600"] = True
    intervals = _intervals()
    intervals["utilization"]["lower"] = -0.0021
    assert not evaluation.assess_budget_comparison(intervals, _shortages(), **kwargs)["passed"]


def test_claims_reject_missing_or_nonfinite_values() -> None:
    intervals = _intervals()
    intervals["fairness"]["mean"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        evaluation.assess_fixed_comparison(
            intervals, _shortages(), all_safe=True, all_eligible=True
        )
    rows = _shortages()
    del rows["mild"]
    with pytest.raises(ValueError, match="shortage"):
        evaluation.assess_fixed_comparison(_intervals(), rows, all_safe=True, all_eligible=True)


def test_summary_claims_exclude_fixed_safety_but_include_each_ppo_run(training_record) -> None:
    training, _ = training_record
    count = len(METRIC_NAMES)
    intervals = np.zeros((3, 3, count))
    for metric, values in _intervals().items():
        intervals[:, :, METRIC_NAMES.index(metric)] = [values[k] for k in evaluation.INTERVAL_NAMES]
    deltas = np.zeros((3, 3, count))
    for metric, values in _intervals().items():
        deltas[:, :, METRIC_NAMES.index(metric)] = values["mean"]
    blocks = {
        "hierarchical": intervals,
        "primary_adjusted": intervals[:2].copy(),
        "shortage_intervals": np.repeat(intervals[:, None], 3, axis=1),
        "conditional": np.repeat(intervals[:, None], 5, axis=1),
        "shortage_deltas": deltas,
        "maxima": np.zeros((12, count)),
        "means": np.zeros((12, count)),
        "shortage_means": np.zeros((12, 3, count)),
        "procedure_means": np.zeros((2, count)),
        "procedure_shortage_means": np.zeros((2, 3, count)),
        "minimum_service_means": np.zeros(12),
    }
    critical = METRIC_NAMES.index("critical_interruption_locality_hours")
    blocks["maxima"][0, critical] = 102
    summary = evaluation._summary_fields(blocks, training)
    assert all(claim["passed"] for claim in summary["claims"].values())
    reward = METRIC_NAMES.index("episode_reward")
    blocks["primary_adjusted"][1, 1, reward] = 0
    summary = evaluation._summary_fields(blocks, training)
    assert not summary["claims"]["PPO - Constrained Optimization"]["passed"]
    blocks["primary_adjusted"][1, 1, reward] = 0.01
    blocks["maxima"][4, critical] = 1
    summary = evaluation._summary_fields(blocks, training)
    assert not any(claim["passed"] for claim in summary["claims"].values())
    blocks["maxima"][4, critical] = 0
    training["runs"][3]["selected"]["iteration"] = 600
    summary = evaluation._summary_fields(blocks, training)
    assert summary["claims"]["PPO - Fixed Rotation"]["passed"]
    assert not summary["claims"]["PPO - PPO at600"]["passed"]


def test_physics_allows_fixed_interruptions_without_using_pooled_ratio_in_reward() -> None:
    cube = torch.zeros((12, 3, 1, 2, len(METRIC_NAMES)), dtype=torch.float64)
    columns = {name: index for index, name in enumerate(METRIC_NAMES)}
    for name, value in {
        "demand_mwh": 100,
        "available_mwh": 95,
        "served_mwh": 90,
        "curtailed_mwh": 10,
        "critical_service_ratio": 1,
        "fairness": 0.95,
        "utilization": 0.9,
        "switching_rate": 0.2,
        "episode_reward": 24 * (5 + 2 * 0.9 + 2 * 0.95 - 0.25 * 0.2) / 9,
    }.items():
        cube[..., columns[name]] = value
    cube[0, ..., columns["critical_service_ratio"]] = 0.45
    cube[0, ..., columns["critical_outage_locality_hours"]] = 100
    cube[0, ..., columns["critical_interruption_locality_hours"]] = 100
    cube[0, ..., columns["episode_reward"]] = 14
    services = torch.full((12, 3, 1, 2, 7), 0.9)
    shortfalls = torch.zeros((3, 1, 2))
    evaluation._validate_physics(cube, services, shortfalls)
    altered = cube.clone()
    altered[1, ..., columns["critical_service_ratio"]] = 0.99
    with pytest.raises(ValueError, match="protect all critical"):
        evaluation._validate_physics(altered, services, shortfalls)
    altered = cube.clone()
    altered[1, ..., columns["episode_reward"]] = 24
    with pytest.raises(ValueError, match="reward does not reconstruct"):
        evaluation._validate_physics(altered, services, shortfalls)


@pytest.mark.gpu
def test_gpu_summary_preserves_run_pairing_and_fixed_reference_layout() -> None:
    cube = torch.zeros((12, 3, 2, 4, len(METRIC_NAMES)), device="cuda:0", dtype=torch.float64)
    cube[1] = 9.5
    for run in range(5):
        cube[run + 2] = 10 + run
        cube[run + 7] = 9 + run
    services = torch.ones((12, 3, 2, 4, 7), device="cuda:0", dtype=torch.float64)
    summary = evaluation._scientific_summary(cube, services)
    assert summary["procedure_means"].shape == (2, len(METRIC_NAMES))
    assert summary["hierarchical"].shape == (3, 3, len(METRIC_NAMES))
    assert summary["conditional"].shape == (3, 5, 3, len(METRIC_NAMES))
    assert summary["primary_adjusted"].shape == (2, 3, len(METRIC_NAMES))
    assert summary["shortage_intervals"].shape == (3, 3, 3, len(METRIC_NAMES))
    assert torch.all(summary["primary_adjusted"][:, 1] <= summary["hierarchical"][:2, 1])
    assert torch.all(summary["primary_adjusted"][:, 2] >= summary["hierarchical"][:2, 2])
    torch.testing.assert_close(
        summary["hierarchical"][2], torch.ones_like(summary["hierarchical"][2]), rtol=0, atol=0
    )
    torch.testing.assert_close(
        summary["shortage_deltas"][2],
        torch.ones_like(summary["shortage_deltas"][2]),
        rtol=0,
        atol=0,
    )

from __future__ import annotations

import copy
import hashlib
from dataclasses import asdict

import numpy as np
import pytest
import torch

from gridguard.allocation_study import METRIC_NAMES, SHORTAGES
from gridguard.ppo_experiment import CONFIG, TRAINING_SEEDS, write_json
from gridguard.ppo_study import select_checkpoint
from gridguard.ppo_training import PPOTrainingConfig
from tools import evaluate_constrained_ppo as evaluation


def _summary() -> dict:
    metrics = dict.fromkeys(METRIC_NAMES, 0.0)
    metrics.update(episode_reward=23.5, fairness=0.96, utilization=0.974)
    return {
        "overall": metrics.copy(),
        "by_shortage": {level: metrics.copy() for level in SHORTAGES},
        "maxima": metrics.copy(),
    }


@pytest.fixture
def training_record(tmp_path, monkeypatch):
    bound = {"source.py": "a" * 64}
    monkeypatch.setattr(evaluation, "source_hashes", lambda _: bound.copy())
    reference = _summary()
    runs = []
    for seed in TRAINING_SEEDS:
        checkpoints = [
            {"iteration": iteration, "summary": copy.deepcopy(reference)}
            for iteration in range(0, 601, 50)
        ]
        run = {
            "seed": seed,
            "checkpoints": checkpoints,
            "selected": select_checkpoint(checkpoints, reference),
            "seconds": 120.0,
            "transitions": 3_686_400,
        }
        for field in ("model", "behavior_cloning_model"):
            path = tmp_path / f"{seed}-{field}.npz"
            path.write_bytes(f"numeric model fixture {seed} {field}".encode())
            run[field] = {
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        runs.append(run)
    record = {
        "schema": "gridguard.constrained_ppo_training.1",
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
            "schema": "gridguard.constrained_ppo_freeze.1",
            "frozen_at_utc": "2026-10-23T12:00:00+00:00",
            **{key: record[key] for key in ("config", "ppo_config", "source_sha256")},
        },
    )
    record["protocol_freeze"] = {"path": freeze_path.name, "sha256": freeze_hash}
    return record, tmp_path


def _publish(record, root):
    path = root / "training.json"
    return path, write_json(path, record)


def test_frozen_training_record_is_validated_before_any_test_scenario(training_record) -> None:
    record, root = training_record
    path, digest = _publish(record, root)
    assert evaluation.load_training_record(path, digest, root=root) == record


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda r: r["runs"].pop(), "all five"),
        (lambda r: r["runs"][0].update(transitions=1), "transition budget"),
        (lambda r: r["runs"][0]["checkpoints"].pop(), "checkpoint sequence"),
        (lambda r: r["runs"][0]["selected"].update(iteration=50), "selection"),
        (lambda r: r.update(deployment_seed=TRAINING_SEEDS[-1]), "deployment seed"),
        (lambda r: r.update(source_sha256={}), "source hashes"),
        (lambda r: r.update(config={}), "configuration"),
        (lambda r: r["runs"][0]["model"].update(path="../outside.npz"), "escapes"),
        (lambda r: r["runs"][0].update(seconds=-1), "runtime"),
        (lambda r: r.pop("protocol_freeze"), "pretraining protocol freeze"),
    ],
)
def test_modified_training_contract_fails_closed(training_record, mutation, message) -> None:
    record, root = training_record
    mutation(record)
    path, digest = _publish(record, root)
    with pytest.raises(ValueError, match=message):
        evaluation.load_training_record(path, digest, root=root)


def test_altered_model_bytes_fail_before_final_evaluation(training_record) -> None:
    record, root = training_record
    path, digest = _publish(record, root)
    (root / record["runs"][0]["model"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="model SHA-256"):
        evaluation.load_training_record(path, digest, root=root)


def test_wrong_training_digest_fails_closed(training_record) -> None:
    record, root = training_record
    path, _ = _publish(record, root)
    with pytest.raises(ValueError, match="SHA-256"):
        evaluation.load_training_record(path, "0" * 64, root=root)


def test_duplicate_model_artifact_path_is_rejected(training_record) -> None:
    record, root = training_record
    record["runs"][1]["model"] = record["runs"][0]["model"].copy()
    path, digest = _publish(record, root)
    with pytest.raises(ValueError, match="separate model artifacts"):
        evaluation.load_training_record(path, digest, root=root)


def test_interval_verification_rejects_changed_values_and_extra_claims() -> None:
    expected = {"interval": {"mean": 0.01, "lower": -0.03, "upper": 0.04}, "passed": False}
    evaluation._compare_nested(copy.deepcopy(expected), expected, "claims")
    changed = copy.deepcopy(expected)
    changed["interval"]["lower"] = 0.03
    with pytest.raises(ValueError, match="does not reconstruct"):
        evaluation._compare_nested(changed, expected, "claims")
    changed = copy.deepcopy(expected)
    changed["passed"] = True
    with pytest.raises(ValueError, match="does not match"):
        evaluation._compare_nested(changed, expected, "claims")
    changed = copy.deepcopy(expected)
    changed["unexpected_win"] = True
    with pytest.raises(ValueError, match="fields do not match"):
        evaluation._compare_nested(changed, expected, "claims")


def test_matching_hashes_require_complete_dimensions_and_valid_sha256() -> None:
    hashes = [["1" * 64, "2" * 64], ["3" * 64, "4" * 64]]
    evaluation._hash_cube(hashes, (2, 2))
    with pytest.raises(ValueError, match="dimensions"):
        evaluation._hash_cube(hashes, (2, 3))
    hashes[0][0] = "not a digest"
    with pytest.raises(ValueError, match="SHA-256"):
        evaluation._hash_cube(hashes, (2, 2))


def test_report_claim_cannot_hide_an_ineligible_training_seed(training_record) -> None:
    training, _ = training_record
    metric_count = len(METRIC_NAMES)
    intervals = np.zeros((2, 3, metric_count))
    intervals[:, :, METRIC_NAMES.index("episode_reward")] = [0.02, 0.015, 0.03]
    deltas = np.zeros((2, 3, metric_count))
    deltas[:, :, METRIC_NAMES.index("episode_reward")] = 0.02
    blocks = {
        "hierarchical": intervals,
        "conditional": np.repeat(intervals[:, None], 5, axis=1),
        "shortage_deltas": deltas,
        "maxima": np.zeros((12, metric_count)),
        "means": np.zeros((12, metric_count)),
        "shortage_means": np.zeros((12, 3, metric_count)),
        "procedure_means": np.zeros((2, metric_count)),
        "procedure_shortage_means": np.zeros((2, 3, metric_count)),
        "minimum_service_means": np.zeros(12),
    }
    fields = evaluation._summary_fields(blocks, training)
    assert all(item["passed"] for item in fields["claims"].values())
    training["runs"][3]["selected"]["eligible"] = False
    fields = evaluation._summary_fields(blocks, training)
    assert not any(item["passed"] for item in fields["claims"].values())


def test_physical_verifier_rejects_changed_reward_energy_and_ratios() -> None:
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
    services = torch.full((12, 3, 1, 2, 7), 0.9)
    shortfalls = torch.zeros((3, 1, 2))
    evaluation._validate_physics(cube, services, shortfalls)
    for name, value, message in (
        ("episode_reward", 24, "reward does not reconstruct"),
        ("curtailed_mwh", 11, "conservation"),
        ("fairness", 1.1, "metric ratios"),
        ("critical_service_ratio", 0.99, "protect all critical"),
        ("critical_interruption_locality_hours", 1, "definitions do not agree"),
    ):
        altered = cube.clone()
        altered[..., columns[name]] = value
        with pytest.raises(ValueError, match=message):
            evaluation._validate_physics(altered, services, shortfalls)

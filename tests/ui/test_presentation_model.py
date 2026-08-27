from __future__ import annotations

import json
from pathlib import Path

import pytest

from gridguard import __version__
from gridguard.robustness import (
    ROBUSTNESS_ARTIFACT_ID,
    ROBUSTNESS_LEGACY_ARTIFACT_ID,
    ROBUSTNESS_SUPPLEMENT_SHA256,
    RobustnessArtifactError,
)
from gridguard.ui.presentation_model import (
    CANONICAL_TRAINING_RECORD,
    HistoricalTrainingRecord,
    PresentationDataError,
    PresentationSnapshot,
    load_presentation_snapshot,
)
from tools.training_provenance import build_training_summary, load_training_provenance

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TRAINING_PROVENANCE_PATH = REPOSITORY_ROOT / "data" / "model" / "training-provenance.yaml"
MODEL_MANIFEST_PATH = REPOSITORY_ROOT / "data" / "model" / "manifest.json"


def _interval(
    mean: float,
    lower: float,
    upper: float,
    *,
    classification: str = "favorable",
) -> dict[str, object]:
    return {
        "mean_delta": mean,
        "ci95_lower": lower,
        "ci95_upper": upper,
        "result_classification": classification,
    }


def _evidence_payload() -> dict[str, object]:
    return {
        "schema": {"name": "gridguard.robustness_supplement", "version": 1},
        "evidence_role": "untouched_confirmatory",
        "provenance": {
            "artifact_id": "gridguard-robustness-supplement-v1",
            "source_revision": "a" * 40,
            "model_manifest": {
                "path": "data/model/manifest.json",
                "sha256": "b" * 64,
                "size_bytes": 100,
            },
            "actor": {
                "path": "data/model/ppo-policy.npz",
                "sha256": "c" * 64,
                "size_bytes": 200,
            },
            "scenario_generator": {
                "path": "src/gridguard/scenarios.py",
                "sha256": "d" * 64,
                "size_bytes": 300,
            },
            "training_provenance": {
                "path": "data/model/training-provenance.yaml",
                "sha256": "e" * 64,
                "size_bytes": 400,
            },
            "implementation_sources": [],
            "seed_separation": {"disjoint_from_all_documented_seeds": True},
        },
        "protocol": {
            "seeds": list(range(81001, 81011)),
            "episodes_per_seed": 32,
            "independent_base_draw_count": 320,
            "shortage_levels": ["mild", "moderate", "severe"],
            "matched_scenario_count": 960,
            "controllers": ["Fixed Rotation", "PPO"],
            "controller_evaluation_count": 1_920,
            "delta_definition": "PPO - Fixed Rotation",
            "bootstrap": {"replicates": 10_000, "seed": 20260826, "ci_percent": 95},
            "sensitivity_parameterization_count": 25,
            "sensitivity_matched_scenario_count": 24_000,
            "sensitivity_controller_evaluation_count": 48_000,
        },
        "metric_definitions": {
            "critical_outage_locality_hours": {
                "unit": "locality-hours",
                "direction": "lower_is_better",
                "role": "primary",
            },
            "fairness": {
                "unit": "ratio",
                "direction": "higher_is_better",
                "role": "primary",
            },
            "utilization": {
                "unit": "ratio",
                "direction": "higher_is_better",
                "role": "primary",
            },
            "power_balance_violation_rate": {
                "unit": "ratio",
                "direction": "lower_is_better",
                "role": "safety",
            },
        },
        "forecast_metric_definitions": {},
        "base_evaluation": {
            "controller_means": {
                "Fixed Rotation": {
                    "critical_outage_locality_hours": 102.0,
                    "fairness": 0.88,
                    "utilization": 0.71,
                    "power_balance_violation_rate": 0.0,
                },
                "PPO": {
                    "critical_outage_locality_hours": 18.7,
                    "fairness": 0.97,
                    "utilization": 0.98,
                    "power_balance_violation_rate": 0.0,
                },
            },
            "paired_inference": {
                "critical_outage_locality_hours": _interval(-83.3, -84.4, -82.2),
                "fairness": _interval(0.09, 0.08, 0.10),
                "utilization": _interval(0.27, 0.26, 0.28),
            },
            "power_balance": {"maximum_violation_rate": 0.0},
        },
        "sensitivity": {},
        "forecast_evaluation": {},
        "limitations": [
            "Synthetic demand traces do not establish real-world forecast accuracy.",
            "Observed interval direction is descriptive and is not a release gate.",
        ],
    }


def test_historical_training_record_matches_strict_canonical_yaml_and_model_summary() -> None:
    payload = load_training_provenance(
        TRAINING_PROVENANCE_PATH,
        model_manifest_path=MODEL_MANIFEST_PATH,
    )
    summary = build_training_summary(payload, provenance_path=TRAINING_PROVENANCE_PATH)
    model_manifest = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))

    projected = HistoricalTrainingRecord.from_validated_payload(
        payload,
        model_manifest=model_manifest,
        provenance_path=summary["training_provenance"]["path"],
    )

    assert projected == CANONICAL_TRAINING_RECORD
    assert projected.provenance_path == "data/model/training-provenance.yaml"
    assert projected.actor_topology == "32 -> 512 -> 512 -> 2,187 logits"
    assert projected.behavior_cloning_teacher_samples == 32_768
    assert projected.behavior_cloning_rollout_batch_size == 128
    assert projected.behavior_cloning_minibatch_size == 2_048
    assert projected.behavior_cloning_epochs == 100
    assert projected.behavior_cloning_optimizer_updates == 1_600
    assert projected.ppo_iterations == 600
    assert projected.ppo_total_episodes == 153_600
    assert projected.ppo_total_transitions == 3_686_400
    assert projected.candidate_experiment_iterations == 1_800
    assert projected.candidate_experiment_episodes == 460_800
    assert projected.candidate_experiment_transitions == 11_059_200
    assert projected.ppo_episodes_per_iteration == 256
    assert projected.ppo_minibatch_size == 2_048
    assert projected.ppo_epochs_per_iteration == 4
    assert projected.learning_rate == pytest.approx(0.0001)
    assert projected.gamma == pytest.approx(0.99)
    assert projected.gae_lambda == pytest.approx(0.95)
    assert projected.clip_epsilon == pytest.approx(0.2)
    assert projected.training_seed == 43
    assert projected.candidate_training_seeds == (42, 43, 44)
    assert projected.validation_seeds == (70_000, 70_001, 70_002)
    assert projected.selected_seed == 43
    assert projected.selected_iteration == 510


def test_snapshot_exposes_only_the_three_documented_holdout_headlines() -> None:
    snapshot = PresentationSnapshot.from_mapping(_evidence_payload())

    assert snapshot.controller_names == ("PPO", "Fixed Rotation")
    assert tuple(metric.key for metric in snapshot.metrics) == (
        "critical_outage_locality_hours",
        "fairness",
        "utilization",
    )
    assert snapshot.metric("critical_outage_locality_hours").ppo == pytest.approx(18.7)
    assert snapshot.metric("critical_outage_locality_hours").fixed_rotation == pytest.approx(102.0)
    assert snapshot.metric("fairness").mean_delta == pytest.approx(0.09)
    with pytest.raises(KeyError):
        snapshot.metric("episode_reward")

    assert snapshot.artifact_id == ROBUSTNESS_LEGACY_ARTIFACT_ID
    assert snapshot.benchmark_id == ROBUSTNESS_LEGACY_ARTIFACT_ID
    assert snapshot.evidence_role == "untouched_confirmatory"
    assert snapshot.independent_base_draw_count == 320
    assert snapshot.matched_scenario_count == 960
    assert snapshot.controller_evaluation_count == 1_920
    assert snapshot.maximum_power_balance_violation_rate == 0.0
    assert snapshot.artifact_sha256 == ROBUSTNESS_SUPPLEMENT_SHA256
    assert snapshot.release_id == f"gridguard-khartoum-v{__version__}"
    assert snapshot.count_statement == (
        "320 documented-seed-disjoint independent base draws × 3 shortage levels = "
        "960 matched scenario "
        "conditions · 1,920 controller evaluations"
    )


def test_snapshot_accepts_verified_schema_two_artifact_identity() -> None:
    payload = _evidence_payload()
    payload["schema"]["version"] = 2
    payload["evidence_role"] = "documented_seed_disjoint_holdout"
    payload["provenance"]["artifact_id"] = ROBUSTNESS_ARTIFACT_ID

    snapshot = PresentationSnapshot.from_mapping(payload)

    assert snapshot.artifact_id == ROBUSTNESS_ARTIFACT_ID


def test_snapshot_accepts_percentile_interval_not_containing_point_estimate() -> None:
    payload = _evidence_payload()
    payload["base_evaluation"]["paired_inference"]["utilization"].update(
        {
            "mean_delta": 0.25,
            "ci95_lower": 0.26,
            "ci95_upper": 0.28,
        }
    )

    snapshot = PresentationSnapshot.from_mapping(payload)

    utilization = snapshot.metric("utilization")
    assert utilization.mean_delta == pytest.approx(0.25)
    assert utilization.ci95_lower == pytest.approx(0.26)
    assert utilization.ci95_upper == pytest.approx(0.28)


def test_snapshot_rejects_reversed_percentile_interval() -> None:
    payload = _evidence_payload()
    payload["base_evaluation"]["paired_inference"]["utilization"].update(
        {
            "ci95_lower": 0.29,
            "ci95_upper": 0.28,
        }
    )

    with pytest.raises(PresentationDataError, match="ci95_lower.*ci95_upper"):
        PresentationSnapshot.from_mapping(payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload["schema"].update({"version": 3}),
            "robustness supplement version 1 or 2",
        ),
        (
            lambda payload: payload["provenance"].update({"artifact_id": "untrusted"}),
            "artifact_id",
        ),
        (
            lambda payload: payload["protocol"].update({"matched_scenario_count": 959}),
            "960",
        ),
        (
            lambda payload: payload["base_evaluation"]["controller_means"]["PPO"].update(
                {"episode_reward": 23.0}
            ),
            "headline metrics",
        ),
        (
            lambda payload: payload["base_evaluation"]["paired_inference"]["utilization"].update(
                {"mean_delta": float("nan")}
            ),
            "finite",
        ),
        (
            lambda payload: payload["base_evaluation"].pop("power_balance"),
            "base_evaluation",
        ),
    ],
)
def test_snapshot_rejects_untrusted_or_inconsistent_scientific_values(
    mutation,
    message: str,
) -> None:
    payload = _evidence_payload()
    mutation(payload)

    with pytest.raises(PresentationDataError, match=message):
        PresentationSnapshot.from_mapping(payload)


def test_file_loader_converts_verification_failure_to_presentation_error(tmp_path) -> None:
    def reject(*args, **kwargs):
        raise RobustnessArtifactError("duplicate JSON field: schema")

    with pytest.raises(PresentationDataError, match="duplicate JSON field"):
        load_presentation_snapshot(tmp_path / "robustness.json", verifier=reject)


def test_file_loader_uses_trusted_robustness_verifier_without_digest_override(tmp_path) -> None:
    calls: list[object] = []

    def verify(path):
        calls.append(path)
        return _evidence_payload()

    evidence = tmp_path / "robustness.json"
    snapshot = load_presentation_snapshot(evidence, verifier=verify)

    assert snapshot.artifact_id == ROBUSTNESS_LEGACY_ARTIFACT_ID
    assert snapshot.metric("utilization").ppo == pytest.approx(0.98)
    assert calls == [evidence]


def test_committed_robustness_artifact_is_ready_for_the_presentation() -> None:
    snapshot = load_presentation_snapshot()

    assert snapshot.artifact_id == ROBUSTNESS_ARTIFACT_ID
    assert snapshot.independent_base_draw_count == 320
    assert snapshot.matched_scenario_count == 960
    assert snapshot.controller_evaluation_count == 1_920
    assert tuple(metric.key for metric in snapshot.metrics) == (
        "critical_outage_locality_hours",
        "fairness",
        "utilization",
    )

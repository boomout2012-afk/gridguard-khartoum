from __future__ import annotations

import pytest

from gridguard import __version__
from gridguard.evaluation import PAIRED_BENCHMARK_SHA256, EvaluationArtifactError
from gridguard.ui.presentation_model import (
    PresentationDataError,
    PresentationSnapshot,
    load_presentation_snapshot,
)


def _evidence_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "benchmark_id": "proposal-ppo-vs-fixed-rotation-paired",
        "protocol": {
            "controllers": ["Fixed Rotation", "PPO"],
            "count_statement": (
                "288 matched scenarios; 576 controller-scenario evaluations (288 per controller)."
            ),
            "matched_scenario_count": 288,
            "independent_base_draw_count": 96,
            "controller_evaluation_count": 576,
            "horizon_hours": 24,
            "shortage_levels": ["mild", "moderate", "severe"],
        },
        "aggregate": {
            "Fixed Rotation": {
                "episode_reward": 13.0,
                "critical_service_ratio": 0.42,
                "power_balance_violation_rate": 0.0,
                "served_mwh": 4500.0,
            },
            "PPO": {
                "episode_reward": 23.0,
                "critical_service_ratio": 0.99,
                "power_balance_violation_rate": 0.0,
                "served_mwh": 6500.0,
            },
        },
        "paired_inference": {
            "episode_reward": {
                "mean_delta": 10.0,
                "ci95_lower": 9.9,
                "ci95_upper": 10.1,
                "direction": "higher_is_better",
                "role": "primary",
                "delta_definition": "PPO - Fixed Rotation",
            },
            "critical_service_ratio": {
                "mean_delta": 0.57,
                "ci95_lower": 0.56,
                "ci95_upper": 0.58,
                "direction": "higher_is_better",
                "role": "primary",
                "delta_definition": "PPO - Fixed Rotation",
            },
            "served_mwh": {
                "mean_delta": 2000.0,
                "ci95_lower": 1980.0,
                "ci95_upper": 2020.0,
                "direction": "higher_is_better",
                "role": "primary",
                "delta_definition": "PPO - Fixed Rotation",
            },
        },
        "limitations": [
            "Synthetic simulation evidence; not real utility measurements.",
        ],
    }


def test_snapshot_preserves_verified_controller_names_and_real_values() -> None:
    snapshot = PresentationSnapshot.from_mapping(_evidence_payload())

    assert snapshot.controller_names == ("PPO", "Fixed Rotation")
    assert snapshot.metrics[0].key == "critical_service_ratio"
    assert snapshot.metrics[0].ppo == pytest.approx(0.99)
    assert snapshot.metrics[0].fixed_rotation == pytest.approx(0.42)
    assert snapshot.metrics[0].mean_delta == pytest.approx(0.57)
    assert snapshot.matched_scenario_count == 288
    assert snapshot.independent_base_draw_count == 96
    assert snapshot.controller_evaluation_count == 576
    assert snapshot.ppo_power_balance_violation_rate == 0.0
    assert snapshot.fixed_rotation_power_balance_violation_rate == 0.0
    assert snapshot.artifact_sha256 == PAIRED_BENCHMARK_SHA256
    assert snapshot.release_id == f"gridguard-khartoum-v{__version__}"
    assert snapshot.count_statement == (
        "96 independent base draws × 3 shortage levels = 288 matched scenario conditions "
        "· 576 controller evaluations"
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload["aggregate"].pop("PPO"), "exactly PPO and Fixed Rotation"),
        (
            lambda payload: payload["protocol"].update(
                {"controllers": ["PPO", "Fixed Rotation", "Extra"]}
            ),
            "exactly PPO and Fixed Rotation",
        ),
        (
            lambda payload: payload["paired_inference"]["served_mwh"].update(
                {"mean_delta": float("nan")}
            ),
            "finite",
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
        raise EvaluationArtifactError("duplicate JSON field: schema_version")

    with pytest.raises(PresentationDataError, match="duplicate JSON field"):
        load_presentation_snapshot(tmp_path / "paired.json", verifier=reject)


def test_file_loader_uses_pinned_semantic_verifier(tmp_path) -> None:
    calls: list[tuple[object, object, object]] = []

    def verify(path, *, expected_sha256, frozen_benchmark_path):
        calls.append((path, expected_sha256, frozen_benchmark_path))
        return _evidence_payload()

    evidence = tmp_path / "paired.json"
    snapshot = load_presentation_snapshot(evidence, verifier=verify)

    assert snapshot.benchmark_id == "proposal-ppo-vs-fixed-rotation-paired"
    assert snapshot.metric("episode_reward").ppo == pytest.approx(23.0)
    assert calls[0][0] == evidence
    assert calls[0][1] == PAIRED_BENCHMARK_SHA256
    assert str(calls[0][2]).endswith("data\\evidence\\frozen-benchmark.json")


def test_committed_paired_artifact_is_ready_for_the_presentation() -> None:
    snapshot = load_presentation_snapshot()

    assert snapshot.controller_names == ("PPO", "Fixed Rotation")
    assert snapshot.matched_scenario_count == 288
    assert snapshot.controller_evaluation_count == 576
    assert {metric.key for metric in snapshot.metrics} == {
        "critical_service_ratio",
        "served_mwh",
        "curtailed_mwh",
        "episode_reward",
        "utilization",
        "fairness",
        "switching_rate",
    }

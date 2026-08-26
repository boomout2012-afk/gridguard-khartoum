from __future__ import annotations

import copy
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pytest
import torch

import gridguard
from gridguard.controllers import PPOController
from gridguard.evaluation import (
    BATCH_SIZE,
    COUNT_STATEMENT,
    FROZEN_ABSOLUTE_TOLERANCES,
    FROZEN_BENCHMARK_SHA256,
    INFERENCE_METRICS,
    METRIC_ORDER,
    SEEDS,
    SEVERITIES,
    EvaluationArtifactError,
    assert_paired_benchmark_integrity,
    build_paired_benchmark_payload,
    canonical_paired_json_bytes,
    evaluate_scenario_pair,
    run_full_paired_evaluation,
    validate_paired_benchmark_payload,
    verify_paired_benchmark,
    write_paired_benchmark,
)
from gridguard.policy import BUNDLED_ARTIFACT_SHA256
from gridguard.scenarios import generate_scenarios

PROJECT_ROOT = Path(__file__).parents[1]
PPO_MANIFEST = PROJECT_ROOT / "data" / "model" / "manifest.json"
FROZEN_BENCHMARK = PROJECT_ROOT / "data" / "evidence" / "frozen-benchmark.json"
PAIRED_BENCHMARK = PROJECT_ROOT / "data" / "evidence" / "paired-benchmark.json"


class _FullServicePolicy:
    def act(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.full(
            (observation.shape[0], 7),
            2,
            dtype=torch.long,
            device=observation.device,
        )


def _provenance() -> dict[str, object]:
    return {
        "frozen_benchmark": {
            "path": "data/evidence/frozen-benchmark.json",
            "sha256": FROZEN_BENCHMARK_SHA256,
            "size_bytes": 9682,
        },
        "ppo_actor": {
            "path": "data/model/ppo-policy.npz",
            "sha256": BUNDLED_ARTIFACT_SHA256,
            "size_bytes": 5_160_058,
        },
        "ppo_manifest": {
            "path": "data/model/manifest.json",
            "sha256": "54300578c34ce18fe54aa57e95e216a81d002da43e82ee09e1ae5bd4e1b51349",
            "size_bytes": 1609,
        },
        "source_checkpoint": {
            "path": "artifacts/checkpoints/best.pt",
            "sha256": "c6e5712f966b46ba18f4c69fb457b9e69f78887c0c5c0279e5ce034f40f9c665",
            "source_revision": "52f6dd82605c993c8094448caecec86faa289c0c",
        },
    }


def _valid_payload(
    *,
    ppo_outperforms: bool = True,
    frozen_benchmark_path: Path = FROZEN_BENCHMARK,
) -> dict[str, object]:
    shape = (len(SEVERITIES), len(SEEDS), BATCH_SIZE, len(METRIC_ORDER))
    fixed = np.zeros(shape, dtype=np.float64)
    ppo = np.zeros(shape, dtype=np.float64)
    values = {
        "episode_reward": (10.0, 11.0),
        "demand_mwh": (100.0, 100.0),
        "available_mwh": (80.0, 80.0),
        "served_mwh": (60.0, 70.0),
        "curtailed_mwh": (40.0, 30.0),
        "critical_service_ratio": (0.4, 0.6),
        "utilization": (0.4, 0.6),
        "fairness": (0.4, 0.6),
        "switching_rate": (0.4, 0.3),
        "oversubscription": (0.1, 0.05),
        "power_balance_violation_rate": (0.0, 0.0),
        "critical_outage_locality_hours": (20.0, 10.0),
    }
    if not ppo_outperforms:
        values = {
            metric: pair if metric in {"demand_mwh", "available_mwh"} else pair[::-1]
            for metric, pair in values.items()
        }
    for metric_index, metric in enumerate(METRIC_ORDER):
        fixed[..., metric_index] = values[metric][0]
        ppo[..., metric_index] = values[metric][1]

    bootstrap = np.zeros((3, len(INFERENCE_METRICS)), dtype=np.float64)
    for index, metric in enumerate(INFERENCE_METRICS):
        delta = float(ppo[..., METRIC_ORDER.index(metric)].mean()) - float(
            fixed[..., METRIC_ORDER.index(metric)].mean()
        )
        bootstrap[:, index] = delta

    return build_paired_benchmark_payload(
        fixed_values=fixed,
        ppo_values=ppo,
        bootstrap_values=bootstrap,
        provenance=_provenance(),
        frozen_benchmark_path=frozen_benchmark_path,
    )


def test_evaluate_pair_reuses_exact_scenario_object() -> None:
    scenario = generate_scenarios(2, 91, "mild", device="cpu")

    evaluated = evaluate_scenario_pair(
        scenario,
        ppo_controller=PPOController(_FullServicePolicy()),
    )

    assert evaluated.fixed_result.scenario is scenario
    assert evaluated.ppo_result.scenario is scenario
    assert evaluated.fixed_metrics.batch_size == scenario.batch_size
    assert evaluated.ppo_metrics.batch_size == scenario.batch_size


def test_payload_uses_unambiguous_matched_counts_and_directions() -> None:
    payload = _valid_payload()

    assert payload["protocol"]["matched_scenario_count"] == 288
    assert payload["protocol"]["controller_evaluation_count"] == 576
    assert payload["protocol"]["independent_base_draw_count"] == 96
    assert payload["protocol"]["severity_variants_per_base_draw"] == 3
    assert payload["protocol"]["count_statement"] == COUNT_STATEMENT
    assert COUNT_STATEMENT == (
        "96 independent base draws crossed with 3 shortage levels = 288 matched scenario "
        "conditions; 576 controller-scenario evaluations (288 per controller)."
    )
    assert payload["protocol"]["delta_definition"] == "PPO - Fixed Rotation"
    assert payload["protocol"]["bootstrap"] == {
        "confidence_level": 0.95,
        "method": "deterministic seed-stratified paired cluster percentile bootstrap",
        "percentile_method": "linear",
        "replicates": 10_000,
        "seed": 20260815,
    }
    assert payload["metric_definitions"]["served_mwh"]["direction"] == "higher_is_better"
    assert payload["metric_definitions"]["curtailed_mwh"]["direction"] == "lower_is_better"
    assert payload["paired_inference"]["curtailed_mwh"]["favorable_sign"] == "negative"


@pytest.mark.parametrize("complex_input", ["fixed", "bootstrap"])
def test_payload_builder_rejects_complex_numeric_arrays(complex_input: str) -> None:
    shape = (len(SEVERITIES), len(SEEDS), BATCH_SIZE, len(METRIC_ORDER))
    fixed = np.zeros(shape, dtype=np.float64)
    ppo = np.zeros(shape, dtype=np.float64)
    bootstrap = np.zeros((3, len(INFERENCE_METRICS)), dtype=np.float64)
    if complex_input == "fixed":
        fixed = fixed.astype(np.complex128)
    else:
        bootstrap = bootstrap.astype(np.complex128)

    with pytest.raises(EvaluationArtifactError, match="real floating"):
        build_paired_benchmark_payload(
            fixed_values=fixed,
            ppo_values=ppo,
            bootstrap_values=bootstrap,
            provenance=_provenance(),
            frozen_benchmark_path=FROZEN_BENCHMARK,
        )


def test_payload_builder_requires_the_hashed_frozen_benchmark(tmp_path: Path) -> None:
    tampered_frozen = tmp_path / "frozen-benchmark.json"
    tampered_frozen.write_bytes(FROZEN_BENCHMARK.read_bytes() + b" ")

    with pytest.raises(EvaluationArtifactError, match="frozen benchmark SHA-256 mismatch"):
        _valid_payload(frozen_benchmark_path=tampered_frozen)


def test_neutral_serialization_accepts_adverse_results_but_trusted_writer_requires_parity(
    tmp_path: Path,
) -> None:
    payload = _valid_payload(ppo_outperforms=False)
    output = tmp_path / "adverse-paired-benchmark.json"

    validate_paired_benchmark_payload(payload)
    digest = write_paired_benchmark(payload, output)

    assert payload["frozen_parity"]["status"] == "failed"
    assert verify_paired_benchmark(output, expected_sha256=digest) == payload
    with pytest.raises(EvaluationArtifactError, match="frozen parity"):
        assert_paired_benchmark_integrity(
            payload,
            frozen_benchmark_path=FROZEN_BENCHMARK,
        )
    trusted_output = tmp_path / "trusted-adverse-paired-benchmark.json"
    with pytest.raises(EvaluationArtifactError, match="frozen parity"):
        write_paired_benchmark(
            payload,
            trusted_output,
            frozen_benchmark_path=FROZEN_BENCHMARK,
        )
    assert not trusted_output.exists()


def test_trusted_integrity_is_sign_neutral_for_ambiguous_interval() -> None:
    payload = json.loads(PAIRED_BENCHMARK.read_text(encoding="utf-8"))
    interval = payload["paired_inference"]["critical_service_ratio"]
    interval["ci95_lower"] = -0.01

    validate_paired_benchmark_payload(payload)
    canonical_paired_json_bytes(payload)

    assert_paired_benchmark_integrity(
        payload,
        frozen_benchmark_path=FROZEN_BENCHMARK,
    )


def test_integrity_check_recomputes_frozen_parity_instead_of_trusting_declared_errors() -> None:
    payload = json.loads(PAIRED_BENCHMARK.read_text(encoding="utf-8"))
    payload["frozen_parity"]["maximum_absolute_error"] = {
        metric: 0.0 for metric in FROZEN_ABSOLUTE_TOLERANCES
    }

    validate_paired_benchmark_payload(payload)

    with pytest.raises(EvaluationArtifactError, match="does not match verified cells"):
        assert_paired_benchmark_integrity(
            payload,
            frozen_benchmark_path=FROZEN_BENCHMARK,
        )


def test_canonical_round_trip_and_hash_verification(tmp_path: Path) -> None:
    payload = _valid_payload()
    output = tmp_path / "paired-benchmark.json"

    digest = write_paired_benchmark(payload, output)
    raw = output.read_bytes()

    assert raw == canonical_paired_json_bytes(payload)
    assert digest == hashlib.sha256(raw).hexdigest()
    assert verify_paired_benchmark(output, expected_sha256=digest) == payload


def test_verifier_rejects_duplicate_and_nonfinite_json(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    duplicate_hash = hashlib.sha256(duplicate.read_bytes()).hexdigest()

    with pytest.raises(EvaluationArtifactError, match="duplicate"):
        verify_paired_benchmark(duplicate, expected_sha256=duplicate_hash)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value":NaN}', encoding="utf-8")
    nonfinite_hash = hashlib.sha256(nonfinite.read_bytes()).hexdigest()

    with pytest.raises(EvaluationArtifactError, match="non-finite"):
        verify_paired_benchmark(nonfinite, expected_sha256=nonfinite_hash)


def test_validator_rejects_missing_scenario_and_identity_drift() -> None:
    missing = _valid_payload()
    missing["paired_rows"].pop()
    with pytest.raises(EvaluationArtifactError, match="288 paired rows"):
        validate_paired_benchmark_payload(missing)

    drifted = _valid_payload()
    drifted["paired_rows"][0]["scenario_id"] = "severe-50003-031"
    with pytest.raises(EvaluationArtifactError, match="scenario_id"):
        validate_paired_benchmark_payload(drifted)


def test_validator_rejects_missing_cell_and_count_drift() -> None:
    missing_cell = _valid_payload()
    missing_cell["cells"].pop()
    with pytest.raises(EvaluationArtifactError, match="exactly 9 cells"):
        validate_paired_benchmark_payload(missing_cell)

    wrong_count = _valid_payload()
    wrong_count["protocol"]["matched_scenario_count"] = 287
    with pytest.raises(EvaluationArtifactError, match="96-draw / 288-condition"):
        validate_paired_benchmark_payload(wrong_count)


def test_validator_rejects_nonfinite_metric_value() -> None:
    payload = _valid_payload()
    payload["paired_rows"][4]["Fixed Rotation"]["served_mwh"] = float("inf")

    with pytest.raises(EvaluationArtifactError, match="finite number"):
        validate_paired_benchmark_payload(payload)


def test_validator_rejects_metric_tampering() -> None:
    payload = _valid_payload()
    tampered = copy.deepcopy(payload)
    tampered["paired_rows"][7]["PPO"]["episode_reward"] += 0.25

    with pytest.raises(EvaluationArtifactError, match="paired_delta"):
        validate_paired_benchmark_payload(tampered)


@pytest.mark.parametrize(
    ("metric", "value", "message"),
    [
        ("served_mwh", 81.0, "available|served"),
        ("curtailed_mwh", 39.0, "energy identity"),
        ("oversubscription", 1.01, "between 0 and 1"),
        ("critical_outage_locality_hours", 168.1, "locality-hours"),
        ("critical_outage_locality_hours", 20.5, "whole locality-hours"),
    ],
)
def test_validator_rejects_physically_impossible_episode_metrics(
    metric: str,
    value: float,
    message: str,
) -> None:
    payload = _valid_payload()
    payload["paired_rows"][0]["Fixed Rotation"][metric] = value

    with pytest.raises(EvaluationArtifactError, match=message):
        validate_paired_benchmark_payload(payload)


def test_validator_allows_percentile_interval_not_containing_point_estimate() -> None:
    payload = _valid_payload()
    inference = payload["paired_inference"]["served_mwh"]
    inference["ci95_lower"] = inference["mean_delta"] + 1.0
    inference["ci95_upper"] = inference["mean_delta"] + 2.0

    validate_paired_benchmark_payload(payload)


def test_validator_rejects_available_energy_above_demand() -> None:
    payload = _valid_payload()
    payload["paired_rows"][0]["Fixed Rotation"]["available_mwh"] = 101.0

    with pytest.raises(EvaluationArtifactError, match="available_mwh must not exceed demand_mwh"):
        validate_paired_benchmark_payload(payload)


def test_neutral_validator_accepts_well_formed_untrusted_provenance_but_integrity_rejects() -> None:
    payload = _valid_payload()
    payload["provenance"]["ppo_actor"]["sha256"] = "f" * 64

    validate_paired_benchmark_payload(payload)

    with pytest.raises(EvaluationArtifactError, match="locked artifact"):
        assert_paired_benchmark_integrity(
            payload,
            frozen_benchmark_path=FROZEN_BENCHMARK,
        )


def test_release_verifier_cross_checks_paired_cells_against_frozen_evidence() -> None:
    digest = hashlib.sha256(PAIRED_BENCHMARK.read_bytes()).hexdigest()

    verified = verify_paired_benchmark(
        PAIRED_BENCHMARK,
        expected_sha256=digest,
        frozen_benchmark_path=FROZEN_BENCHMARK,
    )

    assert verified["frozen_parity"]["status"] == "passed"


def test_verifier_rejects_byte_tampering_before_semantic_use(tmp_path: Path) -> None:
    output = tmp_path / "paired-benchmark.json"
    digest = write_paired_benchmark(_valid_payload(), output)
    output.write_bytes(output.read_bytes().replace(b'"schema_version":2', b'"schema_version":3'))

    with pytest.raises(EvaluationArtifactError, match="SHA-256"):
        verify_paired_benchmark(output, expected_sha256=digest)


@pytest.mark.gpu
def test_full_cuda_protocol_matches_frozen_cells_and_is_self_verifying(tmp_path: Path) -> None:
    payload = run_full_paired_evaluation(
        ppo_manifest_path=PPO_MANIFEST,
        expected_ppo_artifact_sha256=BUNDLED_ARTIFACT_SHA256,
        frozen_benchmark_path=FROZEN_BENCHMARK,
        bootstrap_replicates=10_000,
        bootstrap_seed=20260815,
        device="cuda:0",
    )

    assert len(payload["paired_rows"]) == 288
    assert len(payload["cells"]) == 9
    assert payload["frozen_parity"]["status"] == "passed"
    assert all(
        payload["frozen_parity"]["maximum_absolute_error"][metric] <= tolerance
        for metric, tolerance in FROZEN_ABSOLUTE_TOLERANCES.items()
    )
    output = tmp_path / "paired-benchmark.json"
    digest = write_paired_benchmark(payload, output)
    assert (
        verify_paired_benchmark(
            output,
            expected_sha256=digest,
            frozen_benchmark_path=FROZEN_BENCHMARK,
        )
        == payload
    )


def test_package_version_is_derived_from_distribution_metadata() -> None:
    assert gridguard.__version__ == version("gridguard-khartoum")

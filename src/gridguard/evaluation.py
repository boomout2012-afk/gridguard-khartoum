"""Matched proposal evaluation and strict paired-evidence serialization."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

from gridguard.controllers import FixedRotationController, PPOController
from gridguard.domain import HORIZON_HOURS, LOCALITIES, Scenario, SimulationResult
from gridguard.metrics import EpisodeMetrics, compute_episode_metrics
from gridguard.policy import (
    BUNDLED_ARTIFACT_SHA256,
    SOURCE_CHECKPOINT_PATH,
    SOURCE_CHECKPOINT_SHA256,
    SOURCE_REVISION,
    load_ppo_policy,
)
from gridguard.provenance import sha256_file
from gridguard.scenarios import generate_scenarios
from gridguard.simulation import simulate
from gridguard.statistics import (
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_BOOTSTRAP_SEED,
    stratified_paired_bootstrap,
)

SEVERITIES: Final = ("mild", "moderate", "severe")
SEEDS: Final = (50001, 50002, 50003)
BATCH_SIZE: Final = 32
MATCHED_SCENARIO_COUNT: Final = len(SEVERITIES) * len(SEEDS) * BATCH_SIZE
CONTROLLER_EVALUATION_COUNT: Final = MATCHED_SCENARIO_COUNT * 2
COUNT_STATEMENT: Final = (
    "96 independent base draws crossed with 3 shortage levels = 288 matched scenario "
    "conditions; 576 controller-scenario evaluations (288 per controller)."
)
DELTA_DEFINITION: Final = "PPO - Fixed Rotation"
FROZEN_BENCHMARK_SHA256: Final = "3d720ba87a747a5780119decd7144b4051fc36fc72ab50f229f5750ad749fd15"
PAIRED_BENCHMARK_SHA256: Final = "2c3e71e48c4c7bff4fe609d42a76ca4f770f6dd717cd7d45d5e8e0d99fdf2d11"
FROZEN_BENCHMARK_SIZE_BYTES: Final = 9_682
PPO_ARTIFACT_SIZE_BYTES: Final = 5_160_058
PPO_MANIFEST_SHA256: Final = "89ece396d8d5ef55d11f5f59ed946a593a37a6b4b10a08040cf80aa58ef20fe2"
PPO_MANIFEST_SIZE_BYTES: Final = 1_329

METRIC_ORDER: Final = (
    "episode_reward",
    "demand_mwh",
    "available_mwh",
    "served_mwh",
    "curtailed_mwh",
    "critical_service_ratio",
    "utilization",
    "fairness",
    "switching_rate",
    "oversubscription",
    "power_balance_violation_rate",
    "critical_outage_locality_hours",
)
INFERENCE_METRICS: Final = (
    "episode_reward",
    "critical_service_ratio",
    "utilization",
    "fairness",
    "served_mwh",
    "curtailed_mwh",
    "switching_rate",
)
PRIMARY_METRICS: Final = frozenset(INFERENCE_METRICS[:-1])
FROZEN_METRICS: Final = METRIC_ORDER[:-1]

# These limits are fixed review decisions, not adaptive thresholds. Energy totals admit two
# float32 ULPs at the benchmark scale. The critical-service tolerance also covers the reviewed
# estimand/reduction difference between the source's pooled ratio and the paired artifact's mean
# of per-episode ratios (maximum observed difference 1.268e-6).
FROZEN_ABSOLUTE_TOLERANCES: Final = {
    metric: (0.002 if metric.endswith("_mwh") else 0.000002) for metric in FROZEN_METRICS
}
ENERGY_IDENTITY_ABSOLUTE_TOLERANCE: Final = 0.002
MAX_CRITICAL_OUTAGE_LOCALITY_HOURS: Final = HORIZON_HOURS * len(LOCALITIES)

METRIC_DEFINITIONS: Final[dict[str, dict[str, str]]] = {
    "episode_reward": {
        "definition": "Sum of the accepted 24 hourly reward values.",
        "unit": "reward units",
        "direction": "higher_is_better",
        "role": "primary",
    },
    "demand_mwh": {
        "definition": "Sum of simulated system demand over 24 one-hour steps.",
        "unit": "MWh",
        "direction": "input_only",
        "role": "diagnostic",
    },
    "available_mwh": {
        "definition": "Sum of available system supply over 24 one-hour steps.",
        "unit": "MWh",
        "direction": "input_only",
        "role": "diagnostic",
    },
    "served_mwh": {
        "definition": "Sum of dispatched energy over 24 one-hour steps.",
        "unit": "MWh",
        "direction": "higher_is_better",
        "role": "primary",
    },
    "curtailed_mwh": {
        "definition": "Demand minus dispatched energy over 24 one-hour steps.",
        "unit": "MWh",
        "direction": "lower_is_better",
        "role": "primary",
    },
    "critical_service_ratio": {
        "definition": "Served critical-sector energy divided by critical-sector demand.",
        "unit": "ratio",
        "direction": "higher_is_better",
        "role": "primary",
    },
    "utilization": {
        "definition": "Mean hourly served power divided by min(available supply, demand).",
        "unit": "ratio",
        "direction": "higher_is_better",
        "role": "primary",
    },
    "fairness": {
        "definition": "Mean hourly Jain fairness of cumulative locality service ratios.",
        "unit": "ratio",
        "direction": "higher_is_better",
        "role": "primary",
    },
    "switching_rate": {
        "definition": (
            "Mean hourly fraction of localities changing service tier; the initial hour is zero."
        ),
        "unit": "ratio",
        "direction": "lower_is_better",
        "role": "secondary",
    },
    "oversubscription": {
        "definition": "Mean hourly positive requested-power excess divided by demand.",
        "unit": "ratio",
        "direction": "lower_is_better",
        "role": "diagnostic",
    },
    "power_balance_violation_rate": {
        "definition": "Fraction of hours served power exceeds supply by more than 1e-3 MW.",
        "unit": "ratio",
        "direction": "lower_is_better",
        "role": "safety",
    },
    "critical_outage_locality_hours": {
        "definition": "Locality-hours with critical-sector service below 99 percent.",
        "unit": "locality-hours",
        "direction": "lower_is_better",
        "role": "diagnostic",
    },
}

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_TOP_LEVEL_FIELDS = frozenset(
    {
        "aggregate",
        "benchmark_id",
        "cells",
        "frozen_parity",
        "limitations",
        "metric_definitions",
        "paired_inference",
        "paired_rows",
        "protocol",
        "provenance",
        "schema_version",
    }
)
_ROW_FIELDS = frozenset(
    {
        "scenario_id",
        "shortage_level",
        "seed",
        "batch_index",
        "Fixed Rotation",
        "PPO",
        "paired_delta",
    }
)
_CELL_FIELDS = frozenset(
    {"shortage_level", "seed", "episode_count", "Fixed Rotation", "PPO", "paired_delta"}
)
_INFERENCE_FIELDS = frozenset(
    {
        "mean_delta",
        "ci95_lower",
        "ci95_upper",
        "direction",
        "favorable_sign",
        "role",
        "delta_definition",
    }
)
_LIMITATIONS: Final = [
    "Synthetic simulation evidence; not real utility measurements.",
    (
        "The accepted historical artifact stored 18 batch-cell summaries; these episode rows "
        "and intervals were regenerated from the audited simulator and frozen PPO checkpoint."
    ),
    (
        "Paired bootstrap intervals quantify variability under the fixed synthetic "
        "scenario design, not a field causal effect."
    ),
    "Results describe one frozen PPO actor checkpoint and deterministic inference.",
]


class EvaluationArtifactError(ValueError):
    """Raised when paired evaluation evidence is incomplete or untrusted."""


@dataclass(frozen=True, slots=True, eq=False)
class PairedScenarioEvaluation:
    """Two controller results and metrics sharing one exact scenario object."""

    fixed_result: SimulationResult
    ppo_result: SimulationResult
    fixed_metrics: EpisodeMetrics
    ppo_metrics: EpisodeMetrics

    def __post_init__(self) -> None:
        if self.fixed_result.scenario is not self.ppo_result.scenario:
            raise EvaluationArtifactError("paired controllers must share the exact Scenario object")


@torch.inference_mode()
def evaluate_scenario_pair(
    scenario: Scenario,
    *,
    ppo_controller: PPOController,
) -> PairedScenarioEvaluation:
    """Evaluate both locked controllers against the exact same scenario tensors."""

    if not isinstance(scenario, Scenario):
        raise TypeError("scenario must be a Scenario")
    if not isinstance(ppo_controller, PPOController):
        raise TypeError("ppo_controller must be a PPOController")
    fixed_result = simulate(scenario, FixedRotationController())
    ppo_result = simulate(scenario, ppo_controller)
    return PairedScenarioEvaluation(
        fixed_result=fixed_result,
        ppo_result=ppo_result,
        fixed_metrics=compute_episode_metrics(fixed_result),
        ppo_metrics=compute_episode_metrics(ppo_result),
    )


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def _metric_mapping(row: np.ndarray[Any, Any]) -> dict[str, float]:
    return {metric: float(row[index]) for index, metric in enumerate(METRIC_ORDER)}


def _favorable_sign(metric: str) -> str:
    return (
        "negative" if METRIC_DEFINITIONS[metric]["direction"] == "lower_is_better" else "positive"
    )


def build_paired_benchmark_payload(
    *,
    fixed_values: np.ndarray[Any, Any],
    ppo_values: np.ndarray[Any, Any],
    bootstrap_values: np.ndarray[Any, Any],
    provenance: Mapping[str, object],
    frozen_maximum_absolute_error: Mapping[str, float],
) -> dict[str, Any]:
    """Build canonical-ready evidence at the single final host serialization boundary."""

    expected_shape = (len(SEVERITIES), len(SEEDS), BATCH_SIZE, len(METRIC_ORDER))
    for name, values in (("fixed_values", fixed_values), ("ppo_values", ppo_values)):
        if not isinstance(values, np.ndarray):
            raise TypeError(f"{name} must be a numpy.ndarray")
        if values.shape != expected_shape:
            raise EvaluationArtifactError(f"{name} must have shape {expected_shape}")
        if values.dtype.kind != "f" or not bool(np.isfinite(values).all()):
            raise EvaluationArtifactError(f"{name} must contain only finite real floating values")
    if not isinstance(bootstrap_values, np.ndarray):
        raise TypeError("bootstrap_values must be a numpy.ndarray")
    if bootstrap_values.shape != (3, len(INFERENCE_METRICS)):
        raise EvaluationArtifactError("bootstrap_values must have shape [3, inference_metric]")
    if bootstrap_values.dtype.kind != "f" or not bool(np.isfinite(bootstrap_values).all()):
        raise EvaluationArtifactError(
            "bootstrap_values must contain only finite real floating values"
        )
    if bool((bootstrap_values[1] > bootstrap_values[2]).any()):
        raise EvaluationArtifactError("bootstrap lower limits must not exceed upper limits")

    paired_rows: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    for severity_index, shortage_level in enumerate(SEVERITIES):
        for seed_index, seed in enumerate(SEEDS):
            cell_rows: list[dict[str, Any]] = []
            for batch_index in range(BATCH_SIZE):
                fixed = _metric_mapping(fixed_values[severity_index, seed_index, batch_index])
                ppo = _metric_mapping(ppo_values[severity_index, seed_index, batch_index])
                delta = {metric: ppo[metric] - fixed[metric] for metric in METRIC_ORDER}
                row = {
                    "Fixed Rotation": fixed,
                    "PPO": ppo,
                    "batch_index": batch_index,
                    "paired_delta": delta,
                    "scenario_id": f"{shortage_level}-{seed}-{batch_index:03d}",
                    "seed": seed,
                    "shortage_level": shortage_level,
                }
                paired_rows.append(row)
                cell_rows.append(row)

            cells.append(
                {
                    "Fixed Rotation": {
                        metric: _mean([row["Fixed Rotation"][metric] for row in cell_rows])
                        for metric in METRIC_ORDER
                    },
                    "PPO": {
                        metric: _mean([row["PPO"][metric] for row in cell_rows])
                        for metric in METRIC_ORDER
                    },
                    "episode_count": BATCH_SIZE,
                    "paired_delta": {
                        metric: _mean([row["paired_delta"][metric] for row in cell_rows])
                        for metric in METRIC_ORDER
                    },
                    "seed": seed,
                    "shortage_level": shortage_level,
                }
            )

    aggregate = {
        controller: {
            metric: _mean([cell[controller][metric] for cell in cells]) for metric in METRIC_ORDER
        }
        for controller in ("Fixed Rotation", "PPO")
    }
    paired_inference = {
        metric: {
            "ci95_lower": float(bootstrap_values[1, index]),
            "ci95_upper": float(bootstrap_values[2, index]),
            "delta_definition": DELTA_DEFINITION,
            "direction": METRIC_DEFINITIONS[metric]["direction"],
            "favorable_sign": _favorable_sign(metric),
            "mean_delta": float(bootstrap_values[0, index]),
            "role": "primary" if metric in PRIMARY_METRICS else "secondary",
        }
        for index, metric in enumerate(INFERENCE_METRICS)
    }
    payload: dict[str, Any] = {
        "aggregate": aggregate,
        "benchmark_id": "proposal-ppo-vs-fixed-rotation-paired",
        "cells": cells,
        "frozen_parity": {
            "absolute_tolerances": dict(FROZEN_ABSOLUTE_TOLERANCES),
            "comparison_granularity": "18 controller batch-cell summaries",
            "maximum_absolute_error": dict(frozen_maximum_absolute_error),
            "status": "passed",
        },
        "limitations": list(_LIMITATIONS),
        "metric_definitions": {
            metric: dict(definition) for metric, definition in METRIC_DEFINITIONS.items()
        },
        "paired_inference": paired_inference,
        "paired_rows": paired_rows,
        "protocol": {
            "bootstrap": {
                "confidence_level": 0.95,
                "method": ("deterministic seed-stratified paired cluster percentile bootstrap"),
                "percentile_method": "linear",
                "replicates": 10_000,
                "seed": 20260815,
            },
            "cell_count": len(SEVERITIES) * len(SEEDS),
            "controllers": ["Fixed Rotation", "PPO"],
            "controller_evaluation_count": CONTROLLER_EVALUATION_COUNT,
            "count_statement": COUNT_STATEMENT,
            "delta_definition": DELTA_DEFINITION,
            "episodes_per_cell": BATCH_SIZE,
            "evaluations_per_controller": MATCHED_SCENARIO_COUNT,
            "horizon_hours": HORIZON_HOURS,
            "independent_base_draw_count": len(SEEDS) * BATCH_SIZE,
            "matched_scenario_count": MATCHED_SCENARIO_COUNT,
            "seeds": list(SEEDS),
            "severity_variants_per_base_draw": len(SEVERITIES),
            "shortage_levels": list(SEVERITIES),
        },
        "provenance": dict(provenance),
        "schema_version": 1,
    }
    validate_paired_benchmark_payload(payload)
    return payload


def canonical_paired_json_bytes(payload: object) -> bytes:
    """Serialize strict finite evidence deterministically for stable hashing."""

    try:
        encoded = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise EvaluationArtifactError(f"paired evidence is not canonical JSON: {error}") from error
    return f"{encoded}\n".encode()


def write_paired_benchmark(payload: Mapping[str, object], path: str | Path) -> str:
    """Validate and atomically write a canonical paired benchmark."""

    validate_paired_benchmark_payload(payload)
    encoded = canonical_paired_json_bytes(payload)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        temporary.write_bytes(encoded)
        temporary.replace(output)
    except OSError as error:
        raise EvaluationArtifactError(f"paired benchmark cannot be written: {error}") from error
    return hashlib.sha256(encoded).hexdigest()


def verify_paired_benchmark(
    path: str | Path,
    *,
    expected_sha256: str,
    frozen_benchmark_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify trust anchor, canonical bytes, duplicate safety, and all identities."""

    if type(expected_sha256) is not str or not _SHA256_PATTERN.fullmatch(expected_sha256):
        raise EvaluationArtifactError("expected_sha256 must be a lowercase SHA-256 digest")
    try:
        raw = Path(path).read_bytes()
    except OSError as error:
        raise EvaluationArtifactError(f"paired benchmark cannot be read: {error}") from error
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != expected_sha256:
        raise EvaluationArtifactError(
            f"paired benchmark SHA-256 mismatch: expected {expected_sha256}, found {actual_sha256}"
        )
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_constant,
        )
    except EvaluationArtifactError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise EvaluationArtifactError(
            f"paired benchmark is not valid UTF-8 JSON: {error}"
        ) from error
    mapping = _require_mapping(payload, field="paired benchmark")
    if canonical_paired_json_bytes(mapping) != raw:
        raise EvaluationArtifactError("paired benchmark bytes are not in canonical form")
    validate_paired_benchmark_payload(mapping)
    if frozen_benchmark_path is not None:
        _cross_check_frozen_parity(mapping, Path(frozen_benchmark_path))
    return mapping


def _require_mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise EvaluationArtifactError(f"{field} must be an object")
    return dict(value)


def _require_exact_fields(
    value: Mapping[str, object], expected: frozenset[str], field: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise EvaluationArtifactError(
            f"{field} fields mismatch; missing={sorted(expected - actual)}, "
            f"extras={sorted(actual - expected)}"
        )


def _number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationArtifactError(f"{field} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise EvaluationArtifactError(f"{field} must be a finite number")
    return normalized


def _metric_values(
    value: object,
    *,
    field: str,
    delta: bool = False,
    episode: bool = False,
) -> dict[str, float]:
    mapping = _require_mapping(value, field=field)
    _require_exact_fields(mapping, frozenset(METRIC_ORDER), field)
    result = {
        metric: _number(mapping[metric], field=f"{field}.{metric}") for metric in METRIC_ORDER
    }
    if not delta:
        for metric in METRIC_ORDER:
            if metric != "episode_reward" and result[metric] < 0.0:
                raise EvaluationArtifactError(f"{field}.{metric} must not be negative")
        for metric in (
            "critical_service_ratio",
            "utilization",
            "fairness",
            "switching_rate",
            "oversubscription",
            "power_balance_violation_rate",
        ):
            if not 0.0 <= result[metric] <= 1.0:
                raise EvaluationArtifactError(f"{field}.{metric} must be between 0 and 1")
        if result["demand_mwh"] <= 0.0 or result["available_mwh"] <= 0.0:
            raise EvaluationArtifactError(f"{field} demand and available energy must be positive")
        if result["available_mwh"] > result["demand_mwh"] + ENERGY_IDENTITY_ABSOLUTE_TOLERANCE:
            raise EvaluationArtifactError(f"{field}.available_mwh must not exceed demand_mwh")
        if (
            result["served_mwh"]
            > min(result["demand_mwh"], result["available_mwh"])
            + ENERGY_IDENTITY_ABSOLUTE_TOLERANCE
        ):
            raise EvaluationArtifactError(
                f"{field}.served_mwh must not exceed demand_mwh or available_mwh"
            )
        if not math.isclose(
            result["demand_mwh"],
            result["served_mwh"] + result["curtailed_mwh"],
            abs_tol=ENERGY_IDENTITY_ABSOLUTE_TOLERANCE,
            rel_tol=0.0,
        ):
            raise EvaluationArtifactError(
                f"{field} energy identity demand_mwh = served_mwh + curtailed_mwh failed"
            )
        outage = result["critical_outage_locality_hours"]
        if outage > MAX_CRITICAL_OUTAGE_LOCALITY_HOURS:
            raise EvaluationArtifactError(
                f"{field}.critical_outage_locality_hours exceeds the 168 locality-hours maximum"
            )
        if episode and not math.isclose(outage, round(outage), abs_tol=1e-12, rel_tol=0.0):
            raise EvaluationArtifactError(
                f"{field}.critical_outage_locality_hours must be whole locality-hours per episode"
            )
    return result


def _validate_protocol(value: object) -> None:
    expected = {
        "bootstrap": {
            "confidence_level": 0.95,
            "method": "deterministic seed-stratified paired cluster percentile bootstrap",
            "percentile_method": "linear",
            "replicates": 10_000,
            "seed": 20260815,
        },
        "cell_count": 9,
        "controllers": ["Fixed Rotation", "PPO"],
        "controller_evaluation_count": 576,
        "count_statement": COUNT_STATEMENT,
        "delta_definition": DELTA_DEFINITION,
        "episodes_per_cell": 32,
        "evaluations_per_controller": 288,
        "horizon_hours": 24,
        "independent_base_draw_count": 96,
        "matched_scenario_count": 288,
        "seeds": list(SEEDS),
        "severity_variants_per_base_draw": 3,
        "shortage_levels": list(SEVERITIES),
    }
    if value != expected:
        raise EvaluationArtifactError(
            "protocol does not match the locked 96-draw / 288-condition design"
        )


def _validate_provenance(value: object) -> None:
    provenance = _require_mapping(value, field="provenance")
    _require_exact_fields(
        provenance,
        frozenset({"frozen_benchmark", "ppo_actor", "ppo_manifest", "source_checkpoint"}),
        "provenance",
    )
    expected_paths = {
        "frozen_benchmark": "data/evidence/frozen-benchmark.json",
        "ppo_actor": "data/model/ppo-policy.npz",
        "ppo_manifest": "data/model/manifest.json",
    }
    expected_artifacts = {
        "frozen_benchmark": {
            "path": expected_paths["frozen_benchmark"],
            "sha256": FROZEN_BENCHMARK_SHA256,
            "size_bytes": FROZEN_BENCHMARK_SIZE_BYTES,
        },
        "ppo_actor": {
            "path": expected_paths["ppo_actor"],
            "sha256": BUNDLED_ARTIFACT_SHA256,
            "size_bytes": PPO_ARTIFACT_SIZE_BYTES,
        },
        "ppo_manifest": {
            "path": expected_paths["ppo_manifest"],
            "sha256": PPO_MANIFEST_SHA256,
            "size_bytes": PPO_MANIFEST_SIZE_BYTES,
        },
    }
    for name, expected_path in expected_paths.items():
        artifact = _require_mapping(provenance[name], field=f"provenance.{name}")
        _require_exact_fields(
            artifact, frozenset({"path", "sha256", "size_bytes"}), f"provenance.{name}"
        )
        if artifact["path"] != expected_path:
            raise EvaluationArtifactError(f"provenance.{name}.path is not the locked relative path")
        if type(artifact["sha256"]) is not str or not _SHA256_PATTERN.fullmatch(artifact["sha256"]):
            raise EvaluationArtifactError(f"provenance.{name}.sha256 is invalid")
        if type(artifact["size_bytes"]) is not int or artifact["size_bytes"] <= 0:
            raise EvaluationArtifactError(f"provenance.{name}.size_bytes must be positive")
        if artifact != expected_artifacts[name]:
            raise EvaluationArtifactError(f"provenance.{name} is not the locked artifact")
    source = _require_mapping(provenance["source_checkpoint"], field="provenance.source_checkpoint")
    _require_exact_fields(
        source,
        frozenset({"path", "sha256", "source_revision"}),
        "provenance.source_checkpoint",
    )
    if source["path"] != "artifacts/checkpoints/best.pt":
        raise EvaluationArtifactError("provenance.source_checkpoint.path is invalid")
    if type(source["sha256"]) is not str or not _SHA256_PATTERN.fullmatch(source["sha256"]):
        raise EvaluationArtifactError("provenance.source_checkpoint.sha256 is invalid")
    if type(source["source_revision"]) is not str or not _REVISION_PATTERN.fullmatch(
        source["source_revision"]
    ):
        raise EvaluationArtifactError("provenance.source_checkpoint.source_revision is invalid")
    expected_source = {
        "path": SOURCE_CHECKPOINT_PATH,
        "sha256": SOURCE_CHECKPOINT_SHA256,
        "source_revision": SOURCE_REVISION,
    }
    if source != expected_source:
        raise EvaluationArtifactError("provenance.source_checkpoint is not the locked source")


def validate_paired_benchmark_payload(payload: Mapping[str, object]) -> None:
    """Fail closed on schema, counts, pairing, aggregates, and inference drift."""

    root = _require_mapping(payload, field="paired benchmark")
    _require_exact_fields(root, _TOP_LEVEL_FIELDS, "paired benchmark")
    if root["schema_version"] != 1 or type(root["schema_version"]) is not int:
        raise EvaluationArtifactError("unsupported paired benchmark schema_version")
    if root["benchmark_id"] != "proposal-ppo-vs-fixed-rotation-paired":
        raise EvaluationArtifactError("unexpected paired benchmark identity")
    _validate_protocol(root["protocol"])
    _validate_provenance(root["provenance"])
    if root["limitations"] != _LIMITATIONS:
        raise EvaluationArtifactError("limitations do not match the reviewed evidence scope")
    if root["metric_definitions"] != METRIC_DEFINITIONS:
        raise EvaluationArtifactError("metric_definitions drifted from the locked definitions")

    rows = root["paired_rows"]
    if not isinstance(rows, list) or len(rows) != MATCHED_SCENARIO_COUNT:
        raise EvaluationArtifactError("paired benchmark must contain exactly 288 paired rows")
    rows_by_cell: dict[tuple[str, int], list[dict[str, Any]]] = {}
    position = 0
    for shortage_level in SEVERITIES:
        for seed in SEEDS:
            for batch_index in range(BATCH_SIZE):
                row = _require_mapping(rows[position], field=f"paired_rows[{position}]")
                _require_exact_fields(row, _ROW_FIELDS, f"paired_rows[{position}]")
                expected_id = f"{shortage_level}-{seed}-{batch_index:03d}"
                if row["scenario_id"] != expected_id:
                    raise EvaluationArtifactError(f"paired_rows[{position}].scenario_id drifted")
                if (
                    row["shortage_level"] != shortage_level
                    or row["seed"] != seed
                    or row["batch_index"] != batch_index
                ):
                    raise EvaluationArtifactError(f"paired_rows[{position}] cell identity drifted")
                fixed = _metric_values(
                    row["Fixed Rotation"],
                    field=f"paired_rows[{position}].Fixed Rotation",
                    episode=True,
                )
                ppo = _metric_values(row["PPO"], field=f"paired_rows[{position}].PPO", episode=True)
                delta = _metric_values(
                    row["paired_delta"],
                    field=f"paired_rows[{position}].paired_delta",
                    delta=True,
                )
                for metric in METRIC_ORDER:
                    if not math.isclose(
                        delta[metric], ppo[metric] - fixed[metric], abs_tol=1e-12, rel_tol=0.0
                    ):
                        raise EvaluationArtifactError(
                            f"paired_rows[{position}].paired_delta.{metric} is inconsistent"
                        )
                for metric in ("demand_mwh", "available_mwh"):
                    if fixed[metric] != ppo[metric]:
                        raise EvaluationArtifactError(
                            f"paired_rows[{position}] does not preserve matched {metric}"
                        )
                rows_by_cell.setdefault((shortage_level, seed), []).append(row)
                position += 1

    cells = root["cells"]
    if not isinstance(cells, list) or len(cells) != 9:
        raise EvaluationArtifactError("paired benchmark must contain exactly 9 cells")
    for cell_index, ((shortage_level, seed), cell) in enumerate(
        zip(rows_by_cell, cells, strict=True)
    ):
        summary = _require_mapping(cell, field=f"cells[{cell_index}]")
        _require_exact_fields(summary, _CELL_FIELDS, f"cells[{cell_index}]")
        if (
            summary["shortage_level"] != shortage_level
            or summary["seed"] != seed
            or summary["episode_count"] != BATCH_SIZE
        ):
            raise EvaluationArtifactError(f"cells[{cell_index}] identity or count drifted")
        for controller in ("Fixed Rotation", "PPO", "paired_delta"):
            actual = _metric_values(
                summary[controller],
                field=f"cells[{cell_index}].{controller}",
                delta=controller == "paired_delta",
            )
            expected = {
                metric: _mean(
                    [row[controller][metric] for row in rows_by_cell[(shortage_level, seed)]]
                )
                for metric in METRIC_ORDER
            }
            for metric in METRIC_ORDER:
                if not math.isclose(actual[metric], expected[metric], abs_tol=1e-12, rel_tol=0.0):
                    raise EvaluationArtifactError(
                        f"cells[{cell_index}].{controller}.{metric} drifted"
                    )

    aggregate = _require_mapping(root["aggregate"], field="aggregate")
    _require_exact_fields(aggregate, frozenset({"Fixed Rotation", "PPO"}), "aggregate")
    for controller in ("Fixed Rotation", "PPO"):
        actual = _metric_values(aggregate[controller], field=f"aggregate.{controller}")
        for metric in METRIC_ORDER:
            expected = _mean([cell[controller][metric] for cell in cells])
            if not math.isclose(actual[metric], expected, abs_tol=1e-12, rel_tol=0.0):
                raise EvaluationArtifactError(f"aggregate.{controller}.{metric} drifted")

    inference = _require_mapping(root["paired_inference"], field="paired_inference")
    _require_exact_fields(inference, frozenset(INFERENCE_METRICS), "paired_inference")
    for metric in INFERENCE_METRICS:
        item = _require_mapping(inference[metric], field=f"paired_inference.{metric}")
        _require_exact_fields(item, _INFERENCE_FIELDS, f"paired_inference.{metric}")
        expected_metadata = {
            "delta_definition": DELTA_DEFINITION,
            "direction": METRIC_DEFINITIONS[metric]["direction"],
            "favorable_sign": _favorable_sign(metric),
            "role": "primary" if metric in PRIMARY_METRICS else "secondary",
        }
        if any(item[name] != value for name, value in expected_metadata.items()):
            raise EvaluationArtifactError(f"paired_inference.{metric} metadata drifted")
        mean_delta = _number(item["mean_delta"], field=f"paired_inference.{metric}.mean_delta")
        lower = _number(item["ci95_lower"], field=f"paired_inference.{metric}.ci95_lower")
        upper = _number(item["ci95_upper"], field=f"paired_inference.{metric}.ci95_upper")
        expected_mean = _mean([row["paired_delta"][metric] for row in rows])
        if not math.isclose(mean_delta, expected_mean, abs_tol=1e-9, rel_tol=1e-12):
            raise EvaluationArtifactError(f"paired_inference.{metric}.mean_delta drifted")
        if lower > upper:
            raise EvaluationArtifactError(f"paired_inference.{metric} interval is reversed")
        if not lower <= mean_delta <= upper:
            raise EvaluationArtifactError(
                f"paired_inference.{metric} interval must contain its mean delta"
            )
        direction = METRIC_DEFINITIONS[metric]["direction"]
        if (direction == "higher_is_better" and lower <= 0.0) or (
            direction == "lower_is_better" and upper >= 0.0
        ):
            raise EvaluationArtifactError(
                f"paired_inference.{metric} interval does not support its favorable direction"
            )

    parity = _require_mapping(root["frozen_parity"], field="frozen_parity")
    _require_exact_fields(
        parity,
        frozenset(
            {"absolute_tolerances", "comparison_granularity", "maximum_absolute_error", "status"}
        ),
        "frozen_parity",
    )
    if parity["status"] != "passed" or parity["comparison_granularity"] != (
        "18 controller batch-cell summaries"
    ):
        raise EvaluationArtifactError("frozen_parity did not pass the reviewed comparison")
    if parity["absolute_tolerances"] != FROZEN_ABSOLUTE_TOLERANCES:
        raise EvaluationArtifactError("frozen_parity tolerances drifted")
    errors = _require_mapping(
        parity["maximum_absolute_error"], field="frozen_parity.maximum_absolute_error"
    )
    _require_exact_fields(errors, frozenset(FROZEN_METRICS), "frozen_parity.maximum_absolute_error")
    for metric, tolerance in FROZEN_ABSOLUTE_TOLERANCES.items():
        error = _number(errors[metric], field=f"frozen_parity.maximum_absolute_error.{metric}")
        if error < 0.0 or error > tolerance:
            raise EvaluationArtifactError(
                f"frozen parity failed for {metric}: error {error} exceeds {tolerance}"
            )


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in pairs:
        if name in result:
            raise EvaluationArtifactError(f"paired benchmark contains duplicate field {name!r}")
        result[name] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise EvaluationArtifactError(f"paired benchmark contains non-finite JSON constant {value}")


def _read_strict_json(raw: bytes, *, field: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_constant,
        )
    except EvaluationArtifactError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise EvaluationArtifactError(f"{field} is not valid UTF-8 JSON: {error}") from error
    return _require_mapping(payload, field=field)


def _load_frozen_cells(path: Path) -> tuple[dict[tuple[str, int, str], dict[str, float]], int]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise EvaluationArtifactError(f"frozen benchmark cannot be read: {error}") from error
    actual_hash = hashlib.sha256(raw).hexdigest()
    if actual_hash != FROZEN_BENCHMARK_SHA256:
        raise EvaluationArtifactError(
            "frozen benchmark SHA-256 mismatch: "
            f"expected {FROZEN_BENCHMARK_SHA256}, found {actual_hash}"
        )
    frozen = _read_strict_json(raw, field="frozen benchmark")
    if frozen.get("schema_version") != 1 or frozen.get("benchmark_id") != (
        "proposal-ppo-vs-fixed-rotation"
    ):
        raise EvaluationArtifactError("frozen benchmark identity is unsupported")
    protocol = _require_mapping(frozen.get("protocol"), field="frozen benchmark.protocol")
    expected_protocol_fields = {
        "controllers": ["Fixed Rotation", "PPO"],
        "episode_count_per_cell": BATCH_SIZE,
        "horizon_hours": HORIZON_HOURS,
        "matched_scenario_count": MATCHED_SCENARIO_COUNT,
        "controller_evaluation_count": CONTROLLER_EVALUATION_COUNT,
        "seeds": list(SEEDS),
        "severities": list(SEVERITIES),
        "stored_matched_cell_count": 9,
    }
    if any(protocol.get(name) != value for name, value in expected_protocol_fields.items()):
        raise EvaluationArtifactError("frozen benchmark protocol drifted from the locked design")

    raw_cells = frozen.get("cells")
    if not isinstance(raw_cells, list) or len(raw_cells) != 18:
        raise EvaluationArtifactError("frozen benchmark must contain 18 controller cells")
    cells: dict[tuple[str, int, str], dict[str, float]] = {}
    expected_row_fields = frozenset(
        {"controller", "episode_count", "seed", "severity", *FROZEN_METRICS}
    )
    for index, value in enumerate(raw_cells):
        row = _require_mapping(value, field=f"frozen benchmark.cells[{index}]")
        _require_exact_fields(row, expected_row_fields, f"frozen benchmark.cells[{index}]")
        controller = row["controller"]
        severity = row["severity"]
        seed = row["seed"]
        if (
            controller not in ("Fixed Rotation", "PPO")
            or severity not in SEVERITIES
            or seed not in SEEDS
            or row["episode_count"] != BATCH_SIZE
        ):
            raise EvaluationArtifactError(f"frozen benchmark.cells[{index}] identity drifted")
        key = (severity, seed, controller)
        if key in cells:
            raise EvaluationArtifactError(f"frozen benchmark contains duplicate cell {key}")
        cells[key] = {
            metric: _number(row[metric], field=f"frozen benchmark.cells[{index}].{metric}")
            for metric in FROZEN_METRICS
        }
    expected_keys = {
        (severity, seed, controller)
        for severity in SEVERITIES
        for seed in SEEDS
        for controller in ("Fixed Rotation", "PPO")
    }
    if set(cells) != expected_keys:
        raise EvaluationArtifactError("frozen benchmark does not contain the complete cell grid")
    return cells, len(raw)


def _cross_check_frozen_parity(payload: Mapping[str, object], frozen_path: Path) -> None:
    frozen_cells, frozen_size_bytes = _load_frozen_cells(frozen_path)
    provenance = _require_mapping(payload["provenance"], field="provenance")
    frozen_provenance = _require_mapping(
        provenance["frozen_benchmark"], field="provenance.frozen_benchmark"
    )
    if (
        frozen_provenance["sha256"] != FROZEN_BENCHMARK_SHA256
        or frozen_provenance["size_bytes"] != frozen_size_bytes
    ):
        raise EvaluationArtifactError(
            "paired evidence provenance does not identify the verified frozen benchmark"
        )

    raw_cells = payload["cells"]
    if not isinstance(raw_cells, list):
        raise EvaluationArtifactError("paired benchmark cells must be a list")
    paired_cells = {
        (cell["shortage_level"], cell["seed"]): cell
        for cell in raw_cells
        if isinstance(cell, Mapping)
    }
    maximum_error = {metric: 0.0 for metric in FROZEN_METRICS}
    for (severity, seed, controller), accepted in frozen_cells.items():
        cell = paired_cells[(severity, seed)]
        actual = _require_mapping(cell[controller], field=f"cells.{severity}.{seed}.{controller}")
        for metric in FROZEN_METRICS:
            error = abs(float(actual[metric]) - accepted[metric])
            maximum_error[metric] = max(maximum_error[metric], error)

    parity = _require_mapping(payload["frozen_parity"], field="frozen_parity")
    declared = _require_mapping(
        parity["maximum_absolute_error"], field="frozen_parity.maximum_absolute_error"
    )
    for metric, actual_error in maximum_error.items():
        if not math.isclose(float(declared[metric]), actual_error, abs_tol=1e-12, rel_tol=0.0):
            raise EvaluationArtifactError(
                f"frozen_parity.maximum_absolute_error.{metric} does not match verified cells"
            )


def _episode_metric_matrix(metrics: EpisodeMetrics) -> torch.Tensor:
    values = metrics.as_dict()
    if set(values) != set(METRIC_ORDER):
        raise EvaluationArtifactError("episode metric schema drifted from the evaluation contract")
    matrix = torch.stack(tuple(values[metric] for metric in METRIC_ORDER), dim=-1)
    if matrix.device.type != "cuda" or matrix.device.index != 0:
        raise EvaluationArtifactError("episode metrics left required CUDA device cuda:0")
    return matrix


def _compare_frozen_cells(
    fixed_values: np.ndarray[Any, Any],
    ppo_values: np.ndarray[Any, Any],
    frozen_cells: Mapping[tuple[str, int, str], Mapping[str, float]],
) -> dict[str, float]:
    maximum_error = {metric: 0.0 for metric in FROZEN_METRICS}
    failures: list[str] = []
    for severity_index, severity in enumerate(SEVERITIES):
        for seed_index, seed in enumerate(SEEDS):
            for controller, values in (
                ("Fixed Rotation", fixed_values),
                ("PPO", ppo_values),
            ):
                accepted = frozen_cells[(severity, seed, controller)]
                for metric in FROZEN_METRICS:
                    metric_index = METRIC_ORDER.index(metric)
                    actual = _mean(
                        [
                            float(value)
                            for value in values[severity_index, seed_index, :, metric_index]
                        ]
                    )
                    error = abs(actual - accepted[metric])
                    maximum_error[metric] = max(maximum_error[metric], error)
                    tolerance = FROZEN_ABSOLUTE_TOLERANCES[metric]
                    if error > tolerance:
                        failures.append(
                            f"{severity}/{seed}/{controller}/{metric}: "
                            f"expected={accepted[metric]!r}, actual={actual!r}, "
                            f"absolute_error={error!r}, tolerance={tolerance!r}"
                        )
    if failures:
        raise EvaluationArtifactError(
            "regenerated evidence differs from the frozen benchmark; " + "; ".join(failures)
        )
    return maximum_error


def _build_provenance(
    *,
    manifest_path: Path,
    expected_actor_sha256: str,
    frozen_size_bytes: int,
) -> dict[str, object]:
    try:
        manifest_raw = manifest_path.read_bytes()
        actor_path = manifest_path.parent / "ppo-policy.npz"
        actor_size = actor_path.stat().st_size
    except OSError as error:
        raise EvaluationArtifactError(f"PPO provenance cannot be read: {error}") from error
    try:
        actual_actor_sha256 = sha256_file(actor_path)
    except OSError as error:
        raise EvaluationArtifactError(f"PPO actor cannot be hashed: {error}") from error
    if actual_actor_sha256 != expected_actor_sha256:
        raise EvaluationArtifactError("PPO actor changed after its verified CUDA load")
    return {
        "frozen_benchmark": {
            "path": "data/evidence/frozen-benchmark.json",
            "sha256": FROZEN_BENCHMARK_SHA256,
            "size_bytes": frozen_size_bytes,
        },
        "ppo_actor": {
            "path": "data/model/ppo-policy.npz",
            "sha256": expected_actor_sha256,
            "size_bytes": actor_size,
        },
        "ppo_manifest": {
            "path": "data/model/manifest.json",
            "sha256": hashlib.sha256(manifest_raw).hexdigest(),
            "size_bytes": len(manifest_raw),
        },
        "source_checkpoint": {
            "path": SOURCE_CHECKPOINT_PATH,
            "sha256": SOURCE_CHECKPOINT_SHA256,
            "source_revision": SOURCE_REVISION,
        },
    }


def run_full_paired_evaluation(
    *,
    ppo_manifest_path: str | Path,
    expected_ppo_artifact_sha256: str,
    frozen_benchmark_path: str | Path,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 20260815,
    device: str | torch.device = "cuda:0",
) -> dict[str, Any]:
    """Run the locked CUDA protocol and return canonical-ready paired evidence."""

    if bootstrap_replicates != DEFAULT_BOOTSTRAP_REPLICATES:
        raise EvaluationArtifactError("the reviewed protocol requires exactly 10,000 replicates")
    if bootstrap_seed != DEFAULT_BOOTSTRAP_SEED:
        raise EvaluationArtifactError("the reviewed protocol requires bootstrap seed 20260815")
    try:
        resolved_device = torch.device(device)
    except (RuntimeError, TypeError) as error:
        raise EvaluationArtifactError(f"invalid evaluation device: {device!r}") from error
    if resolved_device.type != "cuda" or resolved_device.index != 0:
        raise EvaluationArtifactError("the proposal evaluation requires device cuda:0")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise EvaluationArtifactError("required CUDA device cuda:0 is unavailable")

    manifest_path = Path(ppo_manifest_path)
    frozen_path = Path(frozen_benchmark_path)
    frozen_cells, frozen_size_bytes = _load_frozen_cells(frozen_path)
    policy = load_ppo_policy(
        manifest_path,
        expected_artifact_sha256=expected_ppo_artifact_sha256,
    )
    ppo_controller = PPOController(policy)

    fixed_cells: list[torch.Tensor] = []
    ppo_cells: list[torch.Tensor] = []
    for shortage_level in SEVERITIES:
        for seed in SEEDS:
            scenario = generate_scenarios(
                BATCH_SIZE,
                seed,
                shortage_level,
                device=resolved_device,
            )
            evaluated = evaluate_scenario_pair(scenario, ppo_controller=ppo_controller)
            if (
                evaluated.fixed_result.scenario is not scenario
                or evaluated.ppo_result.scenario is not scenario
            ):
                raise EvaluationArtifactError("controller pairing lost exact Scenario identity")
            fixed_cells.append(_episode_metric_matrix(evaluated.fixed_metrics))
            ppo_cells.append(_episode_metric_matrix(evaluated.ppo_metrics))

    fixed_cube = torch.stack(fixed_cells).reshape(
        len(SEVERITIES), len(SEEDS), BATCH_SIZE, len(METRIC_ORDER)
    )
    ppo_cube = torch.stack(ppo_cells).reshape_as(fixed_cube)
    inference_indices = torch.tensor(
        [METRIC_ORDER.index(metric) for metric in INFERENCE_METRICS],
        dtype=torch.long,
        device=resolved_device,
    )
    # Compute differences from the persisted float32 episode values in float64;
    # this keeps the bootstrap point estimate identical to the serialized
    # ``PPO - Fixed Rotation`` rows rather than rounding each delta to float32.
    paired_deltas = ppo_cube.to(torch.float64) - fixed_cube.to(torch.float64)
    inference_deltas = paired_deltas.index_select(-1, inference_indices).contiguous()
    bootstrap = stratified_paired_bootstrap(
        inference_deltas,
        metric_names=INFERENCE_METRICS,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )

    # This is the only full numeric device-to-host transfer in the evaluator.
    bootstrap_tensor = bootstrap.as_tensor()
    packed = torch.cat(
        (
            fixed_cube.to(torch.float64).flatten(),
            ppo_cube.to(torch.float64).flatten(),
            bootstrap_tensor.flatten(),
        )
    )
    host = packed.cpu().numpy()
    cube_size = fixed_cube.numel()
    bootstrap_size = bootstrap_tensor.numel()
    fixed_host = host[:cube_size].reshape(fixed_cube.shape)
    ppo_host = host[cube_size : 2 * cube_size].reshape(ppo_cube.shape)
    bootstrap_host = host[2 * cube_size : 2 * cube_size + bootstrap_size].reshape(
        3, len(INFERENCE_METRICS)
    )

    maximum_error = _compare_frozen_cells(fixed_host, ppo_host, frozen_cells)
    provenance = _build_provenance(
        manifest_path=manifest_path,
        expected_actor_sha256=expected_ppo_artifact_sha256,
        frozen_size_bytes=frozen_size_bytes,
    )
    return build_paired_benchmark_payload(
        fixed_values=fixed_host,
        ppo_values=ppo_host,
        bootstrap_values=bootstrap_host,
        provenance=provenance,
        frozen_maximum_absolute_error=maximum_error,
    )

"""Documented-seed-disjoint robustness, sensitivity, and forecast evidence."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Final, Literal

import torch

from gridguard.claim_contract import (
    DOCUMENTED_SEED_DISJOINT_ROLE,
    HOLDOUT_CUSTODY_LIMITATION,
    PPO_ATTRIBUTION_LIMITATION,
    SENSITIVITY_INTERVAL_LIMITATION,
)
from gridguard.controllers import PPOController
from gridguard.domain import (
    HORIZON_HOURS,
    LOCALITIES,
    LOCALITY_DEMAND_WEIGHTS,
    SECTOR_SHARES,
    SECTORS,
    Scenario,
    ShortageLevel,
)
from gridguard.evaluation import evaluate_scenario_pair
from gridguard.forecast import causal_demand_projection
from gridguard.metrics import EpisodeMetrics
from gridguard.policy import SOURCE_REVISION, load_ppo_policy
from gridguard.provenance import sha256_file
from gridguard.robustness_inference import (
    BaseControllerCubeContract,
    DeltaCubeContract,
    ForecastDemandCubeContract,
    PowerBalanceCubeContract,
    RobustnessInferenceError,
    SensitivityControllerCubeContract,
    base_controller_cube_to_cuda,
    build_base_controller_cube_payload,
    build_delta_cube_payload,
    build_forecast_demand_cube_payload,
    build_power_balance_cube_payload,
    build_sensitivity_controller_cube_payload,
    decode_base_controller_cube_payload,
    decode_delta_cube_payload,
    decode_forecast_demand_cube_payload,
    decode_power_balance_cube_payload,
    decode_sensitivity_controller_cube_payload,
    delta_cube_to_cuda,
    forecast_demand_cube_to_cuda,
    power_balance_cube_to_cuda,
    reconstruct_bootstrap_intervals,
    reconstruct_controller_statistics,
    sensitivity_controller_cube_to_cuda,
    validate_reconstruction_cube_semantics,
)
from gridguard.robustness_provenance import (
    ROBUSTNESS_IMPLEMENTATION_SOURCE_BINDINGS,
    ROBUSTNESS_LEGACY_IMPLEMENTATION_SOURCE_BINDINGS,
)
from gridguard.robustness_provenance import (
    ROBUSTNESS_SUPPLEMENT_SHA256 as ROBUSTNESS_SUPPLEMENT_SHA256,
)
from gridguard.scenarios import (
    ScenarioBaseDraws,
    ScenarioDemandParameters,
    generate_scenario_base_draws,
    scenario_from_base_draws,
)
from gridguard.statistics import BootstrapError, stratified_paired_bootstrap

ROBUSTNESS_SCHEMA_NAME: Final = "gridguard.robustness_supplement"
ROBUSTNESS_SCHEMA_VERSION: Final = 2
ROBUSTNESS_LEGACY_SCHEMA_VERSION: Final = 1
ROBUSTNESS_EVIDENCE_ROLE: Final = DOCUMENTED_SEED_DISJOINT_ROLE
ROBUSTNESS_LEGACY_EVIDENCE_ROLE: Final = "untouched_confirmatory"
ROBUSTNESS_ARTIFACT_ID: Final = "gridguard-robustness-supplement-v2"
ROBUSTNESS_LEGACY_ARTIFACT_ID: Final = "gridguard-robustness-supplement-v1"
ROBUSTNESS_SOURCE_REVISION: Final = SOURCE_REVISION
ROBUSTNESS_MODEL_MANIFEST_SHA256: Final = (
    "9803563a6c9820f395f832bc2958fa28c634beb7fe4c87363cd659035e033ee8"
)
ROBUSTNESS_MODEL_MANIFEST_SIZE_BYTES: Final = 1_609
ROBUSTNESS_LEGACY_MODEL_MANIFEST_SHA256: Final = (
    "54300578c34ce18fe54aa57e95e216a81d002da43e82ee09e1ae5bd4e1b51349"
)
ROBUSTNESS_LEGACY_MODEL_MANIFEST_SIZE_BYTES: Final = 1_609
ROBUSTNESS_ACTOR_SHA256: Final = "3e4a7ce44267dd1cf3bbde7e77abafb8f31fdea77ece63acda7be17fa3198ad5"
ROBUSTNESS_ACTOR_SIZE_BYTES: Final = 5_160_058
ROBUSTNESS_LEGACY_ACTOR_SHA256: Final = (
    "3e4a7ce44267dd1cf3bbde7e77abafb8f31fdea77ece63acda7be17fa3198ad5"
)
ROBUSTNESS_LEGACY_ACTOR_SIZE_BYTES: Final = 5_160_058
ROBUSTNESS_SCENARIO_GENERATOR_SHA256: Final = (
    "2b875642c7c89af6881fcc6795d5126bf2d439b4bcacd4335daf9ca8d5d34990"
)
ROBUSTNESS_SCENARIO_GENERATOR_SIZE_BYTES: Final = 14_765
ROBUSTNESS_LEGACY_SCENARIO_GENERATOR_SHA256: Final = (
    "e07c7ee9b41ead44991fbf20e64a7c504883175bd3a6a4b6a51ac7ef0305db84"
)
ROBUSTNESS_LEGACY_SCENARIO_GENERATOR_SIZE_BYTES: Final = 12_513
ROBUSTNESS_TRAINING_PROVENANCE_SHA256: Final = (
    "d2568bce26e9221afa8c266113a704f6fbf2370d3bc949f5aa6c72a7411ba1a8"
)
ROBUSTNESS_TRAINING_PROVENANCE_SIZE_BYTES: Final = 29_126
ROBUSTNESS_LEGACY_TRAINING_PROVENANCE_SHA256: Final = (
    "2cfa158a4b26596fd1f155c57fa79a88dc1c41ce8df946792c559184a815128a"
)
ROBUSTNESS_LEGACY_TRAINING_PROVENANCE_SIZE_BYTES: Final = 14_926
ROBUSTNESS_SEEDS: Final = tuple(range(81001, 81011))
ROBUSTNESS_BATCH_SIZE: Final = 32
ROBUSTNESS_BOOTSTRAP_REPLICATES: Final = 10_000
ROBUSTNESS_BOOTSTRAP_SEED: Final = 20260826
ROBUSTNESS_SHORTAGE_LEVELS: Final = ("mild", "moderate", "severe")
PRIMARY_METRICS: Final = (
    "critical_outage_locality_hours",
    "fairness",
    "utilization",
)
SAFETY_METRIC: Final = "power_balance_violation_rate"
FORECAST_METHODS: Final = ("trailing_mean_3h", "persistence_1h")
FORECAST_TARGET_HOURS: Final = tuple(range(3, HORIZON_HOURS))
SENSITIVITY_FACTORS: Final = (0.75, 1.25)
ROBUSTNESS_MAX_OUTAGE_LOCALITY_HOURS: Final = len(LOCALITIES) * HORIZON_HOURS
ROBUSTNESS_LEGACY_LIMITATIONS: Final = (
    "Synthetic demand traces do not establish real-world forecast accuracy.",
    "Sensitivity covers only one-factor-at-a-time ±25% demand-allocation perturbations.",
    "Observed interval direction is descriptive and is not a release gate.",
)
ROBUSTNESS_LIMITATIONS: Final = (
    *ROBUSTNESS_LEGACY_LIMITATIONS[:2],
    SENSITIVITY_INTERVAL_LIMITATION,
    HOLDOUT_CUSTODY_LIMITATION,
    PPO_ATTRIBUTION_LIMITATION,
)
ROBUSTNESS_LEGACY_SEED_SEPARATION_CLAIM: Final = (
    "Untouched means disjoint from every training, validation, and model-selection seed "
    "documented in data/model/training-provenance.yaml; it does not claim knowledge of "
    "undocumented external activity."
)
ROBUSTNESS_SEED_SEPARATION_CLAIM: Final = (
    "Documented-seed-disjoint means disjoint from every training, validation, and "
    "model-selection seed documented in data/model/training-provenance.yaml; it does not "
    "claim knowledge of undocumented external activity."
)
ROBUSTNESS_METRIC_DEFINITIONS: Final = {
    "critical_outage_locality_hours": {
        "unit": "locality-hours",
        "direction": "lower_is_better",
        "role": "primary",
    },
    "fairness": {"unit": "ratio", "direction": "higher_is_better", "role": "primary"},
    "utilization": {"unit": "ratio", "direction": "higher_is_better", "role": "primary"},
    "power_balance_violation_rate": {
        "unit": "ratio",
        "direction": "lower_is_better",
        "role": "safety",
    },
}
ROBUSTNESS_FORECAST_METRIC_DEFINITIONS: Final = {
    "target": {"quantity": "demand", "unit": "MW"},
    "residual": {"formula": "prediction_mw - target_mw", "unit": "MW"},
    "error_sums": {
        "sae": {"formula": "sum(abs(residual_mw))", "unit": "MW"},
        "sse": {"formula": "sum(residual_mw^2)", "unit": "MW^2"},
    },
    "pooled_metrics": {
        "mae": {"formula": "SAE / count", "unit": "MW"},
        "rmse": {"formula": "sqrt(SSE / count)", "unit": "MW"},
    },
    "aggregation": {
        "locality_sector": {
            "pooling": (
                "episode x target_hour x 7 localities x 5 sectors over absolute cell "
                "residuals and squared cell residuals"
            )
        },
        "system_total": {
            "residual": (
                "sum the 35 signed locality-sector residuals before absolute value or squaring"
            ),
            "pooling": "episode x target_hour after system aggregation",
        },
    },
}
_SHA256_PATTERN: Final = re.compile(r"[0-9a-f]{64}")
_GIT_SHA_PATTERN: Final = re.compile(r"[0-9a-f]{40}")
type _IntervalKind = Literal["paired_effect", "difference_in_differences"]


class RobustnessArtifactError(ValueError):
    """Raised when robustness evidence violates its closed scientific contract."""


def _require_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RobustnessArtifactError(f"{field} must be an object")
    return value


def _require_exact_fields(
    value: Mapping[str, object],
    expected: frozenset[str],
    *,
    field: str,
) -> None:
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RobustnessArtifactError(
            f"{field} fields are not closed; missing={missing}, extra={extra}"
        )


def _number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise RobustnessArtifactError(f"{field} must be a number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise RobustnessArtifactError(f"{field} must be finite")
    return normalized


def _positive_integer(value: object, *, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise RobustnessArtifactError(f"{field} must be a positive integer")
    return value


def _expected_integer(value: object, *, field: str, expected: int) -> int:
    if type(value) is not int:
        raise RobustnessArtifactError(f"{field} must be an integer")
    if value != expected:
        raise RobustnessArtifactError(f"{field} does not match the reviewed protocol")
    return value


def _validate_exact_contract(value: object, expected: object, *, field: str) -> None:
    """Compare closed JSON metadata without Python's bool/int/float equality coercions."""

    if isinstance(expected, Mapping):
        actual = _require_mapping(value, field=field)
        _require_exact_fields(actual, frozenset(expected), field=field)
        for name, expected_value in expected.items():
            _validate_exact_contract(actual[name], expected_value, field=f"{field}.{name}")
        return
    if isinstance(expected, list):
        if not isinstance(value, list) or len(value) != len(expected):
            raise RobustnessArtifactError(f"{field} does not match the reviewed protocol")
        for index, (actual_item, expected_item) in enumerate(zip(value, expected, strict=True)):
            _validate_exact_contract(actual_item, expected_item, field=f"{field}[{index}]")
        return
    if type(expected) is int and type(value) is not int:
        raise RobustnessArtifactError(f"{field} must be an integer")
    if type(expected) is bool and type(value) is not bool:
        raise RobustnessArtifactError(f"{field} must be a boolean")
    if type(value) is not type(expected) or value != expected:
        raise RobustnessArtifactError(f"{field} does not match the reviewed protocol")


def _validate_binding(
    value: object,
    *,
    field: str,
    expected_path: str,
    expected_sha256: str,
    expected_size_bytes: int,
) -> Mapping[str, object]:
    binding = _require_mapping(value, field=field)
    _require_exact_fields(
        binding,
        frozenset({"path", "sha256", "size_bytes"}),
        field=field,
    )
    if binding["path"] != expected_path:
        raise RobustnessArtifactError(f"{field}.path must be {expected_path!r}")
    digest = binding["sha256"]
    if type(digest) is not str or not _SHA256_PATTERN.fullmatch(digest):
        raise RobustnessArtifactError(f"{field}.sha256 must be a lowercase SHA-256 digest")
    if digest != expected_sha256:
        raise RobustnessArtifactError(f"{field}.sha256 does not match the pinned source bytes")
    size_bytes = _positive_integer(binding["size_bytes"], field=f"{field}.size_bytes")
    if size_bytes != expected_size_bytes:
        raise RobustnessArtifactError(f"{field}.size_bytes does not match the pinned source bytes")
    return binding


def _validate_implementation_sources(
    value: object,
    *,
    expected_bindings: tuple[tuple[str, str, int], ...],
) -> None:
    if not isinstance(value, list):
        raise RobustnessArtifactError("provenance.implementation_sources must be an array")
    if len(value) != len(expected_bindings):
        raise RobustnessArtifactError(
            "provenance.implementation_sources does not contain the closed source bundle"
        )
    for index, (actual, expected) in enumerate(zip(value, expected_bindings, strict=True)):
        expected_path, expected_sha256, expected_size_bytes = expected
        _validate_binding(
            actual,
            field=f"provenance.implementation_sources[{index}]",
            expected_path=expected_path,
            expected_sha256=expected_sha256,
            expected_size_bytes=expected_size_bytes,
        )


def _seed_separation_payload(
    *, claim_scope: str = ROBUSTNESS_SEED_SEPARATION_CLAIM
) -> dict[str, object]:
    return {
        "claim_scope": claim_scope,
        "robustness_seeds": list(ROBUSTNESS_SEEDS),
        "documented_seed_sets": {
            "behavior_cloning_teacher_episode_seed_range": {
                "start": 42,
                "end_inclusive": 54,
            },
            "behavior_cloning_exploration_prng_seeds": [10_042, 10_043, 10_044],
            "behavior_cloning_shuffle_prng_seeds": [42, 43, 44],
            "candidate_training_seeds": [42, 43, 44],
            "ppo_rollout_episode_seed_range": {
                "formula": "candidate_training_seed + 1000 + iteration",
                "iteration_start": 1,
                "iteration_end_inclusive": 600,
                "start": 1_043,
                "end_inclusive": 1_644,
            },
            "periodic_validation_seeds": [70_000, 70_001, 70_002],
            "accepted_evaluation_promotion_seeds": [50_001, 50_002, 50_003],
        },
        "disjoint_from_all_documented_seeds": True,
    }


def _validate_seed_separation(value: object, *, is_legacy: bool) -> None:
    claim_scope = (
        ROBUSTNESS_LEGACY_SEED_SEPARATION_CLAIM if is_legacy else ROBUSTNESS_SEED_SEPARATION_CLAIM
    )
    _validate_exact_contract(
        value,
        _seed_separation_payload(claim_scope=claim_scope),
        field="provenance.seed_separation",
    )


def canonical_robustness_json_bytes(payload: object) -> bytes:
    """Serialize finite robustness evidence deterministically."""

    try:
        encoded = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise RobustnessArtifactError(
            f"robustness evidence is not canonical JSON: {error}"
        ) from error
    return f"{encoded}\n".encode()


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise RobustnessArtifactError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _validate_protocol(value: object) -> Mapping[str, object]:
    protocol = _require_mapping(value, field="protocol")
    expected: dict[str, object] = {
        "seeds": list(ROBUSTNESS_SEEDS),
        "episodes_per_seed": ROBUSTNESS_BATCH_SIZE,
        "independent_base_draw_count": len(ROBUSTNESS_SEEDS) * ROBUSTNESS_BATCH_SIZE,
        "shortage_levels": list(ROBUSTNESS_SHORTAGE_LEVELS),
        "matched_scenario_count": 960,
        "controllers": ["Fixed Rotation", "PPO"],
        "controller_evaluation_count": 1_920,
        "delta_definition": "PPO - Fixed Rotation",
        "sensitivity_parameterization_count": 25,
        "sensitivity_matched_scenario_count": 24_000,
        "sensitivity_controller_evaluation_count": 48_000,
        "bootstrap": {
            "replicates": ROBUSTNESS_BOOTSTRAP_REPLICATES,
            "seed": ROBUSTNESS_BOOTSTRAP_SEED,
            "ci_percent": 95,
        },
    }
    _validate_exact_contract(protocol, expected, field="protocol")
    return protocol


def _validate_metric_definitions(value: object) -> None:
    definitions = _require_mapping(value, field="metric_definitions")
    if definitions != ROBUSTNESS_METRIC_DEFINITIONS:
        raise RobustnessArtifactError("metric_definitions do not match the canonical contract")


def _validate_forecast_metric_definitions(value: object) -> None:
    try:
        _validate_exact_contract(
            value,
            ROBUSTNESS_FORECAST_METRIC_DEFINITIONS,
            field="forecast_metric_definitions",
        )
    except RobustnessArtifactError as error:
        raise RobustnessArtifactError(
            f"forecast_metric_definitions do not match the canonical contract: {error}"
        ) from error


def _classify_interval(metric: str, lower: float, upper: float) -> str:
    if metric == "critical_outage_locality_hours":
        if upper < 0.0:
            return "favorable"
        if lower > 0.0:
            return "adverse"
        return "ambiguous"
    if lower > 0.0:
        return "favorable"
    if upper < 0.0:
        return "adverse"
    return "ambiguous"


def _validate_interval(
    value: object,
    *,
    field: str,
    metric: str,
    interval_kind: _IntervalKind,
) -> float:
    interval = _require_mapping(value, field=field)
    _require_exact_fields(
        interval,
        frozenset({"mean_delta", "ci95_lower", "ci95_upper", "result_classification"}),
        field=field,
    )
    mean = _number(interval["mean_delta"], field=f"{field}.mean_delta")
    lower = _number(interval["ci95_lower"], field=f"{field}.ci95_lower")
    upper = _number(interval["ci95_upper"], field=f"{field}.ci95_upper")
    paired_limit = (
        float(HORIZON_HOURS * len(LOCALITIES))
        if metric == "critical_outage_locality_hours"
        else 1.0
    )
    feasible_limit = paired_limit * (2.0 if interval_kind == "difference_in_differences" else 1.0)
    for name, endpoint in (("mean_delta", mean), ("ci95_lower", lower), ("ci95_upper", upper)):
        if not -feasible_limit <= endpoint <= feasible_limit:
            raise RobustnessArtifactError(
                f"{field}.{name} is outside {interval_kind} feasible bounds "
                f"[-{feasible_limit:g}, {feasible_limit:g}]"
            )
    if lower > upper:
        raise RobustnessArtifactError(f"{field} confidence interval is reversed")
    expected = _classify_interval(metric, lower, upper)
    if interval["result_classification"] != expected:
        raise RobustnessArtifactError(f"{field} classification does not match its interval")
    return mean


def _validate_paired_inference(
    value: object,
    *,
    field: str,
    interval_kind: _IntervalKind,
) -> dict[str, float]:
    inference = _require_mapping(value, field=field)
    _require_exact_fields(inference, frozenset(PRIMARY_METRICS), field=field)
    return {
        metric: _validate_interval(
            inference[metric],
            field=f"{field}.{metric}",
            metric=metric,
            interval_kind=interval_kind,
        )
        for metric in PRIMARY_METRICS
    }


def _validate_controller_metrics(value: object, *, field: str) -> None:
    metrics = _require_mapping(value, field=field)
    expected = frozenset((*PRIMARY_METRICS, SAFETY_METRIC))
    _require_exact_fields(metrics, expected, field=field)
    outage = _number(
        metrics["critical_outage_locality_hours"],
        field=f"{field}.critical_outage_locality_hours",
    )
    if not 0.0 <= outage <= HORIZON_HOURS * len(LOCALITIES):
        raise RobustnessArtifactError(f"{field}.critical_outage_locality_hours is out of range")
    for metric in ("fairness", "utilization", SAFETY_METRIC):
        ratio = _number(metrics[metric], field=f"{field}.{metric}")
        if not 0.0 <= ratio <= 1.0:
            raise RobustnessArtifactError(f"{field}.{metric} must be between 0 and 1")
    if metrics[SAFETY_METRIC] != 0.0:
        raise RobustnessArtifactError("power balance violation rate must remain zero")


def _validate_base_evaluation(value: object) -> dict[str, float]:
    base = _require_mapping(value, field="base_evaluation")
    _require_exact_fields(
        base,
        frozenset({"controller_means", "paired_inference", "power_balance"}),
        field="base_evaluation",
    )
    controller_means = _require_mapping(
        base["controller_means"], field="base_evaluation.controller_means"
    )
    _require_exact_fields(
        controller_means,
        frozenset({"Fixed Rotation", "PPO"}),
        field="base_evaluation.controller_means",
    )
    for controller in ("Fixed Rotation", "PPO"):
        _validate_controller_metrics(
            controller_means[controller],
            field=f"base_evaluation.controller_means.{controller}",
        )
    paired_means = _validate_paired_inference(
        base["paired_inference"],
        field="base_evaluation.paired_inference",
        interval_kind="paired_effect",
    )
    fixed = _require_mapping(
        controller_means["Fixed Rotation"],
        field="base_evaluation.controller_means.Fixed Rotation",
    )
    ppo = _require_mapping(
        controller_means["PPO"],
        field="base_evaluation.controller_means.PPO",
    )
    for metric in PRIMARY_METRICS:
        expected_delta = _number(ppo[metric], field=f"PPO.{metric}") - _number(
            fixed[metric], field=f"Fixed Rotation.{metric}"
        )
        if not math.isclose(
            paired_means[metric],
            expected_delta,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RobustnessArtifactError(
                f"base_evaluation paired {metric} mean does not match controller means"
            )
    power_balance = _require_mapping(base["power_balance"], field="base_evaluation.power_balance")
    _require_exact_fields(
        power_balance,
        frozenset({"maximum_violation_rate"}),
        field="base_evaluation.power_balance",
    )
    maximum = _number(
        power_balance["maximum_violation_rate"],
        field="base_evaluation.power_balance.maximum_violation_rate",
    )
    if maximum != 0.0:
        raise RobustnessArtifactError("power balance maximum violation rate must remain zero")
    return paired_means


def _validate_parameter_values(
    value: object,
    *,
    field: str,
    expected: ScenarioDemandParameters,
) -> None:
    parameters = _require_mapping(value, field=field)
    _require_exact_fields(
        parameters,
        frozenset({"locality_weights", "sector_shares"}),
        field=field,
    )
    for name, expected_values in (
        ("locality_weights", expected.locality_weights),
        ("sector_shares", expected.sector_shares),
    ):
        actual_values = parameters[name]
        if not isinstance(actual_values, list) or len(actual_values) != len(expected_values):
            raise RobustnessArtifactError(f"{field}.{name} has the wrong shape")
        normalized = tuple(
            _number(item, field=f"{field}.{name}[{index}]")
            for index, item in enumerate(actual_values)
        )
        if any(value <= 0.0 for value in normalized):
            raise RobustnessArtifactError(f"{field}.{name} values must be positive")
        if any(
            not math.isclose(actual, target, rel_tol=0.0, abs_tol=1e-12)
            for actual, target in zip(normalized, expected_values, strict=True)
        ):
            raise RobustnessArtifactError(f"{field}.{name} does not match the reviewed variant")


def _validate_sensitivity(
    value: object,
    *,
    base_paired_means: Mapping[str, float],
) -> None:
    sensitivity = _require_mapping(value, field="sensitivity")
    _require_exact_fields(
        sensitivity,
        frozenset({"base_parameters", "variants"}),
        field="sensitivity",
    )
    base_parameters = ScenarioDemandParameters(
        locality_weights=LOCALITY_DEMAND_WEIGHTS,
        sector_shares=SECTOR_SHARES,
    )
    _validate_parameter_values(
        sensitivity["base_parameters"],
        field="sensitivity.base_parameters",
        expected=base_parameters,
    )
    variants = sensitivity["variants"]
    if not isinstance(variants, list) or len(variants) != 24:
        raise RobustnessArtifactError("sensitivity must contain exactly 24 variants")
    expected_variants = build_sensitivity_variants()
    for index, (actual_value, expected) in enumerate(zip(variants, expected_variants, strict=True)):
        field = f"sensitivity.variants[{index}]"
        actual = _require_mapping(actual_value, field=field)
        _require_exact_fields(
            actual,
            frozenset(
                {
                    "variant_id",
                    "parameter",
                    "component_index",
                    "component_name",
                    "factor",
                    "parameters",
                    "paired_inference",
                    "difference_in_differences",
                }
            ),
            field=field,
        )
        metadata = {
            "variant_id": expected.variant_id,
            "parameter": expected.parameter,
            "component_index": expected.component_index,
            "component_name": expected.component_name,
            "factor": expected.factor,
        }
        for name, expected_value in metadata.items():
            actual_value = actual[name]
            if name == "component_index" and type(actual_value) is not int:
                raise RobustnessArtifactError(f"{field}.component_index must be an integer")
            if name == "factor":
                actual_value = _number(actual_value, field=f"{field}.factor")
            if actual_value != expected_value:
                raise RobustnessArtifactError(f"{field}.{name} does not match reviewed order")
        _validate_parameter_values(
            actual["parameters"],
            field=f"{field}.parameters",
            expected=expected.parameters,
        )
        paired_means = _validate_paired_inference(
            actual["paired_inference"],
            field=f"{field}.paired_inference",
            interval_kind="paired_effect",
        )
        difference_means = _validate_paired_inference(
            actual["difference_in_differences"],
            field=f"{field}.difference_in_differences",
            interval_kind="difference_in_differences",
        )
        for metric in PRIMARY_METRICS:
            expected_difference = paired_means[metric] - base_paired_means[metric]
            if not math.isclose(
                difference_means[metric],
                expected_difference,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise RobustnessArtifactError(
                    f"{field} {metric} difference-in-differences mean is inconsistent"
                )


def _floating_reduction_tolerance(*values: float) -> float:
    """Allow 64 ULPs for independent float64 reductions and host arithmetic."""

    return 64.0 * max(math.ulp(abs(value)) for value in (*values, 1.0))


def _validate_forecast_sums(
    value: object,
    *,
    field: str,
    expected_count: int,
) -> tuple[int, float, float]:
    sums = _require_mapping(value, field=field)
    _require_exact_fields(
        sums,
        frozenset({"count", "sum_absolute_error", "sum_squared_error"}),
        field=field,
    )
    count = _positive_integer(sums["count"], field=f"{field}.count")
    if count != expected_count:
        raise RobustnessArtifactError(f"{field}.count does not match the aligned target count")
    sum_absolute = _number(sums["sum_absolute_error"], field=f"{field}.sum_absolute_error")
    sum_squared = _number(sums["sum_squared_error"], field=f"{field}.sum_squared_error")
    if sum_absolute < 0.0 or sum_squared < 0.0:
        raise RobustnessArtifactError(f"{field} error sums must be non-negative")
    sum_absolute_squared = sum_absolute * sum_absolute
    if not math.isfinite(sum_absolute_squared):
        raise RobustnessArtifactError(f"{field}.sum_absolute_error is too large")
    lower_bound = sum_absolute_squared / count
    lower_tolerance = _floating_reduction_tolerance(lower_bound, sum_squared)
    upper_tolerance = _floating_reduction_tolerance(sum_absolute_squared, sum_squared)
    if sum_squared < lower_bound - lower_tolerance:
        raise RobustnessArtifactError(f"{field} violates error-sum inequality SAE^2/count <= SSE")
    if sum_squared > sum_absolute_squared + upper_tolerance:
        raise RobustnessArtifactError(f"{field} violates error-sum inequality SSE <= SAE^2")
    return count, sum_absolute, sum_squared


def _validate_forecast_method(
    value: object,
    *,
    field: str,
    expected_window: int,
) -> dict[str, float]:
    method = _require_mapping(value, field=field)
    _require_exact_fields(
        method,
        frozenset({"window_hours", "per_seed", "pooled"}),
        field=field,
    )
    _expected_integer(
        method["window_hours"],
        field=f"{field}.window_hours",
        expected=expected_window,
    )
    per_seed = method["per_seed"]
    if not isinstance(per_seed, list) or len(per_seed) != len(ROBUSTNESS_SEEDS):
        raise RobustnessArtifactError(f"{field}.per_seed must contain ten seed summaries")
    totals = {
        "locality_sector": [0, 0.0, 0.0],
        "system_total": [0, 0.0, 0.0],
    }
    for seed_index, (seed_summary_value, expected_seed) in enumerate(
        zip(per_seed, ROBUSTNESS_SEEDS, strict=True)
    ):
        seed_field = f"{field}.per_seed[{seed_index}]"
        seed_summary = _require_mapping(seed_summary_value, field=seed_field)
        _require_exact_fields(
            seed_summary,
            frozenset({"seed", "locality_sector", "system_total"}),
            field=seed_field,
        )
        _expected_integer(
            seed_summary["seed"],
            field=f"{seed_field}.seed",
            expected=expected_seed,
        )
        seed_values: dict[str, tuple[int, float, float]] = {}
        for aggregation, expected_count in (
            ("locality_sector", ROBUSTNESS_BATCH_SIZE * len(FORECAST_TARGET_HOURS) * 35),
            ("system_total", ROBUSTNESS_BATCH_SIZE * len(FORECAST_TARGET_HOURS)),
        ):
            values = _validate_forecast_sums(
                seed_summary[aggregation],
                field=f"{seed_field}.{aggregation}",
                expected_count=expected_count,
            )
            seed_values[aggregation] = values
            for index, item in enumerate(values):
                totals[aggregation][index] += item
        _, locality_sae, locality_sse = seed_values["locality_sector"]
        _, system_sae, system_sse = seed_values["system_total"]
        sae_tolerance = _floating_reduction_tolerance(locality_sae, system_sae)
        if system_sae > locality_sae + sae_tolerance:
            raise RobustnessArtifactError(f"{seed_field} system SAE exceeds locality-sector SAE")
        system_sse_limit = len(LOCALITIES) * len(SECTORS) * locality_sse
        sse_tolerance = _floating_reduction_tolerance(system_sse_limit, system_sse)
        if system_sse > system_sse_limit + sse_tolerance:
            raise RobustnessArtifactError(
                f"{seed_field} system SSE exceeds 35 times locality-sector SSE"
            )

    pooled = _require_mapping(method["pooled"], field=f"{field}.pooled")
    _require_exact_fields(
        pooled,
        frozenset({"locality_sector", "system_total"}),
        field=f"{field}.pooled",
    )
    measurements: dict[str, float] = {}
    for aggregation in ("locality_sector", "system_total"):
        aggregate_field = f"{field}.pooled.{aggregation}"
        aggregate = _require_mapping(pooled[aggregation], field=aggregate_field)
        _require_exact_fields(
            aggregate,
            frozenset({"count", "mae", "rmse"}),
            field=aggregate_field,
        )
        total_count, absolute_sum, squared_sum = totals[aggregation]
        count = _positive_integer(aggregate["count"], field=f"{aggregate_field}.count")
        if count != total_count:
            raise RobustnessArtifactError(f"{aggregate_field}.count does not match per-seed sums")
        expected_mae = absolute_sum / total_count
        expected_rmse = math.sqrt(squared_sum / total_count)
        for measure, expected_value in (("mae", expected_mae), ("rmse", expected_rmse)):
            actual = _number(aggregate[measure], field=f"{aggregate_field}.{measure}")
            if not math.isclose(actual, expected_value, rel_tol=0.0, abs_tol=1e-12):
                raise RobustnessArtifactError(
                    f"{field} pooled {aggregation} {measure} does not match per-seed sums"
                )
            measurements[f"{aggregation}_{measure}"] = actual
    return measurements


def _validate_forecast_evaluation(value: object) -> None:
    forecast = _require_mapping(value, field="forecast_evaluation")
    _require_exact_fields(
        forecast,
        frozenset(
            {
                "target_hours",
                "trace_count",
                "locality_sector_prediction_count",
                "system_total_prediction_count",
                "methods",
                "comparison",
            }
        ),
        field="forecast_evaluation",
    )
    _validate_exact_contract(
        forecast["target_hours"],
        list(FORECAST_TARGET_HOURS),
        field="forecast_evaluation.target_hours",
    )
    for name, expected in (
        ("trace_count", len(ROBUSTNESS_SEEDS) * ROBUSTNESS_BATCH_SIZE),
        ("locality_sector_prediction_count", 235_200),
        ("system_total_prediction_count", 6_720),
    ):
        _expected_integer(forecast[name], field=f"forecast_evaluation.{name}", expected=expected)
    methods = _require_mapping(forecast["methods"], field="forecast_evaluation.methods")
    _require_exact_fields(methods, frozenset(FORECAST_METHODS), field="forecast_evaluation.methods")
    measurements = {
        "trailing_mean_3h": _validate_forecast_method(
            methods["trailing_mean_3h"],
            field="forecast_evaluation.methods.trailing_mean_3h",
            expected_window=3,
        ),
        "persistence_1h": _validate_forecast_method(
            methods["persistence_1h"],
            field="forecast_evaluation.methods.persistence_1h",
            expected_window=1,
        ),
    }
    comparison = _require_mapping(forecast["comparison"], field="forecast_evaluation.comparison")
    _require_exact_fields(
        comparison,
        frozenset({"trailing_minus_persistence", "lower_error_method"}),
        field="forecast_evaluation.comparison",
    )
    deltas = _require_mapping(
        comparison["trailing_minus_persistence"],
        field="forecast_evaluation.comparison.trailing_minus_persistence",
    )
    winners = _require_mapping(
        comparison["lower_error_method"],
        field="forecast_evaluation.comparison.lower_error_method",
    )
    measurement_names = frozenset(measurements["trailing_mean_3h"])
    _require_exact_fields(
        deltas,
        measurement_names,
        field="forecast_evaluation.comparison.trailing_minus_persistence",
    )
    _require_exact_fields(
        winners,
        measurement_names,
        field="forecast_evaluation.comparison.lower_error_method",
    )
    for measure in sorted(measurement_names):
        expected_delta = (
            measurements["trailing_mean_3h"][measure] - measurements["persistence_1h"][measure]
        )
        actual_delta = _number(
            deltas[measure],
            field=f"forecast_evaluation.comparison.trailing_minus_persistence.{measure}",
        )
        if not math.isclose(actual_delta, expected_delta, rel_tol=0.0, abs_tol=1e-12):
            raise RobustnessArtifactError(f"forecast comparison delta for {measure} is incorrect")
        if math.isclose(expected_delta, 0.0, rel_tol=0.0, abs_tol=1e-12):
            expected_winner = "tie"
        elif expected_delta < 0.0:
            expected_winner = "trailing_mean_3h"
        else:
            expected_winner = "persistence_1h"
        if winners[measure] != expected_winner:
            raise RobustnessArtifactError(f"forecast lower-error method for {measure} is incorrect")


def _host_array(value: object, *, shape: tuple[int, ...], field: str) -> Any:
    # NumPy is intentionally imported only in portable artifact assembly, after
    # the evaluator's single final tensor transfer.
    import numpy as np

    untyped = np.asarray(value)
    if np.iscomplexobj(untyped):
        raise RobustnessArtifactError(f"{field} must contain real values")
    array = np.asarray(untyped, dtype=np.float64)
    if array.shape != shape:
        raise RobustnessArtifactError(f"{field} must have shape {list(shape)}")
    if not bool(np.isfinite(array).all()):
        raise RobustnessArtifactError(f"{field} must contain only finite values")
    return array


def _interval_payload(summary: Any, metric: str, metric_index: int) -> dict[str, object]:
    mean = float(summary[0, metric_index])
    lower = float(summary[1, metric_index])
    upper = float(summary[2, metric_index])
    return {
        "mean_delta": mean,
        "ci95_lower": lower,
        "ci95_upper": upper,
        "result_classification": _classify_interval(metric, lower, upper),
    }


def _paired_inference_payload(summary: Any) -> dict[str, object]:
    return {
        metric: _interval_payload(summary, metric, metric_index)
        for metric_index, metric in enumerate(PRIMARY_METRICS)
    }


def _forecast_payload(statistics: Any) -> dict[str, object]:
    methods: dict[str, object] = {}
    pooled_measurements: dict[str, dict[str, float]] = {}
    for method_index, (method_name, window_hours) in enumerate(
        (("trailing_mean_3h", 3), ("persistence_1h", 1))
    ):
        per_seed = []
        totals = {
            "locality_sector": [0, 0.0, 0.0],
            "system_total": [0, 0.0, 0.0],
        }
        for seed_index, seed in enumerate(ROBUSTNESS_SEEDS):
            summary: dict[str, object] = {"seed": seed}
            for aggregation_index, aggregation in enumerate(("locality_sector", "system_total")):
                raw_count, sum_absolute, sum_squared = statistics[
                    method_index, seed_index, aggregation_index
                ]
                count = int(raw_count)
                if raw_count != count:
                    raise RobustnessArtifactError("forecast count left the integer contract")
                values = {
                    "count": count,
                    "sum_absolute_error": float(sum_absolute),
                    "sum_squared_error": float(sum_squared),
                }
                summary[aggregation] = values
                totals[aggregation][0] += count
                totals[aggregation][1] += float(sum_absolute)
                totals[aggregation][2] += float(sum_squared)
            per_seed.append(summary)

        pooled: dict[str, object] = {}
        method_measurements: dict[str, float] = {}
        for aggregation, (count, sum_absolute, sum_squared) in totals.items():
            mae = sum_absolute / count
            rmse = math.sqrt(sum_squared / count)
            pooled[aggregation] = {"count": count, "mae": mae, "rmse": rmse}
            method_measurements[f"{aggregation}_mae"] = mae
            method_measurements[f"{aggregation}_rmse"] = rmse
        methods[method_name] = {
            "window_hours": window_hours,
            "per_seed": per_seed,
            "pooled": pooled,
        }
        pooled_measurements[method_name] = method_measurements

    deltas: dict[str, float] = {}
    winners: dict[str, str] = {}
    for measure in pooled_measurements["trailing_mean_3h"]:
        delta = (
            pooled_measurements["trailing_mean_3h"][measure]
            - pooled_measurements["persistence_1h"][measure]
        )
        deltas[measure] = delta
        if math.isclose(delta, 0.0, rel_tol=0.0, abs_tol=1e-12):
            winners[measure] = "tie"
        elif delta < 0.0:
            winners[measure] = "trailing_mean_3h"
        else:
            winners[measure] = "persistence_1h"
    return {
        "target_hours": list(FORECAST_TARGET_HOURS),
        "trace_count": len(ROBUSTNESS_SEEDS) * ROBUSTNESS_BATCH_SIZE,
        "locality_sector_prediction_count": 235_200,
        "system_total_prediction_count": 6_720,
        "methods": methods,
        "comparison": {
            "trailing_minus_persistence": deltas,
            "lower_error_method": winners,
        },
    }


def build_robustness_supplement_payload(
    *,
    provenance: Mapping[str, object],
    base_controller_means: object,
    base_bootstrap: object,
    sensitivity_paired_bootstrap: object,
    sensitivity_difference_bootstrap: object,
    bootstrap_delta_cube: object,
    base_controller_metric_cube: object,
    sensitivity_controller_metric_cube: object,
    power_balance_cube: object,
    forecast_demand_cube: object,
    forecast_statistics: object,
    maximum_power_balance_violation_rate: float,
) -> dict[str, Any]:
    """Assemble the canonical host-side document from the final packed transfer."""

    base_means = _host_array(
        base_controller_means,
        shape=(2, 4),
        field="base_controller_means",
    )
    base_intervals = _host_array(base_bootstrap, shape=(3, 3), field="base_bootstrap")
    sensitivity_intervals = _host_array(
        sensitivity_paired_bootstrap,
        shape=(24, 3, 3),
        field="sensitivity_paired_bootstrap",
    )
    difference_intervals = _host_array(
        sensitivity_difference_bootstrap,
        shape=(24, 3, 3),
        field="sensitivity_difference_bootstrap",
    )
    delta_contract = robustness_delta_cube_contract()
    delta_cube = _host_array(
        bootstrap_delta_cube,
        shape=delta_contract.shape,
        field="bootstrap_delta_cube",
    )
    controller_contract = robustness_base_controller_cube_contract()
    controller_cube = _host_array(
        base_controller_metric_cube,
        shape=controller_contract.shape,
        field="base_controller_metric_cube",
    )
    sensitivity_controller_contract = robustness_sensitivity_controller_cube_contract()
    sensitivity_controller_cube = _host_array(
        sensitivity_controller_metric_cube,
        shape=sensitivity_controller_contract.shape,
        field="sensitivity_controller_metric_cube",
    )
    power_contract = robustness_power_balance_cube_contract()
    power_cube = _host_array(
        power_balance_cube,
        shape=power_contract.shape,
        field="power_balance_cube",
    )
    forecast_contract = robustness_forecast_demand_cube_contract()
    forecast_demand_values = _host_array(
        forecast_demand_cube,
        shape=forecast_contract.shape,
        field="forecast_demand_cube",
    )
    forecast_values = _host_array(
        forecast_statistics,
        shape=(2, len(ROBUSTNESS_SEEDS), 2, 3),
        field="forecast_statistics",
    )

    controller_means = {}
    metric_order = (*PRIMARY_METRICS, SAFETY_METRIC)
    for controller_index, controller in enumerate(("Fixed Rotation", "PPO")):
        controller_means[controller] = {
            metric: float(base_means[controller_index, metric_index])
            for metric_index, metric in enumerate(metric_order)
        }
    variants = []
    for variant_index, variant in enumerate(build_sensitivity_variants()):
        variants.append(
            {
                "variant_id": variant.variant_id,
                "parameter": variant.parameter,
                "component_index": variant.component_index,
                "component_name": variant.component_name,
                "factor": variant.factor,
                "parameters": {
                    "locality_weights": list(variant.parameters.locality_weights),
                    "sector_shares": list(variant.parameters.sector_shares),
                },
                "paired_inference": _paired_inference_payload(sensitivity_intervals[variant_index]),
                "difference_in_differences": _paired_inference_payload(
                    difference_intervals[variant_index]
                ),
            }
        )

    payload: dict[str, Any] = {
        "schema": {"name": ROBUSTNESS_SCHEMA_NAME, "version": ROBUSTNESS_SCHEMA_VERSION},
        "evidence_role": ROBUSTNESS_EVIDENCE_ROLE,
        "provenance": dict(provenance),
        "protocol": {
            "seeds": list(ROBUSTNESS_SEEDS),
            "episodes_per_seed": ROBUSTNESS_BATCH_SIZE,
            "independent_base_draw_count": 320,
            "shortage_levels": list(ROBUSTNESS_SHORTAGE_LEVELS),
            "matched_scenario_count": 960,
            "controllers": ["Fixed Rotation", "PPO"],
            "controller_evaluation_count": 1_920,
            "delta_definition": "PPO - Fixed Rotation",
            "bootstrap": {
                "replicates": ROBUSTNESS_BOOTSTRAP_REPLICATES,
                "seed": ROBUSTNESS_BOOTSTRAP_SEED,
                "ci_percent": 95,
            },
            "sensitivity_parameterization_count": 25,
            "sensitivity_matched_scenario_count": 24_000,
            "sensitivity_controller_evaluation_count": 48_000,
        },
        "metric_definitions": {
            metric: dict(definition) for metric, definition in ROBUSTNESS_METRIC_DEFINITIONS.items()
        },
        "forecast_metric_definitions": deepcopy(ROBUSTNESS_FORECAST_METRIC_DEFINITIONS),
        "bootstrap_inputs": {
            "paired_delta_cube": build_delta_cube_payload(delta_cube, contract=delta_contract),
            "base_controller_cube": build_base_controller_cube_payload(
                controller_cube,
                contract=controller_contract,
            ),
            "sensitivity_controller_cube": build_sensitivity_controller_cube_payload(
                sensitivity_controller_cube,
                contract=sensitivity_controller_contract,
            ),
            "power_balance_cube": build_power_balance_cube_payload(
                power_cube,
                contract=power_contract,
            ),
        },
        "forecast_inputs": {
            "demand_cube": build_forecast_demand_cube_payload(
                forecast_demand_values,
                contract=forecast_contract,
            )
        },
        "base_evaluation": {
            "controller_means": controller_means,
            "paired_inference": _paired_inference_payload(base_intervals),
            "power_balance": {
                "maximum_violation_rate": float(maximum_power_balance_violation_rate)
            },
        },
        "sensitivity": {
            "base_parameters": {
                "locality_weights": list(LOCALITY_DEMAND_WEIGHTS),
                "sector_shares": list(SECTOR_SHARES),
            },
            "variants": variants,
        },
        "forecast_evaluation": _forecast_payload(forecast_values),
        "limitations": list(ROBUSTNESS_LIMITATIONS),
    }
    validate_robustness_supplement_payload(payload)
    return payload


def _robustness_metric_matrix(metrics: EpisodeMetrics) -> torch.Tensor:
    values = metrics.as_dict()
    matrix = torch.stack(
        tuple(values[metric] for metric in (*PRIMARY_METRICS, SAFETY_METRIC)),
        dim=-1,
    )
    if matrix.device.type != "cuda" or matrix.device.index != 0:
        raise RobustnessArtifactError("robustness metrics left required CUDA device cuda:0")
    return matrix


def _file_binding(path: Path, *, relative_path: str) -> dict[str, object]:
    try:
        size_bytes = path.stat().st_size
        digest = sha256_file(path)
    except OSError as error:
        raise RobustnessArtifactError(
            f"provenance input is unavailable: {path}: {error}"
        ) from error
    return {"path": relative_path, "sha256": digest, "size_bytes": size_bytes}


def verify_robustness_implementation_sources(
    repository_root: str | Path,
) -> tuple[dict[str, object], ...]:
    """Re-hash the current checkout against the non-circular source trust anchor."""

    root = Path(repository_root).expanduser().resolve()
    verified: list[dict[str, object]] = []
    for index, (relative_path, expected_sha256, expected_size_bytes) in enumerate(
        ROBUSTNESS_IMPLEMENTATION_SOURCE_BINDINGS
    ):
        candidate = (root / relative_path).resolve()
        if not candidate.is_relative_to(root):
            raise RobustnessArtifactError(
                f"implementation source escapes repository root: {relative_path}"
            )
        binding = _file_binding(candidate, relative_path=relative_path)
        _validate_binding(
            binding,
            field=f"implementation_sources[{index}]",
            expected_path=relative_path,
            expected_sha256=expected_sha256,
            expected_size_bytes=expected_size_bytes,
        )
        verified.append(binding)
    return tuple(verified)


def _build_robustness_provenance(manifest_path: Path) -> dict[str, object]:
    if manifest_path.parent.name != "model" or manifest_path.parent.parent.name != "data":
        raise RobustnessArtifactError(
            "PPO manifest must be located at data/model/manifest.json for reproducible provenance"
        )
    repository_root = manifest_path.parent.parent.parent
    scenario_generator_path = repository_root / "src" / "gridguard" / "scenarios.py"
    actor_path = manifest_path.parent / "ppo-policy.npz"
    training_provenance_path = manifest_path.parent / "training-provenance.yaml"
    return {
        "artifact_id": ROBUSTNESS_ARTIFACT_ID,
        "source_revision": SOURCE_REVISION,
        "model_manifest": _file_binding(
            manifest_path,
            relative_path="data/model/manifest.json",
        ),
        "actor": _file_binding(
            actor_path,
            relative_path="data/model/ppo-policy.npz",
        ),
        "scenario_generator": _file_binding(
            scenario_generator_path,
            relative_path="src/gridguard/scenarios.py",
        ),
        "training_provenance": _file_binding(
            training_provenance_path,
            relative_path="data/model/training-provenance.yaml",
        ),
        "implementation_sources": [
            _file_binding(repository_root / relative_path, relative_path=relative_path)
            for relative_path, _, _ in ROBUSTNESS_IMPLEMENTATION_SOURCE_BINDINGS
        ],
        "seed_separation": _seed_separation_payload(),
    }


def _scenario_for_demand_allocation_sensitivity(
    *,
    base_draws: ScenarioBaseDraws,
    shortage_level: ShortageLevel | str,
    demand_parameters: ScenarioDemandParameters,
    reference_scenario: Scenario,
) -> Scenario:
    """Change allocation parameters while holding hourly demand and supply fixed."""

    if not isinstance(base_draws, ScenarioBaseDraws):
        raise TypeError("base_draws must be ScenarioBaseDraws")
    if not isinstance(reference_scenario, Scenario):
        raise TypeError("reference_scenario must be Scenario")
    try:
        normalized_shortage = ShortageLevel(shortage_level)
    except (TypeError, ValueError) as error:
        raise ValueError("shortage_level must be mild, moderate, or severe") from error
    if reference_scenario.shortage_level is not normalized_shortage:
        raise ValueError("reference_scenario must use the requested shortage level")
    if reference_scenario.seed != base_draws.seed:
        raise ValueError("reference_scenario must use the same base-draw seed")
    return scenario_from_base_draws(
        base_draws,
        normalized_shortage,
        demand_parameters=demand_parameters,
        total_demand_override_mw=reference_scenario.demand.sum(dim=(-1, -2)),
        available_supply_override_mw=reference_scenario.supply,
    )


def _require_cuda_zero(device: str | torch.device) -> torch.device:
    try:
        resolved = torch.device(device)
    except (RuntimeError, TypeError) as error:
        raise RobustnessArtifactError(
            f"invalid robustness evaluation device: {device!r}"
        ) from error
    if resolved.type != "cuda" or resolved.index != 0:
        raise RobustnessArtifactError("robustness evaluation requires device cuda:0")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RobustnessArtifactError("required CUDA device cuda:0 is unavailable")
    return resolved


@torch.no_grad()
def run_full_robustness_evaluation(
    *,
    ppo_manifest_path: str | Path,
    expected_ppo_artifact_sha256: str,
    device: str | torch.device = "cuda:0",
) -> dict[str, Any]:
    """Run the locked CUDA robustness protocol and return canonical-ready evidence."""

    resolved_device = _require_cuda_zero(device)
    manifest_path = Path(ppo_manifest_path).expanduser().resolve()
    policy = load_ppo_policy(
        manifest_path,
        expected_artifact_sha256=expected_ppo_artifact_sha256,
    )
    ppo_controller = PPOController(policy)
    provenance = _build_robustness_provenance(manifest_path)
    base_draws = {
        seed: generate_scenario_base_draws(
            ROBUSTNESS_BATCH_SIZE,
            seed,
            device=resolved_device,
        )
        for seed in ROBUSTNESS_SEEDS
    }

    reference_scenarios: dict[tuple[str, int], Scenario] = {}
    fixed_cells: list[torch.Tensor] = []
    ppo_cells: list[torch.Tensor] = []
    forecast_demands: list[torch.Tensor] = []
    for severity_index, shortage_level in enumerate(ROBUSTNESS_SHORTAGE_LEVELS):
        for seed in ROBUSTNESS_SEEDS:
            scenario = scenario_from_base_draws(base_draws[seed], shortage_level)
            reference_scenarios[(shortage_level, seed)] = scenario
            if severity_index == 0:
                forecast_demands.append(scenario.demand)
            evaluated = evaluate_scenario_pair(scenario, ppo_controller=ppo_controller)
            if (
                evaluated.fixed_result.scenario is not scenario
                or evaluated.ppo_result.scenario is not scenario
            ):
                raise RobustnessArtifactError("controller pairing lost exact Scenario identity")
            fixed_cells.append(_robustness_metric_matrix(evaluated.fixed_metrics))
            ppo_cells.append(_robustness_metric_matrix(evaluated.ppo_metrics))

    base_shape = (
        len(ROBUSTNESS_SHORTAGE_LEVELS),
        len(ROBUSTNESS_SEEDS),
        ROBUSTNESS_BATCH_SIZE,
        len(PRIMARY_METRICS) + 1,
    )
    fixed_cube = torch.stack(fixed_cells).reshape(base_shape)
    ppo_cube = torch.stack(ppo_cells).reshape(base_shape)
    base_controller_cube = torch.stack((fixed_cube, ppo_cube)).to(torch.float64)
    base_deltas = ppo_cube[..., :3].to(torch.float64) - fixed_cube[..., :3].to(torch.float64)
    base_bootstrap = stratified_paired_bootstrap(
        base_deltas,
        metric_names=PRIMARY_METRICS,
        replicates=ROBUSTNESS_BOOTSTRAP_REPLICATES,
        seed=ROBUSTNESS_BOOTSTRAP_SEED,
    ).as_tensor()
    base_means = base_controller_cube.mean(dim=(1, 2, 3))
    forecast_demand_tensor = torch.stack(forecast_demands)
    forecast_validation = forecast_input_validation_flags(forecast_demand_tensor)
    forecast_statistics = forecast_error_sufficient_statistics(forecast_demand_tensor)

    sensitivity_controller_cubes: list[torch.Tensor] = []
    for variant in build_sensitivity_variants():
        controller_cells: list[torch.Tensor] = []
        for shortage_level in ROBUSTNESS_SHORTAGE_LEVELS:
            for seed in ROBUSTNESS_SEEDS:
                scenario = _scenario_for_demand_allocation_sensitivity(
                    base_draws=base_draws[seed],
                    shortage_level=shortage_level,
                    demand_parameters=variant.parameters,
                    reference_scenario=reference_scenarios[(shortage_level, seed)],
                )
                evaluated = evaluate_scenario_pair(scenario, ppo_controller=ppo_controller)
                fixed_metrics = _robustness_metric_matrix(evaluated.fixed_metrics)
                ppo_metrics = _robustness_metric_matrix(evaluated.ppo_metrics)
                controller_cells.append(
                    torch.stack((fixed_metrics, ppo_metrics), dim=0).to(torch.float64)
                )
        sensitivity_controller_cubes.append(
            torch.stack(controller_cells)
            .reshape(
                len(ROBUSTNESS_SHORTAGE_LEVELS),
                len(ROBUSTNESS_SEEDS),
                2,
                ROBUSTNESS_BATCH_SIZE,
                len(PRIMARY_METRICS) + 1,
            )
            .permute(2, 0, 1, 3, 4)
        )

    sensitivity_controller_cube = torch.stack(sensitivity_controller_cubes)
    variant_deltas = (
        sensitivity_controller_cube[:, 1, ..., : len(PRIMARY_METRICS)]
        - sensitivity_controller_cube[:, 0, ..., : len(PRIMARY_METRICS)]
    )

    sensitivity_summaries: list[torch.Tensor] = []
    difference_summaries: list[torch.Tensor] = []
    for paired_deltas in variant_deltas:
        sensitivity_summaries.append(
            stratified_paired_bootstrap(
                paired_deltas,
                metric_names=PRIMARY_METRICS,
                replicates=ROBUSTNESS_BOOTSTRAP_REPLICATES,
                seed=ROBUSTNESS_BOOTSTRAP_SEED,
            ).as_tensor()
        )
        difference_summaries.append(
            stratified_paired_bootstrap(
                paired_deltas - base_deltas,
                metric_names=PRIMARY_METRICS,
                replicates=ROBUSTNESS_BOOTSTRAP_REPLICATES,
                seed=ROBUSTNESS_BOOTSTRAP_SEED,
            ).as_tensor()
        )
    power_balance_cube = torch.cat(
        (
            base_controller_cube[..., 3].unsqueeze(0),
            sensitivity_controller_cube[..., 3],
        ),
        dim=0,
    )
    maximum_power_balance = power_balance_cube.amax().reshape(1)
    sensitivity_tensor = torch.stack(sensitivity_summaries)
    difference_tensor = torch.stack(difference_summaries)
    bootstrap_delta_cube = torch.cat(
        (base_deltas.unsqueeze(0), variant_deltas),
        dim=0,
    )

    # This is the evaluator's only full numeric evidence-array device-to-host
    # transfer. Boundary invariant checks elsewhere may legitimately scalar-sync.
    packed = torch.cat(
        (
            base_means.flatten(),
            base_bootstrap.flatten(),
            sensitivity_tensor.flatten(),
            difference_tensor.flatten(),
            bootstrap_delta_cube.flatten(),
            base_controller_cube.flatten(),
            sensitivity_controller_cube.flatten(),
            power_balance_cube.flatten(),
            forecast_demand_tensor.flatten(),
            forecast_statistics.flatten(),
            forecast_validation.flatten(),
            maximum_power_balance.flatten(),
        )
    )
    host = packed.cpu().numpy()
    offset = 0

    def take(shape: tuple[int, ...]) -> Any:
        nonlocal offset
        size = math.prod(shape)
        values = host[offset : offset + size].reshape(shape)
        offset += size
        return values

    base_means_host = take((2, 4))
    base_bootstrap_host = take((3, 3))
    sensitivity_host = take((24, 3, 3))
    difference_host = take((24, 3, 3))
    bootstrap_delta_cube_host = take(robustness_delta_cube_contract().shape)
    base_controller_cube_host = take(robustness_base_controller_cube_contract().shape)
    sensitivity_controller_cube_host = take(robustness_sensitivity_controller_cube_contract().shape)
    power_balance_cube_host = take(robustness_power_balance_cube_contract().shape)
    forecast_demand_host = take(robustness_forecast_demand_cube_contract().shape)
    forecast_host = take((2, len(ROBUSTNESS_SEEDS), 2, 3))
    forecast_validation_host = take((2,))
    maximum_power_balance_host = float(take((1,))[0])
    if offset != host.size:
        raise RobustnessArtifactError("packed robustness transfer was not fully consumed")
    if tuple(forecast_validation_host) != (1.0, 1.0):
        raise RobustnessArtifactError(
            "forecast demand traces must contain only finite, non-negative values"
        )
    return build_robustness_supplement_payload(
        provenance=provenance,
        base_controller_means=base_means_host,
        base_bootstrap=base_bootstrap_host,
        sensitivity_paired_bootstrap=sensitivity_host,
        sensitivity_difference_bootstrap=difference_host,
        bootstrap_delta_cube=bootstrap_delta_cube_host,
        base_controller_metric_cube=base_controller_cube_host,
        sensitivity_controller_metric_cube=sensitivity_controller_cube_host,
        power_balance_cube=power_balance_cube_host,
        forecast_demand_cube=forecast_demand_host,
        forecast_statistics=forecast_host,
        maximum_power_balance_violation_rate=maximum_power_balance_host,
    )


def validate_robustness_supplement_payload(payload: object) -> dict[str, Any]:
    """Validate the closed robustness schema without requiring CUDA."""

    document = _require_mapping(payload, field="robustness supplement")
    schema = _require_mapping(document.get("schema"), field="schema")
    _require_exact_fields(schema, frozenset({"name", "version"}), field="schema")
    version = schema["version"]
    if (
        schema["name"] != ROBUSTNESS_SCHEMA_NAME
        or type(version) is not int
        or version not in {ROBUSTNESS_LEGACY_SCHEMA_VERSION, ROBUSTNESS_SCHEMA_VERSION}
    ):
        raise RobustnessArtifactError(
            "schema does not identify a supported robustness supplement version"
        )
    is_legacy = version == ROBUSTNESS_LEGACY_SCHEMA_VERSION
    expected_document_fields = {
        "schema",
        "evidence_role",
        "provenance",
        "protocol",
        "metric_definitions",
        "forecast_metric_definitions",
        "base_evaluation",
        "sensitivity",
        "forecast_evaluation",
        "limitations",
    }
    if not is_legacy:
        expected_document_fields.add("bootstrap_inputs")
        expected_document_fields.add("forecast_inputs")
    _require_exact_fields(
        document,
        frozenset(expected_document_fields),
        field="robustness supplement",
    )
    expected_role = ROBUSTNESS_LEGACY_EVIDENCE_ROLE if is_legacy else ROBUSTNESS_EVIDENCE_ROLE
    if document["evidence_role"] != expected_role:
        raise RobustnessArtifactError(f"evidence_role must be {expected_role}")

    provenance = _require_mapping(document["provenance"], field="provenance")
    _require_exact_fields(
        provenance,
        frozenset(
            {
                "artifact_id",
                "source_revision",
                "model_manifest",
                "actor",
                "scenario_generator",
                "training_provenance",
                "implementation_sources",
                "seed_separation",
            }
        ),
        field="provenance",
    )
    expected_artifact_id = ROBUSTNESS_LEGACY_ARTIFACT_ID if is_legacy else ROBUSTNESS_ARTIFACT_ID
    if provenance["artifact_id"] != expected_artifact_id:
        raise RobustnessArtifactError("provenance.artifact_id does not match the contract")
    source_revision = provenance["source_revision"]
    if type(source_revision) is not str or not _GIT_SHA_PATTERN.fullmatch(source_revision):
        raise RobustnessArtifactError("provenance.source_revision must be a lowercase git SHA")
    if source_revision != SOURCE_REVISION:
        raise RobustnessArtifactError(
            "provenance.source_revision does not match the pinned PPO source revision"
        )
    _validate_binding(
        provenance["model_manifest"],
        field="provenance.model_manifest",
        expected_path="data/model/manifest.json",
        expected_sha256=(
            ROBUSTNESS_LEGACY_MODEL_MANIFEST_SHA256
            if is_legacy
            else ROBUSTNESS_MODEL_MANIFEST_SHA256
        ),
        expected_size_bytes=(
            ROBUSTNESS_LEGACY_MODEL_MANIFEST_SIZE_BYTES
            if is_legacy
            else ROBUSTNESS_MODEL_MANIFEST_SIZE_BYTES
        ),
    )
    _validate_binding(
        provenance["actor"],
        field="provenance.actor",
        expected_path="data/model/ppo-policy.npz",
        expected_sha256=ROBUSTNESS_LEGACY_ACTOR_SHA256 if is_legacy else ROBUSTNESS_ACTOR_SHA256,
        expected_size_bytes=(
            ROBUSTNESS_LEGACY_ACTOR_SIZE_BYTES if is_legacy else ROBUSTNESS_ACTOR_SIZE_BYTES
        ),
    )
    _validate_binding(
        provenance["scenario_generator"],
        field="provenance.scenario_generator",
        expected_path="src/gridguard/scenarios.py",
        expected_sha256=(
            ROBUSTNESS_LEGACY_SCENARIO_GENERATOR_SHA256
            if is_legacy
            else ROBUSTNESS_SCENARIO_GENERATOR_SHA256
        ),
        expected_size_bytes=(
            ROBUSTNESS_LEGACY_SCENARIO_GENERATOR_SIZE_BYTES
            if is_legacy
            else ROBUSTNESS_SCENARIO_GENERATOR_SIZE_BYTES
        ),
    )
    _validate_binding(
        provenance["training_provenance"],
        field="provenance.training_provenance",
        expected_path="data/model/training-provenance.yaml",
        expected_sha256=(
            ROBUSTNESS_LEGACY_TRAINING_PROVENANCE_SHA256
            if is_legacy
            else ROBUSTNESS_TRAINING_PROVENANCE_SHA256
        ),
        expected_size_bytes=(
            ROBUSTNESS_LEGACY_TRAINING_PROVENANCE_SIZE_BYTES
            if is_legacy
            else ROBUSTNESS_TRAINING_PROVENANCE_SIZE_BYTES
        ),
    )
    _validate_implementation_sources(
        provenance["implementation_sources"],
        expected_bindings=(
            ROBUSTNESS_LEGACY_IMPLEMENTATION_SOURCE_BINDINGS
            if is_legacy
            else ROBUSTNESS_IMPLEMENTATION_SOURCE_BINDINGS
        ),
    )
    _validate_seed_separation(provenance["seed_separation"], is_legacy=is_legacy)
    _validate_protocol(document["protocol"])
    _validate_metric_definitions(document["metric_definitions"])
    _validate_forecast_metric_definitions(document["forecast_metric_definitions"])
    if not is_legacy:
        bootstrap_inputs = _require_mapping(document["bootstrap_inputs"], field="bootstrap_inputs")
        _require_exact_fields(
            bootstrap_inputs,
            frozenset(
                {
                    "paired_delta_cube",
                    "base_controller_cube",
                    "sensitivity_controller_cube",
                    "power_balance_cube",
                }
            ),
            field="bootstrap_inputs",
        )
        try:
            decode_delta_cube_payload(
                bootstrap_inputs["paired_delta_cube"],
                contract=robustness_delta_cube_contract(),
            )
            decode_base_controller_cube_payload(
                bootstrap_inputs["base_controller_cube"],
                contract=robustness_base_controller_cube_contract(),
            )
            decode_sensitivity_controller_cube_payload(
                bootstrap_inputs["sensitivity_controller_cube"],
                contract=robustness_sensitivity_controller_cube_contract(),
            )
            decode_power_balance_cube_payload(
                bootstrap_inputs["power_balance_cube"],
                contract=robustness_power_balance_cube_contract(),
            )
        except RobustnessInferenceError as error:
            raise RobustnessArtifactError(f"bootstrap_inputs: {error}") from error
        forecast_inputs = _require_mapping(document["forecast_inputs"], field="forecast_inputs")
        _require_exact_fields(
            forecast_inputs,
            frozenset({"demand_cube"}),
            field="forecast_inputs",
        )
        try:
            decode_forecast_demand_cube_payload(
                forecast_inputs["demand_cube"],
                contract=robustness_forecast_demand_cube_contract(),
            )
        except RobustnessInferenceError as error:
            raise RobustnessArtifactError(f"forecast_inputs: {error}") from error
    base_paired_means = _validate_base_evaluation(document["base_evaluation"])
    _validate_sensitivity(
        document["sensitivity"],
        base_paired_means=base_paired_means,
    )
    _validate_forecast_evaluation(document["forecast_evaluation"])
    expected_limitations = ROBUSTNESS_LEGACY_LIMITATIONS if is_legacy else ROBUSTNESS_LIMITATIONS
    if document["limitations"] != list(expected_limitations):
        raise RobustnessArtifactError("limitations do not match the reviewed bounded claims")
    return payload  # type: ignore[return-value]


def robustness_delta_cube_contract() -> DeltaCubeContract:
    """Return the closed axis identity for persisted robustness bootstrap inputs."""

    return DeltaCubeContract(
        parameterization_order=(
            "base",
            *(variant.variant_id for variant in build_sensitivity_variants()),
        ),
        shortage_level_order=tuple(ROBUSTNESS_SHORTAGE_LEVELS),
        seed_order=tuple(ROBUSTNESS_SEEDS),
        episode_order=tuple(range(ROBUSTNESS_BATCH_SIZE)),
        metric_order=tuple(PRIMARY_METRICS),
    )


def robustness_base_controller_cube_contract() -> BaseControllerCubeContract:
    """Return the closed axes for base per-controller primary and safety values."""

    return BaseControllerCubeContract(
        controller_order=("Fixed Rotation", "PPO"),
        shortage_level_order=tuple(ROBUSTNESS_SHORTAGE_LEVELS),
        seed_order=tuple(ROBUSTNESS_SEEDS),
        episode_order=tuple(range(ROBUSTNESS_BATCH_SIZE)),
        metric_order=(*PRIMARY_METRICS, SAFETY_METRIC),
    )


def robustness_power_balance_cube_contract() -> PowerBalanceCubeContract:
    """Return the closed axes for safety values across every parameterization."""

    return PowerBalanceCubeContract(
        parameterization_order=(
            "base",
            *(variant.variant_id for variant in build_sensitivity_variants()),
        ),
        controller_order=("Fixed Rotation", "PPO"),
        shortage_level_order=tuple(ROBUSTNESS_SHORTAGE_LEVELS),
        seed_order=tuple(ROBUSTNESS_SEEDS),
        episode_order=tuple(range(ROBUSTNESS_BATCH_SIZE)),
    )


def robustness_sensitivity_controller_cube_contract() -> SensitivityControllerCubeContract:
    """Return closed axes for raw controller draws under all 24 variants."""

    return SensitivityControllerCubeContract(
        parameterization_order=tuple(
            variant.variant_id for variant in build_sensitivity_variants()
        ),
        controller_order=("Fixed Rotation", "PPO"),
        shortage_level_order=tuple(ROBUSTNESS_SHORTAGE_LEVELS),
        seed_order=tuple(ROBUSTNESS_SEEDS),
        episode_order=tuple(range(ROBUSTNESS_BATCH_SIZE)),
        metric_order=(*PRIMARY_METRICS, SAFETY_METRIC),
    )


def robustness_forecast_demand_cube_contract() -> ForecastDemandCubeContract:
    """Return closed axes for the persisted forecast-evaluation demand traces."""

    return ForecastDemandCubeContract(
        seed_order=tuple(ROBUSTNESS_SEEDS),
        episode_order=tuple(range(ROBUSTNESS_BATCH_SIZE)),
        hour_order=tuple(range(HORIZON_HOURS)),
        locality_order=tuple(locality.value for locality in LOCALITIES),
        sector_order=tuple(sector.value for sector in SECTORS),
    )


def _reported_interval_tensor(
    value: Mapping[str, object],
    *,
    device: torch.device,
) -> torch.Tensor:
    rows = []
    for statistic in ("mean_delta", "ci95_lower", "ci95_upper"):
        rows.append(
            [
                _number(
                    _require_mapping(value[metric], field=f"reported.{metric}")[statistic],
                    field=f"reported.{metric}.{statistic}",
                )
                for metric in PRIMARY_METRICS
            ]
        )
    return torch.tensor(rows, dtype=torch.float64, device=device)


def _require_reconstructed_intervals(
    label: str,
    actual: torch.Tensor,
    reported: torch.Tensor,
) -> None:
    if actual.shape != reported.shape or not bool(
        torch.isclose(actual, reported, rtol=0.0, atol=1e-12).all().item()
    ):
        raise RobustnessArtifactError(f"{label} do not match persisted bootstrap inputs")


def _reported_controller_mean_tensor(
    value: object,
    *,
    device: torch.device,
) -> torch.Tensor:
    controller_means = _require_mapping(value, field="base_evaluation.controller_means")
    return torch.tensor(
        [
            [
                _number(
                    _require_mapping(
                        controller_means[controller],
                        field=f"base_evaluation.controller_means.{controller}",
                    )[metric],
                    field=f"base_evaluation.controller_means.{controller}.{metric}",
                )
                for metric in (*PRIMARY_METRICS, SAFETY_METRIC)
            ]
            for controller in ("Fixed Rotation", "PPO")
        ],
        dtype=torch.float64,
        device=device,
    )


def _reported_forecast_reconstruction_tensor(
    value: object,
    *,
    device: torch.device,
) -> torch.Tensor:
    forecast = _require_mapping(value, field="forecast_evaluation")
    methods = _require_mapping(forecast["methods"], field="forecast_evaluation.methods")
    raw_values: list[float] = []
    pooled_values: list[float] = []
    for method_name in FORECAST_METHODS:
        method = _require_mapping(
            methods[method_name],
            field=f"forecast_evaluation.methods.{method_name}",
        )
        per_seed = method["per_seed"]
        if not isinstance(per_seed, list):
            raise RobustnessArtifactError(
                f"forecast_evaluation.methods.{method_name}.per_seed must be a list"
            )
        for seed_index, seed_summary_value in enumerate(per_seed):
            seed_summary = _require_mapping(
                seed_summary_value,
                field=f"forecast_evaluation.methods.{method_name}.per_seed[{seed_index}]",
            )
            for aggregation in ("locality_sector", "system_total"):
                summary = _require_mapping(
                    seed_summary[aggregation],
                    field=(
                        f"forecast_evaluation.methods.{method_name}.per_seed[{seed_index}]."
                        f"{aggregation}"
                    ),
                )
                raw_values.extend(
                    (
                        float(_positive_integer(summary["count"], field="forecast count")),
                        _number(summary["sum_absolute_error"], field="forecast SAE"),
                        _number(summary["sum_squared_error"], field="forecast SSE"),
                    )
                )
        pooled = _require_mapping(
            method["pooled"],
            field=f"forecast_evaluation.methods.{method_name}.pooled",
        )
        for aggregation in ("locality_sector", "system_total"):
            summary = _require_mapping(
                pooled[aggregation],
                field=f"forecast_evaluation.methods.{method_name}.pooled.{aggregation}",
            )
            pooled_values.extend(
                (
                    float(_positive_integer(summary["count"], field="forecast pooled count")),
                    _number(summary["mae"], field="forecast pooled MAE"),
                    _number(summary["rmse"], field="forecast pooled RMSE"),
                )
            )
    comparison = _require_mapping(
        forecast["comparison"],
        field="forecast_evaluation.comparison",
    )
    deltas = _require_mapping(
        comparison["trailing_minus_persistence"],
        field="forecast_evaluation.comparison.trailing_minus_persistence",
    )
    winners = _require_mapping(
        comparison["lower_error_method"],
        field="forecast_evaluation.comparison.lower_error_method",
    )
    winner_codes = {"tie": 0.0, "trailing_mean_3h": 1.0, "persistence_1h": 2.0}
    measurement_order = (
        "locality_sector_mae",
        "locality_sector_rmse",
        "system_total_mae",
        "system_total_rmse",
    )
    comparison_values = [
        _number(deltas[name], field=f"forecast delta {name}") for name in measurement_order
    ]
    comparison_values.extend(winner_codes[str(winners[name])] for name in measurement_order)
    return torch.tensor(
        (*raw_values, *pooled_values, *comparison_values),
        dtype=torch.float64,
        device=device,
    )


def _reconstructed_forecast_tensor(statistics: torch.Tensor) -> torch.Tensor:
    if statistics.shape != (2, len(ROBUSTNESS_SEEDS), 2, 3):
        raise RobustnessArtifactError("reconstructed forecast statistics have the wrong shape")
    if statistics.dtype != torch.float64 or statistics.device != torch.device("cuda:0"):
        raise RobustnessArtifactError(
            "reconstructed forecast statistics must remain float64 on cuda:0"
        )
    statistics_float64 = statistics
    totals = statistics_float64.sum(dim=1)
    counts = totals[..., 0]
    if bool((counts <= 0.0).any().item()):
        raise RobustnessArtifactError("reconstructed forecast counts must remain positive")
    pooled = torch.stack(
        (
            counts,
            totals[..., 1] / counts,
            torch.sqrt(totals[..., 2] / counts),
        ),
        dim=-1,
    )
    deltas = (pooled[0, :, 1:] - pooled[1, :, 1:]).reshape(-1)
    winner_codes = torch.where(
        torch.isclose(
            deltas,
            torch.zeros_like(deltas),
            rtol=0.0,
            atol=1e-12,
        ),
        torch.zeros_like(deltas),
        torch.where(deltas < 0.0, torch.ones_like(deltas), torch.full_like(deltas, 2.0)),
    )
    return torch.cat((statistics_float64.flatten(), pooled.flatten(), deltas, winner_codes))


@torch.inference_mode()
def _locked_forecast_demand_cube_on_cuda() -> torch.Tensor:
    demands = []
    for seed in ROBUSTNESS_SEEDS:
        draws = generate_scenario_base_draws(
            ROBUSTNESS_BATCH_SIZE,
            seed,
            device="cuda:0",
        )
        demands.append(scenario_from_base_draws(draws, "mild").demand)
    return torch.stack(demands)


def verify_robustness_inference_on_cuda(payload: object) -> dict[str, int]:
    """Reconstruct every robustness interval from persisted inputs on ``cuda:0``."""

    document = validate_robustness_supplement_payload(payload)
    schema = _require_mapping(document["schema"], field="schema")
    if schema["version"] != ROBUSTNESS_SCHEMA_VERSION:
        raise RobustnessArtifactError(
            f"CUDA inference reconstruction requires robustness schema version "
            f"{ROBUSTNESS_SCHEMA_VERSION}"
        )
    contract = robustness_delta_cube_contract()
    controller_contract = robustness_base_controller_cube_contract()
    sensitivity_controller_contract = robustness_sensitivity_controller_cube_contract()
    power_contract = robustness_power_balance_cube_contract()
    forecast_contract = robustness_forecast_demand_cube_contract()
    try:
        bootstrap_inputs = _require_mapping(document["bootstrap_inputs"], field="bootstrap_inputs")
        delta_cube = delta_cube_to_cuda(
            bootstrap_inputs["paired_delta_cube"],
            contract=contract,
            device="cuda:0",
        )
        controller_cube = base_controller_cube_to_cuda(
            bootstrap_inputs["base_controller_cube"],
            contract=controller_contract,
            device="cuda:0",
        )
        sensitivity_controller_cube = sensitivity_controller_cube_to_cuda(
            bootstrap_inputs["sensitivity_controller_cube"],
            contract=sensitivity_controller_contract,
            device="cuda:0",
        )
        power_balance_cube = power_balance_cube_to_cuda(
            bootstrap_inputs["power_balance_cube"],
            contract=power_contract,
            device="cuda:0",
        )
        forecast_inputs = _require_mapping(document["forecast_inputs"], field="forecast_inputs")
        forecast_demand_cube = forecast_demand_cube_to_cuda(
            forecast_inputs["demand_cube"],
            contract=forecast_contract,
            device="cuda:0",
        )
        validate_reconstruction_cube_semantics(
            delta_cube,
            controller_cube,
            sensitivity_controller_cube,
            power_balance_cube,
            delta_contract=contract,
            base_controller_contract=controller_contract,
            sensitivity_controller_contract=sensitivity_controller_contract,
            power_balance_contract=power_contract,
            maximum_outage_locality_hours=float(ROBUSTNESS_MAX_OUTAGE_LOCALITY_HOURS),
        )
        base, variants, differences = reconstruct_bootstrap_intervals(
            delta_cube,
            metric_names=tuple(PRIMARY_METRICS),
            replicates=ROBUSTNESS_BOOTSTRAP_REPLICATES,
            seed=ROBUSTNESS_BOOTSTRAP_SEED,
        )
        controller_means, maximum_power_balance = reconstruct_controller_statistics(
            controller_cube,
            power_balance_cube,
        )
        expected_forecast_demand = _locked_forecast_demand_cube_on_cuda()
        if not bool(torch.equal(forecast_demand_cube, expected_forecast_demand)):
            raise RobustnessInferenceError(
                "persisted forecast demand traces do not match locked scenario reconstruction"
            )
        if bool(
            torch.logical_or(
                ~torch.isfinite(forecast_demand_cube),
                forecast_demand_cube < 0.0,
            )
            .any()
            .item()
        ):
            raise RobustnessInferenceError(
                "persisted forecast demand traces must be finite and non-negative"
            )
        forecast_statistics = forecast_error_sufficient_statistics(forecast_demand_cube)
        reconstructed_forecast = _reconstructed_forecast_tensor(forecast_statistics)
    except (BootstrapError, RobustnessInferenceError, RuntimeError, ValueError) as error:
        raise RobustnessArtifactError(f"bootstrap reconstruction failed: {error}") from error

    try:
        base_evaluation = _require_mapping(document["base_evaluation"], field="base_evaluation")
        reported_controller_means = _reported_controller_mean_tensor(
            base_evaluation["controller_means"],
            device=delta_cube.device,
        )
        power_balance = _require_mapping(
            base_evaluation["power_balance"],
            field="base_evaluation.power_balance",
        )
        reported_maximum_power_balance = torch.tensor(
            _number(
                power_balance["maximum_violation_rate"],
                field="base_evaluation.power_balance.maximum_violation_rate",
            ),
            dtype=torch.float64,
            device=delta_cube.device,
        )
        reported_base = _reported_interval_tensor(
            _require_mapping(
                base_evaluation["paired_inference"], field="base_evaluation.paired_inference"
            ),
            device=delta_cube.device,
        )
        reported_forecast = _reported_forecast_reconstruction_tensor(
            document["forecast_evaluation"],
            device=delta_cube.device,
        )
        sensitivity = _require_mapping(document["sensitivity"], field="sensitivity")
        variant_documents = sensitivity["variants"]
        if not isinstance(variant_documents, list):
            raise RobustnessArtifactError("sensitivity.variants must be a list")
        reported_variants = torch.stack(
            tuple(
                _reported_interval_tensor(
                    _require_mapping(
                        _require_mapping(value, field=f"sensitivity.variants[{index}]")[
                            "paired_inference"
                        ],
                        field=f"sensitivity.variants[{index}].paired_inference",
                    ),
                    device=delta_cube.device,
                )
                for index, value in enumerate(variant_documents)
            )
        )
        reported_differences = torch.stack(
            tuple(
                _reported_interval_tensor(
                    _require_mapping(
                        _require_mapping(value, field=f"sensitivity.variants[{index}]")[
                            "difference_in_differences"
                        ],
                        field=f"sensitivity.variants[{index}].difference_in_differences",
                    ),
                    device=delta_cube.device,
                )
                for index, value in enumerate(variant_documents)
            )
        )
        _require_reconstructed_intervals("reconstructed base intervals", base, reported_base)
        _require_reconstructed_intervals(
            "reconstructed controller means",
            controller_means,
            reported_controller_means,
        )
        _require_reconstructed_intervals(
            "reconstructed power-balance maximum",
            maximum_power_balance,
            reported_maximum_power_balance,
        )
        _require_reconstructed_intervals(
            "reconstructed sensitivity intervals",
            variants,
            reported_variants,
        )
        _require_reconstructed_intervals(
            "reconstructed difference-in-differences intervals",
            differences,
            reported_differences,
        )
        _require_reconstructed_intervals(
            "reconstructed forecast summaries",
            reconstructed_forecast,
            reported_forecast,
        )
    except RuntimeError as error:
        raise RobustnessArtifactError(f"CUDA report comparison failed: {error}") from error
    return {
        "parameterization_count": len(contract.parameterization_order),
        "reconstructed_interval_set_count": 1 + 2 * (len(contract.parameterization_order) - 1),
        "forecast_trace_count": len(ROBUSTNESS_SEEDS) * ROBUSTNESS_BATCH_SIZE,
        "reconstructed_forecast_summary_count": 2 * len(ROBUSTNESS_SEEDS) * 2,
    }


def _write_canonical_robustness_bytes(encoded: bytes, path: str | Path) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        temporary.write_bytes(encoded)
        temporary.replace(output)
    except OSError as error:
        raise RobustnessArtifactError(
            f"robustness supplement cannot be written: {error}"
        ) from error
    return hashlib.sha256(encoded).hexdigest()


def _require_current_robustness_schema(payload: Mapping[str, object]) -> None:
    schema = _require_mapping(payload["schema"], field="schema")
    if schema["version"] != ROBUSTNESS_SCHEMA_VERSION:
        raise RobustnessArtifactError(
            f"candidate and release verification require robustness schema version "
            f"{ROBUSTNESS_SCHEMA_VERSION}"
        )


def write_robustness_supplement_candidate(
    payload: Mapping[str, object],
    path: str | Path,
) -> str:
    """Write a canonical schema-v2 candidate without granting release trust."""

    validated = validate_robustness_supplement_payload(payload)
    _require_current_robustness_schema(validated)
    return _write_canonical_robustness_bytes(
        canonical_robustness_json_bytes(validated),
        path,
    )


def write_robustness_supplement(payload: Mapping[str, object], path: str | Path) -> str:
    """Write schema-v2 evidence only when it matches the project trust anchor."""

    validated = validate_robustness_supplement_payload(payload)
    _require_current_robustness_schema(validated)
    encoded = canonical_robustness_json_bytes(validated)
    actual_digest = hashlib.sha256(encoded).hexdigest()
    if actual_digest != ROBUSTNESS_SUPPLEMENT_SHA256:
        raise RobustnessArtifactError(
            "robustness supplement does not match the project trust anchor: "
            f"expected {ROBUSTNESS_SUPPLEMENT_SHA256}, found {actual_digest}"
        )
    return _write_canonical_robustness_bytes(encoded, path)


def _verify_robustness_supplement_bytes(
    path: str | Path,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    if type(expected_sha256) is not str or not _SHA256_PATTERN.fullmatch(expected_sha256):
        raise RobustnessArtifactError("expected_sha256 must be a lowercase SHA-256 digest")
    artifact_path = Path(path)
    try:
        encoded = artifact_path.read_bytes()
    except OSError as error:
        raise RobustnessArtifactError(f"robustness supplement cannot be read: {error}") from error
    actual_digest = hashlib.sha256(encoded).hexdigest()
    if actual_digest != expected_sha256:
        raise RobustnessArtifactError(
            "robustness supplement SHA-256 mismatch: "
            f"expected {expected_sha256}, found {actual_digest}"
        )
    try:
        payload = json.loads(
            encoded,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                RobustnessArtifactError(f"non-finite JSON number: {value}")
            ),
        )
    except (RobustnessArtifactError, UnicodeDecodeError, json.JSONDecodeError) as error:
        if isinstance(error, RobustnessArtifactError):
            raise
        raise RobustnessArtifactError(f"robustness supplement is invalid JSON: {error}") from error
    validated = validate_robustness_supplement_payload(payload)
    if canonical_robustness_json_bytes(validated) != encoded:
        raise RobustnessArtifactError("robustness supplement is not canonical JSON")
    return validated


def verify_robustness_supplement_candidate(
    path: str | Path,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    """Verify canonical schema-v2 candidate bytes without granting release trust."""

    validated = _verify_robustness_supplement_bytes(
        path,
        expected_sha256=expected_sha256,
    )
    _require_current_robustness_schema(validated)
    return validated


def verify_robustness_supplement(path: str | Path) -> dict[str, Any]:
    """Verify canonical schema-v2 bytes against the project trust anchor."""

    return verify_robustness_supplement_candidate(
        path,
        expected_sha256=ROBUSTNESS_SUPPLEMENT_SHA256,
    )


def verify_legacy_robustness_supplement_host(
    path: str | Path,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    """Read a pinned schema-v1 artifact on host without satisfying the release gate."""

    validated = _verify_robustness_supplement_bytes(
        path,
        expected_sha256=expected_sha256,
    )
    schema = _require_mapping(validated["schema"], field="schema")
    if schema["version"] != ROBUSTNESS_LEGACY_SCHEMA_VERSION:
        raise RobustnessArtifactError("legacy host reader requires robustness schema version 1")
    return validated


@dataclass(frozen=True, slots=True)
class SensitivityVariant:
    """One normalized one-factor-at-a-time demand parameter perturbation."""

    variant_id: str
    parameter: str
    component_index: int
    component_name: str
    factor: float
    parameters: ScenarioDemandParameters


def _perturb_and_renormalize(
    values: tuple[float, ...],
    *,
    component_index: int,
    factor: float,
) -> tuple[float, ...]:
    perturbed = list(values)
    perturbed[component_index] *= factor
    scale = sum(values) / sum(perturbed)
    return tuple(value * scale for value in perturbed)


def build_sensitivity_variants() -> tuple[SensitivityVariant, ...]:
    """Return the locked 24 normalized ±25% one-factor-at-a-time variants."""

    variants: list[SensitivityVariant] = []
    for parameter, components, base_values in (
        ("locality_weight", LOCALITIES, LOCALITY_DEMAND_WEIGHTS),
        ("sector_share", SECTORS, SECTOR_SHARES),
    ):
        for component_index, component in enumerate(components):
            for factor in SENSITIVITY_FACTORS:
                locality_weights = LOCALITY_DEMAND_WEIGHTS
                sector_shares = SECTOR_SHARES
                changed = _perturb_and_renormalize(
                    base_values,
                    component_index=component_index,
                    factor=factor,
                )
                if parameter == "locality_weight":
                    locality_weights = changed
                else:
                    sector_shares = changed
                direction = "minus_25_percent" if factor < 1.0 else "plus_25_percent"
                component_name = component.name.lower()
                variants.append(
                    SensitivityVariant(
                        variant_id=f"{parameter}:{component_name}:{direction}",
                        parameter=parameter,
                        component_index=component_index,
                        component_name=component_name,
                        factor=factor,
                        parameters=ScenarioDemandParameters(
                            locality_weights=locality_weights,
                            sector_shares=sector_shares,
                        ),
                    )
                )
    return tuple(variants)


def _error_sufficient_statistics(errors: torch.Tensor) -> torch.Tensor:
    values = errors.to(torch.float64)
    seed_count = values.shape[0]
    count = torch.full(
        (seed_count,),
        values[0].numel(),
        dtype=torch.float64,
        device=values.device,
    )
    return torch.stack(
        (
            count,
            values.sum(dim=tuple(range(1, values.ndim))),
            values.square().sum(dim=tuple(range(1, values.ndim))),
        ),
        dim=-1,
    )


def _validate_forecast_trace_metadata(demand_traces: object) -> torch.Tensor:
    if not isinstance(demand_traces, torch.Tensor):
        raise TypeError("demand_traces must be a torch.Tensor")
    if demand_traces.ndim != 5 or demand_traces.shape[2:] != (
        HORIZON_HOURS,
        len(LOCALITIES),
        len(SECTORS),
    ):
        raise ValueError("demand_traces must have shape [seed, batch, 24, 7, 5]")
    if demand_traces.shape[0] < 1 or demand_traces.shape[1] < 1:
        raise ValueError("demand_traces must have shape [seed, batch, 24, 7, 5]")
    if not demand_traces.is_floating_point():
        raise TypeError("demand_traces must use a floating-point dtype")
    return demand_traces


@torch.inference_mode()
def forecast_input_validation_flags(demand_traces: torch.Tensor) -> torch.Tensor:
    """Return finite/nonnegative flags without synchronizing the tensor device."""

    demand = _validate_forecast_trace_metadata(demand_traces)
    return torch.stack(
        (
            torch.isfinite(demand).all(),
            ~(demand < 0.0).any(),
        )
    ).to(torch.float64)


@torch.inference_mode()
def forecast_error_sufficient_statistics(demand_traces: torch.Tensor) -> torch.Tensor:
    """Return aligned forecast count/error sums without leaving the tensor device.

    Input has shape ``[seed, batch, 24, 7, 5]``. Output has shape
    ``[method, seed, aggregation, statistic]`` where aggregation order is
    locality-sector then system-total, and statistic order is count, absolute
    error sum, then squared error sum.
    """

    demand_traces = _validate_forecast_trace_metadata(demand_traces)

    seed_count, batch_size = demand_traces.shape[:2]
    flattened_demand = demand_traces.reshape(
        seed_count * batch_size,
        HORIZON_HOURS,
        len(LOCALITIES),
        len(SECTORS),
    )
    flattened_trailing_mean = causal_demand_projection(
        flattened_demand,
        window_hours=3,
        horizon_hours=1,
    )
    trailing_mean = flattened_trailing_mean.reshape(
        seed_count,
        batch_size,
        *flattened_trailing_mean.shape[1:],
    )
    targets = demand_traces[:, :, 3:HORIZON_HOURS]
    persistence = demand_traces[:, :, 2 : HORIZON_HOURS - 1]
    methods: list[torch.Tensor] = []
    for predictions in (trailing_mean, persistence):
        locality_sector_error = (predictions - targets).abs()
        system_error = (predictions.sum(dim=(-1, -2)) - targets.sum(dim=(-1, -2))).abs()
        methods.append(
            torch.stack(
                (
                    _error_sufficient_statistics(locality_sector_error),
                    _error_sufficient_statistics(system_error),
                ),
                dim=1,
            )
        )
    return torch.stack(methods, dim=0)

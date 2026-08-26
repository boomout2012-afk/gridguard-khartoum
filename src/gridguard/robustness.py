"""Untouched-seed robustness, sensitivity, and forecast evidence."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Final

import torch

from gridguard.controllers import PPOController
from gridguard.domain import (
    HORIZON_HOURS,
    LOCALITIES,
    LOCALITY_DEMAND_WEIGHTS,
    SECTOR_SHARES,
    SECTORS,
)
from gridguard.evaluation import evaluate_scenario_pair
from gridguard.metrics import EpisodeMetrics
from gridguard.policy import SOURCE_REVISION, load_ppo_policy
from gridguard.provenance import sha256_file
from gridguard.scenarios import (
    ScenarioDemandParameters,
    generate_scenario_base_draws,
    scenario_from_base_draws,
)
from gridguard.statistics import stratified_paired_bootstrap

ROBUSTNESS_SCHEMA_NAME: Final = "gridguard.robustness_supplement"
ROBUSTNESS_SCHEMA_VERSION: Final = 1
ROBUSTNESS_EVIDENCE_ROLE: Final = "untouched_confirmatory"
ROBUSTNESS_ARTIFACT_ID: Final = "gridguard-robustness-supplement-v1"
# Filled after the CUDA artifact is reproduced and accepted. Callers must never
# treat an empty value as trusted; verifier entry points always require a digest.
ROBUSTNESS_SUPPLEMENT_SHA256: Final = ""
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
ROBUSTNESS_LIMITATIONS: Final = (
    "Synthetic demand traces do not establish real-world forecast accuracy.",
    "Sensitivity covers only one-factor-at-a-time ±25% demand-allocation perturbations.",
    "Observed interval direction is descriptive and is not a release gate.",
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
_SHA256_PATTERN: Final = re.compile(r"[0-9a-f]{64}")
_GIT_SHA_PATTERN: Final = re.compile(r"[0-9a-f]{40}")


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


def _validate_binding(
    value: object,
    *,
    field: str,
    expected_path: str,
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
    _positive_integer(binding["size_bytes"], field=f"{field}.size_bytes")
    return binding


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
    expected_values: dict[str, object] = {
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
    }
    expected_fields = frozenset((*expected_values, "bootstrap"))
    _require_exact_fields(protocol, expected_fields, field="protocol")
    for field, expected in expected_values.items():
        if protocol[field] != expected:
            raise RobustnessArtifactError(f"protocol.{field} does not match the reviewed protocol")
    bootstrap = _require_mapping(protocol["bootstrap"], field="protocol.bootstrap")
    _require_exact_fields(
        bootstrap,
        frozenset({"replicates", "seed", "ci_percent"}),
        field="protocol.bootstrap",
    )
    if bootstrap != {
        "replicates": ROBUSTNESS_BOOTSTRAP_REPLICATES,
        "seed": ROBUSTNESS_BOOTSTRAP_SEED,
        "ci_percent": 95,
    }:
        raise RobustnessArtifactError("protocol.bootstrap does not match the reviewed protocol")
    return protocol


def _validate_metric_definitions(value: object) -> None:
    definitions = _require_mapping(value, field="metric_definitions")
    if definitions != ROBUSTNESS_METRIC_DEFINITIONS:
        raise RobustnessArtifactError("metric_definitions do not match the canonical contract")


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


def _validate_interval(value: object, *, field: str, metric: str) -> float:
    interval = _require_mapping(value, field=field)
    _require_exact_fields(
        interval,
        frozenset({"mean_delta", "ci95_lower", "ci95_upper", "result_classification"}),
        field=field,
    )
    mean = _number(interval["mean_delta"], field=f"{field}.mean_delta")
    lower = _number(interval["ci95_lower"], field=f"{field}.ci95_lower")
    upper = _number(interval["ci95_upper"], field=f"{field}.ci95_upper")
    if lower > upper:
        raise RobustnessArtifactError(f"{field} confidence interval is reversed")
    expected = _classify_interval(metric, lower, upper)
    if interval["result_classification"] != expected:
        raise RobustnessArtifactError(f"{field} classification does not match its interval")
    return mean


def _validate_paired_inference(value: object, *, field: str) -> dict[str, float]:
    inference = _require_mapping(value, field=field)
    _require_exact_fields(inference, frozenset(PRIMARY_METRICS), field=field)
    return {
        metric: _validate_interval(inference[metric], field=f"{field}.{metric}", metric=metric)
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
        base["paired_inference"], field="base_evaluation.paired_inference"
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
        )
        difference_means = _validate_paired_inference(
            actual["difference_in_differences"],
            field=f"{field}.difference_in_differences",
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
    if method["window_hours"] != expected_window:
        raise RobustnessArtifactError(f"{field}.window_hours does not match the method")
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
        if seed_summary["seed"] != expected_seed:
            raise RobustnessArtifactError(f"{seed_field}.seed does not match reviewed order")
        for aggregation, expected_count in (
            ("locality_sector", ROBUSTNESS_BATCH_SIZE * len(FORECAST_TARGET_HOURS) * 35),
            ("system_total", ROBUSTNESS_BATCH_SIZE * len(FORECAST_TARGET_HOURS)),
        ):
            values = _validate_forecast_sums(
                seed_summary[aggregation],
                field=f"{seed_field}.{aggregation}",
                expected_count=expected_count,
            )
            for index, item in enumerate(values):
                totals[aggregation][index] += item

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
        if aggregate["count"] != total_count:
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
    exact_values = {
        "target_hours": list(FORECAST_TARGET_HOURS),
        "trace_count": len(ROBUSTNESS_SEEDS) * ROBUSTNESS_BATCH_SIZE,
        "locality_sector_prediction_count": 235_200,
        "system_total_prediction_count": 6_720,
    }
    for name, expected in exact_values.items():
        if forecast[name] != expected:
            raise RobustnessArtifactError(f"forecast_evaluation.{name} is not aligned")
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

    array = np.asarray(value, dtype=np.float64)
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


def _build_robustness_provenance(manifest_path: Path) -> dict[str, object]:
    if manifest_path.parent.name != "model" or manifest_path.parent.parent.name != "data":
        raise RobustnessArtifactError(
            "PPO manifest must be located at data/model/manifest.json for reproducible provenance"
        )
    repository_root = manifest_path.parent.parent.parent
    scenario_generator_path = repository_root / "src" / "gridguard" / "scenarios.py"
    actor_path = manifest_path.parent / "ppo-policy.npz"
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
    }


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


@torch.inference_mode()
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

    fixed_cells: list[torch.Tensor] = []
    ppo_cells: list[torch.Tensor] = []
    forecast_demands: list[torch.Tensor] = []
    for severity_index, shortage_level in enumerate(ROBUSTNESS_SHORTAGE_LEVELS):
        for seed in ROBUSTNESS_SEEDS:
            scenario = scenario_from_base_draws(base_draws[seed], shortage_level)
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
    base_deltas = ppo_cube[..., :3].to(torch.float64) - fixed_cube[..., :3].to(torch.float64)
    base_bootstrap = stratified_paired_bootstrap(
        base_deltas,
        metric_names=PRIMARY_METRICS,
        replicates=ROBUSTNESS_BOOTSTRAP_REPLICATES,
        seed=ROBUSTNESS_BOOTSTRAP_SEED,
    ).as_tensor()
    base_means = torch.stack(
        (
            fixed_cube.to(torch.float64).mean(dim=(0, 1, 2)),
            ppo_cube.to(torch.float64).mean(dim=(0, 1, 2)),
        )
    )
    maximum_power_balance = (
        torch.stack((fixed_cube[..., 3].amax(), ppo_cube[..., 3].amax())).amax().reshape(1)
    )
    forecast_statistics = forecast_error_sufficient_statistics(torch.stack(forecast_demands))

    variant_deltas: list[torch.Tensor] = []
    sensitivity_power_balance: list[torch.Tensor] = []
    for variant in build_sensitivity_variants():
        cells: list[torch.Tensor] = []
        for shortage_level in ROBUSTNESS_SHORTAGE_LEVELS:
            for seed in ROBUSTNESS_SEEDS:
                scenario = scenario_from_base_draws(
                    base_draws[seed],
                    shortage_level,
                    demand_parameters=variant.parameters,
                )
                evaluated = evaluate_scenario_pair(scenario, ppo_controller=ppo_controller)
                fixed_metrics = _robustness_metric_matrix(evaluated.fixed_metrics)
                ppo_metrics = _robustness_metric_matrix(evaluated.ppo_metrics)
                cells.append(
                    ppo_metrics[..., :3].to(torch.float64)
                    - fixed_metrics[..., :3].to(torch.float64)
                )
                sensitivity_power_balance.extend(
                    (fixed_metrics[..., 3].amax(), ppo_metrics[..., 3].amax())
                )
        variant_deltas.append(
            torch.stack(cells).reshape(
                len(ROBUSTNESS_SHORTAGE_LEVELS),
                len(ROBUSTNESS_SEEDS),
                ROBUSTNESS_BATCH_SIZE,
                len(PRIMARY_METRICS),
            )
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
    maximum_power_balance = torch.maximum(
        maximum_power_balance,
        torch.stack(sensitivity_power_balance).amax().reshape(1),
    )
    sensitivity_tensor = torch.stack(sensitivity_summaries)
    difference_tensor = torch.stack(difference_summaries)

    # This is the only full numeric device-to-host transfer in the robustness evaluator.
    packed = torch.cat(
        (
            base_means.flatten(),
            base_bootstrap.flatten(),
            sensitivity_tensor.flatten(),
            difference_tensor.flatten(),
            forecast_statistics.flatten(),
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
    forecast_host = take((2, len(ROBUSTNESS_SEEDS), 2, 3))
    maximum_power_balance_host = float(take((1,))[0])
    if offset != host.size:
        raise RobustnessArtifactError("packed robustness transfer was not fully consumed")
    return build_robustness_supplement_payload(
        provenance=provenance,
        base_controller_means=base_means_host,
        base_bootstrap=base_bootstrap_host,
        sensitivity_paired_bootstrap=sensitivity_host,
        sensitivity_difference_bootstrap=difference_host,
        forecast_statistics=forecast_host,
        maximum_power_balance_violation_rate=maximum_power_balance_host,
    )


def validate_robustness_supplement_payload(payload: object) -> dict[str, Any]:
    """Validate the closed robustness schema without requiring CUDA."""

    document = _require_mapping(payload, field="robustness supplement")
    _require_exact_fields(
        document,
        frozenset(
            {
                "schema",
                "evidence_role",
                "provenance",
                "protocol",
                "metric_definitions",
                "base_evaluation",
                "sensitivity",
                "forecast_evaluation",
                "limitations",
            }
        ),
        field="robustness supplement",
    )
    schema = _require_mapping(document["schema"], field="schema")
    _require_exact_fields(schema, frozenset({"name", "version"}), field="schema")
    if (
        schema["name"] != ROBUSTNESS_SCHEMA_NAME
        or type(schema["version"]) is not int
        or schema["version"] != ROBUSTNESS_SCHEMA_VERSION
    ):
        raise RobustnessArtifactError("schema does not identify robustness supplement version 1")
    if document["evidence_role"] != ROBUSTNESS_EVIDENCE_ROLE:
        raise RobustnessArtifactError("evidence_role must be untouched_confirmatory")

    provenance = _require_mapping(document["provenance"], field="provenance")
    _require_exact_fields(
        provenance,
        frozenset(
            {"artifact_id", "source_revision", "model_manifest", "actor", "scenario_generator"}
        ),
        field="provenance",
    )
    if provenance["artifact_id"] != ROBUSTNESS_ARTIFACT_ID:
        raise RobustnessArtifactError("provenance.artifact_id does not match the contract")
    source_revision = provenance["source_revision"]
    if type(source_revision) is not str or not _GIT_SHA_PATTERN.fullmatch(source_revision):
        raise RobustnessArtifactError("provenance.source_revision must be a lowercase git SHA")
    _validate_binding(
        provenance["model_manifest"],
        field="provenance.model_manifest",
        expected_path="data/model/manifest.json",
    )
    _validate_binding(
        provenance["actor"],
        field="provenance.actor",
        expected_path="data/model/ppo-policy.npz",
    )
    _validate_binding(
        provenance["scenario_generator"],
        field="provenance.scenario_generator",
        expected_path="src/gridguard/scenarios.py",
    )
    _validate_protocol(document["protocol"])
    _validate_metric_definitions(document["metric_definitions"])
    base_paired_means = _validate_base_evaluation(document["base_evaluation"])
    _validate_sensitivity(
        document["sensitivity"],
        base_paired_means=base_paired_means,
    )
    _validate_forecast_evaluation(document["forecast_evaluation"])
    if document["limitations"] != list(ROBUSTNESS_LIMITATIONS):
        raise RobustnessArtifactError("limitations do not match the reviewed bounded claims")
    return payload  # type: ignore[return-value]


def write_robustness_supplement(payload: Mapping[str, object], path: str | Path) -> str:
    """Atomically write a validated canonical robustness supplement."""

    validate_robustness_supplement_payload(payload)
    encoded = canonical_robustness_json_bytes(payload)
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


def verify_robustness_supplement(
    path: str | Path,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    """Verify canonical bytes, the pinned digest, and all scientific invariants on host."""

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


@torch.inference_mode()
def forecast_error_sufficient_statistics(demand_traces: torch.Tensor) -> torch.Tensor:
    """Return aligned forecast count/error sums without leaving the tensor device.

    Input has shape ``[seed, batch, 24, 7, 5]``. Output has shape
    ``[method, seed, aggregation, statistic]`` where aggregation order is
    locality-sector then system-total, and statistic order is count, absolute
    error sum, then squared error sum.
    """

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
    if not bool(torch.isfinite(demand_traces).all().item()):
        raise ValueError("demand_traces must contain only finite values")
    if bool((demand_traces < 0.0).any().item()):
        raise ValueError("demand_traces must not contain negative values")

    targets = demand_traces[:, :, 3:HORIZON_HOURS]
    trailing_mean = demand_traces.unfold(dimension=2, size=3, step=1)[:, :, :21].mean(dim=-1)
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

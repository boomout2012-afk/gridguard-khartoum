"""Validated, read-only evidence boundary for the presentation interface."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from gridguard import __version__
from gridguard.domain import HORIZON_HOURS
from gridguard.resources import runtime_root
from gridguard.robustness import (
    PRIMARY_METRICS,
    ROBUSTNESS_ARTIFACT_ID,
    ROBUSTNESS_EVIDENCE_ROLE,
    ROBUSTNESS_SCHEMA_NAME,
    ROBUSTNESS_SCHEMA_VERSION,
    ROBUSTNESS_SUPPLEMENT_SHA256,
    RobustnessArtifactError,
    verify_robustness_supplement,
)

CONTROLLER_NAMES = ("PPO", "Fixed Rotation")
RELEASE_ID = f"gridguard-khartoum-v{__version__}"

_BASE_EVALUATION_FIELDS = frozenset({"controller_means", "paired_inference", "power_balance"})
_CONTROLLER_MEAN_FIELDS = frozenset((*PRIMARY_METRICS, "power_balance_violation_rate"))
_INTERVAL_FIELDS = frozenset({"mean_delta", "ci95_lower", "ci95_upper", "result_classification"})
_METRIC_PRESENTATION = {
    "critical_outage_locality_hours": (
        "Critical-outage locality-hours",
        "locality-hours",
        1.0,
        "lower_is_better",
    ),
    "fairness": ("Jain fairness", "%", 100.0, "higher_is_better"),
    "utilization": ("Utilization", "%", 100.0, "higher_is_better"),
}
_METRIC_ORDER = tuple(_METRIC_PRESENTATION)
if _METRIC_ORDER != PRIMARY_METRICS:  # pragma: no cover - import-time contract guard
    raise RuntimeError("presentation primary metric order does not match robustness evidence")


class PresentationDataError(ValueError):
    """Raised when verified evidence cannot be presented without changing its meaning."""


@dataclass(frozen=True, slots=True)
class MetricComparison:
    """One untouched aggregate metric and its paired PPO-minus-baseline inference."""

    key: str
    label: str
    unit: str
    display_scale: float
    direction: str
    role: str
    ppo: float
    fixed_rotation: float
    mean_delta: float
    ci95_lower: float
    ci95_upper: float
    result_classification: str


@dataclass(frozen=True, slots=True)
class PresentationSnapshot:
    """Untouched confirmatory values approved for display outside Qt widgets."""

    artifact_id: str
    evidence_role: str
    artifact_sha256: str
    release_id: str
    controller_names: tuple[str, str]
    count_statement: str
    independent_base_draw_count: int
    shortage_level_count: int
    matched_scenario_count: int
    controller_evaluation_count: int
    horizon_hours: int
    metrics: tuple[MetricComparison, ...]
    maximum_power_balance_violation_rate: float
    limitations: tuple[str, ...]

    @property
    def benchmark_id(self) -> str:
        """Retain the shell's evidence-identifier interface during the artifact migration."""

        return self.artifact_id

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> Self:
        """Validate the locked robustness fields consumed by the presentation."""

        schema = _mapping(payload.get("schema"), field="schema")
        _exact_fields(schema, frozenset({"name", "version"}), field="schema")
        if schema != {
            "name": ROBUSTNESS_SCHEMA_NAME,
            "version": ROBUSTNESS_SCHEMA_VERSION,
        }:
            raise PresentationDataError(
                "evidence must identify gridguard robustness supplement version 1"
            )

        evidence_role = _text(payload.get("evidence_role"), field="evidence_role")
        if evidence_role != ROBUSTNESS_EVIDENCE_ROLE:
            raise PresentationDataError("evidence_role must be untouched_confirmatory")
        provenance = _mapping(payload.get("provenance"), field="provenance")
        artifact_id = _text(provenance.get("artifact_id"), field="provenance.artifact_id")
        if artifact_id != ROBUSTNESS_ARTIFACT_ID:
            raise PresentationDataError(f"provenance.artifact_id must be {ROBUSTNESS_ARTIFACT_ID}")

        protocol = _mapping(payload.get("protocol"), field="protocol")
        _validate_controllers(protocol.get("controllers"))
        if protocol.get("delta_definition") != "PPO - Fixed Rotation":
            raise PresentationDataError("protocol.delta_definition must be PPO - Fixed Rotation")
        independent_draws = _exact_positive_int(
            protocol.get("independent_base_draw_count"),
            expected=320,
            field="protocol.independent_base_draw_count",
        )
        matched_conditions = _exact_positive_int(
            protocol.get("matched_scenario_count"),
            expected=960,
            field="protocol.matched_scenario_count",
        )
        controller_evaluations = _exact_positive_int(
            protocol.get("controller_evaluation_count"),
            expected=1_920,
            field="protocol.controller_evaluation_count",
        )
        shortage_levels = _text_sequence(
            protocol.get("shortage_levels"), field="protocol.shortage_levels"
        )
        if shortage_levels != ("mild", "moderate", "severe"):
            raise PresentationDataError(
                "protocol.shortage_levels must be mild, moderate, and severe"
            )

        definitions = _mapping(payload.get("metric_definitions"), field="metric_definitions")
        expected_definition_fields = frozenset((*PRIMARY_METRICS, "power_balance_violation_rate"))
        _exact_fields(definitions, expected_definition_fields, field="metric_definitions")
        for key, (_, unit, _, direction) in _METRIC_PRESENTATION.items():
            definition = _mapping(definitions.get(key), field=f"metric_definitions.{key}")
            _validate_metric_definition(
                definition,
                field=f"metric_definitions.{key}",
                unit="ratio" if unit == "%" else unit,
                direction=direction,
                role="primary",
            )
        safety_definition = _mapping(
            definitions.get("power_balance_violation_rate"),
            field="metric_definitions.power_balance_violation_rate",
        )
        _validate_metric_definition(
            safety_definition,
            field="metric_definitions.power_balance_violation_rate",
            unit="ratio",
            direction="lower_is_better",
            role="safety",
        )

        base = _mapping(payload.get("base_evaluation"), field="base_evaluation")
        _exact_fields(base, _BASE_EVALUATION_FIELDS, field="base_evaluation")
        means = _mapping(base.get("controller_means"), field="base_evaluation.controller_means")
        if set(means) != set(CONTROLLER_NAMES):
            raise PresentationDataError(
                "base_evaluation.controller_means must contain exactly PPO and Fixed Rotation"
            )
        controller_values: dict[str, Mapping[str, Any]] = {}
        for controller_name in CONTROLLER_NAMES:
            values = _mapping(
                means.get(controller_name),
                field=f"base_evaluation.controller_means.{controller_name}",
            )
            _exact_fields(
                values,
                _CONTROLLER_MEAN_FIELDS,
                field=f"{controller_name} headline metrics",
            )
            _nonnegative_number(
                values.get("critical_outage_locality_hours"),
                field=f"{controller_name}.critical_outage_locality_hours",
            )
            for ratio_key in ("fairness", "utilization", "power_balance_violation_rate"):
                _unit_interval_number(values.get(ratio_key), field=f"{controller_name}.{ratio_key}")
            controller_values[controller_name] = values

        inference = _mapping(base.get("paired_inference"), field="base_evaluation.paired_inference")
        _exact_fields(
            inference,
            frozenset(PRIMARY_METRICS),
            field="base_evaluation.paired_inference",
        )
        metrics: list[MetricComparison] = []
        for key in _METRIC_ORDER:
            item = _mapping(inference.get(key), field=f"base_evaluation.paired_inference.{key}")
            _exact_fields(item, _INTERVAL_FIELDS, field=f"paired inference {key}")
            mean = _finite_number(item.get("mean_delta"), field=f"{key}.mean_delta")
            lower = _finite_number(item.get("ci95_lower"), field=f"{key}.ci95_lower")
            upper = _finite_number(item.get("ci95_upper"), field=f"{key}.ci95_upper")
            if lower > mean or mean > upper:
                raise PresentationDataError(f"{key} confidence interval must contain mean_delta")
            classification = _text(
                item.get("result_classification"), field=f"{key}.result_classification"
            )
            if classification not in {"favorable", "ambiguous", "adverse"}:
                raise PresentationDataError(f"{key}.result_classification is invalid")
            label, unit, display_scale, direction = _METRIC_PRESENTATION[key]
            metrics.append(
                MetricComparison(
                    key=key,
                    label=label,
                    unit=unit,
                    display_scale=display_scale,
                    direction=direction,
                    role="primary",
                    ppo=_finite_number(controller_values["PPO"].get(key), field=f"PPO.{key}"),
                    fixed_rotation=_finite_number(
                        controller_values["Fixed Rotation"].get(key),
                        field=f"Fixed Rotation.{key}",
                    ),
                    mean_delta=mean,
                    ci95_lower=lower,
                    ci95_upper=upper,
                    result_classification=classification,
                )
            )

        power_balance = _mapping(base.get("power_balance"), field="base_evaluation.power_balance")
        _exact_fields(
            power_balance,
            frozenset({"maximum_violation_rate"}),
            field="base_evaluation.power_balance",
        )
        maximum_violation_rate = _unit_interval_number(
            power_balance.get("maximum_violation_rate"),
            field="base_evaluation.power_balance.maximum_violation_rate",
        )
        controller_maximum = max(
            float(values["power_balance_violation_rate"]) for values in controller_values.values()
        )
        if maximum_violation_rate != controller_maximum:
            raise PresentationDataError(
                "base_evaluation.power_balance must equal the controller maximum"
            )

        limitations = _text_sequence(payload.get("limitations"), field="limitations")
        count_statement = (
            f"{independent_draws} untouched independent base draws × "
            f"{len(shortage_levels)} shortage levels = {matched_conditions} matched scenario "
            f"conditions · {controller_evaluations:,} controller evaluations"
        )
        return cls(
            artifact_id=artifact_id,
            evidence_role=evidence_role,
            artifact_sha256=ROBUSTNESS_SUPPLEMENT_SHA256,
            release_id=RELEASE_ID,
            controller_names=CONTROLLER_NAMES,
            count_statement=count_statement,
            independent_base_draw_count=independent_draws,
            shortage_level_count=len(shortage_levels),
            matched_scenario_count=matched_conditions,
            controller_evaluation_count=controller_evaluations,
            horizon_hours=HORIZON_HOURS,
            metrics=tuple(metrics),
            maximum_power_balance_violation_rate=maximum_violation_rate,
            limitations=limitations,
        )

    def metric(self, key: str) -> MetricComparison:
        """Return one named headline metric."""

        for metric in self.metrics:
            if metric.key == key:
                return metric
        raise KeyError(key)


def load_presentation_snapshot(
    path: str | Path | None = None,
    *,
    verifier: Callable[..., dict[str, Any]] = verify_robustness_supplement,
) -> PresentationSnapshot:
    """Load only the pinned, semantically verified robustness supplement."""

    evidence_path = (
        runtime_root() / "data" / "evidence" / "robustness-supplement.json"
        if path is None
        else Path(path)
    )
    try:
        payload = verifier(
            evidence_path,
            expected_sha256=ROBUSTNESS_SUPPLEMENT_SHA256,
        )
    except RobustnessArtifactError as error:
        raise PresentationDataError(f"robustness evidence failed verification: {error}") from error
    return PresentationSnapshot.from_mapping(_mapping(payload, field="evidence"))


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PresentationDataError(f"{field} must be an object")
    return value


def _exact_fields(value: Mapping[str, object], expected: frozenset[str], *, field: str) -> None:
    actual = frozenset(value)
    if actual != expected:
        raise PresentationDataError(
            f"{field} fields are not exact; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PresentationDataError(f"{field} must be a non-empty string")
    return value.strip()


def _text_sequence(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise PresentationDataError(f"{field} must be a list of strings")
    return tuple(_text(item, field=f"{field} item") for item in value)


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PresentationDataError(f"{field} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise PresentationDataError(f"{field} must be finite")
    return normalized


def _nonnegative_number(value: object, *, field: str) -> float:
    normalized = _finite_number(value, field=field)
    if normalized < 0.0:
        raise PresentationDataError(f"{field} must be non-negative")
    return normalized


def _unit_interval_number(value: object, *, field: str) -> float:
    normalized = _finite_number(value, field=field)
    if not 0.0 <= normalized <= 1.0:
        raise PresentationDataError(f"{field} must be between 0 and 1")
    return normalized


def _exact_positive_int(value: object, *, expected: int, field: str) -> int:
    if type(value) is not int or value != expected:
        raise PresentationDataError(f"{field} must be exactly {expected:,}")
    return value


def _validate_controllers(value: object) -> None:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != len(CONTROLLER_NAMES)
        or set(value) != set(CONTROLLER_NAMES)
    ):
        raise PresentationDataError("protocol must contain exactly PPO and Fixed Rotation")


def _validate_metric_definition(
    definition: Mapping[str, object],
    *,
    field: str,
    unit: str,
    direction: str,
    role: str,
) -> None:
    expected = {"unit": unit, "direction": direction, "role": role}
    _exact_fields(definition, frozenset(expected), field=field)
    if definition != expected:
        raise PresentationDataError(f"{field} does not match the locked metric definition")

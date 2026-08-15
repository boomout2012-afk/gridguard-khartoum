"""Validated, read-only evidence boundary for the presentation interface."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from gridguard.evaluation import (
    PAIRED_BENCHMARK_SHA256,
    EvaluationArtifactError,
    verify_paired_benchmark,
)
from gridguard.resources import runtime_root

CONTROLLER_NAMES = ("PPO", "Fixed Rotation")

_METRIC_PRESENTATION = {
    "critical_service_ratio": ("Critical service", "%", 100.0),
    "served_mwh": ("Energy served", "MWh", 1.0),
    "curtailed_mwh": ("Energy curtailed", "MWh", 1.0),
    "episode_reward": ("Episode reward", "points", 1.0),
    "utilization": ("Supply utilization", "%", 100.0),
    "fairness": ("Locality fairness", "%", 100.0),
    "switching_rate": ("Switching rate", "%", 100.0),
}
_METRIC_ORDER = tuple(_METRIC_PRESENTATION)


class PresentationDataError(ValueError):
    """Raised when evidence cannot be presented without changing its meaning."""


@dataclass(frozen=True, slots=True)
class MetricComparison:
    """One aggregate metric and its paired PPO-minus-baseline inference."""

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

    @property
    def ppo_is_favorable(self) -> bool:
        if self.direction == "higher_is_better":
            return self.mean_delta > 0
        return self.mean_delta < 0


@dataclass(frozen=True, slots=True)
class PresentationSnapshot:
    """Scientific values approved for display, separated from Qt widgets."""

    benchmark_id: str
    controller_names: tuple[str, str]
    count_statement: str
    independent_base_draw_count: int
    shortage_level_count: int
    matched_scenario_count: int
    controller_evaluation_count: int
    horizon_hours: int
    metrics: tuple[MetricComparison, ...]
    ppo_power_balance_violation_rate: float
    fixed_rotation_power_balance_violation_rate: float
    limitations: tuple[str, ...]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> Self:
        """Validate the paired-evidence fields consumed by the UI."""

        if payload.get("schema_version") != 1:
            raise PresentationDataError("evidence schema_version must be 1")
        benchmark_id = _text(payload.get("benchmark_id"), field="benchmark_id")
        protocol = _mapping(payload.get("protocol"), field="protocol")
        aggregate = _mapping(payload.get("aggregate"), field="aggregate")
        inference = _mapping(payload.get("paired_inference"), field="paired_inference")

        protocol_controllers = protocol.get("controllers")
        if (
            not isinstance(protocol_controllers, Sequence)
            or isinstance(protocol_controllers, (str, bytes))
            or set(protocol_controllers) != set(CONTROLLER_NAMES)
            or len(protocol_controllers) != len(CONTROLLER_NAMES)
        ):
            raise PresentationDataError("evidence must contain exactly PPO and Fixed Rotation")
        if set(aggregate) != set(CONTROLLER_NAMES):
            raise PresentationDataError("aggregate must contain exactly PPO and Fixed Rotation")

        ppo = _mapping(aggregate["PPO"], field="aggregate.PPO")
        fixed = _mapping(aggregate["Fixed Rotation"], field="aggregate.Fixed Rotation")
        ppo_power_balance_violation_rate = _unit_interval_number(
            ppo.get("power_balance_violation_rate"),
            field="aggregate.PPO.power_balance_violation_rate",
        )
        fixed_power_balance_violation_rate = _unit_interval_number(
            fixed.get("power_balance_violation_rate"),
            field="aggregate.Fixed Rotation.power_balance_violation_rate",
        )
        unknown_metrics = set(inference) - set(_METRIC_PRESENTATION)
        if unknown_metrics:
            raise PresentationDataError(
                "paired_inference contains unsupported metric(s): "
                + ", ".join(sorted(unknown_metrics))
            )
        if not inference:
            raise PresentationDataError("paired_inference must contain displayable metrics")

        metrics: list[MetricComparison] = []
        for key in _METRIC_ORDER:
            if key not in inference:
                continue
            item = _mapping(inference[key], field=f"paired_inference.{key}")
            direction = _text(item.get("direction"), field=f"{key}.direction")
            if direction not in {"higher_is_better", "lower_is_better"}:
                raise PresentationDataError(f"{key}.direction is invalid")
            if item.get("delta_definition") != "PPO - Fixed Rotation":
                raise PresentationDataError(f"{key}.delta_definition must be PPO - Fixed Rotation")

            mean = _finite_number(item.get("mean_delta"), field=f"{key}.mean_delta")
            lower = _finite_number(item.get("ci95_lower"), field=f"{key}.ci95_lower")
            upper = _finite_number(item.get("ci95_upper"), field=f"{key}.ci95_upper")
            if lower > mean or mean > upper:
                raise PresentationDataError(f"{key} confidence interval must contain mean_delta")
            label, unit, scale = _METRIC_PRESENTATION[key]
            metrics.append(
                MetricComparison(
                    key=key,
                    label=label,
                    unit=unit,
                    display_scale=scale,
                    direction=direction,
                    role=_text(item.get("role"), field=f"{key}.role"),
                    ppo=_finite_number(ppo.get(key), field=f"aggregate.PPO.{key}"),
                    fixed_rotation=_finite_number(
                        fixed.get(key), field=f"aggregate.Fixed Rotation.{key}"
                    ),
                    mean_delta=mean,
                    ci95_lower=lower,
                    ci95_upper=upper,
                )
            )

        limitations = _text_sequence(payload.get("limitations"), field="limitations")
        _text(protocol.get("count_statement"), field="count_statement")
        independent_draws = _positive_int(
            protocol.get("independent_base_draw_count"),
            field="independent_base_draw_count",
        )
        shortage_levels = _text_sequence(protocol.get("shortage_levels"), field="shortage_levels")
        if len(shortage_levels) != len(set(shortage_levels)):
            raise PresentationDataError("shortage_levels must not contain duplicates")
        matched_conditions = _positive_int(
            protocol.get("matched_scenario_count"), field="matched_scenario_count"
        )
        controller_evaluations = _positive_int(
            protocol.get("controller_evaluation_count"),
            field="controller_evaluation_count",
        )
        if independent_draws * len(shortage_levels) != matched_conditions:
            raise PresentationDataError(
                "independent draws × shortage levels must equal matched scenario conditions"
            )
        if controller_evaluations != matched_conditions * len(CONTROLLER_NAMES):
            raise PresentationDataError(
                "controller evaluations must equal matched conditions × controllers"
            )
        display_count_statement = (
            f"{independent_draws} independent base draws × {len(shortage_levels)} shortage "
            f"levels = {matched_conditions} matched scenario conditions · "
            f"{controller_evaluations} controller evaluations"
        )
        return cls(
            benchmark_id=benchmark_id,
            controller_names=CONTROLLER_NAMES,
            count_statement=display_count_statement,
            independent_base_draw_count=independent_draws,
            shortage_level_count=len(shortage_levels),
            matched_scenario_count=matched_conditions,
            controller_evaluation_count=controller_evaluations,
            horizon_hours=_positive_int(protocol.get("horizon_hours"), field="horizon_hours"),
            metrics=tuple(metrics),
            ppo_power_balance_violation_rate=ppo_power_balance_violation_rate,
            fixed_rotation_power_balance_violation_rate=(fixed_power_balance_violation_rate),
            limitations=limitations,
        )

    def metric(self, key: str) -> MetricComparison:
        """Return one named display metric."""

        for metric in self.metrics:
            if metric.key == key:
                return metric
        raise KeyError(key)


def load_presentation_snapshot(
    path: str | Path | None = None,
    *,
    verifier: Callable[..., dict[str, Any]] = verify_paired_benchmark,
) -> PresentationSnapshot:
    """Load only the pinned, semantically verified paired-evidence payload."""

    root = runtime_root()
    evidence_path = (
        root / "data" / "evidence" / "paired-benchmark.json" if path is None else Path(path)
    )
    frozen_path = root / "data" / "evidence" / "frozen-benchmark.json"
    try:
        payload = verifier(
            evidence_path,
            expected_sha256=PAIRED_BENCHMARK_SHA256,
            frozen_benchmark_path=frozen_path,
        )
    except EvaluationArtifactError as error:
        raise PresentationDataError(f"paired evidence failed verification: {error}") from error
    return PresentationSnapshot.from_mapping(_mapping(payload, field="evidence"))


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PresentationDataError(f"{field} must be an object")
    return value


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


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PresentationDataError(f"{field} must be a positive integer")
    return value


def _unit_interval_number(value: object, *, field: str) -> float:
    normalized = _finite_number(value, field=field)
    if not 0.0 <= normalized <= 1.0:
        raise PresentationDataError(f"{field} must be between 0 and 1")
    return normalized

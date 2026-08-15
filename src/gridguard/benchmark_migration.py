"""Convert the accepted research benchmark into a proposal-only evidence artifact."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_PPO_LABEL_PATTERN = re.compile(r"(?:^|\W)ppo(?:$|\W)", re.IGNORECASE)
_SEVERITIES = ("mild", "moderate", "severe")
_SEEDS = (50001, 50002, 50003)
_PUBLIC_CONTROLLERS = ("Fixed Rotation", "PPO")
_EPISODES_PER_CELL = 32
_SOURCE_REVISION = "52f6dd82605c993c8094448caecec86faa289c0c"
_SOURCE_BENCHMARK_SHA256 = "5f09625ff987440bc6e667550509cb7f1deac03a38c1b905aa284d65bf74023f"
_SOURCE_BENCHMARK_SIZE_BYTES = 26282
_FROZEN_BENCHMARK_SHA256 = "3d720ba87a747a5780119decd7144b4051fc36fc72ab50f229f5750ad749fd15"
_METRICS = (
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
)


class BenchmarkMigrationError(ValueError):
    """Raised when the frozen source benchmark cannot be canonicalized safely."""


def canonical_json_bytes(payload: object) -> bytes:
    """Serialize evidence deterministically for stable hashing."""

    try:
        text = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise BenchmarkMigrationError(f"evidence is not canonical JSON: {error}") from error
    return f"{text}\n".encode()


def canonicalize_source_benchmark(
    source: Mapping[str, object],
    *,
    source_sha256: str,
    source_size_bytes: int,
) -> dict[str, Any]:
    """Return the two-controller proposal slice of the accepted benchmark."""

    if not _SHA256_PATTERN.fullmatch(source_sha256):
        raise BenchmarkMigrationError("source artifact has an invalid SHA-256 digest")
    if (
        isinstance(source_size_bytes, bool)
        or not isinstance(source_size_bytes, int)
        or source_size_bytes <= 0
    ):
        raise BenchmarkMigrationError("source artifact size must be a positive integer")

    raw_aggregate = _require_mapping(source.get("aggregate"), field="aggregate")
    raw_rows = source.get("rows")
    if not isinstance(raw_rows, list):
        raise BenchmarkMigrationError("rows must be a list")

    source_labels = _find_source_controller_labels(raw_aggregate)
    public_by_source = {
        source_labels["Fixed Rotation"]: "Fixed Rotation",
        source_labels["PPO"]: "PPO",
    }

    canonical_cells: list[dict[str, Any]] = []
    seen_cells: set[tuple[str, str, int]] = set()
    for row_index, raw_row in enumerate(raw_rows):
        row = _require_mapping(raw_row, field=f"rows[{row_index}]")
        source_controller = row.get("controller")
        if not isinstance(source_controller, str):
            raise BenchmarkMigrationError(f"rows[{row_index}].controller must be a string")
        if source_controller not in public_by_source:
            continue

        controller = public_by_source[source_controller]
        severity = _require_choice(row, "severity", _SEVERITIES, row_index=row_index)
        seed = _require_integer(row, "seed", row_index=row_index)
        if seed not in _SEEDS:
            raise BenchmarkMigrationError(f"rows[{row_index}].seed is outside the approved grid")
        episodes = _require_integer(row, "episodes", row_index=row_index)
        if episodes != _EPISODES_PER_CELL:
            raise BenchmarkMigrationError(
                f"rows[{row_index}].episodes must be {_EPISODES_PER_CELL}"
            )

        cell_key = (controller, severity, seed)
        if cell_key in seen_cells:
            raise BenchmarkMigrationError(f"duplicate approved benchmark cell: {cell_key}")
        seen_cells.add(cell_key)

        metrics = {
            metric: _require_finite_number(row, metric, row_index=row_index) for metric in _METRICS
        }
        mean_step_reward = _require_finite_number(
            row,
            "mean_step_reward",
            row_index=row_index,
        )
        _validate_cell_identities(metrics, mean_step_reward=mean_step_reward, row_index=row_index)

        canonical_cells.append(
            {
                "controller": controller,
                "episode_count": episodes,
                "seed": seed,
                "severity": severity,
                **metrics,
            }
        )

    if len(canonical_cells) != 18:
        raise BenchmarkMigrationError(f"expected 18 approved rows, found {len(canonical_cells)}")
    _validate_complete_grid(seen_cells)
    _validate_matched_inputs(canonical_cells)

    controller_order = {name: index for index, name in enumerate(_PUBLIC_CONTROLLERS)}
    severity_order = {name: index for index, name in enumerate(_SEVERITIES)}
    canonical_cells.sort(
        key=lambda row: (
            severity_order[row["severity"]],
            row["seed"],
            controller_order[row["controller"]],
        )
    )

    canonical_aggregate = _aggregate_cells(canonical_cells)
    _validate_source_aggregates(
        raw_aggregate,
        source_labels=source_labels,
        canonical_aggregate=canonical_aggregate,
    )

    return {
        "aggregate": canonical_aggregate,
        "benchmark_id": "proposal-ppo-vs-fixed-rotation",
        "cells": canonical_cells,
        "limitations": [
            "Synthetic simulation evidence; not real utility measurements.",
            "The frozen source stores batch-cell summaries, not per-episode rows.",
            (
                "Confidence intervals, p-values, and per-episode win rates are not "
                "supported by this artifact."
            ),
        ],
        "protocol": {
            "controller_evaluation_count": 576,
            "controllers": list(_PUBLIC_CONTROLLERS),
            "episode_count_per_cell": _EPISODES_PER_CELL,
            "evidence_granularity": "batch-cell summaries",
            "horizon_hours": 24,
            "matched_scenario_count": 288,
            "seeds": list(_SEEDS),
            "severities": list(_SEVERITIES),
            "stored_matched_cell_count": 9,
        },
        "schema_version": 1,
        "source": {
            "artifact_sha256": source_sha256,
            "artifact_size_bytes": source_size_bytes,
            "source_revision": _SOURCE_REVISION,
        },
    }


def migrate_frozen_source_benchmark(source_path: str | Path) -> bytes:
    """Reproduce the frozen evidence only from the exact audited source bytes."""

    try:
        source_bytes = Path(source_path).read_bytes()
    except OSError as error:
        raise BenchmarkMigrationError(
            f"trusted source benchmark cannot be read: {error}"
        ) from error
    actual_hash = hashlib.sha256(source_bytes).hexdigest()
    if len(source_bytes) != _SOURCE_BENCHMARK_SIZE_BYTES or actual_hash != _SOURCE_BENCHMARK_SHA256:
        raise BenchmarkMigrationError(
            "trusted source benchmark bytes do not match the audited size and SHA-256"
        )
    try:
        payload = json.loads(
            source_bytes.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_fields,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise BenchmarkMigrationError(
            f"trusted source benchmark is not valid UTF-8 JSON: {error}"
        ) from error
    source = _require_mapping(payload, field="source benchmark")
    migrated = canonicalize_source_benchmark(
        source,
        source_sha256=_SOURCE_BENCHMARK_SHA256,
        source_size_bytes=_SOURCE_BENCHMARK_SIZE_BYTES,
    )
    encoded = canonical_json_bytes(migrated)
    if hashlib.sha256(encoded).hexdigest() != _FROZEN_BENCHMARK_SHA256:
        raise BenchmarkMigrationError(
            "migrated benchmark does not match the reviewed frozen evidence trust anchor"
        )
    return encoded


def _find_source_controller_labels(aggregate: Mapping[str, object]) -> dict[str, str]:
    labels = [label for label in aggregate if isinstance(label, str)]
    fixed = [label for label in labels if label.strip().casefold() == "fixed rotation"]
    learned = [label for label in labels if _PPO_LABEL_PATTERN.search(label)]
    if len(fixed) != 1 or len(learned) != 1:
        raise BenchmarkMigrationError(
            "source benchmark must contain exactly one Fixed Rotation and one PPO aggregate"
        )
    return {"Fixed Rotation": fixed[0], "PPO": learned[0]}


def _aggregate_cells(cells: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    return {
        controller: {
            metric: math.fsum(row[metric] for row in cells if row["controller"] == controller) / 9.0
            for metric in _METRICS
        }
        for controller in _PUBLIC_CONTROLLERS
    }


def _validate_source_aggregates(
    raw_aggregate: Mapping[str, object],
    *,
    source_labels: Mapping[str, str],
    canonical_aggregate: Mapping[str, Mapping[str, float]],
) -> None:
    for controller in _PUBLIC_CONTROLLERS:
        source_values = _require_mapping(
            raw_aggregate.get(source_labels[controller]),
            field=f"aggregate.{controller}",
        )
        for metric in _METRICS:
            source_value = _require_finite_number(
                source_values,
                metric,
                row_index=None,
            )
            recomputed = canonical_aggregate[controller][metric]
            if not math.isclose(source_value, recomputed, abs_tol=1e-12, rel_tol=0.0):
                raise BenchmarkMigrationError(
                    f"aggregate {controller}.{metric} does not match its nine cells"
                )


def _validate_complete_grid(seen: set[tuple[str, str, int]]) -> None:
    expected = {
        (controller, severity, seed)
        for controller in _PUBLIC_CONTROLLERS
        for severity in _SEVERITIES
        for seed in _SEEDS
    }
    if seen != expected:
        raise BenchmarkMigrationError("approved benchmark grid is incomplete or contains extras")


def _validate_matched_inputs(cells: list[dict[str, Any]]) -> None:
    by_scenario: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in cells:
        by_scenario.setdefault((row["severity"], row["seed"]), []).append(row)
    for scenario_key, pair in by_scenario.items():
        if len(pair) != 2:
            raise BenchmarkMigrationError(f"scenario cell is not paired: {scenario_key}")
        for metric in ("demand_mwh", "available_mwh"):
            if pair[0][metric] != pair[1][metric]:
                raise BenchmarkMigrationError(
                    f"matched controller inputs differ for {scenario_key}: {metric}"
                )


def _validate_cell_identities(
    metrics: Mapping[str, float],
    *,
    mean_step_reward: float,
    row_index: int,
) -> None:
    for metric in ("demand_mwh", "available_mwh", "served_mwh", "curtailed_mwh"):
        if metrics[metric] < 0.0:
            raise BenchmarkMigrationError(f"rows[{row_index}].{metric} must be nonnegative")
    for metric in (
        "critical_service_ratio",
        "utilization",
        "fairness",
        "switching_rate",
        "oversubscription",
        "power_balance_violation_rate",
    ):
        if not 0.0 <= metrics[metric] <= 1.0:
            raise BenchmarkMigrationError(
                f"rows[{row_index}].{metric} must be between zero and one"
            )
    if metrics["available_mwh"] > metrics["demand_mwh"] + 0.002:
        raise BenchmarkMigrationError(
            f"rows[{row_index}] has available energy above shortage demand"
        )
    if metrics["served_mwh"] > min(metrics["demand_mwh"], metrics["available_mwh"]) + 0.002:
        raise BenchmarkMigrationError(
            f"rows[{row_index}] serves more energy than demand or availability"
        )
    energy_residual = metrics["demand_mwh"] - (metrics["served_mwh"] + metrics["curtailed_mwh"])
    if abs(energy_residual) > 0.002:
        raise BenchmarkMigrationError(f"rows[{row_index}] violates the energy identity")
    if not math.isclose(
        metrics["episode_reward"],
        24.0 * mean_step_reward,
        abs_tol=5e-6,
        rel_tol=0.0,
    ):
        raise BenchmarkMigrationError(f"rows[{row_index}] violates the reward identity")
    if metrics["power_balance_violation_rate"] != 0.0:
        raise BenchmarkMigrationError(
            f"rows[{row_index}] has a nonzero power-balance violation rate"
        )


def _require_mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BenchmarkMigrationError(f"{field} must be an object")
    return value


def _require_choice(
    row: Mapping[str, Any],
    field: str,
    choices: tuple[str, ...],
    *,
    row_index: int,
) -> str:
    value = row.get(field)
    if value not in choices:
        raise BenchmarkMigrationError(f"rows[{row_index}].{field} is invalid")
    return value


def _require_integer(row: Mapping[str, Any], field: str, *, row_index: int) -> int:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise BenchmarkMigrationError(f"rows[{row_index}].{field} must be an integer")
    return value


def _require_finite_number(
    row: Mapping[str, Any],
    field: str,
    *,
    row_index: int | None,
) -> float:
    value = row.get(field)
    prefix = f"rows[{row_index}]" if row_index is not None else "aggregate"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkMigrationError(f"{prefix}.{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise BenchmarkMigrationError(f"{prefix}.{field} must be a finite number")
    return result


def _reject_duplicate_json_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in pairs:
        if name in result:
            raise BenchmarkMigrationError(
                f"trusted source benchmark contains duplicate field {name!r}"
            )
        result[name] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> None:
    raise BenchmarkMigrationError(
        f"trusted source benchmark contains non-finite JSON value {value}"
    )

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from gridguard.benchmark_migration import (
    BenchmarkMigrationError,
    canonical_json_bytes,
    canonicalize_source_benchmark,
    migrate_frozen_source_benchmark,
)

SEVERITIES = ("mild", "moderate", "severe")
SEEDS = (50001, 50002, 50003)


def _metrics(offset: float) -> dict[str, float]:
    return {
        "episode_reward": 10.0 + offset,
        "mean_step_reward": (10.0 + offset) / 24.0,
        "demand_mwh": 100.0,
        "available_mwh": 60.0,
        "served_mwh": 50.0 + offset,
        "curtailed_mwh": 50.0 - offset,
        "critical_service_ratio": 0.5 + offset / 100.0,
        "utilization": 0.8 + offset / 100.0,
        "fairness": 0.9 + offset / 100.0,
        "switching_rate": 0.25,
        "oversubscription": 0.0,
        "power_balance_violation_rate": 0.0,
    }


def _source_payload() -> dict[str, object]:
    fixed_name = "Fixed rotation"
    learned_name = "trained PPO source"
    rows: list[dict[str, object]] = []
    for severity in SEVERITIES:
        for seed in SEEDS:
            for controller, offset in ((fixed_name, 0.0), (learned_name, 5.0)):
                rows.append(
                    {
                        "controller": controller,
                        "severity": severity,
                        "seed": seed,
                        "episodes": 32,
                        **_metrics(offset),
                    }
                )
    return {
        "aggregate": {
            fixed_name: _metrics(0.0),
            learned_name: _metrics(5.0),
            "Unrelated controller": _metrics(2.0),
        },
        "rows": rows,
    }


def test_canonicalize_source_benchmark_keeps_only_approved_controllers() -> None:
    result = canonicalize_source_benchmark(
        _source_payload(),
        source_sha256="a" * 64,
        source_size_bytes=1234,
    )

    assert result["schema_version"] == 1
    assert result["protocol"]["controllers"] == ["Fixed Rotation", "PPO"]
    assert result["protocol"]["matched_scenario_count"] == 288
    assert result["protocol"]["controller_evaluation_count"] == 576
    assert result["protocol"]["stored_matched_cell_count"] == 9
    assert set(result["aggregate"]) == {"Fixed Rotation", "PPO"}
    assert len(result["cells"]) == 18
    assert {row["controller"] for row in result["cells"]} == {
        "Fixed Rotation",
        "PPO",
    }


def test_canonicalize_source_benchmark_is_stable_and_contains_no_source_label() -> None:
    result = canonicalize_source_benchmark(
        _source_payload(),
        source_sha256="a" * 64,
        source_size_bytes=1234,
    )

    encoded = canonical_json_bytes(result)

    assert encoded.endswith(b"\n")
    assert b"trained PPO source" not in encoded
    assert encoded == canonical_json_bytes(json.loads(encoded))
    assert (
        hashlib.sha256(encoded).hexdigest()
        == "8f5c220225a8bed69ff3b5482fe93d408691bebf81e7503ef03f568284d2e340"
    )


def test_canonicalize_source_benchmark_rejects_missing_cell() -> None:
    source = _source_payload()
    source["rows"].pop()  # type: ignore[union-attr]

    with pytest.raises(BenchmarkMigrationError, match="18 approved rows"):
        canonicalize_source_benchmark(
            source,
            source_sha256="a" * 64,
            source_size_bytes=1234,
        )


def test_canonicalize_source_benchmark_rejects_energy_identity_drift() -> None:
    source = _source_payload()
    source["rows"][0]["served_mwh"] = 40.0  # type: ignore[index]

    with pytest.raises(BenchmarkMigrationError, match="energy identity"):
        canonicalize_source_benchmark(
            source,
            source_sha256="a" * 64,
            source_size_bytes=1234,
        )


def test_canonicalize_source_benchmark_rejects_nonfinite_metric() -> None:
    source = _source_payload()
    source["rows"][0]["fairness"] = float("nan")  # type: ignore[index]

    with pytest.raises(BenchmarkMigrationError, match="finite number"):
        canonicalize_source_benchmark(
            source,
            source_sha256="a" * 64,
            source_size_bytes=1234,
        )


@pytest.mark.parametrize(
    ("metric", "value", "message"),
    [
        ("demand_mwh", -1.0, "nonnegative"),
        ("available_mwh", -1.0, "nonnegative"),
        ("served_mwh", -1.0, "nonnegative"),
        ("curtailed_mwh", -1.0, "nonnegative"),
        ("critical_service_ratio", 1.01, "between zero and one"),
        ("utilization", -0.01, "between zero and one"),
        ("fairness", 1.01, "between zero and one"),
        ("switching_rate", -0.01, "between zero and one"),
        ("oversubscription", 1.01, "between zero and one"),
    ],
)
def test_canonicalize_source_benchmark_rejects_out_of_domain_metrics(
    metric: str,
    value: float,
    message: str,
) -> None:
    source = _source_payload()
    source["rows"][0][metric] = value  # type: ignore[index]

    with pytest.raises(BenchmarkMigrationError, match=message):
        canonicalize_source_benchmark(
            source,
            source_sha256="a" * 64,
            source_size_bytes=1234,
        )


def test_frozen_source_migration_rejects_untrusted_input_bytes(tmp_path: Path) -> None:
    source = tmp_path / "benchmark.json"
    source.write_text(json.dumps(_source_payload()), encoding="utf-8")

    with pytest.raises(BenchmarkMigrationError, match="trusted source"):
        migrate_frozen_source_benchmark(source)


@pytest.mark.parametrize("invalid_size", [True, 1.5, 0, -1])
def test_canonicalize_source_benchmark_rejects_invalid_source_size(
    invalid_size: object,
) -> None:
    with pytest.raises(BenchmarkMigrationError, match="positive integer"):
        canonicalize_source_benchmark(
            _source_payload(),
            source_sha256="a" * 64,
            source_size_bytes=invalid_size,  # type: ignore[arg-type]
        )


def test_canonicalize_source_benchmark_rejects_nonstring_controller() -> None:
    source = _source_payload()
    source["rows"][0]["controller"] = []  # type: ignore[index]

    with pytest.raises(BenchmarkMigrationError, match="controller must be a string"):
        canonicalize_source_benchmark(
            source,
            source_sha256="a" * 64,
            source_size_bytes=1234,
        )

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from gridguard.evaluation import (
    INFERENCE_METRICS,
    METRIC_DEFINITIONS,
    PRIMARY_METRICS,
    assert_paired_benchmark_integrity,
    validate_paired_benchmark_payload,
)

PROJECT_ROOT = Path(__file__).parents[1]
PAIRED_BENCHMARK = PROJECT_ROOT / "data" / "evidence" / "paired-benchmark.json"
FROZEN_BENCHMARK = PROJECT_ROOT / "data" / "evidence" / "frozen-benchmark.json"


def _paired_payload() -> dict[str, object]:
    return json.loads(PAIRED_BENCHMARK.read_text(encoding="utf-8"))


def _legacy_numeric_projection(payload: dict[str, object]) -> dict[str, object]:
    protocol = payload["protocol"]
    inference = payload["paired_inference"]
    frozen_parity = payload["frozen_parity"]
    assert isinstance(protocol, dict)
    assert isinstance(inference, dict)
    assert isinstance(frozen_parity, dict)
    legacy_inference_metrics = INFERENCE_METRICS[:-1]
    return {
        "aggregate": payload["aggregate"],
        "cells": payload["cells"],
        "paired_inference": {
            metric: {
                field: inference[metric][field]
                for field in ("mean_delta", "ci95_lower", "ci95_upper")
            }
            for metric in legacy_inference_metrics
        },
        "frozen_parity": frozen_parity["maximum_absolute_error"],
        "protocol": {
            key: value
            for key, value in protocol.items()
            if key
            not in {
                "controllers",
                "count_statement",
                "delta_definition",
                "severity_coupling",
                "shortage_levels",
            }
        },
    }


def test_metric_roles_make_training_objective_diagnostic() -> None:
    assert frozenset({"critical_outage_locality_hours", "fairness", "utilization"}) == (
        PRIMARY_METRICS
    )
    assert INFERENCE_METRICS[-1] == "critical_outage_locality_hours"
    assert METRIC_DEFINITIONS["episode_reward"]["role"] == "diagnostic"
    assert METRIC_DEFINITIONS["episode_reward"]["label"] == (
        "Training-objective score (diagnostic)"
    )
    assert METRIC_DEFINITIONS["critical_service_ratio"]["role"] == "supporting"
    assert METRIC_DEFINITIONS["served_mwh"]["role"] == "supporting"
    assert METRIC_DEFINITIONS["curtailed_mwh"]["role"] == "complementary"
    assert METRIC_DEFINITIONS["demand_mwh"]["role"] == "input"
    assert METRIC_DEFINITIONS["available_mwh"]["role"] == "input"


def test_paired_v3_is_explicitly_accepted_reproduction_evidence() -> None:
    payload = _paired_payload()

    assert payload["schema_version"] == 3
    assert payload["evidence_role"] == "accepted_reproduction"
    independence = payload["evaluation_independence"]
    assert independence == {
        "gradient_disjoint": True,
        "periodic_validation_disjoint": True,
        "untouched_final_test": False,
        "used_in_model_selection": True,
    }
    assert payload["paired_inference"]["episode_reward"]["role"] == "diagnostic"
    assert payload["paired_inference"]["critical_outage_locality_hours"]["role"] == "primary"


def test_paired_v2_preserves_every_legacy_numeric_leaf() -> None:
    projection = _legacy_numeric_projection(_paired_payload())
    encoded = json.dumps(
        projection,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")

    assert hashlib.sha256(encoded).hexdigest() == (
        "6df5164eab9aa7ce83c741bbfc5ea7b996b4977511e39719d1161e080bf438c6"
    )


def test_integrity_check_is_sign_neutral_but_still_cross_checks_frozen_parity() -> None:
    payload = _paired_payload()
    ambiguous = copy.deepcopy(payload)
    ambiguous["paired_inference"]["critical_outage_locality_hours"]["ci95_upper"] = 1.0

    validate_paired_benchmark_payload(ambiguous)
    assert_paired_benchmark_integrity(
        ambiguous,
        frozen_benchmark_path=FROZEN_BENCHMARK,
    )

    assert hashlib.sha256(FROZEN_BENCHMARK.read_bytes()).hexdigest() == (
        "3d720ba87a747a5780119decd7144b4051fc36fc72ab50f229f5750ad749fd15"
    )

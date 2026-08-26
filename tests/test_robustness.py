from __future__ import annotations

import hashlib
import inspect
import math
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import torch

from gridguard.domain import LOCALITIES, LOCALITY_DEMAND_WEIGHTS, SECTOR_SHARES, SECTORS
from gridguard.robustness import (
    FORECAST_METHODS,
    PRIMARY_METRICS,
    ROBUSTNESS_BATCH_SIZE,
    ROBUSTNESS_SEEDS,
    RobustnessArtifactError,
    build_robustness_supplement_payload,
    build_sensitivity_variants,
    canonical_robustness_json_bytes,
    forecast_error_sufficient_statistics,
    run_full_robustness_evaluation,
    validate_robustness_supplement_payload,
    verify_robustness_supplement,
)


def _interval(metric: str, mean: float = 0.0) -> dict[str, object]:
    lower = mean - 0.1
    upper = mean + 0.1
    if metric == "critical_outage_locality_hours":
        classification = "favorable" if upper < 0 else "adverse" if lower > 0 else "ambiguous"
    else:
        classification = "favorable" if lower > 0 else "adverse" if upper < 0 else "ambiguous"
    return {
        "mean_delta": mean,
        "ci95_lower": lower,
        "ci95_upper": upper,
        "result_classification": classification,
    }


def _forecast_method(window_hours: int) -> dict[str, object]:
    per_seed = []
    for seed in ROBUSTNESS_SEEDS:
        per_seed.append(
            {
                "seed": seed,
                "locality_sector": {
                    "count": 23_520,
                    "sum_absolute_error": 23_520.0,
                    "sum_squared_error": 47_040.0,
                },
                "system_total": {
                    "count": 672,
                    "sum_absolute_error": 1_344.0,
                    "sum_squared_error": 4_032.0,
                },
            }
        )
    return {
        "window_hours": window_hours,
        "per_seed": per_seed,
        "pooled": {
            "locality_sector": {
                "count": 235_200,
                "mae": 1.0,
                "rmse": math.sqrt(2.0),
            },
            "system_total": {
                "count": 6_720,
                "mae": 2.0,
                "rmse": math.sqrt(6.0),
            },
        },
    }


def _valid_payload() -> dict[str, object]:
    variants = []
    for variant in build_sensitivity_variants():
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
                "paired_inference": {metric: _interval(metric) for metric in PRIMARY_METRICS},
                "difference_in_differences": {
                    metric: _interval(metric) for metric in PRIMARY_METRICS
                },
            }
        )
    trailing = _forecast_method(3)
    persistence = _forecast_method(1)
    return {
        "schema": {"name": "gridguard.robustness_supplement", "version": 1},
        "evidence_role": "untouched_confirmatory",
        "provenance": {
            "artifact_id": "gridguard-robustness-supplement-v1",
            "source_revision": "a" * 40,
            "model_manifest": {
                "path": "data/model/manifest.json",
                "sha256": "b" * 64,
                "size_bytes": 100,
            },
            "actor": {
                "path": "data/model/ppo-policy.npz",
                "sha256": "c" * 64,
                "size_bytes": 200,
            },
            "scenario_generator": {
                "path": "src/gridguard/scenarios.py",
                "sha256": "d" * 64,
                "size_bytes": 300,
            },
        },
        "protocol": {
            "seeds": list(ROBUSTNESS_SEEDS),
            "episodes_per_seed": 32,
            "independent_base_draw_count": 320,
            "shortage_levels": ["mild", "moderate", "severe"],
            "matched_scenario_count": 960,
            "controllers": ["Fixed Rotation", "PPO"],
            "controller_evaluation_count": 1_920,
            "delta_definition": "PPO - Fixed Rotation",
            "bootstrap": {"replicates": 10_000, "seed": 20260826, "ci_percent": 95},
            "sensitivity_parameterization_count": 25,
            "sensitivity_matched_scenario_count": 24_000,
            "sensitivity_controller_evaluation_count": 48_000,
        },
        "metric_definitions": {
            "critical_outage_locality_hours": {
                "unit": "locality-hours",
                "direction": "lower_is_better",
                "role": "primary",
            },
            "fairness": {"unit": "ratio", "direction": "higher_is_better", "role": "primary"},
            "utilization": {
                "unit": "ratio",
                "direction": "higher_is_better",
                "role": "primary",
            },
            "power_balance_violation_rate": {
                "unit": "ratio",
                "direction": "lower_is_better",
                "role": "safety",
            },
        },
        "base_evaluation": {
            "controller_means": {
                controller: {
                    "critical_outage_locality_hours": 10.0,
                    "fairness": 0.9,
                    "utilization": 0.8,
                    "power_balance_violation_rate": 0.0,
                }
                for controller in ("Fixed Rotation", "PPO")
            },
            "paired_inference": {metric: _interval(metric) for metric in PRIMARY_METRICS},
            "power_balance": {"maximum_violation_rate": 0.0},
        },
        "sensitivity": {
            "base_parameters": {
                "locality_weights": list(LOCALITY_DEMAND_WEIGHTS),
                "sector_shares": list(SECTOR_SHARES),
            },
            "variants": variants,
        },
        "forecast_evaluation": {
            "target_hours": list(range(3, 24)),
            "trace_count": 320,
            "locality_sector_prediction_count": 235_200,
            "system_total_prediction_count": 6_720,
            "methods": {
                "trailing_mean_3h": trailing,
                "persistence_1h": persistence,
            },
            "comparison": {
                "trailing_minus_persistence": {
                    "locality_sector_mae": 0.0,
                    "locality_sector_rmse": 0.0,
                    "system_total_mae": 0.0,
                    "system_total_rmse": 0.0,
                },
                "lower_error_method": {
                    "locality_sector_mae": "tie",
                    "locality_sector_rmse": "tie",
                    "system_total_mae": "tie",
                    "system_total_rmse": "tie",
                },
            },
        },
        "limitations": [
            "Synthetic demand traces do not establish real-world forecast accuracy.",
            "Sensitivity covers only one-factor-at-a-time ±25% demand-allocation perturbations.",
            "Observed interval direction is descriptive and is not a release gate.",
        ],
    }


def test_sensitivity_protocol_has_exactly_twenty_four_normalized_ofat_variants() -> None:
    variants = build_sensitivity_variants()

    assert len(variants) == 2 * (len(LOCALITIES) + len(SECTORS)) == 24
    assert len({variant.variant_id for variant in variants}) == 24
    assert {variant.factor for variant in variants} == {0.75, 1.25}
    assert {variant.parameter for variant in variants} == {
        "locality_weight",
        "sector_share",
    }
    for variant in variants:
        assert sum(variant.parameters.locality_weights) == pytest.approx(
            sum(LOCALITY_DEMAND_WEIGHTS), abs=1e-12
        )
        assert sum(variant.parameters.sector_shares) == pytest.approx(1.0, abs=1e-12)
        with pytest.raises(FrozenInstanceError):
            variant.factor = 1.0  # type: ignore[misc]


def test_sensitivity_variants_change_only_one_component_before_renormalization() -> None:
    variants = build_sensitivity_variants()

    for variant in variants:
        if variant.parameter == "locality_weight":
            base = LOCALITY_DEMAND_WEIGHTS
            changed = variant.parameters.locality_weights
        else:
            base = SECTOR_SHARES
            changed = variant.parameters.sector_shares
        relative = [value / original for value, original in zip(changed, base, strict=True)]
        unchanged_ratios = [
            ratio for index, ratio in enumerate(relative) if index != variant.component_index
        ]
        assert max(unchanged_ratios) - min(unchanged_ratios) < 1e-12
        assert relative[variant.component_index] / unchanged_ratios[0] == pytest.approx(
            variant.factor,
            abs=1e-12,
        )


def test_robustness_protocol_constants_lock_untouched_workload() -> None:
    assert tuple(range(81001, 81011)) == ROBUSTNESS_SEEDS
    assert ROBUSTNESS_BATCH_SIZE == 32
    assert PRIMARY_METRICS == (
        "critical_outage_locality_hours",
        "fairness",
        "utilization",
    )
    assert FORECAST_METHODS == ("trailing_mean_3h", "persistence_1h")


def test_forecast_statistics_align_both_methods_to_target_hours_three_through_twenty_three() -> (
    None
):
    hours = torch.arange(24, dtype=torch.float64)
    demand = hours.reshape(1, 1, 24, 1, 1).expand(2, 3, 24, 7, 5).clone()

    statistics = forecast_error_sufficient_statistics(demand)

    # Shape: method, seed, aggregation(locality-sector/system), count/sum_abs/sum_squared.
    assert statistics.shape == (2, 2, 2, 3)
    trailing_error = torch.full((21,), 2.0, dtype=torch.float64)
    persistence_error = torch.ones(21, dtype=torch.float64)
    per_seed_local_count = 3 * 21 * 7 * 5
    per_seed_system_count = 3 * 21
    expected = {
        0: (trailing_error, 35.0 * trailing_error),
        1: (persistence_error, 35.0 * persistence_error),
    }
    for method_index, (local_error, system_error) in expected.items():
        assert statistics[method_index, 0, 0, 0].item() == per_seed_local_count
        assert statistics[method_index, 0, 1, 0].item() == per_seed_system_count
        assert statistics[method_index, 0, 0, 1].item() == pytest.approx(
            3 * 7 * 5 * local_error.sum().item()
        )
        assert statistics[method_index, 0, 1, 1].item() == pytest.approx(
            3 * system_error.sum().item()
        )
        assert statistics[method_index, 0, 0, 2].item() == pytest.approx(
            3 * 7 * 5 * local_error.square().sum().item()
        )
        assert statistics[method_index, 0, 1, 2].item() == pytest.approx(
            3 * system_error.square().sum().item()
        )
    assert bool(torch.isfinite(statistics).all().item())


@pytest.mark.parametrize("shape", [(3, 24, 7, 5), (2, 3, 23, 7, 5), (2, 3, 24, 7, 4)])
def test_forecast_statistics_reject_misaligned_trace_cubes(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match=r"\[seed, batch, 24, 7, 5\]"):
        forecast_error_sufficient_statistics(torch.ones(shape))


def test_forecast_statistics_reject_invalid_values() -> None:
    demand = torch.ones((2, 3, 24, 7, 5))
    demand[0, 0, 0, 0, 0] = math.nan

    with pytest.raises(ValueError, match="finite"):
        forecast_error_sufficient_statistics(demand)


def test_strict_validator_accepts_sign_neutral_complete_payload() -> None:
    payload = _valid_payload()
    adverse = payload["base_evaluation"]["paired_inference"]  # type: ignore[index]
    adverse["fairness"] = _interval("fairness", mean=-0.5)  # type: ignore[index]
    payload["base_evaluation"]["controller_means"]["PPO"]["fairness"] = 0.4  # type: ignore[index]
    for variant in payload["sensitivity"]["variants"]:  # type: ignore[index,union-attr]
        variant["paired_inference"]["fairness"] = _interval("fairness", mean=-0.5)

    validated = validate_robustness_supplement_payload(payload)

    assert validated is payload
    assert adverse["fairness"]["result_classification"] == "adverse"  # type: ignore[index]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.__setitem__("unexpected", True), "fields"),
        (
            lambda payload: payload["schema"].__setitem__("version", True),  # type: ignore[union-attr]
            "version",
        ),
        (
            lambda payload: payload["protocol"].__setitem__(  # type: ignore[union-attr]
                "matched_scenario_count", 959
            ),
            "matched_scenario_count",
        ),
        (
            lambda payload: payload["sensitivity"]["variants"].pop(),  # type: ignore[index,union-attr]
            "24 variants",
        ),
        (
            lambda payload: payload["sensitivity"]["variants"][0].__setitem__(  # type: ignore[index]
                "component_index", False
            ),
            "component_index",
        ),
        (
            lambda payload: payload["base_evaluation"]["paired_inference"][  # type: ignore[index]
                "utilization"
            ].__setitem__("result_classification", "favorable"),
            "classification",
        ),
        (
            lambda payload: payload["base_evaluation"]["paired_inference"][  # type: ignore[index]
                "fairness"
            ].__setitem__("mean_delta", 0.05),
            "controller means",
        ),
        (
            lambda payload: payload["sensitivity"]["variants"][0][  # type: ignore[index]
                "paired_inference"
            ]["fairness"].__setitem__("mean_delta", 0.05),
            "difference-in-differences",
        ),
        (
            lambda payload: payload["forecast_evaluation"]["methods"][  # type: ignore[index]
                "trailing_mean_3h"
            ]["pooled"]["locality_sector"].__setitem__("mae", 9.0),
            "pooled",
        ),
        (
            lambda payload: payload["base_evaluation"]["power_balance"].__setitem__(  # type: ignore[index]
                "maximum_violation_rate", 0.1
            ),
            "power balance",
        ),
    ],
)
def test_strict_validator_rejects_schema_and_arithmetic_drift(mutation, message: str) -> None:
    payload = _valid_payload()
    mutation(payload)

    with pytest.raises(RobustnessArtifactError, match=message):
        validate_robustness_supplement_payload(payload)


def test_canonical_bytes_and_host_verifier_reject_duplicates_and_hash_drift(tmp_path: Path) -> None:
    payload = _valid_payload()
    encoded = canonical_robustness_json_bytes(payload)
    path = tmp_path / "robustness.json"
    path.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()

    verified = verify_robustness_supplement(path, expected_sha256=digest)

    assert verified["schema"] == {"name": "gridguard.robustness_supplement", "version": 1}
    with pytest.raises(RobustnessArtifactError, match="SHA-256"):
        verify_robustness_supplement(path, expected_sha256="0" * 64)

    duplicate = encoded.replace(
        b'"evidence_role":',
        b'"evidence_role":"duplicate","evidence_role":',
        1,
    )
    path.write_bytes(duplicate)
    with pytest.raises(RobustnessArtifactError, match="duplicate"):
        verify_robustness_supplement(
            path,
            expected_sha256=hashlib.sha256(duplicate).hexdigest(),
        )


def test_verifier_requires_canonical_serialization(tmp_path: Path) -> None:
    payload = _valid_payload()
    canonical = canonical_robustness_json_bytes(payload)
    noncanonical = canonical.replace(
        b'"evidence_role":"untouched_confirmatory"', b'"evidence_role": "untouched_confirmatory"'
    )
    path = tmp_path / "robustness.json"
    path.write_bytes(noncanonical)

    with pytest.raises(RobustnessArtifactError, match="canonical"):
        verify_robustness_supplement(
            path,
            expected_sha256=hashlib.sha256(noncanonical).hexdigest(),
        )


def test_payload_builder_derives_closed_counts_intervals_and_forecast_measures() -> None:
    base_means = torch.tensor(
        [[20.0, 0.8, 0.7, 0.0], [10.0, 0.9, 0.8, 0.0]],
        dtype=torch.float64,
    ).numpy()
    base_bootstrap = torch.tensor(
        [[-10.0, 0.1, 0.1], [-12.0, 0.05, 0.05], [-8.0, 0.15, 0.15]],
        dtype=torch.float64,
    ).numpy()
    sensitivity_paired = torch.zeros((24, 3, 3), dtype=torch.float64).numpy()
    sensitivity_did = torch.zeros((24, 3, 3), dtype=torch.float64).numpy()
    sensitivity_did[:, 0, :] = torch.tensor([10.0, -0.1, -0.1], dtype=torch.float64).numpy()
    demand = torch.arange(24, dtype=torch.float64).reshape(1, 1, 24, 1, 1)
    demand = demand.expand(10, 32, 24, 7, 5).clone()
    forecast = forecast_error_sufficient_statistics(demand).numpy()
    provenance = _valid_payload()["provenance"]

    payload = build_robustness_supplement_payload(
        provenance=provenance,  # type: ignore[arg-type]
        base_controller_means=base_means,
        base_bootstrap=base_bootstrap,
        sensitivity_paired_bootstrap=sensitivity_paired,
        sensitivity_difference_bootstrap=sensitivity_did,
        forecast_statistics=forecast,
        maximum_power_balance_violation_rate=0.0,
    )

    assert payload["protocol"]["matched_scenario_count"] == 960
    assert payload["base_evaluation"]["paired_inference"]["fairness"] == {
        "mean_delta": 0.1,
        "ci95_lower": 0.05,
        "ci95_upper": 0.15,
        "result_classification": "favorable",
    }
    assert payload["forecast_evaluation"]["methods"]["persistence_1h"]["pooled"]["locality_sector"][
        "mae"
    ] == pytest.approx(1.0)
    assert len(payload["sensitivity"]["variants"]) == 24
    validate_robustness_supplement_payload(payload)


def test_cuda_reproducer_declares_one_final_full_numeric_host_transfer() -> None:
    source = inspect.getsource(run_full_robustness_evaluation)

    assert source.count(".cpu().numpy()") == 1

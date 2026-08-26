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
    ROBUSTNESS_ACTOR_SHA256,
    ROBUSTNESS_ACTOR_SIZE_BYTES,
    ROBUSTNESS_BATCH_SIZE,
    ROBUSTNESS_IMPLEMENTATION_SOURCE_BINDINGS,
    ROBUSTNESS_MODEL_MANIFEST_SHA256,
    ROBUSTNESS_MODEL_MANIFEST_SIZE_BYTES,
    ROBUSTNESS_SCENARIO_GENERATOR_SHA256,
    ROBUSTNESS_SCENARIO_GENERATOR_SIZE_BYTES,
    ROBUSTNESS_SEEDS,
    ROBUSTNESS_SOURCE_REVISION,
    ROBUSTNESS_SUPPLEMENT_SHA256,
    ROBUSTNESS_TRAINING_PROVENANCE_SHA256,
    ROBUSTNESS_TRAINING_PROVENANCE_SIZE_BYTES,
    RobustnessArtifactError,
    build_robustness_supplement_payload,
    build_sensitivity_variants,
    canonical_robustness_json_bytes,
    forecast_error_sufficient_statistics,
    forecast_input_validation_flags,
    run_full_robustness_evaluation,
    validate_robustness_supplement_payload,
    verify_robustness_implementation_sources,
    verify_robustness_supplement,
)
from tools.training_provenance import load_training_provenance

ROOT = Path(__file__).parents[1]


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
            "source_revision": ROBUSTNESS_SOURCE_REVISION,
            "model_manifest": {
                "path": "data/model/manifest.json",
                "sha256": ROBUSTNESS_MODEL_MANIFEST_SHA256,
                "size_bytes": ROBUSTNESS_MODEL_MANIFEST_SIZE_BYTES,
            },
            "actor": {
                "path": "data/model/ppo-policy.npz",
                "sha256": ROBUSTNESS_ACTOR_SHA256,
                "size_bytes": ROBUSTNESS_ACTOR_SIZE_BYTES,
            },
            "scenario_generator": {
                "path": "src/gridguard/scenarios.py",
                "sha256": ROBUSTNESS_SCENARIO_GENERATOR_SHA256,
                "size_bytes": ROBUSTNESS_SCENARIO_GENERATOR_SIZE_BYTES,
            },
            "training_provenance": {
                "path": "data/model/training-provenance.yaml",
                "sha256": ROBUSTNESS_TRAINING_PROVENANCE_SHA256,
                "size_bytes": ROBUSTNESS_TRAINING_PROVENANCE_SIZE_BYTES,
            },
            "implementation_sources": [
                {"path": path, "sha256": sha256, "size_bytes": size_bytes}
                for path, sha256, size_bytes in ROBUSTNESS_IMPLEMENTATION_SOURCE_BINDINGS
            ],
            "seed_separation": {
                "claim_scope": (
                    "Untouched means disjoint from every training, validation, and "
                    "model-selection seed documented in data/model/training-provenance.yaml; "
                    "it does not claim knowledge of undocumented external activity."
                ),
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
        "forecast_metric_definitions": {
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
                        "episode x target_hour x 7 localities x 5 sectors over absolute "
                        "cell residuals and squared cell residuals"
                    )
                },
                "system_total": {
                    "residual": (
                        "sum the 35 signed locality-sector residuals before absolute value "
                        "or squaring"
                    ),
                    "pooling": "episode x target_hour after system aggregation",
                },
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


def test_pinned_provenance_bindings_match_current_source_bytes() -> None:
    bindings = (
        (
            ROOT / "data" / "model" / "manifest.json",
            ROBUSTNESS_MODEL_MANIFEST_SHA256,
            ROBUSTNESS_MODEL_MANIFEST_SIZE_BYTES,
        ),
        (
            ROOT / "data" / "model" / "ppo-policy.npz",
            ROBUSTNESS_ACTOR_SHA256,
            ROBUSTNESS_ACTOR_SIZE_BYTES,
        ),
        (
            ROOT / "src" / "gridguard" / "scenarios.py",
            ROBUSTNESS_SCENARIO_GENERATOR_SHA256,
            ROBUSTNESS_SCENARIO_GENERATOR_SIZE_BYTES,
        ),
        (
            ROOT / "data" / "model" / "training-provenance.yaml",
            ROBUSTNESS_TRAINING_PROVENANCE_SHA256,
            ROBUSTNESS_TRAINING_PROVENANCE_SIZE_BYTES,
        ),
    )

    for path, expected_sha256, expected_size in bindings:
        assert path.stat().st_size == expected_size
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256

    assert tuple(path for path, _, _ in ROBUSTNESS_IMPLEMENTATION_SOURCE_BINDINGS) == (
        "src/gridguard/controllers.py",
        "src/gridguard/domain.py",
        "src/gridguard/evaluation.py",
        "src/gridguard/metrics.py",
        "src/gridguard/policy.py",
        "src/gridguard/robustness.py",
        "src/gridguard/scenarios.py",
        "src/gridguard/simulation.py",
        "src/gridguard/statistics.py",
    )
    for relative_path, expected_sha256, expected_size in ROBUSTNESS_IMPLEMENTATION_SOURCE_BINDINGS:
        path = ROOT / relative_path
        assert path.stat().st_size == expected_size
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256


def test_current_checkout_implementation_sources_are_rehashed() -> None:
    verified = verify_robustness_implementation_sources(ROOT)

    assert tuple(binding["path"] for binding in verified) == tuple(
        path for path, _, _ in ROBUSTNESS_IMPLEMENTATION_SOURCE_BINDINGS
    )


def test_robustness_seeds_are_disjoint_from_every_documented_training_seed_set() -> None:
    training = load_training_provenance(
        ROOT / "data" / "model" / "training-provenance.yaml",
        model_manifest_path=ROOT / "data" / "model" / "manifest.json",
    )
    cloning = training["behavior_cloning_initialization"]
    ppo = training["ppo"]
    selection = training["validation_and_selection"]
    disclosure = training["evaluation_seed_disclosure"]
    candidate_seeds = selection["candidate_training_seeds"]
    selected_training_seed = ppo["training_seed"]
    teacher_start_offset = cloning["teacher_episode_seed_start"] - selected_training_seed
    teacher_end_offset = cloning["teacher_episode_seed_end_inclusive"] - selected_training_seed
    exploration_seed_offset = cloning["exploration_prng_seed"] - selected_training_seed
    documented_sets = {
        "behavior_cloning_teacher_episode_seed_range": {
            episode_seed
            for candidate_seed in candidate_seeds
            for episode_seed in range(
                candidate_seed + teacher_start_offset,
                candidate_seed + teacher_end_offset + 1,
            )
        },
        "behavior_cloning_exploration_prng_seeds": {
            candidate_seed + exploration_seed_offset for candidate_seed in candidate_seeds
        },
        "behavior_cloning_shuffle_prng_seeds": set(candidate_seeds),
        "candidate_training_seeds": set(candidate_seeds),
        "ppo_rollout_episode_seed_range": {
            rollout_seed
            for candidate_seed in candidate_seeds
            for rollout_seed in range(
                candidate_seed + 1_001,
                candidate_seed + 1_001 + ppo["iterations"],
            )
        },
        "periodic_validation_seeds": set(selection["validation_seeds"]),
        "accepted_evaluation_promotion_seeds": set(disclosure["seeds"]),
    }
    robustness_seeds = set(ROBUSTNESS_SEEDS)

    assert all(robustness_seeds.isdisjoint(seeds) for seeds in documented_sets.values())
    metadata = _valid_payload()["provenance"]["seed_separation"]  # type: ignore[index]
    recorded = metadata["documented_seed_sets"]
    teacher_range = recorded["behavior_cloning_teacher_episode_seed_range"]
    assert (
        set(range(teacher_range["start"], teacher_range["end_inclusive"] + 1))
        == (documented_sets["behavior_cloning_teacher_episode_seed_range"])
    )
    assert (
        set(recorded["behavior_cloning_exploration_prng_seeds"])
        == documented_sets["behavior_cloning_exploration_prng_seeds"]
    )
    assert (
        set(recorded["behavior_cloning_shuffle_prng_seeds"])
        == documented_sets["behavior_cloning_shuffle_prng_seeds"]
    )
    assert set(recorded["candidate_training_seeds"]) == documented_sets["candidate_training_seeds"]
    rollout = recorded["ppo_rollout_episode_seed_range"]
    assert (
        set(range(rollout["start"], rollout["end_inclusive"] + 1))
        == documented_sets["ppo_rollout_episode_seed_range"]
    )
    assert (
        set(recorded["periodic_validation_seeds"]) == documented_sets["periodic_validation_seeds"]
    )
    assert (
        set(recorded["accepted_evaluation_promotion_seeds"])
        == documented_sets["accepted_evaluation_promotion_seeds"]
    )
    validate_robustness_supplement_payload(_valid_payload())


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

    flags = forecast_input_validation_flags(demand)
    statistics = forecast_error_sufficient_statistics(demand)

    torch.testing.assert_close(flags, torch.tensor([0.0, 1.0], dtype=torch.float64))
    assert not bool(torch.isfinite(statistics).all())


def test_forecast_validation_flags_reject_negative_values_without_scalarizing() -> None:
    demand = torch.ones((2, 3, 24, 7, 5))
    demand[0, 0, 0, 0, 0] = -1.0

    flags = forecast_input_validation_flags(demand)

    torch.testing.assert_close(flags, torch.tensor([1.0, 0.0], dtype=torch.float64))
    assert ".item()" not in inspect.getsource(forecast_input_validation_flags)
    assert ".item()" not in inspect.getsource(forecast_error_sufficient_statistics)


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
            lambda payload: payload["provenance"].__setitem__(  # type: ignore[union-attr]
                "source_revision", "b" * 40
            ),
            "source_revision",
        ),
        (
            lambda payload: payload["provenance"]["model_manifest"].__setitem__(  # type: ignore[index]
                "sha256", "e" * 64
            ),
            "model_manifest.sha256",
        ),
        (
            lambda payload: payload["provenance"]["scenario_generator"].__setitem__(  # type: ignore[index]
                "size_bytes", ROBUSTNESS_SCENARIO_GENERATOR_SIZE_BYTES + 1
            ),
            "scenario_generator.size_bytes",
        ),
        (
            lambda payload: payload["provenance"]["training_provenance"].__setitem__(  # type: ignore[index]
                "sha256", "d" * 64
            ),
            "training_provenance.sha256",
        ),
        (
            lambda payload: payload["provenance"]["implementation_sources"][0].__setitem__(  # type: ignore[index]
                "path", "src/gridguard/not-the-controller.py"
            ),
            r"implementation_sources\[0\].path",
        ),
        (
            lambda payload: payload["provenance"]["implementation_sources"][1].__setitem__(  # type: ignore[index]
                "sha256", "f" * 64
            ),
            r"implementation_sources\[1\].sha256",
        ),
        (
            lambda payload: payload["provenance"]["implementation_sources"][2].__setitem__(  # type: ignore[index]
                "size_bytes", 1
            ),
            r"implementation_sources\[2\].size_bytes",
        ),
        (
            lambda payload: payload["protocol"].__setitem__(  # type: ignore[union-attr]
                "matched_scenario_count", 959
            ),
            "matched_scenario_count",
        ),
        (
            lambda payload: payload["forecast_metric_definitions"]["pooled_metrics"][  # type: ignore[index]
                "rmse"
            ].__setitem__("formula", "sqrt(SAE / count)"),
            "forecast_metric_definitions",
        ),
        (
            lambda payload: payload["provenance"]["seed_separation"].__setitem__(  # type: ignore[index]
                "disjoint_from_all_documented_seeds", False
            ),
            "disjoint_from_all_documented_seeds",
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
            lambda payload: payload["base_evaluation"]["paired_inference"][  # type: ignore[index]
                "critical_outage_locality_hours"
            ].__setitem__("ci95_lower", -168.01),
            "feasible bounds",
        ),
        (
            lambda payload: payload["sensitivity"]["variants"][0][  # type: ignore[index]
                "paired_inference"
            ]["fairness"].__setitem__("ci95_upper", 1.01),
            "feasible bounds",
        ),
        (
            lambda payload: payload["sensitivity"]["variants"][0][  # type: ignore[index]
                "difference_in_differences"
            ]["utilization"].__setitem__("ci95_upper", 2.01),
            "feasible bounds",
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
            lambda payload: payload["forecast_evaluation"]["methods"][  # type: ignore[index]
                "trailing_mean_3h"
            ][
                "per_seed"
            ][
                0
            ]["locality_sector"].update({"sum_absolute_error": 23_520.0, "sum_squared_error": 0.0}),
            "error-sum inequality",
        ),
        (
            lambda payload: payload["forecast_evaluation"]["methods"][  # type: ignore[index]
                "trailing_mean_3h"
            ][
                "per_seed"
            ][
                0
            ][
                "system_total"
            ].update({"sum_absolute_error": 23_521.0, "sum_squared_error": 1_000_000.0}),
            "system SAE",
        ),
        (
            lambda payload: payload["forecast_evaluation"]["methods"][  # type: ignore[index]
                "trailing_mean_3h"
            ]["per_seed"][0]["system_total"].update({"sum_squared_error": 1_700_000.0}),
            "system SSE",
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


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload["protocol"].__setitem__("episodes_per_seed", 32.0),
            "protocol.episodes_per_seed must be an integer",
        ),
        (
            lambda payload: payload["protocol"]["seeds"].__setitem__(0, 81_001.0),
            r"protocol.seeds\[0\] must be an integer",
        ),
        (
            lambda payload: payload["provenance"]["seed_separation"][  # type: ignore[index]
                "robustness_seeds"
            ].__setitem__(0, 81_001.0),
            r"seed_separation.robustness_seeds\[0\] must be an integer",
        ),
        (
            lambda payload: payload["protocol"]["bootstrap"].__setitem__("replicates", 10_000.0),
            "protocol.bootstrap.replicates must be an integer",
        ),
        (
            lambda payload: payload["sensitivity"]["variants"][0].__setitem__(  # type: ignore[index]
                "component_index", 0.0
            ),
            "component_index must be an integer",
        ),
        (
            lambda payload: payload["forecast_evaluation"]["target_hours"].__setitem__(0, 3.0),
            r"forecast_evaluation.target_hours\[0\] must be an integer",
        ),
        (
            lambda payload: payload["forecast_evaluation"]["methods"][  # type: ignore[index]
                "trailing_mean_3h"
            ].__setitem__("window_hours", 3.0),
            "window_hours must be an integer",
        ),
        (
            lambda payload: payload["forecast_evaluation"]["methods"][  # type: ignore[index]
                "trailing_mean_3h"
            ]["per_seed"][0].__setitem__("seed", 81_001.0),
            "seed must be an integer",
        ),
        (
            lambda payload: payload["forecast_evaluation"]["methods"][  # type: ignore[index]
                "trailing_mean_3h"
            ]["per_seed"][0]["locality_sector"].__setitem__("count", 23_520.0),
            "count must be a positive integer",
        ),
        (
            lambda payload: payload["forecast_evaluation"]["methods"][  # type: ignore[index]
                "trailing_mean_3h"
            ]["pooled"]["locality_sector"].__setitem__("count", 235_200.0),
            "count must be a positive integer",
        ),
        (
            lambda payload: payload["forecast_evaluation"].__setitem__("trace_count", 320.0),
            "forecast_evaluation.trace_count must be an integer",
        ),
    ],
)
def test_strict_validator_rejects_float_encoded_integer_fields(mutation, message: str) -> None:
    payload = _valid_payload()
    mutation(payload)

    with pytest.raises(RobustnessArtifactError, match=message):
        validate_robustness_supplement_payload(payload)


def test_difference_in_differences_accepts_its_expanded_feasible_extrema() -> None:
    payload = _valid_payload()
    payload["base_evaluation"]["controller_means"]["Fixed Rotation"][  # type: ignore[index]
        "critical_outage_locality_hours"
    ] = 168.0
    payload["base_evaluation"]["controller_means"]["PPO"][  # type: ignore[index]
        "critical_outage_locality_hours"
    ] = 0.0
    payload["base_evaluation"]["paired_inference"][  # type: ignore[index]
        "critical_outage_locality_hours"
    ] = {
        "mean_delta": -168.0,
        "ci95_lower": -168.0,
        "ci95_upper": -168.0,
        "result_classification": "favorable",
    }
    for variant in payload["sensitivity"]["variants"]:  # type: ignore[index,union-attr]
        variant["paired_inference"]["critical_outage_locality_hours"] = {
            "mean_delta": 168.0,
            "ci95_lower": 168.0,
            "ci95_upper": 168.0,
            "result_classification": "adverse",
        }
        variant["difference_in_differences"]["critical_outage_locality_hours"] = {
            "mean_delta": 336.0,
            "ci95_lower": 336.0,
            "ci95_upper": 336.0,
            "result_classification": "adverse",
        }

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


def test_committed_robustness_supplement_matches_its_trust_anchor() -> None:
    path = ROOT / "data" / "evidence" / "robustness-supplement.json"
    actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    assert actual_sha256 == ROBUSTNESS_SUPPLEMENT_SHA256
    verified = verify_robustness_supplement(
        path,
        expected_sha256=ROBUSTNESS_SUPPLEMENT_SHA256,
    )
    assert verified["provenance"]["implementation_sources"] == [
        {"path": source_path, "sha256": sha256, "size_bytes": size_bytes}
        for source_path, sha256, size_bytes in ROBUSTNESS_IMPLEMENTATION_SOURCE_BINDINGS
    ]
    assert verified["provenance"]["seed_separation"]["disjoint_from_all_documented_seeds"] is True


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

    # This locks evidence-array movement, not a claim of zero scalar CUDA
    # synchronization in lower-level boundary validators.
    assert source.count(".cpu().numpy()") == 1

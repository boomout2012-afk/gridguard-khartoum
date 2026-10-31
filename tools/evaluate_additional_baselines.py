"""Post-hoc forecast, deterministic-rule and action-availability comparisons.

Historical studies, models and protocols are read only. The three analyses reuse
their original inputs; none is an untouched confirmatory test. Evaluation uses
the existing scientific backend. ``--verify`` reconstructs compact saved summaries
without running models or installing the application.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

if __package__ in {None, ""}:
    sys.path[:0] = [
        str(Path(__file__).resolve().parents[1]),
        str(Path(__file__).resolve().parents[1] / "src"),
    ]

import numpy as np
import torch

from gridguard.constrained_ppo import critical_action_tiers
from gridguard.domain import Scenario
from gridguard.focused_training import TEST_SEEDS, evaluation_scenarios
from gridguard.forecast_model import evaluate_forecast_model, fit_ridge_forecast
from gridguard.proposal_evidence import EVALUATION_SEEDS
from gridguard.scenarios import generate_scenario_base_draws, scenario_from_base_draws
from tools.evaluate_committee_baseline import (
    METRICS,
    NAMES,
    controllers,
    evaluate_conditions,
    hash_rows,
    load_freeze,
)
from tools.evaluate_focused_ppo import _primary_adjusted_bootstrap, _transfer

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/evidence/additional-baselines.json"
SCHEMA = "gridguard.additional_baselines.1"
BOUNDARY_METRICS = (
    "episode_reward",
    "served_mwh",
    "fairness",
    "utilization",
    "switching_rate",
    "critical_outage_locality_hours",
    "critical_interruption_locality_hours",
    "power_balance_violation_rate",
)
METHODS = ("trailing_mean_3h", "persistence_1h", "ridge_ar3", "hour_of_day_mean")
SHORTAGES = ("mild", "moderate", "severe")
INPUT_ARTIFACTS = (
    "data/evidence/committee-baseline-audit.json",
    "data/evidence/proposal-supplement.json",
    "data/evidence/focused-ppo-training.json",
    "data/evidence/committee-baseline-audit.execution-freeze.json",
)
RECIPE = {
    "post_hoc": True,
    "forecast": {
        "training_seeds": [42, 43, 44],
        "training_batch": 256,
        "test_seeds": list(EVALUATION_SEEDS),
        "test_batch": 32,
        "target_hours": list(range(3, 24)),
        "new_method": "training mean for each hour, locality and sector",
        "training_traces": 768,
        "test_traces": 320,
        "information": "known hour of day; no evaluation targets or history used for fitting",
    },
    "rule": {
        "contrast": "Fairness Rule minus Constrained Optimization",
        "bootstrap_replicates": 10000,
        "bootstrap_seed": 20261024,
        "bootstrap": (
            "paired within-seed base draws, severities coupled; deterministic singleton run"
        ),
        "interval_confidence": 0.975,
        "inference": "descriptive pointwise; no new family-wide or confirmatory claim",
        "service_margin_ratio": 0.002,
        "validation_selection": "not applicable to the deterministic rule",
    },
    "boundary": {
        "test_seeds": list(TEST_SEEDS),
        "batch": 32,
        "supply_formula": "critical demand + alpha * minimum locality noncritical demand",
        "alpha": [0.99, 1.01, 2.0],
        "interpretation": (
            "first discretionary-action threshold, not infinitesimal critical headroom"
        ),
        "controllers": list(NAMES),
        "conditions_per_family": 320,
        "hours_per_family": 7680,
        "trajectories": 8640,
        "inference": "descriptive post-hoc challenge; no tuning, selection or significance claim",
    },
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8"
    )


def _validate_cube(demand: torch.Tensor, seeds: tuple[int, ...]) -> None:
    if not isinstance(seeds, tuple) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seed labels must be a nonempty unique tuple")
    if any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("seed labels must be nonnegative integers")
    if (
        not isinstance(demand, torch.Tensor)
        or demand.ndim != 5
        or demand.shape[0] != len(seeds)
        or demand.shape[1] < 1
        or demand.shape[2:] != (24, 7, 5)
        or not demand.is_floating_point()
    ):
        raise ValueError("demand must be floating [seed, trace, 24, 7, 5]")
    if not bool(torch.isfinite(demand).all()) or bool((demand < 0).any()):
        raise ValueError("demand must be finite and nonnegative")


@dataclass(frozen=True)
class HourlyProfile:
    means_mw: torch.Tensor
    training_seeds: tuple[int, ...]
    training_trace_count: int


def fit_hourly_profile(demand: torch.Tensor, *, training_seeds: tuple[int, ...]) -> HourlyProfile:
    """Fit 24 x 35 means over training traces; preserve the input device."""
    _validate_cube(demand, training_seeds)
    return HourlyProfile(
        demand.to(torch.float64).mean((0, 1)), training_seeds, demand.shape[0] * demand.shape[1]
    )


def predict_hourly_profile(
    model: HourlyProfile, demand: torch.Tensor, *, evaluation_seeds: tuple[int, ...]
) -> torch.Tensor:
    """Predict hours 3..23 using the clock; target values never affect the fit."""
    _validate_cube(demand, evaluation_seeds)
    if set(evaluation_seeds) & set(model.training_seeds):
        raise ValueError("training and evaluation seeds must be disjoint")
    if model.means_mw.device != demand.device:
        raise ValueError("model and evaluation tensors must share a device")
    return model.means_mw[None, None, 3:].expand(demand.shape[0], demand.shape[1], 21, 7, 5)


def boundary_scenario(scenario: Scenario, alpha: float) -> Scenario:
    """Bracket the first Full Service choice without changing demand or controller state."""
    if type(alpha) not in (float, int) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be a positive finite number")
    minimum = scenario.demand[..., 3:].sum(-1).amin(-1)
    if bool((minimum <= 0).any()):
        raise ValueError("a positive discretionary demand is required in every locality")
    supply = scenario.demand[..., :3].sum((-1, -2)) + alpha * minimum
    return Scenario(scenario.demand, supply, scenario.shortage_level, scenario.seed)


def feasible_action_counts(scenario: Scenario) -> torch.Tensor:
    """Use the exact request multiplication/reduction order of the shared action mask."""
    demand = scenario.demand.flatten(0, 1)
    tiers = critical_action_tiers(demand.device)
    critical = torch.arange(5, device=demand.device)[None, None, :] < 3
    requests = critical | (tiers == 2)[..., None]
    totals = (demand[:, None] * requests[None]).sum((-1, -2))
    return (totals <= scenario.supply.flatten()[:, None]).sum(-1).reshape_as(scenario.supply)


def _error_statistics(residual: torch.Tensor) -> torch.Tensor:
    flat = residual.flatten(1)
    counts = torch.full_like(flat.sum(-1), flat.shape[-1])
    return torch.stack((counts, flat.abs().sum(-1), flat.square().sum(-1)), dim=-1)


def _forecast_methods(statistics: np.ndarray) -> dict:
    total = statistics.sum(1)
    errors = np.stack((total[..., 1] / total[..., 0], (total[..., 2] / total[..., 0]) ** 0.5), -1)
    return {
        name: {
            agg: dict(zip(("mae_mw", "rmse_mw"), errors[i, j].tolist(), strict=True))
            for j, agg in enumerate(("locality_sector", "system_total"))
        }
        for i, name in enumerate(METHODS)
    }


def _rule_fields(interval_array: np.ndarray, original_cube: np.ndarray) -> dict:
    intervals = {
        name: dict(zip(("mean", "lower", "upper"), interval_array[:, i].tolist(), strict=True))
        for i, name in enumerate(METRICS)
    }
    means = (original_cube[3] - original_cube[2]).mean((1, 2))
    shortage = {
        name: dict(zip(METRICS, means[i].tolist(), strict=True)) for i, name in enumerate(SHORTAGES)
    }
    safety = (
        "critical_outage_locality_hours",
        "critical_interruption_locality_hours",
        "power_balance_violation_rate",
    )
    gates = {
        "both_methods_critical_and_power_constraints": all(
            original_cube[2:4, ..., METRICS.index(name)].max() == 0 for name in safety
        ),
        "reward_improvement": intervals["episode_reward"]["lower"] > 0,
        "fairness_noninferiority": intervals["fairness"]["lower"] >= -0.002,
        "utilization_noninferiority": intervals["utilization"]["lower"] >= -0.002,
        "severity_reward_guardrail": all(r["episode_reward"] >= 0 for r in shortage.values()),
        "severity_service_guardrails": all(
            r["fairness"] >= -0.002 and r["utilization"] >= -0.002 for r in shortage.values()
        ),
    }
    return {
        "intervals": intervals,
        "shortage_deltas": shortage,
        "gates": {name: bool(value) for name, value in gates.items()},
        "numerical_service_gates_passed": bool(all(gates.values())),
    }


def _boundary_summaries(cube: np.ndarray, counts: np.ndarray, ratios: np.ndarray) -> dict:
    means = cube.mean((2, 3))
    maxima = cube.max((2, 3))
    result = {}
    for fi, alpha in enumerate(RECIPE["boundary"]["alpha"]):
        count, ratio = counts[fi], ratios[fi]
        values, frequencies = np.unique(count, return_counts=True)
        result[str(alpha)] = {
            "supply_to_critical_ratio": {
                "mean": float(ratio.mean()),
                "min": float(ratio.min()),
                "max": float(ratio.max()),
            },
            "feasible_action_count": {
                "min": int(count.min()),
                "mean": float(count.mean()),
                "max": int(count.max()),
                "hours_with_multiple_choices": int((count > 1).sum()),
                "total_hours": int(count.size),
                "distribution": dict(
                    zip(map(str, values.astype(int)), frequencies.tolist(), strict=True)
                ),
            },
            "controller_means": {
                name: dict(zip(BOUNDARY_METRICS, means[i, fi].tolist(), strict=True))
                for i, name in enumerate(NAMES)
            },
            "ppo_procedure_mean": dict(
                zip(BOUNDARY_METRICS, means[4:, fi].mean(0).tolist(), strict=True)
            ),
            "maxima": {
                name: dict(zip(BOUNDARY_METRICS, maxima[i, fi].tolist(), strict=True))
                for i, name in enumerate(NAMES)
            },
        }
    return result


@torch.inference_mode()
def evaluate() -> dict:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    frozen, training_record = load_freeze()
    source_paths = sorted(
        set(frozen["source_sha256"])
        | {
            "src/gridguard/forecast_model.py",
            "src/gridguard/proposal_evidence.py",
            "tools/evaluate_additional_baselines.py",
        }
    )
    source_hashes = {p: sha(ROOT / p) for p in source_paths}
    artifact_hashes = {p: sha(ROOT / p) for p in INPUT_ARTIFACTS}
    started_at = datetime.now(UTC).isoformat()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    training_seeds = (42, 43, 44)
    train = torch.stack(
        [
            scenario_from_base_draws(
                generate_scenario_base_draws(256, s, device="cuda:0"), "moderate"
            ).demand
            for s in training_seeds
        ]
    )
    test = torch.stack(
        [
            scenario_from_base_draws(
                generate_scenario_base_draws(32, s, device="cuda:0"), "moderate"
            ).demand
            for s in EVALUATION_SEEDS
        ]
    )
    ridge = fit_ridge_forecast(train, training_seeds=training_seeds, ridge_penalty=0.001)
    original_forecast = evaluate_forecast_model(ridge, test, evaluation_seeds=EVALUATION_SEEDS)
    hourly = fit_hourly_profile(train, training_seeds=training_seeds)
    prediction = predict_hourly_profile(hourly, test, evaluation_seeds=EVALUATION_SEEDS)
    residual = prediction - original_forecast.targets_mw
    hourly_stats = torch.stack(
        (_error_statistics(residual), _error_statistics(residual.sum((-1, -2)))), dim=1
    )
    forecast_stats = torch.cat((original_forecast.sufficient_statistics, hourly_stats.unsqueeze(0)))

    original = json.loads((ROOT / INPUT_ARTIFACTS[0]).read_text())
    cube = torch.tensor(original["episode_metric_cube"], dtype=torch.float64, device="cuda:0")
    interval_array = _primary_adjusted_bootstrap(cube[3:4] - cube[2])

    _, moderate, _ = evaluation_scenarios(TEST_SEEDS)
    families = tuple(
        tuple(boundary_scenario(s, alpha) for s in moderate)
        for alpha in RECIPE["boundary"]["alpha"]
    )
    boundary = evaluate_conditions(controllers(training_record), families, ("0.99", "1.01", "2.0"))
    counts = torch.stack([torch.stack([feasible_action_counts(s) for s in f]) for f in families])
    ratios = torch.stack(
        [torch.stack([s.supply / s.demand[..., :3].sum((-1, -2)) for s in f]) for f in families]
    )
    action_match = (boundary["actions"] == boundary["actions"][2:3]).all(-1).to(torch.float64)
    agreement_counts = action_match.sum((2, 3, 4))
    indices = [METRICS.index(m) for m in BOUNDARY_METRICS]
    blocks = _transfer(
        {
            "forecast_stats": forecast_stats,
            "hourly_means_mw": hourly.means_mw,
            "training_demand": train.to(torch.float64),
            "evaluation_demand": test.to(torch.float64),
            "rule_intervals": interval_array,
            "boundary_cube": boundary["cube"][..., indices],
            "boundary_inputs": boundary["inputs"],
            "boundary_actions": boundary["actions"],
            "feasible_counts": counts,
            "supply_ratios": ratios,
            "agreement_counts": agreement_counts,
        }
    )
    old_forecast = json.loads((ROOT / INPUT_ARTIFACTS[1]).read_text())["forecast"]
    if not np.array_equal(blocks["forecast_stats"][:3], old_forecast["sufficient_statistics"]):
        raise ValueError("the three original forecast statistics did not reproduce exactly")
    payload = {
        "schema": SCHEMA,
        "started_at_utc": started_at,
        "recipe": RECIPE,
        "source_sha256": source_hashes,
        "input_artifact_sha256": artifact_hashes,
        "models": frozen["models"],
        "units": {
            "served_mwh": "MWh/day",
            "forecast_errors": "MW",
            "fairness": "ratio",
            "utilization": "ratio",
            "switching_rate": "fraction of localities per hour",
            "critical_interruption_locality_hours": "locality-hours/day",
            "episode_reward": "sum of 24 normalized hourly scores",
        },
        "forecast": {
            "method_names": list(METHODS),
            "methods": _forecast_methods(blocks["forecast_stats"]),
            "original_statistics_reproduced_exactly": True,
            "sufficient_statistics": blocks["forecast_stats"].tolist(),
            "hourly_training_means_mw": blocks["hourly_means_mw"].tolist(),
            "training_demand_sha256": hash_rows(
                blocks["training_demand"].reshape(3, 256, -1), "<f8"
            ),
            "evaluation_demand_sha256": hash_rows(
                blocks["evaluation_demand"].reshape(10, 32, -1), "<f8"
            ),
        },
        "rule_vs_optimizer": _rule_fields(
            blocks["rule_intervals"], np.asarray(original["episode_metric_cube"])
        ),
        "boundary": {
            "metric_names": list(BOUNDARY_METRICS),
            "metric_cube": blocks["boundary_cube"].tolist(),
            "feasible_action_counts": blocks["feasible_counts"].astype(int).tolist(),
            "supply_to_critical_ratios": blocks["supply_ratios"].tolist(),
            "optimizer_joint_action_match_counts": blocks["agreement_counts"].astype(int).tolist(),
            "summaries": _boundary_summaries(
                blocks["boundary_cube"], blocks["feasible_counts"], blocks["supply_ratios"]
            ),
            "scenario_sha256": hash_rows(blocks["boundary_inputs"], "<f8"),
            "action_sha256": hash_rows(blocks["boundary_actions"].reshape(9, 3, 10, 32, -1), "<i8"),
        },
    }
    validate_payload(payload)
    return payload


def _same(actual, expected, field: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"{field}: mapping keys differ")
        for key, value in expected.items():
            _same(actual[key], value, f"{field}.{key}")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if type(actual) not in (int, float) or not math.isclose(
            actual, expected, abs_tol=1e-10, rel_tol=1e-12
        ):
            raise ValueError(f"{field}: numeric summary differs")
    elif actual != expected:
        raise ValueError(f"{field}: value differs")


def _array(value, shape: tuple[int, ...], field: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{field}: invalid shape or nonfinite data")
    return result


def _hash_cube(value, shape: tuple[int, ...], field: str) -> None:
    values = np.asarray(value)
    if values.shape != shape or any(
        len(str(v)) != 64 or any(c not in "0123456789abcdef" for c in str(v)) for v in values.flat
    ):
        raise ValueError(f"{field}: invalid hash cube")


def validate_payload(payload: dict) -> None:
    """Verify identities and reconstruct saved summaries without executing the simulation."""
    _same(payload.get("schema"), SCHEMA, "schema")
    _same(payload.get("recipe"), RECIPE, "recipe")
    frozen = json.loads((ROOT / INPUT_ARTIFACTS[3]).read_text())
    source_paths = set(frozen["source_sha256"]) | {
        "src/gridguard/forecast_model.py",
        "src/gridguard/proposal_evidence.py",
        "tools/evaluate_additional_baselines.py",
    }
    _same(payload["source_sha256"], {p: sha(ROOT / p) for p in source_paths}, "source hashes")
    _same(
        payload["input_artifact_sha256"],
        {p: sha(ROOT / p) for p in INPUT_ARTIFACTS},
        "input hashes",
    )
    _same(payload["models"], frozen["models"], "models")
    for model in payload["models"].values():
        if sha(ROOT / model["path"]) != model["sha256"]:
            raise ValueError("selected policy checksum mismatch")
    forecast = payload["forecast"]
    _same(forecast["method_names"], list(METHODS), "forecast names")
    stats = _array(forecast["sufficient_statistics"], (4, 10, 2, 3), "forecast statistics")
    if (stats < 0).any() or not np.array_equal(
        stats[:, :, :, 0], np.broadcast_to([23520, 672], (4, 10, 2))
    ):
        raise ValueError("forecast statistics have invalid counts or negative errors")
    original_forecast = json.loads((ROOT / INPUT_ARTIFACTS[1]).read_text())["forecast"]
    if not np.array_equal(stats[:3], original_forecast["sufficient_statistics"]):
        raise ValueError("original forecast statistics differ")
    _same(forecast["original_statistics_reproduced_exactly"], True, "forecast replay")
    _same(forecast["methods"], _forecast_methods(stats), "forecast summary")
    if (_array(forecast["hourly_training_means_mw"], (24, 7, 5), "hourly means") < 0).any():
        raise ValueError("negative hourly means")
    _hash_cube(forecast["training_demand_sha256"], (3, 256), "forecast training hashes")
    _hash_cube(forecast["evaluation_demand_sha256"], (10, 32), "forecast evaluation hashes")

    original = json.loads((ROOT / INPUT_ARTIFACTS[0]).read_text())
    old_cube = np.asarray(original["episode_metric_cube"])
    rule = payload["rule_vs_optimizer"]
    intervals = _array(
        [[rule["intervals"][m][k] for m in METRICS] for k in ("mean", "lower", "upper")],
        (3, len(METRICS)),
        "rule intervals",
    )
    if (intervals[1] > intervals[0]).any() or (intervals[0] > intervals[2]).any():
        raise ValueError("rule interval ordering is invalid")
    mean = (old_cube[3] - old_cube[2]).mean((0, 1, 2))
    if not np.allclose(mean, intervals[0], rtol=0, atol=1e-10):
        raise ValueError("rule means differ from original matched cases")
    _same(rule, _rule_fields(intervals, old_cube), "rule")

    boundary = payload["boundary"]
    _same(boundary["metric_names"], list(BOUNDARY_METRICS), "boundary metrics")
    cube = _array(boundary["metric_cube"], (9, 3, 10, 32, len(BOUNDARY_METRICS)), "boundary cube")
    counts = _array(boundary["feasible_action_counts"], (3, 10, 32, 24), "feasible counts")
    ratios = _array(boundary["supply_to_critical_ratios"], (3, 10, 32, 24), "supply ratios")
    matches = _array(boundary["optimizer_joint_action_match_counts"], (9, 3), "action matches")
    if (cube < 0).any() or (ratios <= 1).any():
        raise ValueError("boundary values are outside their physical domains")
    if not np.equal(counts, counts.astype(int)).all() or (counts < 1).any() or (counts > 128).any():
        raise ValueError("invalid feasible action counts")
    if (counts[0] != 1).any() or (counts[1:] < 2).any():
        raise ValueError("discretionary boundary did not bracket action availability")
    if (
        not np.equal(matches, matches.astype(int)).all()
        or (matches < 0).any()
        or (matches > 7680).any()
    ):
        raise ValueError("invalid action agreement counts")
    if (matches[1:, 0] != 7680).any() or (matches[2] != 7680).any():
        raise ValueError("singleton or self action agreement differs")
    for metric in ("fairness", "utilization", "switching_rate", "power_balance_violation_rate"):
        if (cube[..., BOUNDARY_METRICS.index(metric)] > 1 + 1e-6).any():
            raise ValueError("a boundary ratio exceeds one")
    if (cube[..., BOUNDARY_METRICS.index("power_balance_violation_rate")] != 0).any():
        raise ValueError("boundary power-balance violations")
    for metric in ("critical_outage_locality_hours", "critical_interruption_locality_hours"):
        if (cube[1:, ..., BOUNDARY_METRICS.index(metric)] != 0).any():
            raise ValueError("protected boundary method interrupted critical service")
    _same(boundary["summaries"], _boundary_summaries(cube, counts, ratios), "boundary summary")
    _hash_cube(boundary["scenario_sha256"], (3, 10, 32), "boundary input hashes")
    _hash_cube(boundary["action_sha256"], (9, 3, 10, 32), "boundary action hashes")


def verify(path: Path = OUTPUT, *, expected_sha256: str | None = None) -> dict:
    expected = expected_sha256 or path.with_suffix(".sha256").read_text().split()[0]
    if sha(path) != expected:
        raise ValueError("additional baseline checksum mismatch")
    validate_payload(json.loads(path.read_text(encoding="utf-8")))
    return {
        "passed": True,
        "sha256": expected,
        "forecast_test_traces": 320,
        "boundary_trajectories": 8640,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify", action="store_true", help="check saved evidence without simulation"
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify(args.output)))
    else:
        _write(args.output, evaluate())
        args.output.with_suffix(".sha256").write_text(sha(args.output) + "\n", encoding="ascii")
        print(json.dumps(verify(args.output)))


if __name__ == "__main__":
    main()

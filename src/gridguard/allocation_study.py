"""Protocol and evidence checks for the additive constrained-allocation study."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from gridguard.domain import SimulationResult
from gridguard.duration_metrics import DurationMetrics
from gridguard.metrics import EpisodeMetrics
from gridguard.statistics import stratified_paired_bootstrap

VALIDATION_SEEDS = tuple(range(102101, 102106))
TEST_SEEDS = tuple(range(102201, 102211))
REWARD_TOLERANCES = (0.0, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05)
SHORTAGES = ("mild", "moderate", "severe")
BATCH_SIZE = 32
METRIC_NAMES = tuple(field.name for field in fields(EpisodeMetrics)) + tuple(
    field.name for field in fields(DurationMetrics) if not field.name.startswith("locality_")
)
LOCALITY_METRIC_NAMES = tuple(
    field.name for field in fields(DurationMetrics) if field.name.startswith("locality_")
)
CONTROLLER_NAMES = (
    "Fixed Rotation",
    "PPO",
    "Optimization",
    "Constrained Optimization",
    "Fairness Rule",
    "Feasible PPO",
    "PPO Hybrid",
)
VALIDATION_NAMES = ("Constrained Optimization",) + tuple(
    f"{method} epsilon={epsilon:g}"
    for method in ("Fairness Rule", "PPO Hybrid")
    for epsilon in REWARD_TOLERANCES
)
PAIR_INDICES = ((6, 3), (6, 4), (6, 5), (3, 1), (4, 3), (5, 3), (6, 1))
PROTOCOL = {
    "date": "2026-10-23",
    "validation_seeds": list(VALIDATION_SEEDS),
    "test_seeds": list(TEST_SEEDS),
    "batch_size": BATCH_SIZE,
    "shortages": list(SHORTAGES),
    "reward_tolerances": list(REWARD_TOLERANCES),
    "bootstrap_replicates": 10000,
    "bootstrap_seed": 20261023,
    "selection": "maximize validation episode reward; zero critical interruption and violations; "
    "fairness >= greedy; utilization >= greedy - 0.002; smallest epsilon ties; fallback epsilon 0",
    "reward_improvement_ci_lower_threshold": 0.01,
    "service_noninferiority_margin_ratio": 0.002,
    "claim": "hybrid reward improvement with service guardrails versus both constrained "
    "Optimization and selected Fairness Rule; pointwise 95% intervals",
    "learning_scope": "existing frozen PPO preferences; no new training or learned value model",
}
SOURCE_PATHS = (
    "src/gridguard/domain.py",
    "src/gridguard/controllers.py",
    "src/gridguard/simulation.py",
    "src/gridguard/scenarios.py",
    "src/gridguard/metrics.py",
    "src/gridguard/statistics.py",
    "src/gridguard/optimization.py",
    "src/gridguard/duration_metrics.py",
    "src/gridguard/policy.py",
    "src/gridguard/constrained.py",
    "src/gridguard/hybrid.py",
    "src/gridguard/allocation_study.py",
    "tools/evaluate_allocation_study.py",
    "docs/ALLOCATION_STUDY_PROTOCOL.md",
    "data/model/manifest.json",
    "data/model/ppo-policy.npz",
    "data/model/training-provenance.yaml",
)


def canonical_bytes(payload: dict) -> bytes:
    return (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def source_hashes(root: Path) -> dict[str, str]:
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCE_PATHS}


@torch.inference_mode()
def final_service_ratios(result: SimulationResult) -> torch.Tensor:
    """Final served/demand ratios by locality; exactly zero demand counts as served.

    This separate final-time diagnostic does not redefine canonical Jain fairness.
    No positive demand, including values below the canonical epsilon, is discarded.
    """
    if not isinstance(result, SimulationResult):
        raise TypeError("final service ratios require a SimulationResult")
    result.validate_trace_invariants()
    demand = result.scenario.demand.sum(dim=(1, 3))
    served = result.served.sum(dim=(1, 3))
    denominator = torch.where(demand > 0, demand, torch.ones_like(demand))
    return torch.where(demand > 0, served / denominator, torch.ones_like(demand))


@torch.inference_mode()
def select_validation_settings(cube: torch.Tensor) -> torch.Tensor:
    """Select two tolerance indices on CUDA, with identical criteria for both families."""
    shape = (len(VALIDATION_NAMES), 3, len(VALIDATION_SEEDS), BATCH_SIZE, len(METRIC_NAMES))
    if cube.shape != shape or cube.device.type != "cuda":
        raise ValueError(f"validation cube must have shape {shape} on CUDA")
    if not bool(torch.isfinite(cube).all().item()):
        raise ValueError("validation cube must be finite")
    means = cube.to(torch.float64).mean(dim=(1, 2, 3))
    col = {name: index for index, name in enumerate(METRIC_NAMES)}
    critical = cube[..., col["critical_outage_locality_hours"]].amax(dim=(1, 2, 3))
    violations = cube[..., col["power_balance_violation_rate"]].amax(dim=(1, 2, 3))
    eligible = (
        (critical == 0)
        & (violations == 0)
        & (means[:, col["fairness"]] >= means[0, col["fairness"]])
        & (means[:, col["utilization"]] >= means[0, col["utilization"]] - 0.002)
    )
    choices = []
    for start in (1, 1 + len(REWARD_TOLERANCES)):
        scores = means[start : start + len(REWARD_TOLERANCES), col["episode_reward"]]
        valid = eligible[start : start + len(REWARD_TOLERANCES)]
        # torch.argmax returns the first maximum, which is the smallest epsilon.
        choices.append(scores.masked_fill(~valid, -torch.inf).argmax())
    return torch.stack(choices)


def assess_hybrid_comparison(
    intervals: dict, shortage_deltas: dict, *, critical_safe: bool
) -> dict:
    """Apply declared claim thresholds to already-transferred evidence summaries."""
    metrics = ("episode_reward", "fairness", "utilization")
    if not isinstance(shortage_deltas, dict) or set(shortage_deltas) != set(SHORTAGES):
        raise ValueError("claim requires every declared shortage")
    if type(critical_safe) is not bool:
        raise ValueError("critical safety must be an explicit boolean")
    for metric in metrics:
        item = intervals.get(metric, {})
        if not isinstance(item, dict) or set(item) != {"mean_delta", "ci95_lower", "ci95_upper"}:
            raise ValueError("claim requires complete interval fields")
        if any(
            type(value) not in (int, float) or not math.isfinite(value) for value in item.values()
        ):
            raise ValueError("claim interval values must be finite numbers")
        if item["ci95_lower"] > item["ci95_upper"]:
            raise ValueError("claim interval endpoints are reversed")
    for row in shortage_deltas.values():
        if not isinstance(row, dict) or any(metric not in row for metric in metrics):
            raise ValueError("claim requires complete shortage metrics")
        if any(type(row[m]) not in (int, float) or not math.isfinite(row[m]) for m in metrics):
            raise ValueError("claim shortage values must be finite numbers")
    gates = {
        "critical_and_power_constraints": bool(critical_safe),
        "reward_ci_above_practical_threshold": intervals["episode_reward"]["ci95_lower"] > 0.01,
        "fairness_noninferiority": intervals["fairness"]["ci95_lower"] >= -0.002,
        "utilization_noninferiority": intervals["utilization"]["ci95_lower"] >= -0.002,
        "severity_service_guardrails": all(
            row["fairness"] >= -0.002 and row["utilization"] >= -0.002
            for row in shortage_deltas.values()
        ),
        "severity_reward_guardrail": all(
            row["episode_reward"] >= 0 for row in shortage_deltas.values()
        ),
    }
    return {"passes": all(gates.values()), "gates": gates}


def _numeric_array(value: object, shape: tuple[int, ...], label: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError) as error:
        raise ValueError(f"allocation study {label} must be numerical") from error
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"allocation study {label} shape or finite values mismatch")
    return array


def _hash_array(value: object, shape: tuple[int, ...], label: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=object)
    except (ValueError, TypeError) as error:
        raise ValueError(f"allocation study {label} hash shape mismatch") from error
    if array.shape != shape or any(
        type(item) is not str or re.fullmatch(r"[0-9a-f]{64}", item) is None for item in array.flat
    ):
        raise ValueError(f"allocation study {label} hash shape or digest mismatch")
    return array


def _require_keys(value: object, names: tuple[str, ...], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError(f"allocation study {label} labels mismatch")
    return value


def _close(actual: object, expected: object, label: str, *, atol: float = 1e-10) -> None:
    try:
        values = np.asarray(actual, dtype=np.float64)
    except (ValueError, TypeError) as error:
        raise ValueError(f"allocation study {label} must be numerical") from error
    wanted = np.asarray(expected, dtype=np.float64)
    if (
        values.shape != wanted.shape
        or not np.isfinite(values).all()
        or not np.allclose(values, wanted, rtol=1e-10, atol=atol)
    ):
        raise ValueError(f"allocation study {label} does not reconstruct")


def _validate_physical_arrays(payload: dict, cube: np.ndarray, names: tuple[str, ...]) -> None:
    col = {metric: i for i, metric in enumerate(METRIC_NAMES)}
    for metric in METRIC_NAMES:
        values = cube[..., col[metric]]
        if metric != "episode_reward" and (values < 0).any():
            raise ValueError("allocation study physical metrics must be non-negative")
    for metric in (
        "critical_service_ratio",
        "utilization",
        "fairness",
        "switching_rate",
        "oversubscription",
        "power_balance_violation_rate",
    ):
        if (cube[..., col[metric]] > 1 + 1e-6).any():
            raise ValueError("allocation study metric ratio outside [0, 1]")
    if (
        (cube[..., col["episode_reward"]] < -9) | (cube[..., col["episode_reward"]] > 24.00001)
    ).any():
        raise ValueError("allocation study reward outside possible episode range")
    demand = cube[..., col["demand_mwh"]]
    available = cube[..., col["available_mwh"]]
    served = cube[..., col["served_mwh"]]
    if (served > demand + 0.003).any() or (available > demand + 0.003).any():
        raise ValueError("allocation study energy exceeds scenario demand")
    if (served > available + 0.025).any():
        raise ValueError("allocation study energy exceeds available supply")
    _close(served + cube[..., col["curtailed_mwh"]], demand, "energy identity", atol=0.003)
    for ci in range(1, len(names)):
        if not np.array_equal(demand[ci], demand[0]) or not np.array_equal(
            available[ci], available[0]
        ):
            raise ValueError("allocation study controller input energy is not matched")
    shape = cube.shape[:-1]
    if payload.get("locality_duration_names") != list(LOCALITY_METRIC_NAMES):
        raise ValueError("allocation study locality duration labels mismatch")
    durations = _numeric_array(payload.get("locality_duration_cube"), (*shape, 7, 5), "duration")
    if ((durations < 0) | (durations > 24) | (durations != np.floor(durations))).any():
        raise ValueError("allocation study locality durations must be whole hours from 0 to 24")
    for di, prefix in ((0, "blackout"), (1, "critical_interruption")):
        hours = durations[..., di]
        _close(hours.sum(axis=-1), cube[..., col[f"{prefix}_locality_hours"]], "duration total")
        _close(
            np.ptp(hours, axis=-1),
            cube[..., col[f"{prefix}_duration_range_hours"]],
            "duration range",
        )
        gaps = np.abs(hours[..., :, None] - hours[..., None, :]).sum(axis=(-1, -2)) / 42
        _close(
            gaps,
            cube[..., col[f"{prefix}_duration_mean_pairwise_gap_hours"]],
            "duration gap",
            atol=2e-6,
        )
        streak = durations[..., di + 3]
        if ((streak > hours) | ((hours > 0) & (streak < 1))).any():
            raise ValueError("allocation study duration streak inconsistent with duration")
        _close(streak.max(axis=-1), cube[..., col[f"longest_{prefix}_hours"]], "duration streak")
    _close(
        cube[..., col["critical_outage_locality_hours"]],
        cube[..., col["critical_interruption_locality_hours"]],
        "critical duration identity",
    )
    constrained = range(len(names)) if names == VALIDATION_NAMES else (3, 4, 5, 6)
    if any((durations[ci, ..., 2] != 0).any() for ci in constrained):
        raise ValueError("allocation study constrained actions cannot include Off duration")
    service = _numeric_array(
        payload.get("final_locality_service_ratio_cube"), (*shape, 7), "final service ratio"
    )
    if ((service < 0) | (service > 1 + 1e-6)).any():
        raise ValueError("allocation study final service ratio outside [0, 1]")
    minimum = _require_keys(
        payload.get("mean_final_minimum_locality_service_ratio"), names, "minimum service ratio"
    )
    _close(
        [minimum[name] for name in names],
        service.min(axis=-1).mean(axis=(1, 2, 3)),
        "minimum service ratio",
    )
    shortfalls = _numeric_array(
        payload.get("critical_supply_shortfall_hours"), shape[1:], "shortfall"
    )
    if ((shortfalls < 0) | (shortfalls > 24) | (shortfalls != np.floor(shortfalls))).any():
        raise ValueError("allocation study shortfall counts must be whole hours from 0 to 24")


def _validate_claims(payload: dict, cube: np.ndarray, shortages: np.ndarray) -> None:
    col = {metric: i for i, metric in enumerate(METRIC_NAMES)}
    safe = all(
        np.all(cube[ci, ..., col[metric]] == 0)
        for ci in (3, 4, 5, 6)
        for metric in ("critical_outage_locality_hours", "power_balance_violation_rate")
    )
    comparisons = {}
    for ci in (3, 4):
        label = f"PPO Hybrid - {CONTROLLER_NAMES[ci]}"
        deltas = {
            level: {
                metric: float(shortages[6, si, col[metric]] - shortages[ci, si, col[metric]])
                for metric in ("episode_reward", "fairness", "utilization")
            }
            for si, level in enumerate(SHORTAGES)
        }
        comparisons[label] = assess_hybrid_comparison(
            payload["paired_inference"][label], deltas, critical_safe=safe
        )
    expected = {
        "comparisons": comparisons,
        "hybrid_reward_improvement_with_service_guardrails": all(
            row["passes"] for row in comparisons.values()
        ),
    }
    if payload.get("claims") != expected:
        raise ValueError("allocation study claims do not reconstruct")


def load_study_artifact(path: Path, *, expected_sha256: str, phase: str, root: Path) -> dict:
    """Fail closed on changed identity, source, settings, labels, counts or summaries.

    Small portable evidence checks run on the host only after scientific evaluation.
    The caller supplies the independently recorded artifact identity.
    """
    raw = path.read_bytes()
    if (
        type(expected_sha256) is not str
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
        or hashlib.sha256(raw).hexdigest() != expected_sha256
    ):
        raise ValueError("allocation study artifact SHA-256 mismatch")
    payload = json.loads(raw)
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "gridguard.allocation_study.1"
        or payload.get("phase") != phase
    ):
        raise ValueError("allocation study schema or phase mismatch")
    if phase not in {"validation", "test"} or payload.get("protocol") != PROTOCOL:
        raise ValueError("allocation study protocol mismatch")
    if payload.get("source_sha256") != source_hashes(root):
        raise ValueError("allocation study source identities changed")
    names = VALIDATION_NAMES if phase == "validation" else CONTROLLER_NAMES
    seeds = VALIDATION_SEEDS if phase == "validation" else TEST_SEEDS
    if payload.get("controller_names") != list(names) or payload.get("seeds") != list(seeds):
        raise ValueError("allocation study labels or seed split mismatch")
    if payload.get("metric_names") != list(METRIC_NAMES):
        raise ValueError("allocation study metrics mismatch")
    shape = (len(names), 3, len(seeds), BATCH_SIZE)
    cube = _numeric_array(
        payload.get("episode_metric_cube"), (*shape, len(METRIC_NAMES)), "metric cube"
    )
    _validate_physical_arrays(payload, cube, names)
    scenario_hashes = _hash_array(payload.get("scenario_sha256_cube"), shape[1:], "scenario")
    action_hashes = _hash_array(payload.get("action_sha256_cube"), shape, "action")
    means = cube.mean(axis=(1, 2, 3))
    shortages = cube.mean(axis=(2, 3))
    method_means = _require_keys(payload.get("controller_means"), names, "controller mean")
    shortage_means = _require_keys(payload.get("shortage_means"), SHORTAGES, "shortage")
    for row in shortage_means.values():
        _require_keys(row, names, "shortage method")
    for ci, name in enumerate(names):
        row = _require_keys(method_means[name], METRIC_NAMES, "mean metric")
        _close([row[metric] for metric in METRIC_NAMES], means[ci], "mean")
        for si, level in enumerate(SHORTAGES):
            row = _require_keys(shortage_means[level][name], METRIC_NAMES, "shortage metric")
            _close([row[metric] for metric in METRIC_NAMES], shortages[ci, si], "shortage mean")
    settings = _require_keys(
        payload.get("selected_tolerances"), ("Fairness Rule", "PPO Hybrid"), "selected tolerance"
    )
    if any(
        type(value) not in (int, float) or value not in REWARD_TOLERANCES
        for value in settings.values()
    ):
        raise ValueError("allocation study selected tolerances invalid")
    col = {metric: i for i, metric in enumerate(METRIC_NAMES)}
    if phase == "validation":
        valid = (
            (cube[..., col["critical_outage_locality_hours"]].max(axis=(1, 2, 3)) == 0)
            & (cube[..., col["power_balance_violation_rate"]].max(axis=(1, 2, 3)) == 0)
            & (means[:, col["fairness"]] >= means[0, col["fairness"]])
            & (means[:, col["utilization"]] >= means[0, col["utilization"]] - 0.002)
        )
        eligible = _require_keys(payload.get("validation_eligible"), names, "eligibility")
        if any(
            type(eligible[name]) is not bool or eligible[name] != bool(valid[ci])
            for ci, name in enumerate(names)
        ):
            raise ValueError("allocation study eligibility does not reconstruct")
        fallback = _require_keys(
            payload.get("selection_fallback_used"), ("Fairness Rule", "PPO Hybrid"), "fallback"
        )
        for family, start in (("Fairness Rule", 1), ("PPO Hybrid", 8)):
            scores = means[start : start + 7, col["episode_reward"]]
            selected = np.where(valid[start : start + 7], scores, -np.inf).argmax()
            if settings[family] != REWARD_TOLERANCES[selected]:
                raise ValueError("allocation study setting does not match validation rule")
            if type(fallback[family]) is not bool or fallback[family] != (
                not bool(valid[start : start + 7].any())
            ):
                raise ValueError("allocation study fallback selection does not reconstruct")
            if not np.array_equal(cube[start], cube[0]) or not np.array_equal(
                action_hashes[start], action_hashes[0]
            ):
                raise ValueError(
                    "allocation study zero tolerance must exactly match constrained greedy"
                )
    else:
        relative = payload.get("selection_path")
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise ValueError("allocation study selection path must be root-relative")
        selection_path = (root / relative).resolve()
        if not selection_path.is_relative_to(root.resolve()):
            raise ValueError("allocation study selection path escapes root")
        selection = load_study_artifact(
            selection_path,
            expected_sha256=payload.get("selection_sha256"),
            phase="validation",
            root=root,
        )
        if settings != selection["selected_tolerances"]:
            raise ValueError("allocation study selected tolerances differ from frozen selection")
        validation_hashes = np.asarray(selection["scenario_sha256_cube"], dtype=object)
        if set(scenario_hashes.flat) & set(validation_hashes.flat):
            raise ValueError("allocation study final scenarios overlap validation")
        for ci in (4, 6):
            if settings[names[ci]] == 0 and (
                not np.array_equal(cube[ci], cube[3])
                or not np.array_equal(action_hashes[ci], action_hashes[3])
            ):
                raise ValueError(
                    "allocation study zero tolerance must exactly match constrained greedy"
                )
        pair_names = tuple(f"{names[a]} - {names[b]}" for a, b in PAIR_INDICES)
        paired = _require_keys(payload.get("paired_inference"), pair_names, "paired comparison")
        for a, b in PAIR_INDICES:
            row = _require_keys(paired[f"{names[a]} - {names[b]}"], METRIC_NAMES, "paired metric")
            delta = means[a] - means[b]
            for mi, metric in enumerate(METRIC_NAMES):
                item = _require_keys(
                    row[metric], ("mean_delta", "ci95_lower", "ci95_upper"), "interval"
                )
                if any(
                    type(value) not in (int, float) or not math.isfinite(value)
                    for value in item.values()
                ):
                    raise ValueError("allocation study intervals must be finite numbers")
                if not np.isclose(item["mean_delta"], delta[mi], rtol=1e-10, atol=1e-10):
                    raise ValueError("allocation study paired difference does not reconstruct")
                if item["ci95_lower"] > item["ci95_upper"]:
                    raise ValueError("allocation study interval endpoints reversed")
        _validate_claims(payload, cube, shortages)
    return payload


@torch.inference_mode()
def verify_gpu_intervals(payload: dict) -> None:
    """Reconstruct all final-test intervals on CUDA from saved per-condition outcomes.

    Call after load_study_artifact. This independent scientific verification imports
    the retained host evidence to CUDA, uses the locked CuPy cluster bootstrap, and
    transfers the packed interval results once for comparison. It changes no artifact.
    """
    if payload.get("phase") != "test" or payload.get("protocol") != PROTOCOL:
        raise ValueError("GPU interval reconstruction requires the final-test protocol")
    cube = torch.as_tensor(payload["episode_metric_cube"], dtype=torch.float64, device="cuda:0")
    expected_shape = (len(CONTROLLER_NAMES), 3, len(TEST_SEEDS), BATCH_SIZE, len(METRIC_NAMES))
    if cube.shape != expected_shape or not bool(torch.isfinite(cube).all().item()):
        raise ValueError("GPU interval reconstruction metric cube invalid")
    intervals = (
        torch.stack(
            [
                stratified_paired_bootstrap(
                    cube[a] - cube[b],
                    metric_names=METRIC_NAMES,
                    replicates=PROTOCOL["bootstrap_replicates"],
                    seed=PROTOCOL["bootstrap_seed"],
                ).as_tensor()
                for a, b in PAIR_INDICES
            ]
        )
        .cpu()
        .numpy()
    )
    for pi, (a, b) in enumerate(PAIR_INDICES):
        label = f"{CONTROLLER_NAMES[a]} - {CONTROLLER_NAMES[b]}"
        row = payload["paired_inference"][label]
        retained = [
            [row[metric][field] for metric in METRIC_NAMES]
            for field in ("mean_delta", "ci95_lower", "ci95_upper")
        ]
        _close(retained, intervals[pi], "GPU bootstrap interval")

"""Independent post-study replay; run only after all five runs and final evidence freeze.

This script does not train, select checkpoints or modify scientific source files.
It regenerates all final inputs; independently reruns the first and last final-test seeds
for all 12 controllers at every severity; and reconstructs the complete saved
validation summaries for Fixed Rotation and all ten retained model artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from gridguard.allocation_study import METRIC_NAMES, SHORTAGES
from gridguard.constrained import ConstrainedOptimizationController
from gridguard.constrained_ppo import ConstrainedPPOController, load_model
from gridguard.controllers import FixedRotationController
from gridguard.focused_training import (
    CONFIG,
    TEST_SEEDS,
    TRAINING_SEEDS,
    VALIDATION_SEEDS,
    read_json,
    source_hashes,
    write_json,
)
from gridguard.gpu import run_gpu_preflight
from gridguard.scenarios import generate_scenario_base_draws, scenario_from_base_draws
from gridguard.simulation import simulate
from tools.evaluate_focused_ppo import load_training_record

ROOT = Path(__file__).resolve().parents[1]
CONTROLLER_NAMES = (
    "Fixed Rotation",
    "Constrained Optimization",
    *(f"PPO seed {seed}" for seed in TRAINING_SEEDS),
    *(f"PPO600 seed {seed}" for seed in TRAINING_SEEDS),
)
TOLERANCE = 1e-11


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inside_project(relative: str) -> Path:
    target = (ROOT / relative).resolve()
    if target == ROOT or not target.is_relative_to(ROOT):
        raise ValueError("model path escapes this project")
    return target


def _summary_values(summary: dict) -> list[list[float]]:
    rows = [summary["overall"]]
    rows.extend(summary["by_shortage"][level] for level in SHORTAGES)
    rows.append(summary["maxima"])
    return [[row[name] for name in METRIC_NAMES] for row in rows]


@torch.inference_mode()
def _episode_values(result) -> torch.Tensor:
    """Recalculate all 20 outputs directly, without the metric/evaluator functions.

    Preserve native float32 arithmetic and the declared cumulative-reward order.
    This also independently checks tier requests, global proportional dispatch,
    bounds and the complete hourly reward, including Fixed Rotation's C < 1.
    """
    demand, supply = result.scenario.demand, result.scenario.supply
    served, requested, tiers = result.served, result.requested, result.service_tiers
    dtype = demand.dtype
    epsilon = 1e-8
    expected_request = torch.cat(
        (
            demand[..., :3] * (tiers >= 1)[..., None],
            demand[..., 3:] * (tiers == 2)[..., None],
        ),
        dim=-1,
    )
    demand_total = demand.sum(dim=(-1, -2))
    served_total = served.sum(dim=(-1, -2))
    requested_total = requested.sum(dim=(-1, -2))
    scale = torch.minimum(torch.ones_like(supply), supply / requested_total.clamp_min(epsilon))
    expected_served = requested * scale[..., None, None]
    valid = (
        torch.isfinite(served).all()
        & torch.isfinite(requested).all()
        & torch.isfinite(result.reward).all()
        & (served >= 0).all()
        & (served <= demand).all()
        & (served_total <= supply + 1e-3).all()
        & (requested == expected_request).all()
        & (served == expected_served).all()
    )
    if not bool(valid.item()):
        raise ValueError("independent dispatch, power or demand-bound reconstruction failed")
    locality_demand = demand.sum(-1)
    locality_served = served.sum(-1)
    critical_demand = demand[..., :3].sum(dim=(-1, -2))
    critical_served = served[..., :3].sum(dim=(-1, -2))
    locality_critical_demand = demand[..., :3].sum(-1)
    locality_critical_served = served[..., :3].sum(-1)
    utilization = served_total / torch.minimum(supply, demand_total).clamp_min(epsilon)
    oversubscription = torch.relu(requested_total - supply) / demand_total.clamp_min(epsilon)
    switching = torch.cat(
        (torch.zeros_like(supply[:, :1]), (tiers[:, 1:] != tiers[:, :-1]).to(dtype).mean(-1)),
        dim=1,
    )
    cumulative_ratio = locality_served.cumsum(1) / locality_demand.cumsum(1).clamp_min(epsilon)
    fairness = cumulative_ratio.sum(-1).square() / (7 * cumulative_ratio.square().sum(-1) + epsilon)
    # Reward's sequential accumulation is intentionally separate from the
    # cumsum-based public fairness metric, matching its declared native precision.
    cumulative_demand = torch.zeros_like(locality_demand[:, 0])
    cumulative_served = torch.zeros_like(locality_served[:, 0])
    hourly_reward = []
    for hour in range(24):
        cumulative_demand = cumulative_demand + locality_demand[:, hour]
        cumulative_served = cumulative_served + locality_served[:, hour]
        ratio = cumulative_served / cumulative_demand.clamp_min(epsilon)
        jain = ratio.sum(-1).square() / (7 * ratio.square().sum(-1) + epsilon)
        hour_demand = demand[:, hour].sum((-1, -2))
        hour_served = served[:, hour].sum((-1, -2))
        hour_critical = served[:, hour, :, :3].sum((-1, -2)) / demand[:, hour, :, :3].sum(
            (-1, -2)
        ).clamp_min(epsilon)
        hour_utilization = hour_served / torch.minimum(supply[:, hour], hour_demand).clamp_min(
            epsilon
        )
        hour_oversubscription = torch.relu(
            requested[:, hour].sum((-1, -2)) - supply[:, hour]
        ) / hour_demand.clamp_min(epsilon)
        hourly_reward.append(
            (
                5 * hour_critical
                + 2 * hour_utilization
                + 2 * jain
                - 0.25 * switching[:, hour]
                - 3 * hour_oversubscription
            )
            / 9
        )
    reward = torch.stack(hourly_reward, dim=1)
    if not bool(torch.eq(reward, result.reward).all().item()):
        raise ValueError("independently reconstructed hourly reward differs")
    critical_mask = (locality_critical_demand > epsilon) & (
        locality_critical_served / locality_critical_demand.clamp_min(epsilon) < 0.99
    )
    blackout_mask = (locality_demand > epsilon) & (locality_served == 0)
    values = {
        "episode_reward": reward.sum(1),
        "demand_mwh": demand_total.sum(1),
        "available_mwh": supply.sum(1),
        "served_mwh": served_total.sum(1),
        "curtailed_mwh": (demand_total - served_total).sum(1),
        "critical_service_ratio": critical_served.sum(1)
        / critical_demand.sum(1).clamp_min(epsilon),
        "utilization": utilization.mean(1),
        "fairness": fairness.mean(1),
        "switching_rate": switching.mean(1),
        "oversubscription": oversubscription.mean(1),
        "power_balance_violation_rate": (served_total > supply + 1e-3).to(dtype).mean(1),
        "critical_outage_locality_hours": critical_mask.to(dtype).sum((1, 2)),
    }
    for prefix, mask in (("blackout", blackout_mask), ("critical_interruption", critical_mask)):
        duration = mask.to(dtype).sum(1)
        streak = torch.zeros_like(duration)
        longest = torch.zeros_like(duration)
        for hour in range(24):
            streak = torch.where(mask[:, hour], streak + 1, 0)
            longest = torch.maximum(longest, streak)
        values[f"{prefix}_locality_hours"] = duration.sum(1)
        values[f"{prefix}_duration_range_hours"] = duration.amax(1) - duration.amin(1)
        values[f"{prefix}_duration_mean_pairwise_gap_hours"] = (
            duration[:, :, None] - duration[:, None, :]
        ).abs().sum((1, 2)) / 42
        values[f"longest_{prefix}_hours"] = longest.amax(1)
    if set(values) != set(METRIC_NAMES):
        raise ValueError("independent metric names differ from the frozen schema")
    return torch.stack([values[name] for name in METRIC_NAMES], dim=-1).to(torch.float64)


def _final_services(result) -> torch.Tensor:
    demand = result.scenario.demand.sum((1, 3))
    served = result.served.sum((1, 3))
    denominator = torch.where(demand > 0, demand, 1)
    return torch.where(demand > 0, served / denominator, 1)


@torch.inference_mode()
def _replay_validation(controller, validation_scenarios) -> torch.Tensor:
    severity_rows = []
    for scenarios in validation_scenarios:
        seed_rows = []
        for scenario in scenarios:
            guard = scenario.capture_mutation_guard()
            result = simulate(scenario, controller)
            if guard.changed_fields(scenario):
                raise ValueError("controller mutated validation input during replay")
            seed_rows.append(_episode_values(result))
        severity_rows.append(torch.stack(seed_rows))
    cube = torch.stack(severity_rows)
    return torch.cat(
        (
            cube.mean(dim=(0, 1, 2))[None],
            cube.mean(dim=(1, 2)),
            cube.amax(dim=(0, 1, 2))[None],
        ),
        dim=0,
    )


def _transfer_once(tensors: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    packed = torch.cat([value.reshape(-1).to(torch.float64) for value in tensors.values()])
    host = packed.cpu().numpy()
    blocks = {}
    offset = 0
    for name, value in tensors.items():
        blocks[name] = host[offset : offset + value.numel()].reshape(tuple(value.shape))
        offset += value.numel()
    return blocks


@torch.inference_mode()
def audit(args) -> dict:
    # Both exact evidence files must exist and verify before any final-test input
    # is generated. The training loader also checks the pretraining protocol
    # freeze, all five complete runs, validation selections and model bytes.
    training = load_training_record(args.training, args.training_sha256)
    study = read_json(args.study, args.study_sha256)
    required = {
        "schema": "gridguard.focused_ppo_study.1",
        "training_sha256": args.training_sha256,
        "config": CONFIG,
        "controller_names": list(CONTROLLER_NAMES),
        "metric_names": list(METRIC_NAMES),
        "seeds": list(TEST_SEEDS),
        "shortages": list(SHORTAGES),
    }
    for key, expected in required.items():
        if study.get(key) != expected:
            raise ValueError(f"study {key} is inconsistent before replay")
    bound = source_hashes(ROOT)
    if training["source_sha256"] != bound or study["source_sha256"] != bound:
        raise ValueError("bound scientific sources differ before replay")

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    runtime = run_gpu_preflight()
    start = time.perf_counter()

    controllers = [
        FixedRotationController(),
        ConstrainedOptimizationController(),
    ]
    model_entries = []
    for field in ("model", "iteration600_model"):
        for run in training["runs"]:
            entry = run[field]
            model = load_model(_inside_project(entry["path"]), expected_sha256=entry["sha256"])
            controllers.append(ConstrainedPPOController(model))
            model_entries.append(entry)

    validation_draws = [
        generate_scenario_base_draws(32, seed, device="cuda:0") for seed in VALIDATION_SEEDS
    ]
    validation_scenarios = tuple(
        tuple(scenario_from_base_draws(draw, level) for draw in validation_draws)
        for level in SHORTAGES
    )
    validation_actual = []
    validation_expected = []
    validation_labels = []
    validation_choices = [(0, training["reference_validation"])]
    validation_choices.extend(
        (2 + index, run["selected"]["summary"]) for index, run in enumerate(training["runs"])
    )
    validation_choices.extend(
        (7 + index, next(c["summary"] for c in run["checkpoints"] if c["iteration"] == 600))
        for index, run in enumerate(training["runs"])
    )
    for controller_index, expected in validation_choices:
        label = CONTROLLER_NAMES[controller_index]
        validation_actual.append(
            _replay_validation(controllers[controller_index], validation_scenarios)
        )
        validation_expected.append(_summary_values(expected))
        validation_labels.append(label)
        print(f"Replayed full validation: {label}", flush=True)
    actual_validation = torch.stack(validation_actual)
    expected_validation = torch.tensor(validation_expected, dtype=torch.float64, device="cuda:0")
    validation_errors = (actual_validation - expected_validation).abs().amax(dim=(1, 2))

    # Regenerate all 320 independent draws and their three coupled severities.
    # Controller trajectories are independently replayed for first and last final seeds.
    test_draws = [generate_scenario_base_draws(32, seed, device="cuda:0") for seed in TEST_SEEDS]
    all_inputs, all_shortfalls = [], []
    subset_metrics = [[] for _ in controllers]
    subset_actions = [[] for _ in controllers]
    subset_services = [[] for _ in controllers]
    for level in SHORTAGES:
        input_rows, shortfall_rows = [], []
        for seed_index, draw in enumerate(test_draws):
            scenario = scenario_from_base_draws(draw, level)
            input_rows.append(torch.cat((scenario.demand.flatten(1), scenario.supply), dim=1))
            shortfall_rows.append(
                (scenario.demand[..., :3].sum(dim=(-1, -2)) > scenario.supply).sum(1)
            )
            if seed_index not in (0, len(TEST_SEEDS) - 1):
                continue
            guard = scenario.capture_mutation_guard()
            for index, controller in enumerate(controllers):
                result = simulate(scenario, controller)
                if guard.changed_fields(scenario):
                    raise ValueError("controller mutated final matched input during replay")
                subset_metrics[index].append(_episode_values(result))
                subset_actions[index].append(result.service_tiers)
                subset_services[index].append(_final_services(result).to(torch.float64))
            print(f"Replayed all 12 methods: {level} seed {TEST_SEEDS[seed_index]}", flush=True)
        all_inputs.append(torch.stack(input_rows))
        all_shortfalls.append(torch.stack(shortfall_rows))

    metrics = torch.stack([torch.stack(rows) for rows in subset_metrics])
    services = torch.stack([torch.stack(rows) for rows in subset_services])
    expected_metrics = torch.tensor(
        study["episode_metric_cube"], dtype=torch.float64, device="cuda:0"
    )[:, :, [0, -1]].flatten(1, 2)
    expected_services = torch.tensor(
        study["final_locality_service_ratio_cube"], dtype=torch.float64, device="cuda:0"
    )[:, :, [0, -1]].flatten(1, 2)
    if metrics.shape != expected_metrics.shape or services.shape != expected_services.shape:
        raise ValueError("stored final subset dimensions do not match replay")
    blocks = _transfer_once(
        {
            "validation_errors": validation_errors,
            "validation_summaries": actual_validation,
            "metric_errors": (metrics - expected_metrics).abs().amax(dim=(1, 2, 3)),
            "service_errors": (services - expected_services).abs().amax(dim=(1, 2, 3)),
            "inputs": torch.stack(all_inputs),
            "shortfalls": torch.stack(all_shortfalls),
            "actions": torch.stack([torch.stack(rows) for rows in subset_actions]),
        }
    )
    for key in ("validation_errors", "metric_errors", "service_errors"):
        if not np.isfinite(blocks[key]).all() or bool((blocks[key] > TOLERANCE).any()):
            raise ValueError(f"{key} exceeds exact-replay tolerance: {blocks[key].tolist()}")

    input_hashes = [
        [
            [hashlib.sha256(row.astype("<f8").tobytes()).hexdigest() for row in seed]
            for seed in severity
        ]
        for severity in blocks["inputs"]
    ]
    if input_hashes != study["scenario_sha256_cube"]:
        raise ValueError("regenerated final scenario input hashes differ")
    if blocks["shortfalls"].astype(int).tolist() != study["critical_supply_shortfall_hours"]:
        raise ValueError("regenerated critical-supply shortfall counts differ")
    actual_action_hashes = [
        [
            [hashlib.sha256(row.astype("<i8").tobytes()).hexdigest() for row in severity]
            for severity in method
        ]
        for method in blocks["actions"]
    ]
    expected_action_hashes = [
        [severity[index] for severity in method for index in (0, -1)]
        for method in study["action_sha256_cube"]
    ]
    if actual_action_hashes != expected_action_hashes:
        raise ValueError("replayed final action hashes differ")

    # Prevent time-of-check/time-of-use changes from being accepted after replay.
    if (
        source_hashes(ROOT) != bound
        or _sha256(args.training) != args.training_sha256
        or _sha256(args.study) != args.study_sha256
    ):
        raise ValueError("scientific sources or frozen evidence changed during replay")
    for entry in model_entries:
        if _sha256(_inside_project(entry["path"])) != entry["sha256"]:
            raise ValueError("model artifact changed during replay")
    return {
        "schema": "gridguard.focused_ppo_replay_validation.1",
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "script": Path(__file__).resolve().relative_to(ROOT).as_posix(),
        "script_sha256": _sha256(Path(__file__)),
        "training_sha256": args.training_sha256,
        "study_sha256": args.study_sha256,
        "source_sha256": bound,
        "gpu": runtime.device_name,
        "elapsed_seconds": time.perf_counter() - start,
        "numerical_tolerance": TOLERANCE,
        "passed": True,
        "validation": {
            "conditions_per_method": 480,
            "methods": validation_labels,
            "maximum_absolute_errors": dict(
                zip(validation_labels, blocks["validation_errors"].tolist(), strict=True)
            ),
            "replayed_summaries_row_order": ["overall", *SHORTAGES, "maxima"],
            "replayed_summaries_metric_order": list(METRIC_NAMES),
            "replayed_summaries": blocks["validation_summaries"].tolist(),
        },
        "final_replay": {
            "all_input_hashes_checked": 960,
            "critical_shortfall_conditions_checked": 960,
            "trajectory_seeds": [TEST_SEEDS[0], TEST_SEEDS[-1]],
            "conditions_per_method": 192,
            "controller_trajectories_replayed": 2304,
            "methods": list(CONTROLLER_NAMES),
            "action_hashes_checked": 2304,
            "maximum_metric_absolute_errors": dict(
                zip(CONTROLLER_NAMES, blocks["metric_errors"].tolist(), strict=True)
            ),
            "maximum_final_service_absolute_errors": dict(
                zip(CONTROLLER_NAMES, blocks["service_errors"].tolist(), strict=True)
            ),
        },
        "limits": [
            "The trajectory replay covers 192 of 960 final conditions per method.",
            "All 960 input hashes and critical-supply shortfall counts are regenerated.",
            "Metric formulas are independently reimplemented here; the simulator and "
            "controller implementations are shared frozen primitives.",
            "Bootstrap interval reconstruction is performed by the separate evidence verifier.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training", type=Path, default=ROOT / "data/evidence/focused-ppo-training.json"
    )
    parser.add_argument("--training-sha256", required=True)
    parser.add_argument("--study", type=Path, default=ROOT / "data/evidence/focused-ppo-study.json")
    parser.add_argument("--study-sha256", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/focused-ppo-validation.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite replay evidence: {args.output}")
    payload = audit(args)
    digest = write_json(args.output, payload)
    print(f"Replay passed: {args.output}\nSHA-256: {digest}", flush=True)


if __name__ == "__main__":
    main()

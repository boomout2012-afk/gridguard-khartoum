"""Final matched evaluation of five frozen constrained PPO training runs."""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from gridguard.allocation_study import METRIC_NAMES, SHORTAGES, final_service_ratios
from gridguard.constrained import ConstrainedOptimizationController, FairnessController
from gridguard.constrained_ppo import ConstrainedPPOController, load_model
from gridguard.duration_metrics import compute_duration_metrics
from gridguard.gpu import run_gpu_preflight
from gridguard.metrics import compute_episode_metrics
from gridguard.ppo_experiment import (
    CONFIG,
    TEST_SEEDS,
    TRAINING_SEEDS,
    evaluation_scenarios,
    read_json,
    source_hashes,
    write_json,
)
from gridguard.ppo_study import (
    acceptance_decision,
    conditional_paired_bootstrap,
    hierarchical_paired_bootstrap,
    select_checkpoint,
)
from gridguard.ppo_training import PPOTrainingConfig
from gridguard.simulation import simulate

ROOT = Path(__file__).resolve().parents[1]
CONTROLLER_NAMES = (
    "Fairness Rule",
    "Constrained Optimization",
    *(f"Constrained PPO seed {seed}" for seed in TRAINING_SEEDS),
    *(f"Behavior Cloning seed {seed}" for seed in TRAINING_SEEDS),
)
PROCEDURE_NAMES = ("Constrained PPO", "Behavior Cloning")
PAIR_NAMES = ("Constrained PPO - Fairness Rule", "Constrained PPO - Behavior Cloning")
SCHEMA = "gridguard.constrained_ppo_study.1"
INTERVAL_NAMES = ("mean", "lower", "upper")
SAFETY_METRICS = (
    "critical_interruption_locality_hours",
    "critical_outage_locality_hours",
    "power_balance_violation_rate",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _path_inside(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("artifact paths must be relative strings")
    path = (root / relative).resolve()
    if path == root.resolve() or not path.is_relative_to(root.resolve()):
        raise ValueError("artifact path escapes the project")
    return path


def _hash_text(value: object) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("expected a lowercase SHA-256 digest")


def load_training_record(path: Path, expected_sha256: str, *, root: Path = ROOT) -> dict:
    """Fail closed before creating any final-test scenario or loading an actor."""
    _hash_text(expected_sha256)
    record = read_json(path, expected_sha256)
    if record.get("schema") != "gridguard.constrained_ppo_training.1":
        raise ValueError("training record schema mismatch")
    if record.get("config") != CONFIG or record.get("ppo_config") != asdict(PPOTrainingConfig()):
        raise ValueError("training configuration does not match the frozen protocol")
    if record.get("source_sha256") != source_hashes(root):
        raise ValueError("training source hashes do not match current sources")
    freeze_reference = record.get("protocol_freeze")
    if not isinstance(freeze_reference, dict) or set(freeze_reference) != {"path", "sha256"}:
        raise ValueError("training record requires the pretraining protocol freeze")
    _hash_text(freeze_reference["sha256"])
    freeze = read_json(_path_inside(root, freeze_reference["path"]), freeze_reference["sha256"])
    if freeze.get("schema") != "gridguard.constrained_ppo_freeze.1":
        raise ValueError("pretraining freeze schema mismatch")
    for key in ("config", "ppo_config", "source_sha256"):
        if freeze.get(key) != record[key]:
            raise ValueError("pretraining freeze does not match the training contract")
    try:
        frozen_at = datetime.fromisoformat(freeze["frozen_at_utc"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("pretraining freeze requires a UTC timestamp") from error
    if frozen_at.utcoffset() is None or frozen_at.utcoffset().total_seconds() != 0:
        raise ValueError("pretraining freeze timestamp must use UTC")
    runs = record.get("runs")
    if not isinstance(runs, list) or [run.get("seed") for run in runs] != list(TRAINING_SEEDS):
        raise ValueError("training record must contain all five declared runs in order")
    expected_iterations = list(range(0, CONFIG["iterations"] + 1, CONFIG["checkpoint_interval"]))
    model_paths = set()
    for run in runs:
        if run.get("transitions") != CONFIG["iterations"] * CONFIG["episodes_per_iteration"] * 24:
            raise ValueError("training run does not contain the declared transition budget")
        seconds = run.get("seconds")
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("training runtime must be a finite non-negative number")
        checkpoints = run.get("checkpoints")
        if (
            not isinstance(checkpoints, list)
            or [c.get("iteration") for c in checkpoints] != expected_iterations
        ):
            raise ValueError("training checkpoint sequence is incomplete")
        selected = select_checkpoint(checkpoints, record["reference_validation"])
        if run.get("selected") != selected:
            raise ValueError("training checkpoint selection does not reconstruct")
        for key in ("model", "behavior_cloning_model"):
            entry = run.get(key)
            if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
                raise ValueError("model provenance requires exactly path and SHA-256")
            _hash_text(entry["sha256"])
            model_path = _path_inside(root, entry["path"])
            if _sha256(model_path) != entry["sha256"]:
                raise ValueError("frozen model SHA-256 mismatch")
            if model_path in model_paths:
                raise ValueError("independent runs must have separate model artifacts")
            model_paths.add(model_path)
    eligible = [run for run in runs if run["selected"]["eligible"]]
    deployment = min(
        eligible or runs,
        key=lambda run: (-run["selected"]["summary"]["overall"]["episode_reward"], run["seed"]),
    )["seed"]
    if record.get("deployment_seed") != deployment:
        raise ValueError("deployment seed does not match validation-only selection")
    return record


def _controller_models(record: dict):
    controllers = [
        FairnessController(CONFIG["fairness_reward_tolerance"]),
        ConstrainedOptimizationController(),
    ]
    for field in ("model", "behavior_cloning_model"):
        for run in record["runs"]:
            entry = run[field]
            model = load_model(_path_inside(ROOT, entry["path"]), expected_sha256=entry["sha256"])
            controllers.append(ConstrainedPPOController(model))
    return controllers


def _model_provenance(record: dict) -> dict:
    return {
        CONTROLLER_NAMES[start + index]: run[field]
        for start, field in ((2, "model"), (7, "behavior_cloning_model"))
        for index, run in enumerate(record["runs"])
    }


@torch.inference_mode()
def _scientific_summary(cube: torch.Tensor, services: torch.Tensor) -> dict[str, torch.Tensor]:
    deltas = (cube[2:7] - cube[0], cube[2:7] - cube[7:12])
    arguments = {"n_resamples": CONFIG["bootstrap_replicates"], "seed": CONFIG["bootstrap_seed"]}
    procedures = torch.stack((cube[2:7].mean(0), cube[7:12].mean(0)))
    return {
        "means": cube.mean(dim=(1, 2, 3)),
        "shortage_means": cube.mean(dim=(2, 3)),
        "maxima": cube.amax(dim=(1, 2, 3)),
        "procedure_means": procedures.mean(dim=(1, 2, 3)),
        "procedure_shortage_means": procedures.mean(dim=(2, 3)),
        "minimum_service_means": services.amin(dim=-1).mean(dim=(1, 2, 3)),
        "hierarchical": torch.stack(
            [hierarchical_paired_bootstrap(d, **arguments) for d in deltas]
        ),
        "conditional": torch.stack([conditional_paired_bootstrap(d, **arguments) for d in deltas]),
        "shortage_deltas": torch.stack([d.mean(dim=(0, 2, 3)) for d in deltas]),
    }


def _transfer(tensors: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """One packed scientific GPU-to-host transfer; only artifact assembly follows."""
    packed = torch.cat([tensor.reshape(-1).to(torch.float64) for tensor in tensors.values()])
    host = packed.cpu().numpy()
    offset = 0
    arrays = {}
    for name, tensor in tensors.items():
        arrays[name] = host[offset : offset + tensor.numel()].reshape(tuple(tensor.shape))
        offset += tensor.numel()
    return arrays


def _metric_rows(values: np.ndarray, labels=CONTROLLER_NAMES) -> dict:
    return {
        name: dict(zip(METRIC_NAMES, values[index].tolist(), strict=True))
        for index, name in enumerate(labels)
    }


def _interval(values: np.ndarray) -> dict:
    return {
        metric: dict(zip(INTERVAL_NAMES, values[:, index].tolist(), strict=True))
        for index, metric in enumerate(METRIC_NAMES)
    }


def _summary_fields(blocks: dict[str, np.ndarray], training: dict) -> dict:
    hierarchy = {
        name: _interval(blocks["hierarchical"][index]) for index, name in enumerate(PAIR_NAMES)
    }
    conditional = {
        f"Constrained PPO seed {seed} - {other}": _interval(blocks["conditional"][pair, run])
        for pair in range(2)
        for run, seed in enumerate(TRAINING_SEEDS)
        for other in (("Fairness Rule" if pair == 0 else f"Behavior Cloning seed {seed}"),)
    }
    shortage_deltas = {
        name: {
            shortage: dict(
                zip(METRIC_NAMES, blocks["shortage_deltas"][pair, si].tolist(), strict=True)
            )
            for si, shortage in enumerate(SHORTAGES)
        }
        for pair, name in enumerate(PAIR_NAMES)
    }
    safety_cols = [METRIC_NAMES.index(metric) for metric in SAFETY_METRICS]
    all_eligible = all(run["selected"]["eligible"] for run in training["runs"])
    claims = {}
    for pair, name in enumerate(PAIR_NAMES):
        # Primary safety covers every PPO actor and both deterministic comparators.
        # Attribution additionally covers all paired behavior-cloning actors.
        relevant = blocks["maxima"][:7] if pair == 0 else blocks["maxima"]
        claims[name] = acceptance_decision(
            hierarchy[name],
            shortage_deltas[name],
            all_safe=bool((relevant[:, safety_cols] == 0).all()),
            all_eligible=all_eligible,
        )
    return {
        "controller_means": _metric_rows(blocks["means"]),
        "shortage_means": {
            level: _metric_rows(blocks["shortage_means"][:, si])
            for si, level in enumerate(SHORTAGES)
        },
        "procedure_means": _metric_rows(blocks["procedure_means"], PROCEDURE_NAMES),
        "procedure_shortage_means": {
            level: _metric_rows(blocks["procedure_shortage_means"][:, si], PROCEDURE_NAMES)
            for si, level in enumerate(SHORTAGES)
        },
        "mean_final_minimum_locality_service_ratio": dict(
            zip(CONTROLLER_NAMES, blocks["minimum_service_means"].tolist(), strict=True)
        ),
        "hierarchical_inference": hierarchy,
        "conditional_inference": conditional,
        "shortage_deltas": shortage_deltas,
        "claims": claims,
    }


@torch.inference_mode()
def evaluate(training_path: Path, training_sha256: str) -> dict:
    training = load_training_record(training_path, training_sha256)
    bound_sources = source_hashes(ROOT)
    runtime = run_gpu_preflight()
    print(f"CUDA preflight passed: {runtime.device_name}", flush=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    start = time.perf_counter()
    controllers = _controller_models(training)
    # This is the first point at which final-test inputs may be created.
    scenarios = evaluation_scenarios(TEST_SEEDS)
    controller_rows = [[] for _ in controllers]
    action_rows = [[] for _ in controllers]
    service_rows = [[] for _ in controllers]
    input_rows, shortfall_rows = [], []
    for si, severity_scenarios in enumerate(scenarios):
        rows = [[] for _ in controllers]
        actions = [[] for _ in controllers]
        services = [[] for _ in controllers]
        inputs, shortfalls = [], []
        for zi, scenario in enumerate(severity_scenarios):
            guard = scenario.capture_mutation_guard()
            inputs.append(torch.cat((scenario.demand.flatten(1), scenario.supply), dim=1))
            shortfalls.append((scenario.demand[..., :3].sum((-1, -2)) > scenario.supply).sum(1))
            for index, controller in enumerate(controllers):
                result = simulate(scenario, controller)
                if guard.changed_fields(scenario):
                    raise ValueError("controller mutated a matched final scenario")
                metrics = compute_episode_metrics(result).as_dict()
                metrics.update(compute_duration_metrics(result).as_dict())
                rows[index].append(torch.stack([metrics[name] for name in METRIC_NAMES], dim=-1))
                actions[index].append(result.service_tiers)
                services[index].append(final_service_ratios(result))
            print(f"Final test: {SHORTAGES[si]} seed {TEST_SEEDS[zi]} complete", flush=True)
        for index in range(len(controllers)):
            controller_rows[index].append(torch.stack(rows[index]))
            action_rows[index].append(torch.stack(actions[index]))
            service_rows[index].append(torch.stack(services[index]))
        input_rows.append(torch.stack(inputs))
        shortfall_rows.append(torch.stack(shortfalls))
    cube = torch.stack([torch.stack(rows) for rows in controller_rows]).to(torch.float64)
    service_cube = torch.stack([torch.stack(rows) for rows in service_rows]).to(torch.float64)
    tensors = {
        "cube": cube,
        "services": service_cube,
        "actions": torch.stack([torch.stack(rows) for rows in action_rows]),
        "inputs": torch.stack(input_rows),
        "shortfalls": torch.stack(shortfall_rows),
        **_scientific_summary(cube, service_cube),
    }
    blocks = _transfer(tensors)
    if source_hashes(ROOT) != bound_sources or _sha256(training_path) != training_sha256:
        raise ValueError("frozen scientific sources or training record changed during evaluation")
    # Recheck all model bytes after use; no artifact may change during evaluation.
    load_training_record(training_path, training_sha256)
    payload = {
        "schema": SCHEMA,
        "config": CONFIG,
        "ppo_config": asdict(PPOTrainingConfig()),
        "source_sha256": bound_sources,
        "training_path": training_path.resolve().relative_to(ROOT).as_posix(),
        "training_sha256": training_sha256,
        "model_provenance": _model_provenance(training),
        "deployment_seed": training["deployment_seed"],
        "controller_names": list(CONTROLLER_NAMES),
        "metric_names": list(METRIC_NAMES),
        "seeds": list(TEST_SEEDS),
        "shortages": list(SHORTAGES),
        "matched_scenario_count": 960,
        "controller_trajectory_count": 11520,
        "runtime": asdict(runtime),
        "elapsed_seconds": time.perf_counter() - start,
        "episode_metric_cube": blocks["cube"].tolist(),
        "final_locality_service_ratio_cube": blocks["services"].tolist(),
        "critical_supply_shortfall_hours": blocks["shortfalls"].astype(int).tolist(),
        "scenario_sha256_cube": [
            [
                [hashlib.sha256(row.astype("<f8").tobytes()).hexdigest() for row in seed]
                for seed in severity
            ]
            for severity in blocks["inputs"]
        ],
        "action_sha256_cube": [
            [
                [
                    [hashlib.sha256(row.astype("<i8").tobytes()).hexdigest() for row in seed]
                    for seed in severity
                ]
                for severity in method
            ]
            for method in blocks["actions"]
        ],
        **_summary_fields(blocks, training),
        "limitations": [
            "Synthetic seven-locality, 24-hour decision support; "
            "no real electrical-network validation.",
            "Five training runs give limited information about training variability.",
            "Intervals are pointwise, unadjusted percentile intervals.",
            "Primary inference concerns the entire imitation-plus-PPO training procedure.",
            "PPO attribution additionally requires improvement over "
            "paired behavior-cloning actors.",
            "Iteration-zero selections provide no evidence of improvement caused by PPO updates.",
            "The validation-selected deployment actor is secondary "
            "to the five-run procedure result.",
            "This comparison cannot establish superiority over a full-horizon optimum.",
        ],
    }
    return payload


def _compare_nested(actual: object, expected: object, name: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError(f"{name} fields do not match reconstructed evidence")
        for key, value in expected.items():
            _compare_nested(actual[key], value, f"{name}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"{name} list does not match reconstructed evidence")
        for index, (a, b) in enumerate(zip(actual, expected, strict=True)):
            _compare_nested(a, b, f"{name}[{index}]")
    elif type(expected) in (int, float):
        if (
            type(actual) not in (int, float)
            or not math.isfinite(actual)
            or not math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-11)
        ):
            raise ValueError(f"{name} numerical value does not reconstruct")
    elif actual != expected or type(actual) is not type(expected):
        raise ValueError(f"{name} does not match reconstructed evidence")


def _numeric_tensor(payload: dict, key: str, shape: tuple[int, ...]) -> torch.Tensor:
    try:
        tensor = torch.tensor(payload[key], device="cuda:0", dtype=torch.float64)
    except (TypeError, ValueError, KeyError) as error:
        raise ValueError(f"{key} must be a numeric cube") from error
    if tensor.shape != shape or not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{key} shape or finite-value contract failed")
    return tensor


def _hash_cube(value: object, shape: tuple[int, ...]) -> None:
    if not shape:
        _hash_text(value)
        return
    if not isinstance(value, list) or len(value) != shape[0]:
        raise ValueError("matching-hash cube has incorrect dimensions")
    for child in value:
        _hash_cube(child, shape[1:])


def _validate_physics(cube: torch.Tensor, services: torch.Tensor, shortfalls: torch.Tensor) -> None:
    """Check units, physical bounds and identities on the evidence device."""
    col = {metric: index for index, metric in enumerate(METRIC_NAMES)}
    if not bool((cube[..., 1:] >= 0).all().item()):
        raise ValueError("physical metrics must be non-negative")
    ratios = [
        col[name]
        for name in (
            "critical_service_ratio",
            "utilization",
            "fairness",
            "switching_rate",
            "oversubscription",
            "power_balance_violation_rate",
        )
    ]
    if not bool((cube[..., ratios] <= 1 + 1e-6).all().item()):
        raise ValueError("metric ratios must remain in [0,1]")
    reward = cube[..., col["episode_reward"]]
    if not bool(((reward >= -9) & (reward <= 24.00001)).all().item()):
        raise ValueError("daily reward is outside its physical bounds")
    demand = cube[..., col["demand_mwh"]]
    available = cube[..., col["available_mwh"]]
    served = cube[..., col["served_mwh"]]
    curtailed = cube[..., col["curtailed_mwh"]]
    if not bool(((served <= demand + 0.003) & (available <= demand + 0.003)).all().item()):
        raise ValueError("energy exceeds scenario demand")
    if not bool((served <= available + 0.025).all().item()):
        raise ValueError("served energy exceeds available supply")
    if not bool(torch.isclose(served + curtailed, demand, rtol=1e-10, atol=0.003).all().item()):
        raise ValueError("energy conservation identity failed")
    if not bool(((demand == demand[0]) & (available == available[0])).all().item()):
        raise ValueError("controller scenario energy is not matched")
    for name in (
        "critical_outage_locality_hours",
        "blackout_locality_hours",
        "critical_interruption_locality_hours",
    ):
        hours = cube[..., col[name]]
        if not bool(((hours <= 168) & (hours == hours.floor())).all().item()):
            raise ValueError("locality-hour totals must be whole hours in [0,168]")
    if not bool(
        (
            cube[..., col["critical_outage_locality_hours"]]
            == cube[..., col["critical_interruption_locality_hours"]]
        )
        .all()
        .item()
    ):
        raise ValueError("critical-duration definitions do not agree")
    for name in METRIC_NAMES[14:]:
        if not bool((cube[..., col[name]] <= 24).all().item()):
            raise ValueError("duration ranges, gaps and streaks must not exceed 24 hours")
    if not bool(((services >= 0) & (services <= 1 + 1e-6)).all().item()):
        raise ValueError("final service ratios exceed their physical bounds")
    if not bool(
        ((shortfalls >= 0) & (shortfalls <= 24) & (shortfalls == shortfalls.floor())).all().item()
    ):
        raise ValueError("critical shortfall counts must be integer hours in [0,24]")
    # In the default feasible regime every constrained action serves all critical demand.
    # The public pooled critical ratio is NOT substituted for the hourly mean in general.
    if bool((shortfalls == 0).all().item()):
        if not bool(
            torch.isclose(
                cube[..., col["critical_service_ratio"]], torch.ones_like(reward), rtol=0, atol=1e-6
            )
            .all()
            .item()
        ):
            raise ValueError("critical-feasible dispatch did not protect all critical demand")
        expected_reward = (
            24
            * (
                5
                + 2 * cube[..., col["utilization"]]
                + 2 * cube[..., col["fairness"]]
                - 0.25 * cube[..., col["switching_rate"]]
                - 3 * cube[..., col["oversubscription"]]
            )
            / 9
        )
        if not bool(torch.isclose(reward, expected_reward, rtol=0, atol=1e-5).all().item()):
            raise ValueError(
                "canonical reward does not reconstruct in the critical-feasible regime"
            )


@torch.inference_mode()
def verify(
    path: Path,
    *,
    training_sha256: str,
    expected_sha256: str,
) -> dict:
    """Reconstruct every reported mean, interval and decision on CUDA from stored evidence."""
    _hash_text(expected_sha256)
    payload = read_json(path, expected_sha256)
    if payload.get("schema") != SCHEMA:
        raise ValueError("study schema mismatch")
    required = {
        "config": CONFIG,
        "ppo_config": asdict(PPOTrainingConfig()),
        "source_sha256": source_hashes(ROOT),
        "training_sha256": training_sha256,
        "controller_names": list(CONTROLLER_NAMES),
        "metric_names": list(METRIC_NAMES),
        "seeds": list(TEST_SEEDS),
        "shortages": list(SHORTAGES),
        "matched_scenario_count": 960,
        "controller_trajectory_count": 11520,
    }
    for key, value in required.items():
        if payload.get(key) != value:
            raise ValueError(f"study {key} mismatch")
    training = load_training_record(_path_inside(ROOT, payload["training_path"]), training_sha256)
    if payload.get("model_provenance") != _model_provenance(training):
        raise ValueError("study model provenance mismatch")
    if payload.get("deployment_seed") != training["deployment_seed"]:
        raise ValueError("study deployment actor changed after validation")
    _hash_cube(payload.get("scenario_sha256_cube"), (3, 10, 32))
    _hash_cube(payload.get("action_sha256_cube"), (12, 3, 10, 32))
    run_gpu_preflight()
    cube = _numeric_tensor(payload, "episode_metric_cube", (12, 3, 10, 32, len(METRIC_NAMES)))
    services = _numeric_tensor(payload, "final_locality_service_ratio_cube", (12, 3, 10, 32, 7))
    shortfalls = _numeric_tensor(payload, "critical_supply_shortfall_hours", (3, 10, 32))
    _validate_physics(cube, services, shortfalls)
    summary = _summary_fields(_transfer(_scientific_summary(cube, services)), training)
    for key, expected in summary.items():
        _compare_nested(payload.get(key), expected, key)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training", type=Path, default=ROOT / "data/evidence/constrained-ppo-training.json"
    )
    parser.add_argument("--training-sha256", required=True)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/evidence/constrained-ppo-study.json"
    )
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--sha256", help="explicit SHA-256 of the existing study when verifying")
    args = parser.parse_args()
    if args.verify:
        if args.sha256 is None:
            parser.error("--verify requires an independently recorded --sha256")
        payload = verify(
            args.output, training_sha256=args.training_sha256, expected_sha256=args.sha256
        )
    else:
        if args.output.exists():
            raise FileExistsError("refusing to overwrite an existing constrained PPO study")
        payload = evaluate(args.training, args.training_sha256)
        digest = write_json(args.output, payload)
        payload = verify(args.output, training_sha256=args.training_sha256, expected_sha256=digest)
        print(f"Saved final evidence SHA-256: {digest}", flush=True)
    print(f"Verified {args.output}", flush=True)
    print(f"Primary decisions: {payload['claims']}", flush=True)


if __name__ == "__main__":
    main()

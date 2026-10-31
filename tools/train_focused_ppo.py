"""Train five new long-budget PPO runs against Fixed Rotation; keep final tests unused."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from gridguard.constrained import FairnessController
from gridguard.constrained_ppo import ConstrainedActorCritic, ConstrainedPPOController, save_model
from gridguard.controllers import FixedRotationController
from gridguard.focused_training import (
    CONFIG,
    TRAINING_SEEDS,
    VALIDATION_SEEDS,
    evaluate_metrics,
    evaluation_scenarios,
    read_json,
    scenarios_for_seed,
    source_hashes,
    summarize,
    training_batches,
    write_json,
)
from gridguard.gpu import run_gpu_preflight
from gridguard.ppo_training import (
    PPOTrainingConfig,
    behavior_clone,
    collect_rollout,
    collect_teacher_dataset,
    make_optimizer,
    ppo_update,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "data/model/focused-ppo-study"
RECORD = ROOT / "data/evidence/focused-ppo-training.json"
MONITOR_DIR = ROOT / "tmp/focused-ppo-monitor"


def configure_numerics() -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)


def state_copy(model) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def monitor_row(handle, record: dict) -> None:
    """Flush portable telemetry for an external TensorBoard bridge."""
    handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    handle.flush()


def scalar_statistics(stats: dict[str, torch.Tensor]) -> dict[str, float]:
    """Transfer the small monitoring vector once, outside scientific hot paths."""
    keys = tuple(stats)
    values = torch.stack([stats[key] for key in keys]).detach().cpu().tolist()
    return dict(zip(keys, values, strict=True))


def pilot() -> None:
    """Tiny runtime-only check on a separate training seed, with no held-out data."""
    runtime = run_gpu_preflight()
    print(f"Pilot CUDA: {runtime.device_name}", flush=True)
    torch.manual_seed(991337)
    generator = torch.Generator(device="cuda:0").manual_seed(991337)
    model = ConstrainedActorCritic(device="cuda:0")
    config = PPOTrainingConfig()
    teacher = FairnessController(0.005)
    start = time.perf_counter()
    dataset = collect_teacher_dataset(scenarios_for_seed(9_913_370, (8, 8, 8)), teacher=teacher)
    behavior_clone(model, dataset, epochs=1, generator=generator)
    optimizer = make_optimizer(model, config)
    for iteration in range(3):
        scenarios = scenarios_for_seed(9_913_371 + iteration, (85, 85, 86))
        rollout = collect_rollout(model, scenarios, generator=generator)
        stats = ppo_update(model, optimizer, rollout, config, generator=generator)
    torch.cuda.synchronize()
    values = {key: value.item() for key, value in stats.items()}
    if not all(torch.isfinite(parameter).all().item() for parameter in model.parameters()):
        raise ValueError("pilot produced nonfinite model weights")
    print({"pilot_seconds": time.perf_counter() - start, "last_update": values}, flush=True)


def train_run(run_index: int, scenarios, reference: dict, bound: dict) -> dict:
    seed = TRAINING_SEEDS[run_index]
    MONITOR_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("behavior-cloning", "iteration-600", "selected"):
        if (MODEL_DIR / f"seed-{seed}-{suffix}.npz").exists():
            raise FileExistsError(
                "incomplete focused run already has model files; use a new output directory"
            )
    with (MONITOR_DIR / f"seed-{seed}.jsonl").open("x", encoding="utf-8") as monitor:
        return _train_run(run_index, scenarios, reference, bound, monitor)


def _train_run(run_index: int, scenarios, reference: dict, bound: dict, monitor) -> dict:
    from gridguard.focused_training import select_checkpoint

    seed = TRAINING_SEEDS[run_index]
    torch.manual_seed(seed)
    policy_generator = torch.Generator(device="cuda:0").manual_seed(seed + 1)
    update_generator = torch.Generator(device="cuda:0").manual_seed(seed + 2)
    bc_generator = torch.Generator(device="cuda:0").manual_seed(seed + 3)
    model = ConstrainedActorCritic(device="cuda:0")
    config = PPOTrainingConfig()
    teacher = FairnessController(CONFIG["fairness_reward_tolerance"])
    start = time.perf_counter()
    print(f"Seed {seed}: collecting Fairness Rule demonstrations", flush=True)
    dataset = collect_teacher_dataset(
        scenarios_for_seed(4_500_000 + run_index * 10_000, (455, 455, 455)), teacher=teacher
    )
    bc_stats = behavior_clone(
        model,
        dataset,
        epochs=CONFIG["bc_epochs"],
        minibatch_size=CONFIG["bc_minibatch_size"],
        learning_rate=CONFIG["bc_learning_rate"],
        value_coefficient=CONFIG["bc_value_coefficient"],
        generator=bc_generator,
    )
    bc_path = MODEL_DIR / f"seed-{seed}-behavior-cloning.npz"
    bc_hash = save_model(model, bc_path)
    bc_statistics = scalar_statistics(bc_stats)
    del dataset
    checkpoints = [
        {
            "iteration": 0,
            "summary": summarize(evaluate_metrics(ConstrainedPPOController(model), scenarios)),
        }
    ]
    selected = select_checkpoint(checkpoints, reference)
    selected_state = state_copy(model)
    telemetry = [
        {
            "seed": seed,
            "iteration": 0,
            "transitions": 0,
            "stage": "behavior_cloning",
            "elapsed_seconds": time.perf_counter() - start,
            "bc_statistics": bc_statistics,
            "validation": checkpoints[0]["summary"],
            "selected_iteration": selected["iteration"],
            "selected_eligible": selected["eligible"],
        }
    ]
    monitor_row(monitor, telemetry[-1])
    budget_reference = None
    optimizer = make_optimizer(model, config)
    for iteration in range(CONFIG["iterations"]):
        scenario_seed, batches = training_batches(run_index, iteration)
        rollout = collect_rollout(
            model, scenarios_for_seed(scenario_seed, batches), generator=policy_generator
        )
        stats = ppo_update(
            model,
            optimizer,
            rollout,
            config,
            generator=update_generator,
            progress=iteration / (CONFIG["iterations"] - 1),
        )
        completed = iteration + 1
        validation = None
        if completed % CONFIG["checkpoint_interval"] == 0:
            summary = summarize(evaluate_metrics(ConstrainedPPOController(model), scenarios))
            validation = summary
            checkpoints.append({"iteration": completed, "summary": summary})
            chosen = select_checkpoint(checkpoints, reference)
            if chosen["iteration"] != selected["iteration"]:
                selected_state = state_copy(model)
            selected = chosen
            reward = summary["overall"]["episode_reward"]
            print(
                f"Seed {seed}: {completed}/3000, validation reward {reward:.6f}, "
                f"selected {selected['iteration']}, eligible={selected['eligible']}",
                flush=True,
            )
        if completed == CONFIG["budget_reference_iteration"]:
            reference_path = MODEL_DIR / f"seed-{seed}-iteration-600.npz"
            reference_hash = save_model(model, reference_path)
            budget_reference = {
                "path": reference_path.relative_to(ROOT).as_posix(),
                "sha256": reference_hash,
            }
        if completed % CONFIG["telemetry_interval"] == 0:
            row = {
                "seed": seed,
                "iteration": completed,
                "transitions": completed * CONFIG["episodes_per_iteration"] * 24,
                "stage": "ppo",
                "elapsed_seconds": time.perf_counter() - start,
                **scalar_statistics(stats),
            }
            if validation is not None:
                row.update(
                    validation=validation,
                    selected_iteration=selected["iteration"],
                    selected_eligible=selected["eligible"],
                )
            telemetry.append(row)
            monitor_row(monitor, row)
    if budget_reference is None:
        raise ValueError("required iteration600 model was not retained")
    model.load_state_dict(selected_state)
    selected_path = MODEL_DIR / f"seed-{seed}-selected.npz"
    selected_hash = save_model(model, selected_path)
    if source_hashes(ROOT) != bound:
        raise ValueError("scientific sources changed during training")
    torch.cuda.synchronize()
    return {
        "seed": seed,
        "selected": selected,
        "checkpoints": checkpoints,
        "model": {"path": selected_path.relative_to(ROOT).as_posix(), "sha256": selected_hash},
        "behavior_cloning_model": {
            "path": bc_path.relative_to(ROOT).as_posix(),
            "sha256": bc_hash,
        },
        "iteration600_model": budget_reference,
        "seconds": time.perf_counter() - start,
        "transitions": CONFIG["iterations"] * CONFIG["episodes_per_iteration"] * 24,
        "bc_states": 455 * 3 * 24,
        "bc_statistics": bc_statistics,
        "telemetry": telemetry,
    }


def train() -> None:
    from gridguard.focused_training import select_checkpoint

    if RECORD.exists():
        raise FileExistsError(f"Frozen training record already exists: {RECORD}")
    runtime = run_gpu_preflight()
    print(f"CUDA preflight passed: {runtime.device_name}", flush=True)
    bound = source_hashes(ROOT)
    config = asdict(PPOTrainingConfig())
    freeze_path = MODEL_DIR / "protocol-freeze.json"
    if freeze_path.exists():
        frozen = read_json(freeze_path)
        if (
            frozen.get("schema") != "gridguard.focused_ppo_freeze.1"
            or frozen["source_sha256"] != bound
            or frozen["config"] != CONFIG
            or frozen["ppo_config"] != config
        ):
            raise ValueError("pretraining protocol freeze no longer matches sources/settings")
        freeze_hash = freeze_path.with_suffix(".sha256").read_text().split()[0]
    else:
        freeze_hash = write_json(
            freeze_path,
            {
                "schema": "gridguard.focused_ppo_freeze.1",
                "frozen_at_utc": datetime.now(UTC).isoformat(),
                "source_sha256": bound,
                "config": CONFIG,
                "ppo_config": config,
            },
        )
    scenarios = evaluation_scenarios(VALIDATION_SEEDS)
    reference = summarize(evaluate_metrics(FixedRotationController(), scenarios))
    MONITOR_DIR.mkdir(parents=True, exist_ok=True)
    reference_monitor = MONITOR_DIR / "fixed-rotation.json"
    reference_contents = json.dumps({"summary": reference}, sort_keys=True, allow_nan=False)
    if reference_monitor.exists():
        if json.loads(reference_monitor.read_text(encoding="utf-8")) != {"summary": reference}:
            raise ValueError("monitor Fixed Rotation baseline differs from this validation run")
    else:
        with reference_monitor.open("x", encoding="utf-8") as monitor:
            monitor.write(reference_contents + "\n")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    runs = []
    for index, seed in enumerate(TRAINING_SEEDS):
        run_path = MODEL_DIR / f"seed-{seed}-record.json"
        if run_path.exists():
            record = read_json(run_path)
            if (
                record["source_sha256"] != bound
                or record["config"] != CONFIG
                or record["ppo_config"] != config
            ):
                raise ValueError("resume record does not match frozen source/configuration")
            run = record["run"]
            if (
                run["seed"] != seed
                or run["transitions"]
                != CONFIG["iterations"] * CONFIG["episodes_per_iteration"] * 24
                or run["bc_states"] != 455 * 3 * 24
                or [row["iteration"] for row in run["checkpoints"]]
                != list(range(0, CONFIG["iterations"] + 1, CONFIG["checkpoint_interval"]))
            ):
                raise ValueError("resume run identity or training budget differs")
            from gridguard.constrained_ppo import load_model

            for key in ("model", "behavior_cloning_model", "iteration600_model"):
                model_path = (ROOT / run[key]["path"]).resolve()
                suffix = {
                    "model": "selected",
                    "behavior_cloning_model": "behavior-cloning",
                    "iteration600_model": "iteration-600",
                }[key]
                expected_path = (MODEL_DIR / f"seed-{seed}-{suffix}.npz").resolve()
                if model_path != expected_path:
                    raise ValueError("resume model path differs from its declared run artifact")
                load_model(model_path, expected_sha256=run[key]["sha256"])
            if run["selected"] != select_checkpoint(run["checkpoints"], reference):
                raise ValueError("resume checkpoint selection differs")
            if [row["iteration"] for row in run["telemetry"]] != list(
                range(0, CONFIG["iterations"] + 1, CONFIG["telemetry_interval"])
            ):
                raise ValueError("resume telemetry sequence is incomplete")
            print(f"Seed {seed}: verified completed run", flush=True)
        else:
            run = train_run(index, scenarios, reference, bound)
            write_json(
                run_path,
                {
                    "source_sha256": bound,
                    "config": CONFIG,
                    "ppo_config": config,
                    "run": run,
                },
            )
        runs.append(run)
    eligible = [run for run in runs if run["selected"]["eligible"]]
    deployment = max(
        eligible or runs,
        key=lambda run: (run["selected"]["summary"]["overall"]["episode_reward"], -run["seed"]),
    )
    if source_hashes(ROOT) != bound:
        raise ValueError("scientific sources changed during training")
    payload = {
        "schema": "gridguard.focused_ppo_training.1",
        "protocol_freeze": {
            "path": freeze_path.relative_to(ROOT).as_posix(),
            "sha256": freeze_hash,
        },
        "config": CONFIG,
        "ppo_config": config,
        "source_sha256": bound,
        "runtime": asdict(runtime),
        "runs": runs,
        "reference_validation": reference,
        "deployment_seed": deployment["seed"],
        "numerics": {
            "dtype": "float32",
            "tf32": False,
            "deterministic_algorithms": True,
            "cublas_workspace_config": ":4096:8",
            "random_streams": {
                "initialization": "run_seed",
                "policy_sampling": "run_seed + 1",
                "ppo_minibatch_shuffle": "run_seed + 2",
                "bc_minibatch_shuffle": "run_seed + 3",
            },
        },
        "model_spec": {
            "features": 46,
            "actions": 128,
            "hidden_sizes": [256, 256],
            "activation": "tanh",
            "initialization": "orthogonal",
            "feature_order": "existing32, cumulative_demand_mwh[7], cumulative_served_mwh[7]",
            "cumulative_normalization": "24 * fixed_locality_peak_mw",
            "inference": "deterministic highest feasible logit, lowest index tie",
        },
    }
    digest = write_json(RECORD, payload)
    print(f"Frozen training record: {RECORD}\nSHA-256: {digest}", flush=True)


def main() -> None:
    global MODEL_DIR, RECORD, MONITOR_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true", help="tiny training-only runtime check")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--record", type=Path, default=RECORD)
    parser.add_argument("--monitor-dir", type=Path, default=MONITOR_DIR)
    args = parser.parse_args()
    MODEL_DIR, RECORD, MONITOR_DIR = (
        args.model_dir.resolve(),
        args.record.resolve(),
        args.monitor_dir.resolve(),
    )
    if any(not path.is_relative_to(ROOT) for path in (MODEL_DIR, RECORD, MONITOR_DIR)):
        parser.error("study outputs must remain inside the project for portable provenance")
    configure_numerics()
    pilot() if args.pilot else train()


if __name__ == "__main__":
    main()

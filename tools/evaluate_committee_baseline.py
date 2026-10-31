"""Freeze, execute and verify the explicitly post-hoc committee baseline audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from gridguard.allocation_study import METRIC_NAMES, SHORTAGES
from gridguard.committee_baseline import (
    DIAGNOSTIC_NAMES,
    REWARD_COMPONENT_NAMES,
    REWARD_RECONSTRUCTION_TOLERANCE,
    CriticalAwareFixedRotationController,
    audit_diagnostics,
    critical_supply_stress,
)
from gridguard.constrained import ConstrainedOptimizationController, FairnessController
from gridguard.constrained_ppo import ConstrainedPPOController, load_model
from gridguard.controllers import FixedRotationController
from gridguard.duration_metrics import compute_duration_metrics
from gridguard.focused_training import (
    SOURCE_PATHS,
    TEST_SEEDS,
    TRAINING_SEEDS,
    evaluation_scenarios,
    read_json,
    write_json,
)
from gridguard.gpu import run_gpu_preflight
from gridguard.metrics import compute_episode_metrics
from gridguard.ppo_study import conditional_paired_bootstrap, hierarchical_paired_bootstrap
from gridguard.simulation import simulate
from tools.evaluate_focused_ppo import _compare_nested, _hash_cube, _transfer, load_training_record

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "gridguard.committee_baseline_audit.1"
OUTPUT = ROOT / "data/evidence/committee-baseline-audit.json"
DESIGN = ROOT / "data/evidence/committee-baseline-audit.protocol-freeze.json"
FREEZE = ROOT / "data/evidence/committee-baseline-audit.execution-freeze.json"
TRAINING = ROOT / "data/evidence/focused-ppo-training.json"
ORIGINAL = ROOT / "data/evidence/focused-ppo-study.json"
NAMES = (
    "Fixed Rotation",
    "Critical-aware Fixed Rotation",
    "Constrained Optimization",
    "Fairness Rule",
    *(f"PPO seed {seed}" for seed in TRAINING_SEEDS),
)
METRICS = (*METRIC_NAMES, *DIAGNOSTIC_NAMES)
STRESS_NAMES = ("critical_supply_75pct", "critical_supply_25pct")
RECIPE = {
    "post_hoc": True,
    "test_seeds": list(TEST_SEEDS),
    "training_seeds": list(TRAINING_SEEDS),
    "batch_size": 32,
    "shortages": list(SHORTAGES),
    "fairness_reward_tolerance": 0.005,
    "stress_critical_supply_fractions": [0.75, 0.25],
    "bootstrap_replicates": 10000,
    "bootstrap_seed": 20261031,
    "interval_confidence": 0.95,
    "inference": "descriptive pointwise paired hierarchical; no new confirmatory claim",
    "reward_component_tolerance": REWARD_RECONSTRUCTION_TOLERANCE,
}
AUDIT_SOURCES = (
    *SOURCE_PATHS,
    "src/gridguard/committee_baseline.py",
    "tools/evaluate_committee_baseline.py",
    "docs/COMMITTEE_BASELINE_PROTOCOL.md",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes() -> dict:
    return {name: sha(ROOT / name) for name in AUDIT_SOURCES}


def design_and_training() -> tuple[dict, dict]:
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    if design["protocol"]["sha256"] != sha(ROOT / design["protocol"]["path"]):
        raise ValueError("post-hoc protocol changed after design freeze")
    for path, digest in design["original_evidence"].items():
        if sha(ROOT / path) != digest:
            raise ValueError("original evidence changed after design freeze")
    training = load_training_record(
        TRAINING, design["original_evidence"][TRAINING.relative_to(ROOT).as_posix()]
    )
    return design, training


def freeze() -> str:
    _, training = design_and_training()
    return write_json(
        FREEZE,
        {
            "schema": "gridguard.committee_baseline_execution_freeze.1",
            "frozen_at_utc": datetime.now(UTC).isoformat(),
            "protocol_freeze_sha256": sha(DESIGN),
            "source_sha256": source_hashes(),
            "recipe": RECIPE,
            "models": {f"PPO seed {run['seed']}": run["model"] for run in training["runs"]},
        },
    )


def load_freeze() -> tuple[dict, dict]:
    _, training = design_and_training()
    frozen = read_json(FREEZE)
    if (
        frozen["schema"] != "gridguard.committee_baseline_execution_freeze.1"
        or frozen["protocol_freeze_sha256"] != sha(DESIGN)
        or frozen["source_sha256"] != source_hashes()
        or frozen["recipe"] != RECIPE
        or frozen["models"] != {f"PPO seed {run['seed']}": run["model"] for run in training["runs"]}
    ):
        raise ValueError("audit execution no longer matches its frozen contract")
    return frozen, training


def controllers(training: dict) -> list:
    return [
        FixedRotationController(),
        CriticalAwareFixedRotationController(),
        ConstrainedOptimizationController(),
        FairnessController(reward_tolerance=0.005),
        *[
            ConstrainedPPOController(
                load_model(ROOT / run["model"]["path"], expected_sha256=run["model"]["sha256"])
            )
            for run in training["runs"]
        ],
    ]


@torch.inference_mode()
def evaluate_conditions(methods: list, families: tuple, labels: tuple) -> dict:
    cubes, actions, inputs = [], [], []
    for label, scenarios in zip(labels, families, strict=True):
        family_rows, family_actions, family_inputs = [], [], []
        for scenario in scenarios:
            guard = scenario.capture_mutation_guard()
            rows, action_rows = [], []
            family_inputs.append(torch.cat((scenario.demand.flatten(1), scenario.supply), dim=1))
            for controller in methods:
                result = simulate(scenario, controller)
                if guard.changed_fields(scenario):
                    raise ValueError("controller mutated a matched audit scenario")
                metrics = compute_episode_metrics(result).as_dict()
                metrics.update(compute_duration_metrics(result).as_dict())
                metrics.update(audit_diagnostics(result))
                rows.append(torch.stack([metrics[name] for name in METRICS], dim=-1))
                action_rows.append(result.service_tiers)
            family_rows.append(torch.stack(rows))
            family_actions.append(torch.stack(action_rows))
            print(f"Audit {label}: seed {scenario.seed} complete", flush=True)
        cubes.append(torch.stack(family_rows, dim=1))
        actions.append(torch.stack(family_actions, dim=1))
        inputs.append(torch.stack(family_inputs))
    return {
        "cube": torch.stack(cubes, dim=1).to(torch.float64),
        "actions": torch.stack(actions, dim=1),
        "inputs": torch.stack(inputs).to(torch.float64),
    }


@torch.inference_mode()
def summary_tensors(cube: torch.Tensor, stress: torch.Tensor) -> dict:
    args = {"n_resamples": RECIPE["bootstrap_replicates"], "seed": RECIPE["bootstrap_seed"]}
    deltas = [cube[4:] - cube[index] for index in range(4)]
    return {
        "means": cube.mean((1, 2, 3)),
        "shortage_means": cube.mean((2, 3)),
        "maxima": cube.amax((1, 2, 3)),
        "procedure_means": cube[4:].mean((0, 1, 2, 3)).unsqueeze(0),
        "procedure_shortage_means": cube[4:].mean((0, 2, 3)),
        "paired": torch.stack([hierarchical_paired_bootstrap(d, **args) for d in deltas]),
        "conditional": torch.stack([conditional_paired_bootstrap(d, **args) for d in deltas]),
        "severity_paired": torch.stack(
            [
                torch.stack(
                    [
                        hierarchical_paired_bootstrap(
                            d[:, level : level + 1].expand(-1, 3, -1, -1, -1), **args
                        )
                        for level in range(3)
                    ]
                )
                for d in deltas
            ]
        ),
        "mechanical": hierarchical_paired_bootstrap((cube[1] - cube[0]).unsqueeze(0), **args),
        "stress_means": stress.mean((2, 3)),
        "stress_maxima": stress.amax((2, 3)),
        "stress_procedure_means": stress[4:].mean((0, 2, 3)),
    }


def rows(array: np.ndarray, labels=NAMES) -> dict:
    return {
        label: dict(zip(METRICS, array[index].tolist(), strict=True))
        for index, label in enumerate(labels)
    }


def interval(array: np.ndarray) -> dict:
    return {
        metric: dict(zip(("mean", "lower", "upper"), array[:, index].tolist(), strict=True))
        for index, metric in enumerate(METRICS)
    }


def summary_fields(blocks: dict) -> dict:
    return {
        "controller_means": rows(blocks["means"]),
        "procedure_means": rows(blocks["procedure_means"], ("PPO",)),
        "maxima": rows(blocks["maxima"]),
        "shortage_means": {
            level: rows(blocks["shortage_means"][:, index]) for index, level in enumerate(SHORTAGES)
        },
        "procedure_shortage_means": {
            level: rows(blocks["procedure_shortage_means"][index : index + 1], ("PPO",))
            for index, level in enumerate(SHORTAGES)
        },
        "paired_inference": {
            f"PPO - {baseline}": interval(blocks["paired"][index])
            for index, baseline in enumerate(NAMES[:4])
        },
        "per_run_inference": {
            f"{run} - {baseline}": interval(blocks["conditional"][index, ri])
            for index, baseline in enumerate(NAMES[:4])
            for ri, run in enumerate(NAMES[4:])
        },
        "shortage_paired_inference": {
            f"PPO - {baseline}": {
                level: interval(blocks["severity_paired"][index, si])
                for si, level in enumerate(SHORTAGES)
            }
            for index, baseline in enumerate(NAMES[:4])
        },
        "mechanical_rotation_inference": interval(blocks["mechanical"]),
        "stress": {
            label: {
                "controller_means": rows(blocks["stress_means"][:, index]),
                "procedure_means": rows(
                    blocks["stress_procedure_means"][index : index + 1], ("PPO",)
                ),
                "maxima": rows(blocks["stress_maxima"][:, index]),
            }
            for index, label in enumerate(STRESS_NAMES)
        },
    }


def hash_rows(array: np.ndarray, dtype: str) -> list:
    if array.ndim == 2:
        return [hashlib.sha256(row.astype(dtype).tobytes()).hexdigest() for row in array]
    return [hash_rows(part, dtype) for part in array]


def verify_original_replay(cube: torch.Tensor, scenario_hashes: list, action_hashes: list) -> None:
    original = read_json(ORIGINAL)
    if scenario_hashes != original["scenario_sha256_cube"]:
        raise ValueError("audit scenarios differ from the original 960 final conditions")
    reference = torch.tensor(original["episode_metric_cube"], dtype=torch.float64, device="cuda:0")
    for audit_index, original_index in (
        (0, 0),
        (2, 1),
        *((index + 4, index + 2) for index in range(5)),
    ):
        if not torch.equal(cube[audit_index, ..., : len(METRIC_NAMES)], reference[original_index]):
            raise ValueError("unchanged controller metric replay differs from frozen evidence")
        if action_hashes[audit_index] != original["action_sha256_cube"][original_index]:
            raise ValueError("unchanged controller action replay differs from frozen evidence")


def validate_cubes(cube: torch.Tensor, stress: torch.Tensor) -> None:
    for values, families in ((cube, 3), (stress, 2)):
        if values.shape != (9, families, 10, 32, len(METRICS)) or not bool(
            torch.isfinite(values).all()
        ):
            raise ValueError("audit metric cube shape or finite-value contract failed")
        ix = METRICS.index
        served, available, demand = (
            values[..., ix(name)] for name in ("served_mwh", "available_mwh", "demand_mwh")
        )
        if bool(((served < 0) | (served > available + 0.05) | (served > demand + 0.05)).any()):
            raise ValueError("audit energy balance is invalid")
        if not torch.allclose(values[..., ix("curtailed_mwh")], demand - served, atol=0.05, rtol=0):
            raise ValueError("audit curtailed energy identity failed")
        components = values[..., [ix(name) for name in REWARD_COMPONENT_NAMES]].sum(-1)
        if not torch.allclose(
            components,
            values[..., ix("episode_reward")],
            atol=REWARD_RECONSTRUCTION_TOLERANCE,
            rtol=0,
        ):
            raise ValueError("audit reward decomposition failed")
        if bool((values[..., ix("power_balance_violation_rate")] != 0).any()):
            raise ValueError("audit has a physical power-balance violation")
    if bool((stress[1:, ..., METRICS.index("all_critical_fallback_hours")] != 24).any()):
        raise ValueError("critical infeasibility was not exposed as fallback")
    if bool((stress[1:, ..., METRICS.index("critical_interruption_locality_hours")] != 168).any()):
        raise ValueError("partial critical deprivation was not counted")


@torch.inference_mode()
def evaluate() -> dict:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    frozen, training = load_freeze()
    runtime = run_gpu_preflight()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    started = datetime.now(UTC).isoformat()
    start = time.perf_counter()
    methods = controllers(training)
    scenarios = evaluation_scenarios(TEST_SEEDS)
    nominal = evaluate_conditions(methods, scenarios, SHORTAGES)
    stress_scenarios = tuple(
        tuple(critical_supply_stress(scenario, fraction) for scenario in scenarios[1])
        for fraction in (0.75, 0.25)
    )
    stress = evaluate_conditions(methods, stress_scenarios, STRESS_NAMES)
    validate_cubes(nominal["cube"], stress["cube"])
    summaries = summary_tensors(nominal["cube"], stress["cube"])
    blocks = _transfer(
        {
            **{f"nominal_{k}": v for k, v in nominal.items()},
            **{f"stress_{k}": v for k, v in stress.items()},
            **summaries,
        }
    )
    nominal_hashes = hash_rows(blocks["nominal_inputs"], "<f8")
    nominal_actions = hash_rows(blocks["nominal_actions"].reshape(9, 3, 10, 32, -1), "<i8")
    verify_original_replay(nominal["cube"], nominal_hashes, nominal_actions)
    load_freeze()
    return {
        "schema": SCHEMA,
        "recipe": RECIPE,
        "execution_freeze_sha256": sha(FREEZE),
        "source_sha256": frozen["source_sha256"],
        "models": frozen["models"],
        "started_at_utc": started,
        "elapsed_seconds": time.perf_counter() - start,
        "runtime": asdict(runtime),
        "controller_names": list(NAMES),
        "metric_names": list(METRICS),
        "nominal_matched_conditions": 960,
        "nominal_trajectories": 8640,
        "stress_conditions_per_family": 320,
        "stress_trajectories": 5760,
        "original_replay": (
            "exact input hashes, action hashes and per-condition metrics "
            "for all seven unchanged methods"
        ),
        "episode_metric_cube": blocks["nominal_cube"].tolist(),
        "stress_metric_cube": blocks["stress_cube"].tolist(),
        "scenario_sha256_cube": nominal_hashes,
        "action_sha256_cube": nominal_actions,
        "stress_scenario_sha256_cube": hash_rows(blocks["stress_inputs"], "<f8"),
        "stress_action_sha256_cube": hash_rows(
            blocks["stress_actions"].reshape(9, 2, 10, 32, -1), "<i8"
        ),
        **summary_fields(blocks),
    }


@torch.inference_mode()
def verify(path: Path, expected_sha256: str | None = None) -> dict:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    payload = read_json(path, expected_sha256)
    frozen, _ = load_freeze()
    if (
        payload["schema"] != SCHEMA
        or payload["recipe"] != RECIPE
        or payload["execution_freeze_sha256"] != sha(FREEZE)
        or payload["source_sha256"] != frozen["source_sha256"]
        or payload["models"] != frozen["models"]
        or payload["controller_names"] != list(NAMES)
        or payload["metric_names"] != list(METRICS)
    ):
        raise ValueError("audit artifact does not match its frozen contract")
    if tuple(
        payload[key]
        for key in (
            "nominal_matched_conditions",
            "nominal_trajectories",
            "stress_conditions_per_family",
            "stress_trajectories",
        )
    ) != (960, 8640, 320, 5760):
        raise ValueError("audit condition counts differ from the design")
    cube = torch.tensor(payload["episode_metric_cube"], dtype=torch.float64, device="cuda:0")
    stress = torch.tensor(payload["stress_metric_cube"], dtype=torch.float64, device="cuda:0")
    for key, shape in (
        ("scenario_sha256_cube", (3, 10, 32)),
        ("action_sha256_cube", (9, 3, 10, 32)),
        ("stress_scenario_sha256_cube", (2, 10, 32)),
        ("stress_action_sha256_cube", (9, 2, 10, 32)),
    ):
        _hash_cube(payload[key], shape)
    validate_cubes(cube, stress)
    verify_original_replay(cube, payload["scenario_sha256_cube"], payload["action_sha256_cube"])
    reconstructed = summary_fields(_transfer(summary_tensors(cube, stress)))
    for key, value in reconstructed.items():
        _compare_nested(payload[key], value, key)
    return {
        "passed": True,
        "schema": SCHEMA,
        "nominal_conditions": 960,
        "nominal_trajectories": 8640,
        "stress_trajectories": 5760,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--freeze", action="store_true")
    modes.add_argument("--verify", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--sha256")
    args = parser.parse_args()
    if args.freeze:
        print(f"Frozen audit implementation: {freeze()}")
    elif args.verify:
        print(json.dumps(verify(args.output, args.sha256), indent=2))
    else:
        print(f"Audit artifact SHA-256: {write_json(args.output, evaluate())}")


if __name__ == "__main__":
    main()

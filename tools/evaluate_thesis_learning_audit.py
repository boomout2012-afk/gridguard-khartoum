"""Freeze, execute and verify a fresh matched audit of fine-tuning plus selection."""

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
import yaml

from gridguard.allocation_study import METRIC_NAMES, SHORTAGES
from gridguard.committee_baseline import (
    REWARD_COMPONENT_NAMES,
    CriticalAwareFixedRotationController,
    audit_diagnostics,
)
from gridguard.constrained import ConstrainedOptimizationController, FairnessController
from gridguard.constrained_ppo import ConstrainedPPOController, load_model
from gridguard.controllers import FixedRotationController
from gridguard.duration_metrics import compute_duration_metrics
from gridguard.focused_training import SOURCE_PATHS, TRAINING_SEEDS, evaluation_scenarios
from gridguard.gpu import run_gpu_preflight
from gridguard.metrics import compute_episode_metrics
from gridguard.ppo_experiment import read_json, write_json
from gridguard.ppo_study import conditional_paired_bootstrap, hierarchical_paired_bootstrap
from gridguard.simulation import simulate
from gridguard.thesis_learning_audit import (
    METRICS,
    TEST_SEEDS,
    assert_disjoint_seeds,
    collect_seed_metadata,
    contrast_cubes,
    independent_trace_metrics,
    verify_trace_metrics,
)
from tools.evaluate_focused_ppo import _compare_nested, _hash_cube, _transfer, load_training_record

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "thesis-learning-audit"
OUTPUT = ROOT / f"data/evidence/{PREFIX}.json"
DESIGN = ROOT / f"data/evidence/{PREFIX}.protocol-freeze.json"
FREEZE = ROOT / f"data/evidence/{PREFIX}.execution-freeze.json"
VALIDATION = ROOT / f"data/evidence/{PREFIX}.validation.json"
PROTOCOL = "docs/THESIS_LEARNING_AUDIT_PROTOCOL.md"
TRAINING = ROOT / "data/evidence/focused-ppo-training.json"
SCHEMA = "gridguard.thesis_learning_audit.1"
NAMES = (
    "Fixed Rotation",
    "Critical-aware Fixed Rotation",
    "Constrained Optimization",
    "Fairness Rule",
    *(f"Behavior Cloning seed {seed}" for seed in TRAINING_SEEDS),
    *(f"PPO seed {seed}" for seed in TRAINING_SEEDS),
)
CONTRASTS = ("PPO - own Behavior Cloning", *(f"PPO - {name}" for name in NAMES[:4]))
RECIPE = {
    "post_hoc_motivation": True,
    "fresh_scenario_execution": True,
    "test_seeds": list(TEST_SEEDS),
    "training_seeds": list(TRAINING_SEEDS),
    "batch_size": 32,
    "shortages": list(SHORTAGES),
    "fairness_reward_tolerance": 0.005,
    "bootstrap_replicates": 10000,
    "bootstrap_seed": 20261031,
    "interval_confidence": 0.95,
    "interval_role": "descriptive pointwise nominal intervals; no new primary pass/fail claims",
    "stage_interpretation": "fine-tuning plus validation-selection relative to own fixed BC actor",
    "metric_tolerances": {"energy_mwh": 0.01, "episode_reward": 5e-5, "other": 5e-6},
}
SOURCES = tuple(
    dict.fromkeys(
        (
            *SOURCE_PATHS,
            "src/gridguard/committee_baseline.py",
            "src/gridguard/thesis_learning_audit.py",
            "tools/evaluate_thesis_learning_audit.py",
            "tests/test_thesis_learning_audit.py",
            PROTOCOL,
        )
    )
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes() -> dict:
    return {path: sha(ROOT / path) for path in SOURCES}


def prior_files() -> list[Path]:
    """Documented scientific metadata/artifacts; exclude only this new audit's files."""
    return sorted(
        p
        for base in (ROOT / "data/evidence", ROOT / "data/model")
        for p in base.rglob("*")
        if p.is_file()
        and not p.name.startswith(PREFIX)
        and p.suffix in {".json", ".yaml", ".npz", ".sha256"}
    )


def seed_inventory() -> dict:
    seeds, records, formulas = set(), [], []
    for path in prior_files():
        if path.suffix not in {".json", ".yaml"}:
            continue
        text = path.read_text(encoding="utf-8")
        payload = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
        found, streams = collect_seed_metadata(payload)
        seeds.update(found)
        relative = path.relative_to(ROOT).as_posix()
        records.append({"path": relative, "sha256": sha(path), "seed_count": len(found)})
        formulas.extend({"path": relative, **stream} for stream in streams)
    # Historical selected-run provenance states rollout_seed=training_seed+1000+iteration.
    # Cover every candidate seed 42/43/44, including both possible endpoint conventions.
    historical_rollouts = {s + 1000 + k for s in (42, 43, 44) for k in range(601)}
    historical_teacher = set(range(42, 56))
    seeds.update(historical_rollouts | historical_teacher)
    unresolved = [
        f
        for f in formulas
        if f["expanded_count"] is None
        and f["formula"].replace(" ", "") != "training_seed+1000+iteration"
    ]
    if unresolved:
        raise ValueError(f"unreviewed prior seed formulas: {unresolved}")
    assert_disjoint_seeds(TEST_SEEDS, seeds)
    return {
        "passed": True,
        "proposed_seeds": list(TEST_SEEDS),
        "prior_unique_seed_values": sorted(seeds),
        "metadata_files": records,
        "formula_records": formulas,
        "historical_rollout_bounds": [min(historical_rollouts), max(historical_rollouts)],
        "historical_teacher_conservative_bounds": [42, 55],
        "scope": (
            "all retained evidence/model JSON/YAML seed fields and bounded declared streams; "
            "not independent historical custody"
        ),
    }


def freeze_protocol() -> str:
    inventory = seed_inventory()
    return write_json(
        DESIGN,
        {
            "schema": "gridguard.thesis_learning_protocol_freeze.1",
            "frozen_at_utc": datetime.now(UTC).isoformat(),
            "protocol": {"path": PROTOCOL, "sha256": sha(ROOT / PROTOCOL)},
            "recipe": RECIPE,
            "seed_disjointness": inventory,
            "original_files": {p.relative_to(ROOT).as_posix(): sha(p) for p in prior_files()},
        },
    )


def design_and_training() -> tuple[dict, dict]:
    design = read_json(DESIGN)
    if design.get("schema") != "gridguard.thesis_learning_protocol_freeze.1":
        raise ValueError("learning audit protocol-freeze schema mismatch")
    if (
        design["protocol"] != {"path": PROTOCOL, "sha256": sha(ROOT / PROTOCOL)}
        or design["recipe"] != RECIPE
    ):
        raise ValueError("learning audit protocol changed after freeze")
    for path, digest in design["original_files"].items():
        if sha(ROOT / path) != digest:
            raise ValueError(f"original scientific artifact changed: {path}")
    assert_disjoint_seeds(TEST_SEEDS, set(design["seed_disjointness"]["prior_unique_seed_values"]))
    training = load_training_record(
        TRAINING, design["original_files"][TRAINING.relative_to(ROOT).as_posix()]
    )
    return design, training


def model_records(training: dict) -> dict:
    return {
        label: run[field]
        for field, labels in (("behavior_cloning_model", NAMES[4:9]), ("model", NAMES[9:14]))
        for label, run in zip(labels, training["runs"], strict=True)
    }


def freeze_execution() -> str:
    _, training = design_and_training()
    return write_json(
        FREEZE,
        {
            "schema": "gridguard.thesis_learning_execution_freeze.1",
            "frozen_at_utc": datetime.now(UTC).isoformat(),
            "protocol_freeze_sha256": sha(DESIGN),
            "source_sha256": source_hashes(),
            "recipe": RECIPE,
            "models": model_records(training),
        },
    )


def load_freeze() -> tuple[dict, dict]:
    _, training = design_and_training()
    frozen = read_json(FREEZE)
    if frozen.get("schema") != "gridguard.thesis_learning_execution_freeze.1":
        raise ValueError("learning audit execution-freeze schema mismatch")
    if (
        frozen["protocol_freeze_sha256"] != sha(DESIGN)
        or frozen["source_sha256"] != source_hashes()
        or frozen["recipe"] != RECIPE
        or frozen["models"] != model_records(training)
    ):
        raise ValueError("learning audit implementation/model freeze changed")
    return frozen, training


def controllers(training: dict) -> list:
    return [
        FixedRotationController(),
        CriticalAwareFixedRotationController(),
        ConstrainedOptimizationController(),
        FairnessController(reward_tolerance=0.005),
        *(
            ConstrainedPPOController(
                load_model(ROOT / record["path"], expected_sha256=record["sha256"])
            )
            for record in model_records(training).values()
        ),
    ]


def configure_cuda() -> dict:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    runtime = asdict(run_gpu_preflight())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    return runtime


@torch.inference_mode()
def evaluate_traces(methods: list) -> dict:
    families, action_families, endpoint_families, input_families = [], [], [], []
    error_maximum = torch.zeros(len(METRIC_NAMES), device="cuda:0", dtype=torch.float64)
    for level, scenarios in zip(SHORTAGES, evaluation_scenarios(TEST_SEEDS), strict=True):
        batches, action_batches, endpoint_batches, input_batches = [], [], [], []
        for scenario in scenarios:
            guard = scenario.capture_mutation_guard()
            rows, actions, endpoints = [], [], []
            input_batches.append(torch.cat((scenario.demand.flatten(1), scenario.supply), 1))
            for controller in methods:
                result = simulate(scenario, controller)
                if guard.changed_fields(scenario):
                    raise ValueError("controller mutated matched fresh input")
                metric = compute_episode_metrics(result).as_dict()
                metric.update(compute_duration_metrics(result).as_dict())
                independent, endpoint = independent_trace_metrics(result)
                error = verify_trace_metrics(
                    metric, {name: independent[name] for name in METRIC_NAMES}
                )
                error_maximum = torch.maximum(error_maximum, error)
                metric.update(audit_diagnostics(result))
                metric.update(
                    final_minimum_locality_service_ratio=endpoint.amin(-1),
                    final_locality_service_range_ratio=endpoint.amax(-1) - endpoint.amin(-1),
                )
                rows.append(torch.stack([metric[name] for name in METRICS], -1))
                actions.append(result.service_tiers)
                endpoints.append(endpoint)
            batches.append(torch.stack(rows))
            action_batches.append(torch.stack(actions))
            endpoint_batches.append(torch.stack(endpoints))
            print(f"Fresh learning audit {level}: seed {scenario.seed} complete", flush=True)
        families.append(torch.stack(batches, 1))
        action_families.append(torch.stack(action_batches, 1))
        endpoint_families.append(torch.stack(endpoint_batches, 1))
        input_families.append(torch.stack(input_batches))
    return {
        "cube": torch.stack(families, 1).double(),
        "actions": torch.stack(action_families, 1),
        "endpoints": torch.stack(endpoint_families, 1),
        "inputs": torch.stack(input_families).double(),
        "reconstruction_errors": error_maximum,
    }


def validate_cubes(cube: torch.Tensor, endpoints: torch.Tensor) -> None:
    if cube.shape != (14, 3, 10, 32, len(METRICS)) or endpoints.shape != (14, 3, 10, 32, 7):
        raise ValueError("learning audit cube axes differ from the protocol")
    if not bool(torch.isfinite(cube).all() & torch.isfinite(endpoints).all()):
        raise ValueError("learning audit contains nonfinite results")
    ix = METRICS.index
    demand, served, available = (
        cube[..., ix(k)] for k in ("demand_mwh", "served_mwh", "available_mwh")
    )
    if bool(((served < 0) | (served > available + 0.05) | (served > demand + 0.05)).any()):
        raise ValueError("learning audit energy balance failed")
    if not torch.allclose(cube[..., ix("curtailed_mwh")], demand - served, atol=0.05, rtol=0):
        raise ValueError("learning audit curtailed-energy identity failed")
    if not torch.allclose(
        cube[..., [ix(k) for k in REWARD_COMPONENT_NAMES]].sum(-1),
        cube[..., ix("episode_reward")],
        atol=5e-5,
        rtol=0,
    ):
        raise ValueError("learning audit reward reconstruction failed")
    if bool(((endpoints < 0) | (endpoints > 1 + 5e-6)).any()):
        raise ValueError("locality service ratios out of physical range")
    for name, expected in (
        ("final_minimum_locality_service_ratio", endpoints.amin(-1)),
        ("final_locality_service_range_ratio", endpoints.amax(-1) - endpoints.amin(-1)),
    ):
        if not torch.allclose(cube[..., ix(name)], expected, atol=1e-11, rtol=0):
            raise ValueError("locality endpoint metric disagrees with retained ratios")


@torch.inference_mode()
def summary_tensors(cube: torch.Tensor, endpoints: torch.Tensor) -> dict:
    args = {"n_resamples": RECIPE["bootstrap_replicates"], "seed": RECIPE["bootstrap_seed"]}
    deltas = contrast_cubes(cube)
    return {
        "means": cube.mean((1, 2, 3)),
        "shortage_means": cube.mean((2, 3)),
        "maxima": cube.amax((1, 2, 3)),
        "procedure_means": torch.stack(
            (cube[4:9].mean((0, 1, 2, 3)), cube[9:14].mean((0, 1, 2, 3)))
        ),
        "procedure_shortage_means": torch.stack(
            (cube[4:9].mean((0, 2, 3)), cube[9:14].mean((0, 2, 3)))
        ),
        "locality_means": endpoints.mean((1, 2, 3)),
        "paired": torch.stack([hierarchical_paired_bootstrap(delta, **args) for delta in deltas]),
        "conditional": torch.stack(
            [conditional_paired_bootstrap(delta, **args) for delta in deltas]
        ),
        "severity_paired": torch.stack(
            [
                torch.stack(
                    [
                        hierarchical_paired_bootstrap(
                            delta[:, si : si + 1].expand(-1, 3, -1, -1, -1), **args
                        )
                        for si in range(3)
                    ]
                )
                for delta in deltas
            ]
        ),
    }


def rows(array: np.ndarray, labels=NAMES) -> dict:
    return {
        label: dict(zip(METRICS, array[index].tolist(), strict=True))
        for index, label in enumerate(labels)
    }


def interval(array: np.ndarray) -> dict:
    return {
        name: dict(zip(("mean", "lower", "upper"), array[:, index].tolist(), strict=True))
        for index, name in enumerate(METRICS)
    }


def summary_fields(blocks: dict) -> dict:
    return {
        "controller_means": rows(blocks["means"]),
        "maxima": rows(blocks["maxima"]),
        "procedure_means": rows(blocks["procedure_means"], ("Behavior Cloning", "PPO")),
        "shortage_means": {
            level: rows(blocks["shortage_means"][:, i]) for i, level in enumerate(SHORTAGES)
        },
        "procedure_shortage_means": {
            level: rows(blocks["procedure_shortage_means"][:, i], ("Behavior Cloning", "PPO"))
            for i, level in enumerate(SHORTAGES)
        },
        "final_locality_service_means": {
            name: blocks["locality_means"][i].tolist() for i, name in enumerate(NAMES)
        },
        "paired_inference": {
            name: interval(blocks["paired"][i]) for i, name in enumerate(CONTRASTS)
        },
        "per_run_inference": {
            name: {
                str(seed): interval(blocks["conditional"][i, ri])
                for ri, seed in enumerate(TRAINING_SEEDS)
            }
            for i, name in enumerate(CONTRASTS)
        },
        "shortage_paired_inference": {
            name: {
                level: interval(blocks["severity_paired"][i, si])
                for si, level in enumerate(SHORTAGES)
            }
            for i, name in enumerate(CONTRASTS)
        },
    }


def hash_rows(array: np.ndarray, dtype: str) -> list:
    if array.ndim == 2:
        return [hashlib.sha256(row.astype(dtype).tobytes()).hexdigest() for row in array]
    return [hash_rows(part, dtype) for part in array]


def flatten_hashes(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [item for child in value for item in flatten_hashes(child)]


def old_scenario_hashes() -> set[str]:
    found = set()
    for path in prior_files():
        if path.suffix != ".json" or path.parent != ROOT / "data/evidence":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for key, value in payload.items():
            if "scenario_sha256_cube" in key:
                found.update(flatten_hashes(value))
    return found


def assert_fresh_hashes(hashes: list, previous: set[str]) -> None:
    values = flatten_hashes(hashes)
    if len(values) != len(set(values)):
        raise ValueError("duplicate scenario hashes within fresh conditions")
    if set(values) & previous:
        raise ValueError("fresh scenario hashes overlap previous evidence")


def contract_fields(frozen: dict, freeze_digest: str) -> dict:
    return {
        "schema": SCHEMA,
        "recipe": RECIPE,
        "execution_freeze_sha256": freeze_digest,
        "source_sha256": frozen["source_sha256"],
        "models": frozen["models"],
        "controller_names": list(NAMES),
        "metric_names": list(METRICS),
        "matched_scenario_count": 960,
        "independent_base_draws": 320,
        "controller_trajectory_count": 13440,
    }


def validate_contract(payload: dict, frozen: dict, freeze_digest: str) -> None:
    if any(payload.get(k) != v for k, v in contract_fields(frozen, freeze_digest).items()):
        raise ValueError("learning audit artifact contract mismatch")


@torch.inference_mode()
def evaluate() -> dict:
    frozen, training = load_freeze()
    runtime = configure_cuda()
    started = datetime.now(UTC).isoformat()
    start = time.perf_counter()
    traces = evaluate_traces(controllers(training))
    validate_cubes(traces["cube"], traces["endpoints"])
    blocks = _transfer({**traces, **summary_tensors(traces["cube"], traces["endpoints"])})
    scenario_hashes = hash_rows(blocks["inputs"], "<f8")
    previous = old_scenario_hashes()
    assert_fresh_hashes(scenario_hashes, previous)
    load_freeze()
    return {
        **contract_fields(frozen, sha(FREEZE)),
        "started_at_utc": started,
        "elapsed_seconds": time.perf_counter() - start,
        "runtime": runtime,
        "episode_metric_cube": blocks["cube"].tolist(),
        "final_locality_service_ratio_cube": blocks["endpoints"].tolist(),
        "scenario_sha256_cube": scenario_hashes,
        "action_sha256_cube": hash_rows(blocks["actions"].reshape(14, 3, 10, 32, -1), "<i8"),
        "maximum_independent_metric_discrepancies": dict(
            zip(METRIC_NAMES, blocks["reconstruction_errors"].tolist(), strict=True)
        ),
        "previous_scenario_hashes_checked": len(previous),
        "scenario_hash_overlap": 0,
        "limits": [
            "Synthetic same-generator replication, not distribution shift or field validation.",
            "Five runs; pointwise nominal intervals do not establish simultaneous superiority.",
            "Stage contrast includes fine-tuning, extra computation "
            "and validation checkpoint selection.",
            "No equal-budget additional-imitation or from-scratch comparison.",
            "Independent metric reconstruction shares the frozen simulator traces.",
        ],
        **summary_fields(blocks),
    }


@torch.inference_mode()
def verify(path: Path, *, replay: bool = False) -> dict:
    payload = read_json(path)
    frozen, training = load_freeze()
    configure_cuda()
    validate_contract(payload, frozen, sha(FREEZE))
    cube = torch.tensor(payload["episode_metric_cube"], device="cuda:0", dtype=torch.float64)
    endpoints = torch.tensor(
        payload["final_locality_service_ratio_cube"], device="cuda:0", dtype=torch.float64
    )
    validate_cubes(cube, endpoints)
    _hash_cube(payload["scenario_sha256_cube"], (3, 10, 32))
    _hash_cube(payload["action_sha256_cube"], (14, 3, 10, 32))
    assert_fresh_hashes(payload["scenario_sha256_cube"], old_scenario_hashes())
    reconstructed = summary_fields(_transfer(summary_tensors(cube, endpoints)))
    for key, value in reconstructed.items():
        _compare_nested(payload[key], value, key)
    if replay:
        traces = evaluate_traces(controllers(training))
        if not torch.equal(traces["cube"], cube) or not torch.equal(traces["endpoints"], endpoints):
            raise ValueError("full replay metric/locality endpoint mismatch")
        blocks = _transfer(traces)
        if (
            hash_rows(blocks["inputs"], "<f8") != payload["scenario_sha256_cube"]
            or hash_rows(blocks["actions"].reshape(14, 3, 10, 32, -1), "<i8")
            != payload["action_sha256_cube"]
        ):
            raise ValueError("full replay scenario/action hash mismatch")
        _compare_nested(
            payload["maximum_independent_metric_discrepancies"],
            dict(zip(METRIC_NAMES, blocks["reconstruction_errors"].tolist(), strict=True)),
            "reconstruction errors",
        )
    load_freeze()
    return {
        "schema": "gridguard.thesis_learning_validation.1",
        "passed": True,
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "artifact_sha256": sha(path),
        "execution_freeze_sha256": sha(FREEZE),
        "protocol_freeze_sha256": sha(DESIGN),
        "full_replay": replay,
        "replayed_trajectories": 13440 if replay else 0,
        "means_and_all_intervals_reconstructed_on_cuda": True,
        "original_artifacts_unchanged": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--freeze-protocol", action="store_true")
    mode.add_argument("--freeze-execution", action="store_true")
    mode.add_argument("--verify", action="store_true")
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    if args.freeze_protocol:
        print(f"Protocol frozen: {freeze_protocol()}")
    elif args.freeze_execution:
        print(f"Implementation frozen: {freeze_execution()}")
    elif args.verify:
        receipt = verify(OUTPUT, replay=args.replay)
        print(json.dumps(receipt, indent=2))
        if args.replay:
            print(f"Validation receipt: {write_json(VALIDATION, receipt)}")
    else:
        print(f"Evidence frozen: {write_json(OUTPUT, evaluate())}")


if __name__ == "__main__":
    main()

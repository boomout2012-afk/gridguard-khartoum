"""Evaluate the declared constrained-allocation study without replacing old evidence."""

from __future__ import annotations

import argparse
import hashlib
import math
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from gridguard.allocation_study import (
    BATCH_SIZE,
    CONTROLLER_NAMES,
    METRIC_NAMES,
    PAIR_INDICES,
    PROTOCOL,
    REWARD_TOLERANCES,
    SHORTAGES,
    TEST_SEEDS,
    VALIDATION_NAMES,
    VALIDATION_SEEDS,
    assess_hybrid_comparison,
    canonical_bytes,
    final_service_ratios,
    load_study_artifact,
    select_validation_settings,
    source_hashes,
)
from gridguard.constrained import ConstrainedOptimizationController, FairnessController
from gridguard.controllers import FixedRotationController, PPOController
from gridguard.duration_metrics import compute_duration_metrics
from gridguard.gpu import run_gpu_preflight
from gridguard.hybrid import FeasiblePPOController, PPOHybridController
from gridguard.metrics import compute_episode_metrics
from gridguard.optimization import OptimizationController
from gridguard.policy import BUNDLED_ARTIFACT_SHA256, load_ppo_policy
from gridguard.scenarios import generate_scenario_base_draws, scenario_from_base_draws
from gridguard.simulation import simulate
from gridguard.statistics import stratified_paired_bootstrap

ROOT = Path(__file__).resolve().parents[1]
LOCALITY_DURATION_NAMES = (
    "locality_blackout_hours",
    "locality_critical_interruption_hours",
    "locality_off_hours",
    "locality_longest_blackout_hours",
    "locality_longest_critical_interruption_hours",
)


class TimedController:
    """Synchronized whole-call wall timing; the scientific return stays on CUDA."""

    def __init__(self, controller):
        self.controller = controller
        self.name = controller.name
        self.samples_ms: list[float] = []

    def act(self, context):
        torch.cuda.synchronize()
        start = time.perf_counter()
        action = self.controller.act(context)
        torch.cuda.synchronize()
        self.samples_ms.append((time.perf_counter() - start) * 1000)
        return action


def _controllers(phase: str, actor, selected: dict):
    if phase == "validation":
        return (
            ConstrainedOptimizationController(),
            *(FairnessController(value) for value in REWARD_TOLERANCES),
            *(PPOHybridController(actor, value) for value in REWARD_TOLERANCES),
        )
    return (
        FixedRotationController(),
        PPOController(actor),
        OptimizationController(),
        ConstrainedOptimizationController(),
        FairnessController(selected["Fairness Rule"]),
        FeasiblePPOController(actor),
        PPOHybridController(actor, selected["PPO Hybrid"]),
    )


def _timings(controllers, labels):
    rows = {}
    for name, controller in zip(labels, controllers, strict=True):
        # The first full day is retained as a cold/early sample, excluded from warm summary.
        warm = sorted(controller.samples_ms[24:])
        rows[name] = {
            "median_ms_per_batch": statistics.median(warm),
            "p95_ms_per_batch": warm[math.ceil(0.95 * len(warm)) - 1],
            "warm_call_count": len(warm),
            "initial_day_call_ms": controller.samples_ms[:24],
        }
    return rows


@torch.inference_mode()
def evaluate(
    phase: str,
    *,
    selection_path: Path | None = None,
    selection_sha256: str | None = None,
) -> dict:
    selected = {}
    if phase == "test":
        if selection_path is None or selection_sha256 is None:
            raise ValueError("test requires the frozen validation artifact and its SHA-256")
        selected = load_study_artifact(
            selection_path, expected_sha256=selection_sha256, phase="validation", root=ROOT
        )["selected_tolerances"]
    bound_sources = source_hashes(ROOT)
    runtime = run_gpu_preflight()
    print(f"CUDA preflight passed: {runtime.device_name}", flush=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    actor = load_ppo_policy(
        ROOT / "data/model/manifest.json", expected_artifact_sha256=BUNDLED_ARTIFACT_SHA256
    )
    labels = VALIDATION_NAMES if phase == "validation" else CONTROLLER_NAMES
    seeds = VALIDATION_SEEDS if phase == "validation" else TEST_SEEDS
    controllers = [
        TimedController(controller) for controller in _controllers(phase, actor, selected)
    ]
    by_controller = [[] for _ in controllers]
    by_duration = [[] for _ in controllers]
    by_action = [[] for _ in controllers]
    by_service = [[] for _ in controllers]
    inputs = []
    shortfalls = []
    draws = [generate_scenario_base_draws(BATCH_SIZE, seed, device="cuda:0") for seed in seeds]
    for severity in SHORTAGES:
        rows = [[] for _ in controllers]
        durations = [[] for _ in controllers]
        actions = [[] for _ in controllers]
        services = [[] for _ in controllers]
        severity_inputs = []
        severity_shortfalls = []
        for seed, draw in zip(seeds, draws, strict=True):
            scenario = scenario_from_base_draws(draw, severity)
            guard = scenario.capture_mutation_guard()
            severity_inputs.append(torch.cat((scenario.demand.flatten(1), scenario.supply), dim=1))
            severity_shortfalls.append(
                (scenario.demand[..., :3].sum(dim=(-1, -2)) > scenario.supply).sum(dim=1)
            )
            for ci, controller in enumerate(controllers):
                result = simulate(scenario, controller)
                if guard.changed_fields(scenario):
                    raise ValueError("controller mutated a matched scenario")
                metrics = compute_episode_metrics(result).as_dict()
                duration = compute_duration_metrics(result).as_dict()
                metrics.update(duration)
                rows[ci].append(torch.stack([metrics[name] for name in METRIC_NAMES], dim=-1))
                durations[ci].append(
                    torch.stack([duration[name] for name in LOCALITY_DURATION_NAMES], dim=-1)
                )
                actions[ci].append(result.service_tiers)
                services[ci].append(final_service_ratios(result))
            print(f"{phase}: {severity} seed {seed} complete", flush=True)
        for ci in range(len(controllers)):
            by_controller[ci].append(torch.stack(rows[ci]))
            by_duration[ci].append(torch.stack(durations[ci]))
            by_action[ci].append(torch.stack(actions[ci]))
            by_service[ci].append(torch.stack(services[ci]))
        inputs.append(torch.stack(severity_inputs))
        shortfalls.append(torch.stack(severity_shortfalls))
    cube = torch.stack([torch.stack(rows) for rows in by_controller]).to(torch.float64)
    service_cube = torch.stack([torch.stack(rows) for rows in by_service]).to(torch.float64)
    packed_tensors = {
        "cube": cube,
        "locality_duration": torch.stack([torch.stack(rows) for rows in by_duration]),
        "actions": torch.stack([torch.stack(rows) for rows in by_action]),
        "final_service": service_cube,
        "minimum_service_mean": service_cube.amin(dim=-1).mean(dim=(1, 2, 3)),
        "shortfalls": torch.stack(shortfalls),
        "inputs": torch.stack(inputs),
        "means": cube.mean(dim=(1, 2, 3)),
        "shortage_means": cube.mean(dim=(2, 3)),
    }
    if phase == "validation":
        packed_tensors["choices"] = select_validation_settings(cube)
        col = {name: index for index, name in enumerate(METRIC_NAMES)}
        mean = packed_tensors["means"]
        eligible = (
            (cube[..., col["critical_outage_locality_hours"]].amax(dim=(1, 2, 3)) == 0)
            & (cube[..., col["power_balance_violation_rate"]].amax(dim=(1, 2, 3)) == 0)
            & (mean[:, col["fairness"]] >= mean[0, col["fairness"]])
            & (mean[:, col["utilization"]] >= mean[0, col["utilization"]] - 0.002)
        )
        packed_tensors["eligible"] = eligible
    else:
        packed_tensors["intervals"] = torch.stack(
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
    # One final scientific transfer per phase; host work below assembles portable evidence.
    packed = torch.cat([tensor.reshape(-1).to(torch.float64) for tensor in packed_tensors.values()])
    host = packed.cpu().numpy()
    blocks = {}
    offset = 0
    for name, tensor in packed_tensors.items():
        blocks[name] = host[offset : offset + tensor.numel()].reshape(tuple(tensor.shape))
        offset += tensor.numel()
    if source_hashes(ROOT) != bound_sources:
        raise ValueError("scientific source changed during evaluation")
    means = {
        name: dict(zip(METRIC_NAMES, blocks["means"][ci].tolist(), strict=True))
        for ci, name in enumerate(labels)
    }
    shortage_means = {
        level: {
            name: dict(zip(METRIC_NAMES, blocks["shortage_means"][ci, si].tolist(), strict=True))
            for ci, name in enumerate(labels)
        }
        for si, level in enumerate(SHORTAGES)
    }
    payload = {
        "schema": "gridguard.allocation_study.1",
        "phase": phase,
        "protocol": PROTOCOL,
        "seeds": list(seeds),
        "controller_names": list(labels),
        "metric_names": list(METRIC_NAMES),
        "source_sha256": bound_sources,
        "runtime": asdict(runtime),
        "episode_metric_cube": blocks["cube"].tolist(),
        "controller_means": means,
        "shortage_means": shortage_means,
        "locality_duration_names": list(LOCALITY_DURATION_NAMES),
        "locality_duration_cube": blocks["locality_duration"].tolist(),
        "final_locality_service_ratio_cube": blocks["final_service"].tolist(),
        "mean_final_minimum_locality_service_ratio": dict(
            zip(labels, blocks["minimum_service_mean"].tolist(), strict=True)
        ),
        "critical_supply_shortfall_hours": blocks["shortfalls"].tolist(),
        "scenario_sha256_cube": [
            [
                [
                    hashlib.sha256(blocks["inputs"][si, zi, ei].astype("<f8").tobytes()).hexdigest()
                    for ei in range(BATCH_SIZE)
                ]
                for zi in range(len(seeds))
            ]
            for si in range(3)
        ],
        "action_sha256_cube": [
            [
                [
                    [
                        hashlib.sha256(
                            blocks["actions"][ci, si, zi, ei].astype("<i8").tobytes()
                        ).hexdigest()
                        for ei in range(BATCH_SIZE)
                    ]
                    for zi in range(len(seeds))
                ]
                for si in range(3)
            ]
            for ci in range(len(labels))
        ],
        "decision_timing": {
            "scope": "synchronized wall time for the entire Controller.act call, batch size 32; "
            "first 24 calls retained separately; excludes model loading, simulation and UI; "
            "one local session in fixed controller order, not a formal latency benchmark",
            "controllers": _timings(controllers, labels),
        },
        "limitations": [
            "Synthetic seven-locality decision support; no real electrical-network validation.",
            "One frozen PPO actor; no retraining or learned future value in this experiment.",
            "Raw PPO has different feature information; constrained filters share exact "
            "causal state.",
            "Historical PPO optimizer-imitation initialization prevents attribution to PPO alone.",
            "Pointwise scenario intervals conditional on this actor; not training-run variability.",
            "A result for this hybrid does not settle all RL or lookahead approaches.",
        ],
    }
    if phase == "validation":
        choices = blocks["choices"].astype(int).tolist()
        payload["selected_tolerances"] = {
            name: REWARD_TOLERANCES[index]
            for name, index in zip(("Fairness Rule", "PPO Hybrid"), choices, strict=True)
        }
        payload["validation_eligible"] = dict(
            zip(labels, blocks["eligible"].astype(bool).tolist(), strict=True)
        )
        payload["selection_fallback_used"] = {
            name: not bool(blocks["eligible"][start : start + 7].any())
            for name, start in (("Fairness Rule", 1), ("PPO Hybrid", 8))
        }
    else:
        payload["selected_tolerances"] = selected
        payload["selection_path"] = selection_path.resolve().relative_to(ROOT).as_posix()
        payload["selection_sha256"] = selection_sha256
        payload["paired_inference"] = {
            f"{labels[a]} - {labels[b]}": {
                metric: dict(
                    zip(
                        ("mean_delta", "ci95_lower", "ci95_upper"),
                        blocks["intervals"][pi, :, mi].tolist(),
                        strict=True,
                    )
                )
                for mi, metric in enumerate(METRIC_NAMES)
            }
            for pi, (a, b) in enumerate(PAIR_INDICES)
        }
        safe = all(
            blocks["cube"][ci, ..., METRIC_NAMES.index(metric)].max() == 0
            for ci in (3, 4, 5, 6)
            for metric in ("critical_outage_locality_hours", "power_balance_violation_rate")
        )
        claims = {}
        for comparator in ("Constrained Optimization", "Fairness Rule"):
            key = f"PPO Hybrid - {comparator}"
            deltas = {
                level: {
                    metric: shortage_means[level]["PPO Hybrid"][metric]
                    - shortage_means[level][comparator][metric]
                    for metric in ("episode_reward", "fairness", "utilization")
                }
                for level in SHORTAGES
            }
            claims[key] = assess_hybrid_comparison(
                payload["paired_inference"][key], deltas, critical_safe=safe
            )
        payload["claims"] = {
            "comparisons": claims,
            "hybrid_reward_improvement_with_service_guardrails": all(
                row["passes"] for row in claims.values()
            ),
        }
    return payload


def write_report(payload: dict, path: Path, digest: str) -> None:
    title = "Validation" if payload["phase"] == "validation" else "Final matched comparison"
    lines = [
        f"# Constrained allocation study {title.lower()}",
        "",
        "23 October 2026. Synthetic decision-support simulation, seven localities, 24 hours.",
        "",
        f"Evidence SHA-256: `{digest}`. Settings: `{payload['selected_tolerances']}`.",
        "",
        "Jain fairness and utilization are mean hourly ratios. Critical interruptions sum "
        "locality-hours below 99% critical service. All numbers below are equal-episode means.",
        "",
        "| Method | Critical interruption (locality-h) | Jain fairness (%) | Utilization (%) | "
        "Daily reward | Switching (%) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in payload["controller_means"].items():
        lines.append(
            f"| {name} | {row['critical_outage_locality_hours']:.4f} | "
            f"{100 * row['fairness']:.4f} | {100 * row['utilization']:.4f} | "
            f"{row['episode_reward']:.5f} | {100 * row['switching_rate']:.4f} |"
        )
    if payload["phase"] == "test":
        lines += ["", "## Primary decision", ""]
        passes = payload["claims"]["hybrid_reward_improvement_with_service_guardrails"]
        lines.append(
            "The hybrid passes the predeclared reward and service guardrails against both "
            "constrained comparators."
            if passes
            else "The hybrid does not pass the predeclared reward and service guardrails against "
            "both constrained comparators. Report individual effects as gains, ties or tradeoffs."
        )
        lines += [
            "",
            "| Comparison | Reward delta [95% CI] | Fairness delta pp [95% CI] | "
            "Utilization delta pp [95% CI] |",
            "| --- | ---: | ---: | ---: |",
        ]
        for label, row in payload["paired_inference"].items():
            cells = []
            for metric, scale in (("episode_reward", 1), ("fairness", 100), ("utilization", 100)):
                item = row[metric]
                cells.append(
                    f"{scale * item['mean_delta']:+.4f} "
                    f"[{scale * item['ci95_lower']:+.4f}, {scale * item['ci95_upper']:+.4f}]"
                )
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        for level, methods in payload["shortage_means"].items():
            lines += [
                "",
                f"## {level.capitalize()} shortage",
                "",
                "| Method | Critical interruption (locality-h) | Jain fairness (%) | "
                "Utilization (%) | Daily reward |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
            for name, row in methods.items():
                lines.append(
                    f"| {name} | {row['critical_outage_locality_hours']:.4f} | "
                    f"{100 * row['fairness']:.4f} | {100 * row['utilization']:.4f} | "
                    f"{row['episode_reward']:.5f} |"
                )
        lines += [
            "",
            "## Interruption and locality diagnostics",
            "",
            "Weakest locality is the minimum final served-energy/demand ratio within each "
            "episode, then averaged across episodes. It differs from mean hourly Jain fairness. "
            "Critical duration gap is the within-episode range across localities. The evidence "
            "also retains all locality durations and streaks.",
            "",
            "| Method | Blackout locality-h | Critical duration gap h | Longest critical "
            "interruption h | Weakest locality service % | Power violation % |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for name, row in payload["controller_means"].items():
            weakest = payload["mean_final_minimum_locality_service_ratio"][name]
            lines.append(
                f"| {name} | {row['blackout_locality_hours']:.4f} | "
                f"{row['critical_interruption_duration_range_hours']:.4f} | "
                f"{row['longest_critical_interruption_hours']:.4f} | {100 * weakest:.4f} | "
                f"{100 * row['power_balance_violation_rate']:.4f} |"
            )
        shortfalls = payload["critical_supply_shortfall_hours"]
        fallback_hours = sum(
            value for severity in shortfalls for seed in severity for value in seed
        )
        lines += [
            "",
            f"Critical demand exceeded available supply for {fallback_hours:g} scenario-hours "
            "across the matched conditions. Each constrained method uses the all-Critical-Only "
            "fallback in exactly those hours.",
        ]
        lines += [
            "",
            "## Controller timing",
            "",
            payload["decision_timing"]["scope"],
            "",
            "| Method | Median batch ms | p95 batch ms |",
            "| --- | ---: | ---: |",
        ]
        for name, row in payload["decision_timing"]["controllers"].items():
            lines.append(
                f"| {name} | {row['median_ms_per_batch']:.3f} | {row['p95_ms_per_batch']:.3f} |"
            )
    lines += ["", "## Scope and interpretation", ""]
    lines.extend(f"- {item}" for item in payload["limitations"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("validation", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--selection-sha256")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite an allocation-study artifact")
    payload = evaluate(
        args.phase, selection_path=args.selection, selection_sha256=args.selection_sha256
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_bytes(payload)
    digest = hashlib.sha256(raw).hexdigest()
    args.output.write_bytes(raw)
    load_study_artifact(args.output, expected_sha256=digest, phase=args.phase, root=ROOT)
    args.output.with_suffix(".sha256").write_text(digest + "\n", encoding="ascii")
    write_report(payload, args.output.with_suffix(".md"), digest)
    print(f"Verified {args.output}; SHA-256 {digest}", flush=True)
    print(f"Selected tolerances: {payload['selected_tolerances']}", flush=True)
    if args.phase == "test":
        print(f"Primary result: {payload['claims']}", flush=True)


if __name__ == "__main__":
    main()

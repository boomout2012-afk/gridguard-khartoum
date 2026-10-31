"""Run the predeclared additive proposal experiment on the presentation GPU."""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import asdict
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from gridguard.controllers import FixedRotationController, PPOController
from gridguard.duration_metrics import compute_duration_metrics
from gridguard.forecast_model import evaluate_forecast_model, fit_ridge_forecast
from gridguard.gpu import run_gpu_preflight
from gridguard.metrics import compute_episode_metrics
from gridguard.optimization import OptimizationController
from gridguard.policy import BUNDLED_ARTIFACT_SHA256, load_ppo_policy
from gridguard.proposal_evidence import (
    BATCH_SIZE,
    CONTROLLER_NAMES,
    EVALUATION_SEEDS,
    FORECAST_NAMES,
    LOCALITY_METRIC_NAMES,
    METRIC_NAMES,
    PAIR_INDICES,
    PROTOCOL,
    SHORTAGE_LEVELS,
    canonical_proposal_bytes,
    validate_proposal_supplement,
)
from gridguard.scenarios import generate_scenario_base_draws, scenario_from_base_draws
from gridguard.simulation import simulate
from gridguard.statistics import stratified_paired_bootstrap

ROOT = Path(__file__).resolve().parents[1]


@torch.inference_mode()
def evaluate(root: Path = ROOT) -> dict:
    runtime = run_gpu_preflight()
    print(f"Preflight passed: {runtime.device_name}", flush=True)
    source_paths = (
        "src/gridguard/domain.py",
        "src/gridguard/controllers.py",
        "src/gridguard/simulation.py",
        "src/gridguard/scenarios.py",
        "src/gridguard/metrics.py",
        "src/gridguard/statistics.py",
        "src/gridguard/optimization.py",
        "src/gridguard/duration_metrics.py",
        "src/gridguard/forecast_model.py",
        "src/gridguard/policy.py",
        "tools/evaluate_proposal_extension.py",
        "docs/PROPOSAL_EXTENSION.md",
        "docs/source/graduation-project-proposal.pdf",
        "data/model/manifest.json",
        "data/model/ppo-policy.npz",
        "data/model/training-provenance.yaml",
    )
    source_sha = {
        path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in source_paths
    }
    actor = load_ppo_policy(
        root / "data/model/manifest.json", expected_artifact_sha256=BUNDLED_ARTIFACT_SHA256
    )
    controllers = (FixedRotationController(), PPOController(actor), OptimizationController())
    base_draws = [
        generate_scenario_base_draws(BATCH_SIZE, seed, device="cuda:0") for seed in EVALUATION_SEEDS
    ]
    controller_rows = [[] for _ in controllers]
    locality_rows = [[] for _ in controllers]
    scenario_rows = []
    evaluation_demand = []
    for si, severity in enumerate(SHORTAGE_LEVELS):
        by_controller = [[] for _ in controllers]
        by_locality = [[] for _ in controllers]
        severity_scenarios = []
        for seed_index, draw in enumerate(base_draws):
            scenario = scenario_from_base_draws(draw, severity)
            guard = scenario.capture_mutation_guard()
            severity_scenarios.append(
                torch.cat((scenario.demand.flatten(1), scenario.supply), dim=1)
            )
            if si == 0:
                evaluation_demand.append(scenario.demand)
            for ci, controller in enumerate(controllers):
                result = simulate(scenario, controller)
                if guard.changed_fields(scenario):
                    raise ValueError("controller mutated the shared scenario")
                metrics = compute_episode_metrics(result).as_dict()
                duration = compute_duration_metrics(result).as_dict()
                metrics.update(duration)
                by_controller[ci].append(
                    torch.stack([metrics[name] for name in METRIC_NAMES], dim=-1).to(torch.float64)
                )
                by_locality[ci].append(
                    torch.stack([duration[name] for name in LOCALITY_METRIC_NAMES], dim=-1).to(
                        torch.float64
                    )
                )
            print(
                f"{severity}: completed seed {EVALUATION_SEEDS[seed_index]} ({seed_index + 1}/10)",
                flush=True,
            )
        for ci in range(3):
            controller_rows[ci].append(torch.stack(by_controller[ci]))
            locality_rows[ci].append(torch.stack(by_locality[ci]))
        scenario_rows.append(torch.stack(severity_scenarios))
    cube = torch.stack([torch.stack(rows) for rows in controller_rows])
    locality_cube = torch.stack([torch.stack(rows) for rows in locality_rows])
    input_cube = torch.stack(scenario_rows).to(torch.float64)
    overall = cube.mean(dim=(1, 2, 3))
    shortage = cube.mean(dim=(2, 3))
    locality_means = locality_cube.mean(dim=(1, 2, 3))
    intervals = []
    for a, b in PAIR_INDICES:
        interval = stratified_paired_bootstrap(
            cube[a] - cube[b],
            metric_names=METRIC_NAMES,
            replicates=PROTOCOL["bootstrap_replicates"],
            seed=PROTOCOL["bootstrap_seed"],
        )
        intervals.append(interval.as_tensor())
    print("Controller comparisons and paired cluster bootstrap completed.", flush=True)
    training_seeds = (42, 43, 44)
    training = torch.stack(
        [
            scenario_from_base_draws(
                generate_scenario_base_draws(256, seed, device="cuda:0"), "moderate"
            ).demand
            for seed in training_seeds
        ]
    )
    model = fit_ridge_forecast(training, training_seeds=training_seeds, ridge_penalty=0.001)
    forecast = evaluate_forecast_model(
        model, torch.stack(evaluation_demand), evaluation_seeds=EVALUATION_SEEDS
    )
    stats = forecast.sufficient_statistics
    totals = stats.sum(dim=1)
    errors = torch.stack(
        (totals[..., 1] / totals[..., 0], torch.sqrt(totals[..., 2] / totals[..., 0])), dim=-1
    )
    tensors = {
        "cube": cube,
        "locality_cube": locality_cube,
        "inputs": input_cube,
        "means": overall,
        "shortage": shortage,
        "locality_means": locality_means,
        "intervals": torch.stack(intervals),
        "forecast_stats": stats,
        "forecast_errors": errors,
        "coefficients_mw": model.coefficients_mw,
        "feature_means_mw": model.feature_means_mw,
        "feature_scales_mw": model.feature_scales_mw,
        "target_means_mw": model.target_means_mw,
    }
    # All raw scientific arrays, estimates, inference and intervals stay on GPU
    # until this single explicit bulk transfer for portable artifact assembly.
    packed = torch.cat([tensor.reshape(-1).to(torch.float64) for tensor in tensors.values()])
    host = packed.cpu().numpy()
    blocks = {}
    offset = 0
    for name, tensor in tensors.items():
        blocks[name] = host[offset : offset + tensor.numel()].reshape(tuple(tensor.shape))
        offset += tensor.numel()
    hashes = [
        [
            [
                hashlib.sha256(blocks["inputs"][si, zi, ei].astype("<f8").tobytes()).hexdigest()
                for ei in range(BATCH_SIZE)
            ]
            for zi in range(10)
        ]
        for si in range(3)
    ]
    controller_means = {
        name: dict(zip(METRIC_NAMES, blocks["means"][ci].tolist(), strict=True))
        for ci, name in enumerate(CONTROLLER_NAMES)
    }
    shortage_means = {
        level: {
            name: dict(zip(METRIC_NAMES, blocks["shortage"][ci, si].tolist(), strict=True))
            for ci, name in enumerate(CONTROLLER_NAMES)
        }
        for si, level in enumerate(SHORTAGE_LEVELS)
    }
    paired = {}
    for pi, (a, b) in enumerate(PAIR_INDICES):
        paired[f"{CONTROLLER_NAMES[a]} - {CONTROLLER_NAMES[b]}"] = {
            name: dict(
                zip(
                    ("mean_delta", "ci95_lower", "ci95_upper"),
                    blocks["intervals"][pi, :, mi].tolist(),
                    strict=True,
                )
            )
            for mi, name in enumerate(METRIC_NAMES)
        }
    forecast_methods = {
        method: {
            aggregation: dict(
                zip(("mae_mw", "rmse_mw"), blocks["forecast_errors"][mi, ai].tolist(), strict=True)
            )
            for ai, aggregation in enumerate(("locality_sector", "system_total"))
        }
        for mi, method in enumerate(FORECAST_NAMES)
    }
    payload = {
        "schema": {"name": "gridguard.proposal_supplement", "version": 1},
        "artifact_id": "gridguard-proposal-supplement-v1",
        "protocol": PROTOCOL,
        "controller_names": list(CONTROLLER_NAMES),
        "metric_names": list(METRIC_NAMES),
        "locality_metric_names": list(LOCALITY_METRIC_NAMES),
        "episode_metric_cube": blocks["cube"].tolist(),
        "locality_duration_cube": blocks["locality_cube"].tolist(),
        "controller_means": controller_means,
        "shortage_means": shortage_means,
        "locality_duration_means": {
            name: blocks["locality_means"][ci].tolist() for ci, name in enumerate(CONTROLLER_NAMES)
        },
        "paired_inference": paired,
        "scenario_sha256_cube": hashes,
        "source_sha256": source_sha,
        "runtime": asdict(runtime),
        "forecast": {
            "methods": forecast_methods,
            "training_seeds": list(training_seeds),
            "training_trace_count": 768,
            "evaluation_trace_count": 320,
            "target_hours": list(range(3, 24)),
            "ridge_penalty": 0.001,
            "sufficient_statistics": blocks["forecast_stats"].tolist(),
            "model": {
                name: blocks[name].tolist()
                for name in (
                    "coefficients_mw",
                    "feature_means_mw",
                    "feature_scales_mw",
                    "target_means_mw",
                )
            },
        },
        "method_definitions": {
            "Optimization": (
                "Exact enumeration of all 2187 joint service-tier actions to maximize canonical "
                "one-hour reward using only current demand/supply and prior cumulative state; "
                "deterministic lowest action ID breaks ties. It is not a 24-hour optimum."
            ),
            "blackout": (
                "One modeled hour with positive locality demand above 1e-8 MW and exactly zero "
                "actual delivered power; partial service is separate."
            ),
            "critical_interruption": (
                "One modeled hour with positive critical demand above 1e-8 MW and critical "
                "served/demand below 0.99."
            ),
            "duration_gap": (
                "Within each episode: mean absolute duration difference over the 21 distinct "
                "unordered locality pairs; then equal-episode averaging."
            ),
            "longest_interruption": (
                "Maximum consecutive modeled interrupted hours per locality; episode boundaries "
                "do not wrap; aggregate is the within-episode maximum across localities."
            ),
            "ridge_ar3": (
                "35 independent standardized ridge autoregressions using only t-3,t-2,t-1 demand, "
                "unpenalized intercept, fixed penalty 0.001 and non-negative predictions. "
                "New fit, separate from PPO and display forecast."
            ),
        },
        "limitations": [
            "Synthetic shortage-only simulation; no claim of Sudan utility field effectiveness "
            "or operational grid safety.",
            "Optimization is an exact one-step objective comparator, not a global episode optimum; "
            "it observes the same causal physical state, while PPO consumes its fixed 32-value "
            "observation.",
            "No behavior-cloning-only ablation isolates the incremental effect of PPO fine-tuning.",
            "Zero duration disparity can coexist with uniformly bad service; read duration "
            "inequality alongside total outage burden and critical service.",
            "Intervals are descriptive, pointwise 95% and unadjusted for multiple comparisons; "
            "seed separation does not imply independent custody or preregistration.",
            "Forecasts remain separate from controller decisions; predictive performance is "
            "assessed only on synthetic traces.",
            "The original accepted reproduction and 81001–81010 holdout remain unchanged; "
            "this is additional 91001–91010 evidence.",
        ],
    }
    # Guard the declared implementation/protocol against edits during evaluation.
    for path, digest in source_sha.items():
        if hashlib.sha256((root / path).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"source changed during evaluation: {path}")
    validate_proposal_supplement(payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/evidence/proposal-supplement.json"
    )
    args = parser.parse_args()
    payload = evaluate()
    encoded = canonical_proposal_bytes(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)
    print(f"Wrote {args.output}; SHA-256 {hashlib.sha256(encoded).hexdigest()}", flush=True)


if __name__ == "__main__":
    main()

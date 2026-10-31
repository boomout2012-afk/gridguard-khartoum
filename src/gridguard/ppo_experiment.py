"""Frozen settings and portable provenance for the additive constrained PPO study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from gridguard.allocation_study import METRIC_NAMES, SHORTAGES, canonical_bytes
from gridguard.duration_metrics import compute_duration_metrics
from gridguard.metrics import compute_episode_metrics
from gridguard.scenarios import generate_scenario_base_draws, scenario_from_base_draws
from gridguard.simulation import simulate

TRAINING_SEEDS = (1729, 2718, 3141, 5772, 8111)
VALIDATION_SEEDS = tuple(range(112101, 112106))
TEST_SEEDS = tuple(range(112201, 112211))
CONFIG = {
    "iterations": 600,
    "episodes_per_iteration": 256,
    "checkpoint_interval": 50,
    "bc_episodes_per_severity": 455,
    "bc_epochs": 10,
    "bc_minibatch_size": 1024,
    "bc_learning_rate": 0.0003,
    "bc_value_coefficient": 0.5,
    "training_seeds": list(TRAINING_SEEDS),
    "validation_seeds": list(VALIDATION_SEEDS),
    "test_seeds": list(TEST_SEEDS),
    "evaluation_batch_size": 32,
    "fairness_reward_tolerance": 0.005,
    "bootstrap_replicates": 10000,
    "bootstrap_seed": 20261023,
    "training_scenario_seed": "2000000 + run_index * 10000 + iteration_index",
    "bc_scenario_seed": "3000000 + run_index * 10000",
    "training_severity_batches": "85 each; extra episode rotates iteration_index % 3",
}
SOURCE_PATHS = (
    "src/gridguard/domain.py",
    "src/gridguard/controllers.py",
    "src/gridguard/scenarios.py",
    "src/gridguard/simulation.py",
    "src/gridguard/metrics.py",
    "src/gridguard/duration_metrics.py",
    "src/gridguard/statistics.py",
    "src/gridguard/optimization.py",
    "src/gridguard/constrained.py",
    "src/gridguard/allocation_study.py",
    "src/gridguard/constrained_ppo.py",
    "src/gridguard/ppo_training.py",
    "src/gridguard/ppo_study.py",
    "src/gridguard/ppo_experiment.py",
    "tools/train_constrained_ppo.py",
    "tools/evaluate_constrained_ppo.py",
    "docs/CONSTRAINED_PPO_PROTOCOL.md",
    "data/evidence/allocation-validation.json",
)


def training_batches(run_index: int, iteration_index: int) -> tuple[int, tuple[int, ...]]:
    if not 0 <= run_index < len(TRAINING_SEEDS) or not 0 <= iteration_index < 600:
        raise ValueError("training indices outside frozen protocol")
    return (
        2_000_000 + run_index * 10_000 + iteration_index,
        tuple(85 + int(level == iteration_index % 3) for level in range(3)),
    )


def scenarios_for_seed(seed: int, batch_sizes: tuple[int, ...]):
    """Return separate, correctly labeled severity scenarios on CUDA."""
    return tuple(
        scenario_from_base_draws(generate_scenario_base_draws(batch, seed, device="cuda:0"), level)
        for level, batch in zip(SHORTAGES, batch_sizes, strict=True)
    )


def evaluation_scenarios(seeds: tuple[int, ...]):
    draws = [generate_scenario_base_draws(32, seed, device="cuda:0") for seed in seeds]
    return tuple(
        tuple(scenario_from_base_draws(draw, level) for draw in draws) for level in SHORTAGES
    )


@torch.inference_mode()
def evaluate_metrics(controller, scenarios) -> torch.Tensor:
    rows = []
    for severity_scenarios in scenarios:
        batches = []
        for scenario in severity_scenarios:
            guard = scenario.capture_mutation_guard()
            result = simulate(scenario, controller)
            if guard.changed_fields(scenario):
                raise ValueError("controller mutated matched scenario")
            metrics = compute_episode_metrics(result).as_dict()
            metrics.update(compute_duration_metrics(result).as_dict())
            batches.append(torch.stack([metrics[name] for name in METRIC_NAMES], dim=-1))
        rows.append(torch.stack(batches))
    return torch.stack(rows).to(torch.float64)


def summarize(cube: torch.Tensor) -> dict:
    """Export only small checkpoint selection summaries; metrics remain GPU-computed."""
    overall = cube.mean(dim=(0, 1, 2))
    by_shortage = cube.mean(dim=(1, 2))
    maxima = cube.amax(dim=(0, 1, 2))
    values = torch.cat((overall[None], by_shortage, maxima[None])).cpu().tolist()
    return {
        "overall": dict(zip(METRIC_NAMES, values[0], strict=True)),
        "by_shortage": {
            level: dict(zip(METRIC_NAMES, values[index + 1], strict=True))
            for index, level in enumerate(SHORTAGES)
        },
        "maxima": dict(zip(METRIC_NAMES, values[-1], strict=True)),
    }


def source_hashes(root: Path) -> dict[str, str]:
    return {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in SOURCE_PATHS}


def write_json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_bytes(payload)
    digest = hashlib.sha256(data).hexdigest()
    with path.open("xb") as handle:
        handle.write(data)
    path.with_suffix(".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def read_json(path: Path, expected_sha256: str | None = None) -> dict:
    data = path.read_bytes()
    if expected_sha256 is None:
        expected_sha256 = path.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("evidence SHA-256 mismatch")
    return json.loads(data)

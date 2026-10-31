"""Additive long-budget PPO protocol and selection against Fixed Rotation.

Existing scientific implementations and frozen studies remain unchanged. These
settings declare a new training experiment with fresh scenario streams.
"""

from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path

from gridguard.allocation_study import SHORTAGES
from gridguard.ppo_experiment import evaluate_metrics as evaluate_metrics
from gridguard.ppo_experiment import evaluation_scenarios as evaluation_scenarios
from gridguard.ppo_experiment import read_json as read_json
from gridguard.ppo_experiment import scenarios_for_seed as scenarios_for_seed
from gridguard.ppo_experiment import summarize as summarize
from gridguard.ppo_experiment import write_json as write_json
from gridguard.ppo_study import SAFETY_METRICS, _validate_summary

TRAINING_SEEDS = (1729, 2718, 3141, 5772, 8111)
VALIDATION_SEEDS = tuple(range(122101, 122106))
TEST_SEEDS = tuple(range(122201, 122211))
CONFIG = {
    "iterations": 3000,
    "episodes_per_iteration": 256,
    "checkpoint_interval": 100,
    "telemetry_interval": 25,
    "budget_reference_iteration": 600,
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
    "bootstrap_seed": 20261024,
    "training_scenario_seed": "4000000 + run_index * 10000 + iteration_index",
    "bc_scenario_seed": "4500000 + run_index * 10000",
    "training_severity_batches": "85 each; extra episode rotates iteration_index % 3",
    "selection_comparator": "Fixed Rotation",
    "primary_comparators": ["Fixed Rotation", "Constrained Optimization"],
    "selection": "highest validation reward among zero-critical/power and "
    "fairness/utilization >= Fixed Rotation overall and in every shortage; earliest iteration tie",
    "budget_reference": "same-run iteration600 under the 3000-iteration learning-rate schedule",
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
    "src/gridguard/focused_training.py",
    "tools/train_focused_ppo.py",
    "tools/evaluate_focused_ppo.py",
    "docs/FOCUSED_PPO_PROTOCOL.md",
    "data/evidence/allocation-validation.json",
)


def training_batches(run_index: int, iteration_index: int) -> tuple[int, tuple[int, ...]]:
    """Independent new streams with exactly balanced severity exposure over the run."""
    if (
        type(run_index) is not int
        or type(iteration_index) is not int
        or not 0 <= run_index < len(TRAINING_SEEDS)
        or not 0 <= iteration_index < CONFIG["iterations"]
    ):
        raise ValueError("training indices outside frozen focused protocol")
    return (
        4_000_000 + run_index * 10_000 + iteration_index,
        tuple(85 + int(level == iteration_index % 3) for level in range(3)),
    )


def source_hashes(root: Path) -> dict[str, str]:
    return {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in SOURCE_PATHS}


def checkpoint_eligibility(candidate: dict, reference: dict) -> dict:
    """Require safe PPO and service at least as good as the Fixed Rotation baseline.

    Fixed Rotation itself is allowed to have interruptions. No Fairness Rule
    eligibility threshold is imported from the preceding experiment.
    """
    _validate_summary(candidate, "candidate")
    _validate_summary(reference, "Fixed Rotation reference")
    reasons = []
    for metric in SAFETY_METRICS:
        if candidate["maxima"][metric] != 0:
            reasons.append(f"candidate maximum {metric} is nonzero")
    pairs = [("overall", candidate["overall"], reference["overall"])] + [
        (level, candidate["by_shortage"][level], reference["by_shortage"][level])
        for level in SHORTAGES
    ]
    for level, row, baseline in pairs:
        for metric in ("fairness", "utilization"):
            if row[metric] < baseline[metric]:
                reasons.append(f"{level} {metric} is below Fixed Rotation")
    return {"eligible": not reasons, "reasons": reasons}


def select_checkpoint(records: list[dict], reference: dict) -> dict:
    """Select the eligible reward maximum or label an ineligible reward fallback."""
    if not isinstance(records, list) or not records:
        raise ValueError("checkpoint selection requires a nonempty record list")
    _validate_summary(reference, "Fixed Rotation reference")
    iterations = set()
    evaluated = []
    for record in records:
        if not isinstance(record, dict) or "summary" not in record:
            raise ValueError("checkpoint records require iteration and summary")
        iteration = record.get("iteration")
        if type(iteration) is not int or iteration < 0:
            raise ValueError("checkpoint iteration must be a nonnegative integer")
        if iteration in iterations:
            raise ValueError("checkpoint iterations must be unique")
        iterations.add(iteration)
        candidate = deepcopy(record)
        candidate.update(checkpoint_eligibility(candidate["summary"], reference))
        evaluated.append(candidate)
    eligible = [record for record in evaluated if record["eligible"]]
    selected = max(
        eligible or evaluated,
        key=lambda record: (record["summary"]["overall"]["episode_reward"], -record["iteration"]),
    )
    selected["fallback"] = not bool(eligible)
    return selected


def verify_checkpoint_selection(records: list[dict], reference: dict, selected: dict) -> None:
    if selected != select_checkpoint(records, reference):
        raise ValueError("focused checkpoint selection does not reconstruct")

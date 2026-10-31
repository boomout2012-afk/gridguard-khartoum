"""Thesis prose derived from the separate frozen fresh-scenario learning audit.

This presentation helper does not alter the audit, run simulation or recompute
scientific summaries. The caller verifies the artifact and its validation receipt
before publication; this function validates the schema, design and displayed values.
"""

from __future__ import annotations

import math
from typing import Any

SCHEMA = "gridguard.thesis_learning_audit.1"
TRAINING_SEEDS = (1729, 2718, 3141, 5772, 8111)
SHORTAGES = ("mild", "moderate", "severe")
FIXED_METHODS = (
    "Fixed Rotation",
    "Critical-aware Fixed Rotation",
    "Constrained Optimization",
    "Fairness Rule",
)
DISPLAY_METRICS = (
    "episode_reward",
    "fairness",
    "utilization",
    "served_mwh",
    "switching_rate",
    "final_minimum_locality_service_ratio",
)


def _number(value: Any) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("learning-audit prose requires finite numeric evidence")
    return float(value)


def _interval(row: dict, *, scale: float = 1, digits: int = 4) -> str:
    values = [_number(row[key]) for key in ("mean", "lower", "upper")]
    if values[1] > values[2]:
        raise ValueError("learning-audit interval endpoints are reversed")
    mean, lower, upper = (value * scale for value in values)
    return f"{mean:+.{digits}f} [{lower:+.{digits}f}, {upper:+.{digits}f}]"


def _validate(audit: dict) -> None:
    if not isinstance(audit, dict) or audit.get("schema") != SCHEMA:
        raise ValueError("learning-audit prose schema mismatch")
    counts = {
        "independent_base_draws": 320,
        "matched_scenario_count": 960,
        "controller_trajectory_count": 13440,
        "scenario_hash_overlap": 0,
    }
    if any(type(audit.get(key)) is not int or audit[key] != count for key, count in counts.items()):
        raise ValueError("learning-audit prose count contract mismatch")
    names = [
        *FIXED_METHODS,
        *(f"Behavior Cloning seed {seed}" for seed in TRAINING_SEEDS),
        *(f"PPO seed {seed}" for seed in TRAINING_SEEDS),
    ]
    if audit.get("controller_names") != names:
        raise ValueError("learning-audit prose method identities/order mismatch")
    recipe = audit["recipe"]
    expected = {
        "test_seeds": list(range(132201, 132211)),
        "training_seeds": list(TRAINING_SEEDS),
        "batch_size": 32,
        "shortages": list(SHORTAGES),
        "bootstrap_replicates": 10000,
        "bootstrap_seed": 20261031,
        "interval_confidence": 0.95,
        "post_hoc_motivation": True,
        "fresh_scenario_execution": True,
    }
    if any(recipe.get(key) != value for key, value in expected.items()):
        raise ValueError("learning-audit prose recipe mismatch")
    previous_hashes = audit.get("previous_scenario_hashes_checked")
    if type(previous_hashes) is not int or previous_hashes < 1:
        raise ValueError("learning-audit prose requires prior scenario-hash accounting")
    for row in (
        *(audit["controller_means"][name] for name in FIXED_METHODS),
        *(audit["procedure_means"][name] for name in ("Behavior Cloning", "PPO")),
    ):
        for metric in DISPLAY_METRICS:
            _number(row[metric])
    for comparison in ("PPO - own Behavior Cloning", *(f"PPO - {name}" for name in FIXED_METHODS)):
        for metric in DISPLAY_METRICS:
            _interval(audit["paired_inference"][comparison][metric])
    for name in names[1:]:
        for metric in (
            "critical_interruption_locality_hours",
            "complete_critical_outage_locality_hours",
            "power_balance_violation_rate",
        ):
            if _number(audit["maxima"][name][metric]) != 0:
                raise ValueError("zero-critical/power narrative is unsupported by audit maxima")


def learning_stage_text(audit: dict) -> str:
    """Return Section 5.10 and Tables 17–18 from verified portable audit summaries.

    The caller must also verify ``thesis-learning-audit.validation.json`` before
    publishing the separate full-replay statement. That receipt is not an input
    to this pure presentation function and is not inferred from a positive result.
    """
    _validate(audit)
    recipe = audit["recipe"]
    stage_name = "PPO - own Behavior Cloning"
    stage = audit["paired_inference"][stage_name]
    teacher = audit["paired_inference"]["PPO - Fairness Rule"]
    rotation = audit["paired_inference"]["PPO - Critical-aware Fixed Rotation"]
    per_run = audit["per_run_inference"][stage_name]
    if set(per_run) != {str(seed) for seed in TRAINING_SEEDS}:
        raise ValueError("learning-audit prose requires all five paired runs")
    severity = audit["shortage_paired_inference"][stage_name]
    if set(severity) != set(SHORTAGES):
        raise ValueError("learning-audit prose requires all three shortage summaries")
    positive_runs = sum(_number(row["episode_reward"]["lower"]) > 0 for row in per_run.values())
    positive_severities = sum(
        _number(row["episode_reward"]["lower"]) > 0 for row in severity.values()
    )
    energy_tradeoffs = sum(
        _number(row["utilization"]["upper"]) < 0 and _number(row["served_mwh"]["upper"]) < 0
        for row in severity.values()
    )
    if (positive_runs, positive_severities, energy_tradeoffs) != (5, 3, 3):
        raise ValueError("all-run/all-severity tradeoff narrative is unsupported")
    if _number(teacher["episode_reward"]["mean"]) >= 0.01:
        raise ValueError("below-0.01 teacher-reference narrative is unsupported")
    for comparison in (teacher, rotation):
        if _number(comparison["final_minimum_locality_service_ratio"]["upper"]) >= 0:
            raise ValueError("adverse final-minimum narrative is unsupported")

    table17 = [
        "**Table 17. Fresh matched effects of selected PPO minus its own behavior-cloning "
        "initialization; mean differences and descriptive pointwise 95% intervals.**",
        "",
        "| Outcome | Difference [95% interval] |",
        "| --- | --- |",
    ]
    for metric, label, scale, digits in (
        ("episode_reward", "Daily reward", 1, 4),
        ("fairness", "Jain fairness (pp)", 100, 3),
        ("utilization", "Utilization (pp)", 100, 3),
        ("served_mwh", "Served energy (MWh/day)", 1, 2),
        ("switching_rate", "Switching rate (pp)", 100, 3),
        ("final_minimum_locality_service_ratio", "Final minimum locality service (pp)", 100, 3),
    ):
        table17.append(f"| {label} | {_interval(stage[metric], scale=scale, digits=digits)} |")

    table18 = [
        "**Table 18. Six procedures on the 960 fresh matched conditions; BC and PPO "
        "weight the five runs equally.**",
        "",
        "| Procedure | Reward | Jain (%) | Util. (%) | Served (MWh) | Final min. (%) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    procedures = [
        *((name, audit["controller_means"][name]) for name in FIXED_METHODS),
        ("Behavior Cloning (BC)", audit["procedure_means"]["Behavior Cloning"]),
        ("PPO", audit["procedure_means"]["PPO"]),
    ]
    for name, row in procedures:
        table18.append(
            f"| {name} | {row['episode_reward']:.4f} | {100 * row['fairness']:.2f} "
            f"| {100 * row['utilization']:.2f} | {row['served_mwh']:.2f} "
            f"| {100 * row['final_minimum_locality_service_ratio']:.2f} |"
        )

    severe_switching = severity["severe"]["switching_rate"]
    if not _number(severe_switching["lower"]) <= 0 <= _number(severe_switching["upper"]):
        raise ValueError("Severe switching-uncertainty narrative is unsupported")
    teacher_severity = audit["shortage_paired_inference"]["PPO - Fairness Rule"]
    if not (
        _number(teacher_severity["mild"]["episode_reward"]["upper"]) < 0
        and _number(teacher_severity["moderate"]["episode_reward"]["upper"]) < 0
        and _number(teacher_severity["severe"]["episode_reward"]["lower"]) > 0
    ):
        raise ValueError("teacher shortage-pattern narrative is unsupported")

    bc_reward = audit["procedure_means"]["Behavior Cloning"]["episode_reward"]
    teacher_reward = audit["controller_means"]["Fairness Rule"]["episode_reward"]
    sections = [
        "## 5.10 Fresh audit of the learning stage",
        "The teacher comparison does not measure improvement over the imperfect cloned actor. "
        "A separate follow-up therefore compares each selected PPO actor with its own retained "
        "behavior-cloning (BC) initialization. The protocol was frozen before outcome generation, "
        "followed by an implementation-and-model freeze. The original findings were already known, "
        "so the motivation is post-hoc; fresh execution does not retroactively preregister the "
        "earlier studies. No policy was retrained, tuned or selected again.",
        f"Seeds {recipe['test_seeds'][0]}–{recipe['test_seeds'][-1]} "
        "passed checks against retained "
        "training, demonstration, validation and evaluation metadata. Ten seeds provide 32 base "
        "draws each, giving 320 independent draws and 960 coupled Mild, Moderate and Severe "
        "conditions. None of their input hashes matches the "
        f"{audit['previous_scenario_hashes_checked']:,} "
        "previous scenario hashes checked. Four fixed comparators, five BC actors and five "
        "selected PPO actors produce 13,440 trajectories on identical matched inputs.",
        "The central contrast pairs actors within the five training runs. The hierarchical "
        f"bootstrap uses {recipe['bootstrap_replicates']:,} resamples with seed "
        f"{recipe['bootstrap_seed']}, resampling paired runs and base-draw clusters within ten "
        "seed strata. Sampled scenarios are shared across runs and methods, retaining all three "
        "coupled shortage variants. Means weight runs and conditions equally. The intervals are "
        "descriptive, pointwise and nominally 95%; they do not establish simultaneous superiority "
        "across outcomes or comparators. Table 17 reports the learning-stage effects.",
        "\n".join(table17),
        "All five run-level reward intervals and all three shortage-level reward intervals are "
        "positive. Nevertheless, utilization and served energy decrease in every shortage, with "
        "their intervals entirely below zero. Switching decreases overall, while its Severe-only "
        "interval crosses zero. Thus fine-tuning and selection improve the specified objective "
        "and several distribution measures relative to BC, with an energy cost. This contrast "
        "includes extra computation and validation checkpoint selection; it does not isolate the "
        "PPO algorithm, compare equal training budgets or establish that learning from scratch "
        "is necessary.",
        "\n".join(table18),
        "Final minimum service in Table 18 is the lowest of seven end-of-day locality service "
        "ratios within each condition, averaged afterward. It is distinct from time-averaged "
        "Jain fairness and from the minimum of averaged locality ratios. **PPO's higher Jain "
        "fairness coexists with worse final service for the least-served locality than the "
        "teacher or Critical-aware Fixed Rotation.** The respective final-minimum effects are "
        f"{_interval(teacher['final_minimum_locality_service_ratio'], scale=100, digits=3)} and "
        f"{_interval(rotation['final_minimum_locality_service_ratio'], scale=100, digits=3)} "
        "percentage points. These adverse endpoints prevent a blanket equity claim.",
        "Against Fairness Rule, the fresh reward difference is "
        f"{_interval(teacher['episode_reward'])}, with a mean below the earlier +0.01 reference. "
        f"Served energy changes by {_interval(teacher['served_mwh'], digits=2)} MWh/day. "
        "Mild and Moderate reward effects favor the teacher; Severe favors PPO. The reference "
        "is context, not a new acceptance gate. The four fixed-comparator contrasts are "
        "replication diagnostics and replace none of the original primary decisions.",
        "The size of the learning-stage gain also reflects its starting point. "
        f"Behavior cloning scores {bc_reward:.4f}, "
        f"compared with {teacher_reward:.4f} "
        "for the teacher. The resulting deficit is "
        f"{teacher_reward - bc_reward:.4f} "
        "daily reward, while selected PPO ends only "
        f"{teacher['episode_reward']['mean']:.4f} above the teacher. Most of the "
        f"{stage['episode_reward']['mean']:.4f} stage gain therefore closes the imitation "
        "gap arithmetically. This does not identify how much is caused by PPO rather "
        "than extra computation or checkpoint selection. Further imitation under "
        "an equal budget is an untested alternative, not an established substitute.",
        "All protected methods, including BC, have zero critical interruption below 99% service, "
        "zero complete critical outage and zero power violations in every fresh condition. "
        "A separate validation record confirms exact replay of all 13,440 trajectories and "
        "reconstruction of means and intervals; Appendix A.4 identifies those records. "
        "These fresh seeds test the same synthetic "
        "profiles and perturbations, not changed demand mechanisms or real-grid operation. "
        "Five training runs still limit uncertainty estimation; "
        "the original evidence remains unchanged.",
    ]
    return "\n\n".join(sections) + "\n"

"""Fill thesis result sections from explicitly pinned, previously verified study artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAINING = "data/evidence/focused-ppo-training.json"
STUDY = "data/evidence/focused-ppo-study.json"
DIAGNOSTICS = "data/evidence/focused-ppo-training-diagnostics.json"
SUPPLEMENT = "data/evidence/proposal-supplement.json"
SUPPLEMENT_HASH = "fae5c875b1be148d7a069259b19df9b04666fe97e2182ce8898f45fed7b0a43e"
FIXED = "PPO - Fixed Rotation"
OPT = "PPO - Constrained Optimization"
BUDGET = "PPO - PPO at600"


def load(relative, expected):
    raw = (ROOT / relative).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"Hash mismatch: {relative}")
    return json.loads(raw)


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows],
        ]
    )


def ci_text(interval, scale=1, digits=3):
    mean, low, high = [interval[key] * scale for key in ("mean", "lower", "upper")]
    return f"{mean:+.{digits}f} [{low:+.{digits}f}, {high:+.{digits}f}]"


def main(training_sha, study_sha):
    training = load(TRAINING, training_sha)
    study = load(STUDY, study_sha)
    diagnostics_sha = (ROOT / DIAGNOSTICS).with_suffix(".sha256").read_text().split()[0]
    diagnostics = load(DIAGNOSTICS, diagnostics_sha)
    if diagnostics["training_sha256"] != training_sha:
        raise ValueError("Training diagnostics do not use the pinned training record")
    load(SUPPLEMENT, SUPPLEMENT_HASH)
    if study["training_sha256"] != training_sha:
        raise ValueError("Study does not use the pinned training record")
    if study["matched_scenario_count"] != 960 or study["controller_trajectory_count"] != 11520:
        raise ValueError("Unexpected matched study counts")
    for path, expected in training["source_sha256"].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Scientific source changed: {path}")
    fixed = study["controller_means"]["Fixed Rotation"]
    opt = study["controller_means"]["Constrained Optimization"]
    ppo = study["procedure_means"]["PPO"]
    short = study["procedure_means"]["PPO at600"]
    rows = (fixed, opt, ppo)
    inference = study["primary_family_adjusted_inference"]
    intervals = study["hierarchical_inference"]
    claims = study["claims"]
    runs = training["runs"]
    seconds = sum(run["seconds"] for run in runs)
    selected = [run["selected"]["iteration"] for run in runs]
    main_metrics = [
        ("Critical interruption (locality-hours)", "critical_interruption_locality_hours", 1, 2),
        ("Jain service fairness (%)", "fairness", 100, 2),
        ("Supply utilization (%)", "utilization", 100, 2),
        ("Daily reward", "episode_reward", 1, 4),
        ("Switching rate (%)", "switching_rate", 100, 2),
        ("Critical service (%)", "critical_service_ratio", 100, 2),
        ("Zero-service blackout (locality-hours)", "blackout_locality_hours", 1, 2),
        ("Power-balance violation rate", "power_balance_violation_rate", 1, 4),
    ]
    results_table = table(
        ["Outcome", "Fixed Rotation", "Constrained optimizer", "PPO"],
        [
            [label, *[f"{row[metric] * scale:.{digits}f}" for row in rows]]
            for label, metric, scale, digits in main_metrics
        ],
    )
    durations = (
        ("Blackout duration range (hours)", "blackout_duration_range_hours"),
        ("Blackout mean pairwise gap (hours)", "blackout_duration_mean_pairwise_gap_hours"),
        ("Longest blackout (hours)", "longest_blackout_hours"),
        (
            "Critical-interruption duration range (hours)",
            "critical_interruption_duration_range_hours",
        ),
        (
            "Critical-interruption mean pairwise gap (hours)",
            "critical_interruption_duration_mean_pairwise_gap_hours",
        ),
        ("Longest critical interruption (hours)", "longest_critical_interruption_hours"),
    )
    results_table += "\n\n### Direct interruption-duration outcomes\n\n" + table(
        ["Duration outcome", "Fixed Rotation", "Constrained optimizer", "PPO"],
        [[label, *[f"{row[metric]:.3f}" for row in rows]] for label, metric in durations],
    )
    results_table += (
        "\n\nRanges and pairwise gaps are calculated within each 24-hour scenario before "
        "averaging. A zero duration disparity under shared critical protection is not "
        "evidence of equal Full Service. "
        f"Mean available energy was {fixed['available_mwh']:.2f} MWh per condition. "
        f"Mean served energy was {fixed['served_mwh']:.2f} MWh for Fixed Rotation, "
        f"{opt['served_mwh']:.2f} MWh for the constrained optimizer and "
        f"{ppo['served_mwh']:.2f} MWh for PPO. "
        "These energy totals differ from the reported mean of hourly utilization ratios."
    )
    shortage_rows = []
    for level in study["shortages"]:
        values = [
            study["shortage_means"][level][name]
            for name in ("Fixed Rotation", "Constrained Optimization")
        ]
        values.append(study["procedure_shortage_means"][level]["PPO"])
        for name, value in zip(
            ("Fixed Rotation", "Constrained optimizer", "PPO"), values, strict=True
        ):
            shortage_rows.append(
                [
                    level.title(),
                    name,
                    f"{value['critical_interruption_locality_hours']:.2f}",
                    f"{100 * value['fairness']:.2f}",
                    f"{100 * value['utilization']:.2f}",
                    f"{value['episode_reward']:.4f}",
                ]
            )
    shortages = table(
        [
            "Shortage",
            "Controller",
            "Critical interruption (locality-hours)",
            "Fairness (%)",
            "Utilization (%)",
            "Reward",
        ],
        shortage_rows,
    )
    primary_parts = []
    effect_metrics = (
        ("Critical interruption (locality-hours)", "critical_interruption_locality_hours", 1),
        ("Jain service fairness (pp)", "fairness", 100),
        ("Supply utilization (pp)", "utilization", 100),
        ("Daily reward", "episode_reward", 1),
        ("Switching rate (pp)", "switching_rate", 100),
    )
    for pair in (FIXED, OPT):
        primary_parts.append("### " + pair)
        primary_parts.append(
            table(
                ["Outcome", "Mean difference", "97.5% lower", "97.5% upper"],
                [
                    [
                        label,
                        *[
                            f"{inference[pair][metric][key] * scale:+.4f}"
                            for key in ("mean", "lower", "upper")
                        ],
                    ]
                    for label, metric, scale in effect_metrics
                ],
            )
        )
    primary_intervals = "\n\n".join(primary_parts)

    budget_rows = []
    for run in runs:
        name, earlier = f"PPO seed {run['seed']}", f"PPO600 seed {run['seed']}"
        later_reward = study["controller_means"][name]["episode_reward"]
        earlier_reward = study["controller_means"][earlier]["episode_reward"]
        budget_rows.append(
            [
                run["seed"],
                run["selected"]["iteration"],
                f"{earlier_reward:.4f}",
                f"{later_reward:.4f}",
                f"{later_reward - earlier_reward:+.4f}",
            ]
        )
    budget_table = table(
        [
            "Training seed",
            "Selected iteration",
            "600-iteration reward",
            "Selected reward",
            "Difference",
        ],
        budget_rows,
    )
    budget_table += "\n\n" + table(
        ["Across-run outcome", "Mean difference and 95% interval"],
        [
            [label, ci_text(intervals[BUDGET][metric], scale, 4)]
            for label, metric, scale in effect_metrics[1:]
        ],
    )
    budget_pass = claims[BUDGET]["passed"]
    budget_table += "\n\n" + (
        "The predeclared additional-training criterion passed: the selected longer-run "
        "policies improved daily reward while meeting the service and safety guardrails."
        if budget_pass
        else "The full predeclared additional-training criterion did not pass. A larger budget "
        "alone therefore does not establish improvement on every declared requirement."
    )
    if not budget_pass:
        failed = [
            key.replace("_", " ") for key, value in claims[BUDGET]["gates"].items() if not value
        ]
        budget_table += " Unmet conditions: " + "; ".join(failed) + "."

    abstract = (
        f"Across 960 fresh matched conditions, PPO averaged "
        f"{ppo['critical_interruption_locality_hours']:.2f} critical-interruption locality-hours, "
        f"{100 * ppo['fairness']:.2f}% service fairness and "
        f"{100 * ppo['utilization']:.2f}% utilization. Fixed Rotation recorded "
        f"{fixed['critical_interruption_locality_hours']:.2f}, "
        f"{100 * fixed['fairness']:.2f}% and {100 * fixed['utilization']:.2f}%, respectively; "
        f"the constrained optimizer recorded {opt['critical_interruption_locality_hours']:.2f}, "
        f"{100 * opt['fairness']:.2f}% and {100 * opt['utilization']:.2f}%. "
        f"PPO's daily-reward difference from the optimizer was "
        f"{ci_text(inference[OPT]['episode_reward'], digits=4)} "
        "(97.5% interval). "
    )
    abstract += (
        "The declared optimizer-comparison criterion was met."
        if claims[OPT]["passed"]
        else "The declared optimizer-comparison criterion was not met; individual advantages "
        "must be interpreted alongside the remaining tradeoffs."
    )

    interpretation = []
    interpretation.append(
        "The complete constrained PPO system "
        + ("met" if claims[FIXED]["passed"] else "did not meet")
        + " the predeclared improvement criterion against Fixed Rotation. This comparison "
        "includes differences in action choices and critical-service constraints, so it "
        "does not isolate the neural policy as the sole cause of the observed difference."
    )
    opt_reward = inference[OPT]["episode_reward"]
    interpretation.append(
        f"Against the constrained optimizer, the mean daily-reward difference was "
        f"{ci_text(opt_reward, digits=4)}. The full comparison criterion "
        f"{'passed' if claims[OPT]['passed'] else 'did not pass'}. "
        "Both controllers use the same critical-feasible action choices. The optimizer "
        "maximizes the current-hour objective exactly; PPO can trade a current decision "
        "against its consequences for later cumulative fairness and switching. Any "
        "whole-day advantage therefore concerns this causal myopic comparator and the "
        "specified reward, rather than a superior solution to the same exact optimization problem."
    )
    fair_ci = inference[OPT]["fairness"]
    util_ci = inference[OPT]["utilization"]
    interpretation.append(
        f"The corresponding fairness effect was {ci_text(fair_ci, 100)} percentage points, "
        f"and utilization changed by {ci_text(util_ci, 100)} percentage points. "
        "These outcomes distinguish service and efficiency effects from the weighted "
        "reward score."
    )
    interpretation.append(
        f"Switching increased by {ci_text(inference[OPT]['switching_rate'], 100)} percentage "
        "points, an unfavorable change. The fairness contribution outweighed both the "
        "utilization loss and extra switching penalty in the specified reward. Thus the "
        "observed AI advantage is fairer service allocation and a higher daily objective, "
        "with an efficiency and switching tradeoff."
    )
    practical = study["optimizer_practical_reward_assessment"]
    interpretation.append(
        "The optimizer reward difference "
        + (
            "also exceeded"
            if practical["adjusted_ci_lower_exceeds_threshold"]
            else "did not establish improvement beyond"
        )
        + " the earlier +0.01 daily-reward reference throughout its adjusted interval. "
        "That reference is a research convention; this simulation does not establish "
        "the operational value of a small score difference."
    )
    if not claims[OPT]["passed"]:
        failed = [key.replace("_", " ") for key, value in claims[OPT]["gates"].items() if not value]
        interpretation.append(
            "The unmet optimizer-comparison conditions were: " + "; ".join(failed) + "."
        )
    interpretation.append(
        "All three shortage levels are reported in Section 5.6. Severity-specific intervals are "
        "exploratory, so a favorable subgroup does not replace the overall declared result. "
        "Zero blackout-duration disparities among critical-protected controllers also "
        "do not imply equal access to Full Service."
    )

    eligible_count = sum(r["selected"]["eligible"] for r in runs)
    training_summary = (
        f"Five fresh training seeds completed 3,000 iterations each: 18,432,000 transitions "
        f"per run and 92,160,000 overall. Total recorded training time was {seconds / 60:.2f} "
        f"minutes; final evaluation took {study['elapsed_seconds']:.2f} seconds. "
        f"Validation selected iterations {', '.join(map(str, selected))} for seeds "
        f"{', '.join(str(run['seed']) for run in runs)}, respectively. "
        f"{eligible_count} of the five selections met "
        "the validation eligibility requirements. The final report uses the equal-run "
        "procedure mean rather than selecting the best performer on the final test."
    )
    training_summary += "\n\n" + table(
        ["Training seed", "Selected iteration", "Validation reward", "Training minutes"],
        [
            [
                run["seed"],
                run["selected"]["iteration"],
                f"{run['selected']['summary']['overall']['episode_reward']:.4f}",
                f"{run['seconds'] / 60:.2f}",
            ]
            for run in runs
        ],
    )
    replacements = {
        "ABSTRACT_RESULTS": abstract,
        "TRAINING_SUMMARY": training_summary,
        "RESULTS_TABLE": results_table,
        "RESULTS_BY_SHORTAGE": shortages,
        "PRIMARY_INTERVALS": primary_intervals,
        "BUDGET_COMPARISON_TABLE": budget_table,
        "RESULTS_INTERPRETATION": "\n\n".join(interpretation),
        "TRAINING_CURVE_FIGURE": "[[FIGURE:training]]",
        "RESULTS_FIGURE": "[[FIGURE:outcomes]]",
        "PRIMARY_CI_FIGURE": "[[FIGURE:intervals]]",
        "SCREENSHOT_OVERVIEW": "[[FIGURE:overview]]",
        "SCREENSHOT_LIVE": "[[FIGURE:live]]",
        "RESEARCH_WORKFLOW_FIGURE": "[[FIGURE:workflow]]",
        "LITERATURE_REVIEW_STATUS": "",
    }
    payload = {
        "evidence_sha256": {
            TRAINING: training_sha,
            STUDY: study_sha,
            SUPPLEMENT: SUPPLEMENT_HASH,
            DIAGNOSTICS: diagnostics_sha,
        },
        "replacements": replacements,
    }
    output = ROOT / "tmp/focused-thesis-content.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)
    print("PPO overall reward:", ppo["episode_reward"])
    print("Iteration-600 overall reward:", short["episode_reward"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-sha256", required=True)
    parser.add_argument("--study-sha256", required=True)
    args = parser.parse_args()
    main(args.training_sha256, args.study_sha256)

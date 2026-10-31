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
FOLLOWUP = "data/evidence/committee-baseline-audit.json"
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


def followup_text(audit):
    """Describe the separately verified post-hoc audit without changing primary claims."""
    if audit["schema"] != "gridguard.committee_baseline_audit.1":
        raise ValueError("Unexpected committee audit schema")
    if audit["nominal_matched_conditions"] != 960 or audit["nominal_trajectories"] != 8640:
        raise ValueError("Unexpected committee audit scenario counts")
    if not audit["recipe"]["post_hoc"]:
        raise ValueError("Committee audit must remain explicitly post-hoc")
    labels = [
        "Fixed Rotation",
        "Critical-aware Fixed Rotation",
        "Constrained Optimization",
        "Fairness Rule",
        "PPO",
    ]
    means = {**audit["controller_means"], **audit["procedure_means"]}
    parts = [
        "Fairness Rule retains its teacher setting: among feasible actions within 0.005 "
        "of the highest immediate reward, it maximizes the minimum projected cumulative "
        "service ratio, then resolves ties by reward and candidate index. Critical-aware "
        "Fixed Rotation first requests the critical floor everywhere, then tries Full "
        "Service upgrades in fixed rotating locality order, retaining each upgrade only "
        "when it fits supply. The order advances by three localities every three hours. "
        "It uses no learned model and does not optimize reward. This comparator shares "
        "the critical floor, not the exact discretionary decisions of the original "
        "Off/Full Service schedule.",
        "The follow-up evaluates nine methods: four deterministic rules/optimizers and "
        "the five frozen PPO actors, giving 8,640 nominal trajectories. It regenerates "
        "seeds 122201–122210 with 32 base draws and three coupled shortage levels. "
        "Input hashes, action hashes and per-condition metrics reproduce the seven "
        "unchanged methods from the original experiment exactly. A separate frozen "
        "audit recipe precedes this execution; it is still post-hoc relative to the "
        "original results. No new primary pass/fail claim is registered.",
        "Table 13 compares the five procedures. All four critical-protected procedures "
        "have zero critical interruption, zero complete critical outage and zero power "
        "violations in every nominal condition. Critical-aware rotation alone removes "
        "the original schedule's critical interruptions, showing that protection does "
        "not require learning. Differences in the remaining service outcomes concern "
        "how noncritical supply is allocated.",
        "**Table 13. Post-hoc matched baseline audit on the original 960 conditions; "
        "PPO is the equal mean of five frozen actors.**",
        table(
            ["Procedure", "Reward", "Fairness (%)", "Utilization (%)", "Served (MWh)"],
            [
                [
                    label,
                    f"{means[label]['episode_reward']:.4f}",
                    f"{100 * means[label]['fairness']:.2f}",
                    f"{100 * means[label]['utilization']:.2f}",
                    f"{means[label]['served_mwh']:.2f}",
                ]
                for label in labels
            ],
        ),
        "Table 14 reports PPO minus each added comparator. The paired hierarchical "
        "bootstrap uses 10,000 resamples with seed 20261031, equal run/condition weights "
        "and coupled shortage clusters. Its pointwise 95% intervals are descriptive; "
        "they do not replace the original adjusted 97.5% primary intervals.",
        "**Table 14. Post-hoc PPO effects against the two added baselines; "
        "mean differences and descriptive paired 95% intervals.**",
    ]
    metrics = [
        ("Daily reward", "episode_reward", 1, 4),
        ("Fairness (pp)", "fairness", 100, 3),
        ("Utilization (pp)", "utilization", 100, 3),
        ("Served energy (MWh)", "served_mwh", 1, 2),
        ("Switching rate (pp)", "switching_rate", 100, 3),
    ]
    parts.append(
        table(
            ["Outcome", "PPO - critical-aware rotation", "PPO - Fairness Rule"],
            [
                [
                    label,
                    *[
                        ci_text(audit["paired_inference"][pair][metric], scale, digits)
                        for pair in (
                            "PPO - Critical-aware Fixed Rotation",
                            "PPO - Fairness Rule",
                        )
                    ],
                ]
                for label, metric, scale, digits in metrics
            ],
        )
    )
    teacher = audit["paired_inference"]["PPO - Fairness Rule"]
    parts.append(
        "PPO improves fairness, utilization, served energy and reward over critical-aware "
        "rotation, although it switches more often. This is evidence of discretionary "
        "allocation improvement after sharing critical protection. It is not a pure "
        "learning ablation: action selection and the imitation-plus-PPO procedure also "
        "differ from the rotating rule."
    )
    parts.append(
        f"Against its Fairness Rule teacher, PPO gains {teacher['episode_reward']['mean']:.4f} "
        f"reward and {100 * teacher['fairness']['mean']:.3f} fairness percentage points, "
        f"but serves {-teacher['served_mwh']['mean']:.2f} fewer MWh per day and loses "
        f"{-100 * teacher['utilization']['mean']:.3f} utilization percentage points. "
        "The reward interval's lower endpoint remains below the earlier +0.01 reference. "
        "The utilization interval is entirely below the earlier -0.2-point margin. "
        "These references provide context, not a newly declared confirmatory test. "
        "A positive pooled reward difference is insufficient to claim uniform teacher "
        "superiority or an operational improvement."
    )
    severity = audit["shortage_paired_inference"]["PPO - Fairness Rule"]
    parts.append(
        "Exploratory reward differences against Fairness Rule vary by shortage: "
        + "; ".join(
            f"{level.title()} {ci_text(severity[level]['episode_reward'], digits=4)}"
            for level in ("mild", "moderate", "severe")
        )
        + ". Thus the overall gain is concentrated in Severe conditions, with losses "
        "under Mild and Moderate shortage. Run-level results also vary: seed 1729 has "
        "a negative mean reward difference while the other four have positive means. "
        "All severity and per-run records remain available in the audit artifact; "
        "the favorable subgroup does not replace the overall comparison."
    )
    parts.extend(
        [
            "The feasibility stress test retains the 320 base demand draws and replaces "
            "hourly supply with 75% or 25% of total critical demand. Each family has "
            "320 conditions, with no artificial replication over the original shortage "
            "labels, giving 5,760 additional trajectories across nine methods. These "
            "are constructed infeasibility tests, not estimates of their frequency "
            "in Khartoum. Table 15 reports the resulting critical shortfall.",
            "**Table 15. Infeasible-critical-floor stress test; means over 320 demand "
            "conditions in each family.**",
        ]
    )
    stress_rows = []
    for family, percentage in (("critical_supply_75pct", 75), ("critical_supply_25pct", 25)):
        block = audit["stress"][family]
        for label, values in (
            ("All protected procedures", block["procedure_means"]["PPO"]),
            ("Fixed Rotation", block["controller_means"]["Fixed Rotation"]),
        ):
            stress_rows.append(
                [
                    f"{percentage}%",
                    label,
                    f"{100 * values['critical_service_ratio']:.2f}",
                    f"{values['unmet_critical_mwh']:.2f}",
                    f"{values['critical_interruption_locality_hours']:.0f}",
                    f"{values['complete_critical_outage_locality_hours']:.0f}",
                ]
            )
    parts.append(
        table(
            [
                "Supply / critical demand",
                "Procedure",
                "Critical service (%)",
                "Unmet critical (MWh)",
                "Interruption (locality-h)",
                "Complete outage (locality-h)",
            ],
            stress_rows,
        )
    )
    parts.append(
        "All protected procedures means Critical-aware Fixed Rotation, Constrained "
        "Optimization, Fairness Rule and each PPO actor: their actions and outcomes "
        "coincide under the all-Critical-Only fallback. They fall back for all 24 hours "
        "and incur 168 locality-hours below 99% critical service, with zero complete "
        "critical outage. Their 75% and 25% service fractions follow the physical supply "
        "limit; high fairness and zero total blackouts do not mean adequate critical "
        "service. Every method records zero power-balance violations within the stated "
        "0.001 MW tolerance. This verifies a defined response to infeasibility, not "
        "successful critical protection when supply is insufficient."
    )
    return "\n\n".join(parts)


def main(training_sha, study_sha):
    training = load(TRAINING, training_sha)
    study = load(STUDY, study_sha)
    diagnostics_sha = (ROOT / DIAGNOSTICS).with_suffix(".sha256").read_text().split()[0]
    diagnostics = load(DIAGNOSTICS, diagnostics_sha)
    followup_sha = (ROOT / FOLLOWUP).with_suffix(".sha256").read_text().split()[0]
    followup = load(FOLLOWUP, followup_sha)
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
    results_table = (
        "**Table 6. Final-test controller outcomes on 960 matched conditions; "
        "PPO averages five selected runs.**\n\n"
        + table(
            ["Outcome", "Fixed Rotation", "Constrained optimizer", "PPO"],
            [
                [label, *[f"{row[metric] * scale:.{digits}f}" for row in rows]]
                for label, metric, scale, digits in main_metrics
            ],
        )
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
    results_table += (
        "\n\nTable 7 separates duration burdens and disparities from service-allocation fairness."
        "\n\n**Table 7. Final-test interruption durations and disparities, averaged after "
        "calculation within each episode.**\n\n"
        + table(
            ["Duration outcome", "Fixed Rotation", "Constrained optimizer", "PPO"],
            [[label, *[f"{row[metric]:.3f}" for row in rows]] for label, metric in durations],
        )
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
    shortages = (
        "**Table 10. Final-test outcomes by shortage; 320 matched conditions per level "
        "and equal PPO run weights.**\n\n"
        + table(
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
        table_number = 8 if pair == FIXED else 9
        primary_parts.append(
            f"Table {table_number} reports matched effects in the order shown above; "
            "the primary intervals are adjusted for the two-comparison family."
        )
        primary_parts.append(
            f"**Table {table_number}. Final-test paired effects: {pair}; "
            "adjusted 97.5% intervals.**"
        )
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
    budget_table = (
        "Table 11 gives final-test rewards for validation-selected checkpoints, not the "
        "validation rewards used to choose them.\n\n**Table 11. Same-run final-test daily "
        "reward: iteration 600 versus the validation-selected actor.**\n\n"
        + table(
            [
                "Training seed",
                "Selected iteration",
                "Iteration-600 test reward",
                "Selected-actor test reward",
                "Difference",
            ],
            budget_rows,
        )
    )
    budget_table += (
        "\n\nTable 12 gives the paired uncertainty for continuing the same training schedule."
        "\n\n**Table 12. Final-test effects of selected actors minus same-run iteration-600 "
        "actors; descriptive 95% intervals.**\n\n"
        + table(
            ["Across-run outcome", "Mean difference and 95% interval"],
            [
                [label, ci_text(intervals[BUDGET][metric], scale, 4)]
                for label, metric, scale in effect_metrics[1:]
            ],
        )
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
    budget_table += (
        f" Daily reward changed by {ci_text(intervals[BUDGET]['episode_reward'], digits=4)}, "
        f"while utilization changed by {ci_text(intervals[BUDGET]['utilization'], 100, 3)} "
        "percentage points. The mean utilization loss is therefore about 0.24 percentage "
        "points, and its interval breaches the allowed 0.2-point loss."
    )

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
    abstract += (
        " A post-hoc matched audit finds improved allocation over a critical-aware "
        "rotating rule. Against the Fairness Rule teacher, PPO gains 0.0102 daily reward "
        "but loses 0.308 utilization percentage points and 16.68 served MWh per day; "
        "the pooled gain is concentrated in Severe shortage. Stress tests verify "
        "partial critical-service fallback when the supply cannot cover critical demand."
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
        "whole-day advantage therefore concerns this nonanticipative myopic comparator and the "
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
        "trained procedure achieves fairer service allocation and a higher daily objective, "
        "with an efficiency and switching tradeoff."
    )
    fair_reward = 24 * 2 / 9 * (ppo["fairness"] - opt["fairness"])
    util_reward = 24 * 2 / 9 * (ppo["utilization"] - opt["utilization"])
    switch_reward = -24 * 0.25 / 9 * (ppo["switching_rate"] - opt["switching_rate"])
    reward_difference = ppo["episode_reward"] - opt["episode_reward"]
    if abs(fair_reward + util_reward + switch_reward - reward_difference) > 1e-5:
        raise ValueError("Optimizer reward-component decomposition does not reconstruct reward")
    interpretation.append(
        f"In practical scale, the {reward_difference:.4f} daily-score gain is "
        f"{100 * reward_difference / opt['episode_reward']:.3f}% of the optimizer's "
        f"{opt['episode_reward']:.4f} score, or "
        f"{100 * reward_difference / 24:.3f}% of the approximate ideal daily bound of 24. "
        "Using the reward equation, the measured mean difference decomposes into "
        f"{fair_reward:+.5f} from cumulative fairness, {util_reward:+.5f} from utilization "
        f"and {switch_reward:+.5f} from switching. Critical service and oversubscription "
        "contribute zero difference because both methods achieve full critical service "
        "without oversubscription on these conditions. These are arithmetic components "
        "of the chosen score, not independently identified causal effects."
    )
    interpretation.append(
        f"The corresponding direct energy result is {ppo['served_mwh'] - opt['served_mwh']:+.2f} "
        f"MWh served per synthetic day: {ppo['served_mwh']:.2f} for PPO versus "
        f"{opt['served_mwh']:.2f} for the optimizer, from the same "
        f"{opt['available_mwh']:.2f} MWh available. PPO therefore leaves about "
        f"{opt['served_mwh'] - ppo['served_mwh']:.2f} additional MWh unmet on average. "
        "This energy effect is calculated from the saved energy metrics; it is not a "
        "conversion of reward or of mean hourly utilization into MWh. The experiment "
        "does not measure the social value of the fairness gain, so it cannot establish "
        "that the score gain justifies the energy and switching costs in practice."
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
    training_summary += (
        "\n\nTable 17 reports the validation scores used for selection; Table 11 separately "
        "reports those actors' final-test scores.\n\n**Table 17. Training cost and "
        "validation-selected checkpoints; reward is from validation, not the final test.**\n\n"
        + table(
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
    )
    replacements = {
        "SUPERVISOR": json.loads((ROOT / "docs/academic-cover.json").read_text())["supervisor"],
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
        "FOLLOWUP_AUDIT": followup_text(followup),
    }
    payload = {
        "evidence_sha256": {
            TRAINING: training_sha,
            STUDY: study_sha,
            SUPPLEMENT: SUPPLEMENT_HASH,
            DIAGNOSTICS: diagnostics_sha,
            FOLLOWUP: followup_sha,
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

"""Plot completed, hash-bound PPO telemetry; no simulation or GPU work occurs here.

Run with tmp/tensorboard-runtime/Scripts/python.exe after all five runs finish.
The JSONL rows are matched exactly to telemetry in the frozen training record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (1729, 2718, 3141, 5772, 8111)
ITERATIONS = 3000
TELEMETRY_ITERATIONS = list(range(0, ITERATIONS + 1, 25))
VALIDATION_ITERATIONS = list(range(0, ITERATIONS + 1, 100))
METRICS = (
    "episode_reward",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "clip_fraction",
)
VALIDATION_METRICS = ("episode_reward", "fairness", "utilization")
SEED_COLORS = ("#137B73", "#E77D60", "#3F609D", "#816195", "#9B7B24")
TEAL = "#137B73"
CORAL = "#E77D60"
DEFINITIONS = {
    "episode_reward": "Pre-update rollout mean of 24-hour reward sums over 256 episodes.",
    "policy_loss": "Minimized negative clipped policy surrogate; normalized advantages.",
    "value_loss": "Half mean squared critic error, before the extra 0.5 loss coefficient.",
    "entropy": "Entropy in nats of the masked categorical policy.",
    "approx_kl": (
        "Diagnostic mean of ratio - 1 - log(ratio), using stored rollout log probabilities."
    ),
    "clip_fraction": "Fraction with abs(probability ratio - 1) > 0.2.",
}


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def digest(value: str) -> str:
    value = value.strip().lower()
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("expected SHA-256 must contain exactly 64 hexadecimal characters")
    return value


def portable(path: Path) -> str:
    path = path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("inputs and outputs must remain inside the project")
    return path.relative_to(ROOT).as_posix()


def finite(value: Any, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def summarize(values: list[float]) -> dict[str, float | int]:
    if len(values) != len(SEEDS):
        raise ValueError("every aggregate requires exactly five independent training runs")
    variance = statistics.variance(values)
    return {
        "mean": statistics.fmean(values),
        "sample_variance": variance,
        "sample_sd": math.sqrt(variance),
        "seed_count": len(values),
    }


def load_complete_inputs(
    training_path: Path, expected_sha256: str, monitor_dir: Path
) -> tuple[dict, list[list[dict]], dict[str, str]]:
    """Reject partial runs, altered telemetry and mismatched validation summaries."""
    raw = training_path.read_bytes()
    if sha256(raw) != digest(expected_sha256):
        raise ValueError("frozen training record SHA-256 mismatch")
    training = json.loads(raw)
    if training.get("schema") != "gridguard.focused_ppo_training.1":
        raise ValueError("unexpected training record schema")
    config = training.get("config", {})
    expected_config = {
        "iterations": ITERATIONS,
        "episodes_per_iteration": 256,
        "telemetry_interval": 25,
        "checkpoint_interval": 100,
        "training_seeds": list(SEEDS),
    }
    if any(config.get(key) != value for key, value in expected_config.items()):
        raise ValueError("training identity, budget or sampling schedule differs")
    runs = training.get("runs", [])
    if [run.get("seed") for run in runs] != list(SEEDS):
        raise ValueError("the final training record must contain all five declared runs")
    all_rows = []
    telemetry_hashes = {}
    for run in runs:
        seed = run["seed"]
        if run.get("transitions") != ITERATIONS * 256 * 24:
            raise ValueError(f"seed {seed} has not completed all 3000 iterations")
        path = monitor_dir / f"seed-{seed}.jsonl"
        telemetry_raw = path.read_bytes()
        rows = [json.loads(line) for line in telemetry_raw.decode("utf-8").splitlines()]
        if len(rows) != 121 or [row.get("iteration") for row in rows] != TELEMETRY_ITERATIONS:
            raise ValueError(f"seed {seed} requires exactly 121 complete telemetry rows")
        if rows != run.get("telemetry"):
            raise ValueError(f"seed {seed} telemetry differs from the frozen training record")
        checkpoints = run.get("checkpoints", [])
        if [point.get("iteration") for point in checkpoints] != VALIDATION_ITERATIONS:
            raise ValueError(f"seed {seed} requires all 31 validation checkpoints")
        checkpoint_by_iteration = {point["iteration"]: point for point in checkpoints}
        for row in rows:
            iteration = row["iteration"]
            expected_stage = "behavior_cloning" if iteration == 0 else "ppo"
            if (
                row.get("seed") != seed
                or row.get("stage") != expected_stage
                or row.get("transitions") != iteration * 256 * 24
            ):
                raise ValueError(f"seed {seed} has inconsistent telemetry identity")
            if iteration:
                for metric in METRICS:
                    finite(row.get(metric), f"seed {seed}, iteration {iteration}, {metric}")
            if iteration in checkpoint_by_iteration:
                summary = checkpoint_by_iteration[iteration]["summary"]
                if row.get("validation") != summary:
                    raise ValueError(f"seed {seed} validation differs between record and telemetry")
                for metric in VALIDATION_METRICS:
                    value = finite(summary["overall"].get(metric), f"validation {metric}")
                    if metric != "episode_reward" and not 0 <= value <= 1:
                        raise ValueError(f"validation {metric} is outside ratio bounds")
            elif "validation" in row:
                raise ValueError("validation appears outside its declared checkpoint schedule")
        all_rows.append(rows)
        telemetry_hashes[portable(path)] = sha256(telemetry_raw)
    return training, all_rows, telemetry_hashes


def aggregate(
    training: dict,
    rows_by_seed: list[list[dict]],
    training_path: Path,
    training_sha256: str,
    telemetry_hashes: dict[str, str],
) -> dict:
    diagnostic_points = []
    for index, iteration in enumerate(TELEMETRY_ITERATIONS[1:], start=1):
        diagnostic_points.append(
            {
                "iteration": iteration,
                "metrics": {
                    metric: summarize([float(rows[index][metric]) for rows in rows_by_seed])
                    for metric in METRICS
                },
            }
        )
    validation_points = []
    for index, iteration in enumerate(VALIDATION_ITERATIONS):
        validation_points.append(
            {
                "iteration": iteration,
                "metrics": {
                    metric: summarize(
                        [
                            float(run["checkpoints"][index]["summary"]["overall"][metric])
                            * (1 if metric == "episode_reward" else 100)
                            for run in training["runs"]
                        ]
                    )
                    for metric in VALIDATION_METRICS
                },
            }
        )
    return {
        "schema": "gridguard.focused_ppo_training_diagnostics.1",
        "training_path": portable(training_path),
        "training_sha256": digest(training_sha256),
        "input_telemetry_sha256": telemetry_hashes,
        "plotter_sha256": sha256(Path(__file__).read_bytes()),
        "training_seeds": list(SEEDS),
        "completed_iterations_per_run": ITERATIONS,
        "telemetry_rows_per_seed": 121,
        "diagnostic_samples_per_seed": 120,
        "validation_checkpoints_per_seed": 31,
        "aggregation": {
            "seed_weighting": "equal weight for each of five independent training seeds",
            "sample_variance": "sum((x_seed - mean)**2) / (5 - 1)",
            "band": "mean plus/minus one sample standard deviation; not a confidence interval",
            "smoothing": "none",
            "diagnostic_time_sampling": (
                "One logged PPO update every 25 iterations; not an average of 25 updates. "
                "Iteration zero is behavior cloning and has no PPO diagnostic values."
            ),
            "within_update": (
                "Except episode_reward, diagnostics are minibatch-size-weighted averages "
                "over all minibatches of all four PPO epochs in the logged update."
            ),
            "validation": (
                "Between-seed sample variance of each checkpoint's overall validation mean; "
                "not scenario variance, held-out uncertainty or critic explained variance."
            ),
        },
        "diagnostic_definitions": DEFINITIONS,
        "validation_units": {
            "episode_reward": {"mean": "daily reward", "sample_variance": "daily reward squared"},
            "fairness": {"mean": "percent", "sample_variance": "percentage points squared"},
            "utilization": {"mean": "percent", "sample_variance": "percentage points squared"},
        },
        "explained_variance_logged": any(
            "explained_variance" in row for rows in rows_by_seed for row in rows
        ),
        "explained_variance_figure": "not produced; no critic explained variance is inferred",
        "diagnostics": diagnostic_points,
        "validation": validation_points,
    }


def save_json(path: Path, payload: dict) -> str:
    """Create immutable evidence, permitting an identical repeat without replacing bytes."""
    raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    checksum = sha256(raw)
    checksum_path = path.with_suffix(".sha256")
    checksum_raw = f"{checksum}  {path.name}\n".encode("ascii")
    for target, contents in ((path, raw), (checksum_path, checksum_raw)):
        if target.exists() and target.read_bytes() != contents:
            raise FileExistsError(f"refusing to overwrite different evidence: {target}")
    path.parent.mkdir(parents=True, exist_ok=True)
    for target, contents in ((path, raw), (checksum_path, checksum_raw)):
        if not target.exists():
            with target.open("xb") as handle:
                handle.write(contents)
    return checksum


def plot_figures(payload: dict, rows_by_seed: list[list[dict]], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    font_dir = ROOT / "src/gridguard/assets/fonts"
    regular = font_dir / "AtkinsonHyperlegible-Regular.ttf"
    for filename in (regular, font_dir / "AtkinsonHyperlegible-Bold.ttf"):
        font_manager.fontManager.addfont(str(filename))
    plt.rcParams.update(
        {
            "font.family": font_manager.FontProperties(fname=str(regular)).get_name(),
            "font.size": 10,
            "text.color": "#173F43",
            "axes.labelcolor": "#173F43",
            "axes.edgecolor": "#DCE7E1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "grid.color": "#E6ECE8",
            "xtick.color": "#5C7475",
            "ytick.color": "#5C7475",
            "svg.fonttype": "none",
            "svg.hashsalt": payload["training_sha256"],
        }
    )
    out.mkdir(parents=True, exist_ok=True)

    def save(fig, name: str) -> None:
        fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(
            out / f"{name}.svg", bbox_inches="tight", facecolor="white", metadata={"Date": None}
        )
        plt.close(fig)

    labels = (
        ("Rollout daily reward", "Canonical score / episode"),
        ("Policy loss", "Negative clipped surrogate"),
        ("Value loss", "Half mean squared error"),
        ("Policy entropy", "Nats"),
        ("Approximate KL", r"$\mathbb{E}[\rho - 1 - \log\rho]$"),
        ("Clipped fraction", "Fraction of samples"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(11.7, 7.2))
    fig.suptitle("Training diagnostics across five independent PPO runs", fontsize=16, y=0.99)
    x = TELEMETRY_ITERATIONS[1:]
    for ax, metric, (title, unit) in zip(axes.flat, METRICS, labels, strict=True):
        for rows, color in zip(rows_by_seed, SEED_COLORS, strict=True):
            ax.plot(x, [row[metric] for row in rows[1:]], color=color, alpha=0.27, linewidth=0.9)
        summaries = [point["metrics"][metric] for point in payload["diagnostics"]]
        means = [point["mean"] for point in summaries]
        lower = [point["mean"] - point["sample_sd"] for point in summaries]
        upper = [point["mean"] + point["sample_sd"] for point in summaries]
        ax.fill_between(x, lower, upper, color=TEAL, alpha=0.16, linewidth=0)
        ax.plot(x, means, color=TEAL, linewidth=1.7)
        ax.set_title(title, loc="left", fontsize=12)
        ax.set_ylabel(unit)
        ax.set_xlabel("PPO iteration")
        ax.set_xlim(0, ITERATIONS)
        ax.set_xticks((0, 1000, 2000, 3000))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4), useOffset=False)
        ax.grid(axis="y", alpha=0.8)
    handles = [
        Line2D([], [], color=color, alpha=0.55, linewidth=1, label=str(seed))
        for seed, color in zip(SEEDS, SEED_COLORS, strict=True)
    ]
    handles.extend(
        [
            Line2D([], [], color=TEAL, linewidth=1.8, label="Five-run mean"),
            Patch(facecolor=TEAL, alpha=0.16, label="Mean ± sample SD"),
        ]
    )
    fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.05), ncols=7, frameon=False
    )
    fig.text(
        0.5,
        0.014,
        "120 recorded updates per seed · every 25 iterations · no smoothing · "
        "SD is not a confidence interval",
        ha="center",
        fontsize=10,
        color="#5C7475",
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.9, bottom=0.19, wspace=0.36, hspace=0.5)
    save(fig, "tensorboard-diagnostics")

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 7.3), sharex=True)
    fig.suptitle("Validation variance between five independent training seeds", fontsize=15, y=0.99)
    labels = (
        "Daily reward variance\n(score²)",
        "Jain fairness variance\n(percentage points²)",
        "Utilization variance\n(percentage points²)",
    )
    for ax, metric, label in zip(axes, VALIDATION_METRICS, labels, strict=True):
        values = [point["metrics"][metric]["sample_variance"] for point in payload["validation"]]
        ax.plot(VALIDATION_ITERATIONS, values, color=TEAL, linewidth=1.8)
        ax.scatter(
            VALIDATION_ITERATIONS, values, color=CORAL, s=17, edgecolor="white", linewidth=0.4
        )
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
        ax.set_xlim(0, ITERATIONS)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4), useOffset=False)
        ax.grid(axis="y", alpha=0.8)
    axes[-1].set_xlabel("PPO iteration · iteration 0 is the behavior-cloning starting point")
    fig.text(
        0.5,
        0.017,
        "31 matched validation checkpoints · sample variance uses $n - 1 = 4$ · no smoothing\n"
        "Variance of run means; not scenario variance, "
        "a confidence interval, or critic explained variance",
        ha="center",
        fontsize=10,
        color="#5C7475",
    )
    fig.subplots_adjust(left=0.18, right=0.97, top=0.92, bottom=0.16, hspace=0.26)
    save(fig, "training-variance")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training", type=Path, default=ROOT / "data/evidence/focused-ppo-training.json"
    )
    parser.add_argument("--training-sha256", required=True)
    parser.add_argument("--monitor-dir", type=Path, default=ROOT / "tmp/focused-ppo-monitor")
    parser.add_argument("--out", type=Path, default=ROOT / "docs/assets/focused-study")
    parser.add_argument(
        "--evidence",
        type=Path,
        default=ROOT / "data/evidence/focused-ppo-training-diagnostics.json",
    )
    args = parser.parse_args()
    for path in (args.training, args.monitor_dir, args.out, args.evidence):
        portable(path)
    training, rows, hashes = load_complete_inputs(
        args.training, args.training_sha256, args.monitor_dir
    )
    payload = aggregate(training, rows, args.training, args.training_sha256, hashes)
    checksum = save_json(args.evidence, payload)
    plot_figures(payload, rows, args.out)
    print(f"Created two training figures as PNG/SVG in {args.out}")
    print(f"Diagnostics: {args.evidence}\nSHA-256: {checksum}")


if __name__ == "__main__":
    main()

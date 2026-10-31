"""Plot already verified portable study summaries; no simulation or inference occurs here."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/assets/focused-study"
COLORS = ["#718789", "#E77D60", "#137B73"]
SEED_COLORS = ["#137B73", "#E77D60", "#3F609D", "#816195", "#9B7B24"]


def read_verified(relative, expected=None):
    path = ROOT / relative
    if expected is None:
        expected = path.with_suffix(".sha256").read_text().split()[0]
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"Hash mismatch: {relative}")
    return json.loads(raw)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def forecast_figure():
    forecast = read_verified(
        "data/evidence/proposal-supplement.json",
        "fae5c875b1be148d7a069259b19df9b04666fe97e2182ce8898f45fed7b0a43e",
    )["forecast"]["methods"]
    methods = ("trailing_mean_3h", "persistence_1h", "ridge_ar3")
    labels = ("3-hour\nmean", "Persistence", "Ridge AR(3)")
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.2), layout="constrained")
    fig.suptitle("Next-hour demand prediction · 320 held-out synthetic traces", fontsize=14)
    for ax, scope, metric, title in zip(
        axes.flat,
        ("locality_sector", "system_total", "locality_sector", "system_total"),
        ("mae_mw", "mae_mw", "rmse_mw", "rmse_mw"),
        ("Locality-sector MAE", "System-total MAE", "Locality-sector RMSE", "System-total RMSE"),
        strict=True,
    ):
        values = [forecast[method][scope][metric] for method in methods]
        bars = ax.bar(range(3), values, color=COLORS, width=0.57, zorder=3)
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=5)
        ax.set_xticks(range(3), labels)
        ax.set_title(title, loc="left", fontsize=13, pad=12)
        ax.set_ylabel("Prediction error (MW) · lower is better")
        ax.set_ylim(0, max(values) * 1.18)
        ax.grid(axis="y", zorder=0)
    save(fig, "forecast-errors")


def main(forecast_only=False):
    font = ROOT / "src/gridguard/assets/fonts/AtkinsonHyperlegible-Regular.ttf"
    font_manager.fontManager.addfont(str(font))
    font_manager.fontManager.addfont(str(font.with_name("AtkinsonHyperlegible-Bold.ttf")))
    plt.rcParams.update(
        {
            "font.family": font_manager.FontProperties(fname=str(font)).get_name(),
            "font.size": 11,
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
        }
    )
    forecast_figure()
    if forecast_only:
        return
    training = read_verified("data/evidence/focused-ppo-training.json")
    evidence = read_verified("data/evidence/focused-ppo-study.json")
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 7.2), sharex=True, layout="constrained")
    fig.suptitle("Validation progress across five independent PPO runs", fontsize=15)
    for ax, metric, label, scale in zip(
        axes,
        ("episode_reward", "fairness", "utilization"),
        ("Daily reward", "Jain service fairness (%)", "Supply utilization (%)"),
        (1, 100, 100),
        strict=True,
    ):
        for index, run in enumerate(training["runs"]):
            points = run["checkpoints"]
            x = [p["iteration"] for p in points]
            y = [scale * p["summary"]["overall"][metric] for p in points]
            ax.plot(x, y, color=SEED_COLORS[index], linewidth=1.6, label=str(run["seed"]))
            selected = run["selected"]
            ax.scatter(
                [selected["iteration"]],
                [scale * selected["summary"]["overall"][metric]],
                color=SEED_COLORS[index],
                s=36,
                edgecolor="white",
                linewidth=0.7,
                zorder=4,
            )
        ax.axvline(600, linestyle="--", linewidth=1, color="#8A9B9B")
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.8)
    axes[0].legend(title="Training seed", ncols=5, loc="lower right", fontsize=9, title_fontsize=9)
    axes[-1].set_xlabel("PPO training iterations")
    axes[-1].set_xlim(0, 3050)
    save(fig, "training-curves")

    methods = ("Fixed Rotation", "Constrained Optimization", "PPO")
    rows = [evidence["controller_means"][name] for name in methods[:2]]
    rows.append(evidence["procedure_means"]["PPO"])
    metrics = (
        (
            "critical_interruption_locality_hours",
            "Critical interruption",
            "Locality-hours / day",
            1,
        ),
        ("fairness", "Jain service fairness", "Percent", 100),
        ("utilization", "Supply utilization", "Percent", 100),
        ("episode_reward", "Daily reward", "Canonical score / day", 1),
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.3), layout="constrained")
    fig.suptitle("960 matched conditions · PPO averaged equally across five runs", fontsize=14)
    for ax, (metric, title, units, scale) in zip(axes.flat, metrics, strict=True):
        values = [r[metric] * scale for r in rows]
        bars = ax.bar(range(3), values, color=COLORS, width=0.57, zorder=3)
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=5, fontsize=11)
        ax.set_xticks(range(3), ["Fixed\nRotation", "Constrained\noptimizer", "PPO"])
        ax.set_title(title, loc="left", fontsize=13, pad=12)
        ax.set_ylabel(units)
        ax.set_ylim(0, (max(values) or 1) * 1.17)
        ax.grid(axis="y", zorder=0)
    save(fig, "controller-outcomes")

    comparisons = ("PPO - Constrained Optimization",)
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.4), layout="constrained")
    fig.suptitle("PPO versus constrained optimizer · paired 97.5% intervals", fontsize=14)
    for ax, (metric, title, units, scale) in zip(axes.flat, metrics, strict=True):
        for index, name in enumerate(comparisons):
            data = evidence["primary_family_adjusted_inference"][name][metric]
            mean, low, high = [data[key] * scale for key in ("mean", "lower", "upper")]
            ax.plot([low, high], [index, index], color=COLORS[2], linewidth=3)
            ax.scatter([mean], [index], color=COLORS[2], s=48, zorder=3)
            ax.annotate(
                f"{mean:+.3f} [{low:+.3f}, {high:+.3f}]",
                (mean, index),
                xytext=(0, 13),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
        ax.axvline(0, color="#8A9B9B", linestyle="--", linewidth=1)
        ax.set_yticks([])
        ax.set_ylim(0.8, -0.8)
        ax.set_title(title, loc="left", fontsize=13, pad=12)
        ax.set_xlabel(
            "Percentage-point difference" if scale == 100 else f"Difference ({units.lower()})"
        )
        ax.margins(x=0.35)
        ax.grid(axis="x")
    save(fig, "paired-effects")
    print(f"Wrote 4 figures as PNG and SVG to {OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast-only", action="store_true")
    main(parser.parse_args().forecast_only)

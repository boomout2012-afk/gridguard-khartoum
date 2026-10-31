"""Derive interpretation from frozen evidence on CUDA, then assemble portable artifacts.

No training, new scenarios, model inference, or changes to published evidence occur.
The one packed scientific host transfer follows all CuPy reductions. Plotting uses
only that transferred summary. Scalar validation/preflight synchronization is separate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from gridguard.domain import LOCALITIES, SECTOR_SHARES, ShortageLevel
from gridguard.gpu import run_gpu_preflight
from gridguard.scenarios import SYSTEM_PEAK_MW, _sector_profiles, _supply_fraction
from gridguard.statistics import torch_cuda_to_cupy

ROOT = Path(__file__).resolve().parents[1]
SOURCE_HASHES = {
    "data/evidence/focused-ppo-study.json": (
        "a4d5a849612f06eedef96e0d28e08247e5e3757eba0b4f2e4f1a893e9d4b8247"
    ),
    "data/evidence/focused-ppo-training.json": (
        "e4f8b8f7c73693f713ecef69a06434ce75a3a101d2541654d7009398c4eb5624"
    ),
    "data/evidence/focused-ppo-training-diagnostics.json": (
        "b4c2c76e481e0f30e323ea22f9359212bac1b2018605f833682875d98face500"
    ),
}
PROCEDURES = ("Fixed Rotation", "Constrained Optimization", "PPO")
PAIRS = ("PPO - Fixed Rotation", "PPO - Constrained Optimization")
SHORTAGES = ("Mild", "Moderate", "Severe")
TRAINING_SEEDS = (1729, 2718, 3141, 5772, 8111)
VALIDATION_METRICS = ("episode_reward", "fairness", "utilization")
DIAGNOSTIC_METRICS = (
    "episode_reward",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "clip_fraction",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_frozen(relative: str) -> dict:
    raw = (ROOT / relative).read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_HASHES[relative]:
        raise ValueError(f"Frozen evidence SHA-256 mismatch: {relative}")
    return json.loads(raw)


def _cuda_view(tensor: torch.Tensor) -> Any:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("scientific input must be a torch.Tensor")
    if tensor.device != torch.device("cuda:0"):
        raise ValueError("scientific input must remain on cuda:0")
    if tensor.dtype != torch.float64:
        raise TypeError("portable evidence must be reduced as float64")
    return torch_cuda_to_cupy(tensor.contiguous())


def locality_summary_cuda(
    ratios: torch.Tensor,
    controller_names: tuple[str, ...],
    selected_names: tuple[str, ...],
) -> dict[str, Any]:
    """Reduce per-condition endpoint ratios; every returned array remains on CUDA.

    Input axes are method, shortage, seed, base draw, locality. Select by identity,
    rather than assuming methods happen to occupy a fixed contiguous slice.
    """
    import cupy as cp

    if not isinstance(ratios, torch.Tensor):
        raise TypeError("endpoint ratios must be a torch.Tensor")
    if ratios.ndim != 5 or ratios.shape[-1] != len(LOCALITIES) or min(ratios.shape) < 1:
        raise ValueError("ratios require [method, shortage, seed, draw, seven localities]")
    if len(controller_names) != ratios.shape[0] or len(set(controller_names)) != len(
        controller_names
    ):
        raise ValueError("controller names must be unique and match the method axis")
    if not selected_names or len(set(selected_names)) != len(selected_names):
        raise ValueError("selected names must be nonempty and unique")
    required = (*PROCEDURES[:2], *selected_names)
    if len(set(required)) != len(required) or any(n not in controller_names for n in required):
        raise ValueError("each required baseline and selected actor must occur exactly once")
    values = _cuda_view(ratios)
    if not bool((cp.isfinite(values) & (values >= 0) & (values <= 1)).all()):
        raise ValueError("endpoint ratios must be finite and in [0, 1]")
    fixed = values[controller_names.index(PROCEDURES[0])]
    optimizer = values[controller_names.index(PROCEDURES[1])]
    selected = cp.stack([values[controller_names.index(n)] for n in selected_names])
    condition_axes = (0, 1, 2)
    run_condition_axes = (0, 1, 2, 3)
    minima = (fixed.min(axis=-1), optimizer.min(axis=-1), selected.min(axis=-1))
    return {
        "means_percent": 100
        * cp.stack(
            (
                fixed.mean(axis=condition_axes),
                optimizer.mean(axis=condition_axes),
                selected.mean(axis=run_condition_axes),
            )
        ),
        "paired_shifts_pp": 100
        * cp.stack(
            (
                (selected - fixed[None]).mean(axis=run_condition_axes),
                (selected - optimizer[None]).mean(axis=run_condition_axes),
            )
        ),
        "mean_within_condition_minimum_percent": 100 * cp.stack(tuple(v.mean() for v in minima)),
        "paired_minimum_shifts_pp": 100
        * cp.stack(
            (
                (minima[2] - minima[0][None]).mean(),
                (minima[2] - minima[1][None]).mean(),
            )
        ),
        "by_shortage_means_percent": 100
        * cp.stack(
            (
                fixed.mean(axis=(1, 2)),
                optimizer.mean(axis=(1, 2)),
                selected.mean(axis=(0, 2, 3)),
            )
        ),
        "by_shortage_minimum_percent": 100
        * cp.stack(
            (
                minima[0].mean(axis=(1, 2)),
                minima[1].mean(axis=(1, 2)),
                minima[2].mean(axis=(0, 2, 3)),
            )
        ),
        "selected_run_means_percent": 100 * selected.mean(axis=(1, 2, 3)),
    }


def _learning_blocks(training: dict) -> tuple[dict[str, Any], dict]:
    import cupy as cp

    runs = training["runs"]
    if [r["seed"] for r in runs] != list(TRAINING_SEEDS):
        raise ValueError("training run identities changed")
    iterations = list(range(0, 3001, 100))
    telemetry_iterations = list(range(25, 3001, 25))
    for run in runs:
        if [r["iteration"] for r in run["checkpoints"]] != iterations:
            raise ValueError("validation checkpoint sequence changed")
        if [
            r["iteration"] for r in run["telemetry"] if r["stage"] == "ppo"
        ] != telemetry_iterations:
            raise ValueError("training diagnostic sequence changed")
    validation = torch.tensor(
        [
            [[p["summary"]["overall"][m] for m in VALIDATION_METRICS] for p in run["checkpoints"]]
            for run in runs
        ],
        dtype=torch.float64,
        device="cuda:0",
    )
    telemetry = torch.tensor(
        [
            [[p[m] for m in DIAGNOSTIC_METRICS] for p in run["telemetry"] if p["stage"] == "ppo"]
            for run in runs
        ],
        dtype=torch.float64,
        device="cuda:0",
    )
    val = _cuda_view(validation) * cp.array([1, 100, 100], dtype=cp.float64)
    diag = _cuda_view(telemetry)
    if not bool(cp.isfinite(val).all() & cp.isfinite(diag).all()):
        raise ValueError("learning records contain nonfinite values")
    mean, variance = val.mean(axis=0), val.var(axis=0, ddof=1)
    anchors = (0, 6, 15, 25, 30)
    diagnostic_anchors = (0, 23, 59, 99, 119)
    blocks = {
        "validation_anchor_means": mean[list(anchors)],
        "validation_anchor_sample_variances": variance[list(anchors)],
        "validation_changes": cp.stack(
            (mean[6] - mean[0], mean[30] - mean[6], mean[30] - mean[25], mean[30] - mean[0])
        ),
        "validation_early_reward_gain_fraction": (
            (mean[6, 0] - mean[0, 0]) / (mean[30, 0] - mean[0, 0])
        ).reshape(1),
        "validation_variance_max": cp.stack((variance.max(axis=0), variance.argmax(axis=0) * 100)),
        "diagnostic_anchor_means": diag.mean(axis=0)[list(diagnostic_anchors)],
        "diagnostic_anchor_sample_variances": diag.var(axis=0, ddof=1)[list(diagnostic_anchors)],
    }
    metadata = {
        "validation_metrics": list(VALIDATION_METRICS),
        "validation_units": ["daily reward", "percentage points", "percentage points"],
        "validation_anchor_iterations": [iterations[i] for i in anchors],
        "validation_change_intervals": [[0, 600], [600, 3000], [2500, 3000], [0, 3000]],
        "diagnostic_metrics": list(DIAGNOSTIC_METRICS),
        "diagnostic_anchor_iterations": [telemetry_iterations[i] for i in diagnostic_anchors],
        "variance_convention": "sample variance across the five run means, denominator 4",
        "checkpoint_convention": (
            "iteration 3000 is the fixed final checkpoint, not each run's selected actor"
        ),
    }
    return blocks, metadata


def _generator_blocks() -> dict[str, Any]:
    """Deterministic envelope of the frozen profiles, not a new sampled scenario."""
    import cupy as cp

    profiles = _cuda_view(_sector_profiles(torch.device("cuda:0")).to(torch.float64))
    base = profiles * cp.array(SECTOR_SHARES, dtype=cp.float64)[None] * SYSTEM_PEAK_MW
    critical, noncritical = base[:, :3].sum(axis=1), base[:, 3:].sum(axis=1)
    fractions = torch.stack(
        [
            _supply_fraction(1, level, torch.device("cuda:0"))[0]
            for level in (ShortageLevel.MILD, ShortageLevel.MODERATE, ShortageLevel.SEVERE)
        ]
    ).to(torch.float64)
    # S-C = (q-1)C + qN. For q=.95*f<1 the adverse demand corners are
    # C=1.05*C0 and N=.95*N0. The 95%-of-demand cap never binds: .8*1.05<.95.
    q = 0.95 * _cuda_view(fractions)
    margin = (q - 1) * 1.05 * critical[None] + q * 0.95 * noncritical[None]
    total = base.sum(axis=1)
    return {
        "noiseless_hourly_system_demand_mw": total,
        "noiseless_peak_mw_hour": cp.stack((total.max(), total.argmax())),
        "critical_margin_lower_bound_mw_by_shortage_hour": margin,
        "critical_margin_minimum_mw_hour": cp.stack((margin.min(axis=1), margin.argmin(axis=1))),
    }


def _transfer_once(blocks: dict[str, Any]) -> dict[str, list]:
    import cupy as cp

    shapes = {name: value.shape for name, value in blocks.items()}
    sizes = {name: value.size for name, value in blocks.items()}
    packed = cp.concatenate([value.reshape(-1) for value in blocks.values()])
    host = cp.asnumpy(packed)  # The single scientific host transfer.
    unpacked, offset = {}, 0
    for name in blocks:
        end = offset + sizes[name]
        unpacked[name] = host[offset:end].reshape(shapes[name]).tolist()
        offset = end
    return unpacked


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    path.with_suffix(".sha256").write_text(f"{_sha256(path)}  {path.name}\n", encoding="ascii")


def _plot_locality(payload: dict, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

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
            "grid.color": "#E6ECE8",
            "svg.fonttype": "none",
            "svg.hashsalt": "gridguard-focused-locality-interpretation",
        }
    )
    colors = ("#718789", "#E77D60", "#137B73")
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(10, 5.8), gridspec_kw={"width_ratios": [2.2, 1]}
    )
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.30, top=0.78, wspace=0.48)
    fig.suptitle("Locality service at the end of the day", fontsize=16, y=0.98)
    left.set_title("Mean locality demand fraction served", loc="left", fontsize=12, pad=10)
    means = payload["means_percent"]
    shifts = payload["paired_shifts_pp"][1]
    for i in range(7):
        left.plot([means[1][i], means[2][i]], [i, i], color="#B8CAC3", linewidth=3, zorder=2)
        left.text(96, i, f"{shifts[i]:+.2f} pp", va="center", ha="right", fontsize=9)
    for values, label, color, marker in zip(
        means, PROCEDURES, colors, ("s", "o", "D"), strict=True
    ):
        left.scatter(values, range(7), label=label, color=color, marker=marker, s=37, zorder=3)
    left.set_yticks(range(7), payload["locality_order"])
    left.set_xlim(0, 100)
    left.set_ylim(6.7, -0.7)
    left.set_xticks([0, 20, 40, 60, 80, 100])
    left.set_xlabel("End-of-day demand served (%)")
    left.grid(axis="x", zorder=0)
    left.tick_params(axis="y", length=0)
    left.text(96, -0.97, "PPO minus optimizer", ha="right", fontsize=9)
    right.set_title("Worst-served locality\nwithin each condition", loc="left", fontsize=12, pad=10)
    minima = payload["mean_within_condition_minimum_percent"]
    bars = right.barh(range(3), minima, color=colors, height=0.5, zorder=3)
    right.bar_label(bars, labels=[f"{v:.2f}%" for v in minima], padding=4, fontsize=11)
    right.set_yticks(range(3), ["Fixed\nRotation", "Constrained\nOptimization", "PPO"])
    right.set_xlim(0, 100)
    right.set_ylim(2.8, -0.8)
    right.set_xlabel("Mean minimum service (%)")
    right.grid(axis="x", zorder=0)
    fig.legend(
        *left.get_legend_handles_labels(),
        ncols=3,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.93),
        frameon=False,
        fontsize=10,
    )
    fig.text(
        0.5,
        0.085,
        "960 matched conditions; PPO weights five selected runs equally. Means of episode ratios.\n"
        "Right: take the minimum within each condition before averaging.\n"
        "These endpoint measures differ from hourly-mean Jain fairness and pooled energy ratios.",
        ha="center",
        va="center",
        fontsize=10,
        color="#5C7475",
    )
    for extension in ("png", "svg"):
        fig.savefig(
            output_dir / f"locality-allocation.{extension}",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            metadata={"Date": None} if extension == "svg" else None,
        )
    plt.close(fig)


def main(output_dir: Path, details_path: Path) -> None:
    runtime = asdict(run_gpu_preflight())
    study = _read_frozen("data/evidence/focused-ppo-study.json")
    training = _read_frozen("data/evidence/focused-ppo-training.json")
    _read_frozen("data/evidence/focused-ppo-training-diagnostics.json")
    for relative, expected in study["source_sha256"].items():
        if _sha256(ROOT / relative) != expected:
            raise ValueError(f"Frozen scientific source changed: {relative}")
    names = tuple(study["controller_names"])
    selected = tuple(f"PPO seed {seed}" for seed in TRAINING_SEEDS)
    ratios = torch.tensor(
        study["final_locality_service_ratio_cube"], dtype=torch.float64, device="cuda:0"
    )
    if tuple(ratios.shape) != (12, 3, 10, 32, 7):
        raise ValueError("original study locality cube shape changed")
    locality = locality_summary_cuda(ratios, names, selected)
    learning, learning_metadata = _learning_blocks(training)
    generator = _generator_blocks()
    transferred = _transfer_once({**locality, **learning, **generator})
    provenance = {
        "source_sha256": {
            **SOURCE_HASHES,
            **study["source_sha256"],
            "tools/interpret_focused_study.py": _sha256(Path(__file__)),
        },
        "runtime": runtime,
        "computation": (
            "CUDA float64 CuPy reductions; PyTorch/CuPy DLPack; one final packed host transfer"
        ),
        "new_simulation": False,
    }
    payload = {
        "schema": "gridguard.focused_locality_interpretation.1",
        **provenance,
        "source_key": "final_locality_service_ratio_cube",
        "source_axes": ["method", "shortage", "seed", "base draw", "locality"],
        "source_shape": list(ratios.shape),
        "source_controller_order": list(names),
        "locality_order": [str(n) for n in LOCALITIES],
        "procedure_order": list(PROCEDURES),
        "paired_comparison_order": list(PAIRS),
        "shortage_order": list(SHORTAGES),
        "selected_run_order": list(selected),
        "aggregation": (
            "mean of per-condition 24h served/demand ratios; equal conditions and selected runs"
        ),
        "minimum_aggregation": (
            "minimum across seven localities inside each run-condition, then equal mean"
        ),
        "intervals": "descriptive means only; no new confidence interval or primary claim",
        **{name: transferred[name] for name in locality},
    }
    details = {
        "schema": "gridguard.focused_training_generator_interpretation.1",
        **provenance,
        **learning_metadata,
        **{name: transferred[name] for name in (*learning, *generator)},
        "generator_envelope": (
            "bounded-noise real-arithmetic envelope evaluated using frozen float32 profiles; "
            "no sampled test"
        ),
        "original_training_runtime": training["runtime"],
        "original_training_numerics": training["numerics"],
        "original_run_seconds": [run["seconds"] for run in training["runs"]],
        "original_evaluation_seconds": study["elapsed_seconds"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_locality(payload, output_dir)
    payload["figure_sha256"] = {
        f"locality-allocation.{extension}": _sha256(output_dir / f"locality-allocation.{extension}")
        for extension in ("png", "svg")
    }
    _write_json(output_dir / "locality-allocation.json", payload)
    _write_json(details_path, details)
    print(f"Wrote {output_dir / 'locality-allocation.json'} and .png/.svg")
    print(f"Wrote {details_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/assets/focused-study")
    parser.add_argument(
        "--details-path",
        type=Path,
        default=ROOT / "docs/assets/focused-study/interpretation-details.json",
    )
    args = parser.parse_args()
    main(args.output_dir, args.details_path)

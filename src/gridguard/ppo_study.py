"""Selection rules and CUDA uncertainty for the constrained PPO training study."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

import torch

from gridguard.statistics import (
    CI_PERCENTILES,
    BootstrapError,
    _import_cupy,
    _linear_percentile_rows,
    _validate_positive_int,
    _validate_seed,
    torch_cuda_to_cupy,
)

SHORTAGES = ("mild", "moderate", "severe")
DECISION_METRICS = ("episode_reward", "fairness", "utilization")
SAFETY_METRICS = (
    "critical_outage_locality_hours",
    "critical_interruption_locality_hours",
    "power_balance_violation_rate",
)
SERVICE_MARGIN = 0.002
REWARD_THRESHOLD = 0.01
BOOTSTRAP_SEED = 20261023
BOOTSTRAP_RESAMPLES = 10_000
_BOOTSTRAP_CHUNK = 128


def _finite_number(value: object, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _validate_metric_row(row: object, label: str) -> None:
    if not isinstance(row, dict) or any(metric not in row for metric in DECISION_METRICS):
        raise ValueError(f"{label} requires all decision metrics")
    for metric in DECISION_METRICS:
        _finite_number(row[metric], f"{label} {metric}")


def _validate_shortages(rows: object, label: str) -> None:
    if not isinstance(rows, dict) or set(rows) != set(SHORTAGES):
        raise ValueError(f"{label} requires every declared shortage")
    for level, row in rows.items():
        _validate_metric_row(row, f"{label} {level}")


def _validate_summary(summary: object, label: str) -> None:
    if not isinstance(summary, dict) or any(
        key not in summary for key in ("overall", "by_shortage", "maxima")
    ):
        raise ValueError(f"{label} requires overall, by_shortage and maxima")
    _validate_metric_row(summary["overall"], f"{label} overall")
    _validate_shortages(summary["by_shortage"], label)
    maxima = summary["maxima"]
    if not isinstance(maxima, dict) or any(metric not in maxima for metric in SAFETY_METRICS):
        raise ValueError(f"{label} requires safety maxima")
    for metric in SAFETY_METRICS:
        value = _finite_number(maxima[metric], f"{label} maximum {metric}")
        if value < 0:
            raise ValueError(f"{label} safety maxima must be nonnegative")


def checkpoint_eligibility(candidate: dict, reference: dict) -> dict:
    """Apply all validation gates to already-transferred compact summaries."""
    _validate_summary(candidate, "candidate")
    _validate_summary(reference, "reference")
    reasons: list[str] = []
    for metric in SAFETY_METRICS:
        if candidate["maxima"][metric] != 0:
            reasons.append(f"candidate maximum {metric} is nonzero")
        if reference["maxima"][metric] != 0:
            reasons.append(f"reference maximum {metric} is nonzero")
    pairs = [("overall", candidate["overall"], reference["overall"])] + [
        (level, candidate["by_shortage"][level], reference["by_shortage"][level])
        for level in SHORTAGES
    ]
    for label, row, baseline in pairs:
        if row["episode_reward"] < baseline["episode_reward"]:
            reasons.append(f"{label} episode_reward is below Fairness Rule")
        for metric in ("fairness", "utilization"):
            if row[metric] < baseline[metric] - SERVICE_MARGIN:
                reasons.append(f"{label} {metric} is below the service margin")
    return {"eligible": not reasons, "reasons": reasons}


def select_checkpoint(records: list[dict], reference: dict) -> dict:
    """Select an eligible reward maximum; if absent, flag an ineligible fallback."""
    if not isinstance(records, list) or not records:
        raise ValueError("checkpoint selection requires a nonempty record list")
    _validate_summary(reference, "reference")
    iterations: set[int] = set()
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
        result = deepcopy(record)
        result.update(checkpoint_eligibility(record["summary"], reference))
        evaluated.append(result)
    eligible = [record for record in evaluated if record["eligible"]]
    selected = max(
        eligible or evaluated,
        key=lambda record: (record["summary"]["overall"]["episode_reward"], -record["iteration"]),
    )
    selected["fallback"] = not bool(eligible)
    return selected


def verify_checkpoint_selection(records: list[dict], reference: dict, selected: dict) -> None:
    """Reject altered checkpoint choices, metrics or eligibility declarations."""
    expected = select_checkpoint(records, reference)
    if not isinstance(selected, dict) or selected != expected:
        raise ValueError("checkpoint selection does not match its validation records")


def acceptance_decision(
    intervals: dict,
    shortage_deltas: dict,
    *,
    all_safe: bool,
    all_eligible: bool,
) -> dict:
    """Apply the predeclared final claim to CPU artifact summaries only."""
    if type(all_safe) is not bool or type(all_eligible) is not bool:
        raise ValueError("safety and eligibility must be explicit booleans")
    if not isinstance(intervals, dict):
        raise ValueError("decision requires interval mappings")
    for metric in DECISION_METRICS:
        interval = intervals.get(metric)
        if not isinstance(interval, dict) or set(interval) != {"mean", "lower", "upper"}:
            raise ValueError("decision requires complete mean/lower/upper interval fields")
        for key, value in interval.items():
            _finite_number(value, f"{metric} interval {key}")
        if interval["lower"] > interval["upper"]:
            raise ValueError("interval endpoints are reversed")
    _validate_shortages(shortage_deltas, "decision shortage deltas")
    gates = {
        "critical_and_power_constraints": all_safe,
        "all_validation_selections_eligible": all_eligible,
        "reward_ci_above_practical_threshold": (
            intervals["episode_reward"]["lower"] > REWARD_THRESHOLD
        ),
        "fairness_noninferiority": intervals["fairness"]["lower"] >= -SERVICE_MARGIN,
        "utilization_noninferiority": intervals["utilization"]["lower"] >= -SERVICE_MARGIN,
        "severity_service_guardrails": all(
            row["fairness"] >= -SERVICE_MARGIN and row["utilization"] >= -SERVICE_MARGIN
            for row in shortage_deltas.values()
        ),
        "severity_reward_guardrail": all(
            row["episode_reward"] >= 0 for row in shortage_deltas.values()
        ),
    }
    return {"passed": all(gates.values()), "gates": gates}


def _bootstrap_inputs(
    deltas: torch.Tensor, n_resamples: int, seed: int
) -> tuple[Any, Any, Any, Any]:
    if not isinstance(deltas, torch.Tensor):
        raise TypeError("paired deltas must be a torch.Tensor")
    if deltas.ndim != 5 or deltas.shape[1] != 3 or any(size < 1 for size in deltas.shape):
        raise BootstrapError(
            "paired deltas must have shape [run, severity, seed, base_draw, metric], "
            "with exactly three severities"
        )
    if not deltas.is_floating_point():
        raise BootstrapError("paired deltas must use a floating-point dtype")
    if deltas.device.type != "cuda" or deltas.device.index != 0:
        raise BootstrapError("paired deltas must remain on required CUDA device cuda:0")
    _validate_positive_int("n_resamples", n_resamples)
    _validate_seed(seed)
    cp = _import_cupy()
    values = torch_cuda_to_cupy(deltas.detach().contiguous())
    if not bool(cp.isfinite(values).all()):
        raise BootstrapError("paired deltas must contain only finite values")
    # Reduce severity only after promoting to float64. Coupled variants form one cluster.
    clusters = values.astype(cp.float64).mean(axis=1)
    runs, strata, base_draws, _ = clusters.shape
    random = cp.random.RandomState(seed)
    run_samples = random.randint(0, runs, size=(n_resamples, runs), dtype=cp.int32)
    # There is deliberately no run dimension: scenario uncertainty is shared across runs.
    scenario_samples = random.randint(
        0, base_draws, size=(n_resamples, strata, base_draws), dtype=cp.int32
    )
    return cp, clusters, run_samples, scenario_samples


@torch.inference_mode()
def hierarchical_paired_bootstrap(
    deltas: torch.Tensor,
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> torch.Tensor:
    """Return CUDA ``[mean, lower, upper]`` rows for five-axis paired differences.

    Input is ``[training run, severity, scenario seed, base draw, metric]``.
    Training runs and within-seed base clusters are independently resampled; every sampled
    run uses the same scenario samples and every cluster retains its three severities.
    The caller establishes matched controller inputs before forming paired differences.
    """
    cp, clusters, run_samples, scenario_samples = _bootstrap_inputs(deltas, n_resamples, seed)
    _, strata, _, metric_count = clusters.shape
    stratum_indices = cp.arange(strata, dtype=cp.int32)[None, None, :, None]
    replicates = cp.empty((n_resamples, metric_count), dtype=cp.float64)
    for start in range(0, n_resamples, _BOOTSTRAP_CHUNK):
        end = min(start + _BOOTSTRAP_CHUNK, n_resamples)
        values = clusters[
            run_samples[start:end, :, None, None],
            stratum_indices,
            scenario_samples[start:end, None, :, :],
            :,
        ]
        replicates[start:end] = values.mean(axis=(1, 2, 3))
    interval = _linear_percentile_rows(replicates, percentiles=CI_PERCENTILES, cp=cp)
    mean = clusters.mean(axis=(0, 1, 2))
    return torch.from_dlpack(cp.stack((mean, interval[0], interval[1]), axis=0))


@torch.inference_mode()
def conditional_paired_bootstrap(
    deltas: torch.Tensor,
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> torch.Tensor:
    """Return CUDA ``[run, mean/lower/upper, metric]`` conditional scenario intervals.

    These reuse the hierarchical procedure's scenario draws, keeping each trained run
    fixed. Their intervals quantify scenario uncertainty only, not training uncertainty.
    """
    cp, clusters, _, scenario_samples = _bootstrap_inputs(deltas, n_resamples, seed)
    runs, strata, _, metric_count = clusters.shape
    stratum_indices = cp.arange(strata, dtype=cp.int32)[None, :, None]
    summaries = []
    for run in range(runs):
        replicates = cp.empty((n_resamples, metric_count), dtype=cp.float64)
        for start in range(0, n_resamples, _BOOTSTRAP_CHUNK):
            end = min(start + _BOOTSTRAP_CHUNK, n_resamples)
            values = clusters[run, stratum_indices, scenario_samples[start:end], :]
            replicates[start:end] = values.mean(axis=(1, 2))
        interval = _linear_percentile_rows(replicates, percentiles=CI_PERCENTILES, cp=cp)
        mean = clusters[run].mean(axis=(0, 1))
        summaries.append(cp.stack((mean, interval[0], interval[1]), axis=0))
    return torch.from_dlpack(cp.stack(summaries, axis=0))

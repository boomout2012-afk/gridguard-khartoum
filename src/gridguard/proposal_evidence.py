"""Portable validation for the additive, three-method proposal experiment.

NumPy is used only to reconstruct small host-side evidence arrays after the one
final scientific transfer. Simulation, fitting, inference and bootstrap run on CUDA.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from gridguard.proposal_provenance import PROPOSAL_SUPPLEMENT_SHA256
from gridguard.resources import resource_path

CONTROLLER_NAMES = ("Fixed Rotation", "PPO", "Optimization")
SHORTAGE_LEVELS = ("mild", "moderate", "severe")
EVALUATION_SEEDS = tuple(range(91001, 91011))
BATCH_SIZE = 32
METRIC_NAMES = (
    "episode_reward",
    "demand_mwh",
    "available_mwh",
    "served_mwh",
    "curtailed_mwh",
    "critical_service_ratio",
    "utilization",
    "fairness",
    "switching_rate",
    "oversubscription",
    "power_balance_violation_rate",
    "critical_outage_locality_hours",
    "blackout_locality_hours",
    "critical_interruption_locality_hours",
    "blackout_duration_range_hours",
    "blackout_duration_mean_pairwise_gap_hours",
    "critical_interruption_duration_range_hours",
    "critical_interruption_duration_mean_pairwise_gap_hours",
    "longest_blackout_hours",
    "longest_critical_interruption_hours",
)
LOCALITY_METRIC_NAMES = (
    "locality_blackout_hours",
    "locality_critical_interruption_hours",
    "locality_off_hours",
    "locality_longest_blackout_hours",
    "locality_longest_critical_interruption_hours",
)
PROTOCOL = {
    "evaluation_seeds": list(EVALUATION_SEEDS),
    "batch_size": BATCH_SIZE,
    "shortage_levels": list(SHORTAGE_LEVELS),
    "independent_base_draws": 320,
    "matched_scenario_conditions": 960,
    "controller_evaluations": 2880,
    "horizon_hours": 24,
    "bootstrap_replicates": 10000,
    "bootstrap_seed": 20261022,
    "bootstrap_unit": "base-draw cluster within seed; all three shortage variants retained",
    "intervals": "pointwise percentile 95%; unadjusted for multiple comparisons",
    "evaluation_role": "newly declared synthetic extension; not independently custodied",
}
PAIR_INDICES = ((1, 0), (2, 0), (1, 2))
FORECAST_NAMES = ("trailing_mean_3h", "persistence_1h", "ridge_ar3")


def canonical_proposal_bytes(payload: dict) -> bytes:
    return (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _array(value, shape, name):
    try:
        values = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} is not a numerical array") from error
    if values.shape != shape or not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return values


def _close(value, expected, name, tolerance=1e-8):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    if not math.isfinite(value) or not math.isclose(
        value, float(expected), abs_tol=tolerance, rel_tol=1e-9
    ):
        raise ValueError(f"{name} does not reconstruct from evidence")


def validate_proposal_supplement(payload: dict) -> dict:
    """Check labels, pairing, counts, physical identities and every displayed mean."""
    expected_fields = {
        "schema",
        "artifact_id",
        "protocol",
        "controller_names",
        "metric_names",
        "locality_metric_names",
        "episode_metric_cube",
        "locality_duration_cube",
        "controller_means",
        "shortage_means",
        "locality_duration_means",
        "paired_inference",
        "forecast",
        "source_sha256",
        "scenario_sha256_cube",
        "runtime",
        "limitations",
        "method_definitions",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise ValueError("proposal supplement fields mismatch")
    if payload["schema"] != {"name": "gridguard.proposal_supplement", "version": 1}:
        raise ValueError("proposal supplement schema mismatch")
    if payload["artifact_id"] != "gridguard-proposal-supplement-v1":
        raise ValueError("proposal supplement identity mismatch")
    if payload["protocol"] != PROTOCOL:
        raise ValueError("proposal protocol/counts mismatch")
    for key, expected in (
        ("controller_names", CONTROLLER_NAMES),
        ("metric_names", METRIC_NAMES),
        ("locality_metric_names", LOCALITY_METRIC_NAMES),
    ):
        if payload[key] != list(expected):
            raise ValueError(f"{key} mismatch")
    cube = _array(payload["episode_metric_cube"], (3, 3, 10, 32, len(METRIC_NAMES)), "metrics")
    duration = _array(payload["locality_duration_cube"], (3, 3, 10, 32, 7, 5), "duration")
    if (duration < 0).any() or (duration > 24).any() or (duration != np.floor(duration)).any():
        raise ValueError("locality durations must be whole modeled hours from 0 to 24")
    idx = {name: i for i, name in enumerate(METRIC_NAMES)}
    for col in (
        "critical_service_ratio",
        "utilization",
        "fairness",
        "switching_rate",
        "power_balance_violation_rate",
    ):
        if (cube[..., idx[col]] < 0).any() or (cube[..., idx[col]] > 1 + 1e-6).any():
            raise ValueError(f"{col} must be a ratio")
    if not np.allclose(
        cube[..., idx["served_mwh"]] + cube[..., idx["curtailed_mwh"]],
        cube[..., idx["demand_mwh"]],
        atol=0.002,
        rtol=0,
    ):
        raise ValueError("energy identity mismatch")
    for field in ("demand_mwh", "available_mwh"):
        if not np.array_equal(
            cube[0, ..., idx[field]], cube[1, ..., idx[field]]
        ) or not np.array_equal(cube[0, ..., idx[field]], cube[2, ..., idx[field]]):
            raise ValueError("controller inputs are not matched")
    for column, prefix in ((0, "blackout"), (1, "critical_interruption")):
        hours = duration[..., column]
        total = hours.sum(-1)
        if not np.array_equal(total, cube[..., idx[f"{prefix}_locality_hours"]]):
            raise ValueError("duration total mismatch")
        span = hours.max(-1) - hours.min(-1)
        gap = np.abs(hours[..., :, None] - hours[..., None, :]).sum((-1, -2)) / 42
        for field, expected in (
            (f"{prefix}_duration_range_hours", span),
            (f"{prefix}_duration_mean_pairwise_gap_hours", gap),
        ):
            if not np.allclose(cube[..., idx[field]], expected, rtol=0, atol=2e-6):
                raise ValueError("duration disparity mismatch")
    if not np.array_equal(
        cube[..., idx["critical_outage_locality_hours"]],
        cube[..., idx["critical_interruption_locality_hours"]],
    ):
        raise ValueError("critical interruption differs from canonical outage definition")
    for col, totalcol, name in (
        (3, 0, "longest_blackout_hours"),
        (4, 1, "longest_critical_interruption_hours"),
    ):
        if (duration[..., col] > duration[..., totalcol]).any() or not np.array_equal(
            duration[..., col].max(-1), cube[..., idx[name]]
        ):
            raise ValueError("interruption streak mismatch")
    for ci, name in enumerate(CONTROLLER_NAMES):
        if set(payload["controller_means"][name]) != set(METRIC_NAMES):
            raise ValueError("controller summary field mismatch")
        for mi, metric in enumerate(METRIC_NAMES):
            _close(payload["controller_means"][name][metric], cube[ci, ..., mi].mean(), metric)
            for si, severity in enumerate(SHORTAGE_LEVELS):
                _close(
                    payload["shortage_means"][severity][name][metric],
                    cube[ci, si, ..., mi].mean(),
                    metric,
                )
        expected = duration[ci].mean(axis=(0, 1, 2))
        reported = _array(payload["locality_duration_means"][name], (7, 5), "locality means")
        if not np.allclose(reported, expected, atol=1e-8, rtol=0):
            raise ValueError("locality duration means mismatch")
    expected_pairs = {f"{CONTROLLER_NAMES[a]} - {CONTROLLER_NAMES[b]}" for a, b in PAIR_INDICES}
    if set(payload["paired_inference"]) != expected_pairs:
        raise ValueError("paired comparison labels mismatch")
    for a, b in PAIR_INDICES:
        pair = payload["paired_inference"][f"{CONTROLLER_NAMES[a]} - {CONTROLLER_NAMES[b]}"]
        if set(pair) != set(METRIC_NAMES):
            raise ValueError("paired inference metrics mismatch")
        for mi, metric in enumerate(METRIC_NAMES):
            item = pair[metric]
            if set(item) != {"mean_delta", "ci95_lower", "ci95_upper"}:
                raise ValueError("interval fields mismatch")
            _close(item["mean_delta"], (cube[a, ..., mi] - cube[b, ..., mi]).mean(), metric)
            if (
                not all(
                    isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v)
                    for v in item.values()
                )
                or item["ci95_lower"] > item["ci95_upper"]
            ):
                raise ValueError("invalid paired interval")
    forecast = payload["forecast"]
    if (
        forecast["training_seeds"] != [42, 43, 44]
        or forecast["training_trace_count"] != 768
        or forecast["target_hours"] != list(range(3, 24))
        or forecast["ridge_penalty"] != 0.001
    ):
        raise ValueError("forecast protocol mismatch")
    stats = _array(forecast["sufficient_statistics"], (3, 10, 2, 3), "forecast statistics")
    if (stats < 0).any() or set(forecast["methods"]) != set(FORECAST_NAMES):
        raise ValueError("forecast statistics/method mismatch")
    for mi, method in enumerate(FORECAST_NAMES):
        for ai, aggregation in enumerate(("locality_sector", "system_total")):
            counts, sae, sse = stats[mi, :, ai, :].sum(0)
            expected_count = 320 * 21 * (35 if ai == 0 else 1)
            _close(counts, expected_count, "forecast target count")
            _close(forecast["methods"][method][aggregation]["mae_mw"], sae / counts, "forecast MAE")
            _close(
                forecast["methods"][method][aggregation]["rmse_mw"],
                math.sqrt(sse / counts),
                "forecast RMSE",
            )
    for name, shape in (
        ("coefficients_mw", (7, 5, 3)),
        ("feature_means_mw", (7, 5, 3)),
        ("feature_scales_mw", (7, 5, 3)),
        ("target_means_mw", (7, 5)),
    ):
        values = _array(forecast["model"][name], shape, name)
        if name == "feature_scales_mw" and (values <= 0).any():
            raise ValueError("forecast scales must be positive")
    hashes = payload["scenario_sha256_cube"]
    if (
        not isinstance(hashes, list)
        or len(hashes) != 3
        or any(len(s) != 10 or any(len(rows) != 32 for rows in s) for s in hashes)
    ):
        raise ValueError("scenario digest dimensions mismatch")
    all_hashes = [digest for severity in hashes for seed in severity for digest in seed]
    all_hashes.extend(payload["source_sha256"].values())
    if any(
        not isinstance(d, str) or len(d) != 64 or any(c not in "0123456789abcdef" for c in d)
        for d in all_hashes
    ):
        raise ValueError("invalid provenance digest")
    return payload


def load_proposal_supplement(path: str | Path | None = None) -> dict:
    """Fail closed on altered supplementary evidence, then reconstruct its claims."""
    path = (
        resource_path("data", "evidence", "proposal-supplement.json")
        if path is None
        else Path(path)
    )
    encoded = path.read_bytes()
    if hashlib.sha256(encoded).hexdigest() != PROPOSAL_SUPPLEMENT_SHA256:
        raise ValueError("proposal supplement SHA-256 mismatch")
    payload = json.loads(encoded)
    if canonical_proposal_bytes(payload) != encoded:
        raise ValueError("proposal supplement is not canonical JSON")
    return validate_proposal_supplement(payload)


def verify_proposal_implementation_sources(repository_root: str | Path) -> None:
    """Verify the current experiment source bytes independently of historical evidence."""
    root = Path(repository_root).resolve()
    payload = load_proposal_supplement(root / "data/evidence/proposal-supplement.json")
    for relative, expected in payload["source_sha256"].items():
        candidate = (root / relative).resolve()
        if not candidate.is_relative_to(root) or not candidate.is_file():
            raise ValueError(f"proposal implementation source unavailable: {relative}")
        if hashlib.sha256(candidate.read_bytes()).hexdigest() != expected:
            raise ValueError(f"proposal implementation source SHA-256 mismatch: {relative}")


def verify_proposal_inference_on_cuda(payload: dict) -> None:
    """Reconstruct all three paired interval sets from the recorded episode cube."""
    import torch

    from gridguard.statistics import stratified_paired_bootstrap

    validate_proposal_supplement(payload)
    cube = torch.tensor(payload["episode_metric_cube"], dtype=torch.float64, device="cuda:0")
    actual = []
    expected = []
    for a, b in PAIR_INDICES:
        result = stratified_paired_bootstrap(
            cube[a] - cube[b],
            metric_names=METRIC_NAMES,
            replicates=PROTOCOL["bootstrap_replicates"],
            seed=PROTOCOL["bootstrap_seed"],
        )
        actual.append(result.as_tensor())
        recorded = payload["paired_inference"][f"{CONTROLLER_NAMES[a]} - {CONTROLLER_NAMES[b]}"]
        expected.append(
            [
                [recorded[metric][field] for metric in METRIC_NAMES]
                for field in ("mean_delta", "ci95_lower", "ci95_upper")
            ]
        )
    expected_tensor = torch.tensor(expected, dtype=torch.float64, device="cuda:0")
    if not bool(
        torch.isclose(torch.stack(actual), expected_tensor, rtol=0, atol=1e-10).all().item()
    ):
        raise ValueError("proposal paired intervals do not reconstruct on CUDA")

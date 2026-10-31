"""Trusted CUDA reconstruction of persisted paired confidence intervals."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Final

import numpy as np
import torch

from gridguard.statistics import (
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_BOOTSTRAP_SEED,
    BootstrapError,
    stratified_paired_bootstrap,
)

PAIRED_INFERENCE_ABSOLUTE_TOLERANCE: Final = 1e-12

_LOCKED_BOOTSTRAP_PROTOCOL: Final = {
    "confidence_level": 0.95,
    "method": "deterministic seed-stratified paired cluster percentile bootstrap",
    "percentile_method": "linear",
    "replicates": DEFAULT_BOOTSTRAP_REPLICATES,
    "seed": DEFAULT_BOOTSTRAP_SEED,
}
_ENDPOINT_ORDER: Final = ("mean_delta", "ci95_lower", "ci95_upper")


class PairedInferenceError(ValueError):
    """Raised when trusted paired inference cannot be reconstructed exactly."""


def _require_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise PairedInferenceError(f"{field} must be an object")
    return value


def _require_locked_protocol(
    protocol: object,
    *,
    severity_order: tuple[str, ...],
    seed_order: tuple[int, ...],
    episodes_per_cell: int,
) -> tuple[int, int]:
    root = _require_mapping(protocol, field="protocol")
    bootstrap_value = root.get("bootstrap")
    if not isinstance(bootstrap_value, Mapping):
        raise PairedInferenceError(
            "protocol.bootstrap does not match the locked bootstrap protocol"
        )
    bootstrap = bootstrap_value
    if set(bootstrap) != set(_LOCKED_BOOTSTRAP_PROTOCOL) or any(
        type(bootstrap.get(field)) is not type(expected) or bootstrap.get(field) != expected
        for field, expected in _LOCKED_BOOTSTRAP_PROTOCOL.items()
    ):
        raise PairedInferenceError(
            "protocol.bootstrap does not match the locked bootstrap protocol"
        )

    locked_order = {
        "episodes_per_cell": episodes_per_cell,
        "seeds": list(seed_order),
        "shortage_levels": list(severity_order),
    }
    if any(
        type(root.get(field)) is not type(expected) or root.get(field) != expected
        for field, expected in locked_order.items()
    ):
        raise PairedInferenceError("protocol does not match the locked paired-row order")
    return int(bootstrap["replicates"]), int(bootstrap["seed"])


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PairedInferenceError(f"{field} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise PairedInferenceError(f"{field} must be a finite number")
    return normalized


def _reconstruct_delta_cube(
    paired_rows: object,
    *,
    severity_order: tuple[str, ...],
    seed_order: tuple[int, ...],
    episodes_per_cell: int,
    inference_metric_order: tuple[str, ...],
) -> np.ndarray[tuple[int, int, int, int], np.dtype[np.float64]]:
    if type(paired_rows) is not list:
        raise PairedInferenceError("paired_rows must be an array")
    expected_count = len(severity_order) * len(seed_order) * episodes_per_cell
    if len(paired_rows) != expected_count:
        raise PairedInferenceError(f"paired_rows must contain exactly {expected_count} rows")

    cube = np.empty(
        (
            len(severity_order),
            len(seed_order),
            episodes_per_cell,
            len(inference_metric_order),
        ),
        dtype=np.float64,
    )
    position = 0
    for severity_index, shortage_level in enumerate(severity_order):
        for seed_index, seed in enumerate(seed_order):
            for batch_index in range(episodes_per_cell):
                row = _require_mapping(paired_rows[position], field=f"paired_rows[{position}]")
                if (
                    type(row.get("shortage_level")) is not str
                    or row.get("shortage_level") != shortage_level
                    or type(row.get("seed")) is not int
                    or row.get("seed") != seed
                    or type(row.get("batch_index")) is not int
                    or row.get("batch_index") != batch_index
                ):
                    raise PairedInferenceError(
                        f"paired_rows[{position}] is outside the locked protocol order"
                    )
                delta = _require_mapping(
                    row.get("paired_delta"),
                    field=f"paired_rows[{position}].paired_delta",
                )
                for metric_index, metric in enumerate(inference_metric_order):
                    cube[severity_index, seed_index, batch_index, metric_index] = _finite_float(
                        delta.get(metric),
                        field=f"paired_rows[{position}].paired_delta.{metric}",
                    )
                position += 1
    return cube


def _reported_endpoints(
    paired_inference: object,
    *,
    inference_metric_order: tuple[str, ...],
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    inference = _require_mapping(paired_inference, field="paired_inference")
    endpoints = np.empty((len(_ENDPOINT_ORDER), len(inference_metric_order)), dtype=np.float64)
    for metric_index, metric in enumerate(inference_metric_order):
        item = _require_mapping(inference.get(metric), field=f"paired_inference.{metric}")
        for endpoint_index, endpoint in enumerate(_ENDPOINT_ORDER):
            endpoints[endpoint_index, metric_index] = _finite_float(
                item.get(endpoint),
                field=f"paired_inference.{metric}.{endpoint}",
            )
    return endpoints


def _to_cuda_zero(values: np.ndarray[tuple[int, ...], np.dtype[np.float64]]) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("required CUDA device cuda:0 is unavailable")
    tensor = torch.from_numpy(np.ascontiguousarray(values, dtype=np.float64)).to(
        device="cuda:0",
        dtype=torch.float64,
    )
    if tensor.device.type != "cuda" or tensor.device.index != 0:
        raise RuntimeError("paired inference tensor did not reach required CUDA device cuda:0")
    return tensor


def verify_paired_inference_reconstruction(
    *,
    paired_rows: object,
    paired_inference: object,
    protocol: object,
    severity_order: tuple[str, ...],
    seed_order: tuple[int, ...],
    episodes_per_cell: int,
    inference_metric_order: tuple[str, ...],
) -> float:
    """Recompute every trusted endpoint on ``cuda:0`` and return its maximum error."""

    replicates, seed = _require_locked_protocol(
        protocol,
        severity_order=severity_order,
        seed_order=seed_order,
        episodes_per_cell=episodes_per_cell,
    )
    delta_cube = _reconstruct_delta_cube(
        paired_rows,
        severity_order=severity_order,
        seed_order=seed_order,
        episodes_per_cell=episodes_per_cell,
        inference_metric_order=inference_metric_order,
    )
    reported_endpoints = _reported_endpoints(
        paired_inference,
        inference_metric_order=inference_metric_order,
    )
    try:
        cuda_deltas = _to_cuda_zero(delta_cube)
        cuda_reported = _to_cuda_zero(reported_endpoints)
        reconstructed = stratified_paired_bootstrap(
            cuda_deltas,
            metric_names=inference_metric_order,
            replicates=replicates,
            seed=seed,
        ).as_tensor()
        maximum_absolute_error = float(torch.abs(reconstructed - cuda_reported).amax().item())
    except (BootstrapError, RuntimeError) as error:
        raise PairedInferenceError(
            f"paired inference CUDA reconstruction failed: {error}"
        ) from error

    if maximum_absolute_error > PAIRED_INFERENCE_ABSOLUTE_TOLERANCE:
        raise PairedInferenceError(
            "reported paired inference endpoints do not reconstruct within the declared "
            f"absolute tolerance {PAIRED_INFERENCE_ABSOLUTE_TOLERANCE:.0e}; maximum absolute "
            f"error is {maximum_absolute_error:.17g}"
        )
    return maximum_absolute_error

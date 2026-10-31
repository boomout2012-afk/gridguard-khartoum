"""CUDA/CuPy paired uncertainty estimates for the proposal benchmark."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Final

import torch

DEFAULT_BOOTSTRAP_REPLICATES: Final = 10_000
DEFAULT_BOOTSTRAP_SEED: Final = 20260815
CI_PERCENTILES: Final = (2.5, 97.5)
EXPECTED_SEVERITY_COUNT: Final = 3


class BootstrapError(ValueError):
    """Raised when paired bootstrap inputs or the CUDA runtime are invalid."""


@dataclass(frozen=True, slots=True, eq=False)
class BootstrapIntervals:
    """Device-resident mean deltas and percentile confidence limits."""

    metric_names: tuple[str, ...]
    mean_delta: torch.Tensor
    ci95_lower: torch.Tensor
    ci95_upper: torch.Tensor
    replicates: int
    seed: int

    def __post_init__(self) -> None:
        expected_shape = (len(self.metric_names),)
        tensors = (self.mean_delta, self.ci95_lower, self.ci95_upper)
        if any(not isinstance(value, torch.Tensor) for value in tensors):
            raise TypeError("bootstrap statistics must be torch.Tensor values")
        if any(value.shape != expected_shape for value in tensors):
            raise ValueError("bootstrap statistics must have one value per metric")
        if any(value.device != self.mean_delta.device for value in tensors[1:]):
            raise ValueError("bootstrap statistics must remain on one device")
        if any(value.dtype != torch.float64 for value in tensors):
            raise TypeError("bootstrap statistics must use torch.float64")

    def as_tensor(self) -> torch.Tensor:
        """Return ``[mean, lower, upper]`` rows without leaving CUDA."""

        return torch.stack((self.mean_delta, self.ci95_lower, self.ci95_upper))


def _import_cupy() -> Any:
    try:
        import cupy as cp
    except ImportError as error:  # pragma: no cover - locked dependency in release env
        raise BootstrapError("CuPy is required for GPU bootstrap statistics") from error
    return cp


def torch_cuda_to_cupy(tensor: torch.Tensor) -> Any:
    """Create a zero-copy CuPy view of a contiguous ``cuda:0`` tensor via DLPack."""

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor must be a torch.Tensor")
    if tensor.device.type != "cuda" or tensor.device.index != 0:
        raise BootstrapError("tensor must be on required CUDA device cuda:0")
    if not tensor.is_contiguous():
        raise BootstrapError("tensor must be contiguous for the DLPack boundary")
    cp = _import_cupy()
    try:
        return cp.from_dlpack(tensor.detach())
    except (BufferError, RuntimeError, TypeError) as error:
        raise BootstrapError(f"CUDA DLPack conversion failed: {error}") from error


def _validate_metric_names(metric_names: tuple[str, ...], metric_count: int) -> None:
    if not isinstance(metric_names, tuple):
        raise TypeError("metric_names must be a tuple")
    if len(metric_names) != metric_count:
        raise BootstrapError("metric_names must contain one unique name per metric")
    if any(type(name) is not str or not name for name in metric_names):
        raise BootstrapError("metric_names must contain non-empty strings")
    if len(set(metric_names)) != len(metric_names):
        raise BootstrapError("metric_names must not contain duplicates")


def _validate_positive_int(name: str, value: int) -> None:
    if type(value) is not int or value <= 0:
        raise BootstrapError(f"{name} must be a positive integer")


def _validate_seed(seed: int) -> None:
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise BootstrapError("seed must be an integer between 0 and 2**32 - 1")


def _linear_percentile_rows(values: Any, *, percentiles: tuple[float, ...], cp: Any) -> Any:
    """Return explicit linear percentiles without CuPy's unstable percentile kernel path."""

    sorted_values = cp.sort(values, axis=0)
    sample_count = values.shape[0]
    rows = []
    for percentile in percentiles:
        rank = (sample_count - 1) * (percentile / 100.0)
        lower_index = math.floor(rank)
        upper_index = math.ceil(rank)
        upper_weight = rank - lower_index
        rows.append(
            sorted_values[lower_index]
            + (sorted_values[upper_index] - sorted_values[lower_index]) * upper_weight
        )
    return cp.stack(rows, axis=0)


@torch.inference_mode()
def stratified_paired_bootstrap(
    paired_deltas: torch.Tensor,
    *,
    metric_names: tuple[str, ...],
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> BootstrapIntervals:
    """Bootstrap base-draw clusters within each seed entirely on ``cuda:0``.

    ``paired_deltas`` has shape ``[severity, seed, episode, metric]`` and always
    means ``PPO - Fixed Rotation``. The three severity variants reuse each
    seed/episode base draw, so every replicate resamples episode clusters with
    replacement within each of one or more seed strata and keeps their severity
    variants together.
    """

    if not isinstance(paired_deltas, torch.Tensor):
        raise TypeError("paired_deltas must be a torch.Tensor")
    if (
        paired_deltas.ndim != 4
        or paired_deltas.shape[0] != EXPECTED_SEVERITY_COUNT
        or any(size < 1 for size in paired_deltas.shape)
    ):
        raise BootstrapError(
            "paired_deltas must have shape [severity, seed, episode, metric] "
            "with exactly 3 severities and at least 1 seed"
        )
    if not paired_deltas.is_floating_point():
        raise BootstrapError("paired_deltas must use a floating-point dtype")
    if paired_deltas.device.type != "cuda" or paired_deltas.device.index != 0:
        raise BootstrapError("paired_deltas must remain on required CUDA device cuda:0")
    _validate_metric_names(metric_names, paired_deltas.shape[-1])
    _validate_positive_int("replicates", replicates)
    _validate_seed(seed)

    contiguous = paired_deltas.detach().contiguous()
    cp = _import_cupy()
    values = torch_cuda_to_cupy(contiguous)
    if not bool(cp.isfinite(values).all()):
        raise BootstrapError("paired_deltas must contain only finite values")

    # Float64 accumulation makes the reported uncertainty independent of
    # float32 reduction order while all large arrays remain on the GPU.
    values = values.astype(cp.float64, copy=True)
    _, seed_count, episode_count, _ = values.shape
    cluster_values = values.mean(axis=0)
    random = cp.random.RandomState(seed)
    sample_indices = random.randint(
        0,
        episode_count,
        size=(replicates, seed_count, episode_count),
        dtype=cp.int32,
    )
    seed_indices = cp.arange(seed_count, dtype=cp.int32)[None, :, None]
    replicate_means = cluster_values[seed_indices, sample_indices, :].mean(axis=(1, 2))
    interval = _linear_percentile_rows(
        replicate_means,
        percentiles=CI_PERCENTILES,
        cp=cp,
    )
    mean_delta = cluster_values.mean(axis=(0, 1))
    summary = torch.from_dlpack(cp.stack((mean_delta, interval[0], interval[1]), axis=0))

    return BootstrapIntervals(
        metric_names=metric_names,
        mean_delta=summary[0],
        ci95_lower=summary[1],
        ci95_upper=summary[2],
        replicates=replicates,
        seed=seed,
    )

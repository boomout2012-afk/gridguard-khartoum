"""Strict persisted inputs and CUDA reconstruction for robustness inference."""

from __future__ import annotations

import base64
import binascii
import hashlib
import math
import re
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

import torch

from gridguard.statistics import stratified_paired_bootstrap

_SHA256_PATTERN: Final = re.compile(r"[0-9a-f]{64}")
_AXIS_ORDER: Final = (
    "parameterization",
    "shortage_level",
    "seed",
    "episode",
    "metric",
)


class RobustnessInferenceError(ValueError):
    """Raised when persisted inference inputs cannot prove reported statistics."""


@dataclass(frozen=True, slots=True)
class DeltaCubeContract:
    """Closed axis identities for persisted paired-delta inputs."""

    parameterization_order: tuple[str, ...]
    shortage_level_order: tuple[str, ...]
    seed_order: tuple[int, ...]
    episode_order: tuple[int, ...]
    metric_order: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_string_order(
            "parameterization_order",
            self.parameterization_order,
            minimum_length=2,
        )
        if self.parameterization_order[0] != "base":
            raise RobustnessInferenceError("parameterization_order must begin with 'base'")
        _validate_string_order(
            "shortage_level_order",
            self.shortage_level_order,
            minimum_length=3,
        )
        _validate_integer_order("seed_order", self.seed_order)
        _validate_integer_order("episode_order", self.episode_order)
        if self.episode_order != tuple(range(len(self.episode_order))):
            raise RobustnessInferenceError("episode_order must be contiguous and zero-based")
        _validate_string_order("metric_order", self.metric_order, minimum_length=1)

    @property
    def axis_order(self) -> tuple[str, ...]:
        return _AXIS_ORDER

    @property
    def shape(self) -> tuple[int, ...]:
        return (
            len(self.parameterization_order),
            len(self.shortage_level_order),
            len(self.seed_order),
            len(self.episode_order),
            len(self.metric_order),
        )


def _validate_string_order(name: str, values: object, *, minimum_length: int) -> None:
    if not isinstance(values, tuple) or len(values) < minimum_length:
        raise RobustnessInferenceError(
            f"{name} must be a tuple with at least {minimum_length} entries"
        )
    if any(type(value) is not str or not value for value in values):
        raise RobustnessInferenceError(f"{name} must contain non-empty strings")
    if len(set(values)) != len(values):
        raise RobustnessInferenceError(f"{name} must not contain duplicates")


def _validate_integer_order(name: str, values: object) -> None:
    if not isinstance(values, tuple) or not values:
        raise RobustnessInferenceError(f"{name} must be a non-empty tuple")
    if any(type(value) is not int or value < 0 for value in values):
        raise RobustnessInferenceError(f"{name} must contain non-negative integers")
    if len(set(values)) != len(values):
        raise RobustnessInferenceError(f"{name} must not contain duplicates")


def build_delta_cube_payload(value: object, *, contract: DeltaCubeContract) -> dict[str, Any]:
    """Encode one finite float64 delta cube in canonical little-endian order."""

    if not isinstance(contract, DeltaCubeContract):
        raise TypeError("contract must be DeltaCubeContract")
    # NumPy is intentionally used only after the evaluator's one final host transfer.
    import numpy as np

    array = np.asarray(value, dtype=np.float64)
    if array.shape != contract.shape:
        raise RobustnessInferenceError(f"delta cube must have shape {list(contract.shape)}")
    if not bool(np.isfinite(array).all()):
        raise RobustnessInferenceError("delta cube must contain only finite values")
    little_endian = np.ascontiguousarray(array.astype("<f8", copy=False))
    decoded = little_endian.tobytes(order="C")
    return {
        "encoding": "base64",
        "dtype": "float64-le",
        "shape": list(contract.shape),
        "axis_order": list(contract.axis_order),
        "parameterization_order": list(contract.parameterization_order),
        "shortage_level_order": list(contract.shortage_level_order),
        "seed_order": list(contract.seed_order),
        "episode_order": list(contract.episode_order),
        "metric_order": list(contract.metric_order),
        "decoded_sha256": hashlib.sha256(decoded).hexdigest(),
        "data_base64": base64.b64encode(decoded).decode("ascii"),
    }


def decode_delta_cube_payload(
    payload: object,
    *,
    contract: DeltaCubeContract,
) -> bytes:
    """Validate a closed delta-cube payload and return its exact decoded bytes."""

    if not isinstance(contract, DeltaCubeContract):
        raise TypeError("contract must be DeltaCubeContract")
    if not isinstance(payload, Mapping):
        raise RobustnessInferenceError("delta cube payload must be a mapping")
    expected_fields = {
        "encoding",
        "dtype",
        "shape",
        "axis_order",
        "parameterization_order",
        "shortage_level_order",
        "seed_order",
        "episode_order",
        "metric_order",
        "decoded_sha256",
        "data_base64",
    }
    if set(payload) != expected_fields:
        raise RobustnessInferenceError("delta cube payload fields do not match the contract")
    expected_values = {
        "encoding": "base64",
        "dtype": "float64-le",
        "shape": list(contract.shape),
        "axis_order": list(contract.axis_order),
        "parameterization_order": list(contract.parameterization_order),
        "shortage_level_order": list(contract.shortage_level_order),
        "seed_order": list(contract.seed_order),
        "episode_order": list(contract.episode_order),
        "metric_order": list(contract.metric_order),
    }
    for field, expected in expected_values.items():
        if payload[field] != expected:
            raise RobustnessInferenceError(f"delta cube {field} does not match the contract")
    digest = payload["decoded_sha256"]
    if type(digest) is not str or not _SHA256_PATTERN.fullmatch(digest):
        raise RobustnessInferenceError("delta cube decoded_sha256 must be a lowercase SHA-256")
    encoded = payload["data_base64"]
    if type(encoded) is not str:
        raise RobustnessInferenceError("delta cube data_base64 must be a base64 string")
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise RobustnessInferenceError("delta cube data_base64 is not strict base64") from error
    if base64.b64encode(decoded).decode("ascii") != encoded:
        raise RobustnessInferenceError("delta cube data_base64 is not canonical base64")
    expected_size = math.prod(contract.shape) * 8
    if len(decoded) != expected_size:
        raise RobustnessInferenceError(
            f"delta cube decoded bytes must contain exactly {expected_size} bytes"
        )
    if hashlib.sha256(decoded).hexdigest() != digest:
        raise RobustnessInferenceError("delta cube decoded-byte SHA-256 does not match")
    if any(not math.isfinite(value[0]) for value in struct.iter_unpack("<d", decoded)):
        raise RobustnessInferenceError("delta cube must contain only finite values")
    return decoded


def delta_cube_to_cuda(
    payload: object,
    *,
    contract: DeltaCubeContract,
    device: str | torch.device = "cuda:0",
) -> torch.Tensor:
    """Load validated host bytes onto the required CUDA device for reconstruction."""

    try:
        resolved_device = torch.device(device)
    except (RuntimeError, TypeError) as error:
        raise RobustnessInferenceError(f"invalid reconstruction device: {device!r}") from error
    if resolved_device.type != "cuda" or resolved_device.index != 0:
        raise RobustnessInferenceError("robustness reconstruction requires cuda:0")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RobustnessInferenceError("required CUDA device cuda:0 is unavailable")
    decoded = decode_delta_cube_payload(payload, contract=contract)
    host_buffer = bytearray(decoded)
    host = torch.frombuffer(host_buffer, dtype=torch.float64).reshape(contract.shape)
    return host.to(device=resolved_device)


def reconstruct_bootstrap_intervals(
    delta_cube: torch.Tensor,
    *,
    metric_names: tuple[str, ...],
    replicates: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recompute base, variant, and difference-in-differences interval tensors."""

    if not isinstance(delta_cube, torch.Tensor):
        raise TypeError("delta_cube must be a torch.Tensor")
    if (
        delta_cube.ndim != 5
        or delta_cube.shape[0] < 2
        or delta_cube.shape[1] != 3
        or any(size < 1 for size in delta_cube.shape)
    ):
        raise RobustnessInferenceError(
            "delta_cube must have shape [parameterization, 3, seed, episode, metric]"
        )
    if delta_cube.dtype != torch.float64:
        raise RobustnessInferenceError("delta_cube must use torch.float64")
    if delta_cube.device.type != "cuda" or delta_cube.device.index != 0:
        raise RobustnessInferenceError("delta_cube must remain on required CUDA device cuda:0")
    if len(metric_names) != delta_cube.shape[-1]:
        raise RobustnessInferenceError("metric_names must match the delta cube metric axis")

    base_deltas = delta_cube[0]
    base = stratified_paired_bootstrap(
        base_deltas,
        metric_names=metric_names,
        replicates=replicates,
        seed=seed,
    ).as_tensor()
    variants = []
    differences = []
    for variant_deltas in delta_cube[1:]:
        variants.append(
            stratified_paired_bootstrap(
                variant_deltas,
                metric_names=metric_names,
                replicates=replicates,
                seed=seed,
            ).as_tensor()
        )
        differences.append(
            stratified_paired_bootstrap(
                variant_deltas - base_deltas,
                metric_names=metric_names,
                replicates=replicates,
                seed=seed,
            ).as_tensor()
        )
    return base, torch.stack(variants), torch.stack(differences)

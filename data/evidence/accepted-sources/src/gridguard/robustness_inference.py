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
_BASE_CONTROLLER_AXIS_ORDER: Final = (
    "controller",
    "shortage_level",
    "seed",
    "episode",
    "metric",
)
_POWER_BALANCE_AXIS_ORDER: Final = (
    "parameterization",
    "controller",
    "shortage_level",
    "seed",
    "episode",
)
_SENSITIVITY_CONTROLLER_AXIS_ORDER: Final = (
    "parameterization",
    "controller",
    "shortage_level",
    "seed",
    "episode",
    "metric",
)
_FORECAST_DEMAND_AXIS_ORDER: Final = (
    "seed",
    "episode",
    "hour",
    "locality",
    "sector",
)
RECONSTRUCTION_NUMERIC_TOLERANCE: Final = 1e-12
_PRIMARY_METRIC_ORDER: Final = (
    "critical_outage_locality_hours",
    "fairness",
    "utilization",
)
_SAFETY_METRIC: Final = "power_balance_violation_rate"


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


@dataclass(frozen=True, slots=True)
class BaseControllerCubeContract:
    """Closed axis identities for base per-controller metric inputs."""

    controller_order: tuple[str, ...]
    shortage_level_order: tuple[str, ...]
    seed_order: tuple[int, ...]
    episode_order: tuple[int, ...]
    metric_order: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_string_order("controller_order", self.controller_order, minimum_length=2)
        if len(self.controller_order) != 2:
            raise RobustnessInferenceError("controller_order must contain exactly two controllers")
        _validate_string_order(
            "shortage_level_order",
            self.shortage_level_order,
            minimum_length=3,
        )
        if len(self.shortage_level_order) != 3:
            raise RobustnessInferenceError("shortage_level_order must contain exactly three levels")
        _validate_integer_order("seed_order", self.seed_order)
        _validate_integer_order("episode_order", self.episode_order)
        if self.episode_order != tuple(range(len(self.episode_order))):
            raise RobustnessInferenceError("episode_order must be contiguous and zero-based")
        _validate_string_order("metric_order", self.metric_order, minimum_length=1)

    @property
    def axis_order(self) -> tuple[str, ...]:
        return _BASE_CONTROLLER_AXIS_ORDER

    @property
    def shape(self) -> tuple[int, ...]:
        return (
            len(self.controller_order),
            len(self.shortage_level_order),
            len(self.seed_order),
            len(self.episode_order),
            len(self.metric_order),
        )


@dataclass(frozen=True, slots=True)
class PowerBalanceCubeContract:
    """Closed axis identities for all evaluated power-balance values."""

    parameterization_order: tuple[str, ...]
    controller_order: tuple[str, ...]
    shortage_level_order: tuple[str, ...]
    seed_order: tuple[int, ...]
    episode_order: tuple[int, ...]

    def __post_init__(self) -> None:
        _validate_string_order(
            "parameterization_order",
            self.parameterization_order,
            minimum_length=1,
        )
        if self.parameterization_order[0] != "base":
            raise RobustnessInferenceError("parameterization_order must begin with 'base'")
        _validate_string_order("controller_order", self.controller_order, minimum_length=2)
        if len(self.controller_order) != 2:
            raise RobustnessInferenceError("controller_order must contain exactly two controllers")
        _validate_string_order(
            "shortage_level_order",
            self.shortage_level_order,
            minimum_length=3,
        )
        if len(self.shortage_level_order) != 3:
            raise RobustnessInferenceError("shortage_level_order must contain exactly three levels")
        _validate_integer_order("seed_order", self.seed_order)
        _validate_integer_order("episode_order", self.episode_order)
        if self.episode_order != tuple(range(len(self.episode_order))):
            raise RobustnessInferenceError("episode_order must be contiguous and zero-based")

    @property
    def axis_order(self) -> tuple[str, ...]:
        return _POWER_BALANCE_AXIS_ORDER

    @property
    def shape(self) -> tuple[int, ...]:
        return (
            len(self.parameterization_order),
            len(self.controller_order),
            len(self.shortage_level_order),
            len(self.seed_order),
            len(self.episode_order),
        )


@dataclass(frozen=True, slots=True)
class SensitivityControllerCubeContract:
    """Closed axes for per-controller draws under every sensitivity variant."""

    parameterization_order: tuple[str, ...]
    controller_order: tuple[str, ...]
    shortage_level_order: tuple[str, ...]
    seed_order: tuple[int, ...]
    episode_order: tuple[int, ...]
    metric_order: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_string_order(
            "parameterization_order",
            self.parameterization_order,
            minimum_length=1,
        )
        if "base" in self.parameterization_order:
            raise RobustnessInferenceError(
                "sensitivity parameterization_order must not contain 'base'"
            )
        _validate_string_order("controller_order", self.controller_order, minimum_length=2)
        if len(self.controller_order) != 2:
            raise RobustnessInferenceError("controller_order must contain exactly two controllers")
        _validate_string_order(
            "shortage_level_order",
            self.shortage_level_order,
            minimum_length=3,
        )
        if len(self.shortage_level_order) != 3:
            raise RobustnessInferenceError("shortage_level_order must contain exactly three levels")
        _validate_integer_order("seed_order", self.seed_order)
        _validate_integer_order("episode_order", self.episode_order)
        if self.episode_order != tuple(range(len(self.episode_order))):
            raise RobustnessInferenceError("episode_order must be contiguous and zero-based")
        _validate_string_order("metric_order", self.metric_order, minimum_length=1)

    @property
    def axis_order(self) -> tuple[str, ...]:
        return _SENSITIVITY_CONTROLLER_AXIS_ORDER

    @property
    def shape(self) -> tuple[int, ...]:
        return (
            len(self.parameterization_order),
            len(self.controller_order),
            len(self.shortage_level_order),
            len(self.seed_order),
            len(self.episode_order),
            len(self.metric_order),
        )


@dataclass(frozen=True, slots=True)
class ForecastDemandCubeContract:
    """Closed axes for the demand traces consumed by forecast evaluation."""

    seed_order: tuple[int, ...]
    episode_order: tuple[int, ...]
    hour_order: tuple[int, ...]
    locality_order: tuple[str, ...]
    sector_order: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_integer_order("seed_order", self.seed_order)
        _validate_integer_order("episode_order", self.episode_order)
        _validate_integer_order("hour_order", self.hour_order)
        if self.episode_order != tuple(range(len(self.episode_order))):
            raise RobustnessInferenceError("episode_order must be contiguous and zero-based")
        if self.hour_order != tuple(range(len(self.hour_order))):
            raise RobustnessInferenceError("hour_order must be contiguous and zero-based")
        _validate_string_order("locality_order", self.locality_order, minimum_length=1)
        _validate_string_order("sector_order", self.sector_order, minimum_length=1)

    @property
    def axis_order(self) -> tuple[str, ...]:
        return _FORECAST_DEMAND_AXIS_ORDER

    @property
    def shape(self) -> tuple[int, ...]:
        return (
            len(self.seed_order),
            len(self.episode_order),
            len(self.hour_order),
            len(self.locality_order),
            len(self.sector_order),
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


def _build_cube_payload(
    value: object,
    *,
    shape: tuple[int, ...],
    metadata: Mapping[str, object],
    label: str,
    numpy_dtype: str = "<f8",
    dtype_label: str = "float64-le",
) -> dict[str, Any]:
    # NumPy is intentionally used only after the evaluator's one final host transfer.
    import numpy as np

    untyped = np.asarray(value)
    if np.iscomplexobj(untyped):
        raise RobustnessInferenceError(f"{label} must contain real values")
    array = np.asarray(untyped, dtype=np.dtype(numpy_dtype))
    if array.shape != shape:
        raise RobustnessInferenceError(f"{label} must have shape {list(shape)}")
    if not bool(np.isfinite(array).all()):
        raise RobustnessInferenceError(f"{label} must contain only finite values")
    little_endian = np.ascontiguousarray(array.astype(numpy_dtype, copy=False))
    decoded = little_endian.tobytes(order="C")
    return {
        "encoding": "base64",
        "dtype": dtype_label,
        "shape": list(shape),
        **metadata,
        "decoded_sha256": hashlib.sha256(decoded).hexdigest(),
        "data_base64": base64.b64encode(decoded).decode("ascii"),
    }


def _matches_metadata_contract(value: object, expected: object) -> bool:
    """Compare JSON metadata without accepting bool/int/float equality coercions."""

    if type(value) is not type(expected):
        return False
    if isinstance(expected, list):
        return len(value) == len(expected) and all(
            _matches_metadata_contract(actual, required)
            for actual, required in zip(value, expected, strict=True)
        )
    return value == expected


def _decode_cube_payload(
    payload: object,
    *,
    shape: tuple[int, ...],
    metadata: Mapping[str, object],
    label: str,
    dtype_label: str = "float64-le",
    item_size: int = 8,
    unpack_format: str = "<d",
) -> bytes:
    if not isinstance(payload, Mapping):
        raise RobustnessInferenceError(f"{label} payload must be a mapping")
    expected_fields = {
        "encoding",
        "dtype",
        "shape",
        *metadata,
        "decoded_sha256",
        "data_base64",
    }
    if set(payload) != expected_fields:
        raise RobustnessInferenceError(f"{label} payload fields do not match the contract")
    expected_values = {
        "encoding": "base64",
        "dtype": dtype_label,
        "shape": list(shape),
        **metadata,
    }
    for field, expected in expected_values.items():
        if not _matches_metadata_contract(payload[field], expected):
            raise RobustnessInferenceError(f"{label} {field} does not match the contract")
    digest = payload["decoded_sha256"]
    if type(digest) is not str or not _SHA256_PATTERN.fullmatch(digest):
        raise RobustnessInferenceError(f"{label} decoded_sha256 must be a lowercase SHA-256")
    encoded = payload["data_base64"]
    if type(encoded) is not str:
        raise RobustnessInferenceError(f"{label} data_base64 must be a base64 string")
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise RobustnessInferenceError(f"{label} data_base64 is not strict base64") from error
    if base64.b64encode(decoded).decode("ascii") != encoded:
        raise RobustnessInferenceError(f"{label} data_base64 is not canonical base64")
    expected_size = math.prod(shape) * item_size
    if len(decoded) != expected_size:
        raise RobustnessInferenceError(
            f"{label} decoded bytes must contain exactly {expected_size} bytes"
        )
    if hashlib.sha256(decoded).hexdigest() != digest:
        raise RobustnessInferenceError(f"{label} decoded-byte SHA-256 does not match")
    if any(not math.isfinite(value[0]) for value in struct.iter_unpack(unpack_format, decoded)):
        raise RobustnessInferenceError(f"{label} must contain only finite values")
    return decoded


def _cube_to_cuda(
    decoded: bytes,
    *,
    shape: tuple[int, ...],
    device: str | torch.device = "cuda:0",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    try:
        resolved_device = torch.device(device)
    except (RuntimeError, TypeError) as error:
        raise RobustnessInferenceError(f"invalid reconstruction device: {device!r}") from error
    if resolved_device.type != "cuda" or resolved_device.index != 0:
        raise RobustnessInferenceError("robustness reconstruction requires cuda:0")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RobustnessInferenceError("required CUDA device cuda:0 is unavailable")
    host_buffer = bytearray(decoded)
    host = torch.frombuffer(host_buffer, dtype=dtype).reshape(shape)
    try:
        return host.to(device=resolved_device)
    except RuntimeError as error:
        raise RobustnessInferenceError(f"CUDA reconstruction transfer failed: {error}") from error


def _delta_metadata(contract: DeltaCubeContract) -> dict[str, object]:
    return {
        "axis_order": list(contract.axis_order),
        "parameterization_order": list(contract.parameterization_order),
        "shortage_level_order": list(contract.shortage_level_order),
        "seed_order": list(contract.seed_order),
        "episode_order": list(contract.episode_order),
        "metric_order": list(contract.metric_order),
    }


def _base_controller_metadata(contract: BaseControllerCubeContract) -> dict[str, object]:
    return {
        "axis_order": list(contract.axis_order),
        "controller_order": list(contract.controller_order),
        "shortage_level_order": list(contract.shortage_level_order),
        "seed_order": list(contract.seed_order),
        "episode_order": list(contract.episode_order),
        "metric_order": list(contract.metric_order),
    }


def _power_balance_metadata(contract: PowerBalanceCubeContract) -> dict[str, object]:
    return {
        "axis_order": list(contract.axis_order),
        "parameterization_order": list(contract.parameterization_order),
        "controller_order": list(contract.controller_order),
        "shortage_level_order": list(contract.shortage_level_order),
        "seed_order": list(contract.seed_order),
        "episode_order": list(contract.episode_order),
    }


def _sensitivity_controller_metadata(
    contract: SensitivityControllerCubeContract,
) -> dict[str, object]:
    return {
        "axis_order": list(contract.axis_order),
        "parameterization_order": list(contract.parameterization_order),
        "controller_order": list(contract.controller_order),
        "shortage_level_order": list(contract.shortage_level_order),
        "seed_order": list(contract.seed_order),
        "episode_order": list(contract.episode_order),
        "metric_order": list(contract.metric_order),
    }


def _forecast_demand_metadata(contract: ForecastDemandCubeContract) -> dict[str, object]:
    return {
        "axis_order": list(contract.axis_order),
        "seed_order": list(contract.seed_order),
        "episode_order": list(contract.episode_order),
        "hour_order": list(contract.hour_order),
        "locality_order": list(contract.locality_order),
        "sector_order": list(contract.sector_order),
    }


def build_delta_cube_payload(value: object, *, contract: DeltaCubeContract) -> dict[str, Any]:
    """Encode one finite float64 delta cube in canonical little-endian order."""

    if not isinstance(contract, DeltaCubeContract):
        raise TypeError("contract must be DeltaCubeContract")
    return _build_cube_payload(
        value,
        shape=contract.shape,
        metadata=_delta_metadata(contract),
        label="delta cube",
    )


def build_base_controller_cube_payload(
    value: object,
    *,
    contract: BaseControllerCubeContract,
) -> dict[str, Any]:
    """Encode base per-controller metrics without losing individual draws."""

    if not isinstance(contract, BaseControllerCubeContract):
        raise TypeError("contract must be BaseControllerCubeContract")
    return _build_cube_payload(
        value,
        shape=contract.shape,
        metadata=_base_controller_metadata(contract),
        label="base controller cube",
    )


def build_power_balance_cube_payload(
    value: object,
    *,
    contract: PowerBalanceCubeContract,
) -> dict[str, Any]:
    """Encode per-draw power-balance values for every parameterization."""

    if not isinstance(contract, PowerBalanceCubeContract):
        raise TypeError("contract must be PowerBalanceCubeContract")
    return _build_cube_payload(
        value,
        shape=contract.shape,
        metadata=_power_balance_metadata(contract),
        label="power balance cube",
    )


def build_sensitivity_controller_cube_payload(
    value: object,
    *,
    contract: SensitivityControllerCubeContract,
) -> dict[str, Any]:
    """Encode all sensitivity per-controller metric draws without aggregation."""

    if not isinstance(contract, SensitivityControllerCubeContract):
        raise TypeError("contract must be SensitivityControllerCubeContract")
    return _build_cube_payload(
        value,
        shape=contract.shape,
        metadata=_sensitivity_controller_metadata(contract),
        label="sensitivity controller cube",
    )


def build_forecast_demand_cube_payload(
    value: object,
    *,
    contract: ForecastDemandCubeContract,
) -> dict[str, Any]:
    """Encode canonical per-draw demand traces used by both forecast methods."""

    if not isinstance(contract, ForecastDemandCubeContract):
        raise TypeError("contract must be ForecastDemandCubeContract")
    return _build_cube_payload(
        value,
        shape=contract.shape,
        metadata=_forecast_demand_metadata(contract),
        label="forecast demand cube",
        numpy_dtype="<f4",
        dtype_label="float32-le",
    )


def decode_delta_cube_payload(
    payload: object,
    *,
    contract: DeltaCubeContract,
) -> bytes:
    """Validate a closed delta-cube payload and return its exact decoded bytes."""

    if not isinstance(contract, DeltaCubeContract):
        raise TypeError("contract must be DeltaCubeContract")
    return _decode_cube_payload(
        payload,
        shape=contract.shape,
        metadata=_delta_metadata(contract),
        label="delta cube",
    )


def decode_base_controller_cube_payload(
    payload: object,
    *,
    contract: BaseControllerCubeContract,
) -> bytes:
    """Validate the base per-controller metric cube."""

    if not isinstance(contract, BaseControllerCubeContract):
        raise TypeError("contract must be BaseControllerCubeContract")
    return _decode_cube_payload(
        payload,
        shape=contract.shape,
        metadata=_base_controller_metadata(contract),
        label="base controller cube",
    )


def decode_power_balance_cube_payload(
    payload: object,
    *,
    contract: PowerBalanceCubeContract,
) -> bytes:
    """Validate the all-parameterization power-balance cube."""

    if not isinstance(contract, PowerBalanceCubeContract):
        raise TypeError("contract must be PowerBalanceCubeContract")
    return _decode_cube_payload(
        payload,
        shape=contract.shape,
        metadata=_power_balance_metadata(contract),
        label="power balance cube",
    )


def decode_sensitivity_controller_cube_payload(
    payload: object,
    *,
    contract: SensitivityControllerCubeContract,
) -> bytes:
    """Validate the sensitivity per-controller metric cube."""

    if not isinstance(contract, SensitivityControllerCubeContract):
        raise TypeError("contract must be SensitivityControllerCubeContract")
    return _decode_cube_payload(
        payload,
        shape=contract.shape,
        metadata=_sensitivity_controller_metadata(contract),
        label="sensitivity controller cube",
    )


def decode_forecast_demand_cube_payload(
    payload: object,
    *,
    contract: ForecastDemandCubeContract,
) -> bytes:
    """Validate the persisted forecast demand traces."""

    if not isinstance(contract, ForecastDemandCubeContract):
        raise TypeError("contract must be ForecastDemandCubeContract")
    return _decode_cube_payload(
        payload,
        shape=contract.shape,
        metadata=_forecast_demand_metadata(contract),
        label="forecast demand cube",
        dtype_label="float32-le",
        item_size=4,
        unpack_format="<f",
    )


def delta_cube_to_cuda(
    payload: object,
    *,
    contract: DeltaCubeContract,
    device: str | torch.device = "cuda:0",
) -> torch.Tensor:
    """Load validated delta bytes onto the required CUDA device."""

    decoded = decode_delta_cube_payload(payload, contract=contract)
    return _cube_to_cuda(decoded, shape=contract.shape, device=device)


def base_controller_cube_to_cuda(
    payload: object,
    *,
    contract: BaseControllerCubeContract,
    device: str | torch.device = "cuda:0",
) -> torch.Tensor:
    """Load validated base controller bytes onto the required CUDA device."""

    decoded = decode_base_controller_cube_payload(payload, contract=contract)
    return _cube_to_cuda(decoded, shape=contract.shape, device=device)


def power_balance_cube_to_cuda(
    payload: object,
    *,
    contract: PowerBalanceCubeContract,
    device: str | torch.device = "cuda:0",
) -> torch.Tensor:
    """Load validated power-balance bytes onto the required CUDA device."""

    decoded = decode_power_balance_cube_payload(payload, contract=contract)
    return _cube_to_cuda(decoded, shape=contract.shape, device=device)


def sensitivity_controller_cube_to_cuda(
    payload: object,
    *,
    contract: SensitivityControllerCubeContract,
    device: str | torch.device = "cuda:0",
) -> torch.Tensor:
    """Load validated sensitivity controller bytes onto the required CUDA device."""

    decoded = decode_sensitivity_controller_cube_payload(payload, contract=contract)
    return _cube_to_cuda(decoded, shape=contract.shape, device=device)


def forecast_demand_cube_to_cuda(
    payload: object,
    *,
    contract: ForecastDemandCubeContract,
    device: str | torch.device = "cuda:0",
) -> torch.Tensor:
    """Load validated forecast demand bytes onto the required CUDA device."""

    decoded = decode_forecast_demand_cube_payload(payload, contract=contract)
    return _cube_to_cuda(
        decoded,
        shape=contract.shape,
        device=device,
        dtype=torch.float32,
    )


def _require_cuda_float64_cube(
    name: str,
    value: object,
    *,
    expected_shape: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(value.shape) != expected_shape:
        raise RobustnessInferenceError(
            f"{name} shape must match its persisted reconstruction contract"
        )
    if value.dtype != torch.float64:
        raise RobustnessInferenceError(f"{name} must use torch.float64")
    if value.device.type != "cuda" or value.device.index != 0:
        raise RobustnessInferenceError(f"{name} must remain on required CUDA device cuda:0")
    if not bool(torch.isfinite(value).all().item()):
        raise RobustnessInferenceError(f"{name} must contain only finite values")
    return value


def _require_value_domain(
    name: str,
    value: torch.Tensor,
    *,
    lower: float,
    upper: float,
    non_negative: bool = False,
) -> None:
    tolerance = RECONSTRUCTION_NUMERIC_TOLERANCE
    outside = torch.logical_or(value < lower - tolerance, value > upper + tolerance)
    if bool(outside.any().item()):
        qualifier = "must be non-negative and remain" if non_negative else "must remain"
        raise RobustnessInferenceError(f"{name} values {qualifier} within [{lower:g}, {upper:g}]")


def validate_reconstruction_cube_semantics(
    delta_cube: object,
    base_controller_cube: object,
    sensitivity_controller_cube: object,
    power_balance_cube: object,
    *,
    delta_contract: DeltaCubeContract,
    base_controller_contract: BaseControllerCubeContract,
    sensitivity_controller_contract: SensitivityControllerCubeContract,
    power_balance_contract: PowerBalanceCubeContract,
    maximum_outage_locality_hours: float,
) -> None:
    """Enforce raw metric domains and cross-cube identities on ``cuda:0``."""

    if not isinstance(delta_contract, DeltaCubeContract):
        raise TypeError("delta_contract must be DeltaCubeContract")
    if not isinstance(base_controller_contract, BaseControllerCubeContract):
        raise TypeError("base_controller_contract must be BaseControllerCubeContract")
    if not isinstance(sensitivity_controller_contract, SensitivityControllerCubeContract):
        raise TypeError("sensitivity_controller_contract must be SensitivityControllerCubeContract")
    if not isinstance(power_balance_contract, PowerBalanceCubeContract):
        raise TypeError("power_balance_contract must be PowerBalanceCubeContract")
    if (
        type(maximum_outage_locality_hours) is not float
        or not math.isfinite(maximum_outage_locality_hours)
        or maximum_outage_locality_hours <= 0.0
    ):
        raise RobustnessInferenceError(
            "maximum_outage_locality_hours must be a finite positive float"
        )
    if delta_contract.metric_order != _PRIMARY_METRIC_ORDER:
        raise RobustnessInferenceError("delta metric order does not match the locked contract")
    if base_controller_contract.metric_order != (*_PRIMARY_METRIC_ORDER, _SAFETY_METRIC):
        raise RobustnessInferenceError(
            "base controller metric order does not match the locked contract"
        )
    if sensitivity_controller_contract.metric_order != (
        *_PRIMARY_METRIC_ORDER,
        _SAFETY_METRIC,
    ):
        raise RobustnessInferenceError(
            "sensitivity controller metric order does not match the locked contract"
        )
    if (
        delta_contract.parameterization_order != power_balance_contract.parameterization_order
        or delta_contract.parameterization_order[1:]
        != sensitivity_controller_contract.parameterization_order
        or delta_contract.shortage_level_order != base_controller_contract.shortage_level_order
        or delta_contract.shortage_level_order
        != sensitivity_controller_contract.shortage_level_order
        or delta_contract.shortage_level_order != power_balance_contract.shortage_level_order
        or delta_contract.seed_order != base_controller_contract.seed_order
        or delta_contract.seed_order != sensitivity_controller_contract.seed_order
        or delta_contract.seed_order != power_balance_contract.seed_order
        or delta_contract.episode_order != base_controller_contract.episode_order
        or delta_contract.episode_order != sensitivity_controller_contract.episode_order
        or delta_contract.episode_order != power_balance_contract.episode_order
        or base_controller_contract.controller_order
        != sensitivity_controller_contract.controller_order
        or base_controller_contract.controller_order != power_balance_contract.controller_order
    ):
        raise RobustnessInferenceError("reconstruction cube axes are not mutually aligned")

    deltas = _require_cuda_float64_cube(
        "paired_delta_cube",
        delta_cube,
        expected_shape=delta_contract.shape,
    )
    controllers = _require_cuda_float64_cube(
        "base_controller_cube",
        base_controller_cube,
        expected_shape=base_controller_contract.shape,
    )
    sensitivity_controllers = _require_cuda_float64_cube(
        "sensitivity_controller_cube",
        sensitivity_controller_cube,
        expected_shape=sensitivity_controller_contract.shape,
    )
    power = _require_cuda_float64_cube(
        "power_balance_cube",
        power_balance_cube,
        expected_shape=power_balance_contract.shape,
    )

    _require_value_domain(
        "base_controller_cube critical_outage_locality_hours",
        controllers[..., 0],
        lower=0.0,
        upper=maximum_outage_locality_hours,
        non_negative=True,
    )
    for index, metric in enumerate((*_PRIMARY_METRIC_ORDER[1:], _SAFETY_METRIC), start=1):
        _require_value_domain(
            f"base_controller_cube {metric}",
            controllers[..., index],
            lower=0.0,
            upper=1.0,
        )
    _require_value_domain(
        "sensitivity_controller_cube critical_outage_locality_hours",
        sensitivity_controllers[..., 0],
        lower=0.0,
        upper=maximum_outage_locality_hours,
        non_negative=True,
    )
    for index, metric in enumerate((*_PRIMARY_METRIC_ORDER[1:], _SAFETY_METRIC), start=1):
        _require_value_domain(
            f"sensitivity_controller_cube {metric}",
            sensitivity_controllers[..., index],
            lower=0.0,
            upper=1.0,
        )
    _require_value_domain(
        "paired critical_outage_locality_hours",
        deltas[..., 0],
        lower=-maximum_outage_locality_hours,
        upper=maximum_outage_locality_hours,
    )
    for index, metric in enumerate(_PRIMARY_METRIC_ORDER[1:], start=1):
        _require_value_domain(
            f"paired {metric}",
            deltas[..., index],
            lower=-1.0,
            upper=1.0,
        )
    _require_value_domain(
        "power_balance_cube",
        power,
        lower=0.0,
        upper=1.0,
    )

    expected_base_deltas = (
        controllers[1, ..., : len(_PRIMARY_METRIC_ORDER)]
        - controllers[0, ..., : len(_PRIMARY_METRIC_ORDER)]
    )
    if not bool(
        torch.isclose(
            deltas[0],
            expected_base_deltas,
            rtol=0.0,
            atol=RECONSTRUCTION_NUMERIC_TOLERANCE,
        )
        .all()
        .item()
    ):
        raise RobustnessInferenceError(
            "base paired deltas do not equal PPO minus Fixed Rotation controller draws"
        )
    expected_sensitivity_deltas = (
        sensitivity_controllers[:, 1, ..., : len(_PRIMARY_METRIC_ORDER)]
        - sensitivity_controllers[:, 0, ..., : len(_PRIMARY_METRIC_ORDER)]
    )
    if not bool(
        torch.isclose(
            deltas[1:],
            expected_sensitivity_deltas,
            rtol=0.0,
            atol=RECONSTRUCTION_NUMERIC_TOLERANCE,
        )
        .all()
        .item()
    ):
        raise RobustnessInferenceError(
            "sensitivity paired deltas do not equal PPO minus Fixed Rotation controller draws"
        )
    expected_power = torch.cat(
        (
            controllers[..., len(_PRIMARY_METRIC_ORDER)].unsqueeze(0),
            sensitivity_controllers[..., len(_PRIMARY_METRIC_ORDER)],
        ),
        dim=0,
    )
    if not bool(
        torch.isclose(
            power,
            expected_power,
            rtol=0.0,
            atol=RECONSTRUCTION_NUMERIC_TOLERANCE,
        )
        .all()
        .item()
    ):
        raise RobustnessInferenceError(
            "power-balance slices do not equal persisted controller safety draws"
        )


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


def reconstruct_controller_statistics(
    base_controller_cube: torch.Tensor,
    power_balance_cube: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recompute public controller means and the global safety maximum on CUDA."""

    if not isinstance(base_controller_cube, torch.Tensor):
        raise TypeError("base_controller_cube must be a torch.Tensor")
    if not isinstance(power_balance_cube, torch.Tensor):
        raise TypeError("power_balance_cube must be a torch.Tensor")
    if (
        base_controller_cube.ndim != 5
        or base_controller_cube.shape[0] != 2
        or base_controller_cube.shape[1] != 3
        or base_controller_cube.shape[-1] < 1
        or any(size < 1 for size in base_controller_cube.shape)
    ):
        raise RobustnessInferenceError(
            "base_controller_cube must have shape [2, 3, seed, episode, metric]"
        )
    if (
        power_balance_cube.ndim != 5
        or power_balance_cube.shape[0] < 1
        or power_balance_cube.shape[1:] != base_controller_cube.shape[:4]
    ):
        raise RobustnessInferenceError(
            "power_balance_cube must have shape [parameterization, 2, 3, seed, episode]"
        )
    for name, value in (
        ("base_controller_cube", base_controller_cube),
        ("power_balance_cube", power_balance_cube),
    ):
        if value.dtype != torch.float64:
            raise RobustnessInferenceError(f"{name} must use torch.float64")
        if value.device.type != "cuda" or value.device.index != 0:
            raise RobustnessInferenceError(f"{name} must remain on required CUDA device cuda:0")
        if not bool(torch.isfinite(value).all().item()):
            raise RobustnessInferenceError(f"{name} must contain only finite values")
    return base_controller_cube.mean(dim=(1, 2, 3)), power_balance_cube.amax()

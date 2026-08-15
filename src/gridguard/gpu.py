"""Fail-closed CUDA, PyTorch, and CuPy presentation preflight."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


class GpuPreflightError(RuntimeError):
    """Raised when the required presentation GPU stack is unavailable or inconsistent."""


@dataclass(frozen=True, slots=True)
class GpuRuntimeInfo:
    """Verified identity of the CUDA stack used for simulation and PPO inference."""

    device_name: str
    compute_capability: tuple[int, int]
    total_memory_bytes: int
    torch_version: str
    torch_cuda_version: str
    torch_device_count: int
    cupy_version: str
    cupy_device_count: int
    cupy_runtime_version: int
    cupy_driver_version: int

    @property
    def total_memory_gib(self) -> float:
        return self.total_memory_bytes / 1024**3


def collect_gpu_runtime(torch_module: Any, cupy_module: Any) -> GpuRuntimeInfo:
    """Validate both libraries and return GPU identity without running array work."""

    try:
        if not torch_module.cuda.is_available():
            raise GpuPreflightError("PyTorch CUDA is unavailable")

        torch_cuda_version = torch_module.version.cuda
        if not isinstance(torch_cuda_version, str) or not torch_cuda_version:
            raise GpuPreflightError("PyTorch is not built with CUDA support")
        torch_version = str(torch_module.__version__)
        if torch_version.split("+", 1)[0] != "2.8.0":
            raise GpuPreflightError(f"PyTorch 2.8.0 is required; found {torch_version}")
        if torch_cuda_version != "12.8":
            raise GpuPreflightError(
                f"the PyTorch CUDA 12.8 build is required; found CUDA {torch_cuda_version}"
            )

        torch_device_count = int(torch_module.cuda.device_count())
        if torch_device_count < 1:
            raise GpuPreflightError("PyTorch reports no NVIDIA devices")

        properties = torch_module.cuda.get_device_properties(0)
        cupy_device_count = int(cupy_module.cuda.runtime.getDeviceCount())
        if cupy_device_count < 1:
            raise GpuPreflightError("CuPy cannot see an NVIDIA device")
        cupy_version = str(cupy_module.__version__)
        version_match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", cupy_version)
        if version_match is None:
            raise GpuPreflightError(f"CuPy has an invalid version string: {cupy_version}")
        cupy_version_tuple = tuple(int(value or 0) for value in version_match.groups())
        if cupy_version_tuple < (13, 6, 0) or cupy_version_tuple >= (16, 0, 0):
            raise GpuPreflightError(f"CuPy >=13.6,<16 is required; found {cupy_version}")
        cupy_runtime_version = int(cupy_module.cuda.runtime.runtimeGetVersion())
        cupy_driver_version = int(cupy_module.cuda.runtime.driverGetVersion())
        if not 12000 <= cupy_runtime_version < 13000:
            raise GpuPreflightError(
                f"CuPy must use a CUDA 12 runtime; found {cupy_runtime_version}"
            )
        if cupy_driver_version < cupy_runtime_version:
            raise GpuPreflightError("the NVIDIA driver is older than the CuPy CUDA runtime")

        return GpuRuntimeInfo(
            device_name=str(properties.name),
            compute_capability=(int(properties.major), int(properties.minor)),
            total_memory_bytes=int(properties.total_memory),
            torch_version=torch_version,
            torch_cuda_version=torch_cuda_version,
            torch_device_count=torch_device_count,
            cupy_version=cupy_version,
            cupy_device_count=cupy_device_count,
            cupy_runtime_version=cupy_runtime_version,
            cupy_driver_version=cupy_driver_version,
        )
    except GpuPreflightError:
        raise
    except Exception as error:
        raise GpuPreflightError(f"CUDA runtime inspection failed: {error}") from error


def run_gpu_preflight() -> GpuRuntimeInfo:
    """Require CUDA and prove zero-copy Torch/CuPy interoperability on GPU 0."""

    try:
        import cupy
        import torch
    except ImportError as error:
        raise GpuPreflightError(f"required GPU library is missing: {error.name}") from error

    info = collect_gpu_runtime(torch, cupy)
    try:
        with torch.inference_mode(), torch.cuda.device(0):
            tensor = torch.arange(4, dtype=torch.float32, device="cuda:0")
            cupy_array = cupy.from_dlpack(tensor)
            initial_sum = float(cupy_array.sum().item())
            tensor.add_(1.0)
            torch.cuda.synchronize(0)
            cupy.cuda.runtime.deviceSynchronize()
            shared_sum = float(cupy_array.sum().item())
    except Exception as error:
        raise GpuPreflightError(f"PyTorch/CuPy CUDA interop failed: {error}") from error

    if initial_sum != 6.0 or shared_sum != 10.0:
        raise GpuPreflightError("PyTorch/CuPy DLPack exchange was not zero-copy and coherent")
    return info

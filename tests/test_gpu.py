from __future__ import annotations

from types import SimpleNamespace

import pytest

from gridguard.gpu import GpuPreflightError, collect_gpu_runtime


class FakeTorchCuda:
    def __init__(self, *, available: bool = True, count: int = 1) -> None:
        self._available = available
        self._count = count

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return self._count

    def get_device_properties(self, device: int) -> SimpleNamespace:
        assert device == 0
        return SimpleNamespace(
            name="NVIDIA Test GPU",
            major=9,
            minor=1,
            total_memory=8 * 1024**3,
        )


class FakeCupyRuntime:
    def __init__(self, *, count: int = 1) -> None:
        self._count = count

    def getDeviceCount(self) -> int:
        return self._count

    def runtimeGetVersion(self) -> int:
        return 12080

    def driverGetVersion(self) -> int:
        return 13000


def _fake_torch(*, available: bool = True, count: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        __version__="2.8.0+cu128",
        version=SimpleNamespace(cuda="12.8"),
        cuda=FakeTorchCuda(available=available, count=count),
    )


def _fake_cupy(*, count: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        __version__="14.1.1",
        cuda=SimpleNamespace(runtime=FakeCupyRuntime(count=count)),
    )


def test_collect_gpu_runtime_reports_cuda_stack() -> None:
    info = collect_gpu_runtime(_fake_torch(), _fake_cupy())

    assert info.device_name == "NVIDIA Test GPU"
    assert info.compute_capability == (9, 1)
    assert info.total_memory_bytes == 8 * 1024**3
    assert info.torch_version == "2.8.0+cu128"
    assert info.torch_cuda_version == "12.8"
    assert info.cupy_version == "14.1.1"
    assert info.cupy_runtime_version == 12080
    assert info.cupy_driver_version == 13000


def test_collect_gpu_runtime_rejects_unavailable_torch_cuda() -> None:
    with pytest.raises(GpuPreflightError, match="PyTorch CUDA is unavailable"):
        collect_gpu_runtime(_fake_torch(available=False), _fake_cupy())


def test_collect_gpu_runtime_rejects_zero_visible_torch_devices() -> None:
    with pytest.raises(GpuPreflightError, match="no NVIDIA devices"):
        collect_gpu_runtime(_fake_torch(count=0), _fake_cupy())


def test_collect_gpu_runtime_rejects_zero_visible_cupy_devices() -> None:
    with pytest.raises(GpuPreflightError, match="CuPy cannot see"):
        collect_gpu_runtime(_fake_torch(), _fake_cupy(count=0))


def test_collect_gpu_runtime_rejects_cpu_only_torch_build() -> None:
    torch_module = _fake_torch()
    torch_module.version.cuda = None

    with pytest.raises(GpuPreflightError, match="not built with CUDA"):
        collect_gpu_runtime(torch_module, _fake_cupy())


def test_collect_gpu_runtime_requires_the_pinned_pytorch_and_cuda_build() -> None:
    wrong_torch = _fake_torch()
    wrong_torch.__version__ = "2.7.1+cu128"
    with pytest.raises(GpuPreflightError, match="PyTorch 2.8.0"):
        collect_gpu_runtime(wrong_torch, _fake_cupy())

    wrong_cuda = _fake_torch()
    wrong_cuda.version.cuda = "12.7"
    with pytest.raises(GpuPreflightError, match="CUDA 12.8"):
        collect_gpu_runtime(wrong_cuda, _fake_cupy())


@pytest.mark.parametrize("cupy_version", ["13.5.1", "16.0.0", "invalid"])
def test_collect_gpu_runtime_requires_the_supported_cupy_range(cupy_version: str) -> None:
    cupy_module = _fake_cupy()
    cupy_module.__version__ = cupy_version

    with pytest.raises(GpuPreflightError, match="CuPy"):
        collect_gpu_runtime(_fake_torch(), cupy_module)


def test_collect_gpu_runtime_rejects_incompatible_cupy_runtime_or_driver() -> None:
    wrong_runtime = _fake_cupy()
    wrong_runtime.cuda.runtime.runtimeGetVersion = lambda: 13000
    with pytest.raises(GpuPreflightError, match="CUDA 12"):
        collect_gpu_runtime(_fake_torch(), wrong_runtime)

    old_driver = _fake_cupy()
    old_driver.cuda.runtime.driverGetVersion = lambda: 12000
    with pytest.raises(GpuPreflightError, match="driver"):
        collect_gpu_runtime(_fake_torch(), old_driver)

from __future__ import annotations

import inspect

import pytest

from gridguard.cli import ProjectPreflightResult
from gridguard.gpu import GpuPreflightError, GpuRuntimeInfo
from gridguard.ui import app
from gridguard.ui.app import format_runtime_label, prepare_startup


def _gpu() -> GpuRuntimeInfo:
    return GpuRuntimeInfo(
        device_name="Test GPU",
        compute_capability=(9, 0),
        total_memory_bytes=8 * 1024**3,
        torch_version="2.8.0+cu128",
        torch_cuda_version="12.8",
        torch_device_count=1,
        cupy_version="14.1.1",
        cupy_device_count=1,
        cupy_runtime_version=12090,
        cupy_driver_version=13010,
    )


def _project() -> ProjectPreflightResult:
    return ProjectPreflightResult(
        gpu=_gpu(),
        source_artifact_count=5,
        matched_scenario_count=288,
        ppo_smoke_action=(1, 1, 1, 1, 1, 1, 1),
    )


def test_startup_runs_gpu_gate_before_loading_evidence() -> None:
    order: list[str] = []
    snapshot = object()

    def preflight():
        order.append("gpu")
        return _project()

    def load_snapshot():
        order.append("evidence")
        return snapshot

    bundle = prepare_startup(preflight=preflight, snapshot_loader=load_snapshot)

    assert order == ["gpu", "evidence"]
    assert bundle.gpu.device_name == "Test GPU"
    assert bundle.snapshot is snapshot


def test_failed_gpu_gate_prevents_evidence_and_window_startup() -> None:
    evidence_loaded = False

    def fail_preflight():
        raise GpuPreflightError("CUDA unavailable")

    def load_snapshot():
        nonlocal evidence_loaded
        evidence_loaded = True
        return object()

    with pytest.raises(GpuPreflightError, match="CUDA unavailable"):
        prepare_startup(preflight=fail_preflight, snapshot_loader=load_snapshot)

    assert evidence_loaded is False


def test_runtime_label_uses_readable_ascii_separators() -> None:
    assert format_runtime_label(_gpu()) == (
        "Test GPU | PyTorch 2.8.0+cu128 | CUDA 12.8 | CuPy 14.1.1"
    )


def test_startup_failure_log_uses_ascii_separator() -> None:
    source = inspect.getsource(app.main)

    assert "GridGuard desktop startup: FAIL -" in source
    assert "GridGuard desktop startup: FAIL —" not in source

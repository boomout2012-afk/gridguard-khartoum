from __future__ import annotations

import pytest

from gridguard.gpu import run_gpu_preflight


@pytest.mark.gpu
def test_real_torch_cupy_cuda_interoperability() -> None:
    info = run_gpu_preflight()

    assert info.device_name
    assert info.total_memory_bytes >= 4 * 1024**3
    assert info.torch_cuda_version == "12.8"

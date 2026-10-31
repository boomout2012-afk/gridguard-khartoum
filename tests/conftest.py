"""Hermetic Qt and deterministic CUDA defaults for developer machines and CI."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# Match the desktop entry point before any test initializes a cuBLAS handle.
# Setting this only when the live policy loads is too late after earlier GPU tests.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# This test module's exact bytes are bound by the executed scientific audit.
# Apply collection metadata here so portable CI excludes only its CUDA cases
# without rewriting the test source used to freeze that evidence. The complete
# presentation-GPU gate still selects and runs every one of these tests.
_FROZEN_AUDIT_GPU_TESTS = frozenset(
    {
        "test_contrast_preserves_own_run_pairing",
        "test_independent_metrics_reconstruct_trace_and_endpoints",
        "test_trace_verification_rejects_changed_metric",
        "test_independent_reconstruction_counts_partial_critical_deprivation",
    }
)


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        module, _, name = item.nodeid.partition("::")
        if (
            module == "tests/test_thesis_learning_audit.py"
            and name.split("[", 1)[0] in _FROZEN_AUDIT_GPU_TESTS
        ):
            item.add_marker(pytest.mark.gpu)

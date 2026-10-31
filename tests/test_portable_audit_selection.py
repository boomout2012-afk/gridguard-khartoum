"""Keep the frozen audit's CPU checks runnable when CUDA is unavailable."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_portable_audit_runs_cpu_checks_without_initializing_cuda() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_thesis_learning_audit.py",
            "-m",
            "not gpu",
            "--tb=short",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "5 passed, 5 deselected" in result.stdout

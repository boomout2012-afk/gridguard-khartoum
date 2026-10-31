"""Hermetic Qt and deterministic CUDA defaults for developer machines and CI."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# Match the desktop entry point before any test initializes a cuBLAS handle.
# Setting this only when the live policy loads is too late after earlier GPU tests.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

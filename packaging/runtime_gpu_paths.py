"""Expose bundled CUDA component DLL folders to the frozen Windows process."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_DLL_HANDLES: list[object] = []


def _configure_bundled_cuda_paths() -> None:
    bundle_root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    nvidia_root = bundle_root / "nvidia"
    if not nvidia_root.is_dir():
        return

    binary_directories = sorted(
        directory for directory in nvidia_root.glob("*/bin") if directory.is_dir()
    )
    if not binary_directories:
        return

    existing_path = os.environ.get("PATH", "")
    os.environ["PATH"] = os.pathsep.join(
        [*(str(directory) for directory in binary_directories), existing_path]
    )
    if hasattr(os, "add_dll_directory"):
        for directory in binary_directories:
            _DLL_HANDLES.append(os.add_dll_directory(directory))


_configure_bundled_cuda_paths()

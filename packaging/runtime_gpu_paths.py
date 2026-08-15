"""Expose bundled CUDA component DLL folders to the frozen Windows process."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_DLL_HANDLES: list[object] = []


def _configure_bundled_cuda_paths() -> None:
    bundle_root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    nvidia_root = bundle_root / "nvidia"
    torch_library_directory = bundle_root / "torch" / "lib"
    binary_directories = []
    if torch_library_directory.is_dir():
        binary_directories.append(torch_library_directory)
    binary_directories.extend(
        sorted(directory for directory in nvidia_root.glob("*/bin") if directory.is_dir())
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

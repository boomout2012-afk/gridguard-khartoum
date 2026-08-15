# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

import torch
from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)


PROJECT_ROOT = Path(SPECPATH).resolve().parent


def _filter_nvidia_payload(entries):
    """Drop link-time archives and DLLs already supplied by the pinned torch wheel."""

    torch_library_names = {
        path.name.casefold()
        for path in (Path(torch.__file__).resolve().parent / "lib").glob("*.dll")
    }
    return [
        (source, target)
        for source, target in entries
        if Path(source).suffix.casefold() != ".lib"
        and not (
            Path(source).suffix.casefold() == ".dll"
            and Path(source).name.casefold() in torch_library_names
        )
    ]

cupy_datas, cupy_binaries, cupy_hiddenimports = collect_all("cupy")
nvidia_datas = _filter_nvidia_payload(collect_data_files("nvidia"))
nvidia_binaries = _filter_nvidia_payload(collect_dynamic_libs("nvidia"))

datas = [
    *cupy_datas,
    *nvidia_datas,
    (str(PROJECT_ROOT / "data" / "evidence"), "gridguard/_bundled/data/evidence"),
    (str(PROJECT_ROOT / "data" / "model"), "gridguard/_bundled/data/model"),
    (
        str(PROJECT_ROOT / "docs" / "source" / "graduation-project-proposal.pdf"),
        "gridguard/_bundled/docs/source",
    ),
    (str(PROJECT_ROOT / "src" / "gridguard" / "assets"), "gridguard/assets"),
]

binaries = [*cupy_binaries, *nvidia_binaries]
hiddenimports = [
    "graphlib",
    *cupy_hiddenimports,
    *collect_submodules("cupy_backends"),
    *collect_submodules("cuda.pathfinder"),
    *collect_submodules("nvidia"),
]

analysis = Analysis(
    [str(PROJECT_ROOT / "packaging" / "gridguard_launcher.py")],
    pathex=[str(PROJECT_ROOT / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(PROJECT_ROOT / "packaging" / "runtime_gpu_paths.py")],
    excludes=["pytest", "pytestqt", "ruff", "tkinter"],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(analysis.pure)

executable = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="GridGuard Khartoum",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

distribution = COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="GridGuard Khartoum",
)

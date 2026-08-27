# -*- mode: python ; coding: utf-8 -*-

from hashlib import sha256
from pathlib import Path, PurePosixPath
from runpy import run_path

from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)


PROJECT_ROOT = Path(SPECPATH).resolve().parent
_ALLOWED_DISTRIBUTION_METADATA_FILES = frozenset({"METADATA", "WHEEL", "entry_points.txt"})
_AUDITED_RUNTIME_FILES = (
    "data/evidence/frozen-benchmark.json",
    "data/evidence/paired-benchmark.json",
    "data/evidence/robustness-supplement.json",
    "data/evidence/source-manifest.json",
    "data/model/manifest.json",
    "data/model/ppo-policy.npz",
    "data/model/training-provenance.yaml",
    "docs/source/graduation-project-proposal.pdf",
    "src/gridguard/assets/fonts/AtkinsonHyperlegible-Bold.ttf",
    "src/gridguard/assets/fonts/AtkinsonHyperlegible-Regular.ttf",
    "src/gridguard/assets/fonts/IBMPlexMono-Regular.ttf",
    "src/gridguard/assets/fonts/IBMPlexMono-SemiBold.ttf",
    "src/gridguard/assets/fonts/SourceSerif4-Regular.ttf",
    "src/gridguard/assets/fonts/SourceSerif4-Semibold.ttf",
    "src/gridguard/assets/fonts/THIRD_PARTY_FONTS.md",
    "src/gridguard/assets/fonts/licenses/AtkinsonHyperlegible-OFL.txt",
    "src/gridguard/assets/fonts/licenses/IBMPlexMono-license.txt",
    "src/gridguard/assets/fonts/licenses/SourceSerif4-LICENSE.md",
    "src/gridguard/assets/map/THIRD_PARTY_DATA.md",
    "src/gridguard/assets/map/khartoum-localities.geojson",
    "src/gridguard/assets/map/khartoum-nile.geojson",
    "src/gridguard/assets/map/map-provenance.json",
)


def _minimal_distribution_metadata(distribution):
    """Bundle only portable runtime metadata, never editable-install residue."""

    metadata_directories = copy_metadata(distribution)
    if len(metadata_directories) != 1:
        raise RuntimeError(
            f"expected one metadata directory for {distribution!r}, "
            f"found {len(metadata_directories)}"
        )

    source, target = metadata_directories[0]
    source_root = Path(source)
    target_root = Path(target)
    if not source_root.is_dir():
        raise FileNotFoundError(f"distribution metadata directory is unavailable: {source_root}")
    if (
        target_root.is_absolute()
        or len(target_root.parts) != 1
        or target_root.name != source_root.name
        or not target_root.name.endswith(".dist-info")
    ):
        raise ValueError(
            f"invalid distribution metadata destination for {distribution!r}: {target}"
        )

    entries = []
    for metadata_name in sorted(_ALLOWED_DISTRIBUTION_METADATA_FILES):
        source_path = source_root / metadata_name
        if not source_path.is_file():
            raise FileNotFoundError(f"required distribution metadata is unavailable: {source_path}")
        entries.append((str(source_path), target_root.as_posix()))
    return entries


def _audited_runtime_data():
    """Bundle only the explicit runtime data contract, never ambient directory contents."""

    entries = []
    assets_root = Path("src/gridguard/assets")
    for repository_path in _AUDITED_RUNTIME_FILES:
        relative_path = Path(repository_path)
        source_path = (PROJECT_ROOT / relative_path).resolve()
        if not source_path.is_relative_to(PROJECT_ROOT) or not source_path.is_file():
            raise FileNotFoundError(f"audited runtime file is unavailable: {repository_path}")
        if relative_path.is_relative_to(assets_root):
            packaged_path = Path("gridguard/assets") / relative_path.relative_to(assets_root)
        else:
            packaged_path = Path("gridguard/_bundled") / relative_path
        entries.append((str(source_path), packaged_path.parent.as_posix()))
    return entries


def _bundled_provenance_sources():
    """Bundle every raw source file required by fail-closed evidence verification."""

    trust_anchor_path = PROJECT_ROOT / "src" / "gridguard" / "robustness_provenance.py"
    trust_anchor = run_path(str(trust_anchor_path))
    repository_paths = (
        "src/gridguard/robustness_provenance.py",
        *(
            path
            for path, _sha256, _size_bytes in trust_anchor[
                "ROBUSTNESS_IMPLEMENTATION_SOURCE_BINDINGS"
            ]
        ),
    )
    entries = []
    for repository_path in dict.fromkeys(repository_paths):
        relative_path = Path(repository_path)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"invalid provenance source path: {repository_path}")
        source_path = (PROJECT_ROOT / relative_path).resolve()
        if not source_path.is_relative_to(PROJECT_ROOT) or not source_path.is_file():
            raise FileNotFoundError(f"provenance source is unavailable: {repository_path}")
        target_path = (Path("gridguard/_bundled") / relative_path.parent).as_posix()
        entries.append((str(source_path), target_path))
    return entries


def _payload_file_size(path, label):
    try:
        if not path.is_file():
            raise FileNotFoundError(f"{label} is unavailable: {path}")
        size_bytes = path.stat().st_size
    except OSError as error:
        raise FileNotFoundError(f"{label} is unavailable: {path}") from error
    if size_bytes <= 0:
        raise ValueError(f"{label} must be a non-empty regular file: {path}")
    return size_bytes


def _payload_sha256(path, expected_size_bytes, label):
    digest = sha256()
    bytes_read = 0
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                bytes_read += len(chunk)
        final_size_bytes = path.stat().st_size
    except OSError as error:
        raise FileNotFoundError(f"{label} cannot be read: {path}") from error
    if bytes_read != expected_size_bytes or final_size_bytes != expected_size_bytes:
        raise RuntimeError(f"{label} changed while being fingerprinted: {path}")
    return digest.hexdigest()


def _filter_nvidia_payload(entries):
    """Drop link-time archives; defer DLL deduplication until hooks have run."""

    filtered_entries = []
    for entry in entries:
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise ValueError(f"invalid NVIDIA payload entry: {entry!r}")
        source, target = entry
        if not isinstance(source, str) or not source:
            raise ValueError(f"invalid NVIDIA payload source: {source!r}")
        if not isinstance(target, str) or not target:
            raise ValueError(f"invalid NVIDIA payload target: {target!r}")
        target_path = Path(target)
        if target_path.is_absolute() or ".." in target_path.parts:
            raise ValueError(f"invalid NVIDIA payload target: {target!r}")

        source_path = Path(source)
        suffix = source_path.suffix.casefold()
        if suffix == ".lib":
            continue
        _payload_file_size(source_path, "NVIDIA payload")
        filtered_entries.append((source, target))
    return filtered_entries


def _validated_toc_entry(entry, label):
    if not isinstance(entry, (tuple, list)) or len(entry) != 3:
        raise ValueError(f"invalid {label} entry: {entry!r}")
    destination, source, typecode = entry
    if not isinstance(destination, str) or not destination:
        raise ValueError(f"invalid {label} destination: {destination!r}")
    if not isinstance(source, str) or not source:
        raise ValueError(f"invalid {label} source: {source!r}")
    if not isinstance(typecode, str) or not typecode:
        raise ValueError(f"invalid {label} typecode: {typecode!r}")

    destination_path = PurePosixPath(destination.replace("\\", "/"))
    if (
        destination_path.is_absolute()
        or ".." in destination_path.parts
        or not destination_path.parts
        or ":" in destination_path.parts[0]
    ):
        raise ValueError(f"invalid {label} destination: {destination!r}")
    return destination_path, Path(source), typecode


def _cached_payload_sha256(path, size_bytes, label, cache):
    cache_key = (path, size_bytes)
    digest = cache.get(cache_key)
    if digest is None:
        digest = _payload_sha256(path, size_bytes, label)
        cache[cache_key] = digest
    return digest


def _finalized_torch_dlls_by_name_and_size(entries):
    torch_dlls = {}
    for entry in entries:
        destination, source, typecode = _validated_toc_entry(entry, "analysis.binaries")
        destination_parts = tuple(part.casefold() for part in destination.parts)
        if (
            typecode.casefold() != "binary"
            or len(destination_parts) < 3
            or destination_parts[:2] != ("torch", "lib")
            or destination.suffix.casefold() != ".dll"
        ):
            continue
        size_bytes = _payload_file_size(source, "finalized torch DLL")
        key = (destination.name.casefold(), size_bytes)
        torch_dlls.setdefault(key, []).append(source)
    return torch_dlls


def _filter_finalized_nvidia_toc(entries, label, torch_dlls, fingerprint_cache):
    filtered_entries = []
    for entry in entries:
        destination, source, typecode = _validated_toc_entry(entry, label)
        destination_parts = tuple(part.casefold() for part in destination.parts)
        is_nvidia_payload = len(destination_parts) >= 2 and destination_parts[0] == "nvidia"
        if is_nvidia_payload and destination.suffix.casefold() == ".lib":
            continue
        if (
            typecode.casefold() not in {"binary", "data"}
            or not is_nvidia_payload
            or destination.suffix.casefold() != ".dll"
        ):
            filtered_entries.append(entry)
            continue

        size_bytes = _payload_file_size(source, "finalized NVIDIA payload")
        torch_candidates = torch_dlls.get((destination.name.casefold(), size_bytes), ())
        if not torch_candidates:
            filtered_entries.append(entry)
            continue

        nvidia_sha256 = _cached_payload_sha256(
            source,
            size_bytes,
            "finalized NVIDIA payload",
            fingerprint_cache,
        )
        if not any(
            nvidia_sha256
            == _cached_payload_sha256(
                torch_source,
                size_bytes,
                "finalized torch DLL",
                fingerprint_cache,
            )
            for torch_source in torch_candidates
        ):
            filtered_entries.append(entry)
    return filtered_entries


def _deduplicate_finalized_analysis_payload(analysis):
    """Filter hook-added duplicates once, preserving finalized TOC entries and order."""

    if not isinstance(analysis.binaries, list) or not isinstance(analysis.datas, list):
        raise TypeError("PyInstaller Analysis binaries and datas must be finalized TOC lists")
    torch_dlls = _finalized_torch_dlls_by_name_and_size(analysis.binaries)
    fingerprint_cache = {}
    analysis.binaries[:] = _filter_finalized_nvidia_toc(
        analysis.binaries,
        "analysis.binaries",
        torch_dlls,
        fingerprint_cache,
    )
    analysis.datas[:] = _filter_finalized_nvidia_toc(
        analysis.datas,
        "analysis.datas",
        torch_dlls,
        fingerprint_cache,
    )


cupy_datas, cupy_binaries, cupy_hiddenimports = collect_all("cupy")
nvidia_datas = _filter_nvidia_payload(collect_data_files("nvidia"))
nvidia_binaries = _filter_nvidia_payload(collect_dynamic_libs("nvidia"))

datas = [
    *_minimal_distribution_metadata("gridguard-khartoum"),
    *cupy_datas,
    *nvidia_datas,
    *_audited_runtime_data(),
    *_bundled_provenance_sources(),
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
_deduplicate_finalized_analysis_payload(analysis)

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

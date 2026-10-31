from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import os
import re
import runpy
import shutil
import struct
import subprocess
import sys
import tarfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import PyInstaller.utils.hooks as pyinstaller_hooks
import pytest

from gridguard.robustness_provenance import ROBUSTNESS_IMPLEMENTATION_SOURCE_BINDINGS

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RELEASE_INTEGRITY_HELPER = REPOSITORY_ROOT / "scripts" / "release-integrity.ps1"
POLICY_ERROR_PREFIX = "GRIDGUARD_POLICY_ERROR:"
RUNTIME_PACKAGE_FILES = (
    "data/evidence/frozen-benchmark.json",
    "data/evidence/paired-benchmark.json",
    "data/evidence/robustness-supplement.json",
    "data/evidence/proposal-supplement.json",
    "data/evidence/focused-ppo-study.json",
    "data/evidence/focused-ppo-training.json",
    "docs/focused-ppo-validation.json",
    "data/evidence/source-manifest.json",
    "data/model/manifest.json",
    "data/model/ppo-policy.npz",
    "data/model/focused-ppo-study/seed-8111-selected.npz",
    "data/model/training-provenance.yaml",
    "docs/source/graduation-project-proposal.pdf",
    "docs/PROPOSAL_EXTENSION.md",
    "tools/evaluate_proposal_extension.py",
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
PROVENANCE_SOURCE_FILES = (
    "src/gridguard/robustness_provenance.py",
    *(
        source_path
        for source_path, _sha256, _size_bytes in ROBUSTNESS_IMPLEMENTATION_SOURCE_BINDINGS
    ),
    "src/gridguard/duration_metrics.py",
    "src/gridguard/forecast_model.py",
    "src/gridguard/optimization.py",
    "src/gridguard/proposal_evidence.py",
    "src/gridguard/proposal_provenance.py",
    *(
        f"data/evidence/accepted-sources/{source_path}"
        for source_path, _sha256, _size_bytes in ROBUSTNESS_IMPLEMENTATION_SOURCE_BINDINGS
    ),
)
GRIDGUARD_METADATA_CONTENTS = {
    "METADATA": (
        b"Metadata-Version: 2.5\n"
        b"Name: gridguard-khartoum\n"
        b"Version: 1.2.0\n"
        b"Summary: Offline PPO load-shedding decision-support simulation for Khartoum State\n"
        b"Requires-Python: <3.13,>=3.12\n"
        b"Requires-Dist: cupy-cuda12x[ctk]<16,>=13.6\n"
        b"Requires-Dist: numpy<3,>=2.0\n"
        b"Requires-Dist: pyside6-essentials<7,>=6.8\n"
        b"Requires-Dist: torch==2.8.0\n"
        b"Provides-Extra: dev\n"
        b"Requires-Dist: hatchling==1.32.0; extra == 'dev'\n"
        b"Requires-Dist: pyinstaller<7,>=6.15; extra == 'dev'\n"
        b"Requires-Dist: pytest-qt<5,>=4.5; extra == 'dev'\n"
        b"Requires-Dist: pytest<10,>=8.4; extra == 'dev'\n"
        b"Requires-Dist: ruff<1,>=0.12; extra == 'dev'\n"
        b"Provides-Extra: docs\n"
        b"Requires-Dist: arabic-reshaper==3.0.0; extra == 'docs'\n"
        b"Requires-Dist: lxml<7,>=6; extra == 'docs'\n"
        b"Requires-Dist: pdfplumber<1,>=0.11; extra == 'docs'\n"
        b"Requires-Dist: pypdf<7,>=6; extra == 'docs'\n"
        b"Requires-Dist: python-bidi<0.7,>=0.6; extra == 'docs'\n"
        b"Requires-Dist: python-docx<2,>=1.2; extra == 'docs'\n"
        b"Requires-Dist: pyyaml<7,>=6; extra == 'docs'\n"
        b"Requires-Dist: reportlab<5,>=4.4; extra == 'docs'\n"
    ),
    "WHEEL": (b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"),
    "entry_points.txt": (
        b"[console_scripts]\n"
        b"gridguard-desktop = gridguard.ui.app:main\n"
        b"gridguard-doctor = gridguard.cli:doctor_main\n"
        b"gridguard-evaluate = gridguard.cli:main\n"
    ),
}
GRIDGUARD_METADATA_FILES = tuple(GRIDGUARD_METADATA_CONTENTS)
TEST_SOURCE_EXCLUDED_ROOTS = {
    ".codebase-memory",
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".uv-cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "tmp",
}
SDIST_EXCLUDED_SOURCE_PATHS = {"RELEASE_MANIFEST.json"}
REPRODUCIBLE_ARCHIVE_EPOCH = 1_580_601_600


def _gzip_member(payload: bytes, *, mtime: int = REPRODUCIBLE_ARCHIVE_EPOCH) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(
        filename="",
        mode="wb",
        fileobj=output,
        compresslevel=9,
        mtime=mtime,
    ) as stream:
        stream.write(payload)
    return output.getvalue()


def _write_reproducible_sdist(
    path: Path,
    entries: list[tuple[str, bytes]],
) -> None:
    tar_payload = io.BytesIO()
    with tarfile.open(fileobj=tar_payload, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for member_name, payload in entries:
            info = tarfile.TarInfo(member_name)
            info.mtime = REPRODUCIBLE_ARCHIVE_EPOCH
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    path.write_bytes(_gzip_member(tar_payload.getvalue()))


def _zip_layout(payload: bytes) -> tuple[int, int, list[dict[str, int]]]:
    eocd_offset = payload.rfind(b"PK\x05\x06")
    if eocd_offset < 0:
        raise AssertionError("test wheel has no EOCD")
    (
        _signature,
        disk_number,
        central_disk,
        disk_entries,
        total_entries,
        central_size,
        central_offset,
        comment_size,
    ) = struct.unpack_from("<4s4H2IH", payload, eocd_offset)
    assert disk_number == central_disk == 0
    assert disk_entries == total_entries
    assert comment_size == len(payload) - eocd_offset - 22
    assert central_offset + central_size == eocd_offset

    records: list[dict[str, int]] = []
    cursor = central_offset
    for _ in range(total_entries):
        values = struct.unpack_from("<4s6H3I5H2I", payload, cursor)
        assert values[0] == b"PK\x01\x02"
        name_size, extra_size, member_comment_size = values[10:13]
        records.append(
            {
                "central_offset": cursor,
                "compressed_size": values[8],
                "local_offset": values[16],
            }
        )
        cursor += 46 + name_size + extra_size + member_comment_size
    assert cursor == eocd_offset
    return eocd_offset, central_offset, records


def _local_payload_end(payload: bytes, record: dict[str, int]) -> int:
    local_offset = record["local_offset"]
    values = struct.unpack_from("<4s5H3I2H", payload, local_offset)
    assert values[0] == b"PK\x03\x04"
    name_size, extra_size = values[9:11]
    return local_offset + 30 + name_size + extra_size + record["compressed_size"]


def _insert_wheel_local_bytes(
    path: Path,
    inserted: bytes,
    *,
    count_as_compressed_payload: bool,
) -> None:
    original = path.read_bytes()
    eocd_offset, central_offset, records = _zip_layout(original)
    first = sorted(records, key=lambda record: record["local_offset"])[0]
    insertion_offset = _local_payload_end(original, first)
    mutated = bytearray(original[:insertion_offset] + inserted + original[insertion_offset:])
    shift = len(inserted)
    new_central_offset = central_offset + shift
    new_eocd_offset = eocd_offset + shift

    for record in records:
        new_record_offset = new_central_offset + (record["central_offset"] - central_offset)
        local_offset = record["local_offset"]
        if local_offset >= insertion_offset:
            local_offset += shift
        struct.pack_into("<I", mutated, new_record_offset + 42, local_offset)
    if count_as_compressed_payload:
        local_offset = first["local_offset"]
        struct.pack_into(
            "<I",
            mutated,
            local_offset + 18,
            first["compressed_size"] + shift,
        )
        first_central_offset = new_central_offset + (first["central_offset"] - central_offset)
        struct.pack_into(
            "<I",
            mutated,
            first_central_offset + 20,
            first["compressed_size"] + shift,
        )
    struct.pack_into("<I", mutated, new_eocd_offset + 16, new_central_offset)
    path.write_bytes(mutated)


def _insert_wheel_local_central_gap(path: Path, inserted: bytes) -> None:
    original = path.read_bytes()
    eocd_offset, central_offset, _records = _zip_layout(original)
    mutated = bytearray(original[:central_offset] + inserted + original[central_offset:])
    struct.pack_into(
        "<I", mutated, eocd_offset + len(inserted) + 16, central_offset + len(inserted)
    )
    path.write_bytes(mutated)


def _overlap_first_wheel_member(path: Path, overlap_size: int) -> None:
    payload = bytearray(path.read_bytes())
    _eocd_offset, _central_offset, records = _zip_layout(payload)
    first = sorted(records, key=lambda record: record["local_offset"])[0]
    struct.pack_into(
        "<I",
        payload,
        first["local_offset"] + 18,
        first["compressed_size"] + overlap_size,
    )
    struct.pack_into(
        "<I",
        payload,
        first["central_offset"] + 20,
        first["compressed_size"] + overlap_size,
    )
    path.write_bytes(payload)


def _set_first_tar_octal_field(
    sdist_path: Path,
    *,
    field_offset: int,
    field_size: int,
    value: int,
) -> None:
    tar_payload = bytearray(gzip.decompress(sdist_path.read_bytes()))
    encoded = f"{value:0{field_size - 1}o}\0".encode("ascii")
    assert len(encoded) == field_size
    tar_payload[field_offset : field_offset + field_size] = encoded
    tar_payload[148:156] = b"        "
    checksum = sum(tar_payload[:512])
    tar_payload[148:156] = f"{checksum:06o}\0 ".encode("ascii")
    sdist_path.write_bytes(_gzip_member(bytes(tar_payload)))


def _packaged_runtime_path(source_path: str) -> str:
    if source_path.startswith("src/gridguard/assets/"):
        return "_internal/gridguard/" + source_path.removeprefix("src/gridguard/")
    return "_internal/gridguard/_bundled/" + source_path


def _wheel_runtime_path(source_path: str) -> str:
    return _packaged_runtime_path(source_path).removeprefix("_internal/")


def _test_project_root(distribution: Path) -> Path:
    return distribution.resolve().parents[1]


def _stage_test_source_contract(distribution: Path) -> Path:
    project_root = _test_project_root(distribution)
    source_payloads = {
        "pyproject.toml": (REPOSITORY_ROOT / "pyproject.toml").read_bytes(),
        "src/gridguard/__init__.py": b'__version__ = "1.2.0"\n',
        **{
            source_path: source_path.encode("utf-8")
            for source_path in (*RUNTIME_PACKAGE_FILES, *PROVENANCE_SOURCE_FILES)
        },
    }
    for source_path, payload in source_payloads.items():
        destination = project_root / source_path
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    return project_root


def _test_source_archive_files(project_root: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in project_root.rglob("*")
            if path.is_file()
            and path.relative_to(project_root).parts[0] not in TEST_SOURCE_EXCLUDED_ROOTS
            and path.relative_to(project_root).as_posix() not in SDIST_EXCLUDED_SOURCE_PATHS
        ),
        key=lambda path: path.relative_to(project_root).as_posix(),
    )


def _wheel_record(entries: dict[str, bytes]) -> bytes:
    record_path = "gridguard_khartoum-1.2.0.dist-info/RECORD"
    rows = []
    for member_name, payload in entries.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode()
        rows.append(f"{member_name},sha256={digest},{len(payload)}\n")
    rows.append(f"{record_path},,\n")
    return "".join(rows).encode()


def _stage_required_runtime_payload(distribution: Path) -> None:
    project_root = _stage_test_source_contract(distribution)
    for source_path in RUNTIME_PACKAGE_FILES:
        packaged_path = distribution / _packaged_runtime_path(source_path)
        packaged_path.parent.mkdir(parents=True, exist_ok=True)
        packaged_path.write_bytes((project_root / source_path).read_bytes())


def _stage_required_provenance_sources(distribution: Path) -> None:
    project_root = _stage_test_source_contract(distribution)
    for source_path in PROVENANCE_SOURCE_FILES:
        packaged_path = distribution / "_internal" / "gridguard" / "_bundled" / source_path
        packaged_path.parent.mkdir(parents=True, exist_ok=True)
        packaged_path.write_bytes((project_root / source_path).read_bytes())


def _stage_required_gridguard_metadata(distribution: Path) -> None:
    metadata_root = distribution / "_internal" / "gridguard_khartoum-1.2.0.dist-info"
    metadata_root.mkdir(parents=True, exist_ok=True)
    for name, payload in GRIDGUARD_METADATA_CONTENTS.items():
        (metadata_root / name).write_bytes(payload)


def _stage_release_top_level(
    distribution: Path,
    *,
    include_wheel: bool = True,
    include_sdist: bool = True,
    wheel_metadata_version: str = "1.2.0",
    sdist_metadata_version: str = "1.2.0",
    wheel_metadata_payload: bytes | None = None,
    sdist_metadata_payload: bytes | None = None,
) -> None:
    project_root = _stage_test_source_contract(distribution)
    dist_root = distribution.parent
    distribution.mkdir(parents=True, exist_ok=True)
    placeholder = distribution / "_internal" / "torch" / "lib" / "torch_cuda.dll"
    placeholder.parent.mkdir(parents=True, exist_ok=True)
    if not placeholder.exists():
        placeholder.write_bytes(b"torch")
    embedded_metadata_root = distribution / "_internal" / "gridguard_khartoum-1.2.0.dist-info"
    embedded_metadata_root.mkdir(parents=True, exist_ok=True)
    for metadata_name, metadata_payload in GRIDGUARD_METADATA_CONTENTS.items():
        metadata_path = embedded_metadata_root / metadata_name
        if not metadata_path.exists():
            metadata_path.write_bytes(metadata_payload)
    (dist_root / ".gitignore").write_text("*\n", encoding="utf-8")
    if include_wheel:
        wheel_path = dist_root / "gridguard_khartoum-1.2.0-py3-none-any.whl"
        entries = {
            (
                "gridguard/"
                + source_path.relative_to(project_root / "src" / "gridguard").as_posix()
            ): source_path.read_bytes()
            for source_path in sorted((project_root / "src" / "gridguard").rglob("*.py"))
        }
        entries.update(
            {
                _wheel_runtime_path(source_path): (project_root / source_path).read_bytes()
                for source_path in RUNTIME_PACKAGE_FILES
            }
        )
        entries.update(
            {
                f"gridguard/_bundled/{source_path}": (project_root / source_path).read_bytes()
                for source_path in PROVENANCE_SOURCE_FILES
            }
        )
        metadata_root = "gridguard_khartoum-1.2.0.dist-info"
        entries[f"{metadata_root}/METADATA"] = wheel_metadata_payload or (
            GRIDGUARD_METADATA_CONTENTS["METADATA"].replace(
                b"Version: 1.2.0\n",
                f"Version: {wheel_metadata_version}\n".encode(),
                1,
            )
        )
        entries[f"{metadata_root}/WHEEL"] = GRIDGUARD_METADATA_CONTENTS["WHEEL"]
        entries[f"{metadata_root}/entry_points.txt"] = GRIDGUARD_METADATA_CONTENTS[
            "entry_points.txt"
        ]
        entries[f"{metadata_root}/RECORD"] = _wheel_record(entries)
        with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for member_name, payload in entries.items():
                archive.writestr(member_name, payload)
    if include_sdist:
        sdist_path = dist_root / "gridguard_khartoum-1.2.0.tar.gz"
        metadata = sdist_metadata_payload or GRIDGUARD_METADATA_CONTENTS["METADATA"].replace(
            b"Version: 1.2.0\n",
            f"Version: {sdist_metadata_version}\n".encode(),
            1,
        )
        source_root = "gridguard_khartoum-1.2.0"
        entries = [
            (
                f"{source_root}/{source_path.relative_to(project_root).as_posix()}",
                source_path.read_bytes(),
            )
            for source_path in _test_source_archive_files(project_root)
        ]
        entries.append((f"{source_root}/PKG-INFO", metadata))
        _write_reproducible_sdist(sdist_path, entries)


def _powershell() -> str:
    for candidate in ("pwsh", "powershell"):
        executable = shutil.which(candidate)
        if executable is None:
            continue
        result = subprocess.run(
            [executable, "-NoProfile", "-Command", "$PSVersionTable.PSVersion.ToString()"],
            capture_output=True,
            check=False,
            text=True,
        )
        if result.returncode == 0:
            return executable
    pytest.skip("PowerShell is required for release-integrity script tests")


def _run_distribution_policy(
    distribution: Path,
    *,
    max_size_bytes: int | None = None,
    release_integrity_helper: Path = RELEASE_INTEGRITY_HELPER,
) -> subprocess.CompletedProcess[str]:
    size_argument = "" if max_size_bytes is None else f" -MaxSizeBytes {max_size_bytes}"
    command = (
        f". '{release_integrity_helper}'; try {{ "
        f"Assert-ReleaseDistributionPolicy -DistributionPath '{distribution}'{size_argument} "
        "| ConvertTo-Json -Depth 5 } catch { "
        "$payload = [ordered]@{ "
        "error_type = $_.Exception.GetType().FullName; "
        "message = $_.Exception.Message }; "
        f"[Console]::Error.WriteLine('{POLICY_ERROR_PREFIX}' + "
        "($payload | ConvertTo-Json -Compress)); exit 1 }"
    )
    return subprocess.run(
        [_powershell(), "-NoProfile", "-Command", command],
        capture_output=True,
        check=False,
        text=True,
    )


def _distribution_policy_failure(
    result: subprocess.CompletedProcess[str],
) -> dict[str, str]:
    encoded_records = [
        line.removeprefix(POLICY_ERROR_PREFIX)
        for line in (result.stdout + "\n" + result.stderr).splitlines()
        if line.startswith(POLICY_ERROR_PREFIX)
    ]
    assert len(encoded_records) == 1, result.stdout + result.stderr
    payload = json.loads(encoded_records[0])
    assert set(payload) == {"error_type", "message"}
    assert isinstance(payload["error_type"], str) and payload["error_type"]
    assert isinstance(payload["message"], str) and payload["message"]
    return payload


def _distribution_policy_error(result: subprocess.CompletedProcess[str]) -> str:
    return _distribution_policy_failure(result)["message"]


def _set_named_stream(path: Path, stream_name: str = "gridguard_hidden") -> None:
    result = subprocess.run(
        [
            _powershell(),
            "-NoProfile",
            "-Command",
            (
                f"Set-Content -LiteralPath '{path}' -Stream '{stream_name}' "
                "-Value 'unbound' -NoNewline"
            ),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def _new_junction(path: Path, target: Path) -> None:
    result = subprocess.run(
        [
            _powershell(),
            "-NoProfile",
            "-Command",
            f"New-Item -ItemType Junction -Path '{path}' -Target '{target}' | Out-Null",
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def _new_hard_link(path: Path, target: Path) -> None:
    result = subprocess.run(
        [
            _powershell(),
            "-NoProfile",
            "-Command",
            f"New-Item -ItemType HardLink -Path '{path}' -Target '{target}' | Out-Null",
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@dataclass
class _StubAnalysis:
    pure: list[Any] = field(default_factory=list)
    scripts: list[Any] = field(default_factory=list)
    binaries: list[Any] = field(default_factory=list)
    datas: list[Any] = field(default_factory=list)


def _load_pyinstaller_spec_namespace(
    monkeypatch,
    torch_package_root: Path,
    *,
    finalized_binaries: tuple[tuple[str, str, str], ...] = (),
    finalized_datas: tuple[tuple[str, str, str], ...] = (),
) -> dict[str, Any]:
    (torch_package_root / "lib").mkdir(parents=True, exist_ok=True)
    # The real build deliberately narrows PATH. Keep that side effect inside
    # the test fixture so subsequent PowerShell/uv/Git tests retain their tools.
    monkeypatch.setenv("PATH", os.environ.get("PATH", ""))
    monkeypatch.setattr(pyinstaller_hooks, "collect_all", lambda _package: ([], [], []))
    monkeypatch.setattr(pyinstaller_hooks, "collect_data_files", lambda _package: [])
    monkeypatch.setattr(pyinstaller_hooks, "collect_dynamic_libs", lambda _package: [])
    monkeypatch.setattr(pyinstaller_hooks, "collect_submodules", lambda _package: [])

    def analysis(*_args, **_kwargs):
        return _StubAnalysis(
            binaries=list(finalized_binaries),
            datas=list(finalized_datas),
        )

    return runpy.run_path(
        str(REPOSITORY_ROOT / "packaging" / "gridguard.spec"),
        init_globals={
            "SPECPATH": str(REPOSITORY_ROOT / "packaging"),
            "Analysis": analysis,
            "PYZ": lambda *_args, **_kwargs: object(),
            "EXE": lambda *_args, **_kwargs: object(),
            "COLLECT": lambda *_args, **_kwargs: object(),
        },
    )


def test_spec_namespace_does_not_leak_build_path_into_other_tests(monkeypatch, tmp_path) -> None:
    original_path = os.environ.get("PATH", "")
    with monkeypatch.context() as isolated:
        _load_pyinstaller_spec_namespace(isolated, tmp_path / "torch")
    assert os.environ.get("PATH", "") == original_path


def _stage_complete_release_tree(distribution: Path) -> None:
    torch_dll = distribution / "_internal" / "torch" / "lib" / "torch_cuda.dll"
    nvidia_dll = distribution / "_internal" / "nvidia" / "cublas" / "bin" / "cublas.dll"
    torch_dll.parent.mkdir(parents=True, exist_ok=True)
    nvidia_dll.parent.mkdir(parents=True, exist_ok=True)
    torch_dll.write_bytes(b"torch")
    nvidia_dll.write_bytes(b"nvidia")
    _stage_required_runtime_payload(distribution)
    _stage_required_provenance_sources(distribution)
    _stage_required_gridguard_metadata(distribution)
    _stage_release_top_level(distribution)


def _stage_metadata_variant(
    distribution: Path,
    *,
    target: str,
    metadata_payload: bytes,
) -> None:
    _stage_complete_release_tree(distribution)
    if target == "wheel":
        _stage_release_top_level(distribution, wheel_metadata_payload=metadata_payload)
    elif target == "sdist":
        _stage_release_top_level(distribution, sdist_metadata_payload=metadata_payload)
    elif target == "embedded":
        metadata_path = (
            distribution / "_internal" / "gridguard_khartoum-1.2.0.dist-info" / "METADATA"
        )
        metadata_path.write_bytes(metadata_payload)
    else:
        raise AssertionError(f"unknown metadata target: {target}")


def test_release_distribution_policy_accepts_unique_cuda_payload(tmp_path: Path) -> None:
    distribution = tmp_path / "dist" / "GridGuard Khartoum"
    _stage_complete_release_tree(distribution)

    result = _run_distribution_policy(distribution)

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload == {
        "duplicate_cuda_dll_pair_count": 0,
        "file_count": (
            2
            + len(RUNTIME_PACKAGE_FILES)
            + len(PROVENANCE_SOURCE_FILES)
            + len(GRIDGUARD_METADATA_FILES)
        ),
        "forbidden_editable_metadata_count": 0,
        "packaged_runtime_data_count": len(RUNTIME_PACKAGE_FILES),
        "size_bytes": (
            11
            + sum(len(path.encode("utf-8")) for path in RUNTIME_PACKAGE_FILES)
            + sum(len(path.encode("utf-8")) for path in PROVENANCE_SOURCE_FILES)
            + sum(len(payload) for payload in GRIDGUARD_METADATA_CONTENTS.values())
        ),
        "status": "PASS",
        "unexpected_gridguard_version_artifact_count": 0,
    }


def test_release_distribution_policy_fails_closed_when_uv_cannot_start(
    tmp_path: Path,
) -> None:
    distribution = tmp_path / "dist" / "GridGuard Khartoum"
    _stage_complete_release_tree(distribution)
    test_helper = tmp_path / "scripts" / "release-integrity-launch-failure.ps1"
    test_helper.parent.mkdir()
    helper_text = RELEASE_INTEGRITY_HELPER.read_text(encoding="utf-8")
    launch = "$archiveValidator | & uv run --no-sync python - `"
    broken_launch = (
        "$archiveValidator | & definitely-not-a-command-gridguard run --no-sync python - `"
    )
    assert helper_text.count(launch) == 1
    helper_project_root = '$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path'
    assert helper_text.count(helper_project_root) == 1
    helper_text = helper_text.replace(
        helper_project_root,
        f"$projectRoot = '{REPOSITORY_ROOT.as_posix()}'",
        1,
    )
    test_helper.write_text(helper_text.replace(launch, broken_launch, 1), encoding="utf-8")

    result = _run_distribution_policy(
        distribution,
        release_integrity_helper=test_helper,
    )

    assert result.returncode != 0
    failure = _distribution_policy_failure(result)
    assert failure["error_type"].endswith(".CommandNotFoundException")
    assert "definitely-not-a-command-gridguard" in failure["message"].casefold()


def test_release_metadata_parser_rejects_oversized_embedded_metadata(
    tmp_path: Path,
) -> None:
    distribution = tmp_path / "dist" / "GridGuard Khartoum"
    _stage_complete_release_tree(distribution)
    metadata_path = distribution / "_internal" / "gridguard_khartoum-1.2.0.dist-info" / "METADATA"
    metadata_path.write_bytes(b"X" * (1024 * 1024 + 1))

    result = _run_distribution_policy(distribution)

    assert result.returncode != 0
    output = _distribution_policy_error(result).casefold()
    assert "embedded metadata size" in output


def test_release_metadata_parser_rejects_oversized_pyproject_contract(
    tmp_path: Path,
) -> None:
    distribution = tmp_path / "dist" / "GridGuard Khartoum"
    _stage_complete_release_tree(distribution)
    (tmp_path / "pyproject.toml").write_bytes(b"X" * (1024 * 1024 + 1))
    _stage_release_top_level(distribution)

    result = _run_distribution_policy(distribution)

    assert result.returncode != 0
    output = _distribution_policy_error(result).casefold()
    assert "pyproject metadata contract size" in output


@pytest.mark.parametrize("entry_kind", ["root", "directory", "file", "archive"])
def test_release_distribution_policy_rejects_named_alternate_data_streams(
    tmp_path: Path,
    entry_kind: str,
) -> None:
    distribution = tmp_path / "dist" / "GridGuard Khartoum"
    _stage_complete_release_tree(distribution)
    target = {
        "root": distribution,
        "directory": distribution / "_internal" / "gridguard",
        "file": distribution / "_internal" / "torch" / "lib" / "torch_cuda.dll",
        "archive": distribution.parent / "gridguard_khartoum-1.2.0.tar.gz",
    }[entry_kind]
    _set_named_stream(target)

    result = _run_distribution_policy(distribution)

    assert result.returncode != 0
    output = _distribution_policy_error(result).casefold()
    assert "alternate data stream" in output
    assert "gridguard_hidden" in output


@pytest.mark.parametrize("entry_kind", ["root", "directory"])
def test_release_distribution_policy_rejects_reparse_point_junctions(
    tmp_path: Path,
    entry_kind: str,
) -> None:
    distribution = tmp_path / "dist" / "GridGuard Khartoum"
    _stage_complete_release_tree(distribution)
    if entry_kind == "root":
        link = distribution
        target = tmp_path / "unredirected-release-tree"
    else:
        link = distribution / "_internal" / "gridguard"
        target = tmp_path / "unredirected-gridguard-directory"
    link.rename(target)
    _new_junction(link, target)

    result = _run_distribution_policy(distribution)

    assert result.returncode != 0
    assert "reparse point" in _distribution_policy_error(result).casefold()


def test_release_distribution_policy_rejects_hard_linked_files(tmp_path: Path) -> None:
    distribution = tmp_path / "dist" / "GridGuard Khartoum"
    _stage_complete_release_tree(distribution)
    target = distribution / "_internal" / "torch" / "lib" / "torch_cuda.dll"
    _new_hard_link(tmp_path / "unbound-hard-link.bin", target)

    result = _run_distribution_policy(distribution)

    assert result.returncode != 0
    assert "hard link" in _distribution_policy_error(result).casefold()


def test_release_distribution_policy_rejects_unbound_empty_directories(
    tmp_path: Path,
) -> None:
    distribution = tmp_path / "dist" / "GridGuard Khartoum"
    _stage_complete_release_tree(distribution)
    (distribution / "_internal" / "unbound-empty-directory").mkdir()

    result = _run_distribution_policy(distribution)

    assert result.returncode != 0
    output = _distribution_policy_error(result).casefold()
    assert "empty directory" in output
    assert "unbound-empty-directory" in output


def test_sdist_excludes_release_manifest_without_weakening_source_parity(
    tmp_path: Path,
) -> None:
    distribution = tmp_path / "dist" / "GridGuard Khartoum"
    _stage_complete_release_tree(distribution)
    manifest = tmp_path / "RELEASE_MANIFEST.json"
    manifest.write_text('{"must_not_self_bind":true}\n', encoding="utf-8")
    _stage_release_top_level(distribution)
    sdist_path = distribution.parent / "gridguard_khartoum-1.2.0.tar.gz"

    with tarfile.open(sdist_path, mode="r:gz") as archive:
        names = {member.name for member in archive.getmembers()}

    assert "gridguard_khartoum-1.2.0/RELEASE_MANIFEST.json" not in names
    result = _run_distribution_policy(distribution)
    assert result.returncode == 0, result.stderr

    with tarfile.open(sdist_path, mode="r:gz") as archive:
        entries = [
            (member.name, archive.extractfile(member).read())
            for member in archive.getmembers()
            if member.isfile()
        ]
    metadata = entries.pop()
    assert metadata[0] == "gridguard_khartoum-1.2.0/PKG-INFO"
    entries.append(
        (
            "gridguard_khartoum-1.2.0/RELEASE_MANIFEST.json",
            manifest.read_bytes(),
        )
    )
    entries.append(metadata)
    _write_reproducible_sdist(sdist_path, entries)

    rejected = _run_distribution_policy(distribution)
    assert rejected.returncode != 0
    output = _distribution_policy_error(rejected).replace("\\", "/")
    assert "sdist member inventory mismatch" in output
    assert "RELEASE_MANIFEST.json" in output


@pytest.mark.parametrize("tamper", ["extra", "missing"])
def test_release_distribution_policy_requires_exact_bundled_provenance_sources(
    tmp_path: Path,
    tamper: str,
) -> None:
    distribution = tmp_path / "dist" / "GridGuard Khartoum"
    _stage_required_runtime_payload(distribution)
    _stage_required_provenance_sources(distribution)
    _stage_required_gridguard_metadata(distribution)
    _stage_release_top_level(distribution)
    if tamper == "extra":
        tampered_path = (
            distribution
            / "_internal"
            / "gridguard"
            / "_bundled"
            / "src"
            / "gridguard"
            / "secret.env"
        )
        tampered_path.write_bytes(b"must not ship")
        expected_path = "secret.env"
    else:
        missing_source = PROVENANCE_SOURCE_FILES[0]
        (distribution / "_internal" / "gridguard" / "_bundled" / missing_source).unlink()
        expected_path = missing_source

    result = _run_distribution_policy(distribution)

    assert result.returncode != 0
    output = _distribution_policy_error(result).replace("\\", "/")
    assert "Exact packaged provenance-source inventory mismatch" in output
    assert expected_path in output


@pytest.mark.parametrize("tamper", ["extra", "missing", "wrong_type"])
def test_release_distribution_policy_requires_exact_gridguard_metadata(
    tmp_path: Path,
    tamper: str,
) -> None:
    distribution = tmp_path / "dist" / "GridGuard Khartoum"
    _stage_required_runtime_payload(distribution)
    _stage_required_provenance_sources(distribution)
    _stage_required_gridguard_metadata(distribution)
    _stage_release_top_level(distribution)
    metadata_root = distribution / "_internal" / "gridguard_khartoum-1.2.0.dist-info"
    if tamper == "extra":
        tampered_path = metadata_root / "secret.env"
        tampered_path.write_bytes(b"must not ship")
        expected_name = "secret.env"
    elif tamper == "missing":
        (metadata_root / "WHEEL").unlink()
        expected_name = "WHEEL"
    else:
        (metadata_root / "WHEEL").unlink()
        (metadata_root / "WHEEL").mkdir()
        (metadata_root / "WHEEL" / "not-a-file").write_bytes(b"invalid")
        expected_name = "WHEEL|directory"

    result = _run_distribution_policy(distribution)

    assert result.returncode != 0
    output = _distribution_policy_error(result).replace("\\", "/")
    assert "Exact GridGuard distribution metadata inventory mismatch" in output
    assert expected_name in output


@pytest.mark.parametrize(
    "unexpected_name",
    [
        "secret.env",
        "gridguard_khartoum-1.1.0.tar.gz",
        "gridguard_khartoum-1.2.0-py3-none-win_amd64.whl",
    ],
)
def test_release_distribution_policy_rejects_unknown_stale_or_alternative_dist_sibling(
    tmp_path: Path,
    unexpected_name: str,
) -> None:
    distribution = tmp_path / "dist" / "GridGuard Khartoum"
    _stage_release_top_level(distribution)
    (distribution.parent / unexpected_name).write_bytes(b"must not ship")

    result = _run_distribution_policy(distribution)

    assert result.returncode != 0
    output = _distribution_policy_error(result).replace("\\", "/")
    assert "Exact release dist inventory mismatch" in output
    assert unexpected_name in output


@pytest.mark.parametrize(
    ("include_wheel", "include_sdist", "missing_name"),
    [
        (False, True, "gridguard_khartoum-1.2.0-py3-none-any.whl"),
        (True, False, "gridguard_khartoum-1.2.0.tar.gz"),
    ],
)
def test_release_distribution_policy_rejects_missing_current_archive(
    tmp_path: Path,
    include_wheel: bool,
    include_sdist: bool,
    missing_name: str,
) -> None:
    distribution = tmp_path / "dist" / "GridGuard Khartoum"
    _stage_release_top_level(
        distribution,
        include_wheel=include_wheel,
        include_sdist=include_sdist,
    )

    result = _run_distribution_policy(distribution)

    assert result.returncode != 0
    output = _distribution_policy_error(result).replace("\\", "/")
    assert "Exact release dist inventory mismatch" in output
    assert missing_name in output


@pytest.mark.parametrize(
    "archive_name",
    [
        "gridguard_khartoum-1.2.0-py3-none-any.whl",
        "gridguard_khartoum-1.2.0.tar.gz",
    ],
)
def test_release_distribution_policy_rejects_malformed_current_archive(
    tmp_path: Path,
    archive_name: str,
) -> None:
    distribution = tmp_path / "dist" / "GridGuard Khartoum"
    _stage_release_top_level(distribution)
    (distribution.parent / archive_name).write_bytes(b"not an archive")

    result = _run_distribution_policy(distribution)

    assert result.returncode != 0
    output = _distribution_policy_error(result).replace("\\", "/")
    assert "Release archive structure is invalid" in output
    assert archive_name in output


@pytest.mark.parametrize("archive_kind", ["wheel", "sdist"])
def test_release_distribution_policy_rejects_archive_with_mismatched_embedded_version(
    tmp_path: Path,
    archive_kind: str,
) -> None:
    distribution = tmp_path / "dist" / "GridGuard Khartoum"
    _stage_release_top_level(
        distribution,
        wheel_metadata_version="9.9.9" if archive_kind == "wheel" else "1.2.0",
        sdist_metadata_version="9.9.9" if archive_kind == "sdist" else "1.2.0",
    )

    result = _run_distribution_policy(distribution)

    assert result.returncode != 0
    output = _distribution_policy_error(result).replace("\\", "/")
    assert "Release archive structure is invalid" in output
    assert "cross-artifact metadata mismatch" in output
    assert "version" in output.casefold()


@pytest.mark.parametrize("archive_kind", ["wheel", "sdist"])
@pytest.mark.parametrize("field", ["Name", "Version"])
@pytest.mark.parametrize("duplicate_value", ["same", "contradictory"])
def test_release_archives_reject_duplicate_identity_metadata_headers(
    tmp_path: Path,
    archive_kind: str,
    field: str,
    duplicate_value: str,
) -> None:
    distribution = tmp_path / "dist" / "GridGuard Khartoum"
    expected = "gridguard-khartoum" if field == "Name" else "1.2.0"
    contradictory = "unrelated-project" if field == "Name" else "9.9.9"
    duplicate = expected if duplicate_value == "same" else contradictory
    original_header = f"{field}: {expected}\n".encode()
    metadata = GRIDGUARD_METADATA_CONTENTS["METADATA"].replace(
        original_header,
        original_header + f"{field}: {duplicate}\n".encode(),
        1,
    )
    _stage_release_top_level(
        distribution,
        wheel_metadata_payload=metadata if archive_kind == "wheel" else None,
        sdist_metadata_payload=metadata if archive_kind == "sdist" else None,
    )
    (distribution / "GridGuard Khartoum.exe").write_bytes(b"executable")

    result = _run_distribution_policy(distribution)

    assert result.returncode != 0
    output = _distribution_policy_error(result).casefold()
    assert "release archive structure is invalid" in output
    assert f"exactly one {field}".casefold() in output


@pytest.mark.parametrize("target", ["wheel", "sdist", "embedded"])
@pytest.mark.parametrize(
    "tamper",
    [
        "wrong_runtime_specifier",
        "added_runtime_dependency",
        "removed_runtime_dependency",
        "corrupt_extra_marker",
        "duplicate_requires_dist",
        "extra_optional_requires_dist",
    ],
)
def test_release_metadata_contract_rejects_dependency_semantic_mutations(
    tmp_path: Path,
    target: str,
    tamper: str,
) -> None:
    metadata = GRIDGUARD_METADATA_CONTENTS["METADATA"]
    if tamper == "wrong_runtime_specifier":
        mutated = metadata.replace(
            b"Requires-Dist: numpy<3,>=2.0\n", b"Requires-Dist: numpy<4,>=2.0\n"
        )
    elif tamper == "added_runtime_dependency":
        mutated = metadata.replace(
            b"Requires-Dist: torch==2.8.0\n",
            b"Requires-Dist: torch==2.8.0\nRequires-Dist: malicious-runtime>=1\n",
        )
    elif tamper == "removed_runtime_dependency":
        mutated = metadata.replace(b"Requires-Dist: numpy<3,>=2.0\n", b"")
    elif tamper == "corrupt_extra_marker":
        mutated = metadata.replace(
            b"Requires-Dist: hatchling==1.32.0; extra == 'dev'\n",
            b"Requires-Dist: hatchling==1.32.0; extra == 'docs'\n",
        )
    elif tamper == "duplicate_requires_dist":
        mutated = metadata.replace(
            b"Requires-Dist: torch==2.8.0\n",
            b"Requires-Dist: torch==2.8.0\nRequires-Dist: torch==2.8.0\n",
        )
    else:
        mutated = metadata + b"Requires-Dist: rich>=13; extra == 'dev'\n"
    assert mutated != metadata
    distribution = tmp_path / "dist" / "GridGuard Khartoum"
    _stage_metadata_variant(distribution, target=target, metadata_payload=mutated)

    result = _run_distribution_policy(distribution)

    assert result.returncode != 0
    output = _distribution_policy_error(result).casefold()
    assert "metadata" in output
    assert target in output or "cross-artifact" in output or "project" in output
    if tamper == "duplicate_requires_dist":
        assert "duplicate requires-dist" in output


@pytest.mark.parametrize("target", ["wheel", "sdist", "embedded"])
@pytest.mark.parametrize(
    "field",
    ["Metadata-Version", "Name", "Version", "Summary", "Requires-Python"],
)
def test_release_metadata_contract_rejects_duplicate_singleton_headers(
    tmp_path: Path,
    target: str,
    field: str,
) -> None:
    metadata = GRIDGUARD_METADATA_CONTENTS["METADATA"]
    matching_line = next(
        line
        for line in metadata.splitlines(keepends=True)
        if line.startswith(f"{field}: ".encode())
    )
    mutated = metadata.replace(matching_line, matching_line + matching_line, 1)
    distribution = tmp_path / "dist" / "GridGuard Khartoum"
    _stage_metadata_variant(distribution, target=target, metadata_payload=mutated)

    result = _run_distribution_policy(distribution)

    assert result.returncode != 0
    output = _distribution_policy_error(result).casefold()
    assert f"exactly one {field}".casefold() in output


@pytest.mark.parametrize("target", ["wheel", "sdist", "embedded"])
def test_release_metadata_contract_rejects_email_parser_defects(
    tmp_path: Path,
    target: str,
) -> None:
    malformed = b" orphaned-continuation\n" + GRIDGUARD_METADATA_CONTENTS["METADATA"]
    distribution = tmp_path / "dist" / "GridGuard Khartoum"
    _stage_metadata_variant(distribution, target=target, metadata_payload=malformed)

    result = _run_distribution_policy(distribution)

    assert result.returncode != 0
    output = _distribution_policy_error(result).casefold()
    assert "metadata" in output
    assert "defect" in output or "continuation" in output


@pytest.mark.parametrize("target", ["wheel", "sdist", "embedded"])
def test_release_metadata_contract_rejects_mbox_unix_envelope(
    tmp_path: Path,
    target: str,
) -> None:
    malformed = b"From attacker Thu Jan 1 00:00:00 1970\n" + GRIDGUARD_METADATA_CONTENTS["METADATA"]
    distribution = tmp_path / "dist" / "GridGuard Khartoum"
    _stage_metadata_variant(distribution, target=target, metadata_payload=malformed)

    result = _run_distribution_policy(distribution)

    assert result.returncode != 0
    output = _distribution_policy_error(result).casefold()
    assert "metadata" in output
    assert "unix envelope" in output


def test_release_metadata_expectations_are_derived_from_pyproject(tmp_path: Path) -> None:
    distribution = tmp_path / "dist" / "GridGuard Khartoum"
    _stage_complete_release_tree(distribution)
    pyproject_path = tmp_path / "pyproject.toml"
    pyproject = pyproject_path.read_text(encoding="utf-8")
    mutated = pyproject.replace('"numpy>=2.0,<3"', '"numpy>=2.0,<4"', 1)
    assert mutated != pyproject
    pyproject_path.write_text(mutated, encoding="utf-8")
    _stage_release_top_level(distribution)

    result = _run_distribution_policy(distribution)

    assert result.returncode != 0
    output = _distribution_policy_error(result).casefold()
    assert "metadata" in output
    assert "pyproject" in output or "project contract" in output


def test_release_distribution_policy_rejects_cross_vendor_duplicate_and_stale_archive(
    tmp_path: Path,
) -> None:
    distribution = tmp_path / "dist" / "GridGuard Khartoum"
    torch_dll = distribution / "_internal" / "torch" / "lib" / "cublas.dll"
    nvidia_dll = distribution / "_internal" / "nvidia" / "cublas" / "bin" / "cublas.dll"
    torch_dll.parent.mkdir(parents=True)
    nvidia_dll.parent.mkdir(parents=True)
    torch_dll.write_bytes(b"duplicate")
    nvidia_dll.write_bytes(b"duplicate")
    _stage_required_gridguard_metadata(distribution)
    _stage_release_top_level(distribution)
    (distribution.parent / "gridguard_khartoum-1.0.0.tar.gz").write_bytes(b"stale")

    result = _run_distribution_policy(distribution)

    assert result.returncode != 0
    output = _distribution_policy_error(result)
    assert "gridguard_khartoum-1.0.0.tar.gz" in output


def test_release_distribution_policy_rejects_cross_vendor_duplicate(tmp_path: Path) -> None:
    distribution = tmp_path / "dist" / "GridGuard Khartoum"
    torch_dll = distribution / "_internal" / "torch" / "lib" / "CUBLAS.DLL"
    nvidia_dll = distribution / "_internal" / "nvidia" / "cublas" / "bin" / "cublas.dll"
    torch_dll.parent.mkdir(parents=True)
    nvidia_dll.parent.mkdir(parents=True)
    torch_dll.write_bytes(b"duplicate")
    nvidia_dll.write_bytes(b"duplicate")
    _stage_required_gridguard_metadata(distribution)
    _stage_release_top_level(distribution)

    result = _run_distribution_policy(distribution)

    assert result.returncode != 0
    output = _distribution_policy_error(result)
    assert "Byte-identical Torch/NVIDIA CUDA DLL pairs remain" in output
    assert "_internal/torch/lib/CUBLAS.DLL" in output
    assert "_internal/nvidia/cublas/bin/cublas.dll" in output


def test_release_distribution_policy_allows_byte_identical_dll_aliases(
    tmp_path: Path,
) -> None:
    distribution = tmp_path / "dist" / "GridGuard Khartoum"
    _stage_complete_release_tree(distribution)
    torch_dll = distribution / "_internal" / "torch" / "lib" / "torch_cuda.dll"
    nvidia_alias = (
        distribution / "_internal" / "nvidia" / "runtime" / "bin" / "torch_cuda_alias.dll"
    )
    nvidia_alias.parent.mkdir(parents=True, exist_ok=True)
    nvidia_alias.write_bytes(torch_dll.read_bytes())

    result = _run_distribution_policy(distribution)

    assert result.returncode == 0, result.stdout + result.stderr


def test_release_distribution_policy_rejects_embedded_gridguard_1_1_metadata(
    tmp_path: Path,
) -> None:
    distribution = tmp_path / "dist" / "GridGuard Khartoum"
    metadata = distribution / "_internal" / "gridguard_khartoum-1.1.0.dist-info" / "METADATA"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("Version: 1.1.0\n", encoding="utf-8")
    _stage_release_top_level(distribution)

    result = _run_distribution_policy(distribution)

    assert result.returncode != 0
    output = _distribution_policy_error(result)
    assert "gridguard_khartoum-1.1.0.dist-info/METADATA" in output.replace("\\", "/")
    assert "Unexpected GridGuard release-version artifacts remain" in output


def test_release_distribution_policy_rejects_editable_gridguard_metadata(
    tmp_path: Path,
) -> None:
    distribution = tmp_path / "dist" / "GridGuard Khartoum"
    direct_url = (
        distribution / "_internal" / "gridguard_khartoum-1.2.0.dist-info" / "direct_url.json"
    )
    direct_url.parent.mkdir(parents=True)
    _stage_required_gridguard_metadata(distribution)
    direct_url.write_text('{"url":"file:///C:/developer/worktree"}', encoding="utf-8")
    _stage_release_top_level(distribution)

    result = _run_distribution_policy(distribution)

    assert result.returncode != 0
    output = _distribution_policy_error(result).replace("\\", "/")
    assert "gridguard_khartoum-1.2.0.dist-info/direct_url.json" in output
    assert "Exact GridGuard distribution metadata inventory mismatch" in output


@pytest.mark.parametrize(
    ("metadata_name", "payload", "expected_error"),
    [
        (
            "METADATA",
            GRIDGUARD_METADATA_CONTENTS["METADATA"]
            .replace(
                b"Name: gridguard-khartoum\n",
                b"Name: unrelated-project\n",
                1,
            )
            .decode(),
            "cross-artifact metadata mismatch",
        ),
        (
            "WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: cp39-linux_x86_64\n",
            "wheel contract mismatch",
        ),
        (
            "entry_points.txt",
            "[console_scripts]\ngridguard-desktop = malicious.module:main\n",
            "console-script contract mismatch",
        ),
    ],
)
def test_release_distribution_policy_rejects_invalid_gridguard_metadata_contents(
    tmp_path: Path,
    metadata_name: str,
    payload: str,
    expected_error: str,
) -> None:
    distribution = tmp_path / "dist" / "GridGuard Khartoum"
    _stage_required_runtime_payload(distribution)
    _stage_required_provenance_sources(distribution)
    _stage_required_gridguard_metadata(distribution)
    _stage_release_top_level(distribution)
    metadata_path = (
        distribution / "_internal" / "gridguard_khartoum-1.2.0.dist-info" / metadata_name
    )
    metadata_path.write_text(payload, encoding="utf-8")

    result = _run_distribution_policy(distribution)

    assert result.returncode != 0
    output = _distribution_policy_error(result).casefold()
    if metadata_name == "METADATA":
        assert "release archive structure is invalid" in output
    else:
        assert "gridguard distribution metadata is invalid" in output
    assert expected_error in output


def test_release_distribution_policy_rejects_size_above_budget(tmp_path: Path) -> None:
    distribution = tmp_path / "dist" / "GridGuard Khartoum"
    oversized = distribution / "_internal" / "payload.bin"
    oversized.parent.mkdir(parents=True)
    oversized.write_bytes(b"123456")
    _stage_release_top_level(distribution)

    result = _run_distribution_policy(distribution, max_size_bytes=5)

    assert result.returncode != 0
    assert "5-byte budget" in _distribution_policy_error(result)


@pytest.mark.parametrize(
    "sentinel_relative_path",
    [
        "_internal/gridguard/_bundled/data/model/.env",
        "_internal/gridguard/_bundled/secret.env",
    ],
)
def test_release_distribution_policy_rejects_ambient_runtime_data(
    tmp_path: Path,
    sentinel_relative_path: str,
) -> None:
    distribution = tmp_path / "dist" / "GridGuard Khartoum"
    _stage_required_runtime_payload(distribution)
    _stage_required_provenance_sources(distribution)
    _stage_required_gridguard_metadata(distribution)
    _stage_release_top_level(distribution)
    sentinel = distribution / Path(sentinel_relative_path)
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text("GRIDGUARD_SECRET=must-not-ship\n", encoding="utf-8")

    result = _run_distribution_policy(distribution)

    assert result.returncode != 0
    output = _distribution_policy_error(result).replace("\\", "/")
    assert "Exact packaged GridGuard runtime/provenance inventory mismatch" in output
    assert sentinel_relative_path in output


@pytest.mark.parametrize(
    ("tamper", "expected_error"),
    [
        ("wheel_unrelated_member", "wheel member inventory mismatch"),
        ("wheel_unsafe_member", "unsafe wheel member path"),
        ("wheel_contract", "wheel contract mismatch"),
        ("sdist_secret", "forbidden secret-like archive member"),
        ("sdist_unrelated_member", "sdist member inventory mismatch"),
        ("sdist_unsafe_member", "unsafe sdist member path"),
    ],
)
def test_release_distribution_policy_rejects_unsafe_or_unrelated_archive_members(
    tmp_path: Path,
    tamper: str,
    expected_error: str,
) -> None:
    distribution = tmp_path / "dist" / "GridGuard Khartoum"
    _stage_required_runtime_payload(distribution)
    _stage_required_provenance_sources(distribution)
    _stage_required_gridguard_metadata(distribution)
    _stage_release_top_level(distribution)
    wheel_path = distribution.parent / "gridguard_khartoum-1.2.0-py3-none-any.whl"
    sdist_path = distribution.parent / "gridguard_khartoum-1.2.0.tar.gz"

    if tamper.startswith("wheel_"):
        with zipfile.ZipFile(wheel_path) as archive:
            wheel_members = {member.filename: archive.read(member) for member in archive.infolist()}
        if tamper == "wheel_unrelated_member":
            wheel_members["unrelated.bin"] = b"must-not-ship\n"
        elif tamper == "wheel_unsafe_member":
            wheel_members["../escape.txt"] = b"must-not-ship\n"
        else:
            wheel_members["gridguard_khartoum-1.2.0.dist-info/WHEEL"] = (
                b"Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: cp39-linux_x86_64\n"
            )
        with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for member_name, member_bytes in wheel_members.items():
                archive.writestr(member_name, member_bytes)
    else:
        metadata = b"Metadata-Version: 2.4\nName: gridguard-khartoum\nVersion: 1.2.0\n"
        tampered_member = {
            "sdist_secret": "gridguard_khartoum-1.2.0/.env",
            "sdist_unrelated_member": "gridguard_khartoum-1.2.0/unrelated.bin",
            "sdist_unsafe_member": "gridguard_khartoum-1.2.0/../escape.txt",
        }[tamper]
        _write_reproducible_sdist(
            sdist_path,
            [
                ("gridguard_khartoum-1.2.0/PKG-INFO", metadata),
                (tampered_member, b"GRIDGUARD_SECRET=must-not-ship\n"),
            ],
        )

    result = _run_distribution_policy(distribution)

    assert result.returncode != 0
    output = _distribution_policy_error(result).casefold()
    assert "release archive structure is invalid" in output
    assert expected_error in output


@pytest.mark.parametrize(
    "tamper",
    [
        "archive_comment",
        "member_comment",
        "leading_bytes",
        "trailing_bytes",
        "local_central_gap",
        "local_entry_gap",
        "overlapping_local_ranges",
        "nonexhausted_deflate_payload",
    ],
)
def test_release_distribution_policy_rejects_noncanonical_wheel_container(
    tmp_path: Path,
    tamper: str,
) -> None:
    distribution = tmp_path / "dist" / "GridGuard Khartoum"
    _stage_required_runtime_payload(distribution)
    _stage_required_provenance_sources(distribution)
    _stage_required_gridguard_metadata(distribution)
    _stage_release_top_level(distribution)
    wheel_path = distribution.parent / "gridguard_khartoum-1.2.0-py3-none-any.whl"

    if tamper == "archive_comment":
        with zipfile.ZipFile(wheel_path, "a") as archive:
            archive.comment = b"hidden archive comment"
    elif tamper == "member_comment":
        with zipfile.ZipFile(wheel_path, "r") as archive:
            members = [(info, archive.read(info)) for info in archive.infolist()]
        members[0][0].comment = b"hidden member comment"
        with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for info, payload in members:
                archive.writestr(info, payload)
    elif tamper == "leading_bytes":
        wheel_path.write_bytes(b"hidden-prefix" + wheel_path.read_bytes())
    elif tamper == "trailing_bytes":
        with wheel_path.open("ab") as stream:
            stream.write(b"hidden-suffix")
    elif tamper == "local_central_gap":
        _insert_wheel_local_central_gap(wheel_path, b"hidden-gap")
    elif tamper == "local_entry_gap":
        _insert_wheel_local_bytes(
            wheel_path,
            b"hidden-gap",
            count_as_compressed_payload=False,
        )
    elif tamper == "overlapping_local_ranges":
        _overlap_first_wheel_member(wheel_path, 4)
    else:
        _insert_wheel_local_bytes(
            wheel_path,
            b"hidden-deflate-tail",
            count_as_compressed_payload=True,
        )

    result = _run_distribution_policy(distribution)

    assert result.returncode != 0
    output = _distribution_policy_error(result).casefold()
    assert "release archive structure is invalid" in output
    assert "wheel" in output


@pytest.mark.parametrize(
    "tamper",
    [
        "gzip_mtime",
        "concatenated_gzip_stream",
        "gzip_trailing_bytes",
        "extra_tar_record",
        "nonzero_tar_bytes_after_eof",
        "nonzero_tar_padding",
        "tar_mtime",
        "tar_mode",
        "tar_uid",
    ],
)
def test_release_distribution_policy_rejects_noncanonical_sdist_container(
    tmp_path: Path,
    tamper: str,
) -> None:
    distribution = tmp_path / "dist" / "GridGuard Khartoum"
    _stage_required_runtime_payload(distribution)
    _stage_required_provenance_sources(distribution)
    _stage_required_gridguard_metadata(distribution)
    _stage_release_top_level(distribution)
    sdist_path = distribution.parent / "gridguard_khartoum-1.2.0.tar.gz"

    if tamper == "gzip_mtime":
        payload = bytearray(sdist_path.read_bytes())
        payload[4:8] = (REPRODUCIBLE_ARCHIVE_EPOCH + 1).to_bytes(4, "little")
        sdist_path.write_bytes(payload)
    elif tamper == "concatenated_gzip_stream":
        with sdist_path.open("ab") as stream:
            stream.write(_gzip_member(b"hidden second stream"))
    elif tamper == "gzip_trailing_bytes":
        with sdist_path.open("ab") as stream:
            stream.write(b"hidden trailing bytes")
    elif tamper == "extra_tar_record":
        tar_payload = gzip.decompress(sdist_path.read_bytes()) + (b"\0" * 10_240)
        sdist_path.write_bytes(_gzip_member(tar_payload))
    elif tamper == "nonzero_tar_bytes_after_eof":
        tar_payload = gzip.decompress(sdist_path.read_bytes()) + b"hidden after tar EOF"
        sdist_path.write_bytes(_gzip_member(tar_payload))
    elif tamper == "nonzero_tar_padding":
        tar_payload = bytearray(gzip.decompress(sdist_path.read_bytes()))
        first_size = int(tar_payload[124:136].rstrip(b"\0 ") or b"0", 8)
        first_padding_offset = 512 + first_size
        assert first_padding_offset % 512 != 0
        tar_payload[first_padding_offset] = 1
        sdist_path.write_bytes(_gzip_member(bytes(tar_payload)))
    elif tamper == "tar_mtime":
        _set_first_tar_octal_field(
            sdist_path,
            field_offset=136,
            field_size=12,
            value=REPRODUCIBLE_ARCHIVE_EPOCH + 1,
        )
    elif tamper == "tar_mode":
        _set_first_tar_octal_field(
            sdist_path,
            field_offset=100,
            field_size=8,
            value=0o600,
        )
    else:
        _set_first_tar_octal_field(
            sdist_path,
            field_offset=108,
            field_size=8,
            value=1,
        )

    result = _run_distribution_policy(distribution)

    assert result.returncode != 0
    output = _distribution_policy_error(result).casefold()
    assert "release archive structure is invalid" in output
    assert "sdist" in output


def test_pyinstaller_spec_deduplicates_hook_added_finalized_nvidia_toc_entries(
    monkeypatch,
    tmp_path: Path,
) -> None:
    torch_package_root = tmp_path / "torch"
    torch_dll = torch_package_root / "lib" / "cublasLt64_12.dll"
    torch_dll.parent.mkdir(parents=True)
    torch_dll.write_bytes(b"shared-cuda-bytes")
    nvidia_root = tmp_path / "nvidia"
    identical = nvidia_root / "cublas" / "bin" / torch_dll.name
    alias = nvidia_root / "runtime" / "bin" / "renamed_cuda.dll"
    same_name_different = nvidia_root / "runtime" / "bin" / torch_dll.name
    unique = nvidia_root / "runtime" / "bin" / "nvrtc64_120_0.dll"
    hook_added_static = nvidia_root / "nvjitlink" / "lib" / "nvJitLink_static.lib"
    unrelated = tmp_path / "plugins" / torch_dll.name
    for path in (identical, alias, same_name_different, unique, hook_added_static, unrelated):
        path.parent.mkdir(parents=True, exist_ok=True)
    identical.write_bytes(torch_dll.read_bytes())
    alias.write_bytes(torch_dll.read_bytes())
    same_name_different.write_bytes(b"different-cuda-xx")
    assert same_name_different.stat().st_size == torch_dll.stat().st_size
    unique.write_bytes(b"unique-runtime")
    hook_added_static.write_bytes(b"link-time-only")
    unrelated.write_bytes(torch_dll.read_bytes())

    torch_entry = (f"torch/lib/{torch_dll.name}", str(torch_dll), "BINARY")
    identical_entry = (
        f"nvidia/cublas/bin/{identical.name}",
        str(identical),
        "BINARY",
    )
    alias_entry = (f"nvidia/runtime/bin/{alias.name}", str(alias), "BINARY")
    different_entry = (
        f"nvidia/runtime/bin/{same_name_different.name}",
        str(same_name_different),
        "BINARY",
    )
    unique_entry = (f"nvidia/runtime/bin/{unique.name}", str(unique), "BINARY")
    static_entry = (
        f"nvidia/nvjitlink/lib/{hook_added_static.name}",
        str(hook_added_static),
        "BINARY",
    )
    unrelated_entry = (f"plugins/{unrelated.name}", str(unrelated), "BINARY")
    identical_data_entry = (
        f"nvidia/cublas/bin/{identical.name}",
        str(identical),
        "DATA",
    )
    alias_data_entry = (f"nvidia/runtime/bin/{alias.name}", str(alias), "DATA")
    namespace = _load_pyinstaller_spec_namespace(
        monkeypatch,
        torch_package_root,
        finalized_binaries=(
            unrelated_entry,
            torch_entry,
            identical_entry,
            alias_entry,
            different_entry,
            unique_entry,
            static_entry,
        ),
        finalized_datas=(identical_data_entry, alias_data_entry),
    )

    expected_binaries = [
        unrelated_entry,
        torch_entry,
        alias_entry,
        different_entry,
        unique_entry,
    ]
    assert namespace["analysis"].binaries == expected_binaries
    assert all(
        actual is expected
        for actual, expected in zip(namespace["analysis"].binaries, expected_binaries, strict=True)
    )
    assert namespace["analysis"].datas == [alias_data_entry]
    assert namespace["analysis"].datas[0] is alias_data_entry


def test_finalized_nvidia_filter_fails_closed_when_a_dll_payload_is_unavailable(
    monkeypatch,
    tmp_path: Path,
) -> None:
    torch_package_root = tmp_path / "torch"
    torch_dll = torch_package_root / "lib" / "torch_cuda.dll"
    torch_dll.parent.mkdir(parents=True)
    torch_dll.write_bytes(b"pinned")
    missing_dll = tmp_path / "nvidia" / "runtime" / "bin" / "missing.dll"
    torch_entry = (f"torch/lib/{torch_dll.name}", str(torch_dll), "BINARY")
    missing_entry = (
        f"nvidia/runtime/bin/{torch_dll.name}",
        str(missing_dll),
        "BINARY",
    )

    with pytest.raises(FileNotFoundError, match="NVIDIA payload.*unavailable"):
        _load_pyinstaller_spec_namespace(
            monkeypatch,
            torch_package_root,
            finalized_binaries=(torch_entry, missing_entry),
        )


def test_release_distribution_default_budget_proves_required_reduction() -> None:
    audited_size_bytes = 7_021_231_590
    required_reduction_bytes = 2_000_000_000
    expected_default_max_size_bytes = audited_size_bytes - required_reduction_bytes
    helper = RELEASE_INTEGRITY_HELPER.read_text(encoding="utf-8")
    default_match = re.search(
        r"\[int64\]\$MaxSizeBytes\s*=\s*\$script:DefaultMaximumDistributionSizeBytes",
        helper,
    )
    command = (
        f". '{RELEASE_INTEGRITY_HELPER}'; "
        "[pscustomobject]@{ "
        "audited = $script:AuditedDistributionSizeBytes; "
        "required_reduction = $script:RequiredDistributionReductionBytes; "
        "default_maximum = $script:DefaultMaximumDistributionSizeBytes "
        "} | ConvertTo-Json -Compress"
    )

    result = subprocess.run(
        [_powershell(), "-NoProfile", "-Command", command],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    contract = json.loads(result.stdout)
    assert contract == {
        "audited": audited_size_bytes,
        "required_reduction": required_reduction_bytes,
        "default_maximum": expected_default_max_size_bytes,
    }
    assert default_match is not None
    assert audited_size_bytes - contract["default_maximum"] >= required_reduction_bytes
    assert contract["default_maximum"] <= 5_021_231_590


def test_pyinstaller_spec_resolves_inputs_from_the_repository_root(
    monkeypatch,
    tmp_path: Path,
) -> None:
    # The spec deliberately restricts DLL discovery; restore the runner's tool PATH.
    monkeypatch.setenv("PATH", os.environ.get("PATH", ""))
    captured: dict[str, Any] = {}
    copied_metadata: list[str] = []
    real_copy_metadata = pyinstaller_hooks.copy_metadata
    metadata_directories = real_copy_metadata("gridguard-khartoum")
    assert len(metadata_directories) == 1
    metadata_root = Path(metadata_directories[0][0])
    assert metadata_root.is_dir()
    assert metadata_directories[0][1] == metadata_root.name
    assert {
        "direct_url.json",
        "RECORD",
        *GRIDGUARD_METADATA_FILES,
    }.issubset(path.name for path in metadata_root.iterdir())

    torch_package_root = tmp_path / "torch"
    torch_dll = torch_package_root / "lib" / "torch_cuda_payload.dll"
    torch_dll.parent.mkdir(parents=True)
    torch_dll.write_bytes(b"shared-cublas-bytes")

    cuda_root = tmp_path / "cuda"
    static_library = cuda_root / "vJitLink_static.lib"
    duplicate_dll = cuda_root / "cublasLt64_12.dll"
    license_file = cuda_root / "LICENSE.txt"
    unique_dll = cuda_root / "cupy_only.dll"
    cuda_root.mkdir()
    static_library.write_bytes(b"link-time-only")
    duplicate_dll.write_bytes(torch_dll.read_bytes())
    license_file.write_text("license", encoding="utf-8")
    unique_dll.write_bytes(b"cupy-only-runtime")
    external_cuda_sources = {
        str(static_library),
        str(duplicate_dll),
        str(license_file),
        str(unique_dll),
    }

    monkeypatch.setattr(pyinstaller_hooks, "collect_all", lambda _package: ([], [], []))
    monkeypatch.setattr(
        pyinstaller_hooks,
        "collect_data_files",
        lambda _package: [
            (str(static_library), "nvidia/vjitlink/lib/x64"),
            (str(duplicate_dll), "nvidia/cublas/bin"),
            (str(license_file), "nvidia/cublas"),
        ],
    )
    monkeypatch.setattr(
        pyinstaller_hooks,
        "collect_dynamic_libs",
        lambda _package: [
            (str(duplicate_dll), "nvidia/cublas/bin"),
            (str(unique_dll), "nvidia/runtime/bin"),
        ],
    )
    monkeypatch.setattr(pyinstaller_hooks, "collect_submodules", lambda _package: [])

    def copy_metadata(distribution: str) -> list[tuple[str, str]]:
        copied_metadata.append(distribution)
        return real_copy_metadata(distribution)

    monkeypatch.setattr(pyinstaller_hooks, "copy_metadata", copy_metadata)

    def analysis(scripts, **kwargs):
        captured["scripts"] = scripts
        captured["datas"] = kwargs["datas"]
        captured["hiddenimports"] = kwargs["hiddenimports"]
        return _StubAnalysis()

    namespace = runpy.run_path(
        str(REPOSITORY_ROOT / "packaging" / "gridguard.spec"),
        init_globals={
            "SPECPATH": str(REPOSITORY_ROOT / "packaging"),
            "Analysis": analysis,
            "PYZ": lambda *_args, **_kwargs: object(),
            "EXE": lambda *_args, **_kwargs: object(),
            "COLLECT": lambda *_args, **_kwargs: object(),
        },
    )

    assert namespace["PROJECT_ROOT"] == REPOSITORY_ROOT
    assert captured["scripts"] == [str(REPOSITORY_ROOT / "packaging" / "gridguard_launcher.py")]
    assert all(
        Path(source).is_relative_to(REPOSITORY_ROOT)
        for source, _target in captured["datas"]
        if source not in external_cuda_sources and not Path(source).is_relative_to(metadata_root)
    )
    assert copied_metadata == ["gridguard-khartoum"]
    bundled_gridguard_metadata = {
        Path(source).name
        for source, target in captured["datas"]
        if target == "gridguard_khartoum-1.2.0.dist-info"
    }
    assert bundled_gridguard_metadata == {"METADATA", "WHEEL", "entry_points.txt"}
    assert {
        Path(source) for source, target in captured["datas"] if target == metadata_root.name
    } == {metadata_root / name for name in GRIDGUARD_METADATA_FILES}
    assert "graphlib" in captured["hiddenimports"]
    bundled_cuda_sources = {
        source
        for source, _target in (*captured["datas"], *namespace["binaries"])
        if source in external_cuda_sources
    }
    assert str(static_library) not in bundled_cuda_sources
    assert str(duplicate_dll) in bundled_cuda_sources
    assert str(license_file) in bundled_cuda_sources
    assert str(unique_dll) in bundled_cuda_sources

    bundled_provenance_sources = {
        Path(source).resolve().relative_to(REPOSITORY_ROOT).as_posix()
        for source, target in captured["datas"]
        if target == "gridguard/_bundled/src/gridguard"
    }
    expected_provenance_sources = {
        source_path for source_path in PROVENANCE_SOURCE_FILES if source_path.startswith("src/")
    }
    assert bundled_provenance_sources == expected_provenance_sources

    runtime_sources = {
        Path(source).resolve().relative_to(REPOSITORY_ROOT).as_posix()
        for source, _target in captured["datas"]
        if source not in external_cuda_sources
        and not Path(source).is_relative_to(metadata_root)
        and "_bundled/src/gridguard" not in _target
        and "_bundled/data/evidence/accepted-sources/" not in _target
    }
    assert runtime_sources == set(RUNTIME_PACKAGE_FILES) | {
        "data/evidence/committee-baseline-audit.json",
        "data/evidence/committee-baseline-audit.validation.json",
    }
    assert all(
        not Path(source).is_dir()
        for source, _target in captured["datas"]
        if source not in external_cuda_sources
    )


def test_pyinstaller_spec_is_one_directory_and_bundles_trusted_runtime_assets() -> None:
    spec = (REPOSITORY_ROOT / "packaging" / "gridguard.spec").read_text(encoding="utf-8")

    assert "COLLECT(" in spec
    assert "exclude_binaries=True" in spec
    assert 'name="GridGuard Khartoum"' in spec
    assert "data/evidence" in spec
    assert "data/model" in spec
    assert "graduation-project-proposal.pdf" in spec
    assert "robustness_provenance.py" in spec
    assert "*_bundled_provenance_sources()" in spec
    assert "gridguard/assets" in spec
    assert 'collect_all("cupy")' in spec
    assert 'collect_dynamic_libs("nvidia")' in spec
    assert "_minimal_distribution_metadata" in spec
    assert "_audited_runtime_data" in spec
    assert '(str(PROJECT_ROOT / "data" / "evidence"),' not in spec
    assert '(str(PROJECT_ROOT / "data" / "model"),' not in spec
    assert '(str(PROJECT_ROOT / "src" / "gridguard" / "assets"),' not in spec


def test_wheel_force_includes_only_audited_runtime_files() -> None:
    pyproject = (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'exclude = ["src/gridguard/assets/**"]' in pyproject
    assert '"data/evidence" = ' not in pyproject
    assert '"data/model" = ' not in pyproject
    for source_path in RUNTIME_PACKAGE_FILES:
        assert f'"{source_path}" = ' in pyproject

    assert "[tool.hatch.build.targets.sdist]" in pyproject
    assert 'exclude = ["/RELEASE_MANIFEST.json"]' in pyproject


def test_runtime_hook_searches_torch_libraries_before_unique_nvidia_components() -> None:
    runtime_hook = (REPOSITORY_ROOT / "packaging" / "runtime_gpu_paths.py").read_text(
        encoding="utf-8"
    )

    assert 'bundle_root / "torch" / "lib"' in runtime_hook
    assert 'nvidia_root.glob("*/bin")' in runtime_hook


def test_frozen_launcher_only_delegates_to_the_verified_desktop_entry_point() -> None:
    launcher = (REPOSITORY_ROOT / "packaging" / "gridguard_launcher.py").read_text(encoding="utf-8")

    assert "from gridguard.ui.app import main" in launcher
    assert "raise SystemExit(main())" in launcher


def test_frozen_launcher_has_a_fail_closed_automated_smoke_mode() -> None:
    launcher = (REPOSITORY_ROOT / "packaging" / "gridguard_launcher.py").read_text(encoding="utf-8")

    assert "GRIDGUARD_SMOKE_REPORT" not in launcher
    assert '"--_gridguard-frozen-smoke-report"' in launcher
    assert "_validated_smoke_report_path" in launcher
    assert "prepare_startup()" in launcher
    assert "PresentationWindow(" in launcher
    assert "application.processEvents()" in launcher
    assert '"status": "PASS"' in launcher
    assert '"status": "FAIL"' in launcher
    assert "from gridguard import __version__" in launcher
    assert '"release_id": f"gridguard-khartoum-v{__version__}"' in launcher
    assert '"cause": _exception_summary(error.__cause__)' in launcher


def test_frozen_launcher_ignores_ambient_smoke_controls_for_normal_launch(monkeypatch) -> None:
    namespace = runpy.run_path(
        str(REPOSITORY_ROOT / "packaging" / "gridguard_launcher.py"),
        run_name="gridguard_launcher_test",
    )
    launch = namespace["_run_launcher"]
    launcher_globals = launch.__globals__
    calls: list[str] = []
    launcher_globals["_run_desktop"] = lambda: calls.append("desktop")
    launcher_globals["_run_frozen_smoke"] = lambda _path: calls.append("smoke") or 1
    monkeypatch.setenv("GRIDGUARD_SMOKE_REPORT", "C:/ambient/frozen-smoke.json")
    monkeypatch.setenv("QT_QPA_PLATFORM", " OffScreen:fontengine=freetype ")

    result = launch([])

    assert result == 0
    assert calls == ["desktop"]
    assert "QT_QPA_PLATFORM" not in launcher_globals["os"].environ
    assert launcher_globals["os"].environ["GRIDGUARD_SMOKE_REPORT"] == (
        "C:/ambient/frozen-smoke.json"
    )


def test_frozen_launcher_requires_an_explicit_validated_smoke_report_path(
    monkeypatch,
    tmp_path: Path,
) -> None:
    namespace = runpy.run_path(
        str(REPOSITORY_ROOT / "packaging" / "gridguard_launcher.py"),
        run_name="gridguard_launcher_test",
    )
    launch = namespace["_run_launcher"]
    launcher_globals = launch.__globals__
    captured: dict[str, object] = {}
    report_path = tmp_path / "frozen-smoke.json"

    def run_smoke(path: Path) -> int:
        captured["path"] = path
        captured["qt_platform"] = launcher_globals["os"].environ.get("QT_QPA_PLATFORM")
        return 17

    launcher_globals["_run_frozen_smoke"] = run_smoke
    launcher_globals["_run_desktop"] = lambda: captured.setdefault("desktop", True)
    launcher_globals["_frozen_smoke_report_root"] = lambda: tmp_path.resolve()
    monkeypatch.setenv("QT_QPA_PLATFORM", "windows")

    result = launch(["--_gridguard-frozen-smoke-report", str(report_path.resolve())])

    assert result == 17
    assert captured == {
        "path": report_path.resolve(),
        "qt_platform": "offscreen",
    }


def test_frozen_launcher_rejects_relative_or_non_json_smoke_paths(tmp_path: Path) -> None:
    namespace = runpy.run_path(
        str(REPOSITORY_ROOT / "packaging" / "gridguard_launcher.py"),
        run_name="gridguard_launcher_test",
    )
    validate = namespace["_validated_smoke_report_path"]

    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    for invalid_path in (
        "frozen-smoke.json",
        str(allowed_root / "frozen-smoke.txt"),
        str(tmp_path / "frozen-smoke.json"),
    ):
        with pytest.raises(ValueError):
            validate(invalid_path, allowed_root=allowed_root)


def test_frozen_smoke_root_is_derived_from_the_exact_release_layout(
    monkeypatch,
    tmp_path: Path,
) -> None:
    namespace = runpy.run_path(
        str(REPOSITORY_ROOT / "packaging" / "gridguard_launcher.py"),
        run_name="gridguard_launcher_test",
    )
    resolve_root = namespace["_frozen_smoke_report_root"]
    release_root = tmp_path / "release root with spaces"
    executable = release_root / "dist" / "GridGuard Khartoum" / "GridGuard Khartoum.exe"
    executable.parent.mkdir(parents=True)
    executable.touch()
    expected_root = release_root / "build" / "gridguard"
    expected_root.mkdir(parents=True)
    monkeypatch.setattr(namespace["sys"], "executable", str(executable))
    monkeypatch.setenv("GRIDGUARD_SMOKE_REPORT", str(tmp_path / "ambient"))

    assert resolve_root() == expected_root.resolve()


def test_frozen_smoke_path_must_not_exist_and_report_creation_is_exclusive(
    tmp_path: Path,
) -> None:
    namespace = runpy.run_path(
        str(REPOSITORY_ROOT / "packaging" / "gridguard_launcher.py"),
        run_name="gridguard_launcher_test",
    )
    validate = namespace["_validated_smoke_report_path"]
    write_report = namespace["_write_smoke_report"]
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    report_path = allowed_root / "frozen-smoke.json"

    validated = validate(str(report_path), allowed_root=allowed_root)
    write_report(validated, {"status": "PASS"})
    accepted_bytes = report_path.read_bytes()

    with pytest.raises(FileExistsError):
        write_report(validated, {"status": "FAIL"})
    with pytest.raises(ValueError, match="must not already exist"):
        validate(str(report_path), allowed_root=allowed_root)
    assert report_path.read_bytes() == accepted_bytes
    assert accepted_bytes == b'{\n  "status": "PASS"\n}\n'
    assert json.loads(accepted_bytes) == {"status": "PASS"}


@pytest.mark.skipif(sys.platform != "win32", reason="requires Windows Start-Process semantics")
def test_frozen_smoke_argument_quoting_preserves_exactly_two_child_arguments(
    tmp_path: Path,
) -> None:
    space_root = tmp_path / "process probe with spaces"
    space_root.mkdir()
    report_path = space_root / "frozen-smoke.json"
    capture_path = space_root / "captured arguments.json"
    probe_path = space_root / "argument recorder.py"
    powershell_path = space_root / "argument probe.ps1"
    probe_path.write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        "Path(sys.argv[1]).write_text(json.dumps(sys.argv[2:]), encoding='utf-8')\n",
        encoding="utf-8",
    )
    powershell_path.write_text(
        """param(
    [string]$PythonExecutable,
    [string]$ProbePath,
    [string]$CapturePath,
    [string]$ReportPath
)
$childArguments = @(
    ('"{0}"' -f $ProbePath),
    ('"{0}"' -f $CapturePath),
    "--_gridguard-frozen-smoke-report",
    ('"{0}"' -f $ReportPath)
)
$process = Start-Process `
    -FilePath $PythonExecutable `
    -ArgumentList $childArguments `
    -NoNewWindow `
    -Wait `
    -PassThru
exit $process.ExitCode
""",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(powershell_path),
            "-PythonExecutable",
            sys.executable,
            "-ProbePath",
            str(probe_path),
            "-CapturePath",
            str(capture_path),
            "-ReportPath",
            str(report_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    captured = json.loads(capture_path.read_text(encoding="utf-8"))
    assert captured == ["--_gridguard-frozen-smoke-report", str(report_path)]


def test_frozen_smoke_runs_and_installs_the_primary_live_cuda_scenario() -> None:
    launcher = (REPOSITORY_ROOT / "packaging" / "gridguard_launcher.py").read_text(encoding="utf-8")

    assert 'comparison = run_live_scenario("moderate", 50002)' in launcher
    assert "window.live_page.set_comparison(comparison)" in launcher
    assert "math.isfinite" in launcher
    assert '"shortage_level": comparison.shortage_level' in launcher
    assert '"seed": comparison.seed' in launcher
    assert '"controllers": list(controller_names)' in launcher
    assert '"frame_counts": frame_counts' in launcher
    assert '"finite_core_values": True' in launcher
    assert '"installed_in_live_page": True' in launcher
    assert "expected_results_table_rows = len(startup.snapshot.metrics)" in launcher
    assert "if expected_results_table_rows != 3" in launcher
    assert '"expected_results_table_rows": expected_results_table_rows' in launcher
    assert '"results_table_rows": results_table_rows' in launcher


def test_presentation_scripts_use_the_locked_spec_and_exact_executable() -> None:
    build_script = (REPOSITORY_ROOT / "scripts" / "build-presentation.ps1").read_text(
        encoding="utf-8"
    )
    run_script = (REPOSITORY_ROOT / "scripts" / "run-presentation.ps1").read_text(encoding="utf-8")
    helper_script = RELEASE_INTEGRITY_HELPER.read_text(encoding="utf-8")

    assert "packaging\\gridguard.spec" in build_script
    assert "[switch]$VerifyOnly" in build_script
    assert "if (-not $VerifyOnly)" in build_script
    assert "uv sync --locked --extra dev --extra docs" in build_script
    assert "--no-sync pyinstaller" in build_script
    assert "Assert-ReleaseDistributionPolicy" in build_script
    assert (
        build_script.index("--no-sync pyinstaller")
        < build_script.index("Assert-ReleaseDistributionPolicy")
        < build_script.index("Start-Process")
    )
    assert "GRIDGUARD_SOURCE_REPOSITORY" in build_script
    assert "GRIDGUARD_SOURCE_CHECKPOINT" in build_script
    assert "--junitxml" in build_script
    assert "$LASTEXITCODE" in build_script
    assert "$skippedCount -ne 0" in build_script
    assert "UV_CACHE_DIR" not in build_script
    assert re.search(r"[A-Za-z]:\\", build_script) is None
    assert "GRIDGUARD_SMOKE_REPORT" not in build_script
    assert "QT_QPA_PLATFORM" not in build_script
    assert '"--_gridguard-frozen-smoke-report"' in build_script
    assert "-ArgumentList" in build_script
    assert "('\"{0}\"' -f $smokeReport)" in build_script
    assert "GRIDGUARD_SMOKE_REPORT" not in run_script
    assert "QT_QPA_PLATFORM" not in run_script
    assert "Start-Process" in build_script
    assert "-Wait" in build_script
    assert build_script.count("Get-ValidatedFrozenSmokePayload") == 1
    assert "object_pairs_hook=duplicate_safe_object" in helper_script
    assert "parse_constant=reject_constant" in helper_script
    assert "canonical_bytes(payload) != raw" in helper_script
    assert "ROOT_FIELDS" in helper_script
    assert "GPU_FIELDS" in helper_script
    assert "LIVE_FIELDS" in helper_script
    assert "HOUR_ZERO_FIELDS" in helper_script
    assert "'PASS', 'frozen-smoke status'" in helper_script
    assert "expected_release_id" in helper_script
    assert "'moderate'" in helper_script
    assert "50002" in helper_script
    assert "['PPO', 'Fixed Rotation']" in helper_script
    assert "['PPO'], 24" in helper_script
    assert "'Fixed Rotation'], 24" in helper_script
    assert "live['finite_core_values'], True" in helper_script
    assert "live['installed_in_live_page'], True" in helper_script
    assert "live['expected_results_table_rows'], 3" in helper_script
    assert "live['results_table_rows'], 3" in helper_script
    assert 'Join-Path $projectRoot "RELEASE_MANIFEST.json"' in build_script
    assert "Get-FileHash -Algorithm SHA256" in build_script
    assert "$releaseManifest.windows_build.executable_sha256" in build_script
    assert "$releaseManifest.windows_build.executable_size_bytes" in build_script
    assert "$releaseManifest.windows_build.distribution_file_count" in build_script
    assert "$releaseManifest.windows_build.distribution_size_bytes" in build_script
    assert "$releaseManifest.windows_build.frozen_smoke_report.path" in build_script
    assert "$releaseManifest.windows_build.frozen_smoke_report.sha256" in build_script
    assert "$releaseManifest.windows_build.frozen_smoke_report.size_bytes" in build_script
    assert "$releaseManifest.windows_build.validated_runtime.gpu" in build_script
    assert "$releaseManifest.windows_build.validated_runtime.pytorch" in build_script
    assert "$releaseManifest.windows_build.validated_runtime.cuda" in build_script
    assert "$releaseManifest.windows_build.validated_runtime.cupy" in build_script
    assert "GridGuard Khartoum.exe" in run_script
    assert "Start-Process" in run_script
    assert "Test-Path -LiteralPath" in run_script


def test_standard_presentation_build_is_offline_and_source_parity_is_explicit() -> None:
    build_script = (REPOSITORY_ROOT / "scripts" / "build-presentation.ps1").read_text(
        encoding="utf-8"
    )

    assert "[switch]$StrictSourceParity" in build_script
    assert "[string]$SourceRepositoryPath" in build_script
    assert "[string]$SourceCheckpointPath" in build_script
    assert 'Join-Path $repositoryParent "aipower_rl"' not in build_script
    assert "source parity paths require -StrictSourceParity" in build_script
    assert "strict source parity requires GRIDGUARD_SOURCE_REPOSITORY" in build_script
    assert "strict source parity requires GRIDGUARD_SOURCE_CHECKPOINT" in build_script
    assert re.search(
        r'"GRIDGUARD_SOURCE_REPOSITORY"\s*,\s*\$null\s*,\s*"Process"',
        build_script,
    )
    assert re.search(
        r'"GRIDGUARD_SOURCE_CHECKPOINT"\s*,\s*\$null\s*,\s*"Process"',
        build_script,
    )
    assert "uv sync --locked --extra dev --extra docs --offline" in build_script
    assert "uv run --no-sync ruff check ." in build_script
    assert "uv run --no-sync ruff format --check ." in build_script
    assert "uv run --no-sync gridguard-doctor" in build_script
    assert "uv run --no-sync gridguard-evaluate --verify-evidence" in build_script
    assert (
        "uv run --no-sync gridguard-evaluate --reproduce-paired --output "
        '"tmp/release-validation/paired-benchmark-reproduced.json"'
    ) in build_script
    assert (
        "uv run --no-sync gridguard-evaluate --reproduce-robustness --output "
        '"tmp/release-validation/robustness-supplement-reproduced.json"'
    ) in build_script
    assert (
        "uv run --no-sync python tools/machine_validation.py "
        "--collect-pytest-inventory --collection-path "
        '"tmp/release-validation/release-tests.collection.json"'
    ) in build_script
    assert (
        'uv run --no-sync pytest -q --junitxml "tmp/release-validation/release-tests.xml" '
        '-m "not source_archive and not release_artifact"'
    ) in build_script
    assert "--ignore=tests/test_release_manifest.py" not in build_script
    assert (
        "uv run --no-sync pytest -q --junitxml "
        '"tmp/release-validation/source-parity-tests.xml" -m "source_archive"'
    ) in build_script
    assert (
        '$sourceParityReport = Join-Path $releaseValidationDirectory "source-parity-tests.xml"'
        in build_script
    )
    assert "$skippedCount -ne 0" in build_script
    assert "every selected test is required" in build_script
    assert "Assert-NoAmbientPytestControls" in build_script
    assert "[System.StringComparison]::OrdinalIgnoreCase" in build_script
    assert "$testedSourceIdentity = Get-ReleaseSourceTreeIdentity" in build_script
    assert "New-VerifiedReleaseCandidateSource" in build_script
    assert "Assert-ReleaseCandidateSourceIdentity" in build_script
    assert build_script.count("Assert-ReleaseSourceTreeIdentity") >= 2
    assert (
        "uv run --project $projectRoot --directory $candidateRoot --no-sync pyinstaller"
        in build_script
    )
    assert (
        "uv build $candidateRoot --out-dir $projectRoot\\dist --offline "
        "--no-build-isolation --no-create-gitignore --sdist --wheel"
    ) in build_script
    assert (
        build_script.index("--no-sync pyinstaller")
        < build_script.index("uv build $candidateRoot")
        < build_script.index("Assert-ReleaseDistributionPolicy")
    )

    ordered_gates = (
        "uv sync --locked --extra dev --extra docs --offline",
        "uv run --no-sync ruff check .",
        "uv run --no-sync ruff format --check .",
        "uv run --no-sync gridguard-doctor",
        "uv run --no-sync gridguard-evaluate --verify-evidence",
        "uv run --no-sync gridguard-evaluate --reproduce-paired",
        "uv run --no-sync gridguard-evaluate --reproduce-robustness",
        "uv run --no-sync python tools/machine_validation.py --collect-pytest-inventory",
        "uv run --no-sync pytest -q --junitxml",
        "--no-sync pyinstaller",
        "uv build $candidateRoot",
    )
    positions = tuple(build_script.index(gate) for gate in ordered_gates)
    assert positions == tuple(sorted(positions))


def test_pyinstaller_work_tree_is_isolated_from_release_candidate_and_cleaned() -> None:
    build_script = (REPOSITORY_ROOT / "scripts" / "build-presentation.ps1").read_text(
        encoding="utf-8"
    )

    assert r"--workpath $candidateRoot\build\pyinstaller" not in build_script
    assert r'"tmp\release-pyinstaller-work-"' in build_script
    assert "--workpath $pyinstallerWorkRoot" in build_script
    assert (
        "$resolvedPyinstallerWork = [System.IO.Path]::GetFullPath($pyinstallerWorkRoot)"
        in build_script
    )
    assert r'$pyinstallerWorkPrefix = Join-Path $projectRoot "tmp\release-pyinstaller-work-"' in (
        build_script
    )
    assert "$resolvedPyinstallerWork.StartsWith(" in build_script
    assert "$resolvedCandidateTreePrefix" in build_script
    assert "PyInstaller work path must be outside the release candidate source tree" in (
        build_script
    )
    assert "New-Item -ItemType Directory -Path $resolvedPyinstallerWork -Force" in build_script
    assert "Remove-Item -LiteralPath $resolvedPyinstallerWork -Recurse -Force" in build_script
    assert build_script.index("--workpath $pyinstallerWorkRoot") < build_script.index(
        "uv build $candidateRoot"
    )
    assert build_script.index("uv build $candidateRoot") < build_script.index(
        "Remove-Item -LiteralPath $resolvedPyinstallerWork -Recurse -Force"
    )


def test_build_compares_fresh_inputs_and_verify_only_is_self_contained_without_refresh() -> None:
    build_script = (REPOSITORY_ROOT / "scripts" / "build-presentation.ps1").read_text(
        encoding="utf-8"
    )

    visual_verify = "uv run --no-sync python tools/release_validation.py --verify"
    machine_verify = "uv run --no-sync python tools/machine_validation.py --verify"
    assert build_script.count(visual_verify) == 1
    assert build_script.count(machine_verify) == 2
    assert f"{machine_verify} --verify-inputs" in build_script
    assert "tools/release_validation.py --refresh" not in build_script
    assert "tools/machine_validation.py --refresh" not in build_script
    smoke_validation = "$smokePayload = Get-ValidatedFrozenSmokePayload -ReportPath $smokeReport"
    committed_validation = "Assert-CommittedReleaseValidationRecords"
    process_launch = "$smokeProcess = Start-Process"
    process_guard = "if ($smokeProcess.ExitCode -ne 0)"
    assert (
        build_script.index(process_launch)
        < build_script.index(process_guard)
        < build_script.index(smoke_validation)
        < build_script.rindex(committed_validation)
    )
    assert (f"    }}\n}}\n\n{smoke_validation}\n{committed_validation} `") in build_script
    assert re.search(
        r"Assert-ReleaseSourceSnapshot\s*`\s*"
        r"-ProjectRoot\s+\$projectRoot\s*`\s*"
        r"-ManifestPath\s+\$releaseManifestPath\s*`\s*"
        r"-RequireCleanWorktree:\$VerifyOnly",
        build_script,
    )

from __future__ import annotations

import base64
import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

from tests.test_packaging import (
    GRIDGUARD_METADATA_CONTENTS,
    PROVENANCE_SOURCE_FILES,
    RUNTIME_PACKAGE_FILES,
    SDIST_EXCLUDED_SOURCE_PATHS,
    TEST_SOURCE_EXCLUDED_ROOTS,
    _write_reproducible_sdist,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPOSITORY_ROOT / "RELEASE_MANIFEST.json"
RELEASE_HELPER = REPOSITORY_ROOT / "scripts" / "release-integrity.ps1"
EXPECTED_TRACKED_PATHS = {
    "data/evidence/paired-benchmark.json",
    "data/evidence/robustness-supplement.json",
    "data/evidence/source-manifest.json",
    "data/model/manifest.json",
    "data/model/ppo-policy.npz",
    "data/model/training-provenance.yaml",
    "docs/DEVILS_ADVOCATE_AUDIT.md",
    "docs/REVIEW_REMEDIATION.md",
    "docs/decisions/ADR-002-geographic-boundary-map.md",
    "docs/release-machine-validation.json",
    "docs/release-visual-review.json",
    "docs/source/graduation-project-proposal.pdf",
    "output/pdf/gridguard-khartoum-final-report.pdf",
    "output/presentation/gridguard-khartoum-defense.pptx",
    "src/gridguard/assets/map/THIRD_PARTY_DATA.md",
    "src/gridguard/assets/map/khartoum-localities.geojson",
    "src/gridguard/assets/map/khartoum-nile.geojson",
    "src/gridguard/assets/map/map-provenance.json",
}
CANONICAL_TREE_ALGORITHM = "sha256(canonical-lines-v1)"
UNSIGNED_TRUST_BOUNDARY = "UNSIGNED_INTERNAL_CONSISTENCY_REQUIRES_INDEPENDENTLY_TRUSTED_GIT_COMMIT"
WHEEL_PATH = Path("dist/gridguard_khartoum-1.2.0-py3-none-any.whl")
SDIST_PATH = Path("dist/gridguard_khartoum-1.2.0.tar.gz")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(entries: list[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        line = f"{entry['sha256']}\t{entry['size_bytes']}\t{entry['path']}\n"
        digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def _wheel_runtime_path(source_path: str) -> str:
    if source_path.startswith("src/gridguard/assets/"):
        return "gridguard/" + source_path.removeprefix("src/gridguard/")
    return "gridguard/_bundled/" + source_path


def _stage_test_archive_sources(project_root: Path) -> None:
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


def _wheel_record(entries: dict[str, bytes]) -> bytes:
    record_path = "gridguard_khartoum-1.2.0.dist-info/RECORD"
    rows = []
    for member_name, payload in entries.items():
        encoded_digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=")
        rows.append(f"{member_name},sha256={encoded_digest.decode('ascii')},{len(payload)}\n")
    rows.append(f"{record_path},,\n")
    return "".join(rows).encode()


def _write_test_wheel(path: Path) -> None:
    project_root = path.parents[1]
    _stage_test_archive_sources(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = {
        (
            "gridguard/" + source_path.relative_to(project_root / "src" / "gridguard").as_posix()
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
    for metadata_name, payload in GRIDGUARD_METADATA_CONTENTS.items():
        entries[f"{metadata_root}/{metadata_name}"] = payload
    entries[f"{metadata_root}/RECORD"] = _wheel_record(entries)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for member_name, payload in entries.items():
            archive.writestr(member_name, payload)


def _write_test_sdist(path: Path) -> None:
    project_root = path.parents[1]
    _stage_test_archive_sources(project_root)
    manifest = project_root / "RELEASE_MANIFEST.json"
    if not manifest.exists():
        manifest.write_text("{}\n", encoding="utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = GRIDGUARD_METADATA_CONTENTS["METADATA"]
    root = "gridguard_khartoum-1.2.0"
    source_files = sorted(
        (
            source_path
            for source_path in project_root.rglob("*")
            if source_path.is_file()
            and source_path.relative_to(project_root).parts[0] not in TEST_SOURCE_EXCLUDED_ROOTS
            and source_path.relative_to(project_root).as_posix() not in SDIST_EXCLUDED_SOURCE_PATHS
        ),
        key=lambda source_path: source_path.relative_to(project_root).as_posix(),
    )
    entries = [
        (
            f"{root}/{source_path.relative_to(project_root).as_posix()}",
            source_path.read_bytes(),
        )
        for source_path in source_files
    ]
    entries.append((f"{root}/PKG-INFO", metadata))
    _write_reproducible_sdist(path, entries)


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


def _canonical_json(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_smoke_json(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _valid_frozen_smoke_payload(
    *, release_id: str = "gridguard-khartoum-v1.2.0"
) -> dict[str, object]:
    return {
        "gpu": {
            "cupy_version": "14.1.1",
            "device_name": "Test NVIDIA GPU",
            "torch_cuda_version": "12.8",
            "torch_version": "2.8.0+cu128",
        },
        "live_scenario": {
            "controllers": ["PPO", "Fixed Rotation"],
            "device": "cuda:0",
            "expected_results_table_rows": 3,
            "finite_core_values": True,
            "frame_counts": {"Fixed Rotation": 24, "PPO": 24},
            "hour_zero": {
                "demand_mw": 100.0,
                "fixed_rotation_served_mw": 40.0,
                "ppo_served_mw": 50.0,
                "supply_mw": 60.0,
            },
            "installed_in_live_page": True,
            "results_table_rows": 3,
            "seed": 50002,
            "selected_controller": "Fixed Rotation",
            "selected_hour": 23,
            "shortage_level": "moderate",
            "validated_core_value_count": 1,
        },
        "release_id": release_id,
        "status": "PASS",
        "window_title": "GridGuard Khartoum · Offline Power Allocation",
    }


def _run_frozen_smoke_parser(smoke_path: Path) -> subprocess.CompletedProcess[str]:
    command = (
        f". '{RELEASE_HELPER}'; "
        f"Get-ValidatedFrozenSmokePayload -ReportPath '{smoke_path}' | Out-Null"
    )
    return subprocess.run(
        [_powershell(), "-NoProfile", "-Command", command],
        capture_output=True,
        check=False,
        text=True,
    )


def _valid_schema_three_manifest() -> dict[str, object]:
    binding = {
        "path": "artifact.bin",
        "sha256": "1" * 64,
        "size_bytes": 1,
    }
    distribution_file = {
        "path": "GridGuard Khartoum.exe",
        "sha256": "2" * 64,
        "size_bytes": 1,
    }
    return {
        "release_date": "2026-08-27",
        "release_id": "gridguard-khartoum-v1.2.0",
        "schema_version": 3,
        "source": {
            "git_commit": "3" * 40,
            "uv_lock": {**binding, "path": "uv.lock"},
        },
        "tracked_artifacts": [binding],
        "validation": {
            "authenticity_boundary": UNSIGNED_TRUST_BOUNDARY,
            "machine_validation": {
                **binding,
                "path": "docs/release-machine-validation.json",
            },
            "visual_review": {
                **binding,
                "path": "docs/release-visual-review.json",
            },
        },
        "windows_build": {
            "build_kind": "PyInstaller one-directory",
            "distribution": {
                "file_count": 1,
                "files": [distribution_file],
                "path": "dist/GridGuard Khartoum",
                "size_bytes": 1,
                "tree_hash_algorithm": CANONICAL_TREE_ALGORITHM,
                "tree_sha256": "4" * 64,
            },
            "distribution_file_count": 1,
            "distribution_size_bytes": 1,
            "distribution_tree_sha256": "4" * 64,
            "executable_path": "dist/GridGuard Khartoum/GridGuard Khartoum.exe",
            "executable_sha256": "2" * 64,
            "executable_size_bytes": 1,
            "frozen_smoke": "PASS",
            "frozen_smoke_report": {
                **binding,
                "path": "build/gridguard/frozen-smoke.json",
            },
            "no_gpu_behavior": "FAIL_CLOSED",
            "sdist": {
                **binding,
                "path": "dist/gridguard_khartoum-1.2.0.tar.gz",
            },
            "tracked": False,
            "validated_runtime": {
                "cupy": "14.1.1",
                "cuda": "12.8",
                "gpu": "Test NVIDIA GPU",
                "pytorch": "2.8.0+cu128",
            },
            "wheel": {
                **binding,
                "path": "dist/gridguard_khartoum-1.2.0-py3-none-any.whl",
            },
        },
    }


def _run_release_manifest_parser(manifest_path: Path) -> subprocess.CompletedProcess[str]:
    command = (
        f". '{RELEASE_HELPER}'; "
        f"Get-ReleaseManifestPayload -ManifestPath '{manifest_path}' | Out-Null"
    )
    return subprocess.run(
        [_powershell(), "-NoProfile", "-Command", command],
        capture_output=True,
        check=False,
        text=True,
    )


def _write_final_readiness_fixture(
    root: Path,
    *,
    open_finding: str | None = None,
    machine_version: str = "1.2.0",
) -> Path:
    source_commit = "3" * 40
    machine_record = root / "docs" / "release-machine-validation.json"
    machine_record.parent.mkdir(parents=True, exist_ok=True)
    machine_record.write_bytes(
        _canonical_json(
            {
                "schema_name": "gridguard.release_machine_validation",
                "schema_version": 3,
                "release": {
                    "release_id": "gridguard-khartoum-v1.2.0",
                    "version": machine_version,
                },
                "source": {"git_commit": source_commit},
            }
        )
    )
    ledger = root / "docs" / "DEVILS_ADVOCATE_AUDIT.md"
    rows = []
    for number in range(1, 25):
        finding = f"A{number:02d}"
        if finding in {"A01", "A21"}:
            status = "Post-freeze handoff"
        elif finding == "A02":
            status = "Post-freeze attestation"
        else:
            status = "Open" if finding == open_finding else "Closed"
        rows.append(f"| {finding} | High | Finding | Fixed | 6.1 | {status} | Evidence |")
    ledger.write_text("# Audit\n\n" + "\n".join(rows) + "\n", encoding="utf-8")
    manifest_payload = _valid_schema_three_manifest()
    manifest_payload["source"]["git_commit"] = source_commit
    manifest_payload["validation"]["machine_validation"] = {
        "path": "docs/release-machine-validation.json",
        "sha256": _sha256(machine_record),
        "size_bytes": machine_record.stat().st_size,
    }
    manifest = root / "RELEASE_MANIFEST.json"
    manifest.write_bytes(_canonical_json(manifest_payload))
    return manifest


def test_release_manifest_parser_accepts_canonical_schema_three(tmp_path: Path) -> None:
    manifest = tmp_path / "RELEASE_MANIFEST.json"
    manifest.write_bytes(_canonical_json(_valid_schema_three_manifest()))

    result = _run_release_manifest_parser(manifest)

    assert result.returncode == 0, result.stderr


def test_release_manifest_parser_rejects_duplicate_keys(tmp_path: Path) -> None:
    manifest = tmp_path / "RELEASE_MANIFEST.json"
    canonical = _canonical_json(_valid_schema_three_manifest())
    manifest.write_bytes(
        canonical.replace(
            b'  "schema_version": 3,',
            b'  "schema_version": 3,\n  "schema_version": 3,',
            1,
        )
    )

    result = _run_release_manifest_parser(manifest)

    assert result.returncode != 0
    assert "duplicate" in (result.stdout + result.stderr).casefold()


@pytest.mark.parametrize(
    "tamper",
    [
        '{"schema_version":2,"schema_version":2,"windows_build":{}}',
        '{"schema_version":2,"release_id":NaN,"windows_build":{}}',
    ],
)
def test_release_manifest_migration_reader_rejects_ambiguous_json(
    tmp_path: Path,
    tamper: str,
) -> None:
    manifest = tmp_path / "RELEASE_MANIFEST.json"
    manifest.write_text(tamper, encoding="utf-8")
    command = (
        f". '{RELEASE_HELPER}'; Get-ReleaseManifestForUpdate -ManifestPath '{manifest}' | Out-Null"
    )

    result = subprocess.run(
        [_powershell(), "-NoProfile", "-Command", command],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode != 0
    assert any(
        expected in (result.stdout + result.stderr).casefold()
        for expected in ("duplicate", "number")
    )


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1e10000"])
def test_release_manifest_parser_rejects_nonfinite_numbers(
    tmp_path: Path,
    token: str,
) -> None:
    manifest = tmp_path / "RELEASE_MANIFEST.json"
    canonical = _canonical_json(_valid_schema_three_manifest())
    manifest.write_bytes(
        canonical.replace(
            b'"distribution_file_count": 1', f'"distribution_file_count": {token}'.encode(), 1
        )
    )

    result = _run_release_manifest_parser(manifest)

    assert result.returncode != 0
    assert "number" in (result.stdout + result.stderr).casefold()


@pytest.mark.parametrize(
    "encoding",
    [
        lambda payload: json.dumps(payload).encode("utf-8"),
        lambda payload: _canonical_json(payload).rstrip(b"\n"),
        lambda payload: _canonical_json(payload).replace(b"\n", b"\r\n"),
        lambda payload: b"\xef\xbb\xbf" + _canonical_json(payload),
    ],
)
def test_release_manifest_parser_rejects_noncanonical_json(
    tmp_path: Path,
    encoding,
) -> None:
    manifest = tmp_path / "RELEASE_MANIFEST.json"
    manifest.write_bytes(encoding(_valid_schema_three_manifest()))

    result = _run_release_manifest_parser(manifest)

    assert result.returncode != 0
    assert "canonical" in (result.stdout + result.stderr).casefold()


def test_release_manifest_parser_rejects_wrong_schema(tmp_path: Path) -> None:
    payload = _valid_schema_three_manifest()
    payload["schema_version"] = 2
    manifest = tmp_path / "RELEASE_MANIFEST.json"
    manifest.write_bytes(_canonical_json(payload))

    result = _run_release_manifest_parser(manifest)

    assert result.returncode != 0
    assert "schema_version" in (result.stdout + result.stderr)


@pytest.mark.parametrize(
    ("path", "mutation"),
    [
        ((), "unknown"),
        (("source",), "unknown"),
        (("source", "uv_lock"), "missing"),
        (("validation", "machine_validation"), "unknown"),
        (("windows_build",), "unknown"),
        (("windows_build", "distribution"), "missing"),
        (("windows_build", "distribution", "files", "0"), "unknown"),
        (("windows_build", "validated_runtime"), "missing"),
    ],
)
def test_release_manifest_parser_rejects_unknown_or_missing_recursive_fields(
    tmp_path: Path,
    path: tuple[str, ...],
    mutation: str,
) -> None:
    payload = _valid_schema_three_manifest()
    cursor: object = payload
    for field in path:
        if isinstance(cursor, list):
            cursor = cursor[int(field)]
        else:
            assert isinstance(cursor, dict)
            cursor = cursor[field]
    assert isinstance(cursor, dict)
    if mutation == "unknown":
        cursor["unknown_field"] = "not allowed"
    else:
        cursor.pop(next(iter(cursor)))
    manifest = tmp_path / "RELEASE_MANIFEST.json"
    manifest.write_bytes(_canonical_json(payload))

    result = _run_release_manifest_parser(manifest)

    assert result.returncode != 0
    assert "field" in (result.stdout + result.stderr).casefold()


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("schema_version",), True),
        (("release_id",), 12),
        (("source",), []),
        (("source", "uv_lock", "size_bytes"), False),
        (("tracked_artifacts",), {}),
        (("windows_build", "tracked"), 0),
        (("windows_build", "distribution", "files"), {}),
        (("windows_build", "distribution", "files", "0", "sha256"), None),
        (("windows_build", "validated_runtime", "gpu"), None),
    ],
)
def test_release_manifest_parser_rejects_wrong_recursive_types(
    tmp_path: Path,
    path: tuple[str, ...],
    value: object,
) -> None:
    payload = _valid_schema_three_manifest()
    cursor: object = payload
    for field in path[:-1]:
        if isinstance(cursor, list):
            cursor = cursor[int(field)]
        else:
            assert isinstance(cursor, dict)
            cursor = cursor[field]
    assert isinstance(cursor, dict)
    cursor[path[-1]] = value
    manifest = tmp_path / "RELEASE_MANIFEST.json"
    manifest.write_bytes(_canonical_json(payload))

    result = _run_release_manifest_parser(manifest)

    assert result.returncode != 0
    assert "must be" in (result.stdout + result.stderr).casefold()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("build_kind", "PyInstaller one-file"),
        ("tracked", True),
    ],
)
def test_release_manifest_parser_rejects_wrong_build_semantics(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    payload = _valid_schema_three_manifest()
    payload["windows_build"][field] = value
    manifest = tmp_path / "RELEASE_MANIFEST.json"
    manifest.write_bytes(_canonical_json(payload))

    result = _run_release_manifest_parser(manifest)

    assert result.returncode != 0
    assert field in (result.stdout + result.stderr)


@pytest.mark.parametrize(
    ("open_finding", "machine_version", "expected_success"),
    [
        (None, "1.2.0", True),
        ("A03", "1.2.0", False),
        (None, "9.9.9", False),
    ],
)
def test_final_readiness_cross_checks_machine_release_and_audit_ledger_states(
    tmp_path: Path,
    open_finding: str | None,
    machine_version: str,
    expected_success: bool,
) -> None:
    manifest = _write_final_readiness_fixture(
        tmp_path,
        open_finding=open_finding,
        machine_version=machine_version,
    )
    command = (
        f". '{RELEASE_HELPER}'; "
        f"$manifest = Get-ReleaseManifestPayload -ManifestPath '{manifest}'; "
        f"Assert-FinalReleaseReadiness -ProjectRoot '{tmp_path}' "
        "-ReleaseManifest $manifest"
    )

    result = subprocess.run(
        [_powershell(), "-NoProfile", "-Command", command],
        capture_output=True,
        check=False,
        text=True,
    )

    if expected_success:
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0
        assert any(
            label in (result.stdout + result.stderr).casefold()
            for label in ("ledger", "machine-validation")
        )


@pytest.mark.release_artifact
def test_release_manifest_binds_every_tracked_release_artifact() -> None:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert set(payload) == {
        "release_date",
        "release_id",
        "schema_version",
        "source",
        "tracked_artifacts",
        "validation",
        "windows_build",
    }
    assert payload["schema_version"] == 3
    assert payload["release_id"] == "gridguard-khartoum-v1.2.0"
    assert payload["release_date"] == "2026-08-27"

    source = payload["source"]
    assert set(source) == {"git_commit", "uv_lock"}
    assert re.fullmatch(r"[0-9a-f]{40}", source["git_commit"])
    assert source["uv_lock"]["path"] == "uv.lock"
    assert set(source["uv_lock"]) == {"path", "sha256", "size_bytes"}
    assert source["uv_lock"]["size_bytes"] == (REPOSITORY_ROOT / "uv.lock").stat().st_size
    assert source["uv_lock"]["sha256"] == _sha256(REPOSITORY_ROOT / "uv.lock")

    artifacts = payload["tracked_artifacts"]
    assert {entry["path"] for entry in artifacts} == EXPECTED_TRACKED_PATHS
    assert [entry["path"] for entry in artifacts] == sorted(EXPECTED_TRACKED_PATHS)
    for entry in artifacts:
        assert set(entry) == {"path", "sha256", "size_bytes"}
        relative_path = Path(entry["path"])
        assert not relative_path.is_absolute()
        artifact_path = REPOSITORY_ROOT / relative_path
        assert artifact_path.is_file()
        assert artifact_path.stat().st_size == entry["size_bytes"]
        assert _sha256(artifact_path) == entry["sha256"]


@pytest.mark.release_artifact
def test_release_manifest_binds_validation_records_and_unsigned_trust_boundary() -> None:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    validation = payload["validation"]

    assert set(validation) == {
        "authenticity_boundary",
        "machine_validation",
        "visual_review",
    }
    assert validation["authenticity_boundary"] == UNSIGNED_TRUST_BOUNDARY
    expected_paths = {
        "machine_validation": "docs/release-machine-validation.json",
        "visual_review": "docs/release-visual-review.json",
    }
    for field, relative_path in expected_paths.items():
        binding = validation[field]
        assert set(binding) == {"path", "sha256", "size_bytes"}
        assert binding["path"] == relative_path
        artifact_path = REPOSITORY_ROOT / relative_path
        assert binding["size_bytes"] == artifact_path.stat().st_size
        assert binding["sha256"] == _sha256(artifact_path)


@pytest.mark.release_artifact
def test_release_manifest_binds_the_complete_sorted_distribution_inventory() -> None:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    windows_build = payload["windows_build"]
    distribution = windows_build["distribution"]
    files = distribution["files"]

    assert distribution["path"] == "dist/GridGuard Khartoum"
    assert distribution["tree_hash_algorithm"] == CANONICAL_TREE_ALGORITHM
    assert set(distribution) == {
        "file_count",
        "files",
        "path",
        "size_bytes",
        "tree_hash_algorithm",
        "tree_sha256",
    }
    assert files
    assert [entry["path"] for entry in files] == sorted(entry["path"] for entry in files)
    assert len({entry["path"].casefold() for entry in files}) == len(files)
    assert all(set(entry) == {"path", "sha256", "size_bytes"} for entry in files)
    assert all(re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) for entry in files)
    assert distribution["file_count"] == len(files)
    assert distribution["size_bytes"] == sum(entry["size_bytes"] for entry in files)
    assert distribution["tree_sha256"] == _tree_sha256(files)

    assert windows_build["build_kind"] == "PyInstaller one-directory"
    assert windows_build["tracked"] is False
    assert windows_build["frozen_smoke"] == "PASS"
    assert windows_build["no_gpu_behavior"] == "FAIL_CLOSED"
    assert re.fullmatch(r"[0-9a-f]{64}", windows_build["executable_sha256"])
    assert not Path(windows_build["executable_path"]).is_absolute()

    smoke_binding = windows_build["frozen_smoke_report"]
    assert set(smoke_binding) == {"path", "sha256", "size_bytes"}
    assert re.fullmatch(r"[0-9a-f]{64}", smoke_binding["sha256"])
    assert not Path(smoke_binding["path"]).is_absolute()

    for field, expected_path in (("wheel", WHEEL_PATH), ("sdist", SDIST_PATH)):
        binding = windows_build[field]
        assert set(binding) == {"path", "sha256", "size_bytes"}
        assert binding["path"] == expected_path.as_posix()
        archive_path = REPOSITORY_ROOT / expected_path
        assert binding["size_bytes"] == archive_path.stat().st_size
        assert binding["sha256"] == _sha256(archive_path)

    distribution_path = REPOSITORY_ROOT / distribution["path"]
    assert distribution_path.is_dir()
    actual_paths = {
        path.relative_to(distribution_path).as_posix()
        for path in distribution_path.rglob("*")
        if path.is_file()
    }
    assert actual_paths == {entry["path"] for entry in files}
    actual_size_bytes = sum(
        path.stat().st_size for path in distribution_path.rglob("*") if path.is_file()
    )
    assert actual_size_bytes == distribution["size_bytes"]

    executable = REPOSITORY_ROOT / windows_build["executable_path"]
    assert executable.is_file()
    assert executable.stat().st_size == windows_build["executable_size_bytes"]
    assert _sha256(executable) == windows_build["executable_sha256"]

    smoke_report_path = REPOSITORY_ROOT / smoke_binding["path"]
    assert smoke_report_path.is_file()
    assert smoke_report_path.stat().st_size == smoke_binding["size_bytes"]
    assert _sha256(smoke_report_path) == smoke_binding["sha256"]
    smoke = json.loads(smoke_report_path.read_text(encoding="utf-8"))
    assert smoke["status"] == "PASS"
    assert smoke["live_scenario"]["device"] == "cuda:0"
    assert smoke["live_scenario"]["frame_counts"] == {
        "Fixed Rotation": 24,
        "PPO": 24,
    }
    assert smoke["live_scenario"]["finite_core_values"] is True
    assert smoke["live_scenario"]["installed_in_live_page"] is True
    assert smoke["live_scenario"]["results_table_rows"] == 3


def test_release_integrity_helper_reports_the_exact_tampered_distribution_file(
    tmp_path: Path,
) -> None:
    distribution = tmp_path / "dist"
    (distribution / "_internal").mkdir(parents=True)
    (distribution / "GridGuard Khartoum.exe").write_bytes(b"executable")
    tampered_file = distribution / "_internal" / "runtime.dll"
    tampered_file.write_bytes(b"trusted")
    binding_path = tmp_path / "binding.json"

    create_command = (
        f". '{RELEASE_HELPER}'; "
        f"Get-DistributionBinding -DistributionPath '{distribution}' "
        "-ManifestPathValue 'dist/GridGuard Khartoum' | "
        f"ConvertTo-Json -Depth 10 | Set-Content -LiteralPath '{binding_path}' -Encoding utf8"
    )
    created = subprocess.run(
        [_powershell(), "-NoProfile", "-Command", create_command],
        capture_output=True,
        check=False,
        text=True,
    )
    assert created.returncode == 0, created.stderr

    binding = json.loads(binding_path.read_text(encoding="utf-8-sig"))
    assert [entry["path"] for entry in binding["files"]] == [
        "GridGuard Khartoum.exe",
        "_internal/runtime.dll",
    ]
    assert binding["tree_sha256"] == _tree_sha256(binding["files"])

    tampered_file.write_bytes(b"corrupt")
    verify_command = (
        f". '{RELEASE_HELPER}'; "
        f"$binding = Get-Content -Raw -LiteralPath '{binding_path}' | ConvertFrom-Json; "
        f"Assert-DistributionBinding -DistributionPath '{distribution}' -Binding $binding"
    )
    verified = subprocess.run(
        [_powershell(), "-NoProfile", "-Command", verify_command],
        capture_output=True,
        check=False,
        text=True,
    )

    assert verified.returncode != 0
    assert "_internal/runtime.dll" in (verified.stdout + verified.stderr)
    assert "SHA-256 mismatch" in (verified.stdout + verified.stderr)


@pytest.mark.parametrize("stale_schema_version", [1, 2, 3])
def test_release_integrity_helper_refreshes_source_artifact_and_distribution_bindings(
    tmp_path: Path,
    stale_schema_version: int,
) -> None:
    for relative_path in EXPECTED_TRACKED_PATHS:
        artifact = tmp_path / relative_path
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(f"artifact:{relative_path}".encode())
    (tmp_path / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    machine_record = tmp_path / "docs" / "release-machine-validation.json"
    machine_record.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "source": {
                    "git_commit": "1" * 40,
                    "uv_lock": {
                        "path": "uv.lock",
                        "sha256": _sha256(tmp_path / "uv.lock"),
                        "size_bytes": (tmp_path / "uv.lock").stat().st_size,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    distribution = tmp_path / "dist" / "GridGuard Khartoum"
    (distribution / "_internal").mkdir(parents=True)
    executable = distribution / "GridGuard Khartoum.exe"
    executable.write_bytes(b"executable")
    (distribution / "_internal" / "runtime.dll").write_bytes(b"runtime")
    metadata_root = distribution / "_internal" / "gridguard_khartoum-1.2.0.dist-info"
    metadata_root.mkdir(parents=True)
    for metadata_name, metadata_payload in GRIDGUARD_METADATA_CONTENTS.items():
        (metadata_root / metadata_name).write_bytes(metadata_payload)
    _write_test_wheel(tmp_path / WHEEL_PATH)
    _write_test_sdist(tmp_path / SDIST_PATH)

    smoke_report = tmp_path / "build" / "gridguard" / "frozen-smoke.json"
    smoke_report.parent.mkdir(parents=True)
    smoke_report.write_bytes(_canonical_smoke_json(_valid_frozen_smoke_payload()))

    manifest = tmp_path / "RELEASE_MANIFEST.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": stale_schema_version,
                "release_id": "gridguard-khartoum-v1.0.0",
                "release_date": "2026-08-15",
                "source": {"git_commit": "0" * 40, "uv_lock": {}},
                "tracked_artifacts": [],
                "windows_build": {
                    "build_kind": "untrusted ambient build kind",
                    "tracked": True,
                    "no_gpu_behavior": "FAIL_CLOSED",
                },
                "validation": {},
            }
        ),
        encoding="utf-8",
    )

    refresh_command = (
        f". '{RELEASE_HELPER}'; "
        f"Update-ReleaseManifestBindings -ProjectRoot '{tmp_path}' "
        f"-ManifestPath '{manifest}' -DistributionPath '{distribution}' "
        f"-ExecutablePath '{executable}' -SmokeReportPath '{smoke_report}' "
        f"-GitCommit '{'1' * 40}'"
    )
    refreshed = subprocess.run(
        [_powershell(), "-NoProfile", "-Command", refresh_command],
        capture_output=True,
        check=False,
        text=True,
    )
    assert refreshed.returncode == 0, refreshed.stderr

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 3
    assert payload["release_id"] == "gridguard-khartoum-v1.2.0"
    assert payload["release_date"] == "2026-08-27"
    assert payload["source"]["git_commit"] == "1" * 40
    assert payload["source"]["uv_lock"]["sha256"] == _sha256(tmp_path / "uv.lock")
    assert [item["path"] for item in payload["tracked_artifacts"]] == sorted(EXPECTED_TRACKED_PATHS)
    assert payload["validation"]["authenticity_boundary"] == UNSIGNED_TRUST_BOUNDARY
    assert payload["validation"]["machine_validation"] == {
        "path": "docs/release-machine-validation.json",
        "sha256": _sha256(tmp_path / "docs/release-machine-validation.json"),
        "size_bytes": (tmp_path / "docs/release-machine-validation.json").stat().st_size,
    }
    assert payload["validation"]["visual_review"] == {
        "path": "docs/release-visual-review.json",
        "sha256": _sha256(tmp_path / "docs/release-visual-review.json"),
        "size_bytes": (tmp_path / "docs/release-visual-review.json").stat().st_size,
    }
    assert payload["windows_build"]["distribution"]["file_count"] == 5
    assert payload["windows_build"]["build_kind"] == "PyInstaller one-directory"
    assert payload["windows_build"]["tracked"] is False
    assert (
        payload["windows_build"]["distribution_tree_sha256"]
        == (payload["windows_build"]["distribution"]["tree_sha256"])
    )
    assert payload["windows_build"]["validated_runtime"]["gpu"] == "Test NVIDIA GPU"
    for field, relative_path in (("wheel", WHEEL_PATH), ("sdist", SDIST_PATH)):
        binding = payload["windows_build"][field]
        assert binding == {
            "path": relative_path.as_posix(),
            "sha256": _sha256(tmp_path / relative_path),
            "size_bytes": (tmp_path / relative_path).stat().st_size,
        }

    verify_command = (
        f". '{RELEASE_HELPER}'; "
        f"Assert-ReleaseManifestBinding -ProjectRoot '{tmp_path}' "
        f"-ManifestPath '{manifest}' -DistributionPath '{distribution}' "
        f"-ExecutablePath '{executable}' -SmokeReportPath '{smoke_report}'"
    )
    for relative_path in (WHEEL_PATH, SDIST_PATH):
        archive_path = tmp_path / relative_path
        accepted_bytes = archive_path.read_bytes()
        if relative_path == WHEEL_PATH:
            with zipfile.ZipFile(archive_path, "a") as archive:
                archive.comment = b"same-name replacement"
        else:
            replacement = bytearray(accepted_bytes)
            replacement[4] ^= 1  # Change only the gzip mtime; the tar payload remains valid.
            archive_path.write_bytes(replacement)
        tampered = subprocess.run(
            [_powershell(), "-NoProfile", "-Command", verify_command],
            capture_output=True,
            check=False,
            text=True,
        )
        assert tampered.returncode != 0
        output = tampered.stdout + tampered.stderr
        assert relative_path.as_posix() in output.replace("\\", "/")
        assert "mismatch" in output.lower()
        archive_path.write_bytes(accepted_bytes)

    embedded_metadata = (
        distribution / "_internal" / "gridguard_khartoum-1.2.0.dist-info" / "METADATA"
    )
    embedded_metadata.write_bytes(
        GRIDGUARD_METADATA_CONTENTS["METADATA"].replace(
            b"Requires-Dist: numpy<3,>=2.0\n",
            b"Requires-Dist: numpy<4,>=2.0\n",
            1,
        )
    )
    semantic_tamper = subprocess.run(
        [_powershell(), "-NoProfile", "-Command", refresh_command],
        capture_output=True,
        check=False,
        text=True,
    )
    assert semantic_tamper.returncode != 0
    output = (semantic_tamper.stdout + semantic_tamper.stderr).casefold()
    assert "metadata" in output and "cross-artifact" in output


def test_release_integrity_rejects_machine_record_from_another_source_snapshot(
    tmp_path: Path,
) -> None:
    uv_lock = tmp_path / "uv.lock"
    uv_lock.write_text("version = 1\n", encoding="utf-8")
    machine_record = tmp_path / "docs" / "release-machine-validation.json"
    machine_record.parent.mkdir(parents=True)
    machine_record.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "source": {
                    "git_commit": "2" * 40,
                    "uv_lock": {
                        "path": "uv.lock",
                        "sha256": _sha256(uv_lock),
                        "size_bytes": uv_lock.stat().st_size,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    command = (
        f". '{RELEASE_HELPER}'; "
        f"$machine = Get-Content -Raw -LiteralPath '{machine_record}' | ConvertFrom-Json; "
        f"$uv = Get-FileBinding -AbsolutePath '{uv_lock}' -ManifestPathValue 'uv.lock'; "
        f"Assert-MachineValidationSourceBinding -ProjectRoot '{tmp_path}' "
        f"-MachineValidationPayload $machine -ExpectedGitCommit '{'1' * 40}' "
        "-ExpectedUvLockBinding $uv"
    )

    result = subprocess.run(
        [_powershell(), "-NoProfile", "-Command", command],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode != 0
    assert "machine-validation source Git commit" in (result.stdout + result.stderr)


def test_release_integrity_helper_rejects_smoke_from_a_different_release(
    tmp_path: Path,
) -> None:
    smoke_report = tmp_path / "frozen-smoke.json"
    smoke_report.write_bytes(
        _canonical_smoke_json(_valid_frozen_smoke_payload(release_id="gridguard-khartoum-v1.1.0"))
    )
    command = (
        f". '{RELEASE_HELPER}'; "
        f"Get-ValidatedFrozenSmokePayload -ReportPath '{smoke_report}' | Out-Null"
    )

    result = subprocess.run(
        [_powershell(), "-NoProfile", "-Command", command],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode != 0
    assert "release_id" in (result.stdout + result.stderr)


def test_frozen_smoke_parser_accepts_only_the_canonical_complete_contract(
    tmp_path: Path,
) -> None:
    smoke_report = tmp_path / "frozen-smoke.json"
    smoke_report.write_bytes(_canonical_smoke_json(_valid_frozen_smoke_payload()))

    result = _run_frozen_smoke_parser(smoke_report)

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "tamper",
    [
        "duplicate",
        "unknown",
        "missing",
        "wrong_type",
        "nonfinite",
        "noncanonical",
    ],
)
def test_frozen_smoke_parser_fails_closed_on_ambiguous_or_incomplete_json(
    tmp_path: Path,
    tamper: str,
) -> None:
    payload = _valid_frozen_smoke_payload()
    if tamper == "unknown":
        payload["unbound"] = "not allowed"
        encoded = _canonical_smoke_json(payload)
    elif tamper == "missing":
        payload["live_scenario"].pop("hour_zero")
        encoded = _canonical_smoke_json(payload)
    elif tamper == "wrong_type":
        payload["live_scenario"]["seed"] = True
        encoded = _canonical_smoke_json(payload)
    elif tamper == "noncanonical":
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    else:
        encoded = _canonical_smoke_json(payload)
        if tamper == "duplicate":
            encoded = encoded.replace(
                b'  "status": "PASS",',
                b'  "status": "PASS",\n  "status": "PASS",',
                1,
            )
        else:
            encoded = encoded.replace(b'"demand_mw": 100.0', b'"demand_mw": NaN', 1)
    smoke_report = tmp_path / "frozen-smoke.json"
    smoke_report.write_bytes(encoded)

    result = _run_frozen_smoke_parser(smoke_report)

    assert result.returncode != 0
    output = (result.stdout + result.stderr).casefold()
    assert "frozen" in output and "smoke" in output


def _initialize_release_source_snapshot(
    tmp_path: Path,
) -> tuple[str, Path, Path]:
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "GridGuard Test"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "gridguard@example.invalid"],
        check=True,
    )
    (tmp_path / "app.py").write_text("VERSION = 1\n", encoding="utf-8")
    audit = tmp_path / "docs" / "DEVILS_ADVOCATE_AUDIT.md"
    audit.parent.mkdir(parents=True)
    audit.write_text("# Pre-freeze audit ledger\n", encoding="utf-8")
    visual_review = tmp_path / "docs" / "release-visual-review.json"
    visual_review.write_text(
        json.dumps({"source_stage": "pre-freeze", "status": "PASS"}),
        encoding="utf-8",
    )
    manifest = tmp_path / "RELEASE_MANIFEST.json"
    manifest_payload = _valid_schema_three_manifest()
    manifest_payload["source"]["git_commit"] = "0" * 40
    manifest.write_bytes(_canonical_json(manifest_payload))
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "add",
            "app.py",
            "docs/DEVILS_ADVOCATE_AUDIT.md",
            "docs/release-visual-review.json",
            "RELEASE_MANIFEST.json",
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-m", "source snapshot"],
        check=True,
        capture_output=True,
        text=True,
    )
    source_commit = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return source_commit, manifest, audit


def _commit_release_attestation_envelope(
    tmp_path: Path,
    *,
    source_commit: str,
    manifest: Path,
) -> None:
    manifest_payload = _valid_schema_three_manifest()
    manifest_payload["source"]["git_commit"] = source_commit
    manifest.write_bytes(_canonical_json(manifest_payload))
    (tmp_path / "docs" / "release-machine-validation.json").write_text(
        json.dumps({"source": {"git_commit": source_commit}, "status": "PASS"}),
        encoding="utf-8",
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "add",
            "RELEASE_MANIFEST.json",
            "docs/release-machine-validation.json",
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-m", "freeze attestation envelope"],
        check=True,
        capture_output=True,
        text=True,
    )


def _assert_release_source_snapshot(
    tmp_path: Path,
    manifest: Path,
    *,
    require_clean_worktree: bool = False,
) -> subprocess.CompletedProcess[str]:
    clean_switch = " -RequireCleanWorktree" if require_clean_worktree else ""
    command = (
        f". '{RELEASE_HELPER}'; "
        f"Assert-ReleaseSourceSnapshot -ProjectRoot '{tmp_path}' "
        f"-ManifestPath '{manifest}'{clean_switch}"
    )
    return subprocess.run(
        [_powershell(), "-NoProfile", "-Command", command],
        capture_output=True,
        check=False,
        text=True,
    )


def test_release_source_snapshot_allows_only_machine_record_and_manifest_after_source(
    tmp_path: Path,
) -> None:
    source_commit, manifest, _audit = _initialize_release_source_snapshot(tmp_path)
    _commit_release_attestation_envelope(
        tmp_path,
        source_commit=source_commit,
        manifest=manifest,
    )
    result = _assert_release_source_snapshot(tmp_path, manifest)

    assert result.returncode == 0, result.stderr


def test_release_source_snapshot_rejects_source_renamed_to_an_allowed_attestation_path(
    tmp_path: Path,
) -> None:
    source_commit, manifest, _audit = _initialize_release_source_snapshot(tmp_path)
    manifest_payload = _valid_schema_three_manifest()
    manifest_payload["source"]["git_commit"] = source_commit
    manifest.write_bytes(_canonical_json(manifest_payload))
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "mv",
            "app.py",
            "docs/release-machine-validation.json",
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "RELEASE_MANIFEST.json"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-m", "rename source into attestation"],
        check=True,
        capture_output=True,
        text=True,
    )

    result = _assert_release_source_snapshot(tmp_path, manifest)

    assert result.returncode != 0
    output = (result.stdout + result.stderr).replace("\\", "/")
    assert "app.py" in output
    assert "source snapshot" in output.casefold()


def test_verified_candidate_source_isolated_from_a_live_tree_mutation(
    tmp_path: Path,
) -> None:
    _source_commit, _manifest, _audit = _initialize_release_source_snapshot(tmp_path)
    candidate = tmp_path / "tmp" / "release-candidate"
    identity_path = tmp_path / "tmp" / "tested-source.json"
    identity_path.parent.mkdir(parents=True)
    create_command = (
        f". '{RELEASE_HELPER}'; "
        f"$identity = Get-ReleaseSourceTreeIdentity -ProjectRoot '{tmp_path}'; "
        f"$identity | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath '{identity_path}' "
        "-Encoding utf8; "
        f"New-VerifiedReleaseCandidateSource -ProjectRoot '{tmp_path}' "
        f"-DestinationPath '{candidate}' -Identity $identity | Out-Null; "
        f"Assert-ReleaseCandidateSourceIdentity -CandidateRoot '{candidate}' "
        "-Identity $identity"
    )
    created = subprocess.run(
        [_powershell(), "-NoProfile", "-Command", create_command],
        capture_output=True,
        check=False,
        text=True,
    )
    assert created.returncode == 0, created.stderr
    assert (candidate / "app.py").read_text(encoding="utf-8") == "VERSION = 1\n"

    (tmp_path / "app.py").write_text("VERSION = 2\n", encoding="utf-8")
    verify_command = (
        f". '{RELEASE_HELPER}'; "
        f"$identity = Get-Content -Raw -LiteralPath '{identity_path}' | ConvertFrom-Json; "
        f"Assert-ReleaseCandidateSourceIdentity -CandidateRoot '{candidate}' "
        "-Identity $identity; "
        f"Assert-ReleaseSourceTreeIdentity -ProjectRoot '{tmp_path}' -Identity $identity"
    )
    verified = subprocess.run(
        [_powershell(), "-NoProfile", "-Command", verify_command],
        capture_output=True,
        check=False,
        text=True,
    )

    assert verified.returncode != 0
    output = verified.stdout + verified.stderr
    assert "tested source identity" in output.casefold()
    assert "app.py" in output
    assert (candidate / "app.py").read_text(encoding="utf-8") == "VERSION = 1\n"


def test_release_handoff_requires_the_attestation_envelope_to_be_committed(
    tmp_path: Path,
) -> None:
    source_commit, manifest, _audit = _initialize_release_source_snapshot(tmp_path)
    manifest_payload = _valid_schema_three_manifest()
    manifest_payload["source"]["git_commit"] = source_commit
    manifest.write_bytes(_canonical_json(manifest_payload))
    (tmp_path / "docs" / "release-machine-validation.json").write_text(
        json.dumps({"source": {"git_commit": source_commit}, "status": "PASS"}),
        encoding="utf-8",
    )

    dirty = _assert_release_source_snapshot(
        tmp_path,
        manifest,
        require_clean_worktree=True,
    )

    assert dirty.returncode != 0
    assert "clean worktree" in (dirty.stdout + dirty.stderr).lower()

    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "add",
            "RELEASE_MANIFEST.json",
            "docs/release-machine-validation.json",
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-m", "freeze attestation envelope"],
        check=True,
        capture_output=True,
        text=True,
    )

    clean = _assert_release_source_snapshot(
        tmp_path,
        manifest,
        require_clean_worktree=True,
    )

    assert clean.returncode == 0, clean.stderr


def test_release_source_snapshot_rejects_a_post_source_audit_ledger_edit(
    tmp_path: Path,
) -> None:
    source_commit, manifest, audit = _initialize_release_source_snapshot(tmp_path)
    _commit_release_attestation_envelope(
        tmp_path,
        source_commit=source_commit,
        manifest=manifest,
    )
    audit.write_text("# Ledger edited after source validation\n", encoding="utf-8")

    result = _assert_release_source_snapshot(tmp_path, manifest)

    assert result.returncode != 0
    assert "docs/DEVILS_ADVOCATE_AUDIT.md" in (result.stdout + result.stderr)


def test_release_source_snapshot_rejects_a_post_source_visual_review_edit(
    tmp_path: Path,
) -> None:
    source_commit, manifest, _audit = _initialize_release_source_snapshot(tmp_path)
    _commit_release_attestation_envelope(
        tmp_path,
        source_commit=source_commit,
        manifest=manifest,
    )
    visual_review = tmp_path / "docs" / "release-visual-review.json"
    visual_review.write_text(
        json.dumps({"source_stage": "post-freeze", "status": "PASS"}),
        encoding="utf-8",
    )

    result = _assert_release_source_snapshot(tmp_path, manifest)

    assert result.returncode != 0
    assert "docs/release-visual-review.json" in (result.stdout + result.stderr)


def test_release_source_snapshot_rejects_an_untracked_publication_record(
    tmp_path: Path,
) -> None:
    source_commit, manifest, _audit = _initialize_release_source_snapshot(tmp_path)
    _commit_release_attestation_envelope(
        tmp_path,
        source_commit=source_commit,
        manifest=manifest,
    )
    publication = tmp_path / "docs" / "release-publication-attestation.json"
    publication.write_text("{}\n", encoding="utf-8")

    result = _assert_release_source_snapshot(tmp_path, manifest)

    assert result.returncode != 0
    assert "docs/release-publication-attestation.json" in (result.stdout + result.stderr)


def test_release_source_snapshot_rejects_a_valid_but_stale_commit(tmp_path: Path) -> None:
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "GridGuard Test"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "gridguard@example.invalid"],
        check=True,
    )
    source_file = tmp_path / "app.py"
    source_file.write_text("VERSION = 1\n", encoding="utf-8")
    manifest = tmp_path / "RELEASE_MANIFEST.json"
    manifest_payload = _valid_schema_three_manifest()
    manifest_payload["source"]["git_commit"] = "0" * 40
    manifest.write_bytes(_canonical_json(manifest_payload))
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "app.py", "RELEASE_MANIFEST.json"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-m", "source snapshot"],
        check=True,
        capture_output=True,
        text=True,
    )
    source_commit = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    machine_record = tmp_path / "docs" / "release-machine-validation.json"
    machine_record.parent.mkdir(parents=True)
    machine_record.write_text(
        json.dumps({"source": {"git_commit": source_commit}}),
        encoding="utf-8",
    )
    manifest_payload["source"]["git_commit"] = source_commit
    manifest.write_bytes(_canonical_json(manifest_payload))
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "add",
            "RELEASE_MANIFEST.json",
            "docs/release-machine-validation.json",
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-m", "freeze manifest"],
        check=True,
        capture_output=True,
        text=True,
    )

    verify_command = (
        f". '{RELEASE_HELPER}'; "
        f"Assert-ReleaseSourceSnapshot -ProjectRoot '{tmp_path}' -ManifestPath '{manifest}'"
    )
    current = subprocess.run(
        [_powershell(), "-NoProfile", "-Command", verify_command],
        capture_output=True,
        check=False,
        text=True,
    )
    assert current.returncode == 0, current.stderr

    source_file.write_text("VERSION = 2\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "app.py"], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-m", "change source"],
        check=True,
        capture_output=True,
        text=True,
    )
    stale = subprocess.run(
        [_powershell(), "-NoProfile", "-Command", verify_command],
        capture_output=True,
        check=False,
        text=True,
    )

    assert stale.returncode != 0
    assert "app.py" in (stale.stdout + stale.stderr)
    assert "source snapshot" in (stale.stdout + stale.stderr).lower()


def test_presentation_launcher_verifies_before_start_and_persists_diagnostics() -> None:
    run_script = (REPOSITORY_ROOT / "scripts" / "run-presentation.ps1").read_text(encoding="utf-8")

    assert "release-integrity.ps1" in run_script
    assert "Assert-ReleaseManifestBinding" in run_script
    assert run_script.index("Assert-ReleaseManifestBinding") < run_script.index("Start-Process")
    assert "Write-LaunchDiagnostic" in run_script
    assert "WaitForExit" in run_script
    assert "GRIDGUARD_STARTUP_DIAGNOSTIC" in run_script


def test_refresh_build_runs_manifest_contract_after_writing_final_bindings() -> None:
    build_script = (REPOSITORY_ROOT / "scripts" / "build-presentation.ps1").read_text(
        encoding="utf-8"
    )

    assert "--ignore=tests/test_release_manifest.py" not in build_script
    refresh_index = build_script.index("Update-ReleaseManifestBindings")
    source_snapshot_index = build_script.rindex("Assert-ReleaseSourceSnapshot")
    release_binding_index = build_script.rindex("Assert-ReleaseManifestBinding")
    assert source_snapshot_index < release_binding_index
    artifact_test_index = build_script.rindex('-m "release_artifact"')
    readiness_index = build_script.rindex("Assert-FinalReleaseReadiness")
    assert refresh_index < release_binding_index < artifact_test_index < readiness_index
    assert "Assert-PytestJUnitPassedWithoutSkips" in build_script
    assert '"release-artifact-tests.xml"' in build_script
    assert "every selected test is required" in build_script

    helper_script = RELEASE_HELPER.read_text(encoding="utf-8")
    assert all(
        token in helper_script
        for token in (
            "'status'",
            "'--porcelain=v1'",
            "'--untracked-files=all'",
            "'--no-renames'",
            "'diff'",
            "'--no-ext-diff'",
            "'--name-status'",
            "'-z'",
        )
    )

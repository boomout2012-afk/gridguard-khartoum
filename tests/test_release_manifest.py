from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPOSITORY_ROOT / "RELEASE_MANIFEST.json"
RELEASE_HELPER = REPOSITORY_ROOT / "scripts" / "release-integrity.ps1"
EXPECTED_TRACKED_PATHS = {
    "data/evidence/paired-benchmark.json",
    "data/evidence/source-manifest.json",
    "data/model/manifest.json",
    "data/model/ppo-policy.npz",
    "docs/decisions/ADR-002-geographic-boundary-map.md",
    "docs/source/graduation-project-proposal.pdf",
    "output/pdf/gridguard-khartoum-final-report.pdf",
    "output/presentation/gridguard-khartoum-defense.pptx",
    "src/gridguard/assets/map/THIRD_PARTY_DATA.md",
    "src/gridguard/assets/map/khartoum-localities.geojson",
    "src/gridguard/assets/map/khartoum-nile.geojson",
    "src/gridguard/assets/map/map-provenance.json",
}
CANONICAL_TREE_ALGORITHM = "sha256(canonical-lines-v1)"


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
    assert payload["schema_version"] == 2
    assert payload["release_id"] == "gridguard-khartoum-v1.0.0"
    assert payload["release_date"] == "2026-08-15"

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

    assert windows_build["tracked"] is False
    assert windows_build["frozen_smoke"] == "PASS"
    assert windows_build["no_gpu_behavior"] == "FAIL_CLOSED"
    assert re.fullmatch(r"[0-9a-f]{64}", windows_build["executable_sha256"])
    assert not Path(windows_build["executable_path"]).is_absolute()

    smoke_binding = windows_build["frozen_smoke_report"]
    assert set(smoke_binding) == {"path", "sha256", "size_bytes"}
    assert re.fullmatch(r"[0-9a-f]{64}", smoke_binding["sha256"])
    assert not Path(smoke_binding["path"]).is_absolute()

    distribution_path = REPOSITORY_ROOT / distribution["path"]
    if distribution_path.is_dir():
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
    if executable.is_file():
        assert executable.stat().st_size == windows_build["executable_size_bytes"]
        assert _sha256(executable) == windows_build["executable_sha256"]

    smoke_report_path = REPOSITORY_ROOT / smoke_binding["path"]
    if smoke_report_path.is_file():
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
        assert smoke["live_scenario"]["results_table_rows"] == 7


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


def test_release_integrity_helper_refreshes_source_artifact_and_distribution_bindings(
    tmp_path: Path,
) -> None:
    for relative_path in EXPECTED_TRACKED_PATHS:
        artifact = tmp_path / relative_path
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(f"artifact:{relative_path}".encode())
    (tmp_path / "uv.lock").write_text("version = 1\n", encoding="utf-8")

    distribution = tmp_path / "dist" / "GridGuard Khartoum"
    (distribution / "_internal").mkdir(parents=True)
    executable = distribution / "GridGuard Khartoum.exe"
    executable.write_bytes(b"executable")
    (distribution / "_internal" / "runtime.dll").write_bytes(b"runtime")

    smoke_report = tmp_path / "build" / "gridguard" / "frozen-smoke.json"
    smoke_report.parent.mkdir(parents=True)
    smoke_report.write_text(
        json.dumps(
            {
                "gpu": {
                    "cupy_version": "14.1.1",
                    "device_name": "Test NVIDIA GPU",
                    "torch_cuda_version": "12.8",
                    "torch_version": "2.8.0+cu128",
                },
                "live_scenario": {
                    "controllers": ["PPO", "Fixed Rotation"],
                    "device": "cuda:0",
                    "expected_results_table_rows": 7,
                    "finite_core_values": True,
                    "frame_counts": {"Fixed Rotation": 24, "PPO": 24},
                    "installed_in_live_page": True,
                    "results_table_rows": 7,
                    "seed": 50002,
                    "selected_controller": "Fixed Rotation",
                    "selected_hour": 23,
                    "shortage_level": "moderate",
                    "validated_core_value_count": 1,
                },
                "status": "PASS",
            }
        ),
        encoding="utf-8",
    )

    manifest = tmp_path / "RELEASE_MANIFEST.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "release_id": "gridguard-khartoum-v1.0.0",
                "release_date": "2026-08-15",
                "tracked_artifacts": [],
                "windows_build": {
                    "build_kind": "PyInstaller one-directory",
                    "tracked": False,
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
    assert payload["source"]["git_commit"] == "1" * 40
    assert payload["source"]["uv_lock"]["sha256"] == _sha256(tmp_path / "uv.lock")
    assert [item["path"] for item in payload["tracked_artifacts"]] == sorted(EXPECTED_TRACKED_PATHS)
    assert payload["windows_build"]["distribution"]["file_count"] == 2
    assert (
        payload["windows_build"]["distribution_tree_sha256"]
        == (payload["windows_build"]["distribution"]["tree_sha256"])
    )
    assert payload["windows_build"]["validated_runtime"]["gpu"] == "Test NVIDIA GPU"


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
    manifest.write_text(
        json.dumps({"schema_version": 2, "source": {"git_commit": "0" * 40}}),
        encoding="utf-8",
    )
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
    manifest.write_text(
        json.dumps({"schema_version": 2, "source": {"git_commit": source_commit}}),
        encoding="utf-8",
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "RELEASE_MANIFEST.json"],
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

    assert "--ignore=tests/test_release_manifest.py" in build_script
    refresh_index = build_script.index("Update-ReleaseManifestBindings")
    manifest_test_index = build_script.index("tests/test_release_manifest.py", refresh_index)
    assert refresh_index < manifest_test_index
    source_snapshot_index = build_script.rindex("Assert-ReleaseSourceSnapshot")
    release_binding_index = build_script.rindex("Assert-ReleaseManifestBinding")
    assert source_snapshot_index < release_binding_index

    helper_script = RELEASE_HELPER.read_text(encoding="utf-8")
    assert "status --porcelain --untracked-files=all" in helper_script

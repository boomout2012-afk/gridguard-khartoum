"""Integrity and confinement checks for the independent portable submission."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

BUILDER_PATH = Path(__file__).resolve().parents[1] / "tools/build_submission_bundle.py"
SPEC = importlib.util.spec_from_file_location("submission_bundle", BUILDER_PATH)
assert SPEC is not None and SPEC.loader is not None
bundle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bundle)


def fixture_project(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    project = tmp_path / "project"
    project.mkdir()
    for directory in bundle.SOURCE_DIRECTORIES:
        (project / directory).mkdir()
    for relative in bundle.SOURCE_ROOT_FILES + bundle.SOURCE_OUTPUTS:
        target = project / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(relative.encode())
    (project / "src/valid.py").write_text("pass\n", encoding="utf-8")
    (project / "src/.env").write_text("PRIVATE_TOKEN=example", encoding="utf-8")
    (project / "src/__pycache__").mkdir()
    (project / "src/__pycache__/valid.pyc").write_bytes(b"cache")
    app = tmp_path / "app"
    app.mkdir()
    (app / "GridGuard Khartoum.exe").write_bytes(b"fixture executable")
    (app / "runtime.dll").write_bytes(b"fixture library")
    verification = tmp_path / "reports"
    verification.mkdir()
    (verification / "audit.json").write_text('{"passed":true}', encoding="utf-8")
    return project, app, tmp_path / "Submission", verification


def test_prepare_refresh_preserves_allowlist_and_rejects_junk(tmp_path: Path) -> None:
    project, app, destination, verification = fixture_project(tmp_path)
    bundle.prepare(project, app, destination)
    assert not (destination / "Source/src/.env").exists()
    assert not (destination / "Source/src/__pycache__").exists()
    assert not (destination / bundle.MANIFEST).exists()
    with pytest.raises(FileExistsError):
        bundle.prepare(project, app, destination)
    (project / "src/valid.py").write_text("changed\n", encoding="utf-8")
    bundle.prepare(project, app, destination, refresh=True, verification_dirs=[verification])
    assert (destination / "Source/src/valid.py").read_text() == "changed\n"
    assert (destination / "Verification/audit.json").is_file()
    (destination / "unexpected.txt").write_text("keep me", encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected files"):
        bundle.prepare(project, app, destination, refresh=True, verification_dirs=[verification])
    assert (destination / "unexpected.txt").read_text() == "keep me"


def test_sealed_manifest_and_archive_round_trip_detect_changes(tmp_path: Path) -> None:
    project, app, destination, verification = fixture_project(tmp_path)
    bundle.prepare(project, app, destination, verification_dirs=[verification])
    result = bundle.finalize(destination)
    with zipfile.ZipFile(result["archive"]) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == result["files"] + 2
        assert all(name.startswith("Submission/") for name in archive.namelist())
    assert bundle.main(["verify", "--destination", str(destination)]) == 0
    with pytest.raises(FileExistsError):
        bundle.finalize(destination)
    (destination / "Source/src/valid.py").write_text("evil\n", encoding="utf-8")
    assert bundle.main(["verify", "--destination", str(destination)]) == 1


@pytest.mark.parametrize(
    "relative",
    ["../outside", "/absolute", "C:/drive", "a/../b", "a//b", "a\\b", "a:stream", "trailing."],
)
def test_paths_cannot_escape_or_alias(tmp_path: Path, relative: str) -> None:
    with pytest.raises(ValueError, match="Unsafe"):
        bundle.confined(tmp_path, relative)


def test_links_are_rejected_before_copy(tmp_path: Path) -> None:
    project, app, destination, _ = fixture_project(tmp_path)
    source = project / "src/valid.py"
    link = project / "src/link.py"
    try:
        link.symlink_to(source)
    except OSError:
        # Windows without Developer Mode can still expose a junction/reparse directory.
        if os.name != "nt":
            raise
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(project / "tests")],
            check=True,
            capture_output=True,
        )
    with pytest.raises(ValueError, match="Links/reparse"):
        bundle.prepare(project, app, destination)


def test_powershell_verifier_runs_without_python_and_rejects_tamper(tmp_path: Path) -> None:
    executable = shutil.which("powershell") or shutil.which("pwsh")
    if executable is None:
        pytest.skip("Windows PowerShell or PowerShell Core required for verifier integration")
    project, app, destination, verification = fixture_project(tmp_path)
    bundle.prepare(project, app, destination, verification_dirs=[verification])
    bundle.finalize(destination)
    command = [
        executable,
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(destination / "Verify Submission.ps1"),
        "-Json",
    ]
    clean = subprocess.run(command, capture_output=True, text=True, check=False)
    assert clean.returncode == 0, clean.stderr
    assert json.loads(clean.stdout)["passed"]
    (destination / "App/GridGuard Khartoum/runtime.dll").write_bytes(b"tampered library")
    corrupt = subprocess.run(command, capture_output=True, text=True, check=False)
    assert corrupt.returncode != 0
    assert "mismatch" in corrupt.stderr

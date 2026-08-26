from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_release_lockfile_has_platform_independent_checkout_bytes() -> None:
    attributes = {
        line.strip()
        for line in (ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert "uv.lock text eol=lf" in attributes
    assert b"\r\n" not in (ROOT / "uv.lock").read_bytes()


def test_build_backend_is_exactly_pinned_in_the_locked_dev_environment() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    build_requirements = project["build-system"]["requires"]

    assert len(build_requirements) == 1
    match = re.fullmatch(r"hatchling==(?P<version>\d+\.\d+\.\d+)", build_requirements[0])
    assert match is not None, "Hatchling must be exact-pinned for hermetic no-isolation builds"
    assert build_requirements[0] in project["project"]["optional-dependencies"]["dev"]

    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    locked_hatchling = [package for package in lock["package"] if package["name"] == "hatchling"]
    assert len(locked_hatchling) == 1
    assert locked_hatchling[0]["version"] == match.group("version")


def test_release_artifact_marker_is_registered() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    markers = project["tool"]["pytest"]["ini_options"]["markers"]

    assert any(marker.startswith("release_artifact:") for marker in markers)


def test_quality_workflow_runs_locked_windows_checks_without_release_only_tests() -> None:
    workflow = (ROOT / ".github" / "workflows" / "quality.yml").read_text(encoding="utf-8")

    assert "pull_request:" in workflow
    assert "push:" in workflow
    assert "permissions:\n  contents: read" in workflow
    assert "cancel-in-progress: true" in workflow
    assert "runs-on: windows-latest" in workflow
    assert "actions/checkout@d23441a48e516b6c34aea4fa41551a30e30af803" in workflow
    assert "actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1" in workflow
    assert "astral-sh/setup-uv@c771a70e6277c0a99b617c7a806ffedaca235ff9" in workflow
    assert "uv sync --locked --extra dev --extra docs" in workflow
    assert "uv run --no-sync ruff check ." in workflow
    assert "uv run --no-sync ruff format --check ." in workflow
    assert (
        'uv run --no-sync pytest -m "not gpu and not source_archive and not release_artifact"'
        in workflow
    )
    assert (
        "uv run --no-sync --offline -- uv build --offline --no-index --no-build-isolation"
        in workflow
    )


def test_release_workflow_is_manual_and_targets_the_presentation_gpu() -> None:
    workflow = (ROOT / ".github" / "workflows" / "presentation-release.yml").read_text(
        encoding="utf-8"
    )

    assert "workflow_dispatch:" in workflow
    assert "runs-on: [self-hosted, Windows, X64, gridguard-presentation-gpu]" in workflow
    assert "fetch-depth: 0" in workflow
    assert ".\\scripts\\build-presentation.ps1" in workflow

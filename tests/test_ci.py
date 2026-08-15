from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_quality_workflow_runs_locked_windows_checks_without_release_only_tests() -> None:
    workflow = (ROOT / ".github" / "workflows" / "quality.yml").read_text(
        encoding="utf-8"
    )

    assert "pull_request:" in workflow
    assert "push:" in workflow
    assert "permissions:\n  contents: read" in workflow
    assert "cancel-in-progress: true" in workflow
    assert "runs-on: windows-latest" in workflow
    assert "actions/checkout@d23441a48e516b6c34aea4fa41551a30e30af803" in workflow
    assert "actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1" in workflow
    assert "astral-sh/setup-uv@c771a70e6277c0a99b617c7a806ffedaca235ff9" in workflow
    assert "uv sync --locked --extra dev --extra docs" in workflow
    assert "uv run ruff check ." in workflow
    assert "uv run ruff format --check ." in workflow
    assert 'uv run pytest -m "not gpu and not source_archive"' in workflow
    assert "uv build" in workflow


def test_release_workflow_is_manual_and_targets_the_presentation_gpu() -> None:
    workflow = (
        ROOT / ".github" / "workflows" / "presentation-release.yml"
    ).read_text(encoding="utf-8")

    assert "workflow_dispatch:" in workflow
    assert "runs-on: [self-hosted, Windows, X64, gridguard-presentation-gpu]" in workflow
    assert "fetch-depth: 0" in workflow
    assert ".\\scripts\\build-presentation.ps1" in workflow

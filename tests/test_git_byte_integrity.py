"""Published SHA-256 identities must survive a Windows Git checkout."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHA256 = re.compile(r"[0-9a-f]{64}")
PUBLISHED_FILES = (
    "README.md",
    "SUBMISSION_README.md",
    "docs/REVIEWER_GUIDE.md",
    "docs/REPOSITORY_REVIEW.md",
    "docs/COMMITTEE_REVISION.md",
    "docs/source/REVIEWED_TEMPLATE.md",
    "docs/FOCUSED_THESIS.md",
    "docs/ALLOCATION_STUDY_PROTOCOL.md",
    "docs/CONSTRAINED_PPO_PROTOCOL.md",
    "docs/FOCUSED_PPO_PROTOCOL.md",
    "docs/COMMITTEE_BASELINE_PROTOCOL.md",
    "docs/PROPOSAL_EXTENSION.md",
    "docs/PROPOSAL_RESULTS.md",
    "docs/academic-cover.json",
    "docs/review-package-manifest.json",
    "docs/source/graduation-project-proposal.pdf",
    "docs/source/literature-review-authors.docx",
    "docs/source/reviewed-thesis-template.docx",
    "output/documents/updated/gridguard-khartoum-thesis.docx",
    "output/documents/updated/gridguard-khartoum-thesis.docx.build.json",
    "output/pdf/gridguard-khartoum-gui-pages.pdf",
    "output/pdf/gridguard-khartoum-gui-pages.build.json",
    "output/pdf/updated/gridguard-khartoum-thesis.pdf",
    "output/pdf/updated/gridguard-khartoum-thesis.build.json",
    "output/pdf/updated/gridguard-khartoum-thesis.source.md",
    "output/pdf/updated/gridguard-khartoum-project-presentation.pdf",
    "output/presentation/updated/gridguard-khartoum-defense.pptx",
    "output/presentation/updated/gridguard-khartoum-defense.build.json",
    "tools/build_focused_thesis.py",
    "tools/build_focused_thesis_docx.py",
    "tools/build_focused_presentation.mjs",
    "tools/build_current_gui_pdf.py",
    "tools/capture_current_app.py",
    "scripts/export-submission-thesis.ps1",
    "tools/render_thesis_math.cjs",
    "tools/plot_focused_thesis.py",
    "tools/plot_training_diagnostics.py",
)


def _is_project_file(value: object) -> bool:
    if not isinstance(value, str) or "/" not in value or "\\" in value:
        return False
    path = Path(value)
    if path.parts[0] not in {"data", "docs", "src", "tools", "scripts", "packaging"}:
        return value in PUBLISHED_FILES and (ROOT / path).is_file()
    return not path.is_absolute() and ".." not in path.parts and (ROOT / path).is_file()


def _referenced_paths(value: object) -> set[str]:
    paths: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if _is_project_file(key) and isinstance(item, str) and SHA256.fullmatch(item):
                paths.add(key)
            elif key == "path" and _is_project_file(item) and "sha256" in value:
                paths.add(item)
            if isinstance(item, (dict, list)):
                paths.update(_referenced_paths(item))
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, (dict, list)):
                paths.update(_referenced_paths(item))
    return paths


def _published_paths() -> list[str]:
    paths = set(PUBLISHED_FILES)
    for directory in ("data/evidence", "data/model", "docs/assets"):
        paths.update(
            path.relative_to(ROOT).as_posix()
            for path in (ROOT / directory).rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
        )
    paths.update(path.relative_to(ROOT).as_posix() for path in (ROOT / "docs").glob("*.sha256"))
    paths.update(
        path.relative_to(ROOT).as_posix() for path in (ROOT / "docs").glob("*-validation.json")
    )
    for relative in tuple(paths):
        path = ROOT / relative
        if path.suffix == ".json":
            paths.update(_referenced_paths(json.loads(path.read_text(encoding="utf-8"))))
    return sorted(paths)


def test_git_stage_and_clean_checkout_preserve_published_bytes(tmp_path: Path) -> None:
    paths = _published_paths()
    repository = tmp_path / "repository"
    checkout = tmp_path / "checkout"
    repository.mkdir()
    checkout.mkdir()
    expected = {}
    for relative in (".gitattributes", *paths):
        source = ROOT / relative
        destination = repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        expected[relative] = hashlib.sha256(source.read_bytes()).hexdigest()

    def git(*arguments: str, input_bytes: bytes | None = None) -> bytes:
        return subprocess.run(
            ["git", "-C", str(repository), *arguments],
            input=input_bytes,
            capture_output=True,
            check=True,
            timeout=120,
        ).stdout

    git("init", "-q")
    git("config", "core.autocrlf", "true")
    git("config", "core.safecrlf", "false")
    git("-c", f"core.excludesFile={os.devnull}", "add", "--force", "--all")
    # Inspect the staged blobs, not the still-unchanged worktree.
    names = list(expected)
    staged = git("cat-file", "--batch", input_bytes="".join(f":{p}\n" for p in names).encode())
    cursor = 0
    for relative in names:
        header_end = staged.index(b"\n", cursor)
        _, kind, size = staged[cursor:header_end].split()
        assert kind == b"blob", relative
        start = header_end + 1
        end = start + int(size)
        assert hashlib.sha256(staged[start:end]).hexdigest() == expected[relative], relative
        cursor = end + 1
    assert cursor == len(staged)

    git("checkout-index", "--all", f"--prefix={checkout.as_posix()}/")
    for relative, digest in expected.items():
        assert hashlib.sha256((checkout / relative).read_bytes()).hexdigest() == digest, relative

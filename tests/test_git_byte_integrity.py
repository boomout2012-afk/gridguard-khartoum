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
    "docs/LITERATURE_REVIEW_AUDIT.md",
    "docs/FOCUSED_THESIS.md",
    "docs/ALLOCATION_STUDY_PROTOCOL.md",
    "docs/CONSTRAINED_PPO_PROTOCOL.md",
    "docs/FOCUSED_PPO_PROTOCOL.md",
    "docs/COMMITTEE_BASELINE_PROTOCOL.md",
    "docs/THESIS_LEARNING_AUDIT_PROTOCOL.md",
    "docs/PROPOSAL_EXTENSION.md",
    "docs/academic-cover.json",
    "docs/focused-ppo-validation.json",
    "docs/focused-ppo-validation.sha256",
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
    "tools/assemble_focused_thesis_content.py",
    "tools/thesis_learning_content.py",
    "tools/interpret_focused_study.py",
)
LOCAL_ONLY_FILES = frozenset(
    {"data/evidence/allocation-study.md", "data/evidence/allocation-validation.md"}
)
PUBLIC_ASSET_ROOTS = ("data/evidence", "data/model", "docs/assets")


def _is_public_path(relative: str) -> bool:
    """Keep local notes out of package inventories, including in a source archive.

    Documentation is an explicit allowlist. Scientific records and current visual
    assets remain directory-based so new records cannot be omitted accidentally.
    This contract deliberately does not depend on an available Git worktree.
    """
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or "\\" in relative:
        return False
    if "__pycache__" in path.parts or path.suffix == ".pyc":
        return False
    if relative in LOCAL_ONLY_FILES or relative.startswith("docs/assets/screenshots/"):
        return False
    if relative in PUBLISHED_FILES:
        return True
    if any(relative.startswith(f"{root}/") for root in PUBLIC_ASSET_ROOTS):
        return True
    return bool(path.parts and path.parts[0] in {"src", "tools", "scripts", "packaging", "tests"})


def _is_project_file(value: object) -> bool:
    if not isinstance(value, str) or "/" not in value or "\\" in value:
        return False
    return _is_public_path(value) and (ROOT / value).is_file()


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
    for directory in PUBLIC_ASSET_ROOTS:
        paths.update(
            path.relative_to(ROOT).as_posix()
            for path in (ROOT / directory).rglob("*")
            if path.is_file() and _is_public_path(path.relative_to(ROOT).as_posix())
        )
    for relative in tuple(paths):
        if relative == "docs/review-package-manifest.json":
            # The generated inventory cannot make its old entries public inputs.
            continue
        path = ROOT / relative
        if path.suffix == ".json":
            paths.update(_referenced_paths(json.loads(path.read_text(encoding="utf-8"))))
    return sorted(paths)


def test_local_notes_do_not_reenter_the_public_inventory_through_a_receipt() -> None:
    receipt = {
        "files": [
            {"path": "docs/release-validation.json", "sha256": "0" * 64},
            {"path": "docs/REPOSITORY_REVIEW.md", "sha256": "0" * 64},
            {"path": "data/evidence/allocation-study.md", "sha256": "0" * 64},
            {"path": "docs/assets/screenshots/overview.png", "sha256": "0" * 64},
        ]
    }
    assert not _referenced_paths(receipt)
    assert not (LOCAL_ONLY_FILES & set(_published_paths()))


def test_fresh_audit_frozen_dependencies_are_in_published_package() -> None:
    freeze = json.loads(
        (ROOT / "data/evidence/thesis-learning-audit.execution-freeze.json").read_text()
    )
    assert set(freeze["source_sha256"]).issubset(_published_paths())


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

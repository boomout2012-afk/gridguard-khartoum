from __future__ import annotations

import html
import re
import zipfile
from pathlib import Path

from pypdf import PdfReader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PAIRED_SHA256 = "2c3e71e48c4c7bff4fe609d42a76ca4f770f6dd717cd7d45d5e8e0d99fdf2d11"
REPORT_PATH = REPOSITORY_ROOT / "output" / "pdf" / "gridguard-khartoum-final-report.pdf"
DECK_PATH = REPOSITORY_ROOT / "output" / "presentation" / "gridguard-khartoum-defense.pptx"
TEXT_PATHS = (
    REPOSITORY_ROOT / "README.md",
    REPOSITORY_ROOT / "SPEC.md",
    REPOSITORY_ROOT / "docs" / "METHODOLOGY.md",
    REPOSITORY_ROOT / "docs" / "IMPLEMENTATION.md",
    REPOSITORY_ROOT / "docs" / "RESULTS.md",
    REPOSITORY_ROOT / "docs" / "PROPOSAL_TRACEABILITY.md",
    REPOSITORY_ROOT / "docs" / "PRESENTATION_CHECKLIST.md",
)


def _normalized(value: str) -> str:
    return " ".join(value.split())


def _deck_text() -> str:
    with zipfile.ZipFile(DECK_PATH) as archive:
        xml = "\n".join(
            archive.read(name).decode("utf-8", errors="strict")
            for name in archive.namelist()
            if re.fullmatch(r"ppt/(slides|notesSlides)/[^/]+\.xml", name)
        )
    return html.unescape(" ".join(re.findall(r"<a:t>(.*?)</a:t>", xml)))


def _forbidden_patterns() -> tuple[str, ...]:
    acronym = "".join(("B", "C"))
    return (
        re.escape("behavior " + "cloning"),
        rf"\b{acronym}\s*\+\s*PPO\b",
        rf"\b{acronym}\b",
        re.escape("Smart " + "AI"),
    )


def test_primary_documents_use_only_the_approved_controller_vocabulary() -> None:
    source_text = "\n".join(path.read_text(encoding="utf-8") for path in TEXT_PATHS)
    report_text = "\n".join(page.extract_text() or "" for page in PdfReader(REPORT_PATH).pages)
    corpus = "\n".join((source_text, report_text, _deck_text()))

    assert "PPO" in corpus
    assert "Fixed Rotation" in corpus
    assert not any(
        re.search(pattern, corpus, flags=re.IGNORECASE) for pattern in _forbidden_patterns()
    )


def test_release_claims_keep_counts_hash_and_scope_together() -> None:
    source_text = _normalized("\n".join(path.read_text(encoding="utf-8") for path in TEXT_PATHS))
    report_text = _normalized(
        "\n".join(page.extract_text() or "" for page in PdfReader(REPORT_PATH).pages)
    )

    for corpus in (source_text, report_text):
        assert "96 independent" in corpus
        assert "288 matched scenario conditions" in corpus
        assert "576 controller" in corpus
        assert PAIRED_SHA256 in corpus
        assert "synthetic" in corpus.casefold()
        assert "decision-support" in corpus.casefold()


def test_report_references_are_cited_and_extract_without_broken_glyphs() -> None:
    methodology = (REPOSITORY_ROOT / "docs" / "METHODOLOGY.md").read_text(encoding="utf-8")
    report_text = "\n".join(page.extract_text() or "" for page in PdfReader(REPORT_PATH).pages)

    assert all(f"[{index}]" in methodology for index in range(1, 7))
    assert "References" in report_text
    assert "Proximal Policy Optimization Algorithms" in report_text
    assert "\ufffd" not in report_text
    assert "(cid:" not in report_text


def test_documented_setup_and_artifact_paths_match_the_release() -> None:
    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    spec = (REPOSITORY_ROOT / "SPEC.md").read_text(encoding="utf-8")

    assert "uv sync --extra dev --extra docs" in readme
    assert "uv sync --extra dev --extra docs" in spec
    assert "tools\\build_presentation.ps1" in readme
    assert "Microsoft PowerPoint" in readme
    assert "RELEASE_MANIFEST.json" in readme
    assert "output/presentation/" in spec

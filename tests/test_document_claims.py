from __future__ import annotations

import hashlib
import html
import re
import tomllib
import zipfile
from pathlib import Path

from pypdf import PdfReader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PAIRED_PATH = REPOSITORY_ROOT / "data" / "evidence" / "paired-benchmark.json"
ROBUSTNESS_PATH = REPOSITORY_ROOT / "data" / "evidence" / "robustness-supplement.json"
TRAINING_PATH = REPOSITORY_ROOT / "data" / "model" / "training-provenance.yaml"
AUDIT_PATH = REPOSITORY_ROOT / "docs" / "DEVILS_ADVOCATE_AUDIT.md"
REPORT_PATH = REPOSITORY_ROOT / "output" / "pdf" / "gridguard-khartoum-final-report.pdf"
DECK_PATH = REPOSITORY_ROOT / "output" / "presentation" / "gridguard-khartoum-defense.pptx"
TEXT_PATHS = (
    REPOSITORY_ROOT / "README.md",
    REPOSITORY_ROOT / "SPEC.md",
    REPOSITORY_ROOT / "docs" / "METHODOLOGY.md",
    REPOSITORY_ROOT / "docs" / "RELATED_WORK.md",
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _forbidden_patterns() -> tuple[str, ...]:
    acronym = "".join(("B", "C"))
    return (
        r"\bbehavior\s+cloning\s*\+\s*PPO\b",
        rf"\b{acronym}\s*\+\s*PPO\b",
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


def test_evidence_claims_keep_roles_counts_hashes_and_scope_together() -> None:
    source_text = _normalized("\n".join(path.read_text(encoding="utf-8") for path in TEXT_PATHS))
    report_text = _normalized(
        "\n".join(page.extract_text() or "" for page in PdfReader(REPORT_PATH).pages)
    )

    for corpus in (source_text, report_text):
        assert "96 independent" in corpus
        assert "288 matched scenario conditions" in corpus
        assert "576 controller" in corpus
        assert "accepted reproduction" in corpus.casefold()
        assert "320 independent" in corpus
        assert "960 matched" in corpus
        assert "1,920 controller" in corpus
        assert "25 parameterizations" in corpus
        assert "untouched" in corpus.casefold()
        assert _sha256(PAIRED_PATH) in corpus
        assert _sha256(ROBUSTNESS_PATH) in corpus
        assert _sha256(TRAINING_PATH) in corpus
        assert "synthetic" in corpus.casefold()
        assert "decision-support" in corpus.casefold()


def test_report_references_are_cited_and_extract_without_broken_glyphs() -> None:
    methodology = (REPOSITORY_ROOT / "docs" / "METHODOLOGY.md").read_text(encoding="utf-8")
    related_work = (REPOSITORY_ROOT / "docs" / "RELATED_WORK.md").read_text(encoding="utf-8")
    report_text = "\n".join(page.extract_text() or "" for page in PdfReader(REPORT_PATH).pages)

    assert all(f"[{index}]" in methodology for index in range(1, 9))
    assert all(f"[{index}]" in related_work for index in range(9, 16))
    assert "References" in report_text
    assert "Proximal Policy Optimization Algorithms" in report_text
    assert "Solving the Fair Electric Load Shedding Problem" in report_text
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
    assert "data/model/training-provenance.yaml" in readme
    assert "output/presentation/" in spec


def test_phase_6_audit_ledger_is_complete_and_honest_about_open_work() -> None:
    audit = AUDIT_PATH.read_text(encoding="utf-8")
    rows = [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in audit.splitlines()
        if re.match(r"^\|\s*A\d{2}\s*\|", line)
    ]

    assert [row[0] for row in rows] == [f"A{index:02d}" for index in range(1, 25)]
    assert all(len(row) == 7 for row in rows)
    for finding, severity, issue, disposition, owner, status, evidence in rows:
        assert severity in {"High", "Medium", "Low"}, finding
        assert issue, finding
        assert disposition in {
            "Fixed",
            "Reframed with evidence",
            "Bounded limitation",
            "Optional signing hardening",
        }, finding
        assert re.search(r"\b6\.\d+\b", owner), finding
        assert status in {"Open", "Closed"}, finding
        assert evidence, finding
        if status == "Open":
            assert evidence.startswith("Pending"), finding

    assert (
        "The ledger is a work register, not proof that an open finding is closed."
        in _normalized(audit)
    )


def test_1_2_0_is_the_consistent_unreleased_target_without_rewriting_1_1_0() -> None:
    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    spec = (REPOSITORY_ROOT / "SPEC.md").read_text(encoding="utf-8")
    changelog = (REPOSITORY_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    pyproject = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["version"] == "1.2.0"
    assert "Current development target: **1.2.0 (unreleased)**" in readme
    assert "Target release: 1.2.0 (unreleased)" in spec
    assert "Released baseline: 1.1.0 (2026-08-26)" in spec
    assert "## 1.2.0 - Unreleased" in changelog
    assert "not completed release evidence" in _normalized(changelog)
    assert "## 1.1.0 - 2026-08-26" in changelog


def test_documents_keep_accepted_and_untouched_evidence_roles_distinct() -> None:
    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    spec = (REPOSITORY_ROOT / "SPEC.md").read_text(encoding="utf-8")
    implementation = (REPOSITORY_ROOT / "docs" / "IMPLEMENTATION.md").read_text(encoding="utf-8")
    traceability = (REPOSITORY_ROOT / "docs" / "PROPOSAL_TRACEABILITY.md").read_text(
        encoding="utf-8"
    )

    assert "separately verified\n   untouched robustness supplement" in readme
    assert "Reproduce and verify the frozen benchmark" not in spec
    assert "Frozen benchmark reconstruction matches committed headline values" not in spec
    assert "untouched robustness supplement into presentation data" in implementation
    assert "accepted reproduction projection" in traceability


def test_traceability_exposes_the_complete_training_lineage_and_forecast_horizon() -> None:
    spec = (REPOSITORY_ROOT / "SPEC.md").read_text(encoding="utf-8")
    traceability = (REPOSITORY_ROOT / "docs" / "PROPOSAL_TRACEABILITY.md").read_text(
        encoding="utf-8"
    )

    for source_path in (
        "artifacts/training_manifest.json",
        "artifacts/training_metrics.jsonl",
        "experiments/s2-600-hidden512.json",
        "src/aipower_rl/training.py",
        "src/aipower_rl/learning.py",
        "src/aipower_rl/experiments.py",
    ):
        assert source_path in traceability
    assert "one-hour-ahead forecast comparison using a three-hour trailing window" in _normalized(
        traceability
    )
    assert "Revision: 2026-08-26 (target release 1.2.0 audit contract)" in spec


def test_report_summary_uses_canonical_outage_and_percentage_point_units() -> None:
    report_text = "\n".join(page.extract_text() or "" for page in PdfReader(REPORT_PATH).pages)

    assert "102.1 locality-hours" in report_text
    assert "20.2 locality-hours" in report_text
    assert "+9.15 pp" in report_text
    assert "+26.39 pp" in report_text

from __future__ import annotations

import hashlib
import html
import json
import re
import tomllib
import zipfile
from pathlib import Path

from pypdf import PdfReader

from gridguard.claim_contract import HOLDOUT_CUSTODY_LIMITATION
from gridguard.robustness import ROBUSTNESS_SEED_SEPARATION_CLAIM

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PAIRED_PATH = REPOSITORY_ROOT / "data" / "evidence" / "paired-benchmark.json"
ROBUSTNESS_PATH = REPOSITORY_ROOT / "data" / "evidence" / "robustness-supplement.json"
TRAINING_PATH = REPOSITORY_ROOT / "data" / "model" / "training-provenance.yaml"
AUDIT_PATH = REPOSITORY_ROOT / "docs" / "DEVILS_ADVOCATE_AUDIT.md"
RESULTS_PATH = REPOSITORY_ROOT / "docs" / "RESULTS.md"
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
CURRENT_CLAIM_SOURCE_PATHS = (
    *TEXT_PATHS,
    REPOSITORY_ROOT / "docs" / "REPORT.md",
    REPOSITORY_ROOT / "src" / "gridguard" / "ui" / "presentation_model.py",
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


def test_baseline_tier_names_and_capitalization_are_exact_in_reviewer_surfaces() -> None:
    vocabulary_paths = (
        *CURRENT_CLAIM_SOURCE_PATHS,
        AUDIT_PATH,
        REPOSITORY_ROOT / "docs" / "REVIEW_REMEDIATION.md",
        REPOSITORY_ROOT / "tasks" / "plan.md",
        REPOSITORY_ROOT / "tasks" / "todo.md",
        REPOSITORY_ROOT / "src" / "gridguard" / "claim_contract.py",
        REPOSITORY_ROOT / "src" / "gridguard" / "ui" / "components.py",
        REPOSITORY_ROOT / "tools" / "build_presentation.ps1",
    )
    tier_disclosures = "\n".join(
        (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")
        for relative_path in (
            "docs/REPORT.md",
            "docs/PROPOSAL_TRACEABILITY.md",
            "docs/PRESENTATION_CHECKLIST.md",
            "tools/build_presentation.ps1",
        )
    )

    vocabulary_corpus = "\n".join(path.read_text(encoding="utf-8") for path in vocabulary_paths)

    assert "fixed rotation" not in vocabulary_corpus
    assert "Fixed rotation" not in vocabulary_corpus
    assert "Off and Full Service" in tier_disclosures
    assert "Off/Full" not in vocabulary_corpus
    assert "Off-and-Full-Service-only" not in vocabulary_corpus
    assert "no-Critical-Only" not in vocabulary_corpus
    assert "Mild/Moderate/Severe" not in vocabulary_corpus


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
        assert "documented-seed-disjoint holdout" in corpus.casefold()
        assert "untouched confirmatory" not in corpus.casefold()
        assert _sha256(PAIRED_PATH) in corpus
        assert _sha256(ROBUSTNESS_PATH) in corpus
        assert _sha256(TRAINING_PATH) in corpus
        assert "synthetic" in corpus.casefold()
        assert "decision-support" in corpus.casefold()


def test_current_claim_sources_do_not_use_the_legacy_untouched_label() -> None:
    for path in CURRENT_CLAIM_SOURCE_PATHS:
        assert "untouched" not in path.read_text(encoding="utf-8").casefold(), path


def test_current_assurance_surfaces_do_not_claim_a_confirmatory_comparison() -> None:
    assurance_paths = (
        *CURRENT_CLAIM_SOURCE_PATHS,
        REPOSITORY_ROOT / "docs" / "REVIEW_REMEDIATION.md",
    )
    for path in assurance_paths:
        assert "confirmatory comparison" not in path.read_text(encoding="utf-8").casefold(), path


def test_synthetic_rationale_does_not_claim_matching_removes_generator_bias() -> None:
    methodology = (REPOSITORY_ROOT / "docs" / "METHODOLOGY.md").read_text(encoding="utf-8")
    rationale = methodology.split("### Why synthetic data", maxsplit=1)[1].split(
        "## Controller decisions and dispatch", maxsplit=1
    )[0]
    normalized = _normalized(rationale).casefold()

    assert "largely cancel" not in normalized
    assert "internal validity of the controller comparison is therefore preserved" not in normalized
    assert "not publicly available" not in normalized
    assert "current conditions in sudan prevent" not in normalized
    assert "may affect the controllers differently" in normalized
    assert "no utility measurements were supplied for this study" in normalized


def test_optimization_scope_preserves_the_proposal_conditional_method_objective() -> None:
    for relative_path in ("docs/REPORT.md", "docs/PROPOSAL_TRACEABILITY.md"):
        text = _normalized((REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8"))
        assert "where appropriate" in text, relative_path
        assert "intentionally outside the accepted scope" not in text, relative_path
        assert "no online mathematical dispatch optimizer" in text, relative_path
        assert "optimization-based" in text and "future work" in text, relative_path


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


def test_spec_describes_the_guarded_matched_scenario_without_overclaiming_immutability() -> None:
    spec = (REPOSITORY_ROOT / "SPEC.md").read_text(encoding="utf-8")

    assert "same guarded scenario" in spec
    assert "detected demand/supply mutation fails closed" in spec
    assert "same immutable scenario" not in spec


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
        assert status in {
            "Open",
            "Closed",
            "Post-freeze attestation",
            "Post-freeze handoff",
        }, finding
        assert evidence, finding
        if status == "Open":
            assert evidence.startswith("Pending"), finding

    rows_by_finding = {row[0]: row for row in rows}
    assert rows_by_finding["A01"][5] == "Post-freeze handoff"
    assert rows_by_finding["A02"][5] == "Post-freeze attestation"
    assert rows_by_finding["A21"][5] == "Post-freeze handoff"

    normalized_audit = _normalized(audit)
    assert (
        "The ledger is a work register, not proof that an open finding is closed."
        in normalized_audit
    )
    for required_protocol in (
        "pre-freeze technical closure register",
        "docs/release-visual-review.json must already be frozen",
        "only RELEASE_MANIFEST.json and docs/release-machine-validation.json may change",
        "final handoff response records the exact local HEAD SHA",
        "remote branch SHA",
        "embedding that equality in a new commit would advance HEAD",
        "do not create docs/release-publication-attestation.json",
        "git rev-parse HEAD",
        "git ls-remote --heads",
        "final handoff response is the only A01 publication-evidence record",
        "clean worktree",
    ):
        assert required_protocol.casefold() in normalized_audit.casefold()


def test_1_2_0_is_the_consistent_release_without_rewriting_1_1_0() -> None:
    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    spec = (REPOSITORY_ROOT / "SPEC.md").read_text(encoding="utf-8")
    changelog = (REPOSITORY_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    citation = (REPOSITORY_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    pyproject = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["version"] == "1.2.0"
    assert "Current release: **1.2.0 (2026-08-27)**" in readme
    assert "Release: 1.2.0 (2026-08-27)" in spec
    assert "Released baseline: 1.1.0 (2026-08-26)" in spec
    assert "## 1.2.0 - 2026-08-27" in changelog
    assert "### In progress" not in changelog.split("## 1.1.0", maxsplit=1)[0]
    assert "## 1.1.0 - 2026-08-26" in changelog
    assert re.search(r"^version: 1\.2\.0$", citation, flags=re.MULTILINE)
    assert re.search(r"^date-released: 2026-08-27$", citation, flags=re.MULTILINE)


def test_documents_keep_accepted_and_documented_holdout_roles_distinct() -> None:
    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    spec = (REPOSITORY_ROOT / "SPEC.md").read_text(encoding="utf-8")
    implementation = (REPOSITORY_ROOT / "docs" / "IMPLEMENTATION.md").read_text(encoding="utf-8")
    traceability = (REPOSITORY_ROOT / "docs" / "PROPOSAL_TRACEABILITY.md").read_text(
        encoding="utf-8"
    )

    assert "separately verified\n   documented-seed-disjoint holdout" in readme
    assert "Reproduce and verify the frozen benchmark" not in spec
    assert "Frozen benchmark reconstruction matches committed headline values" not in spec
    assert "documented-seed-disjoint robustness supplement into presentation data" in implementation
    assert "accepted reproduction projection" in traceability


def test_documents_state_holdout_sensitivity_baseline_and_attribution_limits() -> None:
    corpus = _normalized("\n".join(path.read_text(encoding="utf-8") for path in TEXT_PATHS))

    for phrase in (
        "not preregistered or independently custodied",
        "cannot rule out undocumented activity",
        "descriptive, pointwise 95%",
        "unadjusted for multiple comparisons",
        "No behavior-cloning-only ablation",
        "cannot isolate the incremental effect of PPO fine-tuning",
        "supply-blind",
        "cannot select Critical Only",
        "a broad late-stage refactor was deliberately deferred",
    ):
        assert phrase.casefold() in corpus.casefold()


def test_holdout_custody_language_matches_the_verified_seed_separation_scope() -> None:
    assert "training, validation, and model-selection" in HOLDOUT_CUSTODY_LIMITATION
    assert "accepted-evaluation" not in HOLDOUT_CUSTODY_LIMITATION
    assert "robustness sets" not in HOLDOUT_CUSTODY_LIMITATION
    assert "training, validation, and model-selection" in ROBUSTNESS_SEED_SEPARATION_CLAIM


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
    assert "selected seed-43 run" in _normalized(traceability).casefold()
    assert "three-seed candidate experiment" in _normalized(traceability).casefold()
    assert "11,059,200 transitions" in traceability
    assert "Revision: 2026-10-21" in spec
    assert "Release: 1.2.0 (2026-08-27)" in spec


def test_report_summary_uses_canonical_outage_and_percentage_point_units() -> None:
    report_text = "\n".join(page.extract_text() or "" for page in PdfReader(REPORT_PATH).pages)

    assert "102.1 locality-hours" in report_text
    assert "20.2 locality-hours" in report_text
    assert "+9.15 pp" in report_text
    assert "+26.39 pp" in report_text


def test_results_sensitivity_classification_counts_match_the_bound_artifact() -> None:
    payload = json.loads(ROBUSTNESS_PATH.read_text(encoding="utf-8"))
    variants = payload["sensitivity"]["variants"]
    results = RESULTS_PATH.read_text(encoding="utf-8")

    for metric, label in (
        ("critical_outage_locality_hours", "Critical-outage locality-hours"),
        ("fairness", "Jain fairness"),
        ("utilization", "Utilization"),
    ):
        counts = {
            classification: sum(
                variant["difference_in_differences"][metric]["result_classification"]
                == classification
                for variant in variants
            )
            for classification in ("favorable", "ambiguous", "adverse")
        }
        expected_row = (
            f"| {label} | {counts['favorable']} | {counts['ambiguous']} | {counts['adverse']} |"
        )
        assert expected_row in results


def test_release_readme_discloses_the_unsigned_authenticity_boundary() -> None:
    readme = _normalized((REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8"))

    for phrase in (
        "unsigned release manifest",
        "coherent rewrite",
        "private Git branch",
        "exact commit SHA",
        "signing remains optional",
    ):
        assert phrase.casefold() in readme.casefold()


def test_release_documents_keep_source_archive_parity_optional() -> None:
    paths = (
        REPOSITORY_ROOT / "README.md",
        REPOSITORY_ROOT / "SPEC.md",
        REPOSITORY_ROOT / "docs" / "IMPLEMENTATION.md",
        REPOSITORY_ROOT / "docs" / "PRESENTATION_CHECKLIST.md",
    )
    corpus = _normalized("\n".join(path.read_text(encoding="utf-8") for path in paths))

    assert "-StrictSourceParity" in corpus
    assert "source-archive parity" in corpus.casefold()
    assert "normal package build" in corpus.casefold()
    assert "Keep the trusted source archive beside this repository" not in corpus
    assert "mandatory release gate runs on the presentation laptop with the trusted source" not in (
        corpus
    )


def test_reviewer_documents_separate_the_four_release_test_gates() -> None:
    paths = (
        REPOSITORY_ROOT / "README.md",
        REPOSITORY_ROOT / "SPEC.md",
        REPOSITORY_ROOT / "docs" / "METHODOLOGY.md",
        REPOSITORY_ROOT / "docs" / "IMPLEMENTATION.md",
        REPOSITORY_ROOT / "docs" / "decisions" / "ADR-001-gpu-runtime.md",
    )
    portable_command = (
        'uv run --no-sync pytest -m "not gpu and not source_archive and not release_artifact"'
    )

    for path in paths:
        document = path.read_text(encoding="utf-8")
        normalized = _normalized(document).casefold()
        portable_lines = [
            line.strip() for line in document.splitlines() if "pytest" in line and "not gpu" in line
        ]

        assert portable_lines == [portable_command], path
        assert "mandatory presentation-gpu gate" in normalized, path
        assert "strict post-build `release_artifact` gate" in normalized, path
        assert "optional source-archive parity gate" in normalized, path
        assert "normal package build does not require the sibling research archive" in normalized, (
            path
        )


def test_presentation_checklist_points_to_the_durable_source_test_record() -> None:
    checklist = _normalized(
        (REPOSITORY_ROOT / "docs" / "PRESENTATION_CHECKLIST.md").read_text(encoding="utf-8")
    )

    assert "docs/release-machine-validation.json" in checklist
    assert "tmp/release-validation/release-tests.xml" in checklist
    assert "tmp/release-validation/source-tests.xml" not in checklist
    assert "build/release-tests.xml" not in checklist

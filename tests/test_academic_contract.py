from __future__ import annotations

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")


def test_primary_documents_describe_the_verified_administrative_map() -> None:
    corpus = "\n".join(
        _read(path)
        for path in (
            "README.md",
            "SPEC.md",
            "docs/REPORT.md",
            "docs/METHODOLOGY.md",
            "docs/IMPLEMENTATION.md",
            "docs/RESULTS.md",
            "docs/PROPOSAL_TRACEABILITY.md",
            "docs/PRESENTATION_CHECKLIST.md",
        )
    )

    assert "OCHA/HDX" in corpus
    assert "Natural Earth" in corpus
    assert "administrative boundaries" in corpus
    assert "Nile" in corpus
    assert "presentation context only" in corpus
    assert "not electrical-network topology" in corpus
    assert "interactive schematic" not in corpus.casefold()
    assert "schematic map" not in corpus.casefold()


def test_methodology_states_base_draw_reuse_and_canonical_metric_aggregation() -> None:
    methodology = " ".join(_read("docs/METHODOLOGY.md").split())

    assert "severity-neutral" in methodology
    assert "reused unchanged" in methodology
    assert "episode-pooled critical-service ratio" in methodology
    assert "mean hourly supply-utilization ratio" in methodology
    assert "mean hourly cumulative-locality Jain fairness" in methodology
    assert "equally weighted arithmetic mean of the 288 episode values" in methodology


def test_evidence_documents_separate_neutral_integrity_from_release_claims() -> None:
    corpus = "\n".join(
        _read(path)
        for path in (
            "README.md",
            "SPEC.md",
            "docs/METHODOLOGY.md",
            "docs/IMPLEMENTATION.md",
            "docs/RESULTS.md",
        )
    )

    assert "validate_paired_benchmark_payload" in corpus
    assert "assert_paired_benchmark_release_claims" in corpus
    assert "neutral integrity" in corpus
    assert "trusted release gate" in corpus


def test_live_demonstration_contract_covers_aggregate_metrics_and_accessibility() -> None:
    implementation = _read("docs/IMPLEMENTATION.md")
    specification = _read("SPEC.md")

    for phrase in (
        "Live demonstration",
        "Served energy",
        "Critical service",
        "Fairness",
        "Curtailed energy",
        "Power-balance violations",
    ):
        assert phrase in implementation
    assert "synchronized locality list" in implementation
    assert "accessible names and descriptions" in specification
    assert "visible keyboard focus" in specification


def test_report_source_has_complete_academic_structure() -> None:
    report = _read("docs/REPORT.md")

    headings = (
        "## Abstract",
        "## Introduction",
        "## Background",
        "## Discussion",
        "## Conclusion",
    )
    positions = [report.index(heading) for heading in headings]
    assert positions == sorted(positions)
    assert len(report.split("## Conclusion", maxsplit=1)[1].split()) >= 120


def test_results_bind_claims_to_artifacts_and_keep_map_out_of_science() -> None:
    results = " ".join(_read("docs/RESULTS.md").split())

    assert "Results provenance" in results
    assert "3d720ba87a747a5780119decd7144b4051fc36fc72ab50f229f5750ad749fd15" in results
    assert "2c3e71e48c4c7bff4fe609d42a76ca4f770f6dd717cd7d45d5e8e0d99fdf2d11" in results
    assert "df1afbb337047402feb18d2dd26273c671c36761970d0c48c591c62f3ae3d496" in results
    assert "0486a0f7318b5d7c576c13093b27f3f7aa6410329a7c54fb66eb37d3436cdafa" in results
    assert (
        "does not enter scenario generation, controller input, dispatch, metrics, or evidence"
        in results
    )

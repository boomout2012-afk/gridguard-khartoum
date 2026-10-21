from __future__ import annotations

import hashlib
from pathlib import Path

from gridguard.robustness_provenance import ROBUSTNESS_IMPLEMENTATION_SOURCE_BINDINGS

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")


def _sha256(relative_path: str) -> str:
    return hashlib.sha256((REPOSITORY_ROOT / relative_path).read_bytes()).hexdigest()


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


def test_evidence_documents_separate_accepted_reproduction_from_documented_holdout() -> None:
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
    assert "assert_paired_benchmark_integrity" in corpus
    assert "sign-neutral integrity" in corpus
    assert "accepted reproduction" in corpus.casefold()
    assert "documented-seed-disjoint holdout" in corpus.casefold()
    assert "untouched confirmatory" not in corpus.casefold()
    assert "no favorable-result release gate exists" in corpus.casefold()


def test_live_demonstration_contract_covers_aggregate_metrics_and_accessibility() -> None:
    implementation = " ".join(_read("docs/IMPLEMENTATION.md").split()).casefold()
    specification = _read("SPEC.md")

    for phrase in (
        "full-day comparison",
        "served energy",
        "critical service",
        "jain fairness",
        "curtailed energy",
        "power-balance violations",
    ):
        assert phrase in implementation
    assert "for ppo and fixed rotation from the selected episode" in implementation
    assert "tooltip and accessible description" in implementation
    assert "frozen evidence remains separate" in implementation
    assert "synchronized locality dropdown" in implementation
    assert "accessible names and descriptions" in specification
    assert "visible keyboard focus" in specification


def test_report_source_has_complete_academic_structure() -> None:
    report = _read("docs/REPORT.md")

    headings = (
        "## Abstract",
        "## Introduction",
        "## Related Work",
        "## Background",
        "## Discussion",
        "## Conclusion",
    )
    positions = [report.index(heading) for heading in headings]
    assert positions == sorted(positions)
    assert len(report.split("## Conclusion", maxsplit=1)[1].split()) >= 120


def test_report_describes_reference_14_as_a_power_control_survey() -> None:
    report = " ".join(_read("docs/REPORT.md").split())

    assert "survey of reinforcement learning for electric-power control [14]" in report
    assert "reproducible RL evaluation [14]" not in report


def test_implementation_documents_robustness_modules_and_exact_source_bundle() -> None:
    implementation = _read("docs/IMPLEMENTATION.md")

    for module_path in (
        "src/gridguard/robustness.py",
        "src/gridguard/robustness_inference.py",
        "src/gridguard/robustness_provenance.py",
    ):
        assert module_path in implementation
    for source_path, _sha256_digest, _size_bytes in ROBUSTNESS_IMPLEMENTATION_SOURCE_BINDINGS:
        assert f"`{source_path}`" in implementation
    assert "exact bundled scientific source set" in implementation.casefold()


def test_implementation_module_map_documents_extracted_ui_and_claim_modules() -> None:
    implementation = _read("docs/IMPLEMENTATION.md")
    module_map = implementation.split("## Module map", maxsplit=1)[1].split(
        "## Domain and tensor contracts", maxsplit=1
    )[0]

    for module_path in (
        "src/gridguard/claim_contract.py",
        "src/gridguard/ui/window_components.py",
        "src/gridguard/ui/live_components.py",
        "src/gridguard/ui/live_controls.py",
        "src/gridguard/ui/live_visualization.py",
    ):
        assert f"`{module_path}`" in module_map


def test_results_bind_claims_to_artifacts_and_keep_map_out_of_science() -> None:
    results = " ".join(_read("docs/RESULTS.md").split())

    assert "Results provenance" in results
    assert "3d720ba87a747a5780119decd7144b4051fc36fc72ab50f229f5750ad749fd15" in results
    assert _sha256("data/evidence/paired-benchmark.json") in results
    assert _sha256("data/evidence/robustness-supplement.json") in results
    assert _sha256("data/model/training-provenance.yaml") in results
    assert "df1afbb337047402feb18d2dd26273c671c36761970d0c48c591c62f3ae3d496" in results
    assert "0486a0f7318b5d7c576c13093b27f3f7aa6410329a7c54fb66eb37d3436cdafa" in results
    assert (
        "does not enter scenario generation, controller input, dispatch, metrics, or evidence"
        in results
    )


def test_training_and_metric_contract_is_reviewer_discoverable() -> None:
    corpus = " ".join(
        "\n".join(
            _read(path)
            for path in (
                "README.md",
                "SPEC.md",
                "docs/METHODOLOGY.md",
                "docs/IMPLEMENTATION.md",
                "docs/RESULTS.md",
            )
        ).split()
    )

    assert "data/model/training-provenance.yaml" in corpus
    assert "selected seed-43 run" in corpus.casefold()
    assert "600 iterations" in corpus
    assert "153,600 episodes" in corpus
    assert "3,686,400 transitions" in corpus
    assert "three-seed candidate experiment" in corpus.casefold()
    assert "1,800 iterations" in corpus
    assert "460,800 episodes" in corpus
    assert "11,059,200 transitions" in corpus
    assert "directly recorded" in corpus.casefold()
    assert "source-reconstructed" in corpus.casefold()
    assert "arithmetic-derived" in corpus.casefold()
    assert "unavailable historical" in corpus.casefold()
    assert "historical promotion gates" in corpus.casefold()
    assert "not independent headline evidence" in corpus.casefold()
    assert "behavior-cloning initialization" in corpus.casefold()
    assert "no behavior-cloning-only ablation was run" in corpus.casefold()
    assert (
        "cannot isolate the incremental effect of PPO fine-tuning".casefold() in corpus.casefold()
    )
    assert "training-objective score (diagnostic)" in corpus.casefold()
    assert "critical-outage locality-hours" in corpus.casefold()
    assert "fairness" in corpus.casefold()
    assert "utilization" in corpus.casefold()
    assert "power-balance violation" in corpus.casefold()


def test_robustness_scope_and_limitations_are_explicit() -> None:
    corpus = " ".join(
        "\n".join(
            _read(path)
            for path in (
                "README.md",
                "SPEC.md",
                "docs/METHODOLOGY.md",
                "docs/RESULTS.md",
                "docs/REPORT.md",
            )
        ).split()
    )

    for phrase in (
        "seeds 81001-81010",
        "320 independent base draws",
        "960 matched scenario",
        "1,920 controller",
        "24 one-factor-at-a-time",
        "25 parameterizations",
        "persistence",
        "global optimum",
        "shortage-only",
        "supply-blind",
        "cannot select Critical Only",
        "not preregistered or independently custodied",
        "cannot rule out undocumented activity",
        "pointwise 95%",
        "unadjusted for multiple comparisons",
    ):
        assert phrase.casefold() in corpus.casefold()

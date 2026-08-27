from __future__ import annotations

import json
from pathlib import Path

import pytest

from gridguard.evaluation import verify_paired_benchmark
from gridguard.robustness import verify_robustness_supplement
from tools.report_results_contract import (
    ReportResultsClaimError,
    validate_methodology_training_claims,
    validate_rendered_report_sources,
    validate_report_numeric_claims,
    validate_results_numeric_claims,
)
from tools.training_provenance import build_training_summary, load_training_provenance

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = REPOSITORY_ROOT / "docs" / "RESULTS.md"
METHODOLOGY_PATH = REPOSITORY_ROOT / "docs" / "METHODOLOGY.md"
REPORT_PATH = REPOSITORY_ROOT / "docs" / "REPORT.md"
RENDERED_MARKDOWN_PATHS = (
    "docs/REPORT.md",
    "docs/RELATED_WORK.md",
    "docs/METHODOLOGY.md",
    "docs/IMPLEMENTATION.md",
    "docs/RESULTS.md",
    "docs/REFERENCES.md",
)


@pytest.fixture(scope="module")
def verified_evidence() -> tuple[dict, dict]:
    accepted = verify_paired_benchmark(
        REPOSITORY_ROOT / "data" / "evidence" / "paired-benchmark.json",
        frozen_benchmark_path=(REPOSITORY_ROOT / "data" / "evidence" / "frozen-benchmark.json"),
    )
    robustness = verify_robustness_supplement(
        REPOSITORY_ROOT / "data" / "evidence" / "robustness-supplement.json"
    )
    return accepted, robustness


@pytest.fixture(scope="module")
def report_claim_inputs(verified_evidence: tuple[dict, dict]) -> tuple[dict, dict, dict]:
    accepted, robustness = verified_evidence
    training = load_training_provenance(
        REPOSITORY_ROOT / "data" / "model" / "training-provenance.yaml",
        model_manifest_path=REPOSITORY_ROOT / "data" / "model" / "manifest.json",
    )
    return accepted, robustness, training


@pytest.fixture(scope="module")
def report_source_gate_inputs() -> tuple[dict, dict, dict]:
    """Isolate source-gate tests from evidence regeneration performed by another release step."""

    accepted = json.loads(
        (REPOSITORY_ROOT / "data" / "evidence" / "paired-benchmark.json").read_text(
            encoding="utf-8"
        )
    )
    robustness = json.loads(
        (REPOSITORY_ROOT / "data" / "evidence" / "robustness-supplement.json").read_text(
            encoding="utf-8"
        )
    )
    training = load_training_provenance(
        REPOSITORY_ROOT / "data" / "model" / "training-provenance.yaml",
        model_manifest_path=REPOSITORY_ROOT / "data" / "model" / "manifest.json",
    )
    return accepted, robustness, training


def _rendered_markdown_sources() -> dict[str, str]:
    return {
        path: (REPOSITORY_ROOT / path).read_text(encoding="utf-8")
        for path in RENDERED_MARKDOWN_PATHS
    }


def test_committed_results_numeric_claims_match_verified_evidence(
    verified_evidence: tuple[dict, dict],
) -> None:
    accepted, robustness = verified_evidence

    validate_results_numeric_claims(
        RESULTS_PATH.read_text(encoding="utf-8"),
        accepted=accepted,
        robustness=robustness,
    )


@pytest.mark.parametrize(
    ("category", "trusted_text", "tampered_text"),
    (
        (
            "primary outcomes",
            "| Critical-outage locality-hours | 102.1 | 20.2 | -81.9 | "
            "[-82.5, -81.2] | Favorable |",
            "| Critical-outage locality-hours | 102.1 | 20.2 | -80.9 | "
            "[-82.5, -81.2] | Favorable |",
        ),
        (
            "accepted reproduction",
            "| Training-objective score (diagnostic) | 13.969 | 23.512 | "
            "Diagnostic; shares components with reported outcomes |",
            "| Training-objective score (diagnostic) | 14.969 | 23.512 | "
            "Diagnostic; shares components with reported outcomes |",
        ),
        (
            "power-balance safety",
            "Maximum power-balance violation rate is\n0.00%",
            "Maximum power-balance violation rate is\n0.01%",
        ),
        (
            "sensitivity",
            "| Critical-outage locality-hours | 4 | 1 | 19 |",
            "| Critical-outage locality-hours | 5 | 1 | 18 |",
        ),
        (
            "forecast",
            "| Locality-sector MW | Three-hour trailing mean | 2.338 | 3.745 | Persistence |",
            "| Locality-sector MW | Three-hour trailing mean | 2.338 | 3.700 | Persistence |",
        ),
        (
            "forecast",
            "Each method has 235,200 locality-sector predictions and 6,720",
            "Each method has 235,201 locality-sector predictions and 6,720",
        ),
        (
            "baseline interpretation",
            "Mean unused available energy is 2,158.7 MWh for Fixed Rotation versus "
            "150.2 MWh for PPO.",
            "Mean unused available energy is 2,058.7 MWh for Fixed Rotation versus "
            "150.2 MWh for PPO.",
        ),
        (
            "display provenance",
            "df1afbb337047402feb18d2dd26273c671c36761970d0c48c591c62f3ae3d496",
            "0f1afbb337047402feb18d2dd26273c671c36761970d0c48c591c62f3ae3d496",
        ),
    ),
)
def test_numeric_claim_tampering_fails_closed(
    verified_evidence: tuple[dict, dict],
    category: str,
    trusted_text: str,
    tampered_text: str,
) -> None:
    accepted, robustness = verified_evidence
    markdown = RESULTS_PATH.read_text(encoding="utf-8")
    assert trusted_text in markdown

    with pytest.raises(ReportResultsClaimError, match=category):
        validate_results_numeric_claims(
            markdown.replace(trusted_text, tampered_text, 1),
            accepted=accepted,
            robustness=robustness,
        )


def test_report_builder_invokes_numeric_claim_validation_before_rendering() -> None:
    builder_source = (REPOSITORY_ROOT / "tools" / "build_report.py").read_text(encoding="utf-8")

    validation_call = builder_source.index("validate_results_numeric_claims(")
    report_validation_call = builder_source.index("validate_report_numeric_claims(")
    source_bundle_validation = builder_source.index("validate_rendered_report_sources(")
    output_creation = builder_source.index("output_path.parent.mkdir(")
    render_call = builder_source.index("document.multiBuild(")

    assert validation_call < render_call
    assert report_validation_call < output_creation
    assert source_bundle_validation < output_creation


def test_committed_report_numeric_claims_match_verified_inputs(
    report_claim_inputs: tuple[dict, dict, dict],
) -> None:
    accepted, robustness, training = report_claim_inputs

    validate_report_numeric_claims(
        REPORT_PATH.read_text(encoding="utf-8"),
        accepted=accepted,
        robustness=robustness,
        training=training,
    )


def test_committed_rendered_markdown_bundle_matches_the_frozen_release() -> None:
    validate_rendered_report_sources(_rendered_markdown_sources())


def test_rendered_source_digest_rejects_table_row_linebreak_collision() -> None:
    sources = _rendered_markdown_sources()
    trusted = "| Khartoum | 1.00 |"
    assert trusted in sources["docs/METHODOLOGY.md"]
    sources["docs/METHODOLOGY.md"] = sources["docs/METHODOLOGY.md"].replace(
        trusted,
        "| Khartoum\n| 1.00 |",
        1,
    )

    with pytest.raises(ReportResultsClaimError, match="rendered report source digest"):
        validate_rendered_report_sources(sources)


def test_rendered_source_digest_rejects_heading_linebreak_collision() -> None:
    sources = _rendered_markdown_sources()
    trusted = "## Sudan energy-access context"
    assert trusted in sources["docs/RELATED_WORK.md"]
    sources["docs/RELATED_WORK.md"] = sources["docs/RELATED_WORK.md"].replace(
        trusted,
        "##\nSudan energy-access context",
        1,
    )

    with pytest.raises(ReportResultsClaimError, match="rendered report source digest"):
        validate_rendered_report_sources(sources)


@pytest.mark.parametrize("tampered_path", RENDERED_MARKDOWN_PATHS)
def test_builder_rejects_each_tampered_rendered_source_before_output_creation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    report_source_gate_inputs: tuple[dict, dict, dict],
    tampered_path: str,
) -> None:
    from tools import build_report as report_builder

    accepted, robustness, training = report_source_gate_inputs
    source_root = tmp_path / "sources"
    source_root.mkdir()
    copied: dict[str, Path] = {}
    for relative_path, canonical_text in _rendered_markdown_sources().items():
        copied_path = source_root / Path(relative_path).name
        suffix = "\nUnapproved release prose.\n" if relative_path == tampered_path else ""
        copied_path.write_text(canonical_text + suffix, encoding="utf-8")
        copied[relative_path] = copied_path

    monkeypatch.setattr(report_builder, "REPORT_SOURCE_PATH", copied["docs/REPORT.md"])
    monkeypatch.setattr(report_builder, "RELATED_WORK_PATH", copied["docs/RELATED_WORK.md"])
    monkeypatch.setattr(
        report_builder,
        "CHAPTER_PATHS",
        (
            copied["docs/METHODOLOGY.md"],
            copied["docs/IMPLEMENTATION.md"],
            copied["docs/RESULTS.md"],
        ),
    )
    monkeypatch.setattr(report_builder, "REFERENCE_PATH", copied["docs/REFERENCES.md"])
    monkeypatch.setattr(
        report_builder,
        "load_verified_evidence",
        lambda: {
            "accepted": accepted,
            "robustness": robustness,
            "training_record": training,
            "training": build_training_summary(
                training,
                provenance_path="data/model/training-provenance.yaml",
            ),
        },
    )
    output_path = tmp_path / "must-not-exist" / "report.pdf"

    with pytest.raises(ReportResultsClaimError):
        report_builder.build_report(output_path)

    assert not output_path.parent.exists()


@pytest.mark.parametrize(
    ("category", "trusted_text", "tampered_text"),
    (
        (
            "report holdout",
            "seeds 81001-81010:",
            "seeds 81002-81011:",
        ),
        (
            "report holdout",
            "81010: 320 independent base draws",
            "81010: 321 independent base draws",
        ),
        (
            "report holdout",
            "960 matched scenario conditions",
            "961 matched scenario conditions",
        ),
        (
            "report holdout",
            "1,920\ncontroller evaluations",
            "1,921\ncontroller evaluations",
        ),
        (
            "accepted unused supply",
            "2,158.7 MWh for Fixed Rotation",
            "2,058.7 MWh for Fixed Rotation",
        ),
        (
            "report sensitivity",
            "perturbs 12 allocation components",
            "perturbs 13 allocation components",
        ),
        (
            "report sensitivity",
            "giving 24\none-factor-at-a-time variants",
            "giving 23\none-factor-at-a-time variants",
        ),
        (
            "report sensitivity",
            "25 parameterizations including the base",
            "26 parameterizations including the base",
        ),
        (
            "report training scope",
            "selected seed-43 run used 600\nPPO iterations",
            "selected seed-43 run used 601\nPPO iterations",
        ),
        (
            "report training scope",
            "three-seed candidate experiment\ntotaled 1,800 iterations, 460,800 episodes, "
            "and 11,059,200 transitions",
            "three-seed candidate experiment\ntotaled 1,801 iterations, 460,800 episodes, "
            "and 11,059,200 transitions",
        ),
        (
            "report training provenance taxonomy",
            "source hashes, and evidence category of each recovered field",
            "source hashes, with every recovered field treated as directly recorded",
        ),
        (
            "report training promotion",
            "historical promotion gates are not independent headline\nevidence",
            "historical promotion gates are independent headline\nevidence",
        ),
        (
            "report forecast",
            "the three-hour trailing mean with one-hour persistence",
            "the four-hour trailing mean with one-hour persistence",
        ),
    ),
)
def test_report_numeric_claim_tampering_fails_closed(
    report_claim_inputs: tuple[dict, dict, dict],
    category: str,
    trusted_text: str,
    tampered_text: str,
) -> None:
    accepted, robustness, training = report_claim_inputs
    markdown = REPORT_PATH.read_text(encoding="utf-8")
    assert trusted_text in markdown

    with pytest.raises(ReportResultsClaimError, match=category):
        validate_report_numeric_claims(
            markdown.replace(trusted_text, tampered_text, 1),
            accepted=accepted,
            robustness=robustness,
            training=training,
        )


def test_report_rejects_an_unknown_numeric_section(
    report_claim_inputs: tuple[dict, dict, dict],
) -> None:
    accepted, robustness, training = report_claim_inputs
    markdown = REPORT_PATH.read_text(encoding="utf-8")

    with pytest.raises(ReportResultsClaimError, match="unexpected REPORT section"):
        validate_report_numeric_claims(
            markdown + "\n## Fabricated finding\n\nPPO improved by 999%.\n",
            accepted=accepted,
            robustness=robustness,
            training=training,
        )


def test_report_rejects_a_duplicate_known_section(
    report_claim_inputs: tuple[dict, dict, dict],
) -> None:
    accepted, robustness, training = report_claim_inputs
    markdown = REPORT_PATH.read_text(encoding="utf-8")

    with pytest.raises(ReportResultsClaimError, match="duplicate REPORT section"):
        validate_report_numeric_claims(
            markdown + "\n## Discussion\n\nDuplicate narrative.\n",
            accepted=accepted,
            robustness=robustness,
            training=training,
        )


@pytest.mark.parametrize(
    "fabricated_claim",
    (
        "PPO improved nine hundred percent.",
        "PPO delivered x999 percent improvement.",
        "PPO delivered 9e9 percent improvement.",
        "PPO delivered ninefold improvement.",
        "Accepted evidence SHA-256 is " + "0" * 64 + ".",
    ),
)
def test_report_rejects_unbound_numeric_claim_forms(
    report_claim_inputs: tuple[dict, dict, dict],
    fabricated_claim: str,
) -> None:
    accepted, robustness, training = report_claim_inputs
    markdown = REPORT_PATH.read_text(encoding="utf-8")
    tampered = markdown.replace(
        "\n## Conclusion\n",
        f"\n{fabricated_claim}\n\n## Conclusion\n",
        1,
    )

    with pytest.raises(ReportResultsClaimError, match="unbound numeric claim"):
        validate_report_numeric_claims(
            tampered,
            accepted=accepted,
            robustness=robustness,
            training=training,
        )


@pytest.mark.parametrize(
    "fabricated_claim",
    (
        "PPO doubled performance.",
        "PPO cut outages in half.",
        "PPO improved by a dozen percent.",
        "PPO was twice as effective.",
        "PPO cut outages by a third.",
        "PPO cut outages by a quarter.",
        "PPO ranked first.",
        "PPO was second-best.",
        "PPO tripled performance.",
        "PPO delivered triple the score.",
        "PPO cut outages by ½.",
        "PPO improved by Ⅸ percent.",
        "PPO improved by ⁹⁹⁹ percent.",
        "PPO improved by ⑨⑨⑨ percent.",
    ),
)
def test_report_rejects_unapproved_quantitative_prose(
    report_claim_inputs: tuple[dict, dict, dict],
    fabricated_claim: str,
) -> None:
    accepted, robustness, training = report_claim_inputs
    markdown = REPORT_PATH.read_text(encoding="utf-8")
    tampered = markdown.replace(
        "\n## Conclusion\n",
        f"\n{fabricated_claim}\n\n## Conclusion\n",
        1,
    )

    with pytest.raises(ReportResultsClaimError, match="REPORT content digest"):
        validate_report_numeric_claims(
            tampered,
            accepted=accepted,
            robustness=robustness,
            training=training,
        )


def test_report_rejects_numeric_claim_before_abstract(
    report_claim_inputs: tuple[dict, dict, dict],
) -> None:
    accepted, robustness, training = report_claim_inputs
    markdown = REPORT_PATH.read_text(encoding="utf-8")
    tampered = markdown.replace(
        "\n## Abstract\n",
        "\nPPO improved by 999 percent on the holdout.\n\n## Abstract\n",
        1,
    )

    with pytest.raises(ReportResultsClaimError, match="REPORT preamble"):
        validate_report_numeric_claims(
            tampered,
            accepted=accepted,
            robustness=robustness,
            training=training,
        )


def test_unexpected_results_section_with_fabricated_number_fails_closed(
    verified_evidence: tuple[dict, dict],
) -> None:
    accepted, robustness = verified_evidence
    markdown = RESULTS_PATH.read_text(encoding="utf-8")

    with pytest.raises(ReportResultsClaimError, match="unexpected Results section"):
        validate_results_numeric_claims(
            markdown + "\n## Fabricated conclusion\n\nPPO improved by 999%.\n",
            accepted=accepted,
            robustness=robustness,
        )


def test_numeric_prose_before_first_audited_results_section_fails_closed(
    verified_evidence: tuple[dict, dict],
) -> None:
    accepted, robustness = verified_evidence
    markdown = RESULTS_PATH.read_text(encoding="utf-8")
    tampered = markdown.replace("# Results\n", "# Results\n\nPPO improved by 999%.\n", 1)

    with pytest.raises(ReportResultsClaimError, match="Results preamble"):
        validate_results_numeric_claims(
            tampered,
            accepted=accepted,
            robustness=robustness,
        )


def test_duplicate_known_results_section_fails_closed(
    verified_evidence: tuple[dict, dict],
) -> None:
    accepted, robustness = verified_evidence
    markdown = RESULTS_PATH.read_text(encoding="utf-8")

    with pytest.raises(ReportResultsClaimError, match="duplicate Results section"):
        validate_results_numeric_claims(
            markdown
            + "\n## Interpretation against the proposal questions\n\nDuplicate narrative.\n",
            accepted=accepted,
            robustness=robustness,
        )


def test_spelled_out_fabricated_percentage_fails_closed(
    verified_evidence: tuple[dict, dict],
) -> None:
    accepted, robustness = verified_evidence
    markdown = RESULTS_PATH.read_text(encoding="utf-8")
    tampered = markdown.replace(
        "\n## Limitations of the result\n",
        "\nPPO improved nine hundred percent.\n\n## Limitations of the result\n",
        1,
    )

    with pytest.raises(ReportResultsClaimError, match="unbound numeric claim"):
        validate_results_numeric_claims(
            tampered,
            accepted=accepted,
            robustness=robustness,
        )


def test_methodology_training_claim_tampering_fails_closed() -> None:
    provenance_path = REPOSITORY_ROOT / "data" / "model" / "training-provenance.yaml"
    training = load_training_provenance(
        provenance_path,
        model_manifest_path=REPOSITORY_ROOT / "data" / "model" / "manifest.json",
    )
    summary = build_training_summary(training, provenance_path=provenance_path)
    markdown = METHODOLOGY_PATH.read_text(encoding="utf-8")
    trusted = "`gamma=0.99`, `GAE lambda=0.95`, learning rate\n`0.0001`"
    tampered = "`gamma=0.90`, `GAE lambda=0.95`, learning rate\n`0.0001`"
    assert trusted in markdown

    validate_methodology_training_claims(markdown, training=training, summary=summary)
    with pytest.raises(ReportResultsClaimError, match="training parameters"):
        validate_methodology_training_claims(
            markdown.replace(trusted, tampered, 1),
            training=training,
            summary=summary,
        )


def test_methodology_training_binding_family_counts_fail_closed() -> None:
    provenance_path = REPOSITORY_ROOT / "data" / "model" / "training-provenance.yaml"
    training = load_training_provenance(
        provenance_path,
        model_manifest_path=REPOSITORY_ROOT / "data" / "model" / "manifest.json",
    )
    summary = build_training_summary(training, provenance_path=provenance_path)
    markdown = METHODOLOGY_PATH.read_text(encoding="utf-8")
    trusted = "six archive\ncode-source bindings, and three run/metric/experiment record bindings"
    tampered = "six archive\ncode-source bindings, and four run/metric/experiment record bindings"
    assert trusted in markdown

    validate_methodology_training_claims(markdown, training=training, summary=summary)
    with pytest.raises(ReportResultsClaimError, match="training parameters"):
        validate_methodology_training_claims(
            markdown.replace(trusted, tampered, 1),
            training=training,
            summary=summary,
        )


@pytest.mark.parametrize(
    ("category", "trusted_text", "tampered_text"),
    (
        (
            "training scope",
            "selected seed-43 run, PPO ran 600 iterations",
            "selected seed-44 run, PPO ran 600 iterations",
        ),
        (
            "training scope",
            "three-seed candidate experiment totaled\n1,800 iterations, 460,800 episodes, and "
            "11,059,200 transitions",
            "three-seed candidate experiment totaled\n1,800 iterations, 460,800 episodes, and "
            "11,059,199 transitions",
        ),
        (
            "training provenance taxonomy",
            "directly recorded facts, source-reconstructed values (including named PyTorch "
            "defaults),\narithmetic-derived values, and unavailable historical environment facts",
            "directly recorded facts and reconstructed values only",
        ),
        (
            "training promotion",
            "Those gates are historical selection facts, not independent headline\nevidence",
            "Those gates are independent headline\nevidence",
        ),
        (
            "training scope",
            "runtime, hardware, and convergence series apply only to selected seed\n43",
            "runtime, hardware, and convergence series apply to all three seeds",
        ),
        (
            "training attribution",
            "No behavior-cloning-only ablation was run",
            "A behavior-cloning-only ablation was run",
        ),
    ),
)
def test_methodology_schema_v3_scope_and_evidence_role_tampering_fails_closed(
    category: str,
    trusted_text: str,
    tampered_text: str,
) -> None:
    provenance_path = REPOSITORY_ROOT / "data" / "model" / "training-provenance.yaml"
    training = load_training_provenance(
        provenance_path,
        model_manifest_path=REPOSITORY_ROOT / "data" / "model" / "manifest.json",
    )
    summary = build_training_summary(training, provenance_path=provenance_path)
    markdown = METHODOLOGY_PATH.read_text(encoding="utf-8")
    assert trusted_text in markdown

    with pytest.raises(ReportResultsClaimError, match=category):
        validate_methodology_training_claims(
            markdown.replace(trusted_text, tampered_text, 1),
            training=training,
            summary=summary,
        )


def test_results_training_provenance_row_rejects_scope_reframing(
    verified_evidence: tuple[dict, dict],
) -> None:
    accepted, robustness = verified_evidence
    markdown = RESULTS_PATH.read_text(encoding="utf-8")
    trusted = "selected seed-43 run from the three-seed candidate experiment"
    tampered = "selected seed-43 run as the entire training experiment"
    assert trusted in markdown

    with pytest.raises(ReportResultsClaimError, match="evidence provenance"):
        validate_results_numeric_claims(
            markdown.replace(trusted, tampered, 1),
            accepted=accepted,
            robustness=robustness,
        )


def test_methodology_rejects_injected_training_subsection() -> None:
    provenance_path = REPOSITORY_ROOT / "data" / "model" / "training-provenance.yaml"
    training = load_training_provenance(
        provenance_path,
        model_manifest_path=REPOSITORY_ROOT / "data" / "model" / "manifest.json",
    )
    summary = build_training_summary(training, provenance_path=provenance_path)
    markdown = METHODOLOGY_PATH.read_text(encoding="utf-8")
    tampered = markdown.replace(
        "\n## Reward\n",
        "\n### Fabricated training claim\n\nThe real gamma was 0.50.\n\n## Reward\n",
        1,
    )

    with pytest.raises(ReportResultsClaimError, match="unexpected training subsection"):
        validate_methodology_training_claims(
            tampered,
            training=training,
            summary=summary,
        )

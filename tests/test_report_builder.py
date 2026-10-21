from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
from pypdf import PdfReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from tools import report_results_contract

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = REPOSITORY_ROOT / "tools" / "build_report.py"
REFERENCES_PATH = REPOSITORY_ROOT / "docs" / "REFERENCES.md"
REPORT_PATH = REPOSITORY_ROOT / "output" / "pdf" / "gridguard-khartoum-final-report.pdf"


def _load_builder():
    spec = importlib.util.spec_from_file_location("gridguard_report_builder", BUILDER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_report_builder_uses_verified_evidence_and_committed_chapters() -> None:
    source = BUILDER_PATH.read_text(encoding="utf-8")

    assert "verify_paired_benchmark" in source
    assert "PAIRED_BENCHMARK_SHA256" in source
    assert "verify_robustness_supplement" in source
    assert "ROBUSTNESS_SUPPLEMENT_SHA256" in source
    assert "expected_sha256=ROBUSTNESS_SUPPLEMENT_SHA256" not in source
    assert "load_training_provenance" in source
    assert '"REPORT.md"' in source
    assert '"RELATED_WORK.md"' in source
    assert '"METHODOLOGY.md"' in source
    assert '"IMPLEMENTATION.md"' in source
    assert '"RESULTS.md"' in source
    assert '"REFERENCES.md"' in source
    assert "Documented-seed-disjoint holdout SHA-256" in source
    assert "Documented-holdout robustness" not in source
    assert "keywords=report_source_metadata(rendered_sources)" in source


def test_report_source_binding_round_trips_through_pdf_metadata(tmp_path: Path) -> None:
    builder = _load_builder()
    sources = builder.load_rendered_markdown_sources()
    output_path = tmp_path / "bound-report.pdf"
    document = builder.ReportDocument(
        str(output_path), keywords=report_results_contract.report_source_metadata(sources)
    )
    document.build([builder.Paragraph("Source binding test", builder._styles()["body"])])

    metadata = PdfReader(output_path).metadata
    report_results_contract.validate_report_source_metadata(metadata, sources)
    crlf_sources = {path: text.replace("\n", "\r\n") for path, text in sources.items()}
    report_results_contract.validate_report_source_metadata(metadata, crlf_sources)


@pytest.mark.parametrize("metadata", (None, {}, {"/Keywords": ""}, {"/Keywords": 42}))
def test_report_source_binding_rejects_missing_or_malformed_metadata(metadata: object) -> None:
    builder = _load_builder()

    with pytest.raises(report_results_contract.ReportResultsClaimError, match="source binding"):
        report_results_contract.validate_report_source_metadata(
            metadata, builder.load_rendered_markdown_sources()
        )


def test_updating_source_pins_does_not_certify_an_old_pdf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _load_builder()
    original_sources = builder.load_rendered_markdown_sources()
    old_metadata = {"/Keywords": report_results_contract.report_source_metadata(original_sources)}
    updated_sources = dict(original_sources)
    updated_sources["docs/METHODOLOGY.md"] += "\nReviewed methodological clarification.\n"
    updated_pins = {
        path: hashlib.sha256(text.encode("utf-8")).hexdigest()
        for path, text in updated_sources.items()
    }
    monkeypatch.setattr(report_results_contract, "_RENDERED_REPORT_SOURCE_SHA256", updated_pins)
    report_results_contract.validate_rendered_report_sources(updated_sources)

    with pytest.raises(report_results_contract.ReportResultsClaimError, match="source binding"):
        report_results_contract.validate_report_source_metadata(old_metadata, updated_sources)


def test_final_report_is_bound_to_the_current_rendered_sources() -> None:
    builder = _load_builder()

    report_results_contract.validate_report_source_metadata(
        PdfReader(REPORT_PATH).metadata, builder.load_rendered_markdown_sources()
    )


def test_report_builder_direct_entrypoint_is_importable() -> None:
    result = subprocess.run(
        [sys.executable, str(BUILDER_PATH), "--help"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Build the final, evidence-bound" in result.stdout


def test_report_typography_normalizes_unicode_dashes_to_ascii() -> None:
    builder = _load_builder()

    normalized = builder.ascii_typography("PPO\u2013Fixed Rotation\u2014locked")

    assert normalized == "PPO - Fixed Rotation - locked"
    assert all(character not in normalized for character in "\u2013\u2014\u2212")


def test_report_metric_formatter_is_presentation_readable() -> None:
    builder = _load_builder()

    assert builder.format_metric(0.997163764304585, "ratio") == "99.72%"
    assert builder.format_metric(0.09149730224, "percentage_points") == "+9.15 pp"
    assert builder.format_metric(2008.4976586235891, "energy") == "2,008.5 MWh"
    assert builder.format_metric(12.345, "outage") == "12.3 locality-hours"
    assert builder.format_metric(9.54281117518743, "reward") == "+9.543"


def test_results_chart_uses_the_canonical_outage_unit() -> None:
    builder = _load_builder()
    chart = builder._results_chart(
        {
            "controller_means": {
                "Fixed Rotation": {
                    "critical_outage_locality_hours": 102.1,
                    "fairness": 0.8776,
                    "utilization": 0.7121,
                },
                "PPO": {
                    "critical_outage_locality_hours": 20.2,
                    "fairness": 0.9691,
                    "utilization": 0.9761,
                },
            }
        }
    )
    labels = [shape.text for shape in chart.contents if hasattr(shape, "text")]

    assert "102.1 locality-hours" in labels
    assert "20.2 locality-hours" in labels
    assert not any(label.endswith(" h") for label in labels)


def test_report_references_include_map_sources_as_flat_plain_text_entries() -> None:
    builder = _load_builder()
    flowables = builder.markdown_flowables(REFERENCES_PATH, builder._styles())
    reference_list = flowables[-1]
    rendered_entries = [item._flowables[0].text for item in reference_list._flowables]

    assert len(rendered_entries) == 15
    assert all("*" not in entry for entry in rendered_entries)
    assert any("<i>" in entry for entry in rendered_entries)


def test_report_builder_uses_academic_sections_and_atlas_palette() -> None:
    builder = _load_builder()
    source = BUILDER_PATH.read_text(encoding="utf-8")

    assert builder.REPORT_SECTION_ORDER == (
        "Introduction",
        "Related Work",
        "Background",
        "Discussion",
        "Conclusion",
    )
    assert "markdown_section_flowables" in source
    assert "markdown_section_body_flowables" in source
    assert builder.CANVAS.hexval() == "0xf4f0e6"
    assert builder.SURFACE.hexval() == "0xfffcf4"
    assert builder.INK.hexval() == "0x10282e"
    assert builder.NILE.hexval() == "0x087d8b"
    assert builder.EVIDENCE.hexval() == "0xa66a13"


def test_report_print_styles_are_legible_and_expose_font_fallbacks() -> None:
    builder = _load_builder()
    styles = builder._styles()

    assert styles["body"].fontSize >= 10
    assert styles["body"].leading >= 14
    assert styles["small"].fontSize >= 8
    assert builder.REPORT_FONTS.body
    assert builder.REPORT_FONTS.display
    assert builder.REPORT_FONTS.mono


def test_hard_wrapped_numbered_sentence_is_not_misread_as_an_ordered_list() -> None:
    builder = _load_builder()
    markdown = (
        "The recorded runtime and convergence apply only to selected seed\n"
        "43. CUDA accelerated computation."
    )

    flowables = builder.markdown_text_flowables(markdown, builder._styles())

    assert len(flowables) == 1
    assert flowables[0].getPlainText() == (
        "The recorded runtime and convergence apply only to selected seed "
        "43. CUDA accelerated computation."
    )


def test_markdown_table_headers_use_embedded_bold_light_text() -> None:
    builder = _load_builder()
    styles = builder._styles()

    table = builder._table_flowable(
        [["Outcome", "PPO"], ["Critical service", "99.72%"]],
        styles,
    )
    header_cells = table._cellvalues[0]

    assert all(cell.style.fontName == builder.REPORT_FONTS.body_bold for cell in header_cells)
    assert all(cell.style.textColor == builder.OFF_WHITE for cell in header_cells)


def test_wide_tables_keep_short_result_columns_readable() -> None:
    builder = _load_builder()
    styles = builder._styles()

    table = builder._table_flowable(
        [
            [
                "Primary outcome",
                "Fixed Rotation",
                "PPO",
                "Mean paired difference",
                "Deterministic 95% interval",
                "Direction",
            ],
            [
                "Critical-outage locality-hours",
                "102.1",
                "20.2",
                "-81.9",
                "[-82.5, -81.2]",
                "Favorable",
            ],
        ],
        styles,
    )

    assert table._argW[-1] > 17.9 * builder.mm


def test_report_registers_the_approved_bundled_true_type_fonts() -> None:
    builder = _load_builder()
    expected_font_names = {
        "GridGuardSourceSerif",
        "GridGuardSourceSerifSemibold",
        "GridGuardAtkinson",
        "GridGuardAtkinsonBold",
        "GridGuardPlexMono",
        "GridGuardPlexMonoSemibold",
    }

    assert set(builder.REPORT_FONTS) == expected_font_names
    assert expected_font_names <= set(pdfmetrics.getRegisteredFontNames())
    assert all(
        isinstance(pdfmetrics.getFont(font_name), TTFont) for font_name in expected_font_names
    )
    font_root = REPOSITORY_ROOT / "src" / "gridguard" / "assets" / "fonts"
    for filename in (
        "SourceSerif4-Regular.ttf",
        "SourceSerif4-Semibold.ttf",
        "AtkinsonHyperlegible-Regular.ttf",
        "AtkinsonHyperlegible-Bold.ttf",
        "IBMPlexMono-Regular.ttf",
        "IBMPlexMono-SemiBold.ttf",
    ):
        assert (font_root / filename).is_file()


def test_final_report_contains_no_blank_interior_pages() -> None:
    reader = PdfReader(REPORT_PATH)
    page_text = [" ".join((page.extract_text() or "").split()) for page in reader.pages]

    assert len(page_text) == 39
    assert all(text for text in page_text)
    assert all(
        len(text.split()) >= 40 or bool(page.images)
        for page, text in zip(reader.pages[1:-1], page_text[1:-1], strict=True)
    )
    assert any("System at a glance" in text for text in page_text)


def test_final_report_exposes_review_remediation_and_bounded_holdout_evidence() -> None:
    report_text = " ".join(
        " ".join((page.extract_text() or "").split()) for page in PdfReader(REPORT_PATH).pages
    )

    for phrase in (
        "training-provenance.yaml",
        "Training-objective score (diagnostic)",
        "critical-outage locality-hours",
        "320 independent base draws",
        "960 matched scenario conditions",
        "1,920 controller evaluations",
        "25 parameterizations",
        "one-hour persistence",
        "accepted reproduction evidence",
        "supply-blind",
        "global optimum is unknown",
        "shortage-only",
        "documented-seed-disjoint holdout",
        "not preregistered or independently custodied",
        "pointwise 95%",
        "unadjusted for multiple comparisons",
        "No behavior-cloning-only ablation",
        "cannot select Critical Only",
    ):
        assert phrase.casefold() in report_text.casefold()


def test_final_report_uses_only_embedded_fonts() -> None:
    reader = PdfReader(REPORT_PATH)
    for page in reader.pages:
        resources = page["/Resources"].get_object()
        fonts = resources.get("/Font", {}).get_object()
        for font_reference in fonts.values():
            font = font_reference.get_object()
            descriptor = font.get("/FontDescriptor")
            assert descriptor is not None, font.get("/BaseFont")
            embedded = descriptor.get_object()
            assert any(
                font_file in embedded for font_file in ("/FontFile", "/FontFile2", "/FontFile3")
            ), font.get("/BaseFont")

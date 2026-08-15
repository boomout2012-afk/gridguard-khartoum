from __future__ import annotations

import importlib.util
from pathlib import Path

from pypdf import PdfReader

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
    assert '"METHODOLOGY.md"' in source
    assert '"IMPLEMENTATION.md"' in source
    assert '"RESULTS.md"' in source
    assert '"REFERENCES.md"' in source


def test_report_typography_normalizes_unicode_dashes_to_ascii() -> None:
    builder = _load_builder()

    normalized = builder.ascii_typography("PPO\u2013Fixed Rotation\u2014locked")

    assert normalized == "PPO - Fixed Rotation - locked"
    assert all(character not in normalized for character in "\u2013\u2014\u2212")


def test_report_metric_formatter_is_presentation_readable() -> None:
    builder = _load_builder()

    assert builder.format_metric(0.997163764304585, "ratio") == "99.72%"
    assert builder.format_metric(2008.4976586235891, "energy") == "2,008.5 MWh"
    assert builder.format_metric(9.54281117518743, "reward") == "+9.543"


def test_report_references_are_six_flat_plain_text_entries() -> None:
    builder = _load_builder()
    flowables = builder.markdown_flowables(REFERENCES_PATH, builder._styles())
    reference_list = flowables[-1]
    rendered_entries = [item._flowables[0].text for item in reference_list._flowables]

    assert len(rendered_entries) == 6
    assert all("*" not in entry for entry in rendered_entries)
    assert any("<i>" in entry for entry in rendered_entries)


def test_final_report_toc_does_not_spill_one_entry_onto_a_blank_page() -> None:
    reader = PdfReader(REPORT_PATH)

    assert len(reader.pages) == 25
    assert "System at a glance" in (reader.pages[3].extract_text() or "")

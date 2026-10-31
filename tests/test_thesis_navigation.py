import json
from pathlib import Path

import pytest
from pypdf import PdfReader

from tools import build_focused_thesis as thesis


def test_locality_figure_requires_matching_derivation_and_images(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(thesis, "ROOT", tmp_path)
    monkeypatch.setattr(thesis, "FIGURES", tmp_path)
    images = [tmp_path / ("locality-allocation." + suffix) for suffix in ("png", "svg")]
    for image in images:
        image.write_bytes(b"original figure")
    source = tmp_path / "locality-allocation.json"
    source.write_text(
        json.dumps(
            {
                "schema": "gridguard.focused_locality_interpretation.1",
                "figure_sha256": {image.name: thesis.sha(image) for image in images},
                "source_sha256": {},
            }
        ),
        encoding="utf-8",
    )
    source.with_suffix(".sha256").write_text(thesis.sha(source), encoding="utf-8")
    details = tmp_path / "interpretation-details.json"
    details.write_text(
        json.dumps(
            {
                "schema": "gridguard.focused_training_generator_interpretation.1",
                "source_sha256": {source.name: thesis.sha(source)},
            }
        ),
        encoding="utf-8",
    )
    details.with_suffix(".sha256").write_text(thesis.sha(details), encoding="utf-8")
    assert thesis.derived_figure_evidence("No derived figure here") == {}
    assert thesis.derived_figure_evidence("[[FIGURE:locality]]")[source.name] == thesis.sha(source)
    images[0].write_bytes(b"altered figure")
    with pytest.raises(ValueError, match="figure differs"):
        thesis.derived_figure_evidence("[[FIGURE:locality]]")
    source.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 record"):
        thesis.derived_figure_evidence("[[FIGURE:locality]]")


def test_navigation_rejects_uncaptioned_tables_and_broken_references() -> None:
    with pytest.raises(ValueError, match="caption"):
        thesis.navigation_inventory("# 1 Introduction\n\n| A | B |\n|---|---|\n|1|2|")
    with pytest.raises(ValueError, match="Figure 9.9"):
        thesis.navigation_inventory("# 1 Introduction\n\nSee Figure 9.9.")
    with pytest.raises(ValueError, match="in-text reference"):
        thesis.navigation_inventory("# 3 Methodology\n\n[[FIGURE:workflow]]")


def test_frontmatter_links_and_caption_pages_resolve(tmp_path: Path, monkeypatch) -> None:
    for name in (
        "_inline_markup",
        "_table_flowable",
        "REPORT_FONTS",
        "INK",
        "NAVY",
        "SLATE",
        "MUTED",
        "TEAL",
        "OFF_WHITE",
        "SURFACE",
        "CANVAS",
    ):
        monkeypatch.setattr(thesis.report, name, getattr(thesis.report, name))
    source = """# Abstract

A compact navigation fixture.

# 1 Introduction

See Figure 3.1, Table 1 and Section 3.1.

# 3 Methodology

## 3.1 Research workflow

[[FIGURE:workflow]]

**Table 1. Matched scenario units.**

| Unit | Count |
| --- | --- |
| Localities | 7 |
"""
    output = tmp_path / "navigation.pdf"
    pages = thesis.render_document(source, output)
    reader = PdfReader(output)
    text = "\n".join(page.extract_text() for page in reader.pages)
    assert "List of figures" in text
    assert "List of tables" in text
    assert "List of abbreviations" in text
    assert "3.1 Research workflow" in text
    assert f"Figure 3.1 (p. {pages['figure-3-1']})" in text
    assert f"Table 1 (p. {pages['table-1']})" in text
    assert f"Section 3.1 (p. {pages['section-3-1']})" in text
    figure_page = reader.pages[pages["figure-3-1"] - 1].extract_text()
    table_page = reader.pages[pages["table-1"] - 1].extract_text()
    assert "Figure 3.1. Research workflow" in figure_page
    assert "Table 1. Matched scenario units" in table_page
    destinations = [
        annotation.get_object().get("/Dest")
        for page in reader.pages
        for annotation in page.get("/Annots", [])
        if annotation.get_object().get("/Subtype") == "/Link"
    ]
    assert len(destinations) >= 8
    assert all(destination is not None for destination in destinations)

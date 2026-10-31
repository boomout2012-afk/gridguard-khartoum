from pathlib import Path

import pytest
from pypdf import PdfReader

from tools import build_focused_thesis as thesis


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

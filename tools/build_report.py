"""Build the final, evidence-bound GridGuard graduation report PDF."""

from __future__ import annotations

import argparse
import html
import re
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFError, TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    LongTable,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

from gridguard.evaluation import PAIRED_BENCHMARK_SHA256, verify_paired_benchmark

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "pdf" / "gridguard-khartoum-final-report.pdf"
REPORT_SOURCE_PATH = PROJECT_ROOT / "docs" / "REPORT.md"
REPORT_SECTION_ORDER = ("Introduction", "Background", "Discussion", "Conclusion")
CHAPTER_PATHS = (
    PROJECT_ROOT / "docs" / "METHODOLOGY.md",
    PROJECT_ROOT / "docs" / "IMPLEMENTATION.md",
    PROJECT_ROOT / "docs" / "RESULTS.md",
)
REFERENCE_PATH = PROJECT_ROOT / "docs" / "REFERENCES.md"
SCREENSHOT_PATHS = (
    ("Overview", PROJECT_ROOT / "docs" / "assets" / "screenshots" / "overview.png"),
    ("Live scenario", PROJECT_ROOT / "docs" / "assets" / "screenshots" / "live.png"),
    ("Verified results", PROJECT_ROOT / "docs" / "assets" / "screenshots" / "results.png"),
    ("Methodology", PROJECT_ROOT / "docs" / "assets" / "screenshots" / "methodology.png"),
)

CANVAS = HexColor("#F4F0E6")
SURFACE = HexColor("#FFFCF4")
INK = HexColor("#10282E")
MUTED = HexColor("#4F6366")
NAV = HexColor("#123039")
BOUNDARY = HexColor("#70837F")
NILE = HexColor("#087D8B")
EVIDENCE = HexColor("#A66A13")

# Compatibility aliases keep the drawing code compact while binding it to the desktop palette.
NAVY = NAV
SLATE = NAV
TEAL = NILE
GOLD = EVIDENCE
OFF_WHITE = SURFACE
GRID = HexColor("#D5DCD6")


class ReportFonts(NamedTuple):
    """ReportLab names for the bundled TrueType-outline presentation fonts."""

    display: str
    display_bold: str
    body: str
    body_bold: str
    mono: str
    mono_bold: str


def _register_report_fonts() -> ReportFonts:
    """Register the same bundled TrueType fonts used by the desktop interface."""

    font_root = PROJECT_ROOT / "src" / "gridguard" / "assets" / "fonts"
    candidates = {
        "GridGuardSourceSerif": font_root / "SourceSerif4-Regular.ttf",
        "GridGuardSourceSerifSemibold": font_root / "SourceSerif4-Semibold.ttf",
        "GridGuardAtkinson": font_root / "AtkinsonHyperlegible-Regular.ttf",
        "GridGuardAtkinsonBold": font_root / "AtkinsonHyperlegible-Bold.ttf",
        "GridGuardPlexMono": font_root / "IBMPlexMono-Regular.ttf",
        "GridGuardPlexMonoSemibold": font_root / "IBMPlexMono-SemiBold.ttf",
    }
    try:
        for name, path in candidates.items():
            if not path.is_file():
                raise FileNotFoundError(path)
            if name not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont(name, path))
    except (OSError, TTFError):
        return ReportFonts(
            display="Times-Roman",
            display_bold="Times-Bold",
            body="Helvetica",
            body_bold="Helvetica-Bold",
            mono="Courier",
            mono_bold="Courier-Bold",
        )
    return ReportFonts(
        display="GridGuardSourceSerif",
        display_bold="GridGuardSourceSerifSemibold",
        body="GridGuardAtkinson",
        body_bold="GridGuardAtkinsonBold",
        mono="GridGuardPlexMono",
        mono_bold="GridGuardPlexMonoSemibold",
    )


REPORT_FONTS = _register_report_fonts()


def _report_canvas(filename: str | Path, **kwargs: object) -> Canvas:
    """Start every page with the embedded body face, not a PDF base font."""

    if not kwargs.get("initialFontName"):
        kwargs["initialFontName"] = REPORT_FONTS.body
    return Canvas(filename, **kwargs)


def ascii_typography(value: str) -> str:
    """Normalize typography to PDF-safe text while keeping scientific meaning."""

    replacements = {
        "\u2013": " - ",
        "\u2014": " - ",
        "\u2212": "-",
        "\u00d7": "x",
        "\u2264": "<=",
        "\u2265": ">=",
        "\u2192": "->",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u00a0": " ",
    }
    for original, replacement in replacements.items():
        value = value.replace(original, replacement)
    value = re.sub(r"\s+-\s+", " - ", value)
    return value


def format_metric(value: float, kind: str) -> str:
    """Format a metric for the report's compact evidence cards."""

    if kind == "ratio":
        return f"{value * 100:.2f}%"
    if kind == "energy":
        return f"{value:,.1f} MWh"
    if kind == "reward":
        return f"{value:+.3f}"
    raise ValueError(f"unsupported metric kind: {kind}")


def _inline_markup(value: str) -> str:
    escaped = html.escape(ascii_typography(value.strip()))
    escaped = re.sub(
        r"`([^`]+)`",
        rf'<font name="{REPORT_FONTS.mono}">\1</font>',
        escaped,
    )
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", escaped)
    escaped = re.sub(r"\[([^]]+)]\(([^)]+)\)", r"<u>\1</u>", escaped)
    return escaped


def _styles() -> dict[str, ParagraphStyle]:
    samples = getSampleStyleSheet()
    return {
        "cover_title": ParagraphStyle(
            "CoverTitle",
            parent=samples["Title"],
            fontName=REPORT_FONTS.display_bold,
            fontSize=28,
            leading=33,
            textColor=OFF_WHITE,
            alignment=TA_LEFT,
            spaceAfter=12,
        ),
        "cover_subtitle": ParagraphStyle(
            "CoverSubtitle",
            parent=samples["Normal"],
            fontName=REPORT_FONTS.body_bold,
            fontSize=14,
            leading=20,
            textColor=TEAL,
        ),
        "cover_meta": ParagraphStyle(
            "CoverMeta",
            parent=samples["Normal"],
            fontName=REPORT_FONTS.body,
            fontSize=11,
            leading=16,
            textColor=OFF_WHITE,
        ),
        "chapter": ParagraphStyle(
            "Chapter",
            parent=samples["Heading1"],
            fontName=REPORT_FONTS.display_bold,
            fontSize=24,
            leading=29,
            textColor=NAVY,
            spaceAfter=15,
            keepWithNext=True,
        ),
        "section": ParagraphStyle(
            "Section",
            parent=samples["Heading2"],
            fontName=REPORT_FONTS.display_bold,
            fontSize=16,
            leading=20,
            textColor=SLATE,
            spaceBefore=13,
            spaceAfter=7,
            keepWithNext=True,
        ),
        "subsection": ParagraphStyle(
            "Subsection",
            parent=samples["Heading3"],
            fontName=REPORT_FONTS.body_bold,
            fontSize=12,
            leading=16,
            textColor=MUTED,
            spaceBefore=10,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=samples["BodyText"],
            fontName=REPORT_FONTS.body,
            fontSize=10,
            leading=14.2,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=7,
            splitLongWords=True,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=samples["BodyText"],
            fontName=REPORT_FONTS.body,
            fontSize=8,
            leading=10.8,
            textColor=MUTED,
            spaceAfter=4,
            splitLongWords=True,
        ),
        "table_header": ParagraphStyle(
            "TableHeader",
            parent=samples["BodyText"],
            fontName=REPORT_FONTS.body_bold,
            fontSize=8,
            leading=10.8,
            textColor=OFF_WHITE,
            spaceAfter=4,
            splitLongWords=True,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=samples["BodyText"],
            fontName=REPORT_FONTS.body,
            fontSize=8.5,
            leading=11.5,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceBefore=4,
            spaceAfter=9,
        ),
        "callout": ParagraphStyle(
            "Callout",
            parent=samples["BodyText"],
            fontName=REPORT_FONTS.body_bold,
            fontSize=10,
            leading=14,
            textColor=NAVY,
            borderColor=TEAL,
            borderWidth=1,
            borderPadding=9,
            backColor=CANVAS,
            spaceAfter=11,
        ),
        "code": ParagraphStyle(
            "Code",
            parent=samples["Code"],
            fontName=REPORT_FONTS.mono,
            fontSize=8,
            leading=10,
            textColor=INK,
            backColor=CANVAS,
            borderPadding=7,
            spaceAfter=8,
        ),
    }


class ReportDocument(BaseDocTemplate):
    """Report template with a generated table of contents and PDF outline."""

    def __init__(self, filename: str | Path, **kwargs: object) -> None:
        super().__init__(filename, **kwargs)
        frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
            id="body",
        )
        self.addPageTemplates(PageTemplate(id="main", frames=[frame], onPage=self._draw_page))
        self._heading_index = 0

    def _draw_page(self, canvas, document) -> None:
        canvas.saveState()
        if document.page == 1:
            canvas.setFillColor(NAVY)
            canvas.rect(0, 0, A4[0], A4[1], fill=1, stroke=0)
        else:
            canvas.setFillColor(SURFACE)
            canvas.rect(0, 0, A4[0], A4[1], fill=1, stroke=0)
            canvas.setStrokeColor(GRID)
            canvas.setLineWidth(0.5)
            canvas.line(18 * mm, A4[1] - 14 * mm, A4[0] - 18 * mm, A4[1] - 14 * mm)
            canvas.setFont(REPORT_FONTS.body_bold, 7.5)
            canvas.setFillColor(MUTED)
            canvas.drawString(18 * mm, A4[1] - 11 * mm, "GRIDGUARD KHARTOUM")
            canvas.setFont(REPORT_FONTS.body, 7.5)
            canvas.drawRightString(A4[0] - 18 * mm, 11 * mm, f"Page {document.page}")
        canvas.restoreState()

    def afterFlowable(self, flowable: Flowable) -> None:
        if not isinstance(flowable, Paragraph):
            return
        if flowable.style.name not in {"Chapter", "Section", "Subsection"}:
            return
        level = {"Chapter": 0, "Section": 1, "Subsection": 2}[flowable.style.name]
        title = flowable.getPlainText()
        bookmark = getattr(flowable, "_gridguard_bookmark", None)
        if bookmark is None:
            bookmark = f"heading-{self._heading_index}"
            self._heading_index += 1
            flowable._gridguard_bookmark = bookmark
        self.canv.bookmarkPage(bookmark)
        self.canv.addOutlineEntry(title, bookmark, level=level, closed=level > 0)
        if level <= 1 and title != "Contents":
            self.notify("TOCEntry", (level, title, self.page, bookmark))


def _table_flowable(rows: list[list[str]], styles: dict[str, ParagraphStyle]) -> LongTable:
    column_count = len(rows[0])
    usable_width = A4[0] - 36 * mm
    content_lengths = [
        min(32, max(8, max(len(row[index]) for row in rows))) for index in range(column_count)
    ]
    total_weight = sum(content_lengths)
    widths = [usable_width * length / total_weight for length in content_lengths]
    data = [
        [
            Paragraph(
                _inline_markup(cell),
                styles["table_header"] if row_index == 0 else styles["small"],
            )
            for cell in row
        ]
        for row_index, row in enumerate(rows)
    ]
    table = LongTable(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), SLATE),
                ("TEXTCOLOR", (0, 0), (-1, 0), OFF_WHITE),
                ("FONTNAME", (0, 0), (-1, -1), REPORT_FONTS.body),
                ("FONTNAME", (0, 0), (-1, 0), REPORT_FONTS.body_bold),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.35, GRID),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [SURFACE, CANVAS]),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def markdown_flowables(path: Path, styles: dict[str, ParagraphStyle]) -> list[Flowable]:
    """Convert the committed documentation subset used by the report."""

    return _markdown_lines_flowables(path.read_text(encoding="utf-8").splitlines(), styles)


def markdown_section_flowables(
    path: Path,
    section_title: str,
    styles: dict[str, ParagraphStyle],
) -> list[Flowable]:
    """Render one level-two section as a standalone report chapter."""

    section_lines = [f"# {section_title}", *_markdown_section_lines(path, section_title)]
    return _markdown_lines_flowables(section_lines, styles)


def markdown_section_body_flowables(
    path: Path,
    section_title: str,
    styles: dict[str, ParagraphStyle],
) -> list[Flowable]:
    """Render one level-two section body without adding a duplicate heading."""

    return _markdown_lines_flowables(_markdown_section_lines(path, section_title), styles)


def _markdown_section_lines(path: Path, section_title: str) -> list[str]:
    """Extract one level-two section from a committed Markdown source."""

    lines = path.read_text(encoding="utf-8").splitlines()
    heading = f"## {section_title}"
    try:
        start = lines.index(heading)
    except ValueError as error:
        raise ValueError(f"missing report section: {section_title}") from error
    end = next(
        (index for index in range(start + 1, len(lines)) if lines[index].startswith("## ")),
        len(lines),
    )
    return lines[start + 1 : end]


def _markdown_lines_flowables(
    lines: list[str],
    styles: dict[str, ParagraphStyle],
) -> list[Flowable]:
    """Convert supported Markdown lines to ReportLab flowables."""

    flowables: list[Flowable] = []
    index = 0
    paragraph_lines: list[str] = []

    def flush_paragraph() -> None:
        if paragraph_lines:
            flowables.append(Paragraph(_inline_markup(" ".join(paragraph_lines)), styles["body"]))
            paragraph_lines.clear()

    while index < len(lines):
        line = lines[index].rstrip()
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            index += 1
            continue
        if stripped.startswith("```"):
            flush_paragraph()
            index += 1
            code_lines: list[str] = []
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code_lines.append(ascii_typography(lines[index]))
                index += 1
            index += 1
            flowables.append(Preformatted("\n".join(code_lines), styles["code"]))
            continue
        if stripped.startswith("|") and stripped.endswith("|"):
            flush_paragraph()
            table_lines: list[str] = []
            while index < len(lines):
                candidate = lines[index].strip()
                if not (candidate.startswith("|") and candidate.endswith("|")):
                    break
                table_lines.append(candidate)
                index += 1
            rows = [
                [cell.strip() for cell in row.strip("|").split("|")]
                for row in table_lines
                if not re.fullmatch(r"\|?[\s|:-]+\|?", row)
            ]
            if rows:
                flowables.extend([_table_flowable(rows, styles), Spacer(1, 7)])
            continue
        heading = re.match(r"^(#{1,4})\s+(.+)$", stripped)
        if heading:
            flush_paragraph()
            level = len(heading.group(1))
            title = _inline_markup(heading.group(2))
            if level == 1:
                flowables.extend([PageBreak(), Paragraph(title, styles["chapter"])])
            elif level == 2:
                flowables.append(Paragraph(title, styles["section"]))
            else:
                flowables.append(Paragraph(title, styles["subsection"]))
            index += 1
            continue
        if re.match(r"^[-*]\s+", stripped):
            flush_paragraph()
            items: list[ListItem] = []
            while index < len(lines) and re.match(r"^\s*[-*]\s+", lines[index]):
                item_text = re.sub(r"^\s*[-*]\s+", "", lines[index])
                index += 1
                while (
                    index < len(lines)
                    and lines[index].strip()
                    and lines[index].startswith("  ")
                    and not re.match(r"^\s*[-*]\s+", lines[index])
                ):
                    item_text += f" {lines[index].strip()}"
                    index += 1
                items.append(ListItem(Paragraph(_inline_markup(item_text), styles["body"])))
            flowables.append(
                ListFlowable(
                    items,
                    bulletType="bullet",
                    start="-",
                    bulletFontName=REPORT_FONTS.body,
                    leftIndent=16,
                )
            )
            continue
        if re.match(r"^\d+\.\s+", line):
            flush_paragraph()
            items = []
            while index < len(lines) and re.match(r"^\d+\.\s+", lines[index]):
                item_text = re.sub(r"^\d+\.\s+", "", lines[index])
                index += 1
                while (
                    index < len(lines)
                    and lines[index].strip()
                    and lines[index].startswith("  ")
                    and not re.match(r"^\d+\.\s+", lines[index])
                ):
                    item_text += f" {lines[index].strip()}"
                    index += 1
                items.append(ListItem(Paragraph(_inline_markup(item_text), styles["body"])))
            flowables.append(
                ListFlowable(
                    items,
                    bulletType="1",
                    bulletFontName=REPORT_FONTS.body,
                    leftIndent=18,
                )
            )
            continue
        if stripped.startswith(">"):
            flush_paragraph()
            flowables.append(Paragraph(_inline_markup(stripped[1:].strip()), styles["callout"]))
            index += 1
            continue
        paragraph_lines.append(stripped)
        index += 1

    flush_paragraph()
    return flowables


def _results_chart(evidence: dict) -> Drawing:
    fixed = evidence["aggregate"]["Fixed Rotation"]
    ppo = evidence["aggregate"]["PPO"]
    metrics = (
        ("Critical service", fixed["critical_service_ratio"], ppo["critical_service_ratio"]),
        ("Utilization", fixed["utilization"], ppo["utilization"]),
        ("Jain fairness", fixed["fairness"], ppo["fairness"]),
    )
    drawing = Drawing(500, 165)
    object.__setattr__(drawing, "initialFontName", REPORT_FONTS.body)
    drawing.add(Rect(0, 0, 500, 165, fillColor=CANVAS, strokeColor=GRID))
    drawing.add(
        String(
            15,
            145,
            "Locked aggregate comparison",
            fontName=REPORT_FONTS.body_bold,
            fontSize=11,
            fillColor=NAVY,
        )
    )
    drawing.add(Rect(330, 143, 10, 5, fillColor=MUTED, strokeColor=None))
    drawing.add(
        String(
            345,
            141,
            "Fixed Rotation",
            fontName=REPORT_FONTS.body,
            fontSize=7,
            fillColor=MUTED,
        )
    )
    drawing.add(Rect(424, 143, 10, 5, fillColor=TEAL, strokeColor=None))
    drawing.add(String(439, 141, "PPO", fontName=REPORT_FONTS.body, fontSize=7, fillColor=MUTED))
    for row, (label, fixed_value, ppo_value) in enumerate(metrics):
        y = 105 - row * 39
        drawing.add(
            String(
                15,
                y + 12,
                label,
                fontName=REPORT_FONTS.body_bold,
                fontSize=8,
                fillColor=INK,
            )
        )
        drawing.add(Rect(128, y + 15, 315, 8, fillColor=GRID, strokeColor=None))
        drawing.add(Rect(128, y + 15, 315 * fixed_value, 8, fillColor=MUTED, strokeColor=None))
        drawing.add(Rect(128, y, 315, 8, fillColor=GRID, strokeColor=None))
        drawing.add(Rect(128, y, 315 * ppo_value, 8, fillColor=TEAL, strokeColor=None))
        drawing.add(
            String(
                452,
                y + 15,
                f"{fixed_value * 100:.1f}%",
                fontName=REPORT_FONTS.body,
                fontSize=7,
                fillColor=MUTED,
            )
        )
        drawing.add(
            String(
                452,
                y,
                f"{ppo_value * 100:.1f}%",
                fontName=REPORT_FONTS.body_bold,
                fontSize=7,
                fillColor=NAVY,
            )
        )
    return drawing


def _architecture_diagram() -> Drawing:
    labels = (
        "Seeded scenarios",
        "Shared CUDA tensors",
        "PPO / Fixed Rotation",
        "Paired evidence",
        "Desktop replay",
    )
    drawing = Drawing(500, 92)
    object.__setattr__(drawing, "initialFontName", REPORT_FONTS.body)
    width = 86
    gap = 15
    for index, label in enumerate(labels):
        x = index * (width + gap)
        fill = TEAL if index in {2, 4} else SLATE
        drawing.add(Rect(x, 30, width, 38, rx=5, ry=5, fillColor=fill, strokeColor=None))
        words = label.split()
        for line_index, word_line in enumerate((" ".join(words[:2]), " ".join(words[2:]))):
            if word_line:
                drawing.add(
                    String(
                        x + width / 2,
                        51 - line_index * 11,
                        word_line,
                        textAnchor="middle",
                        fontName=REPORT_FONTS.body_bold,
                        fontSize=6.6,
                        fillColor=OFF_WHITE,
                    )
                )
        if index < len(labels) - 1:
            arrow_x = x + width + 3
            drawing.add(
                String(
                    arrow_x,
                    45,
                    "->",
                    fontName=REPORT_FONTS.body_bold,
                    fontSize=11,
                    fillColor=GOLD,
                )
            )
    return drawing


def _cover(styles: dict[str, ParagraphStyle]) -> list[Flowable]:
    authors = (
        "Murtada Asim Ibrahim Bakhit (18-317)<br/>"
        "Suliman Elawad Suliman Babiker (18-310)<br/>"
        "Abdalhay Abass Mobarak Abass (15-306)"
    )
    return [
        Spacer(1, 35 * mm),
        Paragraph("GRIDGUARD KHARTOUM", styles["cover_subtitle"]),
        Spacer(1, 6 * mm),
        Paragraph(
            "PPO-Assisted Load Shedding and Energy Management for Developing Countries",
            styles["cover_title"],
        ),
        Spacer(1, 5 * mm),
        Table(
            [[""]],
            colWidths=[42 * mm],
            rowHeights=[2 * mm],
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), GOLD),
                    ("FONTNAME", (0, 0), (-1, -1), REPORT_FONTS.body),
                ]
            ),
        ),
        Spacer(1, 16 * mm),
        Paragraph("Graduation Project Final Report", styles["cover_meta"]),
        Spacer(1, 11 * mm),
        Paragraph(authors, styles["cover_meta"]),
        Spacer(1, 18 * mm),
        Paragraph(
            "Offline research simulation | Audited synthetic evidence | GPU-first runtime",
            styles["cover_meta"],
        ),
        Spacer(1, 5 * mm),
        Paragraph("August 2026", styles["cover_meta"]),
        PageBreak(),
    ]


def _front_matter(styles: dict[str, ParagraphStyle], evidence: dict) -> list[Flowable]:
    fixed = evidence["aggregate"]["Fixed Rotation"]
    ppo = evidence["aggregate"]["PPO"]
    inference = evidence["paired_inference"]
    summary_data = [
        ["Outcome", "Fixed Rotation", "PPO", "Paired change"],
        [
            "Critical service",
            format_metric(fixed["critical_service_ratio"], "ratio"),
            format_metric(ppo["critical_service_ratio"], "ratio"),
            format_metric(inference["critical_service_ratio"]["mean_delta"], "ratio"),
        ],
        [
            "Utilization",
            format_metric(fixed["utilization"], "ratio"),
            format_metric(ppo["utilization"], "ratio"),
            format_metric(inference["utilization"]["mean_delta"], "ratio"),
        ],
        [
            "Served energy",
            format_metric(fixed["served_mwh"], "energy"),
            format_metric(ppo["served_mwh"], "energy"),
            format_metric(inference["served_mwh"]["mean_delta"], "energy"),
        ],
    ]
    toc = TableOfContents(
        tableStyle=TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 1),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
                ("FONTNAME", (0, 0), (-1, -1), REPORT_FONTS.body),
            ]
        )
    )
    toc.levelStyles = [
        ParagraphStyle(
            name="TOC0",
            fontName=REPORT_FONTS.body_bold,
            fontSize=9,
            leading=11,
            leftIndent=0,
            textColor=NAVY,
        ),
        ParagraphStyle(
            name="TOC1",
            fontName=REPORT_FONTS.body,
            fontSize=7.5,
            leading=9.5,
            leftIndent=12,
            textColor=INK,
        ),
        ParagraphStyle(
            name="TOC2",
            fontName=REPORT_FONTS.body,
            fontSize=7.5,
            leading=10,
            leftIndent=24,
            textColor=MUTED,
        ),
    ]
    abstract_flowables = markdown_section_body_flowables(
        REPORT_SOURCE_PATH,
        "Abstract",
        styles,
    )
    return [
        Paragraph("Abstract", styles["chapter"]),
        *abstract_flowables,
        Paragraph(
            "96 independent base draws x 3 shortage levels = 288 matched scenario "
            "conditions; 576 controller-scenario evaluations.",
            styles["callout"],
        ),
        _results_chart(evidence),
        Spacer(1, 6),
        _table_flowable(summary_data, styles),
        PageBreak(),
        Paragraph("Contents", styles["chapter"]),
        toc,
        PageBreak(),
        Paragraph("System at a glance", styles["chapter"]),
        _architecture_diagram(),
        Spacer(1, 10),
        Paragraph(
            _inline_markup(
                "One severity-neutral base-draw batch is reused across all three shortage "
                "levels, and the same resulting scenario tensor is passed to both controllers. All "
                "performance-critical simulation, policy inference, metric, and bootstrap "
                "operations remain on cuda:0; final display and canonical evidence payloads "
                "cross explicitly to host memory. Startup verifies the "
                "NVIDIA runtime, evidence hashes and semantics, PPO artifact, and a CUDA smoke "
                "inference before the first presentation window appears."
            ),
            styles["body"],
        ),
        Paragraph(
            "Claim boundary: every numeric result in this report describes the locked "
            "synthetic simulator and protocol. It is not operational evidence from Sudan's "
            "electrical grid.",
            styles["callout"],
        ),
    ]


def _screenshots(styles: dict[str, ParagraphStyle]) -> list[Flowable]:
    available = [(caption, path) for caption, path in SCREENSHOT_PATHS if path.is_file()]
    if not available:
        return []
    flowables: list[Flowable] = [PageBreak(), Paragraph("Desktop presentation", styles["chapter"])]
    max_width = A4[0] - 36 * mm
    for caption, path in available:
        image = Image(str(path))
        scale = min(max_width / image.imageWidth, 94 * mm / image.imageHeight)
        image.drawWidth = image.imageWidth * scale
        image.drawHeight = image.imageHeight * scale
        flowables.append(
            KeepTogether(
                [
                    image,
                    Paragraph(
                        f"{caption} - native offline desktop view captured on the "
                        "presentation laptop.",
                        styles["caption"],
                    ),
                ]
            )
        )
    return flowables


def load_verified_evidence() -> dict:
    """Load the canonical result only through the public semantic verifier."""

    return verify_paired_benchmark(
        PROJECT_ROOT / "data" / "evidence" / "paired-benchmark.json",
        expected_sha256=PAIRED_BENCHMARK_SHA256,
        frozen_benchmark_path=PROJECT_ROOT / "data" / "evidence" / "frozen-benchmark.json",
    )


def build_report(output_path: Path = DEFAULT_OUTPUT) -> Path:
    """Render the final report and return its resolved path."""

    if not REPORT_SOURCE_PATH.is_file():
        raise FileNotFoundError(f"missing report source: {REPORT_SOURCE_PATH}")
    for chapter in CHAPTER_PATHS:
        if not chapter.is_file():
            raise FileNotFoundError(f"missing report chapter: {chapter}")
    if not REFERENCE_PATH.is_file():
        raise FileNotFoundError(f"missing report references: {REFERENCE_PATH}")
    evidence = load_verified_evidence()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    styles = _styles()
    story: list[Flowable] = []
    story.extend(_cover(styles))
    story.extend(_front_matter(styles, evidence))
    for section_title in REPORT_SECTION_ORDER[:2]:
        story.extend(markdown_section_flowables(REPORT_SOURCE_PATH, section_title, styles))
    for chapter_path in CHAPTER_PATHS[:2]:
        story.extend(markdown_flowables(chapter_path, styles))
    story.extend(_screenshots(styles))
    story.extend(markdown_flowables(CHAPTER_PATHS[2], styles))
    for section_title in REPORT_SECTION_ORDER[2:]:
        story.extend(markdown_section_flowables(REPORT_SOURCE_PATH, section_title, styles))
    story.append(
        Paragraph(
            f"Canonical paired evidence SHA-256: {PAIRED_BENCHMARK_SHA256}",
            styles["small"],
        )
    )
    story.extend(markdown_flowables(REFERENCE_PATH, styles))
    document = ReportDocument(
        str(output_path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=20 * mm,
        bottomMargin=18 * mm,
        title="GridGuard Khartoum Final Graduation Project Report",
        author=(
            "Murtada Asim Ibrahim Bakhit; Suliman Elawad Suliman Babiker; "
            "Abdalhay Abass Mobarak Abass"
        ),
        subject="PPO-assisted synthetic load-shedding research simulation",
    )
    document.multiBuild(story, canvasmaker=_report_canvas)
    return output_path.resolve()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = build_report(args.output)
    print(f"Report written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

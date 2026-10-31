"""Assemble the standalone thesis from verified study text and figures."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from reportlab.graphics import renderPDF
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    CondPageBreak,
    Flowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    Spacer,
)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import build_extended_report as design  # noqa: E402
from tools import build_report as report  # noqa: E402

SOURCE = ROOT / "docs/FOCUSED_THESIS.md"
OUTPUT = ROOT / "output/pdf/updated/gridguard-khartoum-thesis.pdf"
CONTENT = ROOT / "tmp/focused-thesis-content.json"
FIGURES = ROOT / "docs/assets/focused-study"
SCREENSHOTS = ROOT / "docs/assets/current-app-screenshots"
COVER_PATH = ROOT / "docs/academic-cover.json"
COVER = json.loads(COVER_PATH.read_text(encoding="utf-8"))
TITLE = COVER["project_title"]
DATE = COVER["submission_date"]
INLINE_MARKUP = report._inline_markup
MATH = {}
REFERENCE_TARGETS = {}
REFERENCE_PAGES = {}
REFERENCE_PATTERN = re.compile(
    r"\b(?:Figure|Table|Section|Chapter) (?:\d+(?:\.\d+)*|[A-Z](?:\.\d+)*)\b"
)
FIGURE_DETAILS = {
    "workflow": (
        "1",
        "Research workflow",
        None,
        "Common synthetic inputs support matched control evaluation. Demand prediction "
        "is evaluated separately; forecasts are not fed into the controllers.",
    ),
    "overview": (
        "2",
        "Desktop overview",
        SCREENSHOTS / "overview.png",
        "The seven-locality map and saved evidence support offline inspection "
        "of synthetic scenarios.",
    ),
    "live": (
        "3",
        "Live Scenario inspection",
        SCREENSHOTS / "live-ppo.png",
        "The validation-selected seed-8111 checkpoint-2500 policy runs one matched episode. "
        "This demonstration is distinct from the frozen five-run results.",
    ),
    "outcomes": (
        "4",
        "Main controller outcomes",
        FIGURES / "controller-outcomes.png",
        "Results use 960 fresh matched test conditions. PPO is the equal-weight mean "
        "of five validation-selected policies. Each panel has its own units.",
    ),
    "intervals": (
        "5",
        "Paired PPO effects against Constrained Optimization",
        FIGURES / "paired-effects.png",
        "PPO minus Constrained Optimization, with adjusted 97.5% paired intervals. "
        "Fairness and utilization differences are in percentage points.",
    ),
    "training": (
        "6",
        "Validation learning trajectories",
        FIGURES / "training-curves.png",
        "Five runs on common validation conditions. Dots mark selected checkpoints; "
        "the dashed line marks iteration 600. These are validation results.",
    ),
    "diagnostics": (
        "7",
        "TensorBoard training diagnostics",
        FIGURES / "tensorboard-diagnostics.png",
        "Unsmoothed values every 25 iterations. The thick line is the five-run mean; "
        "shading is one sample standard deviation, not a confidence interval.",
    ),
    "variance": (
        "8",
        "Between-run validation variance",
        FIGURES / "training-variance.png",
        "Sample variance across five run means at 31 validation checkpoints, using "
        "denominator four. Units are squared score or squared percentage points.",
    ),
    "forecast": (
        "9",
        "Held-out forecasting error",
        FIGURES / "forecast-errors.png",
        "MAE and RMSE on 320 held-out traces. The training-fitted hourly profile is "
        "a post-hoc clock-aware baseline; forecasting is separate from control.",
    ),
    "locality": (
        "10",
        "Locality service at the end of the day",
        FIGURES / "locality-allocation.png",
        "Mean final locality service ratios and within-condition minimum, on 960 matched "
        "conditions. Annotations show PPO minus Constrained Optimization in percentage "
        "points. These endpoint measures differ from hourly-mean Jain fairness.",
    ),
}
ABBREVIATIONS = (
    ("AI", "Artificial intelligence"),
    ("API", "Application programming interface"),
    ("AR(3)", "Autoregression with three lags"),
    ("BC", "Behavior cloning"),
    ("CI", "Confidence interval"),
    ("CPU", "Central processing unit"),
    ("CUDA", "Compute Unified Device Architecture"),
    ("GAE", "Generalized advantage estimation"),
    ("GPU", "Graphics processing unit"),
    ("GUI", "Graphical user interface"),
    ("KL", "Kullback-Leibler divergence"),
    ("MAE", "Mean absolute error"),
    ("MDP", "Markov decision process"),
    ("MW / MWh", "Megawatt / megawatt-hour"),
    ("PPO", "Proximal Policy Optimization"),
    ("pp", "Percentage points"),
    ("ReLU", "Rectified linear unit"),
    ("RL", "Reinforcement learning"),
    ("RMSE", "Root mean square error"),
    ("SD", "Sample standard deviation"),
    ("SGD", "Stochastic gradient descent"),
    ("UI", "User interface"),
    ("SHA-256", "Secure Hash Algorithm, 256-bit digest"),
)


def reference_key(label: str) -> str:
    return label.lower().replace(" ", "-").replace(".", "-")


def navigation_inventory(markdown: str) -> dict[str, str]:
    """Reject missing captions and dangling references before rendering any PDF."""
    targets = {}
    for level, title in re.findall(r"(?m)^(#{1,3})\s+(.+)$", markdown):
        number = re.match(r"(\d+(?:\.\d+)*|[A-Z]\.\d+)\s", title)
        if number:
            label = ("Chapter " if len(level) == 1 else "Section ") + number[1]
            targets[label] = reference_key(label)
    for name in re.findall(r"\[\[FIGURE:([a-z]+)\]\]", markdown):
        if name not in FIGURE_DETAILS:
            raise ValueError(f"Unknown figure: {name}")
        label = "Figure " + FIGURE_DETAILS[name][0]
        if label in targets:
            raise ValueError(f"Duplicate caption: {label}")
        targets[label] = reference_key(label)
    pending_caption = None
    in_table = False
    for line in markdown.splitlines():
        caption = re.fullmatch(r"\*\*(Table \d+)\. (.+)\*\*", line.strip())
        if caption:
            if pending_caption:
                raise ValueError(f"Table caption has no table: {pending_caption}")
            label = caption[1]
            if label in targets:
                raise ValueError(f"Duplicate caption: {label}")
            targets[label] = reference_key(label)
            pending_caption = label
        elif line.startswith("|"):
            if not in_table:
                if pending_caption is None:
                    raise ValueError("Every thesis table requires a numbered caption")
                pending_caption = None
            in_table = True
            continue
        elif line.strip() and pending_caption:
            raise ValueError(f"Table caption must immediately precede its table: {pending_caption}")
        in_table = False
    if pending_caption:
        raise ValueError(f"Table caption has no table: {pending_caption}")
    table_numbers = [int(label.split()[1]) for label in targets if label.startswith("Table ")]
    if table_numbers and table_numbers != list(range(1, len(table_numbers) + 1)):
        raise ValueError("Table captions must be numbered consecutively in document order")
    for label in REFERENCE_PATTERN.findall(markdown):
        if label not in targets:
            raise ValueError(f"Unresolved thesis reference: {label}")
    prose = re.sub(r"(?m)^\*\*Table \d+\. .+\*\*$", "", markdown)
    callouts = set(REFERENCE_PATTERN.findall(prose))
    for label in targets:
        if label.startswith(("Figure ", "Table ")) and label not in callouts:
            raise ValueError(f"Missing in-text reference to {label}")
    return targets


def linked_references(markup: str) -> str:
    def link(match):
        label = match[0]
        destination = REFERENCE_TARGETS.get(label)
        if destination is None:
            return label
        return f'<link href="#{destination}" color="#137B73">{label}</link>'

    return REFERENCE_PATTERN.sub(link, markup)


def math_id(latex: str, display: bool) -> str:
    return hashlib.sha256(f"{display}:{latex.strip()}".encode()).hexdigest()[:20]


def prepare_math(markdown: str):
    """Typeset LaTeX before assembling paragraphs; preserve its editable source."""
    expressions = {}
    for match in re.finditer(r"\$\$(.*?)\$\$|(?<!\\)\$([^$\n]+)\$", markdown, re.S):
        display = match[1] is not None
        latex = (match[1] if display else match[2]).strip()
        identifier = math_id(latex, display)
        expressions[identifier] = {"id": identifier, "latex": latex, "display": display}
    directory = FIGURES / "math"
    manifest = ROOT / "tmp/math-typesetting/rendered.json"
    request = manifest.with_name("requests.json")
    request.parent.mkdir(parents=True, exist_ok=True)
    request.write_text(
        json.dumps(
            {
                "directory": str(directory),
                "manifest": str(manifest),
                "expressions": list(expressions.values()),
            }
        ),
        encoding="utf-8",
    )
    subprocess.run(["node", str(ROOT / "tools/render_thesis_math.cjs"), str(request)], check=True)
    MATH.clear()
    MATH.update(json.loads(manifest.read_text(encoding="utf-8")))


def inline_math(value: str) -> str:
    images = {}

    def replace(match):
        item = MATH[math_id(match[1], False)]
        token = f"LATEXINLINE{len(images):04}TOKEN"
        images[token] = (
            f'<img src="{item["png"]}" width="{item["width"]:.4f}" '
            f'height="{item["height"]:.4f}" valign="{item["baseline"]:.4f}"/>'
        )
        return token

    markup = INLINE_MARKUP(re.sub(r"(?<!\\)\$([^$\n]+)\$", replace, value))
    for token, image_markup in images.items():
        markup = markup.replace(token, image_markup)
    return linked_references(markup)


def equation(latex: str):
    item = MATH[math_id(latex, True)]
    scale = min(1, (A4[0] - 48 * mm) / item["width"])
    visual = Image(item["png"], width=item["width"] * scale, height=item["height"] * scale)
    visual.hAlign = "CENTER"
    return KeepTogether([Spacer(1, 6), visual, Spacer(1, 10)])


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derived_figure_evidence(markdown: str) -> dict[str, str]:
    """Bind descriptive figure data and reject images detached from their recorded derivation."""
    if "[[FIGURE:locality]]" not in markdown:
        return {}
    source = FIGURES / "locality-allocation.json"
    sidecar = source.with_suffix(".sha256")
    digest = sha(source)
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    if digest != expected:
        raise ValueError("Locality figure evidence differs from its SHA-256 record")
    evidence = json.loads(source.read_text(encoding="utf-8"))
    if evidence.get("schema") != "gridguard.focused_locality_interpretation.1":
        raise ValueError("Unsupported locality figure evidence schema")
    for name in ("locality-allocation.png", "locality-allocation.svg"):
        if sha(FIGURES / name) != evidence.get("figure_sha256", {}).get(name):
            raise ValueError(f"Locality figure differs from its evidence: {name}")
    details = FIGURES / "interpretation-details.json"
    details_sidecar = details.with_suffix(".sha256")
    if sha(details) != details_sidecar.read_text(encoding="utf-8").split()[0]:
        raise ValueError("Interpretation evidence differs from its SHA-256 record")
    interpretation = json.loads(details.read_text(encoding="utf-8"))
    if interpretation.get("schema") != "gridguard.focused_training_generator_interpretation.1":
        raise ValueError("Unsupported interpretation evidence schema")
    for record in (evidence, interpretation):
        for relative, expected in record["source_sha256"].items():
            path = (ROOT / relative).resolve()
            if not path.is_relative_to(ROOT.resolve()) or sha(path) != expected:
                raise ValueError(f"Interpretation source differs from its evidence: {relative}")
    return {
        path.relative_to(ROOT).as_posix(): sha(path)
        for path in (source, sidecar, details, details_sidecar)
    }


class ThesisDocument(report.ReportDocument):
    def beforeDocument(self):
        self.reference_pages = {}
        self.caption_entries = {"figure": [], "table": []}

    def _draw_page(self, canvas, document):
        canvas.saveState()
        w, h = A4
        canvas.setFillColor(design.WHITE)
        canvas.rect(0, 0, w, h, fill=1, stroke=0)
        if document.page != 1:
            canvas.setFillColor(design.PALE)
            canvas.rect(0, h - 15 * mm, w, 15 * mm, fill=1, stroke=0)
            canvas.setFillColor(design.TEAL)
            canvas.setFont(design.F.body_bold, 7.5)
            canvas.drawString(18 * mm, h - 10 * mm, "GRIDGUARD KHARTOUM")
            canvas.setFillColor(design.MUTED)
            canvas.setFont(design.F.body, 7)
            canvas.drawRightString(w - 18 * mm, h - 10 * mm, "GRADUATION PROJECT  /  THESIS")
            canvas.setStrokeColor(design.RULE)
            canvas.setLineWidth(0.5)
            canvas.line(18 * mm, 16 * mm, w - 18 * mm, 16 * mm)
            canvas.drawString(18 * mm, 11 * mm, DATE)
            canvas.setFillColor(design.DEEP)
            canvas.setFont(design.F.body_bold, 8)
            canvas.drawRightString(w - 18 * mm, 11 * mm, f"{document.page:02}")
        canvas.restoreState()

    def afterFlowable(self, flowable):
        if not isinstance(flowable, Paragraph):
            return
        kind = getattr(flowable, "_navigation_kind", None)
        if kind:
            label = flowable._navigation_label
            key = reference_key(label)
            self.canv.bookmarkPage(key)
            self.reference_pages[key] = self.page
            title = flowable._navigation_title
            self.caption_entries[kind].append((title, self.page, key))
            self.notify(kind + "Entry", (0, title, self.page, key))
            return
        levels = {"Chapter": 0, "Section": 1, "Subsection": 2, "ArabicChapter": 0}
        level = levels.get(flowable.style.name)
        if level is None:
            return
        title = getattr(flowable, "_navigation_heading", flowable.getPlainText())
        number = re.match(r"(\d+(?:\.\d+)*|[A-Z]\.\d+)\s", title)
        if number:
            label = ("Chapter " if level == 0 else "Section ") + number[1]
            key = reference_key(label)
        else:
            key = "heading-" + hashlib.sha256(title.encode()).hexdigest()[:12]
        self.reference_pages[key] = self.page
        self.canv.bookmarkPage(key)
        self.canv.addOutlineEntry(title, key, level=level, closed=level > 0)
        if level <= 1 and title != "Contents":
            self.notify("TOCEntry", (level, title, self.page, key))


class CaptionContents(TableOfContents):
    def __init__(self, kind: str):
        super().__init__()
        self.kind = kind

    def notify(self, kind, stuff):
        if kind == self.kind + "Entry":
            self.addEntry(*stuff)


def arabic_visual(text: str) -> str:
    """Shape a single logical line, then apply Unicode bidirectional ordering."""
    import arabic_reshaper
    from bidi.algorithm import get_display

    return get_display(arabic_reshaper.reshape(text), base_dir="R")


class ArabicParagraph(Flowable):
    """Wrap logical Arabic before visual reordering, preserving each line's order."""

    def __init__(self, text: str, style: ParagraphStyle):
        super().__init__()
        self.text = text
        self.style = style
        self.spaceAfter = style.spaceAfter

    def wrap(self, width, height):
        self.width = width
        self.lines = []
        line = ""
        for word in self.text.split():
            candidate = (line + " " + word).strip()
            visual = arabic_visual(candidate)
            if (
                line
                and pdfmetrics.stringWidth(visual, self.style.fontName, self.style.fontSize) > width
            ):
                self.lines.append(line)
                line = word
            else:
                line = candidate
        if line:
            self.lines.append(line)
        self.height = len(self.lines) * self.style.leading
        return self.width, self.height

    def split(self, width, height):
        self.wrap(width, height)
        count = int(height // self.style.leading)
        if count < 2 or len(self.lines) - count < 2:
            return []
        return [
            ArabicParagraph(" ".join(self.lines[:count]), self.style),
            ArabicParagraph(" ".join(self.lines[count:]), self.style),
        ]

    def draw(self):
        self.canv.setFillColor(self.style.textColor)
        self.canv.setFont(self.style.fontName, self.style.fontSize)
        for index, line in enumerate(self.lines):
            self.canv.drawRightString(
                self.width, self.height - (index + 0.8) * self.style.leading, arabic_visual(line)
            )


def styles():
    for name, filename in (
        ("GridGuardAtkinsonItalic", "AtkinsonHyperlegible-Italic.ttf"),
        ("GridGuardAtkinsonBoldItalic", "AtkinsonHyperlegible-BoldItalic.ttf"),
    ):
        pdfmetrics.registerFont(TTFont(name, ROOT / "docs/assets/fonts" / filename))
    pdfmetrics.registerFontFamily(
        "GridGuardAtkinson",
        normal="GridGuardAtkinson",
        bold="GridGuardAtkinsonBold",
        italic="GridGuardAtkinsonItalic",
        boldItalic="GridGuardAtkinsonBoldItalic",
    )
    for weight in ("Regular", "Bold"):
        pdfmetrics.registerFont(
            TTFont(
                "GridGuardArabic" + weight,
                ROOT / "docs/assets/fonts" / f"NotoNaskhArabic-{weight}.ttf",
            )
        )
    report.REPORT_FONTS = design.F
    report.INK = report.NAVY = report.SLATE = design.DEEP
    report.MUTED = design.MUTED
    report.TEAL = design.TEAL
    report.OFF_WHITE = report.SURFACE = design.WHITE
    report.CANVAS = design.PALE
    report._table_flowable = thesis_table
    report._inline_markup = inline_math
    s = design.styles()
    s["body"].fontSize = 10.5
    s["body"].leading = 15
    s["body"].spaceAfter = 7
    s["chapter"].fontName = design.F.display_bold
    s["chapter"].fontSize = 25
    s["chapter"].leading = 29
    s["small"].fontSize = 8.5
    s["small"].leading = 11.4
    s["table_header"].fontSize = 8.5
    s["table_header"].leading = 11.4
    s["table_header"].textColor = design.WHITE
    s["caption"].fontSize = 8.6
    s["caption"].leading = 12
    s["table_caption"] = ParagraphStyle(
        "TableCaption",
        parent=s["caption"],
        fontName=design.F.body_bold,
        textColor=design.TEAL,
        keepWithNext=True,
        spaceBefore=9,
        spaceAfter=6,
    )
    s["arabic"] = ParagraphStyle(
        "ArabicBody",
        parent=s["body"],
        fontName="GridGuardArabicRegular",
        fontSize=12,
        leading=15.5,
        alignment=2,
        spaceAfter=6,
    )
    s["arabic_chapter"] = ParagraphStyle(
        "ArabicChapter",
        parent=s["chapter"],
        fontName="GridGuardArabicBold",
        fontSize=27,
        leading=34,
        alignment=2,
    )
    return s


def thesis_table(rows, s):
    table = design.table_flowable(rows, s)
    # Keep the nine-row severity comparison together so each controller group is intact.
    return KeepTogether([table]) if rows[0][0] == "Shortage" else table


def cover(s):
    from PIL import Image as PILImage

    def centered(name, size, leading, *, bold=False, color=design.DEEP):
        return ParagraphStyle(
            name,
            fontName=design.F.body_bold if bold else design.F.body,
            fontSize=size,
            leading=leading,
            alignment=1,
            textColor=color,
        )

    label = centered("CoverLabel", 9, 13, bold=True, color=design.TEAL)
    person = centered("CoverPerson", 11, 16)
    logo_path = ROOT / COVER["logos"]
    with PILImage.open(logo_path) as logo:
        logo_width, logo_height = logo.size
    width = A4[0] - 36 * mm
    logo_image = Image(str(logo_path), width=width, height=width * logo_height / logo_width)
    items = [
        Spacer(1, 2 * mm),
        logo_image,
        Spacer(1, 4 * mm),
        Paragraph(COVER["university"], centered("CoverUniversity", 22, 26, bold=True)),
        Paragraph(COVER["faculty"], centered("CoverFaculty", 13, 18)),
        Spacer(1, 9 * mm),
        Paragraph(TITLE, centered("CoverTitle", 25, 29, bold=True, color=design.TEAL)),
        Spacer(1, 3 * mm),
        Paragraph(COVER["project_name"].upper(), centered("CoverProduct", 10, 14, bold=True)),
        Spacer(1, 5 * mm),
        Paragraph("GRADUATION PROJECT FINAL REPORT", label),
        Spacer(1, 3 * mm),
        Paragraph(COVER["cover_statement"], centered("CoverStatement", 10.5, 14)),
        Spacer(1, 6 * mm),
        Paragraph("SUBMITTED BY", label),
    ]
    items.extend(
        Paragraph(f"{author['name']} ({author['index']})", person) for author in COVER["authors"]
    )
    if COVER.get("supervisor"):
        items.extend(
            [
                Spacer(1, 6 * mm),
                Paragraph("SUPERVISED BY", label),
                Paragraph(COVER["supervisor"], person),
            ]
        )
    items.extend(
        [
            Spacer(1, 7 * mm),
            Paragraph(COVER["submission_date"], centered("CoverDate", 10.5, 14)),
        ]
    )
    return items


def workflow():
    width = A4[0] - 36 * mm
    d = Drawing(width, 225)
    cards = [
        (
            0,
            145,
            "1  SCENARIOS",
            ["7 localities · 5 sectors", "24 hours · 3 shortages"],
            110,
        ),
        (
            127,
            145,
            "2  CONTROLLERS",
            ["Rules and optimization", "PPO after imitation"],
            110,
        ),
        (
            254,
            145,
            "3  SHARED DISPATCH",
            ["Tier requests and supply", "Proportional scaling"],
            110,
        ),
        (
            381,
            145,
            "4  OUTCOMES",
            ["Critical interruption", "Fairness · energy"],
            110,
        ),
        (
            0,
            12,
            "PREDICTION STUDY",
            ["Ridge and baseline forecasts", "Evaluated separately from control"],
            151,
        ),
        (
            171,
            12,
            "TRAIN THEN FREEZE",
            ["5 PPO seeds · validation choice", "Fresh final matched scenarios"],
            151,
        ),
        (
            342,
            12,
            "OFFLINE APPLICATION",
            ["Locality map · live scenario", "Verified saved evidence"],
            151,
        ),
    ]
    for x, y, title, lines, card_width in cards:
        d.add(
            Rect(
                x,
                y,
                card_width,
                63,
                rx=7,
                ry=7,
                fillColor=design.PALE,
                strokeColor=design.RULE,
                strokeWidth=0.6,
            )
        )
        d.add(
            String(
                x + 7,
                y + 46,
                title,
                fontName=design.F.body_bold,
                fontSize=7.8,
                fillColor=design.TEAL,
            )
        )
        for index, line in enumerate(lines):
            d.add(
                String(
                    x + 7,
                    y + 29 - index * 13,
                    line,
                    fontName=design.F.body,
                    fontSize=7.6 if card_width == 110 else 8.2,
                    fillColor=design.DEEP,
                )
            )
    for x in (112, 239, 366):
        d.add(Line(x, 176, x + 12, 176, strokeColor=design.TEAL, strokeWidth=1.2))
        d.add(Line(x + 12, 176, x + 8, 179, strokeColor=design.TEAL, strokeWidth=1.2))
        d.add(Line(x + 12, 176, x + 8, 173, strokeColor=design.TEAL, strokeWidth=1.2))
    d.add(
        String(
            0,
            112,
            "Hourly allocation uses identical inputs and dispatch rules",
            fontName=design.F.body,
            fontSize=8.5,
            fillColor=design.MUTED,
        )
    )
    return d


def caption(kind: str, label: str, title: str, explanation: str, s):
    style = s["table_caption"] if kind == "table" else s["caption"]
    text = f"<b>{label}. {INLINE_MARKUP(title.rstrip('.') + '.')}</b>"
    if explanation:
        text += " " + INLINE_MARKUP(explanation)
    paragraph = Paragraph(text, style)
    paragraph._navigation_kind = kind
    paragraph._navigation_label = label
    paragraph._navigation_title = f"{label}. {title}"
    return paragraph


def figure(name: str, s):
    number, title, path, explanation = FIGURE_DETAILS[name]
    if path is None:
        visual = workflow()
    else:
        from PIL import Image as PILImage

        with PILImage.open(path) as image:
            w, h = image.size
        target_width = A4[0] - 36 * mm
        target_height = target_width * h / w
        if target_height > 175 * mm:
            target_width *= 175 * mm / target_height
            target_height = 175 * mm
        visual = Image(str(path), width=target_width, height=target_height)
    return KeepTogether(
        [
            Spacer(1, 6),
            visual,
            caption("figure", "Figure " + number, title, explanation, s),
        ]
    )


def flowables(markdown, s):
    result = []
    pattern = r"(\[\[FIGURE:[a-z]+\]\]|\$\$[\s\S]*?\$\$|^\*\*Table \d+\. .+\*\*$)"
    for part in re.split(pattern, markdown, flags=re.M):
        marker = re.fullmatch(r"\[\[FIGURE:([a-z]+)\]\]", part)
        table_caption = re.fullmatch(r"\*\*(Table \d+)\. (.+)\*\*", part)
        if marker:
            result.append(figure(marker[1], s))
        elif table_caption:
            result.append(caption("table", table_caption[1], table_caption[2], "", s))
        elif part.startswith("$$"):
            if result and isinstance(result[-1], Paragraph):
                result[-1].keepWithNext = True
            result.append(equation(part[2:-2].strip()))
        elif part.strip():
            for item in report.markdown_text_flowables(part, s):
                if isinstance(item, PageBreak):
                    while result and isinstance(result[-1], Spacer):
                        result.pop()
                if (
                    isinstance(item, KeepTogether)
                    and result
                    and getattr(result[-1], "_navigation_kind", "") == "table"
                ):
                    result.append(KeepTogether([result.pop(), *item._content]))
                    continue
                if isinstance(item, Paragraph) and item.getPlainText().endswith(
                    "Computation and verification"
                ):
                    result.append(CondPageBreak(400))
                if isinstance(item, Paragraph) and re.search(
                    r"[\u0600-\u06ff]", item.getPlainText()
                ):
                    if item.style.name == "Chapter":
                        item = Paragraph(arabic_visual(item.getPlainText()), s["arabic_chapter"])
                        item._navigation_heading = "Arabic abstract"
                    else:
                        item = ArabicParagraph(item.getPlainText(), s["arabic"])
                result.append(item)
    return result


def contents_styles():
    return [
        ParagraphStyle(
            "ContentsChapter",
            fontName=design.F.body_bold,
            fontSize=10.5,
            leading=15,
            textColor=design.DEEP,
            spaceBefore=8,
            rightIndent=24,
        ),
        ParagraphStyle(
            "ContentsSection",
            fontName=design.F.body,
            fontSize=9.5,
            leading=13.5,
            textColor=design.MUTED,
            leftIndent=13,
            firstLineIndent=0,
            rightIndent=24,
            spaceBefore=3,
        ),
    ]


def document_story(markdown: str, s):
    first_chapter = markdown.index("\n# 1 ")
    abstract, chapters = markdown[:first_chapter], markdown[first_chapter:]
    toc = TableOfContents()
    toc.levelStyles = contents_styles()
    story = cover(s) + flowables(abstract, s)
    story += [PageBreak(), Paragraph("Contents", s["chapter"]), Spacer(1, 5 * mm), toc]
    for kind in ("figure", "table"):
        entries = CaptionContents(kind)
        entries.levelStyles = [
            ParagraphStyle(
                "CaptionIndex",
                parent=contents_styles()[1],
                leftIndent=0,
                rightIndent=42,
                fontSize=9.5,
                leading=11.5,
                spaceBefore=1,
            )
        ]
        story += [PageBreak(), Paragraph("List of " + kind + "s", s["chapter"]), entries]
    story += [PageBreak(), Paragraph("List of abbreviations", s["chapter"])]
    story += [design.table_flowable([["Abbreviation", "Meaning"], *ABBREVIATIONS], s)]
    chapter_body, reference_heading, reference_text = chapters.partition("\n# References")
    story += flowables(chapter_body, s)
    if reference_heading:
        reference_styles = dict(s)
        for key in ("body", "list_body"):
            reference_styles[key] = ParagraphStyle(
                "Reference" + key,
                parent=s[key],
                fontSize=9.5,
                leading=12.5,
                spaceAfter=7,
                leftIndent=12,
                firstLineIndent=-12,
            )
        story += flowables(reference_heading + reference_text, reference_styles)
    return story


def render_document(markdown: str, output: Path) -> dict[str, int]:
    """Resolve captions, contents and prose page references until pagination is stable."""
    REFERENCE_TARGETS.clear()
    REFERENCE_TARGETS.update(navigation_inventory(markdown))
    REFERENCE_PAGES.clear()
    s = styles()
    output.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(6):
        doc = ThesisDocument(
            str(output),
            pagesize=A4,
            leftMargin=18 * mm,
            rightMargin=18 * mm,
            topMargin=23 * mm,
            bottomMargin=22 * mm,
            title=TITLE,
            author="; ".join(author["name"] for author in COVER["authors"]),
            pageCompression=1,
        )
        doc.multiBuild(document_story(markdown, s), canvasmaker=report._report_canvas)
        if doc.reference_pages == REFERENCE_PAGES:
            missing = set(REFERENCE_TARGETS.values()) - doc.reference_pages.keys()
            if missing:
                raise ValueError(f"Missing rendered destinations: {sorted(missing)}")
            return dict(doc.reference_pages)
        REFERENCE_PAGES.clear()
        REFERENCE_PAGES.update(doc.reference_pages)
    raise ValueError("Thesis page references did not converge")


def build(output: Path, content_path: Path):
    content = json.loads(content_path.read_text(encoding="utf-8"))
    if output == OUTPUT:
        for required in (
            "data/evidence/focused-ppo-training.json",
            "data/evidence/focused-ppo-study.json",
        ):
            if required not in content["evidence_sha256"]:
                raise ValueError(f"Final thesis requires completed study evidence: {required}")
    for relative, expected in content["evidence_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"Evidence changed: {relative}")
    markdown = re.sub(r"<!--.*?-->", "", SOURCE.read_text(encoding="utf-8"), flags=re.S)
    # Source figure notes become a single numbered caption supplied with each visual.
    markdown = re.sub(r"(?ms)^\*\*Figure \d+\..*?(?=\n\s*\n)", "", markdown)
    body, separator, references = markdown.partition("# References")
    body = re.sub(
        r"(?m)((?:^ {4}.+\n?)+)",
        lambda match: "```\n" + "\n".join(line[4:] for line in match[0].splitlines()) + "\n```\n",
        body,
    )
    markdown = body + separator + references
    for key, value in content["replacements"].items():
        markdown = markdown.replace("{{" + key + "}}", value)
    markdown = markdown.replace("{{SUPERVISOR}}", COVER["supervisor"])
    unresolved = re.findall(r"\{\{.*?\}\}", markdown)
    if unresolved:
        raise ValueError(f"Unresolved thesis content: {unresolved}")
    if output == OUTPUT and "LAYOUT PREVIEW" in markdown:
        raise ValueError("Unfinished preview content cannot replace the final thesis")
    if re.search(r"\bproposal\b", markdown, flags=re.I):
        raise ValueError("The final thesis must stand alone without proposal references")
    if re.search(r"\[\d+(?:\s*[,;–-]\s*\d+)*\]", markdown):
        raise ValueError("Use Harvard author-date citations instead of numeric citations")
    if not re.match(r"# (?:Declaration|Dedication|Abstract)\s*\n", markdown.lstrip()):
        raise ValueError("Thesis source must begin with its declared front matter")
    if "\n# Abstract\n" not in "\n" + markdown:
        raise ValueError("Thesis source requires an English abstract")
    if "Original study:" in markdown or "Integrated conclusion" in markdown:
        raise ValueError("Superseded fragmented chapter structure found")
    derived_figures = derived_figure_evidence(markdown)
    prepare_math(markdown)
    # Export the same vector workflow used in the PDF for the editable Word edition.
    workflow_svg = FIGURES / "research-workflow.svg"
    workflow_png = workflow_svg.with_suffix(".png")
    workflow_svg.write_text(
        workflow()
        .asString("svg")
        .replace(
            "font-family: GridGuardAtkinsonBold;",
            "font-family: Atkinson Hyperlegible; font-weight: 700;",
        )
        .replace("font-family: GridGuardAtkinson;", "font-family: Atkinson Hyperlegible;"),
        encoding="utf-8",
    )
    # Rasterize the embedded-font PDF so the Word figure uses the same glyph metrics.
    # SVG rasterizers can silently substitute a wider font for ReportLab's font alias.
    workflow_pdf = ROOT / "tmp/research-workflow-render.pdf"
    renderPDF.drawToFile(workflow(), str(workflow_pdf))
    subprocess.run(
        [
            "pdftoppm",
            "-r",
            "300",
            "-singlefile",
            "-png",
            str(workflow_pdf),
            str(workflow_png.with_suffix("")),
        ],
        check=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        backup = ROOT / "output/pdf/.previous" / f"{output.stem}-{sha(output)[:12]}.pdf"
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            shutil.copy2(output, backup)
    reference_pages = render_document(markdown, output)
    rendered_source = output.with_suffix(".source.md")
    rendered_source.write_text(markdown, encoding="utf-8")
    receipt = {
        "schema": "gridguard.focused_thesis_build.2",
        "navigation_pages": reference_pages,
        "figure_metadata": {
            name: {
                "number": number,
                "title": title,
                "path": (path or workflow_png).relative_to(ROOT).as_posix(),
                "explanation": explanation,
            }
            for name, (number, title, path, explanation) in FIGURE_DETAILS.items()
        },
        "abbreviations": ABBREVIATIONS,
        "front_matter": [
            "Declaration",
            "Dedication",
            "Acknowledgements",
            "Abstract",
            "Arabic abstract",
            "Contents",
            "List of figures",
            "List of tables",
            "List of abbreviations",
        ],
        "pdf_sha256": sha(output),
        "source_sha256": sha(SOURCE),
        "resolved_source_sha256": sha(rendered_source),
        "builder_sha256": sha(Path(__file__)),
        "cover_metadata_sha256": sha(COVER_PATH),
        "cover_logo_sha256": {COVER["logos"]: sha(ROOT / COVER["logos"])},
        "content_sha256": sha(content_path),
        "evidence_sha256": content["evidence_sha256"],
        "literature_review_sha256": sha(ROOT / "docs/source/literature-review-authors.docx"),
        "figures_sha256": {
            p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(FIGURES.rglob("*.png"))
        },
        "derived_figure_evidence_sha256": derived_figures,
        "screenshots_sha256": {
            p.relative_to(ROOT).as_posix(): sha(p)
            for p in (SCREENSHOTS / "overview.png", SCREENSHOTS / "live-ppo.png")
        },
        "latex_renderer_sha256": sha(ROOT / "tools/render_thesis_math.cjs"),
        "latex_engine": "MathJax 3.2.2 TeX input; self-contained SVG and 600-dpi PNG",
        "latex_expression_count": len(MATH),
        "citation_style": "Harvard author-date; alphabetical unnumbered reference list",
        "italic_fonts_sha256": {
            p.relative_to(ROOT).as_posix(): sha(p)
            for p in sorted((ROOT / "docs/assets/fonts").glob("*.ttf"))
        },
        "figure_renderers_sha256": {
            filename: sha(ROOT / filename)
            for filename in ("tools/plot_focused_thesis.py", "tools/plot_training_diagnostics.py")
        },
    }
    output.with_suffix(".build.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(output)
    print("SHA-256:", receipt["pdf_sha256"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--content", type=Path, default=CONTENT)
    args = parser.parse_args()
    build(args.output.resolve(), args.content.resolve())

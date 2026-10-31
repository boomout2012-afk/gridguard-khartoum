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

from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, Spacer
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
DATE = "23 October 2026"
INLINE_MARKUP = report._inline_markup
MATH = {}


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
    return markup


def equation(latex: str):
    item = MATH[math_id(latex, True)]
    scale = min(1, (A4[0] - 48 * mm) / item["width"])
    visual = Image(item["png"], width=item["width"] * scale, height=item["height"] * scale)
    visual.hAlign = "CENTER"
    return KeepTogether([Spacer(1, 6), visual, Spacer(1, 10)])


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ThesisDocument(report.ReportDocument):
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

    def notify(self, kind, stuff):
        if kind == "TOCEntry" and stuff[0] > 0:
            return
        return super().notify(kind, stuff)


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
    d = Drawing(width, 185)
    cards = [
        (
            0,
            112,
            "1  SYNTHETIC SCENARIOS",
            ["7 localities · 5 sectors", "24 hours · 3 shortage levels"],
        ),
        (
            171,
            112,
            "2  MATCHED DECISIONS",
            ["Fixed Rotation · constrained", "optimizer · PPO with constraints"],
        ),
        (
            342,
            112,
            "3  MEASURED OUTCOMES",
            ["Critical downtime · fairness", "utilization · daily reward"],
        ),
        (0, 12, "PREDICTION STUDY", ["Causal ridge forecast", "Evaluated separately from control"]),
        (
            171,
            12,
            "TRAIN THEN FREEZE",
            ["5 PPO seeds · validation choice", "Fresh final matched scenarios"],
        ),
        (
            342,
            12,
            "OFFLINE APPLICATION",
            ["Locality map · live scenario", "Verified saved evidence"],
        ),
    ]
    for x, y, title, lines in cards:
        d.add(
            Rect(
                x,
                y,
                151,
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
                x + 9, y + 46, title, fontName=design.F.body_bold, fontSize=8, fillColor=design.TEAL
            )
        )
        for index, line in enumerate(lines):
            d.add(
                String(
                    x + 9,
                    y + 29 - index * 13,
                    line,
                    fontName=design.F.body,
                    fontSize=8.2,
                    fillColor=design.DEEP,
                )
            )
    for x in (155, 326):
        d.add(Line(x, 143, x + 12, 143, strokeColor=design.TEAL, strokeWidth=1.2))
        d.add(Line(x + 12, 143, x + 8, 146, strokeColor=design.TEAL, strokeWidth=1.2))
        d.add(Line(x + 12, 143, x + 8, 140, strokeColor=design.TEAL, strokeWidth=1.2))
    return d


def figure(name: str, s):
    mapping = {
        "workflow": (
            None,
            "Figure 1. Project structure. Prediction is a separate evaluated component; "
            "forecasts are not fed into the controller study.",
        ),
        "overview": (
            SCREENSHOTS / "overview.png",
            "Figure 2. The desktop overview presents the seven localities and saved evidence. "
            "The application is an offline decision-support demonstration using synthetic data.",
        ),
        "live": (
            SCREENSHOTS / "live-ppo.png",
            "Figure 3. Live Scenario uses the validation-selected PPO policy (seed 8111, "
            "iteration 2500) on matched inputs. Its single episode remains distinct from "
            "the frozen five-run study.",
        ),
        "training": (
            FIGURES / "training-curves.png",
            "Figure 6. Validation learning curves for all five training runs. Dots mark selected "
            "checkpoints; the dashed vertical line marks the retained 600-iteration reference.",
        ),
        "outcomes": (
            FIGURES / "controller-outcomes.png",
            "Figure 4. Main controller outcomes on 960 fresh matched conditions. PPO is the mean "
            "of five validation-selected policies; panels use separate units.",
        ),
        "intervals": (
            FIGURES / "paired-effects.png",
            "Figure 5. Paired PPO effects against the constrained optimizer, with 97.5% intervals "
            "adjusted for the two primary claims. Both baseline comparisons are reported in the "
            "tables. Percentage outcomes use percentage-point differences.",
        ),
        "forecast": (
            FIGURES / "forecast-errors.png",
            "Figure 9. Learned forecasting reduces error on 320 held-out synthetic traces. "
            "MAE and RMSE are reported separately for locality-sector and system-total targets. "
            "This prediction experiment is separate from the controller comparison.",
        ),
        "diagnostics": (
            FIGURES / "tensorboard-diagnostics.png",
            "Figure 7. TensorBoard training diagnostics for all five PPO runs, without "
            "smoothing. Shading is one sample standard deviation across runs, not a "
            "confidence interval. Iteration zero has no PPO update diagnostics.",
        ),
        "variance": (
            FIGURES / "training-variance.png",
            "Figure 8. Sample variance across five independently trained policies on the "
            "fixed validation scenarios. The variance uses a denominator of four; each "
            "panel reports its own squared units.",
        ),
    }
    path, caption = mapping[name]
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
    return KeepTogether([Spacer(1, 6), visual, Paragraph(caption, s["caption"])])


def flowables(markdown, s):
    result = []
    for part in re.split(r"(\[\[FIGURE:[a-z]+\]\]|\$\$[\s\S]*?\$\$)", markdown):
        marker = re.fullmatch(r"\[\[FIGURE:([a-z]+)\]\]", part)
        if marker:
            result.append(figure(marker[1], s))
        elif part.startswith("$$"):
            result.append(equation(part[2:-2].strip()))
        elif part.strip():
            result.extend(report.markdown_text_flowables(part, s))
    return result


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
    unresolved = re.findall(r"\{\{.*?\}\}", markdown)
    if unresolved:
        raise ValueError(f"Unresolved thesis content: {unresolved}")
    if output == OUTPUT and "LAYOUT PREVIEW" in markdown:
        raise ValueError("Unfinished preview content cannot replace the final thesis")
    if re.search(r"\bproposal\b", markdown, flags=re.I):
        raise ValueError("The final thesis must stand alone without proposal references")
    if re.search(r"\[\d+(?:\s*[,;–-]\s*\d+)*\]", markdown):
        raise ValueError("Use Harvard author-date citations instead of numeric citations")
    if not markdown.lstrip().startswith("# Abstract"):
        raise ValueError("Thesis source must begin with # Abstract")
    if "Original study:" in markdown or "Integrated conclusion" in markdown:
        raise ValueError("Superseded fragmented chapter structure found")
    first_chapter = markdown.index("\n# 1 ")
    abstract, chapters = markdown[:first_chapter], markdown[first_chapter:]
    prepare_math(markdown)
    s = styles()
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle(
            "ContentsEntry",
            fontName=design.F.body,
            fontSize=11,
            leading=18,
            textColor=design.DEEP,
            spaceBefore=10,
        )
    ]
    story = cover(s) + flowables(abstract, s)
    story += [PageBreak(), Paragraph("Contents", s["chapter"]), Spacer(1, 8 * mm), toc]
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
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        backup = ROOT / "output/pdf/.previous" / f"{output.stem}-{sha(output)[:12]}.pdf"
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            shutil.copy2(output, backup)
    doc = ThesisDocument(
        str(output),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=23 * mm,
        bottomMargin=22 * mm,
        title=TITLE,
        author="Murtada Bakhit; Suliman Babiker; Abdalhay Abass",
        pageCompression=1,
    )
    doc.multiBuild(story, canvasmaker=report._report_canvas)
    rendered_source = output.with_suffix(".source.md")
    rendered_source.write_text(markdown, encoding="utf-8")
    receipt = {
        "schema": "gridguard.focused_thesis_build.1",
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

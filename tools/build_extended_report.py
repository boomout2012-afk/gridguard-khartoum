"""Build a redesigned review edition without altering release-bound artifacts."""

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    Flowable,
    Image,
    KeepTogether,
    LongTable,
    PageBreak,
    Paragraph,
    Preformatted,
    Spacer,
    TableStyle,
)

from gridguard.proposal_evidence import load_proposal_supplement

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import build_report as report  # noqa: E402

PALE = HexColor("#F5F8F5")
WHITE = HexColor("#FFFFFF")
DEEP = HexColor("#173F43")
TEAL = HexColor("#137B73")
CORAL = HexColor("#E77D60")
MUTED = HexColor("#5C7475")
RULE = HexColor("#DCE7E1")

F = report.REPORT_FONTS._replace(display="GridGuardAtkinson", display_bold="GridGuardAtkinsonBold")
original_styles = report._styles
original_front_matter = report._front_matter
original_notify = report.ReportDocument.notify
original_section = report.markdown_section_text_flowables
original_markdown = report.markdown_text_flowables


def section_flowables(
    markdown: str, section_title: str, s: dict[str, ParagraphStyle]
) -> list[Flowable]:
    flow = original_section(markdown, section_title, s)
    if section_title in ("Introduction", "Discussion", "Conclusion"):
        for item in flow:
            if isinstance(item, Paragraph) and item.style.name == "Chapter":
                item.text = "Original study: " + section_title.lower()
                item.__init__(item.text, s["chapter"])
                break
        position = next(i for i, item in enumerate(flow) if isinstance(item, Paragraph)) + 1
        flow.insert(
            position,
            Paragraph(
                "Historical study context. This section preserves the original PPO–Fixed Rotation "
                "study. The additional Optimization comparator, learned forecast and direct "
                "duration outcomes are reported in the extension chapters and integrated "
                "conclusion.",
                s["callout"],
            ),
        )
    if section_title == "Conclusion":
        flow += [
            PageBreak(),
            Paragraph("Integrated conclusion", s["chapter"]),
            Paragraph(
                "The expanded project covers the proposal’s four selected method categories: "
                "rule-based scheduling, reinforcement learning, explicit dispatch optimization and "
                "learned prediction. It also directly measures blackout duration and inequality "
                "across localities, alongside the original service, fairness and efficiency "
                "outcomes.",
                s["body"],
            ),
            Paragraph(
                "On the additional 960 matched synthetic conditions, Optimization records "
                "6.75 critical interruption locality-hours versus 19.86 for PPO and 102.07 for "
                "Fixed Rotation. PPO retains higher Jain service fairness than Optimization "
                "(96.89% versus 94.89%), while Optimization has higher utilization "
                "(98.24% versus 97.62%). The result is a measured trade-off, not "
                "general PPO superiority.",
                s["body"],
            ),
            Paragraph(
                "Both adaptive controllers have zero zero-service blackout hours in this "
                "sample. Their zero duration disparities partly reflect shared proportional "
                "clipping "
                "in the simulator and do not mean critical interruptions disappear. The learned "
                "ridge predictor reduces system-total MAE to 9.34 MW versus 24.38 MW for "
                "persistence; this is a separate prediction experiment, not an input to control.",
                s["body"],
            ),
            Paragraph(
                "The proposal’s software-simulation scope is retained. The new comparisons "
                "and duration outcomes close the identified coverage gaps without changing the "
                "frozen historical evidence. Representative utility data, electrical-network "
                "validation, a full-horizon scheduling optimum and independent field evaluation "
                "remain future research, not capabilities established by this simulation.",
                s["callout"],
            ),
        ]
    return flow


def markdown_flowables(markdown: str, s: dict[str, ParagraphStyle]) -> list[Flowable]:
    flow = original_markdown(markdown, s)
    if (
        markdown.startswith("# Methodology")
        or markdown.startswith("# Implementation")
        or markdown.startswith("# Results")
        or markdown.startswith("# Related Work")
    ):
        for item in flow:
            if isinstance(item, Paragraph) and item.style.name == "Chapter":
                item.__init__("Original study: " + item.getPlainText().lower(), s["chapter"])
                break
        position = next(i for i, item in enumerate(flow) if isinstance(item, Paragraph)) + 1
        flow.insert(
            position,
            Paragraph(
                "The original study is retained here with its own evidence "
                "and implementation scope. Current additions are defined separately in the "
                "proposal extension chapters; historical statements about absent comparisons "
                "do not describe "
                "the expanded study.",
                s["callout"],
            ),
        )
    flow = [KeepTogether([item]) if isinstance(item, Preformatted) else item for item in flow]
    if markdown.startswith("# Results"):
        for path in ("docs/PROPOSAL_EXTENSION.md", "docs/PROPOSAL_RESULTS.md"):
            flow += original_markdown((ROOT / path).read_text(encoding="utf-8"), s)
    return flow


def editorial_notify(self: report.ReportDocument, kind: str, stuff: tuple) -> Any:
    # The printed contents lists chapters; the PDF outline retains every heading.
    if kind == "TOCEntry" and stuff[0] > 0:
        return
    return original_notify(self, kind, stuff)


def styles() -> dict[str, ParagraphStyle]:
    s = original_styles()
    s["cover_title"].fontSize = 31
    s["cover_title"].leading = 35
    s["cover_title"].textColor = DEEP
    s["cover_meta"].textColor = DEEP
    s["cover_subtitle"].fontSize = 12
    s["cover_subtitle"].leading = 16
    s["chapter"].fontSize = 26
    s["chapter"].leading = 30
    s["chapter"].spaceAfter = 17
    s["section"].textColor = TEAL
    s["body"].allowWidows = 0
    s["body"].allowOrphans = 0
    s["subsection"].textColor = DEEP
    s["callout"].borderWidth = 0
    s["callout"].borderPadding = 8
    s["callout"].fontName = F.body
    s["callout"].fontSize = 9
    s["callout"].leading = 12.5
    s["callout"].spaceBefore = 8
    s["callout"].spaceAfter = 10
    s["caption"].alignment = 0
    s["caption"].spaceBefore = 8
    s["caption"].spaceAfter = 14
    s["kicker"] = ParagraphStyle(
        "Kicker", fontName=F.body_bold, fontSize=9, leading=12, textColor=TEAL, spaceAfter=10
    )
    s["deck"] = ParagraphStyle(
        "Deck", fontName=F.body, fontSize=13, leading=18, textColor=MUTED, spaceAfter=16
    )
    s["alignment_body"] = ParagraphStyle(
        "AlignmentBody", parent=s["body"], fontSize=10.5, leading=15, spaceAfter=9
    )
    return s


def draw_page(self: report.ReportDocument, canvas: Any, document: report.ReportDocument) -> None:
    canvas.saveState()
    width, height = A4
    canvas.setFillColor(PALE if document.page == 1 else WHITE)
    canvas.rect(0, 0, width, height, fill=1, stroke=0)
    if document.page == 1:
        # Two abstract river lines echo the app's Khartoum map without implying topology.
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(2)
        for x, dx in ((410, 85), (485, -40)):
            path = canvas.beginPath()
            path.moveTo(x, 0)
            path.curveTo(x - 35, 70, x + dx, 108, 455, 180)
            canvas.drawPath(path)
        canvas.setFillColor(CORAL)
        canvas.circle(455, 180, 4, stroke=0, fill=1)
        canvas.setFillColor(TEAL)
        canvas.roundRect(18 * mm, height - 17 * mm, 27, 5, 2.5, fill=1, stroke=0)
    else:
        canvas.setFillColor(PALE)
        canvas.rect(0, height - 15 * mm, width, 15 * mm, fill=1, stroke=0)
        canvas.setFont(F.body_bold, 7.5)
        canvas.setFillColor(TEAL)
        canvas.drawString(18 * mm, height - 10 * mm, "GRIDGUARD KHARTOUM")
        canvas.setFont(F.body, 7)
        canvas.setFillColor(MUTED)
        canvas.drawRightString(width - 18 * mm, height - 10 * mm, "GRADUATION PROJECT  /  THESIS")
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.5)
        canvas.line(18 * mm, 16 * mm, width - 18 * mm, 16 * mm)
        canvas.setFont(F.body, 7)
        canvas.drawString(18 * mm, 11 * mm, "Design revision · 22 October 2026")
        canvas.setFont(F.body_bold, 8)
        canvas.setFillColor(DEEP)
        canvas.drawRightString(width - 18 * mm, 11 * mm, f"{document.page:02}")
    canvas.restoreState()


def cover(s: dict[str, ParagraphStyle]) -> list[Flowable]:
    return [
        Spacer(1, 17 * mm),
        Paragraph("GRIDGUARD KHARTOUM", s["cover_subtitle"]),
        Spacer(1, 9 * mm),
        Paragraph(
            "AI-Based Smart Load Shedding and Energy Management System for Developing Countries",
            s["cover_title"],
        ),
        Spacer(1, 7 * mm),
        Paragraph(
            (
                "A Khartoum simulation study<br/>Prediction, scheduling, optimization"
                " and reinforcement learning"
            ),
            s["deck"],
        ),
        Spacer(1, 7 * mm),
        Paragraph("GRADUATION PROJECT FINAL REPORT", s["kicker"]),
        Paragraph(
            (
                "Murtada Asim Ibrahim Bakhit (18-317)<br/>Suliman Elawad Suliman "
                "Babiker (18-310)<br/>Abdalhay Abass Mobarak Abass (15-306)"
            ),
            s["cover_meta"],
        ),
        Spacer(1, 13 * mm),
        Paragraph(
            "August 2026<br/>Design and proposal-alignment revision · 22 October 2026",
            s["cover_meta"],
        ),
        Spacer(1, 7 * mm),
        Paragraph("Synthetic scenarios · Offline decision support", s["kicker"]),
        PageBreak(),
    ]


def table_flowable(rows: list[list[str]], s: dict[str, ParagraphStyle]) -> LongTable:
    width = A4[0] - 36 * mm
    weights = [min(32, max(12, max(len(row[i]) for row in rows))) for i in range(len(rows[0]))]
    widths = [width * w / sum(weights) for w in weights]
    label_widths = {"Controller": 76, "Shortage": 57}
    fixed_widths = {
        index: max(widths[index], label_widths[label])
        for index, label in enumerate(rows[0])
        if label in label_widths
    }
    if fixed_widths:
        remaining = sum(value for i, value in enumerate(widths) if i not in fixed_widths)
        available = width - sum(fixed_widths.values())
        widths = [
            fixed_widths[i] if i in fixed_widths else value * available / remaining
            for i, value in enumerate(widths)
        ]
    data = [
        [
            Paragraph(report._inline_markup(c), s["table_header"] if r == 0 else s["small"])
            for c in row
        ]
        for r, row in enumerate(rows)
    ]
    table = LongTable(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), TEAL),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PALE]),
                ("LINEBELOW", (0, 0), (-1, 0), 1, TEAL),
                ("LINEBELOW", (0, 1), (-1, -1), 0.35, RULE),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def results_chart(evidence: dict) -> Drawing:
    fixed, ppo = (evidence["controller_means"][k] for k in ("Fixed Rotation", "PPO"))
    d = Drawing(493, 158)
    object.__setattr__(d, "initialFontName", F.body)
    metrics = [
        (
            "Critical outage",
            "locality-hours · lower is better",
            "critical_outage_locality_hours",
            168,
            1,
        ),
        ("Jain fairness", "% · higher is better", "fairness", 1, 2),
        ("Utilization", "% · higher is better", "utilization", 1, 2),
    ]
    for x, label, color in ((264, "Fixed Rotation", MUTED), (381, "PPO", TEAL)):
        d.add(Rect(x, 146, 7, 7, fillColor=color, strokeColor=None))
        d.add(String(x + 12, 146, label, fontName=F.body, fontSize=8, fillColor=MUTED))
    for i, (label, unit, key, scale, precision) in enumerate(metrics):
        y = 112 - i * 45
        d.add(String(0, y + 11, label, fontName=F.body_bold, fontSize=10, fillColor=DEEP))
        d.add(String(0, y - 1, unit, fontName=F.body, fontSize=7.5, fillColor=MUTED))
        for j, (values, color) in enumerate(((fixed, MUTED), (ppo, TEAL))):
            yy = y + 12 - j * 15
            d.add(Rect(181, yy, 244, 7, rx=3.5, ry=3.5, fillColor=PALE, strokeColor=None))
            d.add(
                Rect(
                    181,
                    yy,
                    244 * values[key] / scale,
                    7,
                    rx=3.5,
                    ry=3.5,
                    fillColor=color,
                    strokeColor=None,
                )
            )
            val = values[key] * (100 if scale == 1 else 1)
            d.add(
                String(
                    490,
                    yy,
                    f"{val:.{precision}f}" + ("%" if scale == 1 else ""),
                    textAnchor="end",
                    fontName=F.body_bold,
                    fontSize=9,
                    fillColor=color,
                )
            )
    return d


def completed_alignment(s: dict[str, ParagraphStyle]) -> list[Flowable]:
    body = s["alignment_body"]
    return [
        PageBreak(),
        Paragraph("PROPOSAL REVIEW  /  01", s["kicker"]),
        Paragraph("Proposal alignment", s["chapter"]),
        Paragraph(
            "Added methods and direct duration outcomes complete the identified coverage.",
            s["deck"],
        ),
        Paragraph(
            "The accepted proposal defines a simplified AI decision-support simulation, "
            "with Sudan as its main case study and no direct national-grid control [1, pp. 1–3]. "
            "The expanded GridGuard project maps all seven objectives to implemented work below.",
            body,
        ),
        table_flowable(
            [
                ["Proposal objective [1, p. 2]", "Evidence in the expanded project"],
                [
                    "1. Study shortages, with a Sudan focus",
                    "Original Introduction, Related Work and Background; Khartoum locality "
                    "case study.",
                ],
                [
                    "2. Model supply, demand, shortage levels and consumer sectors",
                    (
                        "Seven localities, five sectors, 24-hour demand/supply traces, three "
                        "shortage levels; original Methodology."
                    ),
                ],
                [
                    "3. Develop an AI decision-support model",
                    (
                        "Frozen PPO actor, CUDA simulation and an offline desktop interface; "
                        "original Implementation and extension."
                    ),
                ],
                [
                    "4. Prioritize critical infrastructure",
                    (
                        "Hospitals, water and bakeries represented by the priority-sector "
                        "model; critical service and interruption outcomes."
                    ),
                ],
                [
                    "5. Test prediction, optimization, rules and RL where appropriate",
                    (
                        "Learned ridge AR(3) forecast; exhaustive one-step Optimization; "
                        "Fixed Rotation rules; PPO reinforcement learning."
                    ),
                ],
                [
                    "6. Compare with fixed scheduling",
                    (
                        "All three controllers evaluated on the same 960 additional scenario "
                        "conditions; 2,880 runs, with the original comparisons retained."
                    ),
                ],
                [
                    "7. Evaluate downtime, fairness and efficiency",
                    (
                        "Critical interruption burden, direct blackout/interruption durations"
                        " and disparity, Jain service fairness and utilization."
                    ),
                ],
            ],
            s,
        ),
        Spacer(1, 10),
        Paragraph("Deliverables and scope", s["section"]),
        Paragraph(
            "All seven deliverable categories are represented: documented simulation, AI model, "
            "scenario set, fixed-schedule comparison, evaluation results, final report and "
            "academic "
            "presentation [1, p. 3]. The additions extend the scientific evidence and the current "
            "desktop implementation while preserving the frozen historical artifacts.",
            body,
        ),
        Paragraph(
            "Source: accepted graduation proposal [1]. The proposal permits a simplified "
            "simulation; utility deployment, real-grid control and a guaranteed favorable result "
            "are not claimed as delivered capabilities.",
            s["small"],
        ),
        PageBreak(),
        Paragraph("PROPOSAL REVIEW  /  02", s["kicker"]),
        Paragraph("Added evidence", s["chapter"]),
        Paragraph(
            (
                "Each previously identified gap now has an implementation, a metric "
                "or a measured comparison."
            ),
            s["deck"],
        ),
        Paragraph("01  Explicit dispatch optimization", s["section"]),
        Paragraph(
            "Optimization enumerates all 2,187 joint service-tier actions at each hour, "
            "using the same dispatch rule and canonical current-hour objective as the simulator. "
            "It uses causal physical state and a deterministic tie rule. This supplies a direct "
            "optimization comparator; it is distinct from the optimizer used to train PPO.",
            body,
        ),
        Paragraph("02  Learned prediction", s["section"]),
        Paragraph(
            "A ridge AR(3) model is fitted on 768 newly generated traces from documented "
            "training seeds. It is tested against persistence and the three-hour trailing mean "
            "on the same 320 separate demand traces and identical one-hour targets. No evaluation "
            "targets enter fitting or parameter selection.",
            body,
        ),
        Paragraph("03  Blackout duration and inequality", s["section"]),
        Paragraph(
            "Research question 4 is now assessed directly through hours of zero delivered "
            "service and critical-service interruption for each locality, duration range, mean "
            "pairwise hour gap, and the longest consecutive interruption. These supplement "
            "the original Jain service-allocation metric instead of substituting for it.",
            body,
        ),
        Paragraph("04  Matched comparison and honest interpretation", s["section"]),
        Paragraph(
            "The new protocol uses seeds 91001–91010 across Mild, Moderate and Severe "
            "shortages. It reports all three controller comparisons, paired uncertainty and "
            "forecast errors. Optimization leads on critical interruption and utilization; "
            "PPO leads on Jain service fairness. The zero duration gaps for both adaptive "
            "controllers partly reflect the simulator’s shared proportional clipping.",
            body,
        ),
        Paragraph(
            "Scientific boundaries remain explicit: synthetic inputs, a single global "
            "supply pool, one-step rather than full-horizon optimization, and no independent "
            "field validation. These boundaries preserve the proposal’s simulation scope; "
            "they do not remove any of its requested method or outcome categories.",
            s["callout"],
        ),
    ]


def architecture_diagram() -> Drawing:
    """Show the shared dispatch inputs; forecast assessment remains separate."""
    drawing = Drawing(500, 92)
    object.__setattr__(drawing, "initialFontName", F.body)
    labels = (
        ("Seeded", "scenarios"),
        ("Shared", "scenario tensors"),
        ("Fixed Rotation", "PPO", "Optimization"),
        ("Paired", "evidence"),
        ("Desktop", "replay"),
    )
    for index, lines in enumerate(labels):
        x = index * 101
        drawing.add(Rect(x, 30, 86, 44, rx=5, ry=5, fillColor=TEAL if index in (2, 4) else DEEP))
        for line, text in enumerate(lines):
            drawing.add(
                String(
                    x + 43,
                    58 - line * 10,
                    text,
                    textAnchor="middle",
                    fontName=F.body_bold,
                    fontSize=7,
                    fillColor=WHITE,
                )
            )
        if index < len(labels) - 1:
            drawing.add(
                String(x + 89, 47, "->", fontName=F.body_bold, fontSize=11, fillColor=CORAL)
            )
    return drawing


def front_matter(s: dict[str, ParagraphStyle], evidence: dict, markdown: str) -> list[Flowable]:
    retained = original_front_matter(s, evidence, markdown)
    for index, item in enumerate(retained):
        if isinstance(item, Paragraph) and item.getPlainText().startswith(
            "One severity-neutral base-draw batch"
        ):
            retained[index] = Paragraph(
                "One severity-neutral base-draw batch is reused across all three shortage "
                "levels. The same resulting scenario tensors are passed to Fixed Rotation, "
                "PPO and Optimization. The original frozen benchmark and the additional "
                "three-controller evaluation retain separate evidence records. Simulation, "
                "inference, metrics and bootstrap operations stay on the GPU; only final "
                "display and evidence payloads cross to host memory. Startup verifies the "
                "GPU runtime, evidence and PPO artifact before opening the desktop.",
                s["body"],
            )
            retained.insert(
                index + 1,
                Paragraph(
                    "Learned demand prediction is assessed separately: ridge AR(3), "
                    "persistence and the trailing mean share the same held-out one-hour "
                    "targets. These forecasts do not enter controller decisions.",
                    s["body"],
                ),
            )
            break
    retained[0] = Paragraph("Original study abstract", s["chapter"])
    retained.insert(
        1,
        Paragraph(
            "The following abstract and summary retain the original "
            "two-controller study. The new methods, seeds and outcomes are separate additions "
            "described in the current abstract and proposal extension chapters.",
            s["callout"],
        ),
    )
    means = load_proposal_supplement()["controller_means"]
    rows = [["New evaluation outcome", "Fixed Rotation", "PPO", "Optimization"]]
    for label, key, percent in [
        ("Critical interruption (locality-hours)", "critical_outage_locality_hours", False),
        ("Jain service fairness", "fairness", True),
        ("Supply utilization", "utilization", True),
    ]:
        rows.append(
            [
                label,
                *[
                    (f"{means[name][key]:.2%}" if percent else f"{means[name][key]:.2f}")
                    for name in ("Fixed Rotation", "PPO", "Optimization")
                ],
            ]
        )
    flow = [
        Paragraph("Abstract", s["chapter"]),
        Paragraph(
            "GridGuard Khartoum investigates AI-assisted load shedding in a simplified, "
            "offline simulation of seven Khartoum localities and five service sectors. "
            "The expanded study evaluates a frozen PPO controller, Fixed Rotation and an "
            "explicit one-step Optimization comparator on identical synthetic shortage scenarios. "
            "It adds learned demand prediction and direct measures of blackout duration and "
            "inequality to the original service-allocation outcomes.",
            s["body"],
        ),
        Paragraph(
            "The additional experiment fixes ten evaluation seeds (91001–91010), with "
            "320 independent base trajectories crossed with three shortage levels: 960 matched "
            "conditions and 2,880 controller evaluations. All 2,187 joint service-tier actions "
            "are considered by Optimization at each hour, with no future demand or supply. "
            "Uncertainty uses paired base-trajectory resampling within seed, keeping shortage "
            "variants together. Intervals are pointwise 95% and unadjusted.",
            s["body"],
        ),
        table_flowable(rows, s),
        Spacer(1, 12),
        Paragraph(
            "Optimization records lower critical-interruption burden and higher utilization; "
            "PPO records higher Jain service fairness. Both have zero zero-service blackout "
            "hours in the new sample. Direct duration inequalities are also zero for both, "
            "partly because shared proportional clipping interrupts critical loads across all "
            "localities together. The distinction between interruption burden and its distribution "
            "is therefore essential.",
            s["body"],
        ),
        Paragraph(
            "A causal ridge AR(3) predictor fitted on 768 separate synthetic training "
            "traces reduces system-total forecast MAE from 24.38 MW for persistence to 9.34 MW. "
            "Forecasts are evaluated independently from control. Original reproduction and "
            "81001–81010 holdout results remain unchanged and are retained in this report.",
            s["body"],
        ),
        Paragraph(
            "The work now covers the proposal’s selected prediction, optimization, "
            "rule-based and reinforcement-learning methods and directly assesses blackout "
            "duration fairness. Every result concerns synthetic shortage scenarios; none "
            "establishes real-grid deployment performance or operational safety.",
            s["callout"],
        ),
        PageBreak(),
        *retained,
    ]
    for item in flow:
        if isinstance(item, report.TableOfContents):
            item.levelStyles[0].fontSize = 12
            item.levelStyles[0].leading = 24
    return [*flow, *completed_alignment(s)]


def screenshots(s: dict[str, ParagraphStyle]) -> list[Flowable]:
    descriptions = [
        (
            "Understand the study area",
            "Overview",
            (
                "The Overview page introduces the Khartoum locality map and the "
                "study’s seven localities, 24-hour horizon and five service sectors. "
                "The horizontal navigation, open canvas and floating locality details"
                " keep the geographic context visible."
            ),
            (
                "The map is administrative context. Its shapes and river layers do "
                "not determine demand, network constraints or controller decisions."
            ),
        ),
        (
            "Explore a matched scenario",
            "Live scenario",
            (
                "The Live Scenario page provides shortage and seed controls, "
                "three-controller selection, timeline playback and locality "
                "inspection. A completed 24-hour trace supports direct exploration of"
                " PPO, Fixed Rotation and Optimization on the same generated inputs."
            ),
            (
                "This capture shows an illustrative live trace. Its values must not "
                "replace frozen benchmark or holdout outcomes. The final hour has no "
                "next-hour forecast target."
            ),
        ),
        (
            "Read the verified comparison",
            "Verified results",
            (
                "The Results page presents the documented-seed-disjoint holdout with "
                "aligned comparison bars, clear units and concise outcome labels. "
                "Supporting evidence details retain the full study limitations and "
                "provenance."
            ),
            (
                "Critical-outage locality-hours, Jain fairness and utilization are "
                "separate outcomes. The paired differences in the report come from "
                "full-precision evidence, not subtraction of rounded display labels."
            ),
        ),
    ]
    flow = []
    for i, ((caption, path), (title, short, description, note)) in enumerate(
        zip(report.SCREENSHOT_PATHS, descriptions, strict=True)
    ):
        flow += [PageBreak()]
        if i == 0:
            flow += [Paragraph("Desktop presentation", s["chapter"])]
        flow += [
            Paragraph(f"INTERFACE  /  {i + 1:02}  /  {short.upper()}", s["kicker"]),
            Paragraph(title, s["section"]),
            Paragraph(description, s["body"]),
            Spacer(1, 10),
        ]
        im = Image(str(path))
        ratio = (A4[0] - 36 * mm) / im.imageWidth
        im.drawWidth = im.imageWidth * ratio
        im.drawHeight = im.imageHeight * ratio
        flow += [
            im,
            Paragraph(
                f"Figure {i + 1}. {caption}. Actual current desktop application "
                "capture at 1366 × 768.",
                s["caption"],
            ),
            Paragraph(note, s["callout"]),
        ]
    return flow


def _configure_report() -> None:
    report.CANVAS = PALE
    report.SURFACE = report.OFF_WHITE = WHITE
    report.INK = report.NAV = report.NAVY = report.SLATE = DEEP
    report.TEAL = report.NILE = TEAL
    report.GOLD = report.EVIDENCE = CORAL
    report.MUTED = MUTED
    report.GRID = RULE
    report.REPORT_FONTS = report.REPORT_FONTS._replace(
        display="GridGuardAtkinson", display_bold="GridGuardAtkinsonBold"
    )
    report._styles = styles
    report.markdown_section_text_flowables = section_flowables
    report.markdown_text_flowables = markdown_flowables
    report._cover = cover
    report.ReportDocument._draw_page = draw_page
    report.ReportDocument.notify = editorial_notify
    report._table_flowable = table_flowable
    report._results_chart = results_chart
    report._architecture_diagram = architecture_diagram
    report._front_matter = front_matter
    report._screenshots = screenshots
    report.SCREENSHOT_PATHS = tuple(
        (label, ROOT / "docs/assets/extension-screenshots" / name)
        for label, name in (
            ("Overview", "overview.png"),
            ("Live scenario", "live.png"),
            ("Verified results", "results.png"),
        )
    )


def sha256(path: Path) -> str:
    """Return the digest recorded in the separate document-build receipt."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_extended_report(output: Path | None = None) -> Path:
    """Build the review edition without changing historical release artifacts."""
    output = output or ROOT / "output/pdf/updated/gridguard-khartoum-thesis.pdf"
    output = output.resolve()
    bound = [
        ROOT / path
        for path in (
            "output/pdf/gridguard-khartoum-final-report.pdf",
            "output/presentation/gridguard-khartoum-defense.pptx",
            "docs/release-visual-review.json",
            "docs/release-machine-validation.json",
        )
    ]
    if output in [path.resolve() for path in bound]:
        raise ValueError("The extended report must not overwrite a release-bound artifact")
    before = {str(path.relative_to(ROOT)): sha256(path) for path in bound}
    _configure_report()
    output.parent.mkdir(parents=True, exist_ok=True)
    report.build_report(output)
    if before != {str(path.relative_to(ROOT)): sha256(path) for path in bound}:
        raise RuntimeError("A release-bound artifact changed during the extended report build")
    receipt = ROOT / "tmp/thesis-refresh/restyle-receipt.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(
        json.dumps(
            {
                "output": str(output),
                "sha256": sha256(output),
                "release_bound_unchanged": before,
                "proposal_sha256": sha256(ROOT / "docs/source/graduation-project-proposal.pdf"),
                "authoring_script_sha256": sha256(Path(__file__)),
                "design": (
                    "Approved GUI palette and Atkinson typography; proposal alignment review added"
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def main() -> None:
    output = build_extended_report()
    print(output)
    print("Canonical evidence and source contracts verified; release-bound artifacts unchanged.")


if __name__ == "__main__":
    main()

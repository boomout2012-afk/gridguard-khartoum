"""Methodology page tied to the paired-evidence protocol."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from gridguard.ui.components import CardFrame, page_intro
from gridguard.ui.presentation_model import PresentationSnapshot


class MethodologyPage(QWidget):
    """Scrollable plain-language account of the implemented comparison method."""

    def __init__(self, snapshot: PresentationSnapshot, parent=None) -> None:
        super().__init__(parent)
        self.setAccessibleName("Methodology page")
        self.setAccessibleDescription(
            "Scrollable methodology covering synthetic inputs, controllers, the diagnostic "
            "training objective, untouched evaluation, provenance, and limitations."
        )
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self.scroll_area = QScrollArea()
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setAccessibleName("Scrollable methodology content")
        self.scroll_area.setAccessibleDescription(
            "Use the mouse wheel, scroll bar, Page Up, or Page Down to read all sections."
        )

        content_widget = QWidget()
        content_widget.setAccessibleName("Complete methodology article")
        layout = QVBoxLayout(content_widget)
        layout.setContentsMargins(26, 18, 18, 24)
        layout.setSpacing(10)

        kicker = QLabel("METHODOLOGY  /  PROPOSAL-ALIGNED")
        kicker.setProperty("kicker", True)
        kicker.setAccessibleName("Methodology, proposal-aligned")
        layout.addWidget(kicker)
        title, description = page_intro(
            "How the comparison was produced",
            "A transparent path from synthetic shortage conditions to identical-controller "
            "inputs, CUDA execution, untouched paired inference, and a deliberately narrow claim.",
        )
        layout.addWidget(title)
        layout.addWidget(description)

        protocol = CardFrame(
            accessible_name="Matched evaluation protocol at a glance",
            variant="accent",
        )
        protocol_header = QHBoxLayout()
        protocol_title = QLabel("Study boundary")
        protocol_title.setProperty("heading", "section")
        protocol_badge = QLabel("MATCHED · FROZEN")
        protocol_badge.setProperty("badge", "locked")
        protocol_badge.setProperty("compact", True)
        protocol_header.addWidget(protocol_title)
        protocol_header.addStretch(1)
        protocol_header.addWidget(protocol_badge)
        protocol_text = QLabel(
            f"{snapshot.independent_base_draw_count} independent base draws crossed with "
            f"{snapshot.shortage_level_count} shortage levels produce "
            f"{snapshot.matched_scenario_count} matched {snapshot.horizon_hours}-hour "
            f"scenario conditions and {snapshot.controller_evaluation_count} controller "
            "evaluations. Each pair receives the same demand and supply trace."
        )
        protocol_text.setWordWrap(True)
        protocol_text.setProperty("muted", True)
        protocol_text.setAccessibleName(f"Verified protocol: {protocol_text.text()}")
        protocol.content.addLayout(protocol_header)
        protocol.content.addWidget(protocol_text)
        protocol.setAccessibleDescription(protocol_text.text())
        layout.addWidget(protocol)

        sections = (
            (
                "01",
                "Synthetic scenario generation",
                "Each episode uses deterministic, seeded, 24-hour synthetic demand and "
                "available-supply traces for Mild, Moderate, or Severe shortage. The model "
                "uses one abstract supply pool; it does not model feeders or power flow.",
            ),
            (
                "02",
                "Service tiers and critical-service priorities",
                "At every hour, each locality receives Off, Critical Only, or Full Service. "
                "Critical Only requests Hospital, Water, and Bakery demand; Full Service also "
                "requests Residential and Commercial demand. Dispatch never exceeds supply.",
            ),
            (
                "03",
                "Fixed Rotation",
                "The scheduled baseline works in three-hour blocks. Three consecutive "
                "localities receive Full Service while the others are Off, then the starting "
                "position advances. It does not observe demand, supply, or fairness.",
            ),
            (
                "04",
                "PPO",
                "The frozen learned controller observes the current hour, supply-to-demand "
                "context, locality demand, critical share, service history, and prior tier. "
                "Deterministic argmax inference selects one joint seven-locality action.",
            ),
            (
                "05",
                "Training-objective score (diagnostic)",
                "The accepted hourly reward gives greatest weight to critical service, then "
                "rewards utilization and cumulative locality fairness while penalizing tier "
                "switching and requested-power oversubscription. It is the PPO training-objective "
                "score and remains diagnostic, not an independent primary outcome or economic "
                "valuation.",
            ),
            (
                "06",
                "Untouched confirmatory evidence",
                f"Matched evaluation gives PPO and Fixed Rotation the same immutable scenario in "
                f"every pair. The "
                f"{snapshot.matched_scenario_count} paired conditions are clustered into "
                f"{snapshot.independent_base_draw_count} untouched independent base draws across "
                f"{snapshot.shortage_level_count} linked shortage variants. Results are reported "
                "as observed; there is no favorable/adverse release gate.",
            ),
            (
                "07",
                "Metrics and paired analysis",
                "The three primary outcomes are critical-outage locality-hours, Jain fairness, "
                "and utilization. Every stored difference is PPO minus Fixed Rotation; lower is "
                "beneficial for outage exposure and higher for fairness and utilization. The 95% "
                "intervals keep each base draw's shortage variants together during clustered "
                "paired bootstrap. Power balance is shown separately as a safety result.",
            ),
            (
                "08",
                "Evidence provenance",
                f"The Results page reads {snapshot.artifact_id} only after startup verifies the "
                "committed robustness supplement SHA-256, schema, controller labels, protocol "
                "counts, aggregate values, and scientific invariants. A live scenario cannot "
                "overwrite this frozen evidence.",
            ),
        )
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        self.method_cards: list[CardFrame] = []
        for index, (number, heading, body) in enumerate(sections):
            card = CardFrame(accessible_name=f"Method {number}: {heading}")
            heading_row = QHBoxLayout()
            number_label = QLabel(number)
            number_label.setProperty("badge", "neutral")
            number_label.setProperty("compact", True)
            heading_label = QLabel(heading)
            heading_label.setProperty("heading", "section")
            heading_label.setWordWrap(True)
            heading_row.addWidget(number_label)
            heading_row.addWidget(heading_label, 1)
            body_label = QLabel(body)
            body_label.setWordWrap(True)
            body_label.setProperty("muted", True)
            body_label.setAccessibleName(f"{heading}: {body}")
            card.content.addLayout(heading_row)
            card.content.addWidget(body_label)
            card.setAccessibleDescription(body)
            self.method_cards.append(card)
            grid.addWidget(card, index // 2, index % 2)
        layout.addLayout(grid)

        limits = CardFrame(accessible_name="Evidence limitations", variant="quiet")
        limit_heading = QLabel("Limitations and claim boundary")
        limit_heading.setProperty("heading", "section")
        limits.content.addWidget(limit_heading)
        if snapshot.limitations:
            for limitation in snapshot.limitations:
                label = QLabel(f"• {limitation}")
                label.setWordWrap(True)
                label.setProperty("muted", True)
                label.setAccessibleName(f"Limitation: {limitation}")
                limits.content.addWidget(label)
        else:
            missing = QLabel("No limitation text was supplied by the verified snapshot.")
            missing.setProperty("state", "error")
            missing.setWordWrap(True)
            missing.setAccessibleName("Evidence limitation metadata is unavailable")
            limits.content.addWidget(missing)
        boundary = QLabel(
            "Decision support only: the study does not establish real-grid operational "
            "readiness, field causality, or superiority over every alternative controller."
        )
        boundary.setWordWrap(True)
        boundary.setProperty("muted", True)
        boundary.setAccessibleName(f"Claim boundary: {boundary.text()}")
        limits.content.addWidget(boundary)
        layout.addWidget(limits)

        keyboard = CardFrame(accessible_name="Presenter keyboard guide")
        keyboard_heading = QLabel("Keyboard and interaction guide")
        keyboard_heading.setProperty("heading", "section")
        keyboard_text = QLabel(
            "Keyboard: Ctrl+1 through Ctrl+4 changes pages; F11 toggles fullscreen. On the "
            "map, arrow keys change locality, plus or minus changes zoom, and R or 0 resets "
            "the view. Tab moves through controls and the results table."
        )
        keyboard_text.setWordWrap(True)
        keyboard_text.setProperty("muted", True)
        keyboard_text.setAccessibleName(keyboard_text.text())
        keyboard.content.addWidget(keyboard_heading)
        keyboard.content.addWidget(keyboard_text)
        layout.addWidget(keyboard)

        self.scroll_area.setWidget(content_widget)
        outer.addWidget(self.scroll_area)

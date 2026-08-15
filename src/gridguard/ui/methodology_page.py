"""Methodology page tied to the paired-evidence protocol."""

from __future__ import annotations

from PySide6.QtWidgets import QGridLayout, QLabel, QVBoxLayout, QWidget

from gridguard.ui.components import CardFrame, page_intro
from gridguard.ui.presentation_model import PresentationSnapshot


class MethodologyPage(QWidget):
    """Four-stage explanation of the implemented comparison method."""

    def __init__(self, snapshot: PresentationSnapshot, parent=None) -> None:
        super().__init__(parent)
        self.setAccessibleName("Methodology page")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 18)
        layout.setSpacing(14)
        title, description = page_intro(
            "How the comparison was produced",
            "The presentation separates the learned dispatch policy, the scheduled "
            "baseline, simulation execution, and paired statistical analysis.",
        )
        layout.addWidget(title)
        layout.addWidget(description)

        stages = (
            (
                "01 · Matched conditions",
                f"{snapshot.independent_base_draw_count} independent base draws are each "
                f"evaluated under {snapshot.shortage_level_count} shortage levels, yielding "
                f"{snapshot.matched_scenario_count} matched 24-hour conditions.",
            ),
            (
                "02 · Same inputs",
                "PPO and Fixed Rotation receive the same demand and available-supply trace "
                "for every comparison.",
            ),
            (
                "03 · GPU execution",
                f"{snapshot.controller_evaluation_count} controller-scenario evaluations keep "
                "simulation tensors and PPO inference on CUDA.",
            ),
            (
                "04 · Paired analysis",
                "Each delta is PPO minus Fixed Rotation. The results page reports deterministic "
                "clustered paired-bootstrap 95% intervals that keep each base draw's three "
                "shortage conditions together.",
            ),
        )
        grid = QGridLayout()
        grid.setSpacing(12)
        for index, (heading, body) in enumerate(stages):
            card = CardFrame(accessible_name=heading)
            heading_label = QLabel(heading)
            heading_label.setProperty("heading", "section")
            body_label = QLabel(body)
            body_label.setWordWrap(True)
            body_label.setProperty("muted", True)
            card.content.addWidget(heading_label)
            card.content.addWidget(body_label)
            grid.addWidget(card, index // 2, index % 2)
        layout.addLayout(grid)

        limits = CardFrame(accessible_name="Evidence limitations")
        limit_heading = QLabel("What this evidence does—and does not—claim")
        limit_heading.setProperty("heading", "section")
        limits.content.addWidget(limit_heading)
        for limitation in snapshot.limitations:
            label = QLabel(f"• {limitation}")
            label.setWordWrap(True)
            label.setProperty("muted", True)
            limits.content.addWidget(label)
        layout.addWidget(limits)
        layout.addStretch(1)

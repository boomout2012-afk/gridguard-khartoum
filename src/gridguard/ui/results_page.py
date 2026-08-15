"""Evidence-backed paired comparison page."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gridguard.ui.components import CardFrame, ControllerCard, format_metric_value, page_intro
from gridguard.ui.presentation_model import PresentationSnapshot
from gridguard.ui.theme import MONO_FONT_FAMILY


class ResultsPage(QWidget):
    """Controller cards and paired confidence intervals from committed evidence."""

    def __init__(self, snapshot: PresentationSnapshot, parent=None) -> None:
        super().__init__(parent)
        self.setAccessibleName("Results page")
        self.setAccessibleDescription(
            f"Frozen paired comparison from evidence {snapshot.benchmark_id}."
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(26, 18, 26, 16)
        layout.setSpacing(9)

        kicker = QLabel("RESULTS  /  FROZEN BENCHMARK")
        kicker.setProperty("kicker", True)
        kicker.setAccessibleName("Results, frozen benchmark")
        layout.addWidget(kicker)
        title, description = page_intro(
            "PPO compared with Fixed Rotation",
            "Aggregate values and paired 95% confidence intervals come directly from the "
            "committed matched-scenario evidence. Live demonstrations cannot replace them.",
        )
        layout.addWidget(title)
        layout.addWidget(description)

        self.provenance_card = CardFrame(
            accessible_name="Results evidence provenance",
            variant="accent",
        )
        self.provenance_card.setAccessibleDescription(
            f"Hash-verified committed artifact {snapshot.benchmark_id}; "
            f"SHA-256 {snapshot.artifact_sha256}; release {snapshot.release_id}; "
            f"{snapshot.matched_scenario_count} matched scenario conditions and "
            f"{snapshot.controller_evaluation_count} controller evaluations."
        )
        provenance_grid = QGridLayout()
        provenance_grid.setHorizontalSpacing(18)
        provenance_grid.setVerticalSpacing(3)
        provenance_heading = QLabel("Evidence provenance")
        provenance_heading.setProperty("heading", "section")
        self.provenance_status = QLabel("HASH VERIFIED · FROZEN")
        self.provenance_status.setProperty("badge", "verified")
        self.provenance_status.setProperty("compact", True)
        self.provenance_status.setAccessibleName(
            "Evidence hash verified at startup; frozen results"
        )
        artifact_caption = QLabel("Artifact ID")
        artifact_caption.setProperty("muted", True)
        artifact_caption.setProperty("compact", True)
        self.provenance_id = QLabel(snapshot.benchmark_id)
        self.provenance_id.setProperty("compact", True)
        self.provenance_id.setProperty("mono", True)
        self.provenance_id.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByKeyboard)
        self.provenance_id.setAccessibleName(
            f"Verified evidence identifier: {snapshot.benchmark_id}"
        )
        release_caption = QLabel("Release ID")
        release_caption.setProperty("muted", True)
        release_caption.setProperty("compact", True)
        self.provenance_release = QLabel(snapshot.release_id)
        self.provenance_release.setProperty("compact", True)
        self.provenance_release.setProperty("mono", True)
        self.provenance_release.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        self.provenance_release.setToolTip(
            f"Release identifier: {snapshot.release_id}. Select the text to copy."
        )
        self.provenance_release.setAccessibleName(
            f"Verified application release identifier: {snapshot.release_id}"
        )
        sha256_caption = QLabel("SHA-256")
        sha256_caption.setProperty("muted", True)
        sha256_caption.setProperty("compact", True)
        self.provenance_sha256 = QLabel(snapshot.artifact_sha256)
        self.provenance_sha256.setProperty("compact", True)
        self.provenance_sha256.setProperty("mono", True)
        self.provenance_sha256.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        self.provenance_sha256.setToolTip(
            f"Verified paired-benchmark SHA-256: {snapshot.artifact_sha256}. "
            "Select the text to copy."
        )
        self.provenance_sha256.setAccessibleName(
            f"Verified paired-benchmark SHA-256: {snapshot.artifact_sha256}"
        )
        self.provenance_summary = QLabel(snapshot.count_statement)
        self.provenance_summary.setProperty("muted", True)
        self.provenance_summary.setWordWrap(True)
        self.provenance_summary.setAccessibleName(
            f"Verified evidence sample: {snapshot.count_statement}"
        )
        provenance_grid.addWidget(provenance_heading, 0, 0, 1, 3)
        provenance_grid.addWidget(self.provenance_status, 0, 3, Qt.AlignmentFlag.AlignRight)
        provenance_grid.addWidget(artifact_caption, 1, 0)
        provenance_grid.addWidget(self.provenance_id, 1, 1)
        provenance_grid.addWidget(release_caption, 1, 2)
        provenance_grid.addWidget(self.provenance_release, 1, 3)
        provenance_grid.addWidget(sha256_caption, 2, 0)
        provenance_grid.addWidget(self.provenance_sha256, 2, 1, 1, 3)
        provenance_grid.addWidget(self.provenance_summary, 3, 0, 1, 4)
        provenance_grid.setColumnStretch(0, 0)
        provenance_grid.setColumnStretch(1, 1)
        provenance_grid.setColumnStretch(2, 0)
        provenance_grid.setColumnStretch(3, 1)
        self.provenance_card.content.addLayout(provenance_grid)
        layout.addWidget(self.provenance_card)

        card_row = QHBoxLayout()
        card_row.setSpacing(10)
        self.controller_cards = [
            ControllerCard(snapshot, "PPO"),
            ControllerCard(snapshot, "Fixed Rotation"),
        ]
        for card in self.controller_cards:
            card_row.addWidget(card)
        layout.addLayout(card_row)

        ppo_balance = snapshot.ppo_power_balance_violation_rate
        fixed_balance = snapshot.fixed_rotation_power_balance_violation_rate
        if ppo_balance == 0.0 and fixed_balance == 0.0:
            safety_text = "Safety check: 0.00% power-balance violation rate in both controllers."
        else:
            safety_text = (
                "Safety check: power-balance violation rate "
                f"PPO {ppo_balance:.2%}; Fixed Rotation {fixed_balance:.2%}."
            )
        self.safety_note = QLabel(safety_text)
        self.safety_note.setAccessibleName(f"Power-balance safety result: {safety_text}")
        self.safety_note.setProperty("badge", "verified")
        self.safety_note.setProperty("compact", True)
        layout.addWidget(self.safety_note)

        self.results_table = QTableWidget(len(snapshot.metrics), 4)
        self.results_table.setAccessibleName("Paired controller comparison results")
        self.results_table.setAccessibleDescription(
            f"{len(snapshot.metrics)} rows from {snapshot.benchmark_id} compare PPO with "
            "Fixed Rotation. Delta is PPO minus Fixed Rotation and includes a paired "
            "95 percent confidence interval."
        )
        self.results_table.setHorizontalHeaderLabels(
            ["Metric", "PPO", "Fixed Rotation", "PPO − Fixed (95% CI)"]
        )
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.results_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.results_table.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.horizontalHeader().setMinimumSectionSize(125)
        self.results_table.setColumnWidth(0, 190)
        self.results_table.setColumnWidth(1, 145)
        self.results_table.setColumnWidth(2, 165)
        self.results_table.setMinimumHeight(185)
        for row, metric in enumerate(snapshot.metrics):
            delta = format_metric_value(
                metric,
                metric.mean_delta,
                signed=True,
                as_delta=True,
            )
            interval = (
                f"{delta}  "
                f"[{format_metric_value(metric, metric.ci95_lower, as_delta=True)}, "
                f"{format_metric_value(metric, metric.ci95_upper, as_delta=True)}]"
            )
            values = (
                metric.label,
                format_metric_value(metric, metric.ppo),
                format_metric_value(metric, metric.fixed_rotation),
                interval,
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(
                    f"{metric.label}; {self.results_table.horizontalHeaderItem(column).text()}: "
                    f"{value}"
                )
                if column > 0:
                    item.setFont(QFont(MONO_FONT_FAMILY))
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                self.results_table.setItem(row, column, item)
        layout.addWidget(self.results_table, 1)

        limitation = (
            snapshot.limitations[0].rstrip(".")
            if snapshot.limitations
            else "No limitation text was supplied by the verified snapshot"
        )
        self.scope_note = QLabel(
            f"Scope: {limitation}. Decision support only; not field or operational evidence."
        )
        self.scope_note.setAccessibleName(f"Results scope limitation: {self.scope_note.text()}")
        self.scope_note.setProperty("muted", True)
        self.scope_note.setProperty("compact", True)
        self.scope_note.setWordWrap(True)
        layout.addWidget(self.scope_note)

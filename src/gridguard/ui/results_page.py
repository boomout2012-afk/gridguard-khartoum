"""Evidence-backed paired comparison page."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gridguard.ui.components import ControllerCard, format_metric_value, page_intro
from gridguard.ui.presentation_model import PresentationSnapshot


class ResultsPage(QWidget):
    """Controller cards and paired confidence intervals from committed evidence."""

    def __init__(self, snapshot: PresentationSnapshot, parent=None) -> None:
        super().__init__(parent)
        self.setAccessibleName("Results page")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 18)
        layout.setSpacing(12)
        title, description = page_intro(
            "PPO compared with Fixed Rotation",
            "Aggregate values and paired 95% confidence intervals are read directly "
            "from the committed matched-scenario evidence artifact.",
        )
        layout.addWidget(title)
        layout.addWidget(description)

        card_row = QHBoxLayout()
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
        self.safety_note.setAccessibleName("Power-balance safety result")
        self.safety_note.setProperty("badge", "verified")
        self.safety_note.setProperty("compact", True)
        layout.addWidget(self.safety_note)

        self.results_table = QTableWidget(len(snapshot.metrics), 4)
        self.results_table.setAccessibleName("Paired controller comparison results")
        self.results_table.setAccessibleDescription(
            "Rows compare PPO with Fixed Rotation. Delta is PPO minus Fixed Rotation."
        )
        self.results_table.setHorizontalHeaderLabels(
            ["Metric", "PPO", "Fixed Rotation", "PPO − Fixed (95% CI)"]
        )
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.results_table.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.horizontalHeader().setMinimumSectionSize(145)
        self.results_table.setColumnWidth(0, 190)
        self.results_table.setColumnWidth(1, 150)
        self.results_table.setColumnWidth(2, 170)
        for row, metric in enumerate(snapshot.metrics):
            delta = format_metric_value(
                metric,
                metric.mean_delta,
                signed=True,
                as_delta=True,
            )
            values = (
                metric.label,
                format_metric_value(metric, metric.ppo),
                format_metric_value(metric, metric.fixed_rotation),
                (
                    f"{delta}  "
                    f"[{format_metric_value(metric, metric.ci95_lower, as_delta=True)}, "
                    f"{format_metric_value(metric, metric.ci95_upper, as_delta=True)}]"
                ),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column > 0:
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                self.results_table.setItem(row, column, item)
        layout.addWidget(self.results_table, 1)

        note = QLabel(snapshot.count_statement)
        note.setProperty("muted", True)
        note.setWordWrap(True)
        layout.addWidget(note)

        limitation = snapshot.limitations[0].rstrip(".")
        self.scope_note = QLabel(
            f"Scope: {limitation}. Decision support only; not field or operational evidence."
        )
        self.scope_note.setAccessibleName("Results scope limitation")
        self.scope_note.setProperty("muted", True)
        self.scope_note.setProperty("compact", True)
        self.scope_note.setWordWrap(True)
        layout.addWidget(self.scope_note)

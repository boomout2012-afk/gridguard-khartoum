"""Separate follow-up view; every displayed value arrives formatted from the presenter."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QTableWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from gridguard.committee_results import CommitteeResultsError, CommitteeResultsView
from gridguard.focused_results import DisplayTable
from gridguard.resources import ResourceLocationError
from gridguard.ui.results_components import MUTED, PRIMARY, result_label


class FollowupResultsPage(QWidget):
    """Post-hoc baselines and infeasible-floor diagnostics, independent of primary Results."""

    def __init__(
        self,
        loader: Callable[[], CommitteeResultsView],
        table_factory: Callable[[DisplayTable, str], QTableWidget],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setAccessibleName("Post-hoc follow-up comparison")
        self.means_table: QTableWidget | None = None
        self.paired_tables: list[QTableWidget] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(5)
        try:
            self.view = view = loader()
        except (CommitteeResultsError, ResourceLocationError) as error:
            self.error_label = result_label(
                "Follow-up comparison unavailable. No follow-up values are shown because "
                "the evidence or validation receipt could not be verified.",
                size=20,
                wrap=True,
            )
            self.error_label.setToolTip(str(error))
            layout.addWidget(self.error_label)
            layout.addStretch(1)
            return
        layout.addWidget(result_label(view.count_statement, size=14, bold=True, color=PRIMARY))
        layout.addWidget(result_label(view.scope_statement, size=12, color=MUTED, wrap=True))
        self.detail_tabs = QTabWidget()
        self.detail_tabs.setAccessibleName("Follow-up evidence views")

        overview = QWidget()
        overview_layout = QVBoxLayout(overview)
        overview_layout.setContentsMargins(0, 8, 0, 0)
        overview_layout.setSpacing(7)
        overview_layout.addWidget(result_label(view.baseline_statement, size=13, wrap=True))
        self.means_table = table_factory(view.means, "Five-controller follow-up means")
        self.means_table.horizontalHeader().setFixedHeight(47)
        self.means_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.means_table.setFixedHeight(224)
        overview_layout.addWidget(self.means_table)
        overview_layout.addWidget(
            result_label(view.tradeoff_statement, size=13, bold=True, wrap=True)
        )
        overview_layout.addStretch(1)
        self.detail_tabs.addTab(overview, "Controller means")

        paired = QWidget()
        paired_layout = QVBoxLayout(paired)
        paired_layout.setContentsMargins(0, 8, 0, 0)
        paired_layout.addWidget(result_label(view.interval_statement, size=13, wrap=True))
        comparisons = QHBoxLayout()
        comparisons.setSpacing(18)
        for title, data in view.paired:
            column = QVBoxLayout()
            column.addWidget(result_label(title, size=17, bold=True, color=PRIMARY, wrap=True))
            table = table_factory(data, title)
            table.horizontalHeader().setSectionResizeMode(
                0, QHeaderView.ResizeMode.ResizeToContents
            )
            table.horizontalHeader().setSectionResizeMode(
                1, QHeaderView.ResizeMode.ResizeToContents
            )
            table.setFixedHeight(177)
            column.addWidget(table)
            column.addStretch(1)
            self.paired_tables.append(table)
            comparisons.addLayout(column, 1)
        paired_layout.addLayout(comparisons, 1)
        self.detail_tabs.addTab(paired, "Paired differences")

        stress = QWidget()
        stress_layout = QVBoxLayout(stress)
        stress_layout.setContentsMargins(0, 8, 0, 0)
        stress_layout.setSpacing(5)
        stress_layout.addWidget(result_label(view.stress_statement, size=12, wrap=True))
        self.stress_table = table_factory(view.stress, "Infeasible critical-supply stress outcomes")
        self.stress_table.setStyleSheet("QTableWidget {font-size: 12px;}")
        self.stress_table.horizontalHeader().setFixedHeight(49)
        self.stress_table.verticalHeader().setDefaultSectionSize(27)
        self.stress_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self.stress_table.setFixedHeight(320)
        stress_layout.addWidget(self.stress_table)
        stress_layout.addStretch(1)
        self.detail_tabs.addTab(stress, "Critical shortfall")
        layout.addWidget(self.detail_tabs, 1)
        self.setToolTip(
            "Verified follow-up evidence: " + "; ".join(path for path, _ in view.evidence)
        )

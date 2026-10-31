"""Current five-run results in the established native visual language."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from gridguard.focused_results import (
    DisplayTable,
    FocusedStudyError,
    FocusedStudyView,
    load_focused_results,
)
from gridguard.resources import ResourceLocationError
from gridguard.ui.presentation_model import PresentationSnapshot
from gridguard.ui.results_components import INK, LINE, MUTED, PRIMARY, result_label
from gridguard.ui.results_page import ResultsPage

_STYLE = f"""
QWidget#FocusedResultsPage, QDialog {{background: #F4F7F5;}}
QFrame#CurrentOutcomeCard {{background: white; border: none; border-radius: 18px;}}
QPushButton#StudyAction {{background: white; color: {INK}; border: 1px solid {LINE};
    border-radius: 12px; padding: 8px 18px; font-size: 14px; min-height: 26px;}}
QPushButton#StudyAction:hover {{background: #E9F2EE;}}
QPushButton#StudyAction:focus {{border: 2px solid {PRIMARY};}}
QTableWidget {{background: white; color: {INK}; border: none; font-size: 14px;}}
QTableWidget::item {{padding: 5px; border-bottom: 1px solid {LINE};}}
QTableWidget::item:selected {{background: #DDEDE6; color: {INK};}}
QTableWidget QHeaderView {{padding: 0; margin: 0; border: none;}}
QHeaderView::section {{background: #E9F2EE; color: {MUTED}; padding: 7px;
    border: none; font-size: 13px;}}
QTabWidget::pane {{border: none;}}
QTabBar::tab {{padding: 10px 15px; color: {MUTED};}}
QTabBar::tab:selected {{background: white; color: {PRIMARY};}}
"""


def _table(data: DisplayTable, name: str) -> QTableWidget:
    table = QTableWidget(len(data.rows), len(data.columns))
    table.setAccessibleName(name)
    table.setHorizontalHeaderLabels(data.columns)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    table.verticalHeader().hide()
    table.verticalHeader().setDefaultSectionSize(35)
    table.horizontalHeader().setFixedHeight(36)
    table.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignLeft)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    table.setShowGrid(False)
    table.setFrameShape(QFrame.Shape.NoFrame)
    for row, cells in enumerate(data.rows):
        for column, value in enumerate(cells):
            item = QTableWidgetItem(value)
            item.setToolTip(f"{data.columns[column]}: {value}")
            item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            table.setItem(row, column, item)
    return table


class CurrentStudyDetails(QDialog):
    """Formatted uncertainty, individual runs and source identity from the presenter."""

    def __init__(self, view: FocusedStudyView, parent: QWidget) -> None:
        super().__init__(parent)
        self.setWindowTitle("Current study details")
        self.setStyleSheet(_STYLE)
        self.resize(1050, 650)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 18)
        layout.addWidget(result_label("Current study", size=32, bold=True))
        layout.addWidget(result_label(view.count_statement, color=MUTED))
        self.tabs = QTabWidget()
        for title, table in (
            ("PPO vs Fixed Rotation", view.paired[0][1]),
            ("PPO vs optimizer", view.paired[1][1]),
            ("Individual runs", view.runs),
            ("Shortage levels", view.shortage),
        ):
            page = QWidget()
            content = QVBoxLayout(page)
            if title.startswith("PPO vs"):
                content.addWidget(result_label(view.interval_statement, color=MUTED, wrap=True))
            else:
                content.addWidget(
                    result_label("Frozen final-test means on matched scenarios.", color=MUTED)
                )
            content.addWidget(_table(table, title), 1)
            if title == "PPO vs Fixed Rotation":
                content.addWidget(
                    result_label(
                        "Fixed Rotation uses Off/Full Service without critical reservation. "
                        "This compares complete systems; it does not isolate learning alone.",
                        color=MUTED,
                        wrap=True,
                    )
                )
            elif title == "PPO vs optimizer":
                content.addWidget(
                    result_label(
                        "Both protect critical demand. The optimizer maximizes the current hour; "
                        "the result does not compare PPO with a full-day optimum.",
                        color=MUTED,
                        wrap=True,
                    )
                )
            self.tabs.addTab(page, title)
        evidence_page = QWidget()
        evidence = QVBoxLayout(evidence_page)
        evidence.addWidget(
            result_label("Pinned evidence and independent replay", size=20, bold=True)
        )
        evidence.addWidget(result_label(view.training_statement, color=MUTED, wrap=True))
        evidence.addWidget(
            result_label(
                "PPO starts from fairness-rule imitation, "
                "then trains with critical-service constraints.",
                color=MUTED,
                wrap=True,
            )
        )
        evidence.addWidget(result_label(view.safety_statement, wrap=True))
        evidence.addWidget(
            result_label(
                "Study statistics were verified on CUDA. Independent replay checked every input "
                "hash, 2,304 final trajectories and 5,280 validation trajectories "
                "with zero discrepancy.",
                wrap=True,
            )
        )
        for path, digest in view.evidence:
            evidence.addWidget(result_label(path, size=13, bold=True, wrap=True))
            evidence.addWidget(result_label(f"SHA-256 · {digest}", size=12, color=MUTED, wrap=True))
        evidence.addWidget(result_label(view.live_statement, color=MUTED, wrap=True))
        evidence.addStretch(1)
        self.tabs.addTab(evidence_page, "Evidence")
        layout.addWidget(self.tabs, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class FocusedResultsPage(QWidget):
    """Current study by default, with the historical comparison kept inspectable."""

    def __init__(
        self,
        snapshot: PresentationSnapshot,
        parent=None,
        *,
        loader: Callable[[], FocusedStudyView] = load_focused_results,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("FocusedResultsPage")
        self.setAccessibleName("Results page: current five-run study")
        self.setStyleSheet(_STYLE)
        self.study_view: FocusedStudyView | None = None
        self.current_results_table: QTableWidget | None = None
        self.details_dialog: CurrentStudyDetails | None = None
        self.historical_dialog = QDialog(self)
        self.historical_dialog.setWindowTitle("Earlier results")
        self.historical_dialog.resize(1320, 750)
        historical_layout = QVBoxLayout(self.historical_dialog)
        historical_layout.addWidget(
            result_label(
                "Earlier frozen policy · separate study and scenarios",
                color=PRIMARY,
                bold=True,
            )
        )
        self.historical_page = ResultsPage(snapshot)
        historical_layout.addWidget(self.historical_page, 1)
        close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close.rejected.connect(self.historical_dialog.reject)
        historical_layout.addWidget(close)
        # Backward-compatible integration field explicitly remains the historical
        # three-outcome table. Current visible data is current_results_table.
        self.results_table = self.historical_page.results_table
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 7, 16, 10)
        layout.setSpacing(8)
        header = QHBoxLayout()
        heading = QVBoxLayout()
        heading.setSpacing(1)
        heading.addWidget(result_label("Results", size=48, bold=True))
        heading.addWidget(result_label("Current study", size=16, bold=True, color=PRIMARY))
        header.addLayout(heading, 1)
        self.earlier_button = QPushButton("Earlier results")
        self.earlier_button.setObjectName("StudyAction")
        self.earlier_button.clicked.connect(self.historical_dialog.open)
        self.details_button = QPushButton("Study details")
        self.details_button.setObjectName("StudyAction")
        self.details_button.clicked.connect(self._open_details)
        header.addWidget(self.earlier_button)
        header.addWidget(self.details_button)
        layout.addLayout(header)
        try:
            self.study_view = view = loader()
        except (FocusedStudyError, ResourceLocationError) as error:
            self.error_label = result_label(
                "Current study unavailable. No current study values are shown because "
                "the evidence could not be verified.",
                size=20,
                wrap=True,
            )
            self.error_label.setToolTip(str(error))
            layout.addWidget(self.error_label)
            layout.addStretch(1)
            self.details_button.setEnabled(False)
            return
        self.count_label = result_label(view.count_statement, color=MUTED)
        layout.addWidget(self.count_label)
        cards = QHBoxLayout()
        cards.setSpacing(16)
        for row_index in (0, 1, 2):
            cells = view.means.rows[row_index]
            card = QFrame()
            card.setObjectName("CurrentOutcomeCard")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(22, 14, 22, 14)
            card_layout.setSpacing(4)
            card_layout.addWidget(result_label(cells[0], color=MUTED, size=15))
            card_layout.addWidget(result_label(cells[3], size=33, bold=True, color=PRIMARY))
            card_layout.addWidget(result_label("PPO · five-run mean", size=12, color=MUTED))
            card_layout.addWidget(
                result_label(
                    f"Fixed Rotation {cells[1]} · Optimizer {cells[2]}",
                    size=12,
                    color=MUTED,
                    wrap=True,
                )
            )
            cards.addWidget(card, 1)
        layout.addLayout(cards)
        self.current_results_table = _table(view.means, "Current three-controller study means")
        self.current_results_table.setAccessibleDescription(
            "Frozen matched final-test results. PPO averages five independently trained policies."
        )
        self.current_results_table.setFixedHeight(213)
        layout.addWidget(self.current_results_table)
        self.tradeoff_label = result_label(view.tradeoff, size=14, bold=True, wrap=True)
        layout.addWidget(self.tradeoff_label)
        layout.addStretch(1)
        self.scope_note = result_label(
            "Synthetic scenarios · Decision support only. Live Scenario uses one selected "
            "PPO policy; a single live scenario can differ from these frozen five-run means.",
            size=12,
            color=MUTED,
            wrap=True,
        )
        layout.addWidget(self.scope_note)

    def _open_details(self) -> None:
        if self.study_view is None:
            return
        if self.details_dialog is None:
            self.details_dialog = CurrentStudyDetails(self.study_view, self)
        self.details_dialog.open()

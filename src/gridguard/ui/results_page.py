"""Approved native comparison layout bound to the frozen presentation snapshot."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gridguard.claim_contract import (
    HOLDOUT_CUSTODY_LIMITATION,
    RESULTS_SCOPE_LIMITATION,
    SENSITIVITY_INTERVAL_LIMITATION,
)
from gridguard.ui.components import format_metric_value
from gridguard.ui.presentation_model import PresentationSnapshot
from gridguard.ui.proposal_results import ProposalResultsDialog, build_proposal_results_view
from gridguard.ui.results_components import (
    INK,
    LINE,
    MUTED,
    PRIMARY,
    EvidenceDetailsDialog,
    OutcomeComparisonPanel,
    result_label,
)


class ResultsPage(QWidget):
    """Three evidence-derived comparisons, paired inference, and inspectable provenance."""

    def __init__(
        self,
        snapshot: PresentationSnapshot,
        parent=None,
        *,
        proposal_loader: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        super().__init__(parent)
        self._proposal_loader = proposal_loader
        self.proposal_dialog: ProposalResultsDialog | None = None
        self.proposal_error_dialog: QMessageBox | None = None
        self.setObjectName("ResultsPage")
        self.setAccessibleName("Results page")
        self.setAccessibleDescription(
            f"Frozen documented-seed-disjoint holdout from evidence {snapshot.artifact_id}. "
            f"{HOLDOUT_CUSTODY_LIMITATION} {SENSITIVITY_INTERVAL_LIMITATION} "
            f"{RESULTS_SCOPE_LIMITATION}"
        )
        self.setStyleSheet(
            "QWidget#ResultsPage {background: #F4F7F5;} "
            "QFrame#ResultsComparisonSurface {background: white; border: none; "
            "border-radius: 20px;} "
            f"QPushButton#ResultsEvidenceButton {{background: white; color: {INK}; "
            f"border: 1px solid {LINE}; border-radius: 12px; padding: 0 18px; "
            "font-size: 14px; min-height: 44px;} "
            "QPushButton#ResultsEvidenceButton:hover {background: #E9F2EE;} "
            f"QPushButton#ResultsEvidenceButton:focus {{border: 2px solid {PRIMARY};}} "
            f"QTableWidget#ResultsTable {{background: transparent; color: {INK}; "
            "border: none; font-size: 14px; gridline-color: transparent;} "
            "QTableWidget#ResultsTable::item {padding: 4px 5px; "
            f"border-bottom: 1px solid {LINE};}} "
            "QTableWidget#ResultsTable::item:selected {background: #DDEDE6; color: #183D40;} "
            "QTableWidget#ResultsTable QHeaderView {padding: 0; margin: 0; border: none;} "
            "QTableWidget#ResultsTable QHeaderView::section {background: #F4F7F5; "
            f"color: {MUTED}; "
            f"border: none; border-bottom: 1px solid {LINE}; font-size: 12px; "
            "font-weight: 400; padding: 5px;}"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 7, 16, 10)
        layout.setSpacing(7)
        header = QHBoxLayout()
        heading = QVBoxLayout()
        heading.setSpacing(3)
        self.title_label = result_label("Results", size=54, bold=True)
        heading.addWidget(self.title_label)
        heading.addWidget(
            result_label(
                "PPO and Fixed Rotation on identical matched scenarios.", size=16, color=MUTED
            )
        )
        header.addLayout(heading, 1)
        actions = QVBoxLayout()
        actions.setSpacing(5)
        actions.addWidget(
            result_label("Frozen holdout results", bold=True, color=PRIMARY),
            0,
            Qt.AlignmentFlag.AlignRight,
        )
        self.evidence_details_button = QPushButton("Evidence details")
        self.evidence_details_button.setObjectName("ResultsEvidenceButton")
        self.evidence_details_button.setAccessibleName("Open frozen holdout evidence details")
        self.evidence_details_button.setMinimumHeight(44)
        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        buttons.addWidget(self.evidence_details_button)
        self.extended_study_button = QPushButton("Extended study")
        self.extended_study_button.setObjectName("ResultsEvidenceButton")
        self.extended_study_button.setAccessibleName("Open extended controller and forecast study")
        self.extended_study_button.setMinimumHeight(44)
        self.extended_study_button.clicked.connect(self._open_proposal_results)
        buttons.addWidget(self.extended_study_button)
        actions.addLayout(buttons)
        header.addLayout(actions)
        layout.addLayout(header)

        sample = QHBoxLayout()
        self.count_label = result_label(
            f"{snapshot.matched_scenario_count:,} matched scenario conditions   "
            f"from {snapshot.independent_base_draw_count:,} independent base draws",
            color=MUTED,
        )
        self.count_label.setAccessibleDescription(snapshot.count_statement)
        sample.addWidget(self.count_label, 1)
        sample.addWidget(result_label("Mild, Moderate and Severe shortage", color=MUTED))
        layout.addLayout(sample)

        self.comparison_surface = QFrame()
        self.comparison_surface.setObjectName("ResultsComparisonSurface")
        self.comparison_surface.setAccessibleName("Three primary outcome comparisons")
        self.comparison_surface.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.comparison_surface.setMaximumHeight(290)
        comparisons = QHBoxLayout(self.comparison_surface)
        comparisons.setContentsMargins(0, 0, 0, 0)
        comparisons.setSpacing(0)
        self.outcome_panels: dict[str, OutcomeComparisonPanel] = {}
        for metric in snapshot.metrics:
            if self.outcome_panels:
                separator = QFrame()
                separator.setFixedWidth(1)
                separator.setStyleSheet(f"background: {LINE}; border: none;")
                comparisons.addWidget(separator)
            panel = OutcomeComparisonPanel(metric, self.comparison_surface)
            comparisons.addWidget(panel, 11 if metric.unit == "locality-hours" else 10)
            self.outcome_panels[metric.key] = panel
        layout.addWidget(self.comparison_surface, 1)

        table_heading = QHBoxLayout()
        table_heading.addWidget(result_label("The paired difference", size=21, bold=True), 1)
        table_heading.addWidget(
            result_label("Paired 95% intervals · pp = percentage points", color=MUTED)
        )
        layout.addLayout(table_heading)
        self.results_table = self._comparison_table(snapshot)
        layout.addWidget(self.results_table)

        note_row = QHBoxLayout()
        note_row.addWidget(
            result_label("Differences use unrounded values.", size=13, color=MUTED), 1
        )
        self.safety_note = result_label(
            "Power balance: maximum power-balance violation rate "
            f"{snapshot.maximum_power_balance_violation_rate:.2%} across both controllers.",
            color=MUTED,
        )
        self.safety_note.setAccessibleName(
            f"{self.safety_note.text()} Simulator conservation invariant, "
            "not evidence of real-grid operational safety."
        )
        note_row.addWidget(self.safety_note)
        layout.addLayout(note_row)
        boundaries = QHBoxLayout()
        boundaries.setSpacing(24)
        self.scope_note = result_label(
            "Comparison scope. Fixed Rotation is supply-blind, uses only Off and Full Service, "
            "and cannot select Critical Only.",
            color=MUTED,
            wrap=True,
        )
        self.scope_note.setAccessibleName(f"Scientific claim boundary: {RESULTS_SCOPE_LIMITATION}")
        self.scope_note.setToolTip(RESULTS_SCOPE_LIMITATION)
        boundaries.addWidget(self.scope_note, 3)
        self.synthetic_note = result_label(
            "Synthetic scenarios · Decision support only.\n"
            "Frozen results stay separate from the live scenario.",
            color=MUTED,
            wrap=True,
        )
        self.synthetic_note.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        boundaries.addWidget(self.synthetic_note, 2)
        for note in (self.scope_note, self.synthetic_note):
            # Reserve both lines and breathing room even at the presentation's 768px height.
            note.ensurePolished()
            note.setContentsMargins(0, 4, 0, 4)
            note.setMinimumHeight(2 * note.fontMetrics().height() + 8)
        layout.addLayout(boundaries)

        self.evidence_dialog = EvidenceDetailsDialog(snapshot, self)
        self.evidence_details_button.clicked.connect(self.evidence_dialog.open)
        self.evidence_dialog.finished.connect(
            lambda _result: QTimer.singleShot(0, self._restore_evidence_focus)
        )
        # Retain stable evidence-field access for integrations and assistive tooling.
        for name in (
            "provenance_card",
            "provenance_status",
            "provenance_id",
            "provenance_release",
            "provenance_sha256",
            "provenance_summary",
            "evidence_role_note",
        ):
            setattr(self, name, getattr(self.evidence_dialog, name))

    def _restore_evidence_focus(self) -> None:
        self.window().activateWindow()
        self.evidence_details_button.setFocus(Qt.FocusReason.OtherFocusReason)

    def _open_proposal_results(self) -> None:
        try:
            if self._proposal_loader is None:
                from gridguard.proposal_evidence import load_proposal_supplement

                payload = load_proposal_supplement()
            else:
                payload = self._proposal_loader()
            view = build_proposal_results_view(payload)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            self.proposal_error_dialog = QMessageBox(self)
            self.proposal_error_dialog.setWindowTitle("Extended study unavailable")
            self.proposal_error_dialog.setIcon(QMessageBox.Icon.Warning)
            self.proposal_error_dialog.setText(
                "The additional comparisons could not be verified. No study values are shown."
            )
            self.proposal_error_dialog.setDetailedText(str(error))
            self.proposal_error_dialog.finished.connect(
                lambda _result: QTimer.singleShot(0, self._restore_proposal_focus)
            )
            self.proposal_error_dialog.open()
            return
        if self.proposal_dialog is not None:
            self.proposal_dialog.deleteLater()
        self.proposal_dialog = ProposalResultsDialog(view, self)
        self.proposal_dialog.finished.connect(
            lambda _result: QTimer.singleShot(0, self._restore_proposal_focus)
        )
        self.proposal_dialog.open()

    def _restore_proposal_focus(self) -> None:
        self.window().activateWindow()
        self.extended_study_button.setFocus(Qt.FocusReason.OtherFocusReason)

    @staticmethod
    def _comparison_table(snapshot: PresentationSnapshot) -> QTableWidget:
        table = QTableWidget(len(snapshot.metrics), 4)
        table.setObjectName("ResultsTable")
        table.setAccessibleName("Paired controller comparison results")
        table.setAccessibleDescription(
            f"{len(snapshot.metrics)} primary-outcome rows from {snapshot.artifact_id} compare "
            "PPO with Fixed Rotation. Delta is PPO minus Fixed Rotation and includes a paired "
            "95 percent confidence interval. The sensitivity supplement is not shown in this "
            "table; its intervals are descriptive and pointwise, with no multiplicity adjustment."
        )
        table.setHorizontalHeaderLabels(["Metric", "PPO", "Fixed Rotation", "PPO − Fixed Rotation"])
        table.verticalHeader().hide()
        table.setFrameShape(QFrame.Shape.NoFrame)
        table.setShowGrid(False)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        table.horizontalHeader().setDefaultAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        table.horizontalHeader().setMinimumSectionSize(145)
        for column in range(4):
            mode = (
                QHeaderView.ResizeMode.ResizeToContents
                if column in (1, 2)
                else QHeaderView.ResizeMode.Stretch
            )
            table.horizontalHeader().setSectionResizeMode(column, mode)
        table.horizontalHeader().setFixedHeight(31)
        table.verticalHeader().setDefaultSectionSize(34)
        table.setFixedHeight(136)
        for row, metric in enumerate(snapshot.metrics):
            delta = format_metric_value(
                metric, metric.mean_delta, signed=True, as_delta=True, include_unit=False
            )
            lower = format_metric_value(
                metric, metric.ci95_lower, as_delta=True, include_unit=False
            )
            upper = format_metric_value(
                metric, metric.ci95_upper, as_delta=True, include_unit=False
            )
            interval_unit = "pp" if metric.unit == "%" else metric.unit
            values = (
                metric.label,
                format_metric_value(metric, metric.ppo),
                format_metric_value(metric, metric.fixed_rotation),
                f"{delta} [{lower}, {upper}] {interval_unit}",
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(
                    f"{metric.label}; {table.horizontalHeaderItem(column).text()}: {value}"
                )
                item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                table.setItem(row, column, item)
        return table

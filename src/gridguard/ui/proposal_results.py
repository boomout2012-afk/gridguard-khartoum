"""Display-only view models and native tables for the verified extended study."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from gridguard.ui.results_components import INK, LINE, MUTED, PRIMARY, result_label


@dataclass(frozen=True, slots=True)
class DisplayTable:
    """Already formatted values; Qt never calculates scientific results."""

    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True, slots=True)
class ProposalResultsView:
    artifact_id: str
    count_statement: str
    evaluation_statement: str
    training_statement: str
    means: DisplayTable
    paired: tuple[tuple[str, DisplayTable], ...]
    forecast: DisplayTable


# key, display label with mean unit, interval unit, percentage display
_METRICS = (
    (
        "critical_outage_locality_hours",
        "Critical interruption (locality-hours)",
        "locality-h",
        False,
    ),
    ("fairness", "Cumulative-service Jain fairness", "pp", True),
    ("utilization", "Supply utilization", "pp", True),
    ("critical_service_ratio", "Critical energy served", "pp", True),
    ("blackout_locality_hours", "Zero-service blackout (locality-hours)", "locality-h", False),
    (
        "blackout_duration_mean_pairwise_gap_hours",
        "Blackout-duration disparity (hours)",
        "h",
        False,
    ),
    (
        "critical_interruption_duration_mean_pairwise_gap_hours",
        "Critical-duration disparity (hours)",
        "h",
        False,
    ),
    ("longest_blackout_hours", "Longest blackout per episode (hours)", "h", False),
)
_PAIRED_METRICS = tuple(_METRICS[index] for index in (0, 1, 2, 5, 6))


def build_proposal_results_view(payload: Mapping[str, Any]) -> ProposalResultsView:
    """Format a previously verified payload without aggregating or inferring values."""
    names = tuple(payload["controller_names"])
    means = DisplayTable(
        columns=("Metric", *names),
        rows=tuple(
            (
                label,
                *(
                    f"{payload['controller_means'][name][key]:.2%}"
                    if percentage
                    else f"{payload['controller_means'][name][key]:.2f}"
                    for name in names
                ),
            )
            for key, label, _unit, percentage in _METRICS
        ),
    )
    paired = []
    for pair in (
        "PPO - Fixed Rotation",
        "Optimization - Fixed Rotation",
        "PPO - Optimization",
    ):
        rows = []
        for key, label, unit, percentage in _PAIRED_METRICS:
            interval = payload["paired_inference"][pair][key]
            scale = 100 if percentage else 1
            # Percentage-point conversion is presentation only; use supplied intervals.
            delta = interval["mean_delta"] * scale
            lower = interval["ci95_lower"] * scale
            upper = interval["ci95_upper"] * scale
            rows.append((label, f"{delta:+.2f} {unit}", f"[{lower:+.2f}, {upper:+.2f}] {unit}"))
        paired.append(
            (pair, DisplayTable(("Metric", "Mean difference", "Paired 95% interval"), tuple(rows)))
        )
    forecast = payload["forecast"]
    forecast_rows = []
    for key, label in (
        ("trailing_mean_3h", "Trailing mean (3 hours)"),
        ("persistence_1h", "Persistence (1 hour)"),
        ("ridge_ar3", "Learned ridge AR(3)"),
    ):
        method = forecast["methods"][key]
        forecast_rows.append(
            (
                label,
                *(
                    f"{method[scope][metric]:.4f}"
                    for scope in ("locality_sector", "system_total")
                    for metric in ("mae_mw", "rmse_mw")
                ),
            )
        )
    protocol = payload["protocol"]
    seeds = ", ".join(map(str, protocol["evaluation_seeds"]))
    training_seeds = ", ".join(map(str, forecast["training_seeds"]))
    return ProposalResultsView(
        artifact_id=payload["artifact_id"],
        count_statement=(
            f"{protocol['matched_scenario_conditions']:,} matched scenario conditions from "
            f"{protocol['independent_base_draws']:,} independent base draws; "
            f"{protocol['controller_evaluations']:,} controller evaluations."
        ),
        evaluation_statement=(
            f"Mild, Moderate and Severe shortage. Evaluation seeds: {seeds}. "
            f"Batch size: {protocol['batch_size']}."
        ),
        training_statement=(
            f"Learned forecast fitted on {forecast['training_trace_count']:,} training traces "
            f"from seeds {training_seeds}. "
            f"{forecast['evaluation_trace_count']:,} evaluation traces use the separate "
            f"study seeds above, targeting hours {forecast['target_hours'][0]}–"
            f"{forecast['target_hours'][-1]}."
        ),
        means=means,
        paired=tuple(paired),
        forecast=DisplayTable(
            (
                "Forecast method",
                "Locality-sector\nMAE (MW)",
                "Locality-sector\nRMSE (MW)",
                "System total\nMAE (MW)",
                "System total\nRMSE (MW)",
            ),
            tuple(forecast_rows),
        ),
    )


def _display_table(model: DisplayTable, name: str) -> QTableWidget:
    table = QTableWidget(len(model.rows), len(model.columns))
    table.setAccessibleName(name)
    table.setHorizontalHeaderLabels(model.columns)
    table.verticalHeader().hide()
    table.verticalHeader().setDefaultSectionSize(36)
    table.setShowGrid(False)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    table.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    table.setWordWrap(False)
    header = table.horizontalHeader()
    header.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
    for column in range(1, len(model.columns)):
        header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
    header.setMinimumSectionSize(120)
    header.setFixedHeight(
        header.fontMetrics().height() * max(len(label.splitlines()) for label in model.columns) + 16
    )
    _set_rows(table, model)
    return table


def _set_rows(table: QTableWidget, model: DisplayTable) -> None:
    table.setRowCount(len(model.rows))
    for row, values in enumerate(model.rows):
        for column, value in enumerate(values):
            item = QTableWidgetItem(value)
            item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            item.setToolTip(f"{values[0]}; {model.columns[column]}: {value}")
            table.setItem(row, column, item)


class ProposalResultsDialog(QDialog):
    """Three compact, keyboard-accessible views of an independently verified study."""

    def __init__(self, view: ProposalResultsView, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.view = view
        self.setObjectName("ProposalResultsDialog")
        self.setWindowTitle("Extended study")
        self.setAccessibleName("Extended study: controller, duration and forecast comparisons")
        self.setModal(True)
        self.resize(1040, 680)
        self.setStyleSheet(
            "QDialog#ProposalResultsDialog {background: #F4F7F5;} "
            "QTabWidget::pane {border: none; background: white; border-radius: 16px;} "
            f"QTabBar::tab {{background: transparent; color: {MUTED}; padding: 12px 18px; "
            "border: 2px solid transparent; border-radius: 10px; margin-right: 6px;} "
            f"QTabBar::tab:selected {{background: {PRIMARY}; color: white;}} "
            "QTabBar::tab:focus {border: 2px solid #005FCC;} "
            f"QTableWidget {{background: white; color: {INK}; border: none; font-size: 14px;}} "
            f"QTableWidget::item {{padding: 6px; border-bottom: 1px solid {LINE};}} "
            "QTableWidget::item:selected {background: #DDEDE6; color: #183D40;} "
            "QTableWidget QHeaderView {padding: 0; margin: 0; border: none;} "
            f"QHeaderView::section {{background: white; color: {MUTED}; padding: 8px 6px; "
            f"border: none; border-bottom: 1px solid {LINE}; font-size: 13px;}}"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 18, 24, 16)
        layout.setSpacing(9)
        layout.addWidget(result_label("Extended study", size=32, bold=True))
        layout.addWidget(result_label(view.count_statement, color=PRIMARY, wrap=True))
        layout.addWidget(result_label(view.evaluation_statement, size=13, color=MUTED, wrap=True))
        self.tabs = QTabWidget()
        self.tabs.setAccessibleName("Extended study sections")
        layout.addWidget(self.tabs, 1)

        controllers = QWidget()
        controller_layout = QVBoxLayout(controllers)
        controller_layout.setContentsMargins(14, 12, 14, 12)
        self.means_table = _display_table(view.means, "Extended study controller means")
        self.means_table.setAccessibleDescription(
            "Equal-episode means on the same matched scenarios for all three controllers."
        )
        controller_layout.addWidget(self.means_table, 1)
        controller_layout.addWidget(
            result_label(
                "Optimization exhaustively maximizes the canonical one-hour reward over 2,187 "
                "service-tier actions. It uses current and past state; "
                "it is not a 24-hour optimum.",
                size=13,
                color=MUTED,
                wrap=True,
            )
        )
        controller_layout.addWidget(
            result_label(
                "Blackout means zero service when demand is positive. Critical interruption means "
                "less than 99% of critical demand served. Duration disparity is the mean gap "
                "across the 21 locality pairs within each episode; "
                "equal durations can still mean poor service. Global proportional clipping "
                "can give every locality the same critical interruption hours.",
                size=13,
                color=MUTED,
                wrap=True,
            )
        )
        self.tabs.addTab(controllers, "Controllers")

        paired = QWidget()
        paired_layout = QVBoxLayout(paired)
        paired_layout.setContentsMargins(14, 12, 14, 12)
        selector_row = QHBoxLayout()
        pair_label = result_label("Compare")
        selector_row.addWidget(pair_label)
        self.pair_selector = QComboBox()
        self.pair_selector.setAccessibleName("Controller pair for paired differences")
        self.pair_selector.addItems([pair for pair, _model in view.paired])
        pair_label.setBuddy(self.pair_selector)
        selector_row.addWidget(self.pair_selector, 1)
        paired_layout.addLayout(selector_row)
        self.paired_table = _display_table(
            view.paired[0][1], "Extended study paired confidence intervals"
        )
        paired_layout.addWidget(self.paired_table, 1)
        paired_layout.addWidget(
            result_label(
                "Difference = first controller minus second. Lower interruption durations and "
                "duration gaps are better; higher Jain fairness and utilization are better. "
                "pp means percentage points. Intervals are descriptive and pointwise.",
                size=13,
                color=MUTED,
                wrap=True,
            )
        )
        self.pair_selector.currentIndexChanged.connect(self._select_pair)
        self._select_pair(0)
        self.tabs.addTab(paired, "Paired differences")

        forecast = QWidget()
        forecast_layout = QVBoxLayout(forecast)
        forecast_layout.setContentsMargins(14, 12, 14, 12)
        forecast_layout.addWidget(
            result_label("One-hour-ahead demand prediction", size=20, bold=True)
        )
        forecast_layout.addWidget(result_label(view.training_statement, color=MUTED, wrap=True))
        self.forecast_table = _display_table(
            view.forecast, "Forecast error comparison in megawatts"
        )
        forecast_layout.addWidget(self.forecast_table, 1)
        forecast_layout.addWidget(
            result_label(
                "MAE is mean absolute error; RMSE is root mean square error. Lower is better. "
                "Locality-sector errors and total-system errors use different aggregation scales. "
                "This forecast comparison does not retrain or alter the PPO policy.",
                size=13,
                color=MUTED,
                wrap=True,
            )
        )
        self.tabs.addTab(forecast, "Forecasting")

        footer = QHBoxLayout()
        footer.addWidget(
            result_label(
                f"Synthetic scenarios; decision support only.\nEvidence: {view.artifact_id}",
                size=12,
                color=MUTED,
                wrap=True,
            ),
            1,
        )
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.buttons.rejected.connect(self.reject)
        footer.addWidget(self.buttons)
        layout.addLayout(footer)

    def _select_pair(self, index: int) -> None:
        pair, table = self.view.paired[index]
        _set_rows(self.paired_table, table)
        self.paired_table.setAccessibleDescription(
            f"{pair}. Supplied paired differences and 95 percent confidence intervals."
        )

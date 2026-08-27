"""Scenario input and aggregate-metric widgets for the live page."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAccessible, QAccessibleEvent, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
)

from gridguard.domain import ControllerName
from gridguard.ui.components import CardFrame
from gridguard.ui.live_models import LiveScenarioComparison
from gridguard.ui.theme import MONO_FONT_FAMILY


class LiveScenarioControls(CardFrame):
    """Shortage inputs and an accessible run-state readout."""

    run_requested = Signal(str, int)
    inputs_changed = Signal(str, int)

    def __init__(self, parent=None) -> None:
        super().__init__(accessible_name="Scenario controls", parent=parent)
        self.content.setContentsMargins(14, 9, 14, 9)
        self.content.setSpacing(6)
        control_row = QHBoxLayout()
        control_row.setSpacing(8)

        self.shortage_combo = QComboBox()
        self.shortage_combo.addItems(["Mild", "Moderate", "Severe"])
        self.shortage_combo.setAccessibleName("Shortage level")
        self.shortage_combo.setFixedWidth(132)
        self.seed_input = QSpinBox()
        self.seed_input.setRange(0, 2_147_483_647)
        self.seed_input.setValue(50001)
        self.seed_input.setAccessibleName("Scenario seed")
        self.seed_input.setFixedWidth(122)
        shortage_label = QLabel("Shortage level")
        shortage_label.setBuddy(self.shortage_combo)
        seed_label = QLabel("Seed")
        seed_label.setBuddy(self.seed_input)
        control_row.addWidget(shortage_label)
        control_row.addWidget(self.shortage_combo)
        control_row.addSpacing(8)
        control_row.addWidget(seed_label)
        control_row.addWidget(self.seed_input)
        control_row.addStretch(1)

        self.run_button = QPushButton("Run on GPU")
        self.run_button.setProperty("primary", True)
        self.run_button.setAccessibleName("Run selected scenario on GPU")
        self.run_button.clicked.connect(self._emit_run_request)
        control_row.addWidget(self.run_button)
        self.content.addLayout(control_row)

        self.state_label = QLabel("No live scenario has been run. Choose inputs, then run on GPU.")
        self.state_label.setProperty("state", "empty")
        self.state_label.setProperty("compact", True)
        self.state_label.setWordWrap(True)
        self.content.addWidget(self.state_label)

        self.shortage_combo.currentTextChanged.connect(self._emit_inputs_changed)
        self.seed_input.valueChanged.connect(self._emit_inputs_changed)

    def set_busy(self, busy: bool) -> None:
        """Lock scenario inputs while a background GPU run is active."""

        self.shortage_combo.setEnabled(not busy)
        self.seed_input.setEnabled(not busy)
        self.run_button.setEnabled(not busy)
        if busy:
            self.set_state(
                "Running both controllers on the shared CUDA scenario…",
                "loading",
            )

    def set_state(self, text: str, state: str) -> None:
        _set_state_text(self.state_label, text, state)

    def _emit_run_request(self) -> None:
        self.run_requested.emit(
            self.shortage_combo.currentText().casefold(),
            self.seed_input.value(),
        )

    def _emit_inputs_changed(self, _value: str | int) -> None:
        self.inputs_changed.emit(
            self.shortage_combo.currentText().casefold(),
            self.seed_input.value(),
        )


class LiveComparisonTable(QTableWidget):
    """Read-only aggregate metrics for one matched live comparison."""

    def __init__(self, parent=None) -> None:
        super().__init__(2, 6, parent)
        self.setAccessibleName("Live demonstration aggregate comparison")
        self.setHorizontalHeaderLabels(
            (
                "Controller",
                "Served energy",
                "Critical service",
                "Fairness",
                "Curtailed",
                "Balance violations",
            )
        )
        self.verticalHeader().hide()
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFixedHeight(100)
        self.clear_metrics()

    def clear_metrics(self) -> None:
        self.setAccessibleDescription("No verified live demonstration metrics are available yet.")
        for row, controller in enumerate(_CONTROLLERS):
            self._set_row(row, (controller, "—", "—", "—", "—", "—"))

    def show_comparison(self, comparison: LiveScenarioComparison) -> None:
        self.setAccessibleDescription(
            f"Live demonstration metrics for {comparison.shortage_level} shortage, "
            f"seed {comparison.seed}; these values do not replace frozen evidence."
        )
        for row, controller in enumerate(_CONTROLLERS):
            metrics = comparison.metrics(controller)
            self._set_row(
                row,
                (
                    controller,
                    f"{metrics.served_mwh:,.0f} MWh",
                    f"{metrics.critical_service_ratio:.1%}",
                    f"{metrics.fairness:.1%}",
                    f"{metrics.curtailed_mwh:,.0f} MWh",
                    f"{metrics.power_balance_violation_rate:.1%}",
                ),
                align_values=True,
            )

    def _set_row(
        self,
        row: int,
        values: tuple[str, ...],
        *,
        align_values: bool = False,
    ) -> None:
        for column, value in enumerate(values):
            item = QTableWidgetItem(value)
            if column > 0:
                item.setFont(QFont(MONO_FONT_FAMILY))
                if align_values:
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
            self.setItem(row, column, item)


_CONTROLLERS = (ControllerName.PPO.value, ControllerName.FIXED_ROTATION.value)


def _set_accessible_text(label: QLabel, text: str) -> None:
    """Update a dynamic label and notify assistive technology of its new name."""

    label.setText(text)
    QAccessible.updateAccessibility(QAccessibleEvent(label, QAccessible.Event.NameChanged))


def _set_state_text(label: QLabel, text: str, state: str) -> None:
    """Update a live-region label and its redundant visual state."""

    label.setProperty("state", state)
    label.style().unpolish(label)
    label.style().polish(label)
    _set_accessible_text(label, text)

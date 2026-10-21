"""Scenario input and aggregate-metric widgets for the live page."""

from __future__ import annotations

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QAccessible, QAccessibleEvent, QColor, QFont, QPainter, QPen
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
    QVBoxLayout,
    QWidget,
)

from gridguard.domain import ControllerName
from gridguard.ui.components import CardFrame
from gridguard.ui.live_models import LiveScenarioComparison
from gridguard.ui.theme import MONO_FONT_FAMILY


class LiveComboBox(QComboBox):
    """Native selector with a reliably rendered chevron on offline Qt platforms."""

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        with QPainter(self) as painter:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(QPen(QColor("#183D40" if self.isEnabled() else "#64736C"), 1.8))
            x, y = self.width() - 15, self.height() / 2
            painter.drawLine(QPointF(x - 4, y - 2), QPointF(x, y + 2))
            painter.drawLine(QPointF(x, y + 2), QPointF(x + 4, y - 2))


class LiveScenarioControls(CardFrame):
    """Shortage inputs and an accessible run-state readout."""

    run_requested = Signal(str, int)
    inputs_changed = Signal(str, int)
    state_changed = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(accessible_name="Scenario controls", parent=parent)
        self.setProperty("card", False)
        self.setObjectName("liveScenarioControls")
        self.content.setContentsMargins(0, 0, 0, 0)
        self.content.setSpacing(0)
        control_row = QHBoxLayout()
        control_row.setSpacing(12)
        self.control_row = control_row

        self.shortage_combo = LiveComboBox()
        self.shortage_combo.addItems(["Mild", "Moderate", "Severe"])
        self.shortage_combo.setCurrentText("Moderate")
        self.shortage_combo.setAccessibleName("Shortage level")
        self.shortage_combo.setFixedWidth(132)
        self.seed_input = QSpinBox()
        self.seed_input.setRange(0, 2_147_483_647)
        self.seed_input.setValue(50002)
        self.seed_input.setAccessibleName("Scenario seed")
        self.seed_input.setFixedWidth(122)
        self.seed_input.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
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

        self.controller_slot = QHBoxLayout()
        control_row.addLayout(self.controller_slot)
        control_row.addStretch(1)
        self.run_button = QPushButton("Run scenario")
        self.run_button.setProperty("primary", True)
        self.run_button.setAccessibleName("Run selected scenario")
        self.run_button.setMinimumSize(174, 44)
        self.shortage_combo.setMinimumHeight(44)
        self.seed_input.setMinimumHeight(44)
        self.run_button.clicked.connect(self._emit_run_request)
        control_row.addWidget(self.run_button)
        self.content.addLayout(control_row)

        self.state_label = QLabel(
            "No live scenario has been run. Choose inputs, then run a scenario."
        )
        self.state_label.setProperty("state", "empty")
        self.state_label.setProperty("compact", True)
        self.state_label.setWordWrap(True)
        self.state_label.setObjectName("liveRunState")
        self.state_label.hide()

        self.shortage_combo.currentTextChanged.connect(self._emit_inputs_changed)
        self.seed_input.valueChanged.connect(self._emit_inputs_changed)

    def set_busy(self, busy: bool) -> None:
        """Lock scenario inputs while a background GPU run is active."""

        self.shortage_combo.setEnabled(not busy)
        self.seed_input.setEnabled(not busy)
        self.run_button.setEnabled(not busy)
        if busy:
            self.set_state(
                "Running both controllers on the shared scenario…",
                "loading",
            )

    def set_state(self, text: str, state: str) -> None:
        _set_state_text(self.state_label, text, state)
        self.state_changed.emit(state)

    def add_controller_selector(self, selector: QWidget) -> None:
        """Place the visible controller switch between scenario inputs and Run."""

        self.controller_slot.addWidget(selector)

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


class LiveComparisonSummary(QWidget):
    """Presentation of supplied matched episode metrics, with stable PPO/baseline roles."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("liveComparisonSummary")
        self.setAccessibleName("Matched full-day live comparison")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 2, 20, 2)
        layout.setSpacing(28)
        introduction = QVBoxLayout()
        introduction.setSpacing(3)
        title = QLabel("The full 24 hours")
        title.setObjectName("liveComparisonTitle")
        introduction.addWidget(title)
        self.caption = QLabel("PPO above; Fixed Rotation below.\nThe same scenario.")
        self.caption.setProperty("muted", True)
        introduction.addWidget(self.caption)
        layout.addLayout(introduction, 1)
        self.values: dict[str, QLabel] = {}
        self.baselines: dict[str, QLabel] = {}
        for heading in ("Served energy", "Critical service", "Jain fairness"):
            group = QVBoxLayout()
            group.setSpacing(3)
            label = QLabel(heading)
            label.setProperty("muted", True)
            label.setObjectName("liveMetricCaption")
            value = QLabel("—")
            value.setObjectName("liveMetricValue")
            baseline = QLabel("Fixed Rotation —")
            baseline.setObjectName("liveMetricBaseline")
            self.values[heading] = value
            self.baselines[heading] = baseline
            group.addWidget(label)
            group.addWidget(value)
            group.addWidget(baseline)
            layout.addLayout(group, 1)
        self.clear_metrics()

    def clear_metrics(self) -> None:
        for value in self.values.values():
            _set_accessible_text(value, "—")
        for baseline in self.baselines.values():
            _set_accessible_text(baseline, "Fixed Rotation —")
        self.setToolTip("")
        self.setAccessibleDescription("No verified live demonstration metrics are available yet.")
        QAccessible.updateAccessibility(
            QAccessibleEvent(self, QAccessible.Event.DescriptionChanged)
        )

    def show_comparison(self, comparison: LiveScenarioComparison) -> None:
        ppo = comparison.metrics(ControllerName.PPO.value)
        fixed = comparison.metrics(ControllerName.FIXED_ROTATION.value)
        pairs = {
            "Served energy": (f"{ppo.served_mwh:,.0f} MWh", f"{fixed.served_mwh:,.0f} MWh"),
            "Critical service": (
                f"{ppo.critical_service_ratio:.1%}",
                f"{fixed.critical_service_ratio:.1%}",
            ),
            "Jain fairness": (f"{ppo.fairness:.1%}", f"{fixed.fairness:.1%}"),
        }
        for heading, (learned, baseline) in pairs.items():
            _set_accessible_text(self.values[heading], learned)
            _set_accessible_text(self.baselines[heading], f"Fixed Rotation {baseline}")
        description = (
            f"Live demonstration, {comparison.shortage_level} shortage, seed {comparison.seed}. "
            "PPO is shown above Fixed Rotation. These values do not replace frozen evidence. "
            f"Curtailed energy: PPO {ppo.curtailed_mwh:,.0f} MWh; "
            f"Fixed Rotation {fixed.curtailed_mwh:,.0f} MWh. "
            f"Power-balance violations: PPO {ppo.power_balance_violation_rate:.1%}; "
            f"Fixed Rotation {fixed.power_balance_violation_rate:.1%}."
        )
        self.setAccessibleDescription(description)
        self.setToolTip(description)


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

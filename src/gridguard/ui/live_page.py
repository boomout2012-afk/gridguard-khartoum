"""Live-scenario page and its simulation integration boundary."""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from gridguard.domain import SECTORS, ControllerName
from gridguard.ui.components import CardFrame, StatusLegend, page_intro
from gridguard.ui.live_models import LiveScenarioComparison, LiveScenarioFrame
from gridguard.ui.map_asset import MapAsset
from gridguard.ui.map_view import SchematicMapView


class LiveScenarioPage(QWidget):
    """Scenario controls plus an honest empty state until GPU results arrive."""

    run_requested = Signal(str, int)

    def __init__(self, asset: MapAsset, parent=None) -> None:
        super().__init__(parent)
        self._frames: dict[int, LiveScenarioFrame] = {}
        self._comparison: LiveScenarioComparison | None = None
        self.setAccessibleName("Live Scenario page")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 18)
        layout.setSpacing(12)
        title, description = page_intro(
            "Explore a 24-hour shortage scenario",
            "Choose the shortage level and deterministic seed. The presentation shell "
            "never invents a run; the map updates only when the GPU simulation supplies frames.",
        )
        layout.addWidget(title)
        layout.addWidget(description)

        controls = CardFrame(accessible_name="Scenario controls")
        control_row = QHBoxLayout()
        form = QFormLayout()
        self.shortage_combo = QComboBox()
        self.shortage_combo.addItems(["Mild", "Moderate", "Severe"])
        self.shortage_combo.setAccessibleName("Shortage level")
        self.seed_input = QSpinBox()
        self.seed_input.setRange(0, 2_147_483_647)
        self.seed_input.setValue(50001)
        self.seed_input.setAccessibleName("Scenario seed")
        form.addRow("Shortage level", self.shortage_combo)
        form.addRow("Seed", self.seed_input)
        control_row.addLayout(form)
        control_row.addStretch(1)
        self.run_button = QPushButton("Run on GPU")
        self.run_button.setProperty("primary", True)
        self.run_button.setAccessibleName("Run selected scenario on GPU")
        self.run_button.clicked.connect(self._request_run)
        control_row.addWidget(self.run_button)
        controls.content.addLayout(control_row)
        self.state_label = QLabel("No live scenario has been run. Choose inputs, then run on GPU.")
        self.state_label.setProperty("muted", True)
        self.state_label.setWordWrap(True)
        self.state_label.setAccessibleName("Live scenario state")
        controls.content.addWidget(self.state_label)
        layout.addWidget(controls)

        body = QHBoxLayout()
        body.setSpacing(12)
        map_card = CardFrame(accessible_name="Live service allocation map")
        time_row = QHBoxLayout()
        self.hour_label = QLabel("Hour —")
        self.hour_label.setProperty("heading", "section")
        self.controller_combo = QComboBox()
        self.controller_combo.addItems(
            [ControllerName.PPO.value, ControllerName.FIXED_ROTATION.value]
        )
        self.controller_combo.setEnabled(False)
        self.controller_combo.setAccessibleName("Live map controller")
        self.controller_combo.currentTextChanged.connect(self._select_controller)
        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.setRange(0, 23)
        self.timeline.setEnabled(False)
        self.timeline.setAccessibleName("Scenario hour")
        self.timeline.valueChanged.connect(self._show_hour)
        reset_button = QPushButton("Reset map")
        time_row.addWidget(self.hour_label)
        time_row.addWidget(QLabel("View"))
        time_row.addWidget(self.controller_combo)
        time_row.addWidget(self.timeline, 1)
        time_row.addWidget(reset_button)
        map_card.content.addLayout(time_row)
        self.map_view = SchematicMapView(asset)
        reset_button.clicked.connect(self.map_view.reset_view)
        map_card.content.addWidget(self.map_view, 1)
        body.addWidget(map_card, 8)

        side = QVBoxLayout()
        side.setSpacing(8)
        self.locality_card = CardFrame(accessible_name="Selected locality details")
        self.locality_card.content.setContentsMargins(12, 10, 12, 10)
        self.locality_card.content.setSpacing(4)
        self.locality_name = QLabel("Khartoum")
        self.locality_name.setProperty("heading", "section")
        self.locality_context = QLabel("No controller output · Hour —")
        self.locality_context.setProperty("muted", True)
        self.locality_status = QLabel("Status: not run")
        self.locality_status.setProperty("muted", True)
        self.locality_card.content.addWidget(self.locality_name)
        self.locality_card.content.addWidget(self.locality_context)
        self.locality_card.content.addWidget(self.locality_status)
        detail_grid = QGridLayout()
        detail_grid.setHorizontalSpacing(14)
        self.locality_demand = QLabel("—")
        self.locality_served = QLabel("—")
        self.locality_critical = QLabel("—")
        for column, (label, value) in enumerate(
            (
                ("Demand", self.locality_demand),
                ("Served", self.locality_served),
                ("Critical", self.locality_critical),
            )
        ):
            value.setProperty("detail", "value")
            caption = QLabel(label)
            caption.setProperty("muted", True)
            detail_grid.addWidget(value, 0, column)
            detail_grid.addWidget(caption, 1, column)
        self.locality_card.content.addLayout(detail_grid)
        self.locality_projection = QLabel("Next hour: — · DISPLAY ONLY\nNot PPO input/evidence.")
        self.locality_projection.setProperty("muted", True)
        self.locality_projection.setProperty("compact", True)
        self.locality_projection.setWordWrap(True)
        self.locality_projection.setAccessibleName("Selected locality next-hour demand projection")
        self.locality_projection.setAccessibleDescription(
            "A causal trailing three-hour mean for display only; it is not used by PPO "
            "or frozen evidence."
        )
        self.locality_card.content.addWidget(self.locality_projection)
        sector_heading = QLabel("Sector service delivered")
        sector_heading.setProperty("muted", True)
        self.locality_card.content.addWidget(sector_heading)
        self.sector_service_labels: dict[str, QLabel] = {}
        sector_grid = QGridLayout()
        sector_grid.setHorizontalSpacing(12)
        sector_grid.setVerticalSpacing(2)
        sector_grid.setColumnMinimumWidth(1, 58)
        sector_grid.setColumnMinimumWidth(3, 58)
        sector_grid.setColumnStretch(0, 1)
        sector_grid.setColumnStretch(2, 1)
        for index, sector in enumerate(SECTORS):
            name = QLabel(sector.value)
            value = QLabel("—")
            value.setAlignment(Qt.AlignmentFlag.AlignRight)
            self.sector_service_labels[sector.value] = value
            row = index % 3
            column = (index // 3) * 2
            sector_grid.addWidget(name, row, column)
            sector_grid.addWidget(value, row, column + 1)
        self.locality_card.content.addLayout(sector_grid)
        self.map_view.locality_selected.connect(self._update_selected_locality)
        side.addWidget(self.locality_card)
        side.addWidget(StatusLegend())
        side.addStretch(1)
        body.addLayout(side, 3)
        layout.addLayout(body, 1)

    def set_frames(self, frames: Sequence[LiveScenarioFrame]) -> None:
        """Install 24 genuine simulation frames and show the first hour."""

        self._comparison = None
        self.controller_combo.setEnabled(False)
        self._install_frames(frames)
        self.state_label.setText("GPU simulation complete · use the hour control to explore.")

    def set_comparison(self, comparison: LiveScenarioComparison) -> None:
        """Install both controller timelines from one shared CUDA scenario."""

        if not isinstance(comparison, LiveScenarioComparison):
            raise TypeError("comparison must be a LiveScenarioComparison")
        self._comparison = comparison
        self.controller_combo.blockSignals(True)
        self.controller_combo.setCurrentText(ControllerName.PPO.value)
        self.controller_combo.blockSignals(False)
        self.controller_combo.setEnabled(True)
        self._install_frames(comparison.frames(ControllerName.PPO.value))
        self.state_label.setText("GPU comparison complete · showing PPO on the shared scenario.")

    def set_busy(self, busy: bool) -> None:
        """Lock scenario inputs while a background GPU run is active."""

        self.shortage_combo.setEnabled(not busy)
        self.seed_input.setEnabled(not busy)
        self.run_button.setEnabled(not busy)
        if busy:
            self.state_label.setText("Running both controllers on the shared CUDA scenario…")

    def _install_frames(
        self,
        frames: Sequence[LiveScenarioFrame],
        *,
        reset_hour: bool = True,
    ) -> None:
        indexed = {frame.hour: frame for frame in frames}
        if set(indexed) != set(range(24)) or len(frames) != 24:
            raise ValueError("a live run must contain exactly one frame for each of 24 hours")
        selected_hour = 0 if reset_hour else self.timeline.value()
        self._frames = indexed
        self.timeline.setEnabled(True)
        self.timeline.setValue(selected_hour)
        self._show_hour(selected_hour)

    def show_run_error(self, message: str) -> None:
        """Expose an integration error without substituting demo values."""

        self._frames.clear()
        self._comparison = None
        self.timeline.setEnabled(False)
        self.controller_combo.setEnabled(False)
        self.set_busy(False)
        self.state_label.setText(f"GPU simulation unavailable: {message}")

    def _request_run(self) -> None:
        shortage = self.shortage_combo.currentText().casefold()
        seed = self.seed_input.value()
        self._frames.clear()
        self._comparison = None
        self.timeline.setEnabled(False)
        self.controller_combo.setEnabled(False)
        self.state_label.setText("Scenario requested · awaiting verified GPU simulation output.")
        self.run_requested.emit(shortage, seed)

    def _select_controller(self, controller_name: str) -> None:
        if self._comparison is None:
            return
        self._install_frames(
            self._comparison.frames(controller_name),
            reset_hour=False,
        )
        self.state_label.setText(
            f"GPU comparison complete · showing {controller_name} on the shared scenario."
        )

    def _show_hour(self, hour: int) -> None:
        frame = self._frames.get(hour)
        if frame is None:
            return
        self.hour_label.setText(
            f"Hour {hour:02d}:00  ·  demand {frame.demand_mw:,.0f} MW  ·  "
            f"supply {frame.supply_mw:,.0f} MW"
        )
        self.map_view.set_service_tiers(frame.service_tiers)
        self._update_selected_locality(self.map_view.selected_locality_id or "khartoum")

    def _update_selected_locality(self, locality_id: str) -> None:
        item = self.map_view.locality_items[locality_id]
        self.locality_name.setText(item.locality.name)
        frame = self._frames.get(self.timeline.value())
        if frame is None:
            self.locality_context.setText("No controller output · Hour —")
            self.locality_status.setText("Status: not run")
            self.locality_demand.setText("—")
            self.locality_served.setText("—")
            self.locality_critical.setText("—")
            self.locality_projection.setText("Next hour: — · DISPLAY ONLY\nNot PPO input/evidence.")
            for label in self.sector_service_labels.values():
                label.setText("—")
            return

        controller = self.controller_combo.currentText()
        detail = frame.locality_details[locality_id]
        self.locality_context.setText(f"{controller} · Hour {frame.hour:02d}:00")
        self.locality_status.setText(f"Service tier: {item.status_text}")
        self.locality_demand.setText(f"{detail.demand_mw:.1f} MW")
        self.locality_served.setText(f"{detail.served_mw:.1f} MW")
        self.locality_critical.setText(f"{detail.critical_service_ratio:.1%}")
        projected_demand = detail.next_hour_projected_demand_mw
        if projected_demand is None:
            reason = "needs 3-hour history" if frame.hour < 2 else "end of trace"
            self.locality_projection.setText(
                f"Next hour: unavailable ({reason}) · DISPLAY ONLY\nNot PPO input/evidence."
            )
        else:
            self.locality_projection.setText(
                f"Next hour {frame.hour + 1:02d}:00: {projected_demand:.1f} MW "
                "· DISPLAY ONLY\nTrailing 3-hour mean; not PPO input/evidence."
            )
        for sector, ratio in detail.sector_service_ratio.items():
            self.sector_service_labels[sector].setText(f"{ratio:.1%}")

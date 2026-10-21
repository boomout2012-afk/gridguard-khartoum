"""Live-scenario page and its simulation integration boundary."""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from gridguard.domain import ControllerName
from gridguard.ui.components import ClaimBoundaryNote, page_intro
from gridguard.ui.live_components import (
    LiveComparisonTable,
    LiveLocalityPanel,
    LiveMapPanel,
    LiveScenarioControls,
)
from gridguard.ui.live_models import LiveScenarioComparison, LiveScenarioFrame
from gridguard.ui.map_asset import MapAsset
from gridguard.ui.power_trace import PowerTrace


class LiveScenarioPage(QWidget):
    """Orchestrate genuine live frames across focused presentation widgets."""

    run_requested = Signal(str, int)

    def __init__(self, asset: MapAsset, parent=None) -> None:
        super().__init__(parent)
        self._frames: dict[int, LiveScenarioFrame] = {}
        self._comparison: LiveScenarioComparison | None = None
        self.setAccessibleName("Live Scenario page")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 12, 18, 10)
        layout.setSpacing(8)
        title, description = page_intro(
            "Explore a 24-hour shortage scenario",
            "Run both controllers on identical synthetic inputs. Play the day, inspect a locality, "
            "and switch controllers to compare their decisions.",
        )
        layout.addWidget(title)
        layout.addWidget(description)

        self.controls = LiveScenarioControls()
        self.shortage_combo = self.controls.shortage_combo
        self.seed_input = self.controls.seed_input
        self.run_button = self.controls.run_button
        self.state_label = self.controls.state_label
        self.controls.run_requested.connect(self._request_run)
        self.controls.inputs_changed.connect(self._invalidate_changed_inputs)
        layout.addWidget(self.controls)

        self.comparison_caption = QLabel(
            "Live demonstration · matched aggregate metrics from this run"
        )
        self.comparison_caption.setProperty("eyebrow", True)
        layout.addWidget(self.comparison_caption)
        self.comparison_table = LiveComparisonTable()
        metrics_row = QHBoxLayout()
        metrics_row.setSpacing(10)
        metrics_row.addWidget(self.comparison_table, 3)
        self.power_trace = PowerTrace()
        metrics_row.addWidget(self.power_trace, 1)
        layout.addLayout(metrics_row)
        self.scope_note = ClaimBoundaryNote()
        layout.addWidget(self.scope_note)

        body = QHBoxLayout()
        body.setSpacing(10)
        self.map_panel = LiveMapPanel(asset)
        self.hour_label = self.map_panel.hour_label
        self.controller_combo = self.map_panel.controller_combo
        self.timeline = self.map_panel.timeline
        self.map_view = self.map_panel.map_view
        body.addWidget(self.map_panel, 7)

        self.locality_panel = LiveLocalityPanel(asset, self.map_view)
        self.locality_list = self.locality_panel.locality_list
        self.locality_card = self.locality_panel.locality_card
        self.locality_name = self.locality_panel.locality_name
        self.locality_context = self.locality_panel.locality_context
        self.locality_status = self.locality_panel.locality_status
        self.locality_demand = self.locality_panel.locality_demand
        self.locality_served = self.locality_panel.locality_served
        self.locality_critical = self.locality_panel.locality_critical
        self.locality_projection = self.locality_panel.locality_projection
        self.sector_service_labels = self.locality_panel.sector_service_labels
        body.addWidget(self.locality_panel, 4)
        layout.addLayout(body, 1)

        self.controller_combo.currentTextChanged.connect(self._select_controller)
        self.timeline.valueChanged.connect(self._show_hour)
        self.map_view.locality_selected.connect(self._update_selected_locality)

    def set_frames(self, frames: Sequence[LiveScenarioFrame]) -> None:
        """Install 24 genuine simulation frames and show the first hour."""

        self._comparison = None
        self.controller_combo.setEnabled(False)
        self.comparison_table.clear_metrics()
        self._install_frames(frames)
        self.controls.set_state(
            "GPU simulation complete · use the hour control to explore.",
            "ready",
        )

    def set_comparison(self, comparison: LiveScenarioComparison) -> None:
        """Install both controller timelines from one shared CUDA scenario."""

        if not isinstance(comparison, LiveScenarioComparison):
            raise TypeError("comparison must be a LiveScenarioComparison")
        self._comparison = comparison
        self.controller_combo.blockSignals(True)
        self.controller_combo.setCurrentText(ControllerName.PPO.value)
        self.controller_combo.blockSignals(False)
        self.controller_combo.setEnabled(True)
        self.comparison_table.show_comparison(comparison)
        self._install_frames(comparison.frames(ControllerName.PPO.value))
        self.controls.set_state(
            "GPU comparison complete · showing PPO on the shared scenario.",
            "ready",
        )

    def set_busy(self, busy: bool) -> None:
        """Lock scenario inputs while a background GPU run is active."""

        self.controls.set_busy(busy)

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
        self.map_panel.stop_playback()
        self.power_trace.set_frames(frames)
        self.timeline.setEnabled(True)
        self.timeline.setValue(selected_hour)
        self._show_hour(selected_hour)

    def show_run_error(self, message: str) -> None:
        """Expose an integration error without substituting demo values."""

        self._reset_output()
        self.set_busy(False)
        self.controls.set_state(
            f"GPU simulation unavailable: {message}",
            "error",
        )

    def _request_run(
        self,
        shortage_level: str | None = None,
        seed: int | None = None,
    ) -> None:
        shortage = shortage_level or self.shortage_combo.currentText().casefold()
        scenario_seed = self.seed_input.value() if seed is None else seed
        self._reset_output()
        self.controls.set_state(
            "Scenario requested · awaiting verified GPU simulation output.",
            "loading",
        )
        self.run_requested.emit(shortage, scenario_seed)

    def _reset_output(self) -> None:
        self._frames.clear()
        self._comparison = None
        self.map_panel.reset_output()
        self.locality_panel.reset_output()
        self.comparison_table.clear_metrics()
        self.power_trace.clear()

    def _invalidate_changed_inputs(self, _shortage_level: str, _seed: int) -> None:
        if not self._frames and self._comparison is None:
            return
        self._reset_output()
        self.controls.set_state(
            "Inputs changed; run required before scientific output is available.",
            "empty",
        )

    def _select_controller(self, controller_name: str) -> None:
        if self._comparison is None:
            return
        self._install_frames(
            self._comparison.frames(controller_name),
            reset_hour=False,
        )
        self.controls.set_state(
            f"GPU comparison complete · showing {controller_name} on the shared scenario.",
            "ready",
        )

    def _show_hour(self, hour: int) -> None:
        frame = self._frames.get(hour)
        if frame is None:
            return
        self.map_panel.show_frame(frame)
        self.power_trace.set_hour(hour)
        self.locality_panel.refresh(frame, self.controller_combo.currentText())

    def _update_selected_locality(self, locality_id: str) -> None:
        self.locality_panel.update_selected_locality(
            locality_id,
            self._frames.get(self.timeline.value()),
            self.controller_combo.currentText(),
        )

    def _sync_locality_list(self, locality_id: str) -> None:
        self.locality_panel.sync_locality_list(locality_id)

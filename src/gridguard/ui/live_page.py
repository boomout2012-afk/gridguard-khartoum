"""Live-scenario page and its simulation integration boundary."""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget

from gridguard.claim_contract import RESULTS_SCOPE_LIMITATION
from gridguard.domain import ControllerName
from gridguard.live_policy import DEPLOYED_LIVE_POLICY, LivePolicyIdentity
from gridguard.ui.live_components import compose_live_page
from gridguard.ui.live_models import LiveScenarioComparison, LiveScenarioFrame
from gridguard.ui.map_asset import MapAsset


class LiveScenarioPage(QWidget):
    """Orchestrate genuine live frames across approved native presentation surfaces."""

    run_requested = Signal(str, int)

    def __init__(self, asset: MapAsset, parent=None) -> None:
        super().__init__(parent)
        self._frames: dict[int, LiveScenarioFrame] = {}
        self._comparison: LiveScenarioComparison | None = None
        self.setAccessibleName("Live Scenario page")
        compose_live_page(self, asset)
        self._show_policy_identity(awaiting=True)
        self.controls.run_requested.connect(self._request_run)
        self.controls.inputs_changed.connect(self._invalidate_changed_inputs)
        self.controller_combo.currentTextChanged.connect(self._select_controller)
        self.timeline.valueChanged.connect(self._show_hour)
        self.map_view.locality_selected.connect(self._update_selected_locality)
        self.map_view.select_locality("east_nile")
        self._update_selected_locality("east_nile")

    def set_frames(self, frames: Sequence[LiveScenarioFrame]) -> None:
        """Install 24 genuine simulation frames and show the first hour."""

        self._comparison = None
        self._show_policy_identity()
        self.map_panel.set_controller_available(False)
        self.comparison_table.clear_metrics()
        self.comparison_summary.clear_metrics()
        self._install_frames(frames)
        self.controls.set_state("Scenario complete · use the hour control to explore.", "ready")

    def set_comparison(self, comparison: LiveScenarioComparison) -> None:
        """Install all supplied controller timelines from the same genuine scenario."""

        if not isinstance(comparison, LiveScenarioComparison):
            raise TypeError("comparison must be a LiveScenarioComparison")
        self._comparison = comparison
        self._show_policy_identity(comparison.policy_identity)
        self.map_panel.set_controller_names(tuple(comparison.frames_by_controller))
        for control, value in (
            (self.shortage_combo, comparison.shortage_level.title()),
            (self.seed_input, comparison.seed),
        ):
            blocked = control.blockSignals(True)
            if control is self.shortage_combo:
                control.setCurrentText(value)
            else:
                control.setValue(value)
            control.blockSignals(blocked)
        self.controller_combo.blockSignals(True)
        self.controller_combo.setCurrentText(ControllerName.PPO.value)
        self.controller_combo.blockSignals(False)
        self.map_panel._sync_controller_buttons(ControllerName.PPO.value)
        self.map_panel.set_controller_available(True)
        self.comparison_table.show_comparison(comparison)
        self.comparison_summary.show_comparison(comparison)
        self._install_frames(comparison.frames(ControllerName.PPO.value))
        self.controls.set_state(
            "Comparison complete · showing PPO on the shared scenario.", "ready"
        )

    def set_busy(self, busy: bool) -> None:
        """Lock scenario inputs while the real background run is active."""

        self.controls.set_busy(busy)
        self.map_panel.set_controller_available(not busy and self._comparison is not None)

    def _install_frames(
        self, frames: Sequence[LiveScenarioFrame], *, reset_hour: bool = True
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
        """Expose an integration error without substituting demonstration values."""

        self._reset_output()
        self.set_busy(False)
        self.controls.set_state(f"Scenario unavailable: {message}", "error")

    def _request_run(self, shortage_level: str | None = None, seed: int | None = None) -> None:
        shortage = shortage_level or self.shortage_combo.currentText().casefold()
        scenario_seed = self.seed_input.value() if seed is None else seed
        self._reset_output()
        self.controls.set_state(
            "Scenario requested · awaiting verified simulation output.", "loading"
        )
        self.run_requested.emit(shortage, scenario_seed)

    def _reset_output(self) -> None:
        self._frames.clear()
        self._comparison = None
        self._show_policy_identity(awaiting=True)
        self.map_panel.reset_output()
        self.locality_panel.reset_output()
        self.comparison_table.clear_metrics()
        self.comparison_summary.clear_metrics()
        self.power_trace.clear()

    def _show_policy_identity(
        self, identity: LivePolicyIdentity | None = None, *, awaiting: bool = False
    ) -> None:
        """Identify the supplied policy without claiming a pending run was verified."""

        benchmark_note = (
            "Live Scenario uses one validation-selected PPO policy with critical-service "
            "constraints. Results reports frozen five-run benchmark means; an individual "
            "live scenario can differ. The constrained optimizer protects critical demand "
            "and maximizes the current hour, without a full-day optimality claim."
        )
        if identity is not None:
            title = (
                f"PPO · seed {identity.training_seed} · iteration {identity.checkpoint_iteration}"
            )
            detail = (
                f"Verified live model: {identity.model_relative_path}\n"
                f"Model SHA-256: {identity.model_sha256}\n"
                f"Training evidence SHA-256: {identity.training_sha256}"
            )
        elif awaiting:
            title = "Current trained PPO · ready to run"
            detail = (
                f"Run a scenario to verify and use training seed "
                f"{DEPLOYED_LIVE_POLICY.training_seed}, selected iteration "
                f"{DEPLOYED_LIVE_POLICY.checkpoint_iteration}."
            )
        else:
            title = "PPO identity not supplied"
            detail = "These supplied frames do not identify a deployed PPO checkpoint."
        self.scope_note.setText(f"{title}\nSynthetic scenario · Decision support only")
        description = f"{detail}\n\n{benchmark_note}\n\n{RESULTS_SCOPE_LIMITATION}"
        self.scope_note.setToolTip(description)
        self.scope_note.setAccessibleDescription(description)

    def _invalidate_changed_inputs(self, _shortage_level: str, _seed: int) -> None:
        if not self._frames and self._comparison is None:
            return
        self._reset_output()
        self.controls.set_state(
            "Inputs changed; run required before scientific output is available.", "empty"
        )

    def _select_controller(self, controller_name: str) -> None:
        if self._comparison is None:
            return
        self._install_frames(self._comparison.frames(controller_name), reset_hour=False)
        self.comparison_summary.show_comparison(self._comparison, controller_name=controller_name)
        self.controls.set_state(
            f"Comparison complete · showing {controller_name} on the shared scenario.", "ready"
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

"""Open map, timeline, and floating locality inspector for live simulation output."""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from gridguard.domain import SECTORS, ControllerName
from gridguard.ui.components import CardFrame, StatusLegend
from gridguard.ui.live_controls import LiveComboBox, _set_accessible_text
from gridguard.ui.live_models import LiveScenarioFrame
from gridguard.ui.map_asset import MapAsset
from gridguard.ui.map_items import LocalityGraphicsItem
from gridguard.ui.map_view import LocalityMapView


class LiveMapPanel(QWidget):
    """An open geographic canvas with separately placed selector and timeline."""

    def __init__(self, asset: MapAsset, parent=None) -> None:
        super().__init__(parent)
        self.setAccessibleName("Live service allocation map")
        self.content = QVBoxLayout(self)
        self.content.setContentsMargins(0, 0, 0, 0)
        self.content.setSpacing(3)
        legend = StatusLegend(compact=True)
        legend.setProperty("card", False)
        self.content.addWidget(legend, 0, Qt.AlignmentFlag.AlignRight)
        self.map_view = LocalityMapView(asset)
        self.map_view.setMinimumHeight(210)
        self.content.addWidget(self.map_view, 1)
        footer = QHBoxLayout()
        footer.setSpacing(12)
        credit = QLabel(
            "Boundaries: OCHA COD-AB 2020; Rivers: Natural Earth 5.0.\n"
            "Geographic context, not an electrical network."
        )
        credit.setObjectName("liveMapCredit")
        footer.addWidget(credit, 1)
        for text, name, action in (
            ("+", "Zoom in", self.map_view.zoom_in),
            ("−", "Zoom out", self.map_view.zoom_out),
            ("Fit", "Reset locality map view", self.map_view.reset_view),
        ):
            button = QPushButton(text)
            button.setObjectName("liveMapButton")
            button.setAccessibleName(name)
            button.setToolTip(name)
            button.setFixedSize(44, 44)
            button.clicked.connect(action)
            footer.addWidget(button)
        self.content.addLayout(footer)
        self._build_controller_selector()
        self._build_timeline()

    def _build_controller_selector(self) -> None:
        self.controller_combo = QComboBox(self)
        self.controller_combo.addItems([member.value for member in ControllerName])
        self.controller_combo.setAccessibleName("Live map controller")
        self.controller_combo.hide()
        self.controller_selector = QWidget()
        self.controller_selector.setObjectName("liveControllerSelector")
        row = QHBoxLayout(self.controller_selector)
        row.setContentsMargins(4, 4, 4, 4)
        row.setSpacing(3)
        self.controller_buttons: dict[str, QPushButton] = {}
        group = QButtonGroup(self)
        group.setExclusive(True)
        for controller in ControllerName:
            name = controller.value
            button = QPushButton(name)
            button.setCheckable(True)
            button.setObjectName("liveControllerButton")
            button.setMinimumHeight(44)
            # Qt style-sheet min-height is the content box: 34 + padding 6 + borders 4.
            button.setStyleSheet("min-height: 34px; max-height: 34px;")
            button.setMinimumWidth(72 if name == "PPO" else 132)
            button.setAccessibleName(f"Show {name} live scenario")
            if controller == ControllerName.OPTIMIZATION:
                button.setToolTip(
                    "Constrained optimizer: protects critical demand and exactly maximizes "
                    "the current-hour objective over feasible actions, as in the current study. "
                    "No future demand or full-day optimality claim."
                )
            button.clicked.connect(
                lambda _checked=False, value=name: self.controller_combo.setCurrentText(value)
            )
            group.addButton(button)
            row.addWidget(button)
            self.controller_buttons[name] = button
        self.controller_combo.currentTextChanged.connect(self._sync_controller_buttons)
        self._sync_controller_buttons(self.controller_combo.currentText())
        self.set_controller_available(False)

    def set_controller_names(self, names: tuple[str, ...]) -> None:
        """Show only controller timelines supplied by this comparison."""

        current = self.controller_combo.currentText()
        blocked = self.controller_combo.blockSignals(True)
        self.controller_combo.clear()
        self.controller_combo.addItems(names)
        self.controller_combo.setCurrentText(current if current in names else names[0])
        self.controller_combo.blockSignals(blocked)
        for name, button in self.controller_buttons.items():
            button.setVisible(name in names)
        self._sync_controller_buttons(self.controller_combo.currentText())

    def _sync_controller_buttons(self, name: str) -> None:
        for value, button in self.controller_buttons.items():
            button.setChecked(value == name)

    def set_controller_available(self, available: bool) -> None:
        self.controller_combo.setEnabled(available)
        for button in self.controller_buttons.values():
            button.setEnabled(available)

    def _build_timeline(self) -> None:
        self.timeline_bar = CardFrame(accessible_name="Scenario timeline")
        self.timeline_bar.setObjectName("liveTimeline")
        self.timeline_bar.content.setContentsMargins(16, 7, 16, 7)
        row = QHBoxLayout()
        row.setSpacing(16)
        self.play_button = QPushButton("▶")
        self.play_button.setObjectName("livePlayButton")
        self.play_button.setAccessibleName("Play the 24-hour scenario")
        self.play_button.setFixedSize(44, 44)
        self.play_button.setEnabled(False)
        row.addWidget(self.play_button)
        clock = QVBoxLayout()
        clock.setSpacing(0)
        self.clock_label = QLabel("—")
        self.clock_label.setObjectName("liveHourValue")
        self.clock_label.setMinimumWidth(100)
        self.hour_caption = QLabel("No scenario yet")
        self.hour_caption.setObjectName("liveSmallCaption")
        clock.addWidget(self.clock_label)
        clock.addWidget(self.hour_caption)
        row.addLayout(clock)
        rail = QVBoxLayout()
        rail.setSpacing(0)
        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.setRange(0, 23)
        self.timeline.setEnabled(False)
        self.timeline.setMinimumHeight(27)
        self.timeline.setAccessibleName("Scenario hour")
        rail.addWidget(self.timeline)
        ticks = QHBoxLayout()
        ticks.setContentsMargins(0, 0, 0, 0)
        for index, text in enumerate(
            ("00:00", "04:00", "08:00", "12:00", "16:00", "20:00", "23:00")
        ):
            label = QLabel(text)
            label.setObjectName("liveSmallCaption")
            ticks.addWidget(label)
            if index < 6:
                ticks.addStretch(1)
        rail.addLayout(ticks)
        row.addLayout(rail, 1)
        power = QGridLayout()
        power.setHorizontalSpacing(16)
        power.setVerticalSpacing(3)
        self.supply_label = QLabel("—")
        self.demand_label = QLabel("—")
        for index, (text, value) in enumerate(
            (("Available", self.supply_label), ("Total demand", self.demand_label))
        ):
            label = QLabel(text)
            label.setObjectName("liveSmallCaption")
            value.setObjectName("liveTimelinePower")
            value.setAlignment(Qt.AlignmentFlag.AlignRight)
            power.addWidget(label, index, 0)
            power.addWidget(value, index, 1)
        row.addLayout(power)
        self.timeline_bar.content.addLayout(row)
        self.hour_label = QLabel(self._empty_hour_text(), self)
        self.hour_label.hide()
        self.playback_timer = QTimer(self)
        self.playback_timer.setInterval(900)
        self.playback_timer.timeout.connect(self._advance_hour)
        self.play_button.clicked.connect(self._toggle_playback)

    @staticmethod
    def _empty_hour_text() -> str:
        return "Hour —  ·  demand unavailable  ·  supply unavailable"

    def _toggle_playback(self) -> None:
        if self.playback_timer.isActive():
            self.stop_playback()
            return
        if not self.timeline.isEnabled():
            return
        if self.timeline.value() == self.timeline.maximum():
            self.timeline.setValue(0)
        self.play_button.setText("Ⅱ")
        self.play_button.setAccessibleName("Pause scenario playback")
        self.playback_timer.start()

    def _advance_hour(self) -> None:
        self.timeline.setValue(min(self.timeline.value() + 1, self.timeline.maximum()))
        if self.timeline.value() == self.timeline.maximum():
            self.stop_playback()

    def stop_playback(self) -> None:
        self.playback_timer.stop()
        self.play_button.setText("▶")
        self.play_button.setAccessibleName("Play the 24-hour scenario")

    def hideEvent(self, event) -> None:  # noqa: N802
        self.stop_playback()
        super().hideEvent(event)

    def show_frame(self, frame: LiveScenarioFrame) -> None:
        self.play_button.setEnabled(True)
        _set_accessible_text(
            self.hour_label,
            f"Hour {frame.hour:02d}:00  ·  demand {frame.demand_mw:,.0f} MW  ·  "
            f"supply {frame.supply_mw:,.0f} MW",
        )
        _set_accessible_text(self.clock_label, f"{frame.hour:02d}:00")
        _set_accessible_text(self.hour_caption, f"Hour {frame.hour} of 24")
        _set_accessible_text(self.supply_label, f"{frame.supply_mw:,.1f} MW")
        _set_accessible_text(self.demand_label, f"{frame.demand_mw:,.1f} MW")
        self.map_view.set_service_tiers(frame.service_tiers)

    def reset_output(self) -> None:
        self.stop_playback()
        self.play_button.setEnabled(False)
        _set_accessible_text(self.hour_label, self._empty_hour_text())
        for label in (self.clock_label, self.supply_label, self.demand_label):
            _set_accessible_text(label, "—")
        _set_accessible_text(self.hour_caption, "No scenario yet")
        self.set_controller_names(tuple(member.value for member in ControllerName))
        blocked = self.controller_combo.blockSignals(True)
        self.controller_combo.setCurrentText(ControllerName.PPO.value)
        self.controller_combo.blockSignals(blocked)
        self._sync_controller_buttons(ControllerName.PPO.value)
        blocked = self.timeline.blockSignals(True)
        self.timeline.setValue(0)
        self.timeline.blockSignals(blocked)
        self.set_controller_available(False)
        self.timeline.setEnabled(False)
        selected = self.map_view.selected_locality_id or self.map_view.asset.keyboard_order[0]
        self.map_view.locality_items.clear()
        self.map_view.river_items.clear()
        self.map_view.scene().clear()
        self.map_view._build_scene()
        self.map_view.select_locality(selected)


class LocalityDetailCard(CardFrame):
    """Floating inspector for the selected locality in the supplied display frame."""

    def __init__(self, asset: MapAsset, parent=None) -> None:
        super().__init__(accessible_name="Selected locality details", parent=parent)
        self.setObjectName("liveLocalityCard")
        self.content.setContentsMargins(20, 12, 20, 12)
        self.content.setSpacing(4)
        self.content.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        self.setMinimumHeight(0)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        heading = QHBoxLayout()
        caption = QLabel("Selected locality")
        caption.setObjectName("liveSmallCaption")
        self.locality_clock = QLabel("—")
        self.locality_clock.setObjectName("liveSmallCaption")
        heading.addWidget(caption)
        heading.addStretch(1)
        heading.addWidget(self.locality_clock)
        self.content.addLayout(heading)
        self.locality_combo = LiveComboBox()
        self.locality_combo.setObjectName("liveLocalitySelector")
        self.locality_combo.setAccessibleName("Selected locality")
        self.locality_combo.setMinimumHeight(44)
        localities = {item.locality_id: item for item in asset.localities}
        for locality_id in asset.keyboard_order:
            self.locality_combo.addItem(localities[locality_id].name, locality_id)
        self.content.addWidget(self.locality_combo)
        self.locality_name = QLabel("East Nile", self)
        self.locality_name.hide()
        self.locality_context = QLabel("No controller output · Hour —", self)
        self.locality_context.hide()
        self.locality_status = QLabel("Not run")
        self.locality_status.setObjectName("liveServiceTier")
        self.content.addWidget(self.locality_status)
        power = QGridLayout()
        power.setHorizontalSpacing(20)
        power.setVerticalSpacing(1)
        self.locality_served = QLabel("—")
        self.locality_demand = QLabel("—")
        self.locality_critical = QLabel("—", self)
        self.locality_critical.hide()
        for column, (name, value) in enumerate(
            (("Power served", self.locality_served), ("Demand", self.locality_demand))
        ):
            caption = QLabel(name)
            caption.setObjectName("liveSmallCaption")
            value.setObjectName("livePowerValue")
            if column:
                caption.setAlignment(Qt.AlignmentFlag.AlignRight)
                value.setAlignment(Qt.AlignmentFlag.AlignRight)
            power.addWidget(caption, 0, column)
            power.addWidget(value, 1, column)
        self.content.addLayout(power)
        sector_heading = QLabel("Sector service                                     Demand met")
        sector_heading.setObjectName("liveSmallCaption")
        self.content.addWidget(sector_heading)
        self.sector_widget = QWidget()
        self.sector_widget.setObjectName("liveSectorSurface")
        self.sector_grid = QGridLayout(self.sector_widget)
        self.sector_grid.setContentsMargins(0, 0, 0, 0)
        self.sector_grid.setHorizontalSpacing(8)
        self.sector_grid.setVerticalSpacing(3)
        self.sector_grid.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        self.sector_service_labels: dict[str, QLabel] = {}
        self.sector_bars: dict[str, QProgressBar] = {}
        self.sector_names: dict[str, QLabel] = {}
        for sector in SECTORS:
            name = QLabel(sector.value)
            name.setObjectName("liveSectorLabel")
            value = QLabel("—")
            value.setObjectName("liveSectorLabel")
            value.setAlignment(Qt.AlignmentFlag.AlignRight)
            bar = QProgressBar()
            bar.setObjectName("liveSectorBar")
            bar.setRange(0, 1000)
            bar.setTextVisible(False)
            bar.setFixedHeight(5)
            self.sector_names[sector.value] = name
            self.sector_service_labels[sector.value] = value
            self.sector_bars[sector.value] = bar
        self._compact = None
        self._arrange_sectors(False)
        self.content.addWidget(self.sector_widget, 1)
        self.locality_projection = QLabel("Next hour: unavailable · display only")
        self.locality_projection.setObjectName("liveProjection")
        self.locality_projection.setWordWrap(True)
        self.locality_projection.setAccessibleDescription(
            "A causal trailing three-hour mean for display only; not PPO input/evidence."
        )
        self.content.addWidget(self.locality_projection)

    def _arrange_sectors(self, compact: bool) -> None:
        if compact == self._compact:
            return
        self._compact = compact
        for index, sector in enumerate(SECTORS):
            name = sector.value
            for widget in (
                self.sector_names[name],
                self.sector_bars[name],
                self.sector_service_labels[name],
            ):
                self.sector_grid.removeWidget(widget)
            if compact:
                row, group = divmod(index, 2)
                self.sector_grid.addWidget(self.sector_names[name], row, group * 2)
                self.sector_grid.addWidget(self.sector_service_labels[name], row, group * 2 + 1)
                self.sector_bars[name].hide()
            else:
                self.sector_grid.addWidget(self.sector_names[name], index, 0)
                self.sector_grid.addWidget(self.sector_bars[name], index, 1)
                self.sector_grid.addWidget(self.sector_service_labels[name], index, 2)
                self.sector_bars[name].show()
        self.sector_grid.setColumnStretch(1, 0 if compact else 1)

    def resizeEvent(self, event) -> None:  # noqa: N802
        self._arrange_sectors(self.height() < 300)
        super().resizeEvent(event)

    def update_details(
        self, item: LocalityGraphicsItem, frame: LiveScenarioFrame | None, controller_name: str
    ) -> None:
        _set_accessible_text(self.locality_name, item.locality.name)
        if frame is None:
            self._show_empty_state()
            return
        detail = frame.locality_details[item.locality.locality_id]
        _set_accessible_text(self.locality_context, f"{controller_name} · Hour {frame.hour:02d}:00")
        _set_accessible_text(self.locality_clock, f"{frame.hour:02d}:00")
        _set_accessible_text(self.locality_status, item.status_text)
        _set_accessible_text(self.locality_demand, f"{detail.demand_mw:.1f} MW")
        _set_accessible_text(self.locality_served, f"{detail.served_mw:.1f} MW")
        _set_accessible_text(self.locality_critical, f"{detail.critical_service_ratio:.1%}")
        self._show_projection(frame, detail.next_hour_projected_demand_mw)
        for sector, ratio in detail.sector_service_ratio.items():
            _set_accessible_text(self.sector_service_labels[sector], f"{ratio:.1%}")
            self.sector_bars[sector].setValue(round(ratio * 1000))

    def _show_empty_state(self) -> None:
        for label, text in (
            (self.locality_context, "No controller output · Hour —"),
            (self.locality_clock, "—"),
            (self.locality_status, "Not run"),
            (self.locality_demand, "—"),
            (self.locality_served, "—"),
            (self.locality_critical, "—"),
            (self.locality_projection, "Next hour: unavailable · display only"),
        ):
            _set_accessible_text(label, text)
        for name, label in self.sector_service_labels.items():
            _set_accessible_text(label, "—")
            self.sector_bars[name].setValue(0)

    def _show_projection(self, frame: LiveScenarioFrame, projected_demand_mw: float | None) -> None:
        if projected_demand_mw is None:
            reason = "needs 3-hour history" if frame.hour < 2 else "end of trace"
            text = f"Next hour: unavailable ({reason})\nDisplay only"
        else:
            text = f"Next-hour estimate: {projected_demand_mw:.1f} MW\nDisplay only"
        _set_accessible_text(self.locality_projection, text)
        self.locality_projection.setToolTip("Trailing 3-hour mean; not PPO input/evidence.")


class LiveLocalityPanel(QWidget):
    """Synchronize the locality dropdown, map, and retained selection model."""

    def __init__(self, asset: MapAsset, map_view: LocalityMapView, parent=None) -> None:
        super().__init__(parent)
        self.map_view = map_view
        self.setMinimumWidth(308)
        self.setMaximumWidth(350)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.locality_list = QListWidget(self)
        self.locality_list.setAccessibleName("Khartoum localities")
        self.locality_list.hide()
        localities = {item.locality_id: item for item in asset.localities}
        for locality_id in asset.keyboard_order:
            item = QListWidgetItem(f"{localities[locality_id].name} — not run")
            item.setData(Qt.ItemDataRole.UserRole, locality_id)
            self.locality_list.addItem(item)
        self.locality_card = LocalityDetailCard(asset)
        self.locality_combo = self.locality_card.locality_combo
        for name in (
            "locality_name",
            "locality_context",
            "locality_status",
            "locality_demand",
            "locality_served",
            "locality_critical",
            "locality_projection",
            "sector_service_labels",
        ):
            setattr(self, name, getattr(self.locality_card, name))
        layout.addWidget(self.locality_card)
        self.locality_list.currentItemChanged.connect(self._select_locality_from_list)
        self.locality_combo.currentIndexChanged.connect(self._select_locality_from_combo)
        self.sync_locality_list(map_view.selected_locality_id or asset.keyboard_order[0])

    def refresh(self, frame: LiveScenarioFrame, controller_name: str) -> None:
        for row in range(self.locality_list.count()):
            item = self.locality_list.item(row)
            map_item = self.map_view.locality_items[item.data(Qt.ItemDataRole.UserRole)]
            item.setText(f"{map_item.locality.name} — {map_item.status_text}")
        self.update_selected_locality(
            self.map_view.selected_locality_id or "khartoum", frame, controller_name
        )

    def reset_output(self) -> None:
        for row in range(self.locality_list.count()):
            item = self.locality_list.item(row)
            name = self.map_view.locality_items[item.data(Qt.ItemDataRole.UserRole)].locality.name
            item.setText(f"{name} — not run")
        self.update_selected_locality(
            self.map_view.selected_locality_id or self.map_view.asset.keyboard_order[0],
            None,
            ControllerName.PPO.value,
        )

    def update_selected_locality(
        self, locality_id: str, frame: LiveScenarioFrame | None, controller_name: str
    ) -> None:
        self.sync_locality_list(locality_id)
        self.locality_card.update_details(
            self.map_view.locality_items[locality_id], frame, controller_name
        )

    def _select_locality_from_list(
        self, current: QListWidgetItem | None, _previous: QListWidgetItem | None
    ) -> None:
        if current is not None:
            locality_id = current.data(Qt.ItemDataRole.UserRole)
            if locality_id != self.map_view.selected_locality_id:
                self.map_view.select_locality(locality_id)

    def _select_locality_from_combo(self, _index: int) -> None:
        locality_id = self.locality_combo.currentData()
        if locality_id and locality_id != self.map_view.selected_locality_id:
            self.map_view.select_locality(locality_id)

    def sync_locality_list(self, locality_id: str) -> None:
        for row in range(self.locality_list.count()):
            if self.locality_list.item(row).data(Qt.ItemDataRole.UserRole) == locality_id:
                blocked = self.locality_list.blockSignals(True)
                self.locality_list.setCurrentRow(row)
                self.locality_list.blockSignals(blocked)
                blocked = self.locality_combo.blockSignals(True)
                self.locality_combo.setCurrentIndex(self.locality_combo.findData(locality_id))
                self.locality_combo.blockSignals(blocked)
                return

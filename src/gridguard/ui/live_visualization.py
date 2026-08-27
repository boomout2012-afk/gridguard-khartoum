"""Map timeline and locality-detail widgets for live simulation output."""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListView,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from gridguard.domain import SECTORS, ControllerName
from gridguard.ui.components import CardFrame, StatusLegend
from gridguard.ui.live_controls import _set_accessible_text
from gridguard.ui.live_models import LiveScenarioFrame
from gridguard.ui.map_asset import MapAsset
from gridguard.ui.map_items import LocalityGraphicsItem
from gridguard.ui.map_view import LocalityMapView


class LiveMapPanel(CardFrame):
    """Controller/hour controls and the offline locality map."""

    def __init__(self, asset: MapAsset, parent=None) -> None:
        super().__init__(accessible_name="Live service allocation map", parent=parent)
        self.content.setContentsMargins(12, 9, 12, 9)
        self.content.setSpacing(6)

        readout_row = QHBoxLayout()
        readout_row.setSpacing(8)
        self.hour_label = QLabel(
            "Hour \N{EM DASH}  \N{MIDDLE DOT}  demand unavailable  "
            "\N{MIDDLE DOT}  supply unavailable"
        )
        self.hour_label.setProperty("heading", "section")
        self.hour_label.setProperty("mono", True)
        reset_button = QPushButton("Reset map")
        reset_button.setAccessibleName("Reset locality map view")
        readout_row.addWidget(self.hour_label)
        readout_row.addStretch(1)
        readout_row.addWidget(reset_button)
        self.content.addLayout(readout_row)

        timeline_row = QHBoxLayout()
        timeline_row.setSpacing(8)
        self.controller_combo = QComboBox()
        self.controller_combo.addItems(
            [ControllerName.PPO.value, ControllerName.FIXED_ROTATION.value]
        )
        self.controller_combo.setEnabled(False)
        self.controller_combo.setAccessibleName("Live map controller")
        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.setRange(0, 23)
        self.timeline.setEnabled(False)
        self.timeline.setAccessibleName("Scenario hour")
        view_label = QLabel("Controller")
        view_label.setBuddy(self.controller_combo)
        timeline_row.addWidget(view_label)
        timeline_row.addWidget(self.controller_combo)
        timeline_row.addWidget(self.timeline, 1)
        self.content.addLayout(timeline_row)

        self.map_view = LocalityMapView(asset)
        reset_button.clicked.connect(self.map_view.reset_view)
        self.content.addWidget(self.map_view, 1)

    def show_frame(self, frame: LiveScenarioFrame) -> None:
        _set_accessible_text(
            self.hour_label,
            f"Hour {frame.hour:02d}:00  ·  demand {frame.demand_mw:,.0f} MW  ·  "
            f"supply {frame.supply_mw:,.0f} MW",
        )
        self.map_view.set_service_tiers(frame.service_tiers)

    def reset_output(self) -> None:
        """Restore every scientific map readout to its neutral pre-run state."""

        _set_accessible_text(
            self.hour_label,
            "Hour \N{EM DASH}  \N{MIDDLE DOT}  demand unavailable  "
            "\N{MIDDLE DOT}  supply unavailable",
        )
        controller_signals_blocked = self.controller_combo.blockSignals(True)
        self.controller_combo.setCurrentText(ControllerName.PPO.value)
        self.controller_combo.blockSignals(controller_signals_blocked)
        timeline_signals_blocked = self.timeline.blockSignals(True)
        self.timeline.setValue(0)
        self.timeline.blockSignals(timeline_signals_blocked)
        self.controller_combo.setEnabled(False)
        self.timeline.setEnabled(False)

        selected_locality_id = (
            self.map_view.selected_locality_id or self.map_view.asset.keyboard_order[0]
        )
        self.map_view.locality_items.clear()
        self.map_view.river_items.clear()
        self.map_view.scene().clear()
        self.map_view._build_scene()
        self.map_view.select_locality(selected_locality_id)


class LocalityDetailCard(CardFrame):
    """Selected-locality values for the currently visible genuine frame."""

    def __init__(self, parent=None) -> None:
        super().__init__(accessible_name="Selected locality details", parent=parent)
        self.content.setContentsMargins(10, 7, 10, 7)
        self.content.setSpacing(3)
        self.locality_name = QLabel("Khartoum")
        self.locality_name.setProperty("heading", "section")
        self.locality_context = QLabel("No controller output · Hour —")
        self.locality_context.setProperty("muted", True)
        self.locality_context.setProperty("compact", True)
        self.locality_context.setProperty("mono", True)
        self.locality_status = QLabel("Status: not run")
        self.locality_status.setProperty("muted", True)
        self.locality_status.setProperty("compact", True)
        locality_header = QHBoxLayout()
        locality_header.setSpacing(6)
        locality_header.addWidget(self.locality_name)
        locality_header.addStretch(1)
        locality_header.addWidget(self.locality_status)
        self.content.addLayout(locality_header)
        self.content.addWidget(self.locality_context)
        self._add_power_details()
        self._add_projection()
        self._add_sector_details()

    def _add_power_details(self) -> None:
        detail_grid = QGridLayout()
        detail_grid.setHorizontalSpacing(10)
        detail_grid.setVerticalSpacing(0)
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
        self.content.addLayout(detail_grid)

    def _add_projection(self) -> None:
        self.locality_projection = QLabel("Next hour: — · DISPLAY ONLY\nNot PPO input/evidence.")
        self.locality_projection.setProperty("muted", True)
        self.locality_projection.setProperty("compact", True)
        self.locality_projection.setWordWrap(True)
        self.locality_projection.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Minimum,
        )
        self.locality_projection.setMinimumHeight(44)
        self.locality_projection.setAccessibleDescription(
            "A causal trailing three-hour mean for display only; it is not used by PPO "
            "or frozen evidence."
        )
        self.content.addWidget(self.locality_projection)

    def _add_sector_details(self) -> None:
        sector_heading = QLabel("Sector service delivered")
        sector_heading.setProperty("muted", True)
        self.content.addWidget(sector_heading)
        self.sector_service_labels: dict[str, QLabel] = {}
        sector_grid = QGridLayout()
        sector_grid.setHorizontalSpacing(6)
        sector_grid.setVerticalSpacing(1)
        for column in (0, 2, 4):
            sector_grid.setColumnStretch(column, 1)
        for index, sector in enumerate(SECTORS):
            name = QLabel(sector.value)
            name.setProperty("compact", True)
            value = QLabel("—")
            value.setProperty("compact", True)
            value.setProperty("mono", True)
            value.setAlignment(Qt.AlignmentFlag.AlignRight)
            self.sector_service_labels[sector.value] = value
            row = index % 2
            column = (index // 2) * 2
            sector_grid.addWidget(name, row, column)
            sector_grid.addWidget(value, row, column + 1)
        self.content.addLayout(sector_grid)

    def update_details(
        self,
        item: LocalityGraphicsItem,
        frame: LiveScenarioFrame | None,
        controller_name: str,
    ) -> None:
        _set_accessible_text(self.locality_name, item.locality.name)
        if frame is None:
            self._show_empty_state()
            return

        detail = frame.locality_details[item.locality.locality_id]
        _set_accessible_text(
            self.locality_context,
            f"{controller_name} · Hour {frame.hour:02d}:00",
        )
        _set_accessible_text(self.locality_status, f"Service tier: {item.status_text}")
        _set_accessible_text(self.locality_demand, f"{detail.demand_mw:.1f} MW")
        _set_accessible_text(self.locality_served, f"{detail.served_mw:.1f} MW")
        _set_accessible_text(
            self.locality_critical,
            f"{detail.critical_service_ratio:.1%}",
        )
        self._show_projection(frame, detail.next_hour_projected_demand_mw)
        for sector, ratio in detail.sector_service_ratio.items():
            _set_accessible_text(self.sector_service_labels[sector], f"{ratio:.1%}")

    def _show_empty_state(self) -> None:
        for label, text in (
            (self.locality_context, "No controller output · Hour —"),
            (self.locality_status, "Status: not run"),
            (self.locality_demand, "—"),
            (self.locality_served, "—"),
            (self.locality_critical, "—"),
            (
                self.locality_projection,
                "Next hour: — · DISPLAY ONLY\nNot PPO input/evidence.",
            ),
        ):
            _set_accessible_text(label, text)
        for label in self.sector_service_labels.values():
            _set_accessible_text(label, "—")

    def _show_projection(
        self,
        frame: LiveScenarioFrame,
        projected_demand_mw: float | None,
    ) -> None:
        if projected_demand_mw is None:
            reason = "needs 3-hour history" if frame.hour < 2 else "end of trace"
            text = f"Next hour: unavailable ({reason}) · DISPLAY ONLY\nNot PPO input/evidence."
        else:
            text = (
                f"Next hour {frame.hour + 1:02d}:00: {projected_demand_mw:.1f} MW "
                "· DISPLAY ONLY\nTrailing 3-hour mean; not PPO input/evidence."
            )
        _set_accessible_text(self.locality_projection, text)


class LiveLocalityPanel(QWidget):
    """Bidirectionally synchronized locality list and selected-detail card."""

    def __init__(self, asset: MapAsset, map_view: LocalityMapView, parent=None) -> None:
        super().__init__(parent)
        self.map_view = map_view
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.locality_list = self._build_locality_list(asset)
        self.locality_list.currentItemChanged.connect(self._select_locality_from_list)
        selected_id = map_view.selected_locality_id or asset.keyboard_order[0]
        self.sync_locality_list(selected_id)
        layout.addWidget(self.locality_list)

        self.locality_card = LocalityDetailCard()
        self._expose_detail_aliases()
        layout.addWidget(self.locality_card, 1)
        layout.addWidget(StatusLegend(compact=True))

    @staticmethod
    def _build_locality_list(asset: MapAsset) -> QListWidget:
        locality_list = QListWidget()
        locality_list.setProperty("localityList", True)
        locality_list.setAccessibleName("Khartoum localities")
        locality_list.setAccessibleDescription(
            "Select a locality to synchronize the administrative map and detail panel."
        )
        locality_list.setFlow(QListView.Flow.LeftToRight)
        locality_list.setWrapping(True)
        locality_list.setResizeMode(QListView.ResizeMode.Adjust)
        locality_list.setMovement(QListView.Movement.Static)
        locality_list.setTextElideMode(Qt.TextElideMode.ElideRight)
        locality_cell_size = QSize(208, 23)
        locality_list.setGridSize(locality_cell_size)
        locality_list.setMinimumWidth((locality_cell_size.width() * 2) + 6)
        locality_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        locality_list.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        locality_list.setFixedHeight(95)
        localities = {locality.locality_id: locality for locality in asset.localities}
        for locality_id in asset.keyboard_order:
            item = QListWidgetItem(f"{localities[locality_id].name} — not run")
            item.setData(Qt.ItemDataRole.UserRole, locality_id)
            item.setSizeHint(locality_cell_size)
            item.setToolTip(item.text())
            locality_list.addItem(item)
        return locality_list

    def _expose_detail_aliases(self) -> None:
        self.locality_name = self.locality_card.locality_name
        self.locality_context = self.locality_card.locality_context
        self.locality_status = self.locality_card.locality_status
        self.locality_demand = self.locality_card.locality_demand
        self.locality_served = self.locality_card.locality_served
        self.locality_critical = self.locality_card.locality_critical
        self.locality_projection = self.locality_card.locality_projection
        self.sector_service_labels = self.locality_card.sector_service_labels

    def refresh(
        self,
        frame: LiveScenarioFrame,
        controller_name: str,
    ) -> None:
        for row in range(self.locality_list.count()):
            list_item = self.locality_list.item(row)
            locality_id = list_item.data(Qt.ItemDataRole.UserRole)
            map_item = self.map_view.locality_items[locality_id]
            list_item.setText(f"{map_item.locality.name} — {map_item.status_text}")
            list_item.setToolTip(list_item.text())
        self.update_selected_locality(
            self.map_view.selected_locality_id or "khartoum",
            frame,
            controller_name,
        )

    def reset_output(self) -> None:
        """Clear locality tiers and values without inventing replacement output."""

        for row in range(self.locality_list.count()):
            list_item = self.locality_list.item(row)
            locality_id = list_item.data(Qt.ItemDataRole.UserRole)
            locality_name = self.map_view.locality_items[locality_id].locality.name
            list_item.setText(f"{locality_name} \N{EM DASH} not run")
            list_item.setToolTip(list_item.text())
        selected_locality_id = (
            self.map_view.selected_locality_id or self.map_view.asset.keyboard_order[0]
        )
        self.update_selected_locality(
            selected_locality_id,
            None,
            ControllerName.PPO.value,
        )

    def update_selected_locality(
        self,
        locality_id: str,
        frame: LiveScenarioFrame | None,
        controller_name: str,
    ) -> None:
        self.sync_locality_list(locality_id)
        self.locality_card.update_details(
            self.map_view.locality_items[locality_id],
            frame,
            controller_name,
        )

    def _select_locality_from_list(
        self,
        current: QListWidgetItem | None,
        _previous: QListWidgetItem | None,
    ) -> None:
        if current is None:
            return
        locality_id = current.data(Qt.ItemDataRole.UserRole)
        if locality_id != self.map_view.selected_locality_id:
            self.map_view.select_locality(locality_id)

    def sync_locality_list(self, locality_id: str) -> None:
        for row in range(self.locality_list.count()):
            if self.locality_list.item(row).data(Qt.ItemDataRole.UserRole) == locality_id:
                self.locality_list.blockSignals(True)
                self.locality_list.setCurrentRow(row)
                self.locality_list.blockSignals(False)
                return

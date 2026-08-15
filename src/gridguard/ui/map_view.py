"""Keyboard- and pointer-accessible offline schematic map view."""

from __future__ import annotations

from collections.abc import Mapping

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QKeyEvent, QPainter, QPainterPath, QPen, QWheelEvent
from PySide6.QtWidgets import QGraphicsScene, QGraphicsView

from gridguard.domain import ServiceTier
from gridguard.ui.map_asset import MapAsset
from gridguard.ui.map_items import LocalityGraphicsItem
from gridguard.ui.theme import PALETTE


class SchematicMapView(QGraphicsView):
    """Interactive map with selection, tooltips, panning, and bounded zoom."""

    locality_selected = Signal(str)

    def __init__(self, asset: MapAsset, parent=None) -> None:
        self._scene = QGraphicsScene(parent)
        super().__init__(self._scene, parent)
        self.asset = asset
        self.locality_items: dict[str, LocalityGraphicsItem] = {}
        self._zoom_steps = 0
        self._build_scene()

        rect = asset.scene_rect
        self._scene.setSceneRect(rect.x, rect.y, rect.width, rect.height)
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
        self.setBackgroundBrush(QColor(PALETTE.surface))
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName("Interactive Khartoum locality schematic")
        self.setAccessibleDescription(
            f"{asset.disclaimer} Select a locality with a pointer or arrow keys; "
            "use plus and minus to zoom, and R to reset the view."
        )
        self.setToolTip("Wheel or plus/minus: zoom. Drag background: pan. R: reset.")
        self.select_locality(asset.keyboard_order[0])

    @property
    def selected_locality_id(self) -> str | None:
        selected = self._scene.selectedItems()
        if not selected:
            return None
        item = selected[0]
        return item.locality.locality_id if isinstance(item, LocalityGraphicsItem) else None

    def select_locality(self, locality_id: str) -> None:
        """Select a validated locality and move keyboard focus to its item."""

        try:
            item = self.locality_items[locality_id]
        except KeyError as error:
            raise KeyError(f"unknown locality: {locality_id}") from error
        if item.isSelected():
            item.setFocus()
            return
        self._scene.clearSelection()
        item.setSelected(True)
        item.setFocus()
        self.ensureVisible(item, 24, 24)
        self.locality_selected.emit(locality_id)

    def set_service_tiers(self, tiers: Mapping[str, ServiceTier | int]) -> None:
        """Update any subset of localities while rejecting unknown identifiers."""

        unknown = set(tiers) - set(self.locality_items)
        if unknown:
            raise KeyError(f"unknown locality tier(s): {', '.join(sorted(unknown))}")
        for locality_id, tier in tiers.items():
            self.locality_items[locality_id].set_service_tier(ServiceTier(tier))

    def reset_view(self) -> None:
        """Fit the complete schematic within the viewport."""

        self._zoom_steps = 0
        bounds = self._scene.sceneRect().adjusted(-24, -24, 24, 24)
        self.fitInView(bounds, Qt.AspectRatioMode.KeepAspectRatio)

    def zoom_in(self) -> None:
        self._zoom(1)

    def zoom_out(self) -> None:
        self._zoom(-1)

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        delta = event.angleDelta().y()
        if delta == 0:
            super().wheelEvent(event)
            return
        self._zoom(1 if delta > 0 else -1)
        event.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        direction = {
            Qt.Key.Key_Up: "up",
            Qt.Key.Key_Right: "right",
            Qt.Key.Key_Down: "down",
            Qt.Key.Key_Left: "left",
        }.get(event.key())
        if direction is not None:
            self._navigate(direction)
            event.accept()
            return
        if event.key() in {Qt.Key.Key_Plus, Qt.Key.Key_Equal}:
            self.zoom_in()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Minus:
            self.zoom_out()
            event.accept()
            return
        if event.key() in {Qt.Key.Key_R, Qt.Key.Key_0}:
            self.reset_view()
            event.accept()
            return
        if event.key() in {Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space}:
            selected = self.selected_locality_id
            if selected is not None:
                self.locality_selected.emit(selected)
            event.accept()
            return
        super().keyPressEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._zoom_steps == 0:
            self.reset_view()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        previous = self.selected_locality_id
        super().mousePressEvent(event)
        selected = self.selected_locality_id
        if selected is not None and selected != previous:
            self.locality_selected.emit(selected)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        previous = self.selected_locality_id
        super().mouseReleaseEvent(event)
        selected = self.selected_locality_id
        if selected is not None and selected != previous:
            self.locality_selected.emit(selected)

    def _build_scene(self) -> None:
        river_pen = QPen(QColor(PALETTE.river), 10, Qt.PenStyle.SolidLine)
        river_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        river_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        for river in self.asset.river_context.paths:
            path = QPainterPath()
            path.moveTo(*river.points[0])
            for point in river.points[1:]:
                path.lineTo(*point)
            river_item = self._scene.addPath(path, river_pen)
            river_item.setZValue(-2)
            river_item.setToolTip(river.name)

        by_id = {item.locality_id: item for item in self.asset.localities}
        for locality_id in self.asset.keyboard_order:
            item = LocalityGraphicsItem(by_id[locality_id])
            self.locality_items[locality_id] = item
            self._scene.addItem(item)

    def _navigate(self, direction: str) -> None:
        current_id = self.selected_locality_id or self.asset.keyboard_order[0]
        neighbors = self.locality_items[current_id].locality.neighbors
        destination = getattr(neighbors, direction)
        if destination is not None:
            self.select_locality(destination)

    def _zoom(self, step: int) -> None:
        next_step = self._zoom_steps + step
        if not -4 <= next_step <= 8:
            return
        factor = 1.16 if step > 0 else 1 / 1.16
        self.scale(factor, factor)
        self._zoom_steps = next_step

"""Graphics items used by the interactive Khartoum locality map."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsItemGroup,
    QGraphicsLineItem,
    QGraphicsPathItem,
    QGraphicsRectItem,
    QGraphicsSimpleTextItem,
    QStyle,
)

from gridguard.domain import ServiceTier
from gridguard.ui.map_asset import LocalityShape
from gridguard.ui.theme import BODY_FONT_FAMILY, PALETTE, STATUS_COLORS, STATUS_TEXT

_LABEL_OFFSETS = {
    "khartoum": (100.0, -18.0),
    "omdurman": (-100.0, 12.0),
    "bahri": (60.0, -20.0),
    "east_nile": (93.0, -13.0),
    "jebel_awlia": (85.0, 32.0),
    "karari": (-70.0, -18.0),
    "ombada": (-10.0, -85.0),
}


class LocalityGraphicsItem(QGraphicsPathItem):
    """Selectable Polygon/MultiPolygon locality with redundant status cues."""

    def __init__(self, locality: LocalityShape) -> None:
        super().__init__(_boundary_path(locality))
        self.locality = locality
        self.status_text = "— NOT RUN"
        self._hovered = False
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemIsFocusable
        )
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setBrush(QBrush(QColor(PALETTE.surface_raised)))
        self.setToolTip(self._tooltip())
        self._update_pen()

        self._label_leader = QGraphicsLineItem(self)
        leader_pen = QPen(QColor(PALETTE.boundary), 1, Qt.PenStyle.DashLine)
        leader_pen.setCosmetic(True)
        self._label_leader.setPen(leader_pen)
        self._label_leader.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._label_leader.setZValue(3)

        self._label_group = QGraphicsItemGroup(self)
        self._label_group.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self._label_group.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._label_group.setZValue(4)

        self._label_backing = QGraphicsRectItem(self._label_group)
        self._label_backing.setBrush(QBrush(QColor(PALETTE.surface)))
        backing_pen = QPen(QColor(PALETTE.boundary), 0.75)
        backing_pen.setCosmetic(True)
        self._label_backing.setPen(backing_pen)
        self._label_backing.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._label_backing.setZValue(-1)

        self._name = QGraphicsSimpleTextItem(locality.name, self._label_group)
        name_font = QFont(BODY_FONT_FAMILY)
        name_font.setPointSizeF(9.5)
        name_font.setWeight(QFont.Weight.DemiBold)
        self._name.setFont(name_font)
        self._name.setBrush(QColor(PALETTE.text))
        self._name.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

        self._status = QGraphicsSimpleTextItem(self.status_text, self._label_group)
        status_font = QFont(BODY_FONT_FAMILY)
        status_font.setPointSizeF(7.5)
        status_font.setWeight(QFont.Weight.Bold)
        self._status.setFont(status_font)
        self._status.setBrush(QColor(PALETTE.text_muted))
        self._status.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._position_labels()

    def set_service_tier(self, tier: ServiceTier) -> None:
        """Apply a tier using color, texture, symbol, and text together."""

        normalized = ServiceTier(tier)
        self.status_text = STATUS_TEXT[normalized.value]
        brush_style = {
            ServiceTier.OFF: Qt.BrushStyle.BDiagPattern,
            ServiceTier.CRITICAL_ONLY: Qt.BrushStyle.Dense4Pattern,
            ServiceTier.FULL_SERVICE: Qt.BrushStyle.SolidPattern,
        }[normalized]
        self.setBrush(QBrush(STATUS_COLORS[normalized.value], brush_style))
        self._status.setText(self.status_text)
        self._name.setBrush(QColor(PALETTE.ink))
        self._status.setBrush(STATUS_COLORS[normalized.value])
        self.setToolTip(self._tooltip())
        self._position_labels()

    def hoverEnterEvent(self, event) -> None:  # noqa: N802
        self._hovered = True
        self._update_pen()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:  # noqa: N802
        self._hovered = False
        self._update_pen()
        super().hoverLeaveEvent(event)

    def itemChange(self, change, value):  # noqa: N802
        result = super().itemChange(change, value)
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            self._update_pen(selected=bool(value))
        return result

    def _position_labels(self) -> None:
        anchor_x, anchor_y = self.locality.label_anchor
        offset_x, offset_y = _LABEL_OFFSETS[self.locality.locality_id]
        target_x = anchor_x + offset_x
        target_y = anchor_y + offset_y
        name_bounds = self._name.boundingRect()
        status_bounds = self._status.boundingRect()
        self._label_group.setPos(target_x, target_y)
        self._name.setPos(-name_bounds.width() / 2, -name_bounds.height() - 1)
        self._status.setPos(-status_bounds.width() / 2, 1)
        name_rect = self._name.mapRectToParent(name_bounds)
        status_rect = self._status.mapRectToParent(status_bounds)
        text_rect = name_rect.united(status_rect)
        self._label_backing.setRect(text_rect.adjusted(-5, -3, 5, 3))
        self._label_leader.setLine(anchor_x, anchor_y, target_x, target_y)

    def _update_pen(self, *, selected: bool | None = None) -> None:
        is_selected = self.isSelected() if selected is None else selected
        if is_selected:
            color, width = QColor(PALETTE.focus), 5
        elif self._hovered:
            color, width = QColor(PALETTE.accent), 4
        else:
            color, width = QColor(PALETTE.border), 3
        pen = QPen(color, width)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        self.setPen(pen)

    def _tooltip(self) -> str:
        return (
            f"{self.locality.name} ({self.locality.adm2_pcode})\n"
            f"Status: {self.status_text}\n"
            "Click to select; use arrow keys to move between localities."
        )

    def paint(self, painter: QPainter, option, widget=None) -> None:
        """Suppress Qt's color-only default selection overlay."""

        option.state &= ~QStyle.StateFlag.State_Selected
        super().paint(painter, option, widget)


def _boundary_path(locality: LocalityShape) -> QPainterPath:
    """Build one odd-even path so multipart features and interior holes stay exact."""

    path = QPainterPath()
    path.setFillRule(Qt.FillRule.OddEvenFill)
    for polygon in locality.geometry.polygons:
        for ring in (polygon.outer, *polygon.holes):
            first, *remaining = ring
            path.moveTo(*first)
            for point in remaining:
                path.lineTo(*point)
            path.closeSubpath()
    return path

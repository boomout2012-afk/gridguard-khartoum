"""Graphics items used by the interactive Khartoum locality map."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsSimpleTextItem,
    QStyle,
)

from gridguard.domain import ServiceTier
from gridguard.ui.map_asset import LocalityShape
from gridguard.ui.theme import BODY_FONT_FAMILY, PALETTE, STATUS_COLORS, STATUS_TEXT


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

        self._name = QGraphicsSimpleTextItem(locality.name, self)
        name_font = QFont(BODY_FONT_FAMILY, 13, QFont.Weight.DemiBold)
        self._name.setFont(name_font)
        self._name.setBrush(QColor(PALETTE.text))
        self._name.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._name.setZValue(4)

        self._status = QGraphicsSimpleTextItem(self.status_text, self)
        status_font = QFont(BODY_FONT_FAMILY, 9, QFont.Weight.Bold)
        self._status.setFont(status_font)
        self._status.setBrush(QColor(PALETTE.text_muted))
        self._status.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._status.setZValue(4)
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
        high_contrast_text = QColor(PALETTE.surface)
        self._name.setBrush(high_contrast_text)
        self._status.setBrush(high_contrast_text)
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
        name_bounds = self._name.boundingRect()
        status_bounds = self._status.boundingRect()
        self._name.setPos(anchor_x - name_bounds.width() / 2, anchor_y - 25)
        self._status.setPos(anchor_x - status_bounds.width() / 2, anchor_y + 3)

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

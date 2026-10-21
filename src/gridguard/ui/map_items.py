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
from gridguard.ui.theme import BODY_FONT_FAMILY, PALETTE, STATUS_COLORS, STATUS_FILL_COLORS

_LABEL_OFFSETS = {
    "khartoum": (132.0, 0.0),
    "omdurman": (-93.0, 17.0),
    "bahri": (10.0, -4.0),
    "east_nile": (70.0, -16.0),
    "jebel_awlia": (116.0, 45.0),
    "karari": (0.0, -20.0),
    "ombada": (8.0, -25.0),
}


class LocalityGraphicsItem(QGraphicsPathItem):
    """Pinned geometry with independent service state and coral selection."""

    def __init__(self, locality: LocalityShape) -> None:
        super().__init__(_boundary_path(locality))
        self.locality = locality
        self.service_tier: ServiceTier | None = None
        self.status_text = "Not run"
        self._hovered = False
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemIsFocusable
        )
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._update_fill()
        self.setToolTip(self._tooltip())
        self._update_pen()

        self._label_leader = QGraphicsLineItem(self)
        leader_pen = QPen(QColor("#7D9589"), 1, Qt.PenStyle.SolidLine)
        leader_pen.setCosmetic(True)
        self._label_leader.setPen(leader_pen)
        self._label_leader.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._label_leader.setZValue(3)
        self._label_leader.setVisible(
            locality.locality_id in {"khartoum", "omdurman", "jebel_awlia"}
        )

        self._label_group = QGraphicsItemGroup(self)
        self._label_group.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self._label_group.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._label_group.setZValue(4)

        self._label_backing = QGraphicsRectItem(self._label_group)
        self._label_backing.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self._label_backing.setPen(QPen(Qt.PenStyle.NoPen))
        self._label_backing.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._label_backing.setZValue(-1)

        self._name = _MapNameItem(locality.name, self._label_group)
        name_font = QFont(BODY_FONT_FAMILY)
        name_font.setPixelSize(14)
        self._name.setFont(name_font)
        self._name.setBrush(QColor(PALETTE.text))
        self._name.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

        self._status = QGraphicsSimpleTextItem(self.status_text, self._label_group)
        status_font = QFont(BODY_FONT_FAMILY)
        status_font.setPixelSize(12)
        status_font.setWeight(QFont.Weight.Bold)
        self._status.setFont(status_font)
        self._status.setBrush(QColor(PALETTE.text_muted))
        self._status.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._status.setVisible(False)
        self._position_labels()

    def set_service_tier(self, tier: ServiceTier) -> None:
        """Keep the actual service tier when selection temporarily uses coral."""

        normalized = ServiceTier(tier)
        self.service_tier = normalized
        self.status_text = normalized.label
        self._update_fill()
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
            self._update_fill(selected=bool(value))
            self.setToolTip(self._tooltip())
            if hasattr(self, "_name"):
                font = self._name.font()
                font.setBold(bool(value))
                self._name.setFont(font)
                self._position_labels()
        return result

    def _position_labels(self) -> None:
        anchor_x, anchor_y = self.locality.label_anchor
        offset_x, offset_y = _LABEL_OFFSETS[self.locality.locality_id]
        target_x = anchor_x + offset_x
        target_y = anchor_y + offset_y
        name_bounds = self._name.boundingRect()
        self._label_group.setPos(target_x, target_y)
        self._name.setPos(-name_bounds.width() / 2, -name_bounds.height() / 2)
        name_rect = self._name.mapRectToParent(name_bounds)
        self._label_backing.setRect(name_rect.adjusted(-2, -2, 2, 2))
        self._label_leader.setLine(anchor_x, anchor_y, target_x, target_y)

    def _update_pen(self, *, selected: bool | None = None) -> None:
        is_selected = self.isSelected() if selected is None else selected
        if is_selected:
            color, width = QColor("#A64C35"), 2
        elif self._hovered:
            color, width = QColor(PALETTE.accent), 2
        else:
            color, width = QColor(PALETTE.canvas), 1.5
        pen = QPen(color, width)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        pen.setCosmetic(True)
        self.setPen(pen)

    def _update_fill(self, *, selected: bool | None = None) -> None:
        is_selected = self.isSelected() if selected is None else selected
        if is_selected:
            color = QColor(PALETTE.selection)
        elif self.service_tier is None:
            color = QColor("#D3E4DB")
        else:
            color = STATUS_FILL_COLORS[self.service_tier.value]
        self.setBrush(QBrush(color))

    def _tooltip(self) -> str:
        return (
            f"{self.locality.name} ({self.locality.adm2_pcode})\n"
            f"Service tier: {self.status_text}\n"
            + (
                "Selected locality; coral denotes selection, not a service tier.\n"
                if self.isSelected()
                else ""
            )
            + "Click to select; use arrow keys to move between localities."
        )

    def paint(self, painter: QPainter, option, widget=None) -> None:
        """Suppress Qt's color-only default selection overlay."""

        option.state &= ~QStyle.StateFlag.State_Selected
        super().paint(painter, option, widget)


class _MapNameItem(QGraphicsSimpleTextItem):
    """Crisp device-sized text with a quiet halo instead of a boxed label."""

    def paint(self, painter: QPainter, option, widget=None) -> None:
        text_path = QPainterPath()
        text_path.addText(0, self.font().pixelSize() * 0.94, self.font(), self.text())
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        halo = QPen(QColor(PALETTE.canvas), 3.0)
        halo.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(halo)
        painter.setBrush(self.brush())
        painter.drawPath(text_path)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPath(text_path)


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

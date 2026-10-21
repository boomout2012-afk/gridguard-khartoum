"""A display-only trace of the already calculated live demand and supply frames."""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from gridguard.ui.live_models import LiveScenarioFrame
from gridguard.ui.theme import PALETTE


class PowerTrace(QWidget):
    """Render provided MW values; no forecasts, metrics, or simulation run here."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.frames: tuple[LiveScenarioFrame, ...] = ()
        self.selected_hour = 0
        self.setMinimumWidth(250)
        self.setFixedHeight(100)
        self.setAccessibleName("24-hour demand and available supply trace")
        self.clear()

    def set_frames(self, frames: Sequence[LiveScenarioFrame]) -> None:
        self.frames = tuple(sorted(frames, key=lambda frame: frame.hour))
        self.set_hour(self.selected_hour)

    def set_hour(self, hour: int) -> None:
        self.selected_hour = hour
        frame = next((frame for frame in self.frames if frame.hour == hour), None)
        if frame is not None:
            self.setAccessibleDescription(
                f"Hour {hour:02d}:00: demand {frame.demand_mw:,.1f} MW; "
                f"available supply {frame.supply_mw:,.1f} MW. "
                "Demand is a solid navy line; available supply is a dashed teal line."
            )
            self.setToolTip(self.accessibleDescription())
        self.update()

    def clear(self) -> None:
        self.frames = ()
        self.selected_hour = 0
        self.setAccessibleDescription("Live demand and supply unavailable. Run a scenario first.")
        self.setToolTip(self.accessibleDescription())
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(PALETTE.surface))
        painter.setPen(QColor(PALETTE.ink))
        painter.drawText(10, 16, "— Demand")
        painter.setPen(QColor(PALETTE.nile))
        painter.drawText(100, 16, "- - Supply")
        painter.setPen(QColor(PALETTE.muted))
        painter.drawText(self.width() - 31, 16, "MW")
        if not self.frames:
            painter.setPen(QColor(PALETTE.muted))
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, "Run a scenario to see its trace"
            )
            return
        plot = QRectF(12, 28, self.width() - 24, 48)
        ceiling = max(1.0, *(max(f.demand_mw, f.supply_mw) for f in self.frames))
        for attribute, color, style in (
            ("demand_mw", PALETTE.nav, Qt.PenStyle.SolidLine),
            ("supply_mw", PALETTE.nile, Qt.PenStyle.DashLine),
        ):
            path = QPainterPath()
            for index, frame in enumerate(self.frames):
                point = QPointF(
                    plot.left() + frame.hour / 23 * plot.width(),
                    plot.bottom() - getattr(frame, attribute) / ceiling * plot.height(),
                )
                if index == 0:
                    path.moveTo(point)
                else:
                    path.lineTo(point)
            painter.setPen(QPen(QColor(color), 2, style))
            painter.drawPath(path)
        x = plot.left() + self.selected_hour / 23 * plot.width()
        painter.setPen(QPen(QColor(PALETTE.focus), 1, Qt.PenStyle.DotLine))
        painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
        painter.setPen(QColor(PALETTE.muted))
        painter.drawText(10, 93, "00:00")
        painter.drawText(self.width() - 45, 93, "23:00")

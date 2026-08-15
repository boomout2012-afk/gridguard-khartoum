"""Small, reusable native presentation components."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QVBoxLayout

from gridguard.ui.presentation_model import MetricComparison, PresentationSnapshot
from gridguard.ui.theme import STATUS_COLORS, STATUS_TEXT


class CardFrame(QFrame):
    """Consistent bordered surface with an exposed content layout."""

    def __init__(self, *, accessible_name: str, parent=None) -> None:
        super().__init__(parent)
        self.setProperty("card", True)
        self.setAccessibleName(accessible_name)
        self.content = QVBoxLayout(self)
        self.content.setContentsMargins(16, 14, 16, 14)
        self.content.setSpacing(10)


def page_intro(title: str, description: str) -> tuple[QLabel, QLabel]:
    heading = QLabel(title)
    heading.setProperty("heading", "page")
    heading.setAccessibleName(f"Page title: {title}")
    body = QLabel(description)
    body.setProperty("muted", True)
    body.setWordWrap(True)
    return heading, body


class ControllerCard(CardFrame):
    """Compact controller summary sourced only from an evidence snapshot."""

    def __init__(
        self,
        snapshot: PresentationSnapshot,
        controller_name: str,
        parent=None,
    ) -> None:
        if controller_name not in snapshot.controller_names:
            raise ValueError(f"unsupported controller: {controller_name}")
        super().__init__(accessible_name=f"{controller_name} aggregate results", parent=parent)
        self.controller_name = controller_name

        heading = QLabel(controller_name)
        heading.setProperty("heading", "section")
        descriptor = QLabel(
            "Learned dispatch policy" if controller_name == "PPO" else "Scheduled baseline"
        )
        descriptor.setProperty("muted", True)
        self.content.addWidget(heading)
        self.content.addWidget(descriptor)

        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(4)
        for column, key in enumerate(("critical_service_ratio", "served_mwh", "episode_reward")):
            try:
                metric = snapshot.metric(key)
            except KeyError:
                continue
            value = metric.ppo if controller_name == "PPO" else metric.fixed_rotation
            value_label = QLabel(format_metric_value(metric, value))
            value_label.setProperty("metric", "value")
            value_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
            name_label = QLabel(metric.label)
            name_label.setProperty("muted", True)
            grid.addWidget(value_label, 0, column)
            grid.addWidget(name_label, 1, column)
        self.content.addLayout(grid)


class StatusLegend(CardFrame):
    """Service status key with color, texture name, symbol, and text."""

    def __init__(self, parent=None) -> None:
        super().__init__(accessible_name="Service status legend", parent=parent)
        self.content.setContentsMargins(12, 10, 12, 10)
        self.content.setSpacing(4)
        title = QLabel("Service status")
        title.setProperty("heading", "section")
        self.content.addWidget(title)
        textures = ("hatched", "patterned", "solid")
        status_row = QHBoxLayout()
        status_row.setSpacing(8)
        for tier, texture in enumerate(textures):
            swatch = QFrame()
            swatch.setFixedSize(14, 14)
            swatch.setStyleSheet(
                f"background: {STATUS_COLORS[tier].name()}; border: 1px solid #F4F7F8;"
            )
            swatch.setAccessibleName(f"{STATUS_TEXT[tier]} color swatch; {texture}")
            label = QLabel(STATUS_TEXT[tier])
            label.setProperty("compact", True)
            label.setToolTip(f"{STATUS_TEXT[tier]} areas use a {texture} map fill")
            status_row.addWidget(swatch)
            status_row.addWidget(label)
        status_row.addStretch(1)
        self.content.addLayout(status_row)


def format_metric_value(
    metric: MetricComparison,
    value: float,
    *,
    signed: bool = False,
    as_delta: bool = False,
) -> str:
    """Format a scientific value using display metadata, never recomputation."""

    scaled = value * metric.display_scale
    sign = "+" if signed and scaled > 0 else ""
    if metric.unit == "%":
        if as_delta:
            return f"{sign}{scaled:.2f} pp"
        return f"{sign}{scaled:.2f}%"
    if metric.unit == "MWh":
        return f"{sign}{scaled:,.0f} MWh"
    return f"{sign}{scaled:.2f}"

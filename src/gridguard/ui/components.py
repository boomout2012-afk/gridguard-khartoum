"""Small, reusable native presentation components."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QVBoxLayout

from gridguard.ui.presentation_model import MetricComparison, PresentationSnapshot
from gridguard.ui.theme import PALETTE, STATUS_COLORS, STATUS_TEXT


class CardFrame(QFrame):
    """Consistent bordered surface with an exposed content layout."""

    def __init__(
        self,
        *,
        accessible_name: str,
        variant: str = "default",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setProperty("card", True)
        self.setProperty("cardVariant", variant)
        self.setAccessibleName(accessible_name)
        self.content = QVBoxLayout(self)
        self.content.setContentsMargins(16, 14, 16, 14)
        self.content.setSpacing(10)


def page_intro(title: str, description: str) -> tuple[QLabel, QLabel]:
    """Build a consistently announced page title and purpose statement."""

    heading = QLabel(title)
    heading.setProperty("heading", "page")
    heading.setAccessibleName(f"Page title: {title}")
    body = QLabel(description)
    body.setProperty("muted", True)
    body.setWordWrap(True)
    body.setAccessibleName(f"Page summary: {description}")
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
        heading.setAccessibleName(f"Controller: {controller_name}")
        descriptor = QLabel(
            "Learned dispatch policy" if controller_name == "PPO" else "Scheduled baseline"
        )
        descriptor.setProperty("muted", True)
        descriptor.setAccessibleName(f"{controller_name} role: {descriptor.text()}")
        heading_row = QHBoxLayout()
        heading_row.addWidget(heading)
        heading_row.addStretch(1)
        role_badge = QLabel("LEARNED" if controller_name == "PPO" else "BASELINE")
        role_badge.setProperty("badge", "locked" if controller_name == "PPO" else "neutral")
        role_badge.setProperty("compact", True)
        role_badge.setAccessibleName(f"{controller_name} controller category")
        heading_row.addWidget(role_badge)
        self.content.addLayout(heading_row)
        self.content.addWidget(descriptor)

        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(4)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)
        self.metric_value_labels: dict[str, QLabel] = {}
        displayed_metric_count = 0
        for key in ("critical_service_ratio", "served_mwh", "episode_reward"):
            try:
                metric = snapshot.metric(key)
            except KeyError:
                continue
            value = metric.ppo if controller_name == "PPO" else metric.fixed_rotation
            formatted_value = format_metric_value(metric, value)
            value_label = QLabel(formatted_value)
            value_label.setProperty("metric", "value")
            value_label.setProperty("mono", True)
            value_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
            value_label.setAccessibleName(f"{controller_name}, {metric.label}: {formatted_value}")
            name_label = QLabel(metric.label)
            name_label.setProperty("muted", True)
            name_label.setProperty("compact", True)
            column = displayed_metric_count
            grid.addWidget(value_label, 0, column)
            grid.addWidget(name_label, 1, column)
            self.metric_value_labels[key] = value_label
            displayed_metric_count += 1
        if displayed_metric_count:
            self.content.addLayout(grid)
            self.setAccessibleDescription(
                f"{displayed_metric_count} verified aggregate metrics for {controller_name}."
            )
        else:
            self.state_label = QLabel(
                "No approved headline metrics are available in this evidence snapshot."
            )
            self.state_label.setProperty("state", "empty")
            self.state_label.setWordWrap(True)
            self.state_label.setAccessibleName(f"{controller_name} aggregate result is unavailable")
            self.content.addWidget(self.state_label)
            self.setAccessibleDescription(
                f"The verified snapshot supplied no headline metrics for {controller_name}."
            )


class StatusLegend(CardFrame):
    """Service status key with color, texture name, symbol, and text."""

    def __init__(self, parent=None, *, compact: bool = False) -> None:
        super().__init__(accessible_name="Service status legend", parent=parent)
        horizontal_margin = 8 if compact else 12
        vertical_margin = 5 if compact else 10
        self.content.setContentsMargins(
            horizontal_margin,
            vertical_margin,
            horizontal_margin,
            vertical_margin,
        )
        self.content.setSpacing(2 if compact else 4)
        if not compact:
            title = QLabel("Service status")
            title.setProperty("heading", "section")
            self.content.addWidget(title)
        textures = ("hatched", "patterned", "solid")
        status_row = QHBoxLayout()
        status_row.setSpacing(5 if compact else 8)
        for tier, texture in enumerate(textures):
            swatch = QFrame()
            swatch_size = 12 if compact else 14
            swatch.setFixedSize(swatch_size, swatch_size)
            swatch.setStyleSheet(
                f"background: {STATUS_COLORS[tier].name()}; border: 1px solid {PALETTE.surface};"
            )
            swatch.setAccessibleName(f"{STATUS_TEXT[tier]} color swatch; {texture}")
            label = QLabel(STATUS_TEXT[tier])
            label.setProperty("compact", True)
            label.setAccessibleName(f"Service tier: {STATUS_TEXT[tier]}; {texture} map fill")
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

"""Small, reusable native presentation components."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QVBoxLayout

from gridguard.claim_contract import HOLDOUT_CUSTODY_LIMITATION, RESULTS_SCOPE_LIMITATION
from gridguard.ui.presentation_model import MetricComparison, PresentationSnapshot
from gridguard.ui.theme import PALETTE, STATUS_FILL_COLORS


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


class ClaimBoundaryNote(QLabel):
    """Compact visible claim boundary with the full canonical text for assistive access."""

    def __init__(self, *, include_holdout_custody: bool = False, parent=None) -> None:
        visible_parts = []
        canonical_parts = []
        if include_holdout_custody:
            visible_parts.append(
                "Holdout not preregistered or independently custodied; undocumented activity "
                "cannot be ruled out."
            )
            canonical_parts.append(HOLDOUT_CUSTODY_LIMITATION)
        visible_parts.append(
            "Synthetic decision support only. Fixed Rotation is supply-blind (uses only Off and "
            "Full Service; "
            "cannot select Critical Only). No behavior-cloning-only ablation isolates PPO "
            "fine-tuning."
        )
        canonical_parts.append(RESULTS_SCOPE_LIMITATION)
        canonical_text = " ".join(canonical_parts)

        super().__init__(" ".join(visible_parts), parent)
        self.setProperty("claim", "boundary")
        self.setWordWrap(True)
        self.setAccessibleName(f"Scientific claim boundary: {canonical_text}")
        self.setToolTip(canonical_text)


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
        self.content.setContentsMargins(16, 10, 16, 10)
        self.content.setSpacing(5)

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
        self.metric_value_labels: dict[str, QLabel] = {}
        displayed_metric_count = 0
        for metric in snapshot.metrics:
            key = metric.key
            value = metric.ppo if controller_name == "PPO" else metric.fixed_rotation
            formatted_value = format_metric_value(
                metric, value, include_unit=metric.unit != "locality-hours"
            )
            value_label = QLabel(formatted_value)
            value_label.setProperty("metric", "value")
            value_label.setMinimumHeight(34)
            value_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
            value_label.setAccessibleName(f"{controller_name}, {metric.label}: {formatted_value}")
            name_label = QLabel(metric.label)
            name_label.setProperty("muted", True)
            name_label.setProperty("compact", True)
            name_label.setAccessibleName(f"{metric.label} headline metric")
            if displayed_metric_count == 0:
                value_row, name_row, column, column_span = 0, 1, 0, 2
            else:
                value_row, name_row = 2, 3
                column = displayed_metric_count - 1
                column_span = 1
            grid.addWidget(value_label, value_row, column, 1, column_span)
            grid.addWidget(name_label, name_row, column, 1, column_span)
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


class StatusLegend(QFrame):
    """Quiet service-tier and selection key on the open map canvas."""

    def __init__(self, parent=None, *, compact: bool = False) -> None:
        super().__init__(parent)
        self.setAccessibleName("Service status legend")
        self.setStyleSheet("QFrame { background: transparent; border: 0; }")
        self.content = QVBoxLayout(self)
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
        status_row = QHBoxLayout()
        status_row.setSpacing(5 if compact else 8)
        entries = (
            ("Full Service", STATUS_FILL_COLORS[2].name()),
            ("Critical Only", STATUS_FILL_COLORS[1].name()),
            ("Off", STATUS_FILL_COLORS[0].name()),
            ("Selected locality", PALETTE.selection),
        )
        for name, color in entries:
            swatch = QFrame()
            swatch_size = 12 if compact else 14
            swatch.setFixedSize(swatch_size, swatch_size)
            swatch.setStyleSheet(
                f"background: {color}; border: 1px solid #8EADA0; border-radius: 3px;"
            )
            swatch.setAccessibleName(f"{name} map key")
            label = QLabel(name)
            label.setProperty("compact", True)
            label.setAccessibleName(f"Map legend: {name}")
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
    include_unit: bool = True,
) -> str:
    """Format a scientific value using display metadata, never recomputation."""

    scaled = value * metric.display_scale
    sign = "+" if signed and scaled > 0 else ""
    if metric.unit == "%":
        if as_delta:
            suffix = " pp" if include_unit else ""
            return f"{sign}{scaled:.2f}{suffix}"
        suffix = "%" if include_unit else ""
        return f"{sign}{scaled:.2f}{suffix}"
    if metric.unit == "locality-hours":
        suffix = " locality-hours" if include_unit else ""
        return f"{sign}{scaled:.1f}{suffix}"
    if metric.unit == "MWh":
        suffix = " MWh" if include_unit else ""
        return f"{sign}{scaled:,.1f}{suffix}"
    return f"{sign}{scaled:.2f}"

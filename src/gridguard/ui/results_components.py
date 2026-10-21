"""Native display-only components for the frozen controller comparison."""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from gridguard.claim_contract import (
    HOLDOUT_CUSTODY_LIMITATION,
    RESULTS_SCOPE_LIMITATION,
    SENSITIVITY_INTERVAL_LIMITATION,
)
from gridguard.ui.components import format_metric_value
from gridguard.ui.presentation_model import MetricComparison, PresentationSnapshot
from gridguard.ui.theme import BODY_FONT_FAMILY

INK = "#183D40"
MUTED = "#536A67"
PRIMARY = "#126F68"
LINE = "#D9E4DF"


def result_label(
    text: str, *, size: int = 14, bold: bool = False, color: str = INK, wrap: bool = False
) -> QLabel:
    """Create legible labels independently of legacy page style properties."""
    label = QLabel(text)
    label.setWordWrap(wrap)
    label.setStyleSheet(
        f'font-family: "{BODY_FONT_FAMILY}"; font-size: {size}px; '
        f"font-weight: {700 if bold else 400}; color: {color}; "
        "background: transparent; border: none; padding: 0; margin: 0;"
    )
    label.setAccessibleName(text)
    return label


class ComparisonBar(QWidget):
    """Paint a supplied value on a fixed display scale without scientific aggregation."""

    def __init__(
        self,
        *,
        value: float,
        maximum: float,
        color: str,
        accessible_name: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.raw_value = value
        self.display_maximum = maximum
        self._color = QColor(color)
        self.setFixedHeight(9)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setAccessibleName(accessible_name)
        self.setAccessibleDescription("Frozen evidence on a zero-based display scale.")

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#EDF2EF"))
        track = QRectF(self.rect())
        painter.drawRoundedRect(track, 4, 4)
        fraction = min(1.0, max(0.0, self.raw_value / self.display_maximum))
        if fraction:
            painter.setBrush(self._color)
            painter.drawRoundedRect(QRectF(0, 0, track.width() * fraction, track.height()), 4, 4)


class OutcomeComparisonPanel(QWidget):
    """Two controller values for one authoritative primary outcome."""

    def __init__(self, metric: MetricComparison, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.comparison = metric
        self.setAccessibleName(f"{metric.label} controller comparison")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 16, 28, 16)
        layout.setSpacing(0)
        self.title_label = result_label(
            metric.label.replace("Critical-outage ", "Critical-outage\n"), size=23, bold=True
        )
        self.title_label.setFixedHeight(max(60, self.title_label.sizeHint().height()))
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.title_label)
        direction = (
            "Lower is better" if metric.direction == "lower_is_better" else "Higher is better"
        )
        self.direction_label = result_label(direction, size=13, color=MUTED)
        layout.addWidget(self.direction_label)
        layout.addStretch(1)
        layout.addSpacing(9)
        self.value_labels: dict[str, QLabel] = {}
        self.bars: dict[str, ComparisonBar] = {}
        for controller, value, color, size in (
            ("PPO", metric.ppo, PRIMARY, 32),
            ("Fixed Rotation", metric.fixed_rotation, INK, 27),
        ):
            if self.value_labels:
                layout.addSpacing(12)
            formatted = format_metric_value(
                metric, value, include_unit=metric.unit != "locality-hours"
            )
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(12)
            row.addWidget(result_label(controller, color=color))
            value_label = result_label(formatted, size=size, bold=controller == "PPO", color=color)
            value_label.setProperty("metric", "value")
            value_label.setAccessibleName(f"{controller}, {metric.label}: {formatted}")
            row.addWidget(value_label, 1, Qt.AlignmentFlag.AlignRight)
            layout.addLayout(row)
            layout.addSpacing(5)
            bar = ComparisonBar(
                value=value,
                maximum=168.0 if metric.unit == "locality-hours" else 1.0,
                color=PRIMARY if controller == "PPO" else "#6B8780",
                accessible_name=f"{controller}, {metric.label}: {formatted}",
                parent=self,
            )
            layout.addWidget(bar)
            self.value_labels[controller] = value_label
            self.bars[controller] = bar
        layout.addSpacing(9)
        axis = QHBoxLayout()
        axis.addWidget(result_label("0" if metric.unit == "locality-hours" else "0%", size=12))
        maximum_text = (
            "168 locality-hours per episode" if metric.unit == "locality-hours" else "100%"
        )
        axis.addWidget(
            result_label(maximum_text, size=12, color=MUTED), 1, Qt.AlignmentFlag.AlignRight
        )
        layout.addLayout(axis)


class EvidenceDetailsDialog(QDialog):
    """Keep provenance and complete scientific claim boundaries reviewable."""

    def __init__(self, snapshot: PresentationSnapshot, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Evidence details")
        self.setAccessibleName("Frozen holdout evidence details")
        self.setObjectName("ResultsEvidenceDialog")
        self.resize(810, 650)
        self.setStyleSheet(
            "QDialog#ResultsEvidenceDialog {background: white;} "
            "QScrollArea {background: white; border: none;} "
            "QScrollArea > QWidget > QWidget {background: white;} "
            f"QPushButton {{background: #EDF3EF; color: {INK}; border: 1px solid {LINE}; "
            "border-radius: 12px; padding: 0 16px; min-height: 44px; font-size: 14px;} "
            "QPushButton:focus {border: 2px solid #126F68;}"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(26, 20, 26, 22)
        layout.setSpacing(18)
        heading = QHBoxLayout()
        heading.addWidget(result_label("Evidence details", size=31, bold=True), 1)
        self.close_button = QPushButton("Close")
        self.close_button.setAccessibleName("Close evidence details")
        self.close_button.clicked.connect(self.reject)
        heading.addWidget(self.close_button)
        layout.addLayout(heading)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 14, 0)
        content_layout.setSpacing(13)
        content_layout.addWidget(
            result_label(
                "Frozen documented-seed-disjoint holdout. Live demonstrations cannot replace "
                "these results.",
                color=MUTED,
                wrap=True,
            )
        )
        self.provenance_card = QFrame()
        self.provenance_card.setAccessibleName("Results evidence provenance")
        self.provenance_card.setAccessibleDescription(
            f"Hash-verified committed artifact {snapshot.artifact_id}; "
            f"SHA-256 {snapshot.artifact_sha256}; release {snapshot.release_id}; "
            f"{snapshot.matched_scenario_count} matched scenario conditions and "
            f"{snapshot.controller_evaluation_count} controller evaluations."
        )
        provenance = QVBoxLayout(self.provenance_card)
        provenance.setContentsMargins(0, 0, 0, 0)
        provenance.setSpacing(8)
        self.provenance_status = result_label("Hash verified · Frozen", bold=True, color=PRIMARY)
        self.provenance_id = result_label(snapshot.artifact_id, wrap=True)
        self.provenance_release = result_label(snapshot.release_id)
        self.provenance_sha256 = result_label(snapshot.artifact_sha256, size=12, wrap=True)
        self.provenance_summary = result_label(snapshot.count_statement, wrap=True)
        for caption, label in (
            ("Evidence artifact", self.provenance_id),
            ("Release", self.provenance_release),
            ("SHA-256", self.provenance_sha256),
        ):
            label.setProperty("mono", True)
            label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
                | Qt.TextInteractionFlag.TextSelectableByKeyboard
            )
            label.setAccessibleName(f"Verified {caption}: {label.text()}")
            label.setToolTip(f"Verified robustness supplement {caption}: {label.text()}")
            provenance.addWidget(result_label(caption, size=12, color=MUTED))
            provenance.addWidget(label)
        provenance.addWidget(self.provenance_status)
        provenance.addWidget(self.provenance_summary)
        content_layout.addWidget(self.provenance_card)
        content_layout.addWidget(result_label("Interpreting the evidence", size=20, bold=True))
        self.evidence_role_note = result_label(
            "Base paired 95% confidence intervals appear in the Results table. "
            "Sensitivity: pointwise 95%; no multiplicity adjustment or result gate. "
            "There is no favorable/adverse release gate.",
            wrap=True,
        )
        self.evidence_role_note.setAccessibleName(
            "Documented-seed-disjoint holdout; base paired 95% confidence intervals below; "
            "sensitivity supplement (not shown here): descriptive, pointwise 95%, unadjusted "
            "for multiple comparisons; no favorable/adverse release gate."
        )
        content_layout.addWidget(self.evidence_role_note)
        for text in (
            HOLDOUT_CUSTODY_LIMITATION,
            RESULTS_SCOPE_LIMITATION,
            SENSITIVITY_INTERVAL_LIMITATION,
        ):
            content_layout.addWidget(result_label(text, wrap=True))
        content_layout.addWidget(
            result_label(
                "The power-balance result is a simulator conservation invariant, not evidence "
                "of real-grid operational safety. The accepted reproduction is historical "
                "parity evidence; its three seeds participated in model selection.",
                wrap=True,
            )
        )
        content_layout.addWidget(result_label("Study limitations", size=20, bold=True))
        for limitation in snapshot.limitations:
            content_layout.addWidget(result_label(limitation, color=MUTED, wrap=True))
        content_layout.addStretch(1)
        self.scroll_area.setWidget(content)
        layout.addWidget(self.scroll_area, 1)

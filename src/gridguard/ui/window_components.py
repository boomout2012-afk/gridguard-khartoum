"""Focused presentation-shell widgets shared by the top-level window."""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)


class PageStack(QStackedWidget):
    """Stacked pages with the small QTabWidget API retained for callers."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._page_labels: list[str] = []

    def addTab(self, widget: QWidget, label: str) -> int:  # noqa: N802
        self._page_labels.append(label)
        return self.addWidget(widget)

    def tabText(self, index: int) -> str:  # noqa: N802
        if 0 <= index < len(self._page_labels):
            return self._page_labels[index]
        return ""


class PresentationNavigationRail(QFrame):
    """Four-page navigation with visual and accessible current-page state."""

    page_selected = Signal(int)

    def __init__(self, page_labels: Sequence[str], parent=None) -> None:
        super().__init__(parent)
        self.page_labels = tuple(page_labels)
        self.setObjectName("navigationRail")
        self.setFixedWidth(176)
        self.setAccessibleName("Application navigation")
        self.setAccessibleDescription(
            "Four pages. Use the navigation buttons or Control plus 1 through 4."
        )
        column = QVBoxLayout(self)
        column.setContentsMargins(16, 22, 14, 18)
        column.setSpacing(8)
        self._add_identity(column)

        self.navigation_group = QButtonGroup(self)
        self.navigation_group.setExclusive(True)
        self.buttons: list[QPushButton] = []
        for index, page_label in enumerate(self.page_labels):
            button = self._navigation_button(index, page_label)
            self.navigation_group.addButton(button, index)
            self.buttons.append(button)
            column.addWidget(button)

        column.addStretch(1)
        scope_heading = QLabel("SYNTHETIC DATA")
        scope_heading.setObjectName("railKicker")
        scope = QLabel("Decision support only\nNot grid operations")
        scope.setProperty("railMuted", True)
        scope.setProperty("compact", True)
        scope.setWordWrap(True)
        scope.setAccessibleName("Scope: synthetic data, decision support only, not grid operations")
        column.addWidget(scope_heading)
        column.addWidget(scope)

    @staticmethod
    def _add_identity(column: QVBoxLayout) -> None:
        kicker = QLabel("GRIDGUARD")
        kicker.setObjectName("railKicker")
        title = QLabel("Khartoum")
        title.setObjectName("railTitle")
        subtitle = QLabel("Power allocation atlas")
        subtitle.setObjectName("railSubtitle")
        subtitle.setWordWrap(True)
        column.addWidget(kicker)
        column.addWidget(title)
        column.addWidget(subtitle)
        rule = QFrame()
        rule.setObjectName("railRule")
        column.addWidget(rule)
        navigation_label = QLabel("PRESENTATION")
        navigation_label.setObjectName("railKicker")
        navigation_label.setAccessibleName("Presentation page navigation")
        column.addWidget(navigation_label)

    def _navigation_button(self, index: int, page_label: str) -> QPushButton:
        button = QPushButton(f"{index + 1:02d}  {page_label}")
        button.setProperty("nav", True)
        button.setCheckable(True)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setAccessibleName(f"Open {page_label} page, keyboard shortcut Ctrl+{index + 1}")
        button.setAccessibleDescription(
            f"Selects page {index + 1} of {len(self.page_labels)}: {page_label}."
        )
        button.setToolTip(f"{page_label} · Ctrl+{index + 1}")
        button.clicked.connect(lambda _checked=False, page=index: self.page_selected.emit(page))
        return button

    def set_current_index(self, index: int) -> None:
        """Synchronize the checked button and screen-reader page descriptions."""

        if not 0 <= index < len(self.buttons):
            return
        self.buttons[index].setChecked(True)
        for button_index, button in enumerate(self.buttons):
            state = "current" if button_index == index else "available"
            button.setAccessibleDescription(
                f"{self.page_labels[button_index]} page, {state}; "
                f"page {button_index + 1} of {len(self.page_labels)}."
            )


class PresentationTrustStrip(QFrame):
    """Persistent offline, runtime, and evidence-verification status."""

    def __init__(self, runtime_label: str, evidence_id: str, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("trustStrip")
        self.setAccessibleName("Verified application trust status")
        self.setAccessibleDescription(
            "Offline operation, verified GPU runtime, and locked hash-verified evidence. "
            f"Runtime: {runtime_label}. Evidence: {evidence_id}."
        )
        row = QHBoxLayout(self)
        row.setContentsMargins(22, 8, 22, 8)
        row.setSpacing(10)

        offline = QLabel("OFFLINE")
        offline.setProperty("badge", "verified")
        offline.setProperty("compact", True)
        offline.setAccessibleName("Offline: no external network connection is used")
        self.runtime_status = QLabel("GPU VERIFIED")
        self.runtime_status.setProperty("badge", "verified")
        self.runtime_status.setProperty("compact", True)
        self.runtime_status.setAccessibleName(f"GPU verified: {runtime_label}")
        self.runtime_status.setToolTip(runtime_label)
        evidence = QLabel("EVIDENCE LOCKED")
        evidence.setProperty("badge", "locked")
        evidence.setProperty("compact", True)
        evidence.setAccessibleName(f"Evidence locked: {evidence_id}")
        evidence.setToolTip(evidence_id)
        row.addWidget(offline)
        row.addWidget(self.runtime_status)
        row.addWidget(evidence)
        row.addStretch(1)
        runtime = QLabel(runtime_label)
        runtime.setProperty("muted", True)
        runtime.setProperty("compact", True)
        runtime.setProperty("mono", True)
        runtime.setAccessibleName(f"Verified runtime details: {runtime_label}")
        row.addWidget(runtime)

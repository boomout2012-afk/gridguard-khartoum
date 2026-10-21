"""Native navigation and page stack for the approved GridGuard workspace."""

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


class PresentationNavigationBar(QFrame):
    """Floating three-page navigation with synchronized keyboard and accessible state."""

    page_selected = Signal(int)

    def __init__(self, page_labels: Sequence[str], parent=None) -> None:
        super().__init__(parent)
        self.page_labels = tuple(page_labels)
        self.setObjectName("navigationBar")
        self.setFixedHeight(64)
        self.setAccessibleName("Application navigation")
        self.setAccessibleDescription(
            "Three pages. Use the navigation buttons or Control plus 1 through 3. "
            "F11 toggles full screen."
        )
        row = QHBoxLayout(self)
        row.setContentsMargins(24, 10, 24, 10)
        row.setSpacing(0)
        identity = QWidget()
        identity.setObjectName("navigationIdentity")
        identity.setFixedWidth(250)
        brand_row = QHBoxLayout(identity)
        brand_row.setContentsMargins(0, 0, 0, 0)
        brand_row.setSpacing(9)
        wordmark = QLabel("GridGuard")
        wordmark.setObjectName("brandTitle")
        location = QLabel("Khartoum")
        location.setObjectName("brandSubtitle")
        brand_row.addWidget(wordmark)
        brand_row.addWidget(location)
        brand_row.addStretch(1)
        row.addWidget(identity)
        row.addStretch(1)
        self.navigation_group = QButtonGroup(self)
        self.navigation_group.setExclusive(True)
        self.buttons: list[QPushButton] = []
        for index, page_label in enumerate(self.page_labels):
            button = self._navigation_button(index, page_label)
            self.navigation_group.addButton(button, index)
            self.buttons.append(button)
            row.addWidget(button)
            if index < len(self.page_labels) - 1:
                row.addSpacing(6)
        row.addStretch(1)
        row.addSpacing(250)

    def _navigation_button(self, index: int, page_label: str) -> QPushButton:
        button = QPushButton(page_label)
        button.setProperty("nav", True)
        button.setCheckable(True)
        button.setFixedHeight(44)
        button.setMinimumWidth(116 if index != 1 else 158)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setAccessibleName(f"Open {page_label} page, keyboard shortcut Ctrl+{index + 1}")
        button.setToolTip(f"{page_label} · Ctrl+{index + 1}")
        button.clicked.connect(lambda _checked=False, page=index: self.page_selected.emit(page))
        return button

    def set_current_index(self, index: int) -> None:
        if not 0 <= index < len(self.buttons):
            return
        self.buttons[index].setChecked(True)
        for button_index, button in enumerate(self.buttons):
            state = "current" if button_index == index else "available"
            button.setAccessibleDescription(
                f"{self.page_labels[button_index]} page, {state}; "
                f"page {button_index + 1} of {len(self.page_labels)}."
            )

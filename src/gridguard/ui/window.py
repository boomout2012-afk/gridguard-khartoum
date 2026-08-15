"""Top-level native presentation window."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from gridguard.ui.live_models import LiveScenarioComparison
from gridguard.ui.live_page import LiveScenarioPage
from gridguard.ui.live_worker import LiveScenarioWorker
from gridguard.ui.map_asset import MapAsset, load_map_asset
from gridguard.ui.methodology_page import MethodologyPage
from gridguard.ui.overview_page import OverviewPage
from gridguard.ui.presentation_model import PresentationSnapshot
from gridguard.ui.results_page import ResultsPage


class _PageStack(QStackedWidget):
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


class PresentationWindow(QMainWindow):
    """1366×768 presentation shell with four keyboard-accessible pages."""

    PAGE_LABELS = ("Overview", "Live Scenario", "Results", "Methodology")

    scenario_run_requested = Signal(str, int)

    def __init__(
        self,
        snapshot: PresentationSnapshot,
        *,
        runtime_label: str,
        map_asset: MapAsset | None = None,
        live_runner: Callable[[str, int], LiveScenarioComparison] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        asset = map_asset or load_map_asset()
        self.setWindowTitle("GridGuard Khartoum · Offline Power Allocation")
        self.setAccessibleName("GridGuard Khartoum presentation")
        self.setAccessibleDescription(
            "Offline synthetic load-shedding decision-support presentation with four pages."
        )
        self.setMinimumSize(1024, 650)
        self.resize(1366, 768)
        self._shortcuts: list[QShortcut] = []
        self._live_runner = live_runner
        self._live_thread: QThread | None = None
        self._live_worker: LiveScenarioWorker | None = None

        central = QWidget()
        central.setObjectName("applicationShell")
        shell = QHBoxLayout(central)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)

        self.tabs = _PageStack()
        self.tabs.setObjectName("pageStack")
        self.tabs.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.tabs.setAccessibleName("Presentation pages")
        self.overview_page = OverviewPage(snapshot, asset)
        self.live_page = LiveScenarioPage(asset)
        self.results_page = ResultsPage(snapshot)
        self.methodology_page = MethodologyPage(snapshot)
        self.tabs.addTab(self.overview_page, "Overview")
        self.tabs.addTab(self.live_page, "Live Scenario")
        self.tabs.addTab(self.results_page, "Results")
        self.tabs.addTab(self.methodology_page, "Methodology")
        self.live_page.run_requested.connect(self._handle_live_request)

        self.navigation_rail = self._build_navigation_rail()
        shell.addWidget(self.navigation_rail)
        content = QFrame()
        content.setObjectName("contentSurface")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        content_layout.addWidget(self.tabs, 1)
        self.trust_strip = self._build_trust_strip(runtime_label, snapshot)
        content_layout.addWidget(self.trust_strip)
        shell.addWidget(content, 1)
        self.setCentralWidget(central)
        self.tabs.currentChanged.connect(self._sync_navigation_state)
        self._sync_navigation_state(0)
        self._install_shortcuts()

    @property
    def live_run_active(self) -> bool:
        return self._live_thread is not None and self._live_thread.isRunning()

    def _build_navigation_rail(self) -> QFrame:
        rail = QFrame()
        rail.setObjectName("navigationRail")
        rail.setFixedWidth(176)
        rail.setAccessibleName("Application navigation")
        rail.setAccessibleDescription(
            "Four pages. Use the navigation buttons or Control plus 1 through 4."
        )
        column = QVBoxLayout(rail)
        column.setContentsMargins(16, 22, 14, 18)
        column.setSpacing(8)

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

        self.navigation_group = QButtonGroup(self)
        self.navigation_group.setExclusive(True)
        self.navigation_buttons: list[QPushButton] = []
        for index, page_label in enumerate(self.PAGE_LABELS):
            button = QPushButton(f"{index + 1:02d}  {page_label}")
            button.setProperty("nav", True)
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setAccessibleName(f"Open {page_label} page, keyboard shortcut Ctrl+{index + 1}")
            button.setAccessibleDescription(
                f"Selects page {index + 1} of {len(self.PAGE_LABELS)}: {page_label}."
            )
            button.setToolTip(f"{page_label} · Ctrl+{index + 1}")
            button.clicked.connect(
                lambda _checked=False, page=index: self.tabs.setCurrentIndex(page)
            )
            self.navigation_group.addButton(button, index)
            self.navigation_buttons.append(button)
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
        return rail

    def _build_trust_strip(
        self,
        runtime_label: str,
        snapshot: PresentationSnapshot,
    ) -> QFrame:
        strip = QFrame()
        strip.setObjectName("trustStrip")
        strip.setAccessibleName("Verified application trust status")
        strip.setAccessibleDescription(
            "Offline operation, verified GPU runtime, and locked hash-verified evidence. "
            f"Runtime: {runtime_label}. Evidence: {snapshot.benchmark_id}."
        )
        row = QHBoxLayout(strip)
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
        evidence.setAccessibleName(f"Evidence locked: {snapshot.benchmark_id}")
        evidence.setToolTip(snapshot.benchmark_id)
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
        return strip

    def _sync_navigation_state(self, index: int) -> None:
        if not 0 <= index < len(self.navigation_buttons):
            return
        self.navigation_buttons[index].setChecked(True)
        page_label = self.PAGE_LABELS[index]
        self.tabs.setAccessibleDescription(f"Current page: {page_label}")
        for button_index, button in enumerate(self.navigation_buttons):
            state = "current" if button_index == index else "available"
            button.setAccessibleDescription(
                f"{self.PAGE_LABELS[button_index]} page, {state}; "
                f"page {button_index + 1} of {len(self.PAGE_LABELS)}."
            )

    def _install_shortcuts(self) -> None:
        for index in range(self.tabs.count()):
            shortcut = QShortcut(QKeySequence(f"Ctrl+{index + 1}"), self)
            shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            shortcut.activated.connect(lambda page=index: self.tabs.setCurrentIndex(page))
            self._shortcuts.append(shortcut)
        fullscreen = QShortcut(QKeySequence(Qt.Key.Key_F11), self)
        fullscreen.activated.connect(self._toggle_fullscreen)
        self._shortcuts.append(fullscreen)

    def _toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _handle_live_request(self, shortage_level: str, seed: int) -> None:
        self.scenario_run_requested.emit(shortage_level, seed)
        if self._live_runner is None:
            return
        if self.live_run_active:
            return

        self.live_page.set_busy(True)
        thread = QThread(self)
        thread.setObjectName("gridguard-live-cuda-worker")
        worker = LiveScenarioWorker(self._live_runner, shortage_level, seed)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.succeeded.connect(self._live_succeeded)
        worker.failed.connect(self.live_page.show_run_error)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self._live_thread_finished)
        self._live_thread = thread
        self._live_worker = worker
        thread.start()

    def _live_succeeded(self, comparison: object) -> None:
        if not isinstance(comparison, LiveScenarioComparison):
            self.live_page.show_run_error("worker returned an invalid live comparison")
            return
        self.live_page.set_comparison(comparison)

    def _live_thread_finished(self) -> None:
        self.live_page.set_busy(False)
        thread = self._live_thread
        self._live_thread = None
        self._live_worker = None
        if thread is not None:
            thread.deleteLater()

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.live_run_active:
            self.live_page.state_label.setText(
                "The CUDA comparison is still running; close the window after it completes."
            )
            event.ignore()
            return
        super().closeEvent(event)

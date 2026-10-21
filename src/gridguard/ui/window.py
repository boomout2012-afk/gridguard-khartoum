"""Top-level native presentation window."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QMainWindow,
    QVBoxLayout,
    QWidget,
)

from gridguard.ui.live_models import LiveScenarioComparison
from gridguard.ui.live_page import LiveScenarioPage
from gridguard.ui.live_worker import LiveScenarioWorker
from gridguard.ui.map_asset import MapAsset, load_map_asset
from gridguard.ui.overview_page import OverviewPage
from gridguard.ui.presentation_model import PresentationSnapshot
from gridguard.ui.results_page import ResultsPage
from gridguard.ui.window_components import (
    PageStack,
    PresentationNavigationBar,
)

_PageStack = PageStack


class PresentationWindow(QMainWindow):
    """Three-page native workspace with the approved floating top navigation."""

    PAGE_LABELS = ("Overview", "Live Scenario", "Results")

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
        self.setWindowTitle("GridGuard Khartoum")
        self.setAccessibleName("GridGuard Khartoum presentation")
        self.setAccessibleDescription(
            "Offline synthetic load-shedding decision-support presentation with three pages."
        )
        self.setMinimumSize(1366, 768)
        self.resize(1440, 900)
        self.runtime_label = runtime_label
        self._shortcuts: list[QShortcut] = []
        self._live_runner = live_runner
        self._live_thread: QThread | None = None
        self._live_worker: LiveScenarioWorker | None = None

        central = QWidget()
        central.setObjectName("applicationShell")
        shell = QVBoxLayout(central)
        shell.setContentsMargins(32, 24, 32, 16)
        shell.setSpacing(20)

        self.tabs = _PageStack()
        self.tabs.setObjectName("pageStack")
        self.tabs.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.tabs.setAccessibleName("Presentation pages")
        self.overview_page = OverviewPage(snapshot, asset)
        self.live_page = LiveScenarioPage(asset)
        self.results_page = ResultsPage(snapshot)
        self.tabs.addTab(self.overview_page, "Overview")
        self.tabs.addTab(self.live_page, "Live Scenario")
        self.tabs.addTab(self.results_page, "Results")
        self.overview_page.explore_button.clicked.connect(lambda: self.tabs.setCurrentIndex(1))
        self.overview_page.results_button.clicked.connect(lambda: self.tabs.setCurrentIndex(2))
        self.live_page.run_requested.connect(self._handle_live_request)

        self.navigation_bar = PresentationNavigationBar(self.PAGE_LABELS)
        self.navigation_bar.page_selected.connect(self.tabs.setCurrentIndex)
        self.navigation_group = self.navigation_bar.navigation_group
        self.navigation_buttons = self.navigation_bar.buttons
        shell.addWidget(self.navigation_bar)
        shell.addWidget(self.tabs, 1)
        self.setCentralWidget(central)
        self.tabs.currentChanged.connect(self._sync_navigation_state)
        self._sync_navigation_state(0)
        self._install_shortcuts()

    @property
    def live_run_active(self) -> bool:
        return self._live_thread is not None and self._live_thread.isRunning()

    def _sync_navigation_state(self, index: int) -> None:
        if not 0 <= index < len(self.navigation_buttons):
            return
        self.navigation_bar.set_current_index(index)
        page_label = self.PAGE_LABELS[index]
        self.tabs.setAccessibleDescription(f"Current page: {page_label}")

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
                "The comparison is still running; close the window after it completes."
            )
            event.ignore()
            return
        super().closeEvent(event)

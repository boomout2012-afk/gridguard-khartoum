"""Top-level native presentation window."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QTabWidget,
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


class PresentationWindow(QMainWindow):
    """1366×768 presentation shell with four keyboard-accessible pages."""

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
        self.setMinimumSize(1024, 650)
        self.resize(1366, 768)
        self._shortcuts: list[QShortcut] = []
        self._live_runner = live_runner
        self._live_thread: QThread | None = None
        self._live_worker: LiveScenarioWorker | None = None

        central = QWidget()
        shell = QVBoxLayout(central)
        shell.setContentsMargins(14, 12, 14, 12)
        shell.setSpacing(10)
        shell.addWidget(self._build_header(runtime_label))

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setMovable(False)
        self.tabs.setUsesScrollButtons(False)
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
        shell.addWidget(self.tabs, 1)
        self.setCentralWidget(central)
        self._install_shortcuts()

    @property
    def live_run_active(self) -> bool:
        return self._live_thread is not None and self._live_thread.isRunning()

    def _build_header(self, runtime_label: str) -> QFrame:
        header = QFrame()
        header.setObjectName("appHeader")
        header.setAccessibleName("Application and GPU status")
        row = QHBoxLayout(header)
        row.setContentsMargins(16, 10, 16, 10)
        row.setSpacing(12)
        identity = QVBoxLayout()
        identity.setSpacing(1)
        title = QLabel("GridGuard Khartoum")
        title.setObjectName("brandTitle")
        subtitle = QLabel("Offline GPU load-shedding decision support")
        subtitle.setObjectName("brandSubtitle")
        identity.addWidget(title)
        identity.addWidget(subtitle)
        row.addLayout(identity)
        row.addStretch(1)
        offline = QLabel("OFFLINE")
        offline.setProperty("badge", "verified")
        offline.setAccessibleDescription("No external network connection is used")
        runtime = QLabel(f"● GPU READY  ·  {runtime_label}")
        runtime.setProperty("badge", "verified")
        runtime.setAccessibleName("Verified GPU runtime")
        row.addWidget(offline)
        row.addWidget(runtime)
        return header

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

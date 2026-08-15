"""Qt worker boundary that keeps CUDA simulation off the GUI thread."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QObject, Signal, Slot

from gridguard.ui.live_models import LiveScenarioComparison


class LiveScenarioWorker(QObject):
    """Run one injected live comparison and return it through queued signals."""

    succeeded = Signal(object)
    failed = Signal(str)
    finished = Signal()

    def __init__(
        self,
        runner: Callable[[str, int], LiveScenarioComparison],
        shortage_level: str,
        seed: int,
    ) -> None:
        super().__init__()
        self._runner = runner
        self._shortage_level = shortage_level
        self._seed = seed

    @Slot()
    def run(self) -> None:
        try:
            comparison = self._runner(self._shortage_level, self._seed)
            if not isinstance(comparison, LiveScenarioComparison):
                raise TypeError("live runner must return a LiveScenarioComparison")
        except Exception as error:
            self.failed.emit(str(error) or type(error).__name__)
        else:
            self.succeeded.emit(comparison)
        finally:
            self.finished.emit()

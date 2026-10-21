"""Capture deterministic 1366x768 screenshots from the real offline desktop UI."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

# These settings must be fixed before Qt is imported so host display scaling cannot change pixels.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_SCALE_FACTOR", "1")
os.environ.setdefault("QT_FONT_DPI", "96")

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QApplication

from gridguard.ui.live_models import LiveScenarioComparison
from gridguard.ui.live_runner import run_live_scenario
from gridguard.ui.map_asset import load_map_asset
from gridguard.ui.presentation_model import load_presentation_snapshot
from gridguard.ui.theme import apply_theme
from gridguard.ui.window import PresentationWindow

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "docs" / "assets" / "screenshots"
CAPTURE_SIZE = QSize(1366, 768)
CAPTURE_HOUR = 18
CAPTURE_LOCALITY_ID = "east_nile"
PAGE_CAPTURES = (
    ("overview", 0),
    ("live", 1),
    ("results", 2),
    ("methodology", 3),
)


def build_live_capture_comparison() -> LiveScenarioComparison:
    """Run the canonical CUDA demonstration used by documentation screenshots."""

    return run_live_scenario("moderate", 50002)


def _prepare_window(
    application: QApplication,
    *,
    comparison: LiveScenarioComparison | None = None,
) -> PresentationWindow:
    application.setApplicationName("GridGuard Khartoum")
    application.setOrganizationName("GridGuard Khartoum")
    apply_theme(application)
    snapshot = load_presentation_snapshot()
    window = PresentationWindow(
        snapshot,
        runtime_label="Presentation capture | verified evidence | offline",
        map_asset=load_map_asset(),
    )
    window.setFixedSize(CAPTURE_SIZE)
    window.live_page.shortage_combo.setCurrentText("Moderate")
    window.live_page.seed_input.setValue(50002)
    window.live_page.set_comparison(
        build_live_capture_comparison() if comparison is None else comparison
    )
    window.live_page.timeline.setValue(CAPTURE_HOUR)
    window.live_page.map_view.select_locality(CAPTURE_LOCALITY_ID)
    window.live_page.state_label.setProperty("state", "ready")
    window.live_page.state_label.setText(
        "Canonical CUDA live demonstration | Moderate / seed 50002 | not frozen evidence."
    )
    window.live_page.state_label.setAccessibleName(window.live_page.state_label.text())
    window.live_page.state_label.style().unpolish(window.live_page.state_label)
    window.live_page.state_label.style().polish(window.live_page.state_label)
    return window


def _select_capture_page(window: PresentationWindow, page_index: int) -> None:
    window.tabs.setCurrentIndex(page_index)
    window.navigation_buttons[page_index].setFocus(Qt.FocusReason.OtherFocusReason)


def capture_screenshots(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    application: QApplication | None = None,
) -> tuple[Path, ...]:
    """Render all four real application pages to deterministic PNG files."""

    app = application or QApplication.instance() or QApplication(["gridguard-screenshot-capture"])
    if not isinstance(app, QApplication):
        raise RuntimeError("screenshot capture requires a QApplication")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    comparison = build_live_capture_comparison()
    captured: list[Path] = []
    for name, page_index in PAGE_CAPTURES:
        window = _prepare_window(app, comparison=comparison)
        try:
            window.show()
            _select_capture_page(window, page_index)
            if name == "overview":
                window.overview_page.map_view.reset_view()
            elif name == "live":
                window.live_page.map_view.reset_view()
                window.live_page.map_view.select_locality(CAPTURE_LOCALITY_ID)
            elif name == "methodology":
                window.methodology_page.scroll_area.verticalScrollBar().setValue(0)
            window.navigation_buttons[page_index].setFocus(Qt.FocusReason.OtherFocusReason)
            app.processEvents()
            pixmap = window.grab()
            output_path = destination / f"{name}.png"
            temporary_path = destination / f".{name}.png.tmp"
            if not pixmap.save(str(temporary_path), "PNG"):
                raise OSError(f"could not encode screenshot: {output_path}")
            temporary_path.replace(output_path)
            captured.append(output_path.resolve())
        finally:
            window.close()
            app.processEvents()
    return tuple(captured)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    for path in capture_screenshots(args.output_dir):
        print(f"Screenshot written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

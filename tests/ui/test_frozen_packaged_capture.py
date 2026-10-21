from __future__ import annotations

import runpy
from pathlib import Path

import pytest
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QWidget

from gridguard.ui.presentation_model import load_presentation_snapshot
from gridguard.ui.theme import apply_theme
from gridguard.ui.window import PresentationWindow

ROOT = Path(__file__).resolve().parents[2]


def _capture_function():
    launcher = runpy.run_path(str(ROOT / "packaging" / "gridguard_launcher.py"))
    return launcher["_capture_presentation_pages"]


def test_frozen_smoke_captures_all_three_native_pages(qtbot, qapp, tmp_path):
    capture = _capture_function()
    apply_theme(qapp)
    window = PresentationWindow(load_presentation_snapshot(), runtime_label="Test runtime")
    qtbot.addWidget(window)
    window.show()
    window.tabs.setCurrentIndex(1)

    capture(qapp, window, tmp_path / "frozen-smoke.json")

    files = sorted((tmp_path / "screenshots").glob("*.png"))
    assert [path.name for path in files] == ["live.png", "overview.png", "results.png"]
    assert len({path.read_bytes() for path in files}) == 3
    for path in files:
        image = QImage(str(path))
        assert not image.isNull()
        assert (image.width(), image.height()) == (1366, 768)
    assert window.tabs.currentIndex() == 1


def test_frozen_smoke_rejects_obsolete_four_page_shell(qtbot, qapp, tmp_path):
    capture = _capture_function()
    window = PresentationWindow(load_presentation_snapshot(), runtime_label="Test runtime")
    qtbot.addWidget(window)
    window.tabs.addTab(QWidget(), "Methodology")

    with pytest.raises(RuntimeError, match="three approved pages"):
        capture(qapp, window, tmp_path / "frozen-smoke.json")
    assert not (tmp_path / "screenshots").exists()

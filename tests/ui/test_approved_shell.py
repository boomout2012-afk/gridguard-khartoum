from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame

from gridguard.ui.presentation_model import load_presentation_snapshot
from gridguard.ui.theme import apply_theme
from gridguard.ui.window import PresentationWindow


def test_approved_shell_has_three_horizontal_pages_and_no_runtime_strip(qtbot, qapp):
    apply_theme(qapp)
    window = PresentationWindow(load_presentation_snapshot(), runtime_label="Test CUDA runtime")
    qtbot.addWidget(window)
    window.resize(1366, 768)
    window.show()

    assert [window.tabs.tabText(index) for index in range(window.tabs.count())] == [
        "Overview",
        "Live Scenario",
        "Results",
    ]
    nav = window.navigation_bar
    assert nav.width() > 1000
    assert nav.height() == 64
    assert not window.findChildren(QFrame, "trustStrip")
    assert all(button.height() >= 44 for button in window.navigation_buttons)
    assert (
        len(
            {
                button.mapTo(window, button.rect().topLeft()).y()
                for button in window.navigation_buttons
            }
        )
        == 1
    )
    assert window.tabs.geometry().top() >= nav.geometry().bottom()


def test_top_navigation_and_shortcuts_keep_accessible_state_synchronized(qtbot):
    window = PresentationWindow(load_presentation_snapshot(), runtime_label="Test CUDA runtime")
    qtbot.addWidget(window)
    window.show()
    window.activateWindow()
    qtbot.waitUntil(window.isActiveWindow, timeout=1000)
    for index, button in enumerate(window.navigation_buttons):
        qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
        assert window.tabs.currentIndex() == index
        assert button.isChecked()
        assert "current" in button.accessibleDescription()
    qtbot.keyClick(window, Qt.Key.Key_1, Qt.KeyboardModifier.ControlModifier)
    assert window.tabs.currentIndex() == 0
    qtbot.mouseClick(window.overview_page.results_button, Qt.MouseButton.LeftButton)
    assert window.tabs.currentIndex() == 2

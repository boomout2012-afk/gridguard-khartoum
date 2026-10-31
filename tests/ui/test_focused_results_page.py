"""Current evidence is visible by default without confusing it with the live actor."""

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QLabel

from gridguard.focused_results import FocusedStudyError
from gridguard.ui.focused_results import FocusedResultsPage
from gridguard.ui.presentation_model import load_presentation_snapshot
from gridguard.ui.theme import apply_theme
from gridguard.ui.window import PresentationWindow


@pytest.fixture
def page(qtbot):
    widget = FocusedResultsPage(load_presentation_snapshot())
    qtbot.addWidget(widget)
    widget.resize(1270, 630)
    widget.show()
    return widget


def test_current_results_show_three_methods_and_quantitative_tradeoff(page) -> None:
    text = " ".join(label.text() for label in page.findChildren(QLabel) if label.isVisible())
    assert "Current study" in text
    assert "five independent PPO runs" in text
    assert "did not pass" not in text
    assert "met the service-improvement criterion" not in text
    assert "−0.399 pp" in text
    assert "proposal" not in text.casefold()
    table = page.current_results_table
    assert table.rowCount() == 5
    assert table.columnCount() == 4
    assert table.item(1, 3).text() == "97.02%"
    assert table.item(2, 2).text() == "97.41%"
    assert page.results_table.rowCount() == 3  # retained historical integration field
    assert not page.historical_page.isVisible()


def test_current_results_fit_presentation_surface(page) -> None:
    table = page.current_results_table
    assert table.horizontalScrollBar().maximum() == 0
    assert table.verticalScrollBar().maximum() == 0
    for label in (page.count_label, page.tradeoff_label, page.scope_note):
        bottom = label.mapTo(page, QPoint(0, label.height())).y()
        assert bottom <= page.contentsRect().bottom()
        assert label.height() >= label.heightForWidth(label.width())


def test_intervals_runs_and_earlier_evidence_are_accessible(page, qtbot) -> None:
    qtbot.mouseClick(page.details_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(page.details_dialog.isVisible)
    assert page.details_dialog.tabs.count() == 5
    assert "97.5%" in page.details_dialog.tabs.widget(0).findChildren(QLabel)[0].text()
    details_text = " ".join(label.text() for label in page.details_dialog.findChildren(QLabel))
    assert "did not pass" not in details_text
    assert "met the service-improvement criterion" not in details_text
    page.details_dialog.reject()
    qtbot.mouseClick(page.earlier_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(page.historical_page.isVisible)
    assert page.historical_dialog.windowTitle() == "Earlier results"
    page.historical_dialog.reject()


def test_unverified_current_study_shows_no_current_values(qtbot) -> None:
    def fail():
        raise FocusedStudyError("Cannot verify changed study")

    widget = FocusedResultsPage(load_presentation_snapshot(), loader=fail)
    qtbot.addWidget(widget)
    assert widget.current_results_table is None
    assert widget.study_view is None
    assert "No current study values" in widget.error_label.text()
    assert not widget.details_button.isEnabled()
    assert widget.earlier_button.isEnabled()


def test_window_defaults_to_current_study_at_presentation_size(qtbot, qapp) -> None:
    apply_theme(qapp)
    window = PresentationWindow(load_presentation_snapshot(), runtime_label="Test runtime")
    qtbot.addWidget(window)
    window.resize(1366, 768)
    window.show()
    window.tabs.setCurrentIndex(2)
    qtbot.waitUntil(window.results_page.isVisible)
    page = window.results_page
    assert isinstance(page, FocusedResultsPage)
    assert page.current_results_table.isVisible()
    assert page.current_results_table.verticalScrollBar().maximum() == 0
    for label in (page.count_label, page.tradeoff_label, page.scope_note):
        assert label.height() >= label.heightForWidth(label.width())
        assert label.mapTo(page, QPoint(0, label.height())).y() <= page.height()
    assert "Current trained PPO" in window.live_page.scope_note.text()
    assert "single live scenario" in page.scope_note.text()
    assert "five-run" in page.scope_note.text()

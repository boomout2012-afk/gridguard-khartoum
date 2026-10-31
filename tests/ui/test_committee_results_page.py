from __future__ import annotations

from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QLabel

from gridguard.committee_results import CommitteeResultsError
from gridguard.ui.focused_results import FocusedResultsPage
from gridguard.ui.presentation_model import load_presentation_snapshot
from gridguard.ui.window import PresentationWindow


def test_followup_tab_is_separate_from_original_results_and_keyboard_accessible(qtbot) -> None:
    page = FocusedResultsPage(load_presentation_snapshot())
    qtbot.addWidget(page)
    page.resize(1270, 630)
    page.show()
    assert page.results_tabs.currentIndex() == 0
    assert page.current_results_table.item(1, 3).text() == "97.02%"
    assert page.results_tabs.tabText(1) == "Follow-up comparison"
    bar = page.results_tabs.tabBar()
    bar.setFocus()
    qtbot.keyClick(bar, Qt.Key.Key_Right)
    assert page.results_tabs.currentIndex() == 1
    followup = page.followup_page
    assert followup.means_table.rowCount() == 5
    assert followup.means_table.item(4, 1).text() == "6528.02"
    assert "Post-hoc" in " ".join(label.text() for label in followup.findChildren(QLabel))
    assert followup.detail_tabs.count() == 3
    followup.detail_tabs.setCurrentIndex(2)
    assert followup.stress_table.rowCount() == 10
    assert followup.stress_table.item(4, 3).text() == "168"


def test_followup_failure_does_not_replace_verified_primary_results(qtbot) -> None:
    def fail():
        raise CommitteeResultsError("Cannot verify modified audit")

    page = FocusedResultsPage(load_presentation_snapshot(), followup_loader=fail)
    qtbot.addWidget(page)
    assert page.current_results_table.item(1, 3).text() == "97.02%"
    assert page.followup_page.means_table is None
    assert "No follow-up values" in page.followup_page.error_label.text()


def test_followup_tables_fit_minimum_presentation_view(qtbot) -> None:
    window = PresentationWindow(load_presentation_snapshot(), runtime_label="Test")
    qtbot.addWidget(window)
    window.resize(1366, 768)
    window.show()
    window.tabs.setCurrentIndex(2)
    page = window.results_page
    page.results_tabs.setCurrentIndex(1)
    followup = page.followup_page
    for index, tables in (
        (0, (followup.means_table,)),
        (1, tuple(followup.paired_tables)),
        (2, (followup.stress_table,)),
    ):
        followup.detail_tabs.setCurrentIndex(index)
        qtbot.wait(30)
        for table in tables:
            assert table.horizontalScrollBar().maximum() == 0
            assert table.verticalScrollBar().maximum() == 0
            assert table.mapTo(page, QPoint(0, table.height())).y() <= page.height()

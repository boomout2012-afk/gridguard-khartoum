"""The approved native Results view preserves its frozen evidence boundary."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QLabel

from gridguard.claim_contract import (
    HOLDOUT_CUSTODY_LIMITATION,
    RESULTS_SCOPE_LIMITATION,
    SENSITIVITY_INTERVAL_LIMITATION,
)
from gridguard.ui import theme
from gridguard.ui.components import format_metric_value
from gridguard.ui.presentation_model import PresentationSnapshot
from gridguard.ui.results_page import ResultsPage
from gridguard.ui.window import PresentationWindow


@pytest.fixture
def snapshot() -> PresentationSnapshot:
    evidence = Path(__file__).resolve().parents[2] / "data/evidence/robustness-supplement.json"
    return PresentationSnapshot.from_mapping(json.loads(evidence.read_text(encoding="utf-8")))


def test_comparison_surface_renders_injected_values_without_recomputing_evidence(
    qtbot, snapshot
) -> None:
    changed = replace(snapshot.metrics[1], ppo=0.80123, fixed_rotation=0.95123)
    injected = replace(snapshot, metrics=(snapshot.metrics[0], changed, snapshot.metrics[2]))
    page = ResultsPage(injected)
    qtbot.addWidget(page)

    assert tuple(page.outcome_panels) == tuple(metric.key for metric in injected.metrics)
    for metric in injected.metrics:
        panel = page.outcome_panels[metric.key]
        assert panel.parentWidget() is page.comparison_surface
        for controller, value in (("PPO", metric.ppo), ("Fixed Rotation", metric.fixed_rotation)):
            assert panel.value_labels[controller].text() == format_metric_value(
                metric, value, include_unit=metric.unit != "locality-hours"
            )
            assert panel.bars[controller].raw_value == value
            assert controller in panel.bars[controller].accessibleName()
            assert metric.label in panel.bars[controller].accessibleName()
    assert page.results_table.item(1, 1).text() == "80.12%"
    assert page.results_table.item(1, 3).text() == "+9.15 [9.11, 9.19] pp"
    assert "supply-blind" in page.scope_note.text()
    assert "cannot select Critical Only" in page.scope_note.text()


def test_evidence_dialog_preserves_provenance_and_material_claims(qtbot, snapshot) -> None:
    page = ResultsPage(snapshot)
    qtbot.addWidget(page)
    page.resize(1302, 644)
    page.show()
    page.evidence_details_button.setFocus()
    qtbot.keyClick(page.evidence_details_button, Qt.Key.Key_Space)
    qtbot.waitUntil(page.evidence_dialog.isVisible)

    dialog = page.evidence_dialog
    visible_text = " ".join(label.text() for label in dialog.findChildren(QLabel))
    for text in (
        snapshot.artifact_id,
        snapshot.artifact_sha256,
        snapshot.release_id,
        snapshot.count_statement,
        HOLDOUT_CUSTODY_LIMITATION,
        RESULTS_SCOPE_LIMITATION,
        SENSITIVITY_INTERVAL_LIMITATION,
        *snapshot.limitations,
    ):
        assert text in visible_text
    assert "no favorable/adverse release gate" in visible_text
    assert dialog.scroll_area.widgetResizable()
    assert page.provenance_sha256.textInteractionFlags() & (
        Qt.TextInteractionFlag.TextSelectableByKeyboard
    )
    qtbot.keyClick(dialog, Qt.Key.Key_Escape)
    qtbot.waitUntil(lambda: not dialog.isVisible())
    qtbot.waitUntil(page.evidence_details_button.hasFocus)


@pytest.mark.parametrize(("width", "height"), [(1376, 776), (1302, 644)])
def test_results_fit_the_approved_native_page_sizes(qtbot, qapp, snapshot, width, height) -> None:
    theme.apply_theme(qapp)
    page = ResultsPage(snapshot)
    qtbot.addWidget(page)
    page.resize(width, height)
    page.show()
    qtbot.wait(20)

    assert page.size().width() == width
    assert page.size().height() == height
    assert page.evidence_details_button.height() >= 44
    for panel in page.outcome_panels.values():
        assert panel.title_label.height() >= panel.title_label.sizeHint().height()
        for label in panel.value_labels.values():
            assert label.width() >= label.sizeHint().width()
            assert label.height() >= label.sizeHint().height()
    for label in (page.scope_note, page.safety_note, page.synthetic_note):
        bottom = label.mapTo(page, QPoint(0, label.height())).y()
        assert bottom <= page.height()
        assert label.height() >= label.heightForWidth(label.width())
    assert page.results_table.horizontalScrollBar().maximum() == 0
    assert page.results_table.verticalScrollBar().maximum() == 0
    header = page.results_table.horizontalHeader()
    assert header.viewport().height() >= header.fontMetrics().height() + 10


@pytest.mark.parametrize(("width", "height"), [(1440, 900), (1366, 768)])
def test_footer_text_fits_inside_the_full_window_with_vertical_clearance(
    qtbot, qapp, snapshot, width, height
) -> None:
    theme.apply_theme(qapp)
    window = PresentationWindow(
        snapshot, runtime_label="Test CUDA runtime", results_page_factory=ResultsPage
    )
    qtbot.addWidget(window)
    window.setFixedSize(width, height)
    window.tabs.setCurrentIndex(2)
    window.show()
    qtbot.wait(20)

    page = window.results_page
    page_content_bottom = page.height() - page.layout().contentsMargins().bottom()
    for label in (page.scope_note, page.synthetic_note):
        assert label.isVisible()
        assert label.mapTo(page, QPoint(0, label.height())).y() <= page_content_bottom
        assert label.height() >= label.heightForWidth(label.width())
        assert label.contentsMargins().top() >= 4
        assert label.contentsMargins().bottom() >= 4
        assert label.contentsRect().height() >= 2 * label.fontMetrics().height()
    for panel in page.outcome_panels.values():
        assert panel.height() >= panel.minimumSizeHint().height()

"""The supplemental study is inspectable without changing frozen benchmark values."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QAbstractItemView, QLabel

from gridguard.ui import theme
from gridguard.ui.presentation_model import PresentationSnapshot
from gridguard.ui.proposal_results import ProposalResultsDialog, build_proposal_results_view
from gridguard.ui.results_page import ResultsPage


@pytest.fixture
def payload() -> dict:
    names = ["Fixed Rotation", "PPO", "Optimization"]
    means = {
        "critical_outage_locality_hours": 20.123,
        "fairness": 0.96321,
        "utilization": 0.97876,
        "critical_service_ratio": 0.98765,
        "blackout_locality_hours": 14.125,
        "blackout_duration_mean_pairwise_gap_hours": 1.2345,
        "critical_interruption_duration_mean_pairwise_gap_hours": 2.3456,
        "longest_blackout_hours": 3.4567,
    }
    return {
        "artifact_id": "gridguard-proposal-supplement-v1",
        "protocol": {
            "evaluation_seeds": list(range(91001, 91011)),
            "batch_size": 32,
            "independent_base_draws": 320,
            "matched_scenario_conditions": 960,
            "controller_evaluations": 2880,
        },
        "controller_names": names,
        "controller_means": {name: dict(means) for name in names},
        "paired_inference": {
            pair: {
                key: {"mean_delta": -0.12345, "ci95_lower": -0.23456, "ci95_upper": 0.01234}
                for key in means
            }
            for pair in (
                "PPO - Fixed Rotation",
                "Optimization - Fixed Rotation",
                "PPO - Optimization",
            )
        },
        "forecast": {
            "training_trace_count": 768,
            "training_seeds": [42, 43, 44],
            "evaluation_trace_count": 320,
            "target_hours": list(range(3, 24)),
            "methods": {
                method: {
                    "locality_sector": {"mae_mw": 1.23456, "rmse_mw": 2.34567},
                    "system_total": {"mae_mw": 3.45678, "rmse_mw": 4.56789},
                }
                for method in ("trailing_mean_3h", "persistence_1h", "ridge_ar3")
            },
        },
    }


@pytest.fixture
def snapshot() -> PresentationSnapshot:
    path = Path(__file__).resolve().parents[2] / "data/evidence/robustness-supplement.json"
    return PresentationSnapshot.from_mapping(json.loads(path.read_text(encoding="utf-8")))


def test_view_formats_supplied_means_and_intervals_without_recomputing_them(payload) -> None:
    view = build_proposal_results_view(payload)

    assert view.means.columns == ("Metric", "Fixed Rotation", "PPO", "Optimization")
    assert len(view.means.rows) == 8
    assert view.means.rows[1][1:] == ("96.32%",) * 3
    assert view.means.rows[5][1:] == ("1.23",) * 3
    assert "960 matched scenario conditions" in view.count_statement
    assert "320 independent base draws" in view.count_statement
    assert "2,880 controller evaluations" in view.count_statement
    assert "91001" in view.evaluation_statement and "91010" in view.evaluation_statement
    # Means are identical in this injected fixture; supplied deltas are deliberately not zero.
    pair, intervals = view.paired[0]
    assert pair == "PPO - Fixed Rotation"
    assert intervals.rows[1][1:] == ("-12.35 pp", "[-23.46, +1.23] pp")
    assert intervals.rows[3][1:] == ("-0.12 h", "[-0.23, +0.01] h")
    assert view.forecast.rows[-1] == ("Learned ridge AR(3)", "1.2346", "2.3457", "3.4568", "4.5679")
    assert "768" in view.training_statement
    assert "320 evaluation traces" in view.training_statement
    assert "hours 3–23" in view.training_statement


def test_dialog_has_accessible_read_only_tabs_and_all_comparisons(qtbot, payload) -> None:
    dialog = ProposalResultsDialog(build_proposal_results_view(payload))
    qtbot.addWidget(dialog)
    dialog.open()

    assert dialog.isModal()
    assert dialog.tabs.count() == 3
    assert [dialog.tabs.tabText(i) for i in range(3)] == [
        "Controllers",
        "Paired differences",
        "Forecasting",
    ]
    assert dialog.means_table.rowCount() == 8
    assert dialog.forecast_table.rowCount() == 3
    for table in (dialog.means_table, dialog.paired_table, dialog.forecast_table):
        assert table.accessibleName()
        assert table.editTriggers() == QAbstractItemView.EditTrigger.NoEditTriggers
        assert table.focusPolicy() == Qt.FocusPolicy.StrongFocus
        assert all(table.item(row, 0).toolTip() for row in range(table.rowCount()))
    dialog.tabs.setCurrentIndex(1)
    dialog.pair_selector.setCurrentIndex(2)
    assert dialog.pair_selector.currentText() == "PPO - Optimization"
    assert dialog.paired_table.rowCount() == 5
    assert "PPO - Optimization" in dialog.paired_table.accessibleDescription()
    labels = " ".join(label.text() for label in dialog.findChildren(QLabel))
    for term in ("Synthetic", "decision support", "one-hour", "24-hour optimum", "zero service"):
        assert term in labels
    assert "Global proportional clipping" in labels
    qtbot.keyClick(dialog, Qt.Key.Key_Escape)
    qtbot.waitUntil(lambda: not dialog.isVisible())


@pytest.mark.parametrize(("width", "height"), [(1040, 680), (960, 620)])
def test_study_dialog_keeps_tables_and_text_legible_at_presentation_sizes(
    qtbot, qapp, payload, width, height
) -> None:
    theme.apply_theme(qapp)
    dialog = ProposalResultsDialog(build_proposal_results_view(payload))
    qtbot.addWidget(dialog)
    dialog.resize(width, height)
    dialog.open()
    for tab, table in enumerate((dialog.means_table, dialog.paired_table, dialog.forecast_table)):
        dialog.tabs.setCurrentIndex(tab)
        qtbot.wait(20)
        assert dialog.width() == width
        assert dialog.height() == height
        assert table.horizontalScrollBar().maximum() == 0
        assert table.viewport().height() >= 2 * table.verticalHeader().defaultSectionSize()
        header = table.horizontalHeader()
        header_lines = max(
            len(table.horizontalHeaderItem(column).text().splitlines())
            for column in range(table.columnCount())
        )
        assert header.viewport().height() >= header.fontMetrics().height() * header_lines + 16
        for row in range(table.rowCount()):
            for column in range(table.columnCount()):
                assert (
                    table.columnWidth(column)
                    >= table.fontMetrics().horizontalAdvance(table.item(row, column).text()) + 12
                )
        for label in dialog.findChildren(QLabel):
            if label.isVisible() and label.wordWrap():
                assert label.height() >= label.heightForWidth(label.width())


def test_results_loads_study_only_on_request_and_restores_keyboard_focus(
    qtbot, snapshot, payload
) -> None:
    calls = []

    def loader():
        calls.append(True)
        return payload

    page = ResultsPage(snapshot, proposal_loader=loader)
    qtbot.addWidget(page)
    page.resize(1302, 644)
    page.show()
    before = page.results_table.item(0, 1).text()
    assert not calls
    page.extended_study_button.setFocus()
    qtbot.keyClick(page.extended_study_button, Qt.Key.Key_Space)
    qtbot.waitUntil(lambda: page.proposal_dialog is not None and page.proposal_dialog.isVisible())
    assert calls == [True]
    assert page.results_table.item(0, 1).text() == before
    qtbot.keyClick(page.proposal_dialog, Qt.Key.Key_Escape)
    qtbot.waitUntil(page.extended_study_button.hasFocus)


def test_invalid_study_fails_closed_without_showing_values(qtbot, snapshot) -> None:
    def loader():
        raise ValueError("supplement SHA-256 mismatch")

    page = ResultsPage(snapshot, proposal_loader=loader)
    qtbot.addWidget(page)
    page.show()
    qtbot.mouseClick(page.extended_study_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: page.proposal_error_dialog is not None)
    assert page.proposal_dialog is None
    assert "could not be verified" in page.proposal_error_dialog.text()
    assert "SHA-256 mismatch" in page.proposal_error_dialog.detailedText()
    page.proposal_error_dialog.close()

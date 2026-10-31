"""Three-controller live presentation and direct duration output contracts."""

from __future__ import annotations

from dataclasses import replace

import pytest
from PySide6.QtCore import QPoint, Qt

from gridguard.domain import ServiceTier
from gridguard.ui.live_models import LiveDurationSummary, LiveScenarioComparison
from gridguard.ui.live_page import LiveScenarioPage
from gridguard.ui.map_asset import load_map_asset
from gridguard.ui.theme import apply_theme
from tests.ui.test_live_page import _comparison, _frame, _summary


def _three_controller_comparison() -> LiveScenarioComparison:
    comparison = _comparison()
    duration = LiveDurationSummary(
        blackout_locality_hours=14.0,
        critical_interruption_locality_hours=21.0,
        blackout_duration_range_hours=3.0,
        blackout_duration_mean_pairwise_gap_hours=1.25,
        critical_interruption_duration_range_hours=4.0,
        critical_interruption_duration_mean_pairwise_gap_hours=1.75,
        longest_blackout_hours=2.0,
        longest_critical_interruption_hours=3.0,
    )
    return replace(
        comparison,
        frames_by_controller={
            **comparison.frames_by_controller,
            "Optimization": tuple(_frame(hour, ServiceTier.CRITICAL_ONLY) for hour in range(24)),
        },
        metrics_by_controller={
            **comparison.metrics_by_controller,
            "Optimization": replace(_summary(served_mwh=8_100.0), duration=duration),
        },
    )


def test_optional_third_controller_preserves_two_controller_fixtures(qtbot) -> None:
    page = LiveScenarioPage(load_map_asset())
    qtbot.addWidget(page)
    page.show()
    page.set_comparison(_comparison())

    assert tuple(page._comparison.frames_by_controller) == ("PPO", "Fixed Rotation")
    assert not page.map_panel.controller_buttons["Optimization"].isVisibleTo(page)
    assert page.controller_combo.count() == 2
    assert page._comparison.metrics("PPO").duration is None


def test_controller_switch_updates_map_cards_and_duration_tooltip(qtbot) -> None:
    page = LiveScenarioPage(load_map_asset())
    qtbot.addWidget(page)
    page.show()
    page.set_comparison(_three_controller_comparison())
    assert page.comparison_summary.baselines["Served energy"].text() == "Optimization 8,100 MWh"
    page.timeline.setValue(18)
    button = page.map_panel.controller_buttons["Optimization"]
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)

    assert page.controller_combo.currentText() == "Optimization"
    assert page.timeline.value() == 18
    assert page.map_view.locality_items["khartoum"].status_text == "Critical Only"
    assert page.locality_context.text() == "Optimization · Hour 18:00"
    assert page.comparison_summary.values["Served energy"].text() == "8,100 MWh"
    assert "Optimization above" in page.comparison_summary.caption.text()
    assert page.comparison_summary.baselines["Served energy"].text() == "PPO 7,800 MWh"
    description = page.comparison_summary.toolTip()
    assert "zero-service blackout 14 locality-hours" in description
    assert "critical interruption 21 locality-hours" in description
    assert "mean pairwise gap 1.25 h" in description
    assert page.comparison_summary.accessibleDescription() == description
    assert page.comparison_table.rowCount() == 3
    assert page.comparison_table.item(2, 0).text() == "Optimization"


def test_optimization_is_keyboard_selectable(qtbot) -> None:
    page = LiveScenarioPage(load_map_asset())
    qtbot.addWidget(page)
    page.setFixedSize(1302, 644)
    page.set_comparison(_three_controller_comparison())
    page.show()
    page.activateWindow()
    page.map_panel.controller_buttons["PPO"].setFocus()
    qtbot.waitUntil(page.map_panel.controller_buttons["PPO"].hasFocus, timeout=1000)

    qtbot.keyClick(page.focusWidget(), Qt.Key.Key_Right)
    qtbot.keyClick(page.focusWidget(), Qt.Key.Key_Right)
    optimization = page.map_panel.controller_buttons["Optimization"]
    assert optimization.hasFocus()
    qtbot.keyClick(optimization, Qt.Key.Key_Space)

    assert optimization.isChecked()
    assert page.controller_combo.currentText() == "Optimization"
    assert page.comparison_summary.values["Served energy"].text() == "8,100 MWh"


@pytest.mark.parametrize("size", [(1302, 644), (1376, 776)])
def test_three_controller_layout_fits_1366_and_1440_windows(qtbot, qapp, size) -> None:
    apply_theme(qapp)
    page = LiveScenarioPage(load_map_asset())
    qtbot.addWidget(page)
    page.setFixedSize(*size)
    page.set_comparison(_three_controller_comparison())
    page.show()
    page.controller_combo.setCurrentText("Optimization")
    qtbot.wait(30)

    widgets = (
        page.shortage_combo,
        page.seed_input,
        *page.map_panel.controller_buttons.values(),
        page.run_button,
        page.map_view,
        page.locality_card,
        page.comparison_summary,
    )
    for widget in widgets:
        position = widget.mapTo(page, QPoint(0, 0))
        assert widget.isVisibleTo(page)
        assert position.x() >= 0 and position.y() >= 0
        assert position.x() + widget.width() <= page.width()
        assert position.y() + widget.height() <= page.height()
    for button in page.map_panel.controller_buttons.values():
        assert button.height() >= 44
        assert button.width() >= button.fontMetrics().horizontalAdvance(button.text()) + 24
    assert page.map_view.width() >= 650
    assert page.map_view.height() >= 240
    for value in page.comparison_summary.values.values():
        assert value.height() >= value.sizeHint().height()


def test_comparison_rejects_optimization_with_different_inputs() -> None:
    comparison = _three_controller_comparison()
    frames = list(comparison.frames("Optimization"))
    frames[0] = replace(frames[0], supply_mw=frames[0].supply_mw + 1.0)

    with pytest.raises(ValueError, match="same demand and supply"):
        replace(
            comparison,
            frames_by_controller={**comparison.frames_by_controller, "Optimization": frames},
        )


def test_duration_summary_rejects_out_of_horizon_values() -> None:
    duration = _three_controller_comparison().metrics("Optimization").duration
    with pytest.raises(ValueError, match="longest_blackout_hours"):
        replace(duration, longest_blackout_hours=25.0)

"""Approved native live-page controls, data boundaries, and presentation geometry."""

from __future__ import annotations

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QAccessible, QColor
from PySide6.QtWidgets import QLabel, QPushButton

from gridguard.ui.live_page import LiveScenarioPage
from gridguard.ui.map_asset import load_map_asset
from gridguard.ui.theme import PALETTE, apply_theme
from tests.ui.test_live_page import _comparison


def test_approved_live_controls_run_actual_selected_inputs(qtbot) -> None:
    page = LiveScenarioPage(load_map_asset())
    qtbot.addWidget(page)
    page.show()

    assert page.run_button.text() == "Run scenario"
    assert page.shortage_combo.currentText() == "Moderate"
    assert page.seed_input.value() == 50002
    assert not page.timeline.isEnabled()
    with qtbot.waitSignal(page.run_requested) as emitted:
        qtbot.mouseClick(page.run_button, Qt.MouseButton.LeftButton)
    assert emitted.args == ["moderate", 50002]
    assert not page.timeline.isEnabled()
    assert page.state_label.property("state") == "loading"


def test_approved_live_controller_buttons_preserve_hour_and_real_comparison(qtbot) -> None:
    page = LiveScenarioPage(load_map_asset())
    qtbot.addWidget(page)
    page.show()
    page.set_comparison(_comparison())
    page.timeline.setValue(18)
    qtbot.mouseClick(page.map_panel.controller_buttons["Fixed Rotation"], Qt.MouseButton.LeftButton)

    assert page.controller_combo.currentText() == "Fixed Rotation"
    assert page.timeline.value() == 18
    assert page.comparison_summary.values["Served energy"].text() == "7,800 MWh"
    assert page.comparison_summary.baselines["Served energy"].text() == "Fixed Rotation 5,200 MWh"
    page.seed_input.setValue(50003)
    assert page.comparison_summary.values["Served energy"].text() == "—"
    assert not page.timeline.isEnabled()


@pytest.mark.parametrize("invalidation", ["seed", "shortage", "rerun", "error", "frames"])
def test_live_invalidation_clears_all_comparison_details(qtbot, invalidation) -> None:
    page = LiveScenarioPage(load_map_asset())
    qtbot.addWidget(page)
    comparison = _comparison()
    page.set_comparison(comparison)
    summary = page.comparison_summary
    assert "seed 50001" in summary.toolTip()
    assert "Curtailed energy" in summary.toolTip()

    if invalidation == "seed":
        page.seed_input.setValue(50002)
    elif invalidation == "shortage":
        page.shortage_combo.setCurrentText("Severe")
    elif invalidation == "rerun":
        page.run_button.click()
    elif invalidation == "error":
        page.show_run_error("Simulation failed")
    else:
        page.set_frames(comparison.frames("PPO"))

    assert summary.toolTip() == ""
    accessible = QAccessible.queryAccessibleInterface(summary)
    assert accessible.text(QAccessible.Text.Description) == (
        "No verified live demonstration metrics are available yet."
    )
    for value in summary.values.values():
        assert QAccessible.queryAccessibleInterface(value).text(QAccessible.Text.Name) == "—"


@pytest.mark.parametrize("control", ["PPO", "Fixed Rotation", "play", "locality", "zoom"])
def test_live_keyboard_focus_is_visibly_rendered(qtbot, qapp, control) -> None:
    apply_theme(qapp)
    page = LiveScenarioPage(load_map_asset())
    qtbot.addWidget(page)
    page.setFixedSize(1302, 644)
    page.set_comparison(_comparison())
    page.show()
    page.activateWindow()
    controls = {
        **page.map_panel.controller_buttons,
        "play": page.map_panel.play_button,
        "locality": page.locality_panel.locality_combo,
        "zoom": page.map_panel.findChildren(QPushButton, "liveMapButton")[0],
    }
    target = controls[control]
    page.run_button.setFocus()
    qtbot.mouseMove(page, QPoint(0, 0))
    qtbot.wait(10)
    unfocused = target.grab().toImage()
    geometry = target.geometry()
    for _ in range(30):
        qtbot.keyClick(page.focusWidget(), Qt.Key.Key_Tab)
        if control == "Fixed Rotation" and page.map_panel.controller_buttons["PPO"].hasFocus():
            qtbot.keyClick(page.focusWidget(), Qt.Key.Key_Right)
        if target.hasFocus():
            break
    assert target.hasFocus(), f"{control} must be reachable with Tab"
    focused = target.grab().toImage()
    assert target.geometry() == geometry
    assert focused != unfocused, f"{control} focus must change its rendered appearance"
    focus_color = QColor(PALETTE.focus)
    assert (
        sum(
            focused.pixelColor(x, y) == focus_color
            for x in range(focused.width())
            for y in range(focused.height())
        )
        >= 20
    ), f"{control} needs a visible focus outline"


@pytest.mark.parametrize("size", [(1302, 644), (1376, 776)])
def test_approved_live_layout_fits_native_presentation_content(qtbot, qapp, size) -> None:
    apply_theme(qapp)
    page = LiveScenarioPage(load_map_asset())
    qtbot.addWidget(page)
    page.setFixedSize(*size)
    page.show()
    page.set_comparison(_comparison())
    page.timeline.setValue(18)
    qtbot.wait(30)

    for widget in (
        page.controls,
        page.map_panel,
        page.locality_card,
        page.map_panel.timeline_bar,
        page.comparison_summary,
    ):
        position = widget.mapTo(page, QPoint(0, 0))
        assert position.x() >= 0 and position.y() >= 0
        assert position.x() + widget.width() <= page.width()
        assert position.y() + widget.height() <= page.height()
    assert page.map_view.width() >= 650
    assert page.map_view.height() >= 240
    assert page.locality_panel.locality_combo.height() >= 44
    assert page.map_panel.play_button.height() >= 44
    text = " ".join(
        widget.text()
        for widget in page.findChildren(QLabel) + page.findChildren(QPushButton)
        if widget.isVisibleTo(page)
    ).casefold()
    assert "gpu" not in text
    assert "cuda" not in text
    assert "display only" in page.locality_projection.text().casefold()

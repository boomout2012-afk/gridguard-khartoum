from __future__ import annotations

import inspect

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from gridguard.ui.live_components import (
    LiveComparisonTable,
    LiveLocalityPanel,
    LiveMapPanel,
    LiveScenarioControls,
)
from gridguard.ui.live_page import LiveScenarioPage
from gridguard.ui.map_asset import load_map_asset
from gridguard.ui.window import PresentationWindow
from gridguard.ui.window_components import (
    PageStack,
    PresentationNavigationRail,
    PresentationTrustStrip,
)


@pytest.mark.parametrize("orchestrator", (PresentationWindow, LiveScenarioPage))
def test_ui_orchestrators_stay_within_a_200_line_boundary(orchestrator: type) -> None:
    source_lines, _ = inspect.getsourcelines(orchestrator)

    assert len(source_lines) <= 200, (
        f"{orchestrator.__name__} grew to {len(source_lines)} lines; "
        "extract a cohesive UI component instead"
    )


def test_window_shell_components_preserve_navigation_and_trust_behavior(qtbot) -> None:
    stack = PageStack()
    rail = PresentationNavigationRail(PresentationWindow.PAGE_LABELS)
    trust_strip = PresentationTrustStrip("Test CUDA runtime", "benchmark-001")
    qtbot.addWidget(stack)
    qtbot.addWidget(rail)
    qtbot.addWidget(trust_strip)
    for label in PresentationWindow.PAGE_LABELS:
        stack.addTab(QWidget(), label)
    rail.page_selected.connect(stack.setCurrentIndex)
    stack.currentChanged.connect(rail.set_current_index)

    qtbot.mouseClick(rail.buttons[2], Qt.MouseButton.LeftButton)

    assert stack.currentIndex() == 2
    assert stack.tabText(2) == "Results"
    assert rail.buttons[2].isChecked()
    assert "current" in rail.buttons[2].accessibleDescription()
    assert trust_strip.runtime_status.toolTip() == "Test CUDA runtime"
    assert "benchmark-001" in trust_strip.accessibleDescription()


def test_live_page_exposes_component_widgets_through_existing_aliases(qtbot) -> None:
    page = LiveScenarioPage(load_map_asset())
    qtbot.addWidget(page)

    assert isinstance(page.controls, LiveScenarioControls)
    assert page.shortage_combo is page.controls.shortage_combo
    assert page.seed_input is page.controls.seed_input
    assert page.run_button is page.controls.run_button
    assert page.state_label is page.controls.state_label
    assert isinstance(page.comparison_table, LiveComparisonTable)
    assert isinstance(page.map_panel, LiveMapPanel)
    assert page.hour_label is page.map_panel.hour_label
    assert page.controller_combo is page.map_panel.controller_combo
    assert page.timeline is page.map_panel.timeline
    assert page.map_view is page.map_panel.map_view
    assert isinstance(page.locality_panel, LiveLocalityPanel)
    assert page.locality_list is page.locality_panel.locality_list
    assert page.locality_card is page.locality_panel.locality_card
    assert page.sector_service_labels is page.locality_panel.sector_service_labels


def test_live_controls_emit_normalized_inputs_and_keep_state_accessible(qtbot) -> None:
    controls = LiveScenarioControls()
    qtbot.addWidget(controls)
    controls.shortage_combo.setCurrentText("Severe")
    controls.seed_input.setValue(50003)

    with qtbot.waitSignal(controls.run_requested, timeout=1000) as emitted:
        qtbot.mouseClick(controls.run_button, Qt.MouseButton.LeftButton)

    assert emitted.args == ["severe", 50003]
    controls.set_busy(True)
    assert not controls.run_button.isEnabled()
    assert controls.state_label.property("state") == "loading"
    assert "shared CUDA scenario" in controls.state_label.text()

from __future__ import annotations

import threading

from PySide6.QtCore import Qt

from gridguard.domain import ServiceTier
from gridguard.ui.live_models import (
    LiveScenarioComparison,
    LiveScenarioFrame,
    LocalityHourDetail,
)
from gridguard.ui.map_asset import EXPECTED_LOCALITY_IDS
from gridguard.ui.presentation_model import PresentationSnapshot
from gridguard.ui.window import PresentationWindow


def _snapshot() -> PresentationSnapshot:
    return PresentationSnapshot.from_mapping(
        {
            "schema_version": 1,
            "benchmark_id": "proposal-ppo-vs-fixed-rotation-paired",
            "protocol": {
                "controllers": ["Fixed Rotation", "PPO"],
                "count_statement": "288 matched scenarios; 576 controller evaluations.",
                "matched_scenario_count": 288,
                "independent_base_draw_count": 96,
                "controller_evaluation_count": 576,
                "horizon_hours": 24,
                "shortage_levels": ["mild", "moderate", "severe"],
            },
            "aggregate": {
                "Fixed Rotation": {
                    "episode_reward": 13.0,
                    "critical_service_ratio": 0.42,
                    "power_balance_violation_rate": 0.0,
                    "served_mwh": 4500.0,
                },
                "PPO": {
                    "episode_reward": 23.0,
                    "critical_service_ratio": 0.99,
                    "power_balance_violation_rate": 0.0,
                    "served_mwh": 6500.0,
                },
            },
            "paired_inference": {
                "episode_reward": {
                    "mean_delta": 10.0,
                    "ci95_lower": 9.9,
                    "ci95_upper": 10.1,
                    "direction": "higher_is_better",
                    "role": "primary",
                    "delta_definition": "PPO - Fixed Rotation",
                },
                "critical_service_ratio": {
                    "mean_delta": 0.57,
                    "ci95_lower": 0.56,
                    "ci95_upper": 0.58,
                    "direction": "higher_is_better",
                    "role": "primary",
                    "delta_definition": "PPO - Fixed Rotation",
                },
                "served_mwh": {
                    "mean_delta": 2000.0,
                    "ci95_lower": 1980.0,
                    "ci95_upper": 2020.0,
                    "direction": "higher_is_better",
                    "role": "primary",
                    "delta_definition": "PPO - Fixed Rotation",
                },
            },
            "limitations": ["Synthetic simulation evidence; not real utility measurements."],
        }
    )


def test_window_has_presentation_pages_and_only_approved_controllers(qtbot) -> None:
    window = PresentationWindow(_snapshot(), runtime_label="Test CUDA runtime")
    qtbot.addWidget(window)

    assert [window.tabs.tabText(index) for index in range(window.tabs.count())] == [
        "Overview",
        "Live Scenario",
        "Results",
        "Methodology",
    ]
    assert [card.controller_name for card in window.results_page.controller_cards] == [
        "PPO",
        "Fixed Rotation",
    ]
    assert window.minimumSize().width() <= 1366
    assert window.minimumSize().height() <= 768


def test_live_controls_emit_selected_shortage_and_seed(qtbot) -> None:
    window = PresentationWindow(_snapshot(), runtime_label="Test CUDA runtime")
    qtbot.addWidget(window)
    window.live_page.shortage_combo.setCurrentText("Severe")
    window.live_page.seed_input.setValue(50003)

    with qtbot.waitSignal(window.scenario_run_requested, timeout=1000) as emitted:
        qtbot.mouseClick(window.live_page.run_button, Qt.MouseButton.LeftButton)

    assert emitted.args == ["severe", 50003]
    assert "requested" in window.live_page.state_label.text().casefold()


def test_results_table_displays_values_from_injected_evidence(qtbot) -> None:
    window = PresentationWindow(_snapshot(), runtime_label="Test CUDA runtime")
    qtbot.addWidget(window)

    table = window.results_page.results_table
    assert table.rowCount() == 3
    assert table.item(0, 0).text() == "Critical service"
    assert table.item(0, 1).text() == "99.72%" or table.item(0, 1).text() == "99.00%"
    assert table.item(0, 3).text() == "+57.00 pp  [56.00 pp, 58.00 pp]"
    assert table.accessibleName() == "Paired controller comparison results"


def test_results_page_keeps_safety_and_scope_limits_visible(qtbot) -> None:
    window = PresentationWindow(_snapshot(), runtime_label="Test CUDA runtime")
    qtbot.addWidget(window)

    safety = window.results_page.safety_note.text().casefold()
    scope = window.results_page.scope_note.text().casefold()
    assert "0.00% power-balance violation rate" in safety
    assert "both controllers" in safety
    assert "synthetic simulation evidence" in scope
    assert "decision support only" in scope
    assert "not field or operational evidence" in scope


def test_window_keeps_meaningful_state_before_live_integration(qtbot) -> None:
    window = PresentationWindow(_snapshot(), runtime_label="Test CUDA runtime")
    qtbot.addWidget(window)

    assert "No live scenario has been run" in window.live_page.state_label.text()
    assert window.live_page.map_view.selected_locality_id is not None


def test_page_shortcuts_remain_active_for_keyboard_presenters(qtbot) -> None:
    window = PresentationWindow(_snapshot(), runtime_label="Test CUDA runtime")
    qtbot.addWidget(window)
    window.show()
    window.activateWindow()
    window.tabs.setFocus()
    qtbot.waitUntil(window.isActiveWindow, timeout=1000)

    qtbot.keyClick(window.tabs, Qt.Key.Key_3, modifier=Qt.KeyboardModifier.ControlModifier)

    assert window.tabs.currentIndex() == 2


def _live_comparison() -> LiveScenarioComparison:
    controllers: dict[str, tuple[LiveScenarioFrame, ...]] = {}
    for controller_name, tier in (
        ("PPO", ServiceTier.FULL_SERVICE),
        ("Fixed Rotation", ServiceTier.OFF),
    ):
        controllers[controller_name] = tuple(
            LiveScenarioFrame(
                hour=hour,
                service_tiers={locality_id: tier for locality_id in EXPECTED_LOCALITY_IDS},
                demand_mw=440.0 + hour,
                supply_mw=310.0 + hour,
                locality_details={
                    locality_id: LocalityHourDetail(
                        demand_mw=(440.0 + hour) / len(EXPECTED_LOCALITY_IDS),
                        served_mw=(310.0 + hour) / len(EXPECTED_LOCALITY_IDS),
                        critical_service_ratio=0.95,
                        sector_service_ratio={
                            "Hospital": 1.0,
                            "Water": 0.9,
                            "Bakery": 0.8,
                            "Residential": 0.7,
                            "Commercial": 0.6,
                        },
                    )
                    for locality_id in EXPECTED_LOCALITY_IDS
                },
            )
            for hour in range(24)
        )
    return LiveScenarioComparison(
        shortage_level="severe",
        seed=50003,
        device="cuda:0",
        frames_by_controller=controllers,
    )


def test_live_run_executes_off_the_gui_thread_and_keeps_results_frozen(qtbot) -> None:
    worker_thread_ids: list[int] = []

    def runner(shortage_level: str, seed: int) -> LiveScenarioComparison:
        assert (shortage_level, seed) == ("severe", 50003)
        worker_thread_ids.append(threading.get_ident())
        return _live_comparison()

    window = PresentationWindow(
        _snapshot(),
        runtime_label="Test CUDA runtime",
        live_runner=runner,
    )
    qtbot.addWidget(window)
    frozen_reward = window.results_page.results_table.item(3 - 1, 1).text()
    window.live_page.shortage_combo.setCurrentText("Severe")
    window.live_page.seed_input.setValue(50003)

    qtbot.mouseClick(window.live_page.run_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: not window.live_run_active, timeout=3000)

    assert worker_thread_ids and worker_thread_ids[0] != threading.get_ident()
    assert window.live_page.timeline.isEnabled()
    assert window.live_page.controller_combo.currentText() == "PPO"
    assert window.live_page.map_view.locality_items["khartoum"].status_text == "● FULL"
    assert window.live_page.locality_context.text() == "PPO · Hour 00:00"
    assert window.live_page.sector_service_labels["Water"].text() == "90.0%"
    window.live_page.timeline.setValue(7)
    window.live_page.controller_combo.setCurrentText("Fixed Rotation")
    assert window.live_page.map_view.locality_items["khartoum"].status_text == "× OFF"
    assert window.live_page.timeline.value() == 7
    assert window.live_page.locality_context.text() == "Fixed Rotation · Hour 07:00"
    assert window.results_page.results_table.item(3 - 1, 1).text() == frozen_reward


def test_live_worker_error_is_shown_without_fabricated_frames(qtbot) -> None:
    def runner(shortage_level: str, seed: int) -> LiveScenarioComparison:
        raise RuntimeError(f"test failure for {shortage_level}/{seed}")

    window = PresentationWindow(
        _snapshot(),
        runtime_label="Test CUDA runtime",
        live_runner=runner,
    )
    qtbot.addWidget(window)

    qtbot.mouseClick(window.live_page.run_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: not window.live_run_active, timeout=3000)

    assert not window.live_page.timeline.isEnabled()
    assert "test failure" in window.live_page.state_label.text()

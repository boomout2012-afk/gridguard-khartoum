from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QFontDatabase, QFontMetrics, QRawFont
from PySide6.QtWidgets import QLabel

from gridguard.domain import ServiceTier
from gridguard.ui import theme
from gridguard.ui.live_models import (
    LiveEpisodeSummary,
    LiveScenarioComparison,
    LiveScenarioFrame,
    LocalityHourDetail,
)
from gridguard.ui.map_asset import EXPECTED_LOCALITY_IDS
from gridguard.ui.presentation_model import PresentationSnapshot
from gridguard.ui.theme import PALETTE
from gridguard.ui.window import PresentationWindow


def _snapshot() -> PresentationSnapshot:
    return PresentationSnapshot.from_mapping(
        {
            "schema": {"name": "gridguard.robustness_supplement", "version": 1},
            "evidence_role": "untouched_confirmatory",
            "provenance": {
                "artifact_id": "gridguard-robustness-supplement-v1",
                "source_revision": "a" * 40,
                "model_manifest": {
                    "path": "data/model/manifest.json",
                    "sha256": "b" * 64,
                    "size_bytes": 100,
                },
                "actor": {
                    "path": "data/model/ppo-policy.npz",
                    "sha256": "c" * 64,
                    "size_bytes": 200,
                },
                "scenario_generator": {
                    "path": "src/gridguard/scenarios.py",
                    "sha256": "d" * 64,
                    "size_bytes": 300,
                },
                "training_provenance": {
                    "path": "data/model/training-provenance.yaml",
                    "sha256": "e" * 64,
                    "size_bytes": 400,
                },
                "implementation_sources": [],
                "seed_separation": {"disjoint_from_all_documented_seeds": True},
            },
            "protocol": {
                "seeds": list(range(81001, 81011)),
                "episodes_per_seed": 32,
                "controllers": ["Fixed Rotation", "PPO"],
                "independent_base_draw_count": 320,
                "shortage_levels": ["mild", "moderate", "severe"],
                "matched_scenario_count": 960,
                "controller_evaluation_count": 1_920,
                "delta_definition": "PPO - Fixed Rotation",
                "bootstrap": {"replicates": 10_000, "seed": 20260826, "ci_percent": 95},
                "sensitivity_parameterization_count": 25,
                "sensitivity_matched_scenario_count": 24_000,
                "sensitivity_controller_evaluation_count": 48_000,
            },
            "metric_definitions": {
                "critical_outage_locality_hours": {
                    "unit": "locality-hours",
                    "direction": "lower_is_better",
                    "role": "primary",
                },
                "fairness": {
                    "unit": "ratio",
                    "direction": "higher_is_better",
                    "role": "primary",
                },
                "utilization": {
                    "unit": "ratio",
                    "direction": "higher_is_better",
                    "role": "primary",
                },
                "power_balance_violation_rate": {
                    "unit": "ratio",
                    "direction": "lower_is_better",
                    "role": "safety",
                },
            },
            "forecast_metric_definitions": {},
            "base_evaluation": {
                "controller_means": {
                    "Fixed Rotation": {
                        "critical_outage_locality_hours": 102.0,
                        "fairness": 0.88,
                        "utilization": 0.71,
                        "power_balance_violation_rate": 0.0,
                    },
                    "PPO": {
                        "critical_outage_locality_hours": 18.7,
                        "fairness": 0.97,
                        "utilization": 0.98,
                        "power_balance_violation_rate": 0.0,
                    },
                },
                "paired_inference": {
                    "critical_outage_locality_hours": {
                        "mean_delta": -83.3,
                        "ci95_lower": -84.4,
                        "ci95_upper": -82.2,
                        "result_classification": "favorable",
                    },
                    "fairness": {
                        "mean_delta": 0.09,
                        "ci95_lower": 0.08,
                        "ci95_upper": 0.10,
                        "result_classification": "favorable",
                    },
                    "utilization": {
                        "mean_delta": 0.27,
                        "ci95_lower": 0.26,
                        "ci95_upper": 0.28,
                        "result_classification": "favorable",
                    },
                },
                "power_balance": {"maximum_violation_rate": 0.0},
            },
            "sensitivity": {},
            "forecast_evaluation": {},
            "limitations": [
                "Synthetic demand traces do not establish real-world forecast accuracy.",
                "Observed interval direction is descriptive and is not a release gate.",
            ],
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
    assert [table.item(row, 0).text() for row in range(3)] == [
        "Critical-outage locality-hours",
        "Jain fairness",
        "Utilization",
    ]
    assert table.item(0, 1).text() == "18.7 locality-hours"
    assert table.item(0, 2).text() == "102.0 locality-hours"
    assert table.item(0, 3).text() == "-83.3 [-84.4, -82.2] locality-hours"
    assert table.item(1, 1).text() == "97.00%"
    assert table.item(1, 3).text() == "+9.00 [8.00, 10.00] pp"
    assert table.accessibleName() == "Paired controller comparison results"
    assert all(
        tuple(card.metric_value_labels)
        == (
            "critical_outage_locality_hours",
            "fairness",
            "utilization",
        )
        for card in window.results_page.controller_cards
    )


def test_results_controller_cards_do_not_clip_headlines_at_1366_by_768(qtbot, qapp) -> None:
    theme.apply_theme(qapp)
    snapshot = _snapshot()
    window = PresentationWindow(snapshot, runtime_label="Test CUDA runtime")
    qtbot.addWidget(window)
    window.resize(1366, 768)
    window.show()
    window.tabs.setCurrentIndex(2)
    qtbot.wait(20)

    for card in window.results_page.controller_cards:
        metric_names = {
            label.text(): label
            for label in card.findChildren(QLabel)
            if label.text() in {metric.label for metric in snapshot.metrics}
        }
        assert set(metric_names) == {metric.label for metric in snapshot.metrics}
        for label in (*metric_names.values(), *card.metric_value_labels.values()):
            assert label.width() >= label.sizeHint().width()
            assert label.height() >= label.sizeHint().height()
        assert card.height() >= card.minimumSizeHint().height()

    table = window.results_page.results_table
    for row in range(table.rowCount()):
        for column in range(table.columnCount()):
            item = table.item(row, column)
            painted_width = QFontMetrics(item.font()).horizontalAdvance(item.text()) + 12
            assert table.columnWidth(column) >= painted_width


def test_results_page_keeps_safety_and_scope_limits_visible(qtbot) -> None:
    window = PresentationWindow(_snapshot(), runtime_label="Test CUDA runtime")
    qtbot.addWidget(window)

    safety = window.results_page.safety_note.text().casefold()
    scope = window.results_page.scope_note.text().casefold()
    page_text = " ".join(
        label.text() for label in window.results_page.findChildren(QLabel)
    ).casefold()
    assert "maximum power-balance violation rate 0.00%" in safety
    assert "across both controllers" in safety
    assert "synthetic demand traces" in scope
    assert "decision support only" in scope
    assert "not field or operational evidence" in scope
    assert "untouched confirmatory evidence" in page_text
    assert "no favorable/adverse release gate" in page_text


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


def _contrast_ratio(foreground: str, background: str) -> float:
    def luminance(color: str) -> float:
        channels = [int(color[index : index + 2], 16) / 255 for index in (1, 3, 5)]
        linear = [
            channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
            for channel in channels
        ]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    lighter, darker = sorted((luminance(foreground), luminance(background)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def test_atlas_palette_is_light_and_meets_wcag_aa_text_contrast() -> None:
    assert {
        "canvas": PALETTE.canvas,
        "surface": PALETTE.surface,
        "ink": PALETTE.ink,
        "muted": PALETTE.muted,
        "nav": PALETTE.nav,
        "boundary": PALETTE.boundary,
        "nile": PALETTE.nile,
        "evidence": PALETTE.evidence,
        "full": PALETTE.full,
        "critical": PALETTE.critical,
        "off": PALETTE.off,
        "focus": PALETTE.focus,
    } == {
        "canvas": "#F4F0E6",
        "surface": "#FFFCF4",
        "ink": "#10282E",
        "muted": "#4F6366",
        "nav": "#123039",
        "boundary": "#70837F",
        "nile": "#087D8B",
        "evidence": "#A66A13",
        "full": "#15705C",
        "critical": "#975F19",
        "off": "#963A45",
        "focus": "#005FCC",
    }
    assert _contrast_ratio(PALETTE.canvas, "#000000") > 10
    assert _contrast_ratio(PALETTE.text, PALETTE.canvas) >= 4.5
    assert _contrast_ratio(PALETTE.text_muted, PALETTE.canvas) >= 4.5
    assert _contrast_ratio(PALETTE.rail_text, PALETTE.rail) >= 4.5
    assert _contrast_ratio(PALETTE.surface, PALETTE.accent_dark) >= 4.5
    assert _contrast_ratio(PALETTE.warning, PALETTE.loading_surface) >= 4.5


def test_approved_presentation_fonts_register_from_bundled_assets(qapp) -> None:
    font_directory = Path(theme.__file__).resolve().parents[1] / "assets" / "fonts"
    expected_true_type_files = {
        "SourceSerif4-Regular.ttf",
        "SourceSerif4-Semibold.ttf",
        "AtkinsonHyperlegible-Regular.ttf",
        "AtkinsonHyperlegible-Bold.ttf",
        "IBMPlexMono-Regular.ttf",
        "IBMPlexMono-SemiBold.ttf",
    }
    assert expected_true_type_files <= {path.name for path in font_directory.glob("*.ttf")}
    registered_families = theme.register_presentation_fonts()

    assert registered_families == (
        "Source Serif 4",
        "Atkinson Hyperlegible",
        "IBM Plex Mono",
    )
    available_families = set(QFontDatabase.families())
    assert set(registered_families) <= available_families
    theme.apply_theme(qapp)
    assert qapp.font().family() == "Atkinson Hyperlegible"
    body_font = QRawFont.fromFont(QFont(theme.BODY_FONT_FAMILY, 12))
    for status in theme.STATUS_TEXT.values():
        marker = status.split(maxsplit=1)[0]
        assert all(body_font.supportsCharacter(ord(character)) for character in marker)


def test_window_uses_176px_navigation_rail_and_verified_trust_strip(qtbot) -> None:
    window = PresentationWindow(_snapshot(), runtime_label="Test CUDA runtime")
    qtbot.addWidget(window)

    assert window.navigation_rail.minimumWidth() == 176
    assert window.navigation_rail.maximumWidth() == 176
    assert [button.text() for button in window.navigation_buttons] == [
        "01  Overview",
        "02  Live Scenario",
        "03  Results",
        "04  Methodology",
    ]
    assert window.trust_strip.accessibleName() == "Verified application trust status"
    trust_text = " ".join(
        label.text() for label in window.trust_strip.findChildren(QLabel)
    ).casefold()
    assert "offline" in trust_text
    assert "gpu verified" in trust_text
    assert "evidence locked" in trust_text


def test_navigation_keeps_visual_and_accessible_page_state_in_sync(qtbot) -> None:
    window = PresentationWindow(_snapshot(), runtime_label="Test CUDA runtime")
    qtbot.addWidget(window)

    qtbot.mouseClick(window.navigation_buttons[2], Qt.MouseButton.LeftButton)

    assert window.tabs.currentIndex() == 2
    assert window.navigation_buttons[2].isChecked()
    assert not window.navigation_buttons[0].isChecked()
    assert window.tabs.accessibleDescription() == "Current page: Results"
    assert window.navigation_buttons[2].accessibleName() == (
        "Open Results page, keyboard shortcut Ctrl+3"
    )


def test_navigation_focus_is_blue_and_distinct_from_current_page(qtbot, qapp) -> None:
    theme.apply_theme(qapp)
    window = PresentationWindow(_snapshot(), runtime_label="Test CUDA runtime")
    qtbot.addWidget(window)
    window.resize(1366, 768)
    window.show()
    window.tabs.setCurrentIndex(2)
    window.navigation_buttons[0].setFocus()
    qtbot.wait(20)

    focused = window.navigation_buttons[0]
    current = window.navigation_buttons[2]
    focused_edge = focused.grab().toImage().pixelColor(0, focused.height() // 2).name().upper()
    current_edge = current.grab().toImage().pixelColor(0, current.height() // 2).name().upper()

    assert focused.hasFocus()
    assert not focused.isChecked()
    assert current.isChecked()
    assert focused_edge == PALETTE.focus
    assert current_edge != PALETTE.focus


def test_results_page_displays_evidence_derived_provenance(qtbot) -> None:
    snapshot = _snapshot()
    window = PresentationWindow(snapshot, runtime_label="Test CUDA runtime")
    qtbot.addWidget(window)

    results = window.results_page
    assert results.provenance_id.text() == snapshot.artifact_id
    assert results.provenance_summary.text() == snapshot.count_statement
    assert "hash verified" in results.provenance_status.text().casefold()
    assert snapshot.artifact_id in results.provenance_card.accessibleDescription()
    assert str(snapshot.matched_scenario_count) in results.provenance_card.accessibleDescription()
    assert results.provenance_sha256.text() == snapshot.artifact_sha256
    assert snapshot.artifact_sha256 in results.provenance_sha256.toolTip()
    assert snapshot.artifact_sha256 in results.provenance_sha256.accessibleName()
    assert "robustness supplement" in results.provenance_sha256.toolTip().casefold()
    assert results.provenance_sha256.textInteractionFlags() & (
        Qt.TextInteractionFlag.TextSelectableByKeyboard
    )
    assert results.provenance_release.text() == snapshot.release_id
    assert snapshot.release_id in results.provenance_release.accessibleName()
    assert results.provenance_release.textInteractionFlags() & (
        Qt.TextInteractionFlag.TextSelectableByKeyboard
    )
    assert results.provenance_id.property("mono") is True
    assert all(
        value_label.property("mono") is True
        for card in results.controller_cards
        for value_label in card.metric_value_labels.values()
    )


def test_overview_visibly_attributes_verified_map_sources(qtbot) -> None:
    window = PresentationWindow(_snapshot(), runtime_label="Test CUDA runtime")
    qtbot.addWidget(window)

    attribution = window.overview_page.source_attribution
    assert "OCHA COD-AB 2020" in attribution.text()
    assert "Natural Earth 5.0" in attribution.text()
    assert window.overview_page.map_view.asset.source.source_url in attribution.toolTip()
    assert "Natural Earth 5.0" in attribution.accessibleName()


def test_methodology_is_scrollable_and_covers_the_approved_method(qtbot) -> None:
    snapshot = _snapshot()
    window = PresentationWindow(snapshot, runtime_label="Test CUDA runtime")
    qtbot.addWidget(window)

    page = window.methodology_page
    page_text = " ".join(label.text() for label in page.findChildren(QLabel)).casefold()
    assert page.scroll_area.widgetResizable()
    for topic in (
        "synthetic scenario generation",
        "service tiers",
        "critical-service priorities",
        "fixed rotation",
        "ppo",
        "training-objective score (diagnostic)",
        "matched evaluation",
        "untouched confirmatory evidence",
        "no favorable/adverse release gate",
        "limitations",
        "keyboard",
    ):
        assert topic in page_text
    assert str(snapshot.independent_base_draw_count) in page_text
    assert str(snapshot.matched_scenario_count) in page_text


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
        metrics_by_controller={
            "PPO": LiveEpisodeSummary(
                served_mwh=7716.0,
                critical_service_ratio=0.95,
                fairness=0.91,
                curtailed_mwh=3120.0,
                power_balance_violation_rate=0.0,
            ),
            "Fixed Rotation": LiveEpisodeSummary(
                served_mwh=6500.0,
                critical_service_ratio=0.75,
                fairness=0.82,
                curtailed_mwh=4336.0,
                power_balance_violation_rate=0.0,
            ),
        },
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
    frozen_headline = window.results_page.results_table.item(2, 1).text()
    window.live_page.shortage_combo.setCurrentText("Severe")
    window.live_page.seed_input.setValue(50003)

    qtbot.mouseClick(window.live_page.run_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: not window.live_run_active, timeout=3000)

    assert worker_thread_ids and worker_thread_ids[0] != threading.get_ident()
    assert window.live_page.timeline.isEnabled()
    assert window.live_page.controller_combo.currentText() == "PPO"
    assert window.live_page.map_view.locality_items["khartoum"].status_text == "• FULL"
    assert window.live_page.locality_context.text() == "PPO · Hour 00:00"
    assert window.live_page.sector_service_labels["Water"].text() == "90.0%"
    window.live_page.timeline.setValue(7)
    window.live_page.controller_combo.setCurrentText("Fixed Rotation")
    assert window.live_page.map_view.locality_items["khartoum"].status_text == "× OFF"
    assert window.live_page.timeline.value() == 7
    assert window.live_page.locality_context.text() == "Fixed Rotation · Hour 07:00"
    assert window.results_page.results_table.item(2, 1).text() == frozen_headline


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

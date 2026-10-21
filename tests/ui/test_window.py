from __future__ import annotations

import threading
from pathlib import Path

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QFont, QFontDatabase, QFontMetrics, QRawFont
from PySide6.QtWidgets import QLabel

from gridguard.domain import ServiceTier
from gridguard.ui import presentation_model, theme
from gridguard.ui.live_models import (
    LiveEpisodeSummary,
    LiveScenarioComparison,
    LiveScenarioFrame,
    LocalityHourDetail,
)
from gridguard.ui.map_asset import EXPECTED_LOCALITY_IDS
from gridguard.ui.methodology_page import MethodologyPage
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
                "Forecast-only limitation must not define the comparator scope note.",
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
    ]
    assert all(
        list(panel.value_labels) == ["PPO", "Fixed Rotation"]
        for panel in window.results_page.outcome_panels.values()
    )
    assert window.minimumSize().width() <= 1366
    assert window.minimumSize().height() <= 768


def test_live_controls_emit_selected_shortage_and_seed(qtbot) -> None:
    window = PresentationWindow(_snapshot(), runtime_label="Test CUDA runtime")
    qtbot.addWidget(window)
    window.live_page.shortage_combo.setCurrentText("Severe")
    window.live_page.seed_input.setValue(50003)
    assert "No live scenario has been run" in window.live_page.state_label.text()

    with qtbot.waitSignal(window.scenario_run_requested, timeout=1000) as emitted:
        qtbot.mouseClick(window.live_page.run_button, Qt.MouseButton.LeftButton)

    assert emitted.args == ["severe", 50003]
    assert "requested" in window.live_page.state_label.text().casefold()


def test_results_table_displays_values_from_injected_evidence(qtbot) -> None:
    window = PresentationWindow(_snapshot(), runtime_label="Test CUDA runtime")
    qtbot.addWidget(window)

    table = window.results_page.results_table
    assert table.rowCount() == 3
    assert [table.horizontalHeaderItem(column).text() for column in range(4)] == [
        "Metric",
        "PPO",
        "Fixed Rotation",
        "PPO − Fixed Rotation",
    ]
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
    assert tuple(window.results_page.outcome_panels) == (
        "critical_outage_locality_hours",
        "fairness",
        "utilization",
    )


def test_results_comparisons_do_not_clip_headlines_at_1366_by_768(qtbot, qapp) -> None:
    theme.apply_theme(qapp)
    snapshot = _snapshot()
    window = PresentationWindow(snapshot, runtime_label="Test CUDA runtime")
    qtbot.addWidget(window)
    window.resize(1366, 768)
    window.show()
    window.tabs.setCurrentIndex(2)
    qtbot.wait(20)

    for panel in window.results_page.outcome_panels.values():
        for label in (panel.title_label, *panel.value_labels.values()):
            assert label.width() >= label.sizeHint().width()
            assert label.height() >= label.sizeHint().height()
        assert panel.height() >= panel.minimumSizeHint().height()

    table = window.results_page.results_table
    for row in range(table.rowCount()):
        for column in range(table.columnCount()):
            item = table.item(row, column)
            painted_width = QFontMetrics(item.font()).horizontalAdvance(item.text()) + 12
            assert table.columnWidth(column) >= painted_width


def test_results_page_keeps_safety_and_scope_limits_visible(qtbot) -> None:
    window = PresentationWindow(_snapshot(), runtime_label="Test CUDA runtime")
    qtbot.addWidget(window)
    window.resize(1366, 768)
    window.show()
    window.tabs.setCurrentIndex(2)
    qtbot.waitUntil(window.results_page.scope_note.isVisible, timeout=1000)

    safety = window.results_page.safety_note.text().casefold()
    scope_note = window.results_page.scope_note
    expected_scope = (
        "Comparator boundary: Fixed Rotation is supply-blind, limited to Off and Full Service, and "
        "cannot select Critical Only. Synthetic decision support only, not field evidence. No "
        "behavior-cloning-only ablation was run, so this comparison cannot isolate the "
        "incremental effect of PPO fine-tuning."
    )
    scope = scope_note.text().casefold()
    page_text = " ".join(
        label.text() for label in window.results_page.findChildren(QLabel)
    ).casefold()
    evidence_role_text = window.results_page.evidence_role_note.text().casefold()
    evidence_role_accessible = window.results_page.evidence_role_note.accessibleName().casefold()
    table_description = window.results_page.results_table.accessibleDescription().casefold()
    assert "maximum power-balance violation rate 0.00%" in safety
    assert "across both controllers" in safety
    assert expected_scope == presentation_model.RESULTS_SCOPE_LIMITATION
    assert "supply-blind" in scope
    assert "cannot select critical only" in scope
    assert "forecast-only limitation" not in scope
    assert scope_note.wordWrap()
    assert expected_scope in scope_note.accessibleName()
    assert expected_scope in window.results_page.accessibleDescription()
    assert scope_note.geometry().bottom() <= window.results_page.contentsRect().bottom()
    assert "documented-seed-disjoint holdout" in page_text
    assert "not preregistered or independently custodied" in page_text
    assert "cannot rule out undocumented activity" in page_text
    assert "sensitivity: pointwise 95%" in evidence_role_text
    assert "no multiplicity adjustment or result gate" in evidence_role_text
    assert "base paired 95% confidence intervals below" in evidence_role_accessible
    assert "sensitivity supplement (not shown here)" in evidence_role_accessible
    assert "unadjusted for multiple comparisons" in evidence_role_accessible
    assert "sensitivity supplement is not shown in this table" in table_description
    assert "untouched confirmatory" not in page_text
    assert "no favorable/adverse release gate" in evidence_role_accessible


def test_results_inference_banner_is_fully_visible_at_presentation_width(qtbot) -> None:
    window = PresentationWindow(_snapshot(), runtime_label="Test CUDA runtime")
    qtbot.addWidget(window)
    window.resize(1366, 768)
    window.show()
    window.tabs.setCurrentIndex(2)
    qtbot.mouseClick(window.results_page.evidence_details_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(window.results_page.evidence_role_note.isVisible, timeout=1000)

    banner = window.results_page.evidence_role_note
    assert banner.wordWrap()
    assert banner.height() >= banner.heightForWidth(banner.width())
    accessible = banner.accessibleName().casefold()
    assert "base paired 95% confidence intervals below" in accessible
    assert "sensitivity supplement (not shown here)" in accessible
    assert "unadjusted for multiple comparisons" in accessible
    assert "no favorable/adverse release gate" in accessible


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


def test_window_uses_approved_top_navigation_without_a_trust_strip(qtbot) -> None:
    window = PresentationWindow(_snapshot(), runtime_label="Test CUDA runtime")
    qtbot.addWidget(window)

    assert window.navigation_bar.minimumHeight() == 64
    assert window.navigation_bar.maximumHeight() == 64
    assert [button.text() for button in window.navigation_buttons] == [
        "Overview",
        "Live Scenario",
        "Results",
    ]
    assert not hasattr(window, "trust_strip")
    assert not hasattr(window, "methodology_page")


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
        value_label.property("metric") == "value"
        for panel in results.outcome_panels.values()
        for value_label in panel.value_labels.values()
    )


def test_overview_visibly_attributes_verified_map_sources(qtbot) -> None:
    window = PresentationWindow(_snapshot(), runtime_label="Test CUDA runtime")
    qtbot.addWidget(window)

    attribution = window.overview_page.source_attribution
    assert "OCHA COD-AB 2020" in attribution.text()
    assert "Natural Earth 5.0" in attribution.text()
    assert window.overview_page.map_view.asset.source.source_url in attribution.toolTip()
    assert "Natural Earth 5.0" in attribution.accessibleName()


def test_overview_opens_live_workspace_without_inventing_a_run(qtbot) -> None:
    window = PresentationWindow(_snapshot(), runtime_label="Test CUDA runtime")
    qtbot.addWidget(window)
    qtbot.mouseClick(window.overview_page.explore_button, Qt.MouseButton.LeftButton)
    assert window.tabs.currentIndex() == 1
    assert not window.live_page.timeline.isEnabled()


def test_overview_scope_facts_are_not_clipped_at_presentation_size(qtbot, qapp) -> None:
    theme.apply_theme(qapp)
    window = PresentationWindow(_snapshot(), runtime_label="Test CUDA runtime")
    qtbot.addWidget(window)
    window.setFixedSize(1366, 768)
    window.show()
    qtbot.wait(20)
    values = window.overview_page.scope_fact_labels.values()
    assert len(values) == 3
    for label in values:
        assert label.height() >= label.sizeHint().height()
        assert label.width() >= label.sizeHint().width()


def test_playback_tracks_hours_and_stops_at_end_or_when_leaving_page(qtbot) -> None:
    window = PresentationWindow(_snapshot(), runtime_label="Test CUDA runtime")
    qtbot.addWidget(window)
    window.show()
    window.tabs.setCurrentIndex(1)
    page = window.live_page
    assert not page.map_panel.play_button.isEnabled()
    page.set_comparison(_live_comparison())
    page.map_panel.playback_timer.setInterval(20)
    page.timeline.setValue(22)
    qtbot.mouseClick(page.map_panel.play_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: page.timeline.value() == 23)
    assert not page.map_panel.playback_timer.isActive()
    qtbot.mouseClick(page.map_panel.play_button, Qt.MouseButton.LeftButton)
    assert page.timeline.value() == 0
    assert page.map_panel.playback_timer.isActive()
    window.tabs.setCurrentIndex(2)
    assert not page.map_panel.playback_timer.isActive()


def test_trace_follows_selected_hour_and_clears_on_input_change(qtbot) -> None:
    window = PresentationWindow(_snapshot(), runtime_label="Test CUDA runtime")
    qtbot.addWidget(window)
    page = window.live_page
    page.set_comparison(_live_comparison())
    page.timeline.setValue(7)
    assert page.power_trace.selected_hour == 7
    assert "447" in page.power_trace.accessibleDescription()
    assert "317" in page.power_trace.accessibleDescription()
    page.seed_input.setValue(50005)
    assert page.power_trace.frames == ()
    assert "unavailable" in page.power_trace.accessibleDescription()
    assert not page.map_panel.play_button.isEnabled()


def test_live_details_remain_readable_at_the_allowed_minimum(qtbot, qapp) -> None:
    theme.apply_theme(qapp)
    window = PresentationWindow(_snapshot(), runtime_label="Test CUDA runtime")
    qtbot.addWidget(window)
    window.resize(window.minimumSize())
    window.live_page.set_comparison(_live_comparison())
    window.tabs.setCurrentIndex(1)
    window.show()
    qtbot.wait(20)
    for label in window.live_page.sector_service_labels.values():
        assert label.height() >= label.sizeHint().height()
    clock = window.live_page.map_panel.clock_label
    assert clock.width() >= clock.sizeHint().width()


def test_overview_keeps_synthetic_scope_visible_and_links_to_results(qtbot) -> None:
    window = PresentationWindow(_snapshot(), runtime_label="Test CUDA runtime")
    qtbot.addWidget(window)
    window.resize(1366, 768)
    window.show()
    window.tabs.setCurrentIndex(0)
    qtbot.waitUntil(window.overview_page.scope_note.isVisible, timeout=1000)

    note = window.overview_page.scope_note
    visible = note.text().casefold()
    assert "synthetic" in visible
    assert "decision support only" in visible
    assert note.width() >= note.sizeHint().width()
    assert note.height() >= note.sizeHint().height()
    bottom = note.mapTo(window.overview_page, QPoint(0, note.height())).y()
    assert bottom <= window.overview_page.contentsRect().bottom()
    qtbot.mouseClick(window.overview_page.results_button, Qt.MouseButton.LeftButton)
    assert window.tabs.currentWidget() is window.results_page


def test_methodology_is_scrollable_and_covers_the_approved_method(qtbot) -> None:
    snapshot = _snapshot()
    page = MethodologyPage(snapshot)
    qtbot.addWidget(page)
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
        "documented-seed-disjoint holdout",
        "not preregistered or independently custodied",
        "unadjusted for multiple comparisons",
        "behavior-cloning-only ablation",
        "no favorable/adverse release gate",
        "limitations",
        "keyboard",
    ):
        assert topic in page_text
    assert str(snapshot.independent_base_draw_count) in page_text
    assert str(snapshot.matched_scenario_count) in page_text


def test_methodology_shows_verified_historical_training_record_at_presentation_size(qtbot) -> None:
    snapshot = _snapshot()
    page = MethodologyPage(snapshot)
    qtbot.addWidget(page)
    page.resize(1366, 768)
    page.show()
    qtbot.waitUntil(page.isVisible, timeout=1000)
    card = page.training_record_card
    card_text = " ".join(label.text() for label in card.findChildren(QLabel))

    assert card.accessibleName() == "Verified historical PPO training record"
    assert card.accessibleDescription() == snapshot.training_record.accessible_summary
    assert "Historical PPO training record" in card_text
    assert "VERIFIED PRESENTATION SUMMARY" in card_text
    assert "CANONICAL YAML · data/model/training-provenance.yaml" in card_text
    assert "Actor topology · 32 -> 512 -> 512 -> 2,187 logits" in card_text
    assert (
        "Behavior cloning · 32,768 teacher samples · rollout batch 128 · "
        "minibatch 2,048 · 100 epochs · 1,600 optimizer updates"
    ) in card_text
    assert (
        "Selected seed-43 run · 600 iterations · 153,600 complete episodes · 3,686,400 transitions"
    ) in card_text
    assert (
        "Three-seed candidate experiment · 1,800 iterations · 460,800 complete episodes · "
        "11,059,200 transitions"
    ) in card_text
    assert "PPO batching · 256 episodes/iteration · minibatch 2,048 · 4 epochs/iteration" in (
        card_text
    )
    assert "Learning · LR 0.0001 · gamma 0.99 · GAE lambda 0.95 · clip epsilon 0.20" in (card_text)
    assert (
        "Seeds and selection · training 43 · candidates 42, 43, 44 · validation "
        "70000, 70001, 70002 · selected seed 43 at iteration 510"
    ) in card_text
    assert "historical record, not an executable retraining configuration" in card_text.casefold()
    assert page.scroll_area.horizontalScrollBar().maximum() == 0


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
    assert window.live_page.map_view.locality_items["khartoum"].status_text == "Full Service"
    assert window.live_page.locality_context.text() == "PPO · Hour 00:00"
    assert window.live_page.sector_service_labels["Water"].text() == "90.0%"
    window.live_page.timeline.setValue(7)
    window.live_page.controller_combo.setCurrentText("Fixed Rotation")
    assert window.live_page.map_view.locality_items["khartoum"].status_text == "Off"
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


def _assert_live_output_is_unavailable(window: PresentationWindow) -> None:
    page = window.live_page
    unavailable_hour = page.hour_label.text().casefold()
    assert "hour" in unavailable_hour
    assert "demand unavailable" in unavailable_hour
    assert "supply unavailable" in unavailable_hour
    assert page.controller_combo.currentText() == "PPO"
    assert not page.controller_combo.isEnabled()
    assert page.timeline.value() == 0
    assert not page.timeline.isEnabled()
    assert all(
        map_item.status_text.casefold().endswith("not run")
        for map_item in page.map_view.locality_items.values()
    )
    assert all(map_item.service_tier is None for map_item in page.map_view.locality_items.values())
    assert all(
        page.locality_list.item(row).text().casefold().endswith("not run")
        for row in range(page.locality_list.count())
    )
    assert page.locality_context.text() == "No controller output \N{MIDDLE DOT} Hour \N{EM DASH}"
    assert page.locality_status.text() == "Not run"
    assert page.locality_demand.text() == "\N{EM DASH}"
    assert page.locality_served.text() == "\N{EM DASH}"
    assert page.locality_critical.text() == "\N{EM DASH}"
    assert "Next hour: unavailable" in page.locality_projection.text()
    assert all(label.text() == "\N{EM DASH}" for label in page.sector_service_labels.values())
    assert "no verified live demonstration metrics" in (
        page.comparison_table.accessibleDescription().casefold()
    )
    assert all(
        page.comparison_table.item(row, column).text() == "\N{EM DASH}"
        for row in range(page.comparison_table.rowCount())
        for column in range(1, page.comparison_table.columnCount())
    )


def test_starting_a_new_live_request_clears_the_previous_scientific_output(qtbot) -> None:
    release_runner = threading.Event()

    def runner(_shortage_level: str, _seed: int) -> LiveScenarioComparison:
        release_runner.wait(timeout=3)
        return _live_comparison()

    window = PresentationWindow(
        _snapshot(),
        runtime_label="Test CUDA runtime",
        live_runner=runner,
    )
    qtbot.addWidget(window)
    page = window.live_page
    page.set_comparison(_live_comparison())
    page.timeline.setValue(7)
    page.controller_combo.setCurrentText("Fixed Rotation")

    qtbot.mouseClick(page.run_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window.live_run_active, timeout=1000)

    try:
        _assert_live_output_is_unavailable(window)
        assert page.state_label.property("state") == "loading"
    finally:
        release_runner.set()
        qtbot.waitUntil(lambda: not window.live_run_active, timeout=3000)


def test_live_run_error_clears_the_previous_scientific_output(qtbot) -> None:
    window = PresentationWindow(_snapshot(), runtime_label="Test CUDA runtime")
    qtbot.addWidget(window)
    page = window.live_page
    page.set_comparison(_live_comparison())
    page.timeline.setValue(7)
    page.controller_combo.setCurrentText("Fixed Rotation")

    page.show_run_error("test integration failure")

    _assert_live_output_is_unavailable(window)
    assert page.state_label.property("state") == "error"
    assert "test integration failure" in page.state_label.text()


@pytest.mark.parametrize(
    ("input_name", "changed_value"),
    (
        ("shortage_combo", "Moderate"),
        ("seed_input", 50004),
    ),
    ids=("shortage-level", "seed"),
)
def test_changing_live_inputs_invalidates_completed_scientific_output(
    qtbot,
    input_name: str,
    changed_value: str | int,
) -> None:
    window = PresentationWindow(_snapshot(), runtime_label="Test CUDA runtime")
    qtbot.addWidget(window)
    page = window.live_page
    page.set_comparison(_live_comparison())
    page.timeline.setValue(7)
    page.controller_combo.setCurrentText("Fixed Rotation")

    control = getattr(page, input_name)
    if input_name == "shortage_combo":
        control.setCurrentText(changed_value)
    else:
        control.setValue(changed_value)

    _assert_live_output_is_unavailable(window)
    assert page.state_label.property("state") == "empty"
    assert "inputs changed" in page.state_label.text().casefold()
    assert "run required" in page.state_label.text().casefold()

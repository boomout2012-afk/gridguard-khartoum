from __future__ import annotations

import pytest
from PySide6.QtCore import QPoint, Qt

from gridguard.domain import ControllerName, ServiceTier
from gridguard.ui.live_models import (
    LiveEpisodeSummary,
    LiveScenarioComparison,
    LiveScenarioFrame,
    LocalityHourDetail,
)
from gridguard.ui.live_page import LiveScenarioPage
from gridguard.ui.map_asset import EXPECTED_LOCALITY_IDS, load_map_asset
from gridguard.ui.theme import apply_theme


def _frame(hour: int, tier: ServiceTier = ServiceTier.FULL_SERVICE) -> LiveScenarioFrame:
    demand_per_locality = (440.0 + hour) / len(EXPECTED_LOCALITY_IDS)
    served_per_locality = (310.0 + hour) / len(EXPECTED_LOCALITY_IDS)
    return LiveScenarioFrame(
        hour=hour,
        service_tiers={locality_id: tier for locality_id in EXPECTED_LOCALITY_IDS},
        demand_mw=440.0 + hour,
        supply_mw=310.0 + hour,
        locality_details={
            locality_id: LocalityHourDetail(
                demand_mw=demand_per_locality,
                served_mw=served_per_locality,
                critical_service_ratio=0.95,
                sector_service_ratio={
                    "Hospital": 1.0,
                    "Water": 0.9,
                    "Bakery": 0.8,
                    "Residential": 0.7,
                    "Commercial": 0.6,
                },
                next_hour_projected_demand_mw=(
                    demand_per_locality + 1.0 if 2 <= hour <= 22 else None
                ),
            )
            for locality_id in EXPECTED_LOCALITY_IDS
        },
    )


def _summary(*, served_mwh: float) -> LiveEpisodeSummary:
    return LiveEpisodeSummary(
        served_mwh=served_mwh,
        critical_service_ratio=0.95,
        fairness=0.91,
        curtailed_mwh=120.0,
        power_balance_violation_rate=0.0,
    )


def _comparison() -> LiveScenarioComparison:
    frames = tuple(_frame(hour) for hour in range(24))
    return LiveScenarioComparison(
        shortage_level="mild",
        seed=50001,
        device="cuda:0",
        frames_by_controller={
            ControllerName.PPO.value: frames,
            ControllerName.FIXED_ROTATION.value: frames,
        },
        metrics_by_controller={
            ControllerName.PPO.value: _summary(served_mwh=7_800.0),
            ControllerName.FIXED_ROTATION.value: _summary(served_mwh=5_200.0),
        },
    )


def test_live_frame_requires_every_locality_and_finite_power() -> None:
    with pytest.raises(ValueError, match="all seven"):
        LiveScenarioFrame(
            hour=0,
            service_tiers={"khartoum": ServiceTier.OFF},
            demand_mw=440.0,
            supply_mw=310.0,
            locality_details={},
        )

    with pytest.raises(ValueError, match="finite"):
        LiveScenarioFrame(
            hour=0,
            service_tiers={locality_id: ServiceTier.OFF for locality_id in EXPECTED_LOCALITY_IDS},
            demand_mw=float("nan"),
            supply_mw=310.0,
            locality_details={},
        )


def test_page_displays_only_frames_supplied_by_simulation_integration(qtbot) -> None:
    page = LiveScenarioPage(load_map_asset())
    qtbot.addWidget(page)
    frames = [_frame(hour) for hour in range(24)]
    frames[7] = _frame(7, ServiceTier.CRITICAL_ONLY)

    page.set_frames(frames)
    assert "unavailable" in page.locality_projection.text()
    assert "0.0 MW" not in page.locality_projection.text()
    page.timeline.setValue(7)

    assert page.timeline.isEnabled()
    assert page.hour_label.text() == "Hour 07:00  ·  demand 447 MW  ·  supply 317 MW"
    assert page.map_view.locality_items["khartoum"].status_text == "! CRITICAL ONLY"
    assert page.locality_demand.text() == f"{(447 / 7):.1f} MW"
    assert page.locality_served.text() == f"{(317 / 7):.1f} MW"
    assert page.locality_critical.text() == "95.0%"
    assert page.sector_service_labels["Hospital"].text() == "100.0%"
    assert f"{((447 / 7) + 1):.1f} MW" in page.locality_projection.text()
    assert "DISPLAY ONLY" in page.locality_projection.text()
    assert "not PPO input/evidence" in page.locality_projection.text()
    assert "complete" in page.state_label.text().casefold()


def test_live_frame_rejects_invalid_next_hour_projection() -> None:
    with pytest.raises(ValueError, match="projected"):
        LocalityHourDetail(
            demand_mw=10.0,
            served_mw=8.0,
            critical_service_ratio=0.9,
            sector_service_ratio={
                sector: 0.8
                for sector in (
                    "Hospital",
                    "Water",
                    "Bakery",
                    "Residential",
                    "Commercial",
                )
            },
            next_hour_projected_demand_mw=float("nan"),
        )


def test_live_episode_summary_rejects_invalid_ratios() -> None:
    with pytest.raises(ValueError, match="fairness"):
        LiveEpisodeSummary(
            served_mwh=1.0,
            critical_service_ratio=0.9,
            fairness=1.01,
            curtailed_mwh=0.0,
            power_balance_violation_rate=0.0,
        )


def test_page_labels_matched_aggregate_metrics_as_live_demonstration(qtbot) -> None:
    page = LiveScenarioPage(load_map_asset())
    qtbot.addWidget(page)

    page.set_comparison(_comparison())

    assert page.comparison_table.accessibleName() == "Live demonstration aggregate comparison"
    assert page.comparison_table.horizontalHeaderItem(0).text() == "Controller"
    assert page.comparison_table.horizontalHeaderItem(1).text() == "Served energy"
    assert page.comparison_table.item(0, 0).text() == ControllerName.PPO.value
    assert page.comparison_table.item(0, 1).text() == "7,800 MWh"
    assert page.comparison_table.item(0, 2).text() == "95.0%"
    assert page.comparison_table.item(0, 3).text() == "91.0%"
    assert page.comparison_table.item(0, 5).text() == "0.0%"
    assert "live demonstration" in page.comparison_caption.text().casefold()


def test_visible_locality_list_is_bidirectionally_synchronized_with_map(qtbot) -> None:
    page = LiveScenarioPage(load_map_asset())
    qtbot.addWidget(page)

    assert page.locality_list.accessibleName() == "Khartoum localities"
    assert page.locality_list.count() == 7
    east_nile_row = next(
        row
        for row in range(page.locality_list.count())
        if page.locality_list.item(row).data(Qt.ItemDataRole.UserRole) == "east_nile"
    )

    page.locality_list.setCurrentRow(east_nile_row)
    assert page.map_view.selected_locality_id == "east_nile"

    page.map_view.select_locality("omdurman")
    assert page.locality_list.currentItem().data(Qt.ItemDataRole.UserRole) == "omdurman"


def test_dynamic_status_labels_do_not_mask_their_text_from_accessibility(qtbot) -> None:
    page = LiveScenarioPage(load_map_asset())
    qtbot.addWidget(page)

    assert page.state_label.accessibleName() == ""
    assert page.locality_projection.accessibleName() == ""
    assert page.state_label.property("state") == "empty"

    page.set_busy(True)
    assert "running both controllers" in page.state_label.text().casefold()
    assert page.state_label.property("state") == "loading"

    page.show_run_error("test integration failure")
    assert "test integration failure" in page.state_label.text().casefold()
    assert page.state_label.property("state") == "error"


def test_locality_detail_rows_fit_at_presentation_height(qtbot, qapp) -> None:
    apply_theme(qapp)
    page = LiveScenarioPage(load_map_asset())
    qtbot.addWidget(page)
    page.setFixedSize(1190, 725)
    page.show()
    page.set_frames([_frame(hour, ServiceTier.CRITICAL_ONLY) for hour in range(24)])
    page.timeline.setValue(18)
    qtbot.wait(50)

    detail_labels = (
        page.locality_name,
        page.locality_context,
        page.locality_status,
        page.locality_demand,
        page.locality_served,
        page.locality_critical,
        page.locality_projection,
        *page.sector_service_labels.values(),
    )
    for label in detail_labels:
        bottom = label.mapTo(page.locality_card, QPoint(0, label.height())).y()
        assert label.isVisibleTo(page.locality_card)
        assert label.height() >= label.sizeHint().height()
        assert bottom <= page.locality_card.height()
    viewport_height = page.locality_list.viewport().height()
    for row in range(page.locality_list.count()):
        item = page.locality_list.item(row)
        item_rect = page.locality_list.visualItemRect(item)
        assert item_rect.top() >= 0
        assert item_rect.bottom() < viewport_height
        # Leave enough room for the item delegate's horizontal padding and
        # antialiasing so Qt never substitutes an ellipsis at presentation size.
        assert page.locality_list.fontMetrics().horizontalAdvance(item.text()) <= (
            item_rect.width() - 16
        )
    assert page.timeline.width() >= 180
    assert not page.comparison_table.verticalScrollBar().isVisible()
    comparison_viewport_height = page.comparison_table.viewport().height()
    for row in range(page.comparison_table.rowCount()):
        row_rect = page.comparison_table.visualItemRect(page.comparison_table.item(row, 0))
        assert row_rect.bottom() < comparison_viewport_height


def test_page_rejects_partial_timeline_instead_of_filling_missing_hours(qtbot) -> None:
    page = LiveScenarioPage(load_map_asset())
    qtbot.addWidget(page)

    with pytest.raises(ValueError, match="exactly one frame"):
        page.set_frames([_frame(hour) for hour in range(23)])

    assert not page.timeline.isEnabled()

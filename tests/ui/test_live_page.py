from __future__ import annotations

import pytest
from PySide6.QtCore import QPoint

from gridguard.domain import ServiceTier
from gridguard.ui.live_models import LiveScenarioFrame, LocalityHourDetail
from gridguard.ui.live_page import LiveScenarioPage
from gridguard.ui.map_asset import EXPECTED_LOCALITY_IDS, load_map_asset


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
    assert page.map_view.locality_items["khartoum"].status_text == "◆ CRITICAL ONLY"
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


def test_locality_detail_rows_fit_at_presentation_height(qtbot) -> None:
    page = LiveScenarioPage(load_map_asset())
    qtbot.addWidget(page)
    page.resize(1338, 610)
    page.show()
    page.set_frames([_frame(hour) for hour in range(24)])
    page.timeline.setValue(18)
    qtbot.wait(50)

    for value in page.sector_service_labels.values():
        bottom = value.mapTo(page.locality_card, QPoint(0, value.height())).y()
        assert value.isVisibleTo(page.locality_card)
        assert bottom <= page.locality_card.height()


def test_page_rejects_partial_timeline_instead_of_filling_missing_hours(qtbot) -> None:
    page = LiveScenarioPage(load_map_asset())
    qtbot.addWidget(page)

    with pytest.raises(ValueError, match="exactly one frame"):
        page.set_frames([_frame(hour) for hour in range(23)])

    assert not page.timeline.isEnabled()

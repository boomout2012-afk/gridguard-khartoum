from __future__ import annotations

from itertools import combinations

import pytest
from PySide6.QtCore import QRectF, Qt
from PySide6.QtWidgets import QLabel

from gridguard.domain import HORIZON_HOURS, SECTORS, ServiceTier
from gridguard.ui.map_asset import load_map_asset
from gridguard.ui.map_view import LocalityMapView
from gridguard.ui.overview_page import OverviewPage
from gridguard.ui.presentation_model import load_presentation_snapshot
from gridguard.ui.theme import apply_theme


def test_selected_coral_is_independent_of_the_underlying_service_tier(qtbot) -> None:
    view = LocalityMapView(load_map_asset())
    qtbot.addWidget(view)
    east_nile = view.locality_items["east_nile"]
    assert east_nile.service_tier is None

    view.set_service_tiers({"east_nile": ServiceTier.CRITICAL_ONLY})
    view.select_locality("east_nile")
    assert east_nile.brush().color().name().upper() == "#E47A5E"
    assert east_nile.service_tier is ServiceTier.CRITICAL_ONLY
    assert "Critical Only" in east_nile.toolTip()
    assert "Selected locality" in east_nile.toolTip()

    view.set_service_tiers({"east_nile": ServiceTier.FULL_SERVICE})
    assert east_nile.brush().color().name().upper() == "#E47A5E"
    view.select_locality("bahri")
    assert east_nile.brush().color().name().upper() == "#5C9D8B"
    assert east_nile.brush().style() == Qt.BrushStyle.SolidPattern


@pytest.mark.parametrize("size", [(900, 420), (696, 285)])
def test_map_labels_are_unboxed_legible_and_separated(qtbot, qapp, size) -> None:
    apply_theme(qapp)
    view = LocalityMapView(load_map_asset())
    qtbot.addWidget(view)
    view.resize(*size)
    view.show()
    view.set_service_tiers({key: ServiceTier.CRITICAL_ONLY for key in view.locality_items})
    view.reset_view()
    qtbot.wait(20)

    rects = {}
    for locality_id, item in view.locality_items.items():
        assert item._name.font().pixelSize() >= 14
        assert not item._status.isVisible()
        assert item._label_backing.brush().style() == Qt.BrushStyle.NoBrush
        rects[locality_id] = item._name.deviceTransform(view.viewportTransform()).mapRect(
            item._name.boundingRect()
        )
        assert QRectF(view.viewport().rect()).contains(rects[locality_id]), locality_id
    for (first, first_rect), (second, second_rect) in combinations(rects.items(), 2):
        assert not first_rect.intersects(second_rect), (first, second)


@pytest.mark.parametrize("size, title_size", [((1376, 776), 100), ((1302, 644), 88)])
def test_overview_matches_approved_scope_without_prepopulated_results(
    qtbot, qapp, size, title_size
) -> None:
    apply_theme(qapp)
    page = OverviewPage(load_presentation_snapshot(), load_map_asset())
    qtbot.addWidget(page)
    page.resize(*size)
    page.show()
    qtbot.wait(20)

    assert page.title_label.font().pixelSize() == title_size
    assert page.title_label.font().bold()
    assert page.explore_button.text() == "Explore scenario"
    assert page.results_button.text() == "Compare results"
    assert page.explore_button.height() >= 44
    assert page.results_button.height() >= 44
    assert {key: label.text() for key, label in page.scope_fact_labels.items()} == {
        "Localities": "7",
        "Hours": str(HORIZON_HOURS),
        "Service sectors": str(len(SECTORS)),
    }
    assert not [label for label in page.findChildren(QLabel) if label.property("metric")]
    assert "Synthetic" in page.scope_note.text()
    assert "Decision support only" in page.scope_note.text()
    assert "OCHA COD-AB 2020" in page.source_attribution.text()
    assert "Natural Earth 5.0" in page.source_attribution.text()
    assert page.map_view.selected_locality_id == "east_nile"
    assert all(item.service_tier is None for item in page.map_view.locality_items.values())
    assert page.selected_locality_label.text() == "East Nile"
    page.map_view.select_locality("bahri")
    assert page.selected_locality_label.text() == "Bahri"
    for label in (page.title_label, *page.scope_fact_labels.values()):
        assert label.height() >= label.sizeHint().height()
        assert label.width() >= label.sizeHint().width()
    assert page.map_view.width() >= 650
    assert page.minimumSizeHint().height() <= size[1]

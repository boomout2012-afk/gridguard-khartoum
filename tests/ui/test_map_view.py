from __future__ import annotations

from itertools import combinations

from PySide6.QtCore import QPoint, QPointF, QRectF, Qt
from PySide6.QtGui import QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QApplication, QGraphicsItem, QGraphicsPathItem

from gridguard.domain import ServiceTier
from gridguard.ui.map_asset import load_map_asset
from gridguard.ui.map_view import LocalityMapView, SchematicMapView
from gridguard.ui.theme import PALETTE, STATUS_FILL_COLORS, apply_theme


def test_map_exposes_all_localities_and_accessible_description(qtbot) -> None:
    asset = load_map_asset()
    view = LocalityMapView(asset)
    qtbot.addWidget(view)

    assert tuple(view.locality_items) == asset.keyboard_order
    assert view.accessibleName() == "Interactive Khartoum locality map"
    assert "administrative" in view.accessibleDescription().casefold()
    assert "synthetic" in view.accessibleDescription().casefold()
    assert "schematic" not in view.accessibleDescription().casefold()
    assert SchematicMapView is LocalityMapView


def test_map_renders_complete_administrative_geometry_and_all_river_lines(qtbot) -> None:
    asset = load_map_asset()
    view = LocalityMapView(asset)
    qtbot.addWidget(view)

    assert all(isinstance(item, QGraphicsPathItem) for item in view.locality_items.values())
    for locality in asset.localities:
        rendered = view.locality_items[locality.locality_id]
        source_vertex_count = sum(
            len(polygon.outer) + sum(len(hole) for hole in polygon.holes)
            for polygon in locality.geometry.polygons
        )
        assert rendered.path().fillRule() == Qt.FillRule.OddEvenFill
        assert rendered.path().elementCount() >= source_vertex_count
        assert (
            rendered._label_group.flags()
            & QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations
        )
        assert rendered._label_group.parentItem() is None
        assert rendered._label_group.zValue() > max(
            river_item.zValue() for river_item in view.river_items
        )

    expected_river_lines = sum(len(river.lines) for river in asset.river_context.paths)
    assert len(view.river_items) == expected_river_lines
    assert all(item.pen().isCosmetic() for item in view.river_items)


def test_keyboard_navigation_selects_directional_neighbor(qtbot) -> None:
    view = LocalityMapView(load_map_asset())
    qtbot.addWidget(view)
    view.show()
    view.select_locality("khartoum")
    view.setFocus()

    with qtbot.waitSignal(view.locality_selected, timeout=1000) as emitted:
        qtbot.keyClick(view, Qt.Key.Key_Right)

    assert emitted.args == ["east_nile"]
    assert view.selected_locality_id == "east_nile"


def test_pointer_selection_emits_locality_identifier(qtbot) -> None:
    asset = load_map_asset()
    view = LocalityMapView(asset)
    qtbot.addWidget(view)
    view.resize(700, 500)
    view.show()
    view.reset_view()
    item = view.locality_items["omdurman"]
    anchor = asset.locality("omdurman").label_anchor
    position = view.mapFromScene(QPointF(*anchor))

    with qtbot.waitSignal(view.locality_selected, timeout=1000) as emitted:
        qtbot.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=position)

    assert emitted.args == ["omdurman"]
    assert "Omdurman" in item.toolTip()


def test_service_fills_keep_text_status_independent_of_coral_selection(qtbot) -> None:
    view = LocalityMapView(load_map_asset())
    qtbot.addWidget(view)

    view.set_service_tiers(
        {
            "khartoum": ServiceTier.OFF,
            "omdurman": ServiceTier.CRITICAL_ONLY,
            "bahri": ServiceTier.FULL_SERVICE,
        }
    )

    view.select_locality("east_nile")
    assert "arrow keys" in view.accessibleDescription()
    assert view.focusPolicy() == Qt.FocusPolicy.StrongFocus
    for locality_id, tier in (
        ("khartoum", ServiceTier.OFF),
        ("omdurman", ServiceTier.CRITICAL_ONLY),
        ("bahri", ServiceTier.FULL_SERVICE),
    ):
        item = view.locality_items[locality_id]
        assert item.service_tier is tier
        assert item.status_text == tier.label
        assert item.locality.name in item.toolTip()
        assert f"Service tier: {tier.label}" in item.toolTip()
        assert "Selected locality" not in item.toolTip()
        assert item.brush().style() == Qt.BrushStyle.SolidPattern
        assert item.brush().color() == STATUS_FILL_COLORS[tier.value]
        assert item._name.brush().color().name().upper() == PALETTE.ink

    view.select_locality("omdurman")
    selected = view.locality_items["omdurman"]
    assert selected.brush().color().name().upper() == PALETTE.selection
    assert selected.service_tier is ServiceTier.CRITICAL_ONLY
    assert selected.status_text == "Critical Only"
    assert "Service tier: Critical Only" in selected.toolTip()
    assert "Selected locality; coral denotes selection, not a service tier" in selected.toolTip()

    view.select_locality("bahri")
    assert selected.brush().color() == STATUS_FILL_COLORS[ServiceTier.CRITICAL_ONLY.value]
    assert "Selected locality" not in selected.toolTip()


def test_status_callouts_do_not_overlap_at_live_presentation_size(qtbot, qapp) -> None:
    apply_theme(qapp)
    view = LocalityMapView(load_map_asset())
    qtbot.addWidget(view)
    view.resize(696, 285)
    view.show()
    view.set_service_tiers(
        {locality_id: ServiceTier.CRITICAL_ONLY for locality_id in view.locality_items}
    )
    view.reset_view()
    qtbot.wait(20)

    callout_rects = {
        locality_id: item._label_backing.deviceTransform(view.viewportTransform()).mapRect(
            item._label_backing.rect()
        )
        for locality_id, item in view.locality_items.items()
    }
    viewport_rect = QRectF(view.viewport().rect())
    for locality_id, callout_rect in callout_rects.items():
        assert viewport_rect.contains(callout_rect), locality_id
    for (first_id, first_rect), (second_id, second_rect) in combinations(callout_rects.items(), 2):
        assert not first_rect.intersects(second_rect), (first_id, second_id)


def test_reset_view_restores_initial_zoom(qtbot) -> None:
    view = LocalityMapView(load_map_asset())
    qtbot.addWidget(view)
    view.resize(700, 500)
    view.show()
    view.reset_view()
    initial = view.transform().m11()

    view.zoom_in()
    assert view.transform().m11() > initial

    view.reset_view()
    assert view.transform().m11() == initial


def test_wheel_zoom_and_drag_pan_change_the_view(qtbot) -> None:
    view = LocalityMapView(load_map_asset())
    qtbot.addWidget(view)
    view.resize(700, 500)
    view.show()
    view.reset_view()
    initial_scale = view.transform().m11()

    wheel = QWheelEvent(
        QPointF(350, 250),
        QPointF(view.mapToGlobal(QPoint(350, 250))),
        QPoint(),
        QPoint(0, 120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.ScrollUpdate,
        False,
    )
    QApplication.sendEvent(view.viewport(), wheel)

    assert view.transform().m11() > initial_scale

    view.zoom_in()
    view.zoom_in()
    before = (view.horizontalScrollBar().value(), view.verticalScrollBar().value())
    start = QPointF(20, 20)
    finish = QPointF(180, 150)
    for event in (
        QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            start,
            QPointF(view.viewport().mapToGlobal(start.toPoint())),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        ),
        QMouseEvent(
            QMouseEvent.Type.MouseMove,
            finish,
            QPointF(view.viewport().mapToGlobal(finish.toPoint())),
            Qt.MouseButton.NoButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        ),
        QMouseEvent(
            QMouseEvent.Type.MouseButtonRelease,
            finish,
            QPointF(view.viewport().mapToGlobal(finish.toPoint())),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        ),
    ):
        QApplication.sendEvent(view.viewport(), event)
    after = (view.horizontalScrollBar().value(), view.verticalScrollBar().value())

    assert after != before

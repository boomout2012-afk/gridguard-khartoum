from __future__ import annotations

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QApplication, QGraphicsPathItem

from gridguard.domain import ServiceTier
from gridguard.ui.map_asset import load_map_asset
from gridguard.ui.map_view import LocalityMapView, SchematicMapView


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


def test_statuses_have_symbol_and_text_redundancy(qtbot) -> None:
    view = LocalityMapView(load_map_asset())
    qtbot.addWidget(view)

    view.set_service_tiers(
        {
            "khartoum": ServiceTier.OFF,
            "omdurman": ServiceTier.CRITICAL_ONLY,
            "bahri": ServiceTier.FULL_SERVICE,
        }
    )

    assert view.locality_items["khartoum"].status_text == "× OFF"
    assert view.locality_items["omdurman"].status_text == "◆ CRITICAL ONLY"
    assert view.locality_items["bahri"].status_text == "● FULL"


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

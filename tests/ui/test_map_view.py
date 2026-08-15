from __future__ import annotations

from PySide6.QtCore import Qt

from gridguard.domain import ServiceTier
from gridguard.ui.map_asset import load_map_asset
from gridguard.ui.map_view import SchematicMapView


def test_map_exposes_all_localities_and_accessible_description(qtbot) -> None:
    view = SchematicMapView(load_map_asset())
    qtbot.addWidget(view)

    assert tuple(view.locality_items) == load_map_asset().keyboard_order
    assert view.accessibleName() == "Interactive Khartoum locality schematic"
    assert "schematic" in view.accessibleDescription().casefold()


def test_keyboard_navigation_selects_directional_neighbor(qtbot) -> None:
    view = SchematicMapView(load_map_asset())
    qtbot.addWidget(view)
    view.show()
    view.select_locality("khartoum")
    view.setFocus()

    with qtbot.waitSignal(view.locality_selected, timeout=1000) as emitted:
        qtbot.keyClick(view, Qt.Key.Key_Right)

    assert emitted.args == ["east_nile"]
    assert view.selected_locality_id == "east_nile"


def test_pointer_selection_emits_locality_identifier(qtbot) -> None:
    view = SchematicMapView(load_map_asset())
    qtbot.addWidget(view)
    view.resize(700, 500)
    view.show()
    view.reset_view()
    item = view.locality_items["omdurman"]
    position = view.mapFromScene(item.boundingRect().center())

    with qtbot.waitSignal(view.locality_selected, timeout=1000) as emitted:
        qtbot.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=position)

    assert emitted.args == ["omdurman"]
    assert "Omdurman" in item.toolTip()


def test_statuses_have_symbol_and_text_redundancy(qtbot) -> None:
    view = SchematicMapView(load_map_asset())
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
    view = SchematicMapView(load_map_asset())
    qtbot.addWidget(view)
    view.resize(700, 500)
    view.show()
    view.reset_view()
    initial = view.transform().m11()

    view.zoom_in()
    assert view.transform().m11() > initial

    view.reset_view()
    assert view.transform().m11() == initial

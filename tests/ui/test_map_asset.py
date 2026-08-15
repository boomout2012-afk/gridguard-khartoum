from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest

from gridguard.ui import (
    DEFAULT_MAP_ASSET_PATH,
    MapAssetValidationError,
    load_map_asset,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_LOCALITIES = {
    "khartoum": "Khartoum",
    "omdurman": "Omdurman",
    "bahri": "Bahri",
    "east_nile": "East Nile",
    "jebel_awlia": "Jebel Awlia",
    "karari": "Karari",
    "ombada": "Ombada",
}


@pytest.fixture
def valid_payload() -> dict[str, object]:
    return json.loads(DEFAULT_MAP_ASSET_PATH.read_text(encoding="utf-8"))


def _load_payload(tmp_path: Path, payload: object) -> None:
    path = tmp_path / "map.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    load_map_asset(path)


def _localities(payload: dict[str, object]) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], payload["localities"])


def _river_paths(payload: dict[str, object]) -> list[dict[str, Any]]:
    river_context = cast(dict[str, Any], payload["river_context"])
    return cast(list[dict[str, Any]], river_context["paths"])


def test_default_asset_is_cwd_independent_and_contains_exact_localities(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    asset = load_map_asset()

    assert DEFAULT_MAP_ASSET_PATH.is_file()
    assert "gridguard/assets/map/khartoum-schematic.json" in str(DEFAULT_MAP_ASSET_PATH).replace(
        "\\", "/"
    )
    assert not (REPOSITORY_ROOT / "assets" / "map" / "khartoum-schematic.json").exists()
    assert asset.schema_version == 1
    assert asset.kind == "schematic"
    assert {item.locality_id: item.name for item in asset.localities} == EXPECTED_LOCALITIES
    assert asset.keyboard_order == tuple(EXPECTED_LOCALITIES)
    assert "not a geographic" in asset.disclaimer.casefold()
    assert "not an electrical-network topology" in asset.disclaimer.casefold()
    assert asset.river_context.paths


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 2, "schema_version"),
        ("kind", "geographic", "kind"),
    ],
)
def test_loader_rejects_unknown_schema_or_kind(
    tmp_path: Path,
    valid_payload: dict[str, object],
    field: str,
    value: object,
    message: str,
) -> None:
    valid_payload[field] = value

    with pytest.raises(MapAssetValidationError, match=message):
        _load_payload(tmp_path, valid_payload)


def test_loader_rejects_missing_and_unknown_fields(
    tmp_path: Path,
    valid_payload: dict[str, object],
) -> None:
    del valid_payload["title"]
    valid_payload["projection"] = "EPSG:4326"

    with pytest.raises(MapAssetValidationError, match="field"):
        _load_payload(tmp_path, valid_payload)


def test_loader_requires_explicit_non_geographic_and_non_topology_disclaimer(
    tmp_path: Path,
    valid_payload: dict[str, object],
) -> None:
    valid_payload["disclaimer"] = "A schematic display."

    with pytest.raises(MapAssetValidationError, match="disclaimer"):
        _load_payload(tmp_path, valid_payload)


def test_loader_rejects_unknown_nested_fields(
    tmp_path: Path,
    valid_payload: dict[str, object],
) -> None:
    locality = valid_payload["localities"][0]  # type: ignore[index]
    locality["opacity"] = 0.8  # type: ignore[index]

    with pytest.raises(MapAssetValidationError, match="unknown field"):
        _load_payload(tmp_path, valid_payload)


def test_importing_map_asset_does_not_import_qt() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import gridguard.ui.map_asset; "
                "assert not any(name == 'PySide6' or name.startswith('PySide6.') "
                "for name in sys.modules)"
            ),
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: cast(dict[str, Any], payload["scene_rect"]).__setitem__("x", float("nan")),
        lambda payload: _localities(payload)[0]["polygon"][0].__setitem__(0, float("inf")),
        lambda payload: cast(dict[str, Any], payload["scene_rect"]).__setitem__("x", 10**400),
    ],
)
def test_loader_rejects_nonfinite_coordinates(
    tmp_path: Path,
    valid_payload: dict[str, object],
    mutate: Any,
) -> None:
    mutate(valid_payload)

    with pytest.raises(MapAssetValidationError, match="non-finite|finite"):
        _load_payload(tmp_path, valid_payload)


@pytest.mark.parametrize(
    "polygon",
    [
        [[10, 10], [20, 20]],
        [[10, 10], [20, 20], [10, 10]],
    ],
)
def test_loader_rejects_polygon_with_fewer_than_three_distinct_points(
    tmp_path: Path,
    valid_payload: dict[str, object],
    polygon: list[list[int]],
) -> None:
    _localities(valid_payload)[0]["polygon"] = polygon

    with pytest.raises(MapAssetValidationError, match="three distinct|at least 3"):
        _load_payload(tmp_path, valid_payload)


@pytest.mark.parametrize(
    ("polygon", "message"),
    [
        ([[100, 100], [200, 200], [300, 300]], "area"),
        (
            [[100, 100], [400, 400], [100, 400], [350, 100]],
            "self-intersect|simple",
        ),
        (
            [[100, 100], [300, 100], [300, 300], [100, 300], [300, 100]],
            "duplicate|distinct",
        ),
    ],
)
def test_loader_rejects_unusable_polygon_geometry(
    tmp_path: Path,
    valid_payload: dict[str, object],
    polygon: list[list[int]],
    message: str,
) -> None:
    _localities(valid_payload)[0]["polygon"] = polygon

    with pytest.raises(MapAssetValidationError, match=message):
        _load_payload(tmp_path, valid_payload)


def test_loader_rejects_in_scene_label_anchor_outside_its_polygon(
    tmp_path: Path,
    valid_payload: dict[str, object],
) -> None:
    _localities(valid_payload)[0]["label_anchor"] = [50, 50]

    with pytest.raises(MapAssetValidationError, match="inside.*polygon"):
        _load_payload(tmp_path, valid_payload)


@pytest.mark.parametrize("geometry", ["polygon", "label_anchor", "river"])
def test_loader_rejects_points_outside_scene(
    tmp_path: Path,
    valid_payload: dict[str, object],
    geometry: str,
) -> None:
    if geometry == "polygon":
        _localities(valid_payload)[0]["polygon"][0] = [-1, 100]
    elif geometry == "label_anchor":
        _localities(valid_payload)[0]["label_anchor"] = [1001, 100]
    else:
        _river_paths(valid_payload)[0]["points"][0] = [100, 721]

    with pytest.raises(MapAssetValidationError, match="scene_rect"):
        _load_payload(tmp_path, valid_payload)


@pytest.mark.parametrize("destination", ["unknown_locality", "khartoum"])
def test_loader_rejects_bad_or_self_neighbor(
    tmp_path: Path,
    valid_payload: dict[str, object],
    destination: str,
) -> None:
    _localities(valid_payload)[0]["neighbors"]["right"] = destination

    with pytest.raises(MapAssetValidationError, match="neighbor"):
        _load_payload(tmp_path, valid_payload)


def test_loader_rejects_disconnected_keyboard_navigation(
    tmp_path: Path,
    valid_payload: dict[str, object],
) -> None:
    for locality in _localities(valid_payload):
        neighbors = locality["neighbors"]
        for direction, destination in neighbors.items():
            if destination == "ombada" or locality["id"] == "ombada":
                neighbors[direction] = None

    with pytest.raises(MapAssetValidationError, match="connected"):
        _load_payload(tmp_path, valid_payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate_id", "duplicate locality IDs"),
        ("duplicate_name", "duplicate locality names"),
        ("missing_locality", "seven approved"),
        ("extra_locality", "duplicate locality IDs"),
        ("duplicate_order", "duplicate locality IDs"),
        ("missing_order", "every approved"),
        ("extra_order", "every approved"),
    ],
)
def test_loader_rejects_missing_extra_or_duplicate_locality_identity_and_order(
    tmp_path: Path,
    valid_payload: dict[str, object],
    mutation: str,
    message: str,
) -> None:
    localities = _localities(valid_payload)
    order = cast(list[str], valid_payload["keyboard_order"])
    if mutation == "duplicate_id":
        localities[-1]["id"] = localities[0]["id"]
    elif mutation == "duplicate_name":
        localities[-1]["name"] = localities[0]["name"]
    elif mutation == "missing_locality":
        localities.pop()
    elif mutation == "extra_locality":
        localities.append(dict(localities[0]))
    elif mutation == "duplicate_order":
        order[-1] = order[0]
    elif mutation == "missing_order":
        order.pop()
    else:
        order.append("unknown_locality")

    with pytest.raises(MapAssetValidationError, match=message):
        _load_payload(tmp_path, valid_payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("map_id", "https://example.invalid/khartoum.json"),
        ("map_id", "ssh://host.invalid/khartoum.json"),
        ("map_id", "s3://bucket/khartoum.json"),
        ("map_id", "example.invalid/map.json"),
        ("map_id", "external/map.json"),
        ("title", "../external/map.json"),
        ("title", "C:\\external\\map.json"),
        ("title", "file:///external/map.json"),
    ],
)
def test_loader_rejects_urls_and_external_references(
    tmp_path: Path,
    valid_payload: dict[str, object],
    field: str,
    value: str,
) -> None:
    valid_payload[field] = value

    with pytest.raises(MapAssetValidationError, match="external reference"):
        _load_payload(tmp_path, valid_payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("title", "Controller summary"),
        ("title", "Scientific metric display"),
        ("map_id", "ppo"),
        ("title", "Fixed Rotation"),
        ("title", "Full Service"),
        ("title", "Critical Only"),
        ("title", "Off"),
        (
            "disclaimer",
            "Schematic only; not a geographic boundary map and not an "
            "electrical-network topology. Demand results are excluded.",
        ),
    ],
)
def test_loader_rejects_forbidden_vocabulary_in_string_values(
    tmp_path: Path,
    valid_payload: dict[str, object],
    field: str,
    value: str,
) -> None:
    valid_payload[field] = value

    with pytest.raises(MapAssetValidationError, match="display-only"):
        _load_payload(tmp_path, valid_payload)


@pytest.mark.parametrize(
    "forbidden_field",
    ["controller", "results", "demand_mw", "service_tier", "metrics"],
)
def test_loader_rejects_simulation_and_scientific_data(
    tmp_path: Path,
    valid_payload: dict[str, object],
    forbidden_field: str,
) -> None:
    _localities(valid_payload)[0][forbidden_field] = "PPO"

    with pytest.raises(MapAssetValidationError, match="display-only"):
        _load_payload(tmp_path, valid_payload)


def test_loader_rejects_duplicate_json_object_fields(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version": 1, "schema_version": 1}', encoding="utf-8")

    with pytest.raises(MapAssetValidationError, match="duplicate field"):
        load_map_asset(path)

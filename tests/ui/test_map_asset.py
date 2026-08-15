from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest

from gridguard.ui import MapAssetValidationError, load_map_asset
from gridguard.ui.map_asset import (
    DEFAULT_MAP_ASSET_PATH,
    DEFAULT_MAP_PROVENANCE_PATH,
    DEFAULT_MAP_PROVENANCE_SHA256,
    DEFAULT_RIVER_ASSET_PATH,
)
from tools.build_map_asset import (
    RIVER_SELECTORS,
    BuildError,
    _clip_line_to_mask,
    _river_component,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_LOCALITIES = {
    "khartoum": ("Khartoum", "Khartoum", "SD01007"),
    "omdurman": ("Omdurman", "Um Durman", "SD01006"),
    "bahri": ("Bahri", "Bahri", "SD01003"),
    "east_nile": ("East Nile", "Sharg An Neel", "SD01004"),
    "jebel_awlia": ("Jebel Awlia", "Jebel Awlia", "SD01001"),
    "karari": ("Karari", "Karrari", "SD01005"),
    "ombada": ("Ombada", "Um Bada", "SD01002"),
}
EXPECTED_BOUNDARY_SHA256 = "c13cf1e8515730267d41bd560a1f1a0489552ab2900abb070a9aececacca1bb9"
EXPECTED_RIVER_SHA256 = "bb854a900ecbd3b408df46d5e16e3e0f974ba55993f9d8b5c26e855273c0905a"


@pytest.fixture
def valid_bundle() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    return (
        json.loads(DEFAULT_MAP_ASSET_PATH.read_text(encoding="utf-8")),
        json.loads(DEFAULT_RIVER_ASSET_PATH.read_text(encoding="utf-8")),
        json.loads(DEFAULT_MAP_PROVENANCE_PATH.read_text(encoding="utf-8")),
    )


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode()


def _write_bundle(
    tmp_path: Path,
    localities: object,
    rivers: object,
    provenance: dict[str, Any],
    *,
    refresh_hashes: bool = True,
) -> tuple[Path, Path, Path]:
    locality_path = tmp_path / "khartoum-localities.geojson"
    river_path = tmp_path / "khartoum-nile.geojson"
    provenance_path = tmp_path / "map-provenance.json"
    locality_bytes = _json_bytes(localities)
    river_bytes = _json_bytes(rivers)
    locality_path.write_bytes(locality_bytes)
    river_path.write_bytes(river_bytes)
    if refresh_hashes:
        derived = cast(dict[str, dict[str, object]], provenance["derived_assets"])
        derived[locality_path.name] = {
            "sha256": hashlib.sha256(locality_bytes).hexdigest(),
            "bytes": len(locality_bytes),
        }
        derived[river_path.name] = {
            "sha256": hashlib.sha256(river_bytes).hexdigest(),
            "bytes": len(river_bytes),
        }
    provenance_path.write_bytes(_json_bytes(provenance))
    return locality_path, river_path, provenance_path


def _load_bundle(paths: tuple[Path, Path, Path]) -> object:
    locality_path, river_path, provenance_path = paths
    return load_map_asset(
        locality_path,
        river_path=river_path,
        provenance_path=provenance_path,
    )


def _features(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], payload["features"])


def _square_mask(*, with_hole: bool = False) -> tuple[object, ...]:
    outer = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 0.0))
    holes = (((4.0, 4.0), (6.0, 4.0), (6.0, 6.0), (4.0, 6.0), (4.0, 4.0)),) if with_hole else ()
    return (((outer, *holes),),)


def test_default_geographic_asset_is_cwd_independent_and_pinned(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    asset = load_map_asset()

    assert DEFAULT_MAP_ASSET_PATH.name == "khartoum-localities.geojson"
    assert DEFAULT_RIVER_ASSET_PATH.name == "khartoum-nile.geojson"
    assert DEFAULT_MAP_PROVENANCE_PATH.name == "map-provenance.json"
    assert DEFAULT_MAP_ASSET_PATH.is_file()
    assert DEFAULT_RIVER_ASSET_PATH.is_file()
    assert DEFAULT_MAP_PROVENANCE_PATH.is_file()
    assert hashlib.sha256(DEFAULT_MAP_PROVENANCE_PATH.read_bytes()).hexdigest() == (
        DEFAULT_MAP_PROVENANCE_SHA256
    )
    assert asset.schema_version == 2
    assert asset.map_id == "khartoum-locality-boundaries-2020"
    assert asset.kind == "administrative_boundaries"
    assert asset.projection.crs == "EPSG:3857"
    assert asset.scene_rect.width == 1000
    assert asset.scene_rect.height == 720
    assert asset.projection.padding == 36
    assert asset.keyboard_order == tuple(EXPECTED_LOCALITIES)
    assert "synthetic" in asset.disclaimer.casefold()
    assert "not an electrical-network topology" in asset.disclaimer.casefold()
    assert tuple(path.river_id for path in asset.river_context.paths) == (
        "white_nile",
        "blue_nile",
        "nile",
    )


def test_default_asset_retains_exact_cod_ab_identity_and_attribution() -> None:
    asset = load_map_asset()

    assert asset.source.provider == (
        "United Nations Office for the Coordination of Humanitarian Affairs"
    )
    assert asset.source.dataset == "Sudan administrative boundaries (COD-AB)"
    assert asset.source.boundary_year == 2020
    assert asset.source.valid_from == "2020-08-31"
    assert asset.source.source_revision == "v03"
    assert asset.source.source_sha256 == EXPECTED_BOUNDARY_SHA256
    assert asset.source.license == "CC BY 3.0 IGO"
    assert {
        locality.locality_id: (
            locality.name,
            locality.source_name,
            locality.adm2_pcode,
        )
        for locality in asset.localities
    } == EXPECTED_LOCALITIES
    assert all(locality.source_name_ar for locality in asset.localities)


def test_runtime_exposes_projected_multi_ring_geometry_not_raw_longitude_latitude() -> None:
    asset = load_map_asset()

    assert all(locality.geometry.polygons for locality in asset.localities)
    for locality in asset.localities:
        assert locality.polygon == locality.geometry.polygons[0].outer
        assert locality.polygon[0] != locality.polygon[-1]
        for polygon in locality.geometry.polygons:
            for x, y in (*polygon.outer, *(point for hole in polygon.holes for point in hole)):
                assert asset.scene_rect.x <= x <= asset.scene_rect.right
                assert asset.scene_rect.y <= y <= asset.scene_rect.bottom
                assert not (29 <= x <= 36 and 10 <= y <= 23)
        anchor_x, anchor_y = locality.label_anchor
        assert asset.scene_rect.x <= anchor_x <= asset.scene_rect.right
        assert asset.scene_rect.y <= anchor_y <= asset.scene_rect.bottom


@pytest.mark.parametrize("geometry_type", ["Polygon", "MultiPolygon"])
def test_loader_supports_polygon_multipolygon_and_holes(
    tmp_path: Path,
    valid_bundle: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
    geometry_type: str,
) -> None:
    localities, rivers, provenance = copy.deepcopy(valid_bundle)
    feature = _features(localities)[0]
    anchor_x, anchor_y = cast(list[float], feature["properties"]["label_anchor_wgs84"])
    outer = [
        [anchor_x - 0.2, anchor_y - 0.2],
        [anchor_x + 0.2, anchor_y - 0.2],
        [anchor_x + 0.2, anchor_y + 0.2],
        [anchor_x - 0.2, anchor_y + 0.2],
        [anchor_x - 0.2, anchor_y - 0.2],
    ]
    hole = [
        [anchor_x + 0.08, anchor_y + 0.08],
        [anchor_x + 0.14, anchor_y + 0.08],
        [anchor_x + 0.14, anchor_y + 0.14],
        [anchor_x + 0.08, anchor_y + 0.14],
        [anchor_x + 0.08, anchor_y + 0.08],
    ]
    coordinates: object = [outer, hole]
    if geometry_type == "MultiPolygon":
        second_outer = [
            [anchor_x + 0.3, anchor_y + 0.3],
            [anchor_x + 0.31, anchor_y + 0.3],
            [anchor_x + 0.31, anchor_y + 0.31],
            [anchor_x + 0.3, anchor_y + 0.31],
            [anchor_x + 0.3, anchor_y + 0.3],
        ]
        coordinates = [[outer, hole], [second_outer]]
    feature["geometry"] = {"type": geometry_type, "coordinates": coordinates}

    asset = _load_bundle(_write_bundle(tmp_path, localities, rivers, provenance))
    locality = asset.localities[0]

    assert len(locality.geometry.polygons) == (1 if geometry_type == "Polygon" else 2)
    assert len(locality.geometry.polygons[0].holes) == 1


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.__setitem__("schema_version", 3), "schema_version"),
        (lambda payload: payload.__setitem__("type", "GeometryCollection"), "FeatureCollection"),
        (lambda payload: payload.__setitem__("kind", "schematic"), "kind"),
        (lambda payload: payload.__setitem__("unexpected", True), "unknown field"),
    ],
)
def test_loader_rejects_unknown_or_open_locality_collection_schema(
    tmp_path: Path,
    valid_bundle: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
    mutation: Any,
    message: str,
) -> None:
    localities, rivers, provenance = copy.deepcopy(valid_bundle)
    mutation(localities)

    with pytest.raises(MapAssetValidationError, match=message):
        _load_bundle(_write_bundle(tmp_path, localities, rivers, provenance))


def test_loader_rejects_unknown_feature_properties(
    tmp_path: Path,
    valid_bundle: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
) -> None:
    localities, rivers, provenance = copy.deepcopy(valid_bundle)
    _features(localities)[0]["properties"]["demand_mw"] = 42

    with pytest.raises(MapAssetValidationError, match="unknown field"):
        _load_bundle(_write_bundle(tmp_path, localities, rivers, provenance))


@pytest.mark.parametrize(
    ("geometry", "message"),
    [
        ({"type": "Point", "coordinates": [32.5, 15.6]}, "Polygon|MultiPolygon"),
        (
            {
                "type": "Polygon",
                "coordinates": [[[32.4, 15.5], [32.6, 15.5], [32.6, 15.7], [32.4, 15.7]]],
            },
            "closed",
        ),
        (
            {
                "type": "Polygon",
                "coordinates": [[[32.4, 15.5], [math.inf, 15.5], [32.5, 15.7], [32.4, 15.5]]],
            },
            "finite|non-finite",
        ),
    ],
)
def test_loader_rejects_invalid_geographic_geometry(
    tmp_path: Path,
    valid_bundle: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
    geometry: dict[str, object],
    message: str,
) -> None:
    localities, rivers, provenance = copy.deepcopy(valid_bundle)
    _features(localities)[0]["geometry"] = geometry

    with pytest.raises(MapAssetValidationError, match=message):
        _load_bundle(_write_bundle(tmp_path, localities, rivers, provenance))


def test_loader_rejects_self_intersecting_boundary_ring(
    tmp_path: Path,
    valid_bundle: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
) -> None:
    localities, rivers, provenance = copy.deepcopy(valid_bundle)
    anchor_x, anchor_y = _features(localities)[0]["properties"]["label_anchor_wgs84"]
    _features(localities)[0]["geometry"] = {
        "type": "Polygon",
        "coordinates": [
            [
                [anchor_x - 0.2, anchor_y - 0.2],
                [anchor_x + 0.2, anchor_y + 0.2],
                [anchor_x - 0.2, anchor_y + 0.2],
                [anchor_x + 0.2, anchor_y - 0.2],
                [anchor_x - 0.2, anchor_y - 0.2],
            ]
        ],
    }

    with pytest.raises(MapAssetValidationError, match="simple|self-intersect"):
        _load_bundle(_write_bundle(tmp_path, localities, rivers, provenance))


def test_loader_rejects_hole_that_crosses_its_outer_ring(
    tmp_path: Path,
    valid_bundle: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
) -> None:
    localities, rivers, provenance = copy.deepcopy(valid_bundle)
    anchor_x, anchor_y = _features(localities)[0]["properties"]["label_anchor_wgs84"]
    _features(localities)[0]["geometry"] = {
        "type": "Polygon",
        "coordinates": [
            [
                [anchor_x - 0.2, anchor_y - 0.2],
                [anchor_x + 0.2, anchor_y - 0.2],
                [anchor_x + 0.2, anchor_y + 0.2],
                [anchor_x - 0.2, anchor_y + 0.2],
                [anchor_x - 0.2, anchor_y - 0.2],
            ],
            [
                [anchor_x + 0.1, anchor_y + 0.1],
                [anchor_x + 0.3, anchor_y + 0.1],
                [anchor_x + 0.3, anchor_y + 0.15],
                [anchor_x + 0.1, anchor_y + 0.15],
                [anchor_x + 0.1, anchor_y + 0.1],
            ],
        ],
    }

    with pytest.raises(MapAssetValidationError, match="hole.*inside|intersect"):
        _load_bundle(_write_bundle(tmp_path, localities, rivers, provenance))


def test_loader_rejects_overlapping_multipolygon_components(
    tmp_path: Path,
    valid_bundle: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
) -> None:
    localities, rivers, provenance = copy.deepcopy(valid_bundle)
    anchor_x, anchor_y = _features(localities)[0]["properties"]["label_anchor_wgs84"]
    first = [
        [anchor_x - 0.2, anchor_y - 0.2],
        [anchor_x + 0.2, anchor_y - 0.2],
        [anchor_x + 0.2, anchor_y + 0.2],
        [anchor_x - 0.2, anchor_y + 0.2],
        [anchor_x - 0.2, anchor_y - 0.2],
    ]
    overlapping = [
        [anchor_x + 0.1, anchor_y + 0.1],
        [anchor_x + 0.3, anchor_y + 0.1],
        [anchor_x + 0.3, anchor_y + 0.3],
        [anchor_x + 0.1, anchor_y + 0.3],
        [anchor_x + 0.1, anchor_y + 0.1],
    ]
    _features(localities)[0]["geometry"] = {
        "type": "MultiPolygon",
        "coordinates": [[first], [overlapping]],
    }

    with pytest.raises(MapAssetValidationError, match="MultiPolygon.*overlap|component"):
        _load_bundle(_write_bundle(tmp_path, localities, rivers, provenance))


def test_loader_rejects_boolean_where_river_component_index_requires_integer(
    tmp_path: Path,
    valid_bundle: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
) -> None:
    localities, rivers, provenance = copy.deepcopy(valid_bundle)
    _features(rivers)[0]["properties"]["component_index"] = False

    with pytest.raises(MapAssetValidationError, match="river selector"):
        _load_bundle(_write_bundle(tmp_path, localities, rivers, provenance))


def test_loader_rejects_wrong_locality_pcode_or_source_name(
    tmp_path: Path,
    valid_bundle: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
) -> None:
    localities, rivers, provenance = copy.deepcopy(valid_bundle)
    properties = _features(localities)[0]["properties"]
    properties["adm2_pcode"] = "SD01999"
    properties["source_name"] = "Jabal Awliya"

    with pytest.raises(MapAssetValidationError, match="P-code|identity"):
        _load_bundle(_write_bundle(tmp_path, localities, rivers, provenance))


def test_loader_rejects_label_anchor_outside_locality_geometry(
    tmp_path: Path,
    valid_bundle: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
) -> None:
    localities, rivers, provenance = copy.deepcopy(valid_bundle)
    _features(localities)[0]["properties"]["label_anchor_wgs84"] = [0, 0]

    with pytest.raises(MapAssetValidationError, match="label_anchor.*inside"):
        _load_bundle(_write_bundle(tmp_path, localities, rivers, provenance))


def test_loader_rejects_altered_derived_asset_before_parsing(
    tmp_path: Path,
    valid_bundle: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
) -> None:
    localities, rivers, provenance = copy.deepcopy(valid_bundle)
    paths = _write_bundle(tmp_path, localities, rivers, provenance)
    locality_path, _, _ = paths
    locality_path.write_bytes(locality_path.read_bytes() + b" ")

    with pytest.raises(MapAssetValidationError, match="hash|size"):
        _load_bundle(paths)


@pytest.mark.parametrize(
    ("field_path", "value", "message"),
    [
        (("sources", "boundary_archive", "sha256"), "0" * 64, "boundary archive"),
        (("sources", "boundary_member", "sha256"), "0" * 64, "boundary member"),
        (("sources", "river_source", "sha256"), "0" * 64, "river source"),
        (("transform", "projection"), "EPSG:4326", "projection"),
    ],
)
def test_loader_rejects_unpinned_provenance_contract(
    tmp_path: Path,
    valid_bundle: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
    field_path: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    localities, rivers, provenance = copy.deepcopy(valid_bundle)
    target: dict[str, Any] = provenance
    for field in field_path[:-1]:
        target = target[field]
    target[field_path[-1]] = value

    with pytest.raises(MapAssetValidationError, match=message):
        _load_bundle(_write_bundle(tmp_path, localities, rivers, provenance))


def test_source_urls_are_allowed_only_in_closed_source_metadata(
    tmp_path: Path,
    valid_bundle: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
) -> None:
    localities, rivers, provenance = copy.deepcopy(valid_bundle)
    assert localities["source"]["source_url"].startswith("https://")
    _features(localities)[0]["properties"]["name"] = "https://example.invalid/map"

    with pytest.raises(MapAssetValidationError, match="external reference"):
        _load_bundle(_write_bundle(tmp_path, localities, rivers, provenance))


def test_loader_rejects_duplicate_json_object_fields(tmp_path: Path) -> None:
    localities = tmp_path / "khartoum-localities.geojson"
    localities.write_text(
        '{"type":"FeatureCollection","type":"FeatureCollection"}', encoding="utf-8"
    )
    (tmp_path / "khartoum-nile.geojson").write_text("{}", encoding="utf-8")
    (tmp_path / "map-provenance.json").write_text("{}", encoding="utf-8")

    with pytest.raises(MapAssetValidationError, match="duplicate field"):
        load_map_asset(localities)


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


def test_provenance_records_exact_natural_earth_source() -> None:
    provenance = json.loads(DEFAULT_MAP_PROVENANCE_PATH.read_text(encoding="utf-8"))
    river_source = provenance["sources"]["river_source"]

    assert river_source["sha256"] == EXPECTED_RIVER_SHA256
    assert river_source["bytes"] == 7_307_743
    assert provenance["transform"]["river_selectors"] == [
        {
            "id": "white_nile",
            "feature_index": 278,
            "component_index": 0,
            "featurecla": "River",
            "rivernum": 17,
            "name_en": "White Nile",
        },
        {
            "id": "blue_nile",
            "feature_index": 1075,
            "component_index": 0,
            "featurecla": "River",
            "rivernum": 73,
            "name_en": "Blue Nile",
        },
        {
            "id": "nile",
            "feature_index": 732,
            "component_index": 4,
            "featurecla": "River",
            "rivernum": 4,
            "name_en": "Nile",
        },
    ]
    adjustment = provenance["transform"]["blue_nile_confluence_adjustment"]
    assert adjustment["original_end"] == [32.488862, 15.632595]
    assert adjustment["appended_end"] == [32.490164, 15.634548]
    assert adjustment["distance_m"] == pytest.approx(258.07, abs=0.01)


def test_builder_clips_crossing_line_to_boundary_mask() -> None:
    fragments = _clip_line_to_mask(((-5.0, 5.0), (15.0, 5.0)), _square_mask())

    assert fragments == (((0.0, 5.0), (10.0, 5.0)),)


def test_builder_clipping_emits_disconnected_fragments_around_hole() -> None:
    fragments = _clip_line_to_mask(
        ((-5.0, 5.0), (15.0, 5.0)),
        _square_mask(with_hole=True),
    )

    assert fragments == (
        ((0.0, 5.0), (4.0, 5.0)),
        ((6.0, 5.0), (10.0, 5.0)),
    )


def test_builder_clipping_keeps_collinear_boundary_but_rejects_point_tangent() -> None:
    assert _clip_line_to_mask(((-5.0, 0.0), (15.0, 0.0)), _square_mask()) == (
        ((0.0, 0.0), (10.0, 0.0)),
    )

    with pytest.raises(BuildError, match="does not intersect"):
        _clip_line_to_mask(((-5.0, -5.0), (0.0, 0.0), (-5.0, 5.0)), _square_mask())


def test_builder_fails_closed_when_pinned_river_selector_identity_drifts() -> None:
    selected_feature = {
        "type": "Feature",
        "properties": {
            "featurecla": "River",
            "rivernum": 17,
            "name_en": "Changed White Nile",
        },
        "geometry": {
            "type": "MultiLineString",
            "coordinates": [[[31.0, 9.0], [32.0, 15.0]]],
        },
    }
    source = {
        "type": "FeatureCollection",
        "features": [*({} for _ in range(278)), selected_feature],
    }

    with pytest.raises(BuildError, match="selector identity changed"):
        _river_component(source, RIVER_SELECTORS[0])

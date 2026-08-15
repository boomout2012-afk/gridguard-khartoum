"""Pure-Python loading, provenance verification, and projection of offline map assets."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any

type Point = tuple[float, float]
type Ring = tuple[Point, ...]
type RawPolygon = tuple[Ring, ...]
type RawBoundaryGeometry = tuple[RawPolygon, ...]

_MAP_ASSET_ROOT: Traversable = files("gridguard").joinpath("assets", "map")
DEFAULT_MAP_ASSET_PATH: Traversable = _MAP_ASSET_ROOT.joinpath("khartoum-localities.geojson")
DEFAULT_RIVER_ASSET_PATH: Traversable = _MAP_ASSET_ROOT.joinpath("khartoum-nile.geojson")
DEFAULT_MAP_PROVENANCE_PATH: Traversable = _MAP_ASSET_ROOT.joinpath("map-provenance.json")
DEFAULT_MAP_PROVENANCE_SHA256 = "146833d6c9c4721360b540f71623e27048f3cc3edc563e0342a1040e1957d4f9"

EXPECTED_LOCALITY_IDS = (
    "khartoum",
    "omdurman",
    "bahri",
    "east_nile",
    "jebel_awlia",
    "karari",
    "ombada",
)
_EXPECTED_LOCALITIES = {
    "khartoum": ("Khartoum", "Khartoum", "SD01007"),
    "omdurman": ("Omdurman", "Um Durman", "SD01006"),
    "bahri": ("Bahri", "Bahri", "SD01003"),
    "east_nile": ("East Nile", "Sharg An Neel", "SD01004"),
    "jebel_awlia": ("Jebel Awlia", "Jebel Awlia", "SD01001"),
    "karari": ("Karari", "Karrari", "SD01005"),
    "ombada": ("Ombada", "Um Bada", "SD01002"),
}
_EXPECTED_BOUNDARY_SOURCE = {
    "dataset": "Sudan administrative boundaries (COD-AB)",
    "provider": "United Nations Office for the Coordination of Humanitarian Affairs",
    "boundary_year": 2020,
    "valid_from": "2020-08-31",
    "source_revision": "v03",
    "source_sha256": "c13cf1e8515730267d41bd560a1f1a0489552ab2900abb070a9aececacca1bb9",
    "license": "CC BY 3.0 IGO",
    "source_url": "https://data.humdata.org/dataset/cod-ab-sdn",
    "license_url": "https://creativecommons.org/licenses/by/3.0/igo/",
    "modifications": [
        "Selected ADM1_PCODE SD01 and the seven approved ADM2 P-codes.",
        "Mapped source English names to GridGuard locality display names.",
        "Retained raw WGS84 geometry and source Arabic locality names.",
    ],
}
_EXPECTED_RIVER_SOURCE = {
    "dataset": "Natural Earth 1:10m rivers and lake centerlines",
    "provider": "Natural Earth",
    "source_revision": "v5.0.0",
    "source_sha256": "bb854a900ecbd3b408df46d5e16e3e0f974ba55993f9d8b5c26e855273c0905a",
    "license": "Public domain",
    "source_url": (
        "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
        "b70b7b8766c3c83d5f5145cb6f1c93921d383d3f/geojson/"
        "ne_10m_rivers_lake_centerlines.geojson"
    ),
    "license_url": "https://www.naturalearthdata.com/about/terms-of-use/",
    "modifications": [
        "Selected three pinned river feature components.",
        "Connected the Blue Nile endpoint to the pinned confluence coordinate.",
        "Clipped linework to the dissolved seven-locality boundary mask.",
    ],
}
_EXPECTED_RIVER_SELECTORS = (
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
)
_EXPECTED_SCENE = {"projection": "EPSG:3857", "width": 1000, "height": 720, "padding": 36}

_LOCALITY_COLLECTION_FIELDS = frozenset(
    {
        "type",
        "schema_version",
        "map_id",
        "kind",
        "title",
        "disclaimer",
        "source",
        "scene",
        "keyboard_order",
        "features",
    }
)
_BOUNDARY_SOURCE_FIELDS = frozenset(_EXPECTED_BOUNDARY_SOURCE)
_RIVER_SOURCE_FIELDS = frozenset(_EXPECTED_RIVER_SOURCE)
_SCENE_FIELDS = frozenset({"projection", "width", "height", "padding"})
_FEATURE_FIELDS = frozenset({"type", "properties", "geometry"})
_LOCALITY_PROPERTY_FIELDS = frozenset(
    {
        "id",
        "name",
        "source_name",
        "source_name_ar",
        "adm2_pcode",
        "label_anchor_wgs84",
        "neighbors",
    }
)
_DIRECTION_FIELDS = frozenset({"up", "right", "down", "left"})
_GEOMETRY_FIELDS = frozenset({"type", "coordinates"})
_RIVER_COLLECTION_FIELDS = frozenset({"type", "schema_version", "map_id", "source", "features"})
_RIVER_PROPERTY_FIELDS = frozenset(
    {"id", "name", "feature_index", "component_index", "featurecla", "rivernum"}
)
_PROVENANCE_FIELDS = frozenset(
    {"schema_version", "map_id", "sources", "derived_assets", "transform"}
)
_PROVENANCE_SOURCE_FIELDS = frozenset({"boundary_archive", "boundary_member", "river_source"})
_BOUNDARY_ARCHIVE_FIELDS = frozenset({"source_url", "sha256", "bytes"})
_BOUNDARY_MEMBER_FIELDS = frozenset({"archive_member", "sha256", "bytes"})
_PROVENANCE_RIVER_FIELDS = frozenset({"source_url", "revision", "sha256", "bytes"})
_DERIVED_ASSET_FIELDS = frozenset({"sha256", "bytes"})
_TRANSFORM_FIELDS = frozenset(
    {
        "adm1_pcode",
        "adm2_pcodes",
        "display_name_mapping",
        "projection",
        "scene",
        "river_selectors",
        "blue_nile_confluence_adjustment",
        "river_clip",
    }
)
_ADJUSTMENT_FIELDS = frozenset({"original_end", "appended_end", "distance_m"})
_RIVER_SELECTOR_FIELDS = frozenset(
    {"id", "feature_index", "component_index", "featurecla", "rivernum", "name_en"}
)
_EXTERNAL_REFERENCE_PATTERN = re.compile(
    r"(?:\b[a-z][a-z0-9+.-]*://|\b(?:data|file|mailto|urn):|\bwww\.|[\\/]|"
    r"\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}\b)",
    re.IGNORECASE,
)
_WEB_MERCATOR_LATITUDE_LIMIT = 85.0511287798066


class MapAssetValidationError(ValueError):
    """Raised when a map asset is missing, malformed, unpinned, or unsafe."""


@dataclass(frozen=True, slots=True)
class SceneRect:
    """Scene bounds in projected display coordinates."""

    x: float
    y: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height


@dataclass(frozen=True, slots=True)
class DirectionalNeighbors:
    """Arrow-key destinations for one locality."""

    up: str | None
    right: str | None
    down: str | None
    left: str | None


@dataclass(frozen=True, slots=True)
class MapSource:
    """Pinned administrative-boundary source identity and attribution."""

    dataset: str
    provider: str
    boundary_year: int
    valid_from: str
    source_revision: str
    source_sha256: str
    license: str
    source_url: str
    license_url: str
    modifications: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProjectionSpec:
    """One fixed Web Mercator-to-scene transform shared by all map layers."""

    source_crs: str
    crs: str
    width: float
    height: float
    padding: float
    scale: float
    projected_bounds: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class BoundaryPolygon:
    """One projected polygon with an exterior ring and zero or more holes."""

    outer: Ring
    holes: tuple[Ring, ...]


@dataclass(frozen=True, slots=True)
class BoundaryGeometry:
    """Projected Polygon or MultiPolygon geometry."""

    geometry_type: str
    polygons: tuple[BoundaryPolygon, ...]


@dataclass(frozen=True, slots=True)
class LocalityShape:
    """Projected locality geometry plus pinned source and keyboard metadata."""

    locality_id: str
    name: str
    source_name: str
    source_name_ar: str
    adm2_pcode: str
    geometry: BoundaryGeometry
    label_anchor: Point
    neighbors: DirectionalNeighbors

    @property
    def polygon(self) -> Ring:
        """Return the first exterior ring for version-one graphics-item compatibility."""

        return self.geometry.polygons[0].outer


@dataclass(frozen=True, slots=True)
class RiverPath:
    """One named river feature containing one or more projected line strings."""

    river_id: str
    name: str
    lines: tuple[tuple[Point, ...], ...]

    @property
    def points(self) -> tuple[Point, ...]:
        """Return the longest line for the version-one map-view compatibility path."""

        return max(self.lines, key=len)


@dataclass(frozen=True, slots=True)
class RiverContext:
    """Named geographic river context from the pinned Natural Earth source."""

    label: str
    paths: tuple[RiverPath, ...]


@dataclass(frozen=True, slots=True)
class MapAsset:
    """Validated and projected offline administrative-boundary map definition."""

    schema_version: int
    map_id: str
    kind: str
    title: str
    disclaimer: str
    source: MapSource
    projection: ProjectionSpec
    scene_rect: SceneRect
    keyboard_order: tuple[str, ...]
    localities: tuple[LocalityShape, ...]
    river_context: RiverContext

    def locality(self, locality_id: str) -> LocalityShape:
        """Return one locality shape, raising ``KeyError`` for an unknown ID."""

        for locality in self.localities:
            if locality.locality_id == locality_id:
                return locality
        raise KeyError(locality_id)


@dataclass(frozen=True, slots=True)
class _RawLocality:
    locality_id: str
    name: str
    source_name: str
    source_name_ar: str
    adm2_pcode: str
    geometry_type: str
    geometry: RawBoundaryGeometry
    label_anchor_wgs84: Point
    neighbors: DirectionalNeighbors


@dataclass(frozen=True, slots=True)
class _SceneTransform:
    width: float
    height: float
    padding: float
    scale: float
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    offset_x: float
    offset_y: float

    def point(self, point: Point) -> Point:
        projected_x, projected_y = _web_mercator(point)
        return (
            self.offset_x + (projected_x - self.min_x) * self.scale,
            self.height - (self.offset_y + (projected_y - self.min_y) * self.scale),
        )


def load_map_asset(
    path: str | Path | None = None,
    *,
    river_path: str | Path | None = None,
    provenance_path: str | Path | None = None,
) -> MapAsset:
    """Load, hash-verify, validate, and project the bundled geographic map."""

    locality_asset: Traversable = DEFAULT_MAP_ASSET_PATH if path is None else Path(path)
    if path is None:
        river_asset: Traversable = (
            DEFAULT_RIVER_ASSET_PATH if river_path is None else Path(river_path)
        )
        provenance_asset: Traversable = (
            DEFAULT_MAP_PROVENANCE_PATH if provenance_path is None else Path(provenance_path)
        )
    else:
        locality_parent = Path(path).parent
        river_asset = (
            locality_parent / "khartoum-nile.geojson" if river_path is None else Path(river_path)
        )
        provenance_asset = (
            locality_parent / "map-provenance.json"
            if provenance_path is None
            else Path(provenance_path)
        )

    locality_bytes, locality_payload = _read_json(locality_asset, label="locality map asset")
    river_bytes, river_payload = _read_json(river_asset, label="river map asset")
    provenance_bytes, provenance_payload = _read_json(provenance_asset, label="map provenance")
    if path is None and river_path is None and provenance_path is None:
        actual_provenance_hash = hashlib.sha256(provenance_bytes).hexdigest()
        if actual_provenance_hash != DEFAULT_MAP_PROVENANCE_SHA256:
            raise MapAssetValidationError(
                "map provenance hash mismatch: "
                f"expected {DEFAULT_MAP_PROVENANCE_SHA256}, received {actual_provenance_hash}"
            )
    _verify_provenance(provenance_payload, locality_bytes=locality_bytes, river_bytes=river_bytes)
    return _parse_map_asset(locality_payload, river_payload)


def _read_json(asset: Traversable, *, label: str) -> tuple[bytes, object]:
    try:
        data = asset.read_bytes()
        payload = json.loads(
            data,
            object_pairs_hook=_object_without_duplicate_fields,
            parse_constant=_reject_nonfinite_json_number,
        )
    except FileNotFoundError as error:
        raise MapAssetValidationError(f"{label} is missing: {asset}") from error
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MapAssetValidationError(f"{label} cannot be read: {error}") from error
    return data, payload


def _verify_provenance(
    payload: object,
    *,
    locality_bytes: bytes,
    river_bytes: bytes,
) -> None:
    root = _closed_object(payload, _PROVENANCE_FIELDS, field="map provenance")
    if type(root["schema_version"]) is not int or root["schema_version"] != 1:
        raise MapAssetValidationError("map provenance schema_version must be 1")
    if root["map_id"] != "khartoum-locality-boundaries-2020":
        raise MapAssetValidationError("map provenance map_id does not match the approved map")

    sources = _closed_object(root["sources"], _PROVENANCE_SOURCE_FIELDS, field="sources")
    boundary_archive = _closed_object(
        sources["boundary_archive"], _BOUNDARY_ARCHIVE_FIELDS, field="boundary archive"
    )
    _require_exact(
        boundary_archive,
        {
            "source_url": (
                "https://data.humdata.org/dataset/a66a4b6c-92de-4507-9546-aa1900474180/"
                "resource/018af991-4aa7-4043-a0d5-e429a55851fb/download/"
                "sdn_admin_boundaries.geojson.zip"
            ),
            "sha256": "75330d4e12e53f966df0e5683e04f0919e9964952fbbf827f3729038e603c95e",
            "bytes": 1_200_093,
        },
        field="boundary archive",
    )
    boundary_member = _closed_object(
        sources["boundary_member"], _BOUNDARY_MEMBER_FIELDS, field="boundary member"
    )
    _require_exact(
        boundary_member,
        {
            "archive_member": "sdn_admin2.geojson",
            "sha256": "c13cf1e8515730267d41bd560a1f1a0489552ab2900abb070a9aececacca1bb9",
            "bytes": 1_621_077,
        },
        field="boundary member",
    )
    river_source = _closed_object(
        sources["river_source"], _PROVENANCE_RIVER_FIELDS, field="river source"
    )
    _require_exact(
        river_source,
        {
            "source_url": _EXPECTED_RIVER_SOURCE["source_url"],
            "revision": "b70b7b8766c3c83d5f5145cb6f1c93921d383d3f",
            "sha256": _EXPECTED_RIVER_SOURCE["source_sha256"],
            "bytes": 7_307_743,
        },
        field="river source",
    )

    derived = _closed_object(
        root["derived_assets"],
        frozenset({"khartoum-localities.geojson", "khartoum-nile.geojson"}),
        field="derived_assets",
    )
    _verify_derived_asset(
        derived["khartoum-localities.geojson"],
        locality_bytes,
        field="khartoum-localities.geojson",
    )
    _verify_derived_asset(
        derived["khartoum-nile.geojson"],
        river_bytes,
        field="khartoum-nile.geojson",
    )
    _validate_transform(root["transform"])


def _verify_derived_asset(value: object, data: bytes, *, field: str) -> None:
    item = _closed_object(value, _DERIVED_ASSET_FIELDS, field=f"derived asset {field}")
    expected_hash = _sha256_string(item["sha256"], field=f"derived asset {field}.sha256")
    expected_size = _positive_integer(item["bytes"], field=f"derived asset {field}.bytes")
    if len(data) != expected_size:
        raise MapAssetValidationError(
            f"derived asset {field} size mismatch: expected {expected_size}, received {len(data)}"
        )
    actual_hash = hashlib.sha256(data).hexdigest()
    if actual_hash != expected_hash:
        raise MapAssetValidationError(
            f"derived asset {field} hash mismatch: expected {expected_hash}, received {actual_hash}"
        )


def _validate_transform(value: object) -> None:
    transform = _closed_object(value, _TRANSFORM_FIELDS, field="transform")
    if transform["adm1_pcode"] != "SD01":
        raise MapAssetValidationError("transform ADM1 P-code must be SD01")
    expected_pcodes = sorted(item[2] for item in _EXPECTED_LOCALITIES.values())
    if transform["adm2_pcodes"] != expected_pcodes:
        raise MapAssetValidationError("transform ADM2 P-codes do not match the approved localities")
    expected_names = {
        source_name: display_name
        for display_name, source_name, _ in sorted(
            _EXPECTED_LOCALITIES.values(), key=lambda item: item[1]
        )
    }
    if transform["display_name_mapping"] != expected_names:
        raise MapAssetValidationError("transform display-name mapping does not match the contract")
    if transform["projection"] != "EPSG:3857":
        raise MapAssetValidationError("transform projection must be EPSG:3857")
    scene = _closed_object(transform["scene"], _SCENE_FIELDS, field="transform.scene")
    _require_exact(scene, _EXPECTED_SCENE, field="transform scene")
    selectors = _list(transform["river_selectors"], field="transform.river_selectors")
    parsed_selectors = [
        _closed_object(item, _RIVER_SELECTOR_FIELDS, field=f"river selector {index}")
        for index, item in enumerate(selectors)
    ]
    if not _exact_value_equal(parsed_selectors, list(_EXPECTED_RIVER_SELECTORS)):
        raise MapAssetValidationError("transform river selectors do not match the pinned features")
    adjustment = _closed_object(
        transform["blue_nile_confluence_adjustment"],
        _ADJUSTMENT_FIELDS,
        field="Blue Nile confluence adjustment",
    )
    if not _exact_value_equal(
        adjustment,
        {
            "original_end": [32.488862, 15.632595],
            "appended_end": [32.490164, 15.634548],
            "distance_m": 258.07,
        },
    ):
        raise MapAssetValidationError("Blue Nile confluence adjustment does not match the contract")
    if transform["river_clip"] != "dissolved seven-locality mask":
        raise MapAssetValidationError("transform river clip must use the dissolved locality mask")


def _parse_map_asset(locality_payload: object, river_payload: object) -> MapAsset:
    root = _closed_object(
        locality_payload, _LOCALITY_COLLECTION_FIELDS, field="locality FeatureCollection"
    )
    if root["type"] != "FeatureCollection":
        raise MapAssetValidationError("locality map type must be 'FeatureCollection'")
    if type(root["schema_version"]) is not int or root["schema_version"] != 2:
        raise MapAssetValidationError("locality map schema_version must be 2")
    if root["map_id"] != "khartoum-locality-boundaries-2020":
        raise MapAssetValidationError("locality map_id does not match the approved map")
    if root["kind"] != "administrative_boundaries":
        raise MapAssetValidationError("locality map kind must be 'administrative_boundaries'")
    title = _display_string(root["title"], field="title")
    disclaimer = _display_string(root["disclaimer"], field="disclaimer")
    folded_disclaimer = disclaimer.casefold()
    if (
        "synthetic" not in folded_disclaimer
        or "not an electrical-network topology" not in folded_disclaimer
    ):
        raise MapAssetValidationError(
            "disclaimer must identify synthetic data and exclude electrical-network topology"
        )
    source = _parse_boundary_source(root["source"])
    projection_values = _parse_scene(root["scene"])
    keyboard_order = _parse_keyboard_order(root["keyboard_order"])
    raw_localities = _parse_localities(root["features"])
    _validate_neighbors(raw_localities)
    transform = _create_transform(raw_localities, projection_values)
    localities = tuple(_project_locality(locality, transform) for locality in raw_localities)
    river_context = _parse_rivers(river_payload, transform)
    projection = ProjectionSpec(
        source_crs="EPSG:4326",
        crs="EPSG:3857",
        width=transform.width,
        height=transform.height,
        padding=transform.padding,
        scale=transform.scale,
        projected_bounds=(transform.min_x, transform.min_y, transform.max_x, transform.max_y),
    )
    return MapAsset(
        schema_version=2,
        map_id="khartoum-locality-boundaries-2020",
        kind="administrative_boundaries",
        title=title,
        disclaimer=disclaimer,
        source=source,
        projection=projection,
        scene_rect=SceneRect(0.0, 0.0, transform.width, transform.height),
        keyboard_order=keyboard_order,
        localities=localities,
        river_context=river_context,
    )


def _parse_boundary_source(value: object) -> MapSource:
    source = _closed_object(value, _BOUNDARY_SOURCE_FIELDS, field="boundary source")
    _require_exact(source, _EXPECTED_BOUNDARY_SOURCE, field="boundary source")
    return MapSource(
        dataset=str(source["dataset"]),
        provider=str(source["provider"]),
        boundary_year=int(source["boundary_year"]),
        valid_from=str(source["valid_from"]),
        source_revision=str(source["source_revision"]),
        source_sha256=str(source["source_sha256"]),
        license=str(source["license"]),
        source_url=str(source["source_url"]),
        license_url=str(source["license_url"]),
        modifications=tuple(str(item) for item in source["modifications"]),
    )


def _parse_scene(value: object) -> tuple[float, float, float]:
    scene = _closed_object(value, _SCENE_FIELDS, field="scene")
    _require_exact(scene, _EXPECTED_SCENE, field="scene")
    width = _positive_number(scene["width"], field="scene.width")
    height = _positive_number(scene["height"], field="scene.height")
    padding = _positive_number(scene["padding"], field="scene.padding")
    if padding * 2 >= min(width, height):
        raise MapAssetValidationError("scene padding must leave a positive drawing area")
    return width, height, padding


def _parse_keyboard_order(value: object) -> tuple[str, ...]:
    items = _list(value, field="keyboard_order")
    order = tuple(_display_string(item, field="keyboard_order item") for item in items)
    if order != EXPECTED_LOCALITY_IDS:
        raise MapAssetValidationError(
            "keyboard_order must contain every approved locality ID exactly once and in order"
        )
    return order


def _parse_localities(value: object) -> tuple[_RawLocality, ...]:
    items = _list(value, field="features")
    localities: list[_RawLocality] = []
    for index, raw_feature in enumerate(items):
        feature_field = f"features[{index}]"
        feature = _closed_object(raw_feature, _FEATURE_FIELDS, field=feature_field)
        if feature["type"] != "Feature":
            raise MapAssetValidationError(f"{feature_field}.type must be 'Feature'")
        properties = _closed_object(
            feature["properties"], _LOCALITY_PROPERTY_FIELDS, field=f"{feature_field}.properties"
        )
        locality_id = _display_string(properties["id"], field=f"{feature_field}.properties.id")
        if index >= len(EXPECTED_LOCALITY_IDS) or locality_id != EXPECTED_LOCALITY_IDS[index]:
            raise MapAssetValidationError(
                "locality features must follow the approved keyboard order"
            )
        name = _display_string(properties["name"], field=f"{feature_field}.properties.name")
        source_name = _display_string(
            properties["source_name"], field=f"{feature_field}.properties.source_name"
        )
        source_name_ar = _display_string(
            properties["source_name_ar"], field=f"{feature_field}.properties.source_name_ar"
        )
        expected_name, expected_source_name, expected_pcode = _EXPECTED_LOCALITIES[locality_id]
        identity = (
            name,
            source_name,
            properties["adm2_pcode"],
        )
        if identity != (expected_name, expected_source_name, expected_pcode):
            raise MapAssetValidationError(
                f"locality identity or P-code does not match the contract for {locality_id!r}"
            )
        anchor = _geographic_point(
            properties["label_anchor_wgs84"],
            field=f"{feature_field}.properties.label_anchor_wgs84",
        )
        geometry_type, geometry = _boundary_geometry(
            feature["geometry"], field=f"{feature_field}.geometry"
        )
        if not _point_in_boundary_geometry(anchor, geometry):
            raise MapAssetValidationError(
                f"{feature_field}.properties.label_anchor_wgs84 must lie inside its geometry"
            )
        neighbors = _parse_neighbors(properties["neighbors"], field=f"{feature_field}.neighbors")
        localities.append(
            _RawLocality(
                locality_id=locality_id,
                name=name,
                source_name=source_name,
                source_name_ar=source_name_ar,
                adm2_pcode=expected_pcode,
                geometry_type=geometry_type,
                geometry=geometry,
                label_anchor_wgs84=anchor,
                neighbors=neighbors,
            )
        )
    if len(localities) != len(EXPECTED_LOCALITY_IDS):
        raise MapAssetValidationError("localities must contain exactly the seven approved features")
    return tuple(localities)


def _parse_neighbors(value: object, *, field: str) -> DirectionalNeighbors:
    neighbors = _closed_object(value, _DIRECTION_FIELDS, field=field)
    return DirectionalNeighbors(
        up=_neighbor(neighbors["up"], field=f"{field}.up"),
        right=_neighbor(neighbors["right"], field=f"{field}.right"),
        down=_neighbor(neighbors["down"], field=f"{field}.down"),
        left=_neighbor(neighbors["left"], field=f"{field}.left"),
    )


def _neighbor(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _display_string(value, field=field)


def _validate_neighbors(localities: Sequence[_RawLocality]) -> None:
    by_id = {locality.locality_id: locality for locality in localities}
    adjacency: dict[str, set[str]] = {}
    for locality in localities:
        destinations = (
            locality.neighbors.up,
            locality.neighbors.right,
            locality.neighbors.down,
            locality.neighbors.left,
        )
        adjacency[locality.locality_id] = {item for item in destinations if item is not None}
        for destination in destinations:
            if destination is not None and destination not in by_id:
                raise MapAssetValidationError(
                    f"locality {locality.locality_id!r} has unknown neighbor {destination!r}"
                )
            if destination == locality.locality_id:
                raise MapAssetValidationError(
                    f"locality {locality.locality_id!r} cannot be its own neighbor"
                )
    expected_ids = set(by_id)
    for origin in expected_ids:
        reachable: set[str] = set()
        pending = [origin]
        while pending:
            current = pending.pop()
            if current in reachable:
                continue
            reachable.add(current)
            pending.extend(adjacency[current] - reachable)
        if reachable != expected_ids:
            raise MapAssetValidationError(
                "keyboard navigation must be connected from every locality"
            )


def _boundary_geometry(value: object, *, field: str) -> tuple[str, RawBoundaryGeometry]:
    geometry = _closed_object(value, _GEOMETRY_FIELDS, field=field)
    geometry_type = geometry["type"]
    coordinates = geometry["coordinates"]
    if geometry_type == "Polygon":
        raw_polygons = [coordinates]
    elif geometry_type == "MultiPolygon":
        raw_polygons = coordinates
    else:
        raise MapAssetValidationError(f"{field}.type must be Polygon or MultiPolygon")
    polygons = _list(raw_polygons, field=f"{field}.coordinates")
    if not polygons:
        raise MapAssetValidationError(f"{field} must contain at least one polygon")
    parsed_polygons: list[RawPolygon] = []
    for polygon_index, raw_polygon in enumerate(polygons):
        raw_rings = _list(raw_polygon, field=f"{field}.coordinates[{polygon_index}]")
        if not raw_rings:
            raise MapAssetValidationError(f"{field} polygon must contain an outer ring")
        rings = tuple(
            _geographic_ring(
                raw_ring,
                field=f"{field}.coordinates[{polygon_index}][{ring_index}]",
            )
            for ring_index, raw_ring in enumerate(raw_rings)
        )
        _validate_polygon_rings(rings, field=field)
        parsed_polygons.append(rings)
    parsed_geometry = tuple(parsed_polygons)
    _validate_multipolygon_components(parsed_geometry, field=field)
    return str(geometry_type), parsed_geometry


def _geographic_ring(value: object, *, field: str) -> Ring:
    items = _list(value, field=field)
    if len(items) < 4:
        raise MapAssetValidationError(f"{field} must contain at least four positions")
    points = tuple(
        _geographic_point(item, field=f"{field}[{index}]") for index, item in enumerate(items)
    )
    if points[0] != points[-1]:
        raise MapAssetValidationError(f"{field} must be closed")
    open_ring = points[:-1]
    if len(set(open_ring)) < 3:
        raise MapAssetValidationError(f"{field} must contain at least three distinct positions")
    _validate_simple_ring(open_ring, field=field)
    twice_area = math.fsum(
        start[0] * end[1] - end[0] * start[1]
        for start, end in zip(open_ring, open_ring[1:] + open_ring[:1], strict=True)
    )
    if abs(twice_area) <= 1e-12:
        raise MapAssetValidationError(f"{field} must have nonzero area")
    return points


def _validate_simple_ring(ring: Ring, *, field: str) -> None:
    if len(set(ring)) != len(ring):
        raise MapAssetValidationError(f"{field} must be simple and contain no duplicate vertices")
    edges = tuple(zip(ring, ring[1:] + ring[:1], strict=True))
    for first_index, first_edge in enumerate(edges):
        for second_index in range(first_index + 1, len(edges)):
            adjacent = second_index == first_index + 1 or (
                first_index == 0 and second_index == len(edges) - 1
            )
            if not adjacent and _segments_intersect(first_edge, edges[second_index]):
                raise MapAssetValidationError(f"{field} must be simple and must not self-intersect")


def _segments_intersect(first: tuple[Point, Point], second: tuple[Point, Point]) -> bool:
    first_start, first_end = first
    second_start, second_end = second
    orientations = (
        _orientation(first_start, first_end, second_start),
        _orientation(first_start, first_end, second_end),
        _orientation(second_start, second_end, first_start),
        _orientation(second_start, second_end, first_end),
    )
    first_start_side, first_end_side, second_start_side, second_end_side = orientations
    if first_start_side * first_end_side < 0 and second_start_side * second_end_side < 0:
        return True
    return (
        (first_start_side == 0 and _point_on_segment(second_start, first_start, first_end))
        or (first_end_side == 0 and _point_on_segment(second_end, first_start, first_end))
        or (second_start_side == 0 and _point_on_segment(first_start, second_start, second_end))
        or (second_end_side == 0 and _point_on_segment(first_end, second_start, second_end))
    )


def _validate_polygon_rings(polygon: RawPolygon, *, field: str) -> None:
    outer = polygon[0]
    holes = polygon[1:]
    for hole_index, hole in enumerate(holes, start=1):
        if any(
            not _point_in_ring(point, outer, include_boundary=False) for point in hole[:-1]
        ) or _rings_intersect(outer, hole):
            raise MapAssetValidationError(
                f"{field} hole {hole_index} must lie strictly inside and not intersect "
                "its outer ring"
            )
        for previous_index, previous in enumerate(holes[: hole_index - 1], start=1):
            overlapping = (
                _rings_intersect(previous, hole)
                or _point_in_ring(hole[0], previous, include_boundary=True)
                or _point_in_ring(previous[0], hole, include_boundary=True)
            )
            if overlapping:
                raise MapAssetValidationError(
                    f"{field} holes {previous_index} and {hole_index} must not overlap"
                )


def _validate_multipolygon_components(geometry: RawBoundaryGeometry, *, field: str) -> None:
    for first_index, first in enumerate(geometry):
        for second_index in range(first_index + 1, len(geometry)):
            second = geometry[second_index]
            boundaries_intersect = any(
                _rings_intersect(first_ring, second_ring)
                for first_ring in first
                for second_ring in second
            )
            interiors_overlap = _point_in_boundary_geometry(first[0][0], (second,)) or (
                _point_in_boundary_geometry(second[0][0], (first,))
            )
            if boundaries_intersect or interiors_overlap:
                raise MapAssetValidationError(
                    f"{field} MultiPolygon components {first_index} and {second_index} "
                    "must not overlap or contain one another"
                )


def _rings_intersect(first: Ring, second: Ring) -> bool:
    first_edges = tuple(zip(first, first[1:], strict=False))
    second_edges = tuple(zip(second, second[1:], strict=False))
    return any(
        _segments_intersect(first_edge, second_edge)
        for first_edge in first_edges
        for second_edge in second_edges
    )


def _orientation(start: Point, end: Point, point: Point) -> int:
    first_x = end[0] - start[0]
    first_y = end[1] - start[1]
    second_x = point[0] - start[0]
    second_y = point[1] - start[1]
    scale = max(abs(first_x), abs(first_y), abs(second_x), abs(second_y), 1.0)
    cross_product = (first_x / scale) * (second_y / scale) - (first_y / scale) * (second_x / scale)
    if abs(cross_product) <= 1e-12:
        return 0
    return 1 if cross_product > 0 else -1


def _geographic_point(value: object, *, field: str) -> Point:
    items = _list(value, field=field)
    if len(items) != 2:
        raise MapAssetValidationError(f"{field} must contain exactly longitude and latitude")
    longitude = _number(items[0], field=f"{field}[0]")
    latitude = _number(items[1], field=f"{field}[1]")
    if not -180 <= longitude <= 180:
        raise MapAssetValidationError(f"{field} longitude must be between -180 and 180")
    if not -_WEB_MERCATOR_LATITUDE_LIMIT <= latitude <= _WEB_MERCATOR_LATITUDE_LIMIT:
        raise MapAssetValidationError(f"{field} latitude is outside the EPSG:3857 limit")
    return longitude, latitude


def _point_in_boundary_geometry(point: Point, geometry: RawBoundaryGeometry) -> bool:
    return any(
        _point_in_ring(point, polygon[0], include_boundary=True)
        and not any(_point_in_ring(point, hole, include_boundary=True) for hole in polygon[1:])
        for polygon in geometry
    )


def _point_in_ring(point: Point, ring: Ring, *, include_boundary: bool) -> bool:
    if any(
        _point_on_segment(point, start, end) for start, end in zip(ring, ring[1:], strict=False)
    ):
        return include_boundary
    point_x, point_y = point
    inside = False
    for (start_x, start_y), (end_x, end_y) in zip(ring, ring[1:], strict=False):
        if (start_y > point_y) == (end_y > point_y):
            continue
        crossing_x = start_x + (point_y - start_y) * (end_x - start_x) / (end_y - start_y)
        if point_x < crossing_x:
            inside = not inside
    return inside


def _point_on_segment(point: Point, start: Point, end: Point) -> bool:
    cross = (end[0] - start[0]) * (point[1] - start[1]) - (end[1] - start[1]) * (
        point[0] - start[0]
    )
    scale = max(abs(end[0] - start[0]), abs(end[1] - start[1]), 1.0)
    tolerance = 1e-12 * scale
    return abs(cross) <= tolerance and (
        min(start[0], end[0]) - tolerance <= point[0] <= max(start[0], end[0]) + tolerance
        and min(start[1], end[1]) - tolerance <= point[1] <= max(start[1], end[1]) + tolerance
    )


def _web_mercator(point: Point) -> Point:
    longitude, latitude = point
    earth_radius = 6_378_137.0
    longitude_radians = math.radians(longitude)
    latitude_radians = math.radians(latitude)
    return (
        earth_radius * longitude_radians,
        earth_radius * math.log(math.tan(math.pi / 4 + latitude_radians / 2)),
    )


def _create_transform(
    localities: Sequence[_RawLocality], scene: tuple[float, float, float]
) -> _SceneTransform:
    width, height, padding = scene
    projected = tuple(
        _web_mercator(point)
        for locality in localities
        for polygon in locality.geometry
        for ring in polygon
        for point in ring[:-1]
    )
    if not projected:
        raise MapAssetValidationError("locality geometry must not be empty")
    min_x = min(point[0] for point in projected)
    min_y = min(point[1] for point in projected)
    max_x = max(point[0] for point in projected)
    max_y = max(point[1] for point in projected)
    projected_width = max_x - min_x
    projected_height = max_y - min_y
    if projected_width <= 0 or projected_height <= 0:
        raise MapAssetValidationError("locality projected bounds must have positive area")
    scale = min(
        (width - 2 * padding) / projected_width,
        (height - 2 * padding) / projected_height,
    )
    drawn_width = projected_width * scale
    drawn_height = projected_height * scale
    return _SceneTransform(
        width=width,
        height=height,
        padding=padding,
        scale=scale,
        min_x=min_x,
        min_y=min_y,
        max_x=max_x,
        max_y=max_y,
        offset_x=(width - drawn_width) / 2,
        offset_y=(height - drawn_height) / 2,
    )


def _project_locality(locality: _RawLocality, transform: _SceneTransform) -> LocalityShape:
    polygons = tuple(
        BoundaryPolygon(
            outer=tuple(transform.point(point) for point in polygon[0][:-1]),
            holes=tuple(
                tuple(transform.point(point) for point in hole[:-1]) for hole in polygon[1:]
            ),
        )
        for polygon in locality.geometry
    )
    return LocalityShape(
        locality_id=locality.locality_id,
        name=locality.name,
        source_name=locality.source_name,
        source_name_ar=locality.source_name_ar,
        adm2_pcode=locality.adm2_pcode,
        geometry=BoundaryGeometry(geometry_type=locality.geometry_type, polygons=polygons),
        label_anchor=transform.point(locality.label_anchor_wgs84),
        neighbors=locality.neighbors,
    )


def _parse_rivers(payload: object, transform: _SceneTransform) -> RiverContext:
    root = _closed_object(payload, _RIVER_COLLECTION_FIELDS, field="river FeatureCollection")
    if root["type"] != "FeatureCollection":
        raise MapAssetValidationError("river map type must be 'FeatureCollection'")
    if type(root["schema_version"]) is not int or root["schema_version"] != 1:
        raise MapAssetValidationError("river map schema_version must be 1")
    if root["map_id"] != "khartoum-nile-context-2020":
        raise MapAssetValidationError("river map_id does not match the approved map")
    source = _closed_object(root["source"], _RIVER_SOURCE_FIELDS, field="river source metadata")
    _require_exact(source, _EXPECTED_RIVER_SOURCE, field="river source metadata")
    features = _list(root["features"], field="river features")
    if len(features) != len(_EXPECTED_RIVER_SELECTORS):
        raise MapAssetValidationError("river map must contain exactly three selected features")
    paths: list[RiverPath] = []
    for index, (raw_feature, selector) in enumerate(
        zip(features, _EXPECTED_RIVER_SELECTORS, strict=True)
    ):
        field = f"river features[{index}]"
        feature = _closed_object(raw_feature, _FEATURE_FIELDS, field=field)
        if feature["type"] != "Feature":
            raise MapAssetValidationError(f"{field}.type must be 'Feature'")
        properties = _closed_object(
            feature["properties"], _RIVER_PROPERTY_FIELDS, field=f"{field}.properties"
        )
        expected_properties = {
            "id": selector["id"],
            "name": selector["name_en"],
            "feature_index": selector["feature_index"],
            "component_index": selector["component_index"],
            "featurecla": selector["featurecla"],
            "rivernum": selector["rivernum"],
        }
        if not _exact_value_equal(properties, expected_properties):
            raise MapAssetValidationError(f"{field} does not match the pinned river selector")
        lines = _river_geometry(feature["geometry"], field=f"{field}.geometry")
        projected_lines = tuple(tuple(transform.point(point) for point in line) for line in lines)
        for line_index, line in enumerate(projected_lines):
            for point_index, point in enumerate(line):
                _require_scene_point(
                    point,
                    transform,
                    field=f"{field}.geometry line {line_index} point {point_index}",
                )
        paths.append(
            RiverPath(
                river_id=str(properties["id"]),
                name=str(properties["name"]),
                lines=projected_lines,
            )
        )
    return RiverContext(label="Nile rivers (geographic context)", paths=tuple(paths))


def _river_geometry(value: object, *, field: str) -> tuple[tuple[Point, ...], ...]:
    geometry = _closed_object(value, _GEOMETRY_FIELDS, field=field)
    geometry_type = geometry["type"]
    coordinates = geometry["coordinates"]
    if geometry_type == "LineString":
        raw_lines = [coordinates]
    elif geometry_type == "MultiLineString":
        raw_lines = coordinates
    else:
        raise MapAssetValidationError(f"{field}.type must be LineString or MultiLineString")
    lines = _list(raw_lines, field=f"{field}.coordinates")
    if not lines:
        raise MapAssetValidationError(f"{field} must contain at least one line")
    result: list[tuple[Point, ...]] = []
    for line_index, raw_line in enumerate(lines):
        points = _list(raw_line, field=f"{field}.coordinates[{line_index}]")
        if len(points) < 2:
            raise MapAssetValidationError(f"{field} line must contain at least two positions")
        parsed = tuple(
            _geographic_point(point, field=f"{field}.coordinates[{line_index}][{point_index}]")
            for point_index, point in enumerate(points)
        )
        if len(set(parsed)) < 2:
            raise MapAssetValidationError(f"{field} line must contain two distinct positions")
        result.append(parsed)
    return tuple(result)


def _require_scene_point(point: Point, transform: _SceneTransform, *, field: str) -> None:
    tolerance = 1e-7
    if not (
        -tolerance <= point[0] <= transform.width + tolerance
        and -tolerance <= point[1] <= transform.height + tolerance
    ):
        raise MapAssetValidationError(f"{field} must lie inside the projected scene")


def _closed_object(value: object, fields: frozenset[str], *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MapAssetValidationError(f"{field} must be an object")
    actual = set(value)
    missing = fields - actual
    unknown = actual - fields
    if missing:
        raise MapAssetValidationError(f"{field} is missing field(s): {', '.join(sorted(missing))}")
    if unknown:
        raise MapAssetValidationError(f"{field} has unknown field(s): {', '.join(sorted(unknown))}")
    return value


def _list(value: object, *, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise MapAssetValidationError(f"{field} must be a list")
    return value


def _number(value: object, *, field: str) -> float:
    if type(value) not in (int, float):
        raise MapAssetValidationError(f"{field} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise MapAssetValidationError(f"{field} must be finite")
    return number


def _positive_number(value: object, *, field: str) -> float:
    number = _number(value, field=field)
    if number <= 0:
        raise MapAssetValidationError(f"{field} must be positive")
    return number


def _positive_integer(value: object, *, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise MapAssetValidationError(f"{field} must be a positive integer")
    return value


def _display_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MapAssetValidationError(f"{field} must be a non-empty string")
    if value != value.strip():
        raise MapAssetValidationError(f"{field} must not have surrounding whitespace")
    if _EXTERNAL_REFERENCE_PATTERN.search(value):
        raise MapAssetValidationError(f"{field} contains an external reference")
    return value


def _sha256_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise MapAssetValidationError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _require_exact(actual: Mapping[str, Any], expected: Mapping[str, Any], *, field: str) -> None:
    if not _exact_value_equal(actual, expected):
        raise MapAssetValidationError(f"{field} does not match the pinned contract")


def _exact_value_equal(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _exact_value_equal(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _exact_value_equal(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected, strict=True)
        )
    return actual == expected


def _object_without_duplicate_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MapAssetValidationError(f"map JSON contains duplicate field {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json_number(value: str) -> None:
    raise MapAssetValidationError(f"map JSON contains non-finite JSON number {value}")

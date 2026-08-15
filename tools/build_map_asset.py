"""Build the pinned, offline Khartoum geographic map assets.

This is a build-time tool only.  The desktop runtime reads the generated GeoJSON
with the Python standard library and never performs network or GIS operations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
import urllib.request
import zipfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

type Point = tuple[float, float]
type Ring = tuple[Point, ...]
type Polygon = tuple[Ring, ...]
type Geometry = tuple[Polygon, ...]

BOUNDARY_DATASET_URL = "https://data.humdata.org/dataset/cod-ab-sdn"
BOUNDARY_ARCHIVE_URL = (
    "https://data.humdata.org/dataset/a66a4b6c-92de-4507-9546-aa1900474180/"
    "resource/018af991-4aa7-4043-a0d5-e429a55851fb/download/"
    "sdn_admin_boundaries.geojson.zip"
)
BOUNDARY_ARCHIVE_SHA256 = "75330d4e12e53f966df0e5683e04f0919e9964952fbbf827f3729038e603c95e"
BOUNDARY_ARCHIVE_BYTES = 1_200_093
BOUNDARY_MEMBER = "sdn_admin2.geojson"
BOUNDARY_MEMBER_SHA256 = "c13cf1e8515730267d41bd560a1f1a0489552ab2900abb070a9aececacca1bb9"
BOUNDARY_MEMBER_BYTES = 1_621_077

RIVER_SOURCE_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "b70b7b8766c3c83d5f5145cb6f1c93921d383d3f/geojson/"
    "ne_10m_rivers_lake_centerlines.geojson"
)
RIVER_SOURCE_REVISION = "b70b7b8766c3c83d5f5145cb6f1c93921d383d3f"
RIVER_SOURCE_SHA256 = "bb854a900ecbd3b408df46d5e16e3e0f974ba55993f9d8b5c26e855273c0905a"
RIVER_SOURCE_BYTES = 7_307_743

OUTPUT_MAP_ID = "khartoum-locality-boundaries-2020"
OUTPUT_LOCALITIES = "khartoum-localities.geojson"
OUTPUT_RIVERS = "khartoum-nile.geojson"
OUTPUT_PROVENANCE = "map-provenance.json"
OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "src" / "gridguard" / "assets" / "map"

SCENE = {"projection": "EPSG:3857", "width": 1000, "height": 720, "padding": 36}
DISCLAIMER = (
    "Administrative boundary context only; synthetic decision-support data are not geographic "
    "measurements and this is not an electrical-network topology."
)

LOCALITY_SPECS = (
    {
        "id": "khartoum",
        "name": "Khartoum",
        "source_name": "Khartoum",
        "pcode": "SD01007",
        "neighbors": {
            "up": "bahri",
            "right": "east_nile",
            "down": "jebel_awlia",
            "left": "omdurman",
        },
    },
    {
        "id": "omdurman",
        "name": "Omdurman",
        "source_name": "Um Durman",
        "pcode": "SD01006",
        "neighbors": {
            "up": "karari",
            "right": "khartoum",
            "down": "jebel_awlia",
            "left": "ombada",
        },
    },
    {
        "id": "bahri",
        "name": "Bahri",
        "source_name": "Bahri",
        "pcode": "SD01003",
        "neighbors": {
            "up": "karari",
            "right": "east_nile",
            "down": "khartoum",
            "left": "omdurman",
        },
    },
    {
        "id": "east_nile",
        "name": "East Nile",
        "source_name": "Sharg An Neel",
        "pcode": "SD01004",
        "neighbors": {
            "up": "bahri",
            "right": None,
            "down": "khartoum",
            "left": "bahri",
        },
    },
    {
        "id": "jebel_awlia",
        "name": "Jebel Awlia",
        "source_name": "Jebel Awlia",
        "pcode": "SD01001",
        "neighbors": {
            "up": "khartoum",
            "right": "east_nile",
            "down": None,
            "left": "ombada",
        },
    },
    {
        "id": "karari",
        "name": "Karari",
        "source_name": "Karrari",
        "pcode": "SD01005",
        "neighbors": {
            "up": None,
            "right": "bahri",
            "down": "omdurman",
            "left": "ombada",
        },
    },
    {
        "id": "ombada",
        "name": "Ombada",
        "source_name": "Um Bada",
        "pcode": "SD01002",
        "neighbors": {
            "up": "karari",
            "right": "omdurman",
            "down": "jebel_awlia",
            "left": None,
        },
    },
)

RIVER_SELECTORS = (
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

BLUE_NILE_ORIGINAL_END = (32.488862, 15.632595)
NILE_CONFLUENCE = (32.490164, 15.634548)


class BuildError(RuntimeError):
    """Raised when a pinned source or selected feature differs from the contract."""


def _json_without_duplicate_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BuildError(f"source JSON contains duplicate field {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise BuildError(f"source JSON contains non-finite number {value}")


def _load_json_bytes(data: bytes, *, source: str) -> dict[str, Any]:
    try:
        value = json.loads(
            data,
            object_pairs_hook=_json_without_duplicate_fields,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BuildError(f"{source} is not valid UTF-8 GeoJSON: {error}") from error
    if not isinstance(value, dict):
        raise BuildError(f"{source} root must be an object")
    return value


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _verify_bytes(data: bytes, *, size: int, sha256: str, source: str) -> None:
    if len(data) != size:
        raise BuildError(f"{source} size mismatch: expected {size}, received {len(data)}")
    actual_hash = _sha256(data)
    if actual_hash != sha256:
        raise BuildError(f"{source} hash mismatch: expected {sha256}, received {actual_hash}")


def _download(url: str, path: Path) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "GridGuard-map-builder/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            data = response.read()
    except OSError as error:
        raise BuildError(f"could not download pinned source {url}: {error}") from error
    path.write_bytes(data)
    return data


def _read_boundary_member(archive_data: bytes) -> bytes:
    with tempfile.TemporaryFile() as archive_file:
        archive_file.write(archive_data)
        archive_file.seek(0)
        try:
            with zipfile.ZipFile(archive_file) as archive:
                names = set(archive.namelist())
                if BOUNDARY_MEMBER not in names:
                    raise BuildError(f"boundary archive is missing {BOUNDARY_MEMBER}")
                member_data = archive.read(BOUNDARY_MEMBER)
        except zipfile.BadZipFile as error:
            raise BuildError("boundary archive is not a valid ZIP file") from error
    _verify_bytes(
        member_data,
        size=BOUNDARY_MEMBER_BYTES,
        sha256=BOUNDARY_MEMBER_SHA256,
        source="boundary member",
    )
    return member_data


def _point(value: object, *, field: str) -> Point:
    if not isinstance(value, list) or len(value) < 2:
        raise BuildError(f"{field} must contain longitude and latitude")
    longitude, latitude = value[:2]
    if type(longitude) not in (int, float) or type(latitude) not in (int, float):
        raise BuildError(f"{field} coordinates must be numbers")
    point = (float(longitude), float(latitude))
    if not all(math.isfinite(coordinate) for coordinate in point):
        raise BuildError(f"{field} coordinates must be finite")
    return point


def _ring(value: object, *, field: str) -> Ring:
    if not isinstance(value, list) or len(value) < 4:
        raise BuildError(f"{field} must contain at least four positions")
    ring = tuple(_point(item, field=f"{field}[{index}]") for index, item in enumerate(value))
    if ring[0] != ring[-1]:
        raise BuildError(f"{field} must be closed")
    if len(set(ring[:-1])) < 3:
        raise BuildError(f"{field} must contain at least three distinct positions")
    return ring


def _geometry(value: object, *, field: str) -> Geometry:
    if not isinstance(value, dict) or set(value) != {"type", "coordinates"}:
        raise BuildError(f"{field} must be a closed GeoJSON geometry object")
    geometry_type = value["type"]
    coordinates = value["coordinates"]
    if geometry_type == "Polygon":
        polygons = [coordinates]
    elif geometry_type == "MultiPolygon":
        polygons = coordinates
    else:
        raise BuildError(f"{field} must be Polygon or MultiPolygon")
    if not isinstance(polygons, list) or not polygons:
        raise BuildError(f"{field} must contain at least one polygon")
    result: list[Polygon] = []
    for polygon_index, raw_polygon in enumerate(polygons):
        if not isinstance(raw_polygon, list) or not raw_polygon:
            raise BuildError(f"{field} polygon {polygon_index} must contain an outer ring")
        result.append(
            tuple(
                _ring(raw_ring, field=f"{field}[{polygon_index}][{ring_index}]")
                for ring_index, raw_ring in enumerate(raw_polygon)
            )
        )
    return tuple(result)


def _point_on_segment(point: Point, start: Point, end: Point, tolerance: float = 1e-10) -> bool:
    cross = (end[0] - start[0]) * (point[1] - start[1]) - (end[1] - start[1]) * (
        point[0] - start[0]
    )
    scale = max(abs(end[0] - start[0]), abs(end[1] - start[1]), 1.0)
    if abs(cross) > tolerance * scale:
        return False
    return (
        min(start[0], end[0]) - tolerance <= point[0] <= max(start[0], end[0]) + tolerance
        and min(start[1], end[1]) - tolerance <= point[1] <= max(start[1], end[1]) + tolerance
    )


def _point_in_ring(point: Point, ring: Ring) -> bool:
    if any(
        _point_on_segment(point, start, end) for start, end in zip(ring, ring[1:], strict=False)
    ):
        return True
    x, y = point
    inside = False
    for (start_x, start_y), (end_x, end_y) in zip(ring, ring[1:], strict=False):
        if (start_y > y) == (end_y > y):
            continue
        crossing_x = start_x + (y - start_y) * (end_x - start_x) / (end_y - start_y)
        if x < crossing_x:
            inside = not inside
    return inside


def _point_in_geometry(point: Point, geometry: Geometry) -> bool:
    return any(
        _point_in_ring(point, polygon[0])
        and not any(_point_in_ring(point, hole) for hole in polygon[1:])
        for polygon in geometry
    )


def _point_in_mask(point: Point, mask: Sequence[Geometry]) -> bool:
    return any(_point_in_geometry(point, geometry) for geometry in mask)


def _segment_intersection_parameter(
    start: Point,
    end: Point,
    edge_start: Point,
    edge_end: Point,
) -> float | None:
    segment_x = end[0] - start[0]
    segment_y = end[1] - start[1]
    edge_x = edge_end[0] - edge_start[0]
    edge_y = edge_end[1] - edge_start[1]
    denominator = segment_x * edge_y - segment_y * edge_x
    if abs(denominator) <= 1e-15:
        return None
    delta_x = edge_start[0] - start[0]
    delta_y = edge_start[1] - start[1]
    segment_t = (delta_x * edge_y - delta_y * edge_x) / denominator
    edge_t = (delta_x * segment_y - delta_y * segment_x) / denominator
    tolerance = 1e-10
    if -tolerance <= segment_t <= 1 + tolerance and -tolerance <= edge_t <= 1 + tolerance:
        return min(1.0, max(0.0, segment_t))
    return None


def _mask_edges(mask: Sequence[Geometry]) -> tuple[tuple[Point, Point], ...]:
    return tuple(
        (start, end)
        for geometry in mask
        for polygon in geometry
        for ring in polygon
        for start, end in zip(ring, ring[1:], strict=False)
    )


def _interpolate(start: Point, end: Point, amount: float) -> Point:
    return (
        start[0] + (end[0] - start[0]) * amount,
        start[1] + (end[1] - start[1]) * amount,
    )


def _clip_line_to_mask(line: Sequence[Point], mask: Sequence[Geometry]) -> tuple[Ring, ...]:
    edges = _mask_edges(mask)
    fragments: list[list[Point]] = []
    active: list[Point] = []
    for start, end in zip(line, line[1:], strict=False):
        parameters = [0.0, 1.0]
        for edge_start, edge_end in edges:
            parameter = _segment_intersection_parameter(start, end, edge_start, edge_end)
            if parameter is not None:
                parameters.append(parameter)
        ordered: list[float] = []
        for parameter in sorted(parameters):
            if not ordered or abs(parameter - ordered[-1]) > 1e-10:
                ordered.append(parameter)
        for first, second in zip(ordered, ordered[1:], strict=False):
            interval_start = _interpolate(start, end, first)
            interval_end = _interpolate(start, end, second)
            midpoint = _interpolate(start, end, (first + second) / 2)
            if _point_in_mask(midpoint, mask):
                if not active:
                    active = [interval_start, interval_end]
                elif math.dist(active[-1], interval_start) <= 1e-9:
                    if math.dist(active[-1], interval_end) > 1e-12:
                        active.append(interval_end)
                else:
                    fragments.append(active)
                    active = [interval_start, interval_end]
            elif active:
                fragments.append(active)
                active = []
    if active:
        fragments.append(active)
    cleaned = tuple(tuple(fragment) for fragment in fragments if len(set(fragment)) >= 2)
    if not cleaned:
        raise BuildError("selected river does not intersect the dissolved locality mask")
    return cleaned


def _locality_features(source: dict[str, Any]) -> tuple[list[dict[str, Any]], list[Geometry]]:
    if source.get("type") != "FeatureCollection" or not isinstance(source.get("features"), list):
        raise BuildError("boundary member must be a GeoJSON FeatureCollection")
    by_pcode: dict[str, dict[str, Any]] = {}
    for feature in source["features"]:
        if not isinstance(feature, dict) or not isinstance(feature.get("properties"), dict):
            raise BuildError("boundary source contains an invalid feature")
        properties = feature["properties"]
        if properties.get("adm1_pcode") == "SD01":
            pcode = properties.get("adm2_pcode")
            if not isinstance(pcode, str) or pcode in by_pcode:
                raise BuildError("Khartoum boundary P-codes must be unique strings")
            by_pcode[pcode] = feature
    expected_pcodes = {str(spec["pcode"]) for spec in LOCALITY_SPECS}
    if set(by_pcode) != expected_pcodes:
        raise BuildError(
            f"Khartoum boundary P-codes changed: expected {sorted(expected_pcodes)}, "
            f"received {sorted(by_pcode)}"
        )

    output_features: list[dict[str, Any]] = []
    mask: list[Geometry] = []
    for spec in LOCALITY_SPECS:
        feature = by_pcode[str(spec["pcode"])]
        properties = feature["properties"]
        if properties.get("adm2_name") != spec["source_name"]:
            raise BuildError(f"source name changed for {spec['pcode']}")
        if properties.get("valid_on") != "2020-08-31" or properties.get("version") != "v03":
            raise BuildError(f"source revision changed for {spec['pcode']}")
        source_name_ar = properties.get("adm2_name1")
        if not isinstance(source_name_ar, str) or not source_name_ar.strip():
            raise BuildError(f"Arabic source name is missing for {spec['pcode']}")
        anchor = (properties.get("center_lon"), properties.get("center_lat"))
        anchor_point = _point(list(anchor), field=f"{spec['pcode']} source center")
        geometry = _geometry(feature.get("geometry"), field=f"{spec['pcode']} geometry")
        if not _point_in_geometry(anchor_point, geometry):
            raise BuildError(f"source center is outside the boundary for {spec['pcode']}")
        mask.append(geometry)
        output_features.append(
            {
                "type": "Feature",
                "properties": {
                    "id": spec["id"],
                    "name": spec["name"],
                    "source_name": spec["source_name"],
                    "source_name_ar": source_name_ar,
                    "adm2_pcode": spec["pcode"],
                    "label_anchor_wgs84": list(anchor_point),
                    "neighbors": spec["neighbors"],
                },
                "geometry": feature["geometry"],
            }
        )
    return output_features, mask


def _river_component(source: dict[str, Any], selector: dict[str, Any]) -> tuple[Point, ...]:
    features = source.get("features")
    if source.get("type") != "FeatureCollection" or not isinstance(features, list):
        raise BuildError("Natural Earth source must be a GeoJSON FeatureCollection")
    feature_index = int(selector["feature_index"])
    try:
        feature = features[feature_index]
    except IndexError as error:
        raise BuildError(f"river feature index {feature_index} is missing") from error
    if not isinstance(feature, dict) or not isinstance(feature.get("properties"), dict):
        raise BuildError(f"river feature {feature_index} is invalid")
    properties = feature["properties"]
    expected = {
        "featurecla": selector["featurecla"],
        "rivernum": selector["rivernum"],
        "name_en": selector["name_en"],
    }
    if any(properties.get(field) != value for field, value in expected.items()):
        raise BuildError(f"river selector identity changed at feature {feature_index}")
    geometry = feature.get("geometry")
    if not isinstance(geometry, dict) or geometry.get("type") != "MultiLineString":
        raise BuildError(f"river feature {feature_index} must be a MultiLineString")
    coordinates = geometry.get("coordinates")
    component_index = int(selector["component_index"])
    if not isinstance(coordinates, list) or component_index >= len(coordinates):
        raise BuildError(f"river component {feature_index}:{component_index} is missing")
    component = coordinates[component_index]
    if not isinstance(component, list) or len(component) < 2:
        raise BuildError(f"river component {feature_index}:{component_index} is invalid")
    return tuple(
        _point(item, field=f"river feature {feature_index} component {component_index}[{index}]")
        for index, item in enumerate(component)
    )


def _haversine_metres(start: Point, end: Point) -> float:
    longitude1, latitude1 = map(math.radians, start)
    longitude2, latitude2 = map(math.radians, end)
    latitude_delta = latitude2 - latitude1
    longitude_delta = longitude2 - longitude1
    haversine = math.sin(latitude_delta / 2) ** 2 + (
        math.cos(latitude1) * math.cos(latitude2) * math.sin(longitude_delta / 2) ** 2
    )
    return 2 * 6_371_008.8 * math.asin(math.sqrt(haversine))


def _river_features(source: dict[str, Any], mask: Sequence[Geometry]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for selector in RIVER_SELECTORS:
        component = _river_component(source, selector)
        if selector["id"] == "blue_nile":
            if component[-1] != BLUE_NILE_ORIGINAL_END:
                raise BuildError(
                    f"Blue Nile endpoint changed: expected {BLUE_NILE_ORIGINAL_END}, "
                    f"received {component[-1]}"
                )
            component = (*component, NILE_CONFLUENCE)
        fragments = _clip_line_to_mask(component, mask)
        if len(fragments) == 1:
            geometry: dict[str, object] = {
                "type": "LineString",
                "coordinates": [list(point) for point in fragments[0]],
            }
        else:
            geometry = {
                "type": "MultiLineString",
                "coordinates": [[list(point) for point in fragment] for fragment in fragments],
            }
        output.append(
            {
                "type": "Feature",
                "properties": {
                    "id": selector["id"],
                    "name": selector["name_en"],
                    "feature_index": selector["feature_index"],
                    "component_index": selector["component_index"],
                    "featurecla": selector["featurecla"],
                    "rivernum": selector["rivernum"],
                },
                "geometry": geometry,
            }
        )
    return output


def _encoded(payload: object) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode()


def _write_derived(path: Path, payload: object) -> dict[str, object]:
    data = _encoded(payload)
    path.write_bytes(data)
    return {"sha256": _sha256(data), "bytes": len(data)}


def build_assets(boundary_archive: Path, river_source: Path, output_directory: Path) -> None:
    """Verify the pinned inputs and write the two GeoJSON assets plus provenance."""

    archive_data = boundary_archive.read_bytes()
    _verify_bytes(
        archive_data,
        size=BOUNDARY_ARCHIVE_BYTES,
        sha256=BOUNDARY_ARCHIVE_SHA256,
        source="boundary archive",
    )
    member_data = _read_boundary_member(archive_data)
    river_data = river_source.read_bytes()
    _verify_bytes(
        river_data,
        size=RIVER_SOURCE_BYTES,
        sha256=RIVER_SOURCE_SHA256,
        source="Natural Earth river source",
    )

    boundary_payload = _load_json_bytes(member_data, source=BOUNDARY_MEMBER)
    river_payload = _load_json_bytes(river_data, source="Natural Earth river source")
    locality_features, mask = _locality_features(boundary_payload)
    river_features = _river_features(river_payload, mask)

    localities = {
        "type": "FeatureCollection",
        "schema_version": 2,
        "map_id": OUTPUT_MAP_ID,
        "kind": "administrative_boundaries",
        "title": "Khartoum State Locality Boundaries",
        "disclaimer": DISCLAIMER,
        "source": {
            "dataset": "Sudan administrative boundaries (COD-AB)",
            "provider": "United Nations Office for the Coordination of Humanitarian Affairs",
            "boundary_year": 2020,
            "valid_from": "2020-08-31",
            "source_revision": "v03",
            "source_sha256": BOUNDARY_MEMBER_SHA256,
            "license": "CC BY 3.0 IGO",
            "source_url": BOUNDARY_DATASET_URL,
            "license_url": "https://creativecommons.org/licenses/by/3.0/igo/",
            "modifications": [
                "Selected ADM1_PCODE SD01 and the seven approved ADM2 P-codes.",
                "Mapped source English names to GridGuard locality display names.",
                "Retained raw WGS84 geometry and source Arabic locality names.",
            ],
        },
        "scene": SCENE,
        "keyboard_order": [str(spec["id"]) for spec in LOCALITY_SPECS],
        "features": locality_features,
    }
    rivers = {
        "type": "FeatureCollection",
        "schema_version": 1,
        "map_id": "khartoum-nile-context-2020",
        "source": {
            "dataset": "Natural Earth 1:10m rivers and lake centerlines",
            "provider": "Natural Earth",
            "source_revision": "v5.0.0",
            "source_sha256": RIVER_SOURCE_SHA256,
            "license": "Public domain",
            "source_url": RIVER_SOURCE_URL,
            "license_url": "https://www.naturalearthdata.com/about/terms-of-use/",
            "modifications": [
                "Selected three pinned river feature components.",
                "Connected the Blue Nile endpoint to the pinned confluence coordinate.",
                "Clipped linework to the dissolved seven-locality boundary mask.",
            ],
        },
        "features": river_features,
    }

    output_directory.mkdir(parents=True, exist_ok=True)
    locality_record = _write_derived(output_directory / OUTPUT_LOCALITIES, localities)
    river_record = _write_derived(output_directory / OUTPUT_RIVERS, rivers)
    provenance = {
        "schema_version": 1,
        "map_id": OUTPUT_MAP_ID,
        "sources": {
            "boundary_archive": {
                "source_url": BOUNDARY_ARCHIVE_URL,
                "sha256": BOUNDARY_ARCHIVE_SHA256,
                "bytes": BOUNDARY_ARCHIVE_BYTES,
            },
            "boundary_member": {
                "archive_member": BOUNDARY_MEMBER,
                "sha256": BOUNDARY_MEMBER_SHA256,
                "bytes": BOUNDARY_MEMBER_BYTES,
            },
            "river_source": {
                "source_url": RIVER_SOURCE_URL,
                "revision": RIVER_SOURCE_REVISION,
                "sha256": RIVER_SOURCE_SHA256,
                "bytes": RIVER_SOURCE_BYTES,
            },
        },
        "derived_assets": {
            OUTPUT_LOCALITIES: locality_record,
            OUTPUT_RIVERS: river_record,
        },
        "transform": {
            "adm1_pcode": "SD01",
            "adm2_pcodes": sorted(str(spec["pcode"]) for spec in LOCALITY_SPECS),
            "display_name_mapping": {
                str(spec["source_name"]): str(spec["name"])
                for spec in sorted(LOCALITY_SPECS, key=lambda item: str(item["source_name"]))
            },
            "projection": "EPSG:3857",
            "scene": SCENE,
            "river_selectors": list(RIVER_SELECTORS),
            "blue_nile_confluence_adjustment": {
                "original_end": list(BLUE_NILE_ORIGINAL_END),
                "appended_end": list(NILE_CONFLUENCE),
                "distance_m": round(_haversine_metres(BLUE_NILE_ORIGINAL_END, NILE_CONFLUENCE), 2),
            },
            "river_clip": "dissolved seven-locality mask",
        },
    }
    (output_directory / OUTPUT_PROVENANCE).write_bytes(_encoded(provenance))


def _resolve_sources(arguments: argparse.Namespace, temporary_directory: Path) -> tuple[Path, Path]:
    boundary_archive = arguments.boundary_archive
    if boundary_archive is None:
        boundary_archive = temporary_directory / "sdn_admin_boundaries.geojson.zip"
        _download(BOUNDARY_ARCHIVE_URL, boundary_archive)
    river_source = arguments.river_source
    if river_source is None:
        river_source = temporary_directory / "ne_10m_rivers_lake_centerlines.geojson"
        _download(RIVER_SOURCE_URL, river_source)
    return boundary_archive, river_source


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--boundary-archive",
        type=Path,
        help="Pinned sdn_admin_boundaries.geojson.zip (downloads when omitted)",
    )
    parser.add_argument(
        "--river-source",
        type=Path,
        help="Pinned Natural Earth GeoJSON (downloads when omitted)",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    """Build assets from local verified inputs or download the pinned sources."""

    arguments = _parser().parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="gridguard-map-") as temporary:
        boundary_archive, river_source = _resolve_sources(arguments, Path(temporary))
        build_assets(boundary_archive, river_source, arguments.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

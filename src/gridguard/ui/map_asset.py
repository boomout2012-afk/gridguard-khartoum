"""Pure-Python loading and validation for the bundled locality schematic."""

from __future__ import annotations

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

DEFAULT_MAP_ASSET_PATH: Traversable = files("gridguard").joinpath(
    "assets", "map", "khartoum-schematic.json"
)

EXPECTED_LOCALITY_IDS = (
    "khartoum",
    "omdurman",
    "bahri",
    "east_nile",
    "jebel_awlia",
    "karari",
    "ombada",
)
_EXPECTED_LOCALITIES = dict(
    zip(
        EXPECTED_LOCALITY_IDS,
        (
            "Khartoum",
            "Omdurman",
            "Bahri",
            "East Nile",
            "Jebel Awlia",
            "Karari",
            "Ombada",
        ),
        strict=True,
    )
)
_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "map_id",
        "kind",
        "title",
        "disclaimer",
        "scene_rect",
        "keyboard_order",
        "localities",
        "river_context",
    }
)
_DIRECTION_FIELDS = frozenset({"up", "right", "down", "left"})
_FORBIDDEN_FIELD_TOKENS = frozenset(
    {
        "action",
        "controller",
        "controllers",
        "curtailed",
        "curtailment",
        "demand",
        "demands",
        "episode",
        "fairness",
        "forecast",
        "metric",
        "metrics",
        "observation",
        "policy",
        "prediction",
        "result",
        "results",
        "reward",
        "scenario",
        "sector",
        "served",
        "shortage",
        "supply",
        "tier",
        "tiers",
        "utilization",
        "violation",
    }
)
_FORBIDDEN_VALUE_TOKENS = _FORBIDDEN_FIELD_TOKENS | {
    "energy",
    "kilowatt",
    "kwh",
    "kw",
    "measurement",
    "megawatt",
    "mwh",
    "mw",
    "off",
    "percent",
    "percentage",
    "power",
    "ppo",
    "scientific",
    "service",
    "services",
    "statistic",
    "statistics",
}
_FORBIDDEN_VALUE_PHRASES = frozenset({"critical only", "fixed rotation", "full service"})
_EXTERNAL_REFERENCE_PATTERN = re.compile(
    r"(?:\b[a-z][a-z0-9+.-]*://|\b(?:data|file|mailto|urn):|\bwww\.|[\\/]|"
    r"\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}\b)",
    re.IGNORECASE,
)


class MapAssetValidationError(ValueError):
    """Raised when a schematic asset is missing, malformed, or unsafe."""


@dataclass(frozen=True, slots=True)
class SceneRect:
    """Scene bounds in schematic coordinates."""

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
class LocalityShape:
    """Display-only polygon and keyboard metadata for one locality."""

    locality_id: str
    name: str
    polygon: tuple[Point, ...]
    label_anchor: Point
    neighbors: DirectionalNeighbors


@dataclass(frozen=True, slots=True)
class RiverPath:
    """A schematic river centerline used only for visual context."""

    river_id: str
    name: str
    points: tuple[Point, ...]


@dataclass(frozen=True, slots=True)
class RiverContext:
    """Named river paths that orient the schematic."""

    label: str
    paths: tuple[RiverPath, ...]


@dataclass(frozen=True, slots=True)
class MapAsset:
    """Validated, immutable version-one schematic definition."""

    schema_version: int
    map_id: str
    kind: str
    title: str
    disclaimer: str
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


def load_map_asset(path: str | Path | None = None) -> MapAsset:
    """Load and validate a closed-schema, offline schematic map asset."""

    asset_path: Traversable = DEFAULT_MAP_ASSET_PATH if path is None else Path(path)
    try:
        text = asset_path.read_text(encoding="utf-8")
        payload = json.loads(
            text,
            object_pairs_hook=_object_without_duplicate_fields,
            parse_constant=_reject_nonfinite_json_number,
        )
    except FileNotFoundError as error:
        raise MapAssetValidationError(f"map asset is missing: {asset_path}") from error
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MapAssetValidationError(f"map asset cannot be read: {error}") from error
    return _parse_map_asset(payload)


def _parse_map_asset(payload: object) -> MapAsset:
    _validate_display_only_payload(payload)
    root = _closed_object(payload, _TOP_LEVEL_FIELDS, field="map asset")
    schema_version = root["schema_version"]
    if type(schema_version) is not int or schema_version != 1:
        raise MapAssetValidationError("map asset schema_version must be 1")
    if root["kind"] != "schematic":
        raise MapAssetValidationError("map asset kind must be 'schematic'")

    disclaimer = _nonempty_string(root["disclaimer"], field="disclaimer")
    folded_disclaimer = disclaimer.casefold()
    if (
        "schematic" not in folded_disclaimer
        or "not a geographic" not in folded_disclaimer
        or "not an electrical-network topology" not in folded_disclaimer
    ):
        raise MapAssetValidationError(
            "disclaimer must explicitly identify a schematic that is not a geographic "
            "boundary map and not an electrical-network topology"
        )

    scene_rect = _parse_scene_rect(root["scene_rect"])
    localities = _parse_localities(root["localities"])
    keyboard_order = _parse_keyboard_order(root["keyboard_order"])
    river_context = _parse_river_context(root["river_context"])
    _validate_neighbors(localities)
    _validate_geometry(scene_rect, localities, river_context)
    return MapAsset(
        schema_version=1,
        map_id=_nonempty_string(root["map_id"], field="map_id"),
        kind="schematic",
        title=_nonempty_string(root["title"], field="title"),
        disclaimer=disclaimer,
        scene_rect=scene_rect,
        keyboard_order=keyboard_order,
        localities=localities,
        river_context=river_context,
    )


def _parse_scene_rect(value: object) -> SceneRect:
    item = _closed_object(value, frozenset({"x", "y", "width", "height"}), field="scene_rect")
    rect = SceneRect(
        x=_number(item["x"], field="scene_rect.x"),
        y=_number(item["y"], field="scene_rect.y"),
        width=_number(item["width"], field="scene_rect.width"),
        height=_number(item["height"], field="scene_rect.height"),
    )
    if rect.width <= 0 or rect.height <= 0:
        raise MapAssetValidationError("scene_rect width and height must be positive")
    if not math.isfinite(rect.right) or not math.isfinite(rect.bottom):
        raise MapAssetValidationError("scene_rect bounds must be finite")
    return rect


def _parse_localities(value: object) -> tuple[LocalityShape, ...]:
    items = _list(value, field="localities")
    localities: list[LocalityShape] = []
    for index, raw_item in enumerate(items):
        field = f"localities[{index}]"
        item = _closed_object(
            raw_item,
            frozenset({"id", "name", "polygon", "label_anchor", "neighbors"}),
            field=field,
        )
        neighbors = _closed_object(item["neighbors"], _DIRECTION_FIELDS, field=f"{field}.neighbors")
        localities.append(
            LocalityShape(
                locality_id=_nonempty_string(item["id"], field=f"{field}.id"),
                name=_nonempty_string(item["name"], field=f"{field}.name"),
                polygon=_points(item["polygon"], field=f"{field}.polygon", minimum=3),
                label_anchor=_point(item["label_anchor"], field=f"{field}.label_anchor"),
                neighbors=DirectionalNeighbors(
                    up=_neighbor(neighbors["up"], field=f"{field}.neighbors.up"),
                    right=_neighbor(neighbors["right"], field=f"{field}.neighbors.right"),
                    down=_neighbor(neighbors["down"], field=f"{field}.neighbors.down"),
                    left=_neighbor(neighbors["left"], field=f"{field}.neighbors.left"),
                ),
            )
        )
    _validate_locality_identity(localities)
    return tuple(localities)


def _validate_locality_identity(localities: Sequence[LocalityShape]) -> None:
    ids = [item.locality_id for item in localities]
    names = [item.name for item in localities]
    if len(ids) != len(set(ids)):
        raise MapAssetValidationError("localities contain duplicate locality IDs")
    if len(names) != len({name.casefold() for name in names}):
        raise MapAssetValidationError("localities contain duplicate locality names")
    if set(ids) != set(_EXPECTED_LOCALITIES):
        raise MapAssetValidationError("locality IDs must contain exactly the seven approved IDs")
    if set(names) != set(_EXPECTED_LOCALITIES.values()):
        raise MapAssetValidationError(
            "locality names must contain exactly the seven approved names"
        )
    for locality in localities:
        if _EXPECTED_LOCALITIES[locality.locality_id] != locality.name:
            raise MapAssetValidationError(
                f"locality ID {locality.locality_id!r} does not match name {locality.name!r}"
            )


def _validate_neighbors(localities: Sequence[LocalityShape]) -> None:
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


def _validate_geometry(
    scene_rect: SceneRect,
    localities: Sequence[LocalityShape],
    river_context: RiverContext,
) -> None:
    for locality in localities:
        polygon_field = f"locality {locality.locality_id!r} polygon"
        for index, point in enumerate(locality.polygon):
            _require_point_in_scene(
                point,
                scene_rect,
                field=f"{polygon_field}[{index}]",
            )
        _validate_simple_polygon(locality.polygon, field=polygon_field)
        _require_point_in_scene(
            locality.label_anchor,
            scene_rect,
            field=f"locality {locality.locality_id!r} label_anchor",
        )
        if not _point_strictly_inside_polygon(locality.label_anchor, locality.polygon):
            raise MapAssetValidationError(
                f"locality {locality.locality_id!r} label_anchor must lie inside its polygon"
            )
    for path in river_context.paths:
        for index, point in enumerate(path.points):
            _require_point_in_scene(
                point,
                scene_rect,
                field=f"river path {path.river_id!r} points[{index}]",
            )


def _require_point_in_scene(point: Point, scene_rect: SceneRect, *, field: str) -> None:
    x, y = point
    if not (scene_rect.x <= x <= scene_rect.right and scene_rect.y <= y <= scene_rect.bottom):
        raise MapAssetValidationError(f"{field} must lie inside scene_rect")


def _validate_simple_polygon(polygon: Sequence[Point], *, field: str) -> None:
    if len(set(polygon)) != len(polygon):
        raise MapAssetValidationError(f"{field} must not contain duplicate vertices")

    edges = tuple(
        (polygon[index], polygon[(index + 1) % len(polygon)]) for index in range(len(polygon))
    )
    for first_index, first_edge in enumerate(edges):
        for second_index in range(first_index + 1, len(edges)):
            adjacent = second_index == first_index + 1 or (
                first_index == 0 and second_index == len(edges) - 1
            )
            if not adjacent and _segments_intersect(first_edge, edges[second_index]):
                raise MapAssetValidationError(f"{field} must be simple and must not self-intersect")

    min_x = min(point[0] for point in polygon)
    max_x = max(point[0] for point in polygon)
    min_y = min(point[1] for point in polygon)
    max_y = max(point[1] for point in polygon)
    scale = max(max_x - min_x, max_y - min_y)
    if scale == 0:
        raise MapAssetValidationError(f"{field} must have nonzero area")
    normalized = tuple(((x - min_x) / scale, (y - min_y) / scale) for x, y in polygon)
    twice_area = math.fsum(
        x1 * y2 - x2 * y1
        for (x1, y1), (x2, y2) in zip(
            normalized,
            normalized[1:] + normalized[:1],
            strict=True,
        )
    )
    if abs(twice_area) <= 1e-12:
        raise MapAssetValidationError(f"{field} must have nonzero area")

    for index in range(len(polygon)):
        previous = polygon[index - 1]
        current = polygon[index]
        following = polygon[(index + 1) % len(polygon)]
        if _orientation(previous, current, following) == 0 and (
            _point_on_segment(following, previous, current)
            or _point_on_segment(previous, current, following)
        ):
            raise MapAssetValidationError(f"{field} must be a simple polygon")


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


def _point_on_segment(point: Point, start: Point, end: Point) -> bool:
    if _orientation(start, end, point) != 0:
        return False
    tolerance = 1e-12 * max(abs(end[0] - start[0]), abs(end[1] - start[1]), 1.0)
    return (
        min(start[0], end[0]) - tolerance <= point[0] <= max(start[0], end[0]) + tolerance
        and min(start[1], end[1]) - tolerance <= point[1] <= max(start[1], end[1]) + tolerance
    )


def _point_strictly_inside_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    edges = zip(polygon, polygon[1:] + polygon[:1], strict=True)
    if any(_point_on_segment(point, start, end) for start, end in edges):
        return False

    point_x, point_y = point
    inside = False
    for (start_x, start_y), (end_x, end_y) in zip(
        polygon,
        polygon[1:] + polygon[:1],
        strict=True,
    ):
        if (start_y > point_y) == (end_y > point_y):
            continue
        crossing_x = start_x + (point_y - start_y) * (end_x - start_x) / (end_y - start_y)
        if point_x < crossing_x:
            inside = not inside
    return inside


def _parse_keyboard_order(value: object) -> tuple[str, ...]:
    items = _list(value, field="keyboard_order")
    order = tuple(_nonempty_string(item, field="keyboard_order item") for item in items)
    if len(order) != len(set(order)):
        raise MapAssetValidationError("keyboard_order contains duplicate locality IDs")
    if set(order) != set(_EXPECTED_LOCALITIES):
        raise MapAssetValidationError(
            "keyboard_order must contain every approved locality ID exactly once"
        )
    return order


def _parse_river_context(value: object) -> RiverContext:
    context = _closed_object(value, frozenset({"label", "paths"}), field="river_context")
    raw_paths = _list(context["paths"], field="river_context.paths")
    if not raw_paths:
        raise MapAssetValidationError("river_context.paths must not be empty")
    paths: list[RiverPath] = []
    for index, raw_path in enumerate(raw_paths):
        field = f"river_context.paths[{index}]"
        item = _closed_object(raw_path, frozenset({"id", "name", "points"}), field=field)
        paths.append(
            RiverPath(
                river_id=_nonempty_string(item["id"], field=f"{field}.id"),
                name=_nonempty_string(item["name"], field=f"{field}.name"),
                points=_points(item["points"], field=f"{field}.points", minimum=2),
            )
        )
    if len({path.river_id for path in paths}) != len(paths):
        raise MapAssetValidationError("river_context contains duplicate path IDs")
    if len({path.name.casefold() for path in paths}) != len(paths):
        raise MapAssetValidationError("river_context contains duplicate path names")
    return RiverContext(
        label=_nonempty_string(context["label"], field="river_context.label"),
        paths=tuple(paths),
    )


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


def _list(value: object, *, field: str) -> list[object]:
    if not isinstance(value, list):
        raise MapAssetValidationError(f"{field} must be a list")
    return value


def _nonempty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MapAssetValidationError(f"{field} must be a non-empty string")
    return value


def _number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise MapAssetValidationError(f"{field} must be a number")
    try:
        number = float(value)
    except OverflowError as error:
        raise MapAssetValidationError(f"{field} must be a finite number") from error
    if not math.isfinite(number):
        raise MapAssetValidationError(f"{field} must be a finite number")
    return number


def _point(value: object, *, field: str) -> Point:
    if not isinstance(value, list) or len(value) != 2:
        raise MapAssetValidationError(f"{field} must be a two-number point")
    return (_number(value[0], field=f"{field}[0]"), _number(value[1], field=f"{field}[1]"))


def _points(value: object, *, field: str, minimum: int) -> tuple[Point, ...]:
    items = _list(value, field=field)
    if len(items) < minimum:
        raise MapAssetValidationError(f"{field} must contain at least {minimum} points")
    points = tuple(_point(item, field=f"{field}[{index}]") for index, item in enumerate(items))
    if len(set(points)) < minimum:
        raise MapAssetValidationError(f"{field} must contain at least {minimum} distinct points")
    return points


def _neighbor(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _nonempty_string(value, field=field)


def _object_without_duplicate_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise MapAssetValidationError(f"JSON object contains duplicate field: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json_number(value: str) -> None:
    raise MapAssetValidationError(f"JSON contains non-finite number: {value}")


def _validate_display_only_payload(value: object, *, field: str = "map asset") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            tokens = _text_tokens(key)
            forbidden = tokens & _FORBIDDEN_FIELD_TOKENS
            if forbidden:
                raise MapAssetValidationError(
                    f"map asset is display-only; prohibited scientific or simulation field "
                    f"{key!r} appears at {field}"
                )
            _validate_display_only_payload(child, field=f"{field}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_display_only_payload(child, field=f"{field}[{index}]")
    elif isinstance(value, str):
        if _EXTERNAL_REFERENCE_PATTERN.search(value.strip()):
            raise MapAssetValidationError(f"{field} contains an external reference")
        normalized = _normalized_text(value)
        forbidden = _text_tokens(value) & _FORBIDDEN_VALUE_TOKENS
        if forbidden or any(phrase in normalized for phrase in _FORBIDDEN_VALUE_PHRASES):
            raise MapAssetValidationError(
                f"map asset is display-only; prohibited scientific or controller vocabulary "
                f"appears at {field}"
            )


def _text_tokens(value: str) -> set[str]:
    return set(_normalized_text(value).split())


def _normalized_text(value: str) -> str:
    separated = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
    return " ".join(re.findall(r"[a-z0-9]+", separated.casefold()))

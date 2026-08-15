"""User-interface data contracts that do not require importing Qt."""

from gridguard.ui.map_asset import (
    DEFAULT_MAP_ASSET_PATH,
    EXPECTED_LOCALITY_IDS,
    DirectionalNeighbors,
    LocalityShape,
    MapAsset,
    MapAssetValidationError,
    RiverContext,
    RiverPath,
    SceneRect,
    load_map_asset,
)

__all__ = [
    "DEFAULT_MAP_ASSET_PATH",
    "EXPECTED_LOCALITY_IDS",
    "DirectionalNeighbors",
    "LocalityShape",
    "MapAsset",
    "MapAssetValidationError",
    "RiverContext",
    "RiverPath",
    "SceneRect",
    "load_map_asset",
]

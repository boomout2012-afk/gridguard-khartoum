"""Locate verified runtime assets in a source checkout or installed bundle."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

_SOURCE_SENTINEL = Path("data") / "evidence" / "source-manifest.json"


class ResourceLocationError(RuntimeError):
    """Raised when packaged GridGuard assets cannot be located safely."""


def runtime_root() -> Path:
    """Return the root containing the runtime ``data`` and ``docs/source`` trees."""

    source_root = Path(__file__).resolve().parents[2]
    if (source_root / _SOURCE_SENTINEL).is_file():
        return source_root

    bundled = files("gridguard").joinpath("_bundled")
    bundled_root = Path(str(bundled)).resolve()
    if (bundled_root / _SOURCE_SENTINEL).is_file():
        return bundled_root
    raise ResourceLocationError(
        "GridGuard runtime assets are missing from both the source checkout and installed bundle"
    )


def resource_path(*parts: str | Path) -> Path:
    """Resolve a relative runtime asset path without allowing root escape."""

    relative_path = Path(*parts)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ResourceLocationError(
            "runtime asset path must be relative and remain inside its root"
        )
    root = runtime_root().resolve()
    candidate = (root / relative_path).resolve()
    if not candidate.is_relative_to(root):
        raise ResourceLocationError(
            "runtime asset path must be relative and remain inside its root"
        )
    return candidate

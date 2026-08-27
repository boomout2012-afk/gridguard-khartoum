"""Hatchling hook that bundles the exact audited robustness source bytes."""

from __future__ import annotations

import hashlib
import runpy
import shutil
import tempfile
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

_TRUST_ANCHOR_PATH = "src/gridguard/robustness_provenance.py"
_STAGING_PREFIX = ".gridguard-audited-sources-"


class GridGuardBuildHook(BuildHookInterface):
    """Add raw provenance sources without relocating their importable modules."""

    _staging_directory: tempfile.TemporaryDirectory[str] | None = None

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        del version
        project_root = Path(self.root).resolve()
        build_directory = Path(self.directory).resolve()
        staging_directory = tempfile.TemporaryDirectory(
            prefix=_STAGING_PREFIX,
            dir=build_directory,
        )
        self._staging_directory = staging_directory
        staging_root = Path(staging_directory.name).resolve()

        try:
            force_include = build_data.setdefault("force_include", {})
            for relative_path, source_path in _validated_provenance_sources(project_root):
                staged_path = staging_root / relative_path
                staged_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_path, staged_path)
                force_include[str(staged_path)] = (
                    Path("gridguard/_bundled") / relative_path
                ).as_posix()
        except Exception:
            self._remove_staging()
            raise

    def finalize(
        self,
        version: str,
        build_data: dict[str, Any],
        artifact_path: str,
    ) -> None:
        del version, build_data, artifact_path
        self._remove_staging()

    def _remove_staging(self) -> None:
        staging_directory = self._staging_directory
        if staging_directory is None:
            return

        build_directory = Path(self.directory).resolve()
        resolved_staging_root = Path(staging_directory.name).resolve()
        if (
            resolved_staging_root.parent != build_directory
            or not resolved_staging_root.name.startswith(_STAGING_PREFIX)
        ):
            raise RuntimeError("refusing to remove an invalid wheel-source staging directory")
        staging_directory.cleanup()
        self._staging_directory = None


def _validated_provenance_sources(project_root: Path) -> tuple[tuple[Path, Path], ...]:
    trust_anchor = _resolve_repository_file(project_root, _TRUST_ANCHOR_PATH)
    namespace = runpy.run_path(str(trust_anchor))
    raw_bindings = namespace["ROBUSTNESS_IMPLEMENTATION_SOURCE_BINDINGS"]

    sources: list[tuple[Path, Path]] = [(Path(_TRUST_ANCHOR_PATH), trust_anchor)]
    seen_paths = {_TRUST_ANCHOR_PATH}
    for repository_path, expected_sha256, expected_size_bytes in raw_bindings:
        if repository_path in seen_paths:
            raise ValueError(f"duplicate provenance source path: {repository_path}")
        seen_paths.add(repository_path)

        source_path = _resolve_repository_file(project_root, repository_path)
        source_bytes = source_path.read_bytes()
        if len(source_bytes) != expected_size_bytes:
            raise ValueError(f"provenance source size mismatch: {repository_path}")
        if hashlib.sha256(source_bytes).hexdigest() != expected_sha256:
            raise ValueError(f"provenance source SHA-256 mismatch: {repository_path}")
        sources.append((Path(repository_path), source_path))
    return tuple(sources)


def _resolve_repository_file(project_root: Path, repository_path: str) -> Path:
    relative_path = Path(repository_path)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"invalid provenance source path: {repository_path}")
    source_path = (project_root / relative_path).resolve()
    if not source_path.is_relative_to(project_root) or not source_path.is_file():
        raise FileNotFoundError(f"provenance source is unavailable: {repository_path}")
    return source_path

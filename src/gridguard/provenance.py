"""Fail-closed verification for proposal and migration source artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_TOP_LEVEL_FIELDS = {"schema_version", "source_repository", "artifacts", "lineage"}
_ARTIFACT_FIELDS = {"id", "kind", "path", "sha256"}
_LINEAGE_FIELDS = {
    "git_blob_sha256",
    "git_blob_size_bytes",
    "id",
    "source_path",
    "source_revision",
    "size_bytes",
    "sha256",
}

PROJECT_SOURCE_MANIFEST_SHA256 = "cc06df2492512d1c4de9c73051311f3401d5ddf90eb2563f2f440e53f26194c9"
PROJECT_SOURCE_ARTIFACT_IDS = frozenset(
    {
        "accepted-proposal",
        "frozen-proposal-benchmark",
        "paired-proposal-benchmark",
        "ppo-inference-artifact",
        "ppo-inference-manifest",
    }
)
PROJECT_SOURCE_LINEAGE_IDS = frozenset(
    {
        "source-benchmark",
        "source-ppo-checkpoint",
        "source-simulation-config",
        "source-controllers",
        "source-scenarios",
        "source-simulator",
        "source-dispatch-and-reward",
        "source-policy-network",
        "source-evaluation",
    }
)


class ProvenanceError(ValueError):
    """Raised when source provenance is missing, malformed, or inconsistent."""


@dataclass(frozen=True, slots=True)
class VerifiedArtifact:
    """A manifest artifact whose bytes match its declared digest."""

    artifact_id: str
    kind: str
    relative_path: Path
    sha256: str


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the lowercase SHA-256 digest of a file without loading it all at once."""

    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source_manifest(
    manifest_path: Path,
    *,
    repository_root: Path,
    expected_manifest_sha256: str | None = None,
    expected_artifact_ids: frozenset[str] | None = None,
    expected_lineage_ids: frozenset[str] | None = None,
) -> tuple[VerifiedArtifact, ...]:
    """Verify every local artifact declared by a version-one source manifest."""

    manifest = _read_manifest(
        manifest_path,
        expected_sha256=expected_manifest_sha256,
    )
    schema_version = manifest.get("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        raise ProvenanceError("source manifest schema_version must be 1")
    _validate_manifest_metadata(manifest)

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ProvenanceError("source manifest artifacts must be a list")

    root = repository_root.resolve()
    verified: list[VerifiedArtifact] = []
    seen_ids: set[str] = set()
    seen_paths: set[Path] = set()

    for index, raw_artifact in enumerate(artifacts):
        artifact = _require_mapping(raw_artifact, field=f"artifacts[{index}]")
        _require_exact_fields(
            artifact,
            expected=_ARTIFACT_FIELDS,
            field=f"artifacts[{index}]",
        )
        artifact_id = _require_nonempty_string(artifact, "id", index=index)
        kind = _require_nonempty_string(artifact, "kind", index=index)
        path_text = _require_nonempty_string(artifact, "path", index=index)
        expected_hash = _require_nonempty_string(artifact, "sha256", index=index)

        if artifact_id in seen_ids:
            raise ProvenanceError(f"duplicate artifact id: {artifact_id}")
        seen_ids.add(artifact_id)

        if not _SHA256_PATTERN.fullmatch(expected_hash):
            raise ProvenanceError(f"artifact {artifact_id} has an invalid SHA-256 digest")

        relative_path = Path(path_text)
        candidate = (root / relative_path).resolve()
        if relative_path.is_absolute() or not candidate.is_relative_to(root):
            raise ProvenanceError(f"artifact {artifact_id} must stay inside the repository")
        if candidate in seen_paths:
            raise ProvenanceError(f"duplicate artifact path: {relative_path.as_posix()}")
        seen_paths.add(candidate)

        if not candidate.is_file():
            raise ProvenanceError(f"artifact {artifact_id} is missing: {relative_path.as_posix()}")

        actual_hash = sha256_file(candidate)
        if actual_hash != expected_hash:
            raise ProvenanceError(
                f"artifact {artifact_id} SHA-256 mismatch: "
                f"expected {expected_hash}, found {actual_hash}"
            )

        verified.append(
            VerifiedArtifact(
                artifact_id=artifact_id,
                kind=kind,
                relative_path=relative_path,
                sha256=actual_hash,
            )
        )

    if expected_artifact_ids is not None and seen_ids != expected_artifact_ids:
        raise ProvenanceError(
            "source manifest artifact identities do not match the project trust anchor"
        )
    if expected_lineage_ids is not None:
        actual_lineage_ids = {entry["id"] for entry in manifest.get("lineage", [])}
        if actual_lineage_ids != expected_lineage_ids:
            raise ProvenanceError(
                "source manifest lineage identities do not match the project trust anchor"
            )

    return tuple(verified)


def verify_project_source_manifest(
    manifest_path: Path,
    *,
    repository_root: Path,
) -> tuple[VerifiedArtifact, ...]:
    """Verify the exact, pinned source manifest and its required project identities."""

    return verify_source_manifest(
        manifest_path,
        repository_root=repository_root,
        expected_manifest_sha256=PROJECT_SOURCE_MANIFEST_SHA256,
        expected_artifact_ids=PROJECT_SOURCE_ARTIFACT_IDS,
        expected_lineage_ids=PROJECT_SOURCE_LINEAGE_IDS,
    )


def _read_manifest(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    try:
        raw_bytes = path.read_bytes()
        if expected_sha256 is not None:
            if not _SHA256_PATTERN.fullmatch(expected_sha256):
                raise ProvenanceError("source manifest trust anchor is not a valid SHA-256")
            actual_sha256 = hashlib.sha256(raw_bytes).hexdigest()
            if actual_sha256 != expected_sha256:
                raise ProvenanceError(
                    "source manifest SHA-256 does not match the project trust anchor"
                )
        payload = json.loads(
            raw_bytes.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_constant,
        )
    except FileNotFoundError as error:
        raise ProvenanceError(f"source manifest is missing: {path}") from error
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProvenanceError(f"source manifest cannot be read: {error}") from error
    return _require_mapping(payload, field="source manifest")


def _validate_manifest_metadata(manifest: dict[str, Any]) -> None:
    required_top_level = {"schema_version", "source_repository", "artifacts"}
    missing = required_top_level - manifest.keys()
    unknown = manifest.keys() - _TOP_LEVEL_FIELDS
    if missing:
        raise ProvenanceError(f"source manifest is missing fields: {sorted(missing)}")
    if unknown:
        raise ProvenanceError(f"source manifest has unknown fields: {sorted(unknown)}")

    source_repository = _require_mapping(
        manifest.get("source_repository"),
        field="source_repository",
    )
    _require_exact_fields(
        source_repository,
        expected={"name", "revision"},
        field="source_repository",
    )
    name = source_repository.get("name")
    revision = source_repository.get("revision")
    if not isinstance(name, str) or not name.strip():
        raise ProvenanceError("source_repository.name must be a non-empty string")
    if not isinstance(revision, str) or not _COMMIT_PATTERN.fullmatch(revision):
        raise ProvenanceError(
            "source_repository.revision must be a 40-character lowercase commit hash"
        )

    lineage = manifest.get("lineage", [])
    if not isinstance(lineage, list):
        raise ProvenanceError("source manifest lineage must be a list")
    seen_lineage_ids: set[str] = set()
    for index, raw_entry in enumerate(lineage):
        entry = _require_mapping(raw_entry, field=f"lineage[{index}]")
        _require_exact_fields(
            entry,
            expected=_LINEAGE_FIELDS,
            field=f"lineage[{index}]",
        )
        lineage_id = entry.get("id")
        source_path = entry.get("source_path")
        source_revision = entry.get("source_revision")
        size_bytes = entry.get("size_bytes")
        digest = entry.get("sha256")
        git_blob_size_bytes = entry.get("git_blob_size_bytes")
        git_blob_digest = entry.get("git_blob_sha256")
        if not isinstance(lineage_id, str) or not lineage_id.strip():
            raise ProvenanceError(f"lineage[{index}].id must be a non-empty string")
        if lineage_id in seen_lineage_ids:
            raise ProvenanceError(f"duplicate lineage id: {lineage_id}")
        seen_lineage_ids.add(lineage_id)
        if not isinstance(source_path, str) or not source_path.strip():
            raise ProvenanceError(f"lineage[{index}].source_path must be a non-empty string")
        relative_source = Path(source_path)
        if relative_source.is_absolute() or ".." in relative_source.parts:
            raise ProvenanceError(f"lineage[{index}].source_path must be relative")
        if not isinstance(source_revision, str) or not _COMMIT_PATTERN.fullmatch(source_revision):
            raise ProvenanceError(
                f"lineage[{index}].source_revision must be a 40-character lowercase commit hash"
            )
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes <= 0:
            raise ProvenanceError(f"lineage[{index}].size_bytes must be a positive integer")
        if not isinstance(digest, str) or not _SHA256_PATTERN.fullmatch(digest):
            raise ProvenanceError(f"lineage[{index}].sha256 must be a lowercase SHA-256")
        if (
            isinstance(git_blob_size_bytes, bool)
            or not isinstance(git_blob_size_bytes, int)
            or git_blob_size_bytes <= 0
        ):
            raise ProvenanceError(
                f"lineage[{index}].git_blob_size_bytes must be a positive integer"
            )
        if not isinstance(git_blob_digest, str) or not _SHA256_PATTERN.fullmatch(git_blob_digest):
            raise ProvenanceError(f"lineage[{index}].git_blob_sha256 must be a lowercase SHA-256")


def _require_mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProvenanceError(f"{field} must be an object")
    return value


def _require_exact_fields(
    value: dict[str, Any],
    *,
    expected: set[str],
    field: str,
) -> None:
    missing = expected - value.keys()
    unknown = value.keys() - expected
    if missing:
        raise ProvenanceError(f"{field} is missing fields: {sorted(missing)}")
    if unknown:
        raise ProvenanceError(f"{field} has unknown fields: {sorted(unknown)}")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProvenanceError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise ProvenanceError(f"non-finite JSON value is not allowed: {value}")


def _require_nonempty_string(
    artifact: dict[str, Any],
    field: str,
    *,
    index: int,
) -> str:
    value = artifact.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ProvenanceError(f"artifacts[{index}].{field} must be a non-empty string")
    return value

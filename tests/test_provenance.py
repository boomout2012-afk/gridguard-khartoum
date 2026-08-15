from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from gridguard.provenance import ProvenanceError, sha256_file, verify_source_manifest


def _write_manifest(path: Path, *, artifact_path: str, digest: str) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_repository": {
                    "name": "research archive",
                    "revision": "a" * 40,
                },
                "artifacts": [
                    {
                        "id": "accepted-proposal",
                        "kind": "scope-authority",
                        "path": artifact_path,
                        "sha256": digest,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_sha256_file_returns_lowercase_digest(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"gridguard")

    assert sha256_file(artifact) == hashlib.sha256(b"gridguard").hexdigest()


@pytest.mark.parametrize("chunk_size", [0, -1, 1.5, True])
def test_sha256_file_rejects_invalid_chunk_size(
    tmp_path: Path,
    chunk_size: object,
) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"gridguard")

    with pytest.raises(ValueError, match="positive integer"):
        sha256_file(artifact, chunk_size=chunk_size)  # type: ignore[arg-type]


def test_verify_source_manifest_accepts_matching_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "docs" / "proposal.pdf"
    artifact.parent.mkdir()
    artifact.write_bytes(b"accepted proposal")
    manifest = tmp_path / "source-manifest.json"
    _write_manifest(
        manifest,
        artifact_path="docs/proposal.pdf",
        digest=sha256_file(artifact),
    )

    verified = verify_source_manifest(manifest, repository_root=tmp_path)

    assert [(item.artifact_id, item.kind, item.relative_path) for item in verified] == [
        ("accepted-proposal", "scope-authority", Path("docs/proposal.pdf"))
    ]


@pytest.mark.parametrize("digest", ["0" * 64, "not-a-sha256"])
def test_verify_source_manifest_rejects_invalid_or_mismatched_hash(
    tmp_path: Path,
    digest: str,
) -> None:
    artifact = tmp_path / "proposal.pdf"
    artifact.write_bytes(b"accepted proposal")
    manifest = tmp_path / "source-manifest.json"
    _write_manifest(manifest, artifact_path="proposal.pdf", digest=digest)

    with pytest.raises(ProvenanceError, match="SHA-256"):
        verify_source_manifest(manifest, repository_root=tmp_path)


def test_verify_source_manifest_rejects_missing_artifact(tmp_path: Path) -> None:
    manifest = tmp_path / "source-manifest.json"
    _write_manifest(manifest, artifact_path="missing.pdf", digest="0" * 64)

    with pytest.raises(ProvenanceError, match="missing"):
        verify_source_manifest(manifest, repository_root=tmp_path)


def test_verify_source_manifest_rejects_path_escape(tmp_path: Path) -> None:
    manifest = tmp_path / "source-manifest.json"
    _write_manifest(manifest, artifact_path="../outside.pdf", digest="0" * 64)

    with pytest.raises(ProvenanceError, match="inside the repository"):
        verify_source_manifest(manifest, repository_root=tmp_path)


def test_verify_source_manifest_rejects_unknown_schema(tmp_path: Path) -> None:
    manifest = tmp_path / "source-manifest.json"
    manifest.write_text('{"schema_version": 2, "artifacts": []}', encoding="utf-8")

    with pytest.raises(ProvenanceError, match="schema_version"):
        verify_source_manifest(manifest, repository_root=tmp_path)


@pytest.mark.parametrize("schema_version", [True, 1.0, "1", None])
def test_verify_source_manifest_requires_integer_schema_version(
    tmp_path: Path,
    schema_version: object,
) -> None:
    manifest = tmp_path / "source-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": schema_version,
                "source_repository": {
                    "name": "research archive",
                    "revision": "a" * 40,
                },
                "artifacts": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ProvenanceError, match="schema_version"):
        verify_source_manifest(manifest, repository_root=tmp_path)


def test_verify_source_manifest_rejects_duplicate_json_key(tmp_path: Path) -> None:
    manifest = tmp_path / "source-manifest.json"
    manifest.write_text(
        '{"schema_version":1,"schema_version":1,"source_repository":{},"artifacts":[]}',
        encoding="utf-8",
    )

    with pytest.raises(ProvenanceError, match="duplicate JSON key"):
        verify_source_manifest(manifest, repository_root=tmp_path)


def test_verify_source_manifest_rejects_unknown_artifact_field(tmp_path: Path) -> None:
    artifact = tmp_path / "proposal.pdf"
    artifact.write_bytes(b"accepted proposal")
    manifest = tmp_path / "source-manifest.json"
    _write_manifest(manifest, artifact_path="proposal.pdf", digest=sha256_file(artifact))
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["artifacts"][0]["unexpected"] = True
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProvenanceError, match="unknown fields"):
        verify_source_manifest(manifest, repository_root=tmp_path)


def test_verify_source_manifest_rejects_invalid_source_revision(tmp_path: Path) -> None:
    manifest = tmp_path / "source-manifest.json"
    _write_manifest(manifest, artifact_path="missing.pdf", digest="0" * 64)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["source_repository"]["revision"] = "short"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProvenanceError, match="40-character"):
        verify_source_manifest(manifest, repository_root=tmp_path)


def test_verify_source_manifest_requires_git_blob_lineage_fields(tmp_path: Path) -> None:
    artifact = tmp_path / "proposal.pdf"
    artifact.write_bytes(b"accepted proposal")
    manifest = tmp_path / "source-manifest.json"
    _write_manifest(manifest, artifact_path="proposal.pdf", digest=sha256_file(artifact))
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["lineage"] = [
        {
            "id": "source-file",
            "source_path": "src/source.py",
            "source_revision": "a" * 40,
            "size_bytes": 10,
            "sha256": "b" * 64,
        }
    ]
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProvenanceError, match="missing fields"):
        verify_source_manifest(manifest, repository_root=tmp_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("git_blob_size_bytes", True, "positive integer"),
        ("git_blob_size_bytes", 0, "positive integer"),
        ("git_blob_sha256", "invalid", "lowercase SHA-256"),
    ],
)
def test_verify_source_manifest_rejects_invalid_git_blob_lineage_metadata(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    artifact = tmp_path / "proposal.pdf"
    artifact.write_bytes(b"accepted proposal")
    manifest = tmp_path / "source-manifest.json"
    _write_manifest(manifest, artifact_path="proposal.pdf", digest=sha256_file(artifact))
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    lineage = {
        "git_blob_sha256": "b" * 64,
        "git_blob_size_bytes": 10,
        "id": "source-file",
        "source_path": "src/source.py",
        "source_revision": "a" * 40,
        "size_bytes": 10,
        "sha256": "b" * 64,
    }
    lineage[field] = value
    payload["lineage"] = [lineage]
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProvenanceError, match=message):
        verify_source_manifest(manifest, repository_root=tmp_path)

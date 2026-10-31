from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tools.verify_review_package import ReviewPackageError, verify_review_package


def manifest_for(path: str, raw: bytes) -> dict:
    return {
        "schema": "gridguard.review_package.1",
        "files": [
            {"path": path, "size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        ],
    }


def test_current_review_package_matches_its_manifest() -> None:
    root = Path(__file__).resolve().parents[1]
    result = verify_review_package(root)
    assert result["passed"] is True
    assert result["files"] >= 30


def test_review_package_reports_tampering_and_missing_files(tmp_path: Path) -> None:
    target = tmp_path / "evidence.json"
    target.write_bytes(b"true")
    manifest = manifest_for(target.name, b"true")
    assert verify_review_package(tmp_path, manifest)["files"] == 1
    target.write_bytes(b"fake")
    with pytest.raises(ReviewPackageError, match="SHA-256.*evidence.json"):
        verify_review_package(tmp_path, manifest)
    target.unlink()
    with pytest.raises(ReviewPackageError, match="Missing.*evidence.json"):
        verify_review_package(tmp_path, manifest)


@pytest.mark.parametrize(
    "unsafe", ["../private", "/absolute", "C:/private", "a//b", "a/../b", "a\\b"]
)
def test_review_package_rejects_noncanonical_paths(tmp_path: Path, unsafe: str) -> None:
    with pytest.raises(ReviewPackageError, match="Unsafe"):
        verify_review_package(tmp_path, manifest_for(unsafe, b"value"))


def test_review_package_rejects_duplicate_entries(tmp_path: Path) -> None:
    target = tmp_path / "evidence.json"
    target.write_bytes(b"true")
    manifest = manifest_for(target.name, b"true")
    manifest["files"] *= 2
    with pytest.raises(ReviewPackageError, match="Duplicate"):
        verify_review_package(tmp_path, manifest)

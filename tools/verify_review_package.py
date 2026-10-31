"""Verify the submitted documents, models and evidence using only Python's standard library."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "docs/review-package-manifest.json"
SHA256 = re.compile(r"[0-9a-f]{64}")


class ReviewPackageError(ValueError):
    """A review package is incomplete, changed or has an unsafe inventory."""


def _relative_path(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ReviewPackageError(f"Unsafe inventory path: {value!r}")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != value
        or value == "."
        or any(ord(char) < 32 for char in value)
    ):
        raise ReviewPackageError(f"Unsafe inventory path: {value!r}")
    return path


def verify_review_package(root: Path, manifest: dict | None = None) -> dict:
    """Check every listed regular file; do not import code or load trained models."""
    root = root.resolve(strict=True)
    if manifest is None:
        manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"schema", "files"}
        or manifest["schema"] != "gridguard.review_package.1"
        or not isinstance(manifest["files"], list)
        or not manifest["files"]
    ):
        raise ReviewPackageError("Invalid review package manifest")
    seen = set()
    total_bytes = 0
    for entry in manifest["files"]:
        if not isinstance(entry, dict) or set(entry) != {"path", "size_bytes", "sha256"}:
            raise ReviewPackageError("Invalid inventory entry")
        relative = _relative_path(entry["path"])
        key = relative.as_posix().casefold()
        if key in seen:
            raise ReviewPackageError(f"Duplicate inventory entry: {relative}")
        seen.add(key)
        if (
            type(entry["size_bytes"]) is not int
            or entry["size_bytes"] < 0
            or not isinstance(entry["sha256"], str)
            or SHA256.fullmatch(entry["sha256"]) is None
        ):
            raise ReviewPackageError(f"Invalid size or SHA-256: {relative}")
        path = root.joinpath(*relative.parts)
        cursor = root
        for part in relative.parts:
            cursor /= part
            if cursor.is_symlink() or (hasattr(cursor, "is_junction") and cursor.is_junction()):
                raise ReviewPackageError(f"Unsafe linked inventory path: {relative}")
        if not path.is_file():
            raise ReviewPackageError(f"Missing file: {relative}")
        if not path.resolve().is_relative_to(root):
            raise ReviewPackageError(f"Unsafe inventory path: {relative}")
        if path.stat().st_size != entry["size_bytes"]:
            raise ReviewPackageError(f"Size mismatch: {relative}")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != entry["sha256"]:
            raise ReviewPackageError(f"SHA-256 mismatch: {relative}")
        total_bytes += entry["size_bytes"]
    return {"passed": True, "files": len(seen), "size_bytes": total_bytes, "manifest": MANIFEST}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=ROOT, help="Project root (default: this checkout)"
    )
    args = parser.parse_args()
    try:
        result = verify_review_package(args.root)
    except (OSError, ValueError) as error:
        print(json.dumps({"passed": False, "error": str(error)}))
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from gridguard.benchmark_migration import migrate_frozen_source_benchmark

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def source_repository() -> Path:
    configured_path = os.environ.get("GRIDGUARD_SOURCE_REPOSITORY")
    if configured_path:
        repository = Path(configured_path).expanduser().resolve()
        if not (repository / ".git").exists():
            pytest.fail(f"GRIDGUARD_SOURCE_REPOSITORY is not a Git repository: {repository}")
        return repository

    sibling_repository = ROOT.parent / "aipower_rl"
    if (sibling_repository / ".git").exists():
        return sibling_repository
    pytest.skip(
        "trusted source archive unavailable; set GRIDGUARD_SOURCE_REPOSITORY "
        "to run source_archive tests"
    )


@pytest.mark.source_archive
def test_every_declared_source_lineage_matches_the_trusted_git_blob(
    source_repository: Path,
) -> None:
    manifest = json.loads(
        (ROOT / "data" / "evidence" / "source-manifest.json").read_text(encoding="utf-8")
    )

    for entry in manifest["lineage"]:
        object_name = f"{entry['source_revision']}:{entry['source_path']}"
        completed = subprocess.run(
            ["git", "-C", str(source_repository), "show", object_name],
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr.decode(errors="replace")
        assert len(completed.stdout) == entry["git_blob_size_bytes"], entry["id"]
        assert hashlib.sha256(completed.stdout).hexdigest() == entry["git_blob_sha256"], entry["id"]

        if entry["sha256"] != entry["git_blob_sha256"]:
            checkout_artifact = source_repository / entry["source_path"]
            checkout_bytes = checkout_artifact.read_bytes()
            assert len(checkout_bytes) == entry["size_bytes"], entry["id"]
            assert hashlib.sha256(checkout_bytes).hexdigest() == entry["sha256"], entry["id"]


@pytest.mark.source_archive
def test_frozen_benchmark_reproduces_from_the_exact_audited_source_bytes(
    source_repository: Path,
) -> None:
    migrated = migrate_frozen_source_benchmark(source_repository / "artifacts" / "benchmark.json")

    assert migrated == (ROOT / "data" / "evidence" / "frozen-benchmark.json").read_bytes()

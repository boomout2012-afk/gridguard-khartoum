from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPOSITORY_ROOT / "RELEASE_MANIFEST.json"
EXPECTED_TRACKED_PATHS = {
    "data/evidence/paired-benchmark.json",
    "data/evidence/source-manifest.json",
    "data/model/manifest.json",
    "data/model/ppo-policy.npz",
    "docs/source/graduation-project-proposal.pdf",
    "output/pdf/gridguard-khartoum-final-report.pdf",
    "output/presentation/gridguard-khartoum-defense.pptx",
    "src/gridguard/assets/map/khartoum-schematic.json",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_release_manifest_binds_every_tracked_release_artifact() -> None:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["release_id"] == "gridguard-khartoum-v1.0.0"
    assert payload["release_date"] == "2026-08-15"

    artifacts = payload["tracked_artifacts"]
    assert {entry["path"] for entry in artifacts} == EXPECTED_TRACKED_PATHS
    for entry in artifacts:
        assert set(entry) == {"path", "sha256", "size_bytes"}
        relative_path = Path(entry["path"])
        assert not relative_path.is_absolute()
        artifact_path = REPOSITORY_ROOT / relative_path
        assert artifact_path.is_file()
        assert artifact_path.stat().st_size == entry["size_bytes"]
        assert _sha256(artifact_path) == entry["sha256"]

    windows_build = payload["windows_build"]
    assert windows_build["tracked"] is False
    assert windows_build["frozen_smoke"] == "PASS"
    assert windows_build["no_gpu_behavior"] == "FAIL_CLOSED"
    assert re.fullmatch(r"[0-9a-f]{64}", windows_build["executable_sha256"])
    assert not Path(windows_build["executable_path"]).is_absolute()

    smoke_binding = windows_build["frozen_smoke_report"]
    assert set(smoke_binding) == {"path", "sha256", "size_bytes"}
    assert re.fullmatch(r"[0-9a-f]{64}", smoke_binding["sha256"])
    assert not Path(smoke_binding["path"]).is_absolute()

    executable = REPOSITORY_ROOT / windows_build["executable_path"]
    if executable.is_file():
        assert executable.stat().st_size == windows_build["executable_size_bytes"]
        assert _sha256(executable) == windows_build["executable_sha256"]

        distribution = executable.parent
        distribution_files = [path for path in distribution.rglob("*") if path.is_file()]
        assert len(distribution_files) == windows_build["distribution_file_count"]
        assert (
            sum(path.stat().st_size for path in distribution_files)
            == windows_build["distribution_size_bytes"]
        )

    smoke_report_path = REPOSITORY_ROOT / smoke_binding["path"]
    if smoke_report_path.is_file():
        assert smoke_report_path.stat().st_size == smoke_binding["size_bytes"]
        assert _sha256(smoke_report_path) == smoke_binding["sha256"]
        smoke = json.loads(smoke_report_path.read_text(encoding="utf-8"))
        assert smoke["status"] == "PASS"
        assert smoke["live_scenario"]["device"] == "cuda:0"
        assert smoke["live_scenario"]["frame_counts"] == {
            "Fixed Rotation": 24,
            "PPO": 24,
        }
        assert smoke["live_scenario"]["finite_core_values"] is True
        assert smoke["live_scenario"]["installed_in_live_page"] is True
        assert smoke["live_scenario"]["results_table_rows"] == 7

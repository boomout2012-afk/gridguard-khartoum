from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.export_training_telemetry import export_telemetry


def test_exported_telemetry_matches_frozen_diagnostic_hashes(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    outputs = export_telemetry(root, tmp_path)
    diagnostics = json.loads(
        (root / "data/evidence/focused-ppo-training-diagnostics.json").read_text()
    )
    assert len(outputs) == 5
    for source_name, expected in diagnostics["input_telemetry_sha256"].items():
        path = tmp_path / Path(source_name).name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


def test_export_refuses_to_overwrite_different_telemetry(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    target = tmp_path / "seed-1729.jsonl"
    target.write_text("unrelated data")
    with pytest.raises(FileExistsError):
        export_telemetry(root, tmp_path)
    assert target.read_text() == "unrelated data"

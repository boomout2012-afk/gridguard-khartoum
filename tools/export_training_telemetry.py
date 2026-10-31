"""Restore compact training telemetry from the frozen record, verifying every byte."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pinned_json(path: Path) -> dict:
    raw = path.read_bytes()
    expected = path.with_suffix(".sha256").read_text().split()[0]
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"Evidence checksum mismatch: {path.name}")
    return json.loads(raw)


def export_telemetry(root: Path, destination: Path) -> list[Path]:
    """Reconstruct logged rows, including their original Windows line endings."""
    training_path = root / "data/evidence/focused-ppo-training.json"
    training = pinned_json(training_path)
    diagnostics = pinned_json(root / "data/evidence/focused-ppo-training-diagnostics.json")
    if diagnostics["training_sha256"] != hashlib.sha256(training_path.read_bytes()).hexdigest():
        raise ValueError("Diagnostics and training identities differ")
    expected_hashes = {
        Path(name).name: digest for name, digest in diagnostics["input_telemetry_sha256"].items()
    }
    planned = []
    for run in training["runs"]:
        name = f"seed-{run['seed']}.jsonl"
        lines = [json.dumps(row, sort_keys=True, allow_nan=False) for row in run["telemetry"]]
        candidates = [(separator.join(lines) + separator).encode() for separator in ("\n", "\r\n")]
        matches = [
            raw for raw in candidates if hashlib.sha256(raw).hexdigest() == expected_hashes[name]
        ]
        if len(matches) != 1:
            raise ValueError(f"Cannot reproduce original telemetry bytes: {name}")
        target = destination / name
        if target.exists() and target.read_bytes() != matches[0]:
            raise FileExistsError(f"Refusing to replace different telemetry: {target}")
        planned.append((target, matches[0]))
    destination.mkdir(parents=True, exist_ok=True)
    for target, raw in planned:
        if not target.exists():
            target.write_bytes(raw)
    return [target for target, _ in planned]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "tmp/focused-ppo-monitor")
    args = parser.parse_args()
    for path in export_telemetry(ROOT, args.output.resolve()):
        print(path)


if __name__ == "__main__":
    main()

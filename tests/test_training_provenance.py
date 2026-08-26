from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.training_provenance import (
    TrainingProvenanceError,
    build_training_summary,
    load_training_provenance,
    parse_training_provenance,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROVENANCE_PATH = REPOSITORY_ROOT / "data" / "model" / "training-provenance.yaml"
MODEL_MANIFEST_PATH = REPOSITORY_ROOT / "data" / "model" / "manifest.json"
REVIEWER_ENTRY_POINTS = (
    REPOSITORY_ROOT / "README.md",
    REPOSITORY_ROOT / "SPEC.md",
    REPOSITORY_ROOT / "docs" / "DEVILS_ADVOCATE_AUDIT.md",
)
SOURCE_REVISION = "52f6dd82605c993c8094448caecec86faa289c0c"
SOURCE_HASHES = {
    "artifacts/training_manifest.json": (
        "3b866b6f470e116ff009022032059a1f91b509fadc54dbff6c5962a8f2236a59"
    ),
    "artifacts/training_metrics.jsonl": (
        "f48d2f6da29682e1c47330aa2b33b6bb4941f4e0e248f2c846436f09a4b89bc4"
    ),
    "experiments/s2-600-hidden512.json": (
        "619eb67aa253a35d7a7dc0efdbc6c743d7b0af6db4687575e7fade9dbcfe940b"
    ),
    "src/aipower_rl/training.py": (
        "886931d6c12b0ef45f4f17b89d134e27682c68a2dc4c2f90e2911e2dfec708fb"
    ),
    "src/aipower_rl/learning.py": (
        "18cf22b0b58584b2abef164ec3ea956fa2f61197336a67a7b9c1f4faefc4fac2"
    ),
    "src/aipower_rl/experiments.py": (
        "605e32626caece41508f37cbfac0cb061f7510118ba463cf3ff5843f92a9d753"
    ),
}


def _canonical_text() -> str:
    return PROVENANCE_PATH.read_text(encoding="utf-8")


def _replace_once(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1
    return text.replace(old, new, 1)


def test_training_provenance_exposes_exact_selected_training_contract() -> None:
    payload = load_training_provenance(
        PROVENANCE_PATH,
        model_manifest_path=MODEL_MANIFEST_PATH,
    )

    assert payload["schema"] == "gridguard.ppo_training_provenance"
    assert payload["schema_version"] == 1
    assert payload["model"]["controller"] == "PPO"
    assert payload["model"]["checkpoint"]["sha256"] == (
        "c6e5712f966b46ba18f4c69fb457b9e69f78887c0c5c0279e5ce034f40f9c665"
    )
    assert payload["model"]["actor"]["sha256"] == (
        "3e4a7ce44267dd1cf3bbde7e77abafb8f31fdea77ece63acda7be17fa3198ad5"
    )
    assert payload["model"]["network"]["hidden_layers"] == [512, 512]
    assert payload["behavior_cloning_initialization"]["optimizer"]["weight_decay"] == 0.01
    assert payload["behavior_cloning_initialization"]["optimizer_updates"] == 1600
    assert payload["ppo"]["total_episodes"] == 153_600
    assert payload["ppo"]["total_transitions"] == 3_686_400
    assert payload["ppo"]["planned_optimizer_updates"] == 7200
    assert payload["ppo"]["selected_checkpoint_optimizer_updates"] == 6120
    assert payload["validation_and_selection"]["selected_seed"] == 43
    assert payload["validation_and_selection"]["selected_iteration"] == 510
    assert len(payload["convergence"]["validation_series"]) == 120
    assert payload["convergence"]["validation_series"][-1] == {
        "iteration": 600,
        "training_objective_score": 23.50542449951172,
    }

    summary = build_training_summary(payload, provenance_path=PROVENANCE_PATH)
    assert summary["network"] == "32 -> 512 -> 512 -> 2187"
    assert summary["behavior_cloning_initialization"] == {
        "teacher_samples": 32768,
        "epochs": 100,
        "optimizer_updates": 1600,
    }
    assert summary["selection"]["validation_seeds"] == [70000, 70001, 70002]
    assert summary["training_provenance"]["path"] == "data/model/training-provenance.yaml"
    assert (
        summary["training_provenance"]["sha256"]
        == hashlib.sha256(PROVENANCE_PATH.read_bytes()).hexdigest()
    )


def test_reviewer_entry_points_link_the_canonical_training_parameter_record() -> None:
    for document_path in REVIEWER_ENTRY_POINTS:
        document = document_path.read_text(encoding="utf-8")
        assert "[data/model/training-provenance.yaml]" in document, document_path
        assert "canonical" in document.casefold(), document_path
        assert "strict" in document.casefold(), document_path

    readme = REVIEWER_ENTRY_POINTS[0].read_text(encoding="utf-8")
    spec = REVIEWER_ENTRY_POINTS[1].read_text(encoding="utf-8")
    assert "## Reviewer start here" in readme
    assert "## Reviewer start here" in spec


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            "schema: gridguard.ppo_training_provenance\n"
            "schema: gridguard.ppo_training_provenance\n",
            "duplicate",
        ),
        (
            "schema: &schema gridguard.ppo_training_provenance\ncopy: *schema\n",
            "anchor",
        ),
        ("schema: gridguard.ppo_training_provenance\n<<: {}\n", "merge"),
        (
            "schema: gridguard.ppo_training_provenance\n---\nschema_version: 1\n",
            "single document",
        ),
        ("schema: !!python/object:builtins.object {}\n", "tag"),
    ],
)
def test_strict_loader_rejects_ambiguous_or_unsafe_yaml(payload: str, message: str) -> None:
    with pytest.raises(TrainingProvenanceError, match=message):
        parse_training_provenance(payload)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("schema_version: 1", "schema_version: true", "schema_version"),
        ("learning_rate: 0.0001", "learning_rate: .nan", "finite"),
        (
            "total_transitions: 3686400",
            "total_transitions: 3686399",
            "total_transitions",
        ),
        (
            "path: artifacts/training_manifest.json",
            "path: ../training_manifest.json",
            "relative path",
        ),
    ],
)
def test_semantic_validator_rejects_invalid_types_arithmetic_and_paths(
    old: str,
    new: str,
    message: str,
) -> None:
    tampered = _replace_once(_canonical_text(), old, new)

    with pytest.raises(TrainingProvenanceError, match=message):
        parse_training_provenance(tampered)


def test_semantic_validator_rejects_unknown_fields() -> None:
    tampered = _canonical_text() + "unknown_review_field: true\n"

    with pytest.raises(TrainingProvenanceError, match="unknown fields"):
        parse_training_provenance(tampered)


def test_summary_cli_prints_only_compact_validated_json() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "tools/training_provenance.py",
            "--path",
            str(PROVENANCE_PATH),
            "--model-manifest",
            str(MODEL_MANIFEST_PATH),
            "--summary-json",
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    summary = json.loads(completed.stdout)
    assert summary["network"] == "32 -> 512 -> 512 -> 2187"
    assert "\n" not in completed.stdout.rstrip("\n")


def test_summary_cli_fails_closed_without_printing_json(tmp_path: Path) -> None:
    tampered = tmp_path / "training-provenance.yaml"
    tampered.write_text(
        _replace_once(_canonical_text(), "schema_version: 1", "schema_version: true"),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            "tools/training_provenance.py",
            "--path",
            str(tampered),
            "--model-manifest",
            str(MODEL_MANIFEST_PATH),
            "--summary-json",
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert completed.stdout == ""
    assert "schema_version" in completed.stderr


@pytest.mark.source_archive
def test_declared_training_sources_match_the_pinned_read_only_archive() -> None:
    source_repository = REPOSITORY_ROOT.parent / "aipower_rl"
    if not (source_repository / ".git").exists():
        pytest.skip("trusted read-only source archive is unavailable")
    payload = load_training_provenance(PROVENANCE_PATH)
    declared = {binding["path"]: binding["sha256"] for binding in payload["source_bindings"]}

    assert declared == SOURCE_HASHES
    source_blobs: dict[str, bytes] = {}
    for source_path, expected_hash in SOURCE_HASHES.items():
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(source_repository),
                "show",
                f"{SOURCE_REVISION}:{source_path}",
            ],
            check=True,
            capture_output=True,
        )
        assert hashlib.sha256(completed.stdout).hexdigest() == expected_hash
        source_blobs[source_path] = completed.stdout

    training_manifest = json.loads(source_blobs["artifacts/training_manifest.json"])
    experiment_summary = json.loads(source_blobs["experiments/s2-600-hidden512.json"])
    metric_rows = [
        json.loads(line)
        for line in source_blobs["artifacts/training_metrics.jsonl"].decode("utf-8").splitlines()
    ]
    selected_validation_series = [
        {
            "iteration": row["iteration"],
            "training_objective_score": row["validation_episode_reward"],
        }
        for row in metric_rows
        if row["stage"] == "ppo" and "validation_episode_reward" in row
    ]

    assert payload["software_hardware"]["gpu"] == training_manifest["device_name"]
    assert payload["software_hardware"]["pytorch"] == training_manifest["torch_version"]
    assert payload["software_hardware"]["cuda"] == training_manifest["cuda_version"]
    assert payload["ppo"]["iterations"] == training_manifest["training"]["ppo_iterations"]
    assert payload["ppo"]["learning_rate"] == training_manifest["training"]["ppo"]["learning_rate"]
    assert (
        payload["validation_and_selection"]["candidate_training_seeds"]
        == (experiment_summary["seeds"])
    )
    assert (
        payload["validation_and_selection"]["selected_seed"] == experiment_summary["selected_seed"]
    )
    assert payload["convergence"]["validation_series"] == selected_validation_series

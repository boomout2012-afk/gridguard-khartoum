from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from gridguard.provenance import ProvenanceError, verify_project_source_manifest

ROOT = Path(__file__).parents[1]


def test_repository_source_manifest_verifies_committed_artifacts() -> None:
    verified = verify_project_source_manifest(
        ROOT / "data" / "evidence" / "source-manifest.json",
        repository_root=ROOT,
    )

    assert [artifact.artifact_id for artifact in verified] == [
        "accepted-proposal",
        "frozen-proposal-benchmark",
        "ppo-inference-artifact",
        "ppo-inference-manifest",
        "ppo-training-provenance",
        "paired-proposal-benchmark",
        "robustness-supplement",
        "robustness-trust-anchor-source",
    ]


def test_project_source_manifest_trust_anchor_rejects_a_self_consistent_rewrite(
    tmp_path: Path,
) -> None:
    original = json.loads(
        (ROOT / "data" / "evidence" / "source-manifest.json").read_text(encoding="utf-8")
    )
    original["artifacts"] = []
    rewritten = tmp_path / "source-manifest.json"
    rewritten.write_text(json.dumps(original), encoding="utf-8")

    with pytest.raises(ProvenanceError, match="trust anchor"):
        verify_project_source_manifest(rewritten, repository_root=ROOT)


def test_ppo_manifest_is_bound_to_source_lineage_and_exported_artifact() -> None:
    source_manifest = json.loads(
        (ROOT / "data" / "evidence" / "source-manifest.json").read_text(encoding="utf-8")
    )
    model_manifest = json.loads(
        (ROOT / "data" / "model" / "manifest.json").read_text(encoding="utf-8")
    )

    artifacts = {item["id"]: item for item in source_manifest["artifacts"]}
    lineage = {item["id"]: item for item in source_manifest["lineage"]}
    assert model_manifest["artifact"]["sha256"] == artifacts["ppo-inference-artifact"]["sha256"]
    assert (
        model_manifest["source"]["checkpoint_sha256"] == lineage["source-ppo-checkpoint"]["sha256"]
    )
    assert (
        model_manifest["source"]["source_revision"]
        == lineage["source-ppo-checkpoint"]["source_revision"]
    )
    training_binding = model_manifest["training_provenance"]
    assert artifacts["ppo-training-provenance"]["path"] == ("data/model/training-provenance.yaml")
    assert training_binding["filename"] == "training-provenance.yaml"
    assert training_binding["sha256"] == artifacts["ppo-training-provenance"]["sha256"]
    training_provenance_path = ROOT / "data" / "model" / training_binding["filename"]
    assert training_binding["size_bytes"] == training_provenance_path.stat().st_size


def test_source_manifest_covers_every_migrated_behavior_dependency() -> None:
    source_manifest = json.loads(
        (ROOT / "data" / "evidence" / "source-manifest.json").read_text(encoding="utf-8")
    )

    assert {item["id"]: item["source_path"] for item in source_manifest["lineage"]} == {
        "source-benchmark": "artifacts/benchmark.json",
        "source-ppo-checkpoint": "artifacts/checkpoints/best.pt",
        "source-simulation-config": "src/aipower_rl/config.py",
        "source-controllers": "src/aipower_rl/controllers.py",
        "source-scenarios": "src/aipower_rl/scenarios.py",
        "source-simulator": "src/aipower_rl/simulator.py",
        "source-dispatch-and-reward": "src/aipower_rl/dynamics.py",
        "source-policy-network": "src/aipower_rl/models.py",
        "source-evaluation": "src/aipower_rl/evaluation.py",
        "source-training-manifest": "artifacts/training_manifest.json",
        "source-training-metrics": "artifacts/training_metrics.jsonl",
        "source-candidate-experiment-summary": "experiments/s2-600-hidden512.json",
        "source-training-orchestration": "src/aipower_rl/training.py",
        "source-learning-algorithm": "src/aipower_rl/learning.py",
        "source-experiment-selection": "src/aipower_rl/experiments.py",
    }
    assert {item["source_revision"] for item in source_manifest["lineage"]} == {
        "52f6dd82605c993c8094448caecec86faa289c0c"
    }


def test_robustness_source_bundle_binds_the_canonical_claim_contract() -> None:
    evidence = json.loads(
        (ROOT / "data" / "evidence" / "robustness-supplement.json").read_text(encoding="utf-8")
    )
    bindings = {item["path"]: item for item in evidence["provenance"]["implementation_sources"]}
    source = ROOT / "src" / "gridguard" / "claim_contract.py"

    assert bindings["src/gridguard/claim_contract.py"] == {
        "path": "src/gridguard/claim_contract.py",
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "size_bytes": source.stat().st_size,
    }


def test_frozen_benchmark_has_unambiguous_counts_and_controller_names() -> None:
    evidence = json.loads(
        (ROOT / "data" / "evidence" / "frozen-benchmark.json").read_text(encoding="utf-8")
    )

    assert evidence["protocol"] == {
        "controller_evaluation_count": 576,
        "controllers": ["Fixed Rotation", "PPO"],
        "episode_count_per_cell": 32,
        "evidence_granularity": "batch-cell summaries",
        "horizon_hours": 24,
        "matched_scenario_count": 288,
        "seeds": [50001, 50002, 50003],
        "severities": ["mild", "moderate", "severe"],
        "stored_matched_cell_count": 9,
    }
    assert len(evidence["cells"]) == 18
    assert set(evidence["aggregate"]) == {"Fixed Rotation", "PPO"}


@pytest.mark.parametrize(
    ("metric", "expected_fixed", "expected_ppo"),
    [
        ("episode_reward", 13.969324111938477, 23.512135399712456),
        ("critical_service_ratio", 0.4262974626488156, 0.9971638056966994),
        ("utilization", 0.7122341394424438, 0.9765213396814134),
        ("fairness", 0.8775761524836222, 0.9691991872257657),
        ("served_mwh", 4551.691569010417, 6560.189181857639),
        ("curtailed_mwh", 6142.390950520833, 4133.893283420139),
        ("switching_rate", 0.2500000298023224, 0.21887401905324724),
        ("power_balance_violation_rate", 0.0, 0.0),
    ],
)
def test_frozen_headline_values_match_accepted_source(
    metric: str,
    expected_fixed: float,
    expected_ppo: float,
) -> None:
    evidence = json.loads(
        (ROOT / "data" / "evidence" / "frozen-benchmark.json").read_text(encoding="utf-8")
    )

    assert evidence["aggregate"]["Fixed Rotation"][metric] == expected_fixed
    assert evidence["aggregate"]["PPO"][metric] == expected_ppo


def test_frozen_evidence_states_aggregate_only_limitations() -> None:
    evidence = json.loads(
        (ROOT / "data" / "evidence" / "frozen-benchmark.json").read_text(encoding="utf-8")
    )
    limitations = " ".join(evidence["limitations"])

    assert "Synthetic simulation" in limitations
    assert "not per-episode rows" in limitations
    assert "Confidence intervals" in limitations

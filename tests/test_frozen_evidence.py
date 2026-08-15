from __future__ import annotations

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
        "paired-proposal-benchmark",
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


def test_source_manifest_covers_every_migrated_behavior_dependency() -> None:
    source_manifest = json.loads(
        (ROOT / "data" / "evidence" / "source-manifest.json").read_text(encoding="utf-8")
    )

    assert {item["id"] for item in source_manifest["lineage"]} == {
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
    assert {item["source_revision"] for item in source_manifest["lineage"]} == {
        "52f6dd82605c993c8094448caecec86faa289c0c"
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

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import pytest
import torch

from gridguard import constrained_ppo
from gridguard.ppo_experiment import (
    CONFIG,
    TEST_SEEDS,
    TRAINING_SEEDS,
    VALIDATION_SEEDS,
    read_json,
    training_batches,
    write_json,
)
from gridguard.ppo_study import select_checkpoint
from tools import train_constrained_ppo as runner


def _summary(reward: float) -> dict:
    row = {"episode_reward": reward, "fairness": 0.97, "utilization": 0.98}
    return {
        "overall": row.copy(),
        "by_shortage": {name: row.copy() for name in ("mild", "moderate", "severe")},
        "maxima": {
            "critical_outage_locality_hours": 0.0,
            "critical_interruption_locality_hours": 0.0,
            "power_balance_violation_rate": 0.0,
        },
    }


def test_training_bc_validation_and_final_scenario_streams_are_disjoint():
    training = []
    for run in range(len(TRAINING_SEEDS)):
        totals = [0, 0, 0]
        for iteration in range(CONFIG["iterations"]):
            seed, batches = training_batches(run, iteration)
            training.append(seed)
            assert sum(batches) == CONFIG["episodes_per_iteration"]
            totals = [total + count for total, count in zip(totals, batches, strict=True)]
        assert len(set(totals)) == 1
    bc_seeds = {3_000_000 + run * 10_000 for run in range(len(TRAINING_SEEDS))}
    sets = [set(training), bc_seeds, set(VALIDATION_SEEDS), set(TEST_SEEDS)]
    assert len(training) == len(sets[0]) == 3000
    for index, split in enumerate(sets):
        for other in sets[index + 1 :]:
            assert split.isdisjoint(other)


def test_state_copy_is_independent_from_later_optimizer_mutations():
    model = torch.nn.Linear(2, 1)
    original = runner.state_copy(model)
    expected = {name: tensor.clone() for name, tensor in original.items()}
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(100)
    for name, tensor in original.items():
        torch.testing.assert_close(tensor, expected[name], atol=0, rtol=0)
        assert tensor.data_ptr() != model.state_dict()[name].data_ptr()
        assert not tensor.requires_grad


class _Generator:
    def __init__(self, *, device):
        self.device = device

    def manual_seed(self, seed):
        self.seed = seed
        return self


def _patch_training(monkeypatch, tmp_path, rewards):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(runner, "CONFIG", {**CONFIG, "iterations": 2, "checkpoint_interval": 1})
    monkeypatch.setattr(runner.torch, "Generator", _Generator)
    monkeypatch.setattr(runner.torch, "manual_seed", lambda seed: None)
    monkeypatch.setattr(runner.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(
        runner, "ConstrainedActorCritic", lambda **kwargs: torch.nn.Linear(1, 1, bias=False)
    )
    seen_seeds = []
    monkeypatch.setattr(runner, "scenarios_for_seed", lambda seed, sizes: seen_seeds.append(seed))
    monkeypatch.setattr(runner, "collect_teacher_dataset", lambda *args, **kwargs: None)
    streams = {"bc": [], "policy": [], "update": []}

    def clone(model, dataset, **kwargs):
        streams["bc"].append(kwargs["generator"])
        with torch.no_grad():
            model.weight.zero_()
        return {"loss": torch.tensor(0.0)}

    monkeypatch.setattr(runner, "behavior_clone", clone)
    monkeypatch.setattr(runner, "make_optimizer", lambda model, config: None)

    def rollout(model, scenarios, **kwargs):
        streams["policy"].append(kwargs["generator"])
        return None

    def update(model, optimizer, rollout, config, **kwargs):
        streams["update"].append(kwargs["generator"])
        with torch.no_grad():
            model.weight.add_(1)
        return {"loss": torch.tensor(0.0)}

    monkeypatch.setattr(runner, "collect_rollout", rollout)
    monkeypatch.setattr(runner, "ppo_update", update)
    monkeypatch.setattr(
        runner,
        "evaluate_metrics",
        lambda controller, scenarios: _summary(rewards[int(controller.model.weight.item())]),
    )
    monkeypatch.setattr(runner, "summarize", lambda value: value)
    saved = []
    monkeypatch.setattr(
        runner,
        "save_model",
        lambda model, path: saved.append((path, model.weight.item())) or "a" * 64,
    )
    monkeypatch.setattr(runner, "source_hashes", lambda root: {"source": "stable"})
    return saved, seen_seeds, streams


@pytest.mark.parametrize(
    "rewards, expected_iteration", [([24.0, 23.6, 23.7], 0), ([23.6, 24.0, 23.7], 1)]
)
def test_runner_saves_selected_snapshot_instead_of_final_training_state(
    monkeypatch, tmp_path, rewards, expected_iteration
):
    saved, seeds, streams = _patch_training(monkeypatch, tmp_path, rewards)
    result = runner.train_run(0, object(), _summary(23.5), {"source": "stable"})
    assert result["selected"]["iteration"] == expected_iteration
    assert saved[0][1] == 0
    assert saved[1][1] == expected_iteration
    assert not set(seeds) & set(TEST_SEEDS)
    assert not set(seeds) & set(VALIDATION_SEEDS)
    assert streams["policy"][0] is not streams["update"][0]
    assert len({streams[name][0].seed for name in streams}) == 3


def test_training_rejects_sources_changed_during_run(monkeypatch, tmp_path):
    _patch_training(monkeypatch, tmp_path, [23.6, 23.7, 23.8])
    with pytest.raises(ValueError, match="sources changed"):
        runner.train_run(0, object(), _summary(23.5), {"source": "previous"})


def test_json_evidence_refuses_overwrite_and_detects_changed_bytes(tmp_path):
    path = tmp_path / "evidence.json"
    payload = {"seed": 1729, "selected": 50}
    digest = write_json(path, payload)
    assert read_json(path, digest) == payload
    assert read_json(path) == payload
    with pytest.raises(FileExistsError):
        write_json(path, {"seed": 1729, "selected": 100})
    path.write_text('{"seed":1729,"selected":100}', encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        read_json(path)


@dataclass
class _Runtime:
    device_name: str = "no GPU used by orchestration test"


def _patch_resume(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(runner, "RECORD", tmp_path / "training.json")
    monkeypatch.setattr(runner, "TRAINING_SEEDS", (1729,))
    monkeypatch.setattr(runner, "run_gpu_preflight", _Runtime)
    monkeypatch.setattr(runner, "source_hashes", lambda root: {"source": "stable"})
    monkeypatch.setattr(runner, "evaluation_scenarios", lambda seeds: None)
    monkeypatch.setattr(runner, "evaluate_metrics", lambda controller, scenarios: _summary(23.5))
    monkeypatch.setattr(runner, "summarize", lambda value: value)
    monkeypatch.setattr(constrained_ppo, "load_model", lambda *args, **kwargs: None)
    checkpoints = [
        {"iteration": iteration, "summary": _summary(23.6)}
        for iteration in range(0, CONFIG["iterations"] + 1, CONFIG["checkpoint_interval"])
    ]
    record = {
        "source_sha256": {"source": "stable"},
        "config": deepcopy(CONFIG),
        "ppo_config": runner.asdict(runner.PPOTrainingConfig()),
        "run": {
            "seed": 1729,
            "transitions": 600 * 256 * 24,
            "bc_states": 455 * 3 * 24,
            "selected": select_checkpoint(checkpoints, _summary(23.5)),
            "checkpoints": checkpoints,
            "model": {"path": "selected.npz", "sha256": "a" * 64},
            "behavior_cloning_model": {"path": "bc.npz", "sha256": "b" * 64},
        },
    }
    path = runner.MODEL_DIR / "seed-1729-record.json"
    path.parent.mkdir(parents=True)
    path.touch()
    monkeypatch.setattr(runner, "read_json", lambda path: record)
    return record


def test_resume_rejects_copied_record_from_another_training_seed(monkeypatch, tmp_path):
    record = _patch_resume(monkeypatch, tmp_path)
    record["run"]["seed"] = 2718
    with pytest.raises(ValueError, match="identity"):
        runner.train()


def test_resume_rejects_incomplete_checkpoint_schedule(monkeypatch, tmp_path):
    record = _patch_resume(monkeypatch, tmp_path)
    record["run"]["checkpoints"].pop()
    with pytest.raises(ValueError, match="budget"):
        runner.train()


def test_resume_accepts_complete_matching_record_and_keeps_validation_selection(
    monkeypatch, tmp_path
):
    record = _patch_resume(monkeypatch, tmp_path)
    runner.train()
    payload = read_json(runner.RECORD)
    assert payload["runs"] == [record["run"]]
    assert payload["deployment_seed"] == 1729


def test_resume_rejects_a_changed_source_binding(monkeypatch, tmp_path):
    record = _patch_resume(monkeypatch, tmp_path)
    record["source_sha256"]["source"] = "changed"
    with pytest.raises(ValueError, match="frozen source"):
        runner.train()

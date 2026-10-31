from __future__ import annotations

import copy
import io
import json

import pytest
import torch

from gridguard.allocation_study import METRIC_NAMES, SHORTAGES
from gridguard.focused_training import (
    CONFIG,
    TEST_SEEDS,
    TRAINING_SEEDS,
    VALIDATION_SEEDS,
    checkpoint_eligibility,
    select_checkpoint,
    training_batches,
    verify_checkpoint_selection,
)


def _summary(*, reward=23.4, fairness=0.95, utilization=0.97, interruption=0) -> dict:
    metrics = dict.fromkeys(METRIC_NAMES, 0.0)
    metrics.update(episode_reward=reward, fairness=fairness, utilization=utilization)
    maxima = metrics.copy()
    maxima.update(
        critical_outage_locality_hours=interruption,
        critical_interruption_locality_hours=interruption,
    )
    return {
        "overall": metrics,
        "by_shortage": {level: metrics.copy() for level in SHORTAGES},
        "maxima": maxima,
    }


def _reference() -> dict:
    return _summary(reward=14, fairness=0.88, utilization=0.71, interruption=105)


def test_fixed_rotation_interruptions_do_not_disqualify_a_safe_ppo_candidate() -> None:
    assert checkpoint_eligibility(_summary(), _reference()) == {"eligible": True, "reasons": []}


def test_checkpoint_selection_uses_highest_reward_and_earliest_tie() -> None:
    records = [
        {"iteration": 0, "summary": _summary(reward=23.1)},
        {"iteration": 100, "summary": _summary(reward=23.5)},
        {"iteration": 600, "summary": _summary(reward=23.5)},
        {"iteration": 3000, "summary": _summary(reward=23.4)},
    ]
    selected = select_checkpoint(records, _reference())
    assert selected["iteration"] == 100
    assert selected["eligible"] and not selected["fallback"]
    verify_checkpoint_selection(records, _reference(), selected)
    selected["iteration"] = 600
    with pytest.raises(ValueError, match="selection"):
        verify_checkpoint_selection(records, _reference(), selected)


def test_unsafe_high_reward_cannot_replace_safe_checkpoint() -> None:
    records = [
        {"iteration": 0, "summary": _summary(reward=23.1)},
        {"iteration": 100, "summary": _summary(reward=23.9, interruption=1)},
    ]
    assert select_checkpoint(records, _reference())["iteration"] == 0


@pytest.mark.parametrize("metric", ["fairness", "utilization"])
def test_service_guardrail_applies_within_each_shortage(metric: str) -> None:
    candidate = _summary()
    candidate["by_shortage"]["severe"][metric] = 0.1
    eligible = checkpoint_eligibility(candidate, _reference())
    assert not eligible["eligible"]
    assert any("severe" in reason and metric in reason for reason in eligible["reasons"])


def test_no_eligible_checkpoint_is_explicit_fallback() -> None:
    records = [
        {"iteration": 0, "summary": _summary(reward=23.2, fairness=0.2)},
        {"iteration": 100, "summary": _summary(reward=23.3, fairness=0.2)},
    ]
    selected = select_checkpoint(records, _reference())
    assert selected["iteration"] == 100
    assert selected["fallback"] and not selected["eligible"]


def test_nonfinite_summary_and_duplicate_iteration_fail() -> None:
    invalid = _summary()
    invalid["overall"]["episode_reward"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        checkpoint_eligibility(invalid, _reference())
    one = {"iteration": 0, "summary": _summary()}
    with pytest.raises(ValueError, match="unique"):
        select_checkpoint([one, copy.deepcopy(one)], _reference())


def test_training_budget_seed_splits_and_rotating_counts() -> None:
    assert (
        CONFIG["iterations"] * CONFIG["episodes_per_iteration"] * 24 * len(TRAINING_SEEDS)
        == 92_160_000
    )
    totals = [0, 0, 0]
    scenario_seeds = set()
    for iteration in range(CONFIG["iterations"]):
        seed, batches = training_batches(0, iteration)
        assert seed == 4_000_000 + iteration
        assert sum(batches) == 256
        scenario_seeds.add(seed)
        totals = [a + b for a, b in zip(totals, batches, strict=True)]
    assert len(set(totals)) == 1
    assert not scenario_seeds.intersection(VALIDATION_SEEDS + TEST_SEEDS)
    assert not set(VALIDATION_SEEDS).intersection(TEST_SEEDS)
    assert training_batches(4, 2999)[0] == 4_042_999


@pytest.mark.parametrize("run, iteration", [(-1, 0), (5, 0), (0, -1), (0, 3000), (True, 0)])
def test_out_of_protocol_training_indices_fail(run: int, iteration: int) -> None:
    with pytest.raises(ValueError):
        training_batches(run, iteration)


def test_training_keeps_selected_and_budget_reference_states_separate(
    tmp_path, monkeypatch
) -> None:
    from tools import train_focused_ppo as runner

    class TinyModel:
        def __init__(self, **kwargs):
            self.weight = torch.tensor(0.0)

        def state_dict(self):
            return {"weight": self.weight}

        def load_state_dict(self, state):
            self.weight = state["weight"].clone()

    class Generator:
        def __init__(self, **kwargs):
            pass

        def manual_seed(self, seed):
            return self

    config = {
        **CONFIG,
        "iterations": 3,
        "checkpoint_interval": 1,
        "telemetry_interval": 1,
        "budget_reference_iteration": 2,
    }
    progress = []
    saved = {}

    def update(model, optimizer, rollout, config, **kwargs):
        model.weight += 1
        progress.append(kwargs["progress"])
        return {"episode_reward": model.weight + 22}

    def save(model, path):
        saved[path.name] = model.weight.item()
        return "a" * 64

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(runner, "CONFIG", config)
    monkeypatch.setattr(runner, "ConstrainedActorCritic", TinyModel)
    monkeypatch.setattr(runner.torch, "Generator", Generator)
    monkeypatch.setattr(runner.torch, "manual_seed", lambda _: None)
    monkeypatch.setattr(runner.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(runner, "scenarios_for_seed", lambda *args: None)
    monkeypatch.setattr(runner, "collect_teacher_dataset", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        runner, "behavior_clone", lambda *args, **kwargs: {"loss": torch.tensor(1.0)}
    )
    monkeypatch.setattr(runner, "collect_rollout", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "make_optimizer", lambda *args: None)
    monkeypatch.setattr(runner, "ppo_update", update)
    monkeypatch.setattr(runner, "save_model", save)
    monkeypatch.setattr(runner, "summarize", lambda value: value)
    monkeypatch.setattr(
        runner,
        "evaluate_metrics",
        lambda controller, _: _summary(
            reward={0: 22.0, 1: 23.5, 2: 23.4, 3: 23.3}[int(controller.model.weight)]
        ),
    )
    monkeypatch.setattr(runner, "source_hashes", lambda _: {"source": "bound"})
    monitor = io.StringIO()
    run = runner._train_run(0, None, _reference(), {"source": "bound"}, monitor)
    assert progress == [0.0, 0.5, 1.0]
    assert saved["seed-1729-behavior-cloning.npz"] == 0
    assert saved["seed-1729-iteration-600.npz"] == 2
    assert saved["seed-1729-selected.npz"] == 1
    assert run["selected"]["iteration"] == 1
    rows = [json.loads(line) for line in monitor.getvalue().splitlines()]
    assert [row["iteration"] for row in rows] == [0, 1, 2, 3]
    assert rows == run["telemetry"]

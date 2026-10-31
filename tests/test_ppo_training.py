from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from gridguard.constrained import FairnessController
from gridguard.constrained_ppo import ConstrainedActorCritic, ConstrainedPPOController
from gridguard.controllers import DecisionContext
from gridguard.domain import ControllerName, ShortageLevel
from gridguard.ppo_training import (
    PPOTrainingConfig,
    behavior_clone,
    collect_rollout,
    collect_teacher_dataset,
    make_optimizer,
    ppo_update,
)
from gridguard.scenarios import generate_scenarios
from gridguard.simulation import simulate


@dataclass
class ReplayController:
    tiers: torch.Tensor
    name = ControllerName.PPO

    def act(self, context: DecisionContext) -> torch.Tensor:
        return self.tiers[context.hour]


@pytest.mark.parametrize("level", list(ShortageLevel))
def test_rollout_replays_exact_simulator_physics_and_reward(level: ShortageLevel) -> None:
    scenario = generate_scenarios(2, 781301, level, device="cpu")
    model = ConstrainedActorCritic(device="cpu")
    rollout = collect_rollout(model, scenario, generator=torch.Generator().manual_seed(27))
    replay = simulate(scenario, ReplayController(rollout.service_tiers))
    torch.testing.assert_close(rollout.requested.movedim(0, 1), replay.requested, rtol=0, atol=0)
    torch.testing.assert_close(rollout.served.movedim(0, 1), replay.served, rtol=0, atol=0)
    torch.testing.assert_close(rollout.rewards.movedim(0, 1), replay.reward, rtol=0, atol=0)
    assert rollout.features.shape == (24, 2, 46)
    assert rollout.masks.shape == (24, 2, 128)
    assert rollout.masks.gather(-1, rollout.actions.unsqueeze(-1)).all()
    assert not rollout.features.requires_grad
    assert not rollout.old_log_probs.requires_grad
    assert (rollout.served[..., :3] == scenario.demand.movedim(0, 1)[..., :3]).all()


def test_deterministic_rollout_matches_public_controller() -> None:
    scenario = generate_scenarios(2, 781302, ShortageLevel.SEVERE, device="cpu")
    model = ConstrainedActorCritic(device="cpu")
    rollout = collect_rollout(model, scenario, deterministic=True)
    replay = simulate(scenario, ConstrainedPPOController(model))
    torch.testing.assert_close(rollout.service_tiers.movedim(0, 1), replay.service_tiers)
    torch.testing.assert_close(rollout.rewards.movedim(0, 1), replay.reward)


def test_same_seed_reproduces_sampled_rollout() -> None:
    scenario = generate_scenarios(3, 781303, ShortageLevel.MODERATE, device="cpu")
    model = ConstrainedActorCritic(device="cpu")
    a = collect_rollout(model, scenario, generator=torch.Generator().manual_seed(310))
    b = collect_rollout(model, scenario, generator=torch.Generator().manual_seed(310))
    torch.testing.assert_close(a.actions, b.actions, rtol=0, atol=0)
    torch.testing.assert_close(a.rewards, b.rewards, rtol=0, atol=0)


def test_combined_rollout_preserves_severity_order() -> None:
    scenarios = tuple(generate_scenarios(2, 781304, level, device="cpu") for level in ShortageLevel)
    model = ConstrainedActorCritic(device="cpu")
    combined = collect_rollout(model, scenarios, deterministic=True)
    singles = [collect_rollout(model, scenario, deterministic=True) for scenario in scenarios]
    torch.testing.assert_close(combined.actions, torch.cat([r.actions for r in singles], dim=1))
    torch.testing.assert_close(combined.rewards, torch.cat([r.rewards for r in singles], dim=1))


def test_teacher_dataset_matches_teacher_and_terminal_returns() -> None:
    scenario = generate_scenarios(2, 781305, ShortageLevel.MODERATE, device="cpu")
    teacher = FairnessController(reward_tolerance=0.005)
    dataset = collect_teacher_dataset(scenario, teacher=teacher)
    result = simulate(scenario, teacher)
    actions = dataset.actions.reshape(24, 2)
    binary = 2 ** torch.arange(7)
    expected = ((result.service_tiers - 1) * binary).sum(-1).movedim(0, 1)
    torch.testing.assert_close(actions, expected, rtol=0, atol=0)
    returns = dataset.returns.reshape(24, 2)
    torch.testing.assert_close(returns[-1], result.reward[:, -1])
    torch.testing.assert_close(returns[0], result.reward.sum(1))
    assert dataset.masks.gather(-1, dataset.actions[:, None]).all()


def test_ppo_update_has_finite_gradients_and_changes_parameters() -> None:
    torch.manual_seed(781306)
    scenario = generate_scenarios(3, 781306, ShortageLevel.MODERATE, device="cpu")
    model = ConstrainedActorCritic(device="cpu")
    generator = torch.Generator().manual_seed(301)
    rollout = collect_rollout(model, scenario, generator=generator)
    initial = [p.detach().clone() for p in model.parameters()]
    config = PPOTrainingConfig(epochs=2, minibatch_size=31)
    optimizer = make_optimizer(model, config)
    metrics = ppo_update(model, optimizer, rollout, config, generator=generator, progress=0.5)
    assert all(torch.isfinite(value).all() for value in metrics.values())
    assert any(not torch.equal(a, b) for a, b in zip(initial, model.parameters(), strict=True))
    assert optimizer.param_groups[0]["lr"] == pytest.approx(config.learning_rate * 0.5)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_behavior_cloning_reduces_imitation_loss() -> None:
    torch.manual_seed(781307)
    scenario = generate_scenarios(4, 781307, ShortageLevel.SEVERE, device="cpu")
    dataset = collect_teacher_dataset(scenario, teacher=FairnessController(0.005))
    model = ConstrainedActorCritic(device="cpu")
    with torch.no_grad():
        before = -model.get_action_and_value(
            dataset.features, dataset.masks, action=dataset.actions
        )[1].mean()
    stats = behavior_clone(
        model,
        dataset,
        epochs=10,
        minibatch_size=96,
        learning_rate=0.003,
        generator=torch.Generator().manual_seed(31),
    )
    with torch.no_grad():
        after = -model.get_action_and_value(
            dataset.features, dataset.masks, action=dataset.actions
        )[1].mean()
    assert after < before
    assert all(torch.isfinite(value).all() for value in stats.values())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"epochs": 0},
        {"minibatch_size": 0},
        {"learning_rate": float("nan")},
        {"clip_ratio": -0.2},
        {"gamma": 1.1},
        {"gae_lambda": -0.1},
        {"max_grad_norm": 0},
        {"entropy_start": -1},
    ],
)
def test_training_config_rejects_invalid_parameters(kwargs: dict) -> None:
    with pytest.raises((ValueError, TypeError)):
        PPOTrainingConfig(**kwargs)


@pytest.mark.gpu
def test_cuda_rollout_update_and_teacher_stay_on_device() -> None:
    device = "cuda:0"
    model = ConstrainedActorCritic(device=device)
    scenario = generate_scenarios(4, 781308, ShortageLevel.SEVERE, device=device)
    generator = torch.Generator(device=device).manual_seed(201)
    rollout = collect_rollout(model, scenario, generator=generator)
    config = PPOTrainingConfig(epochs=1, minibatch_size=48)
    stats = ppo_update(model, make_optimizer(model, config), rollout, config, generator=generator)
    dataset = collect_teacher_dataset(scenario, teacher=FairnessController(0.005))
    assert all(
        tensor.device.type == "cuda"
        for tensor in (
            rollout.features,
            rollout.masks,
            rollout.rewards,
            dataset.features,
            dataset.returns,
            *stats.values(),
        )
    )
    replay = simulate(scenario, ReplayController(rollout.service_tiers))
    torch.testing.assert_close(rollout.rewards.movedim(0, 1), replay.reward, rtol=0, atol=0)

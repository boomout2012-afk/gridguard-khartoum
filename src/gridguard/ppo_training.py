"""Device-resident constrained PPO rollouts, imitation initialization and updates.

The rollout uses exactly the existing dispatch physics and canonical reward.
Current and cumulative state enter the policy; future scenario samples do not.
Only complete 24-hour episodes are accepted, so terminal bootstrap value is zero.
No optimizer state or training data is persisted by this module.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from gridguard.constrained_ppo import (
    ConstrainedActorCritic,
    build_training_observation,
    clipped_ppo_loss,
    critical_action_mask,
    critical_action_tiers,
    generalized_advantage_estimate,
)
from gridguard.controllers import Controller, DecisionContext
from gridguard.domain import HORIZON_HOURS, LOCALITIES, Scenario, canonical_step_reward
from gridguard.simulation import _build_observation, _dispatch_service_tiers, _locality_peaks


@dataclass(frozen=True, slots=True)
class PPOTrainingConfig:
    """Fixed update settings; scenario sampling and stopping live in the runner."""

    epochs: int = 4
    minibatch_size: int = 1024
    learning_rate: float = 3e-4
    adam_epsilon: float = 1e-5
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_start: float = 0.01
    entropy_end: float = 0.001
    max_grad_norm: float = 0.5

    def __post_init__(self) -> None:
        for name in ("epochs", "minibatch_size"):
            _positive_integer(name, getattr(self, name))
        for name in ("learning_rate", "adam_epsilon", "max_grad_norm", "clip_ratio"):
            _finite_number(name, getattr(self, name), positive=True)
        if self.clip_ratio >= 1:
            raise ValueError("clip_ratio must be below one")
        for name in ("value_coefficient", "entropy_start", "entropy_end"):
            _finite_number(name, getattr(self, name))
        for name in ("gamma", "gae_lambda"):
            _unit_interval(name, getattr(self, name))


@dataclass(frozen=True, slots=True, eq=False)
class TrainingRollout:
    """Complete episodes in [time, batch, ...] order, all on one device."""

    features: torch.Tensor
    masks: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    service_tiers: torch.Tensor
    requested: torch.Tensor
    served: torch.Tensor


@dataclass(frozen=True, slots=True, eq=False)
class TeacherDataset:
    """Flattened teacher states with feasible labels and undiscounted returns."""

    features: torch.Tensor
    masks: torch.Tensor
    actions: torch.Tensor
    returns: torch.Tensor


def _positive_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer")
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _finite_number(name: str, value: object, *, positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'non-negative'}")


def _unit_interval(name: str, value: object) -> None:
    _finite_number(name, value)
    if value > 1:
        raise ValueError(f"{name} must be between zero and one")


def _scenario_tensors(
    scenarios: Scenario | Sequence[Scenario],
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = (scenarios,) if isinstance(scenarios, Scenario) else tuple(scenarios)
    if not batch or any(not isinstance(scenario, Scenario) for scenario in batch):
        raise TypeError("scenarios must contain at least one Scenario")
    first = batch[0].demand
    for scenario in batch:
        scenario.validate_shortage_invariant()
        if scenario.demand.device != first.device or scenario.demand.dtype != first.dtype:
            raise ValueError("all scenario tensors must use the same device and dtype")
    if len(batch) == 1:
        return batch[0].demand, batch[0].supply
    return torch.cat([s.demand for s in batch]), torch.cat([s.supply for s in batch])


@torch.no_grad()
def _collect_trace(
    scenarios: Scenario | Sequence[Scenario],
    *,
    model: ConstrainedActorCritic | None = None,
    teacher: Controller | None = None,
    generator: torch.Generator | None = None,
    deterministic: bool = False,
) -> TrainingRollout:
    demand_trace, supply_trace = _scenario_tensors(scenarios)
    batch_size = demand_trace.shape[0]
    device = demand_trace.device
    if model is not None:
        parameter = next(model.parameters())
        if parameter.device != device or parameter.dtype != demand_trace.dtype:
            raise ValueError("model and scenarios must use the same device and dtype")
    cumulative_demand = demand_trace.new_zeros((batch_size, len(LOCALITIES)))
    cumulative_served = torch.zeros_like(cumulative_demand)
    previous_tier = torch.zeros((batch_size, len(LOCALITIES)), dtype=torch.long, device=device)
    locality_peaks = _locality_peaks(demand_trace)
    candidate_tiers = critical_action_tiers(device)
    binary_divisors = 2 ** torch.arange(len(LOCALITIES), dtype=torch.long, device=device)
    traces: dict[str, list[torch.Tensor]] = {
        name: [] for name in TrainingRollout.__dataclass_fields__
    }

    for hour in range(HORIZON_HOURS):
        demand, supply = demand_trace[:, hour], supply_trace[:, hour]
        context = DecisionContext(
            hour=hour,
            observation=_build_observation(
                hour=hour,
                demand=demand,
                supply=supply,
                cumulative_demand=cumulative_demand,
                cumulative_served=cumulative_served,
                previous_tier=previous_tier,
                locality_peaks=locality_peaks,
            ),
            _validate_finite=False,
            current_demand_mw=demand,
            current_supply_mw=supply,
            cumulative_demand_mwh=cumulative_demand,
            cumulative_served_mwh=cumulative_served,
            previous_service_tiers=previous_tier,
        )
        features = build_training_observation(context, validate=False)
        masks = critical_action_mask(context, validate=False)
        if model is not None:
            actions, log_probs, _, values = model.get_action_and_value(
                features,
                masks,
                deterministic=deterministic,
                generator=generator,
                validate=False,
            )
            tiers = candidate_tiers[actions]
        else:
            if teacher is None:
                raise TypeError("a model or teacher must be supplied")
            tiers = teacher.act(context)
            if bool(((tiers < 1) | (tiers > 2)).any().item()):
                raise ValueError("teacher must choose Critical Only or Full Service")
            actions = ((tiers - 1) * binary_divisors).sum(-1)
            log_probs = supply.new_zeros(batch_size)
            values = supply.new_zeros(batch_size)
        requested, served = _dispatch_service_tiers(demand, supply, tiers)
        next_demand = cumulative_demand + demand.sum(-1)
        next_served = cumulative_served + served.sum(-1)
        rewards = canonical_step_reward(
            hour=hour,
            demand=demand,
            supply=supply,
            requested=requested,
            served=served,
            tiers=tiers,
            previous_tier=previous_tier,
            next_cumulative_demand=next_demand,
            next_cumulative_served=next_served,
        )
        values_by_name = (
            features,
            masks,
            actions,
            log_probs,
            values,
            rewards,
            tiers,
            requested,
            served,
        )
        for name, value in zip(traces, values_by_name, strict=True):
            traces[name].append(value)
        cumulative_demand, cumulative_served, previous_tier = next_demand, next_served, tiers

    result = TrainingRollout(**{name: torch.stack(trace) for name, trace in traces.items()})
    if not bool(result.masks.gather(-1, result.actions.unsqueeze(-1)).all().item()):
        raise ValueError("rollout selected an action excluded by the critical-service mask")
    return result


def collect_rollout(
    model: ConstrainedActorCritic,
    scenarios: Scenario | Sequence[Scenario],
    *,
    generator: torch.Generator | None = None,
    deterministic: bool = False,
) -> TrainingRollout:
    """Collect complete on-policy episodes, preserving model training-mode state."""
    was_training = model.training
    model.eval()
    try:
        return _collect_trace(
            scenarios, model=model, generator=generator, deterministic=deterministic
        )
    finally:
        model.train(was_training)


@torch.no_grad()
def collect_teacher_dataset(
    scenarios: Scenario | Sequence[Scenario], *, teacher: Controller
) -> TeacherDataset:
    """Collect imitation examples exclusively from caller-supplied training scenarios."""
    rollout = _collect_trace(scenarios, teacher=teacher)
    # With gamma=lambda=1 and zero baseline, GAE is exactly a terminal Monte Carlo return.
    _, returns = generalized_advantage_estimate(
        rollout.rewards, torch.zeros_like(rollout.rewards), gamma=1.0, gae_lambda=1.0
    )
    return TeacherDataset(
        features=rollout.features.flatten(0, 1),
        masks=rollout.masks.flatten(0, 1),
        actions=rollout.actions.flatten(0, 1),
        returns=returns.flatten(0, 1),
    )


def make_optimizer(model: ConstrainedActorCritic, config: PPOTrainingConfig) -> torch.optim.Adam:
    """Create a fresh optimizer; BC optimizer moments are deliberately discarded."""
    return torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=config.adam_epsilon)


def _permutations(
    count: int,
    epochs: int,
    minibatch_size: int,
    device: torch.device,
    generator: torch.Generator | None,
):
    for _ in range(epochs):
        indices = torch.randperm(count, device=device, generator=generator)
        for start in range(0, count, minibatch_size):
            yield indices[start : start + minibatch_size]


def ppo_update(
    model: ConstrainedActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: TrainingRollout,
    config: PPOTrainingConfig,
    *,
    generator: torch.Generator | None = None,
    progress: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Apply clipped PPO with stored feasibility masks and full-episode GAE.

    ``progress`` is the prespecified fraction of the training budget consumed.
    Learning rate decreases linearly to zero; entropy decreases to its final
    coefficient. Returned monitoring statistics remain scalar device tensors.
    """
    _unit_interval("progress", progress)
    model.train()
    for group in optimizer.param_groups:
        group["lr"] = config.learning_rate * (1.0 - progress)
    entropy_coefficient = config.entropy_start + progress * (
        config.entropy_end - config.entropy_start
    )
    advantages, returns = generalized_advantage_estimate(
        rollout.rewards, rollout.values, gamma=config.gamma, gae_lambda=config.gae_lambda
    )
    flat_advantages = advantages.flatten().detach()
    flat_advantages = (flat_advantages - flat_advantages.mean()) / (
        flat_advantages.std(unbiased=False) + 1e-8
    )
    flat_returns = returns.flatten().detach()
    features = rollout.features.flatten(0, 1)
    masks = rollout.masks.flatten(0, 1)
    actions = rollout.actions.flatten()
    old_log_probs = rollout.old_log_probs.flatten().detach()
    stats = features.new_zeros(7)
    count = features.shape[0]

    for indices in _permutations(
        count, config.epochs, config.minibatch_size, features.device, generator
    ):
        _, log_probs, entropy, values = model.get_action_and_value(
            features[indices], masks[indices], action=actions[indices], validate=False
        )
        loss = clipped_ppo_loss(
            log_probs,
            old_log_probs[indices],
            flat_advantages[indices],
            values,
            flat_returns[indices],
            entropy,
            clip_ratio=config.clip_ratio,
            value_coefficient=config.value_coefficient,
            entropy_coefficient=entropy_coefficient,
            validate=False,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        stats += (
            torch.stack(
                (
                    loss.total.detach(),
                    loss.policy.detach(),
                    loss.value.detach(),
                    loss.entropy.detach(),
                    loss.approx_kl.detach(),
                    loss.clip_fraction.detach(),
                    grad_norm.detach(),
                )
            )
            * indices.numel()
        )
    stats /= count * config.epochs
    if not bool(torch.isfinite(stats).all().item()):
        raise FloatingPointError("non-finite PPO training statistic")
    names = (
        "loss",
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "grad_norm",
    )
    metrics = dict(zip(names, stats.unbind(), strict=True))
    metrics.update(
        episode_reward=rollout.rewards.sum(0).mean(),
        learning_rate=features.new_tensor(config.learning_rate * (1.0 - progress)),
        entropy_coefficient=features.new_tensor(entropy_coefficient),
    )
    return metrics


def behavior_clone(
    model: ConstrainedActorCritic,
    dataset: TeacherDataset,
    *,
    epochs: int = 10,
    minibatch_size: int = 1024,
    learning_rate: float = 3e-4,
    value_coefficient: float = 0.5,
    max_grad_norm: float = 0.5,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Fit teacher actions and terminal returns before a separately measured PPO run.

    Actor loss is masked categorical cross entropy. Critic loss is half the mean
    squared return error, weighted by ``value_coefficient``. Save this model as
    the BC-only comparator before calling ``ppo_update`` with a fresh optimizer.
    """
    config = PPOTrainingConfig(
        epochs=epochs,
        minibatch_size=minibatch_size,
        learning_rate=learning_rate,
        value_coefficient=value_coefficient,
        max_grad_norm=max_grad_norm,
    )
    count = dataset.features.shape[0]
    _positive_integer("dataset size", count)
    if dataset.features.shape != (count, 46) or dataset.masks.shape != (count, 128):
        raise ValueError("teacher dataset must have features [N,46] and masks [N,128]")
    if dataset.actions.shape != (count,) or dataset.returns.shape != (count,):
        raise ValueError("teacher actions and returns must have shape [N]")
    if not bool(dataset.masks.gather(-1, dataset.actions[:, None]).all().item()):
        raise ValueError("teacher labels must satisfy the stored critical-service mask")
    model.train()
    optimizer = make_optimizer(model, config)
    stats = dataset.features.new_zeros(3)
    for indices in _permutations(count, epochs, minibatch_size, dataset.features.device, generator):
        _, log_probs, _, values = model.get_action_and_value(
            dataset.features[indices],
            dataset.masks[indices],
            action=dataset.actions[indices],
            validate=False,
        )
        imitation_loss = -log_probs.mean()
        value_loss = 0.5 * (values - dataset.returns[indices]).square().mean()
        loss = imitation_loss + value_coefficient * value_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        stats += (
            torch.stack((loss.detach(), imitation_loss.detach(), value_loss.detach()))
            * indices.numel()
        )
    stats /= count * epochs
    if not bool(torch.isfinite(stats).all().item()):
        raise FloatingPointError("non-finite behavior-cloning statistic")
    return dict(zip(("loss", "imitation_loss", "value_loss"), stats.unbind(), strict=True))

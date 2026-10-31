"""Fresh actor-critic learning over the same critical-first feasible action set.

The actor observes current causal information only. Its 46 inputs contain the
existing 32 features followed by seven cumulative demand and seven cumulative
served energy magnitudes, each normalized by 24 times the fixed locality peak.
Actions use ascending little-endian binary order: Critical Only or Full Service.
If critical demand exceeds supply, only the explicit all-critical fallback is
eligible; this does not assert that critical service is physically feasible.

Training helpers use ordinary no-grad tensors, not inference-mode tensors.
``validate=False`` is reserved for already-validated, internally generated
rollouts; public inference and artifact loading validate their boundaries.
"""

from __future__ import annotations

import hashlib
import io
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from gridguard.constrained import CONSTRAINED_ACTION_COUNT
from gridguard.controllers import OBSERVATION_DIM, DecisionContext
from gridguard.domain import (
    CRITICAL_SECTOR_COUNT,
    HORIZON_HOURS,
    LOCALITIES,
    LOCALITY_DEMAND_WEIGHTS,
    SECTORS,
    ControllerName,
)
from gridguard.optimization import _dispatch_state
from gridguard.scenarios import SYSTEM_PEAK_MW

TRAINING_OBSERVATION_DIM = OBSERVATION_DIM + 2 * len(LOCALITIES)
HIDDEN_SIZE = 256
_FORMAT_VERSION = 1
_FORMAT_KEY = "_format_version"


def critical_action_tiers(device: torch.device | str | None = None) -> torch.Tensor:
    """Return [128, 7] tiers in the existing constrained candidate order."""
    indices = torch.arange(CONSTRAINED_ACTION_COUNT, dtype=torch.long, device=device)
    divisors = 2 ** torch.arange(len(LOCALITIES), dtype=torch.long, device=device)
    return 1 + (indices[:, None] // divisors) % 2


def _context_state(context: DecisionContext, *, validate: bool) -> tuple[torch.Tensor, ...]:
    if validate:
        state = _dispatch_state(context)
        if not bool(torch.isfinite(context.observation).all().item()):
            raise ValueError("training observation must contain only finite values")
        return state
    return (
        context.current_demand_mw,
        context.current_supply_mw,
        context.cumulative_demand_mwh,
        context.cumulative_served_mwh,
        context.previous_service_tiers,
    )


@torch.no_grad()
def build_training_observation(context: DecisionContext, *, validate: bool = True) -> torch.Tensor:
    """Append exact cumulative magnitudes, with no future scenario information."""
    demand, _, cumulative_demand, cumulative_served, _ = _context_state(context, validate=validate)
    peaks = demand.new_tensor(LOCALITY_DEMAND_WEIGHTS) * (
        SYSTEM_PEAK_MW / sum(LOCALITY_DEMAND_WEIGHTS)
    )
    scale = HORIZON_HOURS * peaks
    return torch.cat(
        (context.observation, cumulative_demand / scale, cumulative_served / scale), dim=-1
    )


@torch.no_grad()
def critical_action_mask(context: DecisionContext, *, validate: bool = True) -> torch.Tensor:
    """Return physically feasible choices, or only the explicit index-zero fallback.

    Preserve the dispatcher's multiplication and final 35-sector reduction order
    exactly. Summing critical and noncritical totals separately changes floating
    point rounding at the supply boundary and can admit an infeasible action.
    """
    demand, supply, *_ = _context_state(context, validate=validate)
    tiers = critical_action_tiers(demand.device)
    critical = torch.arange(len(SECTORS), device=demand.device) < CRITICAL_SECTOR_COUNT
    requests = critical[None, None, :] | (tiers == 2)[..., None]
    requested_total = (demand[:, None, :, :] * requests[None, ...]).sum(dim=(-1, -2))
    feasible = requested_total <= supply[:, None]
    fallback = torch.arange(CONSTRAINED_ACTION_COUNT, device=demand.device)[None, :] == 0
    return torch.where(feasible.any(dim=1, keepdim=True), feasible, fallback)


def masked_categorical(
    logits: torch.Tensor, mask: torch.Tensor, *, validate: bool = True
) -> Categorical:
    """A categorical distribution assigning exactly zero mass to excluded actions."""
    if logits.ndim != 2 or mask.shape != logits.shape:
        raise ValueError("logits and mask must have matching [batch, actions] shape")
    if not logits.is_floating_point() or mask.dtype != torch.bool:
        raise TypeError("logits must be floating and mask must be boolean")
    if logits.device != mask.device:
        raise ValueError("logits and mask must share the same device")
    if validate and (
        not bool(torch.isfinite(logits).all().item()) or not bool(mask.any(dim=-1).all().item())
    ):
        raise ValueError("logits must be finite and every mask must admit an action")
    return Categorical(logits=logits.masked_fill(~mask, -torch.inf), validate_args=False)


class ConstrainedActorCritic(nn.Module):
    """Separate two-layer tanh actor and critic with orthogonal initialization."""

    def __init__(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        *,
        initialize: bool = True,
    ) -> None:
        super().__init__()
        if dtype not in (torch.float32, torch.float64):
            raise TypeError("actor-critic dtype must be float32 or float64")
        kwargs = {"device": device, "dtype": dtype}
        self.actor = nn.Sequential(
            nn.Linear(TRAINING_OBSERVATION_DIM, HIDDEN_SIZE, **kwargs),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, **kwargs),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, CONSTRAINED_ACTION_COUNT, **kwargs),
        )
        self.critic = nn.Sequential(
            nn.Linear(TRAINING_OBSERVATION_DIM, HIDDEN_SIZE, **kwargs),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, **kwargs),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, 1, **kwargs),
        )
        self.register_buffer("action_tiers", critical_action_tiers(device), persistent=False)
        if initialize:
            for network, head_gain in ((self.actor, 0.01), (self.critic, 1.0)):
                for layer in network:
                    if isinstance(layer, nn.Linear):
                        gain = head_gain if layer is network[-1] else math.sqrt(2)
                        nn.init.orthogonal_(layer.weight, gain=gain)
                        nn.init.zeros_(layer.bias)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return raw logits [batch,128] and state values [batch]."""
        if observations.ndim != 2 or observations.shape[1] != TRAINING_OBSERVATION_DIM:
            raise ValueError("training observations must have shape [batch, 46]")
        parameter = next(self.parameters())
        if observations.device != parameter.device or observations.dtype != parameter.dtype:
            raise ValueError("training observations must share model device and dtype")
        return self.actor(observations), self.critic(observations).squeeze(-1)

    def get_action_and_value(
        self,
        observations: torch.Tensor,
        mask: torch.Tensor,
        action: torch.Tensor | None = None,
        *,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
        validate: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample/evaluate a masked action with its unchanged rollout log probability."""
        logits, values = self(observations)
        distribution = masked_categorical(logits, mask, validate=validate)
        if action is None:
            action = (
                distribution.logits.argmax(dim=-1)
                if deterministic
                else torch.multinomial(distribution.probs, 1, generator=generator).squeeze(-1)
            )
        elif validate:
            if action.shape != values.shape or action.dtype != torch.long:
                raise ValueError("action must be a long tensor with shape [batch]")
            if action.device != logits.device:
                raise ValueError("action must remain on the policy device")
            if not bool(((action >= 0) & (action < CONSTRAINED_ACTION_COUNT)).all().item()):
                raise ValueError("action index is outside the eligible action range")
            if not bool(mask.gather(1, action[:, None]).all().item()):
                raise ValueError("action must be eligible under its stored rollout mask")
        return action, distribution.log_prob(action), distribution.entropy(), values


@dataclass(frozen=True, slots=True)
class ConstrainedPPOController:
    """Use deterministic masked PPO decisions with exact critical-service constraints."""

    model: ConstrainedActorCritic
    name: ClassVar[ControllerName] = ControllerName.PPO

    @torch.inference_mode()
    def act(self, context: DecisionContext) -> torch.Tensor:
        features = build_training_observation(context)
        mask = critical_action_mask(context)
        action, *_ = self.model.get_action_and_value(features, mask, deterministic=True)
        return self.model.action_tiers[action]


def _unit_interval(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number in [0, 1]")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1]")


@torch.no_grad()
def generalized_advantage_estimate(
    rewards: torch.Tensor,
    values: torch.Tensor,
    *,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
    validate: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE for complete episodes [time,batch], with zero terminal bootstrap.

    Default gamma=1 matches the reported undiscounted 24-hour reward. This helper
    must not be used on truncated rollouts without their bootstrap value.
    """
    _unit_interval("gamma", gamma)
    _unit_interval("gae_lambda", gae_lambda)
    if rewards.ndim != 2 or rewards.shape != values.shape or min(rewards.shape) < 1:
        raise ValueError("rewards and values must have matching nonempty [time, batch] shape")
    if rewards.device != values.device or rewards.dtype != values.dtype:
        raise ValueError("rewards and values must share device and dtype")
    if not rewards.is_floating_point():
        raise TypeError("rewards and values must be floating point")
    if validate and not bool((torch.isfinite(rewards).all() & torch.isfinite(values).all()).item()):
        raise ValueError("rewards and values must be finite")
    advantages = torch.empty_like(rewards)
    continuation = torch.zeros_like(rewards[0])
    next_value = torch.zeros_like(values[0])
    for hour in range(rewards.shape[0] - 1, -1, -1):
        delta = rewards[hour] + gamma * next_value - values[hour]
        continuation = delta + gamma * gae_lambda * continuation
        advantages[hour] = continuation
        next_value = values[hour]
    return advantages, advantages + values


@dataclass(frozen=True, slots=True)
class PPOLoss:
    """Differentiable PPO losses and detached diagnostics on their original device."""

    total: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor
    approx_kl: torch.Tensor
    clip_fraction: torch.Tensor


def clipped_ppo_loss(
    new_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    new_values: torch.Tensor,
    returns: torch.Tensor,
    entropy: torch.Tensor,
    *,
    clip_ratio: float = 0.2,
    value_coefficient: float = 0.5,
    entropy_coefficient: float = 0.01,
    validate: bool = True,
) -> PPOLoss:
    """Clipped policy surrogate plus half-MSE value loss and entropy bonus.

    Advantages are supplied in their desired normalization. Old log probabilities,
    advantages and return targets are detached to prevent updating rollout data.
    """
    _unit_interval("clip_ratio", clip_ratio)
    for name, coefficient in (
        ("value_coefficient", value_coefficient),
        ("entropy_coefficient", entropy_coefficient),
    ):
        if not math.isfinite(coefficient) or coefficient < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    vectors = (new_log_prob, old_log_prob, advantages, new_values, returns, entropy)
    if new_log_prob.ndim != 1 or new_log_prob.numel() < 1:
        raise ValueError("PPO inputs must be nonempty vectors")
    if any(t.shape != new_log_prob.shape for t in vectors):
        raise ValueError("PPO inputs must have identical shapes")
    if any(t.device != new_log_prob.device or t.dtype != new_log_prob.dtype for t in vectors):
        raise ValueError("PPO inputs must share device and dtype")
    if validate and not bool(torch.stack([torch.isfinite(t).all() for t in vectors]).all().item()):
        raise ValueError("PPO inputs must be finite")
    log_ratio = new_log_prob - old_log_prob.detach()
    ratio = log_ratio.exp()
    advantage = advantages.detach()
    policy_loss = -torch.minimum(
        ratio * advantage, ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * advantage
    ).mean()
    value_loss = 0.5 * (new_values - returns.detach()).square().mean()
    mean_entropy = entropy.mean()
    total = policy_loss + value_coefficient * value_loss - entropy_coefficient * mean_entropy
    return PPOLoss(
        total=total,
        policy=policy_loss,
        value=value_loss,
        entropy=mean_entropy,
        approx_kl=((ratio.detach() - 1) - log_ratio.detach()).mean(),
        clip_fraction=((ratio.detach() - 1).abs() > clip_ratio).to(new_log_prob.dtype).mean(),
    )


def save_model(model: ConstrainedActorCritic, path: str | Path) -> str:
    """Atomically publish FP32 weights-only NPZ without overwriting existing files.

    All model parameters are packed for one final GPU-to-host artifact transfer.
    No optimizer state, pickled objects or executable payloads are serialized.
    Return the SHA-256 of the committed bytes for the experiment manifest.
    """
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(destination)
    state = model.state_dict()
    if any(value.dtype != torch.float32 for value in state.values()):
        raise ValueError("checkpoint tensors must use float32")
    flat = torch.cat([value.detach().reshape(-1) for value in state.values()]).cpu().numpy()
    if not np.isfinite(flat).all():
        raise ValueError("checkpoint tensors must contain only finite values")
    arrays = {_FORMAT_KEY: np.array([_FORMAT_VERSION], dtype=np.int64)}
    offset = 0
    for key, value in state.items():
        count = value.numel()
        arrays[key] = flat[offset : offset + count].reshape(tuple(value.shape))
        offset += count
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, suffix=".npz", delete=False
        ) as out:
            temporary_path = Path(out.name)
            np.savez_compressed(out, **arrays)
            out.flush()
            os.fsync(out.fileno())
        # A same-volume hard link publishes complete bytes and fails if the name exists.
        os.link(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return hashlib.sha256(destination.read_bytes()).hexdigest()


def load_model(
    path: str | Path,
    *,
    device: torch.device | str = "cuda:0",
    expected_sha256: str | None = None,
    freeze: bool = True,
) -> ConstrainedActorCritic:
    """Load validated, closed-schema numeric weights with optional provenance hash."""
    source = Path(path)
    payload = source.read_bytes()
    if expected_sha256 is not None:
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise ValueError("expected SHA-256 must be 64 lowercase hexadecimal characters")
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise ValueError("model SHA-256 does not match the expected artifact")
    model = ConstrainedActorCritic(device=device, initialize=False)
    expected = model.state_dict()
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        if len(archive.files) != len(expected) + 1 or set(archive.files) != set(expected) | {
            _FORMAT_KEY
        }:
            raise ValueError("model archive contains missing, duplicate or unexpected tensors")
        version = archive[_FORMAT_KEY]
        if version.dtype != np.int64 or version.shape != (1,) or version[0] != _FORMAT_VERSION:
            raise ValueError("model archive has an unsupported format version")
        tensors = {}
        for key, reference in expected.items():
            value = archive[key]
            if value.dtype != np.float32 or value.shape != tuple(reference.shape):
                raise ValueError(f"model tensor {key} has an incorrect dtype or shape")
            if not np.isfinite(value).all():
                raise ValueError(f"model tensor {key} must be finite")
            tensors[key] = torch.from_numpy(value.copy()).to(device)
    model.load_state_dict(tensors, strict=True)
    model.requires_grad_(not freeze)
    model.train(not freeze)
    return model

"""Frozen PPO action preferences combined with critical-first feasible dispatch.

These are postprocessing experiments with the existing, unchanged PPO actor.
They do not train a new policy, learn a continuation value, or claim that logits
are estimates of future reward. The hybrid admits feasible actions whose current
canonical reward is within a declared tolerance of the best feasible action,
then ranks those actions by the frozen actor's logits. The explicit feasible-PPO
ablation removes the reward cutoff while retaining the same critical constraint.

Both controllers use current demand, supply and past service only. If supply
cannot cover all critical demand, the shared allocator requests Critical Only
everywhere and the existing dispatch clips proportionally. That fallback is
recorded as critical shortfall by the experiment; it is not claimed feasible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import nn

from gridguard.constrained import evaluate_constrained_candidates, select_candidates
from gridguard.controllers import DecisionContext
from gridguard.domain import ControllerName
from gridguard.policy import ACTION_COUNT, OBSERVATION_DIM


def _validate_frozen_policy(policy: nn.Module) -> None:
    if not isinstance(policy, nn.Module):
        raise TypeError("policy must be a frozen torch.nn.Module returning PPO action logits")
    if any(module.training for module in policy.modules()):
        raise ValueError("policy and its submodules must already be in eval mode")
    if any(parameter.requires_grad for parameter in policy.parameters()):
        raise ValueError("policy parameters must already be frozen")


def _validate_context_for_policy(policy: nn.Module, context: DecisionContext) -> None:
    _validate_frozen_policy(policy)
    if not isinstance(context, DecisionContext):
        raise TypeError("PPO postprocessing requires a DecisionContext")
    observation = context.observation
    if not isinstance(observation, torch.Tensor):
        raise TypeError("observation must be a torch.Tensor")
    if observation.ndim != 2 or observation.shape[0] < 1 or observation.shape[1] != OBSERVATION_DIM:
        raise ValueError("observation must have shape [batch, 32] with a nonempty batch")
    if not observation.is_floating_point():
        raise TypeError("observation must use a floating-point dtype")
    state = (*policy.parameters(), *policy.buffers())
    if any(tensor.device != observation.device for tensor in state):
        raise ValueError("policy state and observation must use the same device")
    if any(tensor.is_floating_point() and tensor.dtype != observation.dtype for tensor in state):
        raise ValueError("policy floating-point state and observation must use the same dtype")
    if not bool(torch.isfinite(observation).all().item()):
        raise ValueError("observation must contain only finite values")


def _candidate_preferences(
    policy: nn.Module, context: DecisionContext, action_ids: torch.Tensor
) -> torch.Tensor:
    logits = policy(context.observation)
    if not isinstance(logits, torch.Tensor):
        raise TypeError("policy logits must be a torch.Tensor")
    if logits.shape != (context.observation.shape[0], ACTION_COUNT):
        raise ValueError(f"policy logits must have shape [batch, {ACTION_COUNT}]")
    if not logits.is_floating_point():
        raise TypeError("policy logits must use a floating-point dtype")
    if logits.device != context.observation.device:
        raise ValueError("policy logits and observation must use the same device")
    if logits.dtype != context.observation.dtype:
        raise ValueError("policy logits and observation must use the same dtype")
    if not bool(torch.isfinite(logits).all().item()):
        raise ValueError("policy logits must contain only finite values")
    return logits[:, action_ids]


@dataclass(frozen=True, slots=True)
class FeasiblePPOController:
    """Rank all critical-feasible actions by the existing frozen PPO actor.

    Equal logits resolve by immediate canonical reward and then lowest original
    joint action ID. CPU fake actors support unit tests; CUDA actors stay on CUDA.
    The simulation enum remains PPO for compatibility; the experiment must label
    this ablation separately from the original PPO controller in its results.
    """

    policy: nn.Module
    name: ClassVar[ControllerName] = ControllerName.PPO

    def __post_init__(self) -> None:
        _validate_frozen_policy(self.policy)

    @torch.inference_mode()
    def act(self, context: DecisionContext) -> torch.Tensor:
        _validate_context_for_policy(self.policy, context)
        candidates = evaluate_constrained_candidates(context)
        return select_candidates(
            candidates,
            preference=_candidate_preferences(self.policy, context, candidates.action_ids),
            reward_tolerance=None,
        )


@dataclass(frozen=True, slots=True)
class PPOHybridController:
    """Use frozen PPO preferences within an explicit immediate-reward allowance.

    reward_tolerance is in normalized canonical reward units per hour. Zero
    returns exactly the constrained optimizer, including its action-ID tie rule,
    without evaluating policy logits. Positive tolerances rank eligible actions
    by logits, then canonical reward, then lowest action ID. The allowance limits
    immediate reward loss only; it does not guarantee improved daily outcomes.
    """

    policy: nn.Module
    reward_tolerance: float = 0.01
    name: ClassVar[ControllerName] = ControllerName.PPO

    def __post_init__(self) -> None:
        _validate_frozen_policy(self.policy)
        if isinstance(self.reward_tolerance, bool) or not isinstance(
            self.reward_tolerance, (int, float)
        ):
            raise TypeError("reward_tolerance must be a finite non-negative number")
        if not math.isfinite(self.reward_tolerance) or self.reward_tolerance < 0:
            raise ValueError("reward_tolerance must be a finite non-negative number")

    @torch.inference_mode()
    def act(self, context: DecisionContext) -> torch.Tensor:
        _validate_context_for_policy(self.policy, context)
        candidates = evaluate_constrained_candidates(context)
        if self.reward_tolerance == 0:
            return select_candidates(candidates)
        return select_candidates(
            candidates,
            preference=_candidate_preferences(self.policy, context, candidates.action_ids),
            reward_tolerance=self.reward_tolerance,
        )

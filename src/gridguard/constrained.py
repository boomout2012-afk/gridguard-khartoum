"""Critical-first dispatch candidates and deterministic fairness scheduling.

Every candidate requests Critical Only or Full Service in each locality. A
candidate is physically feasible only when its unscaled total request does not
exceed current supply. Scoring uses the unchanged simulator and canonical reward.
When critical demand itself exceeds supply, no such action is feasible: the
explicit fallback requests Critical Only everywhere and the existing dispatcher
shares the shortage proportionally. That fallback is eligible but is never
reported as feasible, and it cannot guarantee uninterrupted critical service.

These controllers are new research comparators. Their legacy simulator name is
Optimization; evaluation must use separate explicit labels for each method.
Candidate enumeration, scoring, reductions and selection remain on the input
device. Small CPU contexts are supported for independent validation tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import torch

from gridguard.controllers import OBSERVATION_DIM, DecisionContext
from gridguard.domain import LOCALITIES, NUMERIC_EPSILON, ControllerName, canonical_step_reward
from gridguard.optimization import _dispatch_state
from gridguard.simulation import _dispatch_service_tiers

CONSTRAINED_ACTION_COUNT = 2 ** len(LOCALITIES)


@dataclass(frozen=True, slots=True, eq=False)
class ConstrainedCandidates:
    """Scored candidates in ascending original little-endian base-three action ID.

    ``tiers`` is [128, 7] and ``action_ids`` is [128]. ``rewards`` and
    ``feasible`` are [batch, 128]; ``service_ratios`` is [batch, 128, 7].
    ``critical_shortfall`` is [batch] and implies no physically feasible action.
    """

    tiers: torch.Tensor
    action_ids: torch.Tensor
    rewards: torch.Tensor
    feasible: torch.Tensor
    service_ratios: torch.Tensor
    critical_shortfall: torch.Tensor


def _validate_reward_tolerance(value: float | None, *, allow_none: bool = True) -> None:
    if value is None and allow_none:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("reward_tolerance must be a finite non-negative number")
    if not math.isfinite(value) or value < 0:
        raise ValueError("reward_tolerance must be a finite non-negative number")


@torch.inference_mode()
def evaluate_constrained_candidates(context: DecisionContext) -> ConstrainedCandidates:
    """Validate the causal state and score all 128 critical-first assignments."""
    if not isinstance(context, DecisionContext):
        raise TypeError("constrained dispatch requires a DecisionContext")
    observation = context.observation
    if (
        not isinstance(observation, torch.Tensor)
        or observation.ndim != 2
        or observation.shape[0] < 1
        or observation.shape[1] != OBSERVATION_DIM
        or not observation.is_floating_point()
    ):
        raise ValueError("constrained observation must be a floating tensor with shape [batch, 32]")
    if not bool(torch.isfinite(observation).all().item()):
        raise ValueError("constrained observation must contain only finite values")
    demand, supply, cumulative_demand, cumulative_served, previous_tier = _dispatch_state(context)
    device = demand.device
    indices = torch.arange(CONSTRAINED_ACTION_COUNT, dtype=torch.long, device=device)
    binary_divisors = 2 ** torch.arange(len(LOCALITIES), dtype=torch.long, device=device)
    tiers = 1 + (indices[:, None] // binary_divisors) % 2
    base_three_divisors = 3 ** torch.arange(len(LOCALITIES), dtype=torch.long, device=device)
    action_ids = (tiers * base_three_divisors).sum(dim=-1)
    expanded_tiers = tiers.unsqueeze(0).expand(demand.shape[0], -1, -1)
    expanded_demand = demand.unsqueeze(1).expand(-1, CONSTRAINED_ACTION_COUNT, -1, -1)
    expanded_supply = supply.unsqueeze(1).expand(-1, CONSTRAINED_ACTION_COUNT)
    requested, served = _dispatch_service_tiers(expanded_demand, expanded_supply, expanded_tiers)
    next_demand = (cumulative_demand + demand.sum(dim=-1)).unsqueeze(1)
    next_served = cumulative_served.unsqueeze(1) + served.sum(dim=-1)
    rewards = canonical_step_reward(
        hour=context.hour,
        demand=expanded_demand,
        supply=expanded_supply,
        requested=requested,
        served=served,
        tiers=expanded_tiers,
        previous_tier=previous_tier.unsqueeze(1),
        next_cumulative_demand=next_demand,
        next_cumulative_served=next_served,
    )
    feasible = requested.sum(dim=(-1, -2)) <= expanded_supply
    return ConstrainedCandidates(
        tiers=tiers,
        action_ids=action_ids,
        rewards=rewards,
        feasible=feasible,
        service_ratios=next_served / next_demand.clamp_min(NUMERIC_EPSILON),
        critical_shortfall=~feasible.any(dim=1),
    )


@torch.inference_mode()
def eligible_candidates(
    candidates: ConstrainedCandidates, reward_tolerance: float | None = 0.0
) -> torch.Tensor:
    """Admit feasible candidates within absolute one-step canonical reward regret.

    ``None`` admits every feasible action. If none is feasible, admit only index
    zero (all Critical Only). The original physical ``feasible`` mask is unchanged.
    """
    _validate_reward_tolerance(reward_tolerance)
    eligible = candidates.feasible
    if reward_tolerance is not None:
        best_reward = (
            candidates.rewards.masked_fill(~eligible, -torch.inf).max(dim=1, keepdim=True).values
        )
        eligible = eligible & (candidates.rewards >= best_reward - reward_tolerance)
    fallback = (
        torch.arange(candidates.tiers.shape[0], device=candidates.tiers.device).unsqueeze(0) == 0
    )
    return torch.where(candidates.critical_shortfall[:, None], fallback, eligible)


@torch.inference_mode()
def select_candidates(
    candidates: ConstrainedCandidates,
    *,
    preference: torch.Tensor | None = None,
    reward_tolerance: float | None = 0.0,
) -> torch.Tensor:
    """Choose by preference, then canonical reward, then lowest original action ID.

    Without preferences this is exact constrained one-step reward maximization.
    Preferences do not bypass feasibility or the explicit critical-only fallback.
    """
    eligible = eligible_candidates(candidates, reward_tolerance)
    if preference is not None:
        if not isinstance(preference, torch.Tensor):
            raise TypeError("preference must be a tensor")
        if preference.shape != candidates.rewards.shape:
            raise ValueError("preference must match the candidate reward shape")
        if preference.device != candidates.rewards.device:
            raise ValueError("preference must remain on the candidate device")
        if preference.dtype != candidates.rewards.dtype:
            raise TypeError("preference must use the candidate reward dtype")
        if not bool(torch.isfinite(preference).all().item()):
            raise ValueError("preference must contain only finite values")
        best_preference = (
            preference.masked_fill(~eligible, -torch.inf).max(dim=1, keepdim=True).values
        )
        eligible = eligible & (preference == best_preference)
    # torch.max returns the first maximal index; candidate order is ascending ID.
    selected = candidates.rewards.masked_fill(~eligible, -torch.inf).max(dim=1).indices
    return candidates.tiers[selected]


@dataclass(frozen=True, slots=True)
class ConstrainedOptimizationController:
    """Maximize current canonical reward while reserving all critical demand."""

    name: ClassVar[ControllerName] = ControllerName.OPTIMIZATION

    @torch.inference_mode()
    def act(self, context: DecisionContext) -> torch.Tensor:
        return select_candidates(evaluate_constrained_candidates(context))


@dataclass(frozen=True, slots=True)
class FairnessController:
    """Favor the worst-served locality subject to a bounded current reward sacrifice.

    First retain feasible actions within ``reward_tolerance`` of the best current
    reward. Then maximize the smallest projected cumulative locality service
    ratio. Ties favor current reward and the lowest action ID. Zero tolerance
    exactly recovers constrained greedy dispatch, including its tie behavior.
    """

    reward_tolerance: float = 0.01
    name: ClassVar[ControllerName] = ControllerName.OPTIMIZATION

    def __post_init__(self) -> None:
        _validate_reward_tolerance(self.reward_tolerance, allow_none=False)

    @torch.inference_mode()
    def act(self, context: DecisionContext) -> torch.Tensor:
        candidates = evaluate_constrained_candidates(context)
        preference = (
            None if self.reward_tolerance == 0 else candidates.service_ratios.min(dim=-1).values
        )
        return select_candidates(
            candidates, preference=preference, reward_tolerance=self.reward_tolerance
        )

"""Exact myopic dispatch optimization over the simulator's joint tier actions.

At hour t, enumerate a in {Off, Critical Only, Full Service}^7, dispatch with
the same global proportional rule as the other controllers, and maximize
canonical_step_reward for that hour and the resulting cumulative service.
All 2,187 actions are feasible under that dispatch rule; over-requesting is
proportionally clipped and receives the canonical oversubscription penalty.

This is a discrete one-step optimization comparator. It uses current demand,
supply and past service only, with no future trace or learned parameters. It
does not claim a globally optimal 24-hour schedule or isolate any PPO effect.
Equal objective values choose the lowest little-endian base-three action ID,
using the same locality/action encoding as PPO. Candidate scoring, reductions,
and selection stay on the input device; validation synchronizes one scalar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

from gridguard.controllers import DecisionContext
from gridguard.domain import LOCALITIES, SECTORS, ControllerName, canonical_step_reward
from gridguard.simulation import _dispatch_service_tiers

ACTION_COUNT = 3 ** len(LOCALITIES)


def _positive_chunk_size(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer")
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _dispatch_state(context: DecisionContext) -> tuple[torch.Tensor, ...]:
    if not isinstance(context, DecisionContext):
        raise TypeError("Optimization requires a DecisionContext")
    state = (
        context.current_demand_mw,
        context.current_supply_mw,
        context.cumulative_demand_mwh,
        context.cumulative_served_mwh,
        context.previous_service_tiers,
    )
    if any(value is None for value in state):
        raise ValueError("Optimization requires the exact causal dispatch state")
    if any(not isinstance(value, torch.Tensor) for value in state):
        raise TypeError("Optimization causal dispatch state must contain tensors")
    demand, supply, cumulative_demand, cumulative_served, previous_tier = state
    batch_size = context.observation.shape[0]
    expected_shapes = (
        (batch_size, len(LOCALITIES), len(SECTORS)),
        (batch_size,),
        (batch_size, len(LOCALITIES)),
        (batch_size, len(LOCALITIES)),
        (batch_size, len(LOCALITIES)),
    )
    if any(value.shape != shape for value, shape in zip(state, expected_shapes, strict=True)):
        raise ValueError(
            "Optimization causal dispatch state has incompatible batch/locality shapes"
        )
    if any(value.device != context.observation.device for value in state):
        raise ValueError("Optimization state and observation must remain on the same device")
    if previous_tier.dtype != torch.long:
        raise TypeError("Optimization previous_service_tiers must use torch.long")
    if any(value.dtype != context.observation.dtype for value in state[:-1]):
        raise TypeError("Optimization power tensors and observation must use the same dtype")
    valid = torch.stack(
        [torch.isfinite(value).all() & (value >= 0).all() for value in state[:-1]]
        + [
            (cumulative_served <= cumulative_demand).all(),
            ((previous_tier >= 0) & (previous_tier <= 2)).all(),
        ]
    ).all()
    if not bool(valid.item()):
        raise ValueError(
            "Optimization requires finite non-negative power, bounded cumulative service, "
            "and previous tiers between 0 and 2"
        )
    return demand, supply, cumulative_demand, cumulative_served, previous_tier


def _optimize_batch(
    *,
    hour: int,
    state: tuple[torch.Tensor, ...],
    candidates: torch.Tensor,
    candidate_chunk_size: int,
) -> torch.Tensor:
    demand, supply, cumulative_demand, cumulative_served, previous_tier = state
    batch_size = demand.shape[0]
    best_reward = demand.new_full((batch_size,), -torch.inf)
    best_index = torch.zeros(batch_size, dtype=torch.long, device=demand.device)
    for start in range(0, ACTION_COUNT, candidate_chunk_size):
        action_chunk = candidates[start : start + candidate_chunk_size]
        count = action_chunk.shape[0]
        tiers = action_chunk.unsqueeze(0).expand(batch_size, -1, -1)
        current_demand = demand.unsqueeze(1).expand(-1, count, -1, -1)
        current_supply = supply.unsqueeze(1).expand(-1, count)
        requested, served = _dispatch_service_tiers(current_demand, current_supply, tiers)
        scores = canonical_step_reward(
            hour=hour,
            demand=current_demand,
            supply=current_supply,
            requested=requested,
            served=served,
            tiers=tiers,
            previous_tier=previous_tier.unsqueeze(1),
            next_cumulative_demand=(cumulative_demand + demand.sum(dim=-1)).unsqueeze(1),
            next_cumulative_served=cumulative_served.unsqueeze(1) + served.sum(dim=-1),
        )
        chunk_reward, chunk_index = scores.max(dim=1)
        improves = chunk_reward > best_reward
        best_reward = torch.where(improves, chunk_reward, best_reward)
        best_index = torch.where(improves, chunk_index + start, best_index)
    return candidates[best_index]


@dataclass(frozen=True, slots=True)
class OptimizationController:
    """Maximize the canonical current-hour reward, exhaustively and without lookahead.

    Defaults bound each score block to 32 scenarios x 256 joint actions. Both
    chunk sizes affect memory/throughput only; they never prune the action set.
    CPU tensors are accepted for focused tests; CUDA inputs remain on CUDA.
    """

    candidate_chunk_size: int = 256
    scenario_chunk_size: int = 32
    name: ClassVar[ControllerName] = ControllerName.OPTIMIZATION

    def __post_init__(self) -> None:
        _positive_chunk_size("candidate_chunk_size", self.candidate_chunk_size)
        _positive_chunk_size("scenario_chunk_size", self.scenario_chunk_size)
        if self.candidate_chunk_size > ACTION_COUNT:
            raise ValueError(f"candidate_chunk_size must not exceed {ACTION_COUNT}")

    @torch.inference_mode()
    def act(self, context: DecisionContext) -> torch.Tensor:
        state = _dispatch_state(context)
        device = context.observation.device
        indices = torch.arange(ACTION_COUNT, dtype=torch.long, device=device)
        divisors = 3 ** torch.arange(len(LOCALITIES), dtype=torch.long, device=device)
        candidates = (indices[:, None] // divisors) % 3
        selections = [
            _optimize_batch(
                hour=context.hour,
                state=tuple(value[start : start + self.scenario_chunk_size] for value in state),
                candidates=candidates,
                candidate_chunk_size=self.candidate_chunk_size,
            )
            for start in range(0, context.observation.shape[0], self.scenario_chunk_size)
        ]
        return torch.cat(selections, dim=0)

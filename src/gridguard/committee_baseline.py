"""Additive post-hoc baselines and diagnostics; frozen scientific modules are unchanged."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import torch

from gridguard.constrained_ppo import critical_action_mask, critical_action_tiers
from gridguard.controllers import DecisionContext
from gridguard.domain import LOCALITIES, NUMERIC_EPSILON, ControllerName, Scenario, SimulationResult

REWARD_COMPONENT_NAMES = (
    "reward_critical_service",
    "reward_utilization",
    "reward_fairness",
    "reward_switching_penalty",
    "reward_oversubscription_penalty",
)
DIAGNOSTIC_NAMES = (
    *REWARD_COMPONENT_NAMES,
    "unmet_critical_mwh",
    "complete_critical_outage_locality_hours",
    "critical_supply_shortfall_hours",
    "all_critical_fallback_hours",
)
REWARD_RECONSTRUCTION_TOLERANCE = 5e-5


@dataclass(frozen=True, slots=True)
class CriticalAwareFixedRotationController:
    """Reserve critical demand, then use fixed rotating discretionary priority.

    A unique binary score encodes lexicographic priority, not utility or reward.
    The canonical mask retains exact dispatch reduction order and explicit fallback.
    """

    name: ClassVar[ControllerName] = ControllerName.FIXED_ROTATION

    @torch.inference_mode()
    def act(self, context: DecisionContext) -> torch.Tensor:
        mask = critical_action_mask(context)
        tiers = critical_action_tiers(context.observation.device)
        locality = torch.arange(len(LOCALITIES), device=tiers.device)
        start = (context.hour // 3 * 3) % len(LOCALITIES)
        priority = (locality - start) % len(LOCALITIES)
        weights = 2 ** (len(LOCALITIES) - 1 - priority)
        scores = ((tiers - 1) * weights).sum(-1)
        selected = scores.unsqueeze(0).expand_as(mask).masked_fill(~mask, -1).argmax(-1)
        return tiers[selected]


def critical_supply_stress(scenario: Scenario, fraction: float) -> Scenario:
    """Keep matched demand and replace supply by a prespecified critical-demand fraction."""
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
        raise TypeError("critical supply fraction must be a number")
    if not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("critical supply fraction must be strictly between zero and one")
    return Scenario(
        demand=scenario.demand,
        supply=scenario.demand[..., :3].sum((-1, -2)) * fraction,
        shortage_level=scenario.shortage_level,
        seed=scenario.seed,
    )


@torch.inference_mode()
def audit_diagnostics(result: SimulationResult) -> dict[str, torch.Tensor]:
    """Decompose the canonical reward and expose infeasible-critical-supply outcomes.

    Follow the canonical sequential cumulative additions and floating-point order.
    ``critical_outage_locality_hours`` in the existing metrics uses the 0.99 threshold;
    the added complete-outage field separately counts exactly zero critical delivery.
    """
    result.validate_trace_invariants()
    demand, supply, served = result.scenario.demand, result.scenario.supply, result.served
    cumulative_demand = torch.zeros_like(demand[:, 0, :, 0])
    cumulative_served = torch.zeros_like(cumulative_demand)
    previous = torch.zeros_like(result.service_tiers[:, 0])
    components = []
    for hour in range(24):
        hourly_demand, hourly_served = demand[:, hour], served[:, hour]
        demand_total = hourly_demand.sum((-1, -2))
        critical_demand = hourly_demand[..., :3].sum((-1, -2))
        critical_served = hourly_served[..., :3].sum((-1, -2))
        cumulative_demand = cumulative_demand + hourly_demand.sum(-1)
        cumulative_served = cumulative_served + hourly_served.sum(-1)
        ratio = cumulative_served / cumulative_demand.clamp_min(NUMERIC_EPSILON)
        fairness = ratio.sum(-1).square() / (
            len(LOCALITIES) * ratio.square().sum(-1) + NUMERIC_EPSILON
        )
        switching = (
            torch.zeros_like(supply[:, hour])
            if hour == 0
            else (result.service_tiers[:, hour] != previous).to(demand.dtype).mean(-1)
        )
        oversubscription = torch.relu(
            result.requested[:, hour].sum((-1, -2)) - supply[:, hour]
        ) / demand_total.clamp_min(NUMERIC_EPSILON)
        components.append(
            torch.stack(
                (
                    5 * critical_served / critical_demand.clamp_min(NUMERIC_EPSILON) / 9,
                    2
                    * hourly_served.sum((-1, -2))
                    / torch.minimum(supply[:, hour], demand_total).clamp_min(NUMERIC_EPSILON)
                    / 9,
                    2 * fairness / 9,
                    -0.25 * switching / 9,
                    -3 * oversubscription / 9,
                ),
                dim=-1,
            )
        )
        previous = result.service_tiers[:, hour]
    values = torch.stack(components, dim=1).sum(1)
    if not torch.allclose(
        values.sum(-1), result.reward.sum(1), atol=REWARD_RECONSTRUCTION_TOLERANCE, rtol=0
    ):
        raise ValueError("reward component reconstruction differs from the canonical trace")
    critical_demand = demand[..., :3].sum(-1)
    critical_served = served[..., :3].sum(-1)
    shortfall = critical_demand.sum(-1) > supply
    diagnostics = {name: values[:, index] for index, name in enumerate(REWARD_COMPONENT_NAMES)}
    diagnostics.update(
        unmet_critical_mwh=(critical_demand - critical_served).clamp_min(0).sum((1, 2)),
        complete_critical_outage_locality_hours=(
            (critical_demand > NUMERIC_EPSILON) & (critical_served == 0)
        )
        .to(demand.dtype)
        .sum((1, 2)),
        critical_supply_shortfall_hours=shortfall.to(demand.dtype).sum(1),
        all_critical_fallback_hours=(shortfall & (result.service_tiers == 1).all(-1))
        .to(demand.dtype)
        .sum(1),
    )
    return diagnostics

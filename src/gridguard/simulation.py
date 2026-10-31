"""CUDA-preserving 24-hour load-shedding simulation."""

from __future__ import annotations

from math import pi

import torch

from gridguard.controllers import Controller, DecisionContext
from gridguard.domain import (
    CRITICAL_SECTOR_COUNT,
    HORIZON_HOURS,
    LOCALITIES,
    LOCALITY_DEMAND_WEIGHTS,
    NUMERIC_EPSILON,
    SECTORS,
    Scenario,
    ServiceTier,
    SimulationResult,
    canonical_step_reward,
)
from gridguard.scenarios import SYSTEM_PEAK_MW


def _require_tensor(name: str, value: object) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    return value


def _validate_hour(hour: int) -> None:
    if isinstance(hour, bool) or not isinstance(hour, int):
        raise TypeError("hour must be an integer")
    if not 0 <= hour < HORIZON_HOURS:
        raise ValueError("hour must be between 0 and 23")


def _validate_step_inputs(
    *,
    demand: torch.Tensor,
    supply: torch.Tensor,
    cumulative_demand: torch.Tensor,
    cumulative_served: torch.Tensor,
    previous_tier: torch.Tensor,
) -> None:
    if demand.ndim != 3 or demand.shape[1:] != (len(LOCALITIES), len(SECTORS)):
        raise ValueError("demand must have shape [batch, 7, 5]")
    batch_size = demand.shape[0]
    if batch_size < 1:
        raise ValueError("demand batch must contain at least one scenario")
    if supply.shape != (batch_size,):
        raise ValueError("supply must have shape [batch]")
    expected_locality_shape = (batch_size, len(LOCALITIES))
    if cumulative_demand.shape != expected_locality_shape:
        raise ValueError("cumulative_demand must have shape [batch, 7]")
    if cumulative_served.shape != expected_locality_shape:
        raise ValueError("cumulative_served must have shape [batch, 7]")
    if previous_tier.shape != expected_locality_shape:
        raise ValueError("previous_tier must have shape [batch, 7]")
    if previous_tier.dtype != torch.long:
        raise TypeError("previous_tier must use torch.long")
    tensors = (supply, cumulative_demand, cumulative_served, previous_tier)
    if any(tensor.device != demand.device for tensor in tensors):
        raise ValueError("all step tensors must remain on the same device")
    if not demand.is_floating_point():
        raise TypeError("demand must be a floating-point tensor")
    if any(
        tensor.dtype != demand.dtype for tensor in (supply, cumulative_demand, cumulative_served)
    ):
        raise ValueError("power tensors must use the same dtype")
    invalid_values = torch.stack(
        (
            ~torch.isfinite(demand).all(),
            (demand < 0).any(),
            ~torch.isfinite(supply).all(),
            (supply < 0).any(),
            ~torch.isfinite(cumulative_demand).all(),
            (cumulative_demand < 0).any(),
            ~torch.isfinite(cumulative_served).all(),
            (cumulative_served < 0).any(),
            (cumulative_served > cumulative_demand).any(),
            (previous_tier < int(ServiceTier.OFF)).any(),
            (previous_tier > int(ServiceTier.FULL_SERVICE)).any(),
        )
    ).to(device="cpu")
    messages = (
        "demand must contain only finite power values",
        "demand must not contain negative power",
        "supply must contain only finite power values",
        "supply must not contain negative power",
        "cumulative_demand must contain only finite power values",
        "cumulative_demand must not contain negative power",
        "cumulative_served must contain only finite power values",
        "cumulative_served must not contain negative power",
        "cumulative_served must not exceed cumulative_demand",
        "invalid previous service tier: values must be between 0 and 2",
        "invalid previous service tier: values must be between 0 and 2",
    )
    for invalid, message in zip(invalid_values.tolist(), messages, strict=True):
        if invalid:
            raise ValueError(message)


def _locality_peaks(reference: torch.Tensor) -> torch.Tensor:
    weights = reference.new_tensor(LOCALITY_DEMAND_WEIGHTS)
    return SYSTEM_PEAK_MW * weights / weights.sum()


def _build_observation(
    *,
    hour: int,
    demand: torch.Tensor,
    supply: torch.Tensor,
    cumulative_demand: torch.Tensor,
    cumulative_served: torch.Tensor,
    previous_tier: torch.Tensor,
    locality_peaks: torch.Tensor,
) -> torch.Tensor:
    locality_demand = demand.sum(dim=-1)
    total_demand = locality_demand.sum(dim=-1)
    critical_share = demand[..., :CRITICAL_SECTOR_COUNT].sum(dim=-1) / locality_demand.clamp_min(
        NUMERIC_EPSILON
    )
    cumulative_service = cumulative_served / cumulative_demand.clamp_min(NUMERIC_EPSILON)
    previous_tier_fraction = previous_tier.to(demand.dtype) / 2.0

    angle = demand.new_full(supply.shape, 2.0 * pi * hour / HORIZON_HOURS)
    global_features = torch.stack(
        (
            torch.sin(angle),
            torch.cos(angle),
            supply / total_demand.clamp_min(NUMERIC_EPSILON),
            supply / SYSTEM_PEAK_MW,
        ),
        dim=-1,
    )
    locality_features = torch.stack(
        (
            locality_demand / locality_peaks,
            critical_share,
            cumulative_service,
            previous_tier_fraction,
        ),
        dim=-1,
    ).flatten(start_dim=-2)
    return torch.cat((global_features, locality_features), dim=-1)


def build_observation(
    *,
    hour: int,
    demand: torch.Tensor,
    supply: torch.Tensor,
    cumulative_demand: torch.Tensor,
    cumulative_served: torch.Tensor,
    previous_tier: torch.Tensor,
) -> torch.Tensor:
    """Build the accepted 32-wide causal observation on the input tensor device."""

    _validate_hour(hour)
    demand = _require_tensor("demand", demand)
    supply = _require_tensor("supply", supply)
    cumulative_demand = _require_tensor("cumulative_demand", cumulative_demand)
    cumulative_served = _require_tensor("cumulative_served", cumulative_served)
    previous_tier = _require_tensor("previous_tier", previous_tier)
    _validate_step_inputs(
        demand=demand,
        supply=supply,
        cumulative_demand=cumulative_demand,
        cumulative_served=cumulative_served,
        previous_tier=previous_tier,
    )
    return _build_observation(
        hour=hour,
        demand=demand,
        supply=supply,
        cumulative_demand=cumulative_demand,
        cumulative_served=cumulative_served,
        previous_tier=previous_tier,
        locality_peaks=_locality_peaks(demand),
    )


def _validate_dispatch_inputs(
    demand: torch.Tensor,
    supply: torch.Tensor,
    tiers: torch.Tensor,
) -> None:
    if demand.ndim != 3 or demand.shape[1:] != (len(LOCALITIES), len(SECTORS)):
        raise ValueError("demand must have shape [batch, 7, 5]")
    if supply.shape != (demand.shape[0],):
        raise ValueError("supply must have shape [batch]")
    if tiers.shape != (demand.shape[0], len(LOCALITIES)):
        raise ValueError("tiers must have shape [batch, 7]")
    if tiers.dtype != torch.long:
        raise TypeError("tiers must use torch.long")
    if bool(((tiers < ServiceTier.OFF) | (tiers > ServiceTier.FULL_SERVICE)).any().item()):
        raise ValueError("tiers contains an invalid service tier")
    if supply.device != demand.device or tiers.device != demand.device:
        raise ValueError("dispatch tensors must remain on the same device")
    if not demand.is_floating_point() or supply.dtype != demand.dtype:
        raise TypeError("demand and supply must use the same floating-point dtype")
    invalid_power = torch.stack(
        (
            ~torch.isfinite(demand).all(),
            (demand < 0).any(),
            ~torch.isfinite(supply).all(),
            (supply < 0).any(),
        )
    ).to(device="cpu")
    messages = (
        "demand must contain only finite power values",
        "demand must not contain negative power",
        "supply must contain only finite power values",
        "supply must not contain negative power",
    )
    for invalid, message in zip(invalid_power.tolist(), messages, strict=True):
        if invalid:
            raise ValueError(message)


def _dispatch_service_tiers(
    demand: torch.Tensor,
    supply: torch.Tensor,
    tiers: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    critical_request = (
        demand[..., :CRITICAL_SECTOR_COUNT] * (tiers >= ServiceTier.CRITICAL_ONLY)[..., None]
    )
    noncritical_request = (
        demand[..., CRITICAL_SECTOR_COUNT:] * (tiers >= ServiceTier.FULL_SERVICE)[..., None]
    )
    requested = torch.cat((critical_request, noncritical_request), dim=-1)
    requested_total = requested.sum(dim=(-1, -2))
    scale = torch.minimum(
        torch.ones_like(supply),
        supply / requested_total.clamp_min(NUMERIC_EPSILON),
    )
    served = requested * scale[..., None, None]
    return requested, served


def dispatch_service_tiers(
    *,
    demand: torch.Tensor,
    supply: torch.Tensor,
    tiers: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Request tier-selected load and proportionally dispatch one global supply pool."""

    demand = _require_tensor("demand", demand)
    supply = _require_tensor("supply", supply)
    tiers = _require_tensor("tiers", tiers)
    _validate_dispatch_inputs(demand, supply, tiers)
    return _dispatch_service_tiers(demand, supply, tiers)


def _validate_controller_action(
    action: object,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(action, torch.Tensor):
        raise TypeError("controller action must be a torch.Tensor")
    if action.shape != (batch_size, len(LOCALITIES)):
        raise ValueError("controller action must have shape [batch, 7]")
    if action.dtype != torch.long:
        raise TypeError("controller action must use torch.long")
    if action.device != device:
        raise ValueError("controller action must remain on the scenario device")
    # Eager CUDA rejection requires one scalar synchronization; the action tensor stays on-device.
    invalid_tier = (
        (action < int(ServiceTier.OFF)) | (action > int(ServiceTier.FULL_SERVICE))
    ).any()
    if bool(invalid_tier.item()):
        raise ValueError(
            "controller action contains an invalid service tier: values must be between 0 and 2"
        )
    return action


@torch.inference_mode()
def simulate(scenario: Scenario, controller: Controller) -> SimulationResult:
    """Run one matched 24-hour batch without copying hot-path tensors to the CPU."""

    if not isinstance(scenario, Scenario):
        raise TypeError("scenario must be a Scenario")
    if not isinstance(controller, Controller):
        raise TypeError("controller must satisfy the Controller protocol")
    scenario.validate_shortage_invariant()

    batch_size = scenario.batch_size
    demand_dtype = scenario.demand.dtype
    device = scenario.demand.device
    cumulative_demand = torch.zeros(
        (batch_size, len(LOCALITIES)),
        dtype=demand_dtype,
        device=device,
    )
    cumulative_served = torch.zeros_like(cumulative_demand)
    previous_tier = torch.zeros(
        (batch_size, len(LOCALITIES)),
        dtype=torch.long,
        device=device,
    )
    locality_peaks = _locality_peaks(scenario.demand)

    tier_trace: list[torch.Tensor] = []
    requested_trace: list[torch.Tensor] = []
    served_trace: list[torch.Tensor] = []
    reward_trace: list[torch.Tensor] = []

    for hour in range(HORIZON_HOURS):
        demand = scenario.demand[:, hour]
        supply = scenario.supply[:, hour]
        observation = _build_observation(
            hour=hour,
            demand=demand,
            supply=supply,
            cumulative_demand=cumulative_demand,
            cumulative_served=cumulative_served,
            previous_tier=previous_tier,
            locality_peaks=locality_peaks,
        )
        tiers = _validate_controller_action(
            controller.act(
                DecisionContext(
                    hour=hour,
                    observation=observation,
                    _validate_finite=False,
                    current_demand_mw=demand,
                    current_supply_mw=supply,
                    cumulative_demand_mwh=cumulative_demand,
                    cumulative_served_mwh=cumulative_served,
                    previous_service_tiers=previous_tier,
                )
            ),
            batch_size=batch_size,
            device=device,
        )
        requested, served = _dispatch_service_tiers(demand, supply, tiers)
        locality_demand = demand.sum(dim=-1)
        locality_served = served.sum(dim=-1)
        next_cumulative_demand = cumulative_demand + locality_demand
        next_cumulative_served = cumulative_served + locality_served
        reward = canonical_step_reward(
            hour=hour,
            demand=demand,
            supply=supply,
            requested=requested,
            served=served,
            tiers=tiers,
            previous_tier=previous_tier,
            next_cumulative_demand=next_cumulative_demand,
            next_cumulative_served=next_cumulative_served,
        )

        tier_trace.append(tiers)
        requested_trace.append(requested)
        served_trace.append(served)
        reward_trace.append(reward)
        cumulative_demand = next_cumulative_demand
        cumulative_served = next_cumulative_served
        previous_tier = tiers

    return SimulationResult(
        controller_name=controller.name,
        scenario=scenario,
        service_tiers=torch.stack(tier_trace, dim=1),
        requested=torch.stack(requested_trace, dim=1),
        served=torch.stack(served_trace, dim=1),
        reward=torch.stack(reward_trace, dim=1),
    )

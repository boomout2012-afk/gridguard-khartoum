"""Proposal metric tensors derived from immutable simulation traces."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch

from gridguard.domain import CRITICAL_SECTOR_COUNT, LOCALITIES, SimulationResult

EPSILON = 1e-8
POWER_BALANCE_TOLERANCE_MW = 1e-3
CRITICAL_OUTAGE_SERVICE_THRESHOLD = 0.99
STEP_HOURS = 1.0


@dataclass(frozen=True, slots=True, eq=False)
class EpisodeMetrics:
    """One device-resident value per episode for every proposal metric."""

    episode_reward: torch.Tensor
    demand_mwh: torch.Tensor
    available_mwh: torch.Tensor
    served_mwh: torch.Tensor
    curtailed_mwh: torch.Tensor
    critical_service_ratio: torch.Tensor
    utilization: torch.Tensor
    fairness: torch.Tensor
    switching_rate: torch.Tensor
    oversubscription: torch.Tensor
    power_balance_violation_rate: torch.Tensor
    critical_outage_locality_hours: torch.Tensor

    def __post_init__(self) -> None:
        values = tuple(getattr(self, field.name) for field in fields(self))
        if not all(isinstance(value, torch.Tensor) for value in values):
            raise TypeError("episode metrics must be torch.Tensor values")
        reference = values[0]
        if reference.ndim != 1 or reference.shape[0] < 1:
            raise ValueError("episode metrics must have shape [batch]")
        if not reference.is_floating_point():
            raise TypeError("episode metrics must be floating-point tensors")
        if any(value.shape != reference.shape for value in values[1:]):
            raise ValueError("all episode metrics must have the same batch shape")
        if any(value.device != reference.device for value in values[1:]):
            raise ValueError("all episode metrics must remain on the same device")
        if any(value.dtype != reference.dtype for value in values[1:]):
            raise ValueError("all episode metrics must use the same dtype")
        if not bool(torch.stack(values).isfinite().all().item()):
            raise ValueError("episode metrics must contain only finite values")

    @property
    def batch_size(self) -> int:
        return self.episode_reward.shape[0]

    @property
    def device(self) -> torch.device:
        return self.episode_reward.device

    @property
    def episode_pooled_critical_service_ratio(self) -> torch.Tensor:
        """Critical energy served / demanded, pooled within each 24-hour episode."""

        return self.critical_service_ratio

    @property
    def mean_hourly_supply_utilization_ratio(self) -> torch.Tensor:
        """Episode mean of hourly served / min(available supply, demand)."""

        return self.utilization

    @property
    def mean_hourly_cumulative_locality_jain_fairness(self) -> torch.Tensor:
        """Episode mean of hourly Jain fairness over cumulative locality service ratios."""

        return self.fairness

    def as_dict(self) -> dict[str, torch.Tensor]:
        """Expose named tensors without transferring or scalarizing them."""

        return {field.name: getattr(self, field.name) for field in fields(self)}


def _jain_fairness(service_ratio: torch.Tensor) -> torch.Tensor:
    return service_ratio.sum(dim=-1).square() / (
        len(LOCALITIES) * service_ratio.square().sum(dim=-1) + EPSILON
    )


@torch.inference_mode()
def compute_episode_metrics(result: SimulationResult) -> EpisodeMetrics:
    """Compute frozen-source-compatible episode metrics on the result device."""

    if not isinstance(result, SimulationResult):
        raise TypeError("result must be a SimulationResult")

    demand = result.scenario.demand
    supply = result.scenario.supply
    served = result.served
    requested = result.requested
    tiers = result.service_tiers
    dtype = demand.dtype

    demand_total = demand.sum(dim=(-1, -2))
    served_total = served.sum(dim=(-1, -2))
    requested_total = requested.sum(dim=(-1, -2))
    critical_demand = demand[..., :CRITICAL_SECTOR_COUNT].sum(dim=(-1, -2))
    critical_served = served[..., :CRITICAL_SECTOR_COUNT].sum(dim=(-1, -2))
    locality_demand = demand.sum(dim=-1)
    locality_served = served.sum(dim=-1)
    locality_critical_demand = demand[..., :CRITICAL_SECTOR_COUNT].sum(dim=-1)
    locality_critical_served = served[..., :CRITICAL_SECTOR_COUNT].sum(dim=-1)

    cumulative_service = locality_served.cumsum(dim=1) / locality_demand.cumsum(dim=1).clamp_min(
        EPSILON
    )
    hourly_cumulative_locality_jain_fairness = _jain_fairness(cumulative_service)
    hourly_supply_utilization_ratio = served_total / torch.minimum(supply, demand_total).clamp_min(
        EPSILON
    )
    hourly_switching_rate = torch.cat(
        (
            torch.zeros_like(supply[:, :1]),
            (tiers[:, 1:] != tiers[:, :-1]).to(dtype).mean(dim=-1),
        ),
        dim=1,
    )
    hourly_oversubscription_ratio = torch.relu(requested_total - supply) / demand_total.clamp_min(
        EPSILON
    )
    locality_critical_service = locality_critical_served / locality_critical_demand.clamp_min(
        EPSILON
    )
    critical_outage = (locality_critical_demand > EPSILON) & (
        locality_critical_service < CRITICAL_OUTAGE_SERVICE_THRESHOLD
    )

    demand_mwh = demand_total.sum(dim=1) * STEP_HOURS
    served_mwh = served_total.sum(dim=1) * STEP_HOURS
    critical_demand_mwh = critical_demand.sum(dim=1) * STEP_HOURS
    critical_served_mwh = critical_served.sum(dim=1) * STEP_HOURS
    episode_pooled_critical_service_ratio = critical_served_mwh / critical_demand_mwh.clamp_min(
        EPSILON
    )

    return EpisodeMetrics(
        episode_reward=result.reward.sum(dim=1),
        demand_mwh=demand_mwh,
        available_mwh=supply.sum(dim=1) * STEP_HOURS,
        served_mwh=served_mwh,
        curtailed_mwh=(demand_total - served_total).sum(dim=1) * STEP_HOURS,
        critical_service_ratio=episode_pooled_critical_service_ratio,
        utilization=hourly_supply_utilization_ratio.mean(dim=1),
        fairness=hourly_cumulative_locality_jain_fairness.mean(dim=1),
        switching_rate=hourly_switching_rate.mean(dim=1),
        oversubscription=hourly_oversubscription_ratio.mean(dim=1),
        power_balance_violation_rate=(served_total > supply + POWER_BALANCE_TOLERANCE_MW)
        .to(dtype)
        .mean(dim=1),
        critical_outage_locality_hours=critical_outage.to(dtype).sum(dim=(1, 2)) * STEP_HOURS,
    )

"""Direct 24-hour interruption durations and locality disparity on CUDA.

These metrics supplement, and do not redefine, the frozen evidence metrics.
Each one-hour sample is a complete modeled hour, not a sub-hour outage estimate.
Zero-service blackout means positive locality demand and exactly zero actual
delivered power. Partial service is not a zero-service blackout. Critical-service
interruption uses the existing strict service-ratio threshold of 0.99. Hours
without demand above the existing numerical epsilon do not count as outages.

For locality durations d_i in one episode, range is max(d_i) - min(d_i), and
mean pairwise gap is sum_{i<j} abs(d_i - d_j) / 21 for the seven localities.
Both are in hours and lower means more equal durations, not necessarily better
service: all localities interrupted for 24 hours also have zero disparity.
Streaks stop at the episode boundaries; hours 23 and 0 are not joined.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch

from gridguard.domain import (
    CRITICAL_SECTOR_COUNT,
    HORIZON_HOURS,
    LOCALITIES,
    ServiceTier,
    SimulationResult,
)
from gridguard.metrics import CRITICAL_OUTAGE_SERVICE_THRESHOLD, EPSILON, STEP_HOURS


@dataclass(frozen=True, slots=True, eq=False)
class DurationMetrics:
    """CUDA duration tensors: locality fields [batch, 7], other fields [batch].

    All fields use hours except ``blackout_locality_hours`` and
    ``critical_interruption_locality_hours``, which sum locality-hours.
    ``locality_off_hours`` counts requested Off actions regardless of demand or
    actual delivery; it is an action diagnostic, not an outage outcome.
    ``longest_*`` is the maximum locality streak within each episode.
    """

    locality_blackout_hours: torch.Tensor
    locality_critical_interruption_hours: torch.Tensor
    locality_off_hours: torch.Tensor
    locality_longest_blackout_hours: torch.Tensor
    locality_longest_critical_interruption_hours: torch.Tensor
    blackout_locality_hours: torch.Tensor
    critical_interruption_locality_hours: torch.Tensor
    blackout_duration_range_hours: torch.Tensor
    blackout_duration_mean_pairwise_gap_hours: torch.Tensor
    critical_interruption_duration_range_hours: torch.Tensor
    critical_interruption_duration_mean_pairwise_gap_hours: torch.Tensor
    longest_blackout_hours: torch.Tensor
    longest_critical_interruption_hours: torch.Tensor

    def __post_init__(self) -> None:
        values = self.as_dict()
        if not all(isinstance(value, torch.Tensor) for value in values.values()):
            raise TypeError("duration metrics must be torch.Tensor values")
        reference = self.locality_blackout_hours
        if reference.ndim != 2 or reference.shape[0] < 1:
            raise ValueError("locality duration metrics must have shape [batch, 7]")
        if reference.device.type != "cuda":
            raise ValueError("duration metrics require CUDA tensors")
        if not reference.is_floating_point():
            raise TypeError("duration metrics must use floating-point hours")
        for name, value in values.items():
            expected_shape = (
                (self.batch_size, len(LOCALITIES))
                if name.startswith("locality_")
                else (self.batch_size,)
            )
            if value.shape != expected_shape:
                raise ValueError(f"{name} must have shape {list(expected_shape)}")
            if value.device != reference.device or value.dtype != reference.dtype:
                raise ValueError("duration metrics must share the same CUDA device and dtype")
        # Only this validity predicate is synchronized; metric arrays stay on CUDA.
        packed = torch.cat(tuple(value.reshape(-1) for value in values.values()))
        if not bool((torch.isfinite(packed) & (packed >= 0)).all().item()):
            raise ValueError("duration metrics must contain finite, non-negative hours")

    @property
    def batch_size(self) -> int:
        return self.locality_blackout_hours.shape[0]

    @property
    def device(self) -> torch.device:
        return self.locality_blackout_hours.device

    def as_dict(self) -> dict[str, torch.Tensor]:
        """Expose every tensor without copying to host or materializing rows."""

        return {field.name: getattr(self, field.name) for field in fields(self)}

    def mean_over_episodes(self) -> dict[str, torch.Tensor]:
        """Equal-episode means on CUDA, retaining locality order for [7] outputs.

        Disparity and maximum streak are calculated within each episode before
        averaging. This is not disparity of the averaged locality durations.
        """

        return {name: value.mean(dim=0) for name, value in self.as_dict().items()}


def _longest_streak_hours(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Longest consecutive true run per locality, without episode wraparound."""

    hour = torch.arange(1, HORIZON_HOURS + 1, device=mask.device)[None, :, None]
    last_uninterrupted_hour = torch.where(mask, 0, hour).cummax(dim=1).values
    streak_hours = hour - last_uninterrupted_hour
    return streak_hours.amax(dim=1).to(dtype=dtype) * STEP_HOURS


def _duration_disparity_hours(duration_hours: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Range and mean absolute gap over the 21 distinct unordered locality pairs."""

    range_hours = duration_hours.amax(dim=1) - duration_hours.amin(dim=1)
    pairwise_gaps = (duration_hours[:, :, None] - duration_hours[:, None, :]).abs()
    # Every unordered pair appears twice; diagonal terms are zero and excluded
    # from the denominator, unlike an average over all 49 matrix entries.
    ordered_pair_count = len(LOCALITIES) * (len(LOCALITIES) - 1)
    mean_pairwise_gap_hours = pairwise_gaps.sum(dim=(1, 2)) / ordered_pair_count
    return range_hours, mean_pairwise_gap_hours


@torch.inference_mode()
def compute_duration_metrics(result: SimulationResult) -> DurationMetrics:
    """Measure delivered-service interruptions from a validated CUDA trace.

    Required trace fields are demand/served [batch, 24, 7, 5] and requested tiers
    [batch, 24, 7]. The existing trace validator protects provenance and may
    synchronize validity flags. No demand, delivered-power, mask, duration, or
    aggregate tensor is transferred to host memory here. CPU traces fail closed.
    """

    if not isinstance(result, SimulationResult):
        raise TypeError("result must be a SimulationResult")
    if result.scenario.demand.device.type != "cuda":
        raise ValueError("duration metrics require a CUDA simulation trace")
    result.validate_trace_invariants()
    demand_mw = result.scenario.demand
    served_mw = result.served
    dtype = demand_mw.dtype

    locality_demand_mw = demand_mw.sum(dim=-1)
    locality_served_mw = served_mw.sum(dim=-1)
    blackout = (locality_demand_mw > EPSILON) & (locality_served_mw == 0.0)
    critical_demand_mw = demand_mw[..., :CRITICAL_SECTOR_COUNT].sum(dim=-1)
    critical_served_mw = served_mw[..., :CRITICAL_SECTOR_COUNT].sum(dim=-1)
    critical_service_ratio = critical_served_mw / critical_demand_mw.clamp_min(EPSILON)
    critical_interruption = (critical_demand_mw > EPSILON) & (
        critical_service_ratio < CRITICAL_OUTAGE_SERVICE_THRESHOLD
    )

    blackout_hours = blackout.to(dtype=dtype).sum(dim=1) * STEP_HOURS
    critical_interruption_hours = critical_interruption.to(dtype=dtype).sum(dim=1) * STEP_HOURS
    longest_blackout_hours = _longest_streak_hours(blackout, dtype)
    longest_critical_interruption_hours = _longest_streak_hours(critical_interruption, dtype)
    blackout_range_hours, blackout_pairwise_gap_hours = _duration_disparity_hours(blackout_hours)
    critical_range_hours, critical_pairwise_gap_hours = _duration_disparity_hours(
        critical_interruption_hours
    )

    return DurationMetrics(
        locality_blackout_hours=blackout_hours,
        locality_critical_interruption_hours=critical_interruption_hours,
        locality_off_hours=(result.service_tiers == ServiceTier.OFF).to(dtype=dtype).sum(dim=1)
        * STEP_HOURS,
        locality_longest_blackout_hours=longest_blackout_hours,
        locality_longest_critical_interruption_hours=longest_critical_interruption_hours,
        blackout_locality_hours=blackout_hours.sum(dim=1),
        critical_interruption_locality_hours=critical_interruption_hours.sum(dim=1),
        blackout_duration_range_hours=blackout_range_hours,
        blackout_duration_mean_pairwise_gap_hours=blackout_pairwise_gap_hours,
        critical_interruption_duration_range_hours=critical_range_hours,
        critical_interruption_duration_mean_pairwise_gap_hours=critical_pairwise_gap_hours,
        longest_blackout_hours=longest_blackout_hours.amax(dim=1),
        longest_critical_interruption_hours=longest_critical_interruption_hours.amax(dim=1),
    )

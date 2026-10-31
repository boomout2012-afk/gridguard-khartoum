"""Immutable host-side contracts for the interactive live comparison."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from types import MappingProxyType

from gridguard.domain import HORIZON_HOURS, SECTORS, ControllerName, ServiceTier, ShortageLevel
from gridguard.live_policy import LivePolicyIdentity
from gridguard.ui.map_asset import EXPECTED_LOCALITY_IDS


@dataclass(frozen=True, slots=True)
class LocalityHourDetail:
    """Actual demand and service delivered to one locality in one hour."""

    demand_mw: float
    served_mw: float
    critical_service_ratio: float
    sector_service_ratio: Mapping[str, float]
    next_hour_projected_demand_mw: float | None = None

    def __post_init__(self) -> None:
        demand = _nonnegative_number(self.demand_mw, field="locality demand_mw")
        served = _nonnegative_number(self.served_mw, field="locality served_mw")
        if served > demand + 1e-4:
            raise ValueError("locality served_mw must not exceed demand_mw")
        critical = _ratio(self.critical_service_ratio, field="locality critical_service_ratio")
        sector_names = tuple(sector.value for sector in SECTORS)
        if set(self.sector_service_ratio) != set(sector_names):
            raise ValueError("sector_service_ratio must contain all five approved sectors")
        sectors = {
            name: _ratio(self.sector_service_ratio[name], field=f"{name} service ratio")
            for name in sector_names
        }
        object.__setattr__(self, "demand_mw", demand)
        object.__setattr__(self, "served_mw", served)
        object.__setattr__(self, "critical_service_ratio", critical)
        object.__setattr__(self, "sector_service_ratio", MappingProxyType(sectors))
        projection = self.next_hour_projected_demand_mw
        if projection is not None:
            projection = _nonnegative_number(
                projection,
                field="next-hour projected demand_mw",
            )
        object.__setattr__(self, "next_hour_projected_demand_mw", projection)


@dataclass(frozen=True, slots=True)
class LiveScenarioFrame:
    """One validated display hour transferred from the GPU simulation."""

    hour: int
    service_tiers: Mapping[str, ServiceTier | int]
    demand_mw: float
    supply_mw: float
    locality_details: Mapping[str, LocalityHourDetail]

    def __post_init__(self) -> None:
        if isinstance(self.hour, bool) or not isinstance(self.hour, int) or not 0 <= self.hour < 24:
            raise ValueError("hour must be between 0 and 23")
        if set(self.service_tiers) != set(EXPECTED_LOCALITY_IDS):
            raise ValueError("service_tiers must contain all seven locality IDs")
        normalized = {
            locality_id: ServiceTier(self.service_tiers[locality_id])
            for locality_id in EXPECTED_LOCALITY_IDS
        }
        object.__setattr__(self, "service_tiers", MappingProxyType(normalized))
        demand = _nonnegative_number(self.demand_mw, field="demand_mw")
        supply = _nonnegative_number(self.supply_mw, field="supply_mw")
        if set(self.locality_details) != set(EXPECTED_LOCALITY_IDS):
            raise ValueError("locality_details must contain all seven locality IDs")
        details = {
            locality_id: self.locality_details[locality_id] for locality_id in EXPECTED_LOCALITY_IDS
        }
        if any(not isinstance(detail, LocalityHourDetail) for detail in details.values()):
            raise TypeError("locality_details values must be LocalityHourDetail")
        tolerance = max(1e-3, 1e-6 * demand)
        if abs(sum(detail.demand_mw for detail in details.values()) - demand) > tolerance:
            raise ValueError("locality demand must sum to frame demand_mw")
        if sum(detail.served_mw for detail in details.values()) > supply + tolerance:
            raise ValueError("locality served power must not exceed frame supply_mw")
        object.__setattr__(self, "demand_mw", demand)
        object.__setattr__(self, "supply_mw", supply)
        object.__setattr__(self, "locality_details", MappingProxyType(details))


@dataclass(frozen=True, slots=True)
class LiveDurationSummary:
    """Already-calculated duration outcomes after the single display transfer.

    Totals use locality-hours; disparity and longest interruption use hours.
    Requested Off actions are not substituted for actual delivered-service loss.
    """

    blackout_locality_hours: float
    critical_interruption_locality_hours: float
    blackout_duration_range_hours: float
    blackout_duration_mean_pairwise_gap_hours: float
    critical_interruption_duration_range_hours: float
    critical_interruption_duration_mean_pairwise_gap_hours: float
    longest_blackout_hours: float
    longest_critical_interruption_hours: float

    def __post_init__(self) -> None:
        for field in fields(self):
            value = _nonnegative_number(getattr(self, field.name), field=field.name)
            maximum = (
                HORIZON_HOURS * len(EXPECTED_LOCALITY_IDS)
                if field.name.endswith("locality_hours")
                else HORIZON_HOURS
            )
            if value > maximum:
                raise ValueError(f"{field.name} must not exceed {maximum} hours")
            object.__setattr__(self, field.name, value)


@dataclass(frozen=True, slots=True)
class LiveEpisodeSummary:
    """Host-side aggregate metrics for one live controller episode."""

    served_mwh: float
    critical_service_ratio: float
    fairness: float
    curtailed_mwh: float
    power_balance_violation_rate: float
    duration: LiveDurationSummary | None = None
    utilization: float | None = None
    episode_reward: float | None = None

    def __post_init__(self) -> None:
        if self.duration is not None and not isinstance(self.duration, LiveDurationSummary):
            raise TypeError("duration must be a LiveDurationSummary")
        if self.utilization is not None:
            object.__setattr__(self, "utilization", _ratio(self.utilization, field="utilization"))
        if self.episode_reward is not None:
            reward = self.episode_reward
            if isinstance(reward, bool) or not isinstance(reward, (int, float)):
                raise TypeError("episode_reward must be a number")
            if not math.isfinite(reward):
                raise ValueError("episode_reward must be finite")
            object.__setattr__(self, "episode_reward", float(reward))
        object.__setattr__(
            self,
            "served_mwh",
            _nonnegative_number(self.served_mwh, field="served_mwh"),
        )
        object.__setattr__(
            self,
            "critical_service_ratio",
            _ratio(self.critical_service_ratio, field="critical_service_ratio"),
        )
        object.__setattr__(self, "fairness", _ratio(self.fairness, field="fairness"))
        object.__setattr__(
            self,
            "curtailed_mwh",
            _nonnegative_number(self.curtailed_mwh, field="curtailed_mwh"),
        )
        object.__setattr__(
            self,
            "power_balance_violation_rate",
            _ratio(
                self.power_balance_violation_rate,
                field="power_balance_violation_rate",
            ),
        )


@dataclass(frozen=True, slots=True)
class LiveScenarioComparison:
    """Matched live timelines, with optional Optimization for legacy fixture compatibility."""

    shortage_level: str
    seed: int
    device: str
    frames_by_controller: Mapping[str, Sequence[LiveScenarioFrame]]
    metrics_by_controller: Mapping[str, LiveEpisodeSummary]
    policy_identity: LivePolicyIdentity | None = None

    def __post_init__(self) -> None:
        if self.policy_identity is not None and not isinstance(
            self.policy_identity, LivePolicyIdentity
        ):
            raise TypeError("policy_identity must be a LivePolicyIdentity")
        try:
            shortage = ShortageLevel(self.shortage_level)
        except (TypeError, ValueError) as error:
            raise ValueError("shortage_level must be mild, moderate, or severe") from error
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed < 2**63
        ):
            raise ValueError("seed must be between 0 and 2**63 - 1")
        if self.device != "cuda:0":
            raise ValueError("live comparison must originate from cuda:0")
        required = {ControllerName.PPO.value, ControllerName.FIXED_ROTATION.value}
        present = set(self.frames_by_controller)
        if not required <= present <= required | {ControllerName.OPTIMIZATION.value}:
            raise ValueError(
                "live comparison requires PPO, Fixed Rotation, and optional Optimization"
            )
        if set(self.metrics_by_controller) != present:
            raise ValueError("live metrics must contain the same controllers as the frames")
        names = tuple(member.value for member in ControllerName if member.value in present)

        normalized: dict[str, tuple[LiveScenarioFrame, ...]] = {}
        for name in names:
            frames = tuple(self.frames_by_controller[name])
            if len(frames) != 24 or tuple(frame.hour for frame in frames) != tuple(range(24)):
                raise ValueError(f"{name} must contain exactly one frame for each of 24 hours")
            normalized[name] = frames
        ppo_frames = normalized[ControllerName.PPO.value]
        for name in names[1:]:
            for ppo, other in zip(ppo_frames, normalized[name], strict=True):
                if ppo.demand_mw != other.demand_mw or ppo.supply_mw != other.supply_mw:
                    raise ValueError(
                        "all controllers must display the exact same demand and supply"
                    )
                if any(
                    ppo.locality_details[locality_id].demand_mw
                    != other.locality_details[locality_id].demand_mw
                    for locality_id in EXPECTED_LOCALITY_IDS
                ):
                    raise ValueError("all controllers must display the exact same locality demand")
                if any(
                    ppo.locality_details[locality_id].next_hour_projected_demand_mw
                    != other.locality_details[locality_id].next_hour_projected_demand_mw
                    for locality_id in EXPECTED_LOCALITY_IDS
                ):
                    raise ValueError("all controllers must display the same demand projection")

        object.__setattr__(self, "shortage_level", shortage.value)
        object.__setattr__(self, "frames_by_controller", MappingProxyType(normalized))
        metrics = {name: self.metrics_by_controller[name] for name in names}
        if any(not isinstance(value, LiveEpisodeSummary) for value in metrics.values()):
            raise TypeError("live metrics values must be LiveEpisodeSummary")
        object.__setattr__(self, "metrics_by_controller", MappingProxyType(metrics))

    def frames(self, controller_name: str) -> tuple[LiveScenarioFrame, ...]:
        """Return the immutable timeline for one approved controller."""

        try:
            return self.frames_by_controller[ControllerName(controller_name).value]
        except (KeyError, ValueError) as error:
            raise KeyError(controller_name) from error

    def metrics(self, controller_name: str) -> LiveEpisodeSummary:
        """Return aggregate metrics for one approved live controller run."""

        try:
            return self.metrics_by_controller[ControllerName(controller_name).value]
        except (KeyError, ValueError) as error:
            raise KeyError(controller_name) from error


def _nonnegative_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError(f"{field} must be finite and non-negative")
    return normalized


def _ratio(value: object, *, field: str) -> float:
    normalized = _nonnegative_number(value, field=field)
    if normalized > 1:
        raise ValueError(f"{field} must be between 0 and 1")
    return normalized

"""Proposal-aligned vocabulary and immutable tensor contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from numbers import Real

import torch

HORIZON_HOURS = 24


class Locality(StrEnum):
    """Khartoum State localities in the locked simulation order."""

    KHARTOUM = "Khartoum"
    OMDURMAN = "Omdurman"
    BAHRI = "Bahri"
    EAST_NILE = "East Nile"
    JEBEL_AWLIA = "Jebel Awlia"
    KARARI = "Karari"
    OMBADA = "Ombada"


class Sector(StrEnum):
    """Demand sectors in priority order, with critical services first."""

    HOSPITAL = "Hospital"
    WATER = "Water"
    BAKERY = "Bakery"
    RESIDENTIAL = "Residential"
    COMMERCIAL = "Commercial"


class ShortageLevel(StrEnum):
    """Public shortage choices supported by the presentation simulation."""

    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class ServiceTier(IntEnum):
    """Per-locality service action encoded for tensor dispatch."""

    OFF = 0
    CRITICAL_ONLY = 1
    FULL_SERVICE = 2

    @property
    def label(self) -> str:
        return ("Off", "Critical Only", "Full Service")[self.value]


class ControllerName(StrEnum):
    """Controllers included in the proposal comparison."""

    PPO = "PPO"
    FIXED_ROTATION = "Fixed Rotation"


LOCALITIES = tuple(Locality)
SECTORS = tuple(Sector)
CRITICAL_SECTOR_COUNT = 3
CRITICAL_SECTORS = SECTORS[:CRITICAL_SECTOR_COUNT]
LOCALITY_DEMAND_WEIGHTS = (1.00, 0.95, 0.85, 0.80, 0.70, 0.75, 0.90)
SECTOR_SHARES = (0.08, 0.07, 0.05, 0.55, 0.25)


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not 0 <= seed < 2**63:
        raise ValueError("seed must be between 0 and 2**63 - 1")


def _require_tensor(name: str, value: object) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    return value


def _validate_floating_power(name: str, value: torch.Tensor) -> None:
    if not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must contain only finite power values")
    if bool((value < 0).any().item()):
        raise ValueError(f"{name} must not contain negative power")


@dataclass(frozen=True, slots=True, eq=False)
class Scenario:
    """An immutable-reference batch of matched 24-hour demand and supply traces."""

    demand: torch.Tensor
    supply: torch.Tensor
    shortage_level: ShortageLevel
    seed: int

    def __post_init__(self) -> None:
        demand = _require_tensor("demand", self.demand)
        supply = _require_tensor("supply", self.supply)
        if demand.ndim != 4 or demand.shape[1:] != (
            HORIZON_HOURS,
            len(LOCALITIES),
            len(SECTORS),
        ):
            raise ValueError("demand must have shape [batch, 24, 7, 5]")
        if demand.shape[0] < 1:
            raise ValueError("demand batch must contain at least one scenario")
        expected_supply_shape = (demand.shape[0], HORIZON_HOURS)
        if supply.shape != expected_supply_shape:
            raise ValueError(f"supply must have shape {list(expected_supply_shape)}")
        _validate_floating_power("demand", demand)
        _validate_floating_power("supply", supply)
        if demand.device != supply.device:
            raise ValueError("demand and supply must remain on the same device")
        if demand.dtype != supply.dtype:
            raise ValueError("demand and supply must use the same dtype")
        try:
            shortage_level = ShortageLevel(self.shortage_level)
        except (TypeError, ValueError) as error:
            raise ValueError("shortage_level must be mild, moderate, or severe") from error
        object.__setattr__(self, "shortage_level", shortage_level)
        _validate_seed(self.seed)

    @property
    def batch_size(self) -> int:
        return self.demand.shape[0]

    @property
    def horizon_hours(self) -> int:
        return self.demand.shape[1]

    @property
    def demand_mw(self) -> torch.Tensor:
        """Demand tensor with shape ``[batch, hour, locality, sector]``."""

        return self.demand

    @property
    def available_supply_mw(self) -> torch.Tensor:
        """Available grid supply with shape ``[batch, hour]``."""

        return self.supply


@dataclass(frozen=True, slots=True, eq=False)
class SimulationResult:
    """Validated raw controller trace for one immutable scenario batch."""

    controller_name: ControllerName
    scenario: Scenario
    service_tiers: torch.Tensor
    requested: torch.Tensor
    served: torch.Tensor
    reward: torch.Tensor

    def __post_init__(self) -> None:
        try:
            controller_name = ControllerName(self.controller_name)
        except (TypeError, ValueError) as error:
            raise ValueError("controller_name must be PPO or Fixed Rotation") from error
        object.__setattr__(self, "controller_name", controller_name)
        if not isinstance(self.scenario, Scenario):
            raise TypeError("scenario must be a Scenario")

        tiers = _require_tensor("service_tiers", self.service_tiers)
        requested = _require_tensor("requested", self.requested)
        served = _require_tensor("served", self.served)
        reward = _require_tensor("reward", self.reward)
        batch_shape = (self.scenario.batch_size, HORIZON_HOURS)
        if tiers.shape != (*batch_shape, len(LOCALITIES)):
            raise ValueError("service_tiers must have shape [batch, 24, 7]")
        if tiers.dtype != torch.long:
            raise TypeError("service_tiers must use torch.long")
        expected_power_shape = self.scenario.demand.shape
        if requested.shape != expected_power_shape:
            raise ValueError("requested must have shape [batch, 24, 7, 5]")
        if served.shape != expected_power_shape:
            raise ValueError("served must have shape [batch, 24, 7, 5]")
        if reward.shape != batch_shape:
            raise ValueError("reward must have shape [batch, 24]")

        device = self.scenario.demand.device
        if any(tensor.device != device for tensor in (tiers, requested, served, reward)):
            raise ValueError("all result tensors must remain on the scenario device")
        dtype = self.scenario.demand.dtype
        if any(tensor.dtype != dtype for tensor in (requested, served, reward)):
            raise ValueError("power and reward tensors must use the scenario dtype")
        if not requested.is_floating_point():
            raise TypeError("requested must be a floating-point tensor")
        if not served.is_floating_point():
            raise TypeError("served must be a floating-point tensor")
        if not reward.is_floating_point():
            raise ValueError("reward must be a finite floating-point tensor")

        tolerance = 1e-4
        served_total = served.sum(dim=(-1, -2))
        failed_checks = torch.stack(
            (
                ((tiers < ServiceTier.OFF) | (tiers > ServiceTier.FULL_SERVICE)).any(),
                ~torch.isfinite(requested).all(),
                (requested < 0).any(),
                ~torch.isfinite(served).all(),
                (served < 0).any(),
                ~torch.isfinite(reward).all(),
                (requested > self.scenario.demand + tolerance).any(),
                (served > requested + tolerance).any(),
                (served_total > self.scenario.supply + tolerance).any(),
            )
        ).to(device="cpu")
        messages = (
            "service_tiers contains an invalid service tier",
            "requested must contain only finite power values",
            "requested must not contain negative power",
            "served must contain only finite power values",
            "served must not contain negative power",
            "reward must be a finite floating-point tensor",
            "requested power must not exceed scenario demand",
            "served power must not exceed requested power",
            "served power must not exceed available supply",
        )
        for failed, message in zip(failed_checks.tolist(), messages, strict=True):
            if failed:
                raise ValueError(message)

    @property
    def requested_mw(self) -> torch.Tensor:
        return self.requested

    @property
    def served_mw(self) -> torch.Tensor:
        return self.served


@dataclass(frozen=True, slots=True)
class Metrics:
    """Proposal metric record for one controller/scenario result."""

    episode_reward: float
    demand_mwh: float
    available_mwh: float
    served_mwh: float
    curtailed_mwh: float
    critical_service_ratio: float
    fairness: float
    utilization: float
    oversubscription: float
    switching_rate: float
    power_balance_violation_rate: float
    critical_outage_locality_hours: float

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number")
            normalized = float(value)
            if not math.isfinite(normalized):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, normalized)

        nonnegative = (
            "demand_mwh",
            "available_mwh",
            "served_mwh",
            "curtailed_mwh",
            "critical_outage_locality_hours",
        )
        for name in nonnegative:
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative")

        ratios = (
            "critical_service_ratio",
            "fairness",
            "utilization",
            "oversubscription",
            "switching_rate",
            "power_balance_violation_rate",
        )
        for name in ratios:
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")

        energy_tolerance = max(1e-3, 1e-7 * self.demand_mwh)
        if abs(self.demand_mwh - self.served_mwh - self.curtailed_mwh) > energy_tolerance:
            raise ValueError("demand, served, and curtailed MWh violate the energy identity")
        if self.available_mwh > self.demand_mwh + energy_tolerance:
            raise ValueError("available_mwh must not exceed shortage demand_mwh")
        if self.served_mwh > min(self.demand_mwh, self.available_mwh) + energy_tolerance:
            raise ValueError("served_mwh must not exceed demand or available energy")
        if self.critical_outage_locality_hours > HORIZON_HOURS * len(LOCALITIES):
            raise ValueError(
                "critical_outage_locality_hours must not exceed the 24-hour locality horizon"
            )

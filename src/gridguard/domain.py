"""Proposal-aligned vocabulary and immutable tensor contracts."""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from numbers import Real

import torch

HORIZON_HOURS = 24
NUMERIC_EPSILON = 1e-8


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
    OPTIMIZATION = "Optimization"


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


def _validate_shortage_supply(demand: torch.Tensor, supply: torch.Tensor) -> None:
    """Reduce on-device and synchronize only the one fail-closed boundary predicate."""

    exceeds_total_demand = (supply > demand.sum(dim=(-1, -2))).any()
    if bool(exceeds_total_demand.item()):
        raise ValueError(
            "scenario must model a shortage: supply must not exceed total demand in any batch hour"
        )


@dataclass(frozen=True, slots=True)
class _TensorMutationSnapshot:
    """Original tensor identity, storage contract, version, and device-resident bytes."""

    tensor: torch.Tensor
    expected: torch.Tensor
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    version: int | None
    data_ptr: int
    storage_data_ptr: int
    storage_offset: int
    stride: tuple[int, ...]
    layout: torch.layout
    requires_grad: bool


def _capture_tensor_mutation_snapshot(name: str, tensor: torch.Tensor) -> _TensorMutationSnapshot:
    try:
        version = tensor._version
    except RuntimeError:
        version = None
    return _TensorMutationSnapshot(
        tensor=tensor,
        expected=tensor.detach().contiguous().clone(),
        shape=tuple(tensor.shape),
        dtype=tensor.dtype,
        device=torch.device(tensor.device),
        version=version,
        data_ptr=tensor.data_ptr(),
        storage_data_ptr=tensor.untyped_storage().data_ptr(),
        storage_offset=tensor.storage_offset(),
        stride=tensor.stride(),
        layout=tensor.layout,
        requires_grad=tensor.requires_grad,
    )


def _changed_tensor_fields(
    fields: tuple[tuple[str, object, _TensorMutationSnapshot], ...],
) -> tuple[str, ...]:
    changed: list[str] = []
    comparable: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    for name, current, snapshot in fields:
        if not isinstance(current, torch.Tensor):
            changed.append(name)
            continue
        try:
            version = current._version
        except RuntimeError:
            version = None
        if (
            current is not snapshot.tensor
            or tuple(current.shape) != snapshot.shape
            or current.dtype != snapshot.dtype
            or current.device != snapshot.device
            or current.layout != snapshot.layout
            or current.stride() != snapshot.stride
            or current.storage_offset() != snapshot.storage_offset
            or current.data_ptr() != snapshot.data_ptr
            or current.untyped_storage().data_ptr() != snapshot.storage_data_ptr
            or current.requires_grad != snapshot.requires_grad
            or version != snapshot.version
        ):
            changed.append(name)
        else:
            comparable.append((name, current, snapshot.expected))

    if comparable:
        byte_changes = torch.stack(
            tuple(
                torch.ne(
                    current.detach().contiguous().view(torch.uint8),
                    expected.view(torch.uint8),
                ).any()
                for _, current, expected in comparable
            )
        ).to(device="cpu")
        changed.extend(
            name
            for (name, _, _), was_changed in zip(
                comparable,
                byte_changes.tolist(),
                strict=True,
            )
            if was_changed
        )
    return tuple(name for name, _, _ in fields if name in changed)


@dataclass(frozen=True, slots=True)
class ScenarioMutationGuard:
    """Snapshot detecting ordinary PyTorch writes or tensor/storage replacement."""

    demand: _TensorMutationSnapshot
    supply: _TensorMutationSnapshot
    shortage_level: ShortageLevel
    seed: int

    @property
    def device(self) -> torch.device:
        return self.demand.expected.device

    def changed_fields(self, scenario: Scenario) -> tuple[str, ...]:
        """Return tensor fields whose shape, storage contract, or bytes changed."""

        if not isinstance(scenario, Scenario):
            raise TypeError("scenario must be a Scenario")
        if scenario.shortage_level != self.shortage_level or scenario.seed != self.seed:
            raise ValueError("scenario identity changed after the mutation guard was captured")

        return _changed_tensor_fields(
            (
                ("demand", scenario.demand, self.demand),
                ("supply", scenario.supply, self.supply),
            )
        )


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
        _validate_shortage_supply(demand, supply)
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

    def validate_shortage_invariant(self) -> None:
        """Recheck the mutable tensor payload before controller execution."""

        _validate_shortage_supply(self.demand, self.supply)

    def capture_mutation_guard(self) -> ScenarioMutationGuard:
        """Clone demand and supply on their current device for matched-use checks."""

        return ScenarioMutationGuard(
            demand=_capture_tensor_mutation_snapshot("demand", self.demand),
            supply=_capture_tensor_mutation_snapshot("supply", self.supply),
            shortage_level=self.shortage_level,
            seed=self.seed,
        )


def canonical_scenario_sha256(scenario: Scenario, *, batch_index: int) -> str:
    """Hash one CPU-resident trace in a stable identity-and-float64 byte format."""

    if not isinstance(scenario, Scenario):
        raise TypeError("scenario must be a Scenario")
    if isinstance(batch_index, bool) or not isinstance(batch_index, int):
        raise TypeError("batch_index must be an integer")
    if not 0 <= batch_index < scenario.batch_size:
        raise ValueError("batch_index is outside the scenario batch")
    if scenario.demand.device.type != "cpu":
        raise ValueError(
            "canonical scenario hashing requires tensors after the final host transfer"
        )

    shortage = scenario.shortage_level.value.encode("ascii")
    digest = hashlib.sha256()
    digest.update(b"gridguard.scenario-trace.sha256.v1\0")
    digest.update(struct.pack("<B", len(shortage)))
    digest.update(shortage)
    digest.update(struct.pack("<QI", scenario.seed, batch_index))
    for label, values in (
        (b"demand_mw", scenario.demand[batch_index]),
        (b"supply_mw", scenario.supply[batch_index]),
    ):
        canonical = values.detach().contiguous().to(dtype=torch.float64).reshape(-1)
        digest.update(struct.pack("<B", len(label)))
        digest.update(label)
        digest.update(struct.pack("<I", canonical.numel()))
        digest.update(struct.pack(f"<{canonical.numel()}d", *canonical.tolist()))
    return digest.hexdigest()


def _reward_jain_fairness(service_ratio: torch.Tensor) -> torch.Tensor:
    return service_ratio.sum(dim=-1).square() / (
        len(LOCALITIES) * service_ratio.square().sum(dim=-1) + NUMERIC_EPSILON
    )


def canonical_step_reward(
    *,
    hour: int,
    demand: torch.Tensor,
    supply: torch.Tensor,
    requested: torch.Tensor,
    served: torch.Tensor,
    tiers: torch.Tensor,
    previous_tier: torch.Tensor,
    next_cumulative_demand: torch.Tensor,
    next_cumulative_served: torch.Tensor,
) -> torch.Tensor:
    """Calculate the locked five-component training-objective reward for one hour."""

    demand_total = demand.sum(dim=(-1, -2))
    served_total = served.sum(dim=(-1, -2))
    critical_demand = demand[..., :CRITICAL_SECTOR_COUNT].sum(dim=(-1, -2))
    critical_served = served[..., :CRITICAL_SECTOR_COUNT].sum(dim=(-1, -2))
    critical_service = critical_served / critical_demand.clamp_min(NUMERIC_EPSILON)
    utilization = served_total / torch.minimum(supply, demand_total).clamp_min(NUMERIC_EPSILON)
    service_ratio = next_cumulative_served / next_cumulative_demand.clamp_min(NUMERIC_EPSILON)
    fairness = _reward_jain_fairness(service_ratio)
    switching = (
        torch.zeros_like(supply)
        if hour == 0
        else (tiers != previous_tier).to(demand.dtype).mean(dim=-1)
    )
    requested_total = requested.sum(dim=(-1, -2))
    oversubscription = torch.relu(requested_total - supply) / demand_total.clamp_min(
        NUMERIC_EPSILON
    )
    return (
        5.0 * critical_service
        + 2.0 * utilization
        + 2.0 * fairness
        - 0.25 * switching
        - 3.0 * oversubscription
    ) / 9.0


@torch.inference_mode()
def canonical_reward_trace(
    *,
    scenario: Scenario,
    service_tiers: torch.Tensor,
    requested: torch.Tensor,
    served: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct the canonical reward trace from a validated simulation trace."""

    cumulative_demand = torch.zeros(
        (scenario.batch_size, len(LOCALITIES)),
        dtype=scenario.demand.dtype,
        device=scenario.demand.device,
    )
    cumulative_served = torch.zeros_like(cumulative_demand)
    previous_tier = torch.zeros(
        (scenario.batch_size, len(LOCALITIES)),
        dtype=torch.long,
        device=scenario.demand.device,
    )
    reward_trace: list[torch.Tensor] = []
    for hour in range(HORIZON_HOURS):
        demand = scenario.demand[:, hour]
        hour_served = served[:, hour]
        next_cumulative_demand = cumulative_demand + demand.sum(dim=-1)
        next_cumulative_served = cumulative_served + hour_served.sum(dim=-1)
        tiers = service_tiers[:, hour]
        reward_trace.append(
            canonical_step_reward(
                hour=hour,
                demand=demand,
                supply=scenario.supply[:, hour],
                requested=requested[:, hour],
                served=hour_served,
                tiers=tiers,
                previous_tier=previous_tier,
                next_cumulative_demand=next_cumulative_demand,
                next_cumulative_served=next_cumulative_served,
            )
        )
        cumulative_demand = next_cumulative_demand
        cumulative_served = next_cumulative_served
        previous_tier = tiers
    return torch.stack(reward_trace, dim=1)


@dataclass(frozen=True, slots=True)
class SimulationResultMutationGuard:
    """Device-resident snapshot binding a validated result to its original trace."""

    scenario: Scenario
    scenario_guard: ScenarioMutationGuard
    controller_name: ControllerName
    service_tiers: _TensorMutationSnapshot
    requested: _TensorMutationSnapshot
    served: _TensorMutationSnapshot
    reward: _TensorMutationSnapshot

    @classmethod
    def capture(cls, result: SimulationResult) -> SimulationResultMutationGuard:
        """Capture scenario identity and exact result tensor contracts and bytes."""

        return cls(
            scenario=result.scenario,
            scenario_guard=result.scenario.capture_mutation_guard(),
            controller_name=result.controller_name,
            service_tiers=_capture_tensor_mutation_snapshot("service_tiers", result.service_tiers),
            requested=_capture_tensor_mutation_snapshot("requested", result.requested),
            served=_capture_tensor_mutation_snapshot("served", result.served),
            reward=_capture_tensor_mutation_snapshot("reward", result.reward),
        )

    def changed_fields(self, result: SimulationResult) -> tuple[str, ...]:
        """Return identity, metadata, version-history, or byte-level changes."""

        if not isinstance(result, SimulationResult):
            raise TypeError("result must be a SimulationResult")
        changed: list[str] = []
        if result.controller_name != self.controller_name:
            changed.append("controller_name")

        scenario = result.scenario
        scenario_is_valid = isinstance(scenario, Scenario)
        if (
            not scenario_is_valid
            or scenario is not self.scenario
            or scenario.shortage_level != self.scenario_guard.shortage_level
            or scenario.seed != self.scenario_guard.seed
        ):
            changed.append("scenario")

        tensor_fields: list[tuple[str, object, _TensorMutationSnapshot]] = []
        if scenario_is_valid:
            tensor_fields.extend(
                (
                    ("scenario.demand", scenario.demand, self.scenario_guard.demand),
                    ("scenario.supply", scenario.supply, self.scenario_guard.supply),
                )
            )
        tensor_fields.extend(
            (
                ("service_tiers", result.service_tiers, self.service_tiers),
                ("requested", result.requested, self.requested),
                ("served", result.served, self.served),
                ("reward", result.reward, self.reward),
            )
        )
        changed.extend(_changed_tensor_fields(tuple(tensor_fields)))
        return tuple(dict.fromkeys(changed))


@dataclass(frozen=True, slots=True, eq=False)
class SimulationResult:
    """Validated raw controller trace for one immutable scenario batch."""

    controller_name: ControllerName
    scenario: Scenario
    service_tiers: torch.Tensor
    requested: torch.Tensor
    served: torch.Tensor
    reward: torch.Tensor
    _mutation_guard: SimulationResultMutationGuard = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        inference_trace_names = tuple(
            name
            for name in ("service_tiers", "requested", "served", "reward")
            if isinstance(getattr(self, name), torch.Tensor)
            and torch.is_inference(getattr(self, name))
        )
        if inference_trace_names:
            with torch.inference_mode(False):
                versioned = tuple(getattr(self, name).clone() for name in inference_trace_names)
            for name, tensor in zip(inference_trace_names, versioned, strict=True):
                object.__setattr__(self, name, tensor)
        self.validate_trace_invariants()
        object.__setattr__(self, "_mutation_guard", SimulationResultMutationGuard.capture(self))

    @torch.inference_mode()
    def validate_trace_invariants(self) -> torch.Tensor:
        """Validate the complete trace and return its reconstructed canonical reward."""

        mutation_guard = getattr(self, "_mutation_guard", None)
        if mutation_guard is not None:
            changed = mutation_guard.changed_fields(self)
            if changed:
                raise ValueError(
                    "simulation result changed after validation: " + ", ".join(changed)
                )
            return self.reward

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
        expected_critical_request = (
            self.scenario.demand[..., :CRITICAL_SECTOR_COUNT]
            * (tiers >= ServiceTier.CRITICAL_ONLY)[..., None]
        )
        expected_noncritical_request = (
            self.scenario.demand[..., CRITICAL_SECTOR_COUNT:]
            * (tiers >= ServiceTier.FULL_SERVICE)[..., None]
        )
        expected_requested = torch.cat(
            (expected_critical_request, expected_noncritical_request), dim=-1
        )
        requested_total = requested.sum(dim=(-1, -2))
        dispatch_scale = torch.minimum(
            torch.ones_like(self.scenario.supply),
            self.scenario.supply / requested_total.clamp_min(NUMERIC_EPSILON),
        )
        expected_served = requested * dispatch_scale[..., None, None]
        expected_reward = canonical_reward_trace(
            scenario=self.scenario,
            service_tiers=tiers,
            requested=requested,
            served=served,
        )
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
                ~torch.eq(requested, expected_requested).all(),
                ~torch.eq(served, expected_served).all(),
                ~torch.eq(reward, expected_reward).all(),
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
            "requested power must match the selected service-tier requests",
            "served power must match the proportional dispatch rule",
            "reward must match the canonical training-objective reward",
        )
        for failed, message in zip(failed_checks.tolist(), messages, strict=True):
            if failed:
                raise ValueError(message)
        return expected_reward

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

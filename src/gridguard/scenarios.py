"""Deterministic synthetic demand and shortage scenario generation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import torch

from gridguard.domain import (
    HORIZON_HOURS,
    LOCALITIES,
    LOCALITY_DEMAND_WEIGHTS,
    SECTOR_SHARES,
    Scenario,
    ShortageLevel,
)

DEFAULT_DEVICE = "cuda:0"
SYSTEM_PEAK_MW = 700.0
DEFAULT_DEMAND_NOISE = 0.05
DEFAULT_SUPPLY_NOISE = 0.05


@dataclass(frozen=True, slots=True, eq=False)
class ScenarioBaseDraws:
    """Immutable references to one batch of severity-neutral stochastic draws."""

    demand_uniform: torch.Tensor
    supply_uniform: torch.Tensor
    seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.demand_uniform, torch.Tensor):
            raise TypeError("demand_uniform must be a torch.Tensor")
        if not isinstance(self.supply_uniform, torch.Tensor):
            raise TypeError("supply_uniform must be a torch.Tensor")
        if self.demand_uniform.ndim != 4 or self.demand_uniform.shape[1:] != (
            HORIZON_HOURS,
            len(LOCALITIES),
            len(SECTOR_SHARES),
        ):
            raise ValueError("demand_uniform must have shape [batch, 24, 7, 5]")
        if self.demand_uniform.shape[0] < 1:
            raise ValueError("base draws must contain at least one scenario")
        expected_supply_shape = (self.demand_uniform.shape[0], HORIZON_HOURS)
        if self.supply_uniform.shape != expected_supply_shape:
            raise ValueError(f"supply_uniform must have shape {list(expected_supply_shape)}")
        if self.demand_uniform.dtype != torch.float32 or self.supply_uniform.dtype != torch.float32:
            raise TypeError("scenario base draws must use torch.float32")
        if self.demand_uniform.device != self.supply_uniform.device:
            raise ValueError("scenario base draws must remain on one device")
        draws = (self.demand_uniform, self.supply_uniform)
        if any(not bool(torch.isfinite(value).all().item()) for value in draws):
            raise ValueError("scenario base draws must contain only finite values")
        if any(bool(((value < 0.0) | (value >= 1.0)).any().item()) for value in draws):
            raise ValueError("scenario base draws must be in the interval [0, 1)")
        _validate_seed(self.seed)

    @property
    def batch_size(self) -> int:
        return self.demand_uniform.shape[0]

    @property
    def device(self) -> torch.device:
        return self.demand_uniform.device


def _validate_batch_size(batch_size: int) -> None:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not 0 <= seed < 2**63:
        raise ValueError("seed must be between 0 and 2**63 - 1")


def _validate_noise(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized < 1.0:
        raise ValueError(f"{name} must be finite and in the interval [0, 1)")
    return normalized


def _resolve_device(device: str | torch.device) -> torch.device:
    try:
        resolved = torch.device(device)
    except (RuntimeError, TypeError) as error:
        raise ValueError(f"device is not a valid PyTorch device: {device!r}") from error
    if resolved.type not in {"cpu", "cuda"}:
        raise ValueError("device must select CPU or CUDA")
    return resolved


def _gaussian(hours: torch.Tensor, center: float, width: float) -> torch.Tensor:
    return torch.exp(-0.5 * ((hours - center) / width).square())


def _sector_profiles(device: torch.device) -> torch.Tensor:
    hours = torch.arange(HORIZON_HOURS, device=device, dtype=torch.float32)
    hospital = torch.full_like(hours, 0.95)
    water = 0.45 + 0.35 * _gaussian(hours, 6.0, 2.0) + 0.40 * _gaussian(hours, 18.0, 2.5)
    bakery = 0.35 + 0.60 * _gaussian(hours, 5.0, 1.5) + 0.45 * _gaussian(hours, 16.0, 2.0)
    residential = 0.40 + 0.30 * _gaussian(hours, 7.0, 2.0) + 0.65 * _gaussian(hours, 20.0, 3.0)
    commercial = 0.25 + 0.75 * _gaussian(hours, 13.0, 4.0)
    profiles = torch.stack((hospital, water, bakery, residential, commercial), dim=-1)
    return profiles / profiles.amax(dim=0, keepdim=True)


def _supply_fraction(
    batch_size: int,
    shortage_level: ShortageLevel,
    device: torch.device,
) -> torch.Tensor:
    base, start, duration, depth = {
        ShortageLevel.MILD: (0.80, 0, 0, 0.0),
        ShortageLevel.MODERATE: (0.65, 17, 4, 0.10),
        ShortageLevel.SEVERE: (0.50, 16, 6, 0.15),
    }[shortage_level]
    fraction = torch.full(
        (batch_size, HORIZON_HOURS),
        base,
        dtype=torch.float32,
        device=device,
    )
    if duration:
        hours = torch.arange(HORIZON_HOURS, device=device)
        shortage_window = (hours >= start) & (hours < start + duration)
        # Subtraction preserves exact float32 parity with the accepted source.
        fraction[:, shortage_window] -= depth
    return fraction


def generate_scenario_base_draws(
    batch_size: int,
    seed: int,
    *,
    device: str | torch.device = DEFAULT_DEVICE,
) -> ScenarioBaseDraws:
    """Draw one severity-neutral stochastic batch for reuse across shortage levels."""

    _validate_batch_size(batch_size)
    _validate_seed(seed)
    resolved_device = _resolve_device(device)
    try:
        generator = torch.Generator(device=resolved_device).manual_seed(seed)
    except RuntimeError as error:
        raise RuntimeError(
            f"could not initialize the scenario generator on {resolved_device}: {error}"
        ) from error
    demand_uniform = torch.rand(
        (batch_size, HORIZON_HOURS, len(LOCALITIES), len(SECTOR_SHARES)),
        dtype=torch.float32,
        device=resolved_device,
        generator=generator,
    )
    supply_uniform = torch.rand(
        (batch_size, HORIZON_HOURS),
        dtype=torch.float32,
        device=resolved_device,
        generator=generator,
    )
    return ScenarioBaseDraws(
        demand_uniform=demand_uniform,
        supply_uniform=supply_uniform,
        seed=seed,
    )


def scenario_from_base_draws(
    base_draws: ScenarioBaseDraws,
    shortage_level: ShortageLevel | str,
    *,
    demand_noise: float = DEFAULT_DEMAND_NOISE,
    supply_noise: float = DEFAULT_SUPPLY_NOISE,
) -> Scenario:
    """Purely transform shared stochastic draws into one shortage-level scenario."""

    if not isinstance(base_draws, ScenarioBaseDraws):
        raise TypeError("base_draws must be ScenarioBaseDraws")
    try:
        normalized_shortage = ShortageLevel(shortage_level)
    except (TypeError, ValueError) as error:
        raise ValueError("shortage_level must be mild, moderate, or severe") from error
    normalized_demand_noise = _validate_noise("demand_noise", demand_noise)
    normalized_supply_noise = _validate_noise("supply_noise", supply_noise)

    locality_weights = torch.tensor(
        LOCALITY_DEMAND_WEIGHTS,
        dtype=torch.float32,
        device=base_draws.device,
    )
    locality_peaks = SYSTEM_PEAK_MW * locality_weights / locality_weights.sum()
    sector_shares = torch.tensor(SECTOR_SHARES, dtype=torch.float32, device=base_draws.device)
    base_demand = locality_peaks[:, None] * sector_shares[None, :]
    profiles = _sector_profiles(base_draws.device)

    demand = base_demand[None, None, :, :] * profiles[None, :, None, :]
    demand = demand.expand(base_draws.batch_size, -1, len(LOCALITIES), -1).clone()
    demand_multiplier = 1.0 + normalized_demand_noise * (2.0 * base_draws.demand_uniform - 1.0)
    demand *= demand_multiplier

    fraction = _supply_fraction(base_draws.batch_size, normalized_shortage, base_draws.device)
    supply_multiplier = 1.0 + normalized_supply_noise * (2.0 * base_draws.supply_uniform - 1.0)
    total_demand = demand.sum(dim=(-1, -2))
    supply = total_demand * fraction * supply_multiplier
    supply = torch.minimum(supply, total_demand * 0.95)

    return Scenario(
        demand=demand,
        supply=supply,
        shortage_level=normalized_shortage,
        seed=base_draws.seed,
    )


def generate_scenarios(
    batch_size: int,
    seed: int,
    shortage_level: ShortageLevel | str,
    *,
    device: str | torch.device = DEFAULT_DEVICE,
    demand_noise: float = DEFAULT_DEMAND_NOISE,
    supply_noise: float = DEFAULT_SUPPLY_NOISE,
) -> Scenario:
    """Generate a matched batch on ``device`` without leaving the PyTorch hot path."""

    _validate_batch_size(batch_size)
    _validate_seed(seed)
    try:
        normalized_shortage = ShortageLevel(shortage_level)
    except (TypeError, ValueError) as error:
        raise ValueError("shortage_level must be mild, moderate, or severe") from error
    normalized_demand_noise = _validate_noise("demand_noise", demand_noise)
    normalized_supply_noise = _validate_noise("supply_noise", supply_noise)
    resolved_device = _resolve_device(device)

    base_draws = generate_scenario_base_draws(batch_size, seed, device=resolved_device)
    return scenario_from_base_draws(
        base_draws,
        normalized_shortage,
        demand_noise=normalized_demand_noise,
        supply_noise=normalized_supply_noise,
    )


def generate_scenario(
    seed: int,
    shortage_level: ShortageLevel | str,
    *,
    device: str | torch.device = DEFAULT_DEVICE,
    demand_noise: float = DEFAULT_DEMAND_NOISE,
    supply_noise: float = DEFAULT_SUPPLY_NOISE,
) -> Scenario:
    """Generate one scenario while retaining the standard leading batch dimension."""

    return generate_scenarios(
        1,
        seed,
        shortage_level,
        device=device,
        demand_noise=demand_noise,
        supply_noise=supply_noise,
    )

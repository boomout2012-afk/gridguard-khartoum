"""Controller contracts for the proposal-aligned simulation."""

from __future__ import annotations

from dataclasses import InitVar, dataclass
from typing import ClassVar, Protocol, runtime_checkable

import torch

from gridguard.domain import HORIZON_HOURS, LOCALITIES, ControllerName, ServiceTier

OBSERVATION_DIM = 32


@dataclass(frozen=True, slots=True, eq=False)
class DecisionContext:
    """The causal information available to a controller at one simulation hour."""

    hour: int
    observation: torch.Tensor
    _validate_finite: InitVar[bool] = True
    current_demand_mw: torch.Tensor | None = None
    current_supply_mw: torch.Tensor | None = None
    cumulative_demand_mwh: torch.Tensor | None = None
    cumulative_served_mwh: torch.Tensor | None = None
    previous_service_tiers: torch.Tensor | None = None

    def __post_init__(self, _validate_finite: bool) -> None:
        if isinstance(self.hour, bool) or not isinstance(self.hour, int):
            raise TypeError("hour must be an integer")
        if not 0 <= self.hour < HORIZON_HOURS:
            raise ValueError("hour must be between 0 and 23")
        if not isinstance(self.observation, torch.Tensor):
            raise TypeError("observation must be a torch.Tensor")
        if self.observation.ndim != 2 or self.observation.shape[1] != OBSERVATION_DIM:
            raise ValueError("observation must have shape [batch, 32]")
        if self.observation.shape[0] < 1:
            raise ValueError("observation batch must contain at least one scenario")
        if not self.observation.is_floating_point():
            raise TypeError("observation must be a floating-point tensor")
        if _validate_finite and not bool(torch.isfinite(self.observation).all().item()):
            raise ValueError("observation must contain only finite values")
        state = (
            self.current_demand_mw,
            self.current_supply_mw,
            self.cumulative_demand_mwh,
            self.cumulative_served_mwh,
            self.previous_service_tiers,
        )
        if all(value is None for value in state):
            return
        if any(value is None for value in state):
            raise ValueError("all dispatch-state tensors must be provided together")
        batch = self.observation.shape[0]
        shapes = ((batch, 7, 5), (batch,), (batch, 7), (batch, 7), (batch, 7))
        for index, (value, shape) in enumerate(zip(state, shapes, strict=True)):
            if not isinstance(value, torch.Tensor) or value.shape != shape:
                raise ValueError(f"dispatch-state tensor {index} must have shape {shape}")
            if value.device != self.observation.device:
                raise ValueError("dispatch-state tensors must share the observation device")
            dtype = torch.long if index == 4 else self.observation.dtype
            if value.dtype != dtype:
                raise TypeError("dispatch-state tensors must use the declared dtype")
        if _validate_finite:
            floating = torch.cat(tuple(value.flatten() for value in state[:4]))
            if bool((~torch.isfinite(floating) | (floating < 0)).any().item()):
                raise ValueError("dispatch-state power must be finite and non-negative")
            if bool((self.cumulative_served_mwh > self.cumulative_demand_mwh).any().item()):
                raise ValueError("cumulative served energy cannot exceed cumulative demand")
            if bool(
                ((self.previous_service_tiers < 0) | (self.previous_service_tiers > 2)).any().item()
            ):
                raise ValueError("previous service tiers must be between 0 and 2")


@runtime_checkable
class Controller(Protocol):
    """Structural interface shared by presentation controllers."""

    @property
    def name(self) -> ControllerName: ...

    def act(self, context: DecisionContext) -> torch.Tensor: ...


@runtime_checkable
class PPOInferencePolicy(Protocol):
    """The narrow policy surface needed by the simulator."""

    def act(self, observation: torch.Tensor) -> torch.Tensor: ...


@runtime_checkable
class PrevalidatedPPOInferencePolicy(Protocol):
    """Optional fast path for observations validated by the simulation boundary."""

    def act_prevalidated(self, observation: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True, slots=True)
class FixedRotationController:
    """Accepted three-hour rotation with three full-service localities per block."""

    name: ClassVar[ControllerName] = ControllerName.FIXED_ROTATION
    active_localities: ClassVar[int] = 3
    block_hours: ClassVar[int] = 3

    @torch.inference_mode()
    def act(self, context: DecisionContext) -> torch.Tensor:
        locality = torch.arange(len(LOCALITIES), device=context.observation.device)
        start = (context.hour // self.block_hours * self.active_localities) % len(LOCALITIES)
        active = (locality - start) % len(LOCALITIES) < self.active_localities
        action = torch.where(
            active,
            int(ServiceTier.FULL_SERVICE),
            int(ServiceTier.OFF),
        ).to(torch.long)
        return action.unsqueeze(0).expand(context.observation.shape[0], -1).clone()


@dataclass(frozen=True, slots=True)
class PPOController:
    """Adapt an explicitly loaded PPO policy to the simulation controller contract."""

    policy: PPOInferencePolicy
    name: ClassVar[ControllerName] = ControllerName.PPO

    @torch.inference_mode()
    def act(self, context: DecisionContext) -> torch.Tensor:
        if isinstance(self.policy, PrevalidatedPPOInferencePolicy):
            return self.policy.act_prevalidated(context.observation)
        return self.policy.act(context.observation)

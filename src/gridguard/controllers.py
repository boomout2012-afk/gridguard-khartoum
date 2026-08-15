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

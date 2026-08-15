"""Deterministic causal demand projection for the presentation display only.

This projection is not an input to PPO dispatch and is not part of benchmark evidence.
"""

from __future__ import annotations

import torch

from gridguard.domain import LOCALITIES, SECTORS

DEFAULT_WINDOW_HOURS = 3
DEFAULT_HORIZON_HOURS = 1


def _validate_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_demand(demand_mw: object) -> torch.Tensor:
    if not isinstance(demand_mw, torch.Tensor):
        raise TypeError("demand_mw must be a torch.Tensor")
    if demand_mw.ndim != 4 or demand_mw.shape[2:] != (
        len(LOCALITIES),
        len(SECTORS),
    ):
        raise ValueError("demand_mw must have shape [batch, hour, 7, 5]")
    if demand_mw.shape[0] < 1:
        raise ValueError("demand_mw batch must contain at least one trace")
    if not demand_mw.is_floating_point():
        raise TypeError("demand_mw must be a floating-point tensor")
    if not bool(torch.isfinite(demand_mw).all().item()):
        raise ValueError("demand_mw must contain only finite power values")
    if bool((demand_mw < 0).any().item()):
        raise ValueError("demand_mw must not contain negative power")
    return demand_mw


@torch.inference_mode()
def causal_demand_projection(
    demand_mw: torch.Tensor,
    *,
    window_hours: int = DEFAULT_WINDOW_HOURS,
    horizon_hours: int = DEFAULT_HORIZON_HOURS,
) -> torch.Tensor:
    """Project demand without allowing target or future samples into their estimate.

    ``demand_mw`` has shape ``[batch, hour, locality, sector]``. Each returned
    time step is the trailing-window mean that predicts a target
    ``horizon_hours`` later. The first returned value therefore targets input
    hour ``window_hours + horizon_hours - 1``. The result stays on the input
    tensor's device and preserves its floating-point dtype.

    The default is a one-hour projection intended only for the desktop
    presentation. PPO decisions and frozen evaluation evidence do not consume
    this output.
    """

    demand = _validate_demand(demand_mw)
    _validate_positive_integer("window_hours", window_hours)
    _validate_positive_integer("horizon_hours", horizon_hours)

    history_hours = demand.shape[1]
    if window_hours + horizon_hours > history_hours:
        raise ValueError(
            "window_hours + horizon_hours must not exceed the available demand history"
        )

    valid_target_count = history_hours - window_hours - horizon_hours + 1
    rolling_windows = demand.unfold(dimension=1, size=window_hours, step=1)
    return rolling_windows[:, :valid_target_count].mean(dim=-1)

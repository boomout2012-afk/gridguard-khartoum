"""A separate, causal learned forecast experiment for the scientific supplement.

This module never changes the desktop projection or frozen benchmark. Ridge AR(3)
learns 35 independent locality-sector regressions from training traces only. Each
forecast uses the preceding three demand observations, with no future values,
hour-of-day features, supply, controller decisions, or scenario identifiers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Final

import torch

from gridguard.domain import HORIZON_HOURS, LOCALITIES, SECTORS

FORECAST_TRAINING_SEEDS: Final = (42, 43, 44)
FORECAST_METHOD_NAMES: Final = ("trailing_mean_3h", "persistence_1h", "ridge_ar3")
FORECAST_TARGET_HOURS: Final = tuple(range(3, HORIZON_HOURS))
DEFAULT_RIDGE_PENALTY: Final = 1e-3
FEATURE_SCALE_FLOOR_MW: Final = 1e-8
_CELL_SHAPE: Final = (len(LOCALITIES), len(SECTORS))


def _validate_seeds(seeds: tuple[int, ...], *, field: str) -> None:
    if not isinstance(seeds, tuple) or not seeds:
        raise ValueError(f"{field} must be a nonempty tuple of seeds")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise TypeError(f"{field} must contain integer seeds")
    if any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError(f"{field} must contain unique non-negative seeds")


def _validate_training_seeds(seeds: tuple[int, ...]) -> None:
    _validate_seeds(seeds, field="training_seeds")
    if not set(seeds).issubset(FORECAST_TRAINING_SEEDS):
        raise ValueError("training_seeds must use only documented candidate seeds 42, 43, 44")


def _validate_penalty(ridge_penalty: float) -> None:
    if isinstance(ridge_penalty, bool) or not isinstance(ridge_penalty, Real):
        raise TypeError("ridge_penalty must be a positive finite number")
    if not math.isfinite(ridge_penalty) or ridge_penalty <= 0:
        raise ValueError("ridge_penalty must be a positive finite number")


def _validate_demand_cube(demand_mw: object, *, seeds: tuple[int, ...], field: str) -> torch.Tensor:
    _validate_seeds(seeds, field=f"{field}_seeds")
    if not isinstance(demand_mw, torch.Tensor):
        raise TypeError(f"{field}_demand_mw must be a torch.Tensor")
    if (
        demand_mw.ndim != 5
        or demand_mw.shape[0] != len(seeds)
        or demand_mw.shape[1] < 1
        or demand_mw.shape[2:] != (HORIZON_HOURS, *_CELL_SHAPE)
    ):
        raise ValueError(f"{field} demand must have shape [len(seeds), batch, 24, 7, 5]")
    if not demand_mw.is_floating_point():
        raise TypeError(f"{field} demand must use a floating-point dtype")
    if demand_mw.device.type != "cuda":
        raise ValueError(f"{field} demand must remain on CUDA; CPU fallback is disabled")
    if not bool(torch.isfinite(demand_mw).all().item()):
        raise ValueError(f"{field} demand must be finite")
    if bool((demand_mw < 0).any().item()):
        raise ValueError(f"{field} demand must be non-negative")
    return demand_mw


@dataclass(frozen=True, slots=True, eq=False)
class RidgeDemandForecaster:
    """Fitted AR(3) parameters, retained on CUDA for inference and provenance.

    The chronological feature order is t-3, t-2, t-1. Prediction in MW is
    ``target_means_mw + sum(coefficients_mw * (history - feature_means_mw)
    / feature_scales_mw)`` followed by a non-negative clamp. Centering leaves
    the intercept unpenalized. Training minimizes mean squared residual in MW
    plus ``ridge_penalty * sum(coefficients_mw**2)`` per locality-sector.
    """

    coefficients_mw: torch.Tensor
    feature_means_mw: torch.Tensor
    feature_scales_mw: torch.Tensor
    target_means_mw: torch.Tensor
    training_seeds: tuple[int, ...]
    training_trace_count: int
    ridge_penalty: float

    def __post_init__(self) -> None:
        _validate_training_seeds(self.training_seeds)
        _validate_penalty(self.ridge_penalty)
        if (
            isinstance(self.training_trace_count, bool)
            or not isinstance(self.training_trace_count, int)
            or self.training_trace_count < len(self.training_seeds)
            or self.training_trace_count % len(self.training_seeds)
        ):
            raise ValueError("training_trace_count must describe equal nonempty seed batches")
        parameters = (
            ("coefficients_mw", self.coefficients_mw, (*_CELL_SHAPE, 3)),
            ("feature_means_mw", self.feature_means_mw, (*_CELL_SHAPE, 3)),
            ("feature_scales_mw", self.feature_scales_mw, (*_CELL_SHAPE, 3)),
            ("target_means_mw", self.target_means_mw, _CELL_SHAPE),
        )
        devices = set()
        for name, tensor, shape in parameters:
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if tensor.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
            if tensor.dtype != torch.float64 or tensor.device.type != "cuda":
                raise ValueError(f"{name} must remain float64 on CUDA")
            if not bool(torch.isfinite(tensor).all().item()):
                raise ValueError(f"{name} must be finite")
            devices.add(tensor.device)
        if len(devices) != 1:
            raise ValueError("forecast parameters must share one CUDA device")
        if bool((self.feature_scales_mw <= 0).any().item()):
            raise ValueError("feature_scales_mw must be positive")
        if bool((self.feature_means_mw < 0).any().item()) or bool(
            (self.target_means_mw < 0).any().item()
        ):
            raise ValueError("training demand means must be non-negative")


@dataclass(frozen=True, slots=True, eq=False)
class ForecastModelEvaluation:
    """Aligned predictions and sufficient statistics retained on CUDA.

    Predictions have axes [method, seed, batch, target_hour, locality, sector].
    Sufficient statistics have axes [method, seed, aggregation, statistic], with
    aggregations locality-sector then system-total, and statistics count, sum
    absolute error in MW, sum squared error in MW squared. System residuals sum
    the 35 signed cell residuals before absolute value or squaring. Divide the
    summed errors by the summed counts to obtain pooled MAE or squared RMSE.
    """

    method_names: tuple[str, ...]
    target_hours: tuple[int, ...]
    evaluation_seeds: tuple[int, ...]
    predictions_mw: torch.Tensor
    targets_mw: torch.Tensor
    sufficient_statistics: torch.Tensor


@torch.inference_mode()
def fit_ridge_forecast(
    training_demand_mw: torch.Tensor,
    *,
    training_seeds: tuple[int, ...] = FORECAST_TRAINING_SEEDS,
    ridge_penalty: float = DEFAULT_RIDGE_PENALTY,
) -> RidgeDemandForecaster:
    """Fit float64 ridge AR(3) on CUDA using only the supplied training cube.

    The supplement protocol uses 256 newly generated traces for each documented
    candidate seed 42, 43, 44, yielding 768 traces. These are a new forecasting
    fit, not the historical PPO training trajectories. No holdout data are
    accepted for fitting or normalization. The default penalty is fixed before
    evaluation and is not selected by comparing holdout outcomes.
    """

    _validate_training_seeds(training_seeds)
    _validate_penalty(ridge_penalty)
    demand = _validate_demand_cube(training_demand_mw, seeds=training_seeds, field="training").to(
        torch.float64
    )
    windows = demand.unfold(dimension=2, size=3, step=1)[:, :, : len(FORECAST_TARGET_HOURS)]
    features = windows.permute(3, 4, 0, 1, 2, 5).reshape(*_CELL_SHAPE, -1, 3)
    targets = demand[:, :, 3:].permute(3, 4, 0, 1, 2).reshape(*_CELL_SHAPE, -1, 1)
    feature_means_mw = features.mean(dim=-2)
    feature_scales_mw = features.std(dim=-2, correction=0).clamp_min(FEATURE_SCALE_FLOOR_MW)
    target_means_mw = targets.mean(dim=-2).squeeze(-1)
    standardized = (features - feature_means_mw.unsqueeze(-2)) / feature_scales_mw.unsqueeze(-2)
    centered_targets = targets - target_means_mw[..., None, None]
    sample_count = features.shape[-2]
    transpose = standardized.transpose(-1, -2)
    gram = (transpose @ standardized) / sample_count
    gram += float(ridge_penalty) * torch.eye(3, dtype=torch.float64, device=demand.device)
    rhs = (transpose @ centered_targets) / sample_count
    coefficients_mw = torch.linalg.solve(gram, rhs).squeeze(-1)
    return RidgeDemandForecaster(
        coefficients_mw=coefficients_mw,
        feature_means_mw=feature_means_mw,
        feature_scales_mw=feature_scales_mw,
        target_means_mw=target_means_mw,
        training_seeds=training_seeds,
        training_trace_count=demand.shape[0] * demand.shape[1],
        ridge_penalty=float(ridge_penalty),
    )


def _error_statistics(residual_mw: torch.Tensor) -> torch.Tensor:
    flattened = residual_mw.flatten(start_dim=1)
    sum_absolute_error_mw = flattened.abs().sum(dim=1)
    sum_squared_error_mw2 = flattened.square().sum(dim=1)
    counts = torch.full_like(sum_absolute_error_mw, flattened.shape[1])
    return torch.stack((counts, sum_absolute_error_mw, sum_squared_error_mw2), dim=-1)


@torch.inference_mode()
def evaluate_forecast_model(
    model: RidgeDemandForecaster,
    evaluation_demand_mw: torch.Tensor,
    *,
    evaluation_seeds: tuple[int, ...],
) -> ForecastModelEvaluation:
    """Compare three causal methods on identical one-hour targets t=3..23.

    AR(3) and the trailing mean use exactly t-3..t-1. Persistence uses the latest
    observation t-1 from that same available history. Neither evaluation targets
    nor evaluation history update fitted parameters or normalization. This is
    a separate synthetic forecast experiment, not an operational forecast or an
    input to the PPO policy.
    """

    if not isinstance(model, RidgeDemandForecaster):
        raise TypeError("model must be a RidgeDemandForecaster")
    demand = _validate_demand_cube(evaluation_demand_mw, seeds=evaluation_seeds, field="evaluation")
    if set(evaluation_seeds).intersection(model.training_seeds):
        raise ValueError("training and evaluation seeds must be disjoint")
    if demand.device != model.coefficients_mw.device:
        raise ValueError("model and evaluation demand must share one CUDA device")
    demand = demand.to(torch.float64)
    windows = demand.unfold(dimension=2, size=3, step=1)[:, :, : len(FORECAST_TARGET_HOURS)]
    standardized = (windows - model.feature_means_mw) / model.feature_scales_mw
    learned = (
        (standardized * model.coefficients_mw).sum(dim=-1) + model.target_means_mw
    ).clamp_min(0)
    predictions = torch.stack((windows.mean(dim=-1), demand[:, :, 2:23], learned))
    targets_mw = demand[:, :, 3:24]
    statistics = []
    for prediction_mw in predictions:
        residual_mw = prediction_mw - targets_mw
        statistics.append(
            torch.stack(
                (
                    _error_statistics(residual_mw),
                    _error_statistics(residual_mw.sum(dim=(-1, -2))),
                ),
                dim=1,
            )
        )
    sufficient_statistics = torch.stack(statistics)
    if not bool(torch.isfinite(sufficient_statistics).all().item()):
        raise ValueError("forecast error statistics must be finite")
    return ForecastModelEvaluation(
        method_names=FORECAST_METHOD_NAMES,
        target_hours=FORECAST_TARGET_HOURS,
        evaluation_seeds=evaluation_seeds,
        predictions_mw=predictions,
        targets_mw=targets_mw,
        sufficient_statistics=sufficient_statistics,
    )

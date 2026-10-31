from __future__ import annotations

import pytest
import torch

from gridguard.domain import LOCALITIES, SECTORS, ShortageLevel
from gridguard.forecast_model import (
    FORECAST_METHOD_NAMES,
    FORECAST_TRAINING_SEEDS,
    RidgeDemandForecaster,
    evaluate_forecast_model,
    fit_ridge_forecast,
)
from gridguard.scenarios import generate_scenarios


def _recurrence_cube(seed_count: int, *, initial_offset: float = 0.0) -> torch.Tensor:
    initial = torch.arange(32 * 7 * 5, device="cuda:0", dtype=torch.float64)
    initial = initial.reshape(32, 7, 5) / 100 + 1 + initial_offset
    hours = [initial, initial * 0.8 + 1, initial * 0.6 + 2]
    for _ in range(21):
        hours.append(0.5 * hours[-1] + 0.25 * hours[-2] + 0.1 * hours[-3] + 2)
    return torch.stack(hours, dim=1).unsqueeze(0).expand(seed_count, -1, -1, -1, -1).clone()


def test_cpu_fit_fails_without_silent_fallback() -> None:
    demand = torch.ones(3, 2, 24, len(LOCALITIES), len(SECTORS))
    with pytest.raises(ValueError, match="CUDA"):
        fit_ridge_forecast(demand)


@pytest.mark.gpu
def test_fit_learns_known_recurrence_and_generalizes_to_unseen_traces() -> None:
    training = _recurrence_cube(3)
    model = fit_ridge_forecast(training, ridge_penalty=1e-9)
    held_out = _recurrence_cube(1, initial_offset=8.0)

    result = evaluate_forecast_model(model, held_out, evaluation_seeds=(91001,))

    assert result.method_names == FORECAST_METHOD_NAMES
    learned = result.predictions_mw[result.method_names.index("ridge_ar3")]
    torch.testing.assert_close(learned, held_out[:, :, 3:], rtol=1e-5, atol=1e-5)
    assert result.predictions_mw.device == torch.device("cuda:0")
    assert result.predictions_mw.dtype == torch.float64
    assert result.predictions_mw.shape == (3, 1, 32, 21, 7, 5)
    assert model.training_seeds == FORECAST_TRAINING_SEEDS
    assert model.training_trace_count == 96


@pytest.mark.gpu
def test_future_values_cannot_change_earlier_predictions_or_fitted_parameters() -> None:
    model = fit_ridge_forecast(_recurrence_cube(3))
    coefficient_before = model.coefficients_mw.clone()
    test_demand = _recurrence_cube(1)
    changed = test_demand.clone()
    changed[:, :, 8:] += 10_000

    first = evaluate_forecast_model(model, test_demand, evaluation_seeds=(91001,))
    second = evaluate_forecast_model(model, changed, evaluation_seeds=(91001,))

    # These six predictions target hours 3..8 and observe no value at hour 8 or later.
    torch.testing.assert_close(
        first.predictions_mw[:, :, :, :6],
        second.predictions_mw[:, :, :, :6],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(model.coefficients_mw, coefficient_before, rtol=0, atol=0)


@pytest.mark.gpu
def test_all_methods_share_exactly_the_same_target_hours_and_available_history() -> None:
    training = _recurrence_cube(3)
    model = fit_ridge_forecast(training)
    held_out = torch.arange(24, dtype=torch.float64, device="cuda:0")
    held_out = held_out.reshape(1, 1, 24, 1, 1).expand(1, 2, 24, 7, 5)

    result = evaluate_forecast_model(model, held_out, evaluation_seeds=(91001,))

    assert result.target_hours == tuple(range(3, 24))
    persistence = result.predictions_mw[result.method_names.index("persistence_1h")]
    trailing = result.predictions_mw[result.method_names.index("trailing_mean_3h")]
    torch.testing.assert_close(persistence, held_out[:, :, 2:23])
    torch.testing.assert_close(trailing, held_out[:, :, 1:22])
    torch.testing.assert_close(result.targets_mw, held_out[:, :, 3:24])


@pytest.mark.gpu
def test_error_statistics_preserve_locality_sector_and_system_aggregation() -> None:
    device = torch.device("cuda:0")
    model = RidgeDemandForecaster(
        coefficients_mw=torch.zeros(7, 5, 3, dtype=torch.float64, device=device),
        feature_means_mw=torch.zeros(7, 5, 3, dtype=torch.float64, device=device),
        feature_scales_mw=torch.ones(7, 5, 3, dtype=torch.float64, device=device),
        target_means_mw=torch.ones(7, 5, dtype=torch.float64, device=device),
        training_seeds=(42,),
        training_trace_count=2,
        ridge_penalty=0.001,
    )
    held_out = torch.full((1, 2, 24, 7, 5), 2.0, dtype=torch.float64, device=device)

    result = evaluate_forecast_model(model, held_out, evaluation_seeds=(91001,))

    expected = torch.tensor(
        [[1470.0, 1470.0, 1470.0], [42.0, 1470.0, 51450.0]],
        device=device,
        dtype=torch.float64,
    )
    learned_index = result.method_names.index("ridge_ar3")
    torch.testing.assert_close(result.sufficient_statistics[learned_index, 0], expected)
    assert result.sufficient_statistics.shape == (3, 1, 2, 3)
    assert result.sufficient_statistics.device == device


@pytest.mark.gpu
def test_training_and_evaluation_seed_overlap_is_rejected() -> None:
    model = fit_ridge_forecast(_recurrence_cube(3))
    with pytest.raises(ValueError, match="disjoint"):
        evaluate_forecast_model(model, _recurrence_cube(1), evaluation_seeds=(43,))


@pytest.mark.gpu
@pytest.mark.parametrize("seeds", [(91001, 91002, 91003), (42, 42, 43), (42, 43), (True, 43, 44)])
def test_fit_rejects_unapproved_duplicate_or_misaligned_seed_labels(seeds: tuple[int, ...]) -> None:
    with pytest.raises((TypeError, ValueError), match="seeds"):
        fit_ridge_forecast(_recurrence_cube(3), training_seeds=seeds)


@pytest.mark.gpu
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -0.1])
def test_fit_rejects_invalid_power(invalid: float) -> None:
    training = _recurrence_cube(3)
    training[0, 0, 0, 0, 0] = invalid
    with pytest.raises(ValueError, match="finite|non-negative"):
        fit_ridge_forecast(training)


@pytest.mark.gpu
def test_constant_training_demand_is_stable_and_predictions_are_nonnegative() -> None:
    training = torch.zeros((3, 2, 24, 7, 5), device="cuda:0")
    model = fit_ridge_forecast(training)
    held_out = torch.ones((1, 2, 24, 7, 5), device="cuda:0")

    result = evaluate_forecast_model(model, held_out, evaluation_seeds=(91001,))

    assert bool(torch.isfinite(result.predictions_mw).all())
    assert bool((result.predictions_mw >= 0).all())


@pytest.mark.gpu
def test_real_scenarios_fit_and_evaluate_without_changing_inputs() -> None:
    training = torch.stack(
        [generate_scenarios(4, seed, ShortageLevel.MODERATE).demand_mw for seed in (42, 43, 44)]
    )
    held_out = generate_scenarios(2, 91001, ShortageLevel.MODERATE).demand_mw.unsqueeze(0)
    before = held_out.clone()
    model = fit_ridge_forecast(training)

    result = evaluate_forecast_model(model, held_out, evaluation_seeds=(91001,))

    torch.testing.assert_close(held_out, before, rtol=0, atol=0)
    assert bool(torch.isfinite(result.sufficient_statistics).all())
    assert model.coefficients_mw.shape == (7, 5, 3)
    assert model.feature_means_mw.shape == (7, 5, 3)
    assert model.feature_scales_mw.shape == (7, 5, 3)
    assert model.target_means_mw.shape == (7, 5)

from __future__ import annotations

import inspect

import pytest
import torch

from gridguard.domain import LOCALITIES, SECTORS, ShortageLevel
from gridguard.forecast import causal_demand_projection
from gridguard.scenarios import generate_scenario


def _demand_trace(values: list[float], *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    hourly = torch.tensor(values, dtype=dtype).reshape(1, -1, 1, 1)
    return hourly.expand(1, -1, len(LOCALITIES), len(SECTORS)).clone()


def test_default_projection_is_one_hour_ahead() -> None:
    horizon_default = (
        inspect.signature(causal_demand_projection).parameters["horizon_hours"].default
    )

    assert horizon_default == 1


def test_one_hour_projection_uses_the_preceding_trailing_window() -> None:
    demand = _demand_trace([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])

    projection = causal_demand_projection(demand, window_hours=3)

    assert projection.shape == (1, 3, len(LOCALITIES), len(SECTORS))
    expected = _demand_trace([7.0 / 3.0, 14.0 / 3.0, 28.0 / 3.0])
    torch.testing.assert_close(projection, expected)


def test_future_samples_cannot_leak_into_earlier_projections() -> None:
    demand = _demand_trace([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    changed_future = demand.clone()
    changed_future[:, 4:] = 10_000.0

    baseline = causal_demand_projection(demand, window_hours=3)
    changed = causal_demand_projection(changed_future, window_hours=3)

    # These outputs predict target hours 3 and 4. Neither may observe hour 4 or later.
    torch.testing.assert_close(changed[:, :2], baseline[:, :2], rtol=0.0, atol=0.0)
    assert not torch.equal(changed[:, 2:], baseline[:, 2:])


def test_larger_horizon_preserves_strict_target_alignment() -> None:
    demand = _demand_trace([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

    projection = causal_demand_projection(demand, window_hours=2, horizon_hours=2)

    # Targets are hours 3..6; each estimate uses a two-hour window ending two hours earlier.
    expected = _demand_trace([1.5, 2.5, 3.5, 4.5])
    torch.testing.assert_close(projection, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_projection_preserves_cpu_dtype_and_device(dtype: torch.dtype) -> None:
    demand = _demand_trace([1.0, 2.0, 3.0, 4.0], dtype=dtype)

    first = causal_demand_projection(demand, window_hours=2)
    second = causal_demand_projection(demand, window_hours=2)

    assert first.dtype is dtype
    assert first.device == demand.device
    assert torch.equal(first, second)


@pytest.mark.parametrize("value", [None, [[1.0]], "demand"])
def test_projection_rejects_non_tensor_input(value: object) -> None:
    with pytest.raises(TypeError, match="demand_mw must be a torch.Tensor"):
        causal_demand_projection(value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "shape",
    [
        (1, 5, len(LOCALITIES)),
        (1, 5, len(LOCALITIES), len(SECTORS), 1),
        (1, 5, len(LOCALITIES) - 1, len(SECTORS)),
        (1, 5, len(LOCALITIES), len(SECTORS) - 1),
        (0, 5, len(LOCALITIES), len(SECTORS)),
    ],
)
def test_projection_rejects_invalid_demand_shape(shape: tuple[int, ...]) -> None:
    demand = torch.empty(shape)

    with pytest.raises(ValueError, match="demand_mw"):
        causal_demand_projection(demand)


def test_projection_rejects_non_floating_demand() -> None:
    demand = torch.ones((1, 5, len(LOCALITIES), len(SECTORS)), dtype=torch.int64)

    with pytest.raises(TypeError, match="floating-point"):
        causal_demand_projection(demand)


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), -0.01])
def test_projection_rejects_invalid_power_values(invalid_value: float) -> None:
    demand = _demand_trace([1.0, 2.0, 3.0, 4.0])
    demand[0, 1, 0, 0] = invalid_value

    with pytest.raises(ValueError, match="finite|negative"):
        causal_demand_projection(demand, window_hours=2)


@pytest.mark.parametrize("window_hours", [0, -1, True, 1.5])
def test_projection_rejects_invalid_window(window_hours: object) -> None:
    demand = _demand_trace([1.0, 2.0, 3.0, 4.0])

    with pytest.raises((TypeError, ValueError), match="window_hours"):
        causal_demand_projection(
            demand,
            window_hours=window_hours,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("horizon_hours", [0, -1, True, 1.5])
def test_projection_rejects_invalid_horizon(horizon_hours: object) -> None:
    demand = _demand_trace([1.0, 2.0, 3.0, 4.0])

    with pytest.raises((TypeError, ValueError), match="horizon_hours"):
        causal_demand_projection(
            demand,
            window_hours=2,
            horizon_hours=horizon_hours,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("window_hours", "horizon_hours"),
    [(4, 1), (3, 2), (1, 4)],
)
def test_projection_requires_at_least_one_valid_target(
    window_hours: int,
    horizon_hours: int,
) -> None:
    demand = _demand_trace([1.0, 2.0, 3.0, 4.0])

    with pytest.raises(ValueError, match=r"window_hours \+ horizon_hours"):
        causal_demand_projection(
            demand,
            window_hours=window_hours,
            horizon_hours=horizon_hours,
        )


@pytest.mark.gpu
def test_projection_keeps_real_scenario_data_on_cuda_zero() -> None:
    scenario = generate_scenario(809, ShortageLevel.MODERATE)

    projection = causal_demand_projection(scenario.demand_mw, window_hours=3)

    assert projection.device == torch.device("cuda:0")
    assert projection.dtype == scenario.demand_mw.dtype
    assert projection.shape == (1, 21, len(LOCALITIES), len(SECTORS))
    assert bool(torch.isfinite(projection).all().item())
    assert bool((projection >= 0).all().item())

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from gridguard.controllers import DecisionContext
from gridguard.domain import (
    HORIZON_HOURS,
    LOCALITIES,
    SECTORS,
    ControllerName,
    Scenario,
    ServiceTier,
    ShortageLevel,
    SimulationResult,
)
from gridguard.duration_metrics import compute_duration_metrics
from gridguard.metrics import compute_episode_metrics
from gridguard.simulation import simulate


@dataclass(frozen=True)
class _TraceController:
    tiers: torch.Tensor
    name: ControllerName = ControllerName.FIXED_ROTATION

    def act(self, context: DecisionContext) -> torch.Tensor:
        return self.tiers[:, context.hour]


def _result(
    tiers: torch.Tensor,
    *,
    demand_mw: torch.Tensor | None = None,
    supply_fraction: float | torch.Tensor = 1.0,
) -> SimulationResult:
    if demand_mw is None:
        demand_mw = torch.ones(
            (tiers.shape[0], HORIZON_HOURS, len(LOCALITIES), len(SECTORS)),
            dtype=torch.float64,
            device=tiers.device,
        )
    scenario = Scenario(
        demand=demand_mw,
        supply=demand_mw.sum(dim=(-1, -2)) * supply_fraction,
        shortage_level=ShortageLevel.SEVERE,
        seed=0,
    )
    return simulate(scenario, _TraceController(tiers))


@pytest.fixture
def cuda_device() -> torch.device:
    assert torch.cuda.is_available(), "the presentation CUDA runtime is required"
    return torch.device("cuda:0")


def _full_service_tiers(batch_size: int, device: torch.device | str) -> torch.Tensor:
    return torch.full(
        (batch_size, HORIZON_HOURS, len(LOCALITIES)),
        ServiceTier.FULL_SERVICE,
        dtype=torch.long,
        device=device,
    )


@pytest.mark.gpu
def test_duration_totals_disparity_and_streaks_have_analytic_values(
    cuda_device: torch.device,
) -> None:
    tiers = _full_service_tiers(2, cuda_device)
    tiers[0, (0, 1, 3), 0] = ServiceTier.OFF
    tiers[0, 21:24, 1] = ServiceTier.OFF
    tiers[0, (0, 23), 2] = ServiceTier.OFF
    tiers[0, :, 4] = ServiceTier.OFF
    tiers[0, :2, 5] = ServiceTier.OFF

    metrics = compute_duration_metrics(_result(tiers))

    totals = torch.tensor(
        [[3, 3, 2, 0, 24, 2, 0], [0, 0, 0, 0, 0, 0, 0]],
        dtype=torch.float64,
        device=cuda_device,
    )
    streaks = torch.tensor(
        [[2, 3, 1, 0, 24, 2, 0], [0, 0, 0, 0, 0, 0, 0]],
        dtype=torch.float64,
        device=cuda_device,
    )
    for actual in (
        metrics.locality_blackout_hours,
        metrics.locality_critical_interruption_hours,
        metrics.locality_off_hours,
    ):
        torch.testing.assert_close(actual, totals)
    torch.testing.assert_close(metrics.locality_longest_blackout_hours, streaks)
    torch.testing.assert_close(metrics.locality_longest_critical_interruption_hours, streaks)
    # The 21 unordered pairs have a total absolute gap of 158 hours.
    for actual, expected in (
        (metrics.blackout_locality_hours, [34.0, 0.0]),
        (metrics.critical_interruption_locality_hours, [34.0, 0.0]),
        (metrics.blackout_duration_range_hours, [24.0, 0.0]),
        (metrics.critical_interruption_duration_range_hours, [24.0, 0.0]),
        (metrics.blackout_duration_mean_pairwise_gap_hours, [158 / 21, 0.0]),
        (metrics.critical_interruption_duration_mean_pairwise_gap_hours, [158 / 21, 0.0]),
        (metrics.longest_blackout_hours, [24.0, 0.0]),
        (metrics.longest_critical_interruption_hours, [24.0, 0.0]),
    ):
        torch.testing.assert_close(actual, totals.new_tensor(expected))


@pytest.mark.gpu
def test_critical_shortfall_is_not_a_zero_service_blackout_or_an_off_action(
    cuda_device: torch.device,
) -> None:
    metrics = compute_duration_metrics(
        _result(_full_service_tiers(1, cuda_device), supply_fraction=0.5)
    )

    assert torch.count_nonzero(metrics.locality_blackout_hours) == 0
    assert torch.count_nonzero(metrics.locality_off_hours) == 0
    torch.testing.assert_close(
        metrics.locality_critical_interruption_hours,
        torch.full((1, 7), 24.0, dtype=torch.float64, device=cuda_device),
    )
    assert metrics.critical_interruption_duration_range_hours.item() == 0.0
    assert metrics.critical_interruption_duration_mean_pairwise_gap_hours.item() == 0.0
    assert metrics.longest_critical_interruption_hours.item() == 24.0


@pytest.mark.gpu
def test_zero_supply_counts_actual_blackouts_without_requested_off_tiers(
    cuda_device: torch.device,
) -> None:
    metrics = compute_duration_metrics(
        _result(_full_service_tiers(1, cuda_device), supply_fraction=0.0)
    )

    assert torch.count_nonzero(metrics.locality_off_hours) == 0
    assert metrics.blackout_locality_hours.item() == 168.0
    assert metrics.critical_interruption_locality_hours.item() == 168.0
    assert metrics.blackout_duration_mean_pairwise_gap_hours.item() == 0.0
    assert metrics.longest_blackout_hours.item() == 24.0


@pytest.mark.gpu
def test_hours_with_no_demand_are_not_interruption_hours(cuda_device: torch.device) -> None:
    tiers = _full_service_tiers(1, cuda_device)
    tiers[:, :, 6] = ServiceTier.OFF
    demand_mw = torch.ones((1, 24, 7, 5), dtype=torch.float64, device=cuda_device)
    demand_mw[:, :, 5, :3] = 0.0
    demand_mw[:, :, 6] = 0.0

    metrics = compute_duration_metrics(_result(tiers, demand_mw=demand_mw, supply_fraction=0.0))

    torch.testing.assert_close(
        metrics.locality_blackout_hours,
        demand_mw.new_tensor([[24, 24, 24, 24, 24, 24, 0]]),
    )
    torch.testing.assert_close(
        metrics.locality_critical_interruption_hours,
        demand_mw.new_tensor([[24, 24, 24, 24, 24, 0, 0]]),
    )
    assert metrics.locality_off_hours[0, 6].item() == 24.0


@pytest.mark.gpu
def test_critical_threshold_is_strict_and_positive_delivery_is_not_zero_service(
    cuda_device: torch.device,
) -> None:
    tiers = _full_service_tiers(1, cuda_device)
    demand_mw = torch.zeros((1, 24, 7, 5), dtype=torch.float64, device=cuda_device)
    demand_mw[:, :, 0, 0] = 1.0
    supply_fraction = torch.ones((1, 24), dtype=torch.float64, device=cuda_device)
    supply_fraction[0, 0] = 0.99
    supply_fraction[0, 1] = torch.nextafter(
        supply_fraction.new_tensor(0.99), supply_fraction.new_tensor(0.0)
    )
    supply_fraction[0, 2] = 1e-12
    supply_fraction[0, 3] = 0.0
    result = _result(tiers, demand_mw=demand_mw, supply_fraction=supply_fraction)

    metrics = compute_duration_metrics(result)

    assert metrics.locality_blackout_hours[0, 0].item() == 1.0
    assert metrics.locality_critical_interruption_hours[0, 0].item() == 3.0
    assert metrics.locality_longest_critical_interruption_hours[0, 0].item() == 3.0
    torch.testing.assert_close(
        metrics.critical_interruption_locality_hours,
        compute_episode_metrics(result).critical_outage_locality_hours,
    )


@pytest.mark.gpu
def test_zero_outage_metrics_and_episode_aggregation_remain_on_cuda(
    cuda_device: torch.device,
) -> None:
    metrics = compute_duration_metrics(_result(_full_service_tiers(3, cuda_device)))

    assert metrics.batch_size == 3
    assert metrics.device == cuda_device
    for name, value in metrics.as_dict().items():
        expected_shape = (3, 7) if name.startswith("locality_") else (3,)
        assert value.shape == expected_shape
        assert value.device == cuda_device
        assert value.dtype == torch.float64
        assert not value.requires_grad
        assert torch.count_nonzero(value) == 0
    for name, value in metrics.mean_over_episodes().items():
        assert value.shape == ((7,) if name.startswith("locality_") else ())
        assert value.device == cuda_device
        assert torch.isfinite(value).all()
        assert torch.count_nonzero(value) == 0


@pytest.mark.gpu
def test_aggregate_is_equal_episode_mean_after_per_episode_disparity(
    cuda_device: torch.device,
) -> None:
    tiers = _full_service_tiers(2, cuda_device)
    tiers[0, :6, 0] = ServiceTier.OFF
    tiers[1, :6, 1] = ServiceTier.OFF

    aggregate = compute_duration_metrics(_result(tiers)).mean_over_episodes()

    # Each episode has range 6; taking the range of averaged locality hours gives 3.
    assert aggregate["blackout_duration_range_hours"].item() == 6.0
    assert aggregate["blackout_duration_mean_pairwise_gap_hours"].item() == pytest.approx(12 / 7)
    torch.testing.assert_close(
        aggregate["locality_blackout_hours"],
        torch.tensor([3, 3, 0, 0, 0, 0, 0], dtype=torch.float64, device=cuda_device),
    )


@pytest.mark.gpu
def test_duration_metrics_reject_mutated_trace(cuda_device: torch.device) -> None:
    result = _result(_full_service_tiers(1, cuda_device))
    with torch.inference_mode():
        result.served[0, 0, 0, 0] = 0.0

    with pytest.raises(ValueError, match="changed after validation"):
        compute_duration_metrics(result)


def test_duration_metrics_reject_cpu_trace() -> None:
    result = _result(_full_service_tiers(1, "cpu"))

    with pytest.raises(ValueError, match="CUDA"):
        compute_duration_metrics(result)


def test_duration_metrics_require_a_simulation_result() -> None:
    with pytest.raises(TypeError, match="SimulationResult"):
        compute_duration_metrics(None)  # type: ignore[arg-type]

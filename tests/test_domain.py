from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

from gridguard.domain import (
    CRITICAL_SECTOR_COUNT,
    CRITICAL_SECTORS,
    LOCALITIES,
    LOCALITY_DEMAND_WEIGHTS,
    SECTOR_SHARES,
    SECTORS,
    ControllerName,
    Locality,
    Metrics,
    Scenario,
    Sector,
    ServiceTier,
    ShortageLevel,
    SimulationResult,
)


def _scenario() -> Scenario:
    return Scenario(
        demand=torch.ones((2, 24, 7, 5), dtype=torch.float32),
        supply=torch.full((2, 24), 20.0, dtype=torch.float32),
        shortage_level=ShortageLevel.MODERATE,
        seed=17,
    )


def test_domain_vocabulary_has_the_locked_public_order_and_values() -> None:
    assert tuple(item.value for item in LOCALITIES) == (
        "Khartoum",
        "Omdurman",
        "Bahri",
        "East Nile",
        "Jebel Awlia",
        "Karari",
        "Ombada",
    )
    assert tuple(item.value for item in SECTORS) == (
        "Hospital",
        "Water",
        "Bakery",
        "Residential",
        "Commercial",
    )
    assert CRITICAL_SECTOR_COUNT == 3
    assert CRITICAL_SECTORS == (Sector.HOSPITAL, Sector.WATER, Sector.BAKERY)
    assert LOCALITY_DEMAND_WEIGHTS == (1.0, 0.95, 0.85, 0.80, 0.70, 0.75, 0.90)
    assert SECTOR_SHARES == (0.08, 0.07, 0.05, 0.55, 0.25)


def test_domain_enums_use_only_proposal_vocabulary() -> None:
    assert list(ShortageLevel) == [
        ShortageLevel.MILD,
        ShortageLevel.MODERATE,
        ShortageLevel.SEVERE,
    ]
    assert [tier.label for tier in ServiceTier] == [
        "Off",
        "Critical Only",
        "Full Service",
    ]
    assert [int(tier) for tier in ServiceTier] == [0, 1, 2]
    assert [controller.value for controller in ControllerName] == ["PPO", "Fixed Rotation"]
    assert Locality("East Nile") is Locality.EAST_NILE
    with pytest.raises(ValueError):
        ServiceTier(3)


def test_scenario_is_an_immutable_validated_24_hour_batch() -> None:
    scenario = _scenario()

    assert scenario.demand.shape == (2, 24, 7, 5)
    assert scenario.supply.shape == (2, 24)
    assert scenario.batch_size == 2
    assert scenario.horizon_hours == 24
    assert scenario.demand_mw is scenario.demand
    assert scenario.available_supply_mw is scenario.supply
    with pytest.raises(FrozenInstanceError):
        scenario.seed = 18  # type: ignore[misc]


@pytest.mark.parametrize(
    ("demand_shape", "supply_shape", "message"),
    [
        ((1, 23, 7, 5), (1, 24), "demand must have shape"),
        ((1, 24, 6, 5), (1, 24), "demand must have shape"),
        ((1, 24, 7, 4), (1, 24), "demand must have shape"),
        ((1, 24, 7, 5), (1, 23), "supply must have shape"),
        ((2, 24, 7, 5), (1, 24), "supply must have shape"),
    ],
)
def test_scenario_rejects_invalid_shapes(
    demand_shape: tuple[int, ...],
    supply_shape: tuple[int, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        Scenario(
            demand=torch.ones(demand_shape),
            supply=torch.ones(supply_shape),
            shortage_level=ShortageLevel.MILD,
            seed=1,
        )


@pytest.mark.parametrize("field", ["demand", "supply"])
@pytest.mark.parametrize("invalid_value", [-0.1, float("nan"), float("inf")])
def test_scenario_rejects_invalid_power(field: str, invalid_value: float) -> None:
    demand = torch.ones((1, 24, 7, 5))
    supply = torch.ones((1, 24))
    (demand if field == "demand" else supply).flatten()[0] = invalid_value

    with pytest.raises(ValueError, match=field):
        Scenario(
            demand=demand,
            supply=supply,
            shortage_level=ShortageLevel.MILD,
            seed=1,
        )


def test_scenario_rejects_non_floating_or_mixed_tensor_contracts() -> None:
    with pytest.raises(TypeError, match="floating-point"):
        Scenario(
            demand=torch.ones((1, 24, 7, 5), dtype=torch.int64),
            supply=torch.ones((1, 24), dtype=torch.int64),
            shortage_level=ShortageLevel.MILD,
            seed=1,
        )

    with pytest.raises(ValueError, match="same dtype"):
        Scenario(
            demand=torch.ones((1, 24, 7, 5), dtype=torch.float32),
            supply=torch.ones((1, 24), dtype=torch.float64),
            shortage_level=ShortageLevel.MILD,
            seed=1,
        )


@pytest.mark.parametrize("seed", [-1, True, 2**63])
def test_scenario_rejects_invalid_seed(seed: object) -> None:
    with pytest.raises((TypeError, ValueError), match="seed"):
        Scenario(
            demand=torch.ones((1, 24, 7, 5)),
            supply=torch.ones((1, 24)),
            shortage_level=ShortageLevel.MILD,
            seed=seed,  # type: ignore[arg-type]
        )


def test_simulation_result_validates_tiers_power_and_batch_shapes() -> None:
    scenario = _scenario()
    result = SimulationResult(
        controller_name=ControllerName.PPO,
        scenario=scenario,
        service_tiers=torch.full((2, 24, 7), ServiceTier.CRITICAL_ONLY, dtype=torch.long),
        requested=torch.full((2, 24, 7, 5), 0.5),
        served=torch.full((2, 24, 7, 5), 0.5),
        reward=torch.zeros((2, 24)),
    )

    assert result.controller_name is ControllerName.PPO
    assert result.requested_mw is result.requested
    assert result.served_mw is result.served

    invalid_tiers = result.service_tiers.clone()
    invalid_tiers[0, 0, 0] = 3
    with pytest.raises(ValueError, match="service_tiers"):
        SimulationResult(
            controller_name=ControllerName.FIXED_ROTATION,
            scenario=scenario,
            service_tiers=invalid_tiers,
            requested=result.requested,
            served=result.served,
            reward=result.reward,
        )

    negative_served = result.served.clone()
    negative_served[0, 0, 0, 0] = -1.0
    with pytest.raises(ValueError, match="served"):
        SimulationResult(
            controller_name=ControllerName.PPO,
            scenario=scenario,
            service_tiers=result.service_tiers,
            requested=result.requested,
            served=negative_served,
            reward=result.reward,
        )


def test_metrics_reject_nonfinite_energy_and_out_of_range_ratios() -> None:
    valid = dict(
        episode_reward=1.0,
        demand_mwh=10.0,
        available_mwh=8.0,
        served_mwh=7.0,
        curtailed_mwh=3.0,
        critical_service_ratio=0.9,
        fairness=0.8,
        utilization=0.875,
        oversubscription=0.0,
        switching_rate=0.25,
        power_balance_violation_rate=0.0,
        critical_outage_locality_hours=12.0,
    )
    assert Metrics(**valid).served_mwh == 7.0

    with pytest.raises(ValueError, match="demand_mwh"):
        Metrics(**(valid | {"demand_mwh": float("nan")}))
    with pytest.raises(ValueError, match="fairness"):
        Metrics(**(valid | {"fairness": 1.1}))

    with pytest.raises(ValueError, match="energy identity"):
        Metrics(**(valid | {"curtailed_mwh": 1.0}))
    with pytest.raises(ValueError, match="critical_outage_locality_hours"):
        Metrics(**(valid | {"critical_outage_locality_hours": 169.0}))
    with pytest.raises(ValueError, match="oversubscription"):
        Metrics(**(valid | {"oversubscription": 1.1}))

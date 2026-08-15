from __future__ import annotations

import hashlib
import inspect
from dataclasses import FrozenInstanceError

import pytest
import torch

from gridguard.domain import ShortageLevel
from gridguard.scenarios import (
    ScenarioBaseDraws,
    generate_scenario,
    generate_scenario_base_draws,
    generate_scenarios,
    scenario_from_base_draws,
)


def _fingerprint(demand: torch.Tensor, supply: torch.Tensor) -> str:
    payload = demand.detach().cpu().contiguous().numpy().tobytes()
    payload += supply.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def test_generator_defaults_to_the_cuda_production_device() -> None:
    default = inspect.signature(generate_scenarios).parameters["device"].default

    assert default == "cuda:0"


@pytest.mark.gpu
def test_default_generator_keeps_the_production_batch_on_cuda() -> None:
    scenario = generate_scenarios(2, 811, ShortageLevel.MILD)

    assert scenario.demand.device == torch.device("cuda:0")
    assert scenario.supply.device == torch.device("cuda:0")


@pytest.mark.parametrize("shortage_level", list(ShortageLevel))
def test_same_seed_and_shortage_level_are_deterministic_on_one_device(
    shortage_level: ShortageLevel,
) -> None:
    first = generate_scenarios(3, 917, shortage_level, device="cpu")
    second = generate_scenarios(3, 917, shortage_level, device="cpu")

    assert torch.equal(first.demand, second.demand)
    assert torch.equal(first.supply, second.supply)
    assert first.demand.shape == (3, 24, 7, 5)
    assert first.supply.shape == (3, 24)
    assert first.demand.dtype == torch.float32
    assert first.demand.device.type == "cpu"


def test_different_seeds_change_both_noisy_traces() -> None:
    first = generate_scenarios(2, 100, ShortageLevel.MODERATE, device="cpu")
    second = generate_scenarios(2, 101, ShortageLevel.MODERATE, device="cpu")

    assert not torch.equal(first.demand, second.demand)
    assert not torch.equal(first.supply, second.supply)


def test_all_shortage_levels_are_pure_transforms_of_one_immutable_base_draw() -> None:
    base_draws = generate_scenario_base_draws(3, 917, device="cpu")
    demand_draws_before = base_draws.demand_uniform.clone()
    supply_draws_before = base_draws.supply_uniform.clone()

    scenarios = tuple(
        scenario_from_base_draws(base_draws, shortage_level) for shortage_level in ShortageLevel
    )

    assert isinstance(base_draws, ScenarioBaseDraws)
    assert all(torch.equal(scenario.demand, scenarios[0].demand) for scenario in scenarios[1:])
    assert all(scenario.seed == base_draws.seed for scenario in scenarios)
    assert torch.equal(base_draws.demand_uniform, demand_draws_before)
    assert torch.equal(base_draws.supply_uniform, supply_draws_before)
    with pytest.raises(FrozenInstanceError):
        base_draws.seed = 918  # type: ignore[misc]


@pytest.mark.parametrize("shortage_level", list(ShortageLevel))
def test_compatibility_generator_matches_explicit_base_draw_transform(
    shortage_level: ShortageLevel,
) -> None:
    base_draws = generate_scenario_base_draws(3, 481, device="cpu")

    explicit = scenario_from_base_draws(base_draws, shortage_level)
    compatible = generate_scenarios(3, 481, shortage_level, device="cpu")

    assert torch.equal(explicit.demand, compatible.demand)
    assert torch.equal(explicit.supply, compatible.supply)


@pytest.mark.parametrize(
    ("shortage_level", "available_mwh", "fingerprint"),
    [
        (
            ShortageLevel.MILD,
            8554.734375,
            "9a4818b8db1a0535be01a310e52c58d4a6d3058ca467b3f3df518882a3c4000e",
        ),
        (
            ShortageLevel.MODERATE,
            6724.16357421875,
            "7b15b6e4061380bbdad1b18c2140c4d70823d60a5ce030b89bee11f132d413a7",
        ),
        (
            ShortageLevel.SEVERE,
            4848.4638671875,
            "a06ce1687a299f1c4de7b75999dcede2dc18efde96bbd948943c44c48378ce04",
        ),
    ],
)
def test_noise_free_scenarios_match_the_audited_source_golden_cases(
    shortage_level: ShortageLevel,
    available_mwh: float,
    fingerprint: str,
) -> None:
    scenario = generate_scenarios(
        1,
        0,
        shortage_level,
        device="cpu",
        demand_noise=0.0,
        supply_noise=0.0,
    )

    assert scenario.demand.sum().item() == pytest.approx(10693.41796875, abs=1e-6)
    assert scenario.supply.sum().item() == pytest.approx(available_mwh, abs=1e-6)
    assert _fingerprint(scenario.demand, scenario.supply) == fingerprint


@pytest.mark.parametrize(
    ("shortage_level", "base", "start", "stop", "shortage_fraction"),
    [
        (ShortageLevel.MILD, 0.80, 0, 0, 0.80),
        (ShortageLevel.MODERATE, 0.65, 17, 21, 0.55),
        (ShortageLevel.SEVERE, 0.50, 16, 22, 0.35),
    ],
)
def test_noise_free_supply_uses_the_locked_hourly_shortage_profiles(
    shortage_level: ShortageLevel,
    base: float,
    start: int,
    stop: int,
    shortage_fraction: float,
) -> None:
    scenario = generate_scenarios(
        1,
        33,
        shortage_level,
        device="cpu",
        demand_noise=0.0,
        supply_noise=0.0,
    )
    fraction = scenario.supply / scenario.demand.sum(dim=(-1, -2))
    expected = torch.full((24,), base)
    if stop:
        expected[start:stop] -= base - shortage_fraction

    torch.testing.assert_close(fraction[0], expected, rtol=1e-6, atol=1e-6)


def test_noise_free_demand_preserves_peak_weights_and_sector_shares() -> None:
    scenario = generate_scenarios(
        1,
        5,
        ShortageLevel.MILD,
        device="cpu",
        demand_noise=0.0,
        supply_noise=0.0,
    )
    peak_by_sector = scenario.demand[0].sum(dim=1).amax(dim=0)

    torch.testing.assert_close(
        peak_by_sector,
        torch.tensor([56.0, 49.0, 35.0, 385.0, 175.0]),
        rtol=1e-6,
        atol=1e-5,
    )
    locality_totals = scenario.demand[0].sum(dim=(0, 2))
    torch.testing.assert_close(
        locality_totals / locality_totals[0],
        torch.tensor([1.0, 0.95, 0.85, 0.80, 0.70, 0.75, 0.90]),
        rtol=1e-6,
        atol=1e-6,
    )


def test_default_noise_is_bounded_and_supply_is_capped_below_demand() -> None:
    baseline = generate_scenarios(
        4,
        44,
        ShortageLevel.MODERATE,
        device="cpu",
        demand_noise=0.0,
        supply_noise=0.0,
    )
    noisy = generate_scenarios(4, 44, ShortageLevel.MODERATE, device="cpu")
    demand_multiplier = noisy.demand / baseline.demand
    total_demand = noisy.demand.sum(dim=(-1, -2))

    assert demand_multiplier.amin().item() >= 0.95 - 1e-6
    assert demand_multiplier.amax().item() <= 1.05 + 1e-6
    assert bool((noisy.supply <= total_demand * 0.95 + 1e-5).all().item())
    assert bool((noisy.supply < total_demand).all().item())


def test_supply_cap_applies_even_with_a_large_valid_noise_range() -> None:
    scenario = generate_scenarios(
        64,
        71,
        ShortageLevel.MILD,
        device="cpu",
        demand_noise=0.0,
        supply_noise=0.5,
    )
    total_demand = scenario.demand.sum(dim=(-1, -2))

    assert bool((scenario.supply <= total_demand * 0.95 + 1e-5).all().item())


def test_single_scenario_wrapper_returns_a_one_row_batch() -> None:
    scenario = generate_scenario(19, ShortageLevel.SEVERE, device="cpu")

    assert scenario.batch_size == 1
    assert scenario.seed == 19
    assert scenario.shortage_level is ShortageLevel.SEVERE


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_generator_rejects_invalid_batch_size(batch_size: object) -> None:
    with pytest.raises((TypeError, ValueError), match="batch_size"):
        generate_scenarios(
            batch_size,  # type: ignore[arg-type]
            1,
            ShortageLevel.MILD,
            device="cpu",
        )


@pytest.mark.parametrize("noise_name", ["demand_noise", "supply_noise"])
@pytest.mark.parametrize("noise", [-0.1, 1.0, float("nan"), True])
def test_generator_rejects_invalid_noise(noise_name: str, noise: object) -> None:
    options = {noise_name: noise}
    with pytest.raises((TypeError, ValueError), match=noise_name):
        generate_scenarios(
            1,
            1,
            ShortageLevel.MILD,
            device="cpu",
            **options,  # type: ignore[arg-type]
        )


def test_generator_rejects_non_public_shortage_levels() -> None:
    with pytest.raises(ValueError, match="shortage_level"):
        generate_scenarios(1, 1, "training", device="cpu")  # type: ignore[arg-type]

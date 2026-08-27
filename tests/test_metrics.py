from __future__ import annotations

import pytest
import torch

from gridguard.controllers import FixedRotationController
from gridguard.domain import ServiceTier, ShortageLevel, canonical_reward_trace
from gridguard.metrics import EpisodeMetrics, compute_episode_metrics
from gridguard.scenarios import generate_scenarios
from gridguard.simulation import simulate

# Locked outputs copied from the accepted pre-cleanup implementation. They are
# intentionally independent of the implementation under test.
FIXED_ROTATION_GOLDENS = {
    ShortageLevel.MILD: {
        "available_mwh": 8554.734375,
        "served_mwh": 4629.950195,
        "episode_reward": 13.189439,
        "critical_service_ratio": 0.43271279,
        "critical_outage_locality_hours": 96.0,
        "utilization": 0.54227942,
        "fairness": 0.87743145,
        "oversubscription": 0.0,
    },
    ShortageLevel.MODERATE: {
        "available_mwh": 6724.163574,
        "served_mwh": 4629.950195,
        "episode_reward": 13.957674,
        "critical_service_ratio": 0.43271279,
        "critical_outage_locality_hours": 96.0,
        "utilization": 0.68632358,
        "fairness": 0.87743145,
        "oversubscription": 0.0,
    },
    ShortageLevel.SEVERE: {
        "available_mwh": 4848.463867,
        "served_mwh": 4395.165039,
        "episode_reward": 14.755190,
        "critical_service_ratio": 0.41345379,
        "critical_outage_locality_hours": 114.0,
        "utilization": 0.90686268,
        "fairness": 0.87799525,
        "oversubscription": 0.01789216,
    },
}


@pytest.mark.parametrize("shortage_level", tuple(ShortageLevel))
def test_fixed_rotation_reproduces_accepted_noise_free_goldens(
    shortage_level: ShortageLevel,
) -> None:
    scenario = generate_scenarios(
        1,
        seed=0,
        shortage_level=shortage_level,
        device="cpu",
        demand_noise=0.0,
        supply_noise=0.0,
    )

    metrics = compute_episode_metrics(simulate(scenario, FixedRotationController()))
    expected = FIXED_ROTATION_GOLDENS[shortage_level]

    assert metrics.demand_mwh.item() == pytest.approx(10693.417969, abs=1e-3)
    assert metrics.available_mwh.item() == pytest.approx(expected["available_mwh"], abs=1e-3)
    assert metrics.served_mwh.item() == pytest.approx(expected["served_mwh"], abs=1e-3)
    assert metrics.episode_reward.item() == pytest.approx(expected["episode_reward"], abs=1e-5)
    assert metrics.critical_service_ratio.item() == pytest.approx(
        expected["critical_service_ratio"], abs=1e-6
    )
    assert (
        metrics.critical_outage_locality_hours.item() == expected["critical_outage_locality_hours"]
    )
    assert metrics.utilization.item() == pytest.approx(expected["utilization"], abs=1e-6)
    assert metrics.fairness.item() == pytest.approx(expected["fairness"], abs=1e-6)
    assert metrics.switching_rate.item() == pytest.approx(0.25, abs=1e-7)
    assert metrics.oversubscription.item() == pytest.approx(expected["oversubscription"], abs=1e-7)
    assert metrics.power_balance_violation_rate.item() == 0.0
    assert metrics.curtailed_mwh.item() == pytest.approx(
        metrics.demand_mwh.item() - metrics.served_mwh.item(), abs=1e-3
    )


def test_metrics_expose_one_tensor_per_episode_without_materializing_rows() -> None:
    scenario = generate_scenarios(
        4,
        seed=7,
        shortage_level=ShortageLevel.MODERATE,
        device="cpu",
    )

    metrics = compute_episode_metrics(simulate(scenario, FixedRotationController()))

    assert isinstance(metrics, EpisodeMetrics)
    assert metrics.batch_size == 4
    assert metrics.device == torch.device("cpu")
    assert all(value.shape == (4,) for value in metrics.as_dict().values())
    assert all(value.device == scenario.demand.device for value in metrics.as_dict().values())
    torch.testing.assert_close(
        metrics.demand_mwh,
        metrics.served_mwh + metrics.curtailed_mwh,
        rtol=1e-6,
        atol=1e-3,
    )


def test_metric_aliases_state_the_episode_aggregation_rules_explicitly() -> None:
    scenario = generate_scenarios(
        2,
        seed=23,
        shortage_level=ShortageLevel.MODERATE,
        device="cpu",
    )

    metrics = compute_episode_metrics(simulate(scenario, FixedRotationController()))

    assert metrics.episode_pooled_critical_service_ratio is metrics.critical_service_ratio
    assert metrics.mean_hourly_supply_utilization_ratio is metrics.utilization
    assert metrics.mean_hourly_cumulative_locality_jain_fairness is metrics.fairness


def test_metrics_reject_reward_forged_after_result_validation() -> None:
    scenario = generate_scenarios(
        2,
        seed=23,
        shortage_level=ShortageLevel.MODERATE,
        device="cpu",
    )
    result = simulate(scenario, FixedRotationController())
    with torch.inference_mode():
        result.reward[0, 0] += 0.01

    with pytest.raises(ValueError, match="changed after validation"):
        compute_episode_metrics(result)


def test_metrics_reject_coherent_served_and_reward_rewrite() -> None:
    scenario = generate_scenarios(
        2,
        seed=23,
        shortage_level=ShortageLevel.MODERATE,
        device="cpu",
    )
    result = simulate(scenario, FixedRotationController())
    with torch.inference_mode():
        result.served[0, 0, 0, 0] *= 0.5
        result.reward.copy_(
            canonical_reward_trace(
                scenario=result.scenario,
                service_tiers=result.service_tiers,
                requested=result.requested,
                served=result.served,
            )
        )

    with pytest.raises(ValueError, match="changed after validation"):
        compute_episode_metrics(result)


@pytest.mark.parametrize("field", ("requested", "service_tiers"))
def test_metrics_reject_requested_or_tier_rewrite_with_recomputed_reward(field: str) -> None:
    scenario = generate_scenarios(
        2,
        seed=23,
        shortage_level=ShortageLevel.MODERATE,
        device="cpu",
    )
    result = simulate(scenario, FixedRotationController())
    with torch.inference_mode():
        if field == "requested":
            result.requested[0, 0, 0, 0] *= 0.5
        else:
            result.service_tiers[0, 0, 0] = ServiceTier.OFF
        result.reward.copy_(
            canonical_reward_trace(
                scenario=result.scenario,
                service_tiers=result.service_tiers,
                requested=result.requested,
                served=result.served,
            )
        )

    with pytest.raises(ValueError):
        compute_episode_metrics(result)


def test_metrics_reject_fully_coherent_post_validation_trace_rewrite() -> None:
    scenario = generate_scenarios(
        2,
        seed=23,
        shortage_level=ShortageLevel.MODERATE,
        device="cpu",
    )
    result = simulate(scenario, FixedRotationController())
    with torch.inference_mode():
        result.service_tiers.fill_(ServiceTier.OFF)
        result.requested.zero_()
        result.served.zero_()
        result.reward.copy_(
            canonical_reward_trace(
                scenario=result.scenario,
                service_tiers=result.service_tiers,
                requested=result.requested,
                served=result.served,
            )
        )

    with pytest.raises(ValueError, match="changed after validation"):
        compute_episode_metrics(result)


def test_metrics_reject_trace_mutation_even_after_values_are_restored() -> None:
    scenario = generate_scenarios(
        2,
        seed=23,
        shortage_level=ShortageLevel.MODERATE,
        device="cpu",
    )
    result = simulate(scenario, FixedRotationController())
    original = result.served.clone()
    with torch.inference_mode():
        result.served[0, 0, 0, 0] *= 0.5
        result.served.copy_(original)
    assert torch.equal(result.served, original)

    with pytest.raises(ValueError, match="changed after validation"):
        compute_episode_metrics(result)


def test_metrics_reject_unversioned_data_write_from_snapshot_bytes() -> None:
    scenario = generate_scenarios(
        2,
        seed=23,
        shortage_level=ShortageLevel.MODERATE,
        device="cpu",
    )
    result = simulate(scenario, FixedRotationController())
    original_version = result.served._version
    with torch.inference_mode():
        result.served.data[0, 0, 0, 0] *= 0.5
    assert result.served._version == original_version

    with pytest.raises(ValueError, match="changed after validation"):
        compute_episode_metrics(result)


def test_metrics_reject_same_object_trace_storage_replacement() -> None:
    scenario = generate_scenarios(
        2,
        seed=23,
        shortage_level=ShortageLevel.MODERATE,
        device="cpu",
    )
    result = simulate(scenario, FixedRotationController())
    original_tensor = result.served
    original_bytes = result.served.clone()

    result.served.data = result.served.clone()

    assert result.served is original_tensor
    assert torch.equal(result.served, original_bytes)
    with pytest.raises(ValueError, match="changed after validation"):
        compute_episode_metrics(result)


def test_metrics_reject_scenario_mutation_after_result_validation() -> None:
    scenario = generate_scenarios(
        2,
        seed=23,
        shortage_level=ShortageLevel.MODERATE,
        device="cpu",
    )
    result = simulate(scenario, FixedRotationController())
    with torch.inference_mode():
        result.scenario.supply[0, 0] *= 0.9

    with pytest.raises(ValueError, match="scenario.supply"):
        compute_episode_metrics(result)


def test_episode_metrics_reject_nonfinite_values() -> None:
    values = {
        name: torch.ones(2, dtype=torch.float32) for name in EpisodeMetrics.__dataclass_fields__
    }
    values["fairness"][0] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        EpisodeMetrics(**values)


@pytest.mark.gpu
def test_episode_metrics_remain_cuda_native() -> None:
    assert torch.cuda.is_available(), "the presentation CUDA runtime is required"
    scenario = generate_scenarios(
        16,
        seed=91,
        shortage_level=ShortageLevel.SEVERE,
        device="cuda:0",
    )

    metrics = compute_episode_metrics(simulate(scenario, FixedRotationController()))

    assert metrics.device == torch.device("cuda:0")
    assert all(value.device == torch.device("cuda:0") for value in metrics.as_dict().values())
    assert torch.count_nonzero(metrics.power_balance_violation_rate) == 0


@pytest.mark.gpu
def test_cuda_metrics_reject_coherent_served_and_reward_rewrite() -> None:
    assert torch.cuda.is_available(), "the presentation CUDA runtime is required"
    scenario = generate_scenarios(
        2,
        seed=91,
        shortage_level=ShortageLevel.SEVERE,
        device="cuda:0",
    )
    result = simulate(scenario, FixedRotationController())
    with torch.inference_mode():
        result.served[0, 0, 0, 0] *= 0.5
        result.reward.copy_(
            canonical_reward_trace(
                scenario=result.scenario,
                service_tiers=result.service_tiers,
                requested=result.requested,
                served=result.served,
            )
        )

    with pytest.raises(ValueError, match="changed after validation"):
        compute_episode_metrics(result)


@pytest.mark.gpu
def test_cuda_metrics_reject_fully_coherent_post_validation_trace_rewrite() -> None:
    assert torch.cuda.is_available(), "the presentation CUDA runtime is required"
    scenario = generate_scenarios(
        2,
        seed=91,
        shortage_level=ShortageLevel.SEVERE,
        device="cuda:0",
    )
    result = simulate(scenario, FixedRotationController())
    with torch.inference_mode():
        result.service_tiers.fill_(ServiceTier.OFF)
        result.requested.zero_()
        result.served.zero_()
        result.reward.copy_(
            canonical_reward_trace(
                scenario=result.scenario,
                service_tiers=result.service_tiers,
                requested=result.requested,
                served=result.served,
            )
        )

    with pytest.raises(ValueError, match="changed after validation"):
        compute_episode_metrics(result)

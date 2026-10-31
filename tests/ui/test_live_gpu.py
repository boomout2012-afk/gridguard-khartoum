from __future__ import annotations

import pytest
import torch

from gridguard.duration_metrics import compute_duration_metrics
from gridguard.forecast import causal_demand_projection
from gridguard.gpu import run_gpu_preflight
from gridguard.live_policy import DEPLOYED_LIVE_POLICY
from gridguard.metrics import compute_episode_metrics
from gridguard.ui.live_runner import (
    LiveCudaExecution,
    build_live_comparison,
    execute_live_scenario_cuda,
)


@pytest.mark.gpu
def test_live_runner_uses_one_shared_cuda_scenario_for_three_controllers(monkeypatch) -> None:
    run_gpu_preflight()

    execution = execute_live_scenario_cuda("mild", 50001)

    assert execution.scenario.demand.device.type == "cuda"
    assert execution.scenario.demand.device.index == 0
    assert execution.ppo.scenario is execution.scenario
    assert execution.fixed_rotation.scenario is execution.scenario
    assert execution.optimization is not None
    assert execution.optimization.scenario is execution.scenario
    assert execution.ppo.service_tiers.device == execution.scenario.demand.device
    assert execution.fixed_rotation.service_tiers.device == execution.scenario.demand.device
    assert execution.optimization.service_tiers.device == execution.scenario.demand.device

    original_to = torch.Tensor.to
    bulk_transfers = []

    def record_to(tensor, *args, **kwargs):
        destination = kwargs.get("device", args[0] if args else None)
        if (
            tensor.device.type == "cuda"
            and destination in ("cpu", torch.device("cpu"))
            and tensor.numel() > 100
        ):
            bulk_transfers.append(tensor.numel())
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", record_to)

    comparison = build_live_comparison(execution)

    assert comparison.device == "cuda:0"
    assert execution.policy_identity == DEPLOYED_LIVE_POLICY
    assert comparison.policy_identity == DEPLOYED_LIVE_POLICY
    assert tuple(comparison.frames_by_controller) == ("PPO", "Fixed Rotation", "Optimization")
    assert len(bulk_transfers) == 1
    assert all(len(frames) == 24 for frames in comparison.frames_by_controller.values())
    ppo_frames = comparison.frames_by_controller["PPO"]
    fixed_frames = comparison.frames_by_controller["Fixed Rotation"]
    optimization_frames = comparison.frames_by_controller["Optimization"]
    assert [(frame.demand_mw, frame.supply_mw) for frame in ppo_frames] == [
        (frame.demand_mw, frame.supply_mw) for frame in fixed_frames
    ]
    assert [(frame.demand_mw, frame.supply_mw) for frame in ppo_frames] == [
        (frame.demand_mw, frame.supply_mw) for frame in optimization_frames
    ]
    for name, result in (
        ("PPO", execution.ppo),
        ("Fixed Rotation", execution.fixed_rotation),
        ("Optimization", execution.optimization),
    ):
        summary = comparison.metrics(name)
        canonical_metrics = compute_episode_metrics(result)
        assert summary.utilization == canonical_metrics.utilization.item()
        assert summary.episode_reward == canonical_metrics.episode_reward.item()
        duration = comparison.metrics(name).duration
        assert duration is not None
        expected = compute_duration_metrics(result)
        assert duration.blackout_locality_hours == expected.blackout_locality_hours.item()
        assert duration.critical_interruption_locality_hours == (
            expected.critical_interruption_locality_hours.item()
        )
        assert duration.blackout_duration_mean_pairwise_gap_hours == pytest.approx(
            expected.blackout_duration_mean_pairwise_gap_hours.item()
        )
    assert all(len(frame.locality_details) == 7 for frame in ppo_frames + fixed_frames)
    detail = ppo_frames[0].locality_details["khartoum"]
    assert detail.served_mw <= detail.demand_mw
    assert tuple(detail.sector_service_ratio) == (
        "Hospital",
        "Water",
        "Bakery",
        "Residential",
        "Commercial",
    )
    assert detail.next_hour_projected_demand_mw is None
    projected = causal_demand_projection(execution.scenario.demand_mw)[0].sum(dim=-1)
    expected_hour_3 = projected[0, 0].item()
    ppo_hour_2 = ppo_frames[2].locality_details["khartoum"]
    fixed_hour_2 = fixed_frames[2].locality_details["khartoum"]
    assert ppo_hour_2.next_hour_projected_demand_mw == pytest.approx(expected_hour_3)
    assert fixed_hour_2.next_hour_projected_demand_mw == pytest.approx(expected_hour_3)
    assert ppo_frames[23].locality_details["khartoum"].next_hour_projected_demand_mw is None

    # Existing injected two-controller executions remain valid, without fabricated
    # Optimization frames; the real entry point above always evaluates all three.
    bulk_transfers.clear()
    legacy = build_live_comparison(
        LiveCudaExecution(execution.scenario, execution.ppo, execution.fixed_rotation)
    )
    assert tuple(legacy.frames_by_controller) == ("PPO", "Fixed Rotation")
    assert len(bulk_transfers) == 1
    assert legacy.metrics("PPO").duration == comparison.metrics("PPO").duration

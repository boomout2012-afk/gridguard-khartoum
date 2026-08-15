from __future__ import annotations

import pytest

from gridguard.forecast import causal_demand_projection
from gridguard.gpu import run_gpu_preflight
from gridguard.ui.live_runner import build_live_comparison, execute_live_scenario_cuda


@pytest.mark.gpu
def test_live_runner_uses_one_shared_cuda_scenario_for_both_controllers() -> None:
    run_gpu_preflight()

    execution = execute_live_scenario_cuda("mild", 50001)

    assert execution.scenario.demand.device.type == "cuda"
    assert execution.scenario.demand.device.index == 0
    assert execution.ppo.scenario is execution.scenario
    assert execution.fixed_rotation.scenario is execution.scenario
    assert execution.ppo.service_tiers.device == execution.scenario.demand.device
    assert execution.fixed_rotation.service_tiers.device == execution.scenario.demand.device

    comparison = build_live_comparison(execution)

    assert comparison.device == "cuda:0"
    assert tuple(comparison.frames_by_controller) == ("PPO", "Fixed Rotation")
    assert all(len(frames) == 24 for frames in comparison.frames_by_controller.values())
    ppo_frames = comparison.frames_by_controller["PPO"]
    fixed_frames = comparison.frames_by_controller["Fixed Rotation"]
    assert [(frame.demand_mw, frame.supply_mw) for frame in ppo_frames] == [
        (frame.demand_mw, frame.supply_mw) for frame in fixed_frames
    ]
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

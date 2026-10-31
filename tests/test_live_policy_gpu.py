"""Real CUDA parity between the deployed actor and the unchanged study methods."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from gridguard.allocation_study import METRIC_NAMES
from gridguard.constrained import ConstrainedOptimizationController
from gridguard.constrained_ppo import ConstrainedPPOController, load_model
from gridguard.controllers import FixedRotationController
from gridguard.duration_metrics import compute_duration_metrics
from gridguard.focused_training import evaluate_metrics
from gridguard.gpu import run_gpu_preflight
from gridguard.live_policy import DEPLOYED_LIVE_POLICY
from gridguard.metrics import compute_episode_metrics
from gridguard.simulation import simulate
from gridguard.ui.live_runner import execute_live_scenario_cuda

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.gpu
@pytest.mark.parametrize("shortage", ["mild", "moderate", "severe"])
def test_live_deployment_exactly_matches_frozen_study_methods(shortage) -> None:
    run_gpu_preflight()
    execution = execute_live_scenario_cuda(shortage, 122201)
    assert execution.policy_identity == DEPLOYED_LIVE_POLICY
    scenario = execution.scenario
    guard = scenario.capture_mutation_guard()
    study_actor = load_model(
        ROOT / DEPLOYED_LIVE_POLICY.model_relative_path,
        expected_sha256=DEPLOYED_LIVE_POLICY.model_sha256,
    )
    controllers = (
        ConstrainedPPOController(study_actor),
        FixedRotationController(),
        ConstrainedOptimizationController(),
    )
    for actual, controller in zip(execution.controller_results, controllers, strict=True):
        expected = simulate(scenario, controller)
        for field in ("service_tiers", "requested", "served", "reward"):
            assert torch.equal(getattr(actual, field), getattr(expected, field)), field
        metrics = compute_episode_metrics(actual).as_dict()
        metrics.update(compute_duration_metrics(actual).as_dict())
        actual_row = torch.stack([metrics[name] for name in METRIC_NAMES], dim=-1)
        expected_row = evaluate_metrics(controller, ((scenario,),))[0, 0]
        assert torch.equal(actual_row.double(), expected_row)
        assert torch.count_nonzero(metrics["power_balance_violation_rate"]).item() == 0
        if actual is not execution.fixed_rotation:
            assert torch.count_nonzero(metrics["critical_interruption_locality_hours"]).item() == 0
            assert torch.count_nonzero(metrics["critical_outage_locality_hours"]).item() == 0
    assert not guard.changed_fields(scenario)

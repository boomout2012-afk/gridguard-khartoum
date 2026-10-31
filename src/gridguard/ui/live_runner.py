"""CUDA-only live comparison runner and explicit presentation transfer."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from gridguard.constrained import ConstrainedOptimizationController
from gridguard.controllers import FixedRotationController
from gridguard.domain import (
    CRITICAL_SECTOR_COUNT,
    SECTORS,
    ControllerName,
    Scenario,
    SimulationResult,
)
from gridguard.duration_metrics import compute_duration_metrics
from gridguard.forecast import causal_demand_projection
from gridguard.live_policy import LivePolicyIdentity, load_live_policy
from gridguard.metrics import EpisodeMetrics, compute_episode_metrics
from gridguard.scenarios import generate_scenario
from gridguard.simulation import simulate
from gridguard.ui.live_models import (
    LiveDurationSummary,
    LiveEpisodeSummary,
    LiveScenarioComparison,
    LiveScenarioFrame,
    LocalityHourDetail,
)
from gridguard.ui.map_asset import EXPECTED_LOCALITY_IDS

_LIVE_METRIC_FIELDS = (
    "served_mwh",
    "critical_service_ratio",
    "fairness",
    "curtailed_mwh",
    "power_balance_violation_rate",
    "utilization",
    "episode_reward",
)
_LIVE_DURATION_FIELDS = (
    "blackout_locality_hours",
    "critical_interruption_locality_hours",
    "blackout_duration_range_hours",
    "blackout_duration_mean_pairwise_gap_hours",
    "critical_interruption_duration_range_hours",
    "critical_interruption_duration_mean_pairwise_gap_hours",
    "longest_blackout_hours",
    "longest_critical_interruption_hours",
)


@dataclass(frozen=True, slots=True, eq=False)
class LiveCudaExecution:
    """Controller traces retaining one exact shared CUDA scenario."""

    scenario: Scenario
    ppo: SimulationResult
    fixed_rotation: SimulationResult
    optimization: SimulationResult | None = None
    policy_identity: LivePolicyIdentity | None = None

    def __post_init__(self) -> None:
        if self.scenario.batch_size != 1:
            raise ValueError("live execution requires exactly one scenario")
        if self.policy_identity is not None and not isinstance(
            self.policy_identity, LivePolicyIdentity
        ):
            raise TypeError("live policy identity must be a LivePolicyIdentity")
        if any(result.scenario is not self.scenario for result in self.controller_results):
            raise ValueError("live controllers must share the exact same Scenario object")
        if self.ppo.controller_name != ControllerName.PPO:
            raise ValueError("ppo result must be labeled PPO")
        if self.fixed_rotation.controller_name != ControllerName.FIXED_ROTATION:
            raise ValueError("baseline result must be labeled Fixed Rotation")
        if (
            self.optimization is not None
            and self.optimization.controller_name != ControllerName.OPTIMIZATION
        ):
            raise ValueError("optimization result must be labeled Optimization")

        device = self.scenario.demand.device
        if device.type != "cuda" or device.index != 0:
            raise ValueError("live execution must remain on cuda:0")
        tensors = (self.scenario.supply,) + tuple(
            tensor
            for result in self.controller_results
            for tensor in (result.service_tiers, result.requested, result.served, result.reward)
        )
        if any(tensor.device != device for tensor in tensors):
            raise ValueError("all live scenario and result tensors must remain on cuda:0")

    @property
    def controller_results(self) -> tuple[SimulationResult, ...]:
        return (self.ppo, self.fixed_rotation) + (
            (self.optimization,) if self.optimization is not None else ()
        )


def execute_live_scenario_cuda(
    shortage_level: str,
    seed: int,
    *,
    training_path: str | Path | None = None,
    device: str | torch.device = "cuda:0",
) -> LiveCudaExecution:
    """Generate once and run three controllers against the same CUDA scenario."""

    resolved_device = torch.device(device)
    if resolved_device.type != "cuda" or resolved_device.index != 0:
        raise ValueError("live execution requires explicit device cuda:0")
    policy = load_live_policy(training_path=training_path, device=resolved_device)
    scenario = generate_scenario(
        seed,
        shortage_level,
        device=resolved_device,
    )
    guard = scenario.capture_mutation_guard()
    fixed_result = simulate(scenario, FixedRotationController())
    if guard.changed_fields(scenario):
        raise ValueError("Fixed Rotation mutated the matched live scenario")
    ppo_result = simulate(scenario, policy.controller)
    if guard.changed_fields(scenario):
        raise ValueError("PPO mutated the matched live scenario")
    optimization_result = simulate(scenario, ConstrainedOptimizationController())
    if guard.changed_fields(scenario):
        raise ValueError("Constrained Optimization mutated the matched live scenario")
    return LiveCudaExecution(
        scenario=scenario,
        ppo=ppo_result,
        fixed_rotation=fixed_result,
        optimization=optimization_result,
        policy_identity=policy.identity,
    )


def build_live_comparison(execution: LiveCudaExecution) -> LiveScenarioComparison:
    """Perform one explicit, final 24-row CUDA-to-host display transfer."""

    if not isinstance(execution, LiveCudaExecution):
        raise TypeError("execution must be a LiveCudaExecution")
    demand = execution.scenario.demand[0]
    results = execution.controller_results
    supply_mw = execution.scenario.supply[0].unsqueeze(-1)
    projection = causal_demand_projection(execution.scenario.demand_mw)[0].sum(dim=-1)
    next_hour_projection = torch.zeros(
        (24, len(EXPECTED_LOCALITY_IDS)),
        dtype=demand.dtype,
        device=demand.device,
    )
    next_hour_projection[2:23] = projection
    packed_display = torch.cat(
        (
            supply_mw,
            demand.flatten(start_dim=1),
            next_hour_projection,
        )
        + tuple(
            tensor
            for result in results
            for tensor in (
                result.service_tiers[0].to(dtype=demand.dtype),
                result.served[0].flatten(start_dim=1),
            )
        ),
        dim=-1,
    )
    locality_count = len(EXPECTED_LOCALITY_IDS)
    sector_count = len(SECTORS)
    expected_width = (
        1
        + locality_count * sector_count
        + locality_count
        + len(results) * locality_count * (sector_count + 1)
    )
    if packed_display.shape != (24, expected_width):
        raise ValueError("live display tensor has an unexpected shape")
    if packed_display.device.type != "cuda" or packed_display.device.index != 0:
        raise ValueError("live display tensor left cuda:0 before final transfer")

    # Frame and summary tensors remain device-resident until the transfer below.
    episode_metrics = tuple(compute_episode_metrics(result) for result in results)
    duration_metrics = tuple(compute_duration_metrics(result) for result in results)
    packed_metrics = torch.cat(
        tuple(
            value
            for metrics, durations in zip(episode_metrics, duration_metrics, strict=True)
            for value in (
                *(getattr(metrics, name) for name in _LIVE_METRIC_FIELDS),
                *(getattr(durations, name) for name in _LIVE_DURATION_FIELDS),
            )
        )
    )
    packed_transfer = torch.cat((packed_display.reshape(-1), packed_metrics))

    # Simulation, inference, frame data, and aggregate metrics stay on cuda:0
    # until this single bulk device-to-host presentation transfer.
    host_values = packed_transfer.to(device="cpu", non_blocking=False).tolist()
    frame_value_count = 24 * expected_width
    host_rows = [
        host_values[start : start + expected_width]
        for start in range(0, frame_value_count, expected_width)
    ]
    host_metrics = host_values[frame_value_count:]
    frames_by_controller: dict[str, list[LiveScenarioFrame]] = {
        result.controller_name.value: [] for result in results
    }
    for hour, row in enumerate(host_rows):
        supply = row[0]
        offset = 1
        demand_values = row[offset : offset + locality_count * sector_count]
        offset += locality_count * sector_count
        projection_values = row[offset : offset + locality_count]
        offset += locality_count
        next_projection = projection_values if 2 <= hour <= 22 else None
        for result in results:
            tiers = row[offset : offset + locality_count]
            offset += locality_count
            served = row[offset : offset + locality_count * sector_count]
            offset += locality_count * sector_count
            frames_by_controller[result.controller_name.value].append(
                _frame(hour, supply, demand_values, next_projection, tiers, served)
            )
    summary_width = len(_LIVE_METRIC_FIELDS) + len(_LIVE_DURATION_FIELDS)
    return LiveScenarioComparison(
        shortage_level=execution.scenario.shortage_level.value,
        seed=execution.scenario.seed,
        device="cuda:0",
        policy_identity=execution.policy_identity,
        frames_by_controller=frames_by_controller,
        metrics_by_controller={
            result.controller_name.value: _episode_summary(
                host_metrics[index * summary_width : (index + 1) * summary_width], metrics
            )
            for index, (result, metrics) in enumerate(zip(results, episode_metrics, strict=True))
        },
    )


def run_live_scenario(shortage_level: str, seed: int) -> LiveScenarioComparison:
    """Run the complete CUDA live flow used by the desktop worker."""

    return build_live_comparison(execute_live_scenario_cuda(shortage_level, seed))


def _frame(
    hour: int,
    supply_mw: float,
    demand_values: Sequence[float],
    next_hour_projection_values: Sequence[float] | None,
    tier_values: Sequence[float],
    served_values: Sequence[float],
) -> LiveScenarioFrame:
    sector_count = len(SECTORS)
    tiers = {
        locality_id: int(tier)
        for locality_id, tier in zip(EXPECTED_LOCALITY_IDS, tier_values, strict=True)
    }
    details: dict[str, LocalityHourDetail] = {}
    for locality_index, locality_id in enumerate(EXPECTED_LOCALITY_IDS):
        start = locality_index * sector_count
        demand = demand_values[start : start + sector_count]
        served = served_values[start : start + sector_count]
        critical_demand = sum(demand[:CRITICAL_SECTOR_COUNT])
        critical_served = sum(served[:CRITICAL_SECTOR_COUNT])
        details[locality_id] = LocalityHourDetail(
            demand_mw=sum(demand),
            served_mw=sum(served),
            critical_service_ratio=critical_served / max(critical_demand, 1e-8),
            sector_service_ratio={
                sector.value: delivered / max(requested, 1e-8)
                for sector, requested, delivered in zip(
                    SECTORS,
                    demand,
                    served,
                    strict=True,
                )
            },
            next_hour_projected_demand_mw=(
                None
                if next_hour_projection_values is None
                else next_hour_projection_values[locality_index]
            ),
        )
    return LiveScenarioFrame(
        hour=hour,
        service_tiers=tiers,
        demand_mw=sum(demand_values),
        supply_mw=supply_mw,
        locality_details=details,
    )


def _episode_summary(
    values: Sequence[float],
    metrics: EpisodeMetrics,
) -> LiveEpisodeSummary:
    metric_count = len(_LIVE_METRIC_FIELDS)
    if metrics.batch_size != 1 or len(values) != metric_count + len(_LIVE_DURATION_FIELDS):
        raise ValueError("live aggregate metrics have an unexpected shape")
    return LiveEpisodeSummary(
        **dict(zip(_LIVE_METRIC_FIELDS, values[:metric_count], strict=True)),
        duration=LiveDurationSummary(
            **dict(zip(_LIVE_DURATION_FIELDS, values[metric_count:], strict=True)),
        ),
    )

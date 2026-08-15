"""CUDA-only live comparison runner and explicit presentation transfer."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from gridguard.controllers import FixedRotationController, PPOController
from gridguard.domain import (
    CRITICAL_SECTOR_COUNT,
    SECTORS,
    ControllerName,
    Scenario,
    SimulationResult,
)
from gridguard.forecast import causal_demand_projection
from gridguard.policy import BUNDLED_ARTIFACT_SHA256, load_ppo_policy
from gridguard.resources import resource_path
from gridguard.scenarios import generate_scenario
from gridguard.simulation import simulate
from gridguard.ui.live_models import (
    LiveScenarioComparison,
    LiveScenarioFrame,
    LocalityHourDetail,
)
from gridguard.ui.map_asset import EXPECTED_LOCALITY_IDS


@dataclass(frozen=True, slots=True, eq=False)
class LiveCudaExecution:
    """Two controller traces that retain one exact shared CUDA scenario."""

    scenario: Scenario
    ppo: SimulationResult
    fixed_rotation: SimulationResult

    def __post_init__(self) -> None:
        if self.scenario.batch_size != 1:
            raise ValueError("live execution requires exactly one scenario")
        if (
            self.ppo.scenario is not self.scenario
            or self.fixed_rotation.scenario is not self.scenario
        ):
            raise ValueError("live controllers must share the exact same Scenario object")
        if self.ppo.controller_name != ControllerName.PPO:
            raise ValueError("ppo result must be labeled PPO")
        if self.fixed_rotation.controller_name != ControllerName.FIXED_ROTATION:
            raise ValueError("baseline result must be labeled Fixed Rotation")

        device = self.scenario.demand.device
        if device.type != "cuda" or device.index != 0:
            raise ValueError("live execution must remain on cuda:0")
        tensors = (
            self.scenario.supply,
            self.ppo.service_tiers,
            self.ppo.requested,
            self.ppo.served,
            self.ppo.reward,
            self.fixed_rotation.service_tiers,
            self.fixed_rotation.requested,
            self.fixed_rotation.served,
            self.fixed_rotation.reward,
        )
        if any(tensor.device != device for tensor in tensors):
            raise ValueError("all live scenario and result tensors must remain on cuda:0")


def execute_live_scenario_cuda(
    shortage_level: str,
    seed: int,
    *,
    manifest_path: str | Path | None = None,
    device: str | torch.device = "cuda:0",
) -> LiveCudaExecution:
    """Generate once and run both controllers against the same CUDA scenario."""

    resolved_device = torch.device(device)
    if resolved_device.type != "cuda" or resolved_device.index != 0:
        raise ValueError("live execution requires explicit device cuda:0")
    actor_manifest = (
        resource_path("data", "model", "manifest.json")
        if manifest_path is None
        else Path(manifest_path)
    )
    scenario = generate_scenario(
        seed,
        shortage_level,
        device=resolved_device,
    )
    actor = load_ppo_policy(
        actor_manifest,
        expected_artifact_sha256=BUNDLED_ARTIFACT_SHA256,
    )
    fixed_result = simulate(scenario, FixedRotationController())
    ppo_result = simulate(scenario, PPOController(actor))
    return LiveCudaExecution(
        scenario=scenario,
        ppo=ppo_result,
        fixed_rotation=fixed_result,
    )


def build_live_comparison(execution: LiveCudaExecution) -> LiveScenarioComparison:
    """Perform one explicit, final 24-row CUDA-to-host display transfer."""

    if not isinstance(execution, LiveCudaExecution):
        raise TypeError("execution must be a LiveCudaExecution")
    demand = execution.scenario.demand[0]
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
            execution.ppo.service_tiers[0].to(dtype=torch.float32),
            execution.ppo.served[0].flatten(start_dim=1),
            execution.fixed_rotation.service_tiers[0].to(dtype=torch.float32),
            execution.fixed_rotation.served[0].flatten(start_dim=1),
        ),
        dim=-1,
    )
    locality_count = len(EXPECTED_LOCALITY_IDS)
    sector_count = len(SECTORS)
    expected_width = 1 + locality_count * sector_count * 3 + locality_count * 3
    if packed_display.shape != (24, expected_width):
        raise ValueError("live display tensor has an unexpected shape")
    if packed_display.device.type != "cuda" or packed_display.device.index != 0:
        raise ValueError("live display tensor left cuda:0 before final transfer")

    # This is the only bulk device-to-host transfer in the UI runner. All simulation,
    # dispatch, and PPO inference tensors stay on cuda:0 until this 24×127 display tensor.
    host_rows = packed_display.to(device="cpu", non_blocking=False).tolist()
    ppo_frames: list[LiveScenarioFrame] = []
    fixed_frames: list[LiveScenarioFrame] = []
    for hour, row in enumerate(host_rows):
        supply = row[0]
        offset = 1
        demand_values = row[offset : offset + locality_count * sector_count]
        offset += locality_count * sector_count
        projection_values = row[offset : offset + locality_count]
        offset += locality_count
        ppo_tiers = row[offset : offset + locality_count]
        offset += locality_count
        ppo_served = row[offset : offset + locality_count * sector_count]
        offset += locality_count * sector_count
        fixed_tiers = row[offset : offset + locality_count]
        offset += locality_count
        fixed_served = row[offset : offset + locality_count * sector_count]
        next_projection = projection_values if 2 <= hour <= 22 else None
        ppo_frames.append(
            _frame(hour, supply, demand_values, next_projection, ppo_tiers, ppo_served)
        )
        fixed_frames.append(
            _frame(hour, supply, demand_values, next_projection, fixed_tiers, fixed_served)
        )
    return LiveScenarioComparison(
        shortage_level=execution.scenario.shortage_level.value,
        seed=execution.scenario.seed,
        device="cuda:0",
        frames_by_controller={
            ControllerName.PPO.value: tuple(ppo_frames),
            ControllerName.FIXED_ROTATION.value: tuple(fixed_frames),
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

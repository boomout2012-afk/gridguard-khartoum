"""Additive fresh-scenario learning-stage audit; all prior science stays frozen."""

from __future__ import annotations

import re

import torch

from gridguard.allocation_study import METRIC_NAMES
from gridguard.committee_baseline import DIAGNOSTIC_NAMES
from gridguard.domain import SimulationResult

TEST_SEEDS = tuple(range(132201, 132211))
ENDPOINT_NAMES = ("final_minimum_locality_service_ratio", "final_locality_service_range_ratio")
METRICS = (*METRIC_NAMES, *DIAGNOSTIC_NAMES, *ENDPOINT_NAMES)


def collect_seed_metadata(payload: object) -> tuple[set[int], list[dict]]:
    """Read seed fields and bounded declared arithmetic streams, never execute strings.

    Metric-array integers are not seeds. Formula expansion is restricted to the
    two published stream grammars, using the enclosing config's run/iteration counts.
    Other textual seed descriptions are recorded for review, without claiming that
    arbitrary prose is executable provenance.
    """
    values: set[int] = set()
    formulas: list[dict] = []

    def add_numbers(value):
        if type(value) is int:
            values.add(value)
        elif isinstance(value, list):
            for item in value:
                add_numbers(item)

    def walk(value, path):
        if isinstance(value, dict):
            for key, item in value.items():
                label = f"{path}.{key}"
                if "seed" in key.lower():
                    add_numbers(item)
                    if key.endswith("seed_start") and type(item) is int:
                        end = value.get(key.removesuffix("start") + "end_inclusive")
                        if type(end) is int and item <= end < item + 1_000_000:
                            values.update(range(item, end + 1))
                    if isinstance(item, str) and "+" in item:
                        normalized = item.replace(" ", "")
                        match = re.fullmatch(
                            r"(\d+)\+run_index\*(\d+)(\+iteration_index)?", normalized
                        )
                        if match:
                            runs = len(value.get("training_seeds", ()))
                            iterations = value.get("iterations", 1) if match[3] else 1
                            if (
                                not runs
                                or type(iterations) is not int
                                or not 0 < iterations < 100000
                            ):
                                raise ValueError(f"seed formula lacks bounded dimensions: {label}")
                            expanded = {
                                int(match[1]) + run * int(match[2]) + iteration
                                for run in range(runs)
                                for iteration in range(iterations)
                            }
                            values.update(expanded)
                            formulas.append(
                                {"field": label, "formula": item, "expanded_count": len(expanded)}
                            )
                        else:
                            formulas.append(
                                {"field": label, "formula": item, "expanded_count": None}
                            )
                if isinstance(item, (dict, list)):
                    walk(item, label)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, (dict, list)):
                    walk(item, f"{path}[{index}]")

    walk(payload, "root")
    return values, formulas


def assert_disjoint_seeds(proposed: tuple[int, ...], previous: set[int]) -> None:
    if (
        not proposed
        or len(set(proposed)) != len(proposed)
        or any(type(s) is not int for s in proposed)
    ):
        raise ValueError("fresh scenario seeds must be unique integers")
    overlap = set(proposed) & previous
    if overlap:
        raise ValueError(f"fresh scenario seeds overlap prior metadata: {sorted(overlap)}")


def contrast_cubes(cube: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """First contrast pairs selected actors with their own BC initialization."""
    if cube.ndim != 5 or cube.shape[0] != 14 or cube.shape[1] != 3:
        raise ValueError("learning audit cube must have [14,3,seed,draw,metric] axes")
    return (cube[9:14] - cube[4:9], *(cube[9:14] - cube[index] for index in range(4)))


@torch.inference_mode()
def independent_trace_metrics(
    result: SimulationResult,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Reconstruct all 20 canonical scalar metrics independently in CUDA float64.

    This consumes the same frozen simulator trace, so it verifies metric arithmetic
    rather than constituting an independent simulator or physical-model validation.
    """
    if result.served.device.type != "cuda":
        raise ValueError("independent audit metrics require CUDA")
    demand, served, requested = (
        v.to(torch.float64) for v in (result.scenario.demand, result.served, result.requested)
    )
    supply = result.scenario.supply.to(torch.float64)
    tiers = result.service_tiers
    dt, yt, qt = (v.sum((-1, -2)) for v in (demand, served, requested))
    ld, ly = demand.sum(-1), served.sum(-1)
    cd, cy = demand[..., :3].sum(-1), served[..., :3].sum(-1)
    ratios = ly.cumsum(1) / ld.cumsum(1).clamp_min(1e-8)
    fairness = ratios.sum(-1).square() / (7 * ratios.square().sum(-1) + 1e-8)
    utilization = yt / torch.minimum(supply, dt).clamp_min(1e-8)
    switching = torch.cat(
        (torch.zeros_like(supply[:, :1]), (tiers[:, 1:] != tiers[:, :-1]).double().mean(-1)), 1
    )
    oversub = (qt - supply).clamp_min(0) / dt.clamp_min(1e-8)
    critical_hourly = cy.sum(-1) / cd.sum(-1).clamp_min(1e-8)
    reward = (
        5 * critical_hourly + 2 * utilization + 2 * fairness - 0.25 * switching - 3 * oversub
    ) / 9
    blackout = (ld > 1e-8) & (ly == 0)
    interruption = (cd > 1e-8) & (cy / cd.clamp_min(1e-8) < 0.99)
    out = {
        "episode_reward": reward.sum(1),
        "demand_mwh": dt.sum(1),
        "available_mwh": supply.sum(1),
        "served_mwh": yt.sum(1),
        "curtailed_mwh": (dt - yt).sum(1),
        "critical_service_ratio": cy.sum((1, 2)) / cd.sum((1, 2)).clamp_min(1e-8),
        "utilization": utilization.mean(1),
        "fairness": fairness.mean(1),
        "switching_rate": switching.mean(1),
        "oversubscription": oversub.mean(1),
        "power_balance_violation_rate": (yt > supply + 0.001).double().mean(1),
        "critical_outage_locality_hours": interruption.double().sum((1, 2)),
    }
    pair = torch.triu_indices(7, 7, offset=1, device=demand.device)
    for name, mask in (("blackout", blackout), ("critical_interruption", interruption)):
        durations = mask.double().sum(1)
        current = torch.zeros_like(durations)
        longest = current.clone()
        for hour in range(24):
            current = torch.where(mask[:, hour], current + 1, 0)
            longest = torch.maximum(longest, current)
        out[f"{name}_locality_hours"] = durations.sum(1)
        out[f"{name}_duration_range_hours"] = durations.amax(1) - durations.amin(1)
        out[f"{name}_duration_mean_pairwise_gap_hours"] = (
            (durations[:, pair[0]] - durations[:, pair[1]]).abs().mean(1)
        )
        out[f"longest_{name}_hours"] = longest.amax(1)
    endpoints = ly.sum(1) / ld.sum(1).clamp_min(1e-8)
    return out, endpoints


def verify_trace_metrics(canonical: dict, reconstructed: dict) -> torch.Tensor:
    """Reject metric disagreement using predeclared float32 accumulation tolerances."""
    errors = []
    for name, reference in reconstructed.items():
        if name not in canonical:
            raise ValueError(f"canonical metric missing: {name}")
        tolerance = 0.01 if name.endswith("_mwh") else (5e-5 if name == "episode_reward" else 5e-6)
        error = (canonical[name].double() - reference).abs().amax()
        if not bool(torch.isfinite(error)) or bool(error > tolerance):
            raise ValueError(f"independent metric reconstruction failed: {name}")
        errors.append(error)
    return torch.stack(errors)

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pytest
import torch

import gridguard
import gridguard.evaluation as evaluation_module
from gridguard.controllers import PPOController
from gridguard.domain import ControllerName, Scenario, canonical_scenario_sha256
from gridguard.evaluation import (
    BATCH_SIZE,
    COUNT_STATEMENT,
    FROZEN_ABSOLUTE_TOLERANCES,
    FROZEN_BENCHMARK_SHA256,
    INFERENCE_METRICS,
    METRIC_ORDER,
    PAIRED_BENCHMARK_SHA256,
    SEEDS,
    SEVERITIES,
    EvaluationArtifactError,
    assert_paired_benchmark_integrity,
    build_paired_benchmark_payload,
    canonical_paired_json_bytes,
    evaluate_scenario_pair,
    run_full_paired_evaluation,
    validate_paired_benchmark_payload,
    verify_paired_benchmark,
    verify_paired_benchmark_candidate,
    verify_paired_inference_on_cuda,
    write_paired_benchmark,
    write_paired_benchmark_candidate,
)
from gridguard.paired_inference import PairedInferenceError
from gridguard.policy import BUNDLED_ARTIFACT_SHA256
from gridguard.scenarios import generate_scenarios

PROJECT_ROOT = Path(__file__).parents[1]
PPO_MANIFEST = PROJECT_ROOT / "data" / "model" / "manifest.json"
FROZEN_BENCHMARK = PROJECT_ROOT / "data" / "evidence" / "frozen-benchmark.json"
PAIRED_BENCHMARK = PROJECT_ROOT / "data" / "evidence" / "paired-benchmark.json"


class _FullServicePolicy:
    def act(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.full(
            (observation.shape[0], 7),
            2,
            dtype=torch.long,
            device=observation.device,
        )


class _MutatingPolicy:
    def __init__(self, scenario: Scenario, field: str) -> None:
        self._scenario = scenario
        self._field = field
        self._mutated = False

    def act(self, observation: torch.Tensor) -> torch.Tensor:
        if not self._mutated:
            getattr(self._scenario, self._field).flatten()[0].add_(1.0)
            self._mutated = True
        return torch.full(
            (observation.shape[0], 7),
            2,
            dtype=torch.long,
            device=observation.device,
        )


class _MutatingFixedController:
    def __init__(self, scenario: Scenario) -> None:
        self._scenario = scenario
        self._mutated = False

    @property
    def name(self) -> ControllerName:
        return ControllerName.FIXED_ROTATION

    def act(self, context) -> torch.Tensor:
        if not self._mutated:
            self._scenario.supply.flatten()[0].add_(1.0)
            self._mutated = True
        return torch.zeros(
            (context.observation.shape[0], 7),
            dtype=torch.long,
            device=context.observation.device,
        )


class _RestoringMutatingPolicy:
    def __init__(self, scenario: Scenario) -> None:
        self._scenario = scenario
        self._original = scenario.demand.flatten()[0].clone()
        self._calls = 0

    def act(self, observation: torch.Tensor) -> torch.Tensor:
        self._calls += 1
        if self._calls == 1:
            self._scenario.demand.flatten()[0].mul_(0.5)
        elif self._calls == 2:
            self._scenario.demand.flatten()[0].copy_(self._original)
        return torch.full(
            (observation.shape[0], 7),
            2,
            dtype=torch.long,
            device=observation.device,
        )


class _RestoringMutatingFixedController:
    def __init__(self, scenario: Scenario) -> None:
        self._scenario = scenario
        self._original = scenario.supply.flatten()[0].clone()
        self._calls = 0

    @property
    def name(self) -> ControllerName:
        return ControllerName.FIXED_ROTATION

    def act(self, context) -> torch.Tensor:
        self._calls += 1
        if self._calls == 1:
            self._scenario.supply.flatten()[0].add_(1.0)
        elif self._calls == 2:
            self._scenario.supply.flatten()[0].copy_(self._original)
        return torch.zeros(
            (context.observation.shape[0], 7),
            dtype=torch.long,
            device=context.observation.device,
        )


class _DataRestoringMutatingPolicy:
    def __init__(self, scenario: Scenario, field: str) -> None:
        self._tensor = getattr(scenario, field)
        self._original = self._tensor.flatten()[0].clone()
        self._calls = 0

    def act(self, observation: torch.Tensor) -> torch.Tensor:
        self._calls += 1
        if self._calls == 1:
            self._tensor.data.flatten()[0].mul_(0.5)
        elif self._calls == 2:
            self._tensor.data.flatten()[0].copy_(self._original)
        return torch.full(
            (observation.shape[0], 7),
            2,
            dtype=torch.long,
            device=observation.device,
        )


class _DataRestoringMutatingFixedController:
    def __init__(self, scenario: Scenario, field: str) -> None:
        self._tensor = getattr(scenario, field)
        self._original = self._tensor.flatten()[0].clone()
        self._calls = 0

    @property
    def name(self) -> ControllerName:
        return ControllerName.FIXED_ROTATION

    def act(self, context) -> torch.Tensor:
        self._calls += 1
        if self._calls == 1:
            self._tensor.data.flatten()[0].mul_(0.5)
        elif self._calls == 2:
            self._tensor.data.flatten()[0].copy_(self._original)
        return torch.zeros(
            (context.observation.shape[0], 7),
            dtype=torch.long,
            device=context.observation.device,
        )


class _MetadataRestoringMutatingPolicy:
    def __init__(self, scenario: Scenario, field: str) -> None:
        self._tensor = getattr(scenario, field)
        self._calls = 0

    def act(self, observation: torch.Tensor) -> torch.Tensor:
        self._calls += 1
        if self._calls == 1:
            self._tensor.data = self._tensor.data.view(torch.int32)
        elif self._calls == 2:
            self._tensor.data = self._tensor.data.view(torch.float32)
        return torch.full(
            (observation.shape[0], 7),
            2,
            dtype=torch.long,
            device=observation.device,
        )


class _MetadataRestoringMutatingFixedController:
    def __init__(self, scenario: Scenario, field: str) -> None:
        self._tensor = getattr(scenario, field)
        self._calls = 0

    @property
    def name(self) -> ControllerName:
        return ControllerName.FIXED_ROTATION

    def act(self, context) -> torch.Tensor:
        self._calls += 1
        if self._calls == 1:
            self._tensor.data = self._tensor.data.view(torch.int32)
        elif self._calls == 2:
            self._tensor.data = self._tensor.data.view(torch.float32)
        return torch.zeros(
            (context.observation.shape[0], 7),
            dtype=torch.long,
            device=context.observation.device,
        )


def _provenance() -> dict[str, object]:
    return {
        "frozen_benchmark": {
            "path": "data/evidence/frozen-benchmark.json",
            "sha256": FROZEN_BENCHMARK_SHA256,
            "size_bytes": 9682,
        },
        "ppo_actor": {
            "path": "data/model/ppo-policy.npz",
            "sha256": BUNDLED_ARTIFACT_SHA256,
            "size_bytes": 5_160_058,
        },
        "ppo_manifest": {
            "path": "data/model/manifest.json",
            "sha256": "d79170340280c1a33512bdc91565a352bdb071b840afc2f9aee40593c92fcf74",
            "size_bytes": 1609,
        },
        "source_checkpoint": {
            "path": "artifacts/checkpoints/best.pt",
            "sha256": "c6e5712f966b46ba18f4c69fb457b9e69f78887c0c5c0279e5ce034f40f9c665",
            "source_revision": "52f6dd82605c993c8094448caecec86faa289c0c",
        },
    }


def _valid_payload(
    *,
    ppo_outperforms: bool = True,
    frozen_benchmark_path: Path = FROZEN_BENCHMARK,
    supply_multiplier: float = 1.0,
) -> dict[str, object]:
    shape = (len(SEVERITIES), len(SEEDS), BATCH_SIZE, len(METRIC_ORDER))
    fixed = np.zeros(shape, dtype=np.float64)
    ppo = np.zeros(shape, dtype=np.float64)
    values = {
        "episode_reward": (10.0, 11.0),
        "demand_mwh": (100.0, 100.0),
        "available_mwh": (80.0, 80.0),
        "served_mwh": (60.0, 70.0),
        "curtailed_mwh": (40.0, 30.0),
        "critical_service_ratio": (0.4, 0.6),
        "utilization": (0.4, 0.6),
        "fairness": (0.4, 0.6),
        "switching_rate": (0.4, 0.3),
        "oversubscription": (0.1, 0.05),
        "power_balance_violation_rate": (0.0, 0.0),
        "critical_outage_locality_hours": (20.0, 10.0),
    }
    if not ppo_outperforms:
        values = {
            metric: pair if metric in {"demand_mwh", "available_mwh"} else pair[::-1]
            for metric, pair in values.items()
        }
    demand_shape = (len(SEVERITIES), len(SEEDS), BATCH_SIZE, 24, 7, 5)
    supply_shape = (len(SEVERITIES), len(SEEDS), BATCH_SIZE, 24)
    scenario_demand = np.full(demand_shape, 100.0 / (24 * 7 * 5), dtype=np.float64)
    for seed_index in range(len(SEEDS)):
        for batch_index in range(BATCH_SIZE):
            perturbation = float(seed_index * BATCH_SIZE + batch_index + 1) * 1e-6
            scenario_demand[:, seed_index, batch_index, 0, 0, 0] += perturbation
            scenario_demand[:, seed_index, batch_index, 0, 0, 1] -= perturbation

    supply_fractions = np.asarray(
        [
            [0.80] * 24,
            [0.55 if 17 <= hour < 21 else 0.65 for hour in range(24)],
            [0.35 if 16 <= hour < 22 else 0.50 for hour in range(24)],
        ],
        dtype=np.float64,
    )
    hourly_demand = scenario_demand.sum(axis=(-1, -2))
    scenario_supply = hourly_demand * supply_fractions[:, None, None, :] * supply_multiplier
    assert scenario_supply.shape == supply_shape

    for metric_index, metric in enumerate(METRIC_ORDER):
        fixed[..., metric_index] = values[metric][0]
        ppo[..., metric_index] = values[metric][1]
    demand_mwh = scenario_demand.sum(axis=(-1, -2, -3))
    available_mwh = scenario_supply.sum(axis=-1)
    fixed_service_fraction, ppo_service_fraction = (
        (0.75, 0.875) if ppo_outperforms else (0.875, 0.75)
    )
    for controller_values, service_fraction in (
        (fixed, fixed_service_fraction),
        (ppo, ppo_service_fraction),
    ):
        controller_values[..., METRIC_ORDER.index("demand_mwh")] = demand_mwh
        controller_values[..., METRIC_ORDER.index("available_mwh")] = available_mwh
        served_mwh = available_mwh * service_fraction
        controller_values[..., METRIC_ORDER.index("served_mwh")] = served_mwh
        controller_values[..., METRIC_ORDER.index("curtailed_mwh")] = demand_mwh - served_mwh

    bootstrap = np.zeros((3, len(INFERENCE_METRICS)), dtype=np.float64)
    for index, metric in enumerate(INFERENCE_METRICS):
        delta = float(ppo[..., METRIC_ORDER.index(metric)].mean()) - float(
            fixed[..., METRIC_ORDER.index(metric)].mean()
        )
        bootstrap[:, index] = delta

    return build_paired_benchmark_payload(
        fixed_values=fixed,
        ppo_values=ppo,
        bootstrap_values=bootstrap,
        scenario_demand_values=scenario_demand,
        scenario_supply_values=scenario_supply,
        provenance=_provenance(),
        frozen_benchmark_path=frozen_benchmark_path,
    )


def _refresh_scenario_hash(payload: dict[str, object], row_position: int) -> None:
    rows = payload["paired_rows"]
    cell_start = row_position // BATCH_SIZE * BATCH_SIZE
    cell_rows = rows[cell_start : cell_start + BATCH_SIZE]
    target = rows[row_position]
    scenario = Scenario(
        demand=torch.tensor(
            [row["scenario_trace"]["demand_mw"] for row in cell_rows],
            dtype=torch.float64,
        ),
        supply=torch.tensor(
            [row["scenario_trace"]["supply_mw"] for row in cell_rows],
            dtype=torch.float64,
        ),
        shortage_level=target["shortage_level"],
        seed=target["seed"],
    )
    target["scenario_sha256"] = canonical_scenario_sha256(
        scenario,
        batch_index=row_position - cell_start,
    )


def _trusted_schema_three_payload() -> dict[str, object]:
    committed = json.loads(PAIRED_BENCHMARK.read_text(encoding="utf-8"))
    if committed.get("schema_version") != 3:
        raise AssertionError("the committed paired benchmark has not been promoted to schema 3")
    return committed


def test_evaluate_pair_reuses_exact_scenario_object() -> None:
    scenario = generate_scenarios(2, 91, "mild", device="cpu")

    evaluated = evaluate_scenario_pair(
        scenario,
        ppo_controller=PPOController(_FullServicePolicy()),
    )

    assert evaluated.fixed_result.scenario is scenario
    assert evaluated.ppo_result.scenario is scenario
    assert evaluated.fixed_metrics.batch_size == scenario.batch_size
    assert evaluated.ppo_metrics.batch_size == scenario.batch_size


@pytest.mark.gpu
def test_evaluate_pair_fails_closed_when_ppo_mutates_cuda_demand() -> None:
    scenario = generate_scenarios(2, 91, "mild", device="cuda:0")

    with pytest.raises(EvaluationArtifactError, match="PPO mutated scenario demand"):
        evaluate_scenario_pair(
            scenario,
            ppo_controller=PPOController(_MutatingPolicy(scenario, "demand")),
        )


@pytest.mark.gpu
def test_evaluate_pair_fails_closed_when_fixed_rotation_mutates_cuda_supply(
    monkeypatch,
) -> None:
    scenario = generate_scenarios(2, 91, "mild", device="cuda:0")
    mutating_controller = _MutatingFixedController(scenario)
    monkeypatch.setattr(
        evaluation_module,
        "FixedRotationController",
        lambda: mutating_controller,
    )

    with pytest.raises(EvaluationArtifactError, match="Fixed Rotation mutated scenario supply"):
        evaluate_scenario_pair(
            scenario,
            ppo_controller=PPOController(_FullServicePolicy()),
        )


@pytest.mark.gpu
def test_evaluate_pair_fails_closed_when_ppo_restores_cuda_demand_bytes() -> None:
    scenario = generate_scenarios(2, 91, "mild", device="cuda:0")

    with pytest.raises(EvaluationArtifactError, match="PPO mutated scenario demand"):
        evaluate_scenario_pair(
            scenario,
            ppo_controller=PPOController(_RestoringMutatingPolicy(scenario)),
        )


@pytest.mark.gpu
def test_evaluate_pair_fails_closed_when_fixed_rotation_restores_cuda_supply(
    monkeypatch,
) -> None:
    scenario = generate_scenarios(2, 91, "mild", device="cuda:0")
    mutating_controller = _RestoringMutatingFixedController(scenario)
    monkeypatch.setattr(
        evaluation_module,
        "FixedRotationController",
        lambda: mutating_controller,
    )

    with pytest.raises(EvaluationArtifactError, match="Fixed Rotation mutated scenario supply"):
        evaluate_scenario_pair(
            scenario,
            ppo_controller=PPOController(_FullServicePolicy()),
        )


@pytest.mark.gpu
@pytest.mark.parametrize("field", ["demand", "supply"])
def test_evaluate_pair_fails_closed_on_ppo_data_write_before_cross_step_restore(
    field: str,
) -> None:
    scenario = generate_scenarios(2, 91, "mild", device="cuda:0")

    with pytest.raises(EvaluationArtifactError, match=rf"PPO mutated scenario {field}"):
        evaluate_scenario_pair(
            scenario,
            ppo_controller=PPOController(_DataRestoringMutatingPolicy(scenario, field)),
        )


@pytest.mark.gpu
@pytest.mark.parametrize("field", ["demand", "supply"])
def test_evaluate_pair_fails_closed_on_fixed_data_write_before_cross_step_restore(
    monkeypatch,
    field: str,
) -> None:
    scenario = generate_scenarios(2, 91, "mild", device="cuda:0")
    monkeypatch.setattr(
        evaluation_module,
        "FixedRotationController",
        lambda: _DataRestoringMutatingFixedController(scenario, field),
    )

    with pytest.raises(
        EvaluationArtifactError,
        match=rf"Fixed Rotation mutated scenario {field}",
    ):
        evaluate_scenario_pair(
            scenario,
            ppo_controller=PPOController(_FullServicePolicy()),
        )


@pytest.mark.gpu
@pytest.mark.parametrize("field", ["demand", "supply"])
def test_evaluate_pair_fails_closed_on_ppo_metadata_rebind_before_restore(
    field: str,
) -> None:
    scenario = generate_scenarios(2, 91, "mild", device="cuda:0")

    with pytest.raises(EvaluationArtifactError, match=rf"PPO mutated scenario {field}"):
        evaluate_scenario_pair(
            scenario,
            ppo_controller=PPOController(_MetadataRestoringMutatingPolicy(scenario, field)),
        )


@pytest.mark.gpu
@pytest.mark.parametrize("field", ["demand", "supply"])
def test_evaluate_pair_fails_closed_on_fixed_metadata_rebind_before_restore(
    monkeypatch,
    field: str,
) -> None:
    scenario = generate_scenarios(2, 91, "mild", device="cuda:0")
    monkeypatch.setattr(
        evaluation_module,
        "FixedRotationController",
        lambda: _MetadataRestoringMutatingFixedController(scenario, field),
    )

    with pytest.raises(
        EvaluationArtifactError,
        match=rf"Fixed Rotation mutated scenario {field}",
    ):
        evaluate_scenario_pair(
            scenario,
            ppo_controller=PPOController(_FullServicePolicy()),
        )


def test_payload_uses_unambiguous_matched_counts_and_directions() -> None:
    payload = _valid_payload()

    assert payload["protocol"]["matched_scenario_count"] == 288
    assert payload["protocol"]["controller_evaluation_count"] == 576
    assert payload["protocol"]["independent_base_draw_count"] == 96
    assert payload["protocol"]["severity_variants_per_base_draw"] == 3
    assert payload["protocol"]["count_statement"] == COUNT_STATEMENT
    assert COUNT_STATEMENT == (
        "96 independent base draws crossed with 3 shortage levels = 288 matched scenario "
        "conditions; 576 controller-scenario evaluations (288 per controller)."
    )
    assert payload["protocol"]["delta_definition"] == "PPO - Fixed Rotation"
    assert payload["protocol"]["bootstrap"] == {
        "confidence_level": 0.95,
        "method": "deterministic seed-stratified paired cluster percentile bootstrap",
        "percentile_method": "linear",
        "replicates": 10_000,
        "seed": 20260815,
    }
    assert payload["metric_definitions"]["served_mwh"]["direction"] == "higher_is_better"
    assert payload["metric_definitions"]["curtailed_mwh"]["direction"] == "lower_is_better"
    assert payload["paired_inference"]["curtailed_mwh"]["favorable_sign"] == "negative"


def test_payload_binds_unique_canonical_scenario_hashes_in_protocol_order() -> None:
    payload = _valid_payload()
    rows = payload["paired_rows"]
    hashes = [row["scenario_sha256"] for row in rows]

    assert payload["schema_version"] == 3
    assert all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in hashes)
    assert len(set(hashes)) == len(rows) == 288

    first_cell_rows = rows[:BATCH_SIZE]
    demand = np.asarray(
        [row["scenario_trace"]["demand_mw"] for row in first_cell_rows],
        dtype=np.float64,
    )
    supply = np.asarray(
        [row["scenario_trace"]["supply_mw"] for row in first_cell_rows],
        dtype=np.float64,
    )
    first_cell = Scenario(
        demand=torch.from_numpy(demand),
        supply=torch.from_numpy(supply),
        shortage_level=SEVERITIES[0],
        seed=SEEDS[0],
    )
    assert hashes[:2] == [
        canonical_scenario_sha256(first_cell, batch_index=0),
        canonical_scenario_sha256(first_cell, batch_index=1),
    ]
    assert demand.shape == (BATCH_SIZE, 24, 7, 5)
    assert supply.shape == (BATCH_SIZE, 24)
    assert sum(first_cell_rows[0]["scenario_trace"]["demand_mw"][0][0]) > 0.0


def test_validator_rejects_fabricated_or_reordered_scenario_hashes() -> None:
    fabricated = _valid_payload()
    fabricated["paired_rows"][0]["scenario_sha256"] = hashlib.sha256(b"fabricated").hexdigest()
    with pytest.raises(EvaluationArtifactError, match="does not match scenario_trace"):
        validate_paired_benchmark_payload(fabricated)

    reordered = _valid_payload()
    first = reordered["paired_rows"][0]["scenario_sha256"]
    reordered["paired_rows"][0]["scenario_sha256"] = reordered["paired_rows"][1]["scenario_sha256"]
    reordered["paired_rows"][1]["scenario_sha256"] = first
    with pytest.raises(EvaluationArtifactError, match="does not match scenario_trace"):
        validate_paired_benchmark_payload(reordered)


def test_validator_rejects_cross_severity_demand_drift_with_self_consistent_hash() -> None:
    payload = _valid_payload()
    moderate_row_position = len(SEEDS) * BATCH_SIZE
    trace = payload["paired_rows"][moderate_row_position]["scenario_trace"]["demand_mw"]
    trace[0][0][0] += 0.001
    trace[0][0][1] -= 0.001
    _refresh_scenario_hash(payload, moderate_row_position)

    with pytest.raises(EvaluationArtifactError, match="shared base draw demand"):
        validate_paired_benchmark_payload(payload)


def test_validator_rejects_cross_severity_supply_draw_drift_with_preserved_total() -> None:
    payload = _valid_payload()
    moderate_row_position = len(SEEDS) * BATCH_SIZE
    supply = payload["paired_rows"][moderate_row_position]["scenario_trace"]["supply_mw"]
    supply[0] += 0.001
    supply[1] -= 0.001
    _refresh_scenario_hash(payload, moderate_row_position)

    with pytest.raises(EvaluationArtifactError, match="shared base draw supply"):
        validate_paired_benchmark_payload(payload)


@pytest.mark.parametrize(
    ("target", "value"),
    [
        (("protocol", "matched_scenario_count"), 288.0),
        (("paired_rows", 0, "seed"), 50001.0),
        (("paired_rows", 0, "batch_index"), False),
    ],
)
def test_validator_rejects_identity_values_with_equal_but_wrong_json_types(
    target: tuple[object, ...],
    value: object,
) -> None:
    payload = _valid_payload()
    owner = payload
    for part in target[:-1]:
        owner = owner[part]
    owner[target[-1]] = value

    with pytest.raises(EvaluationArtifactError, match="type|identity|protocol"):
        validate_paired_benchmark_payload(payload)


def test_trusted_integrity_rejects_non_golden_scenario_bundle() -> None:
    payload = _valid_payload()

    with pytest.raises(EvaluationArtifactError, match="reviewed CUDA scenario bundle"):
        assert_paired_benchmark_integrity(
            payload,
            frozen_benchmark_path=FROZEN_BENCHMARK,
        )


@pytest.mark.parametrize(
    ("trace_field", "metric"),
    (("demand_mw", "demand_mwh"), ("supply_mw", "available_mwh")),
)
def test_validator_recomputes_trace_energy_totals(trace_field: str, metric: str) -> None:
    payload = _valid_payload()
    rows = payload["paired_rows"]
    trace = rows[0]["scenario_trace"][trace_field]
    if trace_field == "demand_mw":
        trace[0][0][0] += 0.01
    else:
        trace[0] += 0.01

    first_cell = Scenario(
        demand=torch.tensor(
            [row["scenario_trace"]["demand_mw"] for row in rows[:BATCH_SIZE]],
            dtype=torch.float64,
        ),
        supply=torch.tensor(
            [row["scenario_trace"]["supply_mw"] for row in rows[:BATCH_SIZE]],
            dtype=torch.float64,
        ),
        shortage_level=SEVERITIES[0],
        seed=SEEDS[0],
    )
    rows[0]["scenario_sha256"] = canonical_scenario_sha256(first_cell, batch_index=0)

    with pytest.raises(EvaluationArtifactError, match=rf"trace-derived {metric}"):
        validate_paired_benchmark_payload(payload)


def test_validator_rejects_non_strict_scenario_trace_shape_and_values() -> None:
    wrong_shape = _valid_payload()
    wrong_shape["paired_rows"][0]["scenario_trace"]["supply_mw"].pop()
    with pytest.raises(EvaluationArtifactError, match="scenario_trace.supply_mw must have shape"):
        validate_paired_benchmark_payload(wrong_shape)

    boolean_value = _valid_payload()
    boolean_value["paired_rows"][0]["scenario_trace"]["demand_mw"][0][0][0] = True
    with pytest.raises(EvaluationArtifactError, match="must contain only finite floating values"):
        validate_paired_benchmark_payload(boolean_value)


def test_schema_two_is_host_readable_but_not_trusted_evidence(tmp_path: Path) -> None:
    legacy = _valid_payload()
    legacy["schema_version"] = 2
    for row in legacy["paired_rows"]:
        row.pop("scenario_sha256")
        row.pop("scenario_trace")
    output = tmp_path / "paired-benchmark-v2.json"

    validate_paired_benchmark_payload(legacy)
    digest = write_paired_benchmark_candidate(legacy, output)
    assert verify_paired_benchmark_candidate(output, expected_sha256=digest) == legacy
    with pytest.raises(EvaluationArtifactError, match="SHA-256 mismatch"):
        verify_paired_benchmark(
            output,
            frozen_benchmark_path=FROZEN_BENCHMARK,
        )
    with pytest.raises(EvaluationArtifactError, match="requires paired schema_version 3"):
        assert_paired_benchmark_integrity(
            legacy,
            frozen_benchmark_path=FROZEN_BENCHMARK,
        )


@pytest.mark.parametrize("complex_input", ["fixed", "bootstrap"])
def test_payload_builder_rejects_complex_numeric_arrays(complex_input: str) -> None:
    shape = (len(SEVERITIES), len(SEEDS), BATCH_SIZE, len(METRIC_ORDER))
    fixed = np.zeros(shape, dtype=np.float64)
    ppo = np.zeros(shape, dtype=np.float64)
    bootstrap = np.zeros((3, len(INFERENCE_METRICS)), dtype=np.float64)
    if complex_input == "fixed":
        fixed = fixed.astype(np.complex128)
    else:
        bootstrap = bootstrap.astype(np.complex128)

    with pytest.raises(EvaluationArtifactError, match="real floating"):
        build_paired_benchmark_payload(
            fixed_values=fixed,
            ppo_values=ppo,
            bootstrap_values=bootstrap,
            scenario_demand_values=np.ones(
                (len(SEVERITIES), len(SEEDS), BATCH_SIZE, 24, 7, 5),
                dtype=np.float64,
            ),
            scenario_supply_values=np.ones(
                (len(SEVERITIES), len(SEEDS), BATCH_SIZE, 24),
                dtype=np.float64,
            ),
            provenance=_provenance(),
            frozen_benchmark_path=FROZEN_BENCHMARK,
        )


def test_payload_builder_requires_the_hashed_frozen_benchmark(tmp_path: Path) -> None:
    tampered_frozen = tmp_path / "frozen-benchmark.json"
    tampered_frozen.write_bytes(FROZEN_BENCHMARK.read_bytes() + b" ")

    with pytest.raises(EvaluationArtifactError, match="frozen benchmark SHA-256 mismatch"):
        _valid_payload(frozen_benchmark_path=tampered_frozen)


def test_neutral_serialization_accepts_adverse_results_but_trusted_writer_rejects_non_golden(
    tmp_path: Path,
) -> None:
    payload = _valid_payload(ppo_outperforms=False)
    output = tmp_path / "adverse-paired-benchmark.json"

    validate_paired_benchmark_payload(payload)
    digest = write_paired_benchmark_candidate(payload, output)

    assert payload["frozen_parity"]["status"] == "failed"
    assert verify_paired_benchmark_candidate(output, expected_sha256=digest) == payload
    with pytest.raises(EvaluationArtifactError, match="reviewed CUDA scenario bundle"):
        assert_paired_benchmark_integrity(
            payload,
            frozen_benchmark_path=FROZEN_BENCHMARK,
        )
    trusted_output = tmp_path / "trusted-adverse-paired-benchmark.json"
    with pytest.raises(EvaluationArtifactError, match="reviewed CUDA scenario bundle"):
        write_paired_benchmark(
            payload,
            trusted_output,
            frozen_benchmark_path=FROZEN_BENCHMARK,
        )
    assert not trusted_output.exists()


def test_neutral_validation_is_sign_neutral_for_ambiguous_interval() -> None:
    payload = _trusted_schema_three_payload()
    interval = payload["paired_inference"]["critical_service_ratio"]
    interval["ci95_lower"] = -0.01

    validate_paired_benchmark_payload(payload)
    canonical_paired_json_bytes(payload)


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("endpoint", "offset"),
    (
        ("mean_delta", 5e-10),
        ("ci95_lower", -0.01),
        ("ci95_upper", 0.01),
    ),
)
def test_trusted_integrity_rejects_fabricated_paired_inference_endpoint(
    endpoint: str,
    offset: float,
) -> None:
    payload = _trusted_schema_three_payload()
    interval = payload["paired_inference"]["critical_service_ratio"]
    interval[endpoint] += offset

    validate_paired_benchmark_payload(payload)

    with pytest.raises(EvaluationArtifactError, match="paired inference.*reconstruct"):
        verify_paired_inference_on_cuda(payload)


def test_candidate_paths_do_not_cross_the_cuda_reconstruction_boundary(
    monkeypatch,
    tmp_path: Path,
) -> None:
    payload = _trusted_schema_three_payload()
    monkeypatch.setattr(
        evaluation_module,
        "verify_paired_inference_reconstruction",
        lambda **_kwargs: pytest.fail("candidate verification touched trusted CUDA inference"),
    )
    output = tmp_path / "candidate.json"

    digest = write_paired_benchmark_candidate(payload, output)

    assert verify_paired_benchmark_candidate(output, expected_sha256=digest) == payload


def test_integrity_check_recomputes_frozen_parity_instead_of_trusting_declared_errors() -> None:
    payload = _trusted_schema_three_payload()
    payload["frozen_parity"]["maximum_absolute_error"] = {
        metric: 0.0 for metric in FROZEN_ABSOLUTE_TOLERANCES
    }

    validate_paired_benchmark_payload(payload)

    with pytest.raises(EvaluationArtifactError, match="does not match verified cells"):
        assert_paired_benchmark_integrity(
            payload,
            frozen_benchmark_path=FROZEN_BENCHMARK,
        )


def test_canonical_round_trip_and_hash_verification(tmp_path: Path) -> None:
    payload = _valid_payload()
    output = tmp_path / "paired-benchmark.json"

    digest = write_paired_benchmark_candidate(payload, output)
    raw = output.read_bytes()

    assert raw == canonical_paired_json_bytes(payload)
    assert digest == hashlib.sha256(raw).hexdigest()
    assert verify_paired_benchmark_candidate(output, expected_sha256=digest) == payload


def test_verifier_rejects_duplicate_and_nonfinite_json(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    duplicate_hash = hashlib.sha256(duplicate.read_bytes()).hexdigest()

    with pytest.raises(EvaluationArtifactError, match="duplicate"):
        verify_paired_benchmark_candidate(duplicate, expected_sha256=duplicate_hash)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value":NaN}', encoding="utf-8")
    nonfinite_hash = hashlib.sha256(nonfinite.read_bytes()).hexdigest()

    with pytest.raises(EvaluationArtifactError, match="non-finite"):
        verify_paired_benchmark_candidate(nonfinite, expected_sha256=nonfinite_hash)


def test_validator_rejects_missing_scenario_and_identity_drift() -> None:
    missing = _valid_payload()
    missing["paired_rows"].pop()
    with pytest.raises(EvaluationArtifactError, match="288 paired rows"):
        validate_paired_benchmark_payload(missing)

    drifted = _valid_payload()
    drifted["paired_rows"][0]["scenario_id"] = "severe-50003-031"
    with pytest.raises(EvaluationArtifactError, match="scenario_id"):
        validate_paired_benchmark_payload(drifted)


def test_validator_rejects_missing_cell_and_count_drift() -> None:
    missing_cell = _valid_payload()
    missing_cell["cells"].pop()
    with pytest.raises(EvaluationArtifactError, match="exactly 9 cells"):
        validate_paired_benchmark_payload(missing_cell)

    wrong_count = _valid_payload()
    wrong_count["protocol"]["matched_scenario_count"] = 287
    with pytest.raises(EvaluationArtifactError, match="96-draw / 288-condition"):
        validate_paired_benchmark_payload(wrong_count)


def test_validator_rejects_nonfinite_metric_value() -> None:
    payload = _valid_payload()
    payload["paired_rows"][4]["Fixed Rotation"]["served_mwh"] = float("inf")

    with pytest.raises(EvaluationArtifactError, match="finite number"):
        validate_paired_benchmark_payload(payload)


def test_validator_rejects_metric_tampering() -> None:
    payload = _valid_payload()
    tampered = copy.deepcopy(payload)
    tampered["paired_rows"][7]["PPO"]["episode_reward"] += 0.25

    with pytest.raises(EvaluationArtifactError, match="paired_delta"):
        validate_paired_benchmark_payload(tampered)


@pytest.mark.parametrize(
    ("metric", "value", "message"),
    [
        ("served_mwh", 81.0, "available|served"),
        ("curtailed_mwh", 39.0, "energy identity"),
        ("oversubscription", 1.01, "between 0 and 1"),
        ("critical_outage_locality_hours", 168.1, "locality-hours"),
        ("critical_outage_locality_hours", 20.5, "whole locality-hours"),
    ],
)
def test_validator_rejects_physically_impossible_episode_metrics(
    metric: str,
    value: float,
    message: str,
) -> None:
    payload = _valid_payload()
    payload["paired_rows"][0]["Fixed Rotation"][metric] = value

    with pytest.raises(EvaluationArtifactError, match=message):
        validate_paired_benchmark_payload(payload)


def test_validator_allows_percentile_interval_not_containing_point_estimate() -> None:
    payload = _valid_payload()
    inference = payload["paired_inference"]["served_mwh"]
    inference["ci95_lower"] = inference["mean_delta"] + 1.0
    inference["ci95_upper"] = inference["mean_delta"] + 2.0

    validate_paired_benchmark_payload(payload)


def test_validator_rejects_available_energy_above_demand() -> None:
    payload = _valid_payload()
    payload["paired_rows"][0]["Fixed Rotation"]["available_mwh"] = 101.0

    with pytest.raises(EvaluationArtifactError, match="available_mwh must not exceed demand_mwh"):
        validate_paired_benchmark_payload(payload)


def test_neutral_validator_accepts_well_formed_untrusted_provenance_but_integrity_rejects() -> None:
    payload = _trusted_schema_three_payload()
    payload["provenance"]["ppo_actor"]["sha256"] = "f" * 64

    validate_paired_benchmark_payload(payload)

    with pytest.raises(EvaluationArtifactError, match="locked artifact"):
        assert_paired_benchmark_integrity(
            payload,
            frozen_benchmark_path=FROZEN_BENCHMARK,
        )


def test_neutral_validator_rejects_supply_draw_outside_locked_generator_range() -> None:
    with pytest.raises(EvaluationArtifactError, match="supply multiplier.*range"):
        _valid_payload(supply_multiplier=1.2)


@pytest.mark.gpu
def test_rehashed_candidate_with_endpoint_tamper_remains_untrusted(
    tmp_path: Path,
) -> None:
    payload = _trusted_schema_three_payload()
    payload["paired_inference"]["served_mwh"]["ci95_lower"] = -999.0
    candidate_output = tmp_path / "rehashed-candidate.json"
    digest = write_paired_benchmark_candidate(payload, candidate_output)

    assert (
        verify_paired_benchmark_candidate(
            candidate_output,
            expected_sha256=digest,
        )
        == payload
    )

    with pytest.raises(EvaluationArtifactError, match="project trust anchor"):
        write_paired_benchmark(
            payload,
            tmp_path / "trusted-output.json",
            frozen_benchmark_path=FROZEN_BENCHMARK,
        )
    with pytest.raises(EvaluationArtifactError, match="paired inference.*reconstruct"):
        verify_paired_inference_on_cuda(payload)


def test_trusted_integrity_translates_cuda_reconstruction_failure(monkeypatch) -> None:
    payload = _trusted_schema_three_payload()

    def fail_reconstruction(**_kwargs):
        raise PairedInferenceError("CUDA reconstruction failed: cuda:0 is unavailable")

    monkeypatch.setattr(
        evaluation_module,
        "verify_paired_inference_reconstruction",
        fail_reconstruction,
    )

    with pytest.raises(
        EvaluationArtifactError,
        match="paired inference reconstruction failed.*cuda:0 is unavailable",
    ):
        verify_paired_inference_on_cuda(payload)


def test_trusted_verifier_uses_project_anchor_instead_of_a_caller_digest(
    tmp_path: Path,
) -> None:
    payload = _trusted_schema_three_payload()
    payload["paired_inference"]["served_mwh"]["ci95_lower"] = -999.0
    output = tmp_path / "self-accepted.json"
    digest = write_paired_benchmark_candidate(payload, output)
    assert digest != PAIRED_BENCHMARK_SHA256

    with pytest.raises(EvaluationArtifactError, match="SHA-256 mismatch"):
        verify_paired_benchmark(
            output,
            frozen_benchmark_path=FROZEN_BENCHMARK,
        )


def test_trusted_verifier_is_host_only_when_cuda_is_hidden() -> None:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "-1"
    command = (
        "from pathlib import Path; "
        "from gridguard.evaluation import verify_paired_benchmark; "
        "root = Path.cwd(); "
        "payload = verify_paired_benchmark("
        "root / 'data/evidence/paired-benchmark.json', "
        "frozen_benchmark_path=root / 'data/evidence/frozen-benchmark.json'); "
        "print(payload['protocol']['matched_scenario_count'])"
    )

    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "288"


@pytest.mark.gpu
def test_explicit_trusted_cuda_reconstruction_verifies_committed_endpoints() -> None:
    payload = verify_paired_benchmark(
        PAIRED_BENCHMARK,
        frozen_benchmark_path=FROZEN_BENCHMARK,
    )

    assert verify_paired_inference_on_cuda(payload) <= 1e-10


def test_release_verifier_cross_checks_paired_cells_against_frozen_evidence(
    tmp_path: Path,
) -> None:
    output = tmp_path / "paired-benchmark-v3.json"
    payload = _trusted_schema_three_payload()
    digest = write_paired_benchmark(
        payload,
        output,
        frozen_benchmark_path=FROZEN_BENCHMARK,
    )
    assert digest == PAIRED_BENCHMARK_SHA256

    verified = verify_paired_benchmark(
        output,
        frozen_benchmark_path=FROZEN_BENCHMARK,
    )

    assert verified["frozen_parity"]["status"] == "passed"


def test_verifier_rejects_byte_tampering_before_semantic_use(tmp_path: Path) -> None:
    output = tmp_path / "paired-benchmark.json"
    digest = write_paired_benchmark_candidate(_valid_payload(), output)
    output.write_bytes(output.read_bytes().replace(b'"schema_version":3', b'"schema_version":4'))

    with pytest.raises(EvaluationArtifactError, match="SHA-256"):
        verify_paired_benchmark_candidate(output, expected_sha256=digest)


@pytest.mark.gpu
def test_full_cuda_protocol_matches_frozen_cells_and_is_self_verifying(tmp_path: Path) -> None:
    payload = run_full_paired_evaluation(
        ppo_manifest_path=PPO_MANIFEST,
        expected_ppo_artifact_sha256=BUNDLED_ARTIFACT_SHA256,
        frozen_benchmark_path=FROZEN_BENCHMARK,
        bootstrap_replicates=10_000,
        bootstrap_seed=20260815,
        device="cuda:0",
    )

    assert len(payload["paired_rows"]) == 288
    assert len(payload["cells"]) == 9
    assert payload["frozen_parity"]["status"] == "passed"
    assert all(
        payload["frozen_parity"]["maximum_absolute_error"][metric] <= tolerance
        for metric, tolerance in FROZEN_ABSOLUTE_TOLERANCES.items()
    )
    committed = json.loads(PAIRED_BENCHMARK.read_text(encoding="utf-8"))
    preserved_fields = (
        "scenario_id",
        "shortage_level",
        "seed",
        "batch_index",
        "Fixed Rotation",
        "PPO",
        "paired_delta",
    )
    assert all(
        all(generated[field] == accepted[field] for field in preserved_fields)
        for generated, accepted in zip(
            payload["paired_rows"],
            committed["paired_rows"],
            strict=True,
        )
    )
    output = tmp_path / "paired-benchmark.json"
    digest = write_paired_benchmark(
        payload,
        output,
        frozen_benchmark_path=FROZEN_BENCHMARK,
    )
    assert digest == PAIRED_BENCHMARK_SHA256
    assert (
        verify_paired_benchmark(
            output,
            frozen_benchmark_path=FROZEN_BENCHMARK,
        )
        == payload
    )
    assert verify_paired_inference_on_cuda(payload) <= 1e-10


def test_package_version_is_derived_from_distribution_metadata() -> None:
    assert gridguard.__version__ == version("gridguard-khartoum")

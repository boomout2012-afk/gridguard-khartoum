from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from gridguard.domain import Scenario, ShortageLevel
from tools.evaluate_additional_baselines import (
    OUTPUT,
    boundary_scenario,
    feasible_action_counts,
    fit_hourly_profile,
    predict_hourly_profile,
    validate_payload,
    verify,
)


def training_cube() -> torch.Tensor:
    hour = torch.arange(24, dtype=torch.float64)[None, None, :, None, None]
    offsets = torch.tensor([0.0, 2.0, 4.0, 6.0]).reshape(2, 2, 1, 1, 1)
    return (hour + offsets).expand(2, 2, 24, 7, 5).clone()


def test_hourly_profile_matches_known_training_average_and_ignores_test_values() -> None:
    training = training_cube()
    before = training.clone()
    model = fit_hourly_profile(training, training_seeds=(42, 43))
    expected = (torch.arange(24, dtype=torch.float64) + 3)[:, None, None].expand(24, 7, 5)
    torch.testing.assert_close(model.means_mw, expected)
    evaluation = torch.full((1, 3, 24, 7, 5), 10000.0, dtype=torch.float64)
    first = predict_hourly_profile(model, evaluation, evaluation_seeds=(91001,))
    evaluation.fill_(0)
    second = predict_hourly_profile(model, evaluation, evaluation_seeds=(91001,))
    assert first.shape == (1, 3, 21, 7, 5)
    assert torch.equal(first, second)
    assert torch.equal(training, before)
    torch.testing.assert_close(first[0, 0], expected[3:])


def test_hourly_profile_rejects_overlapping_seed_labels() -> None:
    model = fit_hourly_profile(training_cube(), training_seeds=(42, 43))
    with pytest.raises(ValueError, match="disjoint"):
        predict_hourly_profile(model, training_cube(), evaluation_seeds=(43, 44))


@pytest.mark.parametrize("mutation", ["shape", "negative", "nonfinite"])
def test_hourly_profile_rejects_invalid_training_data(mutation: str) -> None:
    cube = training_cube()
    if mutation == "shape":
        cube = cube[:, :, :-1]
    elif mutation == "negative":
        cube[0, 0, 0, 0, 0] = -1
    else:
        cube[0, 0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        fit_hourly_profile(cube, training_seeds=(42, 43))


@pytest.mark.parametrize("alpha, expected", [(0.99, 1), (1.01, 8), (2.0, 29)])
def test_boundary_brackets_known_discrete_action_counts(alpha: float, expected: int) -> None:
    demand = torch.ones((1, 24, 7, 5), dtype=torch.float32)
    original = Scenario(demand, torch.full((1, 24), 30.0), ShortageLevel.MODERATE, 9)
    stress = boundary_scenario(original, alpha)
    torch.testing.assert_close(stress.supply, torch.full((1, 24), 21 + alpha * 2))
    assert feasible_action_counts(stress).tolist() == [[expected] * 24]
    assert torch.equal(original.supply, torch.full((1, 24), 30.0))
    assert stress.demand is original.demand


@pytest.mark.parametrize("alpha", [0, -1, True, float("nan")])
def test_boundary_rejects_invalid_headroom(alpha) -> None:
    original = Scenario(torch.ones((1, 24, 7, 5)), torch.full((1, 24), 30.0), "moderate", 9)
    with pytest.raises((TypeError, ValueError)):
        boundary_scenario(original, alpha)


def test_published_additional_baselines_verify_without_execution() -> None:
    assert verify(OUTPUT)["passed"]


@pytest.mark.parametrize("target", ["forecast", "rule", "boundary", "source", "seed"])
def test_portable_validation_rejects_altered_scientific_fields(target: str) -> None:
    payload = copy.deepcopy(json.loads(OUTPUT.read_text(encoding="utf-8")))
    if target == "forecast":
        payload["forecast"]["methods"]["hour_of_day_mean"]["system_total"]["mae_mw"] += 1
    elif target == "rule":
        payload["rule_vs_optimizer"]["gates"]["utilization_noninferiority"] = False
    elif target == "boundary":
        payload["boundary"]["summaries"]["1.01"]["feasible_action_count"]["min"] = 1
    elif target == "source":
        payload["source_sha256"]["src/gridguard/scenarios.py"] = "0" * 64
    else:
        payload["recipe"]["forecast"]["training_seeds"][0] = 91001
    with pytest.raises(ValueError):
        validate_payload(payload)


def test_checksum_rejects_tampered_payload(tmp_path: Path) -> None:
    path = tmp_path / "additional-baselines.json"
    path.write_bytes(OUTPUT.read_bytes() + b" ")
    expected = OUTPUT.with_suffix(".sha256").read_text().split()[0]
    with pytest.raises(ValueError, match="checksum"):
        verify(path, expected_sha256=expected)

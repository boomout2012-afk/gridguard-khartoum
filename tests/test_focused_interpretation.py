"""Check scientific interpretation against small examples with known meanings."""

from __future__ import annotations

import pytest
import torch

from tools.interpret_focused_study import locality_summary_cuda


def test_locality_interpretation_requires_cuda() -> None:
    ratios = torch.ones((4, 1, 1, 2, 7), dtype=torch.float64)
    with pytest.raises(ValueError, match="cuda:0"):
        locality_summary_cuda(
            ratios,
            ("PPO seed 2", "Fixed Rotation", "PPO seed 1", "Constrained Optimization"),
            ("PPO seed 1", "PPO seed 2"),
        )


@pytest.mark.gpu
def test_endpoint_units_label_order_and_minimum_before_mean() -> None:
    # The first two localities exchange deprivation between the two conditions.
    # For seed 1, locality 0 could have demand 1 then 9 MWh, served 0.2 then 7.2:
    # its mean ratio is 50%, whereas pooling those energies would give 74%.
    # PPO's within-condition worst ratio averages to 30%; taking the minimum
    # after averaging locality outcomes would incorrectly report 50%.
    ratios = torch.ones((4, 1, 1, 2, 7), dtype=torch.float64, device="cuda:0")
    ratios[0, 0, 0, :, :2] = ratios.new_tensor([[0.4, 0.6], [0.6, 0.4]])
    ratios[1, 0, 0, :, :2] = ratios.new_tensor([[0.1, 0.9], [0.5, 0.9]])
    ratios[2, 0, 0, :, :2] = ratios.new_tensor([[0.2, 0.8], [0.8, 0.2]])
    ratios[3, 0, 0, :, :2] = ratios.new_tensor([[0.3, 0.7], [0.5, 0.7]])
    blocks = locality_summary_cuda(
        ratios,
        ("PPO seed 2", "Fixed Rotation", "PPO seed 1", "Constrained Optimization"),
        ("PPO seed 1", "PPO seed 2"),
    )
    expected = torch.tensor(
        [
            [30, 90, 100, 100, 100, 100, 100],
            [40, 70, 100, 100, 100, 100, 100],
            [50, 50, 100, 100, 100, 100, 100],
        ],
        dtype=torch.float64,
        device="cuda:0",
    )
    torch.testing.assert_close(
        torch.from_dlpack(blocks["means_percent"]), expected, rtol=0, atol=2e-6
    )
    torch.testing.assert_close(
        torch.from_dlpack(blocks["paired_shifts_pp"]),
        torch.tensor(
            [[20, -40, 0, 0, 0, 0, 0], [10, -20, 0, 0, 0, 0, 0]],
            dtype=torch.float64,
            device="cuda:0",
        ),
        rtol=0,
        atol=2e-6,
    )
    torch.testing.assert_close(
        torch.from_dlpack(blocks["mean_within_condition_minimum_percent"]),
        torch.tensor([30, 40, 30], dtype=torch.float64, device="cuda:0"),
        rtol=0,
        atol=2e-6,
    )
    torch.testing.assert_close(
        torch.from_dlpack(blocks["paired_minimum_shifts_pp"]),
        torch.tensor([0, -10], dtype=torch.float64, device="cuda:0"),
        rtol=0,
        atol=2e-6,
    )


@pytest.mark.gpu
def test_locality_interpretation_rejects_nonfinite_or_out_of_range_ratio() -> None:
    names = ("Fixed Rotation", "Constrained Optimization", "PPO seed 1")
    for invalid in (float("nan"), -0.1, 1.1):
        ratios = torch.ones((3, 1, 1, 1, 7), dtype=torch.float64, device="cuda:0")
        ratios[0, 0, 0, 0, 0] = invalid
        with pytest.raises(ValueError, match=r"finite.*\[0, 1\]"):
            locality_summary_cuda(ratios, names, ("PPO seed 1",))


@pytest.mark.gpu
def test_locality_interpretation_rejects_ambiguous_controller_identity() -> None:
    ratios = torch.ones((3, 1, 1, 1, 7), dtype=torch.float64, device="cuda:0")
    with pytest.raises(ValueError, match="unique"):
        locality_summary_cuda(
            ratios,
            ("Fixed Rotation", "Constrained Optimization", "Constrained Optimization"),
            ("PPO seed 1",),
        )

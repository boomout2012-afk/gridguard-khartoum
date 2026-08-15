from __future__ import annotations

import gc

import pytest
import torch

from gridguard.statistics import (
    BootstrapError,
    stratified_paired_bootstrap,
    torch_cuda_to_cupy,
)


def test_bootstrap_rejects_non_tensor_input() -> None:
    with pytest.raises(TypeError, match="torch.Tensor"):
        stratified_paired_bootstrap([[1.0]], metric_names=("reward",))  # type: ignore[arg-type]


def test_bootstrap_rejects_invalid_cube_before_cuda_work() -> None:
    malformed = torch.zeros((9, 32), dtype=torch.float32)

    with pytest.raises(BootstrapError, match=r"\[severity, seed, episode, metric\]"):
        stratified_paired_bootstrap(malformed, metric_names=("reward",))


def test_bootstrap_rejects_cpu_input() -> None:
    deltas = torch.zeros((3, 3, 32, 1), dtype=torch.float32)

    with pytest.raises(BootstrapError, match="CUDA"):
        stratified_paired_bootstrap(deltas, metric_names=("reward",))


@pytest.mark.gpu
def test_cupy_view_shares_cuda_storage_through_dlpack() -> None:
    source = torch.arange(6, device="cuda:0", dtype=torch.float32)

    view = torch_cuda_to_cupy(source)
    view += 10.0

    assert view.data.ptr == source.data_ptr()
    torch.testing.assert_close(
        source,
        torch.arange(10, 16, device="cuda:0", dtype=torch.float32),
    )


@pytest.mark.gpu
def test_stratified_bootstrap_is_deterministic_and_preserves_seed_strata() -> None:
    # Every episode within a severity/seed cell is constant. A correctly seed-stratified
    # cluster bootstrap therefore has a zero-width interval even though cell means differ.
    cells = torch.arange(9, device="cuda:0", dtype=torch.float32).reshape(3, 3)
    deltas = cells[:, :, None, None].expand(3, 3, 32, 2).clone()
    deltas[..., 1] *= -2.0

    first = stratified_paired_bootstrap(
        deltas,
        metric_names=("reward", "curtailed_mwh"),
        replicates=1_000,
        seed=20260815,
    )
    second = stratified_paired_bootstrap(
        deltas,
        metric_names=("reward", "curtailed_mwh"),
        replicates=1_000,
        seed=20260815,
    )

    assert first.metric_names == ("reward", "curtailed_mwh")
    assert first.replicates == 1_000
    assert first.seed == 20260815
    assert first.mean_delta.device == torch.device("cuda:0")
    torch.testing.assert_close(first.as_tensor(), second.as_tensor(), rtol=0.0, atol=0.0)
    expected = torch.tensor([4.0, -8.0], device="cuda:0", dtype=torch.float64)
    torch.testing.assert_close(first.mean_delta, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(first.ci95_lower, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(first.ci95_upper, expected, rtol=0.0, atol=0.0)


@pytest.mark.gpu
def test_bootstrap_rejects_nonfinite_cuda_deltas() -> None:
    deltas = torch.zeros((3, 3, 32, 1), device="cuda:0", dtype=torch.float32)
    deltas[2, 0, 7, 0] = torch.nan

    with pytest.raises(BootstrapError, match="finite"):
        stratified_paired_bootstrap(deltas, metric_names=("reward",))


@pytest.mark.gpu
def test_bootstrap_reuses_each_seed_sample_across_correlated_severities() -> None:
    episode_pattern = torch.linspace(-5.0, 7.0, 32, device="cuda:0", dtype=torch.float64)
    deltas = torch.zeros((3, 3, 32, 1), device="cuda:0", dtype=torch.float64)
    deltas[0, :, :, 0] = episode_pattern
    deltas[1, :, :, 0] = -episode_pattern

    result = stratified_paired_bootstrap(
        deltas,
        metric_names=("cancelled_severity_effect",),
        replicates=1_000,
        seed=20260815,
    )

    zero = torch.zeros(1, device="cuda:0", dtype=torch.float64)
    torch.testing.assert_close(result.mean_delta, zero, rtol=0.0, atol=1e-15)
    torch.testing.assert_close(result.ci95_lower, zero, rtol=0.0, atol=1e-15)
    torch.testing.assert_close(result.ci95_upper, zero, rtol=0.0, atol=1e-15)


@pytest.mark.gpu
def test_bootstrap_summary_dlpack_ownership_survives_prior_result_teardown() -> None:
    warmup = torch.randn((3, 3, 32, 2), device="cuda:0", dtype=torch.float64)
    first = stratified_paired_bootstrap(
        warmup,
        metric_names=("first", "second"),
        replicates=1_000,
        seed=17,
    )
    del first
    gc.collect()

    pattern = torch.linspace(-5.0, 7.0, 32, device="cuda:0", dtype=torch.float64)
    cancelled = torch.zeros((3, 3, 32, 1), device="cuda:0", dtype=torch.float64)
    cancelled[0, :, :, 0] = pattern
    cancelled[1, :, :, 0] = -pattern
    second = stratified_paired_bootstrap(
        cancelled,
        metric_names=("cancelled",),
        replicates=1_000,
        seed=20260815,
    )

    zero = torch.zeros(1, device="cuda:0", dtype=torch.float64)
    torch.testing.assert_close(second.as_tensor(), zero.expand(3, 1), rtol=0.0, atol=1e-15)

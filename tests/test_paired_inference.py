from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import gridguard.paired_inference as paired_inference_module
from gridguard.evaluation import BATCH_SIZE, INFERENCE_METRICS, SEEDS, SEVERITIES
from gridguard.paired_inference import (
    PairedInferenceError,
    verify_paired_inference_reconstruction,
)

PROJECT_ROOT = Path(__file__).parents[1]
PAIRED_BENCHMARK = PROJECT_ROOT / "data" / "evidence" / "paired-benchmark.json"


def _committed_payload() -> dict[str, object]:
    return json.loads(PAIRED_BENCHMARK.read_text(encoding="utf-8"))


def _verify(payload: dict[str, object]) -> float:
    return verify_paired_inference_reconstruction(
        paired_rows=payload["paired_rows"],
        paired_inference=payload["paired_inference"],
        protocol=payload["protocol"],
        severity_order=SEVERITIES,
        seed_order=SEEDS,
        episodes_per_cell=BATCH_SIZE,
        inference_metric_order=INFERENCE_METRICS,
    )


@pytest.mark.gpu
def test_committed_paired_inference_reconstructs_exactly_on_cuda_zero() -> None:
    assert _verify(_committed_payload()) == 0.0


def test_reconstruction_rejects_missing_bootstrap_protocol_before_cuda(monkeypatch) -> None:
    payload = _committed_payload()
    payload["protocol"] = copy.deepcopy(payload["protocol"])
    payload["protocol"].pop("bootstrap")
    monkeypatch.setattr(
        paired_inference_module,
        "_to_cuda_zero",
        lambda _values: pytest.fail("invalid protocol reached the CUDA boundary"),
    )

    with pytest.raises(PairedInferenceError, match="locked bootstrap protocol"):
        _verify(payload)


def test_reconstruction_rejects_wrong_bootstrap_protocol_before_cuda(monkeypatch) -> None:
    payload = _committed_payload()
    payload["protocol"] = copy.deepcopy(payload["protocol"])
    payload["protocol"]["bootstrap"]["percentile_method"] = "nearest"
    monkeypatch.setattr(
        paired_inference_module,
        "_to_cuda_zero",
        lambda _values: pytest.fail("invalid protocol reached the CUDA boundary"),
    )

    with pytest.raises(PairedInferenceError, match="locked bootstrap protocol"):
        _verify(payload)


def test_reconstruction_translates_cuda_device_failure(monkeypatch) -> None:
    payload = _committed_payload()

    def fail_cuda_transfer(_values):
        raise RuntimeError("cuda:0 is unavailable")

    monkeypatch.setattr(paired_inference_module, "_to_cuda_zero", fail_cuda_transfer)

    with pytest.raises(
        PairedInferenceError,
        match="CUDA reconstruction failed.*cuda:0 is unavailable",
    ):
        _verify(payload)

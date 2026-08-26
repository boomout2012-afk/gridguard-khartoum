from __future__ import annotations

import base64
import hashlib
import math
import struct

import numpy as np
import pytest
import torch

from gridguard.robustness_inference import (
    DeltaCubeContract,
    RobustnessInferenceError,
    build_delta_cube_payload,
    decode_delta_cube_payload,
    delta_cube_to_cuda,
    reconstruct_bootstrap_intervals,
)


def _contract() -> DeltaCubeContract:
    return DeltaCubeContract(
        parameterization_order=("base", "variant-a"),
        shortage_level_order=("mild", "moderate", "severe"),
        seed_order=(81001, 81002),
        episode_order=(0, 1),
        metric_order=("critical_outage_locality_hours", "fairness"),
    )


def test_delta_cube_payload_round_trips_strict_little_endian_bytes() -> None:
    contract = _contract()
    values = np.arange(math.prod(contract.shape), dtype=np.float64).reshape(contract.shape)
    values[0, 0, 0, 0, 0] = 0.5

    payload = build_delta_cube_payload(values, contract=contract)
    decoded = decode_delta_cube_payload(payload, contract=contract)

    assert payload["encoding"] == "base64"
    assert payload["dtype"] == "float64-le"
    assert payload["shape"] == list(contract.shape)
    assert payload["axis_order"] == list(contract.axis_order)
    assert decoded[:8] == struct.pack("<d", 0.5)
    assert hashlib.sha256(decoded).hexdigest() == payload["decoded_sha256"]
    assert base64.b64encode(decoded).decode("ascii") == payload["data_base64"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.__setitem__("unexpected", True), "fields"),
        (lambda payload: payload.__setitem__("encoding", "hex"), "encoding"),
        (lambda payload: payload["shape"].__setitem__(0, 3), "shape"),
        (
            lambda payload: payload["parameterization_order"].__setitem__(0, "not-base"),
            "parameterization_order",
        ),
        (lambda payload: payload.__setitem__("decoded_sha256", "0" * 64), "SHA-256"),
        (lambda payload: payload.__setitem__("data_base64", "not base64!"), "base64"),
    ],
)
def test_delta_cube_decoder_rejects_contract_or_byte_tampering(mutation, message: str) -> None:
    contract = _contract()
    payload = build_delta_cube_payload(np.zeros(contract.shape), contract=contract)
    mutation(payload)

    with pytest.raises(RobustnessInferenceError, match=message):
        decode_delta_cube_payload(payload, contract=contract)


def test_delta_cube_decoder_rejects_nonfinite_encoded_values() -> None:
    contract = _contract()
    payload = build_delta_cube_payload(np.zeros(contract.shape), contract=contract)
    decoded = bytearray(decode_delta_cube_payload(payload, contract=contract))
    decoded[:8] = struct.pack("<d", float("nan"))
    payload["data_base64"] = base64.b64encode(decoded).decode("ascii")
    payload["decoded_sha256"] = hashlib.sha256(decoded).hexdigest()

    with pytest.raises(RobustnessInferenceError, match="finite"):
        decode_delta_cube_payload(payload, contract=contract)


def test_delta_cube_contract_rejects_duplicate_or_misaligned_axes() -> None:
    with pytest.raises(RobustnessInferenceError, match="parameterization_order"):
        DeltaCubeContract(
            parameterization_order=("base", "base"),
            shortage_level_order=("mild", "moderate", "severe"),
            seed_order=(81001,),
            episode_order=(0,),
            metric_order=("fairness",),
        )
    with pytest.raises(RobustnessInferenceError, match="episode_order"):
        DeltaCubeContract(
            parameterization_order=("base", "variant-a"),
            shortage_level_order=("mild", "moderate", "severe"),
            seed_order=(81001,),
            episode_order=(1,),
            metric_order=("fairness",),
        )


@pytest.mark.gpu
def test_delta_cube_reconstructs_base_variant_and_difference_intervals_on_cuda() -> None:
    contract = _contract()
    values = np.zeros(contract.shape, dtype=np.float64)
    values[0, ..., 0] = 2.0
    values[0, ..., 1] = 0.1
    values[1, ..., 0] = 3.0
    values[1, ..., 1] = 0.4
    payload = build_delta_cube_payload(values, contract=contract)

    cube = delta_cube_to_cuda(payload, contract=contract, device="cuda:0")
    base, variants, differences = reconstruct_bootstrap_intervals(
        cube,
        metric_names=contract.metric_order,
        replicates=20,
        seed=20260826,
    )

    assert cube.device == torch.device("cuda:0")
    assert cube.dtype == torch.float64
    torch.testing.assert_close(
        base[0], torch.tensor([2.0, 0.1], dtype=torch.float64, device="cuda:0")
    )
    torch.testing.assert_close(
        variants[0, 0], torch.tensor([3.0, 0.4], dtype=torch.float64, device="cuda:0")
    )
    torch.testing.assert_close(
        differences[0, 0],
        torch.tensor([1.0, 0.3], dtype=torch.float64, device="cuda:0"),
    )
    assert base.shape == (3, 2)
    assert variants.shape == (1, 3, 2)
    assert differences.shape == (1, 3, 2)

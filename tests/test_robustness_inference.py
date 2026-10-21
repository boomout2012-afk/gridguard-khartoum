from __future__ import annotations

import base64
import hashlib
import math
import struct

import numpy as np
import pytest
import torch

import gridguard.robustness_inference as robustness_inference_module
from gridguard.robustness_inference import (
    BaseControllerCubeContract,
    DeltaCubeContract,
    ForecastDemandCubeContract,
    PowerBalanceCubeContract,
    RobustnessInferenceError,
    SensitivityControllerCubeContract,
    base_controller_cube_to_cuda,
    build_base_controller_cube_payload,
    build_delta_cube_payload,
    build_forecast_demand_cube_payload,
    build_power_balance_cube_payload,
    build_sensitivity_controller_cube_payload,
    decode_base_controller_cube_payload,
    decode_delta_cube_payload,
    decode_forecast_demand_cube_payload,
    decode_power_balance_cube_payload,
    decode_sensitivity_controller_cube_payload,
    delta_cube_to_cuda,
    forecast_demand_cube_to_cuda,
    power_balance_cube_to_cuda,
    reconstruct_bootstrap_intervals,
    reconstruct_controller_statistics,
    sensitivity_controller_cube_to_cuda,
)


def _contract() -> DeltaCubeContract:
    return DeltaCubeContract(
        parameterization_order=("base", "variant-a"),
        shortage_level_order=("mild", "moderate", "severe"),
        seed_order=(81001, 81002),
        episode_order=(0, 1),
        metric_order=("critical_outage_locality_hours", "fairness"),
    )


def _base_controller_contract() -> BaseControllerCubeContract:
    return BaseControllerCubeContract(
        controller_order=("Fixed Rotation", "PPO"),
        shortage_level_order=("mild", "moderate", "severe"),
        seed_order=(81001, 81002),
        episode_order=(0, 1),
        metric_order=(
            "critical_outage_locality_hours",
            "fairness",
            "utilization",
            "power_balance_violation_rate",
        ),
    )


def _semantic_delta_contract() -> DeltaCubeContract:
    return DeltaCubeContract(
        parameterization_order=("base", "variant-a"),
        shortage_level_order=("mild", "moderate", "severe"),
        seed_order=(81001, 81002),
        episode_order=(0, 1),
        metric_order=(
            "critical_outage_locality_hours",
            "fairness",
            "utilization",
        ),
    )


def _power_balance_contract() -> PowerBalanceCubeContract:
    return PowerBalanceCubeContract(
        parameterization_order=("base", "variant-a"),
        controller_order=("Fixed Rotation", "PPO"),
        shortage_level_order=("mild", "moderate", "severe"),
        seed_order=(81001, 81002),
        episode_order=(0, 1),
    )


def _sensitivity_controller_contract() -> SensitivityControllerCubeContract:
    return SensitivityControllerCubeContract(
        parameterization_order=("variant-a",),
        controller_order=("Fixed Rotation", "PPO"),
        shortage_level_order=("mild", "moderate", "severe"),
        seed_order=(81001, 81002),
        episode_order=(0, 1),
        metric_order=(
            "critical_outage_locality_hours",
            "fairness",
            "utilization",
            "power_balance_violation_rate",
        ),
    )


def _forecast_demand_contract() -> ForecastDemandCubeContract:
    return ForecastDemandCubeContract(
        seed_order=(81001, 81002),
        episode_order=(0, 1),
        hour_order=tuple(range(24)),
        locality_order=("Khartoum", "Omdurman"),
        sector_order=("Hospital", "Water"),
    )


def _consistent_cuda_reconstruction_cubes() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    delta_contract = _semantic_delta_contract()
    controller_contract = _base_controller_contract()
    power_contract = _power_balance_contract()
    controller = torch.zeros(
        controller_contract.shape,
        dtype=torch.float64,
        device="cuda:0",
    )
    controller[..., 0] = 10.0
    controller[..., 1] = 0.9
    controller[..., 2] = 0.8
    delta = torch.zeros(delta_contract.shape, dtype=torch.float64, device="cuda:0")
    sensitivity = torch.zeros(
        _sensitivity_controller_contract().shape,
        dtype=torch.float64,
        device="cuda:0",
    )
    sensitivity[..., 0] = 10.0
    sensitivity[..., 1] = 0.9
    sensitivity[..., 2] = 0.8
    power = torch.zeros(power_contract.shape, dtype=torch.float64, device="cuda:0")
    return delta, controller, sensitivity, power


def _validate_reconstruction_cubes(
    delta: torch.Tensor,
    controller: torch.Tensor,
    sensitivity: torch.Tensor,
    power: torch.Tensor,
) -> None:
    robustness_inference_module.validate_reconstruction_cube_semantics(
        delta,
        controller,
        sensitivity,
        power,
        delta_contract=_semantic_delta_contract(),
        base_controller_contract=_base_controller_contract(),
        sensitivity_controller_contract=_sensitivity_controller_contract(),
        power_balance_contract=_power_balance_contract(),
        maximum_outage_locality_hours=168.0,
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


def test_delta_cube_payload_rejects_complex_values_instead_of_truncating_them() -> None:
    contract = _contract()
    values = np.ones(contract.shape, dtype=np.complex128) * (1.0 + 2.0j)

    with pytest.raises(RobustnessInferenceError, match="real"):
        build_delta_cube_payload(values, contract=contract)


@pytest.mark.parametrize(
    ("contract_factory", "build_payload", "decode_payload"),
    (
        (_contract, build_delta_cube_payload, decode_delta_cube_payload),
        (
            _base_controller_contract,
            build_base_controller_cube_payload,
            decode_base_controller_cube_payload,
        ),
        (
            _power_balance_contract,
            build_power_balance_cube_payload,
            decode_power_balance_cube_payload,
        ),
        (
            _sensitivity_controller_contract,
            build_sensitivity_controller_cube_payload,
            decode_sensitivity_controller_cube_payload,
        ),
        (
            _forecast_demand_contract,
            build_forecast_demand_cube_payload,
            decode_forecast_demand_cube_payload,
        ),
    ),
)
@pytest.mark.parametrize(
    ("field", "replacement_type"),
    (
        ("shape", float),
        ("seed_order", float),
        ("episode_order", float),
        ("episode_order", bool),
    ),
)
def test_cube_metadata_rejects_equal_values_with_wrong_json_types(
    contract_factory,
    build_payload,
    decode_payload,
    field: str,
    replacement_type,
) -> None:
    contract = contract_factory()
    payload = build_payload(np.zeros(contract.shape, dtype=np.float64), contract=contract)
    payload[field] = [replacement_type(value) for value in payload[field]]

    with pytest.raises(RobustnessInferenceError, match=field):
        decode_payload(payload, contract=contract)


def test_controller_and_power_balance_cubes_round_trip_strict_contracts() -> None:
    controller_contract = _base_controller_contract()
    power_contract = _power_balance_contract()
    controller_values = np.arange(math.prod(controller_contract.shape), dtype=np.float64).reshape(
        controller_contract.shape
    )
    power_values = np.zeros(power_contract.shape, dtype=np.float64)

    controller_payload = build_base_controller_cube_payload(
        controller_values,
        contract=controller_contract,
    )
    power_payload = build_power_balance_cube_payload(power_values, contract=power_contract)

    assert (
        len(
            decode_base_controller_cube_payload(
                controller_payload,
                contract=controller_contract,
            )
        )
        == controller_values.nbytes
    )
    assert (
        len(decode_power_balance_cube_payload(power_payload, contract=power_contract))
        == power_values.nbytes
    )
    controller_payload["controller_order"][0] = "untrusted"
    with pytest.raises(RobustnessInferenceError, match="controller_order"):
        decode_base_controller_cube_payload(
            controller_payload,
            contract=controller_contract,
        )


def test_sensitivity_controller_and_forecast_demand_cubes_round_trip_strict_contracts() -> None:
    sensitivity_contract = _sensitivity_controller_contract()
    forecast_contract = _forecast_demand_contract()
    sensitivity_values = np.arange(math.prod(sensitivity_contract.shape), dtype=np.float64).reshape(
        sensitivity_contract.shape
    )
    forecast_values = np.arange(math.prod(forecast_contract.shape), dtype=np.float64).reshape(
        forecast_contract.shape
    )

    sensitivity_payload = build_sensitivity_controller_cube_payload(
        sensitivity_values,
        contract=sensitivity_contract,
    )
    forecast_payload = build_forecast_demand_cube_payload(
        forecast_values,
        contract=forecast_contract,
    )

    assert (
        len(
            decode_sensitivity_controller_cube_payload(
                sensitivity_payload,
                contract=sensitivity_contract,
            )
        )
        == sensitivity_values.nbytes
    )
    assert (
        len(
            decode_forecast_demand_cube_payload(
                forecast_payload,
                contract=forecast_contract,
            )
        )
        == forecast_values.size * 4
    )
    assert forecast_payload["dtype"] == "float32-le"
    sensitivity_payload["parameterization_order"][0] = "untrusted"
    with pytest.raises(RobustnessInferenceError, match="parameterization_order"):
        decode_sensitivity_controller_cube_payload(
            sensitivity_payload,
            contract=sensitivity_contract,
        )


@pytest.mark.gpu
def test_sensitivity_controller_and_forecast_demand_cubes_transfer_to_cuda_zero() -> None:
    sensitivity_contract = _sensitivity_controller_contract()
    forecast_contract = _forecast_demand_contract()
    sensitivity_values = np.zeros(sensitivity_contract.shape, dtype=np.float64)
    forecast_values = np.ones(forecast_contract.shape, dtype=np.float64)

    sensitivity = sensitivity_controller_cube_to_cuda(
        build_sensitivity_controller_cube_payload(
            sensitivity_values,
            contract=sensitivity_contract,
        ),
        contract=sensitivity_contract,
    )
    forecast = forecast_demand_cube_to_cuda(
        build_forecast_demand_cube_payload(forecast_values, contract=forecast_contract),
        contract=forecast_contract,
    )

    assert sensitivity.device == torch.device("cuda:0")
    assert sensitivity.shape == sensitivity_contract.shape
    assert forecast.device == torch.device("cuda:0")
    assert forecast.shape == forecast_contract.shape
    assert forecast.dtype == torch.float32


@pytest.mark.parametrize(
    ("builder", "contract"),
    [
        (build_base_controller_cube_payload, _base_controller_contract()),
        (build_sensitivity_controller_cube_payload, _sensitivity_controller_contract()),
        (build_power_balance_cube_payload, _power_balance_contract()),
        (build_forecast_demand_cube_payload, _forecast_demand_contract()),
    ],
)
def test_reconstruction_cube_builders_reject_complex_values(builder, contract) -> None:
    values = np.ones(contract.shape, dtype=np.complex128) * (1.0 + 2.0j)

    with pytest.raises(RobustnessInferenceError, match="real"):
        builder(values, contract=contract)


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
def test_cube_semantics_accepts_roundoff_within_named_strict_tolerance() -> None:
    tolerance = robustness_inference_module.RECONSTRUCTION_NUMERIC_TOLERANCE
    assert tolerance == 1e-12
    delta, controller, sensitivity, power = _consistent_cuda_reconstruction_cubes()
    controller[1, 0, 0, 0, 1] += tolerance / 2.0
    controller[:, 0, 0, 1, 0] = 168.0 + tolerance / 2.0
    controller[:, 0, 1, 0, 2] = 1.0 + tolerance / 2.0
    controller[:, 0, 1, 1, 3] = 1.0 + tolerance / 2.0
    power[0] = controller[..., 3]
    sensitivity[0, 0, 0, 1, 0, 3] = 1.0 + tolerance / 2.0
    power[1, 0, 0, 1, 0] = sensitivity[0, 0, 0, 1, 0, 3]
    delta[1, 0, 0, 0, 0] = 168.0 + tolerance / 2.0
    sensitivity[0, 0, 0, 0, 0, 0] = 0.0
    sensitivity[0, 1, 0, 0, 0, 0] = 168.0 + tolerance / 2.0
    delta[1, 0, 0, 0, 1] = -1.0 - tolerance / 2.0
    sensitivity[0, 0, 0, 0, 0, 1] = 1.0 + tolerance / 2.0
    sensitivity[0, 1, 0, 0, 0, 1] = 0.0

    _validate_reconstruction_cubes(delta, controller, sensitivity, power)


@pytest.mark.gpu
def test_cube_semantics_rejects_cross_cube_drift_beyond_named_tolerance() -> None:
    tolerance = robustness_inference_module.RECONSTRUCTION_NUMERIC_TOLERANCE
    delta, controller, sensitivity, power = _consistent_cuda_reconstruction_cubes()
    delta[1, 0, 0, 0, 0] = tolerance * 2.0

    with pytest.raises(RobustnessInferenceError, match="sensitivity paired deltas"):
        _validate_reconstruction_cubes(delta, controller, sensitivity, power)


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


@pytest.mark.gpu
def test_bootstrap_reconstruction_matches_nonconstant_axis_coded_literal_reference() -> None:
    contract = _contract()
    values = np.empty(contract.shape, dtype=np.float64)
    for parameterization in range(2):
        for severity in range(3):
            for seed_index in range(2):
                for episode in range(2):
                    base = 10 * severity + 3 * seed_index + 4 * episode
                    values[parameterization, severity, seed_index, episode, 0] = (
                        base if parameterization == 0 else 1.5 * base + 2 * episode + 5 * seed_index
                    )
                    values[parameterization, severity, seed_index, episode, 1] = (
                        (severity + 1) * (episode + 1) + 0.25 * seed_index
                        if parameterization == 0
                        else -0.5 * (severity + 1) * (episode + 1) + 2 * seed_index + episode
                    )
    cube = delta_cube_to_cuda(
        build_delta_cube_payload(values, contract=contract),
        contract=contract,
    )

    base, variants, differences = reconstruct_bootstrap_intervals(
        cube,
        metric_names=contract.metric_order,
        replicates=9,
        seed=17,
    )

    torch.testing.assert_close(
        base,
        torch.tensor(
            [[13.5, 3.125], [12.7, 2.725], [15.3, 4.025]],
            dtype=torch.float64,
            device="cuda:0",
        ),
        rtol=0.0,
        atol=1e-12,
    )
    torch.testing.assert_close(
        variants[0],
        torch.tensor(
            [[23.75, 0.0], [22.15, 0.0], [27.35, 0.0]],
            dtype=torch.float64,
            device="cuda:0",
        ),
        rtol=0.0,
        atol=1e-12,
    )
    torch.testing.assert_close(
        differences[0],
        torch.tensor(
            [[10.25, -3.125], [9.45, -4.025], [12.05, -2.725]],
            dtype=torch.float64,
            device="cuda:0",
        ),
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.gpu
def test_controller_statistics_reconstruct_means_and_global_power_balance_on_cuda() -> None:
    controller_contract = _base_controller_contract()
    power_contract = _power_balance_contract()
    controller_values = np.zeros(controller_contract.shape, dtype=np.float64)
    controller_values[0, ..., 0] = 10.0
    controller_values[0, ..., 1] = 0.8
    controller_values[0, ..., 2] = 0.7
    controller_values[1, ..., 0] = 4.0
    controller_values[1, ..., 1] = 0.9
    controller_values[1, ..., 2] = 0.95
    power_values = np.zeros(power_contract.shape, dtype=np.float64)
    power_values[1, 1, 2, 1, 1] = 0.125

    controller_cube = base_controller_cube_to_cuda(
        build_base_controller_cube_payload(controller_values, contract=controller_contract),
        contract=controller_contract,
    )
    power_cube = power_balance_cube_to_cuda(
        build_power_balance_cube_payload(power_values, contract=power_contract),
        contract=power_contract,
    )
    means, maximum = reconstruct_controller_statistics(controller_cube, power_cube)

    torch.testing.assert_close(
        means,
        torch.tensor(
            [[10.0, 0.8, 0.7, 0.0], [4.0, 0.9, 0.95, 0.0]],
            dtype=torch.float64,
            device="cuda:0",
        ),
    )
    torch.testing.assert_close(
        maximum,
        torch.tensor(0.125, dtype=torch.float64, device="cuda:0"),
    )

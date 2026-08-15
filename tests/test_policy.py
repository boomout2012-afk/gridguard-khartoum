from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from gridguard.policy import (
    ACTION_COUNT,
    BUNDLED_ARTIFACT_SHA256,
    EXPECTED_ACTOR_TENSORS,
    HIDDEN_SIZE,
    LOCALITY_COUNT,
    MODEL_SCHEMA,
    MODEL_SCHEMA_VERSION,
    OBSERVATION_DIM,
    PolicyArtifactError,
    PPOActor,
    load_ppo_policy,
    verify_ppo_artifact,
)
from tools.export_ppo_policy import (
    SOURCE_CHECKPOINT_SHA256,
    export_ppo_policy,
    load_trusted_actor_state,
    write_ppo_artifact,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BUNDLED_MANIFEST_PATH = REPOSITORY_ROOT / "data" / "model" / "manifest.json"
ARCHIVED_MODELS_PATH = "src/aipower_rl/models.py"
ARCHIVED_MODELS_SHA256 = "6fedcc55675a72aff048a3b1a9a98c86a87962427031c6a4450239fdccbbde03"
SOURCE_REVISION = "52f6dd82605c993c8094448caecec86faa289c0c"


@pytest.fixture(scope="module")
def source_checkpoint() -> Path:
    configured_path = os.environ.get("GRIDGUARD_SOURCE_CHECKPOINT")
    if configured_path:
        checkpoint = Path(configured_path).expanduser().resolve()
        if not checkpoint.is_file():
            pytest.fail(f"GRIDGUARD_SOURCE_CHECKPOINT is not a file: {checkpoint}")
        return checkpoint

    sibling_checkpoint = (
        REPOSITORY_ROOT.parent / "aipower_rl" / "artifacts" / "checkpoints" / "best.pt"
    )
    if sibling_checkpoint.is_file():
        return sibling_checkpoint
    pytest.skip(
        "trusted source archive unavailable; set GRIDGUARD_SOURCE_CHECKPOINT "
        "to run source_archive tests"
    )


def _synthetic_actor_state(*, selected_index: int = 1598) -> dict[str, torch.Tensor]:
    state = {
        "actor_encoder.0.weight": torch.zeros(HIDDEN_SIZE, OBSERVATION_DIM),
        "actor_encoder.0.bias": torch.zeros(HIDDEN_SIZE),
        "actor_encoder.2.weight": torch.zeros(HIDDEN_SIZE, HIDDEN_SIZE),
        "actor_encoder.2.bias": torch.zeros(HIDDEN_SIZE),
        "actor.weight": torch.zeros(ACTION_COUNT, HIDDEN_SIZE),
        "actor.bias": torch.zeros(ACTION_COUNT),
    }
    state["actor.bias"][selected_index] = 1.0
    return state


def _write_synthetic_artifact(tmp_path: Path) -> tuple[Path, str]:
    exported = write_ppo_artifact(
        _synthetic_actor_state(),
        output_dir=tmp_path,
        source_checkpoint_sha256=SOURCE_CHECKPOINT_SHA256,
        source_checkpoint_path="artifacts/checkpoints/best.pt",
        source_revision="52f6dd82605c993c8094448caecec86faa289c0c",
    )
    return exported.manifest_path, exported.artifact_sha256


def test_actor_decodes_little_endian_base_three() -> None:
    actor = PPOActor()

    decoded = actor.decode_action_indices(torch.tensor([1598]))

    assert decoded.tolist() == [[2, 1, 0, 2, 1, 0, 2]]


@pytest.mark.parametrize(
    "encoded_action",
    [
        torch.tensor([1.0]),
        torch.tensor([True]),
        torch.tensor([1.0 + 0.0j]),
    ],
)
def test_actor_rejects_noninteger_encoded_actions(encoded_action: torch.Tensor) -> None:
    actor = PPOActor()

    with pytest.raises(TypeError, match="integer"):
        actor.decode_action_indices(encoded_action)


@pytest.mark.parametrize("encoded_action", [torch.tensor([-1]), torch.tensor([ACTION_COUNT])])
def test_actor_rejects_out_of_range_encoded_actions(encoded_action: torch.Tensor) -> None:
    actor = PPOActor()

    with pytest.raises(ValueError, match="range"):
        actor.decode_action_indices(encoded_action)


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), -float("inf")])
def test_actor_rejects_nonfinite_observations(invalid_value: float) -> None:
    actor = PPOActor()
    observation = torch.zeros(OBSERVATION_DIM)
    observation[0] = invalid_value

    with pytest.raises(ValueError, match="finite"):
        actor(observation)


def test_actor_rejects_nonfloating_or_mismatched_observation_dtype() -> None:
    actor = PPOActor()

    with pytest.raises(TypeError, match="floating-point"):
        actor(torch.zeros(OBSERVATION_DIM, dtype=torch.long))
    with pytest.raises(ValueError, match="dtype"):
        actor(torch.zeros(OBSERVATION_DIM, dtype=torch.float64))


@pytest.mark.gpu
def test_synthetic_bias_golden_loads_frozen_actor_on_cuda(tmp_path: Path) -> None:
    manifest_path, digest = _write_synthetic_artifact(tmp_path)

    policy = load_ppo_policy(manifest_path, expected_artifact_sha256=digest)
    action = policy.act(torch.zeros(OBSERVATION_DIM, device="cuda:0"))

    assert action.tolist() == [2, 1, 0, 2, 1, 0, 2]
    assert action.device == torch.device("cuda:0")
    assert next(policy.parameters()).device == torch.device("cuda:0")
    assert policy.training is False
    assert all(not parameter.requires_grad for parameter in policy.parameters())


def test_loader_rejects_cpu_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(PolicyArtifactError, match="CUDA"):
        load_ppo_policy(
            tmp_path / "missing-manifest.json",
            expected_artifact_sha256="0" * 64,
        )


@pytest.mark.parametrize("trust_anchor", [None, True, 123, b"0" * 64, "short"])
def test_host_verifier_rejects_invalid_trust_anchor_type_or_value(
    tmp_path: Path,
    trust_anchor: object,
) -> None:
    manifest_path, _digest = _write_synthetic_artifact(tmp_path)

    with pytest.raises(PolicyArtifactError, match="trust anchor"):
        verify_ppo_artifact(
            manifest_path,
            expected_artifact_sha256=trust_anchor,  # type: ignore[arg-type]
        )


def test_loader_rejects_tampered_actor_bytes(tmp_path: Path) -> None:
    manifest_path, digest = _write_synthetic_artifact(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_path = manifest_path.parent / manifest["artifact"]["filename"]
    payload = bytearray(artifact_path.read_bytes())
    payload[len(payload) // 2] ^= 0x01
    artifact_path.write_bytes(payload)

    with pytest.raises(PolicyArtifactError, match="SHA-256"):
        verify_ppo_artifact(manifest_path, expected_artifact_sha256=digest)


def test_loader_rejects_unknown_schema(tmp_path: Path) -> None:
    manifest_path, digest = _write_synthetic_artifact(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = MODEL_SCHEMA_VERSION + 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(PolicyArtifactError, match="schema"):
        verify_ppo_artifact(manifest_path, expected_artifact_sha256=digest)


def test_loader_rejects_boolean_schema_version(tmp_path: Path) -> None:
    manifest_path, digest = _write_synthetic_artifact(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(PolicyArtifactError, match="schema"):
        verify_ppo_artifact(manifest_path, expected_artifact_sha256=digest)


def test_loader_rejects_wrong_tensor_shape(tmp_path: Path) -> None:
    state = _synthetic_actor_state()
    state["actor.bias"] = state["actor.bias"][:-1]
    exported = write_ppo_artifact(
        state,
        output_dir=tmp_path,
        source_checkpoint_sha256=SOURCE_CHECKPOINT_SHA256,
        source_checkpoint_path="artifacts/checkpoints/best.pt",
        source_revision="52f6dd82605c993c8094448caecec86faa289c0c",
        validate_shapes=False,
    )

    with pytest.raises(PolicyArtifactError, match="shape"):
        verify_ppo_artifact(
            exported.manifest_path,
            expected_artifact_sha256=exported.artifact_sha256,
        )


def test_host_verification_does_not_probe_cuda(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, digest = _write_synthetic_artifact(tmp_path)

    def fail_cuda_probe() -> bool:
        raise AssertionError("host artifact verification must not probe CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", fail_cuda_probe)

    verified = verify_ppo_artifact(manifest_path, expected_artifact_sha256=digest)

    assert verified.artifact_sha256 == digest
    assert verified.artifact_path.name == "ppo-policy.npz"


def test_exported_artifact_contains_only_actor_state_and_provenance(tmp_path: Path) -> None:
    exported = write_ppo_artifact(
        _synthetic_actor_state(),
        output_dir=tmp_path,
        source_checkpoint_sha256=SOURCE_CHECKPOINT_SHA256,
        source_checkpoint_path="artifacts/checkpoints/best.pt",
        source_revision="52f6dd82605c993c8094448caecec86faa289c0c",
    )
    manifest = json.loads(exported.manifest_path.read_text(encoding="utf-8"))

    with np.load(exported.artifact_path, allow_pickle=False) as archive:
        assert set(archive.files) == set(EXPECTED_ACTOR_TENSORS)

    assert manifest["schema"] == MODEL_SCHEMA
    assert manifest["schema_version"] == MODEL_SCHEMA_VERSION
    assert manifest["architecture"] == {
        "action_count": ACTION_COUNT,
        "action_encoding": "little-endian-base-3",
        "activation": "tanh",
        "hidden_size": HIDDEN_SIZE,
        "locality_count": LOCALITY_COUNT,
        "observation_dim": OBSERVATION_DIM,
    }
    assert manifest["source"]["checkpoint_sha256"] == SOURCE_CHECKPOINT_SHA256
    serialized = json.dumps(manifest, sort_keys=True)
    for forbidden in ("critic", "optimizer", "training_state", '"step"'):
        assert forbidden not in serialized


def test_artifact_writer_is_byte_reproducible(tmp_path: Path) -> None:
    first = write_ppo_artifact(
        _synthetic_actor_state(),
        output_dir=tmp_path / "first",
        source_checkpoint_sha256=SOURCE_CHECKPOINT_SHA256,
        source_checkpoint_path="artifacts/checkpoints/best.pt",
        source_revision="52f6dd82605c993c8094448caecec86faa289c0c",
    )
    second = write_ppo_artifact(
        _synthetic_actor_state(),
        output_dir=tmp_path / "second",
        source_checkpoint_sha256=SOURCE_CHECKPOINT_SHA256,
        source_checkpoint_path="artifacts/checkpoints/best.pt",
        source_revision="52f6dd82605c993c8094448caecec86faa289c0c",
    )

    assert first.artifact_path.read_bytes() == second.artifact_path.read_bytes()
    assert first.manifest_path.read_bytes() == second.manifest_path.read_bytes()


def test_bundled_manifest_hash_matches_committed_actor() -> None:
    manifest = json.loads(BUNDLED_MANIFEST_PATH.read_text(encoding="utf-8"))
    artifact = BUNDLED_MANIFEST_PATH.parent / manifest["artifact"]["filename"]

    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == manifest["artifact"]["sha256"]
    assert artifact.stat().st_size == manifest["artifact"]["size_bytes"]


@pytest.mark.source_archive
def test_export_is_identical_to_bundled_artifact(
    tmp_path: Path,
    source_checkpoint: Path,
) -> None:
    exported = export_ppo_policy(source_checkpoint, output_dir=tmp_path)
    bundled_manifest = json.loads(BUNDLED_MANIFEST_PATH.read_text(encoding="utf-8"))
    bundled_artifact = BUNDLED_MANIFEST_PATH.parent / bundled_manifest["artifact"]["filename"]

    assert exported.artifact_path.read_bytes() == bundled_artifact.read_bytes()
    assert exported.manifest_path.read_bytes() == BUNDLED_MANIFEST_PATH.read_bytes()


@pytest.mark.gpu
@pytest.mark.source_archive
def test_bundled_cuda_actor_matches_trusted_source_checkpoint(
    source_checkpoint: Path,
) -> None:
    source_repository = source_checkpoint.parents[2]
    source_actor_class = _load_hash_verified_archived_actor(source_repository)
    source_state = load_trusted_actor_state(source_checkpoint)
    source_actor = source_actor_class(OBSERVATION_DIM, HIDDEN_SIZE).to("cuda:0")
    incompatible = source_actor.load_state_dict(source_state, strict=False)
    assert incompatible.unexpected_keys == []
    assert set(incompatible.missing_keys) == {
        "critic_encoder.0.weight",
        "critic_encoder.0.bias",
        "critic_encoder.2.weight",
        "critic_encoder.2.bias",
        "critic.weight",
        "critic.bias",
    }
    source_actor.requires_grad_(False).eval()
    exported_actor = load_ppo_policy(
        BUNDLED_MANIFEST_PATH,
        expected_artifact_sha256=BUNDLED_ARTIFACT_SHA256,
    )
    observations = torch.stack(
        (
            torch.zeros(OBSERVATION_DIM),
            torch.linspace(-1.0, 1.0, OBSERVATION_DIM),
            torch.sin(torch.arange(OBSERVATION_DIM) * 0.37),
            torch.cos(torch.arange(OBSERVATION_DIM) * 0.19) * 0.5,
        )
    ).to("cuda:0")

    with torch.inference_mode():
        source_logits, _source_values = source_actor(observations)
        exported_logits = exported_actor(observations)
        exported_actions = exported_actor.act(observations)
        source_actions = source_actor.act(observations, deterministic=True).action

    torch.testing.assert_close(exported_logits, source_logits, rtol=0.0, atol=0.0)
    expected_actions = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 1, 2, 2],
            [2, 2, 2, 2, 1, 2, 2],
        ],
        device="cuda:0",
    )
    assert torch.equal(exported_actions, expected_actions)
    assert torch.equal(source_actions, expected_actions)


def _load_hash_verified_archived_actor(source_repository: Path) -> type[Any]:
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(source_repository),
            "show",
            f"{SOURCE_REVISION}:{ARCHIVED_MODELS_PATH}",
        ],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        pytest.fail(
            "trusted source repository cannot provide the accepted actor implementation: "
            + completed.stderr.decode("utf-8", errors="replace")
        )
    source_bytes = completed.stdout
    assert hashlib.sha256(source_bytes).hexdigest() == ARCHIVED_MODELS_SHA256

    package_name = "aipower_rl"
    config_name = "aipower_rl.config"
    module_name = "_gridguard_hash_verified_archived_models"
    package = types.ModuleType(package_name)
    package.__path__ = []  # type: ignore[attr-defined]
    config = types.ModuleType(config_name)
    config.LOCALITIES = tuple(range(LOCALITY_COUNT))  # type: ignore[attr-defined]
    module = types.ModuleType(module_name)
    module.__file__ = f"{SOURCE_REVISION}:{ARCHIVED_MODELS_PATH}"

    previous = {name: sys.modules.get(name) for name in (package_name, config_name, module_name)}
    try:
        sys.modules[package_name] = package
        sys.modules[config_name] = config
        sys.modules[module_name] = module
        exec(compile(source_bytes, module.__file__, "exec"), module.__dict__)
    finally:
        for name, prior in previous.items():
            if prior is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior

    actor_class = module.__dict__.get("ActorCriticPolicy")
    if not isinstance(actor_class, type):
        pytest.fail("accepted source actor class was not defined by the verified source blob")
    return actor_class


@pytest.mark.gpu
def test_runtime_loader_does_not_use_torch_pickle_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, digest = _write_synthetic_artifact(tmp_path)

    def fail_torch_load(*args: object, **kwargs: object) -> None:
        raise AssertionError("runtime must not call torch.load")

    monkeypatch.setattr(torch, "load", fail_torch_load)

    policy = load_ppo_policy(manifest_path, expected_artifact_sha256=digest)

    assert policy.act(torch.zeros(OBSERVATION_DIM, device="cuda:0")).shape == (LOCALITY_COUNT,)

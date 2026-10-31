"""Strict inference-only PPO actor loading and deterministic CUDA inference."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from torch import nn

from gridguard.provenance import sha256_file

OBSERVATION_DIM: Final = 32
HIDDEN_SIZE: Final = 512
LOCALITY_COUNT: Final = 7
ACTION_LEVELS: Final = 3
ACTION_COUNT: Final = ACTION_LEVELS**LOCALITY_COUNT

MODEL_SCHEMA: Final = "gridguard.ppo_actor"
MODEL_SCHEMA_VERSION: Final = 2
MODEL_ARTIFACT_FILENAME: Final = "ppo-policy.npz"
MODEL_MANIFEST_FILENAME: Final = "manifest.json"
TRAINING_PROVENANCE_FILENAME: Final = "training-provenance.yaml"
TRAINING_PROVENANCE_FORMAT: Final = "yaml"
TRAINING_PROVENANCE_SCHEMA: Final = "gridguard.ppo_training_provenance"
TRAINING_PROVENANCE_SCHEMA_VERSION: Final = 3
SOURCE_CHECKPOINT_SHA256: Final = "c6e5712f966b46ba18f4c69fb457b9e69f78887c0c5c0279e5ce034f40f9c665"
SOURCE_CHECKPOINT_PATH: Final = "artifacts/checkpoints/best.pt"
SOURCE_REVISION: Final = "52f6dd82605c993c8094448caecec86faa289c0c"

EXPECTED_ACTOR_TENSORS: Final[dict[str, tuple[int, ...]]] = {
    "actor_encoder.0.weight": (HIDDEN_SIZE, OBSERVATION_DIM),
    "actor_encoder.0.bias": (HIDDEN_SIZE,),
    "actor_encoder.2.weight": (HIDDEN_SIZE, HIDDEN_SIZE),
    "actor_encoder.2.bias": (HIDDEN_SIZE,),
    "actor.weight": (ACTION_COUNT, HIDDEN_SIZE),
    "actor.bias": (ACTION_COUNT,),
}

BUNDLED_ARTIFACT_SHA256: Final = "3e4a7ce44267dd1cf3bbde7e77abafb8f31fdea77ece63acda7be17fa3198ad5"

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "artifact",
        "architecture",
        "source",
        "tensors",
        "training_provenance",
    }
)
_ARTIFACT_FIELDS = frozenset({"filename", "format", "sha256", "size_bytes"})
_TRAINING_PROVENANCE_FIELDS = frozenset(
    {"filename", "format", "schema", "schema_version", "sha256", "size_bytes"}
)
_ARCHITECTURE_FIELDS = frozenset(
    {
        "observation_dim",
        "hidden_size",
        "action_count",
        "locality_count",
        "activation",
        "action_encoding",
    }
)
_SOURCE_FIELDS = frozenset({"checkpoint_path", "checkpoint_sha256", "source_revision"})
_TENSOR_FIELDS = frozenset({"dtype", "shape"})


class PolicyArtifactError(RuntimeError):
    """Raised when a PPO actor artifact cannot be trusted or loaded on CUDA."""


@dataclass(frozen=True, slots=True)
class VerifiedPPOArtifact:
    """Host-verified PPO actor bytes ready for an explicit CUDA load."""

    manifest_path: Path
    artifact_path: Path
    artifact_sha256: str
    size_bytes: int
    _actor_state: tuple[tuple[str, np.ndarray[Any, np.dtype[np.float32]]], ...]


class PPOActor(nn.Module):
    """The accepted PPO actor architecture, without critic or training state."""

    def __init__(self) -> None:
        super().__init__()
        self.actor_encoder = nn.Sequential(
            nn.Linear(OBSERVATION_DIM, HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.Tanh(),
        )
        self.actor = nn.Linear(HIDDEN_SIZE, ACTION_COUNT)
        divisors = ACTION_LEVELS ** torch.arange(LOCALITY_COUNT, dtype=torch.long)
        self.register_buffer("action_divisors", divisors, persistent=False)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Return joint-action logits for one observation or a batch."""

        self._validate_observation_metadata(observation)
        if not bool(torch.isfinite(observation).all().item()):
            raise ValueError("observation must contain only finite values")
        return self._forward_layers(observation)

    def _validate_observation_metadata(self, observation: torch.Tensor) -> None:
        if observation.ndim == 0 or observation.shape[-1] != OBSERVATION_DIM:
            raise ValueError(
                f"expected observation width {OBSERVATION_DIM}, received {tuple(observation.shape)}"
            )
        if not observation.is_floating_point():
            raise TypeError("observation must be a floating-point tensor")
        parameter = next(self.parameters())
        if observation.dtype != parameter.dtype:
            raise ValueError(
                f"observation dtype must be {parameter.dtype}; received {observation.dtype}"
            )

    def _forward_layers(self, observation: torch.Tensor) -> torch.Tensor:
        return self.actor(self.actor_encoder(observation))

    def _decode_action_indices_unchecked(self, encoded_action: torch.Tensor) -> torch.Tensor:
        encoded = encoded_action.to(dtype=torch.long)
        return (encoded[..., None] // self.action_divisors) % ACTION_LEVELS

    def decode_action_indices(self, encoded_action: torch.Tensor) -> torch.Tensor:
        """Decode indices using the source policy's little-endian base-three order."""

        if (
            encoded_action.dtype == torch.bool
            or encoded_action.is_floating_point()
            or encoded_action.is_complex()
        ):
            raise TypeError("encoded action indices must use an integer tensor dtype")
        if encoded_action.device != self.action_divisors.device:
            raise ValueError(
                "encoded action indices must be on the actor device "
                f"{self.action_divisors.device}; received {encoded_action.device}"
            )
        if not bool(((encoded_action >= 0) & (encoded_action < ACTION_COUNT)).all().item()):
            raise ValueError(f"encoded action indices must be in range [0, {ACTION_COUNT})")
        return self._decode_action_indices_unchecked(encoded_action)

    @torch.inference_mode()
    def act(self, observation: torch.Tensor) -> torch.Tensor:
        """Return deterministic argmax action tiers on the actor's CUDA device."""

        parameter = next(self.parameters())
        if parameter.device.type != "cuda" or parameter.device.index != 0:
            raise PolicyArtifactError("PPO actor is not loaded on required device cuda:0")
        if observation.device != parameter.device:
            raise ValueError(
                f"observation must be on {parameter.device}; received {observation.device}"
            )
        return self._decode_action_indices_unchecked(self(observation).argmax(dim=-1))

    @torch.inference_mode()
    def act_prevalidated(self, observation: torch.Tensor) -> torch.Tensor:
        """Infer after simulation-boundary validation without host scalar synchronization."""

        parameter = next(self.parameters())
        if parameter.device.type != "cuda" or parameter.device.index != 0:
            raise PolicyArtifactError("PPO actor is not loaded on required device cuda:0")
        if observation.device != parameter.device:
            raise ValueError(
                f"observation must be on {parameter.device}; received {observation.device}"
            )
        self._validate_observation_metadata(observation)
        encoded = self._forward_layers(observation).argmax(dim=-1)
        return self._decode_action_indices_unchecked(encoded)


def load_ppo_policy(
    manifest_path: str | Path,
    *,
    expected_artifact_sha256: str,
) -> PPOActor:
    """Require CUDA, verify an injected artifact, freeze it, and set eval mode."""

    _require_cuda_zero()
    verified = verify_ppo_artifact(
        manifest_path,
        expected_artifact_sha256=expected_artifact_sha256,
    )

    try:
        device = torch.device("cuda:0")
        actor = PPOActor().to(device)
        actor.load_state_dict(
            {name: torch.from_numpy(array).to(device) for name, array in verified._actor_state},
            strict=True,
        )
        actor.requires_grad_(False)
        actor.eval()
    except (RuntimeError, TypeError, ValueError) as error:
        raise PolicyArtifactError(f"failed to load PPO actor on cuda:0: {error}") from error

    if any(parameter.device != device for parameter in actor.parameters()):
        raise PolicyArtifactError("PPO actor parameters did not load entirely on cuda:0")
    return actor


def verify_ppo_artifact(
    manifest_path: str | Path,
    *,
    expected_artifact_sha256: str,
) -> VerifiedPPOArtifact:
    """Verify schema, provenance, bytes, and tensors without probing CUDA."""

    if not isinstance(expected_artifact_sha256, str) or not _SHA256_PATTERN.fullmatch(
        expected_artifact_sha256
    ):
        raise PolicyArtifactError("a valid expected actor SHA-256 trust anchor is required")

    resolved_manifest_path = Path(manifest_path)
    manifest = _read_and_validate_manifest(resolved_manifest_path)
    artifact_metadata = _require_mapping(manifest["artifact"], field="artifact")
    training_metadata = _require_mapping(
        manifest["training_provenance"],
        field="training_provenance",
    )
    training_path = resolved_manifest_path.parent / training_metadata["filename"]
    try:
        training_size_bytes = training_path.stat().st_size
    except OSError as error:
        raise PolicyArtifactError(f"PPO training provenance is unavailable: {error}") from error
    if training_size_bytes != training_metadata["size_bytes"]:
        raise PolicyArtifactError(
            "training provenance size mismatch: "
            f"expected {training_metadata['size_bytes']}, found {training_size_bytes}"
        )
    try:
        training_hash = sha256_file(training_path)
    except OSError as error:
        raise PolicyArtifactError(f"PPO training provenance cannot be hashed: {error}") from error
    if training_hash != training_metadata["sha256"]:
        raise PolicyArtifactError(
            "training provenance SHA-256 mismatch: "
            f"expected {training_metadata['sha256']}, found {training_hash}"
        )

    declared_hash = artifact_metadata["sha256"]
    if declared_hash != expected_artifact_sha256:
        raise PolicyArtifactError("actor SHA-256 trust anchor does not match the artifact manifest")

    artifact_path = resolved_manifest_path.parent / artifact_metadata["filename"]
    try:
        size_bytes = artifact_path.stat().st_size
    except OSError as error:
        raise PolicyArtifactError(f"PPO actor artifact is unavailable: {error}") from error
    if size_bytes != artifact_metadata["size_bytes"]:
        raise PolicyArtifactError(
            f"actor size mismatch: expected {artifact_metadata['size_bytes']}, found {size_bytes}"
        )
    try:
        actual_hash = sha256_file(artifact_path)
    except OSError as error:
        raise PolicyArtifactError(f"PPO actor artifact cannot be hashed: {error}") from error
    if actual_hash != declared_hash:
        raise PolicyArtifactError(
            f"actor SHA-256 mismatch: expected {declared_hash}, found {actual_hash}"
        )

    state = _read_actor_arrays(artifact_path)
    return VerifiedPPOArtifact(
        manifest_path=resolved_manifest_path,
        artifact_path=artifact_path,
        artifact_sha256=actual_hash,
        size_bytes=size_bytes,
        _actor_state=tuple(state.items()),
    )


def _require_cuda_zero() -> None:
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise PolicyArtifactError("required PyTorch CUDA device cuda:0 is unavailable")
    except PolicyArtifactError:
        raise
    except Exception as error:
        raise PolicyArtifactError(
            f"required CUDA device cuda:0 cannot be inspected: {error}"
        ) from error


def _read_and_validate_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_fields,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PolicyArtifactError(f"PPO actor manifest cannot be read: {error}") from error
    manifest = _require_mapping(payload, field="manifest")
    _require_exact_fields(manifest, expected=_TOP_LEVEL_FIELDS, field="manifest")

    if (
        type(manifest["schema"]) is not str
        or manifest["schema"] != MODEL_SCHEMA
        or type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != MODEL_SCHEMA_VERSION
    ):
        raise PolicyArtifactError("unsupported PPO actor manifest schema")

    artifact = _require_mapping(manifest["artifact"], field="artifact")
    _require_exact_fields(artifact, expected=_ARTIFACT_FIELDS, field="artifact")
    if artifact["filename"] != MODEL_ARTIFACT_FILENAME or artifact["format"] != "numpy.npz":
        raise PolicyArtifactError("unsupported PPO actor artifact format")
    if not isinstance(artifact["sha256"], str) or not _SHA256_PATTERN.fullmatch(artifact["sha256"]):
        raise PolicyArtifactError("artifact.sha256 must be a lowercase SHA-256 digest")
    if type(artifact["size_bytes"]) is not int or artifact["size_bytes"] <= 0:
        raise PolicyArtifactError("artifact.size_bytes must be a positive integer")

    training_provenance = _require_mapping(
        manifest["training_provenance"],
        field="training_provenance",
    )
    _require_exact_fields(
        training_provenance,
        expected=_TRAINING_PROVENANCE_FIELDS,
        field="training_provenance",
    )
    if (
        training_provenance["filename"] != TRAINING_PROVENANCE_FILENAME
        or training_provenance["format"] != TRAINING_PROVENANCE_FORMAT
        or training_provenance["schema"] != TRAINING_PROVENANCE_SCHEMA
        or type(training_provenance["schema_version"]) is not int
        or training_provenance["schema_version"] != TRAINING_PROVENANCE_SCHEMA_VERSION
    ):
        raise PolicyArtifactError("unsupported PPO training provenance format")
    if not isinstance(training_provenance["sha256"], str) or not _SHA256_PATTERN.fullmatch(
        training_provenance["sha256"]
    ):
        raise PolicyArtifactError("training_provenance.sha256 must be a lowercase SHA-256 digest")
    if type(training_provenance["size_bytes"]) is not int or training_provenance["size_bytes"] <= 0:
        raise PolicyArtifactError("training_provenance.size_bytes must be a positive integer")

    architecture = _require_mapping(manifest["architecture"], field="architecture")
    _require_exact_fields(
        architecture,
        expected=_ARCHITECTURE_FIELDS,
        field="architecture",
    )
    expected_architecture = {
        "observation_dim": OBSERVATION_DIM,
        "hidden_size": HIDDEN_SIZE,
        "action_count": ACTION_COUNT,
        "locality_count": LOCALITY_COUNT,
        "activation": "tanh",
        "action_encoding": "little-endian-base-3",
    }
    if any(
        type(architecture[name]) is not type(expected_value) or architecture[name] != expected_value
        for name, expected_value in expected_architecture.items()
    ):
        raise PolicyArtifactError("PPO actor architecture does not match the accepted schema")

    source = _require_mapping(manifest["source"], field="source")
    _require_exact_fields(source, expected=_SOURCE_FIELDS, field="source")
    if type(source["checkpoint_path"]) is not str or source["checkpoint_path"] != (
        SOURCE_CHECKPOINT_PATH
    ):
        raise PolicyArtifactError("unexpected PPO source checkpoint path")
    if type(source["checkpoint_sha256"]) is not str or source["checkpoint_sha256"] != (
        SOURCE_CHECKPOINT_SHA256
    ):
        raise PolicyArtifactError("unexpected PPO source checkpoint SHA-256")
    if (
        type(source["source_revision"]) is not str
        or source["source_revision"] != SOURCE_REVISION
        or not _REVISION_PATTERN.fullmatch(source["source_revision"])
    ):
        raise PolicyArtifactError("unexpected PPO source revision")

    tensors = _require_mapping(manifest["tensors"], field="tensors")
    if set(tensors) != set(EXPECTED_ACTOR_TENSORS):
        raise PolicyArtifactError("PPO artifact tensor schema is incomplete or contains extras")
    for name, expected_shape in EXPECTED_ACTOR_TENSORS.items():
        metadata = _require_mapping(tensors[name], field=f"tensors.{name}")
        _require_exact_fields(metadata, expected=_TENSOR_FIELDS, field=f"tensors.{name}")
        shape = metadata["shape"]
        if (
            type(metadata["dtype"]) is not str
            or metadata["dtype"] != "float32"
            or not isinstance(shape, list)
            or any(type(dimension) is not int for dimension in shape)
            or shape != list(expected_shape)
        ):
            raise PolicyArtifactError(f"tensor {name} metadata has an invalid dtype or shape")
    return manifest


def _read_actor_arrays(path: Path) -> dict[str, np.ndarray[Any, np.dtype[np.float32]]]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if len(archive.files) != len(EXPECTED_ACTOR_TENSORS) or set(archive.files) != set(
                EXPECTED_ACTOR_TENSORS
            ):
                raise PolicyArtifactError(
                    "PPO artifact tensor set is incomplete or contains non-actor state"
                )
            state: dict[str, np.ndarray[Any, np.dtype[np.float32]]] = {}
            for name, expected_shape in EXPECTED_ACTOR_TENSORS.items():
                array = archive[name]
                if array.dtype != np.dtype(np.float32):
                    raise PolicyArtifactError(f"tensor {name} must use float32")
                if array.shape != expected_shape:
                    raise PolicyArtifactError(
                        f"tensor {name} shape must be {expected_shape}, found {array.shape}"
                    )
                if not np.isfinite(array).all():
                    raise PolicyArtifactError(f"tensor {name} contains non-finite values")
                state[name] = np.array(array, dtype=np.float32, copy=True, order="C")
    except PolicyArtifactError:
        raise
    except (OSError, ValueError) as error:
        raise PolicyArtifactError(f"PPO actor archive cannot be read safely: {error}") from error
    return state


def _require_mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PolicyArtifactError(f"{field} must be an object")
    return dict(value)


def _reject_duplicate_json_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in pairs:
        if name in result:
            raise PolicyArtifactError(f"PPO actor manifest contains duplicate field {name!r}")
        result[name] = value
    return result


def _require_exact_fields(
    value: Mapping[str, object],
    *,
    expected: frozenset[str],
    field: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extras = sorted(actual - expected)
        raise PolicyArtifactError(
            f"{field} schema fields mismatch; missing={missing}, extras={extras}"
        )

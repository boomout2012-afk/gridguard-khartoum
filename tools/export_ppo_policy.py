"""Export the trusted PPO checkpoint as a deterministic actor-only NumPy artifact."""

from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from gridguard.policy import (
    ACTION_COUNT,
    EXPECTED_ACTOR_TENSORS,
    HIDDEN_SIZE,
    LOCALITY_COUNT,
    MODEL_ARTIFACT_FILENAME,
    MODEL_MANIFEST_FILENAME,
    MODEL_SCHEMA,
    MODEL_SCHEMA_VERSION,
    OBSERVATION_DIM,
    SOURCE_CHECKPOINT_PATH,
    SOURCE_CHECKPOINT_SHA256,
    SOURCE_REVISION,
)
from gridguard.provenance import sha256_file

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


class PolicyExportError(RuntimeError):
    """Raised when the trusted source cannot produce the strict PPO artifact."""


@dataclass(frozen=True, slots=True)
class ExportedArtifact:
    artifact_path: Path
    manifest_path: Path
    artifact_sha256: str
    size_bytes: int


def load_trusted_actor_state(source_checkpoint: str | Path) -> dict[str, torch.Tensor]:
    """Read and validate only the accepted actor tensors from the trusted checkpoint."""

    source_path = Path(source_checkpoint)
    try:
        actual_hash = sha256_file(source_path)
    except OSError as error:
        raise PolicyExportError(f"source checkpoint cannot be hashed: {error}") from error
    if actual_hash != SOURCE_CHECKPOINT_SHA256:
        raise PolicyExportError(
            f"source checkpoint SHA-256 mismatch: expected {SOURCE_CHECKPOINT_SHA256}, "
            f"found {actual_hash}"
        )

    try:
        checkpoint = torch.load(source_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise PolicyExportError(
            f"trusted PPO checkpoint cannot be loaded safely: {error}"
        ) from error
    if not isinstance(checkpoint, Mapping):
        raise PolicyExportError("trusted PPO checkpoint must contain a mapping")

    architecture = checkpoint.get("architecture")
    if not isinstance(architecture, Mapping):
        raise PolicyExportError("trusted PPO checkpoint is missing architecture metadata")
    if architecture.get("observation_dim") != OBSERVATION_DIM:
        raise PolicyExportError("trusted PPO checkpoint has an unexpected observation width")
    if architecture.get("hidden_size") != HIDDEN_SIZE:
        raise PolicyExportError("trusted PPO checkpoint has an unexpected actor hidden size")

    model = checkpoint.get("model")
    if not isinstance(model, Mapping):
        raise PolicyExportError("trusted PPO checkpoint is missing its model state")

    actor_state: dict[str, torch.Tensor] = {}
    for name, expected_shape in EXPECTED_ACTOR_TENSORS.items():
        tensor = model.get(name)
        if not isinstance(tensor, torch.Tensor):
            raise PolicyExportError(f"trusted PPO checkpoint is missing tensor {name}")
        cpu_tensor = tensor.detach().to(device="cpu").contiguous()
        if cpu_tensor.dtype != torch.float32:
            raise PolicyExportError(f"trusted PPO tensor {name} must use float32")
        if tuple(cpu_tensor.shape) != expected_shape:
            raise PolicyExportError(
                f"trusted PPO tensor {name} shape must be {expected_shape}, "
                f"found {tuple(cpu_tensor.shape)}"
            )
        if not torch.isfinite(cpu_tensor).all():
            raise PolicyExportError(f"trusted PPO tensor {name} contains non-finite values")
        actor_state[name] = cpu_tensor.clone()
    return actor_state


def write_ppo_artifact(
    actor_state: Mapping[str, torch.Tensor | np.ndarray],
    *,
    output_dir: str | Path,
    source_checkpoint_sha256: str,
    source_checkpoint_path: str,
    source_revision: str,
    validate_shapes: bool = True,
) -> ExportedArtifact:
    """Serialize validated actor tensors with deterministic bytes and a strict manifest."""

    if set(actor_state) != set(EXPECTED_ACTOR_TENSORS):
        raise PolicyExportError("actor state must contain exactly the accepted actor tensors")
    if source_checkpoint_sha256 != SOURCE_CHECKPOINT_SHA256 or not _SHA256_PATTERN.fullmatch(
        source_checkpoint_sha256
    ):
        raise PolicyExportError("unexpected source checkpoint SHA-256")
    if source_checkpoint_path != SOURCE_CHECKPOINT_PATH:
        raise PolicyExportError("unexpected source checkpoint path")
    if source_revision != SOURCE_REVISION or not _REVISION_PATTERN.fullmatch(source_revision):
        raise PolicyExportError("unexpected source revision")

    arrays: dict[str, np.ndarray] = {}
    for name, expected_shape in EXPECTED_ACTOR_TENSORS.items():
        value = actor_state[name]
        if isinstance(value, torch.Tensor):
            array = value.detach().to(device="cpu").contiguous().numpy()
        elif isinstance(value, np.ndarray):
            array = value
        else:
            raise PolicyExportError(f"actor tensor {name} has an unsupported type")
        if array.dtype != np.dtype(np.float32):
            raise PolicyExportError(f"actor tensor {name} must use float32")
        if validate_shapes and array.shape != expected_shape:
            raise PolicyExportError(
                f"actor tensor {name} shape must be {expected_shape}, found {array.shape}"
            )
        if not np.isfinite(array).all():
            raise PolicyExportError(f"actor tensor {name} contains non-finite values")
        arrays[name] = np.array(array, dtype=np.float32, copy=True, order="C")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    artifact_path = destination / MODEL_ARTIFACT_FILENAME
    temporary_artifact = destination / f".{MODEL_ARTIFACT_FILENAME}.tmp"
    try:
        _write_deterministic_npz(temporary_artifact, arrays)
        temporary_artifact.replace(artifact_path)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        temporary_artifact.unlink(missing_ok=True)
        raise PolicyExportError(f"PPO actor artifact could not be written: {error}") from error

    try:
        artifact_sha256 = sha256_file(artifact_path)
        size_bytes = artifact_path.stat().st_size
    except OSError as error:
        raise PolicyExportError(f"PPO actor artifact cannot be inspected: {error}") from error
    manifest = {
        "schema": MODEL_SCHEMA,
        "schema_version": MODEL_SCHEMA_VERSION,
        "artifact": {
            "filename": MODEL_ARTIFACT_FILENAME,
            "format": "numpy.npz",
            "sha256": artifact_sha256,
            "size_bytes": size_bytes,
        },
        "architecture": {
            "observation_dim": OBSERVATION_DIM,
            "hidden_size": HIDDEN_SIZE,
            "action_count": ACTION_COUNT,
            "locality_count": LOCALITY_COUNT,
            "activation": "tanh",
            "action_encoding": "little-endian-base-3",
        },
        "source": {
            "checkpoint_path": source_checkpoint_path,
            "checkpoint_sha256": source_checkpoint_sha256,
            "source_revision": source_revision,
        },
        "tensors": {
            name: {"dtype": "float32", "shape": list(array.shape)}
            for name, array in sorted(arrays.items())
        },
    }
    manifest_path = destination / MODEL_MANIFEST_FILENAME
    temporary_manifest = destination / f".{MODEL_MANIFEST_FILENAME}.tmp"
    try:
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary_manifest.replace(manifest_path)
    except OSError as error:
        temporary_manifest.unlink(missing_ok=True)
        raise PolicyExportError(f"PPO actor manifest could not be written: {error}") from error
    return ExportedArtifact(
        artifact_path=artifact_path,
        manifest_path=manifest_path,
        artifact_sha256=artifact_sha256,
        size_bytes=size_bytes,
    )


def export_ppo_policy(
    source_checkpoint: str | Path,
    *,
    output_dir: str | Path,
) -> ExportedArtifact:
    """Export the trusted checkpoint's actor and omit every non-inference field."""

    actor_state = load_trusted_actor_state(source_checkpoint)
    return write_ppo_artifact(
        actor_state,
        output_dir=output_dir,
        source_checkpoint_sha256=SOURCE_CHECKPOINT_SHA256,
        source_checkpoint_path=SOURCE_CHECKPOINT_PATH,
        source_revision=SOURCE_REVISION,
    )


def _write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.save(buffer, arrays[name], allow_pickle=False)
            entry = zipfile.ZipInfo(f"{name}.npy", date_time=_FIXED_ZIP_TIMESTAMP)
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.create_system = 3
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, buffer.getvalue(), compresslevel=9)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-checkpoint",
        required=True,
        type=Path,
        help="Path to the trusted source checkpoint with the pinned SHA-256",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "model",
        help="Directory for ppo-policy.npz and manifest.json",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    exported = export_ppo_policy(args.source_checkpoint, output_dir=args.output_dir)
    print(
        json.dumps(
            {
                "artifact": str(exported.artifact_path),
                "manifest": str(exported.manifest_path),
                "sha256": exported.artifact_sha256,
                "size_bytes": exported.size_bytes,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

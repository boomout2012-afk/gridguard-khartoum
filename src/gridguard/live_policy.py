"""Pinned deployment of the validation-selected actor from the focused PPO study.

The runtime consumes one selected policy. It does not average five policies or
substitute a live episode for the thesis's frozen matched evaluation. Existing
training, masking, simulation and metric implementations remain unchanged.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch

from gridguard.constrained_ppo import ConstrainedPPOController, load_model
from gridguard.focused_training import CONFIG, TRAINING_SEEDS, select_checkpoint
from gridguard.resources import runtime_root

TRAINING_RELATIVE_PATH = "data/evidence/focused-ppo-training.json"


@dataclass(frozen=True, slots=True)
class LivePolicyIdentity:
    """Verified model identity carried through the live presentation boundary."""

    training_seed: int
    checkpoint_iteration: int
    model_relative_path: str
    model_sha256: str
    training_sha256: str


DEPLOYED_LIVE_POLICY = LivePolicyIdentity(
    training_seed=8111,
    checkpoint_iteration=2500,
    model_relative_path="data/model/focused-ppo-study/seed-8111-selected.npz",
    model_sha256="9c3dbed2cdd979bdd566004ecaada8c78a3aef26397eb7e90a2b953d1da4f2d3",
    training_sha256="e4f8b8f7c73693f713ecef69a06434ce75a3a101d2541654d7009398c4eb5624",
)


@dataclass(frozen=True, slots=True)
class LoadedLivePolicy:
    """Frozen inference controller and the exact provenance verified at loading."""

    controller: ConstrainedPPOController
    identity: LivePolicyIdentity


def _inside(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("live policy asset must remain inside its runtime root")
    return candidate


def _verify_selection(record: dict) -> None:
    if record.get("schema") != "gridguard.focused_ppo_training.1":
        raise ValueError("live training record schema mismatch")
    if record.get("config") != CONFIG:
        raise ValueError("live training configuration differs from the frozen study")
    runs = record.get("runs")
    if not isinstance(runs, list) or [run.get("seed") for run in runs] != list(TRAINING_SEEDS):
        raise ValueError("live training record requires the five declared runs")
    for run in runs:
        if run["selected"] != select_checkpoint(run["checkpoints"], record["reference_validation"]):
            raise ValueError("live checkpoint selection does not reconstruct")
    eligible = [run for run in runs if run["selected"]["eligible"]]
    if not eligible:
        raise ValueError("live deployment requires a validation-eligible policy")
    selected = min(
        eligible,
        key=lambda run: (-run["selected"]["summary"]["overall"]["episode_reward"], run["seed"]),
    )
    identity = DEPLOYED_LIVE_POLICY
    if (
        record.get("deployment_seed") != identity.training_seed
        or selected["seed"] != identity.training_seed
        or selected["selected"]["iteration"] != identity.checkpoint_iteration
        or selected["selected"]["fallback"]
        or selected["model"]
        != {"path": identity.model_relative_path, "sha256": identity.model_sha256}
    ):
        raise ValueError("live deployment identity differs from validation selection")


def load_live_policy(
    *,
    root: Path | None = None,
    training_path: str | Path | None = None,
    device: torch.device | str = "cuda:0",
) -> LoadedLivePolicy:
    """Verify the pinned record and weights, then load deterministic masked PPO.

    CPU loading supports small provenance tests only; the live runner separately
    requires cuda:0. Missing or modified assets never trigger a legacy fallback.
    """
    base = (runtime_root() if root is None else Path(root)).resolve()
    source = (
        _inside(base, TRAINING_RELATIVE_PATH)
        if training_path is None
        else Path(training_path).resolve()
    )
    payload = source.read_bytes()
    if hashlib.sha256(payload).hexdigest() != DEPLOYED_LIVE_POLICY.training_sha256:
        raise ValueError("live training record SHA-256 does not match the pinned study")
    _verify_selection(json.loads(payload))

    # Match the frozen study's FP32 settings before actor or scenario arithmetic.
    # Entry points set the workspace environment before CUDA preflight as well.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    model = load_model(
        _inside(base, DEPLOYED_LIVE_POLICY.model_relative_path),
        device=device,
        expected_sha256=DEPLOYED_LIVE_POLICY.model_sha256,
    )
    return LoadedLivePolicy(ConstrainedPPOController(model), DEPLOYED_LIVE_POLICY)

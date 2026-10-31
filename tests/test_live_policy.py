"""Deployment provenance must fail closed before model or scenario execution."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch

from gridguard.live_policy import DEPLOYED_LIVE_POLICY, load_live_policy

ROOT = Path(__file__).resolve().parents[1]


def _runtime_copy(tmp_path: Path) -> Path:
    for relative in (
        "data/evidence/focused-ppo-training.json",
        DEPLOYED_LIVE_POLICY.model_relative_path,
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    return tmp_path


def test_live_policy_is_the_validation_selected_frozen_actor() -> None:
    loaded = load_live_policy(root=ROOT, device="cpu")
    identity = loaded.identity

    assert identity == DEPLOYED_LIVE_POLICY
    assert identity.training_seed == 8111
    assert identity.checkpoint_iteration == 2500
    assert identity.model_sha256 == (
        "9c3dbed2cdd979bdd566004ecaada8c78a3aef26397eb7e90a2b953d1da4f2d3"
    )
    assert identity.training_sha256 == (
        "e4f8b8f7c73693f713ecef69a06434ce75a3a101d2541654d7009398c4eb5624"
    )
    model = loaded.controller.model
    assert model.actor[0].in_features == 46
    assert model.actor[-1].out_features == 128
    assert not model.training
    assert not any(parameter.requires_grad for parameter in model.parameters())
    assert torch.get_float32_matmul_precision() == "highest"
    assert not torch.backends.cuda.matmul.allow_tf32


@pytest.mark.parametrize("mutation", ["training", "model", "missing-model"])
def test_live_policy_rejects_modified_or_missing_assets(tmp_path, mutation) -> None:
    root = _runtime_copy(tmp_path)
    if mutation == "training":
        path = root / "data/evidence/focused-ppo-training.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["deployment_seed"] = 1729
        path.write_text(json.dumps(record), encoding="utf-8")
    else:
        path = root / DEPLOYED_LIVE_POLICY.model_relative_path
        if mutation == "missing-model":
            path.unlink()
        else:
            payload = bytearray(path.read_bytes())
            payload[-1] ^= 1
            path.write_bytes(payload)

    with pytest.raises((ValueError, FileNotFoundError), match="SHA-256|cannot find|No such file"):
        load_live_policy(root=root, device="cpu")


def test_live_policy_does_not_fall_back_to_earlier_actor(tmp_path) -> None:
    (tmp_path / "data/model").mkdir(parents=True)
    shutil.copyfile(ROOT / "data/model/manifest.json", tmp_path / "data/model/manifest.json")
    with pytest.raises(FileNotFoundError):
        load_live_policy(root=tmp_path, device="cpu")

"""The frozen smoke must prove the deployed research policy and constraints."""

from dataclasses import replace
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace

import pytest


def _comparison():
    from gridguard.live_policy import DEPLOYED_LIVE_POLICY

    clean = SimpleNamespace(
        power_balance_violation_rate=0.0,
        duration=SimpleNamespace(critical_interruption_locality_hours=0.0),
    )
    return SimpleNamespace(
        policy_identity=DEPLOYED_LIVE_POLICY,
        metrics_by_controller={"PPO": clean, "Optimization": clean},
    )


def _check():
    launcher = run_path(str(Path(__file__).parents[1] / "packaging/gridguard_launcher.py"))
    return launcher["_deployed_policy_smoke_facts"]


def test_frozen_smoke_records_validation_selected_policy() -> None:
    facts = _check()(_comparison())
    assert facts["training_seed"] == 8111
    assert facts["checkpoint_iteration"] == 2500
    assert facts["critical_and_power_checks_passed"] is True


@pytest.mark.parametrize("mutation", ["missing", "wrong_hash", "critical", "power"])
def test_frozen_smoke_rejects_wrong_policy_or_constraint_failure(mutation: str) -> None:
    comparison = _comparison()
    if mutation == "missing":
        comparison.policy_identity = None
    elif mutation == "wrong_hash":
        comparison.policy_identity = replace(comparison.policy_identity, model_sha256="0" * 64)
    elif mutation == "critical":
        comparison.metrics_by_controller["PPO"].duration.critical_interruption_locality_hours = 1
    else:
        comparison.metrics_by_controller["Optimization"].power_balance_violation_rate = 0.1
    with pytest.raises(RuntimeError, match="policy|constraint"):
        _check()(comparison)

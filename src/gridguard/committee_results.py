"""Pinned display-only follow-up results; no Qt, models or scientific calculations."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

from gridguard.focused_results import DisplayTable
from gridguard.resources import runtime_root

AUDIT_PATH = "data/evidence/committee-baseline-audit.json"
VALIDATION_PATH = "data/evidence/committee-baseline-audit.validation.json"
PINNED_FILES = {
    AUDIT_PATH: "8029c5a6020388d412ee3263849cfe55c786227d1b16a45f665e83918e7fab6b",
    VALIDATION_PATH: "ac65f72300936713cde15e66817ae040e64761730aced9d853c7f8fd37fd3ea4",
}
BASELINES = (
    "Fixed Rotation",
    "Critical-aware Fixed Rotation",
    "Constrained Optimization",
    "Fairness Rule",
)
RUNS = (1729, 2718, 3141, 5772, 8111)
METHODS = (*BASELINES, *(f"PPO seed {seed}" for seed in RUNS))
DISPLAY_METHODS = (*BASELINES, "PPO")
_METRICS = (
    ("served_mwh", "Served energy", "MWh", 1, 2),
    ("critical_interruption_locality_hours", "Critical interruption", "locality-h", 1, 2),
    ("fairness", "Fairness", "pp", 100, 2),
    ("utilization", "Utilization", "pp", 100, 2),
    ("episode_reward", "Daily reward", "", 1, 4),
    ("switching_rate", "Switching", "pp", 100, 2),
)


class CommitteeResultsError(RuntimeError):
    """The follow-up artifact or its independent validation receipt is unavailable."""


@dataclass(frozen=True, slots=True)
class CommitteeResultsView:
    scope_statement: str
    count_statement: str
    baseline_statement: str
    tradeoff_statement: str
    interval_statement: str
    stress_statement: str
    means: DisplayTable
    paired: tuple[tuple[str, DisplayTable], ...]
    stress: DisplayTable
    evidence: tuple[tuple[str, str], ...]


def _load(root: Path, relative: str) -> dict:
    raw = (root / relative).read_bytes()
    if hashlib.sha256(raw).hexdigest() != PINNED_FILES[relative]:
        raise ValueError(f"SHA-256 mismatch: {relative}")
    return json.loads(raw)


def _number(value: object) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("display values must be finite numbers")
    return float(value)


def _signed(value: float, digits: int) -> str:
    return f"{_number(value):+.{digits}f}".replace("-", "−")


def _row(payload: dict, name: str) -> dict:
    return payload["procedure_means"][name] if name == "PPO" else payload["controller_means"][name]


def load_committee_results(*, root: Path | None = None) -> CommitteeResultsView:
    """Present code-pinned, already-verified means and intervals without recomputing them."""
    try:
        base = runtime_root() if root is None else root
        audit = _load(base, AUDIT_PATH)
        receipt = _load(base, VALIDATION_PATH)
        if (
            audit["schema"] != "gridguard.committee_baseline_audit.1"
            or receipt["schema"] != "gridguard.committee_baseline_validation.1"
            or receipt["passed"] is not True
            or receipt["audit"] != {"path": AUDIT_PATH, "sha256": PINNED_FILES[AUDIT_PATH]}
            or receipt["verification"]["passed"] is not True
            or audit["recipe"]["post_hoc"] is not True
            or audit["recipe"]["training_seeds"] != list(RUNS)
            or audit["recipe"]["interval_confidence"] != 0.95
            or audit["nominal_matched_conditions"] != 960
            or audit["nominal_trajectories"] != 8640
            or audit["stress_conditions_per_family"] != 320
            or audit["stress_trajectories"] != 5760
            or audit["controller_names"] != list(METHODS)
            or set(audit["controller_means"]) != set(METHODS)
            or set(audit["procedure_means"]) != {"PPO"}
        ):
            raise ValueError("follow-up identity, counts or verification differ")
        return _build_view(audit)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        raise CommitteeResultsError(f"Cannot verify the follow-up comparison: {error}") from error


def _build_view(audit: dict) -> CommitteeResultsView:
    means = tuple(
        (
            name,
            *(
                f"{_number(_row(audit, name)[key]) * scale:.{digits}f}"
                for key, _, _, scale, digits in _METRICS
            ),
        )
        for name in DISPLAY_METHODS
    )
    paired = []
    for baseline in ("Critical-aware Fixed Rotation", "Fairness Rule"):
        rows = []
        for key, label, unit, scale, digits in (_METRICS[0], _METRICS[2], _METRICS[3], _METRICS[4]):
            effect = audit["paired_inference"][f"PPO - {baseline}"][key]
            mean, lower, upper = (_number(effect[field]) for field in ("mean", "lower", "upper"))
            if lower > upper:
                raise ValueError("follow-up interval endpoints are reversed")
            rows.append(
                (
                    label,
                    _signed(mean * scale, digits),
                    (
                        f"[{_signed(lower * scale, digits)}, "
                        f"{_signed(upper * scale, digits)}] {unit}"
                    ).rstrip(),
                )
            )
        paired.append(
            (
                f"PPO vs {baseline}",
                DisplayTable(("Outcome", "PPO difference", "95% interval"), tuple(rows)),
            )
        )
    stress_rows = []
    for family, label in (("critical_supply_75pct", "75%"), ("critical_supply_25pct", "25%")):
        payload = audit["stress"][family]
        for name in DISPLAY_METHODS:
            row = _row(payload, name)
            stress_rows.append(
                (
                    label,
                    name,
                    *(
                        f"{_number(row[key]):.{digits}f}"
                        for key, digits in (
                            ("all_critical_fallback_hours", 0),
                            ("critical_interruption_locality_hours", 0),
                            ("complete_critical_outage_locality_hours", 0),
                            ("unmet_critical_mwh", 2),
                            ("power_balance_violation_rate", 0),
                        )
                    ),
                )
            )
    effect = audit["paired_inference"]["PPO - Fairness Rule"]
    return CommitteeResultsView(
        scope_statement=(
            "Post-hoc follow-up · previously evaluated scenarios · "
            "no retraining or new confirmatory claim."
        ),
        count_statement=(
            "Same 960 matched conditions · 320 base draws · five frozen selected PPO policies"
        ),
        baseline_statement=(
            "Critical-aware Fixed Rotation reserves critical demand, then rotates discretionary "
            "priority without reward optimization. Fairness Rule uses the frozen imitation-teacher "
            "setting. These comparisons do not isolate learning alone."
        ),
        tradeoff_statement=(
            "PPO versus Fairness Rule is a tradeoff: "
            f"daily reward {_signed(effect['episode_reward']['mean'], 4)}, "
            f"served energy {_signed(effect['served_mwh']['mean'], 2)} MWh/day. "
            "Reward is dimensionless; it cannot be converted into energy or cost."
        ),
        interval_statement=(
            "Descriptive pointwise paired 95% intervals; pp = percentage points. "
            "All five PPO runs receive equal weight. Original primary criteria remain unchanged."
        ),
        stress_statement=(
            "Supply is 75% or 25% of critical demand: the critical floor is physically infeasible. "
            "Critical-only fallback shares the shortage; it is not uninterrupted critical service. "
            "Each stress family has 320 conditions and stays separate from nominal results."
        ),
        means=DisplayTable(
            (
                "Controller",
                "Served\nMWh",
                "Critical\nlocality-h",
                "Fairness\n%",
                "Utilization\n%",
                "Daily\nreward",
                "Switching\n%",
            ),
            means,
        ),
        paired=tuple(paired),
        stress=DisplayTable(
            (
                "Supply /\ncritical",
                "Controller",
                "Fallback\nhours",
                "Interrupted\nlocality-h",
                "Zero critical\nlocality-h",
                "Unmet critical\nMWh",
                "Power violation\nrate",
            ),
            tuple(stress_rows),
        ),
        evidence=tuple(PINNED_FILES.items()),
    )

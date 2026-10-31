"""Pinned, display-only current-study data; no Qt, simulation or model deployment."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from gridguard.resources import runtime_root

STUDY_PATH = "data/evidence/focused-ppo-study.json"
TRAINING_PATH = "data/evidence/focused-ppo-training.json"
AUDIT_PATH = "docs/focused-ppo-validation.json"
PINNED_FILES = {
    STUDY_PATH: "a4d5a849612f06eedef96e0d28e08247e5e3757eba0b4f2e4f1a893e9d4b8247",
    TRAINING_PATH: "e4f8b8f7c73693f713ecef69a06434ce75a3a101d2541654d7009398c4eb5624",
    AUDIT_PATH: "df1cc59a3a1cd94f8333f77693dc47ef1afb6007093d940dd9deab06383a1fe1",
}
LIVE_POLICY_STATEMENT = (
    "Live Scenario uses one validation-selected PPO policy from this training study "
    "with the same critical-service constraints and constrained optimizer. "
    "A single scenario can differ from these frozen five-run benchmark means."
)
_METRICS = (
    ("critical_interruption_locality_hours", "Critical interruption", "locality-h", 1, 2),
    ("fairness", "Jain service fairness", "%", 100, 2),
    ("utilization", "Supply utilization", "%", 100, 2),
    ("episode_reward", "Daily reward", "", 1, 4),
    ("switching_rate", "Switching rate", "%", 100, 2),
)


class FocusedStudyError(RuntimeError):
    """The pinned current study is missing, changed or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class DisplayTable:
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True, slots=True)
class FocusedStudyView:
    count_statement: str
    training_statement: str
    fixed_conclusion: str
    optimizer_conclusion: str
    tradeoff: str
    interval_statement: str
    live_statement: str
    safety_statement: str
    means: DisplayTable
    paired: tuple[tuple[str, DisplayTable], ...]
    runs: DisplayTable
    shortage: DisplayTable
    evidence: tuple[tuple[str, str], ...]


def _signed(value: float, digits: int) -> str:
    return f"{value:+.{digits}f}".replace("-", "−")


def _load(root: Path, relative: str) -> dict:
    raw = (root / relative).read_bytes()
    if hashlib.sha256(raw).hexdigest() != PINNED_FILES[relative]:
        raise ValueError(f"SHA-256 mismatch: {relative}")
    return json.loads(raw)


def load_focused_results(*, root: Path | None = None) -> FocusedStudyView:
    """Load exact already-verified evidence without recalculating scientific outputs.

    Hashes are code-pinned, not accepted from mutable adjacent checksum files.
    The separate audit is itself pinned and linked to the exact study/training.
    Frozen scientific source hashes remain historical provenance; UI edits do not
    cause a new claim that those sources or a new actor have been deployed.
    """
    try:
        base = runtime_root() if root is None else root
        study = _load(base, STUDY_PATH)
        training = _load(base, TRAINING_PATH)
        audit = _load(base, AUDIT_PATH)
        if (
            study["schema"] != "gridguard.focused_ppo_study.1"
            or training["schema"] != "gridguard.focused_ppo_training.1"
            or audit["schema"] != "gridguard.focused_ppo_replay_validation.1"
            or study["training_sha256"] != PINNED_FILES[TRAINING_PATH]
            or audit["training_sha256"] != PINNED_FILES[TRAINING_PATH]
            or audit["study_sha256"] != PINNED_FILES[STUDY_PATH]
            or audit["passed"] is not True
            or study["matched_scenario_count"] != 960
            or study["controller_trajectory_count"] != 11520
            or len(training["runs"]) != 5
        ):
            raise ValueError("study identity, counts or replay verification differ")
        return _build_view(study, training)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise FocusedStudyError(f"Cannot verify the current study: {error}") from error


def _build_view(study: dict, training: dict) -> FocusedStudyView:
    # Only unit conversion and display formatting occur here. Published means,
    # interval endpoints and decisions come directly from the pinned artifacts.
    fixed = study["controller_means"]["Fixed Rotation"]
    optimizer = study["controller_means"]["Constrained Optimization"]
    ppo = study["procedure_means"]["PPO"]
    mean_rows = tuple(
        (
            label,
            *(
                f"{row[key] * scale:.{digits}f}{'%' if unit == '%' else ''}"
                + (" locality-h" if unit == "locality-h" else "")
                for row in (fixed, optimizer, ppo)
            ),
        )
        for key, label, unit, scale, digits in _METRICS
    )
    paired = []
    for pair in ("PPO - Fixed Rotation", "PPO - Constrained Optimization"):
        rows = []
        for key, label, unit, scale, _digits in _METRICS:
            effect = study["primary_family_adjusted_inference"][pair][key]
            delta_unit = "pp" if unit == "%" else unit
            rows.append(
                (
                    label,
                    _signed(effect["mean"] * scale, 4),
                    f"[{_signed(effect['lower'] * scale, 4)}, "
                    f"{_signed(effect['upper'] * scale, 4)}] {delta_unit}".rstrip(),
                )
            )
        paired.append(
            (pair, DisplayTable(("Outcome", "PPO difference", "97.5% interval"), tuple(rows)))
        )
    runs = tuple(
        (
            str(run["seed"]),
            str(run["selected"]["iteration"]),
            f"{study['controller_means'][f'PPO seed {run["seed"]}']['fairness']:.2%}",
            f"{study['controller_means'][f'PPO seed {run["seed"]}']['utilization']:.2%}",
            f"{study['controller_means'][f'PPO seed {run["seed"]}']['episode_reward']:.4f}",
        )
        for run in training["runs"]
    )
    shortage_rows = []
    for severity in study["shortages"]:
        for name, row in (
            ("Fixed Rotation", study["shortage_means"][severity]["Fixed Rotation"]),
            (
                "Constrained optimizer",
                study["shortage_means"][severity]["Constrained Optimization"],
            ),
            ("PPO", study["procedure_shortage_means"][severity]["PPO"]),
        ):
            shortage_rows.append(
                (
                    severity.title(),
                    name,
                    f"{row['fairness']:.2%}",
                    f"{row['utilization']:.2%}",
                    f"{row['episode_reward']:.4f}",
                )
            )
    effect = study["primary_family_adjusted_inference"]["PPO - Constrained Optimization"]
    fixed_passed = study["claims"]["PPO - Fixed Rotation"]["passed"]
    optimizer_passed = study["claims"]["PPO - Constrained Optimization"]["passed"]
    return FocusedStudyView(
        count_statement="960 matched conditions · 320 base draws · five independent PPO runs",
        training_statement="3,000 iterations per run · 92.16 million training transitions total",
        fixed_conclusion=(
            "PPO met the service-improvement criterion against Fixed Rotation."
            if fixed_passed
            else "PPO did not meet the full criterion against Fixed Rotation."
        ),
        optimizer_conclusion=(
            "PPO met the full criterion against the constrained optimizer."
            if optimizer_passed
            else (
                "The full optimizer comparison did not pass: utilization exceeded the allowed loss."
            )
        ),
        tradeoff=(
            "Against the constrained optimizer: "
            f"fairness {_signed(effect['fairness']['mean'] * 100, 3)} pp"
            f" · utilization {_signed(effect['utilization']['mean'] * 100, 3)} pp"
            f" · daily reward {_signed(effect['episode_reward']['mean'], 4)}."
        ),
        interval_statement=(
            "Paired 97.5% intervals adjusted for the two primary comparisons. "
            "pp = percentage points; PPO means weight all five runs equally."
        ),
        live_statement=LIVE_POLICY_STATEMENT,
        safety_statement=(
            "Zero critical interruptions and power-balance violations "
            "for PPO and the constrained optimizer."
        ),
        means=DisplayTable(
            ("Outcome", "Fixed Rotation", "Constrained optimizer", "PPO"), mean_rows
        ),
        paired=tuple(paired),
        runs=DisplayTable(
            ("PPO seed", "Selected iteration", "Fairness", "Utilization", "Reward"), runs
        ),
        shortage=DisplayTable(
            ("Shortage", "Controller", "Fairness", "Utilization", "Reward"), tuple(shortage_rows)
        ),
        evidence=tuple(PINNED_FILES.items()),
    )

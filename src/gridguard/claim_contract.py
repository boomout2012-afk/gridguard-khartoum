"""Canonical bounded language for reviewer-facing scientific claims."""

from typing import Final

DOCUMENTED_SEED_DISJOINT_ROLE: Final = "documented_seed_disjoint_holdout"
DOCUMENTED_SEED_DISJOINT_LABEL: Final = "documented-seed-disjoint holdout"

HOLDOUT_CUSTODY_LIMITATION: Final = (
    "Seeds are disjoint only from every documented training, validation, and model-selection "
    "seed in data/model/training-provenance.yaml. The holdout was not preregistered or "
    "independently custodied and cannot rule out undocumented activity."
)
SENSITIVITY_INTERVAL_LIMITATION: Final = (
    "Sensitivity intervals are descriptive, pointwise 95%, and unadjusted for multiple "
    "comparisons; they do not support family-wide significance claims."
)
PPO_ATTRIBUTION_LIMITATION: Final = (
    "No behavior-cloning-only ablation was run, so this comparison cannot isolate the "
    "incremental effect of PPO fine-tuning."
)
FIXED_ROTATION_LIMITATION: Final = (
    "Fixed Rotation is supply-blind, limited to Off and Full Service, "
    "and cannot select Critical Only."
)
SYNTHETIC_EVIDENCE_LIMITATION: Final = "Synthetic decision support only, not field evidence."
RESULTS_SCOPE_LIMITATION: Final = (
    f"Comparator boundary: {FIXED_ROTATION_LIMITATION} {SYNTHETIC_EVIDENCE_LIMITATION} "
    f"{PPO_ATTRIBUTION_LIMITATION}"
)

# Changelog

## 1.2.0 - Unreleased

This section tracks the active adversarial-remediation release. It is not completed release
evidence: each A01-A24 row must be closed against current artifacts and the exact reviewed commit
before 1.2.0 is promoted.

### Added

- Added a reviewer entry point and a severity-classified A01-A24 audit register with one required
  disposition, implementation owner, live status, and closure-evidence slot per finding.
- Made the canonical, strict `data/model/training-provenance.yaml` record conspicuous from both the
  README and specification without claiming that excluded optimizer state or raw runs exist.

### In progress

- Correct scientific interval, scenario-identity, sensitivity, seed-role, and attribution
  semantics while preserving the approved proposal boundary.
- Rebuild claim surfaces, release validation, Windows packaging, and the manifest, then publish
  the exact reviewed branch SHA privately.

## 1.1.0 - 2026-08-26

- Reframed the historical paired benchmark as accepted reproduction evidence after verifying that
  its seeds participated in model selection; numerical results and the frozen benchmark remain
  unchanged.
- Made critical-outage locality-hours, Jain fairness, and utilization the three primary outcomes;
  relabeled reward as Training-objective score (diagnostic) and kept power balance as safety.
- Added a strict, model-manifest-bound `data/model/training-provenance.yaml` record containing the
  network, behavior-cloning initialization, PPO hyperparameters, workload, seeds, selection,
  hardware, convergence, and source hashes.
- Added untouched confirmatory evidence over ten new seeds, 24 one-factor-at-a-time sensitivity
  variants, and a causal trailing-mean forecast comparison against persistence.
- Added domain Related Work, a criticism disposition ledger, explicit baseline/dispatch/scaling/
  optimum/shortage limitations, and reviewer-facing training/evidence summaries.

## 1.0.0 - 2026-08-15

- Rebuilt the accepted graduation-project scope in a clean, consistently named repository.
- Added a fail-closed CUDA runtime using PyTorch and CuPy with no silent CPU fallback.
- Exported the frozen PPO policy to a pickle-free, hash-verified runtime artifact.
- Reproduced the accepted Fixed Rotation comparison and added a paired, cluster-aware evaluation.
- Added a native offline desktop presentation with an interactive Khartoum schematic map.
- Added auditable methodology, implementation, results, provenance, and release documentation.
- Added a reproducible one-directory Windows presentation build.

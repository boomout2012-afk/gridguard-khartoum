# Changelog

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

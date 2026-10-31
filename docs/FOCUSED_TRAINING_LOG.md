# Focused training execution log

The pretraining protocol freeze was created on 23 October 2026 with SHA-256
`58eaaf418ddabee63552b13163c0b97b1cdebf1d85b0a07e84cdcad5a930b8ff`.

The initial process stopped unexpectedly after seed 1729 reported iteration 1925
(301.65 seconds in its last telemetry row). At 09:36 UTC, the training process and
the owned TensorBoard server and bridge were absent; the execution session was no
longer available. The cause was not established from an error log. No final-test
scenarios had been generated and no complete training run had been published.

The incomplete actor snapshots, telemetry and TensorBoard events were preserved
under `tmp/focused-ppo-interrupted-20261023-0936`. The original source and protocol
freeze were retained. Training restarted from the same fresh seed and unchanged
configuration as a hidden background process, with stdout and stderr saved to
`tmp/focused-ppo-training.log` and `tmp/focused-ppo-training-error.log`.

Training times in the final study record describe completed runs. They exclude
the interrupted attempt and document-authoring time. No partial attempt is used
as a completed run, a validation selection or final-test evidence.

## Completed run and evaluation

All five restarted runs completed 3,000 iterations with 256 episodes of 24 steps per
iteration: 92,160,000 PPO transitions in total. Completed training took 35.976683
minutes. Validation selected iterations 2900, 3000, 2900, 3000 and 2500 for seeds
1729, 2718, 3141, 5772 and 8111, respectively. Final evaluation used 960 fresh matched
conditions after selection was frozen.

Published SHA-256 identities:

- Training: `e4f8b8f7c73693f713ecef69a06434ce75a3a101d2541654d7009398c4eb5624`.
- Final study: `a4d5a849612f06eedef96e0d28e08247e5e3757eba0b4f2e4f1a893e9d4b8247`.
- Diagnostic aggregates: `b4c2c76e481e0f30e323ea22f9359212bac1b2018605f833682875d98face500`.

The evidence verifier reconstructed final estimates, bootstrap intervals and claim
decisions on CUDA. The independent replay in `docs/focused-ppo-validation.json`
checked all scenario identities, 2,304 final trajectories and 5,280 validation
trajectories with maximum absolute error zero. The full declared comparison passes
against Fixed Rotation and fails against Constrained Optimization because of the
utilization and severity guardrails, despite reward and fairness gains.

TensorBoard retains the five completed runs and an Across-run variability view.
The PDF diagnostics use unsmoothed logged updates, equal seed weighting and sample
variance with denominator four. The exact original aggregate creator is archived at
`data/evidence/accepted-sources/plot_training_diagnostics-3ac68a10ebff.py`; subsequent
font-size changes use verified render-only mode and leave aggregate evidence intact.

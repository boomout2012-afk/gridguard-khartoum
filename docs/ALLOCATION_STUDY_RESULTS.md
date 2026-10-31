# Constrained allocation: what the comparison established

23 October 2026

**The Fairness Rule achieved the highest overall canonical reward among the four constrained
methods. The PPO Hybrid improved on Constrained Optimization, but did not improve on the
Fairness Rule.** Its predeclared success criterion required passing against both, so the
study does not demonstrate an overall hybrid advantage.

This follows the [frozen protocol](ALLOCATION_STUDY_PROTOCOL.md). Validation used 160
independent base draws, 480 coupled shortage conditions and 7,200 controller episodes.
It selected epsilon **0.005** for Fairness Rule and **0.0025** for PPO Hybrid. The separate
final test used 320 base draws, 960 conditions and 6,720 controller episodes. All seven
methods received matched 24-hour scenarios. See the complete [validation report](../data/evidence/allocation-validation.md)
and [final report](../data/evidence/allocation-study.md).

## Final outcomes

Numbers are equal-episode means. Interruption burden sums locality-hours below 99% critical
service; fairness is mean hourly Jain fairness of cumulative service ratios. Utilization
and switching also use the existing mean-hourly definitions.

| Method | Critical interruption, locality-h | Fairness % | Utilization % | Daily reward | Switching % |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed Rotation | 102.0625 | 87.7626 | 71.2315 | 13.97002 | 25.0000 |
| PPO | 19.9937 | 96.9181 | 97.6327 | 23.50701 | 22.1708 |
| Optimization | 6.5479 | 94.8863 | 98.2004 | 23.48261 | 18.8170 |
| Constrained Optimization | 0 | 94.6535 | 97.4264 | 23.44233 | 20.2896 |
| Fairness Rule | 0 | 96.0479 | 97.3581 | 23.48396 | 24.6540 |
| Feasible PPO | 0 | 96.4435 | 96.5352 | 23.46382 | 24.2566 |
| PPO Hybrid | 0 | 95.2042 | 97.3177 | 23.45862 | 21.3827 |

All four constrained methods had zero critical interruptions, zero zero-service blackouts
and zero power-balance violations. No scenario required the insufficient-critical-supply
fallback. This confirms behavior in these default synthetic conditions, not on a real grid.

## Paired differences and tradeoffs

Intervals are pointwise 95% paired bootstrap intervals from 10,000 resamples, retaining
the coupled shortage variants. Percentage-point differences are denoted pp.

| Comparison | Reward difference [95% CI] | Fairness difference, pp [95% CI] | Utilization difference, pp [95% CI] |
| --- | ---: | ---: | ---: |
| PPO Hybrid minus Constrained Optimization | +0.0163 [+0.0135, +0.0191] | +0.5508 [+0.4886, +0.6133] | -0.1087 [-0.1316, -0.0861] |
| PPO Hybrid minus Fairness Rule | -0.0253 [-0.0278, -0.0228] | -0.8437 [-0.9017, -0.7840] | -0.0405 [-0.0640, -0.0171] |
| Fairness Rule minus Constrained Optimization | +0.0416 [+0.0389, +0.0443] | +1.3944 [+1.3313, +1.4566] | -0.0683 [-0.0879, -0.0489] |

The hybrid passed every reward/service gate against Constrained Optimization. Against
Fairness Rule it failed the reward and fairness requirements, including shortage-specific
guardrails. Its Severe mean reward was slightly higher than Fairness Rule, by approximately
0.0019; that descriptive subgroup result does not overturn the overall finding.

Fairness Rule is not best on every metric or shortage level. It switches more often than
Constrained Optimization and PPO Hybrid. Feasible PPO has the highest overall Jain fairness
among constrained methods, but lower utilization; it also has the highest constrained
reward under Severe shortage. Fairness Rule gives the strongest average weakest-locality
final service: 50.7460%, versus 44.7881% for Constrained Optimization and 46.5280% for PPO Hybrid.

Raw PPO retains higher overall fairness and reward, alongside approximately 20 critical
interruption locality-hours per episode. Its different constraints and feature information
prevent treating that comparison as an isolated test of learned preferences.

Both structured rules are computationally practical: median complete decision calls for
32 scenarios were 1.294 ms for Constrained Optimization and 1.429 ms for Fairness Rule,
versus 1.827 ms for PPO Hybrid. These are local fixed-order timing observations.

**For the declared critical-service requirement and canonical score, the current evidence
favors the deterministic Fairness Rule.** No new training or learned future-value model
was tested. The result concerns one frozen actor with historical optimizer-imitation
initialization; it neither isolates PPO fine-tuning nor rules out future learning methods.

## Evidence and reproduction

The [final evidence](../data/evidence/allocation-study.json) binds the
[validation evidence](../data/evidence/allocation-validation.json), source files, model and
protocol. Both files passed structural and physical checks; all seven paired interval sets
were independently reconstructed on CUDA/CuPy. Recorded SHA-256 values:

- Validation: `292a3f734558a1de8c818e8ab2daec14b976ca5352b82b9bef398ca0cd30ca90`
- Final: `f41adb611ae3e323644d38abdc6244f26f17f0c029f567d4621c5969c1e31060`

From the repository root, reproduce into new paths; existing outputs are never overwritten:

```powershell
uv run python tools/evaluate_allocation_study.py --phase validation --output tmp/allocation-reproduction/validation.json
uv run python tools/evaluate_allocation_study.py --phase test --output tmp/allocation-reproduction/test.json --selection data/evidence/allocation-validation.json --selection-sha256 292a3f734558a1de8c818e8ab2daec14b976ca5352b82b9bef398ca0cd30ca90
```

Timing and output-reference metadata can change reproduced file hashes. Verify the published
evidence and reconstruct its intervals with:

```powershell
@'
from pathlib import Path
from gridguard.allocation_study import load_study_artifact, verify_gpu_intervals
root = Path.cwd()
payload = load_study_artifact(
    root / "data/evidence/allocation-study.json",
    expected_sha256="f41adb611ae3e323644d38abdc6244f26f17f0c029f567d4621c5969c1e31060",
    phase="test", root=root,
)
verify_gpu_intervals(payload)
print("Verified sources, selection, claims and paired intervals")
'@ | uv run python -
```

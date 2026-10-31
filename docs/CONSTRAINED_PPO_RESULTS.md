# Constrained PPO training: results against the Fairness Rule

23 October 2026

**Newly trained constrained PPO did not beat the Fairness Rule overall.** Across five
independent training runs, it improved reward and fairness over its own behavior-cloning
initialization, but the Fairness Rule retained higher average daily reward, slightly
higher Jain fairness and substantially less switching. All methods protected critical
service in every final-test condition.

This follows the [frozen protocol](CONSTRAINED_PPO_PROTOCOL.md),
[training record](../data/evidence/constrained-ppo-training.json) and
[final evidence](../data/evidence/constrained-ppo-study.json). It is a new training study;
the earlier [frozen-policy allocation study](ALLOCATION_STUDY_RESULTS.md) remains separate.
Earlier model weights, benchmark values and published evidence were preserved.

## What was trained and compared

Five independent runs each used 32,760 Fairness Rule demonstration states, ten imitation
epochs and 600 PPO iterations of 256 complete 24-hour episodes: **3,686,400 PPO transitions
per run, 18,432,000 total**. Separate actor/critic networks use two 256-unit tanh layers,
46 causal features and 128 masked Critical Only/Full Service choices.

Physics, reward and metrics stayed unchanged. Fairness Rule retained epsilon 0.005.
Validation used 480 conditions and 13 checkpoints per run. Final testing used **960 fresh
matched conditions**: 320 independent base draws with coupled Mild, Moderate and Severe
variants. Two deterministic methods, five PPO actors and five paired behavior-cloning
actors produced **11,520 trajectories**. There was no final-test selection or tuning.

## Overall outcomes

Learned-method rows equally average five runs. Fairness, utilization and switching retain
their existing mean-hourly definitions.

| Method | Daily reward | Jain fairness % | Utilization % | Switching % | Critical interruption, locality-h |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fairness Rule | **23.48759** | **96.1925** | 97.3541 | 25.2350 | 0 |
| Constrained Optimization | 23.44485 | 94.7485 | 97.4280 | **20.6851** | 0 |
| Constrained PPO, five-run mean | 23.44892 | 96.0311 | 97.3921 | 30.0470 | 0 |
| Behavior Cloning, five-run mean | 23.31276 | 93.0156 | **98.2551** | 33.2517 | 0 |

All 12 methods had zero critical interruptions, zero zero-service blackouts and zero
power-balance violations in every condition. Critical service was 100%; no critical-supply
fallback was needed. Zero interruption duration does **not** imply equal Full Service: Critical Only
supplies every locality while noncritical allocation differs. Mean final minimum locality
service was **50.9335% for Fairness Rule**, **46.1889% for PPO** and **43.3015% for imitation**.

## Uncertainty and the predeclared decision

These PPO-minus-comparator differences use 10,000 paired hierarchical bootstrap resamples
over training runs and matched base-draw clusters. Coupled severities and shared scenario
samples stay together. Intervals are pointwise 95%; pp means percentage points.

| Comparator | Daily reward difference [95% CI] | Fairness difference, pp [95% CI] | Utilization difference, pp [95% CI] |
| --- | ---: | ---: | ---: |
| Fairness Rule | -0.03867 [-0.04425, -0.03319] | -0.1615 [-0.2913, -0.0443] | +0.0379 [-0.0227, +0.0925] |
| Paired Behavior Cloning | +0.13616 [+0.12786, +0.14523] | +3.0155 [+2.8283, +3.2282] | -0.8630 [-0.9295, -0.8069] |

**The primary criterion failed:** reward, fairness noninferiority, shortage-specific
guardrails and validation eligibility failed. Safety and aggregate utilization
noninferiority passed. Every PPO run had lower mean reward than Fairness Rule.

PPO improved reward and fairness over imitation in every shortage level, but lost
utilization. Its full service-guarded criterion against imitation therefore also failed.

PPO switched **4.8120 pp** more than Fairness Rule [95% CI: +4.2192, +5.4774]. The reward
formula attributes -0.03208 of its -0.03867 reward gap to switching, -0.00861 to fairness
and +0.00202 to utilization. This explains the score difference, not its learning cause.

## All training runs and checkpoint status

No checkpoint in any run met every validation condition. Each run retained its highest
validation-reward checkpoint under the predeclared fallback rule, explicitly **ineligible**.

| Training seed | Selected iteration | PPO final reward | Behavior-cloning final reward | PPO fairness % | PPO utilization % |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1729 | 600 | 23.44271 | 23.31658 | 96.0342 | 97.4101 |
| 2718 | 550 | 23.45701 | 23.30628 | 96.2072 | 97.2874 |
| 3141 | 550 | 23.45044 | 23.31648 | 96.0592 | 97.4574 |
| 5772 | 550 | 23.44200 | 23.31262 | 95.7992 | 97.4480 |
| 8111 | 550 | 23.45244 | 23.31183 | 96.0555 | 97.3574 |

Seed 2718 is the validation-selected deployment candidate, also ineligible. It is not a
demonstrated improvement. The evidence retains all checkpoint records and per-run intervals.

## Differences by shortage level

These descriptive five-run subgroup differences do not overturn the primary result.

| Comparator | Shortage | Reward difference | Fairness difference, pp | Utilization difference, pp | Switching difference, pp |
| --- | --- | ---: | ---: | ---: | ---: |
| Fairness Rule | Mild | -0.06637 | -0.6969 | +0.3076 | +6.8408 |
| Fairness Rule | Moderate | -0.06436 | -0.4822 | -0.0533 | +5.3698 |
| Fairness Rule | Severe | +0.01472 | +0.6948 | -0.1406 | +2.2254 |
| Paired Behavior Cloning | Mild | +0.05193 | +1.1643 | -0.5923 | -3.2135 |
| Paired Behavior Cloning | Moderate | +0.14316 | +2.9149 | -1.0255 | -6.3586 |
| Paired Behavior Cloning | Severe | +0.21340 | +4.9672 | -0.9712 | -0.0420 |

PPO improved Severe reward and fairness, with less utilization and more switching.
Mild and Moderate losses outweighed that gain overall.

## Verification, cost and limits

Recorded training times total **483.679 seconds (8.06 minutes)** on the RTX 5060 Laptop
GPU; final evaluation took **28.257 seconds**, excluding later audit. No decision latency
was measured. CUDA verification reconstructed means, intervals and decisions and checked
source/model hashes, physical bounds, energy and duration identities, matching and reward.
The [replay audit](constrained-ppo-validation.json) additionally checked:

- 5,280 validation trajectories with zero summary reconstruction error;
- all 960 held-out input hashes and critical-supply shortfall counts;
- 1,152 held-out trajectories, covering all 12 methods on 96 conditions each, with exact
  action hashes and zero error in all 20 metrics and seven final locality service ratios.

Trajectory replay covers 10% of final conditions per method and uses frozen project
formulas. Source, model and evidence files remained unchanged.

Recorded SHA-256 values:

- Training record: `4ee4edbcec4b95f8fa53148a304fb2fb5aa03af136bef6544b9ae104b39ca071`
- Final study: `f2a82b7059eec389ae04bf849d225f4a6776bfb0ea42172688829f39c1d5696c`
- Replay audit: `d1242346aba9da40327b5921f546a6328ea1d03fe2f8c1174687253b26af8f07`

Verify the published evidence from the repository root:

```powershell
uv run python tools/evaluate_constrained_ppo.py --verify --training-sha256 4ee4edbcec4b95f8fa53148a304fb2fb5aa03af136bef6544b9ae104b39ca071 --sha256 f2a82b7059eec389ae04bf849d225f4a6776bfb0ea42172688829f39c1d5696c
```

**This experiment favors Fairness Rule.** Five runs and one training recipe cannot settle
every RL approach. These synthetic decision-support results do not establish real-grid
performance. The Severe gain and improvement over imitation remain valid scoped findings.

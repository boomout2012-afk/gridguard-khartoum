# Constrained allocation study final matched comparison

23 October 2026. Synthetic decision-support simulation, seven localities, 24 hours.

Evidence SHA-256: `f41adb611ae3e323644d38abdc6244f26f17f0c029f567d4621c5969c1e31060`. Settings: `{'Fairness Rule': 0.005, 'PPO Hybrid': 0.0025}`.

Jain fairness and utilization are mean hourly ratios. Critical interruptions sum locality-hours below 99% critical service. All numbers below are equal-episode means.

| Method | Critical interruption (locality-h) | Jain fairness (%) | Utilization (%) | Daily reward | Switching (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed Rotation | 102.0625 | 87.7626 | 71.2315 | 13.97002 | 25.0000 |
| PPO | 19.9937 | 96.9181 | 97.6327 | 23.50701 | 22.1708 |
| Optimization | 6.5479 | 94.8863 | 98.2004 | 23.48261 | 18.8170 |
| Constrained Optimization | 0.0000 | 94.6535 | 97.4264 | 23.44233 | 20.2896 |
| Fairness Rule | 0.0000 | 96.0479 | 97.3581 | 23.48396 | 24.6540 |
| Feasible PPO | 0.0000 | 96.4435 | 96.5352 | 23.46382 | 24.2566 |
| PPO Hybrid | 0.0000 | 95.2042 | 97.3177 | 23.45862 | 21.3827 |

## Primary decision

The hybrid does not pass the predeclared reward and service guardrails against both constrained comparators. Report individual effects as gains, ties or tradeoffs.

| Comparison | Reward delta [95% CI] | Fairness delta pp [95% CI] | Utilization delta pp [95% CI] |
| --- | ---: | ---: | ---: |
| PPO Hybrid - Constrained Optimization | +0.0163 [+0.0135, +0.0191] | +0.5508 [+0.4886, +0.6133] | -0.1087 [-0.1316, -0.0861] |
| PPO Hybrid - Fairness Rule | -0.0253 [-0.0278, -0.0228] | -0.8437 [-0.9017, -0.7840] | -0.0405 [-0.0640, -0.0171] |
| PPO Hybrid - Feasible PPO | -0.0052 [-0.0078, -0.0026] | -1.2393 [-1.2998, -1.1798] | +0.7825 [+0.7556, +0.8102] |
| Constrained Optimization - PPO | -0.0647 [-0.0689, -0.0604] | -2.2647 [-2.3529, -2.1781] | -0.2063 [-0.2428, -0.1694] |
| Fairness Rule - Constrained Optimization | +0.0416 [+0.0389, +0.0443] | +1.3944 [+1.3313, +1.4566] | -0.0683 [-0.0879, -0.0489] |
| Feasible PPO - Constrained Optimization | +0.0215 [+0.0184, +0.0246] | +1.7901 [+1.7222, +1.8597] | -0.8912 [-0.9219, -0.8611] |
| PPO Hybrid - PPO | -0.0484 [-0.0522, -0.0447] | -1.7139 [-1.7908, -1.6367] | -0.3150 [-0.3532, -0.2777] |

## Mild shortage

| Method | Critical interruption (locality-h) | Jain fairness (%) | Utilization (%) | Daily reward |
| --- | ---: | ---: | ---: | ---: |
| Fixed Rotation | 96.0000 | 87.7436 | 54.2701 | 13.19153 |
| PPO | 12.6219 | 97.6816 | 98.0106 | 23.61721 |
| Optimization | 2.4500 | 96.4055 | 98.4351 | 23.60539 |
| Constrained Optimization | 0.0000 | 96.3993 | 98.0890 | 23.58811 |
| Fairness Rule | 0.0000 | 97.9028 | 98.1609 | 23.63230 |
| Feasible PPO | 0.0000 | 97.7270 | 97.4009 | 23.59651 |
| PPO Hybrid | 0.0000 | 96.9518 | 97.9864 | 23.60311 |

## Moderate shortage

| Method | Critical interruption (locality-h) | Jain fairness (%) | Utilization (%) | Daily reward |
| --- | ---: | ---: | ---: | ---: |
| Fixed Rotation | 96.0000 | 87.7436 | 68.6846 | 13.96030 |
| PPO | 26.1406 | 96.8331 | 98.2835 | 23.50193 |
| Optimization | 9.8000 | 95.1671 | 98.9259 | 23.50631 |
| Constrained Optimization | 0.0000 | 94.8101 | 98.0119 | 23.45367 |
| Fairness Rule | 0.0000 | 96.7942 | 97.8642 | 23.51224 |
| Feasible PPO | 0.0000 | 96.6241 | 96.9852 | 23.45852 |
| PPO Hybrid | 0.0000 | 95.3247 | 97.8365 | 23.46349 |

## Severe shortage

| Method | Critical interruption (locality-h) | Jain fairness (%) | Utilization (%) | Daily reward |
| --- | ---: | ---: | ---: | ---: |
| Fixed Rotation | 114.1875 | 87.8007 | 90.7397 | 14.75822 |
| PPO | 21.2188 | 96.2397 | 96.6039 | 23.40188 |
| Optimization | 7.3938 | 93.0863 | 97.2400 | 23.33612 |
| Constrained Optimization | 0.0000 | 92.7509 | 96.1783 | 23.28520 |
| Fairness Rule | 0.0000 | 93.4467 | 96.0493 | 23.30735 |
| Feasible PPO | 0.0000 | 94.9795 | 95.2195 | 23.33644 |
| PPO Hybrid | 0.0000 | 93.3362 | 96.1301 | 23.30926 |

## Interruption and locality diagnostics

Weakest locality is the minimum final served-energy/demand ratio within each episode, then averaged across episodes. It differs from mean hourly Jain fairness. Critical duration gap is the within-episode range across localities. The evidence also retains all locality durations and streaks.

| Method | Blackout locality-h | Critical duration gap h | Longest critical interruption h | Weakest locality service % | Power violation % |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed Rotation | 96.0000 | 3.6667 | 8.0000 | 36.0268 | 0.0000 |
| PPO | 0.0000 | 0.0000 | 1.3552 | 49.2529 | 0.0000 |
| Optimization | 0.0000 | 0.0000 | 0.6708 | 43.9595 | 0.0000 |
| Constrained Optimization | 0.0000 | 0.0000 | 0.0000 | 44.7881 | 0.0000 |
| Fairness Rule | 0.0000 | 0.0000 | 0.0000 | 50.7460 | 0.0000 |
| Feasible PPO | 0.0000 | 0.0000 | 0.0000 | 48.6730 | 0.0000 |
| PPO Hybrid | 0.0000 | 0.0000 | 0.0000 | 46.5280 | 0.0000 |

Critical demand exceeded available supply for 0 scenario-hours across the matched conditions. Each constrained method uses the all-Critical-Only fallback in exactly those hours.

## Controller timing

synchronized wall time for the entire Controller.act call, batch size 32; first 24 calls retained separately; excludes model loading, simulation and UI; one local session in fixed controller order, not a formal latency benchmark

| Method | Median batch ms | p95 batch ms |
| --- | ---: | ---: |
| Fixed Rotation | 0.106 | 0.191 |
| PPO | 0.201 | 0.316 |
| Optimization | 5.249 | 6.519 |
| Constrained Optimization | 1.294 | 1.730 |
| Fairness Rule | 1.429 | 1.834 |
| Feasible PPO | 1.742 | 2.129 |
| PPO Hybrid | 1.827 | 2.478 |

## Scope and interpretation

- Synthetic seven-locality decision support; no real electrical-network validation.
- One frozen PPO actor; no retraining or learned future value in this experiment.
- Raw PPO has different feature information; constrained filters share exact causal state.
- Historical PPO optimizer-imitation initialization prevents attribution to PPO alone.
- Pointwise scenario intervals conditional on this actor; not training-run variability.
- A result for this hybrid does not settle all RL or lookahead approaches.

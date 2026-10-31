# Constrained allocation study validation

23 October 2026. Synthetic decision-support simulation, seven localities, 24 hours.

Evidence SHA-256: `292a3f734558a1de8c818e8ab2daec14b976ca5352b82b9bef398ca0cd30ca90`. Settings: `{'Fairness Rule': 0.005, 'PPO Hybrid': 0.0025}`.

Jain fairness and utilization are mean hourly ratios. Critical interruptions sum locality-hours below 99% critical service. All numbers below are equal-episode means.

| Method | Critical interruption (locality-h) | Jain fairness (%) | Utilization (%) | Daily reward | Switching (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Constrained Optimization | 0.0000 | 94.6039 | 97.4282 | 23.43970 | 20.3013 |
| Fairness Rule epsilon=0 | 0.0000 | 94.6039 | 97.4282 | 23.43970 | 20.3013 |
| Fairness Rule epsilon=0.001 | 0.0000 | 94.9224 | 97.4223 | 23.45308 | 20.7949 |
| Fairness Rule epsilon=0.0025 | 0.0000 | 95.4410 | 97.4142 | 23.47112 | 22.1739 |
| Fairness Rule epsilon=0.005 | 0.0000 | 96.0200 | 97.3499 | 23.48240 | 24.5995 |
| Fairness Rule epsilon=0.01 | 0.0000 | 96.9196 | 97.0803 | 23.45723 | 33.4139 |
| Fairness Rule epsilon=0.02 | 0.0000 | 98.1664 | 95.9236 | 23.36871 | 47.4132 |
| Fairness Rule epsilon=0.05 | 0.0000 | 98.4661 | 93.7287 | 23.18621 | 59.6267 |
| PPO Hybrid epsilon=0 | 0.0000 | 94.6039 | 97.4282 | 23.43970 | 20.3013 |
| PPO Hybrid epsilon=0.001 | 0.0000 | 94.8258 | 97.4086 | 23.44708 | 20.8135 |
| PPO Hybrid epsilon=0.0025 | 0.0000 | 95.1924 | 97.3064 | 23.45704 | 21.4335 |
| PPO Hybrid epsilon=0.005 | 0.0000 | 95.6453 | 97.0201 | 23.46012 | 22.3053 |
| PPO Hybrid epsilon=0.01 | 0.0000 | 96.1854 | 96.7021 | 23.46330 | 23.6049 |
| PPO Hybrid epsilon=0.02 | 0.0000 | 96.3709 | 96.4994 | 23.45815 | 24.2398 |
| PPO Hybrid epsilon=0.05 | 0.0000 | 96.3860 | 96.4772 | 23.45746 | 24.2870 |

## Scope and interpretation

- Synthetic seven-locality decision support; no real electrical-network validation.
- One frozen PPO actor; no retraining or learned future value in this experiment.
- Raw PPO has different feature information; constrained filters share exact causal state.
- Historical PPO optimizer-imitation initialization prevents attribution to PPO alone.
- Pointwise scenario intervals conditional on this actor; not training-run variability.
- A result for this hybrid does not settle all RL or lookahead approaches.

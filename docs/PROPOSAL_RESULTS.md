# Proposal extension: measured results

Results compiled on 22 October 2026 from the separate, validated proposal supplement. The protocol in `docs/PROPOSAL_EXTENSION.md` was declared in the development workspace before the complete evaluation, with unfavorable and inconclusive outcomes retained. This is not external preregistration or independent custody.

Evidence: `data/evidence/proposal-supplement.json`; SHA-256 `fae5c875b1be148d7a069259b19df9b04666fe97e2182ce8898f45fed7b0a43e`. The loader verifies this pinned identity, schema, matched inputs, counts, duration identities, means, paired deltas, and forecast sufficient statistics before this chapter is generated. Interval endpoints are read from the validated frozen artifact. Displayed numbers are rounded to four decimal places; differences are calculated before rounding.

The new evaluation uses seeds 91001-91010, 32 base trajectories per seed, and a 24-hour horizon. The 320 independent base trajectories each generate coupled Mild, Moderate, and Severe conditions: 960 matched conditions and 2,880 controller-scenario runs. These results do not replace the historical accepted reproduction or the earlier 81001-81010 holdout. All data remain synthetic and support a research decision-support demonstration.

## Overall controller outcomes

Artifact storage clarification: scenario inputs are reproducible from the pinned generator,
seeds and protocol; the serialized artifact records their SHA-256 digests rather than raw demand
and supply arrays. Per-target forecast predictions exist during CUDA evaluation. The saved
artifact retains fitted parameters and per-seed count/absolute-error/squared-error statistics,
not those full prediction arrays. All three sets of controller confidence intervals have also
been independently reconstructed from the saved metric cube on CUDA.

Each entry is an equal-episode mean over the 960 matched conditions. Critical outage counts locality-hours below 99% of critical demand. Fairness is the existing Jain index of cumulative service ratios, not a direct measure of blackout duration. Lower outage and higher fairness/utilization are favorable for their respective outcomes.

| Outcome | Fixed Rotation | PPO | Optimization |
| --- | ---: | ---: | ---: |
| Critical outage (locality-hours) | 102.0719 | 19.8552 | 6.7521 |
| Jain service fairness (%) | 87.7584 | 96.8922 | 94.8924 |
| Supply utilization (%) | 71.2448 | 97.6224 | 98.2407 |

Optimization has the lowest critical-outage burden and highest utilization in this experiment. PPO has the highest Jain service fairness. Thus the stronger comparator reveals a tradeoff rather than general PPO superiority. Optimization exhaustively evaluates all 2,187 joint tier assignments against the canonical current-hour objective; it is not a 24-hour optimum. Its explicit causal state differs from PPO's fixed 32-value observation.

| Supporting diagnostic | Fixed Rotation | PPO | Optimization |
| --- | ---: | ---: | ---: |
| Critical service (%) | 42.6250 | 99.7038 | 99.8701 |
| Power-balance violation rate | 0.0000 | 0.0000 | 0.0000 |
| Canonical episode reward | 13.9698 | 23.5058 | 23.4838 |

The reward remains an objective diagnostic. Maximizing a current-hour objective does not guarantee the largest accumulated episode reward or every public outcome.

## Direct interruption duration and disparity

A zero-service blackout requires positive total demand and exactly zero delivered power. A critical interruption requires positive critical demand and service below 99%. The latter sums to the canonical critical-outage outcome. Duration range and the mean absolute gap across 21 locality pairs are calculated within each episode before averaging. Longest interruption is the maximum consecutive duration across localities within an episode, without wrapping its boundaries.

| Duration outcome | Fixed Rotation | PPO | Optimization |
| --- | ---: | ---: | ---: |
| Zero-service blackout (locality-hours) | 96.0000 | 0.0000 | 0.0000 |
| Critical interruption (locality-hours) | 102.0719 | 19.8552 | 6.7521 |
| Blackout duration range (hours) | 3.0000 | 0.0000 | 0.0000 |
| Blackout mean pairwise gap (hours) | 1.7143 | 0.0000 | 0.0000 |
| Critical duration range (hours) | 3.6667 | 0.0000 | 0.0000 |
| Critical mean pairwise gap (hours) | 1.9025 | 0.0000 | 0.0000 |
| Longest blackout (hours) | 6.0000 | 0.0000 | 0.0000 |
| Longest critical interruption (hours) | 8.0000 | 1.3573 | 0.6969 |

PPO and Optimization have zero zero-service blackout hours and zero duration gaps in this sample. They still have critical interruptions. Under the simulator's global proportional supply clipping, all seven localities fall below the critical-service threshold at the same affected hours for these two controllers. Their critical interruption durations are therefore equal across localities, even though their total burdens differ. The zero disparity is partly structural and cannot distinguish their performance here. Equal duration alone never establishes adequate service.

## Outcomes by shortage level

Each shortage/controller row averages 320 conditions; the three shortage variants reuse the same base trajectories. Percentage columns are on a 0-100 scale.

| Shortage | Controller | Critical outage (locality-hours) | Jain fairness (%) | Utilization (%) |
| --- | --- | ---: | ---: | ---: |
| Mild | Fixed Rotation | 96.0000 | 87.7395 | 54.2830 |
| Mild | PPO | 12.5781 | 97.6741 | 98.0257 |
| Mild | Optimization | 2.6250 | 96.2779 | 98.4827 |
| Moderate | Fixed Rotation | 96.0000 | 87.7395 | 68.7025 |
| Moderate | PPO | 25.9219 | 96.8097 | 98.2398 |
| Moderate | Optimization | 9.9094 | 95.2357 | 98.9097 |
| Severe | Fixed Rotation | 114.2156 | 87.7962 | 90.7489 |
| Severe | PPO | 21.0656 | 96.1928 | 96.6017 |
| Severe | Optimization | 7.7219 | 93.1636 | 97.3296 |

| Shortage | Controller | Blackout burden (locality-hours) | Blackout range (h) | Blackout pairwise gap (h) | Longest blackout (h) |
| --- | --- | ---: | ---: | ---: | ---: |
| Mild | Fixed Rotation | 96.0000 | 3.0000 | 1.7143 | 6.0000 |
| Mild | PPO | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Mild | Optimization | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Moderate | Fixed Rotation | 96.0000 | 3.0000 | 1.7143 | 6.0000 |
| Moderate | PPO | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Moderate | Optimization | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Severe | Fixed Rotation | 96.0000 | 3.0000 | 1.7143 | 6.0000 |
| Severe | PPO | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Severe | Optimization | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

| Shortage | Controller | Critical burden (locality-hours) | Critical range (h) | Critical pairwise gap (h) | Longest critical interruption (h) |
| --- | --- | ---: | ---: | ---: | ---: |
| Mild | Fixed Rotation | 96.0000 | 3.0000 | 1.7143 | 6.0000 |
| Mild | PPO | 12.5781 | 0.0000 | 0.0000 | 0.9875 |
| Mild | Optimization | 2.6250 | 0.0000 | 0.0000 | 0.3375 |
| Moderate | Fixed Rotation | 96.0000 | 3.0000 | 1.7143 | 6.0000 |
| Moderate | PPO | 25.9219 | 0.0000 | 0.0000 | 1.5844 |
| Moderate | Optimization | 9.9094 | 0.0000 | 0.0000 | 0.9281 |
| Severe | Fixed Rotation | 114.2156 | 5.0000 | 2.2789 | 12.0000 |
| Severe | PPO | 21.0656 | 0.0000 | 0.0000 | 1.5000 |
| Severe | Optimization | 7.7219 | 0.0000 | 0.0000 | 0.8250 |

The same primary-outcome tradeoff occurs at all three shortage levels: Optimization has lower critical outage and higher utilization, while PPO has higher Jain fairness. Outage counts need not change monotonically with shortage severity because the controllers make discrete service-tier choices and the outcome applies a strict critical-service threshold.

## Paired effects and pointwise uncertainty

Subtraction order is explicit in each heading. These are paired differences over matched conditions. Intervals use 10,000 seed-stratified bootstrap resamples (bootstrap seed 20261022), clustering each base trajectory with all three shortage variants and all controllers. The 95% percentile intervals are pointwise and unadjusted for multiple comparisons; no family-wide significance claim is made. Fairness and utilization effects are percentage points (pp). Other effects retain the units shown. Degenerate intervals reflect identical observed paired effects, not certainty outside this synthetic design.

### PPO - Fixed Rotation

| Outcome | Mean difference | 95% lower | 95% upper |
| --- | ---: | ---: | ---: |
| Critical outage (locality-hours) | -82.2167 | -82.9136 | -81.5271 |
| Jain service fairness (pp) | 9.1337 | 9.0995 | 9.1688 |
| Supply utilization (pp) | 26.3776 | 26.3259 | 26.4289 |
| Zero-service blackout (locality-hours) | -96.0000 | -96.0000 | -96.0000 |
| Critical interruption (locality-hours) | -82.2167 | -82.9136 | -81.5271 |
| Blackout duration range (hours) | -3.0000 | -3.0000 | -3.0000 |
| Blackout mean pairwise gap (hours) | -1.7143 | -1.7143 | -1.7143 |
| Critical duration range (hours) | -3.6667 | -3.6667 | -3.6667 |
| Critical mean pairwise gap (hours) | -1.9025 | -1.9033 | -1.9016 |
| Longest blackout (hours) | -6.0000 | -6.0000 | -6.0000 |
| Longest critical interruption (hours) | -6.6427 | -6.6844 | -6.6000 |

### Optimization - Fixed Rotation

| Outcome | Mean difference | 95% lower | 95% upper |
| --- | ---: | ---: | ---: |
| Critical outage (locality-hours) | -95.3198 | -95.7240 | -94.9125 |
| Jain service fairness (pp) | 7.1340 | 7.0523 | 7.2142 |
| Supply utilization (pp) | 26.9959 | 26.9450 | 27.0462 |
| Zero-service blackout (locality-hours) | -96.0000 | -96.0000 | -96.0000 |
| Critical interruption (locality-hours) | -95.3198 | -95.7240 | -94.9125 |
| Blackout duration range (hours) | -3.0000 | -3.0000 | -3.0000 |
| Blackout mean pairwise gap (hours) | -1.7143 | -1.7143 | -1.7143 |
| Critical duration range (hours) | -3.6667 | -3.6667 | -3.6667 |
| Critical mean pairwise gap (hours) | -1.9025 | -1.9033 | -1.9016 |
| Longest blackout (hours) | -6.0000 | -6.0000 | -6.0000 |
| Longest critical interruption (hours) | -7.3031 | -7.3406 | -7.2646 |

### PPO - Optimization

| Outcome | Mean difference | 95% lower | 95% upper |
| --- | ---: | ---: | ---: |
| Critical outage (locality-hours) | 13.1031 | 12.3229 | 13.8615 |
| Jain service fairness (pp) | 1.9998 | 1.9253 | 2.0753 |
| Supply utilization (pp) | -0.6183 | -0.6524 | -0.5851 |
| Zero-service blackout (locality-hours) | 0.0000 | 0.0000 | 0.0000 |
| Critical interruption (locality-hours) | 13.1031 | 12.3229 | 13.8615 |
| Blackout duration range (hours) | 0.0000 | 0.0000 | 0.0000 |
| Blackout mean pairwise gap (hours) | 0.0000 | 0.0000 | 0.0000 |
| Critical duration range (hours) | 0.0000 | 0.0000 | 0.0000 |
| Critical mean pairwise gap (hours) | 0.0000 | 0.0000 | 0.0000 |
| Longest blackout (hours) | 0.0000 | 0.0000 | 0.0000 |
| Longest critical interruption (hours) | 0.6604 | 0.6062 | 0.7156 |

## Locality duration means

All entries below are hours per locality, averaged across the 960 conditions. Off hours describe requested actions and are retained as a diagnostic. Locality means are not used to recompute the within-episode disparity outcomes.

### Fixed Rotation

| Locality | Blackout | Critical interruption | Off action | Longest blackout | Longest critical interruption |
| --- | ---: | ---: | ---: | ---: | ---: |
| Khartoum | 12.0000 | 12.3573 | 12.0000 | 6.0000 | 6.3333 |
| Omdurman | 12.0000 | 13.0240 | 12.0000 | 6.0000 | 6.0094 |
| Bahri | 12.0000 | 13.0000 | 12.0000 | 6.0000 | 6.0000 |
| East Nile | 15.0000 | 15.6667 | 15.0000 | 6.0000 | 6.6667 |
| Jebel Awlia | 15.0000 | 16.0000 | 15.0000 | 6.0000 | 8.0000 |
| Karari | 15.0000 | 16.0000 | 15.0000 | 6.0000 | 7.0000 |
| Ombada | 15.0000 | 16.0240 | 15.0000 | 6.0000 | 7.0000 |

### PPO

| Locality | Blackout | Critical interruption | Off action | Longest blackout | Longest critical interruption |
| --- | ---: | ---: | ---: | ---: | ---: |
| Khartoum | 0.0000 | 2.8365 | 0.0000 | 0.0000 | 1.3573 |
| Omdurman | 0.0000 | 2.8365 | 0.0000 | 0.0000 | 1.3573 |
| Bahri | 0.0000 | 2.8365 | 0.0000 | 0.0000 | 1.3573 |
| East Nile | 0.0000 | 2.8365 | 0.0000 | 0.0000 | 1.3573 |
| Jebel Awlia | 0.0000 | 2.8365 | 0.0000 | 0.0000 | 1.3573 |
| Karari | 0.0000 | 2.8365 | 0.0000 | 0.0000 | 1.3573 |
| Ombada | 0.0000 | 2.8365 | 0.0000 | 0.0000 | 1.3573 |

### Optimization

| Locality | Blackout | Critical interruption | Off action | Longest blackout | Longest critical interruption |
| --- | ---: | ---: | ---: | ---: | ---: |
| Khartoum | 0.0000 | 0.9646 | 0.0000 | 0.0000 | 0.6969 |
| Omdurman | 0.0000 | 0.9646 | 0.0000 | 0.0000 | 0.6969 |
| Bahri | 0.0000 | 0.9646 | 0.0000 | 0.0000 | 0.6969 |
| East Nile | 0.0000 | 0.9646 | 0.0000 | 0.0000 | 0.6969 |
| Jebel Awlia | 0.0000 | 0.9646 | 0.0000 | 0.0000 | 0.6969 |
| Karari | 0.0000 | 0.9646 | 0.0000 | 0.0000 | 0.6969 |
| Ombada | 0.0000 | 0.9646 | 0.0000 | 0.0000 | 0.6969 |

## Learned causal forecast results

Ridge AR(3) was fitted once on 768 newly generated training traces: seeds 42, 43, and 44 with 256 traces each. The 35 regressions use only demand at t-3, t-2, and t-1, training-only normalization, an unpenalized intercept, fixed ridge penalty 0.001, and non-negative predictions. Coefficients and normalization parameters are stored in the supplement. Evaluation uses the 320 seed-disjoint unique demand traces from 91001-91010, once each, and target hours 3-23 for every method. No evaluation values are used for fitting or hyperparameter selection.

| Method | Locality-sector MAE (MW) | Locality-sector RMSE (MW) | System-total MAE (MW) | System-total RMSE (MW) |
| --- | ---: | ---: | ---: | ---: |
| Three-hour trailing mean | 2.3385 | 3.7436 | 42.7512 | 49.9759 |
| One-hour persistence | 1.3465 | 2.1776 | 24.3809 | 28.5917 |
| Ridge AR(3) | 0.6573 | 1.1882 | 9.3376 | 11.9242 |

Each method contributes 235,200 locality-sector and 6,720 system-total predictions. System error sums signed cell residuals before taking absolute value or squaring. MAE and RMSE are reconstructed from pooled counts, absolute-error sums, and squared-error sums; no confidence interval is asserted for this descriptive forecast comparison. Ridge AR(3) has the lowest error on all four reported measures. This improves the selected synthetic prediction comparison but does not establish real-demand calibration or benefits to control: all forecasts remain separate from controller decisions, and the existing display forecast is unchanged.

## Proposal coverage and remaining limits

The extension supplies explicit optimization and learned prediction comparisons for objective 5 and direct blackout/critical-interruption duration disparities for research question 4. It retains evidence unfavorable to PPO: Optimization performs better on critical-outage burden and utilization. The direct metrics also reveal a simulator-induced tie that requires careful interpretation.

The study remains synthetic, shortage-only, and without representative utility measurements or electrical-network validation. The one-step Optimization comparator does not establish a 24-hour scheduling optimum. There is no behavior-cloning-only ablation to isolate incremental PPO fine-tuning. Seed separation and declared protocols do not establish independent custody. The application remains an offline research decision-support demonstration, not an operational grid controller.

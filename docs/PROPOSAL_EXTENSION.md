# Proposal extension: comparison methods and interruption durations

Protocol declaration: 22 October 2026, before the complete extension evaluation.

This chapter defines an additional scientific experiment for GridGuard Khartoum. It addresses
the accepted proposal's objective 5 through an explicit dispatch-optimization comparison and a
learned prediction model, and research question 4 through direct measures of interruption
duration and disparity between localities. It supplements the existing report and evidence.
The frozen PPO actor, historical accepted reproduction, and documented-seed-disjoint robustness
artifact retain their original identities and results.

The protocol is declared in the development workspace before inspecting the full set of new
evaluation outcomes. This is not an external preregistration or independent custody claim.
Unit tests and small correctness checks are implementation validation, not the full experiment.
The experiment retains every reported outcome, including unfavorable and inconclusive results.
No favorable result is required for artifact creation, verification, or reporting.

## Fixed evaluation design

| Element | Declared design |
| --- | --- |
| Evaluation seeds | 91001 through 91010, inclusive |
| Independent base trajectories | 32 trajectories per seed, 320 in total |
| Horizon | 24 one-hour samples per trajectory |
| Geographic units and sectors | Seven localities and five service sectors |
| Shortage levels | Mild, Moderate, Severe |
| Coupling | Reuse each severity-neutral base draw across all three shortage levels |
| Matched scenario conditions | 320 base trajectories x 3 shortage levels = 960 |
| Controllers | Fixed Rotation, PPO, Optimization |
| Controller-scenario evaluations | 960 matched conditions x 3 controllers = 2,880 |
| Learned forecast training | Seeds 42, 43, 44, with 256 new synthetic traces per seed |
| Forecast evaluation inputs | The 320 unique demand traces, without duplicating shortage variants |
| Forecast targets | One hour ahead, target hours 3 through 23, inclusive |

Each controller receives the same demand and available-supply trace for a matched condition.
The three shortage variants of one base trajectory are dependent conditions, not three
independent draws. Demand does not change with shortage severity in this locked generator, so
forecasting evaluates each of the 320 demand traces once.

The new evaluation seeds are separate from the documented PPO training, validation, and
selection seeds, the historical accepted-reproduction seeds, the earlier 81001-81010 holdout,
and the new forecasting fit. Seed separation refers to documented activity and cannot rule out
undocumented prior use. All scenarios remain synthetic.

## Controller methods

**Fixed Rotation** retains the existing fixed three-hour schedule. Its actions are Off or Full
Service. It does not observe demand or fairness and cannot select Critical Only.

**PPO** retains the existing frozen actor and its 32-value observation. It selects one of the
2,187 joint locality actions across Off, Critical Only, and Full Service. The extension does not
retrain the actor or change its weights.

**Optimization** is a discrete, exhaustive, one-step comparator. At each hour it enumerates all
3^7 = 2,187 joint service-tier assignments and scores each assignment with the same canonical
current-hour reward and proportional dispatch rule used by the simulator. It receives current
demand, current supply, cumulative demand and delivered service, and the previous service tiers.
It receives no future demand or supply. Requests exceeding supply undergo the same proportional
clipping and canonical oversubscription penalty as other controllers.

Optimization selects the assignment with the largest canonical one-step score. A tie selects
the lowest action ID in the shared little-endian base-three locality encoding. Candidate and
scenario chunks bound memory usage without removing any assignment from the search. The
scoring, dispatch, reductions, and selection run on CUDA in the primary evaluation path.

The Optimization comparator is exact for its stated current-hour objective. It does not solve
the complete 24-hour scheduling problem, anticipate future shortages, or establish an optimum
for any public episode metric. Its richer explicit causal state also differs from the PPO
actor's compressed observation. These differences belong to the method definition and limit
causal attribution. Comparing these controllers does not isolate the incremental effect of
PPO fine-tuning after its historical initialization.

## Direct duration definitions

Duration measurements use actual delivered power from the validated simulation trace. A
one-hour sample represents one complete modeled hour. These metrics do not estimate sub-hour
interruptions. Let d_i be the duration for locality i within one 24-hour episode, for i = 1..7.

**Zero-service blackout:** a locality has total demand above 1e-8 MW and receives exactly zero
total delivered power during that hour. Partial service is not a zero-service blackout. An Off
action alone does not establish a blackout when there is no positive demand.

**Critical-service interruption:** a locality has critical demand above 1e-8 MW and receives
less than 99% of that demand. Critical demand comprises Hospital, Water, and Bakery. This uses
the existing strict 0.99 service threshold. Summing critical-interruption durations over the
seven localities gives the existing critical-outage locality-hours outcome.

**Duration range:** max(d_i) - min(d_i), in hours.

**Mean pairwise duration gap:** sum of abs(d_i - d_j) over the 21 distinct unordered locality
pairs, divided by 21, in hours. Diagonal self-pairs do not enter the denominator.

**Longest interruption:** the longest consecutive run of affected one-hour samples. Streaks
stop at the episode boundaries. Hour 23 does not connect to hour 0. Each locality has a longest
streak, and the episode summary is the maximum of those seven locality streaks.

| Field | Unit | Aggregation and interpretation |
| --- | --- | --- |
| `locality_blackout_hours` | hours | Zero-service duration for each of the seven localities |
| `locality_critical_interruption_hours` | hours | Critical-service interruption duration for each locality |
| `locality_off_hours` | hours | Requested Off actions for each locality, an action diagnostic |
| `locality_longest_blackout_hours` | hours | Longest zero-service streak for each locality |
| `locality_longest_critical_interruption_hours` | hours | Longest critical-service interruption streak for each locality |
| `blackout_locality_hours` | locality-hours | Sum of zero-service durations across the seven localities |
| `critical_interruption_locality_hours` | locality-hours | Sum of critical-service interruption durations across localities |
| `blackout_duration_range_hours` | hours | Range of the seven locality blackout durations |
| `blackout_duration_mean_pairwise_gap_hours` | hours | Mean absolute gap over 21 pairs of blackout durations |
| `critical_interruption_duration_range_hours` | hours | Range of the seven locality critical-interruption durations |
| `critical_interruption_duration_mean_pairwise_gap_hours` | hours | Mean absolute gap over 21 pairs of critical-interruption durations |
| `longest_blackout_hours` | hours | Maximum locality blackout streak within the episode |
| `longest_critical_interruption_hours` | hours | Maximum locality critical-interruption streak within the episode |

Range, pairwise gaps, and longest streaks are computed within each episode before averaging
over episodes. Disparity is not computed from already-averaged locality durations. Lower
duration disparity means more equal durations, not necessarily better service. Every locality
losing service for 24 hours would have zero disparity. Duration disparity therefore accompanies
interruption burden, critical service, utilization, and power-balance checks. Jain fairness
remains the existing cumulative-service-ratio metric and is not renamed as blackout-duration
fairness. All new duration fields are reported, including ties and zero-valued results.

## Learned prediction experiment

The learned method is ridge autoregression with three lags, labeled **Ridge AR(3)**. It fits one
regression for each of the 35 locality-sector combinations. For target hour t, the three
features are demand at t-3, t-2, and t-1, in chronological order. The model receives no target or
future demand, hour-of-day features, supply, controller actions, or scenario identity.

The training cube contains 768 newly generated synthetic traces: 256 for each documented
candidate seed 42, 43, and 44. This is a new forecasting fit using the current scenario generator,
not a reconstruction of the historical PPO training trajectories. Each regression uses 16,128
training target samples: 768 traces x 21 target hours.

Feature means, population standard deviations, and target means are calculated from the
training cube only. Feature scales have a fixed floor of 1e-8 MW. Each regression centers its
target, standardizes its three features, and solves the float64 ridge normal equations on CUDA.
The objective is mean squared residual plus lambda times the squared coefficient norm, with
lambda fixed at 0.001 before the full evaluation. Centering leaves the intercept unpenalized.
The prediction restores the training target mean and clamps negative predictions to zero.

The fitted coefficients, feature means, feature scales, target means, training seed labels,
training trace count, and ridge penalty are retained for reproducibility. Fitting and inference
remain on CUDA. Evaluation data neither update these parameters nor select hyperparameters.
The model API rejects overlapping training and evaluation seed labels.

All three methods predict the same target hours 3..23 on the same 320 unique evaluation traces:

| Method | Information used for a target at hour t |
| --- | --- |
| Three-hour trailing mean | Arithmetic mean of demand at t-3, t-2, t-1 |
| One-hour persistence | Demand at t-1, the latest value in the available history |
| Ridge AR(3) | Fixed learned linear combination of demand at t-3, t-2, t-1 plus intercept |

Forecast evaluation retains per-target predictions and count, sum absolute error, and sum
squared error for each method and evaluation seed. The signed residual is predicted demand
minus target demand. Locality-sector metrics pool all 35 cell residuals over traces and target
hours. System-total metrics sum the 35 signed cell residuals for a target before taking absolute
value or squaring. Each method therefore has 235,200 locality-sector predictions and 6,720
system-total predictions. Pooled MAE is total absolute error divided by total count. Pooled RMSE
is the square root of total squared error divided by total count. Both errors are in MW.

The forecast experiment remains separate from controller decisions and the existing display
projection. Lower synthetic prediction error would not establish operational calibration or
improved controller performance. An unfavorable learned-model result is retained.

## Paired uncertainty and reporting

Controller effects use matched within-condition differences. The report identifies the
subtraction order in every comparison, including PPO minus Fixed Rotation, Optimization minus
Fixed Rotation, and PPO minus Optimization. Existing critical-outage locality-hours, Jain
fairness, and utilization retain their canonical definitions. Direct duration measures add
explicit interruption and disparity outcomes. Power-balance violations remain a separate safety
invariant, and the canonical reward remains an objective diagnostic rather than a new primary
outcome.

Paired controller intervals use 10,000 deterministic, seed-stratified bootstrap resamples. Each
resample draws base-trajectory indices with replacement within each evaluation seed. All three
shortage variants and every controller for a selected base trajectory stay coupled. The same
selected indices form both sides of each effect. Resampling localities, severity conditions,
or individual controller runs independently would break the matched design and is excluded.

The interval endpoints are the 2.5th and 97.5th percentiles of the bootstrap effect distribution.
These are descriptive pointwise 95% intervals, unadjusted for multiple comparisons. They do not
support a family-wide significance claim. Aggregate estimates give equal weight to episodes
and the equally sized shortage-by-seed cells. Results are presented overall and by shortage
level so the aggregate cannot conceal a shortage-specific tradeoff.

The new evidence records the protocol, scenario inputs, controller identities, fitted forecast
parameters, per-episode metrics, paired inference, software/runtime identities, and source
hashes. Simulation, inference, duration metrics, forecast calculations, and bootstrap remain on
CUDA. Scalar validation flags may synchronize. Numeric evidence arrays undergo one final bulk
host transfer for portable artifact assembly and validation.

The new artifact is a separate proposal-extension supplement. Its values never replace the
historical accepted-reproduction or earlier robustness tables, and live demonstration values
never replace any frozen evidence. The extension makes no claim about a real electrical grid,
electrical-network feasibility, representative utility measurements, or general superiority
over all scheduling or forecasting methods.

## Implementation and acceptance evidence

| Concern | Implementation | Focused verification |
| --- | --- | --- |
| Exact one-step dispatch comparator | `src/gridguard/optimization.py` and shared dispatch/reward state | `tests/test_optimization.py` and `tests/test_proposal_integration.py` |
| Direct duration and disparity measures | `src/gridguard/duration_metrics.py` | `tests/test_duration_metrics.py` |
| Learned causal forecasting | `src/gridguard/forecast_model.py` | `tests/test_forecast_model.py` |
| Existing display forecast | `src/gridguard/forecast.py`, unchanged | Existing causal forecast tests |
| Historical evidence | Existing hash-bound evidence and model files, unchanged | Existing integrity verification |

The completed experiment must pass focused scientific tests, Ruff checks, real CUDA execution,
and evidence reconstruction before numerical conclusions are added to the updated thesis and
presentation. Final academic artifacts require visual inspection of every page and slide.

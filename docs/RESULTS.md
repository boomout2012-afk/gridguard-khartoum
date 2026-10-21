# Results

## Result statement

The reviewer-facing result is the documented-seed-disjoint holdout on seeds 81001-81010. It
contains 320 independent base draws crossed with Mild, Moderate, and Severe shortage-only
conditions: 960 matched scenario conditions and 1,920 controller-scenario evaluations. Under this
synthetic protocol, PPO has lower critical-outage locality-hours, higher Jain fairness, and higher
utilization than Fixed Rotation. Both controllers record zero power-balance violations.

All three deterministic 95% seed-stratified paired cluster intervals lie in the direction favoring
PPO. That direction is an observed result, not a release criterion: the artifact verifier is
sign-neutral and would accept a well-formed favorable, adverse, or ambiguous result. The finding
applies to one frozen PPO actor, one supply-blind Fixed Rotation baseline, and the stated synthetic
generator. It is decision-support evidence, not a real-grid operational claim.

The seeds are disjoint only from every documented training, validation, and model-selection seed in
`data/model/training-provenance.yaml`. The holdout was not preregistered or independently custodied
and cannot rule out undocumented activity. No behavior-cloning-only ablation was run,
so this comparison cannot isolate the incremental effect of PPO fine-tuning.

## Results provenance and evidence roles

The project keeps historical parity, documented seed disjointness, and training lineage distinct.

| Evidence role | File and SHA-256 | Permitted interpretation |
|---|---|---|
| Accepted aggregate source | `data/evidence/frozen-benchmark.json` - `3d720ba87a747a5780119decd7144b4051fc36fc72ab50f229f5750ad749fd15` | Preserve the accepted 18 batch-cell summaries; it contains no historical episode rows or intervals |
| Accepted reproduction evidence | `data/evidence/paired-benchmark.json` - `7f2009c4f7efbbedf098069ced9218addec51d55a18b969e6c54433027020af5` | Reproduce 96 independent base draws, 288 matched scenario conditions, 576 controller evaluations, paired rows, and frozen parity; seeds `50001-50003` participated in model selection, so this is not an independently custodied final test |
| Documented-seed-disjoint holdout | `data/evidence/robustness-supplement.json` - `6b56d149331ee191302cd9e6f22365988db0ae301f9441190b61214acf832e48` | Report the 320-draw primary comparison, 24 sensitivity variants, safety invariant, and synthetic forecast evaluation; disjointness covers documented activity only and does not imply preregistration or independent custody |
| Historical PPO training record | `data/model/training-provenance.yaml` - `d2568bce26e9221afa8c266113a704f6fbf2370d3bc949f5aa6c72a7411ba1a8` | Distinguish the selected seed-43 run from the three-seed candidate experiment; scope hardware/convergence to the selected run; expose promotion checks, provenance categories, unavailable historical facts, and source lineage; this is not a bitwise-retraining bundle and historical gates are not independent headline evidence |

`validate_paired_benchmark_payload` reconstructs closed schema, pairing, physical bounds, and
arithmetic. `assert_paired_benchmark_integrity` adds locked provenance and direct frozen-cell
parity. The robustness verifier independently reconstructs protocol counts, paired means,
difference-in-differences, forecast sufficient-statistic bounds, and source bindings. These are
sign-neutral integrity paths; no favorable-result release gate exists.

Full numeric precision remains in canonical JSON. Human-facing prose, tables, UI, and slides use
one decimal place for MWh and locality-hours, two decimals for percentages, and three decimals for
the Training-objective score (diagnostic).

The display geography has independent provenance and no evidentiary role. OCHA/HDX administrative
boundaries are pinned at SHA-256
`df1afbb337047402feb18d2dd26273c671c36761970d0c48c591c62f3ae3d496`; Natural Earth Nile
linework is pinned at
`0486a0f7318b5d7c576c13093b27f3f7aa6410329a7c54fb66eb37d3436cdafa`. Both are presentation
context only and not electrical-network topology. The map does not enter scenario generation,
controller input, dispatch, metrics, or evidence.

## Documented-seed-disjoint primary outcomes

Every difference is `PPO - Fixed Rotation`. Percentage-point differences are shown for ratio
outcomes. Values are rounded only for presentation.

| Primary outcome | Fixed Rotation | PPO | Mean paired difference | Deterministic 95% interval | Direction |
|---|---:|---:|---:|---:|---|
| Critical-outage locality-hours | 102.1 | 20.2 | -81.9 | [-82.5, -81.2] | Favorable |
| Jain fairness | 87.76% | 96.91% | +9.15 percentage points | [+9.11, +9.19] | Favorable |
| Utilization | 71.21% | 97.61% | +26.39 percentage points | [+26.34, +26.45] | Favorable |

Critical-outage locality-hours can range from 0 to 168 per episode. It counts each locality-hour in
which critical-sector service is below the explicit 99% threshold. Jain fairness is the mean of
hourly fairness over cumulative locality service ratios. Utilization is the mean hourly served
power divided by `min(available supply, demand)`. These definitions are not interchangeable with
the reward components or with feeder-level outage and stability measures.

The safety result is separate from the primary table. Maximum power-balance violation rate is
0.00% across the base comparison and all 24 sensitivity variants for both controllers under the
0.001 MW simulator tolerance. This verifies an implemented conservation invariant; it is not a
claim about voltage, frequency, protection, or real-grid safety.

## Accepted reproduction context

The accepted reproduction remains useful because it exactly preserves the prior committee-facing
comparison while exposing its evidence role. It has 96 independent base draws, 288 matched
scenario conditions, and 576 controller-scenario evaluations. Its legacy numeric projection is
unchanged after reward reframing.

| Outcome | Fixed Rotation | PPO | Public role |
|---|---:|---:|---|
| Training-objective score (diagnostic) | 13.969 | 23.512 | Diagnostic; shares components with reported outcomes |
| Critical-service ratio | 42.63% | 99.72% | Supporting |
| Critical-outage locality-hours | 102.1 | 18.7 | Primary |
| Jain fairness | 87.76% | 96.92% | Primary |
| Utilization | 71.22% | 97.65% | Primary |
| Served energy | 4,551.7 MWh | 6,560.2 MWh | Supporting |
| Curtailed energy | 6,142.4 MWh | 4,133.9 MWh | Complementary energy identity |
| Switching rate | 25.00% | 21.89% | Secondary |
| Power-balance violation rate | 0.00% | 0.00% | Safety |

The accepted critical-outage paired difference is -83.4 locality-hours with a deterministic 95%
interval of [-84.6, -82.2]. The Training-objective score difference is +9.543, but it is not an
independent primary outcome: the objective directly includes critical service, utilization,
fairness, switching, and oversubscription. Served and curtailed energy are likewise complementary
under the episode energy identity and are not two independent endpoints.

The accepted cell summaries still match the frozen aggregate artifact within every locked
tolerance. That parity proves source-compatible reproduction; it does not make the three reused
seeds independently custodied.

## Sensitivity across illustrative demand allocation

The one-factor-at-a-time analysis changes each of seven locality weights and five sector shares by
-25% and +25%, renormalizing the affected vector. The 24 variants plus the unchanged base give 25
parameterizations. Each variant reuses the same 320 base draws across all three shortage levels
and both controllers while holding the base available-supply trace exactly fixed, producing 24,000
matched sensitivity scenarios and 48,000 controller-scenario evaluations when the base is included.

All 144 sensitivity intervals are descriptive, pointwise 95% intervals and are unadjusted for
multiple comparisons. They do not establish family-wide statistical significance.

For every one of the 24 variants, the paired interval remains favorable for PPO on all three
primary outcomes. Difference-in-differences asks a different question: whether the perturbed
paired effect is larger or smaller than the base paired effect. Those classifications are mixed:

| Difference-in-differences outcome | Favorable | Ambiguous | Adverse |
|---|---:|---:|---:|
| Critical-outage locality-hours | 4 | 1 | 19 |
| Jain fairness | 6 | 9 | 9 |
| Utilization | 6 | 5 | 13 |

Thus, the direction of the controller comparison persists in this local +/-25% neighborhood, but
many perturbations reduce the size of the base advantage. The study supports bounded local
sensitivity, not parameter invariance, interaction coverage, global robustness, or calibration to
Khartoum.

## Synthetic forecast comparison

Both display methods are evaluated on the same 320 documented-seed-disjoint synthetic demand traces and common
target hours 3-23. The three-hour trailing mean uses only hours `t-3` through `t-1`; the one-hour
persistence baseline uses `t-1`. Each method has 235,200 locality-sector predictions and 6,720
system-total predictions. Errors are in MW at each one-hour target; locality-sector statistics pool
episode x target x 7 x 5 absolute cell errors, while system-total error is calculated after summing
all 35 locality-sector residuals at a target. Squared-error sums use MW squared, MAE uses MW, and
RMSE uses MW.

| Aggregation | Method | MAE | RMSE | Lower-error method |
|---|---|---:|---:|---|
| Locality-sector MW | Three-hour trailing mean | 2.338 | 3.745 | Persistence |
| Locality-sector MW | One-hour persistence | 1.346 | 2.178 | Persistence |
| System-total MW | Three-hour trailing mean | 42.725 | 49.973 | Persistence |
| System-total MW | One-hour persistence | 24.377 | 28.584 | Persistence |

Persistence has lower error on all four stored measures. Trailing-minus-persistence differences
are +0.992 MW locality-sector MAE, +1.567 MW locality-sector RMSE, +18.348 MW system MAE, and
+21.390 MW system RMSE. This is a useful negative result: the more complex display projection does
not beat the simple baseline under this synthetic generator. It remains display-only and is never
PPO input. No real-world forecast-accuracy claim follows.

## Baseline interpretation

Fixed Rotation serves three consecutive localities at Full Service for each three-hour block and
turns the other four Off. It does not observe supply, demand, critical need, or accumulated
fairness; it cannot select Critical Only. In the accepted reproduction its critical-service ratios
are 43.27%, 43.27%, and 41.35%, while utilization rises from 54.26% to 68.68% to 90.73% across
Mild, Moderate, and Severe shortages. The utilization rise is a denominator effect as usable
supply falls, not evidence that the schedule adapts.

Mean unused available energy is 2,158.7 MWh for Fixed Rotation versus 150.2 MWh for PPO. Part of
the utilization difference is therefore attributable to the supply-blind baseline design, which
uses only Off and Full Service. The evidence supports comparison with this proposal baseline only. It does not establish
PPO superiority over a supply-aware critical-preserving heuristic or every rule-based controller.

## Interpretation against the proposal questions

- **Critical services:** PPO reduces the modeled critical-outage burden relative to Fixed Rotation,
  but proportional dispatch does not implement feeder-level within-hour hospital priority.
- **Fairness:** PPO raises mean hourly Jain fairness of cumulative locality service ratios in the
  locked synthetic generator.
- **Distribution efficiency:** PPO raises utilization; supporting served energy and complementary
  curtailment explain the energy allocation without being counted as extra primary outcomes.
- **Prediction:** one-hour persistence beats the display-only trailing mean on all stored synthetic
  MAE/RMSE measures.
- **Data and rules needed for realism (RQ3):** a field-facing model would need feeder/locality
  demand, independent supply and generation constraints, network topology, critical-facility
  loads and priorities, outage/restoration histories, operating and switching rules,
  weather/calendar effects, and validated asset-to-locality mappings.
- **Optimization:** Adam optimized PPO parameters during historical training; CUDA accelerates
  computation. The desktop implements no separate online mathematical dispatch optimizer.

## Limitations of the result

- The scenarios are synthetic, Khartoum-inspired, and uncalibrated. Supply is constructed partly
  from synthetic demand, so matched scenarios do not remove shared generator misspecification.
- Fixed Rotation is supply-blind and limited to Off and Full Service. A stronger supply-aware baseline
  is outside the accepted scope but necessary future work.
- Performance relative to the global optimum is unknown; no valid upper-bound comparator is
  included.
- When requests exceed supply, one proportional factor scales critical and non-critical sectors
  together. Critical-service outcomes are not feeder-level restoration evidence.
- The actor enumerates `3^N` actions (`3^7 = 2,187` here), limiting scalability beyond seven
  localities.
- The operating domain is shortage-only: Mild, Moderate, and Severe. Surplus, normal operation,
  recovery, and restoration transients are not evaluated.
- Bootstrap intervals are conditional on the fixed synthetic seed strata and shortage design.
  Ten documented-seed-disjoint seeds improve separation and coverage but were not preregistered or
  independently custodied and do not represent all future conditions.
- Sensitivity intervals are pointwise and unadjusted for multiple comparisons; no family-wide
  significance claim is supported.
- No behavior-cloning-only ablation was run, so the incremental contribution of PPO fine-tuning
  beyond the initialization is not isolated.
- The +/-25% one-factor sensitivity excludes interactions and cannot replace calibration data.
- Forecast errors use synthetic traces only. Persistence's lower error does not establish a
  real-world forecasting conclusion.
- Power-balance compliance is a simulator invariant, not evidence of real-grid stability or
  operational safety.
- Administrative boundaries and Nile linework are presentation context only; they contribute no
  scientific input.

# Results

## Result statement

Under the locked synthetic evaluation design, PPO outperformed Fixed Rotation on all six
primary outcomes: episode reward, critical-service ratio, utilization, Jain fairness, served
energy, and curtailed energy. The newly reproduced per-episode artifact crosses 96 independent
base draws with three shortage levels, producing 288 matched scenario conditions and 576
controller-scenario evaluations. For every primary outcome, the seed-stratified paired cluster
95% percentile interval lies wholly in the direction favoring PPO.

This is a simulation result, not a claim about real utility operation. It applies to one frozen
PPO actor, one Fixed Rotation baseline, and the stated synthetic shortage design.

## Two evidence layers

The project retains two complementary artifacts and does not treat them as interchangeable.

| Evidence layer | File and SHA-256 | Granularity | Permitted use |
|---|---|---|---|
| Accepted aggregate evidence | `data/evidence/frozen-benchmark.json` - `3d720ba87a747a5780119decd7144b4051fc36fc72ab50f229f5750ad749fd15` | 18 controller batch-cell summaries: 2 controllers x 3 shortage levels x 3 seeds; each summary represents 32 episodes | Preserve and reproduce the accepted aggregate comparison; no confidence intervals, p-values, or episode win rates |
| Reproduced paired evidence | `data/evidence/paired-benchmark.json` - `2c3e71e48c4c7bff4fe609d42a76ca4f770f6dd717cd7d45d5e8e0d99fdf2d11` | 288 scenario-condition rows from 96 base-draw clusters, each containing both controller outcomes and their paired difference, plus 9 cell summaries | Aggregate results, matched differences, deterministic seed-stratified paired cluster intervals, and critical-outage locality-hours |

The accepted artifact was migrated without inventing episode records. The new artifact was then
regenerated from the pinned simulator and PPO actor, retained every episode pair, and passed a
cell-level parity comparison against the accepted summaries.

## Accepted aggregate comparison

The following values are the exact numeric values stored in the accepted aggregate artifact.
They are means per 24-hour synthetic episode, not sums across 288 episodes. Ratios are shown as
stored; multiply by 100 only when presenting them as percentages.

| Metric | Fixed Rotation | PPO | Direction / role |
|---|---:|---:|---|
| Episode reward | 13.969324111938477 | 23.512135399712456 | Higher / primary |
| Critical-service ratio | 0.4262974626488156 | 0.9971638056966994 | Higher / primary |
| Utilization | 0.7122341394424438 | 0.9765213396814134 | Higher / primary |
| Jain fairness | 0.8775761524836222 | 0.9691991872257657 | Higher / primary |
| Served energy (MWh) | 4551.691569010417 | 6560.189181857639 | Higher / primary |
| Curtailed energy (MWh) | 6142.390950520833 | 4133.893283420139 | Lower / primary |
| Switching rate | 0.2500000298023224 | 0.21887401905324724 | Lower / secondary |
| Oversubscription | 0.005967141646477912 | 0.0017833372629765007 | Lower / diagnostic |
| Power-balance violation rate | 0 | 0 | Zero / safety |
| Demand energy (MWh) | 10694.082356770834 | 10694.082356770834 | Matched input |
| Available energy (MWh) | 6710.389431423611 | 6710.389431423611 | Matched input |

The source stored only these cell-derived aggregates, so inferential statistics are intentionally
absent from this evidence layer.

## Reproduced paired aggregate comparison

The following are the exact aggregate values stored in the new per-episode paired artifact.
They are means per 24-hour synthetic episode. Small differences from the preceding table
reflect reproduced float32 reduction order and, for critical service, episode-ratio averaging
rather than pooled energy. All are within the locked parity tolerances shown below.

| Metric | Fixed Rotation | PPO | Direction / role |
|---|---:|---:|---|
| Episode reward | 13.969324383470747 | 23.512135558658176 | Higher / primary |
| Critical-service ratio | 0.4262975813407037 | 0.997163764304585 | Higher / primary |
| Utilization | 0.712234134061469 | 0.976521334093478 | Higher / primary |
| Jain fairness | 0.8775761361337371 | 0.9691991855700811 | Higher / primary |
| Served energy (MWh) | 4551.691485934787 | 6560.189144558377 | Higher / primary |
| Curtailed energy (MWh) | 6142.390977647569 | 4133.893338521321 | Lower / primary |
| Switching rate | 0.2500000298023224 | 0.21887402122633326 | Lower / secondary |
| Oversubscription | 0.005967141698218054 | 0.0017833372334684504 | Lower / diagnostic |
| Power-balance violation rate | 0 | 0 | Zero / safety |
| Critical outage locality-hours | 102.07291666666667 | 18.666666666666668 | Lower / diagnostic |
| Demand energy (MWh) | 10694.08246866862 | 10694.08246866862 | Matched input |
| Available energy (MWh) | 6710.389377170139 | 6710.389377170139 | Matched input |

The aggregate critical-service ratio corresponds to approximately 42.63% for Fixed Rotation
and 99.72% for PPO. PPO also reduced mean critical outage exposure from approximately 102.07 to
18.67 locality-hours per episode. Critical outage is a diagnostic threshold measure and was not
included in the bootstrap interval set.

Both controllers recorded an aggregate power-balance violation rate of exactly zero in the
paired artifact. This confirms the implemented simulator invariant for these runs; it is not a
real-grid stability claim.

## Paired differences and 95% intervals

Each difference below is exactly `PPO - Fixed Rotation`. The table reproduces the stored mean
differences and deterministic 95% stratified paired percentile intervals. Values are copied
from the pinned paired artifact without re-rounding.

| Metric | Mean difference | 95% lower | 95% upper | Favorable sign |
|---|---:|---:|---:|---|
| Episode reward | 9.54281117518743 | 9.53561696095599 | 9.550073716955053 | Positive |
| Critical-service ratio | 0.5708661829638811 | 0.570613363581813 | 0.5711048985391648 | Positive |
| Utilization | 0.26428720003200906 | 0.2632496295703782 | 0.2652864054621508 | Positive |
| Jain fairness | 0.09162304943634403 | 0.09091882564437886 | 0.09230119484062825 | Positive |
| Served energy (MWh) | 2008.4976586235891 | 2000.3781833224828 | 2016.7392950693768 | Positive |
| Curtailed energy (MWh) | -2008.4976391262478 | -2016.7392876095244 | -2000.3781435224746 | Negative |
| Switching rate | -0.031126008575989142 | -0.03478424262462391 | -0.0274884424974314 | Negative |

All seven intervals exclude zero in the favorable direction. This statement is descriptive of
the locked, seed-stratified cluster bootstrap distribution; no p-value or multiplicity-adjusted
hypothesis test was calculated. In presentation terms, the ratio differences correspond to about
+57.09 percentage points in critical service, +26.43 percentage points in utilization, +9.16
percentage points in Jain fairness, and -3.11 percentage points in switching.

Served and curtailed energy are complementary under the energy identity. Their paired results
describe the same allocation change with opposite signs and should not be interpreted as two
independent endpoints.

## Consistency across the nine evaluation cells

Each shortage-level and seed combination is a separate 32-episode cell. PPO had the favorable
cell mean in all 9 of 9 cells for every primary outcome. The observed cell-mean difference
ranges were:

| Primary outcome | Favorable cells | Minimum cell difference | Maximum cell difference |
|---|---:|---:|---:|
| Episode reward | 9 / 9 | 8.649745643138885 | 10.428759038448334 |
| Critical-service ratio | 9 / 9 | 0.5634181443601847 | 0.5836229948326945 |
| Utilization | 9 / 9 | 0.05924975872039795 | 0.43822693079710007 |
| Jain fairness | 9 / 9 | 0.082805709913373 | 0.09917639568448067 |
| Served energy (MWh) | 9 / 9 | 296.61602783203125 | 3757.100311279297 |
| Curtailed energy (MWh) | 9 / 9 | -3757.100296020508 | -296.6159973144531 |

The secondary switching-rate cell mean favored PPO in 7 of 9 cells even though its overall
paired interval favored PPO. Oversubscription was lower in aggregate for PPO but its direction
was mixed across shortage levels and cells: PPO was lower in 3 cells and higher in 6. For that
reason, oversubscription remains diagnostic and is not presented as a consistent primary advantage.
Power-balance violations were zero in every cell for both controllers.

## Parity with the accepted aggregate evidence

Before the per-episode results were serialized, every reproduced controller cell summary was
compared with its accepted counterpart. The comparison covered 18 summaries and used fixed,
code-locked absolute release tolerances. All metrics passed.

| Metric | Maximum absolute cell error | Allowed absolute tolerance |
|---|---:|---:|
| Available energy (MWh) | 0.00048828125 | 0.002 |
| Critical-service ratio | 0.0000012684613466262817 | 0.000002 |
| Curtailed energy (MWh) | 0.000274658203125 | 0.002 |
| Demand energy (MWh) | 0.00091552734375 | 0.002 |
| Episode reward | 0.0000011920928955078125 | 0.000002 |
| Jain fairness | 8.754432201385498e-8 | 0.000002 |
| Oversubscription | 8.149072527885437e-10 | 0.000002 |
| Power-balance violation rate | 0 | 0.000002 |
| Served energy (MWh) | 0.0006103515625 | 0.002 |
| Switching rate | 8.847564458847046e-9 | 0.000002 |
| Utilization | 8.754432201385498e-8 | 0.000002 |

Critical outage locality-hours is absent from the accepted aggregate artifact, so it has no
frozen parity row. It is calculated from the newly retained per-episode traces using the locked
99% service threshold.

For critical-service ratio specifically, the accepted source cells used pooled critical energy,
whereas the paired artifact calculates episode ratios before averaging. The recorded maximum
cell difference of `0.0000012684613466262817` remains below the locked tolerance. The paired
rows are therefore a compatible reproduction, not reconstructed historical episode records.

## Interpretation against the proposal questions

- **Critical services:** PPO delivered nearly all modeled critical-sector demand and sharply
  reduced critical outage locality-hours relative to Fixed Rotation.
- **Fairness:** PPO increased the mean hourly Jain fairness of cumulative locality service
  ratios.
- **Distribution efficiency:** PPO increased utilization and served energy while reducing
  curtailed energy.
- **Operational consistency inside the model:** both controllers respected the simulator's
  supply constraint, and PPO had a lower aggregate switching rate.
- **Traditional schedule comparison:** the matched design isolates controller behavior from
  scenario differences because both controllers receive identical demand and supply tensors.

These findings satisfy the proposal-level software demonstration and comparison objectives.
They do not establish national-grid optimization, field readiness, or performance under real
utility data.

## Limitations of the result

- The scenarios are synthetic, use fixed demand shares and hourly profiles, and contain no
  measurements from a utility.
- The simulator uses one abstract supply pool and omits electrical-network constraints and
  sub-hour dynamics.
- The paired intervals resample 96 base-draw clusters inside three fixed seed strata while
  keeping each cluster's three shortage variants together. They do not describe uncertainty
  across all possible seeds, shortage prevalence, or future grid conditions.
- Only PPO and Fixed Rotation are included in the confirmatory comparison.
- Results describe one frozen PPO actor under deterministic argmax inference.
- The reward weights and 99% critical-outage threshold are modeling choices.
- The display-only demand projection is not part of this evidence and has no reported accuracy.
- The offline locality map is schematic and contributes no scientific input to the evaluation.

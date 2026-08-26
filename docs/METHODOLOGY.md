# Methodology

## Study objective and claim boundary

GridGuard Khartoum evaluates whether a trained PPO controller can allocate limited
electricity more effectively than a traditional Fixed Rotation schedule in a simplified,
software-only model of Khartoum State. The accepted proposal [1] defines this as a
decision-support simulation, not a controller for a real electricity network. Accordingly,
the evidence supports a comparison inside the stated synthetic scenario design only.

The controller comparison is PPO versus Fixed Rotation. Evidence has two roles. Seeds
`50001-50003` reproduce the accepted historical comparison but participated in checkpoint
promotion and model selection, so that artifact is labeled **accepted reproduction**, not an
untouched final test. Seeds `81001-81010` are disjoint from every documented training, validation,
selection, and earlier evaluation seed and provide the bounded **untouched confirmatory**
comparison. The proposal permits prediction, rule-based logic,
optimization, and reinforcement learning where appropriate; it does not require every method to
be a controller in the final comparison. The presentation includes a small causal demand
projection as a separate explanatory feature. It is not an input to PPO and does not change
dispatch.

## Simulation domain

The model uses seven Khartoum State localities in a fixed tensor order: Khartoum, Omdurman,
Bahri, East Nile, Jebel Awlia, Karari, and Ombada. Each locality contains five demand sectors:
Hospital, Water, Bakery, Residential, and Commercial. Hospital, Water, and Bakery are the
three critical sectors.

Each episode contains 24 one-hour steps. Demand and supply are expressed as MW at each step;
energy totals are the sum over the one-hour steps and are therefore reported in MWh. The
simulator uses one global supply pool. It does not model feeders, voltages, frequencies,
transmission constraints, or power flow.

The presentation locates those seven units using second-level administrative boundaries from
OCHA/HDX Sudan COD-AB revision v03 [7], with White Nile, Blue Nile, and Nile linework from Natural
Earth 1:10m rivers [8]. These verified layers are presentation context only. They are not
electrical-network topology and do not enter scenario generation, controller observations,
dispatch, reward, metrics, or evidence.

### Synthetic demand

The system peak parameter is 700 MW. Locality peak shares are proportional to the locked
weights below. Locked sector coefficients define the base peak allocation before distinct
hourly profiles are applied, so instantaneous sector proportions can vary by hour.

| Locality | Relative weight |
|---|---:|
| Khartoum | 1.00 |
| Omdurman | 0.95 |
| Bahri | 0.85 |
| East Nile | 0.80 |
| Jebel Awlia | 0.70 |
| Karari | 0.75 |
| Ombada | 0.90 |

| Sector | Base peak coefficient | Profile assumption |
|---|---:|---|
| Hospital | 0.08 | Nearly constant through the day |
| Water | 0.07 | Morning and evening peaks |
| Bakery | 0.05 | Early-morning and afternoon peaks |
| Residential | 0.55 | Morning and stronger evening peaks |
| Commercial | 0.25 | Midday peak |

The sector curves are deterministic normalized profiles built from constant and Gaussian
components. Each generated episode then receives independent uniform demand perturbations in
the interval [-5%, +5%]. The perturbations are produced by a seeded PyTorch generator on the
selected device.

### Synthetic supply and shortage levels

Available supply is current total demand multiplied by a shortage fraction and an independent
uniform perturbation in [-5%, +5%]. It is capped at 95% of demand so every episode remains a
shortage case.

| Shortage level | Base supply fraction | Additional shortage window |
|---|---:|---|
| Mild | 0.80 | None |
| Moderate | 0.65 | Subtract 0.10 during simulation hours 17-20 |
| Severe | 0.50 | Subtract 0.15 during simulation hours 16-21 |

The same seed reproduces the same batched demand and supply tensors in the pinned CUDA
environment. Accepted reproduction uses seeds 50001-50003; the untouched confirmatory protocol
uses the separate seeds 81001-81010.

The simulator parameters are reproducible, but reproducibility is not calibration. Their
provenance classification is:

| Parameter family | Classification | Basis and limitation |
|---|---|---|
| Seven locality names | Proposal-constrained | Defines the accepted case-study vocabulary; names do not supply electrical data |
| OCHA/HDX boundaries and Natural Earth rivers | Sourced | Presentation context only; not topology, demand, or controller input |
| 700 MW peak, locality weights, and sector shares | Illustrative | Locked research assumptions, not utility measurements |
| Hourly sector profiles and +/-5% demand noise | Illustrative/derived | Deterministic profile construction plus seeded synthetic variation |
| Mild, Moderate, and Severe shortage fractions/windows | Proposal-constrained/illustrative | Reproducible shortage-only test conditions, not historical generation traces |
| Supply tied to total demand and capped below demand | Illustrative | Creates controlled shortages but a structural demand-supply relationship |
| PPO actor and Fixed Rotation schedule | Sourced/derived | Hash-bound accepted archive lineage and audited clean implementation |

A realistic calibration would require feeder/locality demand, independent generation and supply
constraints, electrical-network topology, critical-facility loads and priorities, outage and
restoration histories, operating and switching rules, weather/calendar effects, and validated
asset-to-locality mappings. Matched scenarios isolate controller differences within this generator;
they do not remove misspecification shared by both controllers. The scientific case is therefore
synthetic and Khartoum-inspired.

Shortage variants are intentionally coupled. For each seed, the generator first creates one
severity-neutral `ScenarioBaseDraws` batch containing demand-uniform and supply-uniform tensors.
That base batch is reused unchanged for Mild, Moderate, and Severe; a pure transformation applies
only the level's shortage fraction and additional shortage window. For a fixed
`(seed, batch_index)` pair, demand perturbations and the supply-noise draw are therefore identical
across all three variants. The design contains 96 base-draw clusters, each observed under three
shortage levels, rather than 288 independent random draws.

## Controller decisions and dispatch

At every hour, each controller chooses one service tier for each locality:

- Off: request no load.
- Critical Only: request Hospital, Water, and Bakery load.
- Full Service: request all five sectors.

The simulator concatenates the requested critical and non-critical loads. If total requested
power exceeds available supply, one proportional scale factor is applied to every requested
load. This guarantees that simulated served power does not exceed the global supply pool, but it
does not add a second within-hour critical-first dispatch stage. Critical priority affects the
selected tier, not feeder-level hospital restoration after oversubscription.

### Fixed Rotation

Fixed Rotation divides the day into three-hour blocks. In each block, three consecutive
localities receive Full Service and the remaining localities are Off. The starting locality
advances by three positions at each block and wraps around the seven-locality order. The rule
does not observe current demand, supply, critical-service need, or cumulative fairness.
It is supply-blind, uses only Off and Full Service, and never selects Critical Only. This makes it
transparent and proposal-aligned but weaker than a supply-aware critical-preserving heuristic;
the comparison must not be generalized to every reasonable rule-based policy.

In the accepted reproduction, its critical-service ratios are 43.27%, 43.27%, and 41.35% under
Mild, Moderate, and Severe shortages; utilization is 54.26%, 68.68%, and 90.73%. The rise in
utilization as shortage worsens is a denominator effect because utilization divides by available
usable power, not evidence that the schedule adapts. Mean unused available energy is 2,158.7 MWh
for Fixed Rotation versus 150.2 MWh for PPO, so part of the utilization contrast is attributable
to the baseline's design.

### PPO

PPO [2] receives a causal 32-value observation at each hour:

- four global values: sine and cosine of hour, supply divided by demand, and supply divided by
  the 700 MW system peak; and
- four values for each of seven localities: normalized current demand, current critical-sector
  share, cumulative service ratio, and previous service tier divided by two.

The inference-only actor has two 512-unit tanh layers and 2,187 output logits. The 2,187
outputs represent all `3^7` joint locality actions. Deterministic argmax inference selects one
joint action, which is decoded into seven service tiers using the pinned little-endian base-3
encoding. The actor is loaded from a hash-verified, pickle-free NumPy artifact and runs on
`cuda:0` in inference mode.

This release reproduces the frozen actor's architecture, weights, actions, and evaluation. It
does not include optimizer state and does not claim bitwise reproduction of the original training
run. The joint-action head grows as `3^N`: the accepted seven-locality actor has `3^7 = 2,187`
outputs, so the architecture does not scale gracefully to a much larger locality count.

### Historical PPO training record

The strict, model-manifest-bound record is `data/model/training-provenance.yaml`, SHA-256
`2cfa158a4b26596fd1f155c57fa79a88dc1c41ce8df946792c559184a815128a`. It records the 32-value
input, two 512-unit tanh layers, separate actor/critic encoders, 2,187 action logits, optimizer
settings, workload, seeds, validation, selection, hardware, convergence, and six archive source
bindings. Behavior-cloning initialization supplied the starting PPO actor from 32,768 teacher
samples; it is initialization, not a third controller.

PPO then ran 600 iterations with 256 complete 24-hour episodes per iteration: 153,600 episodes
and 3,686,400 transitions. Each iteration used four epochs, 2,048-sample minibatches, 12 optimizer
updates, `gamma=0.99`, `GAE lambda=0.95`, learning rate `0.0001`, clip epsilon `0.2`, value
coefficient `0.5`, normalized entropy coefficient `0.001`, and separate actor/critic maximum
gradient norms of `0.5`. Adam optimized historical PPO parameters; the selected seed-43 checkpoint
was iteration 510 after 6,120 updates. CUDA accelerated computation. No separate online
mathematical dispatch optimizer runs in the desktop or confirmatory comparison.

## Reward

The accepted hourly reward combines critical service, utilization, fairness, switching, and
oversubscription:

```text
r_t = (5 C_t + 2 U_t + 2 J_t - 0.25 S_t - 3 O_t) / 9
```

Here, `C_t` is the critical-service ratio, `U_t` is utilization, `J_t` is Jain fairness [3] of
cumulative locality service ratios, `S_t` is the fraction of localities that changed tier, and
`O_t` is positive requested-power excess divided by demand. Switching is zero at the first
hour. Episode reward is the sum of the 24 hourly rewards.

For locality `i`, let `x_i,t` be cumulative served energy divided by cumulative demand through
hour `t`. The hourly fairness term is:

```text
J_t = (sum_i x_i,t)^2 / (7 sum_i x_i,t^2 + 1e-8)
```

Using cumulative ratios makes fairness depend on service history rather than only the current
hour.

The reward is a research design choice, not an economic valuation of electricity or service
interruption. Because it directly includes critical service, utilization, fairness, switching,
and oversubscription, it is publicly labeled **Training-objective score (diagnostic)** rather than
treated as independent outcome evidence.

## Outcome definitions and signs

All ratios are unitless and remain in the interval [0, 1] unless a paired difference is being
reported.

| Outcome | Definition | Preferred sign for `PPO - Fixed Rotation` | Role |
|---|---|---:|---|
| Training-objective score (diagnostic) | Sum of 24 accepted hourly rewards | Positive | Diagnostic |
| Critical-service ratio | Critical-sector energy served / critical-sector demand | Positive | Supporting |
| Utilization | Mean hourly served power / `min(supply, demand)` | Positive | Primary |
| Jain fairness | Mean hourly Jain fairness of cumulative locality service ratios | Positive | Primary |
| Served energy | Sum of dispatched energy | Positive | Supporting |
| Curtailed energy | Demand minus served energy | Negative | Energy complement |
| Switching rate | Mean hourly fraction of localities changing tier; hour zero is zero | Negative | Secondary |
| Oversubscription | Mean positive requested-power excess / demand | Negative | Diagnostic |
| Power-balance violation rate | Fraction of hours served power exceeds supply by more than 0.001 MW | Non-positive | Safety |
| Critical-outage locality-hours | Locality-hours with critical-sector service below 99% | Negative | Primary |
| Demand and available energy | Scenario inputs summed over the episode | Not comparative | Input |

Because every time step is one hour, power sums in the energy metrics have the numerical unit
MWh. Critical outage locality-hours can range from 0 to 168 in one episode: 24 hours multiplied
by seven localities.

### Canonical aggregation rules

Every public metric has one calculation level and one aggregation rule:

- The **episode-pooled critical-service ratio** is total critical energy served divided by total
  critical energy demanded within one 24-hour episode. Controller aggregates are arithmetic means
  of those episode ratios; they do not pool energy across episodes.
- The **mean hourly supply-utilization ratio** is first calculated within an episode as the mean
  of 24 hourly `served / min(supply, demand)` values, then averaged across episodes.
- The **mean hourly cumulative-locality Jain fairness** is first calculated within an episode as
  the mean of 24 hourly fairness values over cumulative locality service ratios, then averaged
  across episodes.
- Reward is the 24-step sum per episode. Energy and locality-hour outcomes are 24-step sums per
  episode. Switching, oversubscription, and power-balance violation rate are hourly means per
  episode.

For each controller in the accepted reproduction, the aggregate is the equally weighted arithmetic
mean of the 288 episode values. Equivalently, it is the equally weighted mean of the nine cell means
because every cell contains 32 episodes. The untouched confirmatory aggregate is the same canonical
calculation over 960 episodes: 10 seeds x 32 base draws x 3 shortage variants. Paired inference
always uses within-scenario `PPO - Fixed Rotation` episode differences while keeping the three
shortage variants of each base draw in the same resampled cluster.

## Accepted reproduction design

The evaluation is a deterministic, within-scenario paired comparison.

| Design element | Locked value |
|---|---:|
| Shortage levels | Mild, Moderate, Severe |
| Seeds per shortage level | 50001, 50002, 50003 |
| Cells | 9 |
| Episodes per cell | 32 |
| Independent `(seed, batch_index)` base-draw clusters | 96 |
| Shortage variants per base draw | 3 |
| Matched scenario conditions | 288 |
| Controllers evaluated per scenario | 2 |
| Controller-scenario evaluations | 576 |
| Episode horizon | 24 one-hour steps |

For each shortage-level and seed cell, the generator creates one 32-episode `Scenario` batch.
The exact same immutable-reference `Scenario` object is passed first to Fixed Rotation and then
to PPO; the two controllers therefore receive identical demand and supply tensors. Metrics
remain on the GPU until the final serialization boundary. After the one packed host transfer,
the canonical artifact builder deterministically materializes episode rows and computes the
small row, cell, and aggregate summaries in NumPy/Python. Every stored paired difference is
defined as:

```text
delta = PPO - Fixed Rotation
```

This sign convention means positive values favor PPO for higher-is-better metrics, while
negative values favor PPO for lower-is-better metrics.

The estimand is the equally weighted mean paired difference across the nine shortage-level and
seed cells. Equal weighting is exact here because every cell contains 32 variants. The 288 rows
are valid matched controller comparisons, but their 96 base draws are the sampling clusters.
The estimand does not estimate the real-world prevalence of any shortage level or seed-defined
scenario family.

The `50001-50003` seeds were not used for PPO gradient updates or periodic validation, but the
archived promotion workflow reused them during model selection. Consequently, this design supports
historical parity and accepted reproduction only. It is not described as an untouched final test.

## Paired uncertainty calculation

Confidence intervals are calculated from the regenerated per-episode evidence. The
accepted frozen artifact contains cell summaries rather than episode rows and therefore cannot
support confidence intervals, p-values, or episode-level win rates.

For each inferential metric, the 288 paired differences are arranged as a `[3 shortage levels,
3 seeds, 32 base draws, metric]` tensor. Each of 10,000 bootstrap replicates samples 32 base-draw
indices with replacement within each seed. The same sampled index is reused for that base
draw's Mild, Moderate, and Severe variants, so correlated shortage variants stay together. The
replicate statistic is the equally weighted mean over shortage levels, seeds, and sampled base
draws. The reported 95% interval uses an explicit CuPy CUDA sort followed by linear
interpolation at the 2.5th and 97.5th percentile ranks.

This is a deterministic seed-stratified paired cluster percentile bootstrap [4]. It holds the three
shortage levels and three seed strata fixed; it does not resample shortage levels or seeds. This
preserves the intended dependence and stratification but limits the intervals to the locked
design.

The bootstrap seed is 20260815. Differences and reductions use float64. PyTorch [5] CUDA tensors
are exposed to CuPy [6] through zero-copy DLPack; sampling and percentile calculation stay on
`cuda:0`. Mean, lower, and upper statistics return through one stacked DLPack transfer, and the
evaluator performs one final packed full numeric evidence-array transfer for canonical JSON
serialization. Boundary validators may synchronize scalar invariant flags; they do not move the
simulation, metric, forecast, or bootstrap arrays or calculations to CPU.

These intervals quantify resampling variability of the 96 base-draw clusters conditional on the
nine synthetic cells. They are not p-values, do not estimate a field causal effect, and do not
establish performance on a real Sudanese electricity network.

The reproduced critical-service estimand first calculates a ratio for each episode and then
averages those episode ratios. The accepted cell summaries used pooled critical energy. Their
maximum cell difference is `0.0000012684613466262817`, which is below the locked `0.000002`
tolerance. The new rows should therefore be understood as a source-compatible episode-level
reproduction, not reconstructed historical episode data.

## Untouched robustness, sensitivity, and forecast design

The confirmatory supplement uses seeds 81001-81010 with 32 base draws per seed. Crossing these
320 independent base draws with all three shortage levels gives 960 matched scenario conditions
and 1,920 controller-scenario evaluations. The two controllers reuse the exact same immutable
scenario within every pair. The deterministic seed-stratified paired cluster bootstrap uses 10,000
replicates and seed `20260826`; it resamples 32 base-draw indices within each of 10 seed strata and
keeps all three shortage variants together. Result classification is descriptive: sign-neutral
integrity accepts favorable, adverse, or ambiguous intervals.

Sensitivity uses the same base draws. Each of seven locality weights and five sector shares is
multiplied separately by 0.75 and 1.25, then its affected vector is renormalized to preserve the
original total. This gives 24 one-factor-at-a-time +/-25% variants, or 25 parameterizations when
the unperturbed base is included. Every variant is evaluated over the same 960 matched scenarios.
For each primary outcome the supplement stores the variant paired effect and the
difference-in-differences relative to the base paired effect. This is a bounded local sensitivity
study, not a proof of global robustness.

On the 320 untouched synthetic demand traces, forecast evaluation aligns both methods to target
hours 3-23. The three-hour trailing mean uses only hours `t-3` through `t-1`; one-hour persistence
uses hour `t-1`. The supplement stores per-seed sufficient statistics and recomputable pooled MAE
and RMSE for 235,200 locality-sector predictions and 6,720 system-total predictions per method.
At each one-hour target, residuals are in MW. Locality-sector statistics pool absolute and squared
cell residuals across episode x target x 7 x 5; system-total error is calculated after summing the
35 signed cell residuals. SAE is in MW, SSE in MW squared, and pooled MAE/RMSE in MW. These values
compare two synthetic display methods and do not establish real-world forecast accuracy.

## Causal display-only demand projection

The desktop presentation can display a one-hour-ahead demand projection. For each locality and
sector, the default projection is the mean of the preceding three observed hours. A target is
never included in its own input window, and changes to future observations cannot affect an
earlier projection. The operation remains on the input PyTorch device.

With a three-hour window, current hours 2 through 22 display locality projections for hours 3
through 23. The first two hours have insufficient history and the final hour has no next-hour
target, so the UI shows no projection for those edge cases.

This projection is intentionally simple. It demonstrates causal prediction logic for
decision-support presentation, but it is not consumed by the PPO observation, simulator, reward,
or controller comparison. Its synthetic-trace MAE and RMSE are evaluated against persistence in
the robustness supplement; no real-world forecast-accuracy claim is made.

## Reproducibility and provenance

The proposal, accepted source benchmark, source checkpoint, clean inference artifact, historical
training record, accepted reproduction, and untouched robustness supplement are bound by SHA-256
manifests. The key final evidence hashes are:

| Artifact | SHA-256 |
|---|---|
| Accepted proposal PDF | `3432e64a21ab9741bf47646fd0f4f561522ddfd81d4005deb892d38672c65e48` |
| Frozen aggregate benchmark | `3d720ba87a747a5780119decd7144b4051fc36fc72ab50f229f5750ad749fd15` |
| PPO inference artifact | `3e4a7ce44267dd1cf3bbde7e77abafb8f31fdea77ece63acda7be17fa3198ad5` |
| Accepted paired reproduction | `6052e9e47d92e82a0f1db0d3643a2e9be6edc5f3e9fc615b0b185964b54c1677` |
| Untouched robustness supplement | `c89e04b1bdca371e1728f6b4cb19f4e899261ab805c2e400c15e835eaf75a30b` |
| PPO training provenance YAML | `2cfa158a4b26596fd1f155c57fa79a88dc1c41ce8df946792c559184a815128a` |
| Project source manifest | `7f0a865686de7ff4f7264ff9211fb60ad226abad9445573ab72615172b963fa4` |

Run the release checks from the repository root on the presentation laptop:

```powershell
uv sync --frozen --extra dev --extra docs
$sourceArchive = (Resolve-Path -LiteralPath "..\aipower_rl" -ErrorAction Stop).Path
$env:GRIDGUARD_SOURCE_REPOSITORY = $sourceArchive
$env:GRIDGUARD_SOURCE_CHECKPOINT = (
    Resolve-Path -LiteralPath (
        Join-Path $sourceArchive "artifacts\checkpoints\best.pt"
    ) -ErrorAction Stop
).Path
uv run gridguard-doctor
uv run pytest -ra
uv run gridguard-evaluate --verify-evidence
uv run ruff check .
```

The explicit archive variables turn a missing source archive or checkpoint into a test failure
instead of an optional skip. Release review must confirm that pytest reports zero skipped tests.
The full test command is deliberately unfiltered: release verification requires the NVIDIA GPU,
the CUDA PyTorch/CuPy stack, and the trusted source archive. The CPU-safe development subset is
useful feedback but is not a release certificate:

```powershell
uv run pytest -m "not gpu and not source_archive"
```

The paired artifact can be independently regenerated to a temporary file without overwriting
the committed evidence. The command refuses to target the trusted artifact and accepts the
reproduction only when its bytes produce the pinned SHA-256:

```powershell
$reproduction = Join-Path $env:TEMP "gridguard-paired-benchmark.json"
uv run gridguard-evaluate --reproduce-paired --output $reproduction
```

The required output digest is
`6052e9e47d92e82a0f1db0d3643a2e9be6edc5f3e9fc615b0b185964b54c1677`.

The untouched supplement has the same overwrite protection and byte-reproduction requirement:

```powershell
$robustness = Join-Path $env:TEMP "gridguard-robustness-supplement.json"
uv run gridguard-evaluate --reproduce-robustness --output $robustness
```

Evidence verification deliberately separates scientific integrity from result direction.
`validate_paired_benchmark_payload` performs neutral integrity checks: closed schema, matched
inputs, finite physical bounds, energy identities, row deltas, recomputed summaries, and a parity
status consistent with the stored errors. It can therefore validate a well-formed adverse or
inconclusive artifact. `assert_paired_benchmark_integrity` adds locked provenance, a direct
cross-check against the hash-pinned frozen benchmark, and passing 18-cell parity without checking
the sign of any interval. The robustness verifier independently reconstructs protocol counts,
controller-minus-baseline means, difference-in-differences, forecast sufficient-statistic
identities, and result classifications. No favorable-result release gate exists.

## Limitations and threats to validity

- All demand and supply traces are synthetic and are not calibrated against utility records.
- The illustrative supply trace is constructed partly from synthetic demand; matched controller
  scenarios do not remove this shared generator assumption.
- The seven localities share one abstract supply pool; the model omits network topology,
  feeder limits, losses, voltage, frequency, protection, restoration, and operator constraints.
- Proportional dispatch scales critical and non-critical requests together when total requests
  exceed supply; it does not model feeder-level within-hour critical restoration priority.
- The one-hour resolution cannot represent sub-hour switching or transient grid behavior.
- Fixed Rotation is supply-blind and limited to Off/Full Service. The study does not establish
  superiority over a supply-aware critical-preserving heuristic, every optimization method, or
  every learned controller.
- Performance relative to the global optimum is unknown; no episode-level optimization or valid
  upper-bound comparator is included.
- The PPO joint-action head has `3^N` scaling (`3^7 = 2,187` here), limiting architectural
  scalability beyond the accepted seven-locality scope.
- Both designs remain conditional on fixed synthetic seed strata and shortage construction, even
  though the untouched supplement expands coverage to ten seeds and adds local sensitivity.
- Power-balance compliance is an invariant of this simulator, not evidence of real-grid
  stability or operational safety.
- The 99% critical-outage threshold and reward weights are explicit research assumptions.
- The evaluated operating domain is shortage-only: Mild, Moderate, and Severe. It does not cover
  surplus, normal operation, recovery, or restoration transients.
- The source-pinned administrative boundaries and Nile linework are presentation context only;
  they are not electrical-network topology and do not improve the simulator's external validity.
- The causal projection is evaluated only on synthetic demand against persistence; it has no
  real-data validation and remains excluded from controller decisions.

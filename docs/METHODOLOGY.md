# Methodology

## Study objective and claim boundary

GridGuard Khartoum evaluates whether a trained PPO controller can allocate limited
electricity more effectively than a traditional Fixed Rotation schedule in a simplified,
software-only model of Khartoum State. The accepted proposal [1] defines this as a
decision-support simulation, not a controller for a real electricity network. Accordingly,
the evidence supports a comparison inside the stated synthetic scenario design only.

The confirmatory comparison is PPO versus Fixed Rotation. The proposal permits prediction,
rule-based logic, optimization, and reinforcement learning where appropriate; it does not
require every method to be a controller in the final comparison. The presentation includes a
small causal demand projection as a separate explanatory feature. It is not an input to PPO,
does not change dispatch, and is excluded from every reported performance claim.

## Simulation domain

The model uses seven Khartoum State localities in a fixed tensor order: Khartoum, Omdurman,
Bahri, East Nile, Jebel Awlia, Karari, and Ombada. Each locality contains five demand sectors:
Hospital, Water, Bakery, Residential, and Commercial. Hospital, Water, and Bakery are the
three critical sectors.

Each episode contains 24 one-hour steps. Demand and supply are expressed as MW at each step;
energy totals are the sum over the one-hour steps and are therefore reported in MWh. The
simulator uses one global supply pool. It does not model feeders, voltages, frequencies,
transmission constraints, or power flow.

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
environment. The fixed evaluation seeds are 50001, 50002, and 50003.

Shortage variants are intentionally coupled. For a fixed `(seed, batch_index)` pair, the
generator reuses the same demand perturbations and supply-noise draw for Mild, Moderate, and
Severe; only the shortage fraction and window change. The design therefore contains 96 base
draw clusters, each observed under three shortage levels, rather than 288 independent random
draws.

## Controller decisions and dispatch

At every hour, each controller chooses one service tier for each locality:

- Off: request no load.
- Critical Only: request Hospital, Water, and Bakery load.
- Full Service: request all five sectors.

The simulator concatenates the requested critical and non-critical loads. If total requested
power exceeds available supply, one proportional scale factor is applied to every requested
load. This guarantees that simulated served power does not exceed the global supply pool.

### Fixed Rotation

Fixed Rotation divides the day into three-hour blocks. In each block, three consecutive
localities receive Full Service and the remaining localities are Off. The starting locality
advances by three positions at each block and wraps around the seven-locality order. The rule
does not observe current demand, supply, critical-service need, or cumulative fairness.

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
does not include optimizer state and does not claim to reproduce the original training run.

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
interruption. The individual physical and service metrics are therefore reported alongside
reward.

## Outcome definitions and signs

All ratios are unitless and remain in the interval [0, 1] unless a paired difference is being
reported.

| Outcome | Definition | Preferred sign for `PPO - Fixed Rotation` | Role |
|---|---|---:|---|
| Episode reward | Sum of 24 accepted hourly rewards | Positive | Primary |
| Critical-service ratio | Critical-sector energy served / critical-sector demand | Positive | Primary |
| Utilization | Mean hourly served power / `min(supply, demand)` | Positive | Primary |
| Jain fairness | Mean hourly Jain fairness of cumulative locality service ratios | Positive | Primary |
| Served energy | Sum of dispatched energy | Positive | Primary |
| Curtailed energy | Demand minus served energy | Negative | Primary |
| Switching rate | Mean hourly fraction of localities changing tier; hour zero is zero | Negative | Secondary |
| Oversubscription | Mean positive requested-power excess / demand | Negative | Diagnostic |
| Power-balance violation rate | Fraction of hours served power exceeds supply by more than 0.001 MW | Non-positive | Safety |
| Critical outage locality-hours | Locality-hours with critical-sector service below 99% | Negative | Diagnostic |
| Demand and available energy | Scenario inputs summed over the episode | Not comparative | Diagnostic |

Because every time step is one hour, power sums in the energy metrics have the numerical unit
MWh. Critical outage locality-hours can range from 0 to 168 in one episode: 24 hours multiplied
by seven localities.

## Matched evaluation design

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

## Paired uncertainty calculation

Confidence intervals are calculated only from the newly reproduced per-episode evidence. The
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
evaluator performs one final packed device-to-host transfer for canonical JSON serialization.

These intervals quantify resampling variability of the 96 base-draw clusters conditional on the
nine synthetic cells. They are not p-values, do not estimate a field causal effect, and do not
establish performance on a real Sudanese electricity network.

The reproduced critical-service estimand first calculates a ratio for each episode and then
averages those episode ratios. The accepted cell summaries used pooled critical energy. Their
maximum cell difference is `0.0000012684613466262817`, which is below the locked `0.000002`
tolerance. The new rows should therefore be understood as a source-compatible episode-level
reproduction, not reconstructed historical episode data.

## Causal display-only demand projection

The desktop presentation can display a one-hour-ahead demand projection. For each locality and
sector, the default projection is the mean of the preceding three observed hours. A target is
never included in its own input window, and changes to future observations cannot affect an
earlier projection. The operation remains on the input PyTorch device.

With a three-hour window, current hours 2 through 22 display locality projections for hours 3
through 23. The first two hours have insufficient history and the final hour has no next-hour
target, so the UI shows no projection for those edge cases.

This projection is intentionally simple. It demonstrates causal prediction logic for
decision-support presentation, but it is not consumed by the PPO observation, simulator,
reward, evaluation, or frozen claim. No forecast-accuracy claim is made.

## Reproducibility and provenance

The proposal, accepted source benchmark, source checkpoint, clean inference artifact, frozen
benchmark, and paired benchmark are bound by SHA-256 manifests. The key final evidence hashes
are:

| Artifact | SHA-256 |
|---|---|
| Accepted proposal PDF | `3432e64a21ab9741bf47646fd0f4f561522ddfd81d4005deb892d38672c65e48` |
| Frozen aggregate benchmark | `3d720ba87a747a5780119decd7144b4051fc36fc72ab50f229f5750ad749fd15` |
| PPO inference artifact | `3e4a7ce44267dd1cf3bbde7e77abafb8f31fdea77ece63acda7be17fa3198ad5` |
| Paired per-episode benchmark | `2c3e71e48c4c7bff4fe609d42a76ca4f770f6dd717cd7d45d5e8e0d99fdf2d11` |
| Project source manifest | `cc06df2492512d1c4de9c73051311f3401d5ddf90eb2563f2f440e53f26194c9` |

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
`2c3e71e48c4c7bff4fe609d42a76ca4f770f6dd717cd7d45d5e8e0d99fdf2d11`.

## Limitations and threats to validity

- All demand and supply traces are synthetic and are not calibrated against utility records.
- The seven localities share one abstract supply pool; the model omits network topology,
  feeder limits, losses, voltage, frequency, protection, restoration, and operator constraints.
- The one-hour resolution cannot represent sub-hour switching or transient grid behavior.
- The study compares one frozen PPO actor with one Fixed Rotation rule. It does not establish
  superiority over every heuristic, optimization method, or learned controller.
- The fixed seeds and shortage construction limit external validity. Bootstrap intervals remain
  conditional on this scenario design.
- Power-balance compliance is an invariant of this simulator, not evidence of real-grid
  stability or operational safety.
- The 99% critical-outage threshold and reward weights are explicit research assumptions.
- The schematic map is a presentation aid, not a geographic boundary or electrical-network map.
- The causal projection has no real-data validation and is excluded from controller claims.

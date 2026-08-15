# Implementation

## System overview

GridGuard Khartoum is an offline Python 3.12 desktop application and reproducible CUDA
simulation package. It separates four concerns: immutable domain contracts, deterministic
scenario and controller execution, audited evidence generation, and presentation-only UI
rendering.

```text
synthetic Scenario batch
        |
        +--> Fixed Rotation --> 24-hour simulator --> episode metrics --+
        |                                                            |
        +--> PPO -------------> 24-hour simulator --> episode metrics --+--> paired deltas
                                                                          --> CuPy bootstrap
                                                                          --> canonical JSON

validated map asset + replay traces + hash-verified evidence --> offline desktop presentation
```

The two controller paths share the exact same `Scenario` object. The desktop presentation may
replay a selected scenario, but its Results view reads committed evidence rather than silently
recalculating the headline claim.

## Module map

| Path | Responsibility |
|---|---|
| `src/gridguard/domain.py` | Locality, sector, shortage, service-tier, controller, scenario, result, and metric contracts |
| `src/gridguard/scenarios.py` | Seeded 24-hour synthetic demand and supply batches |
| `src/gridguard/controllers.py` | Shared controller protocol, Fixed Rotation rule, and PPO adapter |
| `src/gridguard/simulation.py` | Causal 32-value observation, tier dispatch, reward, and immutable result traces |
| `src/gridguard/metrics.py` | Per-episode proposal outcomes on the source tensor device |
| `src/gridguard/policy.py` | Strict PPO actor schema, artifact verification, CUDA load, and deterministic inference |
| `src/gridguard/statistics.py` | CuPy stratified paired bootstrap through DLPack |
| `src/gridguard/evaluation.py` | Exact scenario pairing, per-episode evidence, frozen parity, canonical serialization, and semantic verification |
| `src/gridguard/forecast.py` | Causal trailing-window demand projection for display only |
| `src/gridguard/gpu.py` | Fail-closed CUDA, PyTorch, CuPy, driver, and DLPack preflight |
| `src/gridguard/provenance.py` | Pinned source-manifest and artifact verification |
| `src/gridguard/resources.py` | Safe source-checkout or installed-bundle resource resolution |
| `src/gridguard/ui/map_asset.py` | Closed-schema validation for the offline locality schematic |
| `src/gridguard/ui/app.py` | Desktop entry point and fail-closed startup sequence |
| `src/gridguard/ui/window.py` | 1366 x 768 shell, page navigation, and presentation shortcuts |
| `src/gridguard/ui/presentation_model.py` | Strict projection of paired evidence into presentation data |
| `src/gridguard/ui/map_view.py` and `src/gridguard/ui/map_items.py` | Interactive `QGraphicsView` map and redundant service-state rendering |
| `src/gridguard/ui/live_models.py` | Immutable, validated host-side contracts for two 24-hour controller timelines |
| `src/gridguard/ui/live_runner.py` | Shared-scenario CUDA execution and one explicit display transfer |
| `src/gridguard/ui/live_worker.py` | Qt worker signals that keep CUDA execution off the GUI thread |
| `src/gridguard/ui/overview_page.py` | Scope, protocol, and locality overview |
| `src/gridguard/ui/live_page.py` | Shortage/seed controls, 24-hour replay timeline, and locality detail surface |
| `src/gridguard/ui/results_page.py` | Artifact-derived controller comparison and paired intervals |
| `src/gridguard/ui/methodology_page.py` | In-application method and limitation summary |
| `src/gridguard/ui/components.py` and `src/gridguard/ui/theme.py` | Reusable accessible cards, metric formatting, and visual theme |
| `src/gridguard/cli.py` | `gridguard-doctor` and evidence-verification entry points |

## Domain and tensor contracts

The public vocabulary is encoded directly as enums. There are seven localities, five sectors,
three shortage levels, and three service tiers. `Scenario` tensors have shapes:

```text
demand: [batch, 24, 7, 5]
supply: [batch, 24]
```

Simulation traces have shapes:

```text
service_tiers: [batch, 24, 7]
requested:     [batch, 24, 7, 5]
served:        [batch, 24, 7, 5]
reward:        [batch, 24]
```

Constructors reject invalid dimensions, mismatched devices or dtypes, negative power,
non-finite values, invalid tiers, and inconsistent energy identities. Frozen dataclasses limit
accidental reference reassignment, and the simulator does not mutate the scenario tensors.

## GPU-first execution

The presentation and benchmark require `cuda:0`; there is no silent CPU fallback. PyTorch runs
scenario generation, the 24-hour batched simulator, PPO inference, and metric calculation. CuPy
runs the bootstrap sampling and percentiles. The large arrays remain on the GPU.

`run_gpu_preflight()` checks:

- PyTorch 2.8.0 with its CUDA 12.8 build;
- at least one visible NVIDIA device;
- CuPy in the supported range `>=13.6,<16` with a CUDA 12 runtime;
- an NVIDIA driver new enough for the CuPy runtime; and
- coherent zero-copy PyTorch/CuPy sharing by mutating a Torch CUDA tensor and observing the
  updated values through a CuPy DLPack view.

Any failed check stops launch or evaluation with an actionable error. Small unit tests can
inject a CPU device for local logic checks, but the release gate runs the unfiltered GPU and
source-provenance tests on the presentation laptop.

## PPO artifact and inference

The desktop does not deserialize the source training checkpoint. A dedicated exporter extracts
only the actor tensors into `data/model/ppo-policy.npz`. Runtime loading uses
`numpy.load(..., allow_pickle=False)` and accepts exactly six float32 tensors with locked names
and shapes.

| Layer | Shape |
|---|---|
| First actor encoder weight | `[512, 32]` |
| First actor encoder bias | `[512]` |
| Second actor encoder weight | `[512, 512]` |
| Second actor encoder bias | `[512]` |
| Joint-action output weight | `[2187, 512]` |
| Joint-action output bias | `[2187]` |

`data/model/manifest.json` binds the artifact to the accepted source checkpoint and source
revision. Before loading CUDA parameters, the verifier checks the externally pinned SHA-256,
file size, schema, architecture metadata, tensor set, dtype, shape, and finiteness. The loaded
actor is moved to `cuda:0`, frozen, placed in evaluation mode, and restricted to deterministic
argmax actions.

## Scenario, dispatch, and metrics pipeline

`generate_scenarios()` creates a complete batch directly on the requested PyTorch device. A
device-local seeded generator supplies both demand and supply perturbations. The simulator then
iterates the 24 hours while keeping cumulative demand, cumulative served energy, and previous
tiers on the same device.

At each hour:

1. Build the causal 32-value observation from current and accumulated information.
2. Ask the selected controller for seven integer tiers.
3. Reject malformed controller output before dispatch.
4. Convert tiers to requested sector loads.
5. Scale all requested loads proportionally if requests exceed available supply.
6. Update cumulative state and calculate the accepted reward.
7. Append immutable tensor traces for final metric calculation and presentation replay.

The metric module calculates one device-resident value per episode. Result and metric contracts,
together with invariant tests, enforce `served + curtailed = demand`, verify served energy
cannot exceed demand or availability, and report any hour exceeding the 0.001 MW power-balance
tolerance.

## Paired evaluation and statistics

`evaluate_scenario_pair()` executes Fixed Rotation and PPO on the exact same `Scenario` object.
The full evaluator loops through three shortage levels and three seeds, generating a batch of 32
scenario variants for each of nine cells. The three shortage variants at a given
`(seed, batch_index)` reuse one noise draw, so the design has 96 base-draw clusters. Crossing
those draws with three shortage levels produces 288 matched scenario conditions, 288 paired
rows, and 576 controller-scenario evaluations.

Episode metrics are stacked into `[3 shortage levels, 3 seeds, 32 episodes, 12 metrics]` cubes.
On CUDA, the evaluator calculates `PPO - Fixed Rotation` in float64, selects the seven inferential
metrics, and retains the `[3, 3, 32, 7]` dependence structure. For each of 10,000 deterministic
replicates, CuPy samples 32 cluster indices with replacement inside each seed and reuses those
indices across all three shortage variants. The statistic then gives equal weight to shortage
levels, seeds, and sampled clusters. Endpoint calculation uses an explicit CUDA sort and linear
interpolation, `a + (b - a) * weight`. This fixes the operation order and avoids reliance on the
library percentile helper while preserving the defined linear percentile. Mean, lower, and
upper values return to PyTorch in one stacked DLPack transfer. Only one packed numeric transfer
returns the two metric cubes and bootstrap output to host memory. The canonical builder then
materializes the 288 small row mappings and computes their cell/aggregate summaries in
NumPy/Python for deterministic JSON serialization and validation; no controller, simulator,
metric, or bootstrap computation falls back to CPU.

The resulting `data/evidence/paired-benchmark.json` contains:

- all 288 controller-paired episode rows;
- nine cell summaries;
- aggregate controller means;
- paired mean differences and 95% percentile intervals;
- metric definitions and direction metadata;
- frozen-cell parity errors and tolerances;
- exact scenario counts, seeds, and bootstrap protocol;
- the 96-base-draw cluster structure and three shortage variants per draw;
- PPO and source checkpoint provenance; and
- explicit scientific limitations.

Serialization is canonical UTF-8 JSON with sorted keys, finite numbers only, compact separators,
and a final newline. The verifier requires the external hash, canonical byte representation,
closed schema, exact row ordering, unique scenario identities, matched input energy, recomputed
row deltas, metric physical bounds and energy identities, recomputed cell and aggregate means,
inference metadata, and passing frozen parity. Verification also reloads the frozen artifact and
cross-checks all 18 controller cell summaries rather than trusting the stored parity declaration
alone.

## Evidence provenance and fail-closed behavior

`data/evidence/source-manifest.json` is itself pinned in code. It contains exact IDs and hashes
for the accepted proposal, frozen benchmark, PPO artifact, PPO manifest, paired benchmark, and
nine source-lineage files. Verification rejects a rewritten manifest even if its internal
entries are self-consistent.

Additional safeguards include:

- duplicate JSON fields and non-finite JSON constants are rejected;
- resource paths must remain relative to the runtime root;
- model files must match both an external trust anchor and their manifest;
- evidence cell counts, controller labels, seeds, and scenario IDs are closed rather than
  inferred from arbitrary input;
- the paired file must match SHA-256
  `2c3e71e48c4c7bff4fe609d42a76ca4f770f6dd717cd7d45d5e8e0d99fdf2d11`; and
- the evaluator refuses devices other than `cuda:0`, bootstrap counts other than 10,000, or a
  bootstrap seed other than 20260815.

## Display-only forecast

`causal_demand_projection()` accepts `[batch, hour, 7, 5]` demand tensors. Its default output is
a one-hour-ahead trailing mean over the previous three hours. The function validates shape,
dtype, finiteness, non-negativity, window, and horizon, preserves the input dtype and device,
and uses `torch.inference_mode()`.

The live CUDA runner sums the sector projections into one value per locality and aligns them so
current hours 2 through 22 display forecasts for hours 3 through 23. The first two hours lack a
three-hour history and hour 23 has no next-hour target, so those frames display no projection.
The module, runner, UI labels, and tests enforce the boundary: the projection is displayed only
and is never passed to PPO, dispatch, reward, or the evidence generator.

## Offline schematic map

The bundled map is `src/gridguard/assets/map/khartoum-schematic.json`. It contains only display
geometry, locality labels, directional keyboard neighbors, and schematic Nile paths. Its own
disclaimer states that it is neither a geographic boundary map nor an electrical-network
topology.

The loader rejects duplicate fields, non-finite coordinates, unsupported schema values,
unknown or missing localities, invalid keyboard order, external references, scientific values,
controller values, degenerate or self-intersecting polygons, out-of-scene points, label anchors
outside polygons, and invalid neighbor targets. Keeping simulation data out of the map asset
prevents presentation geometry from becoming an unreviewed scientific input.

## Desktop presentation

`gridguard-desktop` starts the PySide6 application through `ui/app.py`. Before it creates the
main window, startup runs the complete project preflight: GPU and DLPack checks, pinned source
manifest, paired-evidence semantics and frozen parity, PPO artifact verification, and a golden
CUDA inference action. It then loads the same pinned paired-evidence snapshot and bundled map.
A GPU, evidence, model, resource, or map failure blocks the presentation instead of opening a
partly trusted interface.

The main window is designed at 1366 x 768 and contains four pages: Overview, Live Scenario,
Results, and Methodology. Ctrl+1 through Ctrl+4 select the pages and F11 toggles fullscreen.
Controls expose accessible names and descriptions, accept keyboard focus, and use a visible
gold focus outline.

The `QGraphicsView` map supports pointer selection, hover details, arrow-key navigation through
explicit locality neighbors, wheel or plus/minus zoom, drag-to-pan, and R or 0 to reset the
view. Service state is never communicated by color alone: fill, texture, symbol, and text
distinguish Off, Critical Only, and Full Service. The map retains its schematic disclaimer in
the presentation.

The Live Scenario page accepts a shortage level and seed and starts a `QThread` worker so CUDA
execution does not block the Qt event loop. The runner generates one `cuda:0` scenario, passes
the exact same object through PPO and Fixed Rotation, and validates that every scenario and
result tensor remains on `cuda:0`. Inputs are disabled while the worker is active, and the
window refuses to close until the CUDA run has completed safely.

After both controllers finish, the runner packs hourly supply, 35 locality-sector demand values,
seven next-hour locality projections, seven PPO tiers, 35 PPO served values, seven Fixed
Rotation tiers, and 35 Fixed Rotation served values into one `[24, 127]` display tensor. This is
the UI runner's only bulk device-to-host transfer. The resulting immutable comparison must
contain exactly 24 frames for each named controller and identical supply, system demand,
locality demand, and projection values at every paired hour.

A controller selector and 24-hour slider let the presenter compare tier allocations on the same
scenario. The selected-locality panel derives actual demand MW, served MW, critical-service
ratio, and service ratios for Hospital, Water, Bakery, Residential, and Commercial from the
transferred simulation values. It also labels the causal next-hour demand projection as display
only and explicitly states that the value is not PPO input or evidence. Until a valid comparison
arrives, the page displays an honest waiting state; it never fabricates replay data. Runtime
errors clear the frames, disable replay controls, and remain visible to the presenter.

The Results page renders only values present in the paired evidence snapshot, including paired
intervals. It does not calculate replacement headline values. The Methodology page summarizes
the locked design and repeats the evidence limitations for presentation use.

## Runtime resources and packaging

`runtime_root()` supports two layouts:

1. a source checkout containing `data/evidence/source-manifest.json`; or
2. an installed package containing a `_bundled` runtime tree.

The wheel configuration force-includes `data/evidence`, `data/model`, and the accepted proposal
PDF. The map JSON is package data under `src/gridguard/assets`. The same safe resolver is used
for source and installed execution, and path traversal outside the chosen root is rejected.

The application is designed for an offline Windows presentation. No evidence, map geometry,
model weight, or replay dependency requires a network request after installation.

## Verification strategy

The tests are organized by the same boundaries as the implementation:

| Test group | What it proves |
|---|---|
| Domain and scenario tests | Shapes, vocabulary, seed determinism, shortage construction, and GPU generation |
| Controller and simulation tests | Fixed schedule, action validation, dispatch, reward, and golden traces |
| Metric tests | Energy identities, proposal definitions, sign conventions, and finite device-resident outputs |
| Policy tests | Safe artifact parsing, actor architecture, source parity, decoding, and real CUDA inference |
| Evaluation and statistics tests | Exact pairing, counts, bootstrap stratification, DLPack sharing, canonical evidence, and frozen parity |
| Provenance and source tests | Pinned manifest, source Git blobs, accepted benchmark migration, and tamper rejection |
| Forecast tests | Causality, shape, dtype, device preservation, and invalid-input rejection |
| Map and package tests | Geometry, keyboard metadata, display-only schema, wheel inclusion, and isolated install loading |
| Desktop UI tests | Startup-gate ordering, evidence projection, tabs, shortcuts, map interaction, and asynchronous live success/error states |
| GPU integration tests | Real PyTorch/CuPy/CUDA preflight on the presentation laptop |

The portable command excludes release-only GPU and source-archive checks:

```powershell
uv run pytest -m "not gpu and not source_archive"
```

The presentation release requires all checks without skips:

```powershell
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

Release review requires zero skipped tests. Setting both source paths explicitly prevents the
source-parity fixtures from treating an absent research archive as an optional environment.

## Deliberate implementation limits

- Training code and optimizer state are not part of the clean presentation runtime.
- The simulator is a decision-support abstraction and has no hardware control interface.
- The map contains no network topology or real administrative boundaries.
- NumPy/Python is restricted to artifact parsing and deterministic row/cell/aggregate assembly
  after the single final evidence transfer; simulation, inference, metrics, and bootstrap hot
  paths remain on the GPU.
- The result claim is fixed to the two named controllers, three shortage levels, three seeds,
  and one frozen PPO actor.

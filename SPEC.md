# Specification: GridGuard Khartoum

Status: Approved implementation contract; October source refresh under readiness review
Date: 2026-08-15
Revision: 2026-10-21 (source refresh; final release verification pending)
Release: 1.2.0 (2026-08-27)
Released baseline: 1.1.0 (2026-08-26)
Source proposal: `docs/source/graduation-project-proposal.pdf`
Source proposal SHA-256: `3432e64a21ab9741bf47646fd0f4f561522ddfd81d4005deb892d38672c65e48`

## Reviewer start here

- The historical A01-A24 remediation contract is in
  [docs/DEVILS_ADVOCATE_AUDIT.md](docs/DEVILS_ADVOCATE_AUDIT.md). Current findings and closure gates
  are in [docs/GRADUATION_REVIEW.md](docs/GRADUATION_REVIEW.md). An open ledger row is not release
  evidence, and a historical closure does not certify later source changes.
- The canonical, strict historical training record is
  [data/model/training-provenance.yaml](data/model/training-provenance.yaml). It exposes the actor
  architecture, behavior-cloning initialization, PPO hyperparameters, workload, seeds,
  validation and selection, hardware, source bindings, and convergence in one reviewer-facing
  YAML file.
- The August release records identify their own tested source snapshot. The October source
  refresh needs a new tested snapshot, reviewed artifacts, and final package verification before
  handoff. The final handoff separately reports exact local/private-remote branch SHA equality
  because a Git branch cannot record its own future head SHA without changing that SHA.

## Assumptions

1. The presentation target is a Windows laptop.
2. The primary application and benchmark run on the presentation laptop's NVIDIA GPU and require a successful CUDA preflight; internet access is never required.
3. The public scientific claim is limited to the hash-pinned evidence roles: the accepted
   reproduction preserves historical parity, while the documented-seed-disjoint holdout reports
   the reviewer-facing comparison of PPO and Fixed Rotation on three primary outcomes inside the
   stated synthetic shortage protocol.
4. Legacy experiment versions are research history, not part of this clean project.
5. The seven-locality map uses source-pinned OCHA/HDX administrative boundaries with Natural
   Earth Nile context. It is presentation context only, not electrical-network topology, and
   cannot provide scientific inputs.
6. The final academic package includes a native desktop application, source repository, methodology/implementation/results chapters, a final PDF report, and a slide deck.
7. No real utility measurements are available; all demand and supply traces remain synthetic and this limitation is displayed in the application and documents.

## Objective

Build a clean, presentation-ready implementation of the accepted graduation proposal: an AI-assisted load-shedding simulation for seven Khartoum State localities that protects critical services, considers fairness, and is evaluated against a traditional fixed-rotation schedule.

The primary user is an academic review committee. A presenter must be able to select a shortage severity, replay a deterministic scenario, move through the 24-hour timeline, inspect each locality on an interactive map, and compare PPO with Fixed Rotation. The application must clearly distinguish live scenario demonstration data from frozen benchmark evidence.

## Product vocabulary

- Product: **GridGuard Khartoum**
- Learned controller: **PPO**
- Traditional baseline: **Fixed Rotation**
- Service tiers: **Off**, **Critical Only**, **Full Service**
- Critical sectors: **Hospitals**, **Water**, **Bakeries**
- Other sectors: **Residential**, **Commercial**
- Shortage levels: **Mild**, **Moderate**, **Severe**
- Geographic units: **Localities**, not zones, districts, feeders, or areas

Internal experiment labels must not appear in the normal presentation flow.

## User flows

### 1. Run a matched scenario

The presenter selects Mild, Moderate, or Severe shortage and a seed, then runs the same generated demand and supply trace through PPO and Fixed Rotation.

Acceptance conditions:

- Both controllers receive the identical scenario.
- The run is deterministic for the same severity and seed.
- The application reports demand, available supply, served energy, critical service, fairness, curtailment, and power-balance violations.
- A table labeled **Live demonstration** shows Served energy, Critical service, Fairness,
  Curtailed energy, and Power-balance violations for both controllers from that selected episode.
- Live aggregates remain visually and semantically separate from frozen Results evidence.

### 2. Explore the interactive map

The presenter moves an hour slider and selects a locality on a pannable and zoomable offline
administrative map or its synchronized locality list.

Acceptance conditions:

- Every locality is visible and selectable by mouse and keyboard.
- Locality color and text show the active service tier without relying on color alone.
- The detail panel reports total demand, served power, critical-service percentage, and per-sector status for the selected hour.
- The map retains the exact ADM2 P-code and source identity for all seven OCHA/HDX units and shows
  source-pinned Natural Earth White Nile, Blue Nile, and Nile linework.
- The interface exposes accessible names and descriptions, a synchronized locality list, and
  visible keyboard focus.
- A permanent notice says that boundaries are presentation context only, results are synthetic,
  and the view is not electrical-network topology.

### 3. Present the frozen result

The presenter opens a Results view containing the proposal-level comparison and its provenance.

Acceptance conditions:

- Headline values are loaded from the committed, hash-verified documented-seed-disjoint holdout.
- The primary table contains exactly critical-outage locality-hours, Jain fairness, and
  utilization for PPO and Fixed Rotation across seeds 81001-81010: 320 independent base draws,
  960 matched scenario conditions, and 1,920 controller-scenario evaluations.
- Training-objective score is diagnostic, while power-balance violation rate is shown separately
  as a safety invariant.
- The accepted reproduction remains available for historical parity but is visibly labeled as
  not an independently custodied final test because its three seeds participated in model selection.
- The synthetic-data and decision-support-only limitations remain visible.
- All evidence verification is sign-neutral. Result direction is descriptive and never a release
  gate.

### 4. Inspect evidence and explain the methodology

The approved October interface contains Overview, Live Scenario, and Results, with floating
top navigation, open cartography, and the selected A/D/F visual design. Results opens native
Evidence details for artifact identity, hash, counts, custody, sensitivity, and comparison limits.
The Methodology tab and runtime status strip are removed at the user's request. The offline
methodology chapter and canonical training record retain the full account of scenario generation,
action tiers, proportional dispatch, Fixed Rotation, PPO, reward, training, and forecast evaluation.

## Tech stack

- Python 3.12
- PySide6 / Qt 6 for the native desktop interface
- PyTorch 2.8 with CUDA 12.8 for batched simulation and PPO inference
- CuPy for GPU-native numerical/statistical work and zero-copy DLPack interchange
- NumPy after the single final full numeric evidence-array transfer for canonical artifact
  assembly and small host-side validation; scalar invariant checks may synchronize CUDA, but no
  simulation, inference, metric, forecast, or bootstrap array computation moves to CPU
- Standard-library JSON and hashing for evidence provenance
- pytest and pytest-qt for unit, integration, and GUI interaction tests
- Ruff for linting and formatting checks
- uv for dependency and command management
- PyInstaller for the Windows presentation build

The GUI uses `QGraphicsScene` and `QGraphicsView` for the offline interactive map. Qt's documented
scene/item event model provides selection and hover behavior, while `ScrollHandDrag` and wheel
scaling provide panning and zooming. Runtime projects verified WGS84 geometry to EPSG:3857 and
aspect-fits it into a fixed scene without a GIS or network dependency.

## Commands

The release test contract has four distinct stages: the portable gate for ordinary development,
the mandatory presentation-GPU gate for the complete source/runtime suite, the strict post-build
`release_artifact` gate for the frozen Windows tree, and the optional source-archive parity gate.
The normal package build does not require the sibling research archive.

```powershell
# Install development dependencies
uv sync --extra dev --extra docs

# Run the desktop application
uv run gridguard-desktop

# Reproduce the paired benchmark without overwriting the trusted artifact
uv run gridguard-evaluate --reproduce-paired --output .\paired-benchmark-reproduced.json

# Reproduce the documented-seed-disjoint robustness supplement without overwriting the trusted artifact
uv run gridguard-evaluate --reproduce-robustness --output .\robustness-supplement-reproduced.json

# Print the validated PPO training summary
uv run --extra docs python tools/training_provenance.py --path data/model/training-provenance.yaml --model-manifest data/model/manifest.json --summary-json

# Portable gate: no GPU, source archive, or built release tree required
uv run --no-sync pytest -m "not gpu and not source_archive and not release_artifact"

# Mandatory presentation-GPU gate: source/runtime tests and evidence, with zero skips
uv run --no-sync gridguard-doctor
uv run --no-sync pytest -ra -m "not source_archive and not release_artifact"
uv run --no-sync gridguard-evaluate --verify-evidence

# Build, then run the strict post-build release_artifact gate
.\scripts\build-presentation.ps1
uv run --no-sync pytest -ra -m "release_artifact"

# Optional source-archive parity gate when the read-only research archive is supplied
.\scripts\build-presentation.ps1 -StrictSourceParity `
  -SourceRepositoryPath ..\aipower_rl `
  -SourceCheckpointPath ..\aipower_rl\artifacts\checkpoints\best.pt

# Run lint checks
uv run ruff check .

```

## Project structure

```text
data/
  evidence/                     accepted reproduction, seed-disjoint robustness, and provenance
  model/                        PPO actor, manifest, and training-provenance.yaml
docs/
  source/                       accepted proposal PDF
  METHODOLOGY.md                academic methodology chapter
  IMPLEMENTATION.md             system implementation chapter
  RESULTS.md                    audited results chapter
  REPORT.md                     assembled final report source
output/presentation/            final slide deck with embedded speaker notes
src/gridguard/
  assets/map/                   pinned boundaries, Nile context, provenance, and attribution
  domain.py                     shared vocabulary and typed data contracts
  scenarios.py                  synthetic demand/supply generation
  forecast.py                   separate display-only causal demand projection
  simulation.py                 24-hour load-shedding state transitions
  controllers.py                Fixed Rotation and learned-policy adapters
  policy.py                     inference-only PPO network and loader
  metrics.py                    proposal-aligned outcome calculations
  evaluation.py                 matched evaluation and evidence verification
  robustness.py                 seed-disjoint evaluation, sensitivity, and forecast evidence
  cli.py                        reproducibility command
  ui/                           native desktop views and presentation state
tasks/                          reviewed plan and completion checklist
tests/                          unit, parity, integration, and GUI tests
```

## Code style

Use explicit domain names, immutable data where practical, and small functions with typed inputs and outputs.

```python
def run_matched_scenario(
    scenario: Scenario,
    controllers: tuple[Controller, ...],
) -> dict[ControllerName, SimulationResult]:
    """Evaluate every controller against the same guarded scenario;
    detected demand/supply mutation fails closed."""
    return {
        controller.name: simulate(scenario=scenario, controller=controller)
        for controller in controllers
    }
```

Rules:

- Use `locality`, `sector`, `service_tier`, and `shortage_level` consistently.
- Avoid version-number names in product code.
- Keep UI widgets below roughly 200 lines and separate presentation state from rendering.
- Never silently recompute frozen headline values in the GUI.
- Show units in variable names or labels where ambiguity is possible (`mw`, `mwh`, `hours`, `ratio`).

## Metric and evidence contracts

Canonical public metrics have one definition and aggregation path:

- critical-outage locality-hours, Jain fairness, and utilization are the three primary outcomes;
- critical service is a supporting ratio: critical energy served divided by critical energy
  demanded, pooled within each 24-hour episode;
- utilization is the episode mean of hourly served power divided by
  `min(available supply, demand)`;
- fairness is the episode mean of hourly Jain fairness calculated from cumulative locality
  service ratios;
- the accepted episode reward is exposed only as **Training-objective score (diagnostic)**;
- power-balance violation rate is a separate safety invariant; and
- aggregate controller results are equally weighted arithmetic means of episode metrics. In the
  accepted reproduction, nine shortage-by-seed cell means give the same aggregate because every
  cell contains 32 episodes; the documented-seed-disjoint supplement applies the same rule across 30 equal-size
  shortage-by-seed cells per controller.

`validate_paired_benchmark_payload` is a neutral integrity validator. It enforces closed schema,
matched inputs, physical bounds, arithmetic identities, row/cell/aggregate reconstruction, and
internally consistent parity status without assuming that PPO must win.
`assert_paired_benchmark_integrity` additionally verifies locked provenance, reloads the
hash-pinned frozen benchmark, and requires direct 18-cell parity. The accepted reproduction is
explicitly model-selection-involved. The separate robustness verifier locks seeds 81001-81010,
320 independent base draws, 960 matched scenarios, 1,920 controller evaluations, 10,000 bootstrap
replicates, 24 one-factor-at-a-time variants (25 parameterizations with the base), and the aligned
three-hour-versus-persistence forecast comparison. This is sign-neutral integrity: no favorable
result is required for writing, verification, packaging, display, or release.

[data/model/training-provenance.yaml](data/model/training-provenance.yaml) is the canonical
historical training record. Its SHA-256 is
`d2568bce26e9221afa8c266113a704f6fbf2370d3bc949f5aa6c72a7411ba1a8`; the model manifest binds it.
It records the actor architecture, behavior-cloning initialization, Adam settings, validation and
selection, seed roles, source lineage, and the provenance class of every recovered field. The
selected seed-43 run used 600 PPO iterations, 153,600 episodes, and 3,686,400 transitions; the
three-seed candidate experiment totaled 1,800 iterations, 460,800 episodes, and 11,059,200
transitions. Runtime, hardware, and convergence values are scoped to selected seed 43. Historical
promotion gates are model-selection facts, not independent headline evidence. The record explicitly
labels unavailable historical environment facts and is not an executable bitwise-retraining
configuration.

## Testing strategy

### Unit tests

- Severity-neutral base-draw determinism, explicit reuse across shortages, and shortage coverage
- Fixed-rotation schedule behavior
- Service-tier dispatch and power-balance invariants
- Critical-service, fairness, utilization, curtailment, and outage metrics
- Policy artifact schema and CUDA inference
- Sign-neutral evidence integrity, frozen-parity assertion, robustness arithmetic, hashing, and
  claim reconstruction

### Parity tests

Golden cases generated from the accepted source implementation must prove that the clean scenario generator, simulator, fixed baseline, learned policy actions, and aggregate metrics match the source within declared numeric tolerances.

### Integration tests

- Same scenario is passed to both controllers
- All three shortage levels run end to end
- Accepted-reproduction reconstruction matches the frozen benchmark's 18 committed cell summaries
- Documented-seed-disjoint reviewer-facing headline values verify independently from the robustness supplement
- Desktop presentation model loads offline after the CUDA/PyTorch/CuPy preflight

### GUI tests

- Window opens in Qt offscreen mode
- Map supports selection, keyboard focus, panning, and zooming
- Hour and shortage controls update the map and detail panel
- Empty, loading, and error states are legible
- Frozen Results and its Evidence details render without changing evidence
- Live demonstration aggregates match the selected controller episode metrics
- Dynamic state is exposed through accessible text, a synchronized locality list, and visible
  keyboard focus at 1366 x 768

### Release checks

- All tests pass with no skipped proposal-critical tests
- Ruff passes
- Evidence verifier passes after all documentation is final
- Windows executable launches and completes the primary presentation flow

## Scientific claim boundary

Supported:

- The accepted reproduction preserves 96 independent base draws, 288 matched scenario conditions,
  and 576 controller-scenario evaluations, but its seeds were involved in model selection.
- The documented-seed-disjoint holdout covers seeds 81001-81010, 320 independent base draws, 960
  matched scenario conditions, and 1,920 controller-scenario evaluations under Mild, Moderate,
  and Severe shortage-only operation.
- Seed disjointness is established only against every documented training, validation, and
  model-selection seed in `data/model/training-provenance.yaml`. The holdout was not preregistered
  or independently custodied and cannot rule out undocumented activity.
- Sensitivity intervals are descriptive, pointwise 95%, and unadjusted for multiple comparisons;
  they do not support family-wide significance claims.
- No behavior-cloning-only ablation was run, so this comparison cannot isolate the
  incremental effect of PPO fine-tuning.
- Fixed Rotation is supply-blind, limited to Off and Full Service, and cannot select Critical Only.
- Descriptive comparison of critical-outage locality-hours, fairness, utilization, and the
  separate zero-tolerance power-balance safety invariant under the locked synthetic protocol.
- The software demonstrates how AI can support load-shedding decisions under simplified shortage scenarios.

Not supported:

- Real-grid operational readiness
- National-grid optimization
- Universal superiority over every optimization or rule-based controller
- Performance relative to a global optimum or a stronger supply-aware rule
- Accuracy on real Sudanese utility data
- Behavior during surplus, recovery, normal operation, or restoration transients
- A claim that administrative polygons or rivers represent feeders, service territories,
  substations, measured demand, or any other electrical-network topology

## Boundaries

### Always do

- Preserve the original proposal and source-artifact hashes.
- Use matched scenarios for controller comparisons.
- Generate one severity-neutral base-draw batch per seed and reuse it unchanged across Mild,
  Moderate, and Severe before applying severity-specific deterministic transformations.
- Keep synthetic-data and decision-support-only notices visible.
- Write tests before behavioral implementation.
- Run focused tests, lint, and evidence verification before every release commit.
- Freeze the final provenance manifest only after source, results, report, and presentation content are stable.

### Ask first

- Change the primary scientific claim or metric definitions.
- Add real operational data or external online services.
- Change the pinned geographic source, selected P-codes, river features, projection, attribution,
  or license treatment.
- Add dependencies outside the approved stack.
- Make the GitHub repository public.

### Never do

- Import superseded experiment code or claims into this repository.
- Present synthetic values as real utility measurements.
- Commit optimizer state, raw training runs, caches, secrets, or unnecessary experiment artifacts.
- Modify a frozen evidence file without regenerating and reviewing its provenance manifest.
- Require internet access during the presentation.
- Silently fall back to CPU for the presentation or frozen benchmark.

## Success criteria

- Every proposal objective and expected deliverable has an explicit implementation or document reference in `docs/PROPOSAL_TRACEABILITY.md`.
- The repository contains only proposal-relevant product code, tests, frozen evidence, final documents, and presentation assets.
- PPO versus Fixed Rotation can be reproduced on CUDA across Mild, Moderate, and Severe matched scenarios.
- Both the accepted 96-base-draw / 288-condition / 576-evaluation reproduction and the documented-seed-disjoint
  320-base-draw / 960-condition / 1,920-evaluation supplement are verified from committed artifacts.
- Reviewers can find every historical training parameter in the strict, hash-bound
  `data/model/training-provenance.yaml` record.
- The desktop GUI provides a keyboard-accessible interactive seven-locality map and a complete offline presentation flow.
- Methodology, Implementation, and Results chapters agree with code, metrics, labels, and claim boundaries.
- The final report PDF and slide deck pass visual review.
- The Windows presentation build launches successfully.
- Git status is clean and the final reviewed branch is pushed to GitHub.

## Approved delivery decisions

1. Publish the final repository privately on GitHub.
2. Use a clearly labeled, offline administrative locality map derived from pinned OCHA/HDX
   boundaries with Natural Earth Nile context. It is presentation context only and not
   electrical-network topology.
3. Proceed with a neutral academic report and slide design; adapt to a university template later if one is supplied.
4. Use **PPO** as the learned-controller name everywhere in the clean project.

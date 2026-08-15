# Specification: GridGuard Khartoum

Status: Approved
Date: 2026-08-15
Source proposal: `docs/source/graduation-project-proposal.pdf`
Source proposal SHA-256: `3432e64a21ab9741bf47646fd0f4f561522ddfd81d4005deb892d38672c65e48`

## Assumptions

1. The presentation target is a Windows laptop.
2. The primary application and benchmark run on the presentation laptop's NVIDIA GPU and require a successful CUDA preflight; internet access is never required.
3. The public scientific claim is limited to the accepted frozen summaries and the clean,
   hash-pinned paired reproduction: the trained PPO policy outperforms Fixed Rotation on the
   reported synthetic proposal metrics.
4. Legacy experiment versions are research history, not part of this clean project.
5. The seven-locality map is an interactive schematic, not an engineering-grade geographic or electrical-network map.
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

### 2. Explore the interactive map

The presenter moves an hour slider and clicks a locality on a pannable and zoomable schematic map.

Acceptance conditions:

- Every locality is visible and selectable by mouse and keyboard.
- Locality color and text show the active service tier without relying on color alone.
- The detail panel reports total demand, served power, critical-service percentage, and per-sector status for the selected hour.

### 3. Present the frozen result

The presenter opens a Results view containing the proposal-level comparison and its provenance.

Acceptance conditions:

- Headline values are loaded from a committed, hash-verified evidence artifact.
- The primary table compares PPO with Fixed Rotation across 96 independent base draws crossed
  with three shortage levels: 288 matched scenario conditions and 576 controller evaluations.
- Later experimental failures or unfinished branches cannot alter the headline claim.
- The synthetic-data and decision-support-only limitations remain visible.

### 4. Explain the methodology

The presenter opens a Methodology view that explains scenario generation, action tiers, critical-service priorities, the fixed baseline, PPO, reward design, matched evaluation, and limitations in plain academic language.

## Tech stack

- Python 3.12
- PySide6 / Qt 6 for the native desktop interface
- PyTorch 2.8 with CUDA 12.8 for batched simulation and PPO inference
- CuPy for GPU-native numerical/statistical work and zero-copy DLPack interchange
- NumPy after the single final evidence transfer for canonical artifact assembly and small
  host-side validation
- Standard-library JSON and hashing for evidence provenance
- pytest and pytest-qt for unit, integration, and GUI interaction tests
- Ruff for linting and formatting checks
- uv for dependency and command management
- PyInstaller for the Windows presentation build

The GUI uses `QGraphicsScene` and `QGraphicsView` for the offline interactive map. Qt's documented scene/item event model provides selection and hover behavior, while `ScrollHandDrag` and wheel scaling provide panning and zooming.

## Commands

```powershell
# Install development dependencies
uv sync --extra dev --extra docs

# Run the desktop application
uv run gridguard-desktop

# Verify the required NVIDIA GPU, CUDA, PyTorch, CuPy, model, and evidence
uv run gridguard-doctor

# Reproduce and verify the frozen benchmark
uv run gridguard-evaluate --verify-evidence

# Reproduce the paired benchmark without overwriting the trusted artifact
uv run gridguard-evaluate --reproduce-paired --output .\paired-benchmark-reproduced.json

# Portable unit gate, then mandatory presentation-laptop release gate
uv run pytest -m "not gpu and not source_archive"
uv run pytest

# Run lint checks
uv run ruff check .

# Build the Windows presentation application
uv run pyinstaller --noconfirm packaging/gridguard.spec
```

## Project structure

```text
data/
  evidence/                     frozen benchmark and provenance manifest
  model/                        inference-only trained policy checkpoint
docs/
  source/                       accepted proposal PDF
  METHODOLOGY.md                academic methodology chapter
  IMPLEMENTATION.md             system implementation chapter
  RESULTS.md                    audited results chapter
  REPORT.md                     assembled final report source
output/presentation/            final slide deck with embedded speaker notes
src/gridguard/
  assets/map/                   packaged offline schematic-map data
  domain.py                     shared vocabulary and typed data contracts
  scenarios.py                  synthetic demand/supply generation
  forecast.py                   separate display-only causal demand projection
  simulation.py                 24-hour load-shedding state transitions
  controllers.py                fixed rotation and learned-policy adapters
  policy.py                     inference-only PPO network and loader
  metrics.py                    proposal-aligned outcome calculations
  evaluation.py                 matched evaluation and evidence verification
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
    """Evaluate every controller against the same immutable scenario."""
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

## Testing strategy

### Unit tests

- Scenario determinism and shortage coverage
- Fixed-rotation schedule behavior
- Service-tier dispatch and power-balance invariants
- Critical-service, fairness, utilization, curtailment, and outage metrics
- Policy artifact schema and CUDA inference
- Evidence hashing and claim reconstruction

### Parity tests

Golden cases generated from the accepted source implementation must prove that the clean scenario generator, simulator, fixed baseline, learned policy actions, and aggregate metrics match the source within declared numeric tolerances.

### Integration tests

- Same scenario is passed to both controllers
- All three shortage levels run end to end
- Frozen benchmark reconstruction matches committed headline values
- Desktop presentation model loads offline after the CUDA/PyTorch/CuPy preflight

### GUI tests

- Window opens in Qt offscreen mode
- Map supports selection, keyboard focus, panning, and zooming
- Hour and shortage controls update the map and detail panel
- Empty, loading, and error states are legible
- Frozen Results and Methodology views render without changing evidence

### Release checks

- All tests pass with no skipped proposal-critical tests
- Ruff passes
- Evidence verifier passes after all documentation is final
- Windows executable launches and completes the primary presentation flow

## Scientific claim boundary

Supported:

- Across 96 independent synthetic base draws crossed with three shortage levels (288 matched
  scenario conditions and 576 controller evaluations), PPO materially outperformed Fixed Rotation
  on reward, critical-service ratio, supply utilization, fairness, served energy, and curtailment
  while recording zero power-balance violations.
- The software demonstrates how AI can support load-shedding decisions under simplified shortage scenarios.

Not supported:

- Real-grid operational readiness
- National-grid optimization
- Universal superiority over every optimization or rule-based controller
- Accuracy on real Sudanese utility data
- A claim that schematic localities represent feeder topology or exact administrative boundaries

## Boundaries

### Always do

- Preserve the original proposal and source-artifact hashes.
- Use matched scenarios for controller comparisons.
- Keep synthetic-data and decision-support-only notices visible.
- Write tests before behavioral implementation.
- Run focused tests, lint, and evidence verification before every release commit.
- Freeze the final provenance manifest only after source, results, report, and presentation content are stable.

### Ask first

- Change the primary scientific claim or metric definitions.
- Add real operational data or external online services.
- Replace the schematic map with externally licensed geographic boundaries.
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
- The 96-base-draw / 288-condition / 576-controller-evaluation result and all displayed deltas are
  verified from committed artifacts.
- The desktop GUI provides a keyboard-accessible interactive seven-locality map and a complete offline presentation flow.
- Methodology, Implementation, and Results chapters agree with code, metrics, labels, and claim boundaries.
- The final report PDF and slide deck pass visual review.
- The Windows presentation build launches successfully.
- Git status is clean and the final reviewed branch is pushed to GitHub.

## Approved delivery decisions

1. Publish the final repository privately on GitHub.
2. Use a clearly labeled, offline schematic locality map.
3. Proceed with a neutral academic report and slide design; adapt to a university template later if one is supplied.
4. Use **PPO** as the learned-controller name everywhere in the clean project.

# Implementation Plan: GridGuard Khartoum Final Delivery

Status: Complete

## Overview

Create a proposal-focused repository that preserves the proven PPO versus Fixed Rotation result, removes later-version research complexity, provides a native offline desktop presentation with an interactive schematic map, and ships coherent methodology, implementation, results, report, presentation, and GitHub artifacts.

## Architecture decisions

- Treat the accepted proposal PDF as the scope authority.
- Keep `aipower_rl` read-only as the research archive.
- Use `gridguard` as the only Python package and GridGuard Khartoum as the only product name.
- Extract only the accepted 24-hour, seven-locality simulation path.
- Use the trained PPO actor with PyTorch CUDA inference and keep the simulation hot path on the GPU.
- Use CuPy for GPU-native statistics and explicit DLPack interchange rather than host copies.
- Keep live demonstration outputs separate from the frozen benchmark.
- Use a built-in schematic map so presentation behavior is offline and licensing is simple.
- Freeze provenance after documents and presentation outputs are final.

## Dependency graph

```text
Approved specification
  -> frozen source inventory and golden cases
      -> clean domain and scenario contracts
          -> simulator and fixed baseline
              -> learned policy inference
                  -> matched evaluator and audited evidence
                      -> presentation view model
                          -> interactive map and results UI
                              -> Windows build
                                  -> report and slide deck
                                      -> final provenance audit
                                          -> GitHub publication
```

## Phase 0: Scope and provenance

### Task 0.1 - Approve specification and traceability baseline (complete)

Description: Review assumptions, vocabulary, claim boundaries, deliverables, and open questions before behavioral code is written.

Acceptance criteria:

- The user approves `SPEC.md` or requested edits are incorporated.
- Map type, GitHub visibility, and any university templates are resolved.
- No unresolved question changes the application architecture or public claim.

Verification:

- Manual review of `SPEC.md` and `docs/PROPOSAL_TRACEABILITY.md`.

Dependencies: None
Likely files: `SPEC.md`, `docs/PROPOSAL_TRACEABILITY.md`
Scope: Small

### Task 0.2 - Freeze migration sources and golden cases (complete)

Description: Copy only the proposal, benchmark, inference weights, and small golden source outputs needed to prove parity. Record source commits and SHA-256 hashes.

Acceptance criteria:

- Every migrated binary or JSON artifact has a source hash and source commit.
- Optimizer state and raw training runs are excluded.
- Golden cases cover all three shortage levels and both primary controllers.

Verification:

- `uv run pytest tests/test_provenance.py tests/test_golden_sources.py`

Dependencies: Task 0.1
Likely files: `data/evidence/source-manifest.json`, `data/evidence/golden-cases.json`, `tests/test_provenance.py`, `tests/test_golden_sources.py`
Scope: Medium

## Phase 1: Proposal simulation core

### Task 1.1 - Define clean domain contracts (complete)

Description: Define localities, sectors, shortage levels, service tiers, scenario data, controller results, and metric records with consistent names and units.

Acceptance criteria:

- Exactly seven localities, five sectors, and three service tiers are represented.
- Contracts reject invalid shapes, tiers, negative power, and mismatched horizons.
- Public names match `SPEC.md`.

Verification:

- Write failing contract tests, implement, then run `uv run pytest tests/test_domain.py`.

Dependencies: Task 0.2
Likely files: `src/gridguard/domain.py`, `tests/test_domain.py`
Scope: Small

### Task 1.2 - Extract scenario generation and causal forecast (complete)

Description: Implement deterministic 24-hour synthetic demand/supply generation for Mild, Moderate, and Severe shortages plus a clearly separated next-hour causal forecast used for decision support.

Acceptance criteria:

- Same seed and shortage level produce identical scenarios.
- Demand remains positive and supply remains below demand in shortage windows.
- Golden source demand/supply cases match within tolerance.

Verification:

- Red/green tests in `tests/test_scenarios.py` and `tests/test_forecast.py`.

Dependencies: Task 1.1
Likely files: `src/gridguard/scenarios.py`, `src/gridguard/forecast.py`, `tests/test_scenarios.py`, `tests/test_forecast.py`
Scope: Medium

### Task 1.3 - Implement simulator and Fixed Rotation (complete)

Description: Implement service-tier dispatch, cumulative service state, fixed three-hour rotation, and proposal metrics without later-version reserve/rebound/flexible-load machinery.

Acceptance criteria:

- Fixed Rotation changes the active group every three hours.
- Served power never exceeds available supply within tolerance.
- Source golden actions, served power, and core metrics match.

Verification:

- Red/green tests in `tests/test_simulation.py`, `tests/test_controllers.py`, and `tests/test_metrics.py`.

Dependencies: Task 1.2
Likely files: `src/gridguard/simulation.py`, `src/gridguard/controllers.py`, `src/gridguard/metrics.py`, `tests/test_simulation.py`, `tests/test_controllers.py`
Scope: Medium

### Checkpoint: Simulation foundation

- All Phase 1 tests pass.
- Ruff passes.
- A fixed-rotation scenario runs end to end on CUDA.
- Golden parity report contains no unexplained difference.

## Phase 2: Learned policy and evidence

### Task 2.1 - Create inference-only policy artifact (complete)

Description: Reconstruct the accepted PPO actor, strip optimizer and critic state from the trusted source checkpoint, and load the compact artifact safely with PyTorch on CUDA.

Acceptance criteria:

- Inference artifact contains only actor weights, architecture metadata, training provenance, and validation metadata.
- CUDA actions match the source checkpoint on golden observations.
- Invalid or tampered checkpoint schemas fail closed.

Verification:

- Red/green tests in `tests/test_policy.py` and hash verification in `tests/test_provenance.py`.

Dependencies: Task 1.3
Likely files: `src/gridguard/policy.py`, `data/model/ppo-policy.npz`, `tests/test_policy.py`, `data/model/manifest.json`
Scope: Medium

### Task 2.2 - Run matched evaluation and rebuild paired evidence (complete)

Description: Evaluate PPO and Fixed Rotation on identical held-out scenarios, store per-scenario rows, calculate paired statistics, and resolve the source episode-count ambiguity.

Acceptance criteria:

- Protocol explicitly reports matched-scenario count and controller-run count.
- Aggregate metrics reproduce the accepted benchmark within tolerance or differences are explained before publication.
- Paired deltas and confidence intervals are computed from per-scenario rows.

Verification:

- `uv run gridguard-evaluate --reproduce-paired --output paired-benchmark-reproduced.json`
- `uv run pytest tests/test_evaluation.py tests/test_statistics.py`

Dependencies: Task 2.1
Likely files: `src/gridguard/evaluation.py`, `src/gridguard/statistics.py`, `src/gridguard/cli.py`, `tests/test_evaluation.py`, `tests/test_statistics.py`
Scope: Medium

### Task 2.3 - Add evidence manifest and verifier (complete)

Description: Bind the benchmark, model, source proposal, metric definitions, and final claim values with canonical hashes and a fail-closed verifier.

Acceptance criteria:

- Any evidence or documented headline drift fails verification.
- Verifier reconstructs every displayed result from committed artifacts.
- Manifest is not finalized until report and slides are stable.

Verification:

- Red/green tamper tests in `tests/test_evidence.py`.
- `uv run gridguard-evaluate --verify-evidence`.

Dependencies: Task 2.2
Likely files: `src/gridguard/provenance.py`, `src/gridguard/resources.py`, `data/evidence/source-manifest.json`, `tests/test_provenance.py`
Scope: Medium

### Checkpoint: Scientific foundation

- PPO and Fixed Rotation run on CUDA.
- Primary result is reproducible and count wording is unambiguous.
- Evidence tampering is detected.
- Proposal metrics and limitations are frozen for UI integration.

## Phase 3: Native desktop presentation

### Task 3.1 - Build presentation state and main window shell (complete)

Description: Create a native PySide6 application shell and a presentation view model that loads scenario, evidence, selection, loading, and error states without embedding scientific logic in widgets.

Acceptance criteria:

- App passes a CUDA/PyTorch/CuPy preflight, then opens offline and shows Overview, Live Scenario, Results, and Methodology navigation.
- Missing model/evidence produces a useful recovery message.
- Presentation state is testable without rendering the full window.

Verification:

- Red/green tests in `tests/ui/test_presenter.py` and offscreen smoke test.

Dependencies: Task 2.3
Likely files: `src/gridguard/ui/app.py`, `src/gridguard/ui/window.py`, `src/gridguard/ui/presentation_model.py`, `tests/ui/test_app.py`, `tests/ui/test_window.py`
Scope: Medium

### Task 3.2 - Implement interactive locality map (complete)

Description: Create a purpose-designed schematic map with clickable locality items, river context, service-tier status, pan/zoom, keyboard selection, and a synchronized detail panel.

Acceptance criteria:

- All seven localities are visible and selectable.
- Map updates for controller and hour changes.
- Service tier is conveyed by label/pattern as well as color.

Verification:

- Red/green GUI tests in `tests/ui/test_map_view.py`.
- Manual keyboard, pan, zoom, and 1366x768 presentation check.

Dependencies: Task 3.1
Likely files: `src/gridguard/assets/map/khartoum-schematic.json`, `src/gridguard/ui/map_asset.py`, `src/gridguard/ui/map_view.py`, `src/gridguard/ui/map_items.py`, `tests/ui/test_map_view.py`
Scope: Medium

### Task 3.3 - Implement live comparison and results views (complete)

Description: Add scenario controls, 24-hour timeline, controller comparison, metric cards, demand/supply chart, frozen result table, and clear provenance/limitation states.

Acceptance criteria:

- Running a scenario updates map, timeline, chart, and locality detail coherently.
- Frozen benchmark values cannot be replaced by live demo values.
- All labels, units, counts, and deltas match the evidence contract.

Verification:

- Red/green tests in `tests/ui/test_scenario_view.py` and `tests/ui/test_results_view.py`.
- Manual end-to-end presentation rehearsal.

Dependencies: Task 3.2
Likely files: `src/gridguard/ui/live_page.py`, `src/gridguard/ui/live_runner.py`, `src/gridguard/ui/results_page.py`, `tests/ui/test_live_page.py`, `tests/ui/test_window.py`
Scope: Medium

### Task 3.4 - Implement methodology view and visual polish (complete)

Description: Add a concise explanation of inputs, methods, reward, metrics, limitations, keyboard guidance, and cohesive visual design suitable for projection.

Acceptance criteria:

- Methodology agrees with the final chapter text and code.
- Contrast, focus visibility, typography, and 1366x768 layout are presentation-safe.
- UI contains no V-number history or unsupported claim.

Verification:

- GUI text-contract tests and manual visual review at 100%, 125%, and 150% Windows scaling.

Dependencies: Task 3.3
Likely files: `src/gridguard/ui/methodology_page.py`, `src/gridguard/ui/theme.py`, `tests/ui/test_window.py`
Scope: Medium

### Checkpoint: Presentation application

- Full GUI test suite passes offscreen.
- Primary presenter flow works offline on the validated NVIDIA GPU.
- Map is interactive and accessible.
- No console errors, broken labels, clipped content, or evidence drift.

## Phase 4: Academic deliverables and release

### Task 4.1 - Write Methodology, Implementation, and Results chapters (complete)

Description: Produce committee-ready chapters grounded in the accepted proposal, clean source, and frozen evidence.

Acceptance criteria:

- Every algorithm, metric, result, and limitation is traceable to code or evidence.
- Results use the final matched-scenario/controller-run terminology.
- Chapters contain no contradictory legacy-experiment narrative.

Verification:

- Documentation claim checker passes.
- Human academic review.

Dependencies: Task 3.4
Likely files: `docs/METHODOLOGY.md`, `docs/IMPLEMENTATION.md`, `docs/RESULTS.md`, `tests/test_document_claims.py`
Scope: Medium

### Task 4.2 - Assemble and visually verify the final report PDF (complete)

Description: Assemble proposal context, methodology, implementation, results, conclusion, limitations, and references into a polished final report.

Acceptance criteria:

- Report includes all proposal deliverables and consistent figures/tables.
- Rendered pages have no clipping, overlap, broken glyphs, or unreadable charts.
- Final report hash is included in the release manifest.

Verification:

- Render every PDF page to PNG and inspect visually.
- Reopen and extract headings/table labels for structural checks.

Dependencies: Task 4.1
Likely files: `docs/REPORT.md`, `output/pdf/gridguard-khartoum-final-report.pdf`
Scope: Medium

### Task 4.3 - Create and verify the academic slide deck (complete)

Description: Create a concise deck and speaker notes aligned with the desktop demonstration and report.

Acceptance criteria:

- Deck covers problem, objectives, method, architecture, demo flow, fixed comparison, results, limitations, and conclusion.
- Every result matches frozen evidence.
- Rendered slides are legible on a projector.

Verification:

- Render every slide and visually inspect.
- Run presentation rehearsal against the desktop app.

Dependencies: Task 4.2
Likely files: `tools/build_presentation.ps1`, `output/presentation/gridguard-khartoum-defense.pptx`
Scope: Medium

### Task 4.4 - Build and smoke-test the Windows application (complete)

Description: Package the final native GUI and add a simple launcher and presentation checklist.

Acceptance criteria:

- Executable starts without network access and passes the presentation laptop's CUDA/PyTorch/CuPy preflight.
- Primary scenario and results flow completes from the packaged build.
- Package excludes caches, source research artifacts, and secrets.

Verification:

- Build with `uv run pyinstaller --noconfirm packaging/gridguard.spec`.
- Launch packaged executable and complete smoke checklist.

Dependencies: Task 4.3
Likely files: `packaging/gridguard.spec`, `packaging/gridguard_launcher.py`, `scripts/build-presentation.ps1`, `scripts/run-presentation.ps1`, `docs/PRESENTATION_CHECKLIST.md`
Scope: Medium

### Task 4.5 - Final audit, release commit, and GitHub publication (complete)

Description: Run the final checks, freeze provenance, verify repository cleanliness, create intentional release commits, push to GitHub, and prepare the reviewed branch/PR.

Acceptance criteria:

- Tests, lint, executable smoke, document claim audit, and evidence verifier pass.
- Git history is atomic, worktree is clean, and no secret or unnecessary artifact is tracked.
- Repository is pushed to the user-approved GitHub visibility and a draft PR is opened if a feature branch is used.

Verification:

- `uv run pytest`
- `uv run ruff check .`
- `uv run gridguard-evaluate --verify-evidence`
- `git status --short --branch`
- GitHub branch and PR inspection

Dependencies: Task 4.4
Likely files: `RELEASE_MANIFEST.json`, `CHANGELOG.md`, `README.md`, `tests/test_release_manifest.py`, `tests/test_document_claims.py`
Scope: Medium

## Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Source benchmark episode wording is ambiguous | High | Rebuild per-scenario evidence and report scenario count separately from controller runs |
| Clean extraction changes behavior | High | Golden parity tests for scenario, action, dispatch, and metrics before UI work |
| Evidence hashes drift after documentation edits | High | Freeze final manifest after report and deck are complete |
| PyTorch/PySide6 packaged application is large | Medium | Strip optimizer state, exclude unused Qt modules, and smoke-test the actual build |
| Presentation GPU or CUDA stack is unavailable | High | Fail-closed launch preflight, explicit diagnostics, and a rehearsal checklist; all assets remain offline |
| Schematic map is mistaken for real topology | Medium | Permanent visible schematic/synthetic disclaimer |
| Academic template requirements arrive late | Medium | Keep report/deck content modular and resolve template question before final rendering |

## Approved delivery decisions

- GitHub visibility: private.
- Report and slide style: neutral academic design until a university template is supplied.
- Map: offline schematic with an explicit non-geographic disclaimer.
- Learned-controller label: PPO throughout the clean project.

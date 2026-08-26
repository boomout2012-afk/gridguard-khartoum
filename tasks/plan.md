# Implementation Plan: GridGuard Khartoum Final Delivery

Status: Active - Phase 6 adversarial hardening and 1.2.0 release

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

## Phase 5: Review remediation (complete)

### Task 5.1 - Establish the review ledger and domain literature (complete)

Description: Preserve every substantive colleague-review point in clean English, record its
verdict and closure evidence, answer the realism portion of research question 3, and position the
project against relevant energy literature without expanding claims.

Acceptance criteria:

- `docs/REVIEW_REMEDIATION.md` contains R01–R14 with criticism, verdict, action, and acceptance
  evidence and binds the raw review snapshot by SHA-256.
- `docs/RELATED_WORK.md` distinguishes contextual literature from project evidence and states that
  the scientific case is synthetic and not utility-calibrated.
- `docs/REFERENCES.md` contains exactly 15 entries, including seven vetted domain sources.

Verification:

- `uv run pytest tests/test_review_remediation.py`

Dependencies: Task 4.5
Likely files: `docs/REVIEW_REMEDIATION.md`, `docs/RELATED_WORK.md`, `docs/REFERENCES.md`
Scope: Small

### Task 5.2 - Publish PPO training provenance (complete)

Description: Recover the selected PPO training record from the pinned research archive and publish
it as a strict, hash-bound YAML artifact that documentation tooling can validate and reviewers can
find directly.

Acceptance criteria:

- Architecture, initialization, PPO updates, validation, selection, seeds, hardware, and convergence
  are explicit and tied to immutable archive hashes.
- The model manifest, source manifest, wheel, and Windows package verify the YAML without adding a
  runtime YAML dependency.

Verification:

- Focused training-provenance, model, provenance, and packaging tests.

Dependencies: Task 5.1
Scope: Medium

### Task 5.3 - Reframe outcomes and accepted reproduction evidence (complete)

Description: Make critical-outage locality-hours, Jain fairness, and utilization the three primary
outcomes; retain episode reward only as a training-objective diagnostic; and preserve frozen numeric
evidence while updating semantic metadata and trust anchors.

Acceptance criteria:

- Metric roles agree across code, artifacts, UI, report, and deck.
- The accepted frozen benchmark and PPO actor remain byte-identical.
- Release verification tests integrity and arithmetic rather than requiring a favorable direction.

Verification:

- Focused evaluation-role, evidence, UI, document, and immutable-hash tests.

Dependencies: Task 5.2
Scope: Medium

### Task 5.4 - Add untouched robustness and forecast evidence (complete)

Description: Evaluate seeds 81001–81010, one-factor ±25% demand-parameter sensitivity, and the
three-hour trailing-mean forecast against one-hour persistence on matched synthetic traces.

Acceptance criteria:

- The supplemental artifact reports exact workload counts, deterministic intervals, sensitivity
  classifications, forecast MAE/RMSE, and bounded synthetic-only claims.
- Two CUDA reproductions are byte-identical and semantic verification fails closed on tampering.

Verification:

- Focused scenario, robustness, forecast, statistics, CLI, and evidence tests.

Dependencies: Task 5.3
Scope: Large

### Task 5.5 - Rebuild and audit the reviewer-facing release (complete)

Description: Apply all verified dispositions to the application and academic package, rebuild the
report, 13-slide deck, and Windows distribution, then refresh provenance in dependency order.

Acceptance criteria:

- Human-facing values use the approved precision and every limitation is consistent across media.
- Every report page and slide is rendered and inspected, the packaged application passes the CUDA
  smoke path, and the final release tree verifies cleanly offline.
- The raw `point.md` snapshot is removed only after R01–R14 acceptance evidence is complete.

Verification:

- Full pytest and Ruff checks, doctor, evidence reproduction/verification, visual inspection,
  packaged smoke test, release-manifest verification, and clean-tree audit.

Dependencies: Tasks 5.2–5.4
Scope: Large

## Phase 6: Adversarial hardening and concrete 1.2.0 release (active)

### Delivery contract

Phase 6 closes the confirmed defects and assurance gaps found by the post-remediation devil's-
advocate audit. It does not add a new controller, field-data claim, or post-proposal experiment.
The corrected evidence semantics and release bindings are published as version 1.2.0 so the 1.1.0
history is preserved rather than silently rewritten.

The work proceeds in four gates:

```text
Audit contract and hermetic gates
  -> scientific invariants, reconstruction, and claim contract
      -> regenerated evidence and reviewer-facing surfaces
          -> hardened package, generated validation, manifest freeze, and publication
```

### Task 6.1 - Freeze the audit ledger and 1.2.0 contract

Description: Record every confirmed finding A01-A24, its severity, required treatment, owner task,
and closure evidence; establish 1.2.0 as the next release; and keep the proposal boundary explicit.

Acceptance criteria:

- A tracked audit ledger maps every A01-A24 finding to exactly one closed disposition: fixed,
  reframed with evidence, bounded limitation, or optional signing hardening.
- `SPEC.md`, version metadata, changelog, and reviewer entry points identify 1.2.0 without rewriting
  the frozen 1.1.0 history.
- `data/model/training-provenance.yaml` is directly linked from reviewer-facing documentation and
  remains the canonical, strict record of architecture, PPO parameters, seeds, hardware, selection,
  and convergence.

Verification:

- Document-claim and training-provenance tests.
- Manual audit-ledger completeness review against A01-A24.

Dependencies: Task 5.5
Likely files: `docs/DEVILS_ADVOCATE_AUDIT.md`, `SPEC.md`, `README.md`, `CHANGELOG.md`,
`pyproject.toml`, `src/gridguard/__init__.py`, `tests/test_document_claims.py`,
`tests/test_training_provenance.py`
Scope: Small

### Task 6.2 - Make build and test gates hermetic

Description: Pin the build backend in the locked development environment, prove the nested wheel
smoke without network access, separate portable tests from tests that require final release
artifacts, and remove the sibling research archive as a required release/runtime dependency.

Acceptance criteria:

- After one locked environment sync, the wheel smoke succeeds with `PIP_NO_INDEX=1`, offline mode,
  no dependency sync, and an empty task-specific uv cache; no network request occurs.
- Portable and `release_artifact` suites each run with zero skips, and missing executables, smoke
  reports, or distribution files fail the release-artifact suite instead of passing conditionally.
- The release build consumes only committed, hash-bound inputs; the sibling `aipower_rl` archive is
  optional for parity revalidation and is never required by the installed offline application.

Verification:

- `uv run --no-sync pytest -q tests/ui/test_map_asset_wheel.py tests/test_ci.py`
- Portable CI selection plus post-package `uv run --no-sync pytest -m release_artifact`.
- Network-denied build test and source-input provenance check.

Dependencies: Task 6.1
Likely files: `pyproject.toml`, `uv.lock`, `.github/workflows/quality.yml`,
`tests/ui/test_map_asset_wheel.py`, `tests/test_ci.py`, `tests/test_release_manifest.py`,
`scripts/build-presentation.ps1`, `scripts/release-integrity.ps1`
Scope: Medium

### Task 6.3 - Enforce matched-scenario integrity and trace identity

Description: Replace shallow immutability claims with an enforced read-only-use contract, detect
controller mutation on CUDA, and bind every paired evaluation row to a canonical scenario trace
fingerprint.

Acceptance criteria:

- `evaluate_scenario_pair` fails closed if either controller mutates demand or supply before the
  matched comparison finishes, while both controllers still receive the same validated scenario.
- Every paired row contains a unique, validated 64-character `scenario_sha256` derived from the
  canonical demand/supply trace and scenario identity.
- Default generation, accepted numeric rows, and the frozen benchmark remain unchanged.

Verification:

- Focused domain/evaluation mutation, fingerprint, ordering, and schema tests on CUDA.
- Golden parity and immutable frozen-benchmark hash tests.

Dependencies: Task 6.1
Likely files: `src/gridguard/domain.py`, `src/gridguard/evaluation.py`,
`tests/test_domain.py`, `tests/test_evaluation.py`, evidence schema fixtures
Scope: Medium

### Task 6.4 - Make sensitivity a supply-isolated demand-allocation intervention

Description: Regenerate each demand sensitivity from locked base draws while preserving the exact
base available-supply trace, so the experiment changes demand allocation rather than demand and
supply together.

Acceptance criteria:

- For all seeds, shortage levels, and 24 variants, base and variant supply tensors are exactly equal
  while the intended allocation parameter and demand distribution change.
- The supply override boundary validates shape, dtype, device, finiteness, non-negativity, and
  feasibility; all power-balance violations remain zero.
- Ordinary scenario generation remains bit-identical; only sensitivity values are permitted to
  change for this correction.

Verification:

- `uv run --no-sync pytest -q tests/test_scenarios.py tests/test_robustness.py`
- Golden default-scenario and power-balance checks.

Dependencies: Task 6.1; may run in parallel with Task 6.3
Likely files: `src/gridguard/scenarios.py`, `src/gridguard/robustness.py`,
`tests/test_scenarios.py`, `tests/test_robustness.py`
Scope: Medium

### Task 6.5 - Reconstruct every reported confidence interval

Description: Make trusted verification independently recompute paired, base, variant, and
difference-in-differences bootstrap endpoints from persisted per-draw inputs using the canonical
CUDA/CuPy implementation.

Acceptance criteria:

- Paired intervals reconstruct from the existing paired rows, and robustness evidence persists a
  strict, hash-bound delta cube with locked shape, order, dtype, and byte encoding.
- Trusted verification recomputes every public mean and interval endpoint within the declared
  tolerance; changing an endpoint while rehashing JSON still fails semantic verification.
- Schema validation remains sign-neutral: adverse or ambiguous results pass when honestly derived,
  while altered inputs, ordering, shapes, hashes, or endpoints fail closed.

Verification:

- Focused evaluation, robustness, CLI, schema-tamper, and independent algorithm-reference tests.
- `uv run --no-sync gridguard-evaluate --verify-evidence` after real CUDA preflight.

Dependencies: Tasks 6.3 and 6.4
Likely files: `src/gridguard/evaluation.py`, `src/gridguard/robustness.py`,
`src/gridguard/cli.py`, focused scientific tests, evidence schema fixtures
Scope: Large

### Task 6.6 - Correct the scientific claim contract and training disclosure

Description: Use bounded terminology for the seed-disjoint holdout and sensitivity intervals, add
the missing behavior-cloning attribution limitation, state the scenario mutation contract exactly,
and make the training YAML unmistakably discoverable and complete.

Acceptance criteria:

- `documented-seed-disjoint holdout` replaces positive `untouched confirmatory` claims and explicitly
  disclaims preregistration, independent custody, and knowledge of undocumented activity.
- Every sensitivity result is labelled descriptive, pointwise 95%, and unadjusted for multiple
  comparisons; no family-wide significance claim is made.
- UI, evidence, report, deck, and training YAML state that no behavior-cloning-only ablation was run,
  so the observed comparison cannot isolate the incremental effect of PPO fine-tuning.

Verification:

- Artifact-schema, training-provenance, UI, report, presentation, and document-claim tests.
- Repository claim scan for prohibited positive wording and legacy metric-role language.

Dependencies: Task 6.5 schema decisions
Likely files: `src/gridguard/robustness.py`, `src/gridguard/ui/presentation_model.py`,
`src/gridguard/ui/results_page.py`, `src/gridguard/ui/methodology_page.py`,
`data/model/training-provenance.yaml`, documentation and claim tests
Scope: Medium

### Task 6.7 - Regenerate, compare, and promote scientific evidence

Description: Reproduce paired and robustness artifacts on the validated presentation GPU, compare
their scientific projections before promotion, refresh all dependent bindings, and demand a second
byte-identical reproduction.

Acceptance criteria:

- Frozen benchmark bytes, paired numeric results, robustness base outcomes, forecast results, seeds,
  counts, and zero balance violations remain unchanged; only paired metadata/fingerprints and the
  corrected sensitivity values may differ.
- Two clean CUDA reproductions of each final artifact are byte-identical and pass semantic
  reconstruction, tamper, source-provenance, and model-provenance verification.
- New schemas, hashes, classifications, and any changed sensitivity conclusions are reviewed before
  promotion; no favorable direction is required.

Verification:

- `gridguard-doctor`, temporary paired/robustness reproductions, scientific-projection comparison,
  `gridguard-evaluate --verify-evidence`, focused tests, Ruff check, and Ruff format check.

Dependencies: Tasks 6.3-6.6
Likely files: `data/evidence/paired-benchmark.json`,
`data/evidence/robustness-supplement.json`, `data/evidence/source-manifest.json`,
`data/model/manifest.json`, `data/model/training-provenance.yaml`, scientific trust anchors
Scope: Large

### Task 6.8 - Put the weak-baseline boundary beside every result and simplify the UI

Description: Add one canonical reviewer-facing Fixed Rotation limitation, display it adjacent to
the frozen result, and decompose the oversized presentation shell and live-scenario page without
changing behavior or moving scientific logic into widgets.

Acceptance criteria:

- Results state visibly and accessibly that Fixed Rotation is supply-blind, Off/Full Service only,
  and cannot select Critical Only, alongside the synthetic decision-support/field-evidence boundary.
- `PresentationWindow`, `LiveScenarioPage`, and extracted components meet the roughly 200-line UI
  boundary while preserving tested public aliases, worker lifecycle, keyboard behavior, and state.
- The 1366x768 screenshots show no movement, clipping, lost accessible text, or live/frozen evidence
  leakage; all scientific calculations remain outside widgets.

Verification:

- Focused results, live-page, window, screenshot, accessibility, Ruff, and format tests.
- Pixel/layout inspection at 1366x768 and Windows scaling settings.

Dependencies: Tasks 6.6 and 6.7
Likely files: `src/gridguard/ui/results_page.py`, `src/gridguard/ui/window.py`,
`src/gridguard/ui/live_page.py`, new focused UI component modules, UI/screenshot tests
Scope: Large

### Task 6.9 - Correct and rebuild the academic claim surfaces

Description: Fix the report citation and module/package descriptions, title-case shortage levels,
and bind the slide narrative and conclusion explicitly to the limited Fixed Rotation comparator and
the final scientific contract.

Acceptance criteria:

- Reference [14] is described accurately; implementation documentation covers robustness,
  provenance, extracted UI modules, and the exact bundled trust-anchor/source set.
- Slide 4 uses Mild, Moderate, and Severe and discloses the comparator design; slide 13 and its notes
  bound the headline to that comparator and do not generalize to stronger baselines.
- Report, deck, notes, screenshots, and all supporting chapters use the final evidence-neutral
  holdout, interval, attribution, and sensitivity wording.

Verification:

- Document-claim, report-builder, presentation-builder, per-slide XML, and note-source tests.
- Rebuild draft report/deck and inspect the affected pages/slides before final release rendering.

Dependencies: Tasks 6.7 and 6.8
Likely files: `docs/REPORT.md`, `docs/IMPLEMENTATION.md`, `docs/RESULTS.md`,
`docs/METHODOLOGY.md`, `docs/REFERENCES.md`, `tools/build_report.py`,
`tools/build_presentation.ps1`, document/report/presentation tests
Scope: Medium

### Task 6.10 - Harden and shrink the Windows package

Description: Clean only validated generated outputs, remove stale 1.0.0 distributions, use an
explicit private smoke argument instead of ambient environment control, deduplicate identical CUDA
DLLs after PyInstaller analysis, and build only the current 1.2.0 release.

Acceptance criteria:

- Cleanup rejects unknown targets and leaves `dist` containing only the current versioned archives,
  `.gitignore`, and `GridGuard Khartoum`; no stale 1.0.0 archive survives.
- Normal launch ignores poisoned smoke/offscreen environment variables, while the explicit frozen
  smoke path writes only to a validated temporary output and succeeds on `cuda:0`.
- No SHA-identical NVIDIA/`torch/lib` DLL pair remains; the package is at least 2,000,000,000 bytes
  smaller than the audited 7,021,231,590-byte tree and is at most 5,100,000,000 bytes.

Verification:

- Packaging unit tests plus inspection of the actual built tree for duplicate hashes and total size.
- Packaged CUDA doctor, PyTorch/CuPy initialization, live simulation, evidence verification, and
  frozen smoke.

Dependencies: Task 6.2 for build mechanics; final package waits for Task 6.9
Likely files: `packaging/gridguard.spec`, `packaging/gridguard_launcher.py`,
`scripts/build-presentation.ps1`, `scripts/run-presentation.ps1`, `tests/test_packaging.py`,
`tests/test_release_manifest.py`, `dist/`
Scope: Large

### Task 6.11 - Generate validation evidence and bind the release manifest

Description: Replace carried-forward PASS claims with fresh machine validation and a complete
hash-bound visual attestation, then reconstruct the release manifest from those records and state
its authenticity boundary honestly.

Acceptance criteria:

- A machine report records commands, exit codes, JUnit totals, zero skips, Ruff/format results,
  doctor/evidence/reproduction hashes, and actual report/deck/notes counts; no PASS value is copied.
- A tracked attestation binds final PDF/PPTX hashes to contiguous reviewed page, slide, and note lists;
  current expected counts are 36 pages and 13 slides/notes, but detected final counts are authoritative.
- The bumped manifest binds both records, hashes every required release file, rejects substitution or
  stale counts, and discloses that the unsigned hashes require an independently trusted Git commit.

Verification:

- Release-validation and release-manifest tamper tests.
- Render and inspect every final report page, slide, speaker-note source, screenshot, and packaged
  smoke report; then run manifest verification from the built tree.

Dependencies: Tasks 6.7, 6.9, and 6.10
Likely files: `tools/release_validation.py`, `tests/test_release_validation.py`,
`docs/release-visual-review.json`, `scripts/release-integrity.ps1`,
`scripts/build-presentation.ps1`, `RELEASE_MANIFEST.json`, release-manifest tests
Scope: Large

### Task 6.12 - Run the final gate and publish the reviewed branch

Description: Rebuild in dependency order, freeze the manifest last, produce atomic reviewed commits,
prove a clean release from the committed tree, and publish the reviewed feature branch to the
approved private GitHub repository.

Acceptance criteria:

- Locked sync, Ruff check/format, full portable and release-artifact pytest selections, CUDA doctor,
  semantic evidence verification, two reproductions, report/deck review, package smoke, and manifest
  verification all pass with zero required skips.
- The worktree is clean, the final manifest source commit is reachable from the published branch,
  and `origin/feature/review-remediation` exactly equals local `HEAD`.
- The release is explicitly unsigned unless the user supplies an external signing identity;
  Authenticode or detached signing remains optional and promotion into `main` remains a separate
  explicit decision.

Verification:

- `uv sync --locked --extra dev --extra docs`
- `uv run --no-sync ruff check .` and `uv run --no-sync ruff format --check .`
- `uv run --no-sync pytest`, post-build `pytest -m release_artifact`, evidence/doctor/reproduction,
  `build-presentation.ps1 -VerifyOnly`, clean-tree, `git push --set-upstream`, and `git ls-remote`.

Dependencies: Tasks 6.1-6.11
Likely files: all final source/artifact changes, `RELEASE_MANIFEST.json`, `tasks/todo.md`, Git history
Scope: Large

### Phase 6 audit closure map

| Finding | Required treatment | Owner task(s) |
|---|---|---|
| A01 unpublished reviewed branch | Publish and verify the exact remote SHA | 6.12 |
| A02 stale/unverified validation block | Generate validation; reconstruct the manifest | 6.11 |
| A03 stale 1.0.0 archives | Validated cleanup and current-only output | 6.10 |
| A04 weak baseline hidden in deck conclusion | Adjacent canonical disclosure in UI/deck/notes | 6.8, 6.9 |
| A05 wheel test reaches unpinned network build backend | Locked, offline, no-sync wheel smoke | 6.2 |
| A06 interval endpoints not independently reconstructed | Persist inputs and recompute all endpoints | 6.5, 6.7 |
| A07 aggregate-only scenario match and shallow mutability | Mutation guards and trace fingerprints | 6.3 |
| A08 sensitivity also changes supply | Hold base supply exactly fixed | 6.4, 6.7 |
| A09 `untouched confirmatory` overclaims custody | Documented-seed-disjoint terminology | 6.6 |
| A10 no multiplicity treatment for 144 summaries | Pointwise/descriptive/no-adjustment labels | 6.6 |
| A11 no behavior-cloning-only ablation | Explicit attribution limitation | 6.6 |
| A12 Results UI chooses the wrong limitation | Canonical adjacent baseline limitation | 6.8 |
| A13 report mischaracterizes reference [14] | Correct citation description | 6.9 |
| A14 module/package documentation is stale | Update exact module and bundle maps | 6.9 |
| A15 slide shortage terms are lowercased | Use approved title-case vocabulary | 6.9 |
| A16 visual attestation has wrong/unbound counts | Complete hash-bound review record | 6.11 |
| A17 oversized UI coordinators violate the working boundary | Extract tested child components | 6.8 |
| A18 oversized scientific modules are a maintenance risk | Extract only new schema/reconstruction helpers while touched; avoid a risky broad rewrite | 6.3-6.5 |
| A19 package contains 2+ GB duplicate CUDA DLLs | Post-analysis deduplication and size budget | 6.10 |
| A20 release tests skip when artifacts are absent | Separate strict post-build release tests | 6.2, 6.11 |
| A21 self-authenticating unsigned manifest | Honest trust model plus published Git anchor; signing optional | 6.11, 6.12 |
| A22 ambient smoke environment can suppress the GUI | Explicit private smoke argument and path validation | 6.10 |
| A23 release path relies on sibling archive | Committed release inputs; archive only for optional parity | 6.2, 6.10 |
| A24 mixed historical/current distribution versions | 1.2.0-only release output | 6.1, 6.10 |

### Phase 6 checkpoints

Scientific checkpoint after Task 6.7:

- Real CUDA/PyTorch/CuPy preflight passes on the presentation GPU.
- Every public interval is reconstructed from persisted inputs.
- Paired and robustness artifacts reproduce byte-identically twice.
- Any changed sensitivity conclusion is reported honestly, without direction-based acceptance.

Claim and artifact checkpoint after Task 6.9:

- Baseline, seed-disjoint, pointwise-interval, attribution, and synthetic-only boundaries agree
  across evidence, YAML, UI, chapters, report, deck, and notes.
- All corrected UI pages fit and remain accessible at 1366x768.
- Draft PDF and deck contain no stale citation, module, packaging, terminology, or conclusion claim.

Release checkpoint after Task 6.12:

- Both test selections have zero failures and zero required skips; Ruff check and format pass.
- Final artifacts, visual attestation, validation report, package tree, and manifest agree by hash.
- No stale release, duplicate CUDA DLL, unbound PASS claim, hidden network dependency, or ambient smoke
  switch remains.
- The clean reviewed branch is present on the approved private remote at the exact local SHA.

### Phase 6 risks and controls

| Risk | Impact | Control |
|---|---|---|
| Corrected supply-isolated sensitivity changes the narrative | High | Accept evidence-neutral results; review deltas before promotion; never gate on favorable direction |
| Evidence schema work accidentally changes the frozen benchmark | High | Immutable source hashes, projection comparison, golden parity, and two reproductions |
| Large scientific refactor creates hidden regressions | High | Test-first focused helpers; no broad rewrite; scientific checkpoint before consumer updates |
| PyInstaller DLL removal breaks CUDA/CuPy loading | High | Remove only byte-identical duplicates after analysis; test the actual package on the RTX GPU |
| Offline build appears green only because of ambient cache | High | Empty task-specific cache, no-index/offline environment, and network-denied test |
| Generated files invalidate the manifest late | High | Stabilize sources and artifacts first; generate validation next; freeze manifest last |
| Signing credentials are unavailable | Medium | Disclose unsigned status and anchor to the private Git commit; add signing only with user-supplied identity |

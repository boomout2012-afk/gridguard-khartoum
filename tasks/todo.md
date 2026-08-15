# GridGuard Khartoum Completion Checklist

## Phase 0 - Scope and provenance

- [x] Approve `SPEC.md` assumptions, claim boundary, architecture, and deliverables.
- [x] Resolve GitHub visibility, university templates, and schematic-map acceptance.
- [x] Freeze proposal-relevant source artifacts and golden parity cases.
- [x] Record source commit and SHA-256 for every migrated artifact.

## Phase 1 - Proposal simulation core

- [x] Define clean locality, sector, shortage, tier, scenario, result, and metric contracts.
- [x] Extract deterministic Mild, Moderate, and Severe scenario generation.
- [x] Add the separate causal next-hour forecast used for decision-support display.
- [x] Implement the 24-hour simulator and power-balance invariants.
- [x] Implement and verify Fixed Rotation.
- [x] Match golden source actions, dispatch, and metrics.

## Checkpoint - Simulation foundation

- [x] Phase 1 tests pass.
- [x] Ruff passes.
- [x] CUDA fixed-rotation scenario runs end to end.
- [x] Golden parity has no unexplained difference.

## Phase 2 - Learned policy and evidence

- [x] Create a safe inference-only PPO artifact without critic or optimizer state.
- [x] Match CUDA policy logits and actions exactly against the trusted source checkpoint.
- [x] Run matched PPO and Fixed Rotation evaluation.
- [x] Resolve base-draw, scenario-condition, and controller-evaluation wording.
- [x] Calculate paired cluster-bootstrap deltas and confidence intervals.
- [x] Create a fail-closed evidence manifest and verifier.

## Checkpoint - Scientific foundation

- [x] Both controllers run on CUDA across all shortage levels.
- [x] Proposal metrics reproduce within reviewed tolerance.
- [x] Evidence tamper tests pass.
- [x] Primary claims and limitations are frozen for UI work.

## Phase 3 - Native desktop presentation

- [x] Build the PySide6 application shell and presentation state model.
- [x] Add offline interactive schematic map with all seven localities.
- [x] Add click, keyboard selection, pan, zoom, and hour synchronization.
- [x] Add scenario controls and matched live comparison.
- [x] Add demand/supply, service-tier, and metric visualizations.
- [x] Add frozen Results view with provenance and limitations.
- [x] Add Methodology view with consistent terminology.
- [x] Verify 1366x768 layout and Windows display scaling.

## Checkpoint - Presentation application

- [x] GUI tests pass in offscreen mode.
- [x] Primary presenter flow works without internet after the CUDA/PyTorch/CuPy preflight.
- [x] UI is keyboard accessible and does not rely on color alone.
- [x] No evidence value can drift through live demo state.

## Phase 4 - Academic deliverables and release

- [x] Write `docs/METHODOLOGY.md`.
- [x] Write `docs/IMPLEMENTATION.md`.
- [x] Write `docs/RESULTS.md`.
- [x] Assemble and visually verify the final report PDF.
- [x] Create and visually verify the slide deck and speaker notes.
- [x] Build and smoke-test the Windows desktop application.
- [x] Run full tests, lint, claim audit, and evidence verification.
- [x] Freeze the final provenance manifest after all documents are stable.
- [x] Review tracked files and secrets; confirm the intended release tree is clean.
- [ ] Commit in reviewed atomic increments.
- [ ] Push to the approved GitHub repository and open the appropriate PR.

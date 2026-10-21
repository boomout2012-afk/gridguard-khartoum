# GridGuard Khartoum Completion Checklist

## October 2026 source refresh

Release identity remains 1.2.0. This section records work completed before the new source freeze;
the older phase checkboxes below preserve their historical planning state. The detailed findings
and evidence are in [docs/GRADUATION_REVIEW.md](../docs/GRADUATION_REVIEW.md).

- [x] Review the incoming Abdalhay update and identify the stale source/artifact bindings.
- [x] Correct academic overclaims and disclose partial dispatch-optimization coverage.
- [x] Reject incorrect integer-metadata types and reproduce the provenance refresh twice on CUDA;
  preserve all scientific results and persisted arrays.
- [x] Improve the native presentation interface, timeline playback, and power trace; pass the
  37 focused UI tests at the supported 1366 x 768 minimum.
- [x] Capture and inspect all four final pages using a real CUDA scenario.
- [x] Rebuild and inspect all 39 report pages with the verified PDF source-bundle binding.
- [x] Inspect all 13 deck slides and their speaker notes; rebuild and visually recheck the
  final screenshot aspect-ratio correction.
- [x] Fix startup diagnostic persistence and distinguish historical CLI verification from
  current-source release acceptance.

Final acceptance occurs after source freeze. Read the actual source/GPU test results, zero-skip
counts, reproduction identities, and tested source SHA from `docs/release-machine-validation.json`;
read package, smoke, artifact, and source bindings from `RELEASE_MANIFEST.json`. Require the clean
current-source `scripts/build-presentation.ps1 -VerifyOnly` gate and final offline rehearsal.
The final handoff records the exact local/private-remote SHA evidence. These post-freeze records
establish completion; do not edit source checkboxes afterward merely to restate their results.

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
- [x] Create reviewed release commits from sanitized history.
- [ ] Publish the current reviewed private branch and verify its remote SHA; the adversarial audit
  found that the local feature branch had no upstream.

## Phase 5 - Review remediation

- [x] Preserve and dispose R01–R14 in a clean review ledger bound to the raw review SHA-256.
- [x] Add bounded Related Work and seven vetted domain references, for 15 references total.
- [x] Publish and strictly validate the hash-bound PPO training-provenance YAML.
- [x] Reclassify the training-objective score as diagnostic and enforce the three primary outcomes.
- [x] Preserve byte-identical frozen benchmark and PPO actor trust anchors.
- [x] Generate and verify the ten-seed seed-disjoint supplemental evidence; Phase 6 corrects the
  overstrong historical `untouched confirmatory` label.
- [x] Generate and verify all 24 one-factor ±25% sensitivity variants.
- [x] Compare the trailing-mean forecast with one-hour persistence using MAE and RMSE.
- [x] Apply R01–R14 language, precision, and claim boundaries across UI, report, and deck.
- [x] Rebuild and visually inspect every report page and all 13 slides.
- [x] Package the YAML and robustness evidence in wheel and PyInstaller distributions.
- [x] Pass full tests, Ruff, CUDA doctor, evidence reproduction, package smoke, and release audit.
- [x] Remove `point.md` only after all remediation evidence is complete and verified.

## Phase 6 - Adversarial hardening and concrete 1.2.0 release

### Contract and hermetic gates

- [x] Record A01-A24 in a tracked audit ledger with owner, disposition, and closure evidence.
- [x] Set 1.2.0 consistently in specification, package metadata, changelog, and release paths.
- [x] Link the strict training-parameter YAML directly from README, report, and packaged reviewer assets.
- [x] Pin the build backend and prove the wheel smoke with no index, no sync, and no network.
- [ ] Split portable tests from strict post-build `release_artifact` tests; require zero skips in both.
- [x] Remove the sibling research archive as a required release-build or runtime input.

### Scientific assurance

- [x] Add CUDA mutation guards around matched controller evaluation.
- [x] Add and validate a unique canonical scenario SHA-256 on every paired row.
- [x] Hold the base available-supply trace exactly fixed for every demand-allocation sensitivity.
- [x] Persist strict per-draw robustness inputs and reconstruct every reported bootstrap endpoint.
- [x] Rename the holdout to documented-seed-disjoint and state its custody limitations.
- [x] Label all sensitivity intervals pointwise, descriptive, and unadjusted for multiplicity.
- [x] State that no behavior-cloning-only ablation isolates PPO fine-tuning's incremental effect.
- [x] Separate the selected seed-43 run workload from the aggregate three-seed candidate experiment
  in the strict YAML, UI, report, deck, and reviewer entry points.
- [x] Classify direct, reconstructed, derived, and unavailable historical training facts; retain
  exact promotion gates only as historical model-selection evidence.
- [x] Reproduce paired and robustness evidence twice on CUDA and require byte-identical outputs.
- [x] Confirm frozen benchmark, paired values, robustness base values, forecast values, counts, seeds,
  and zero balance violations remain unchanged; review sensitivity-only changes without a direction gate.

### Claims, UI, report, and deck

- [x] Put the supply-blind Fixed Rotation limitation—uses only Off and Full Service and cannot
  select Critical Only—beside Results.
- [x] Split `PresentationWindow` and `LiveScenarioPage` into tested focused components.
- [x] Re-run keyboard, accessibility, live/frozen separation, and 1366x768 visual tests.
- [x] Correct reference [14], robustness/UI module maps, and bundled provenance-source documentation.
- [x] Use Mild, Moderate, and Severe consistently on slide 4.
- [x] Bound slide 13 and its notes explicitly to the weak Fixed Rotation comparator.
- [x] Apply seed-disjoint, interval, attribution, and synthetic-only wording across every claim surface.

### Windows release and provenance

- [x] Safely remove only known stale 1.0.0/generated outputs and emit current 1.2.0 artifacts only.
- [x] Replace ambient smoke/offscreen control with an explicit private argument and validated output path.
- [x] Remove byte-identical CUDA DLL duplicates after PyInstaller analysis.
- [x] Keep the final package at or below 5,021,231,590 bytes and prove all GPU smoke paths still work.
- [ ] Generate a fresh machine validation report from actual commands, JUnit totals, hashes, and counts.
- [x] Render and inspect every final report page, slide, note source, and screenshot.
- [x] Bind a complete visual-review attestation to the final PDF/PPTX hashes.
- [ ] Rebuild the release manifest from validation evidence; disclose the unsigned Git-anchor trust model.

### Final proof and publication

- [ ] Pass locked sync, Ruff check/format, full pytest, release-artifact pytest, CUDA doctor,
  evidence reconstruction, two reproductions, packaged smoke, and manifest verification with zero skips.
- [ ] Confirm the release tree contains no stale version, duplicate CUDA DLL, cache, secret, or unknown file.
- [ ] Freeze the manifest last, create reviewed atomic commits, and finish with a clean worktree.
- [ ] Push `feature/review-remediation`, set its upstream, and verify the remote SHA equals local `HEAD`.
- [ ] Update this checklist to complete only after every closure artifact and remote proof exists.

# Presentation checklist

The October 2026 source refresh is not certified by the historical August release records.
Complete the current source/artifact sequence in [GRADUATION_REVIEW.md](GRADUATION_REVIEW.md)
before using this checklist for final rehearsal. A passing historical-record verifier does not
substitute for a current source snapshot, fresh CUDA tests, or inspection of regenerated artifacts.

## Build and frozen-runtime validation

- Finish source and document changes, capture the real CUDA scenario, render and inspect every
  report page and slide, and review speaker notes before freezing the new source snapshot.
- Generate the fresh exact test collection, zero-skip JUnit results, and both byte-identical
  evidence reproductions; refresh and commit the machine record against the actual tested source
  commit. `scripts/build-presentation.ps1` verifies this record but does not generate it.
- For this source refresh, use `scripts/build-presentation.ps1 -RefreshManifest` after the
  machine record is committed, then commit the manifest before the clean `-VerifyOnly` gate.
- Run `scripts/build-presentation.ps1` from the repository before copying the package.
- After finalizing `RELEASE_MANIFEST.json`, run
  `scripts/build-presentation.ps1 -VerifyOnly` to bind the existing executable,
  distribution, runtime, and frozen-smoke report without rebuilding.
- Confirm the normal package build succeeds without the sibling research archive and that the
  committed `docs/release-machine-validation.json` records zero skipped selected source tests in
  its `junit` summary. The retained transient JUnit input is
  `tmp/release-validation/release-tests.xml`.
- Confirm that `source.git_commit` and the `source.uv_lock` binding in that machine record exactly
  match `RELEASE_MANIFEST.json`. The named Git commit is the tested source snapshot; the later
  machine-record and manifest commits are its non-self-referential attestation envelope.
- Treat source-archive parity as an optional audit only. When the read-only archive is available,
  invoke `scripts/build-presentation.ps1 -StrictSourceParity` with explicit
  `-SourceRepositoryPath` and `-SourceCheckpointPath` values; never make it a runtime or normal
  package-build dependency.
- Require both final lines: `Presentation package ready` and
  `Frozen presentation smoke: PASS`. The build must not be used if either is missing.
- Keep `build/gridguard/frozen-smoke.json` with the presentation backup and confirm it names
  the expected NVIDIA GPU, PyTorch 2.8.0 CUDA 12.8 build, and supported CuPy version.
- Confirm the smoke report records the deterministic `moderate` / `50002` live CUDA scenario,
  both `PPO` and `Fixed Rotation`, 24 frames per controller, finite values, and successful
  installation in the Live Scenario page. Its expected and rendered results-table row counts
  must both be three.
- Compare the distribution byte count and file count with the current `RELEASE_MANIFEST.json`;
  copy the complete one-directory package rather than the executable alone.

## Before leaving for the presentation

- Use the presentation laptop with the validated NVIDIA GPU and current driver.
- Keep the full `dist/GridGuard Khartoum` directory together; the executable is not standalone.
- Copy the complete distribution directory to a second offline drive as a backup.
- Keep the final report PDF and slide deck beside the distribution directory.
- Keep `data/model/training-provenance.yaml` available for reviewer questions; its SHA-256 is
  `d2568bce26e9221afa8c266113a704f6fbf2370d3bc949f5aa6c72a7411ba1a8`.
- Connect power and disable battery-saving mode before opening GridGuard.
- Run `scripts/run-presentation.ps1` once and confirm the startup trust gate passes.

## Five-minute technical rehearsal

- Open each tab and confirm no text is clipped at the presentation resolution.
- On Live Scenario, select a shortage level and seed, then run the GPU simulation.
- Change controller and hour and confirm the map and locality detail update together.
- Confirm the real OCHA/HDX locality outlines and Natural Earth White Nile, Blue Nile, and Nile
  context are visible, then state that they are presentation context only and not
  electrical-network topology.
- Select localities with both the map and synchronized locality list; confirm selection, text,
  and visible keyboard focus stay aligned.
- Select at least two localities and check demand, served power, critical service, and sectors.
- Read the **Live demonstration** aggregate rows for PPO and Fixed Rotation, including Served
  energy, Critical service, Fairness, Curtailed energy, and Power-balance violations. State that
  these values describe the selected run and do not replace frozen Results evidence.
- Open Results and state that every delta is `PPO - Fixed Rotation`.
- State both evidence roles precisely. Accepted reproduction: 96 independent base draws, 288
  matched scenario conditions, and 576 controller-scenario evaluations; its seeds participated in
  model selection. Documented-seed-disjoint holdout: seeds 81001-81010, 320 independent base draws, 960
  matched scenario conditions, and 1,920 controller-scenario evaluations.
- State that the holdout was not preregistered or independently custodied and cannot rule out
  undocumented activity.
- State that sensitivity intervals are descriptive, pointwise 95%, and unadjusted for multiple
  comparisons.
- State that no behavior-cloning-only ablation isolates the incremental effect of PPO fine-tuning.
- State that the three primary outcomes are critical-outage locality-hours, fairness, and
  utilization; power-balance violation is separate safety, and the training-objective score is
  diagnostic.
- State that the demand projection is display-only and is not PPO input. Its synthetic MAE/RMSE
  comparison uses one-hour persistence on the same target hours.

## Scientific wording to preserve

- Say `PPO`, never an earlier experimental training label.
- Say `Fixed Rotation`, not an undefined baseline.
- Say `CRITICAL ONLY` for the middle service tier.
- Say the results are simulator evidence for the locked protocol, not field deployment evidence.
- Say the historical artifact stored batch summaries; paired episode rows and confidence
  intervals were regenerated from the audited simulator and frozen PPO checkpoint.
- Say sign-neutral integrity validation can accept a well-formed adverse result; no favorable
  interval is required for verification, display, packaging, or release.
- Say Fixed Rotation is a transparent but supply-blind baseline limited to Off and Full Service;
  it cannot select Critical Only. Do not generalize the comparison to every rule-based controller.
- Say Adam optimized PPO parameters historically, while CUDA accelerates computation; no online
  mathematical dispatch optimizer is implemented.
- Say the `3^7 = 2,187` joint-action head has `3^N` scaling, performance relative to the global
  optimum is unknown, and the evaluated domain is shortage-only.
- If asked for training settings, open `data/model/training-provenance.yaml`: the selected seed-43
  run used 600 PPO iterations, 153,600 episodes, and 3,686,400 transitions; the three-seed candidate
  experiment totaled 1,800 iterations, 460,800 episodes, and 11,059,200 transitions. Hardware and
  convergence are selected-run facts; historical promotion gates are not independent headline
  evidence.
- Say the map uses source-pinned administrative boundaries and Nile context for presentation
  only; never describe polygons as feeders, service territories, or measured grid data.

## If the app refuses to start

Do not bypass the gate. Record the message and retain `build/gridguard/launch-diagnostic.json`
when launching through `scripts/run-presentation.ps1`. The application records `FAILED` before
showing a blocked-startup dialog and `READY` after opening the verified window; `STARTED` alone
means startup is pending. Confirm the NVIDIA driver is active and use the report/slides backup
if the gate still fails. A blocked startup is safer than presenting unverified or CPU-fallback
results.

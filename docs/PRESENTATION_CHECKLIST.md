# Presentation checklist

## Build and frozen-runtime validation

- Run `scripts/build-presentation.ps1` from the repository before copying the package.
- After finalizing `RELEASE_MANIFEST.json`, run
  `scripts/build-presentation.ps1 -VerifyOnly` to bind the existing executable,
  distribution, runtime, and frozen-smoke report without rebuilding.
- Keep the trusted source archive beside this repository as `aipower_rl`, or set
  `GRIDGUARD_SOURCE_REPOSITORY` and `GRIDGUARD_SOURCE_CHECKPOINT` explicitly. The release
  gate must report zero skipped tests in `build/release-tests.xml`.
- Require both final lines: `Presentation package ready` and
  `Frozen presentation smoke: PASS`. The build must not be used if either is missing.
- Keep `build/gridguard/frozen-smoke.json` with the presentation backup and confirm it names
  the expected NVIDIA GPU, PyTorch 2.8.0 CUDA 12.8 build, and supported CuPy version.
- Confirm the smoke report records the deterministic `moderate` / `50002` live CUDA scenario,
  both `PPO` and `Fixed Rotation`, 24 frames per controller, finite values, and successful
  installation in the Live Scenario page. Its expected and rendered results-table row counts
  must both be seven.
- The validated distribution is a one-directory package of 8,021,341,500 bytes (about
  7.47 GiB); copy all 4,647 files rather than the executable alone.

## Before leaving for the presentation

- Use the presentation laptop with the validated NVIDIA GPU and current driver.
- Keep the full `dist/GridGuard Khartoum` directory together; the executable is not standalone.
- Copy the complete distribution directory to a second offline drive as a backup.
- Keep the final report PDF and slide deck beside the distribution directory.
- Connect power and disable battery-saving mode before opening GridGuard.
- Run `scripts/run-presentation.ps1` once and confirm the startup trust gate passes.

## Five-minute technical rehearsal

- Open each tab and confirm no text is clipped at the presentation resolution.
- On Live Scenario, select a shortage level and seed, then run the GPU simulation.
- Change controller and hour and confirm the map and locality detail update together.
- Select at least two localities and check demand, served power, critical service, and sectors.
- Open Results and state that every delta is `PPO - Fixed Rotation`.
- State the evaluation count precisely: 96 independent base draws, 288 matched scenario
  conditions, and 576 controller-scenario evaluations.
- State that the demand projection is display-only and is not PPO input or evidence.

## Scientific wording to preserve

- Say `PPO`, never an earlier experimental training label.
- Say `Fixed Rotation`, not an undefined baseline.
- Say `CRITICAL ONLY` for the middle service tier.
- Say the results are simulator evidence for the locked protocol, not field deployment evidence.
- Say the historical artifact stored batch summaries; paired episode rows and confidence
  intervals were regenerated from the audited simulator and frozen PPO checkpoint.

## If the app refuses to start

Do not bypass the gate. Record the message, restart the laptop once, confirm the NVIDIA driver
is active, and use the report/slides backup if the gate still fails. A blocked startup is safer
than presenting unverified or CPU-fallback results.

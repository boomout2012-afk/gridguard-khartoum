# Graduation readiness review

Review date: 2026-10-21. Incoming source revision:
`85917d5f082c9c33d6d5010e2ceb0a30052b4669` (Abdalhay Abass).

The findings below describe the incoming October source; the remediation table records the work
completed before the new source freeze. Final acceptance is established afterward by the matching
machine record, release manifest, current-source gate, and final handoff described below. This
source document is not rewritten after freeze to claim those later results. The accepted proposal
and `SPEC.md` remain the scope authority. Release identity remains 1.2.0.

## Confirmed findings

| ID | Priority | Finding | Evidence and consequence | Closure requirement |
| --- | --- | --- | --- | --- |
| G01 | Release blocker | The release source snapshot is stale. | `RELEASE_MANIFEST.json` and `docs/release-machine-validation.json` identify source commit `c95135a625c81b207684abfdf41fa20a0026d3fa`. `Assert-ReleaseSourceSnapshot` rejects the incoming revision because `docs/METHODOLOGY.md`, `docs/PROPOSAL_TRACEABILITY.md`, `docs/REPORT.md`, `src/gridguard/ui/methodology_page.py`, and `tools/report_results_contract.py` changed after that snapshot. | Finish changes, freeze a new tested source commit, then regenerate the machine record and release manifest against it. |
| G02 | Release blocker | The report, deck, screenshots, and packaged application have not been refreshed for the incoming changes. | The existing PDF, deck, and executable are the August 27 artifacts. Their hashes still match the historical visual record and release manifest. Commit `85917d5` changes source-text hash pins only; it does not regenerate or review the resulting artifacts. The historical methodology screenshot/package cannot show the October methodology-page changes. | Recapture the final UI, rebuild the academic artifacts and Windows application, inspect every PDF page and slide, review speaker notes and screenshots, and refresh their bindings. |
| G03 | Medium | Historical verification commands can be mistaken for current-source readiness. | Both `tools/release_validation.py --verify` and `tools/machine_validation.py --verify` pass at the incoming revision. The former checks the recorded artifact identities; the latter checks the historical record and its committed inputs. Neither checks the current source tree against the recorded source commit. The full build does perform that additional check. | Clearly distinguish historical verification from current readiness in reviewer documentation, and require the existing source-snapshot gate at final release. Preserve self-contained historical verification for source archives. |
| G04 | Medium | The presentation launch diagnostic does not receive application failures. | `scripts/run-presentation.ps1` sets `GRIDGUARD_STARTUP_DIAGNOSTIC`, but no Python code reads it. `src/gridguard/ui/app.py` shows a blocking dialog and writes only to `sys.stderr`. The windowed executable has no console streams. If the dialog remains open beyond the launcher's eight-second observation, the wrapper records `RUNNING`; after dismissal it has stopped observing the process. | Persist a structured startup failure from the application before showing the dialog, preserve the underlying error/cause, and keep the wrapper from overwriting an application failure with `RUNNING`. Add a behavioral regression test with no console streams. |
| G05 | High | Incoming academic prose overstates what matching and the proposal establish. | `docs/METHODOLOGY.md` said generator assumptions largely cancel and internal validity is preserved; shared misspecification can affect controllers differently. `docs/REPORT.md` and traceability used confirmatory wording and treated dispatch optimization as outside the proposal, although the proposal lists optimization among methods to consider. | Bound conclusions to the simulator, state only that no utility measurements were supplied, describe the documented-seed-disjoint comparison accurately, and record dispatch-optimization coverage as partial rather than completed. |
| G06 | High | Persisted scientific-array metadata accepts incorrect JSON types. | `_decode_cube_payload` in `src/gridguard/robustness_inference.py` used ordinary equality, which accepts boolean/integer/float equivalence in integer metadata and shape fields. A malformed artifact could pass the declared metadata contract. | Require exact metadata types recursively; retain the original computation, metrics, and arrays; reproduce and review the changed implementation provenance. |
| G07 | Medium | Documentation screenshot values were constructed separately from the scientific runner. | `tools/capture_screenshots.py` previously constructed its own service tiers, power values, and episode summaries while labeling the comparison with a CUDA device. The screenshot state disclosed that it was a fixture, but it was not an actual PPO run. | Capture the real canonical CUDA scenario (`moderate`, seed `50002`) through `run_live_scenario`, reuse that result across the four captures, and retain a clear live-versus-frozen distinction. |
| G08 | Medium | Source-text hash pins do not prove that a committed PDF was rebuilt. | The incoming commit updated Markdown hash pins while the old PDF remained byte-identical. Its artifact hash alone could not reveal correspondence with changed source text. | Embed the normalized source-bundle identity into PDF metadata and verify it against the current rendered Markdown sources; reject missing, malformed, or stale metadata. Rebuild and inspect the report afterward. |
| G09 | Low | Forecast-error values on slide 10 lack their power unit. | The forecast comparison presents MAE/RMSE for system demand without making MW explicit in the slide subtitle. | Include MW in the forecast slide and verify the regenerated slide visually. |

G04 is a diagnostic defect, not evidence that the existing trust checks permit execution after
failure. The application still refuses to open the main window after a recognized failed gate.
PyInstaller documents the unavailable standard streams for Windows windowed executables in its
[common issues guide](https://pyinstaller.org/en/stable/common-issues-and-pitfalls.html#sys-stdin-sys-stdout-and-sys-stderr-in-noconsole-windowed-applications-windows-only).

G04 source remediation is implemented. The application atomically records `FAILED`, the error
type, and its underlying cause before opening the blocking dialog. It records `READY` only after
the verified main window opens. Missing console streams and an unwritable diagnostic path do not
mask the original failure. The launcher preserves these application records and leaves startup
pending when no readiness signal exists; process survival alone is no longer called `RUNNING`.
Focused application and PowerShell regression tests passed, including the reproduced blocked-dialog
case. The changed executable still requires the final package build and frozen-runtime check.

## Current remediation evidence

This table distinguishes source-level progress from final release acceptance. Test counts refer
to the focused runs performed during this review, not to a new complete release-suite result.

| Finding | Evidence recorded during the October review | Final acceptance evidence |
| --- | --- | --- |
| G01-G02 | Historical mismatch reproduced. All four final real-CUDA screenshots and all 39 rebuilt PDF pages have been inspected; the PDF includes its verified source-bundle identity. All 13 deck slides and their notes have been inspected; the final screenshot aspect-ratio correction was rebuilt and visually rechecked. | The new source commit, visual record, machine record, Windows package, and release manifest must agree. |
| G03 | Reviewer documentation and the visual/machine `--verify` success messages explicitly identify historical record/artifact verification and point to `scripts/build-presentation.ps1 -VerifyOnly` for current-source release. The existing source-snapshot policy is unchanged. | Clarity correction implemented; execute the current-source and final package gates. Historical verifier passes alone remain insufficient. |
| G04 | Failing regressions reproduced before implementation; 18 focused application/PowerShell tests passed. Failures are recorded before the dialog, no-console and unwritable destinations preserve the original error, and the wrapper no longer invents a ready state. | Validate the rebuilt executable and launcher. |
| G05 | Methodology, report, and proposal traceability corrected; 33 focused academic/document tests passed. The rebuilt report and deck have been inspected. No new comparator or field-data claim was introduced. | Reviewed artifact hashes in the visual record; partial optimization-method coverage remains a disclosed limitation. |
| G06 | Exact recursive metadata-type comparison implemented; 20 regression cases passed. Two real CUDA candidate reproductions are byte-identical to each other, and their scientific values/arrays are unchanged from the historical supplement. | Final evidence/source gates and machine-record refresh for the changed provenance bindings. |
| G07 | Screenshot capture delegates to the real CUDA runner and labels the live demonstration. Custom screenshot-only dispatch/metric calculations have been removed. All four final 1366 x 768 captures have been inspected. | Exact screenshot hashes in the final visual record. |
| G08 | The rebuilt 39-page PDF embeds its verified normalized source-bundle identity, and every page has been inspected. Regression coverage checks metadata round trips and rejects missing/malformed/stale bindings. | Exact PDF hash and page count in the final visual record, plus the committed-PDF source-binding test. |
| G09 | Deck builder subtitle states MW; a regression checks the unit. Slide 10 was inspected with all 13 slides and their notes. | Final corrected deck and note identities in the visual record. |

The GUI redesign, timeline playback, and 24-hour power trace are source-complete. The UI review
reports 37 passing focused tests, including clipping checks at the minimum supported 1366 x 768
size. The former 1024 x 650 minimum could not contain the presentation layout; the application
now enforces 1366 x 768 and exposes an F11 fullscreen hint. The final four screenshots have been
inspected. Packaged-runtime rehearsal remains governed by the final release gates; focused
source tests and screenshots do not certify the rebuilt executable.

The corrected slide was rechecked; the other twelve rendered slides and all slide/note text
remained unchanged. Once the source is frozen, final acceptance is read from
`docs/release-machine-validation.json`, `RELEASE_MANIFEST.json`, the successful current-source
`scripts/build-presentation.ps1 -VerifyOnly` gate, and the final handoff. They must identify the
new tested source and the reviewed artifacts. Earlier findings or pre-freeze task checkboxes do
not override these later attestations, and their existence alone does not establish a pass.

The G06 provenance comparison is retained at
`tmp/graduation-review/robustness-numeric-comparison.json`. Both candidate reproductions have
SHA-256 `6b56d149331ee191302cd9e6f22365988db0ae301f9441190b61214acf832e48`. The historical supplement
has SHA-256 `fabb5a2f13ab31cd51f641f4e6a6db3cce8a28a6e2c4fa144612f634bad0beb1`. Comparison confirms
that the only JSON difference is the implementation-source binding for
`src/gridguard/robustness_inference.py` under inference provenance, changing its hash and size.
All scientific sections, numeric results, persisted cubes, and protocol fields remain identical.
The accepted paired benchmark, frozen aggregate benchmark, PPO actor, model manifest, and training
provenance retain their original bytes. Dependent source-manifest and document pins are refreshed
to identify the new, reproduced provenance; that is not a replacement with demonstration values.

## Read-only checks performed during this audit

| Check | Observed result | Meaning |
| --- | --- | --- |
| `uv run --no-sync python tools/release_validation.py --verify` | Passed | Existing recorded artifacts still match the historical visual-review bindings. It does not repeat visual inspection or prove correspondence with changed Markdown/UI source. |
| `uv run --no-sync python tools/machine_validation.py --verify` | Passed | The historical machine record and bound committed inputs are internally consistent. It does not rerun its commands or validate the present source snapshot. |
| Dot-source `scripts/release-integrity.ps1`, then run `Assert-ReleaseSourceSnapshot` against `RELEASE_MANIFEST.json` | Failed with the five changed paths listed in G01 | The incoming source is not the source certified by the existing package. This is the correct fail-closed result. |
| Inspect the recorded historical source-test summary | 1,136 tests, zero failures, errors, or skips | Historical evidence only; the complete current source and GPU gate must run after the final changes. |

These read-only results describe the incoming revision before remediation. Subsequent provenance
and source changes require fresh verification; the earlier passes must not be reused as evidence
for the changed tree. No final release certification or visual-inspection attestation is made by
this document.

## Proposed focused regression coverage

These changes should preserve the existing scientific evidence and unsigned provenance boundary.

- For G03, retain the existing `Assert-ReleaseSourceSnapshot` policy and tests for stale or dirty
  source versus permitted later attestation files. Invoke that gate as part of final release;
  a second independent source-validation policy is not required for this remediation.
- For G04, inject a trust-gate error, set both console streams to `None`, and verify that the
  diagnostic contains the error before the dialog opens, the main window is never constructed,
  and startup returns a nonzero code. Verify that an unwritable diagnostic location cannot mask
  the original failure. Exercise wrapper behavior when the application records a failure while
  its dialog is still open.
- For G02, bind rendered artifacts to their reviewed input identities as well as to their own
  output hashes. A changed report source or screenshot source must invalidate readiness until
  the artifact and review record are refreshed. Hash updates by themselves are not review.

## Required closure sequence

The build script verifies an existing machine record; it does not create one. A new source
snapshot therefore requires the following order. Do not begin the expensive package build while
source or document review is still changing its inputs.

1. Complete the scientific, product, UI, and academic corrections. Preserve scientific values,
   model, proposal, metric definitions, and approved scope. Any implementation-provenance refresh
   must retain the dual-reproduction and numeric-comparison evidence described above. Run focused
   regression tests and Ruff for each behavior change.
2. Capture all three current application pages at 1366 x 768 with
   `uv run --no-sync python tools/capture_screenshots.py`. Inspect the screenshots, loading/error
   states, keyboard navigation, locality selection, timeline playback, and map pan/zoom. Captures
   now use the real CUDA demonstration and remain visibly separate from frozen evidence.
3. Build the report with `uv run --no-sync python tools/build_report.py`. Build the deck using
   `powershell -NoProfile -File tools/build_presentation.ps1` with Microsoft PowerPoint
   installed. Render and inspect every page and slide, and read every speaker note. Correct any
   defect before proceeding.
4. Only after that inspection, refresh the visual record with
   `uv run --no-sync python tools/release_validation.py --refresh --confirm-agent-content-and-visual-inspection-completed`.
   Commit the reviewed code, documentation, screenshots, report, deck, and visual record as the
   final source snapshot. Record its full commit ID. The machine record and release manifest are
   the only permitted subsequent attestation changes.
5. On the presentation GPU, execute the exact commands returned by
   `tools.machine_validation.expected_machine_validation_commands`, preserving their actual exit
   codes. They cover locked offline sync, Ruff lint/format, CUDA doctor, evidence verification,
   both byte-identical evidence reproductions, exact pytest collection, and the full source
   suite selected by `not source_archive and not release_artifact`. Use the canonical transient
   paths below and require zero skipped selected tests.
6. Refresh `docs/release-machine-validation.json` with `tools/machine_validation.py --refresh`,
   the actual source commit, exact retained input paths, and the actual nine gate exit codes.
   Verify it with `--verify --verify-inputs`, then commit that record. Do not substitute old test
   results or assumed zero exit codes.
7. Run `scripts/build-presentation.ps1 -RefreshManifest`. It reruns the source gates, creates a
   verified source candidate, builds PyInstaller plus wheel/sdist, checks distribution policy,
   runs a real frozen CUDA smoke scenario, rechecks machine/visual records, refreshes manifest
   bindings, checks the source snapshot, and runs the post-build `release_artifact` suite with
   zero skips. Require both final success messages.
8. Commit the manifest, then run `scripts/build-presentation.ps1 -VerifyOnly` from the clean
   worktree. Rehearse the packaged application offline on the presentation laptop, using Mild,
   Moderate, and Severe scenarios and both controllers. Verify the full interaction flow and
   keep the complete distribution together with the report/deck backup.
9. Record the final reviewed source and handoff commit IDs and the fresh test/verification
   results. Publish the private branch only after all final artifacts are verified. Preserve
   the sibling research repository as read-only.

Canonical transient inputs:

| Purpose | Path |
| --- | --- |
| Exact selected test collection | `tmp/release-validation/release-tests.collection.json` |
| Source/GPU JUnit results | `tmp/release-validation/release-tests.xml` |
| Accepted reproduction | `tmp/release-validation/paired-benchmark-reproduced.json` |
| Robustness reproduction | `tmp/release-validation/robustness-supplement-reproduced.json` |
| Frozen executable smoke report | `build/gridguard/frozen-smoke.json` |
| Post-build JUnit results | `tmp/release-validation/release-artifact-tests.xml` |

The package identity is currently hard-coded as release 1.2.0, dated 2026-08-27, across the
release scripts, machine/visual contracts, and packaging tests. If this work receives a new
release identity, update those identity contracts together before the final source freeze;
changing the manifest alone is insufficient.

## Planning allowance

These are provisional allowances, not measured completion promises for this review. They exclude
time spent fixing newly discovered defects and depend on the local GPU, disk, and PowerPoint.

| Stage after implementation | Planning allowance |
| --- | --- |
| Screenshot capture and interaction review | 5-15 minutes |
| PDF/deck generation and inspection of all pages, slides, and notes | 20-45 minutes |
| Source/GPU gates, collection, reproductions, and machine record | 15-40 minutes; the historical source suite alone took 227.956 seconds |
| Full package build, repeated gates, frozen smoke, and artifact checks | 20-60 minutes |
| Clean final verification and offline rehearsal | 10-20 minutes |

Use measured results from the current run to replace these allowances. The current release CI
job has a 120-minute timeout and the portable quality job has a 30-minute timeout; those are
limits, not evidence of normal duration. Until the closure sequence passes, describe the project
as under readiness review rather than graduation-ready.

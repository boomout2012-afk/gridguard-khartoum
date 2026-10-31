# Proposal Traceability and Source Audit

Status: Original release audit with additive proposal implementation and evidence
Original audit: 2026-08-26; latest additive review: 2026-10-23

## Current interface captures and institutional covers, 2026-10-23

The user requested removing two conclusion sentences from the app's Results views,
refreshing all GUI pages and the academic screenshots, and using matching university
and faculty covers. The app's numerical data and the academic findings are unchanged.
The interface PDF includes a cover and 18 actual app views, including the current
study tabs and historical evidence views. The thesis and presentation use the same
verified screenshot set and shared cover metadata, with the original university and
faculty logos extracted from the supplied reference document. The degree and supervisor
are the user-provided Bachelor of Computer Science and Mrs. Abubakr Siraj.

The updated executable passed packaged CUDA checks across all three shortage levels.
The new outputs and source identities are recorded in
`docs/gui-document-refresh-validation.json`, superseding the stable-path bindings in
the preceding historical refresh receipts.

## Trained PPO live deployment, 2026-10-23

The user authorized replacing the earlier Live Scenario actor with the newly trained
study policy. The predeclared validation-only deployment choice is training seed 8111,
checkpoint 2500, SHA-256
`9c3dbed2cdd979bdd566004ecaada8c78a3aef26397eb7e90a2b953d1da4f2d3`.
The live comparison now uses that actor and the study's constrained current-hour
optimizer, with identical causal inputs and critical-service constraints. Fixed Rotation
remains the matched baseline. The Results page retains the original five-run study,
which is distinct from a single live policy and scenario.

This additive deployment preserves the frozen scientific implementation, original model,
all research actors, and published evidence. It supersedes the earlier-policy live
boundary recorded in the academic refresh below. Validation and executable identity are
recorded in `docs/live-ppo-deployment-validation.json`.

## Academic and application refresh, 2026-10-23

The publication artifacts now stand alone: the thesis and presentation omit the
project-proposal reference and all discussion of that source document. Internal
scope records and frozen evidence retain their original identities. The thesis uses
Harvard author-date citations with 31 cited, alphabetized references, real italic
publication titles and hanging indents. Its final length is 35 pages.

The presentation now follows the same study and results, with LaTeX equations,
training diagnostics, between-run variance and explicit utilization tradeoffs.
The app opens the current five-run study on Results and offers the earlier evidence
separately. Its live demonstration explicitly identifies the earlier policy; the new
actors are not silently substituted. Three pinned JSON artifacts supply the current
view, and the packaged smoke check compares every displayed table cell with that
verified view. Final artifact identities and verification are recorded in
`docs/academic-refresh-validation.json`.

## Longer PPO study and thesis, 2026-10-23

The proposal-wide thesis now follows one sequence: problem and related work, methods,
application, matched experiments, discussion and conclusion. The authors' literature
review is integrated with reference corrections recorded in `docs/LITERATURE_REVIEW_AUDIT.md`.
Prediction, critical-service protection, optimization, fixed scheduling, PPO and direct
duration metrics remain in scope. The main controller comparison is PPO, Fixed Rotation
and Constrained Optimization. Fairness Rule demonstrations are disclosed as a training
ingredient, with earlier comparisons retained in the appendix and original evidence.

Five fresh PPO runs completed 3,000 iterations each (92,160,000 PPO transitions; 35.98
minutes of completed training). Validation selected checkpoints before the final 960
matched conditions were evaluated. The pretraining freeze remains unchanged. The full
declared comparison passes against Fixed Rotation. Against Constrained Optimization,
PPO improves daily reward by 0.04990 and service fairness by 2.2368 percentage points,
but utilization falls by 0.3989 percentage points and switching increases. The full
optimizer-comparison criterion therefore fails. The longer-budget comparison also
fails its full criterion despite reward and fairness gains. These outcomes are stated
in the abstract, results and conclusion without presenting an overall optimizer win.

Accepted records are `data/evidence/focused-ppo-training.json`,
`data/evidence/focused-ppo-study.json` and `data/evidence/focused-ppo-training-diagnostics.json`,
each with a SHA-256 sidecar. `docs/focused-ppo-validation.json` records independent CUDA
metric reconstruction and trajectory replay: all 960 scenario identities, 2,304 final
trajectories and 5,280 validation trajectories checked, with zero observed numerical
error. Final confidence intervals and claim decisions were separately reconstructed
by the evidence verifier.

The redesigned 36-page thesis is `output/pdf/updated/gridguard-khartoum-thesis.pdf`.
All formulas use LaTeX; its six scientific graphs include training curves, TensorBoard
loss/entropy/KL/clipping diagnostics, between-run sample variance, controller outcomes,
paired intervals and forecast errors. Variability bands are explicitly sample standard
deviations, not confidence intervals. Every page was rendered and visually reviewed.
The PDF build sidecar and `docs/focused-thesis-validation.json` bind the final output,
sources, graph renderers, evidence and verification. This thesis refresh does not
redeploy the new actor into the existing executable or refresh the earlier slide deck.
Screenshots identify that version boundary. Historical evidence and actor bytes remain
unchanged.

## Additive proposal coverage, 2026-10-22

The original audit below is retained as historical evidence. Its statements that a dispatch
optimizer was future work and prediction was naive-only are superseded by this extension.

| Proposal objective | Current implementation and evidence |
|---|---|
| 1. Study shortages with Sudan as the case study | Thesis context, Related Work, Khartoum localities and declared synthetic assumptions |
| 2. Model supply, demand, shortages and sectors | Existing scenario generator, seven localities, five sectors, three shortage levels |
| 3. AI decision-support model | Frozen PPO, matched CUDA simulation, native offline desktop |
| 4. Critical-service priority | Hospital/water/bakery demand, Critical Only action, service and interruption outcomes |
| 5. Selected prediction, optimization, rules and RL methods | Ridge AR(3), exhaustive one-step Optimization, Fixed Rotation, PPO |
| 6. Fixed-schedule comparison | Original comparisons plus the additional three-controller matched study |
| 7. Downtime, fairness and efficiency | Critical burden; direct duration range, pairwise gap and streaks; Jain service fairness; utilization and energy |

Research question 4 now has direct blackout-duration and critical-interruption-duration evidence.
The new supplement contains 320 independent base draws, 960 coupled matched conditions and
2,880 controller evaluations. Its seed labels, formulas, sources, per-episode metric arrays,
locality durations, fitted forecasting parameters and error statistics are pinned in
`data/evidence/proposal-supplement.json`. All three paired confidence-interval sets are
reconstructed on CUDA before acceptance. The protocol is declared before the complete run,
without claiming external preregistration or independent custody.

The seven proposal deliverable categories are represented by the simulation, model, scenarios,
controller comparisons, measured results, updated thesis and project presentation. See
`docs/PROPOSAL_EXTENSION.md` and `docs/PROPOSAL_RESULTS.md`. This addition does not establish
field effectiveness, a full-horizon optimum, or operational grid control. Package/release
records continue to identify the exact versions they validated.

Verification for the addition covers exhaustive-search agreement with a scalar oracle,
analytic duration cases, forecast leakage checks, matched input and source hashes,
reconstruction of every paired interval on CUDA, and original golden/parity cases across
all three shortage levels. The source desktop and an extracted wheel pass real CUDA startup.
The historical study's 13 byte-bound source files are preserved under
`data/evidence/accepted-sources/`; current implementation hashes are checked independently
against the new supplement. This preserves both provenance chains rather than replacing the
accepted evidence with a new run.

The redesigned thesis and project presentation are in `output/pdf/updated/`, with the editable
deck in `output/presentation/updated/`. Results remain within those academic documents.
The separate Windows build in `output/app/GridGuard Khartoum/` passes actual packaged CUDA
startup and all three live controllers through all 24 hours. Its clean build excludes ambient
document-tool DLL paths. Final artifact identities and checks are recorded in
`docs/proposal-extension-validation.json`; the historical release records remain unchanged.

## Native GUI design update, 2026-10-21

The approved A/D/F design is implemented in the source desktop application: three pages with
top navigation, an open locality map, a floating locality inspector, and evidence-derived
comparison bars. The user's review removes the Methodology tab, sidebar notices, and persistent
runtime strip. Compact synthetic/decision-support notes remain in page content; the Results
evidence dialog retains the complete canonical limitations and provenance. The academic
methodology and training records remain unchanged.

Focused native tests cover navigation, both presentation sizes, map selection and pan/zoom,
controller switching, timeline playback, real-run integration, empty/loading/error states,
changed-input invalidation, and frozen-evidence isolation. Documentation screenshots are
captured from the actual CUDA Moderate/50002 scenario. Scientific modules, model weights, and
frozen evidence are unchanged. This source UI update does not constitute a new frozen release;
the Windows release process still requires its complete build, integrity, and smoke gates.

Source verification on 2026-10-21: all 147 tests in `tests/ui` and
`tests/test_screenshot_capture.py` passed with no skips; Ruff lint and format checks passed;
`gridguard-evaluate --verify-evidence` passed. The redesigned application also reached its
verified startup READY state on the presentation GPU. Every current page was rendered and
inspected at 1366 x 768 and 1440 x 900.

## Accepted scope baseline

The only scope authority for this clean project is
`docs/source/graduation-project-proposal.pdf`, SHA-256
`3432e64a21ab9741bf47646fd0f4f561522ddfd81d4005deb892d38672c65e48`.

The proposal requires a simplified software simulation, an AI decision-support model, shortage
scenarios, critical-service priorities, comparison with fixed scheduling, proposal-aligned
results, a final report, and an academic presentation. It explicitly excludes direct control of
a real national grid.

## Verified source foundation

The read-only sibling source repository was inspected without modification. The accepted
benchmark and training record identify commit `52f6dd8` as their source revision, so every
migrated behavior below is pinned to the exact file bytes at that commit rather than a later
research revision.

| Source item | Evidence | Source SHA-256 | Source commit |
|---|---|---|---|
| Frozen controller benchmark | `artifacts/benchmark.json` | `5f09625ff987440bc6e667550509cb7f1deac03a38c1b905aa284d65bf74023f` | `52f6dd82605c993c8094448caecec86faa289c0c` |
| Trained PPO checkpoint | `artifacts/checkpoints/best.pt` | `c6e5712f966b46ba18f4c69fb457b9e69f78887c0c5c0279e5ce034f40f9c665` | `52f6dd82605c993c8094448caecec86faa289c0c` |
| Simulation configuration | `src/aipower_rl/config.py` | `5f1f9febaaf7cdae980ed67c83948362b378cd290118f5712f47da1a09d65d04` | `52f6dd82605c993c8094448caecec86faa289c0c` |
| Fixed Rotation controller | `src/aipower_rl/controllers.py` | `b75eaf60da7d3c4e5225327d05029c8b5191583c4853bb17b8bc32f228118538` | `52f6dd82605c993c8094448caecec86faa289c0c` |
| Scenario generator | `src/aipower_rl/scenarios.py` | `689bb4319194baf2f593d98c4b3309d843167e039ed48fff20ee0a9b14aef186` | `52f6dd82605c993c8094448caecec86faa289c0c` |
| Simulator | `src/aipower_rl/simulator.py` | `88df6575a6b2821cde6abcd63f2effe81596210da364bc02bd416b0981e658e6` | `52f6dd82605c993c8094448caecec86faa289c0c` |
| Dispatch and reward | `src/aipower_rl/dynamics.py` | `4fe504105c5f030a143f5583a788de9a4ba9c028b34d49aabc8ff9c3da64c9d8` | `52f6dd82605c993c8094448caecec86faa289c0c` |
| PPO actor network | `src/aipower_rl/models.py` | `6fedcc55675a72aff048a3b1a9a98c86a87962427031c6a4450239fdccbbde03` | `52f6dd82605c993c8094448caecec86faa289c0c` |
| Proposal metrics | `src/aipower_rl/evaluation.py` | `1d0aeef17ed84c3184731cc89c3a39c7f982981183c98047c92fb9c6ffd4cf68` | `52f6dd82605c993c8094448caecec86faa289c0c` |
| Training manifest | `artifacts/training_manifest.json` | `3b866b6f470e116ff009022032059a1f91b509fadc54dbff6c5962a8f2236a59` | `52f6dd82605c993c8094448caecec86faa289c0c` |
| Training metrics | `artifacts/training_metrics.jsonl` | `f48d2f6da29682e1c47330aa2b33b6bb4941f4e0e248f2c846436f09a4b89bc4` | `52f6dd82605c993c8094448caecec86faa289c0c` |
| Selected experiment summary | `experiments/s2-600-hidden512.json` | `619eb67aa253a35d7a7dc0efdbc6c743d7b0af6db4687575e7fade9dbcfe940b` | `52f6dd82605c993c8094448caecec86faa289c0c` |
| Training orchestration | `src/aipower_rl/training.py` | `886931d6c12b0ef45f4f17b89d134e27682c68a2dc4c2f90e2911e2dfec708fb` | `52f6dd82605c993c8094448caecec86faa289c0c` |
| PPO learning algorithm | `src/aipower_rl/learning.py` | `18cf22b0b58584b2abef164ec3ea956fa2f61197336a67a7b9c1f4faefc4fac2` | `52f6dd82605c993c8094448caecec86faa289c0c` |
| Experiment selection | `src/aipower_rl/experiments.py` | `605e32626caece41508f37cbfac0cb061f7510118ba463cf3ff5843f92a9d753` | `52f6dd82605c993c8094448caecec86faa289c0c` |

The benchmark hash identifies the exact audited Windows-checkout bytes. Git stores the same JSON
with normalized LF line endings; `source-manifest.json` records and verifies both byte
representations (`5f09625f...` for the checkout and `37d7e51c...` for the Git blob). All other
listed source hashes equal their Git-blob hashes.

The clean implementation uses parity tests and inference-only artifacts instead of copying the
full research archive.

### Verified display geography

The presentation map is separately source-pinned and never treated as proposal evidence.
OCHA/HDX Sudan COD-AB revision v03 supplies the exact Khartoum State ADM2 units `SD01001` through
`SD01007`; Natural Earth 1:10m rivers version 5.0.0 supplies White Nile, Blue Nile, and Nile
context. The upstream archive, selected member, immutable river revision, derived GeoJSON hashes,
selectors, transformation, licenses, and attribution are recorded in
`src/gridguard/assets/map/THIRD_PARTY_DATA.md` and `map-provenance.json`. The boundaries are
presentation context only and not electrical-network topology.

## Accepted reproduction and documented-seed-disjoint holdout

The historical artifact records 32 episodes for each of nine shortage-level/seed cells. It stores
only cell summaries and does not expose the common base draws reused across shortage levels. The
clean paired artifact therefore reports all three counts: **96 independent base draws, 288 matched
scenario conditions, and 576 controller-scenario evaluations**. Its seeds were reused in archived
model selection, so this is accepted reproduction evidence rather than an independently custodied final test.
The separate documented-seed-disjoint holdout uses seeds 81001-81010: **320 independent base
draws, 960 matched scenario conditions, and 1,920 controller-scenario evaluations**.
Disjointness is established only against documented activity; this holdout was not preregistered
or independently custodied and cannot rule out undocumented activity.

The following table is the **accepted reproduction projection** retained for historical parity; it
is not the documented-seed-disjoint reviewer-facing result table.

| Metric | Fixed Rotation | PPO | Direction |
|---|---:|---:|---|
| Training-objective score (diagnostic) | 13.969 | 23.512 | Higher; diagnostic only |
| Critical-service ratio | 42.63% | 99.72% | Higher; supporting |
| Supply utilization | 71.22% | 97.65% | Higher; primary |
| Jain fairness | 87.76% | 96.92% | Higher; primary |
| Critical-outage locality-hours | 102.1 | 18.7 | Lower; primary |
| Served energy | 4,551.7 MWh | 6,560.2 MWh | Higher; supporting |
| Curtailed energy | 6,142.4 MWh | 4,133.9 MWh | Lower; complementary |
| Switching rate | 25.00% | 21.89% | Lower is better |
| Power-balance violation rate | 0.00% | 0.00% | Must remain zero |

The clean project resolves the historical count ambiguity with
`data/evidence/paired-benchmark.json`. Its episode rows and confidence intervals were regenerated
from the audited simulator and frozen PPO checkpoint; they were not present in the historical
aggregate-only artifact.
`data/evidence/robustness-supplement.json` adds the documented-seed-disjoint seed set, 24 one-factor-at-a-time
sensitivity variants (25 parameterizations with the base), and a one-hour-ahead forecast
comparison using a three-hour trailing window against one-hour persistence. Neither verifier
requires a favorable result.

## Proposal coverage matrix

| Proposal requirement | Source status | Clean-repository acceptance evidence |
|---|---|---|
| Simplified shortage simulation | Implemented among broader experimental branches | Small proposal-equivalent simulator, severity-neutral base-draw reuse, parity tests, architecture documentation |
| Supply, demand, shortage levels, sectors | Implemented | Typed domain contract and deterministic Mild, Moderate, and Severe scenarios |
| AI decision-support model | Trained PPO checkpoint exists | Inference-only checkpoint, CUDA loader, policy parity tests |
| Critical infrastructure priorities | Hospitals, water, and bakeries modeled | Per-sector map detail, critical-service invariant tests |
| Prediction method | Forecast work existed outside the accepted path | Causal one-hour-ahead trailing-window projection for display only, isolated from PPO; synthetic MAE/RMSE compared with one-hour persistence on common targets |
| Rule-based method | Fixed scheduling and a deterministic fairness rule implemented | Fixed Rotation remains the historical baseline. The additive allocation study includes a Fairness Rule over critical-feasible actions, with its tolerance selected on validation only |
| Optimization method | Dispatch optimization implemented in additive proposal studies | Exact current-step action enumeration and a critical-feasible constrained optimizer are implemented and compared on matched scenarios in `data/evidence/proposal-supplement.json` and `data/evidence/allocation-study.json`. This is immediate-reward dispatch optimization, not a full-horizon optimum; Adam remains the separate neural-network parameter optimizer |
| Reinforcement learning | Historical PPO plus constrained retraining workflow implemented | Clean historical inference remains unchanged. `docs/CONSTRAINED_PPO_PROTOCOL.md` specifies fresh masked PPO training, imitation-only ablation, five training seeds, validation-only checkpoint selection and a separate final matched comparison |
| Multiple shortage scenarios | Mild, Moderate, Severe implemented | Matched deterministic CUDA replay and paired evidence across all three levels |
| Fixed-schedule comparison | Implemented | Accepted 288-row reproduction plus documented-seed-disjoint 960-condition holdout, cluster-aware intervals, sign-neutral integrity, and Results view; Fixed Rotation is disclosed as supply-blind, limited to Off and Full Service, and unable to select Critical Only |
| Critical-service downtime | Implemented in evaluation metrics | Frozen table plus per-scenario GUI detail |
| Fairness between areas | Jain fairness and critical-outage locality-hours implemented | Frozen table, definitions, and tests |
| Distribution efficiency | Utilization, served MWh, curtailment implemented | Frozen table, definitions, and tests |
| Data and rules needed for realism (RQ3) | Historical work used illustrative inputs | Parameter-provenance table plus required feeder/locality demand, independent supply/generation, topology, critical-facility, outage/restoration, operating-rule, weather/calendar, and asset-mapping inputs |
| PPO training traceability | Archive contains manifests, metrics, experiment selection, and source | Strict hash-bound `data/model/training-provenance.yaml`: architecture, behavior-cloning initialization, PPO parameters, selected seed-43 run (600 iterations, 153,600 episodes, 3,686,400 transitions), three-seed candidate experiment (1,800 iterations, 460,800 episodes, 11,059,200 transitions), seeds, promotion checks, selected-run hardware/convergence, provenance categories, historical unknowns, and source hashes |
| Final report | Earlier research material was fragmented | Integrated report with methodology, implementation, results, limitations, and references |
| Academic presentation | Earlier dashboard lacked the approved focused flow | Offline native desktop app, verified administrative map, synchronized accessible locality list, live aggregate table, Windows build, deck, and speaker notes |

## Resolved migration risks

1. Out-of-scope experimental branches and claims were excluded from the clean project.
2. Product, controller, locality, service-tier, and shortage terminology was normalized.
3. Frozen evidence is isolated from the interactive demonstration and cannot be replaced by live
   scenario state.
4. Raw runs, optimizer state, caches, and unrelated research artifacts were excluded.
5. Ambiguous episode wording was replaced with separate independent base-draw, matched-condition,
   and controller-evaluation counts.
6. Every migrated source is commit-pinned and byte-hashed, and final release artifacts are bound in
   `RELEASE_MANIFEST.json`.
7. Geographic realism is isolated from scientific inputs: the OCHA/HDX boundaries and Natural
   Earth Nile context are verified presentation assets, not electrical-network topology.
8. Evidence integrity and frozen parity are sign-neutral; validation cannot silently encode a
   favorable research conclusion.
9. The accepted reproduction is distinguished from the documented-seed-disjoint holdout.
10. Reward is a training-objective diagnostic; outage burden, fairness, and utilization are the
    three primary public outcomes, with power balance shown separately as safety.

## Migration rule

The sibling repository remains an immutable research archive. Nothing in it was deleted, renamed,
or rewritten for this delivery. Only proposal-relevant behavior and audited artifacts were
migrated, and every migrated behavior has a clean test or recorded parity check.

## Constrained PPO training evidence, 23 October 2026

The additive experiment in `docs/CONSTRAINED_PPO_RESULTS.md` completed five independent
training runs (18,432,000 PPO transitions) and evaluated their validation-selected models
on 320 fresh base draws with three coupled shortage levels: 960 matched conditions.
Every tested constrained method protected critical service and satisfied power balance.
PPO improved its imitation-only initialization but did not beat the Fairness Rule under
the predeclared overall reward/service criteria. All unfavorable findings are retained.

`data/evidence/constrained-ppo-training.json` binds the pretraining protocol/source freeze,
checkpoint histories and selected weight files. `data/evidence/constrained-ppo-study.json`
contains all 11,520 final controller trajectories' metric rows and uncertainty estimates.
`docs/constrained-ppo-validation.json` records exact validation and final-scenario replay.
These are additional research findings; the prior accepted model and evidence remain
separately identifiable by their unchanged hashes.

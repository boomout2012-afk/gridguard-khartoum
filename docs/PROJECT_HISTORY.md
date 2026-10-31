# Project history and earlier release contracts

This document preserves the detailed project overview as it stood on 23 October 2026.
Use the [current README](../README.md) and [reviewer guide](REVIEWER_GUIDE.md) for the
submission entry points. The earlier release contracts, training records and audit
findings below describe their own evidence snapshots; they do not replace the current
five-run study or certify a later Git revision. Commands in this history assume the
repository root as their working directory.

## Archived project overview

GridGuard Khartoum is an offline desktop simulation that demonstrates PPO-assisted load-shedding
decisions for seven Khartoum State localities. It compares PPO with a traditional Fixed Rotation
schedule on identical synthetic shortage scenarios, with explicit protection metrics for
hospitals, water services, and bakeries.

The current submission (23 October 2026) compares constrained PPO, Fixed Rotation and
Constrained Optimization. Five fresh PPO runs each completed 3,000 iterations. The app's
Results page shows the five-run study; Live Scenario uses the validation-selected seed-8111
policy at iteration 2500. Live examples are separate from the study averages.
The complete declared comparison passes against Fixed Rotation. PPO improves fairness and
reward over Constrained Optimization, but loses enough utilization that the overall optimizer
comparison does not pass. The optimizer maximizes current-hour reward, not full-day reward.

Current submission deliverables:

- Windows app: `output/app/GridGuard Khartoum/GridGuard Khartoum.exe` (keep its folder intact).
- Thesis: `output/pdf/updated/gridguard-khartoum-thesis.pdf` (35 pages).
- Presentation: `output/pdf/updated/gridguard-khartoum-project-presentation.pdf` (24 slides).
- Editable slides: `output/presentation/updated/gridguard-khartoum-defense.pptx`.
- GUI collection: `output/pdf/gridguard-khartoum-gui-pages.pdf` (cover and 18 app views).

See [SUBMISSION_README.md](../SUBMISSION_README.md) for the portable app, source setup, evidence
verification and document rebuild instructions. The current document/app identity receipt is
`docs/gui-document-refresh-validation.json`. Earlier records below describe earlier artifacts.

The 22 October extension also provides direct interruption-duration/disparity metrics and a
separate learned Ridge AR(3) demand forecast. Its additional 960-condition study remains
available under Results > Extended study and in [the extension results](../docs/PROPOSAL_RESULTS.md).

The project implements the scope accepted in `docs/source/graduation-project-proposal.pdf`. It is
a research and presentation tool, not a real-grid controller. All demand and supply values are
synthetic. The offline map uses pinned OCHA/HDX administrative boundaries with Natural Earth Nile
context. These layers are for presentation context only; they are not electrical-network topology
and supply no scientific input.

## Reviewer start here

Historical release: **1.2.0 (2026-08-27)** and its original validation records remain intact;
release 1.1.0 remains the earlier baseline. The October proposal extension has the separate app,
academic outputs and verification record listed above. The original release records describe
their exact historical artifacts and are not relabeled as certification of the extension.

- Read the approved implementation contract in [SPEC.md](../SPEC.md).
- Follow current findings and remaining release gates in the
  [October graduation-readiness review](../docs/GRADUATION_REVIEW.md).
- Track every confirmed remediation finding in the
  [devil's-advocate audit register](../docs/DEVILS_ADVOCATE_AUDIT.md).
- Find all historical PPO training parameters in the
  [data/model/training-provenance.yaml](../data/model/training-provenance.yaml) file. It is the
  canonical, strict, model-manifest-bound record of architecture, initialization, optimizer and
  loss settings, workload, seeds, selection, hardware, lineage, and convergence.

The training record is intentionally provenance rather than an executable bitwise-retraining
configuration: optimizer state and raw training runs are excluded. The validated summary command
is shown in [PPO training parameters](#ppo-training-parameters).

## Presentation requirements

- Windows laptop with a compatible NVIDIA GPU
- CUDA-capable PyTorch 2.8 and CuPy CUDA 12.x runtime
- Python 3.12 for source execution
- No internet connection after installation

Run `uv run gridguard-doctor` before every rehearsal. The presentation and benchmark fail closed instead of silently switching to the CPU.

The GPU runtime rationale and trade-offs are recorded in `docs/decisions/ADR-001-gpu-runtime.md`.

## Verified evidence

The evidence package has two deliberately different roles. The accepted reproduction at
`data/evidence/paired-benchmark.json` (SHA-256
`7f2009c4f7efbbedf098069ced9218addec51d55a18b969e6c54433027020af5`) preserves the historical
comparison: 96 independent synthetic base draws crossed with Mild, Moderate, and Severe shortages,
giving 288 matched scenario conditions and 576 controller-scenario evaluations. Seeds
`50001-50003` were gradient-disjoint, but the archived promotion process reused them during model
selection. This layer is therefore accepted reproduction evidence, not an independently custodied final test.

The documented-seed-disjoint holdout at `data/evidence/robustness-supplement.json` (SHA-256
`6b56d149331ee191302cd9e6f22365988db0ae301f9441190b61214acf832e48`) uses seeds 81001-81010: 320 independent base draws, 960 matched scenario
conditions, and 1,920 controller-scenario evaluations. It also evaluates 24 one-factor-at-a-time
locality-weight and sector-share perturbations, for 25 parameterizations including the base, and
compares the display-only three-hour trailing-mean forecast with one-hour persistence on common
target hours. Those seeds are disjoint only from every documented training, validation, and
model-selection seed in `data/model/training-provenance.yaml`. The holdout was not preregistered or
independently custodied and cannot rule out undocumented activity. PPO and Fixed Rotation use the same validated
scenario within each pair; device-side guards reject detected demand or supply mutation.

Sensitivity intervals are descriptive, pointwise 95%, and unadjusted for multiple comparisons;
they do not support family-wide significance claims. No behavior-cloning-only ablation was run, so
this comparison cannot isolate the incremental effect of PPO fine-tuning. Fixed Rotation is
supply-blind, limited to Off and Full Service, and cannot select Critical Only.

The public primary outcomes are critical-outage locality-hours, Jain fairness, and utilization.
Power-balance violation rate is a separate safety invariant. The episode reward is labeled
**Training-objective score (diagnostic)** because it contains several reported outcomes and is not
independent confirmation. Full-precision values remain in canonical JSON; human-facing views use
one decimal place for MWh/locality-hours, two decimals for percentages, and three decimals for the
diagnostic score.

`validate_paired_benchmark_payload` checks closed schema, pairing, arithmetic, and physical
identities. `assert_paired_benchmark_integrity` adds locked provenance and direct parity with the
frozen aggregate artifact. Both are sign-neutral integrity checks: adverse or inconclusive
evidence remains valid evidence, and no favorable-result release gate exists. These are synthetic
simulation results for decision-support research, not measurements from Sudan's electrical grid.

## PPO training parameters

The canonical reviewer-facing historical record is
[data/model/training-provenance.yaml](../data/model/training-provenance.yaml) (SHA-256
`d2568bce26e9221afa8c266113a704f6fbf2370d3bc949f5aa6c72a7411ba1a8`). It records the
32 -> 512 -> 512 -> 2,187 actor, behavior-cloning initialization, PPO optimizer and loss
parameters, all seeds, validation and checkpoint-selection rules, source hashes, and the selected
run's hardware and convergence series. The selected seed-43 run used 600 iterations, 153,600
episodes, and 3,686,400 transitions; its checkpoint is iteration 510 after 6,120 optimizer updates.
The three-seed candidate experiment totaled 1,800 iterations, 460,800 episodes, and 11,059,200
transitions. Historical promotion gates selected the actor; they are not independent headline
evidence. Schema v3 classifies each historical field as directly recorded, source-reconstructed
(including named PyTorch defaults), arithmetic-derived, or an unavailable historical environment
fact. Adam optimized network parameters during training. CUDA accelerates training, inference,
simulation, and statistics. The historical desktop performed no online optimization; the current
desktop also evaluates a constrained current-hour dispatch optimizer in Live Scenario.

The YAML is a strict provenance record, not a bitwise-retraining promise: optimizer state and raw
training runs are intentionally excluded. Print a validated machine-readable summary with:

```powershell
uv run --extra docs python .\tools\training_provenance.py --path .\data\model\training-provenance.yaml --model-manifest .\data\model\manifest.json --summary-json
```

## Offline geographic context

The application bundles seven second-level Khartoum State units from OCHA's Sudan COD-AB revision
v03 and White Nile, Blue Nile, and Nile linework from Natural Earth 1:10m rivers. Source sizes,
SHA-256 pins, selectors, transformations, licenses, and derived hashes are recorded in
`src/gridguard/assets/map/THIRD_PARTY_DATA.md` and `map-provenance.json`. Runtime performs no
download and fails closed if the packaged geography drifts. The design decision and rejected
alternatives are recorded in `docs/decisions/ADR-002-geographic-boundary-map.md`.

## Desktop presentation flow

The current source interface uses the approved A/D/F design: a floating top navigation bar,
open locality map, white control panels, bold Atkinson typography, and coral selection accents.
The three pages are Overview, Live Scenario, and Results. Open **Evidence details** on Results
for provenance and complete scientific limitations. Ctrl+1 through Ctrl+3 switch pages;
F11 toggles full screen. To run the current source application:

```powershell
uv run --no-sync gridguard-desktop
```

1. Select Mild, Moderate, or Severe shortage and a deterministic seed.
2. Replay the same 24-hour scenario through PPO and Fixed Rotation.
3. Inspect service tiers and sector outcomes on the interactive administrative map or synchronized
   keyboard-accessible locality list.
4. Compare the live demonstration's matched episode aggregates with the separately verified
   documented-seed-disjoint holdout; live values never replace reviewer-facing frozen evidence.
5. Inspect evidence provenance and limitations in Results; consult the offline methodology
   and implementation chapters for the complete academic account.

## Development

Install the locked environment, then use the portable test gate for ordinary development or
CPU-only CI:

```powershell
uv sync --extra dev --extra docs
uv run --no-sync pytest -m "not gpu and not source_archive and not release_artifact"
uv run --no-sync ruff check .
```

That portable gate is useful feedback, but it does not certify a presentation release. The gate
contract is deliberately separated: the mandatory presentation-GPU gate validates all source and
GPU tests from committed, hash-bound inputs; the strict post-build `release_artifact` gate validates
the completed Windows tree; and the optional source-archive parity gate audits the supplied
read-only research archive. The normal package build does not require the sibling research archive.

```powershell
uv run --no-sync gridguard-doctor
uv run --no-sync pytest -ra -m "not source_archive and not release_artifact"
uv run --no-sync gridguard-evaluate --verify-evidence

# Run only after the Windows package and release manifest are final.
uv run --no-sync pytest -ra -m "release_artifact"
```

Both mandatory gates intentionally fail when their required GPU, CUDA stack, evidence, or packaged
outputs are unavailable, and both require zero skipped selected tests. When the read-only archive
is available, enable the optional parity audit explicitly with:

```powershell
.\scripts\build-presentation.ps1 -StrictSourceParity `
  -SourceRepositoryPath ..\aipower_rl `
  -SourceCheckpointPath ..\aipower_rl\artifacts\checkpoints\best.pt
```

Strict parity validates the supplied archive and checkpoint and includes `source_archive` tests;
the normal package build never probes the sibling repository or ambient source-path variables.

To reproduce the paired artifact without overwriting the trusted copy:

```powershell
uv run gridguard-evaluate --reproduce-paired --output .\paired-benchmark-reproduced.json
```

To reproduce the documented-seed-disjoint robustness artifact without overwriting the trusted copy:

```powershell
uv run gridguard-evaluate --reproduce-robustness --output .\robustness-supplement-reproduced.json
```

The implementation plan and completion evidence are tracked in `tasks/plan.md` and `tasks/todo.md`.

## Build the academic artifacts

Build the report from the verified evidence and committed chapters:

```powershell
uv run python .\tools\build_report.py
```

The editable slide deck requires Microsoft PowerPoint on Windows because the builder uses the
native PowerPoint automation interface for shapes, speaker notes, and rendered QA images:

```powershell
powershell -NoProfile -File .\tools\build_presentation.ps1
```

The final PDF, deck, evidence, model, proposal, map, and Windows executable are bound by size and
SHA-256 in `RELEASE_MANIFEST.json`.

The unsigned release manifest proves internal byte consistency against its pinned inventory, but
it does not authenticate authorship. A coherent rewrite could replace both artifacts and their
hashes. Reviewers must therefore compare the published private Git branch with the exact commit SHA
recorded in the manifest and reported at handoff. Artifact signing remains optional unless the
project team supplies a signing identity; this release does not claim signature-backed
authenticity.

Historical visual-record or machine-record verification alone does not establish that changed
source has been released. For a source refresh, follow the source freeze, fresh test/evidence
commands, visual review, machine-record refresh, and final package sequence in
`docs/GRADUATION_REVIEW.md`. The existing build enforces a separate source-snapshot check.

## Build the Windows presentation package

Run the locked release gate and create the one-directory executable with:

```powershell
.\scripts\build-presentation.ps1
```

Launch it with `.\scripts\run-presentation.ps1`. Keep the entire
`dist\GridGuard Khartoum` directory together because the executable depends on its adjacent
PyTorch, CuPy, Qt, model, and evidence files. The final rehearsal steps and approved scientific
wording are in `docs/PRESENTATION_CHECKLIST.md`.

The final methodology, implementation, and results chapters are under `docs/`. Generated report
and slide artifacts are written under `output/` and are intentionally separate from the source
package.

## Constrained allocation research study

The additive [allocation study protocol](../docs/ALLOCATION_STUDY_PROTOCOL.md) compares the existing
controllers with Constrained Optimization, a deterministic Fairness Rule, Feasible PPO, and a
PPO Hybrid that filters the frozen actor's preferences by feasibility and immediate reward.
This experiment leaves the accepted evidence and model unchanged. It does not retrain PPO or
implement learned future-value planning.

Run validation before the final comparison, using new output paths because the runner refuses
to overwrite evidence:

```powershell
uv run python tools/evaluate_allocation_study.py --phase validation --output output/research/allocation-validation.json
$allocationSelectionSha = (Get-FileHash output/research/allocation-validation.json -Algorithm SHA256).Hash.ToLower()
uv run python tools/evaluate_allocation_study.py --phase test --selection output/research/allocation-validation.json --selection-sha256 $allocationSelectionSha --output output/research/allocation-study.json
```

The validation record freezes both selected reward tolerances and scientific source hashes.
Final evaluation rejects changed sources or a mismatched validation identity. Each phase writes
JSON evidence, its SHA-256, and a readable Markdown report. The completed comparison and its
interpretation are recorded in [allocation study results](../docs/ALLOCATION_STUDY_RESULTS.md).

## Training PPO with critical-service constraints

The additive [constrained PPO protocol](../docs/CONSTRAINED_PPO_PROTOCOL.md) trains five fresh
actors with the same feasible action set as the Fairness Rule. Each run starts with
Fairness Rule imitation, retains an imitation-only comparator, then receives a fixed
PPO training budget. Checkpoint selection uses validation data; final comparisons use
960 fresh matched conditions and account for both scenario and training-run uncertainty.

To reproduce into new paths without overwriting published model or evidence files:

```powershell
uv run python tools/train_constrained_ppo.py --model-dir output/research/constrained-ppo-models --record output/research/constrained-ppo-training.json
$constrainedTrainingSha = (Get-FileHash output/research/constrained-ppo-training.json -Algorithm SHA256).Hash.ToLower()
uv run python tools/evaluate_constrained_ppo.py --training output/research/constrained-ppo-training.json --training-sha256 $constrainedTrainingSha --output output/research/constrained-ppo-study.json
$constrainedStudySha = (Get-FileHash output/research/constrained-ppo-study.json -Algorithm SHA256).Hash.ToLower()
uv run python tools/evaluate_constrained_ppo.py --verify --training output/research/constrained-ppo-training.json --training-sha256 $constrainedTrainingSha --output output/research/constrained-ppo-study.json --sha256 $constrainedStudySha
```

Training writes a protocol/source freeze before validation, saves weights without optimizer
state, and refuses to overwrite existing artifacts. A completed run can be resumed only
when its source, configuration, model hashes and selection record still match. A positive
result is never required for evidence verification.

The completed comparison is in [constrained PPO results](../docs/CONSTRAINED_PPO_RESULTS.md),
with frozen training and final-test records under `data/evidence/` and independent replay
checks in `docs/constrained-ppo-validation.json`.

## Longer PPO study and proposal-wide thesis

The [focused study protocol](../docs/FOCUSED_PPO_PROTOCOL.md) specifies five fresh PPO runs
with 3,000 iterations each, five times the preceding budget. Its main comparators are
Fixed Rotation and Constrained Optimization. Both constrained controllers share the
same critical-service feasibility rule; the optimizer exhaustively maximizes the
current-hour reward. The study tests daily outcomes and does not treat this comparator
as a full-day optimum. The earlier Fairness Rule findings remain preserved.

Reproduce into new paths, keeping final scenarios separate from checkpoint selection:

```powershell
uv run python tools/train_focused_ppo.py --model-dir output/research/focused-ppo-models --record output/research/focused-ppo-training.json --monitor-dir output/research/focused-ppo-monitor
$focusedTrainingSha = (Get-FileHash output/research/focused-ppo-training.json -Algorithm SHA256).Hash.ToLower()
uv run python tools/evaluate_focused_ppo.py --training output/research/focused-ppo-training.json --training-sha256 $focusedTrainingSha --output output/research/focused-ppo-study.json
$focusedStudySha = (Get-FileHash output/research/focused-ppo-study.json -Algorithm SHA256).Hash.ToLower()
uv run python tools/evaluate_focused_ppo.py --verify --training output/research/focused-ppo-training.json --training-sha256 $focusedTrainingSha --output output/research/focused-ppo-study.json --sha256 $focusedStudySha
```

The thesis source is [FOCUSED_THESIS.md](../docs/FOCUSED_THESIS.md). It covers the full
proposal: shortage simulation, learned forecasting, critical priorities, optimization,
PPO, service outcomes and the offline application. The authors' supplied literature
review is retained unchanged at `docs/source/literature-review-authors.docx`, with
reference corrections documented in [the literature audit](../docs/LITERATURE_REVIEW_AUDIT.md).
`tools/assemble_focused_thesis_content.py`, `tools/plot_focused_thesis.py` and
`tools/build_focused_thesis.py` assemble the pinned results and render the redesigned
PDF. The build preserves its previous PDF and writes source and provenance sidecars.
Formulas remain editable LaTeX in the source and are rendered with MathJax to SVG and
600-dpi images. Install the document-only renderer with
`npm install --prefix tmp/math-typesetting --no-audit --no-fund mathjax-full@3.2.2`.
It uses the bundled Sharp library, or `GRIDGUARD_SHARP_PATH` when provided. The chart
script requires Matplotlib; the scientific simulation and training dependencies are unchanged.

The completed 35-page thesis is `output/pdf/updated/gridguard-khartoum-thesis.pdf`.
It includes LaTeX equations and six scientific graphs, including TensorBoard training
diagnostics and variance across the five runs. The full declared PPO comparison passes
against Fixed Rotation. PPO improves fairness and reward over Constrained Optimization,
but its utilization tradeoff prevents an overall pass. Evidence replay and final document
checks are recorded in `docs/focused-ppo-validation.json` and
`docs/focused-thesis-validation.json`. The standalone manuscript uses Harvard author-date
citations and an alphabetized reference list. The current presentation PDF is
`output/pdf/updated/gridguard-khartoum-project-presentation.pdf`, with its editable
companion at `output/presentation/updated/gridguard-khartoum-defense.pptx`.

The desktop Results page now opens the current five-run study, including adjusted
intervals, individual runs and shortage details. Earlier results remain accessible
separately. Live Scenario now uses the study's validation-selected PPO policy
(training seed 8111, checkpoint 2500) and the same constrained current-hour optimizer.
Both use the study's critical-service feasibility rule on identical live inputs.
Live values describe one selected policy and one scenario; they do not replace the
five-run study averages. The earlier academic refresh is recorded in
`docs/academic-refresh-validation.json`; the subsequent live deployment and rebuilt
executable are recorded in `docs/live-ppo-deployment-validation.json`.

The latest interface export contains a cover and 18 captured app views at
`output/pdf/gridguard-khartoum-gui-pages.pdf`. The thesis and presentation use the
same current app captures and shared university/faculty cover details from
`docs/academic-cover.json`. The rebuilt app removes the two requested conclusion
sentences from its Results views while retaining the quantitative comparisons.
Current document and executable identities are recorded in
`docs/gui-document-refresh-validation.json`; preceding receipts remain historical.

`tools/plot_training_diagnostics.py --render-only --training-sha256 <accepted-training-sha256>`
verifies and redraws the published diagnostics without replacing their frozen aggregate.
Run it in the document plotting environment with Matplotlib available.

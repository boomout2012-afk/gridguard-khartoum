# GridGuard Khartoum

GridGuard Khartoum is an offline desktop simulation that demonstrates PPO-assisted load-shedding
decisions for seven Khartoum State localities. It compares PPO with a traditional Fixed Rotation
schedule on identical synthetic shortage scenarios, with explicit protection metrics for
hospitals, water services, and bakeries.

The project implements the scope accepted in `docs/source/graduation-project-proposal.pdf`. It is
a research and presentation tool, not a real-grid controller. All demand and supply values are
synthetic. The offline map uses pinned OCHA/HDX administrative boundaries with Natural Earth Nile
context. These layers are for presentation context only; they are not electrical-network topology
and supply no scientific input.

## Reviewer start here

Current release: **1.2.0 (2026-08-27)** identifies the historical packaged release. Release 1.1.0
remains the earlier baseline. The October 2026 source refresh is under graduation-readiness
review; the August package and its validation records do not certify these changes. Final
screenshots, report, deck, Windows build, machine validation, and manifest must be regenerated
and verified before this source refresh is handed over as presentation-ready.

- Read the approved implementation contract in [SPEC.md](SPEC.md).
- Follow current findings and remaining release gates in the
  [October graduation-readiness review](docs/GRADUATION_REVIEW.md).
- Track every confirmed remediation finding in the
  [devil's-advocate audit register](docs/DEVILS_ADVOCATE_AUDIT.md).
- Find all historical PPO training parameters in the
  [data/model/training-provenance.yaml](data/model/training-provenance.yaml) file. It is the
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
[data/model/training-provenance.yaml](data/model/training-provenance.yaml) (SHA-256
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
simulation, and statistics; it is not a mathematical dispatch optimizer, and the shipped desktop
performs no online optimization.

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

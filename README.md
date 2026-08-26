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
`6052e9e47d92e82a0f1db0d3643a2e9be6edc5f3e9fc615b0b185964b54c1677`) preserves the historical
comparison: 96 independent synthetic base draws crossed with Mild, Moderate, and Severe shortages,
giving 288 matched scenario conditions and 576 controller-scenario evaluations. Seeds
`50001-50003` were gradient-disjoint, but the archived promotion process reused them during model
selection. This layer is therefore accepted reproduction evidence, not an untouched final test.

The untouched confirmatory supplement at `data/evidence/robustness-supplement.json` (SHA-256
`c89e04b1bdca371e1728f6b4cb19f4e899261ab805c2e400c15e835eaf75a30b`) uses seeds 81001-81010: 320 independent base draws, 960 matched scenario
conditions, and 1,920 controller-scenario evaluations. It also evaluates 24 one-factor-at-a-time
locality-weight and sector-share perturbations, for 25 parameterizations including the base, and
compares the display-only three-hour trailing-mean forecast with one-hour persistence on common
target hours. PPO and Fixed Rotation always receive identical scenario tensors within a pair.

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

The complete reviewer-facing historical record is
`data/model/training-provenance.yaml` (SHA-256
`2cfa158a4b26596fd1f155c57fa79a88dc1c41ce8df946792c559184a815128a`). It records the
32 -> 512 -> 512 -> 2,187 actor, behavior-cloning initialization, PPO optimizer and loss
parameters, all seeds, validation and checkpoint-selection rules, hardware, source hashes, and the
full convergence series. Historical PPO training ran for 600 iterations, 153,600 episodes, and
3,686,400 transitions; the selected seed-43 checkpoint is iteration 510 after 6,120 optimizer
updates. Adam optimized network parameters during training. CUDA accelerates training, inference,
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

1. Select Mild, Moderate, or Severe shortage and a deterministic seed.
2. Replay the same 24-hour scenario through PPO and Fixed Rotation.
3. Inspect service tiers and sector outcomes on the interactive administrative map or synchronized
   keyboard-accessible locality list.
4. Compare the live demonstration's matched episode aggregates with the separately verified
   untouched robustness supplement; live values never replace reviewer-facing frozen evidence.
5. Review methodology, implementation, limitations, and evidence provenance offline.

## Development

Install the locked environment, then use the portable test gate for ordinary development or
CPU-only CI:

```powershell
uv sync --extra dev --extra docs
uv run pytest -m "not gpu and not source_archive"
uv run ruff check .
```

That portable gate is useful feedback, but it does not certify a presentation release. The
mandatory release gate runs on the presentation laptop with the trusted source archive available:

```powershell
uv run gridguard-doctor
uv run pytest
uv run gridguard-evaluate --verify-evidence
uv run gridguard-desktop
```

The unfiltered release command intentionally fails when CUDA, CuPy, the PPO source archive, or the
packaged evidence is unavailable. Tests are not silently skipped in a release rehearsal.

To reproduce the paired artifact without overwriting the trusted copy:

```powershell
uv run gridguard-evaluate --reproduce-paired --output .\paired-benchmark-reproduced.json
```

To reproduce the untouched robustness artifact without overwriting the trusted copy:

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

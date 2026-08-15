# GridGuard Khartoum

GridGuard Khartoum is an offline desktop simulation that demonstrates PPO-assisted load-shedding decisions for seven Khartoum State localities. It compares PPO with a traditional Fixed Rotation schedule on identical synthetic shortage scenarios, with explicit protection metrics for hospitals, water services, and bakeries.

The project implements the scope accepted in `docs/source/graduation-project-proposal.pdf`. It is a research and presentation tool, not a real-grid controller. All demand and supply values are synthetic, and the locality map is a schematic rather than an electrical or geographic boundary map.

## Presentation requirements

- Windows laptop with a compatible NVIDIA GPU
- CUDA-capable PyTorch 2.8 and CuPy CUDA 12.x runtime
- Python 3.12 for source execution
- No internet connection after installation

Run `uv run gridguard-doctor` before every rehearsal. The presentation and benchmark fail closed instead of silently switching to the CPU.

The GPU runtime rationale and trade-offs are recorded in `docs/decisions/ADR-001-gpu-runtime.md`.

## Verified evidence

The committed paired benchmark contains 96 independent synthetic base draws crossed with Mild,
Moderate, and Severe shortages: 288 matched scenario conditions and 576 controller evaluations.
PPO and Fixed Rotation always receive the same scenario tensors. Uncertainty intervals use 10,000
deterministic seed-stratified paired cluster bootstrap replicates, keeping the three severity
variants of each base draw together.

The accepted historical benchmark stored only 18 batch-cell summaries. The clean project
reproduces those summaries within fixed tolerances, then regenerates the episode-level rows and
paired intervals from the audited simulator and frozen PPO checkpoint. These are synthetic
simulation results, not measurements from Sudan's electrical grid.

## Desktop presentation flow

1. Select Mild, Moderate, or Severe shortage and a deterministic seed.
2. Replay the same 24-hour scenario through PPO and Fixed Rotation.
3. Inspect service tiers and sector outcomes on the interactive locality map.
4. Compare the live demonstration with the separately frozen benchmark result.
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

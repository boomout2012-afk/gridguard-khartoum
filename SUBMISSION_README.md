# GridGuard Khartoum submission guide

Submission date: October 2026. University of Khartoum, Faculty of Mathematical Sciences and
Informatics. Bachelor of Computer Science. Supervisor: Mrs. Abubakr Siraj.

Authors: Murtada Asim Ibrahim Bakhit (18-317), Suliman Elawad Suliman Babiker (18-310),
and Abdalhay Abass Mobarak Abass (15-306).

## Open the submitted project

The portable submission has Documents, App, Source and Verification folders. Documents contains
the final 35-page thesis, 24-slide presentation PDF, editable PowerPoint and 19-page GUI collection.
Double-click `Run GridGuard.cmd` to start the app. Keep the complete App folder together.
No Python installation or network connection is needed to run the packaged application.

The app requires 64-bit Windows and a compatible NVIDIA GPU/driver supporting the bundled
CUDA 12.8 runtime. It was verified on an NVIDIA GeForce RTX 5060 Laptop GPU. The bundle includes
PyTorch, CuPy, Qt, trained weights, geographic assets and evidence. It cannot supply the operating
system's NVIDIA driver. A CPU-only computer can read all documents and inspect the source.

The submitted Results page displays the five-run study. Live Scenario uses the validation-selected
PPO actor (seed 8111, checkpoint 2500) on one selected scenario, so its numbers should not equal
the five-run averages. Both constrained controllers use the same critical-service feasibility rule.
The complete comparison passes against Fixed Rotation. Against Constrained Optimization, PPO
improves fairness and daily reward but loses utilization; it does not pass the full comparison.
These are synthetic decision-support results, not measurements or operational recommendations
for Sudan's electrical grid. The constrained optimizer optimizes each hour, not the full day.

## Source contents

- `src/gridguard`: simulation, controllers, CUDA metrics/evaluation, evidence validation and Qt UI.
- `tests`: unit, integration, real GPU, UI and packaging checks.
- `data/evidence`: frozen results, training/diagnostic summaries, checksums and accepted source snapshots.
- `data/model`: shipped actors, all five study selections, iteration-600 and imitation-only comparators.
- `docs/FOCUSED_THESIS.md`: current manuscript with editable LaTeX formulas and Harvard references.
- `docs/academic-cover.json`: shared cover fields; `docs/assets`: logos, fonts, screenshots and graphs.
- `tools`, `scripts`, `packaging`: training, evaluation, academic authoring and Windows build scripts.
- `pyproject.toml`, `uv.lock`: locked application and test environment.

The code is copied as ordinary files. No virtual environment, Git working tree, cache, unrelated
download or secret is included. Compact training telemetry is recoverable from the frozen record;
optimizer state and bulky intermediate training runs are not required to replay the submitted models.
The sibling research repository is not required. Its accepted source snapshots are already included.

The old `RELEASE_MANIFEST.json`, historical chapters and original report/deck under `output/pdf`
and `output/presentation` are retained as explicitly historical audit/test fixtures. They do not
certify this submission. Use the top-level submission manifest and Verification report for the final
identities. All current academic deliverables are in Documents and in Source's `output/*/updated`
paths, except the current GUI PDF, which is directly under `output/pdf`.

## Install and test source

Use Python 3.12 and uv. The complete test suite also uses Git and PowerShell 7 to exercise
temporary build/release fixtures; it does not require this folder to be a Git checkout.
Initial dependency installation needs internet access or a pre-populated
package cache; the packaged application above is already offline-ready.

```powershell
uv sync --locked --extra dev --extra docs
uv run --no-sync gridguard-doctor
uv run --no-sync gridguard-evaluate --verify-evidence
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pytest -ra -m "not source_archive and not release_artifact"
uv run --no-sync gridguard-desktop
```

The source/GPU gate requires NVIDIA CUDA and must report zero skipped selected tests. The
`source_archive` group is an optional audit of the separate original research Git repository.
The historical `release_artifact` group validates that older release's exact Git commit and dist
layout, not this portable submission. The submission's per-file integrity check is
`Verify Submission.ps1`, next to START HERE.md; it requires only Windows PowerShell.

## Reproduce the current evidence

Verify the frozen current study without changing it:

```powershell
$trainingSha = (Get-FileHash data/evidence/focused-ppo-training.json -Algorithm SHA256).Hash.ToLower()
$studySha = (Get-FileHash data/evidence/focused-ppo-study.json -Algorithm SHA256).Hash.ToLower()
uv run --no-sync python tools/evaluate_focused_ppo.py --verify --training data/evidence/focused-ppo-training.json --training-sha256 $trainingSha --output data/evidence/focused-ppo-study.json --sha256 $studySha
```

For fresh evaluation with the selected actors, use the same command without `--verify` and
`--sha256`, and choose a new output such as `output/research/focused-replay.json`. Training and
evaluation runners refuse to replace existing evidence. Re-training is a separate experiment:

```powershell
uv run --no-sync python tools/train_focused_ppo.py --model-dir output/research/retrained-models --record output/research/retrained-training.json --monitor-dir output/research/retrained-monitor
```

The frozen seeds, training budget, model selection and uncertainty methods are documented in
`docs/FOCUSED_PPO_PROTOCOL.md`. Re-training is not promised to be bitwise identical on different
hardware; exact submitted model weights and full-precision evaluation evidence are included.

## Rebuild or edit the documents

The finalized PDFs can be submitted directly. The editable deck includes native charts, tables,
speaker notes and chart workbooks. Install the bundled Atkinson Hyperlegible regular/bold fonts
from `src/gridguard/assets/fonts` when editing it to preserve typography. Use Microsoft PowerPoint
for the verified rendering path; other viewers may lay out editable content differently.

To export the editable deck as PDF using Microsoft PowerPoint on Windows:

```powershell
powershell -NoProfile -File scripts/export-submission-presentation.ps1
```

To regenerate thesis figures, LaTeX and GUI PDF from the included source/assets, install Node.js
with npm, then run:

```powershell
powershell -NoProfile -File scripts/rebuild-academic.ps1 -InstallDependencies
```

That script installs document-only plotting/rendering dependencies and reconstructs the original
five telemetry logs byte-for-byte from the frozen training JSON. It does not retrain models.
Subsequent builds can omit `-InstallDependencies`. For graphs only, use
`tools/export_training_telemetry.py` and the existing `tools/plot_training_diagnostics.py --render-only`
with the training hash above. The SD bands are not confidence intervals; the variance denominator
is five minus one.

The original `tools/build_focused_presentation.mjs` generator is also preserved for traceability.
It uses OpenAI's Codex artifact authoring runtime and its finalizer and is not a standalone npm
dependency. The portable editing/re-export route is the included PowerPoint and script above.

## Build the Windows executable

From the installed source environment:

```powershell
uv run --no-sync pyinstaller --noconfirm --clean --distpath output/app --workpath build/submission packaging/gridguard.spec
```

This creates a one-directory distribution. Keep its `_internal` directory with the executable.
Do not run a rebuild while the destination application is open. Validate source tests and the
real CUDA doctor before rebuilding. The historical Git-attested release pipeline remains in
`scripts/build-presentation.ps1`; it targets the older release contract and is not needed for
the portable build command above.

All fonts and geography retain their third-party attribution/license files. Included university
logos are original assets from the supplied university reference document.

# Setup and reproduction

## Run the application

Install Python 3.12 and [uv](https://docs.astral.sh/uv/), then run these commands
from the repository root:

```powershell
uv sync --locked --extra dev --extra docs
uv run --no-sync gridguard-desktop
```

Choose a shortage level in Live Scenario to compare controllers on the same day.
Results shows the saved study; a single live scenario need not match its averages.

<details>
<summary>Execution requirements</summary>

The desktop application targets 64-bit Windows. Its simulation backend uses PyTorch
and CuPy and requires a compatible NVIDIA graphics card and driver. Exact dependency
versions are in `pyproject.toml` and `uv.lock`. The installed application runs offline;
initial dependency installation needs internet access or a populated package cache.
The thesis, presentation and saved results can be read without running the application.

</details>

## Reproduce the experiments

The [research guide](docs/REVIEWER_GUIDE.md) identifies each study, its protocol and
the corresponding results. Check the main experiment without overwriting it:

```powershell
$trainingSha = (Get-Content data/evidence/focused-ppo-training.sha256).Split()[0]
$studySha = (Get-Content data/evidence/focused-ppo-study.sha256).Split()[0]
uv run --no-sync python tools/evaluate_focused_ppo.py --verify --training data/evidence/focused-ppo-training.json --training-sha256 $trainingSha --output data/evidence/focused-ppo-study.json --sha256 $studySha
```

Check the additional comparisons separately:

```powershell
$baselineSha = (Get-Content data/evidence/committee-baseline-audit.sha256).Split()[0]
uv run --no-sync python tools/evaluate_committee_baseline.py --verify --sha256 $baselineSha
uv run --no-sync python tools/evaluate_thesis_learning_audit.py --verify
uv run --no-sync python tools/evaluate_additional_baselines.py --verify
```

These commands check saved evidence and reconstruct its summaries. A complete replay
of the fresh learning-stage study can be run without replacing its saved receipt:

```powershell
uv run --no-sync python -c "from tools.evaluate_thesis_learning_audit import OUTPUT, verify; verify(OUTPUT, replay=True)"
```

For a new training experiment, choose new output paths:

```powershell
uv run --no-sync python tools/train_focused_ppo.py --model-dir output/research/models --record output/research/training.json --monitor-dir output/research/monitor
```

The published policies are retained for exact replay. Retraining is a new experiment
and may produce different weights. The earlier actor's configuration is documented in
[training-provenance.yaml](data/model/training-provenance.yaml); it is separate from
the five-run study used in the thesis.

## Development

```powershell
uv run --no-sync pytest -m "not source_archive"
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
```

Tests use Git and PowerShell 7 for temporary packaging fixtures. Optional archive-parity
tests use material outside this repository. GitHub runs the portable subset.

## Documents and packaging

The thesis source is [FOCUSED_THESIS.md](docs/FOCUSED_THESIS.md). It contains the
LaTeX equations and references; results are filled from saved evidence. Figures,
screenshots, institutional logos and licensed fonts are in `docs/assets/`.

Rebuild the thesis and application-page PDFs with Node.js, npm and Poppler's
`pdftoppm` available on `PATH`:

```powershell
powershell -NoProfile -File scripts/rebuild-academic.ps1 -InstallDependencies
```

The plotting script reconstructs training telemetry from the saved record rather than
retraining. The original locality analysis can be regenerated with
`uv run --no-sync python tools/interpret_focused_study.py`.

The [editable thesis](output/documents/updated/gridguard-khartoum-thesis.docx) uses native
Word equations. The [editable presentation](output/presentation/updated/gridguard-khartoum-defense.pptx)
contains charts, tables and speaker notes. Use Microsoft Office and the supplied fonts
for the established export layout. Export helpers are in `scripts/`.

Build the Windows application with:

```powershell
uv run --no-sync pyinstaller --noconfirm --clean --distpath output/app --workpath build/submission packaging/gridguard.spec
```

Keep the generated application folder together. To check the included research files
without installing the application, run `python tools/verify_review_package.py`.

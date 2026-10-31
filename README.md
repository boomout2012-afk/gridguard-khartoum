# GridGuard Khartoum

**AI-Based Smart Load Shedding and Energy Management System for Developing Countries**

An offline Windows research application for seven Khartoum State localities. It simulates
24-hour electricity shortages, prioritizes critical services, and compares **PPO**, **Fixed
Rotation**, and **Constrained Optimization** on identical scenarios. The project also evaluates
demand forecasting, interruption duration, fairness and energy utilization.

All demand and supply data are synthetic. This is a decision-support research project;
it has not been validated for operating a real electrical grid.

## Reviewer start here

| Deliverable | Contents |
| --- | --- |
| [Thesis PDF](output/pdf/updated/gridguard-khartoum-thesis.pdf) | 35 pages; methodology, LaTeX formulas, results, training graphs and Harvard references |
| [Presentation PDF](output/pdf/updated/gridguard-khartoum-project-presentation.pdf) | 24 slides for the project defense |
| [Editable PowerPoint](output/presentation/updated/gridguard-khartoum-defense.pptx) | Native charts, tables and speaker notes |
| [GUI collection](output/pdf/gridguard-khartoum-gui-pages.pdf) | Cover and 18 current application views |

The [reviewer guide](docs/REVIEWER_GUIDE.md) maps the implementation, evidence and verification
steps. The [submission guide](SUBMISSION_README.md) explains document rebuilding and the
separate portable Windows package. Academic cover details are shared in
[academic-cover.json](docs/academic-cover.json).

## Main findings

The current study trains five PPO policies for 3,000 iterations each and evaluates them on
960 matched conditions. The complete declared comparison passes against Fixed Rotation.
Against Constrained Optimization, PPO improves fairness and daily reward, but its utilization
loss means the full comparison does not pass. The optimizer maximizes current-hour reward;
the study does not compare PPO with a full-day optimum.

The app's **Results** page reports the five-run study. **Live Scenario** uses the policy selected
by validation, training seed 8111 at checkpoint 2500. One live scenario is separate from the
study averages. See the [frozen study protocol](docs/FOCUSED_PPO_PROTOCOL.md) for selection rules,
uncertainty, outcome criteria and limitations.

## Run from source

Use **Python 3.12**, **uv**, 64-bit Windows and a compatible NVIDIA GPU/driver for CUDA 12.8.
The locked environment provides PyTorch 2.8, CuPy and PySide6. Initial dependency installation
needs internet or a populated package cache; the installed app runs offline. The scientific
runtime requires CUDA and does not silently switch to CPU.

```powershell
uv sync --locked --extra dev --extra docs
uv run --no-sync gridguard-doctor
uv run --no-sync gridguard-evaluate --verify-evidence
uv run --no-sync gridguard-desktop
```

Choose Mild, Moderate or Severe shortage, run the matched scenario, then inspect the timeline
and locality service details. The map supplies geographic context only, not electrical-network
topology or scientific inputs.

## Verify the source

Check the submitted documents, evidence, policy weights and scientific source files with
`python tools/verify_review_package.py`. This integrity check needs only Python's standard
library and works without a GPU or installed application dependencies.
The [repository review map](docs/REPOSITORY_REVIEW.md) explains the manifest and artifact scope.

The complete test suite also requires Git and PowerShell 7 for temporary packaging fixtures.

```powershell
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
# Portable checks used by GitHub CI; no presentation-GPU certification.
uv run --no-sync pytest -m "not gpu and not source_archive and not release_artifact"
# Complete source and GPU checks on the presentation machine.
uv run --no-sync pytest -ra -m "not source_archive and not release_artifact"
```

The [reviewer guide](docs/REVIEWER_GUIDE.md#verification-and-reproduction) explains the current
study verification and separates it from optional archive audits and historical release checks.

## Repository map

| Path | Purpose |
| --- | --- |
| `src/gridguard/` | Domain, scenarios, controllers, simulation, metrics, evidence and Qt interface |
| `tests/` | Unit, integration, GPU, GUI and packaging tests |
| `data/evidence/` | Frozen results, training summaries, checksums and accepted source snapshots |
| `data/model/` | Policy weights and training provenance |
| `docs/` | Manuscript, protocols, figures, screenshots and review records |
| `tools/`, `scripts/`, `packaging/` | Training, evaluation, document generation and Windows packaging |
| `output/pdf/updated/`, `output/presentation/updated/` | Current academic deliverables |

The large prebuilt CUDA application and local build caches are excluded from Git. The
repository includes the source, models, evidence and documents needed for review and reproduction.
Earlier reports and release fixtures remain documented in [project history](docs/PROJECT_HISTORY.md).
Their historical `RELEASE_MANIFEST.json` does not certify the current submission.
The canonical historical training record is
[data/model/training-provenance.yaml](data/model/training-provenance.yaml), which is checked by
strict provenance validation. It describes the earlier actor; use the focused study protocol
above for the current five-run study and deployed policy.

# GridGuard Khartoum Working Agreement

## Mission

Deliver the accepted graduation proposal as a clean, offline, presentation-ready Windows desktop application and academic package. The only scope authority is `docs/source/graduation-project-proposal.pdf`; `SPEC.md` defines the approved implementation contract.

## Product vocabulary

- Product: GridGuard Khartoum
- Learned controller: PPO
- Baseline controller: Fixed Rotation
- Geographic unit: locality
- Service tiers: Off, Critical Only, Full Service
- Shortage levels: Mild, Moderate, Severe

Use these terms in code, tests, UI, documentation, artifacts, and commits. Do not use experimental version labels or the legacy compound name for the learned controller.

## Scope boundaries

- Keep the sibling `aipower_rl` repository read-only. It is a research archive and parity source.
- Do not import post-proposal experimental features or claims.
- Keep synthetic-data and decision-support-only limitations explicit.
- Compare controllers only on identical matched scenarios.
- Never replace frozen benchmark values with live demonstration values.
- Require a validated NVIDIA CUDA GPU for the primary presentation and benchmark path; never require internet access.
- Do not commit optimizer state, raw training runs, caches, secrets, or unrelated research artifacts.

## Architecture

- Python 3.12 package under `src/gridguard`.
- PyTorch 2.8 with CUDA 12.8 for PPO inference and batched simulation.
- CuPy for GPU-native statistics/array work, interoperating with PyTorch through DLPack where data crosses the library boundary.
- NumPy only after the single final evidence transfer, for portable artifact assembly and small
  host-side validation; do not move simulation, inference, metric, or bootstrap hot paths to CPU.
- PySide6 native UI with scientific logic outside widgets.
- `QGraphicsScene`/`QGraphicsView` for the offline schematic map.
- JSON evidence artifacts with SHA-256 provenance and fail-closed verification.
- pytest and pytest-qt for tests, Ruff for lint/format, uv for environments, PyInstaller for Windows packaging.

Keep dependencies directed as follows:

```text
domain -> scenarios -> controllers/simulation -> metrics/evaluation/evidence
                                                    |
                                                    v
                                            presenter -> Qt views
```

Domain and scientific modules must not import Qt. Widgets must not calculate scientific results.

## Development workflow

1. Write or update a focused failing test before changing behavior.
2. Implement the smallest complete vertical slice.
3. Run focused tests and Ruff.
4. Update traceability or task state when acceptance evidence changes.
5. Commit one reviewed concern at a time with an imperative conventional message.

Useful commands:

```powershell
uv sync --extra dev --extra docs
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run gridguard-desktop
uv run gridguard-doctor
uv run gridguard-evaluate --verify-evidence
```

## Code conventions

- Use typed, explicit units such as `_mw`, `_mwh`, `_hours`, and `_ratio`.
- Prefer immutable dataclasses and pure calculation functions.
- Validate tensor/array shapes and finite, non-negative power values at boundaries.
- Inject controllers, clocks, paths, and loaders rather than reading hidden global state.
- Keep UI widgets small and expose presentation state through typed view models.
- Any public metric must have one canonical definition, test, unit, aggregation rule, and evidence source.

## Verification rules

- Proposal-critical tests may not be skipped in a release.
- Release verification includes a real CUDA/PyTorch/CuPy preflight on the presentation GPU.
- Golden parity cases must cover all three shortage levels and both controllers.
- Power-balance violations must remain zero within the declared numeric tolerance.
- Evidence verification must fail on altered artifacts, hashes, schemas, labels, counts, or claim values.
- GUI tests run offscreen and cover keyboard interaction, selection, pan/zoom, and 1366x768 layout.
- Render and inspect every page/slide before accepting final PDF or presentation output.

## Code discovery

Prefer codebase-memory-mcp over repository text search for source-code discovery:

1. `search_graph`
2. `trace_path`
3. `get_code_snippet`
4. `query_graph`
5. `get_architecture`

Use text search for non-code documents, configuration values, and literal error messages. Re-index the graph after meaningful structural changes.

## Libraries

Use Context7 official/current documentation before implementing or troubleshooting library-specific behavior. Resolve the library ID first, then query one focused concept at a time.

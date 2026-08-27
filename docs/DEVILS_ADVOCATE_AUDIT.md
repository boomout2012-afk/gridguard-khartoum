# Devil's Advocate Audit Register

Status: Active pre-freeze technical closure register

Audit date: 2026-08-26

Target release: 1.2.0 (unreleased)

Released baseline retained for history: 1.1.0 (2026-08-26)

This source-tracked ledger is the pre-freeze technical closure register for the confirmed
post-1.1.0 adversarial findings. Version 1.2.0 is a target, not a completed release. The ledger is
a work register, not proof that an open finding is closed. Neither a task number nor a planned
command is acceptance evidence. Its final state is part of the tested source snapshot and must
not be edited after that snapshot is committed.

The approved proposal remains the scope authority. Remediation may correct evidence semantics,
claims, verification, packaging, and maintainability needed for those corrections. It must not add
a new controller, operational field-data claim, online dependency, or post-proposal experiment.

## Reviewer start here

- The implementation contract is [SPEC.md](../SPEC.md).
- The complete A01-A24 work register is the ledger below.
- The **canonical, strict training-parameter and lineage record** is
  [data/model/training-provenance.yaml](../data/model/training-provenance.yaml). The model manifest
  binds its bytes; the record covers network architecture, behavior-cloning initialization, PPO
  optimizer and loss settings, workload, seeds, selection, hardware, source lineage, and
  convergence. It separates the selected seed-43 run (600 iterations, 153,600 episodes,
  3,686,400 transitions) from the aggregate three-seed candidate experiment (1,800 iterations,
  460,800 episodes, 11,059,200 transitions), classifies recorded/reconstructed/derived/unknown
  facts, and preserves the exact historical promotion gates. Those gates are model-selection
  history, not independent current headline evidence. The record is historical provenance, not
  an executable or bitwise-retraining promise.
- Execution order and acceptance gates are in [tasks/plan.md](../tasks/plan.md); checklist state is
  in [tasks/todo.md](../tasks/todo.md).

## Register rules

Severity describes the consequence if a finding survives into 1.2.0:

- **High**: release-blocking scientific, integrity, runtime, or publication defect.
- **Medium**: material assurance, documentation, or maintainability gap that must be resolved or
  explicitly bounded before release.
- **Low**: visible consistency defect that still must be corrected before release.

Each finding has exactly one required closure disposition:

- **Fixed**: eliminate the defect and prove the corrected behavior.
- **Reframed with evidence**: replace an overclaim with precise wording and bind it to supporting
  evidence.
- **Bounded limitation**: retain the proposal-sized implementation while making the unresolved
  attribution or maintenance boundary explicit.
- **Optional signing hardening**: make the unsigned trust boundary honest and externally anchored;
  apply signing only if the user supplies an identity.

`Open` means the evidence named in the final column is still pending. `Closed` may be used only
after that evidence exists before the tested source commit and passes its stated gate.
`Post-freeze attestation` is reserved for A02 because its machine-validation record is generated
from the tested source commit and therefore cannot exist inside that commit. `Post-freeze
handoff` is reserved for A01 and A21 because private-remote publication and the resulting external
Git anchor can be observed only after the final attestation-envelope commit is pushed. Neither
special status is a disguised `Closed`.

## Finding ledger

| Finding | Severity | Confirmed issue | Required disposition | Owner task(s) | Status | Closure evidence |
|---|---|---|---|---|---|---|
| A01 | High | The reviewed feature branch is not published and has no verified upstream SHA. | Fixed | 6.12 | Post-freeze handoff | After the final push, the final handoff response records exact local and private-remote branch SHA equality without changing repository bytes or advancing `HEAD`. |
| A02 | High | The release manifest carries stale validation and wrong report/deck counts instead of fresh machine evidence. | Fixed | 6.11 | Post-freeze attestation | After the tested source commit, `docs/release-machine-validation.json` records its exact source SHA and detected counts; the reconstructed `RELEASE_MANIFEST.json` binds that record. |
| A03 | Medium | Stale 1.0.0 wheel and source archives remain beside the current distribution. | Fixed | 6.10 | Closed | V5 is the validated cleanup/current-inventory record: the standard build produced exactly `.gitignore`, `GridGuard Khartoum`, `gridguard_khartoum-1.2.0-py3-none-any.whl`, and `gridguard_khartoum-1.2.0.tar.gz` under `dist`; `Assert-ReleaseDistributionPolicy` passed and no historical or alternative-version distribution remained. Final manifest binding remains a separate post-freeze gate. |
| A04 | High | The deck conclusion hides that Fixed Rotation is supply-blind, uses only Off and Full Service, and cannot select Critical Only. | Fixed | 6.8 and 6.9 | Closed | Slide 13 and its speaker note state the exact comparator boundary beside the conclusion. `docs/release-visual-review.json` binds all 13 rendered slides and all 13 note contents; `tests/test_presentation_builder.py`, `tests/test_document_claims.py`, and `tests/test_release_validation.py` pass the claim-adjacency, note-content, and visual-binding contracts. |
| A05 | High | The wheel smoke can reach PyPI for an unpinned Hatchling build dependency and is not a hermetic offline proof. | Fixed | 6.2 | Closed | `pyproject.toml` and `uv.lock` exact-pin Hatchling 1.32.0. `tests/test_ci.py` and `tests/ui/test_map_asset_wheel.py` pass V1; the wheel smoke sets `PIP_NO_INDEX=1`, points `UV_CACHE_DIR` at a new empty temporary directory, builds with `--offline --no-index --no-build-isolation`, installs outside the source tree, and verifies the bundled reviewer assets. |
| A06 | High | Reported confidence-interval endpoints are accepted from artifact bytes rather than independently reconstructed from persisted per-draw inputs. | Fixed | 6.5 and 6.7 | Closed | `src/gridguard/paired_inference.py` and `src/gridguard/robustness_inference.py` reconstruct paired, base, sensitivity, difference-in-differences, forecast, and power-balance results from persisted strict cubes. V2 passes the CUDA reconstruction and rehashed/coherent endpoint-tamper tests in `tests/test_evaluation.py`, `tests/test_robustness.py`, and `tests/test_robustness_inference.py`. |
| A07 | High | Matched-scenario evidence proves aggregate demand/supply equality only, while frozen dataclasses do not prevent tensor mutation. | Fixed | 6.3 | Closed | `src/gridguard/domain.py` supplies the canonical trace digest and device-side mutation guard; every paired row carries a unique validated `scenario_sha256`. V2 passes the fingerprint/order/storage-replacement/write-restore tests in `tests/test_domain.py` and the controller-mutation and fabricated-fingerprint tests in `tests/test_evaluation.py`. |
| A08 | High | Demand-allocation sensitivity also regenerates supply, so it does not isolate demand allocation. | Fixed | 6.4 and 6.7 | Closed | `src/gridguard/robustness.py` reuses the exact base available-supply tensor for each demand-allocation variant. V2 passes `tests/test_robustness.py::test_sensitivity_scenario_holds_reference_total_demand_and_supply_fixed` and the strict robustness reconstruction tests. |
| A09 | High | The phrase untouched confirmatory overstates what documented seed-set disjointness can prove without preregistration or independent custody. | Reframed with evidence | 6.6 | Closed | `src/gridguard/claim_contract.py` defines the documented-seed-disjoint role and the no-preregistration/no-independent-custody boundary. V3 passes the legacy-label exclusion, custody-language, seed-set-disjointness, YAML, UI, and document projection tests. |
| A10 | High | The 144 sensitivity summaries use pointwise intervals without a multiplicity treatment or an explicit family-wide inference boundary. | Reframed with evidence | 6.6 | Closed | The canonical claim contract, evidence, UI, and reviewer documents label all 144 sensitivity summaries descriptive, pointwise 95%, unadjusted for multiple comparisons, and unsuitable for a family-wide significance claim. V3 passes the cross-surface claim tests. |
| A11 | Medium | No behavior-cloning-only ablation exists, so the comparison cannot isolate the incremental effect of PPO fine-tuning. | Bounded limitation | 6.6 | Closed | `src/gridguard/claim_contract.py` and schema-v3 `data/model/training-provenance.yaml` define the no-ablation attribution limit. The strict YAML also distinguishes the selected seed-43 run from the aggregate three-seed candidate experiment, classifies direct/reconstructed/derived/unknown facts, and records promotion gates as historical model-selection facts rather than independent headline evidence. Evidence, UI, report source, and deck source project those boundaries. V3 passes the canonical-YAML, source-bundle, UI, and cross-surface document tests. |
| A12 | High | The Results UI surfaces only the first general limitation and can omit the weak-baseline limitation beside the result. | Fixed | 6.8 | Closed | `src/gridguard/ui/results_page.py` renders the canonical supply-blind comparator boundary—uses only Off and Full Service and cannot select Critical Only—beside Results. V3 and V4 pass the visibility, accessibility, keyboard, and 1366x768 clipping tests, including `tests/ui/test_window.py::test_results_page_keeps_safety_and_scope_limits_visible`. |
| A13 | Medium | Report reference 14 is described as reproducible RL evaluation although the source is a survey of RL for electric-power control. | Fixed | 6.9 | Closed | Reference [14] is identified as a survey in `docs/REPORT.md` and the rebuilt 38-page PDF. `docs/release-visual-review.json` binds the inspected report; `tests/test_report_builder.py` and `tests/test_document_claims.py` pass the citation-description and extraction checks. |
| A14 | Medium | Implementation and packaging documentation omit robustness modules and the exact bundled provenance-source set. | Fixed | 6.9 | Closed | `docs/IMPLEMENTATION.md` names all robustness modules, extracted UI/claim modules, and the exact bundled provenance-source set. V3 passes `tests/test_academic_contract.py::test_implementation_documents_robustness_modules_and_exact_source_bundle`, its extracted-module-map companion, and the frozen source-binding tests. |
| A15 | Low | Slide 4 lowercases Mild, Moderate, and Severe contrary to the approved vocabulary. | Fixed | 6.9 | Closed | Slide 4 uses Mild, Moderate, and Severe in source, XML, and the inspected render. `tests/test_presentation_builder.py` and `tests/test_document_claims.py` enforce the exact vocabulary. |
| A16 | High | The visual attestation is unbound and records stale counts rather than all current report pages, slides, and notes. | Fixed | 6.11 | Closed | Schema-v2 `docs/release-visual-review.json` binds the final PDF, PPTX, 38 contiguous report pages, 13 contiguous slides, the actual relationship-resolved contents of all 13 speaker notes, and all four screenshots by path, SHA-256, and size. `tools/release_validation.py --verify` and all `tests/test_release_validation.py` cases pass. |
| A17 | Medium | PresentationWindow and LiveScenarioPage exceed the approximate 200-line UI coordination boundary. | Fixed | 6.8 | Closed | Coordination is split across `src/gridguard/ui/window_components.py`, `live_components.py`, `live_controls.py`, and `live_visualization.py`. V4 passes all 65 focused UI tests; `tests/ui/test_ui_decomposition.py` enforces the 200-line orchestrator boundary and preserves navigation, trust, aliases, controls, and accessibility. |
| A18 | Medium | Large scientific modules increase maintenance risk, but a broad refactor would add unnecessary scientific regression risk. | Bounded limitation | 6.3 through 6.5 | Closed | The required new logic is isolated in `src/gridguard/paired_inference.py`, `robustness_inference.py`, `robustness_provenance.py`, and `claim_contract.py`; the reviewer documents explicitly defer a broad late-stage refactor. V2 and V3 pass the focused helper, reconstruction, source-binding, and boundary tests without a scope-expanding rewrite. |
| A19 | High | The Windows package contains more than 2 GB of byte-identical CUDA DLL pairs. | Fixed | 6.10 | Closed | V5 passed the actual-tree release policy with 4,665 files and 4,873,683,241 bytes, zero same-basename-and-SHA-identical Torch/NVIDIA CUDA DLL pairs, and a 2,147,548,349-byte reduction from the audited 7,021,231,590-byte tree. The canonical 994-byte frozen smoke report contained zero CRLF sequences and reported PASS on the RTX 5060 with PyTorch 2.8.0+cu128, CUDA 12.8, CuPy 14.1.1, and matched 24-frame PPO and 24-frame Fixed Rotation live results. |
| A20 | High | Release tests can pass conditionally when executable, smoke, or distribution artifacts are absent. | Fixed | 6.2 and 6.11 | Closed | V5 passed 1,136 selected source tests with zero failures, errors, or skips and built and smoke-tested the actual package. V6 passed the independent portable selection with 1,072 tests and zero failures, errors, or skips, all 129 packaging tests, and the fresh structured-error-path and smoke-order review; the build contracts now fail closed when required executable, smoke, or distribution inputs are absent. The exact-source post-build `release_artifact` JUnit and manifest binding remain mandatory post-freeze release gates and are not claimed by this pre-freeze technical closure. |
| A21 | Medium | A self-hashed unsigned manifest proves internal consistency but not authenticity against a coherent rewrite. | Optional signing hardening | 6.11 and 6.12 | Post-freeze handoff | The frozen source and manifest disclose the unsigned/coherent-rewrite boundary; after the final push, the same external local/private-remote SHA equality recorded for A01 supplies the independently trusted Git anchor. Signing remains optional and requires a user-supplied identity. |
| A22 | High | An ambient GRIDGUARD_SMOKE_REPORT variable can suppress the normal GUI and redirect smoke output. | Fixed | 6.10 | Closed | `packaging/gridguard_launcher.py` accepts smoke mode only through the explicit private `--_gridguard-frozen-smoke-report` argument, validates an exclusive build-owned JSON path, and removes ambient offscreen control for a normal launch. V6 passed all 129 packaging tests, including poisoned-environment, invalid-path, exclusive-create, and Windows argument-quoting cases; V5 then produced and validated the canonical packaged smoke report through that private path. |
| A23 | High | The release path relies on the sibling research archive even though installed runtime inputs must be committed, bound, and offline. | Fixed | 6.2 and 6.10 | Closed | V5 invoked the standard offline build without `-StrictSourceParity`; locked sync, Ruff, CUDA doctor, evidence verification, both reproduction commands, 1,136 selected source tests, the Windows package, distribution policy, and frozen smoke all ran without requiring the sibling archive. V6 separately passed all 8 `source_archive` tests with zero skips using explicit trusted repository and checkpoint paths, preserving the archive as optional parity evidence only. |
| A24 | Medium | Historical and current distribution versions are mixed, obscuring which package represents the reviewed release. | Fixed | 6.1 and 6.10 | Closed | V5 produced only the 1.2.0 wheel, 1.2.0 source distribution, and 1.2.0 frozen application. The passing distribution policy validated the exact top-level inventory and the mutually consistent wheel `METADATA`, source-distribution `PKG-INFO`, frozen distribution metadata, and `pyproject.toml` contract; no historical GridGuard version artifact remained. Final hashes remain subject to the post-freeze manifest gate. |

## Pre-freeze technical verification

These current-tree command records were produced on 2026-08-27 from the repository root. They are
technical closure evidence only; the final full, package, visual, machine-attestation, and remote
publication gates remain separate.

- **V1 — hermetic build contract:**
  `uv run --no-sync pytest -q tests/test_ci.py tests/ui/test_map_asset_wheel.py` exited 0 with
  6 tests passed, including a real wheel build against a new empty temporary uv cache.
- **V2 — scientific assurance:**
  `uv run --no-sync pytest -q tests/test_domain.py tests/test_evaluation.py
  tests/test_robustness.py tests/test_robustness_inference.py` exited 0.
- **V3 — claims, provenance, and documentation:**
  `uv run --no-sync pytest -q -m "not source_archive and not release_artifact"
  tests/test_document_claims.py tests/test_academic_contract.py tests/test_training_provenance.py
  tests/test_frozen_evidence.py tests/ui/test_window.py tests/ui/test_presentation_model.py
  tests/ui/test_ui_decomposition.py` exited 0.
- **V4 — focused desktop behavior:**
  `uv run --no-sync pytest -q tests/ui/test_app.py tests/ui/test_window.py
  tests/ui/test_live_page.py tests/ui/test_presentation_model.py tests/ui/test_ui_decomposition.py`
  exited 0 with 65 tests passed.
- **V5 — standard offline Windows package rehearsal:**
  `scripts/build-presentation.ps1` was invoked without `-StrictSourceParity`. Offline locked sync,
  Ruff check and format, CUDA doctor, evidence verification, paired and robustness reproduction,
  and the selected source suite all exited 0; its JUnit record contained 1,136 tests, zero failures,
  zero errors, and zero skips. The build produced the exact four-entry `dist` inventory, and
  `Assert-ReleaseDistributionPolicy` passed with 4,665 files and 4,873,683,241 bytes. That is
  2,147,548,349 bytes smaller than the audited 7,021,231,590-byte tree, with zero actual-tree
  same-basename-and-SHA-identical Torch/NVIDIA CUDA DLL pairs. The canonical 994-byte frozen smoke
  report contained zero CRLF sequences and reported PASS on the NVIDIA GeForce RTX 5060 Laptop GPU
  with PyTorch 2.8.0+cu128, CUDA 12.8, CuPy 14.1.1, and matched 24-frame PPO and 24-frame Fixed
  Rotation live results. After validating the smoke report, the rehearsal stopped only at the
  intentionally absent pre-freeze machine-validation record; it did not claim the final manifest
  or `release_artifact` gates.
- **V6 — packaging, portable execution, optional archive parity, and error-path assurance:**
  `uv run --no-sync pytest -q tests/test_packaging.py` exited 0 with 129 tests passed. The
  `source_archive` selection, run separately with explicit trusted repository and checkpoint paths,
  exited 0 with 8 tests passed and zero skips. A fresh structured-error-path and smoke-order review
  also returned PASS. The portable `not gpu and not source_archive and not release_artifact`
  selection exited 0 with 1,072 tests passed and zero failures, errors, or skips. These are
  pre-freeze technical records only; final exact-source `release_artifact`, machine-record,
  manifest, clean-tree, and publication gates remain post-freeze.

## Closure protocol

1. Implement each owner task test-first and retain evidence-neutral acceptance: an adverse result
   is still valid when honestly derived.
2. Before the tested source commit, replace `Pending` only with exact files, hashes, command
   records, or rendered-review records that already exist. Change a technical finding to `Closed`
   only while all of its closure evidence is present in that source snapshot.
3. The rebuilt PDF and PPTX must be fully inspected before source freeze.
   docs/release-visual-review.json must already be frozen in the tested source commit; later visual
   edits invalidate the source snapshot.
4. Commit and test that source snapshot. From that point onward this ledger is immutable. Only
   RELEASE_MANIFEST.json and docs/release-machine-validation.json may change while forming the
   post-source attestation envelope. The machine record must name the tested source SHA, and the
   manifest must bind the final machine and pre-frozen visual records.
5. Commit the machine record and reconstructed manifest, rerun the strict release-integrity gates,
   and require a clean worktree. Ledger drift, visual-record drift, source drift, and unknown or
   untracked release files all fail the source-snapshot gate.
6. Push that final attestation-envelope commit to the private remote without another commit. The
   final handoff response records the exact local HEAD SHA, the private remote branch ref, and the
   remote branch SHA obtained from Git; it states whether the two 40-character values are equal.
   An annotated private tag may point to that same final commit because creating a tag does not
   advance `HEAD`.
7. A01 and A21 remain `Post-freeze handoff` in this immutable ledger. Embedding that equality in a
   new commit would advance HEAD and make the embedded final-SHA statement self-invalidating;
   embedding the resulting external anchor would have the same defect. Do not create
   docs/release-publication-attestation.json, tracked or untracked. The remote-ref output and final
   handoff response are deliberately external publication and authenticity evidence, not
   source-tested or manifest-authenticated evidence.
8. Keep Authenticode or detached signing optional unless the user supplies a signing identity; do
   not describe an unsigned artifact as authenticated.

### A01 publication evidence location

Task 6.12 reads the final local commit with `git rev-parse HEAD` and the published private branch
with `git ls-remote --heads <private-remote> refs/heads/<private-branch>`. The final handoff response
is the only A01 publication-evidence record: it names the remote and branch, prints both exact
40-character SHAs, and states their equality. This response is external to the immutable source
snapshot and unsigned release manifest; it does not claim source-tested authentication.

Task 6.12 must audit every row against current files, commands, rendered artifacts, packaged
runtime behavior, the clean final worktree, and the remote Git refs before 1.2.0 can be called
released. A02 is satisfied only by the manifest-bound post-freeze machine record; A01 is satisfied
only by the external final-handoff SHA comparison.

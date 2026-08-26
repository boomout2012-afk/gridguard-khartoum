# Devil's Advocate Audit Register

Status: Active remediation register

Audit date: 2026-08-26

Target release: 1.2.0 (unreleased)

Released baseline retained for history: 1.1.0 (2026-08-26)

This register converts the confirmed post-1.1.0 adversarial findings into a reviewable closure
contract. Version 1.2.0 is a target, not a completed release. The ledger is a work register, not
proof that an open finding is closed. Neither a task number nor a planned command is acceptance
evidence.

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
  convergence. It is historical provenance, not a bitwise-retraining promise.
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
after that evidence exists at the reviewed commit and passes its stated gate.

## Finding ledger

| Finding | Severity | Confirmed issue | Required disposition | Owner task(s) | Status | Closure evidence |
|---|---|---|---|---|---|---|
| A01 | High | The reviewed feature branch is not published and has no verified upstream SHA. | Fixed | 6.12 | Open | Pending — exact local and private-remote branch SHA equality from Task 6.12. |
| A02 | High | The release manifest carries stale validation and wrong report/deck counts instead of fresh machine evidence. | Fixed | 6.11 | Open | Pending — generated validation record, detected counts, and reconstructed manifest from Task 6.11. |
| A03 | Medium | Stale 1.0.0 wheel and source archives remain beside the current distribution. | Fixed | 6.10 | Open | Pending — validated cleanup log and current-version-only distribution inventory from Task 6.10. |
| A04 | High | The deck conclusion hides that Fixed Rotation is supply-blind, uses only Off and Full Service, and cannot select Critical Only. | Fixed | 6.8 and 6.9 | Open | Pending — adjacent UI/deck/notes disclosure plus rendered review from Tasks 6.8 and 6.9. |
| A05 | High | The wheel smoke can reach PyPI for an unpinned Hatchling build dependency and is not a hermetic offline proof. | Fixed | 6.2 | Open | Pending — locked, no-index, offline, empty-cache wheel-smoke evidence from Task 6.2. |
| A06 | High | Reported confidence-interval endpoints are accepted from artifact bytes rather than independently reconstructed from persisted per-draw inputs. | Fixed | 6.5 and 6.7 | Open | Pending — endpoint reconstruction and rehashed-endpoint tamper failures from Tasks 6.5 and 6.7. |
| A07 | High | Matched-scenario evidence proves aggregate demand/supply equality only, while frozen dataclasses do not prevent tensor mutation. | Fixed | 6.3 | Open | Pending — canonical scenario fingerprints and controller-mutation rejection tests from Task 6.3. |
| A08 | High | Demand-allocation sensitivity also regenerates supply, so it does not isolate demand allocation. | Fixed | 6.4 and 6.7 | Open | Pending — exact base-supply reuse tests and regenerated sensitivity evidence from Tasks 6.4 and 6.7. |
| A09 | High | The phrase untouched confirmatory overstates what documented seed-set disjointness can prove without preregistration or independent custody. | Reframed with evidence | 6.6 | Open | Pending — documented-seed-disjoint wording and explicit custody disclaimers across all claim surfaces from Task 6.6. |
| A10 | High | The 144 sensitivity summaries use pointwise intervals without a multiplicity treatment or an explicit family-wide inference boundary. | Reframed with evidence | 6.6 | Open | Pending — descriptive pointwise 95% and unadjusted-for-multiplicity labels from Task 6.6. |
| A11 | Medium | No behavior-cloning-only ablation exists, so the comparison cannot isolate the incremental effect of PPO fine-tuning. | Bounded limitation | 6.6 | Open | Pending — claim-adjacent attribution limitation in evidence, YAML, UI, report, and deck from Task 6.6. |
| A12 | High | The Results UI surfaces only the first general limitation and can omit the weak-baseline limitation beside the result. | Fixed | 6.8 | Open | Pending — canonical adjacent baseline limitation and accessibility/UI tests from Task 6.8. |
| A13 | Medium | Report reference 14 is described as reproducible RL evaluation although the source is a survey of RL for electric-power control. | Fixed | 6.9 | Open | Pending — corrected citation description and rebuilt report inspection from Task 6.9. |
| A14 | Medium | Implementation and packaging documentation omit robustness modules and the exact bundled provenance-source set. | Fixed | 6.9 | Open | Pending — corrected module and bundle maps with document tests from Task 6.9. |
| A15 | Low | Slide 4 lowercases Mild, Moderate, and Severe contrary to the approved vocabulary. | Fixed | 6.9 | Open | Pending — rebuilt slide XML and rendered slide inspection from Task 6.9. |
| A16 | High | The visual attestation is unbound and records stale counts rather than all current report pages, slides, and notes. | Fixed | 6.11 | Open | Pending — hash-bound contiguous visual-review record using detected final counts from Task 6.11. |
| A17 | Medium | PresentationWindow and LiveScenarioPage exceed the approximate 200-line UI coordination boundary. | Fixed | 6.8 | Open | Pending — extracted components, preserved behavior, and 1366x768 UI evidence from Task 6.8. |
| A18 | Medium | Large scientific modules increase maintenance risk, but a broad refactor would add unnecessary scientific regression risk. | Bounded limitation | 6.3 through 6.5 | Open | Pending — focused helper extraction only where required by Tasks 6.3-6.5, with regression tests. |
| A19 | High | The Windows package contains more than 2 GB of byte-identical CUDA DLL pairs. | Fixed | 6.10 | Open | Pending — actual-tree duplicate-hash scan, size budget, and packaged CUDA smoke from Task 6.10. |
| A20 | High | Release tests can pass conditionally when executable, smoke, or distribution artifacts are absent. | Fixed | 6.2 and 6.11 | Open | Pending — separate zero-skip portable and strict post-build release-artifact gates from Tasks 6.2 and 6.11. |
| A21 | Medium | A self-hashed unsigned manifest proves internal consistency but not authenticity against a coherent rewrite. | Optional signing hardening | 6.11 and 6.12 | Open | Pending — honest unsigned threat boundary and independently trusted Git commit anchor; signing only with a user-supplied identity. |
| A22 | High | An ambient GRIDGUARD_SMOKE_REPORT variable can suppress the normal GUI and redirect smoke output. | Fixed | 6.10 | Open | Pending — explicit private smoke argument, validated temporary path, and poisoned-environment tests from Task 6.10. |
| A23 | High | The release path relies on the sibling research archive even though installed runtime inputs must be committed, bound, and offline. | Fixed | 6.2 and 6.10 | Open | Pending — clean release-input proof without the archive and optional parity-only archive checks from Tasks 6.2 and 6.10. |
| A24 | Medium | Historical and current distribution versions are mixed, obscuring which package represents the reviewed release. | Fixed | 6.1 and 6.10 | Open | Pending — 1.2.0 metadata plus current-version-only built outputs from Tasks 6.1 and 6.10. |

## Closure protocol

1. Implement the owner task test-first and retain evidence-neutral acceptance: an adverse result is
   still valid when honestly derived.
2. Replace `Pending` with exact files, hashes, command records, rendered-review records, or remote
   SHA proof only after they exist.
3. Change a finding to `Closed` only when its complete evidence is reachable from the reviewed
   commit. Reopen it if a later regeneration invalidates that evidence.
4. Freeze the final release manifest last. A manifest generated from an uncommitted or subsequently
   modified tree is not closure evidence.
5. Keep Authenticode or detached signing optional unless the user supplies a signing identity; do
   not describe an unsigned artifact as authenticated.

Task 6.12 must audit every row against current files, commands, rendered artifacts, packaged
runtime behavior, and remote state before 1.2.0 can be called released.

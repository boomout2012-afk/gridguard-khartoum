# Review Remediation Ledger

This is the historical colleague-review ledger. Its recorded closures describe the August
remediation and do not certify later source changes. The October 2026 findings, verified source
fixes, and remaining graduation-readiness gates are tracked in
[GRADUATION_REVIEW.md](GRADUATION_REVIEW.md); final refreshed artifacts remain subject to that gate.

This ledger gives a clean English disposition for every substantive point in the colleague review.
The raw review snapshot was `point.md`, 29,537 bytes, with SHA-256
`1624389a763f283cae828a7bf5f0922b2acb724b8b687c88f19ad9c869cb3701`. It was removed after verified closure of all
14 dispositions; this ledger retains its identity and clean-English substance without preserving
the snapshot's damaged display encoding.

The approved remediation keeps the accepted proposal boundary: PPO and Fixed Rotation remain the
only controllers, the frozen benchmark and exported PPO actor remain immutable, and no dispatch
rule or PPO architecture is changed. New work is limited to honest reframing, training provenance,
documented-seed-disjoint evidence, sensitivity and forecast evaluation, documentation, and packaging.

## R01 — Structural weakness of Fixed Rotation

**Criticism:** Fixed Rotation cannot select Critical Only, does not observe available supply, and
requests Full Service for only three of seven localities while the remainder are Off. Its nearly
fixed Mild/Moderate critical-service result and large unused-supply gap can make PPO's utilization
advantage partly a consequence of a weak comparator. The review also treated the baseline's rising
utilization under worsening shortage as evidence of a defect.

**Verdict:** Partially valid. The structural comparison concern is valid. Verified cell aggregates
give Fixed Rotation critical-service ratios of approximately 0.4327, 0.4327, and 0.4135 and
utilization of approximately 0.5426, 0.6868, and 0.9073 for Mild, Moderate, and Severe shortages.
Mean unused available energy is approximately 2,158.7 MWh for Fixed Rotation versus 150.2 MWh for
PPO. However, utilization is served power divided by available usable power; its increase as supply
falls is a mathematical denominator effect, not by itself a simulator defect or evidence that Fixed
Rotation becomes smarter.

**Action:** Describe Fixed Rotation as supply-blind and limited to Off and Full Service. Attribute part
of the utilization gap to baseline design, keep results descriptive, and retain a stronger
supply-aware rule as future work rather than adding a new controller to the accepted comparison.

**Acceptance evidence:** The corrected explanation and rounded values must appear consistently in
Methodology, Results, Discussion, the reviewer-facing ledger, and document-claim tests.

## R02 — Episode reward presented as independent evidence

**Criticism:** The episode reward `(5C + 2U + 2J − 0.25S − 3O) / 9` contains several reported
outcomes, so presenting reward as an independent primary result overstates the comparison.

**Verdict:** Valid. Optimizing and reporting a training objective is normal, but it is not an
independent confirmation of the component outcomes.

**Action:** Preserve the internal `episode_reward` key for parity, relabel it publicly as
“Training-objective score (diagnostic),” and remove it from headline and favorable-result gates.
Use critical-outage locality-hours, Jain fairness, and utilization as the three primary outcomes.

**Acceptance evidence:** Canonical metric-role metadata, artifact validation, UI cards, report and
deck tables, and release tests must all enforce the primary/supporting/diagnostic distinction.

## R03 — Optimization traceability conflated with GPU acceleration

**Criticism:** The traceability matrix treated GPU acceleration as the proposal's mathematical
optimization method, although faster computation and optimization of dispatch decisions are
different concepts.

**Verdict:** Valid.

**Action:** State that Adam optimized PPO parameters during historical training, CUDA accelerates
inference and evaluation, and no separate online mathematical dispatch optimizer is implemented in
the documented-seed-disjoint comparison. Mark that proposal item as intentionally outside the
accepted scope.

**Acceptance evidence:** The proposal traceability row, methodology, discussion, and presentation
must use this distinction and must not imply that CUDA is an optimization algorithm.

## R04 — Missing domain Related Work

**Criticism:** The original eight references covered the proposal, methods, software, statistics,
and map data but did not position the work against load rationing, fair load shedding, critical-load
restoration, energy reinforcement learning, forecasting, or Sudan's energy context.

**Verdict:** Valid.

**Action:** Add a bounded Related Work chapter spanning those five domains and add seven vetted
domain sources as references [9]–[15]. Do not convert contextual literature into a calibration or
novelty claim.

**Acceptance evidence:** `docs/RELATED_WORK.md`, the 15-entry `docs/REFERENCES.md`, and
`tests/test_review_remediation.py` must agree on all citations and claim boundaries.

## R05 — Illustrative parameters and the unanswered realism question

**Criticism:** Locality demand weights, sector shares, the 700 MW peak, hourly profiles, and shortage
settings were locked for reproducibility but not derived from utility evidence. Available supply is
also constructed partly as a fraction of synthetic demand rather than from an independent generation
and network series, creating a structural demand–supply relationship. The locality names and map do
not make the scientific generator specific to Khartoum. The review correctly connected these gaps to
research question 3: “What types of data and rules are needed to simulate load-shedding decisions in
a realistic way?”

**Verdict:** Valid.

**Action:** Classify every important parameter as sourced, derived, proposal-constrained, or
illustrative. Describe the required calibration inputs: feeder and locality demand, supply and
generation constraints, network topology, critical-facility loads and priorities, outage and
restoration histories, operating and switching rules, weather/calendar effects, and validated
asset-to-locality mappings. Run one-factor sensitivity analysis by perturbing each locality weight
and sector share by ±25% and renormalizing the affected vector. Describe the case as synthetic and
Khartoum-inspired; geographic names and boundaries are presentation context, not electrical data.
Explain that simulation is suitable for a controlled algorithm comparison, while matched scenarios
do not remove misspecification in a generator shared by both controllers.

**Acceptance evidence:** A parameter-provenance table, an explicit RQ3 response, and the
hash-verified robustness supplement must disclose the base values, 24 variants, result
classifications, and remaining need for real calibration data.

## R06 — No within-hour priority during proportional dispatch

**Criticism:** When requested power exceeds supply, one proportional factor scales critical and
non-critical requested load together. Critical priority exists in tier selection but not as a
second dispatch stage within the hour.

**Verdict:** Valid limitation.

**Action:** Disclose the proportional-dispatch behavior and avoid interpreting the critical-service
metric as feeder-level hospital priority. Do not alter dispatch because the frozen PPO actor was
trained against this rule and changing it would invalidate parity.

**Acceptance evidence:** Methodology, limitations, Related Work, and tests must describe the same
dispatch semantics while the existing power-balance and parity tests remain unchanged.

## R07 — Excess precision in human-facing results

**Criticism:** Tables displayed float values with more digits than are useful or scientifically
meaningful for the synthetic model.

**Verdict:** Valid.

**Action:** Retain full precision in canonical JSON. In prose, UI, report, and slides use two
decimal places for percentages, one decimal place for MWh and locality-hours, and three decimal
places for the training-objective score. Reserve scientific notation for parity errors.

**Acceptance evidence:** Shared formatting contracts and document/deck tests must reject raw
machine-precision values in human-facing output while evidence recomputation retains exact values.

## R08 — Too few and not independently custodied evaluation seeds

**Criticism:** Seeds `50001–50003` yield only 96 independent base draws and provide a narrow basis
for generalization.

**Verdict:** Valid. The provenance audit found a stronger concern: those seeds were disjoint from
gradient updates and periodic validation, but the archived promotion process reused them during
model selection, so they are accepted reproduction evidence rather than an untouched final test.

**Action:** Preserve their artifact for historical parity and add documented-seed-disjoint seeds
`81001–81010`, 32 base draws per seed, crossed with all three shortage levels and both controllers.
Report the result honestly regardless of direction and do not use result favorability as a release
gate. This disjointness covers documented activity only; it is not preregistration or independent
custody.

**Acceptance evidence:** The training provenance must disclose the old-seed role; the robustness
artifact must record 320 independent draws, 960 matched scenario conditions, 1,920 controller
evaluations, deterministic intervals, and verified seed-set separation.

## R09 — Display forecast lacked evaluation

**Criticism:** The causal demand projection was described without MAE, RMSE, or a simple forecast
baseline.

**Verdict:** Valid.

**Action:** On the documented-seed-disjoint synthetic demand traces, compare the three-hour trailing
mean with one-hour persistence over common target hours 3–23. Report pooled and disaggregated MAE
and RMSE, alignment rules, and the lower-error method or tie. Keep the projection display-only.

**Acceptance evidence:** Forecast sufficient statistics and recomputed metrics must be stored in
the robustness supplement, with tests for causal alignment, common targets, and no real-world
accuracy claim.

## R10 — Request for a supply-aware controller baseline

**Criticism:** A supply-aware, critical-preserving rule would be a fairer comparator than Fixed
Rotation and would better isolate the value added by learning.

**Verdict:** Scientifically useful but excluded by the approved scope.

**Action:** Keep Fixed Rotation as the proposal baseline, explain its limitations, and list a
supply-aware heuristic as future work. Do not add a controller, change the headline comparison, or
import post-proposal experimental behavior without a separately approved study.

**Acceptance evidence:** Controller registries and evidence remain limited to PPO and Fixed
Rotation; documentation explicitly avoids treating the current comparison as superiority over all
reasonable rule-based policies.

## R11 — Missing PPO training parameters and convergence record

**Criticism:** A reviewer could not find the network, initialization, PPO hyperparameters, training
workload, validation, selection rule, seeds, optimizer settings, hardware, or convergence history.

**Verdict:** Strongly valid.

**Action:** Add `data/model/training-provenance.yaml` as a strict, hash-bound historical record,
link it from the model manifest and reviewer-facing documents, and generate summaries from the
validated record. Describe behavior cloning only as PPO initialization. Do not commit optimizer
state or raw training runs, and do not promise bitwise retraining.

**Acceptance evidence:** Strict YAML validation, pinned archive lineage, arithmetic and seed
invariants, model-manifest binding, package inclusion, and visible report/deck/UI training summaries
must pass together.

## R12 — No optimal or upper-bound comparator

**Criticism:** Without an optimization benchmark, the experiment cannot show how far PPO is from an
optimal allocation.

**Verdict:** Valid limitation.

**Action:** State that performance relative to the global optimum is unknown. Do not call a myopic
joint-action selector an episode-level upper bound. Treat a network-constrained or episode-level
optimization benchmark as future work.

**Acceptance evidence:** Limitations and Related Work must contain this boundary, and no public
artifact or presentation may use “optimal,” “upper bound,” or equivalent wording for PPO results.

## R13 — Exponential joint-action scaling

**Criticism:** The actor enumerates all three service tiers for each locality, so the output space is
`3^N` and will not scale well beyond seven localities.

**Verdict:** Valid limitation.

**Action:** Report the current `3^7 = 2,187` outputs and identify factored, autoregressive, or
shared-locality policies with explicit constraint handling as future research. Do not change the
accepted actor architecture.

**Acceptance evidence:** Architecture and limitation text must expose the scaling law and keep the
exported actor hash and action-decoding parity unchanged.

## R14 — Shortage-only operating domain

**Criticism:** The evidence covers Mild, Moderate, and Severe shortages but not surplus, recovery,
restoration transients, or normal operation.

**Verdict:** Valid limitation.

**Action:** Describe the evaluated domain as shortage-only. Do not infer behavior during surplus or
recovery even if lower-level data structures can represent additional supply. List those operating
states, with suitable constraints and transition rules, as future work.

**Acceptance evidence:** Scenario protocol, report, deck, UI limitations, and artifact metadata must
all restrict claims to the three matched shortage levels.

## Closure rule

A finding is closed only when its acceptance evidence passes on the committed repository and the
rendered report, deck, and packaged application agree with it. The raw review snapshot may then be
removed as approved; this clean ledger and its tests remain as the auditable disposition.

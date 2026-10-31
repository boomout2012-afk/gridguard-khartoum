# Thesis evidence and explanation improvements

31 October 2026. This record describes the further thesis revision following the
[committee response](COMMITTEE_REVISION.md). It is a change map, not a university grade
or an assertion that an examiner must award a particular mark.

## What changed

| Concern | Revised evidence or explanation |
| --- | --- |
| Exact synthetic inputs | Section 3.2 distinguishes the 700 MW normalization scale from the simultaneous profile peak. Appendix A.3 gives the hourly Gaussian profiles, coefficients, noise, supply equations and bounded critical margin. Table 1 no longer implies locality–sector pairings. |
| Reproducible learning method | Section 3.9 states finite-episode GAE, terminal handling, critic targets and advantage normalization with the original GAE reference. Section 3.10 states the learning-rate and entropy schedules including the zero final learning rate. |
| What the curves mean | Section 5.6 quantifies early gains and the later plateau, distinguishes common-iteration validation means from selected-policy test means, and explains why final near-zero KL is not proof of optimal convergence. |
| Who receives service | Figure 5.7 shows per-locality endpoint ratios and the mean within-condition minimum from the original saved study. These are separate from time-averaged Jain fairness and pooled energy ratios. |
| Contribution after imitation | Section 5.9 adds a prospectively frozen execution on fresh seeds, comparing each selected PPO actor with its own saved behavior-cloning initialization and four deterministic baselines. All 13,440 trajectories were replayed. |
| Adverse equity outcome | Tables 17–18 and the discussion retain lower energy/utilization after fine-tuning and lower final worst-locality service than the teacher and critical-aware rotation despite higher Jain fairness. |
| Literature and attribution | Section 2.8 compares the closest fair-allocation studies more specifically. GAE has its own Harvard reference, and geographic dataset entries have retrievable provenance. |
| Reader navigation | Word contents include the lists of figures, tables and abbreviations. Figure-list entries use short titles while full explanatory captions remain with the figures. |

## Scientific interpretation

The original five-run study, selected policies, primary criteria and failed optimizer
comparison remain unchanged. The earlier teacher/critical-aware follow-up used the
original conditions and remains explicitly post-hoc. The new audit uses 320 independent
base draws and their three coupled shortage variants, with five paired learned runs.
Its pointwise nominal 95% intervals are descriptive rather than a new family of
confirmatory success claims.

Fine-tuning plus validation selection increases daily reward by 0.1801 relative to the
same runs' initial actors, while serving 77.20 fewer MWh/day. This is a result about the
whole stage, including extra computation and selection. It does not isolate the PPO
algorithm, equalize training budgets, establish learning from scratch or test new demand
mechanisms. Higher Jain fairness does not improve every locality or every equity measure.

The application and defense presentation retain the original benchmark and preceding
baseline audit. The new learning-stage study is a separately identified thesis analysis;
it changes neither the deployed actor nor the app's frozen displayed results.

## Review and reproduction

The [learning-stage protocol](THESIS_LEARNING_AUDIT_PROTOCOL.md),
[evidence](../data/evidence/thesis-learning-audit.json) and
[full-replay receipt](../data/evidence/thesis-learning-audit.validation.json) retain the
new design, all per-condition outcomes and verification identity. The
[reviewer guide](REVIEWER_GUIDE.md) provides verification/replay commands. Descriptive
original-study reductions are retained under `docs/assets/focused-study/` with their
source and figure hashes. Independent review remains a judgment about the submitted
artifacts, not a replacement for these reproducible checks.

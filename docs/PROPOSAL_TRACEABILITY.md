# Proposal Traceability and Source Audit

Status: Final release traceability
Audit date: 2026-08-26

## Accepted scope baseline

The only scope authority for this clean project is
`docs/source/graduation-project-proposal.pdf`, SHA-256
`3432e64a21ab9741bf47646fd0f4f561522ddfd81d4005deb892d38672c65e48`.

The proposal requires a simplified software simulation, an AI decision-support model, shortage
scenarios, critical-service priorities, comparison with fixed scheduling, proposal-aligned
results, a final report, and an academic presentation. It explicitly excludes direct control of
a real national grid.

## Verified source foundation

The read-only sibling source repository was inspected without modification. The accepted
benchmark and training record identify commit `52f6dd8` as their source revision, so every
migrated behavior below is pinned to the exact file bytes at that commit rather than a later
research revision.

| Source item | Evidence | Source SHA-256 | Source commit |
|---|---|---|---|
| Frozen controller benchmark | `artifacts/benchmark.json` | `5f09625ff987440bc6e667550509cb7f1deac03a38c1b905aa284d65bf74023f` | `52f6dd82605c993c8094448caecec86faa289c0c` |
| Trained PPO checkpoint | `artifacts/checkpoints/best.pt` | `c6e5712f966b46ba18f4c69fb457b9e69f78887c0c5c0279e5ce034f40f9c665` | `52f6dd82605c993c8094448caecec86faa289c0c` |
| Simulation configuration | `src/aipower_rl/config.py` | `5f1f9febaaf7cdae980ed67c83948362b378cd290118f5712f47da1a09d65d04` | `52f6dd82605c993c8094448caecec86faa289c0c` |
| Fixed Rotation controller | `src/aipower_rl/controllers.py` | `b75eaf60da7d3c4e5225327d05029c8b5191583c4853bb17b8bc32f228118538` | `52f6dd82605c993c8094448caecec86faa289c0c` |
| Scenario generator | `src/aipower_rl/scenarios.py` | `689bb4319194baf2f593d98c4b3309d843167e039ed48fff20ee0a9b14aef186` | `52f6dd82605c993c8094448caecec86faa289c0c` |
| Simulator | `src/aipower_rl/simulator.py` | `88df6575a6b2821cde6abcd63f2effe81596210da364bc02bd416b0981e658e6` | `52f6dd82605c993c8094448caecec86faa289c0c` |
| Dispatch and reward | `src/aipower_rl/dynamics.py` | `4fe504105c5f030a143f5583a788de9a4ba9c028b34d49aabc8ff9c3da64c9d8` | `52f6dd82605c993c8094448caecec86faa289c0c` |
| PPO actor network | `src/aipower_rl/models.py` | `6fedcc55675a72aff048a3b1a9a98c86a87962427031c6a4450239fdccbbde03` | `52f6dd82605c993c8094448caecec86faa289c0c` |
| Proposal metrics | `src/aipower_rl/evaluation.py` | `1d0aeef17ed84c3184731cc89c3a39c7f982981183c98047c92fb9c6ffd4cf68` | `52f6dd82605c993c8094448caecec86faa289c0c` |
| Training manifest | `artifacts/training_manifest.json` | `3b866b6f470e116ff009022032059a1f91b509fadc54dbff6c5962a8f2236a59` | `52f6dd82605c993c8094448caecec86faa289c0c` |
| Training metrics | `artifacts/training_metrics.jsonl` | `f48d2f6da29682e1c47330aa2b33b6bb4941f4e0e248f2c846436f09a4b89bc4` | `52f6dd82605c993c8094448caecec86faa289c0c` |
| Selected experiment summary | `experiments/s2-600-hidden512.json` | `619eb67aa253a35d7a7dc0efdbc6c743d7b0af6db4687575e7fade9dbcfe940b` | `52f6dd82605c993c8094448caecec86faa289c0c` |
| Training orchestration | `src/aipower_rl/training.py` | `886931d6c12b0ef45f4f17b89d134e27682c68a2dc4c2f90e2911e2dfec708fb` | `52f6dd82605c993c8094448caecec86faa289c0c` |
| PPO learning algorithm | `src/aipower_rl/learning.py` | `18cf22b0b58584b2abef164ec3ea956fa2f61197336a67a7b9c1f4faefc4fac2` | `52f6dd82605c993c8094448caecec86faa289c0c` |
| Experiment selection | `src/aipower_rl/experiments.py` | `605e32626caece41508f37cbfac0cb061f7510118ba463cf3ff5843f92a9d753` | `52f6dd82605c993c8094448caecec86faa289c0c` |

The benchmark hash identifies the exact audited Windows-checkout bytes. Git stores the same JSON
with normalized LF line endings; `source-manifest.json` records and verifies both byte
representations (`5f09625f...` for the checkout and `37d7e51c...` for the Git blob). All other
listed source hashes equal their Git-blob hashes.

The clean implementation uses parity tests and inference-only artifacts instead of copying the
full research archive.

### Verified display geography

The presentation map is separately source-pinned and never treated as proposal evidence.
OCHA/HDX Sudan COD-AB revision v03 supplies the exact Khartoum State ADM2 units `SD01001` through
`SD01007`; Natural Earth 1:10m rivers version 5.0.0 supplies White Nile, Blue Nile, and Nile
context. The upstream archive, selected member, immutable river revision, derived GeoJSON hashes,
selectors, transformation, licenses, and attribution are recorded in
`src/gridguard/assets/map/THIRD_PARTY_DATA.md` and `map-provenance.json`. The boundaries are
presentation context only and not electrical-network topology.

## Accepted reproduction and untouched confirmation

The historical artifact records 32 episodes for each of nine shortage-level/seed cells. It stores
only cell summaries and does not expose the common base draws reused across shortage levels. The
clean paired artifact therefore reports all three counts: **96 independent base draws, 288 matched
scenario conditions, and 576 controller-scenario evaluations**. Its seeds were reused in archived
model selection, so this is accepted reproduction evidence rather than an untouched final test.
The separate untouched confirmatory supplement uses seeds 81001-81010: **320 independent base
draws, 960 matched scenario conditions, and 1,920 controller-scenario evaluations**.

The following table is the **accepted reproduction projection** retained for historical parity; it
is not the untouched reviewer-facing result table.

| Metric | Fixed Rotation | PPO | Direction |
|---|---:|---:|---|
| Training-objective score (diagnostic) | 13.969 | 23.512 | Higher; diagnostic only |
| Critical-service ratio | 42.63% | 99.72% | Higher; supporting |
| Supply utilization | 71.22% | 97.65% | Higher; primary |
| Jain fairness | 87.76% | 96.92% | Higher; primary |
| Critical-outage locality-hours | 102.1 | 18.7 | Lower; primary |
| Served energy | 4,551.7 MWh | 6,560.2 MWh | Higher; supporting |
| Curtailed energy | 6,142.4 MWh | 4,133.9 MWh | Lower; complementary |
| Switching rate | 25.00% | 21.89% | Lower is better |
| Power-balance violation rate | 0.00% | 0.00% | Must remain zero |

The clean project resolves the historical count ambiguity with
`data/evidence/paired-benchmark.json`. Its episode rows and confidence intervals were regenerated
from the audited simulator and frozen PPO checkpoint; they were not present in the historical
aggregate-only artifact.
`data/evidence/robustness-supplement.json` adds the untouched seed set, 24 one-factor-at-a-time
sensitivity variants (25 parameterizations with the base), and a one-hour-ahead forecast
comparison using a three-hour trailing window against one-hour persistence. Neither verifier
requires a favorable result.

## Proposal coverage matrix

| Proposal requirement | Source status | Clean-repository acceptance evidence |
|---|---|---|
| Simplified shortage simulation | Implemented among broader experimental branches | Small proposal-equivalent simulator, severity-neutral base-draw reuse, parity tests, architecture documentation |
| Supply, demand, shortage levels, sectors | Implemented | Typed domain contract and deterministic Mild/Moderate/Severe scenarios |
| AI decision-support model | Trained PPO checkpoint exists | Inference-only checkpoint, CUDA loader, policy parity tests |
| Critical infrastructure priorities | Hospitals, water, and bakeries modeled | Per-sector map detail, critical-service invariant tests |
| Prediction method | Forecast work existed outside the accepted path | Causal one-hour-ahead trailing-window projection for display only, isolated from PPO; synthetic MAE/RMSE compared with one-hour persistence on common targets |
| Rule-based method | Fixed scheduling existed | Fixed Rotation is the primary baseline and its behavior is tested and documented |
| Optimization method | Adam used in historical training | Training YAML records Adam parameter optimization; CUDA accelerates inference/evaluation; no separate online mathematical dispatch optimizer is claimed |
| Reinforcement learning | PPO policy implemented | Clean inference code and audited PPO methodology |
| Multiple shortage scenarios | Mild, Moderate, Severe implemented | Matched deterministic CUDA replay and paired evidence across all three levels |
| Fixed-schedule comparison | Implemented | Accepted 288-row reproduction plus untouched 960-condition supplement, cluster-aware intervals, sign-neutral integrity, and Results view; Fixed Rotation is disclosed as supply-blind Off/Full only |
| Critical-service downtime | Implemented in evaluation metrics | Frozen table plus per-scenario GUI detail |
| Fairness between areas | Jain fairness and critical-outage locality-hours implemented | Frozen table, definitions, and tests |
| Distribution efficiency | Utilization, served MWh, curtailment implemented | Frozen table, definitions, and tests |
| Data and rules needed for realism (RQ3) | Historical work used illustrative inputs | Parameter-provenance table plus required feeder/locality demand, independent supply/generation, topology, critical-facility, outage/restoration, operating-rule, weather/calendar, and asset-mapping inputs |
| PPO training traceability | Archive contains manifests, metrics, experiment selection, and source | Strict hash-bound `data/model/training-provenance.yaml`: architecture, behavior-cloning initialization, PPO parameters, 600 iterations, 153,600 episodes, 3,686,400 transitions, seeds, selection, hardware, convergence, and source hashes |
| Final report | Earlier research material was fragmented | Integrated report with methodology, implementation, results, limitations, and references |
| Academic presentation | Earlier dashboard lacked the approved focused flow | Offline native desktop app, verified administrative map, synchronized accessible locality list, live aggregate table, Windows build, deck, and speaker notes |

## Resolved migration risks

1. Out-of-scope experimental branches and claims were excluded from the clean project.
2. Product, controller, locality, service-tier, and shortage terminology was normalized.
3. Frozen evidence is isolated from the interactive demonstration and cannot be replaced by live
   scenario state.
4. Raw runs, optimizer state, caches, and unrelated research artifacts were excluded.
5. Ambiguous episode wording was replaced with separate independent base-draw, matched-condition,
   and controller-evaluation counts.
6. Every migrated source is commit-pinned and byte-hashed, and final release artifacts are bound in
   `RELEASE_MANIFEST.json`.
7. Geographic realism is isolated from scientific inputs: the OCHA/HDX boundaries and Natural
   Earth Nile context are verified presentation assets, not electrical-network topology.
8. Evidence integrity and frozen parity are sign-neutral; validation cannot silently encode a
   favorable research conclusion.
9. The accepted reproduction is distinguished from the untouched confirmatory seed set.
10. Reward is a training-objective diagnostic; outage burden, fairness, and utilization are the
    three primary public outcomes, with power balance shown separately as safety.

## Migration rule

The sibling repository remains an immutable research archive. Nothing in it was deleted, renamed,
or rewritten for this delivery. Only proposal-relevant behavior and audited artifacts were
migrated, and every migrated behavior has a clean test or recorded parity check.

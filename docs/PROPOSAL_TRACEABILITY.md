# Proposal Traceability and Source Audit

Status: Final release traceability
Audit date: 2026-08-15

## Accepted scope baseline

The only scope authority for this clean project is
`docs/source/graduation-project-proposal.pdf`, SHA-256
`3432e64a21ab9741bf47646fd0f4f561522ddfd81d4005deb892d38672c65e48`.

The proposal requires a simplified software simulation, an AI decision-support model, shortage
scenarios, critical-service priorities, comparison with fixed scheduling, proposal-aligned
results, a final report, and an academic presentation. It explicitly excludes direct control of
a real national grid.

## Verified source foundation

The read-only sibling source repository was inspected at commit `fd2ba18`. The accepted benchmark
identifies commit `52f6dd8` as its source revision, so every migrated behavior below is pinned to
the exact file bytes at that commit rather than a later research revision.

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

## Frozen headline comparison

The historical artifact records 32 episodes for each of nine shortage-level/seed cells. It stores
only cell summaries and does not expose the common base draws reused across shortage levels. The
clean paired artifact therefore reports all three counts: **96 independent base draws, 288 matched
scenario conditions, and 576 controller-scenario evaluations**.

| Metric | Fixed Rotation | PPO | Direction |
|---|---:|---:|---|
| Episode reward | 13.9693 | 23.5121 | Higher is better |
| Critical-service ratio | 42.63% | 99.72% | Higher is better |
| Supply utilization | 71.22% | 97.65% | Higher is better |
| Jain fairness | 0.8776 | 0.9692 | Higher is better |
| Served energy | 4,551.69 MWh | 6,560.19 MWh | Higher is better |
| Curtailed energy | 6,142.39 MWh | 4,133.89 MWh | Lower is better |
| Switching rate | 25.00% | 21.89% | Lower is better |
| Power-balance violation rate | 0.00% | 0.00% | Must remain zero |

The clean project resolves the historical count ambiguity with
`data/evidence/paired-benchmark.json`. Its episode rows and confidence intervals were regenerated
from the audited simulator and frozen PPO checkpoint; they were not present in the historical
aggregate-only artifact.

## Proposal coverage matrix

| Proposal requirement | Source status | Clean-repository acceptance evidence |
|---|---|---|
| Simplified shortage simulation | Implemented among broader experimental branches | Small proposal-equivalent simulator, severity-neutral base-draw reuse, parity tests, architecture documentation |
| Supply, demand, shortage levels, sectors | Implemented | Typed domain contract and deterministic Mild/Moderate/Severe scenarios |
| AI decision-support model | Trained PPO checkpoint exists | Inference-only checkpoint, CUDA loader, policy parity tests |
| Critical infrastructure priorities | Hospitals, water, and bakeries modeled | Per-sector map detail, critical-service invariant tests |
| Prediction method | Forecast work existed outside the accepted path | Causal trailing-window projection for display only, isolated from PPO and evidence |
| Rule-based method | Fixed scheduling existed | Fixed Rotation is the primary baseline and its behavior is tested and documented |
| Optimization method | GPU optimization reference exists | Training context is documented; the desktop uses GPU acceleration for inference and simulation |
| Reinforcement learning | PPO policy implemented | Clean inference code and audited PPO methodology |
| Multiple shortage scenarios | Mild, Moderate, Severe implemented | Matched deterministic CUDA replay and paired evidence across all three levels |
| Fixed-schedule comparison | Implemented | Rebuilt 288-row paired artifact with cluster-aware intervals, neutral integrity validation, trusted release assertion, and Results view |
| Critical-service downtime | Implemented in evaluation metrics | Frozen table plus per-scenario GUI detail |
| Fairness between areas | Jain fairness and critical-outage locality-hours implemented | Frozen table, definitions, and tests |
| Distribution efficiency | Utilization, served MWh, curtailment implemented | Frozen table, definitions, and tests |
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
8. Neutral evidence integrity is testable independently of the stronger trusted release gate, so
   validation cannot silently encode a favorable research conclusion.

## Migration rule

The sibling repository remains an immutable research archive. Nothing in it was deleted, renamed,
or rewritten for this delivery. Only proposal-relevant behavior and audited artifacts were
migrated, and every migrated behavior has a clean test or recorded parity check.

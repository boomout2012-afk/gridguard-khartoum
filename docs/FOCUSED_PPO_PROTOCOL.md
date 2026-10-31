# Focused PPO study against Fixed Rotation and constrained optimization

23 October 2026

This additive study implements the user-authorized fivefold training budget and focuses
the main comparisons on the proposal's fixed-schedule and optimization approaches. Its scope
remains the seven-locality, 24-hour synthetic decision-support simulation. The proposal
requires comparison and evaluation; it does not require a predetermined PPO victory.
All earlier model, protocol and evidence files remain unchanged.

The protocol, configuration and scientific source hashes are frozen before study
training and validation. Final-test scenarios are not generated or inspected until all
five runs and their validation selections have been frozen. The preceding Fairness Rule
and optimization studies remain disclosed as additional analyses, with separate results.

## Methods and attribution

The main comparisons are **newly trained constrained PPO versus Fixed Rotation and
Constrained Optimization**, equally
averaged across training seeds 1729, 2718, 3141, 5772 and 8111. Fixed Rotation retains the
existing three-hour rotation of Off/Full Service tiers. PPO chooses from 128 combinations
of Critical Only/Full Service, subject to the same explicit critical-service constraint
used in the previous training study:

    C + sum_i x_i b_i <= S,  x_i in {0,1}

Here C is aggregate critical demand, S available supply and b_i locality noncritical
demand. Direct comparison uses no extra MW feasibility slack. If no candidate fits,
PPO uses the all-Critical-Only fallback with existing proportional dispatch; fallback
conditions are counted. The default generator is expected to make that fallback
unnecessary. Feasibility masking is applied to demonstration training, rollout sampling,
PPO likelihood recomputation and final inference. Stored rollout masks remain fixed
during PPO updates.

The comparison tests the **complete constrained PPO system against Fixed Rotation**.
Because their action rules and critical-service constraints differ, any advantage cannot
be attributed to learning alone. Critical protection is an explicit allocation rule.
Both methods receive matched causal scenarios; neither receives future samples.

Constrained Optimization exhaustively maximizes the current-hour canonical reward over
the same 128 critical-feasible candidates and receives the same exact causal state as
PPO. It does not plan the full day. This stronger comparison evaluates objective-specific
advantages of the learned policy under matching allocation constraints; it cannot show
superiority to every optimization method or a full-horizon optimum.

Fairness Rule with epsilon 0.005 supplies behavior-cloning demonstrations. Its teacher
role is disclosed even though it is not a primary comparator in this focused study.
Five same-run iteration-600 actors provide the secondary budget reference. The final
evaluation therefore has 12 methods: Fixed Rotation, Constrained Optimization, five
selected PPO actors and five iteration-600 actors. Demonstration-only actors are retained
in training provenance.

The secondary comparison asks whether continuing this training schedule beyond iteration
600 improves the selected procedure. The iteration-600 snapshots belong to a planned
3,000-iteration schedule; their learning-rate/entropy settings differ from a separately
tuned 600-iteration recipe. They are an exploratory within-run budget reference, not a
claim about the globally best use of a smaller compute budget.

## Fixed training recipe

Each run collects 455 teacher episodes per severity: 1,365 episodes and 32,760 states.
Behavior cloning uses ten epochs, batch size 1,024 including the final partial batch,
Adam learning rate 0.0003, and masked cross-entropy plus 0.25 times critic mean squared
error against undiscounted teacher return-to-go. PPO starts a fresh optimizer after
imitation.

Each run completes **3,000 iterations × 256 episodes × 24 steps = 18,432,000 PPO
transitions**. Five runs total **92,160,000 transitions**, excluding teacher collection
and evaluation. The 85/85/86 severity allocation rotates its extra episode with iteration
index modulo three, giving equal total severity counts. The budget is not extended based
on observed validation or final-test outcomes.

Actor and critic are separate two-hidden-layer networks with 256 tanh units per layer,
orthogonal initialization, 128 actor logits and one critic output. The 46 features are
the existing 32 causal features plus seven cumulative-demand and seven cumulative-served
energy values, normalized by 24 times fixed locality peak MW. Final decisions use the
largest feasible logit, breaking ties with the lowest candidate ID.

| Setting | Value |
| --- | --- |
| Discount / GAE lambda | 1.0 / 0.95 |
| PPO clipping | 0.2 |
| Epochs / minibatch size | 4 / 1,024 |
| Value coefficient | 0.5 multiplying half mean squared error |
| Adam learning rate | 0.0003 linearly to zero |
| Entropy coefficient | 0.01 linearly to 0.001 |
| Maximum gradient norm | 0.5 |

Schedules use zero-based iteration k divided by 2,999. Advantages are normalized over
the complete rollout before minibatch updates. Terminal bootstrap value is exactly zero;
episodes are not linked. The canonical reward, metric definitions, physics and generator
remain unchanged. Training is float32 with deterministic algorithms and TF32 disabled.

For run index j and iteration k, scenario seeds are 4,000,000 + 10,000j + k. Teacher
scenario seeds are 4,500,000 + 10,000j, shared across coupled shortage variants. Network
initialization uses the training seed; separate CUDA generators use seed + 1 for policy
sampling, seed + 2 for PPO minibatch order and seed + 3 for imitation minibatch order.

## Splits and validation selection

| Split | Scenario seeds | Base draws per seed | Independent base draws | Conditions |
| --- | --- | ---: | ---: | ---: |
| Validation | 122101–122105 | 32 | 160 | 480 |
| Final test | 122201–122210 | 32 | 320 | 960 |

Each base draw has matched Mild, Moderate and Severe variants. These sets are disjoint
from training, demonstrations and earlier published studies. All methods use the exact
same inputs for each condition, protected by mutation guards and input hashes.

Evaluate iteration zero and every 100 iterations through 3,000: 31 checkpoints per run.
A checkpoint is eligible when every validation condition has zero PPO critical-outage
and critical-interruption locality-hours and zero power-balance violations, and PPO mean
fairness and utilization are each at least Fixed Rotation both overall and within every
severity. Fixed Rotation is allowed its observed critical interruptions.

Select the highest overall daily canonical reward among eligible checkpoints; break exact
ties with the earliest iteration. If none is eligible, retain the highest-reward checkpoint
under the same tie rule, mark it ineligible, and forbid a positive primary claim. Keep all
31 validation summaries. Freeze the selected actor, raw iteration-600 actor, demonstration
actor, training record, configuration and source/protocol hashes before final evaluation.

The optional deployment candidate is the eligible run with highest validation reward,
breaking ties with the lowest training seed; if none is eligible, explicitly mark the
same highest-reward choice as a fallback. Its result never replaces the five-run result.
No final-test-best seed, checkpoint, horizon or severity-specific policy is selected.

## Final outcomes and decisions

Final evaluation contains **960 conditions × 12 methods = 11,520 trajectories**. Report
every run and the equal-run aggregate, including energy, critical-service ratio,
critical/zero-service interruption durations and streaks, Jain fairness, utilization,
switching, oversubscription, power violations and final minimum locality service. Report
all severity means and negative results. Zero interruption duration does not imply equal
Full Service when Critical Only protects every locality.

The Fixed Rotation primary claim is **the constrained PPO system improves critical service, fairness
and utilization relative to Fixed Rotation**. It requires all of the following:

1. The upper endpoint of the adjusted 97.5% interval for PPO minus Fixed Rotation critical-interruption
   locality-hours is strictly below zero.
2. The lower endpoints for Jain-fairness and utilization differences are both strictly
   above zero.
3. Every selected PPO actor has zero critical-outage/interruption locality-hours and zero
   power-balance violations in every test condition. The Fixed Rotation comparator is
   not required to have zero interruptions.
4. Every severity has nonpositive mean critical-interruption difference and nonnegative
   mean fairness/utilization differences.
5. All five validation selections are eligible.

Daily reward is a diagnostic in the Fixed Rotation comparison, not a substitute for the
proposal's critical-service, fairness and distribution-performance outcomes.

The optimizer primary claim is **PPO improves daily reward under service guardrails
versus this constrained optimizer**. The adjusted 97.5% interval for PPO minus Constrained
Optimization reward must have lower endpoint strictly above zero. Fairness and utilization
lower endpoints must each be at least -0.002. Every severity must have nonnegative mean
reward difference and fairness/utilization differences at least -0.002. Every selected
PPO actor and Constrained Optimization must have zero critical interruptions/outages and
power violations in every condition, and all five selections must be eligible. Report
the effect size and separately whether the mean and adjusted lower endpoint exceed the
earlier practical reference of +0.01 daily reward. A statistically positive tiny effect
must not be relabeled as an established operational improvement.

The secondary claim of **benefit from additional training in this schedule** requires:
daily reward difference versus paired iteration-600 actors has 95% lower endpoint above
zero; fairness and utilization lower endpoints are at least -0.002; every severity has
nonnegative mean reward difference and fairness/utilization differences at least -0.002;
both sets of PPO actors satisfy zero critical/power violations; all selections are eligible;
and **all selected iterations are greater than 600**. Otherwise report the difference as
an inconclusive result, earlier-checkpoint selection or tradeoff, as applicable. The
0.002 ratio margin is 0.2 percentage points and is a research choice, not an operational
standard. Do not infer benefit merely because the training budget was larger.

## Uncertainty, evidence and limitations

Use 10,000 paired hierarchical bootstrap resamples with seed 20261024. Resample training
runs and base-draw clusters within each scenario-seed stratum, sharing sampled scenario
clusters across runs and methods and preserving the three coupled severities. Use
equal-run, equal-condition means. The two primary comparison claims use Bonferroni-adjusted
97.5% two-sided intervals, with linear 1.25th/98.75th percentile endpoints, to account for
the two-comparison family. Each claim requires all its predeclared outcome gates. Also
report descriptive 95% intervals, secondary budget-comparison 95% intervals, and per-run
conditional 95% intervals, using 2.5th/97.5th percentile endpoints. Severity-specific
hierarchical 95% intervals are exploratory; do not select a favorable subgroup as a new
confirmatory claim. Severity mean guardrails remain part of the predeclared primary gates.
The study has only five
independent training runs; 960 conditions do not represent 960 independent policies.

Simulation, training, candidate scoring, metrics and bootstrap stay on CUDA; CuPy uses
the existing DLPack boundary. Final scientific arrays have one packed host transfer for
portable artifact assembly. Small validation/telemetry summaries may synchronize with
the host. Record all model/config/source hashes, inputs, action hashes, per-condition
metrics, final locality service ratios and selection decisions. Verify schema, labels,
counts, physical identities, matching and interval/claim reconstruction against pinned
hashes. Do not substitute the public pooled critical-service ratio for the hourly mean
when reconstructing Fixed Rotation's reward.

Report total training and evaluation elapsed time; no decision-latency claim is planned.
This experiment adds no storage, forecasts, network physics or real-grid validation. A
positive result establishes performance on the declared comparator and outcomes, not
universal RL superiority, superiority over all optimization, learning's isolated
contribution against Fixed Rotation, or superiority to the previously tested Fairness Rule.

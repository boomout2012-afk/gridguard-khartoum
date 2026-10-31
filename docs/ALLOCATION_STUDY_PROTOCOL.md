# Critical-service allocation study protocol

23 October 2026

This additive experiment tests whether constrained optimization, a deterministic fairness
rule, or a frozen-PPO preference filter improves scheduling in GridGuard Khartoum. Its scope
is the existing seven-locality, 24-hour synthetic decision-support simulator. It does not
establish electrical-network feasibility or performance on a real grid.

The protocol is declared before inspecting validation and final-test outcomes. Original
model files, benchmark evidence and proposal-extension evidence remain unchanged. New
results belong to a separate allocation study and never replace frozen benchmark values.

## Methods and common allocation rule

The final comparison uses these exact labels:

| Controller | Definition |
| --- | --- |
| Fixed Rotation | Existing fixed three-hour rotation, unchanged |
| PPO | Existing frozen actor and action selection, unchanged |
| Optimization | Existing exhaustive current-hour canonical-reward optimizer, unchanged |
| Constrained Optimization | Maximize current-hour canonical reward over the common critical-feasible candidate set |
| Fairness Rule | Within the reward-tolerance candidate set, maximize the worst projected cumulative locality service ratio; break ties by immediate reward, then lowest action ID |
| Feasible PPO | Select the frozen actor's highest-logit action from the common critical-feasible candidate set, without a reward-tolerance restriction; break ties by immediate reward, then lowest action ID |
| PPO Hybrid | Within the reward-tolerance candidate set, select the frozen actor's highest-logit action; break ties by immediate reward, then lowest action ID |

For the four constrained methods, every locality receives Critical Only or Full Service.
Let C be aggregate critical demand, S available supply, and b_i the noncritical demand of
locality i. The 128 candidate allocations satisfy:

    x_i in {0, 1}
    C + sum_i x_i b_i <= S

Candidate feasibility uses direct floating-point comparison with zero additional MW slack.
The all-Critical-Only
candidate guarantees a nonempty feasible set whenever critical demand fits available
supply. If S < C and no feasible candidate exists, all four constrained methods use the
same all-Critical-Only fallback and the existing proportional dispatch. Fallback use is
counted and reported; zero critical interruptions are not guaranteed for such inputs.
The default generator is expected to make this fallback unnecessary.

The existing dispatch calculation, canonical reward, action encoding, scenario generator
and public metric definitions remain unchanged. For a given state, let r* be the largest
canonical immediate reward among critical-feasible candidates. The reward-tolerance set
contains feasible actions whose immediate reward is at least r* - epsilon, with zero
additional numerical comparison slack. Epsilon is measured in
normalized per-step canonical-reward units, not percentage points or episode reward.
For both tunable families, epsilon zero bypasses preference ranking and exactly recovers
Constrained Optimization, including its lowest-action-ID reward tie rule.

Projected cumulative service for locality i is cumulative served energy including the
candidate's current-hour service divided by cumulative demand including current demand,
using the existing numerical denominator convention. The Fairness Rule maximizes the
minimum of these seven ratios. This max-min criterion is distinct from Jain fairness,
which remains a reported outcome and a validation guardrail.

All four constrained methods receive the same exact causal current demand, supply,
cumulative demand, cumulative served energy and previous service tiers. The PPO preference
head uses its unchanged 32-feature observation. Raw PPO receives cumulative service ratios
but not exact cumulative-energy magnitudes; it is therefore a reference method rather than
an information-identical ablation. No controller receives the future scenario trace.

PPO Hybrid is a post-processing combination of optimization and a frozen actor. There is
no retraining, new value-function learning, forecasting or lookahead in this study. Feasible
PPO isolates the effect of feasibility masking without the reward filter. The frozen actor's
historical optimizer-imitation initialization remains an attribution limitation.

## Scenario split and matching

| Split | Seeds, inclusive | Base draws per seed | Independent base draws | Matched conditions |
| --- | --- | --- | --- | --- |
| Validation | 102101 through 102105 | 32 | 160 | 480 |
| Final test | 102201 through 102210 | 32 | 320 | 960 |

Every base draw has coupled Mild, Moderate and Severe variants. All methods in a split
receive the exact same scenario object for each condition. Scenario mutation guards and
input hashes verify matching. Every episode lasts 24 one-hour samples. Split seed sets are
disjoint, and final-test outcomes must not participate in selection or further tuning.

Validation evaluates exactly 15 configurations: Constrained Optimization, seven Fairness
Rule settings and seven PPO Hybrid settings. Each family uses this fixed epsilon grid:

    0, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05

Thus validation contains 7,200 controller episodes. Final evaluation uses the seven
controllers listed above, with one selected epsilon per tunable family, for 6,720 controller
episodes. No final-test epsilon sweep is performed.

## Validation selection and freeze

Fairness Rule and PPO Hybrid select epsilon independently by the same rule. A setting is
eligible only when it has zero critical-interruption locality-hours and zero power-balance
violations in every validation condition, mean utilization no more than 0.002 below
Constrained Optimization, and mean Jain fairness at least that of Constrained Optimization.
Both utilization and Jain fairness are ratios; 0.002 is 0.2 percentage points.

Among eligible settings, select the highest mean daily canonical episode reward. Resolve
equal selection scores with the lowest epsilon. If no setting is eligible, select epsilon
zero and record that the fallback selection rule was used. This fallback does not excuse
failure of any final-test acceptance condition.

Selection produces a JSON record containing all validation configurations and outcomes,
eligibility decisions, selected epsilons, selection-rule identity, model hashes and hashes
of the scientific sources and this protocol. Freeze that record before generating or
inspecting final-test outcomes. The final-test tool must reject changed source or protocol
hashes, changed model hashes, changed settings, or a nonconforming selection record. The
recorded source list must cover the allocation logic, evaluation and evidence logic,
scenario generation, dispatch, reward, metrics, bootstrap and model-loading code actually
used by the study.

## Outcomes and primary decision rule

Report existing energy, critical-service, Jain-fairness, utilization, switching,
oversubscription, power-balance, zero-service blackout and critical-interruption outcomes.
Include direct locality durations and streaks, final minimum locality service,
fallback counts, daily canonical reward and decision timing. Report aggregate and
shortage-specific results, including unfavorable and zero-valued findings. Duration
equality alone does not demonstrate good service.

The final locality service ratio is total energy delivered to that locality over the
24 hours divided by its total demand. Exactly zero demand is assigned ratio one; positive
demand is divided by its actual magnitude. Take the minimum of the seven locality ratios
within each episode, then average those episode minima. This final-time diagnostic does
not replace canonical Jain fairness, which averages hourly cumulative service fairness.

The primary claim is narrowly named **hybrid reward improvement with service guardrails**.
It requires every condition below against both Constrained Optimization and the
validation-selected Fairness Rule:

1. The lower endpoint of the paired 95% interval for PPO Hybrid minus comparator daily
   canonical reward is strictly greater than 0.01.
2. The lower endpoint of the paired 95% interval for each of utilization and Jain fairness
   is at least -0.002.
3. Within each of Mild, Moderate and Severe, mean utilization and Jain-fairness differences
   are each at least -0.002, and mean reward difference is at least zero.
4. All four constrained methods have zero critical-interruption locality-hours and zero
   power-balance violations in every final-test condition.

The +0.01 reward threshold and 0.002 ratio margins are pragmatic, predeclared study choices.
They are not calibrated operational or utility standards. Passing these criteria is not a
claim of universal RL superiority, improvement on every service metric, or superiority to
a full-horizon optimum. A failure, tie, tradeoff or inconclusive interval must be retained
and described directly. A favorable result against raw PPO alone does not establish the
value of learned preferences over constrained allocation.

## Paired uncertainty and reproducibility

Use 10,000 deterministic seed-stratified paired bootstrap resamples with seed 20261023.
Resample base-draw clusters with replacement within each final-test seed; retain all three
shortage variants and every controller together. Bootstrap controller differences with
the subtraction order printed explicitly. Do not resample localities or controller runs
independently. Equal-size cells and episodes receive equal weight.

Report percentile 95% intervals at the 2.5th and 97.5th percentiles. These are pointwise,
unadjusted intervals; exploratory pairwise results do not support a family-wide significance
claim. They quantify scenario variability conditional on this frozen actor and simulator,
not variability across independent PPO training runs. Shortage-specific means are explicit
guardrails, not separate interval claims.

Simulation, inference, candidate scoring, metrics and bootstrap remain on CUDA. CuPy
statistics use the existing DLPack boundary. Small validation flags may synchronize with
the host; scientific outcome arrays stay on the device. Numeric scientific arrays have one final bulk
host transfer per evaluation phase for portable evidence assembly and small reconstruction
checks. Runtime details, source/model/protocol hashes, per-condition inputs or their hashes,
per-episode outcomes, validation selection and paired inference are retained in separate
study artifacts. Timing includes the complete controller call, including any candidate
filter and PPO inference, with loading and offline costs identified separately.

This is a synthetic comparison of one frozen policy and declared allocation rules. It
cannot establish the benefit of PPO fine-tuning over imitation, and a negative result for
this filter does not rule out a separately trained scheduler or learned future-value model.

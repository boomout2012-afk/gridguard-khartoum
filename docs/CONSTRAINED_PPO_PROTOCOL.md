# Training PPO with critical-service constraints

23 October 2026

This protocol is frozen by its SHA-256 in the study configuration before study training
or inspection of validation and final-test outcomes.

This experiment asks whether a newly trained, constrained PPO policy improves the existing
Fairness Rule on fresh matched GridGuard Khartoum scenarios. It tests the existing
seven-locality, 24-hour synthetic decision-support model. No new physical dynamics,
forecast information, reward weights or real-grid claims are introduced to favor learning.
Earlier model files, release evidence and allocation-study artifacts remain unchanged.

## Question and attribution

The primary question is whether the complete procedure—Fairness Rule imitation followed
by constrained PPO training and validation-only checkpoint selection—improves daily
canonical reward while protecting critical service, utilization and Jain fairness.
Five independent training seeds are evaluated; the best final-test seed is never selected.

Each run retains its behavior-cloning-only actor before reinforcement learning. Comparing
PPO with that actor separates improvement from PPO fine-tuning from performance obtained
by imitating the Fairness Rule. A favorable comparison with Constrained Optimization alone
does not establish an advantage over the stronger Fairness Rule. A selected checkpoint at
iteration zero is a behavior-cloning result, not evidence of improvement from PPO updates.

## Shared allocation constraints and information

All new methods use the same 128 candidates. Each locality receives Critical Only or
Full Service. For total critical demand C, supply S, and locality noncritical demand b_i:

    x_i in {0, 1}
    C + sum_i x_i b_i <= S

Feasibility uses the existing direct floating-point comparison without additional MW
slack. If no feasible candidate exists, the shared all-Critical-Only fallback applies and
its use is counted. The default scenario generator is expected to avoid this case; the
constraint is not a guarantee of zero interruptions for arbitrary insufficient supply.

The PPO categorical distribution masks infeasible candidates during behavior cloning,
rollout sampling, likelihood recomputation, validation and final inference. Stored rollout
masks are reused during PPO updates. Every sampled action must have positive support in
the distribution used for its old log probability. Validation and final evaluation use
deterministic highest-logit feasible selection, with the lowest candidate ID breaking ties.

All compared constrained methods have current locality demand, critical demand, supply,
cumulative demand, cumulative served energy, time and previous service tiers. The learned
model receives 46 causal features: the existing 32 features plus seven cumulative-demand
and seven cumulative-served-energy features with fixed normalization. The feature order
and normalization are recorded in the model manifest. No future scenario samples are
available to any method. Exact same scenario inputs are used for every comparison.

The canonical per-step reward and existing public metrics remain unchanged. PPO maximizes
the undiscounted sum over 24 steps. This is a changed learning and action-selection
procedure, not a claim of a new physical simulator or a different definition of fairness.

## Methods

| Method | Role |
| --- | --- |
| Fairness Rule | Primary comparator; fixed epsilon 0.005 selected by the previous allocation study, with no new sweep |
| Constrained Optimization | Immediate-reward comparator over the same critical-feasible candidates |
| Constrained PPO, seeds 1729, 2718, 3141, 5772, 8111 | Five separately initialized and trained actors, each with its own validation-selected checkpoint |
| Behavior Cloning, corresponding five seeds | Each actor immediately after Fairness Rule imitation and before PPO updates |

The prior frozen PPO may be reported as a clearly labeled reference. It has different
constraints and input features and does not isolate the value of this learning procedure.

## Training budget

The planned fixed budget per run is 600 iterations of 256 complete 24-step episodes:
3,686,400 environment transitions per run and 18,432,000 across five runs. This excludes
behavior-cloning data collection and evaluation. Training outcomes do not determine an
extension of the budget or a hyperparameter sweep.

Actor and critic are separate two-hidden-layer networks with 256 tanh units in each layer.
The actor has 128 logits; the critic has one scalar output. The planned PPO settings are:

| Setting | Value |
| --- | --- |
| Discount gamma | 1.0 |
| GAE lambda | 0.95 |
| PPO clip | 0.2 |
| Update epochs per rollout | 4 |
| Minibatch size | 1,024 |
| Critic-loss coefficient | 0.5 multiplying half the mean squared value error |
| Initial Adam learning rate | 0.0003, linearly decayed to zero |
| Entropy coefficient | Linear decay from 0.01 to 0.001 |
| Maximum gradient norm | 0.5 |

True episode termination uses zero terminal value; transitions are not linked across
episodes. Advantages are normalized over the complete rollout before minibatch updates.
For zero-based iteration k, the learning-rate and entropy schedules use progress k/599.
The extra episode in the 85/85/86 severity allocation rotates with k modulo three, giving
equal total severity counts over all 600 iterations. Numerical settings, initialization
and random-generator seeds are also recorded in the frozen configuration.

Behavior cloning uses 455 complete Fairness Rule episodes per shortage level: 1,365
episodes and 32,760 supervised states per run. Train for ten epochs with minibatches of
1,024, including the last partial minibatch, and Adam at learning rate 0.0003. The loss is
masked action cross-entropy plus 0.25 times mean squared critic error against undiscounted
teacher return-to-go. The critic is fitted only using training data. A fresh PPO optimizer
state starts after behavior cloning; no final-test data participate in any update.

## Scenario split

| Split | Seeds | Base draws per seed | Base draws | Conditions |
| --- | --- | ---: | ---: | ---: |
| Validation | 112101 through 112105 | 32 | 160 | 480 |
| Final test | 112201 through 112210 | 32 | 320 | 960 |

Each independent base draw has coupled Mild, Moderate and Severe variants. Keep these
variants together for matching and uncertainty estimates. These seeds are distinct from
the previously inspected proposal-extension and allocation-study scenarios. Final-test
inputs and outcomes remain unused until every training run and checkpoint selection is
frozen. Training and demonstration scenario ranges are separate from both held-out sets
and from each other. For zero-based run index j and iteration k, training scenario seeds
are 2,000,000 + 10,000j + k. Behavior-cloning scenarios use seed 3,000,000 + 10,000j,
shared across the three coupled shortage variants. Network initialization uses
`torch.manual_seed(training_seed)`. Separate CUDA generators use training_seed + 1 for
policy sampling, training_seed + 2 for PPO minibatch shuffling, and training_seed + 3 for
behavior-cloning minibatch shuffling. The training record retains these derivations.

All five runs use the same validation scenarios and final scenarios. The run seed controls
initialization, demonstration sampling, policy sampling and minibatch order. Training
scenario streams are independent across runs. Each severity receives equal weight in
selection and final aggregate results.

## Checkpoint selection

Evaluate iteration zero and iterations 50, 100, ..., 600 on validation only, for 13 candidate
checkpoints per run. Eligibility requires zero critical-interruption locality-hours and
zero power-balance violations in every condition, mean daily reward at least that of
Fairness Rule both overall and within each shortage level, and mean Jain fairness and
utilization each no more than 0.002 below Fairness Rule both overall and within each shortage level.
The margin is a ratio difference of 0.002, or 0.2 percentage points.

Within each run, select the eligible checkpoint with the highest overall mean daily
canonical reward. Break exact reward ties with the earliest iteration. If none is eligible,
retain the highest-reward checkpoint under the same tie rule and explicitly record an
ineligible fallback. Fallback does not excuse failure of the service requirements or
support a positive primary claim. Report all checkpoint validation summaries and every
run's selection status.

Freeze the selected model files, behavior-cloning actors, per-run selection records,
training configuration, source hashes and this protocol before final evaluation. The
deployment actor is selected by highest overall validation reward among eligible per-run
choices, breaking exact ties with the lowest training seed. If none is eligible, record
the highest-reward run as an ineligible fallback under the same tie rule. This result is
secondary and cannot replace the five-run learning-procedure result.

## Outcomes and decision rule

Report each run and the equally weighted five-run mean, plus both deterministic comparators.
Retain canonical reward, critical-service ratio, critical-interruption locality-hours,
Jain fairness, utilization, switching, oversubscription, power-balance violations,
zero-service blackout durations, final minimum locality service and fallback counts.
Include all shortage-specific means, unfavorable findings, total training elapsed time
and total final-evaluation elapsed time. This study does not measure per-decision latency;
earlier latency measurements of different actors do not establish this policy's speed.

The primary claim is **trained constrained PPO improves reward over Fairness Rule with
service guardrails**. Every gate must pass:

1. The lower endpoint of the two-sided 95% interval for the five-run mean daily reward
   difference, PPO minus Fairness Rule, is strictly greater than +0.01.
2. The lower endpoints for both Jain-fairness and utilization differences are at least
   -0.002.
3. Within each shortage level, mean reward difference is nonnegative, and mean fairness
   and utilization differences are each at least -0.002.
4. Every selected PPO actor and both constrained deterministic comparators have zero
   critical interruptions and zero power-balance violations in every final condition.
5. Every training run has an eligible validation-selected checkpoint.

These margins are pragmatic research choices, not calibrated operational standards.
The same reward/service gates are also reported against paired behavior-cloning-only
actors. Improvement against Fairness Rule without improvement over behavior cloning
does not establish that PPO fine-tuning was responsible. No improvement, a tradeoff,
mixed training-seed performance or an inconclusive interval must be stated directly.
The experiment cannot demonstrate superiority over a full-horizon optimum.

## Uncertainty and reproducibility

Use 10,000 paired hierarchical bootstrap resamples on CUDA/CuPy. For each replicate,
resample the five training runs with replacement, and resample base-draw clusters with
replacement within each of the ten final-test scenario seeds. The sampled scenario
clusters must be shared across sampled runs, methods and paired behavior-cloning actors;
all three shortage variants stay together. Compute equal-run, equal-condition mean
differences. Report percentile intervals at the 2.5th and 97.5th percentiles.

Also report per-run paired scenario intervals, explicitly conditional on that trained
actor. They are secondary descriptive comparisons. Five runs give limited information
about training variability; 960 conditions do not constitute 960 independent training
runs. All intervals are pointwise, unadjusted intervals, not a family-wide guarantee.
The deterministic bootstrap seed is 20261023.

Training, simulation, inference, candidate scoring, metrics and bootstrap remain on CUDA.
CuPy uses the established DLPack boundary. Host-side checks and artifact assembly may
read small validation summaries and the final evidence transfer. Record GPU/runtime
details, time, transition counts, configuration and protocol hashes, input matching hashes,
model hashes, per-condition outcomes and independent interval reconstruction. Model
weights and compact provenance may be retained; raw optimizer states and training caches
are not committed. Earlier published artifacts are not overwritten.

## Research basis

- Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
  introduces the clipped policy update used here.
- Huang and Ontanon, [A Closer Look at Invalid Action Masking in Policy Gradient Algorithms](https://arxiv.org/abs/2006.14171)
  studies policy-gradient training with state-dependent invalid-action masks.
- Agarwal et al., [Deep Reinforcement Learning at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264)
  motivates reporting training-run uncertainty instead of relying on a best-run point estimate.

These sources motivate the methods and evaluation discipline. They do not predict a win
for PPO in this simulator.

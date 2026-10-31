# Committee baseline audit: post-hoc matched follow-up

31 October 2026. This document is frozen before generating the follow-up outcomes.
The earlier five-run final-test results are already known. Reusing those conditions
makes this an explicitly **post-hoc explanatory audit**, not a new confirmatory test
or an untouched holdout. No policy is retrained, selected again or tuned. The earlier
protocol, models, evidence and primary decision rules remain unchanged.

## Fixed comparison

Use the original 960 final conditions: scenario seeds 122201 through 122210, 32 base
draws per seed, and coupled Mild, Moderate and Severe variants. Verify the regenerated
input hashes against the published focused study before accepting an audit. Evaluate
these nine methods on identical inputs:

1. Original Fixed Rotation, unchanged (three Full Service localities every three hours).
2. Critical-aware Fixed Rotation, defined below.
3. Constrained Optimization, unchanged current-hour exhaustive reward maximization.
4. Existing Fairness Rule, with reward tolerance 0.005, the frozen demonstration-teacher
   setting. Do not select a tolerance from the follow-up results.
5. All five previously selected PPO actors, training seeds 1729, 2718, 3141, 5772 and
   8111, with exactly the model hashes and checkpoints in the frozen training record.

There are 8,640 nominal trajectories. Summaries of PPO give equal weight to each of the
five runs; the validation-selected deployed actor does not replace this aggregate.

## Critical-aware Fixed Rotation

Use the same 128 Critical Only/Full Service assignments and the exact physical
feasibility condition used by constrained PPO and Constrained Optimization. Reserve
all critical demand first. Begin discretionary priority at locality
`(floor(hour / 3) * 3) mod 7`, matching the original rotation's block cadence. Visit
all seven localities cyclically from this start; grant Full Service when that
locality's complete noncritical demand fits the remaining supply, otherwise keep
Critical Only and continue to the next locality. This is a deterministic greedy
rotation, not reward, fairness, value or forecast optimization. There is no cap of
three Full Service localities: the original cap would confound available-capacity use.
An equivalent implementation selects the lexicographically greatest feasible binary
Full Service vector in that fixed priority order. Its unique binary rank is independent
of candidate reward, cumulative service and PPO outputs.

When total critical demand exceeds supply there is no feasible critical-protecting
assignment. The controller explicitly selects all Critical Only, and the unchanged
physical dispatcher proportionally scales delivery. That fallback is never labeled
feasible and does not guarantee uninterrupted critical service.

The new baseline controls for the explicit critical floor and access to all 128 actions.
It does not isolate PPO updates from imitation, architecture, validation selection or
policy optimization. Fairness Rule is the imitation teacher and contains a bounded
one-step reward tradeoff; it is not an independent learned policy.

## Infeasible-critical-supply stress audit

Separately reuse the 320 base draws underlying the Moderate variant, with demand
unchanged and supply set hourly to 0.75 or 0.25 times total critical demand. These are
two prespecified stress families, each with 320 conditions and nine methods: 5,760
additional trajectories. They intentionally violate the normal critical-floor
feasibility assumption in every hour. They are not additional normal generator
severities and must not be pooled into the nominal comparison.

Report physically infeasible hours, selected all-Critical-Only fallback hours for each
method, critical interruption and complete-outage locality-hours, unmet critical MWh,
and power-balance violations. Expect the constrained controllers to use proportional
fallback; test that partial critical deprivation is counted even when no locality is
completely without power. Report every stress result, including adverse outcomes.

## Outcomes and uncertainty

Report all existing canonical episode/duration metrics, every severity and each PPO
run. Emphasize served MWh, critical interruption locality-hours, Jain fairness,
utilization, daily reward and switching. Add unmet critical MWh and a five-component
reconstruction of the locked daily reward (critical service, utilization, fairness,
switching penalty, oversubscription penalty). Its components must sum to the actual
reward within declared float32 accumulation tolerance; do not convert the dimensionless
reward difference into MWh. Report observed matched served-MWh differences directly.

For PPO minus each of the four deterministic baselines, use the existing CUDA/CuPy
paired hierarchical bootstrap: 10,000 resamples, seed 20261031, resampling five training
runs and base draws within ten seed strata, sharing draws across methods/runs and
keeping coupled severities together. Report descriptive pointwise 95% intervals and
per-run conditional intervals. Severity intervals are also descriptive. No new binary
"PPO wins" claim or replacement primary acceptance gate is introduced. Report the
mechanical baseline change (Critical-aware Rotation minus original Rotation) with
paired scenario intervals; it has no training-run uncertainty. Report component and
MWh changes alongside the previously observed reward difference, including tradeoffs.

Stress outcomes are descriptive with no additional significance tests. Record all
per-condition metrics, action and input hashes, safety counts, model/source identities,
the protocol-freeze hash and elapsed evaluation time. One packed scientific transfer
to the host follows CUDA simulation, metrics and bootstrap computation. Assembly and
hashing of the portable artifact occur afterward. Report evaluation time only; do not
infer production decision latency from this batched experiment.

## Verification and limits

Before interpreting new outcomes: test greedy priority independently against a small
enumerated feasible set; test reward-independence, cadence, infeasible fallback and
partial-critical accounting; preserve all existing frozen hashes. Reproduce original
Fixed Rotation, Constrained Optimization and five PPO per-condition metrics/actions
against the earlier focused evidence. Verify artifact hashes and reconstruction of
means and paired uncertainty on CUDA. Bind the final source files before executing
the audit, and reject changes during execution.

This remains synthetic seven-locality, 24-hour decision support without power-flow,
network protection, real operational costs or field validation. A fairer baseline can
explain mechanical benefits and expose residual tradeoffs; it cannot establish that
RL generally outperforms optimization or that a dimensionless reward advantage is an
operational or economic benefit.

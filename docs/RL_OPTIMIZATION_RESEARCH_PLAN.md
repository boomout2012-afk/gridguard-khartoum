# Testing where learning helps GridGuard Khartoum

22 October 2026

Implementation update, 23 October 2026: the constrained optimizer, deterministic Fairness Rule,
Feasible PPO and frozen-PPO preference hybrid are implemented and evaluated. On 960 fresh matched
conditions the hybrid improved reward over constrained greedy allocation, but the Fairness Rule
achieved higher overall reward, fairness and utilization than the hybrid, with more switching.
All four constrained methods had zero critical interruptions. See the
[completed study](ALLOCATION_STUDY_RESULTS.md) and [verification record](allocation-study-validation.json).
The learned continuation-value and rolling-planning experiments below remain future work.

Optimization is implementable at the current seven-locality scale. PPO has not demonstrated an overall advantage over it. The most useful next experiment separates guaranteed critical service from scheduling the remaining electricity fairly over the day, then tests whether learning improves that scheduling.

These experiments extend the comparison within the existing synthetic, decision-support setting. They retain Fixed Rotation, the frozen PPO actor, and the existing Optimization results. They do not replace the accepted evidence or establish behavior on a real electrical network.

## What the current comparison establishes

In the [960-condition study](PROPOSAL_RESULTS.md), PPO has higher Jain service fairness and slightly higher accumulated canonical reward than the current-hour Optimization controller. Optimization has fewer critical interruptions and higher supply utilization. This is a tradeoff, not an overall PPO win. The optimizer searches every current-hour action; it does not optimize the full day.

The frozen PPO actor was initialized by imitating an optimizer. Its [training provenance](../data/model/training-provenance.yaml) contains no cloning-only ablation, so the existing comparison cannot isolate the benefit of PPO fine-tuning from imitation.

A [local decision-latency diagnostic](../tmp/rl-opportunity/decision-latency.json) measured median calls of 0.196 ms for PPO and 5.022 ms for Optimization on the presentation GPU. Both are practical for hourly decisions. These are current implementation timings from one session, excluding loading, training, simulation and UI. They do not establish a fundamental speed advantage: the optimizer can be improved, and PPO's current 2,187-output action head also grows exponentially with locality count.

An exact optimum cannot be beaten on the identical objective, constraints, information and horizon. Learning can potentially outperform a greedy or time-limited planner, approximate a costly solution cheaply, or improve expected outcomes under uncertainty. Each possibility requires its own comparison.

## Separate critical service from discretionary service

Let C be total critical demand, S available supply, and b_i the noncritical demand of locality i. Reserve C first. Choose which localities receive Full Service using:

    x_i in {0, 1}
    sum_i x_i b_i <= S - C

Every locality receives Critical Only or Full Service. Seven binary choices require checking only 128 combinations. Existing dispatch physics remain unchanged: restricting total requests to supply prevents proportional clipping of critical loads.

This is a new constrained objective. Apply the same critical-service constraint to every new competing scheduler and retain the original soft-reward experiment separately.

Under the default generator, serving all critical demand is always feasible. Define A_h and B_h as the default critical and noncritical sector shares multiplied by their normalized hourly profiles. With default independent demand noise of at most 5%, the critical fraction is bounded by:

    C_h / D_h <= 1.05 A_h / (1.05 A_h + 0.95 B_h)

Supply is at least 0.95 f_h D_h, where f_h is the shortage-dependent supply fraction. Evaluating this conservative bound across all 24 hours gives minimum supply-to-critical-demand ratios of 2.064 for Mild, 1.677 for Moderate and 1.242 for Severe. The 95%-of-demand supply cap never binds in these default cases. This bound applies to the default shares, profiles and noise; it does not apply to arbitrary overrides or actual grid conditions.

An exploratory CUDA run of the constrained rule produced zero critical interruptions and 100% critical service in all 48 conditions: 16 matched base trajectories across three shortage levels, seed 92021. Mean utilization was approximately 97.45% and Jain fairness 94.60%. These are diagnostic outcomes, not a matched comparison against the existing 960-condition results. The [script](../tmp/rl-structure-diagnostic/critical_first_diagnostic.py) and [recorded output](../tmp/rl-structure-diagnostic/results.json) preserve the calculation; the script was saved after its original stdin execution.

## Where learning could add value

| Decision | First method to test | Possible learning contribution |
| --- | --- | --- |
| Protect critical demand now | Exact feasible allocation | None needed when the feasible set is cheaply known |
| Share remaining power fairly | Rule favoring underserved localities | Learn the later consequences of serving one locality now |
| Avoid unnecessary tier changes | Explicit switching cost and lookahead | Learn when a switch is worthwhile over the remaining day |
| Respond to uncertain future demand and supply | Shared causal forecasts and rolling planning | Estimate future consequences without a long online search |

The current simulator has no battery, carried-over electricity, demand backlog or action-dependent future supply. The action-dependent future state consists of accumulated service and previous tiers. A locality's service deficit is an accounting measure, not electricity that can be delivered retroactively.

Current PPO can reconstruct current locality demand, critical demand and supply from its observation. The meaningful information difference is that it receives cumulative service ratios while Optimization also receives exact cumulative demand and delivered-energy magnitudes. New comparisons should expose equivalent causal information, including identical forecasts, to both methods.

## The creative candidate

Use exact allocation for immediate outcomes and a learned value for their later consequences:

    choose the feasible action maximizing
    immediate reward + gamma * learned value of the resulting state

For example, two allocations may deliver similar energy now, but one repeatedly neglects a locality before its evening demand peak. A learned value could recognize that consequence without searching every possible daily schedule. A debt rule or forecast planner may recognize it just as well; that is the test.

Combining short planning with learned future value has research precedent in [Deep Value Model Predictive Control](https://proceedings.mlr.press/v100/hoeller20a.html) and [Temporal Difference Learning for Model Predictive Control](https://proceedings.mlr.press/v162/hansen22a.html). Their results motivate an experiment here; they do not establish a GridGuard performance guarantee.

This hybrid does not inherently require PPO. To test PPO specifically, a PPO supervisor could choose locality priorities while the exact allocator enforces feasibility. Compare that supervisor with fixed priorities and a deterministic service-deficit rule using the identical allocator. A successful hybrid would establish the value of that combination, not that PPO alone beats optimization.

## Experiments in order

1. **Constrained greedy allocation.** Formalize the 128-combination diagnostic, test its feasibility boundary, and provide an explicit fallback for inputs where critical demand exceeds supply.
2. **Service-deficit scheduling.** Add a deterministic rule that favors low cumulative service ratios. This is the cheapest test of whether learning is necessary.
3. **Rolling planning.** Test two- and four-hour horizons with shared causal forecasts. Execute one action, observe actual conditions, and plan again. Label limited or beam search as approximate. Even 128 choices per hour make naive four-hour enumeration large; use pruning or bounded search and measure the actual cost.
4. **Learning-assisted scheduling.** Compare zero future value, fixed service-deficit value, learned future value, and the rolling planner. If using PPO, include imitation-only and PPO alternatives to isolate its contribution.

Validate the simpler baselines before investing in training. Keep the existing seven localities and physics. Adding storage, physical network constraints or a larger region would require separately justified scope; those features should not be introduced merely to produce an RL win.

## Decide what counts as success before the final test

Use fresh, disjoint training, validation and final-test trajectories. Previously inspected scenarios are development material. Keep coupled shortage variants together. Select thresholds, forecast models, tuning budgets and checkpoints on validation, then freeze the protocol before opening final outcomes.

Require zero avoidable critical interruptions under the default feasibility constraint. Compare daily fairness, utilization, switching, direct interruption duration and total reward together. Declare practical improvement and allowable tradeoff margins before testing. Use at least five independent learning runs for an initial study and report training-run variability separately from paired scenario uncertainty. This distinction follows the concerns in [Agarwal et al. on reliable RL evaluation](https://arxiv.org/abs/2108.13264).

Report full online decision cost, including any feasibility or optimization layer, plus offline training and teacher costs. Compare quality across reasonable planning-time budgets rather than choosing a deadline that disadvantages the optimizer.

The possible conclusions are:

- **Learning improves service:** a repeatable, practically meaningful daily improvement survives the common critical-service constraint and the strongest practical planning baseline.
- **Learning improves efficiency:** outcomes meet prespecified equivalence margins and online savings matter for the actual deployment deadline.
- **No learning advantage is demonstrated:** structured optimization or a simple scheduling rule meets the requirements as well or better. Prefer it operationally and retain PPO as an evaluated research alternative.
- **Evidence remains inconclusive:** uncertainty is too wide to distinguish a meaningful benefit or equivalence.

The first implementation priority is the constrained allocator and service-deficit baseline. The strongest potential learning contribution is anticipating the consequences of daily scheduling; current evidence does not show that PPO is necessary for this task.

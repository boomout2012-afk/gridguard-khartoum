# GridGuard Khartoum final report

**Full title:** GridGuard Khartoum: PPO-Assisted Load Shedding and Energy Management for
Developing Countries

**Document type:** Graduation Project Final Report

**Authors:**

- Murtada Asim Ibrahim Bakhit (Index 18-317)
- Suliman Elawad Suliman Babiker (Index 18-310)
- Abdalhay Abass Mobarak Abass (Index 15-306)

**Date:** August 2026

## Abstract

GridGuard Khartoum is an offline, GPU-first research simulation for proposal-aligned load
shedding across seven Khartoum State localities. It compares a frozen PPO controller with Fixed
Rotation on identical synthetic 24-hour shortage scenarios. The release separates an accepted
reproduction whose seeds participated in model selection from a documented-seed-disjoint holdout on
seeds 81001-81010: 320 independent base draws, 960 matched scenario conditions, and 1,920
controller evaluations. Critical-outage locality-hours, fairness, and utilization are the three
primary outcomes; the accepted reward is a Training-objective score (diagnostic), and power
balance is reported separately as safety. A strict `data/model/training-provenance.yaml` exposes
the historical network, behavior-cloning initialization, optimizer settings, seeds, selection,
source hashes, and evidence category of each recovered field. The selected seed-43 run used 600
PPO iterations, 153,600 episodes, and 3,686,400 transitions; the three-seed candidate experiment
totaled 1,800 iterations, 460,800 episodes, and 11,059,200 transitions. Hardware and convergence
are explicitly selected-run facts, and historical promotion gates are not independent headline
evidence. The robustness
supplement also covers 25 parameterizations and compares the display forecast with one-hour
persistence. All traces are synthetic, the operating domain is shortage-only, and the work is
decision-support research rather than a deployed controller or field-performance claim.

## Introduction

Planned electricity shortages create an allocation problem: limited supply must be distributed
while essential services remain protected and repeated exclusion of the same communities is
avoided. The accepted proposal asks whether artificial intelligence can support that decision in
a simplified model of Khartoum State. GridGuard Khartoum turns that question into a reproducible
software experiment and an offline presentation tool. Its controller comparison is deliberately
narrow: one frozen PPO policy is evaluated against one transparent but supply-blind Fixed Rotation
schedule on identical synthetic scenarios.

The project has two deliverables that must not be confused. The scientific package produces and
verifies matched controller evidence. The desktop application explains that evidence and can run
a separate live scenario for demonstration. Live values never replace the verified documented-seed-disjoint
robustness supplement used by the Results page. The
system also shows real locality boundaries and the White Nile, Blue Nile, and Nile for geographic
orientation, but those layers contribute no demand, topology, controller input, or metric value.

## Related Work

The detailed Related Work chapter positions the project against Sudan's energy-access context [9],
long-duration rationing policy and fair load shedding [10, 11], critical-load restoration [12],
reinforcement learning for demand response [13], a survey of reinforcement learning for
electric-power control [14], and a review of electricity load forecasting [15]. That literature supports the
evaluation questions and exposes limitations;
it does not calibrate the illustrative generator or establish novelty or field readiness.

## Background

Fixed Rotation is a useful academic baseline because its schedule is deterministic and easy to
audit, yet it does not observe current shortage conditions, critical demand, or accumulated
service fairness. PPO represents the contrasting learned strategy: it maps a causal observation
of current supply, demand, previous tiers, and service history to one joint action across seven
localities. The study therefore asks whether the learned mapping improves the proposal outcomes
inside a controlled simulator, not whether it can operate Sudan's electricity network.

The simulation models 24 one-hour steps, five demand sectors per locality, and three service
tiers. Hospital, water, and bakery demand is treated as critical. Mild, Moderate, and Severe
shortages are deterministic transformations of the same severity-neutral stochastic base draws.
This matched construction isolates controller differences and motivates the paired analysis.
Administrative geometry comes from the pinned OCHA/HDX Sudan COD-AB revision v03 [7], while river
orientation comes from pinned Natural Earth 1:10m data [8]. Both are packaged offline and
verified independently of scientific evidence.

## Discussion

The documented-seed-disjoint holdout is the reviewer-facing result because its ten seeds are
disjoint from every documented training, validation, and model-selection seed in
`data/model/training-provenance.yaml`. This does not establish preregistration or independent
custody and cannot rule out undocumented activity. The accepted
96-draw artifact remains valuable for frozen parity, but it is labeled accepted reproduction
evidence because its seeds participated in model selection. Both layers use matched controller
pairs and sign-neutral verification; no favorable interval is a release condition.

The three primary outcomes avoid double-counting the optimized reward: critical-outage
locality-hours captures outage burden, Jain fairness captures distribution between localities,
and utilization captures use of available energy. The reward is a Training-objective score
(diagnostic), critical-service ratio and served energy are supporting, curtailed energy is
complementary, and power-balance violation is separate safety. This framing does not erase the
accepted numbers; it limits what each number is allowed to prove.

Fixed Rotation is transparent but supply-blind, is limited to Off and Full Service, and cannot select
Critical Only. Its rising utilization as shortage worsens is a denominator effect, not adaptive
behavior. Mean unused supply in the accepted reproduction is 2,158.7 MWh for Fixed Rotation versus
150.2 MWh for PPO, so part of the utilization contrast reflects baseline design. A supply-aware
critical-preserving heuristic is necessary future work before making broad claims against
rule-based control. Performance relative to the global optimum is unknown. The proposal's
optimization method is intentionally outside the accepted scope: Adam optimized the PPO parameters
during historical training and CUDA accelerates inference and evaluation, but no online mathematical
dispatch optimizer is implemented or compared. The confirmatory comparison was deliberately limited
to a learned controller against a rule-based schedule, and an optimization-based comparator is
future work.

These findings remain about allocation logic in this simulator. The single global supply pool
omits feeder capacity, voltage, frequency, protection, losses, restoration procedures, operator
constraints, and sub-hour dynamics. When requests exceed supply, proportional dispatch scales
critical and non-critical demand together. The demand profiles, shortage construction, reward
weights, and outage threshold are illustrative assumptions rather than utility measurements.
Matched scenarios isolate controllers but cannot repair a misspecified generator.

The sensitivity study perturbs 12 allocation components in both directions, giving 24
one-factor-at-a-time variants and 25 parameterizations including the base. It measures local
dependence on those assumptions, not all plausible parameter interactions. Its intervals are
descriptive, pointwise 95%, and unadjusted for multiple comparisons; no family-wide significance
claim follows. Forecast MAE and RMSE
compare the three-hour trailing mean with one-hour persistence on synthetic targets; they do not
establish real-world forecast accuracy. The evaluated operating domain is shortage-only and says
nothing about surplus, recovery, normal operation, or restoration transients. The PPO actor also
enumerates `3^N` joint actions (`3^7 = 2,187` here), which limits scaling beyond seven localities.
Factored, autoregressive, or shared per-locality policies with explicit constraint handling are the
standard routes to larger systems and remain future work.

The presentation design reinforces that boundary. The real administrative map is more legible and
recognizable than the former abstract drawing, but visual realism must not be mistaken for model
realism. OCHA/HDX boundaries and Natural Earth rivers are presentation context only and are not
electrical-network topology. Likewise, the live aggregate table helps the committee compare the
two controllers on one selected run, while the Results page remains bound to the separately
verified frozen evidence.

## Conclusion

GridGuard Khartoum fulfills the accepted proposal as a focused, auditable decision-support
simulation and academic presentation package. It demonstrates a complete chain from deterministic
synthetic scenario generation, through matched PPO and Fixed Rotation execution, to canonical
primary and safety metrics, paired uncertainty, sensitivity, forecast comparison, provenance
checks, and an offline native interface. The conclusion is intentionally bounded: it establishes
comparative behavior for one frozen actor and one weak baseline in the reviewed simulator, not
operational superiority on a real grid, over all rule-based methods, or relative to an optimum.

The release also improves how that conclusion can be inspected. Sign-neutral evidence validation
tests schema, arithmetic, physical identities, pairing, source bindings, and frozen parity without
assuming a favorable result. The strict training YAML makes each recovered parameter, provenance
category, scope, historical unknown, and selection caveat discoverable. The live view exposes matched episode metrics without altering the
benchmark, and the source-pinned administrative map gives geographic orientation without entering
the scientific pipeline. These separations make the project easier to explain and harder to
misrepresent.

The training record also makes the attribution boundary explicit. No behavior-cloning-only
ablation was run, so this comparison cannot isolate the incremental effect of PPO fine-tuning.

Future work should begin with representative operational data and an engineering network model,
followed by independent validation of constraints, reward choices, uncertainty, robustness, and
human operating procedures. Only after those steps could field-facing trials be considered. Until
then, GridGuard Khartoum remains what its evidence supports: a reproducible software demonstration
of how PPO can assist simplified load-shedding decisions under synthetic shortage conditions.

## Assembly

The generated report uses the Introduction, Related Work, and Background above, followed by the
committed Methodology, Implementation, and Results chapters, then the Discussion, Conclusion, and
References. Build with `uv run --extra docs python tools/build_report.py`. The output is
`output/pdf/gridguard-khartoum-final-report.pdf`.

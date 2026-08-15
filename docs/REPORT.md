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
Rotation on identical synthetic 24-hour shortage scenarios. The release preserves the accepted
aggregate comparison and regenerates auditable paired episode evidence from the frozen PPO
checkpoint and reviewed simulator. Across 96 independent base draws replayed under three
shortage levels, PPO improves reward, critical-service protection, utilization, fairness, served
energy, curtailed energy, and switching rate; both controllers have zero recorded power-balance
violations under the locked tolerance. A native offline desktop application provides a verified
OCHA/HDX administrative-boundary map with Natural Earth Nile context and deterministic live
replay. Geography is presentation context only, results remain synthetic, and the map is not
electrical-network topology. The work is decision-support research, not a deployed controller,
and makes no field-performance claim.

## Introduction

Planned electricity shortages create an allocation problem: limited supply must be distributed
while essential services remain protected and repeated exclusion of the same communities is
avoided. The accepted proposal asks whether artificial intelligence can support that decision in
a simplified model of Khartoum State. GridGuard Khartoum turns that question into a reproducible
software experiment and an offline presentation tool. Its confirmatory comparison is deliberately
narrow: one frozen PPO policy is evaluated against one transparent Fixed Rotation schedule on
identical synthetic scenarios.

The project has two deliverables that must not be confused. The scientific package produces and
verifies matched controller evidence. The desktop application explains that evidence and can run
a separate live scenario for demonstration. Live values never replace the frozen benchmark. The
system also shows real locality boundaries and the White Nile, Blue Nile, and Nile for geographic
orientation, but those layers contribute no demand, topology, controller input, or metric value.

## Background

Fixed rotation is a useful academic baseline because its schedule is deterministic and easy to
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

Within the locked synthetic protocol, the result is internally consistent. PPO has the favorable
aggregate value for every primary outcome, the favorable cell mean in all nine shortage-by-seed
cells, and a paired cluster interval wholly on the favorable side of zero for each inferential
metric. The largest practical contrast is critical-service protection: the reproduced aggregate
critical-service ratio is about 99.72% for PPO and 42.63% for Fixed Rotation. Improved utilization
and served energy show that the gain is not obtained by violating the modeled supply constraint;
both controllers record zero power-balance violations under the declared 0.001 MW tolerance.

These findings should be interpreted as evidence about allocation logic in this simulator. The
single global supply pool omits feeder capacity, voltage, frequency, protection, losses,
restoration procedures, operator constraints, and sub-hour dynamics. The demand profiles,
shortage construction, reward weights, and outage threshold are research assumptions rather than
utility measurements. The cluster bootstrap quantifies resampling variability among 96 base
draws while holding the three seed strata and three shortage variants fixed; it does not describe
all future operating conditions.

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
metrics, paired uncertainty, provenance checks, and an offline native interface. Under the locked
protocol, PPO improves reward, episode-pooled critical service, mean hourly utilization, mean
hourly cumulative-locality Jain fairness, served energy, curtailed energy, and switching rate.
The conclusion is intentionally bounded: it establishes comparative behavior for one frozen actor
and one baseline in the reviewed simulator, not operational superiority on a real grid.

The release also improves how that conclusion can be inspected. Neutral evidence validation can
test schema, arithmetic, physical identities, and pairing without assuming a favorable result;
the trusted release gate separately requires locked provenance, frozen-benchmark parity, and
favorable intervals before a public claim is accepted. The live view exposes matched episode
metrics without altering the benchmark, and the source-pinned administrative map gives geographic
orientation without entering the scientific pipeline. These separations make the project easier
to explain and harder to misrepresent.

Future work should begin with representative operational data and an engineering network model,
followed by independent validation of constraints, reward choices, uncertainty, robustness, and
human operating procedures. Only after those steps could field-facing trials be considered. Until
then, GridGuard Khartoum remains what its evidence supports: a reproducible software demonstration
of how PPO can assist simplified load-shedding decisions under synthetic shortage conditions.

## Assembly

The generated report uses the Introduction and Background above, followed by the committed
Methodology, Implementation, and Results chapters, then the Discussion, Conclusion, and
References. Build with `uv run --extra docs python tools/build_report.py`. The output is
`output/pdf/gridguard-khartoum-final-report.pdf`.

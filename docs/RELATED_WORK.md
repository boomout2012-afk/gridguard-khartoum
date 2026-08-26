# Related Work

GridGuard Khartoum sits at the intersection of electricity rationing, fair allocation, critical-load
protection, reinforcement learning, and short-horizon demand forecasting. These fields contain
substantially richer physical and institutional models than the present application. This review
therefore positions the project as a bounded, offline decision-support prototype and identifies the
evidence that would be required to move beyond its Khartoum-inspired synthetic case. It does not
treat geographic labels or a schematic map as validation of an electrical model.

## Sudan energy-access context

The World Bank's 2025 appraisal-stage project document describes severe disruption to Sudan's
electricity sector, including grid damage, generation shortfalls, fuel-supply constraints, and
outages affecting households and essential services [9]. It also describes a current emphasis on
decentralized energy access for public facilities and other underserved uses. This establishes the
importance of resilient access and critical services in the national context. It does not provide
the feeder measurements, hourly locality demand, switching constraints, or operating records
needed to parameterize this simulator.

Accordingly, reference [9] is used only for problem context. It is not used to infer the project's
700 MW peak, locality demand weights, sector shares, hourly load profiles, shortage fractions, or
service priorities. Those quantities are proposal-constrained or illustrative assumptions. The
scientific path in this release is therefore not calibrated to the Khartoum grid. Calibration would
require time-aligned utility measurements, network and feeder topology, generation and import
availability, equipment limits, outage and restoration histories, critical-facility inventories,
operator rules, weather and calendar effects, and a defensible mapping from physical assets to the
seven modeled localities.

## Long-duration shortage, rationing, and fairness

Heggie et al. study rationing policy under persistent shortage, where demand routinely exceeds
supply and a policy must balance fair and efficient allocation amid operational uncertainty [10].
That framing is closer to GridGuard's shortage episodes than emergency-only under-frequency load
shedding, but their work also highlights why a rotation schedule is a policy choice rather than a
neutral benchmark. In GridGuard, Fixed Rotation does not observe available supply, requests Full
Service for three localities, and turns the other four Off. Its low utilization and critical-service
performance must therefore be interpreted partly as consequences of that rule design.

Oluwasuji et al. formulate fair electric load shedding for developing-country settings as a resource
allocation problem with heterogeneous household needs [11]. Their analysis motivates looking
beyond aggregate energy served and treating fairness as an explicit outcome. GridGuard uses Jain
fairness over cumulative locality service ratios, but this is a coarser abstraction: a locality is
not a household, individual preferences are absent, and the fixed sector shares do not establish
actual vulnerability. Sensitivity analysis can show whether conclusions change under specified
perturbations; it cannot replace locally observed demand or community-defined service priorities.

These studies also limit how the comparison should be worded. Matched scenarios isolate controller
differences under one generator, but they do not cancel misspecification shared by every scenario.
The comparison can support statements about the implemented synthetic experiment. It does not
establish that one policy would be fairer or more efficient in field operation.

## Critical loads and operational constraints

Poudel and Dubey formulate critical-load restoration using distributed energy resources, network
connectivity, radial-operation constraints, restoration timing, and resource availability [12].
Their restoration problem differs from GridGuard's daily shortage allocation, yet it demonstrates
that protecting critical services normally depends on physical topology and operational limits as
well as a priority label. GridGuard instead chooses one of three service tiers for each locality and
then applies a single proportional scale factor when requested power exceeds supply. Within a
selected locality, critical and non-critical requested load are reduced by the same factor.

The present critical-service metric is therefore evidence about the simulator's tier requests and
proportional dispatch, not about feeder-level hospital restoration. A realistic extension would
model feeder connectivity, switching feasibility, minimum service blocks, ramping and restoration
rules, critical-facility demand, and explicit priority constraints. A mathematical dispatcher or
episode-level optimum would also be needed before describing any comparator as an upper bound.

## Reinforcement learning in energy control

Vázquez-Canteli and Nagy review reinforcement learning for demand response across buildings,
storage, electric vehicles, appliances, and related flexible-demand applications [13]. Their review
notes the importance of human comfort, feedback, non-stationarity, and coordination. Glavic surveys
reinforcement learning across normal, preventive, emergency, and restorative electric-power
control contexts and across several control levels [14]. Together, these reviews support PPO as a
recognizable research approach while emphasizing that deployment depends on a faithful environment,
constraints, and evaluation beyond a training objective.

GridGuard uses a frozen PPO actor trained in a synthetic simulator and performs deterministic
offline inference. It does not learn online, command utility equipment, estimate network state, or
enforce alternating-current power-flow constraints. The joint action contains `3^7 = 2,187`
possibilities for seven localities; this direct enumeration grows as `3^N`. Factored,
autoregressive, or shared-locality policies with explicit constraint handling are possible future
directions, not capabilities of this release.

The distinction between training and evaluation is also important. The episode reward is the
training-objective score and shares components with several reported measures. It is therefore a
diagnostic of objective attainment rather than independent outcome evidence. The proposal-aligned
primary outcomes are critical-outage locality-hours, Jain fairness, and utilization, each reported
separately from the training objective.

## Demand forecasting and evaluation

Nti et al. review electricity load-forecasting methods, input variables, horizons, and error
measures [15]. Their synthesis reports common use of historical consumption and contextual inputs
and identifies MAE and RMSE among established error measures. This supports evaluating GridGuard's
causal three-hour trailing-mean projection against a simple one-hour persistence baseline on common
target hours.

That experiment is deliberately narrow. Both forecast methods operate on synthetic demand traces,
so their MAE and RMSE measure consistency with this generator rather than accuracy for Khartoum.
No result from it supports operational scheduling, general forecast superiority, or a real-world
accuracy claim. A field study would require held-out metered data, clearly defined forecast
horizons, seasonal and event coverage, missing-data rules, and comparisons with appropriate
statistical and operational baselines.

## Position of GridGuard Khartoum

GridGuard integrates established ideas into an offline Windows demonstration: a deterministic
synthetic scenario generator, Fixed Rotation and PPO controllers, matched evaluation, explicit
fairness and critical-service measures, hash-verified evidence, and a schematic locality display.
The project does not make a claim of algorithmic novelty, grid deployment, or utility calibration.
Its contribution is a traceable academic implementation that makes assumptions and evidence
boundaries inspectable.

Within those boundaries, the review literature motivates three improvements now incorporated into
the evaluation design: independent physical and service outcomes rather than reward alone,
one-factor sensitivity tests for illustrative demand parameters, and a forecast comparison with
standard error measures. It also identifies work that remains outside the approved architecture:
a supply-aware comparator, feeder-constrained optimization, an episode-level optimal benchmark,
real demand calibration, larger or factored action spaces, and recovery or surplus operation.
Consequently, favorable synthetic results may justify further investigation; they do not establish
safe or effective operation on Sudan's power system.

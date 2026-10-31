# Declaration

We declare that this project, entitled “AI-Based Smart Load Shedding and Energy
Management System for Developing Countries”, is our original work, carried out under
the supervision of {{SUPERVISOR}} at the Faculty of Mathematical Sciences and
Informatics, University of Khartoum. This work has not previously been submitted for
another degree or professional qualification, and sources consulted are acknowledged.

Murtada Asim Ibrahim Bakhit (18-317) — Signature: ____________________

Suliman Elawad Suliman Babiker (18-310) — Signature: ____________________

Abdalhay Abass Mobarak Abass (15-306) — Signature: ____________________

Date: ____________________

# Dedication

To our families, whose patience and support made this work possible.

To the people of Sudan, whose resilience in the face of energy hardship inspired this study.

To our supervisor, {{SUPERVISOR}}, for guidance and encouragement.

# Abstract

Electricity shortages create a distribution problem as well as a shortage of generation. Allocating the same supply differently can change interruption durations, protect essential services and distribute unequal burdens between communities. This study develops and evaluates GridGuard Khartoum, an AI-assisted load-shedding and energy-management system combining software simulation, history-only demand prediction, priority rules, constrained optimization, reinforcement learning and an offline application. Khartoum State provides the geographic setting; all demand and supply traces are synthetic.

The simulation represents seven localities, five sectors and 24 hourly decisions under Mild, Moderate and Severe shortages. Hospital, Water and Bakery demand receives explicit priority. The main experiment compares Fixed Rotation, Constrained Optimization and Proximal Policy Optimization (PPO) on identical fresh scenarios. PPO and Constrained Optimization share 128 Critical Only/Full Service assignments and the same supply-feasibility rule. PPO uses 46 features available at decision time and Fairness Rule demonstrations before reinforcement learning. Five independent runs each complete 3,000 iterations, followed by validation selection and evaluation on 960 fresh matched conditions. Critical interruption, service fairness, utilization and direct duration disparities are reported separately from the training score.

{{ABSTRACT_RESULTS}}

A separate forecasting experiment compares Ridge AR(3), persistence and a three-hour trailing mean on unseen synthetic targets. Ridge AR(3) has the lowest error on all four measures, but forecasts are not controller inputs. The application presents locality context, matched replay and verified results offline. The contribution is an inspectable decision-support system and outcome-specific evidence, not a real-grid controller or a general proof that learning surpasses optimization.

**Keywords:** load shedding; energy management; PPO; constrained optimization; critical services; fairness; prediction; Khartoum.

# الملخص

تتناول هذه الدراسة توزيع الكهرباء أثناء نقص الإمداد من خلال تطوير نظام «جريد جارد الخرطوم» لدعم القرار. يجمع النظام المحاكاة والتنبؤ بالطلب وقواعد أولوية الخدمات الأساسية والتحسين المقيد والتعلم المعزز، مع تطبيق مكتبي يعمل دون اتصال بالإنترنت. تمثل ولاية الخرطوم الإطار الجغرافي للدراسة، لكن بيانات الطلب والإمداد جميعها اصطناعية وليست قياسات للشبكة الفعلية.

تمثل المحاكاة سبع محليات وخمسة قطاعات وقرارات ساعية لمدة أربع وعشرين ساعة، تحت ثلاثة مستويات للنقص. تقارن الدراسة خوارزمية التحسين القريب للسياسة «PPO» بالتناوب الثابت والتحسين المقيد على الظروف نفسها. يشترك PPO والتحسين المقيد في مئة وثمانية وعشرين تخصيصا ممكنا وقيد لحماية طلب المستشفيات والمياه والمخابز عندما يسمح الإمداد. تستخدم السياسة ستا وأربعين سمة متاحة عند اتخاذ القرار، ويبدأ التدريب بتقليد قاعدة للإنصاف ثم بالتعلم المعزز. أجريت خمس عمليات تدريب مستقلة، لكل منها ثلاثة آلاف تكرار، واختيرت السياسات باستخدام مجموعة تحقق منفصلة قبل اختبار تسعمئة وستين حالة متطابقة.

بلغ متوسط PPO صفرا من ساعات انقطاع الخدمات الأساسية على مستوى المحليات، و97.02% للإنصاف و97.02% لاستغلال الإمداد. حقق النظام المقيد معيار المقارنة المعلن مقابل التناوب الثابت، لكن حماية الخدمات الأساسية تعود إلى القيد الصريح وليست دليلا منفردا على فائدة التعلم. مقارنة بالتحسين المقيد الذي يعظم مكافأة الساعة الحالية، زاد PPO متوسط الإنصاف بمقدار 2.24 نقطة مئوية والمكافأة اليومية بمقدار 0.0499، بفاصل ثقة معدل بنسبة 97.5% من 0.0406 إلى 0.0577. في المقابل انخفض استغلال الإمداد بمقدار 0.40 نقطة مئوية وزاد تبديل مستويات الخدمة، ولذلك لم يتحقق معيار التحسن الكامل المشروط بمؤشرات الخدمة. هذه النتائج لا تثبت تفوق التعلم على التحسين لجميع الأهداف أو على تخطيط أمثل لليوم الكامل.

أضاف تحليل لاحق مقارنات على حالات الاختبار نفسها. تحسن تخصيص PPO مقارنة بقاعدة تناوب تحمي الطلب الأساسي. وأمام قاعدة الإنصاف المستخدمة لتوليد أمثلة التدريب، زادت المكافأة اليومية بمقدار 0.0102، لكن انخفض استغلال الإمداد بمقدار 0.308 نقطة مئوية والطاقة المقدمة بمقدار 16.68 ميغاواط ساعة يوميا. تركزت زيادة المكافأة في النقص الحاد؛ ولا يمثل التحليل اختبارا تأكيديا مستقلا جديدا. وأكدت اختبارات الإجهاد استمرار انقطاعات جزئية للخدمات الأساسية عندما يقل الإمداد عن طلبها.

خفض نموذج الانحدار الذاتي المنتظم «Ridge AR(3)» خطأ التنبؤ بالطلب الاصطناعي مقارنة باستمرار آخر قيمة ومتوسط الساعات الثلاث السابقة. لا تدخل هذه التنبؤات في سياسة التحكم. تتمثل مساهمة المشروع في نظام قابل للفحص ونتائج محددة النطاق توضح المفاضلات، وليس في إثبات الجاهزية التشغيلية لشبكة كهربائية حقيقية.

**الكلمات المفتاحية:** فصل الأحمال؛ إدارة الطاقة؛ التعلم المعزز؛ التحسين المقيد؛ الخدمات الأساسية؛ الإنصاف؛ الخرطوم.

# 1 Introduction

## 1.1 Background and problem

When demand exceeds available electricity, some consumption must be reduced or interrupted. A schedule cannot remove the physical deficit, but it can change its consequences. Different allocations may preserve essential facilities, avoid prolonged interruptions or use supply that a rigid schedule leaves unrequested. These are distinct objectives: serving more energy does not automatically distribute it fairly, and equal blackout durations do not necessarily mean adequate service.

Sudan provides the setting for this software study of electricity allocation under shortage. Hospitals, water facilities and bakeries motivate explicit priorities. Residential and commercial demand also matter because critical protection does not settle how remaining supply should be shared. The World Bank (2025) describes electricity-sector disruption and the importance of energy access for essential facilities in Sudan. It provides context, not the hourly measurements used in this experiment.

The project asks how prediction, rules, optimization and learning can support shortage allocation. These methods answer related but different questions. Forecasting estimates later demand from observed history. Priority rules specify protected services. Optimization finds the best admissible assignment for a defined objective and horizon. Reinforcement learning seeks a policy whose sequence of decisions produces a favorable accumulated outcome. An informative comparison must identify what each method knows, can choose and is asked to improve.

GridGuard represents chronic hourly shortage rather than a sub-second voltage or frequency emergency. Actions affect later cumulative service ratios and switching history, creating a sequential allocation problem without storage. Unused electricity cannot be carried forward. Any advantage of learning must therefore appear in measured service and scheduling outcomes; it cannot be assumed from the use of a neural network.

## 1.2 Aim, objectives and research questions

The aim is to develop and evaluate an offline decision-support system for electricity allocation under shortage. Seven objectives form one workflow: study shortage with Sudan as the motivating context; model supply, demand and sectors; develop AI-assisted decision support; protect critical services; test prediction, rules, optimization and reinforcement learning where appropriate; compare with a traditional fixed schedule; and evaluate downtime, fairness and distribution efficiency.

The study addresses five research questions:

1. How can synthetic scenarios represent supply, locality demand and service priorities transparently?
2. Does a learned history-only forecast improve next-hour prediction over simple baselines?
3. How do PPO and Constrained Optimization change critical interruption, service fairness, utilization and blackout-duration inequality relative to Fixed Rotation?
4. Under matching critical constraints, does PPO improve declared outcomes over the optimizer, and what tradeoffs remain?
5. How can an offline interface explain the decisions and evidence clearly?

These questions evaluate AI without requiring a predetermined advantage over other methods. A result favoring a transparent optimizer is useful. An improvement on one outcome is also useful when its scope and costs are explicit. Direct duration measures assess blackout inequality alongside Jain service fairness; the two are not treated as interchangeable.

## 1.3 Scope and contribution

The model covers seven localities, five sectors, three shortage levels and a 24-hour horizon. Demand profiles and shortage parameters are illustrative assumptions. A shared supply pool replaces network power flow. There are no feeders, losses, voltage or frequency dynamics, storage or restoration transients. Administrative boundaries provide geographic orientation, not electrical service territories.

The contribution is an inspectable comparison framework connecting repeatable simulation,
learned prediction, explicit critical constraints, exact current-hour optimization,
trained PPO and separate service measures. It identifies when a learned objective
improves and when the service costs prevent the declared claim from passing. This is an
implemented case study, not a new PPO algorithm. The reduction in critical interruption
against an unprotected schedule is not presented as the central learning contribution.
Matched inputs support procedure comparisons, while independent runs expose training
variability and the follow-up audit tests stronger transparent alternatives.

The Fixed Rotation comparison evaluates the complete system. It cannot assign every improvement to learning because Fixed Rotation lacks Critical Only actions and a critical-feasibility mask. PPO and Constrained Optimization, by contrast, share the same feasible assignments and information available at decision time. That comparison is stronger for assessing learned scheduling, while still not establishing superiority over a full-day optimum or every optimization method. Section 5.7 adds a post-hoc matched comparison with the Fairness Rule teacher and a critical-aware rotating baseline; it does not reclassify that follow-up as part of the original confirmatory design.
# 2 Related work

{{LITERATURE_REVIEW_STATUS}}

## 2.1 Traditional scheduling and optimization

The chapter has nine connected subsections: eight literature themes followed by a synthesis positioning this study. It begins with conventional load shedding and optimization. Emergency protection and recurring rationing solve different problems. Larik et al. (2019) review under-voltage shedding, distinguishing static and dynamic, open-loop and closed-loop approaches. This informs the need to specify the control task, but transient performance measures cannot be transferred directly to a daily simulator without voltage dynamics.

Heggie et al. (2018) study rationing during long-term shortage, emphasizing repeated allocation and its efficiency consequences. This is closer to GridGuard's setting. Fixed rotation is inspectable and easy to explain, but a predetermined service block can ignore changing demand and the accumulated burden of earlier interruptions. It is a useful baseline, not the strongest possible conventional policy.

Optimization supplies a necessary stronger comparison. Sarwar et al. (2020) formulate priority-aware load shedding through mixed-integer programming for islanded distribution systems. Poudel and Dubey (2019) connect critical-load restoration to distributed resources and network restrictions. Together these studies show that explicit priorities and adaptive decisions are compatible with mathematical optimization; learning is not the only mechanism available.

The literature does not justify declaring all optimization too slow. Cost depends on the formulation, horizon, scale and implementation. GridGuard has seven locality decisions, so exhaustively scoring 128 constrained assignments is practical. This optimizer is exact for its current-hour objective. The research question is whether a learned sequence improves full-day outcomes, not whether PPO exceeds the exact maximum of the identical one-step problem.

## 2.2 Machine learning and demand forecasting

Ramchurn et al. (2012) frame the smart grid as a broad challenge for AI, connecting coordination and uncertainty across distributed decisions. Glavic (2019) distinguishes power-system operating tasks and illustrates why a learning method's relevance depends on the state, objective and restrictions. A method useful for a market, building or emergency-control task is not automatically useful for fair shortage allocation.

Forecasting is a separate AI contribution. Nti et al. (2020) review electricity-load forecasting methods and evaluation. Bedi and Toshniwal (2018) investigate empirical-mode decomposition with deep learning. Convolutional, recurrent and ensemble approaches offer further alternatives for exploiting temporal structure, but complexity alone does not establish superiority for a particular dataset.

GridGuard uses compact ridge autoregression with three hourly lags for every locality-sector pair. Its inputs, coefficients and training-only normalization can be inspected directly. Persistence and a trailing mean provide transparent comparisons. Every method must predict the same target hours; otherwise differences in warm-up or forecast horizon could be mistaken for a model improvement.

The forecasting experiment answers a prediction question. PPO receives current observations and accumulated history, not a forecast or future trace. Improved forecast accuracy therefore cannot explain a measured controller gain. Integrating predictions into planning would require another matched experiment, including forecast errors and comparable information for competing controllers.

## 2.3 Reinforcement learning for load shedding

Reinforcement learning connects observations, actions and state transitions through an accumulated reward. Huang et al. (2020) investigate learned emergency control, including under-voltage shedding. Wang et al. (2021) use dueling deep Q-learning for emergency shedding in microgrids. Chen et al. (2023) apply PPO to frequency restoration in a modified IEEE 39-bus system. These are relevant precedents for learned decisions, while their physical tasks differ from hourly rationing.

GridGuard contains no voltage trajectory, frequency deviation or electrical graph. It therefore cannot reproduce transient-stability or response-time claims from those studies. Its sequential effects arise through cumulative service ratios and switching history. The policy must be assessed against those mechanisms rather than the broader promise of emergency-control research.

Schulman et al. (2017) introduce PPO's clipped policy objective and minibatch updates. The actor assigns probabilities to discrete allocation choices, while the critic estimates expected return from the current state. Performance depends on observations, reward, initialization and training; the algorithm's name provides no guarantee that it will beat a transparent rule.

García and Fernández (2015) distinguish changing the learning objective from restricting exploration. A finite reward penalty does not make an unsafe action impossible. GridGuard therefore imposes critical protection through an action mask, separately from reward weights. Huang and Ontañón (2022) support using the same valid-action distribution during sampling and policy updates. This consistency is essential when PPO recomputes probabilities for stored actions.

Fairness Rule demonstrations initialize the actor before PPO training. That prior
knowledge changes the interpretation: the system is imitation followed by reinforcement
learning, not learning from scratch. A favorable comparison against Fixed Rotation does
not isolate the benefit of PPO fine-tuning. Section 5.7 compares the current frozen
policies with this teacher on matched conditions; Appendix A retains the earlier,
separate teacher comparison, including its negative result.

## 2.4 Microgrid energy management

Microgrid studies provide related examples of sequential energy decisions. Ji et al. (2019) investigate daily microgrid operation through deep reinforcement learning, while Vázquez-Canteli and Nagy (2019) review demand-response algorithms and modeling choices. Storage, flexible consumption and prices can create substantial opportunities to trade present service against future benefits.

GridGuard adopts sequential allocation with a simpler physical model. It has no battery, price signal, energy trading or controllable generation. Time coupling comes from service history and tier changes. Learning's potential planning advantage is therefore narrower than in a system that can store electricity for later demand. An observed result must be explained by the mechanisms actually present.

This literature also raises the transfer problem. A policy that succeeds in simulation may fail under different demand patterns or equipment restrictions. Synthetic experimentation is useful for controlled comparison, but repetition does not validate the environment. The simulator's assumptions remain part of every conclusion.

## 2.5 Resilience and critical-service priority

Resilience research concerns maintaining and restoring important functions during disruption. Rosales-Asensio et al. (2022) investigate metrics that weight critical electricity customers according to the system and operator context. Poudel and Dubey (2019) connect restoration priority to feasible resources and network constraints. These studies support explicit priorities while requiring the protected quantity and its delivery mechanism to be defined.

Afzal et al. (2020) distinguish established reliability concepts from resilience under
major disruptions and review assessment frameworks. Their network-restoration setting
supports the need to define the scope of protection; this single-day allocation
simulator does not model damage recovery or measure whole-grid resilience.

GridGuard designates Hospital, Water and Bakery as critical to represent essential health, water and food services. Critical Only supplies their entire modeled sector demand. This is not a measurement of any particular facility's minimum operating load. A real hospital may contain essential and deferrable consumption, and real feeders may connect several sectors that cannot be independently switched.

Protection is also conditional on supply. If total critical demand fits, the mask permits only assignments that request all of it within capacity. If it does not fit, the all-Critical-Only fallback still experiences shortage. The evaluation records that infeasibility rather than reporting unconditional critical protection.

## 2.6 Humanitarian consequences of interruption

Irwin et al. (2020) review relationships between electricity access, reliability and health in low- and middle-income countries. They report relevant associations while rating the overall evidence quality as weak. Nduhuura et al. (2021) study household outage impacts in Accra and their association with household circumstances. These sources motivate attention to essential services and uneven burdens without establishing quantitative health benefits for this controller.

The project measures supplied critical energy, critical interruption duration, zero-service blackouts and locality allocation. It does not estimate mortality, clinical risk, food security or economic loss. The same interrupted hour may have different consequences for different facilities and people. Operational priorities would require richer data and stakeholder participation.

Locality averages also conceal internal differences. The model has no household income, vulnerability or backup-generator access. Better locality-level fairness therefore describes the represented allocation, not equality of every consequence experienced by residents.

## 2.7 Developing-country conditions and Sudan

Studies of Sudan connect chronic shortage to infrastructure, finance and unequal access. Al-Rikabi et al. (2025) discuss Sudan's renewable-energy situation, and Basheir and Abdelrahman (2022) analyze the electricity crisis and energy-transition choices. The World Bank (2025) provides additional context for essential-service access. These are contextual sources rather than measurements of the modeled localities.

National statistics in those documents are tied to their dates and source definitions. They do not calibrate the experiment's 700 MW illustrative peak, sector shares or locality weights. The Moderate and Severe shortage windows are controlled scenarios, not reproductions of a documented Khartoum day. Fresh synthetic traces permit repeatable comparison while remaining insufficient for operational calibration.

Familiar locality names and administrative geometry make the geographic setting understandable. They do not establish electricity-network topology. This separation supports an honest Sudan-inspired study without turning geographic context into an unsupported claim of physical fidelity.

## 2.8 Fairness, equity and evaluation

Jenkins et al. (2016) distinguish distributional, recognition and procedural dimensions of energy justice. This explains why total delivered energy is incomplete: who receives service, whose needs count and how decisions are governed all matter. GridGuard measures a narrow distributional component and names critical services explicitly. It does not implement participatory governance or claim to resolve energy justice.

Oluwasuji et al. (2020) formulate fair shedding with heterogeneous consumer utility and optimization. Their work motivates explicit allocation objectives, but their utility-based formulations are not interchangeable with the implemented metrics. GridGuard uses Jain's index (Jain et al., 1984) for cumulative demand-normalized locality service and direct duration disparities. It does not implement envy-freeness, and a fairness reward term is not a hard equality guarantee.

Agarwal et al. (2021) explain how uncertainty and limited training runs affect reinforcement-learning conclusions. This motivates reporting all five independently trained policies rather than selecting the best final-test actor. Bootstrap reasoning (Efron and Tibshirani, 1993) supports retaining controller pairing and the dependence between shortage variants of the same underlying scenario.

## 2.9 Position of this study

This ninth subsection synthesizes the preceding eight themes into a bounded contribution:
combine history-only prediction, transparent priority rules, exact current-hour
optimization, a constrained learned policy and direct service outcomes in an inspectable
Sudan-inspired simulation. The claim concerns an implemented combination and evaluation,
not proof that no previous study considered related priorities and fairness.

The literature also sets the comparison standard. Optimization is implemented as a competitor. Critical protection is enforced rather than inferred from a penalty. Forecasting remains distinct from control. Fairness and duration are both measured. Favorable, unfavorable and inconclusive results remain visible. These principles connect the review directly to the methods that follow.
# 3 Methodology

## 3.1 Simulation units and temporal resolution

The model uses the locality order Khartoum, Omdurman, Bahri, East Nile, Jebel Awlia, Karari
and Ombada. Each locality contains Hospital, Water, Bakery, Residential and Commercial
demand. The first three sectors are critical. These categories represent essential
services while keeping the action space small enough to inspect.

An episode contains 24 steps, each representing one hour. Demand and supply are measured
in MW at each step. Multiplying by the one-hour step duration gives energy in MWh; daily
energy is the sum over the 24 steps. The model uses one shared supply pool and imposes no
network distance or transfer-capacity cost between localities. Administrative geometry
has no role in the calculations.

Figure 3.1 connects the common controller inputs, shared dispatch and outcome calculations.
The forecasting branch is evaluated separately and supplies no future information to PPO.

{{RESEARCH_WORKFLOW_FIGURE}}

## 3.2 Synthetic demand

The illustrative system peak is 700 MW. It is divided among localities using the weights
in Table 1. The weights specify relative allocation rather than measured locality peaks.
Within each locality, sector coefficients define base demand before hourly profiles and
random perturbations are applied.

The two parameter lists are independent; every locality contains all five sectors.

**Table 1. Illustrative allocation parameters.**

| Locality | Relative demand weight | Sector | Base peak coefficient |
| --- | ---: | --- | ---: |
| Khartoum | 1.00 | Hospital | 0.08 |
| Omdurman | 0.95 | Water | 0.07 |
| Bahri | 0.85 | Bakery | 0.05 |
| East Nile | 0.80 | Residential | 0.55 |
| Jebel Awlia | 0.70 | Commercial | 0.25 |
| Karari | 0.75 | | |
| Ombada | 0.90 | | |

Hospital demand is approximately constant. Water demand has morning and evening peaks,
Bakery demand has early-morning and afternoon peaks, Residential demand has morning and
stronger evening peaks, and Commercial demand peaks around midday. The generator uses
deterministic normalized profiles, with independent uniform demand perturbations between
$-5\%$ and $+5\%$. The sector coefficients are base peak shares; because profiles differ,
instantaneous sector proportions vary over the day.

This construction gives every experiment an inspectable source of variation. It allows
the same episode to be reproduced from its seed, and permits changes in shortage severity
without changing the underlying random demand draw. It does not reproduce a measured
Khartoum load curve. The parameters should therefore be read as research assumptions
whose consequences can be examined, not estimates of actual energy requirements.

## 3.3 Available supply and shortage levels

Supply is derived from current total demand, a shortage fraction and a separate uniform
perturbation between $-5\%$ and $+5\%$. The generator caps supply at 95% of demand to retain a
shortage-only operating domain. The three shortage levels are defined in Table 2.

**Table 2. Shortage construction.**

| Shortage level | Base supply fraction | Additional reduction |
| --- | ---: | --- |
| Mild | 0.80 | None |
| Moderate | 0.65 | Subtract 0.10 during hours 17-20 |
| Severe | 0.50 | Subtract 0.15 during hours 16-21 |

A severity-neutral draw supplies the demand and supply disturbances for all three
versions of an episode. The resulting Mild, Moderate and Severe scenarios are coupled:
they share the same base randomness while applying different shortage transformations.
This makes a severity comparison easier to interpret, but also means that the three
versions are not statistically independent observations.

The default generator provides enough power to cover the modeled critical sectors under
the defined perturbation limits. This property is important to the meaning of the
critical-service constraint. When the constraint is satisfied, zero critical interruption
is feasible by construction. Other supply conditions could make that requirement
impossible; the controller has an explicit fallback for that case rather than an
unconditional guarantee.

## 3.4 Service tiers and dispatch

The simulator recognizes three locality service tiers. Off requests no power. Critical
Only requests Hospital, Water and Bakery demand. Full Service requests all five sectors.
Let requested sector demand be $q$, available system supply be $S$, and total requested
power be $Q$. The delivered load is calculated using one global factor:

$$
\alpha = \min\left(1,\frac{S}{\max(Q,\varepsilon)}\right)
$$

$$
\mathrm{delivered} = \alpha\,\mathrm{requested}
$$

Here $\varepsilon$ is a small numerical safeguard. If requests fit within supply, they are
delivered without scaling. If they exceed supply, every requested sector is reduced by
the same proportion. All controllers use this exact dispatch calculation. In
particular, dispatch does not silently restore critical demand after an excessive
request. Critical protection is achieved by the constrained controllers' admissible
choices before dispatch.

At each step the simulator records the requested and delivered sector loads, locality
tiers and reward. Cumulative demand and cumulative served energy are updated for the
next decision. The external demand and supply traces do not depend on the chosen
actions. The action's consequences for subsequent decisions are the changed service
history and previous tier.

## 3.5 Fixed Rotation

Fixed Rotation assigns Full Service to three consecutive localities for a three-hour
block and switches the remaining four localities Off. The starting locality advances
by three positions for each new block, wrapping around the seven-locality order.
Given the hour, the schedule is deterministic. It does not inspect current demand,
available supply, critical demand or cumulative service.

This rule provides a fixed-schedule baseline for evaluating adaptive allocation.
Its transparency makes it possible to explain why a locality is selected at a given
hour. However, it does not select Critical Only, and its Full Service requests can
underuse or exceed current supply. The result is a baseline for this specified
traditional scheduling strategy, not for the best achievable non-learning controller.

Because supply decreases under greater shortage, the rule's utilization ratio can rise
even though the schedule itself is unchanged. A smaller denominator does not indicate
adaptation. This is why utilization is interpreted alongside served energy and direct
interruption outcomes.

## 3.6 Shared critical-service constraints

PPO and Constrained Optimization reserve critical demand in every locality whenever physically
feasible. Their action space contains only Critical Only and Full Service. Seven binary
choices produce $2^7 = 128$ joint assignments. Let $C$ be total critical demand, $b_i$ be the
noncritical demand of locality $i$, and $x_i$ indicate whether that locality receives Full
Service. A candidate is admissible when:

$$
x_i \in \{0,1\}
$$

$$
C+\sum_{i=1}^{7}x_i b_i \leq S
$$

The implementation evaluates this condition using the same requested-load reduction
as the dispatcher, preserving the floating-point boundary. It does not add a hidden
supply tolerance that could admit a request above availability. Infeasible candidates
receive zero probability in the policy's categorical distribution.

If no candidate satisfies the condition, only the all-Critical-Only fallback is allowed.
Dispatch then shares the insufficient supply proportionally, and the episode records
the resulting critical interruption. The fallback is a declared response to infeasibility;
it is not counted as successful satisfaction of the constraint.

The same feasibility mask is used for demonstrations, policy sampling, probability
recomputation during PPO updates, validation and final inference. The mask therefore
changes what the policy is trained to choose, rather than merely replacing an unwanted
action after it has been selected. Within the feasible set, the network determines
which localities receive the remaining noncritical supply.

## 3.7 Constrained Optimization

Constrained Optimization enumerates all 128 assignments defined above, filters by the same feasibility condition, applies shared dispatch and scores the canonical current-hour reward. It receives current demand, supply, cumulative demand and served energy, and previous tiers. Exact score ties select the lowest candidate index. It receives no future demand or supply. The computation remains on CUDA and evaluates every candidate rather than sampling a subset.

This method solves its immediate decision exactly. It does not solve the complete 24-hour schedule. Each choice changes the cumulative service and switching history seen at the next step, so maximizing immediate reward need not maximize daily reward or every public metric. PPO can demonstrate a benefit only through measured episode outcomes; it cannot exceed the exact one-step maximum for an identical state, objective and action set.

The historical Optimization controller searched all 2,187 assignments of Off, Critical Only and Full Service under proportional dispatch. Its earlier results are retained separately. The main comparator here uses the same 128 protected assignments as PPO, making the attribution clearer. Neither formulation is a proven full-horizon optimum.
## 3.8 Decision process, observation and network

The environment is a finite-horizon decision process with $T=24$ and step duration
$\Delta t=1\,\mathrm{h}$. Write $d_{i,k,t}$ for sector demand in MW, $S_t$ for available
MW, $D_{i,t}^{-}$ and $E_{i,t}^{-}$ for locality cumulative demanded and served MWh
before hour $t$, and $a_{i,t-1}$ for the previous tier. The full simulator state can be
written as follows:

$$
X_t=(t,d_t,S_t,D_t^{-},E_t^{-},a_{t-1},\xi_t).
$$

Here $\xi_t$ denotes the exogenous scenario-generator state, including the fixed shortage
regime and the random process that supplies later demand and availability. Including
that state makes the simulator's transition well defined; it does not give the policy
access to future samples. The controller uses the observed projection described below.
This is a finite-horizon MDP formulation of the environment, not a proof that every
compressed observation is a sufficient Markov state for every future real-world process.

For PPO and Constrained Optimization, $a_t\in\mathcal A_t$ is one of the 128 binary
Critical Only/Full Service assignments surviving the mask in Section 3.6. The dispatch
in Section 3.4 gives delivered sector power $y_{i,k,t}$. The action-dependent transitions are:

$$
D_{i,t+1}^{-}=D_{i,t}^{-}+\Delta t\sum_k d_{i,k,t},\qquad
E_{i,t+1}^{-}=E_{i,t}^{-}+\Delta t\sum_k y_{i,k,t}.
$$

The next previous-tier vector is $a_t$, and time advances by one hour. The generator
advances $\xi_t$ and supplies the next demand and supply independently of the action.
Both cumulative vectors start at zero; the previous-tier vector starts Off, while the
first-hour switching penalty is explicitly zero. At hour 24 the episode terminates,
with no storage, terminal reward or carry-over into another episode. The policy seeks
the expected sum of the hourly reward in Section 3.9 with discount $\gamma=1$.

The policy observes 46 nonanticipative quantities. Here nonanticipative means using
only information available at decision time; it does not mean causal identification.
Four global features encode time using sine
and cosine, current supply relative to demand, and supply relative to the fixed system
peak. Each locality contributes normalized current demand, its current critical-demand
share, cumulative service ratio and previous service tier. These components form the
original 32-value state representation. Seven cumulative-demand and seven cumulative-
served-energy features complete the 46-value input.

The additional magnitudes matter because the same cumulative ratio can arise from
different amounts of past demand. They are normalized using 24 times the fixed peak of
the corresponding locality. Their normalization does not use future episode demand.
All features come from current observations and accumulated history; the policy receives
no future supply, future demand, target-hour value or final evaluation outcome.

With $P_i=700w_i/\sum_j w_j$ MW as the fixed locality peak and
$d_{i,t}=\sum_k d_{i,k,t}$, the four global inputs are
$\sin(2\pi t/24)$, $\cos(2\pi t/24)$, $S_t/\max(\sum_i d_{i,t},\varepsilon)$ and $S_t/700$.
Each locality contributes $d_{i,t}/P_i$, its critical-demand fraction,
$E_{i,t}^{-}/\max(D_{i,t}^{-},\varepsilon)$ and $a_{i,t-1}/2$.
The final fourteen inputs are $D_{i,t}^{-}/(24P_i\Delta t)$ and
$E_{i,t}^{-}/(24P_i\Delta t)$ for all seven localities. Thus every network input is
dimensionless. Ratio denominators use the implementation's $10^{-8}$ numerical floor
in their respective units; the floor is not an additional physical supply allowance.

Actor and critic use separate networks with two hidden layers of 256 $\tanh$ units. The
actor produces 128 unnormalized action scores, or logits, one for each joint assignment. The critic predicts one
scalar value. Hidden layers use orthogonal initialization; the actor output uses a
small initialization gain. During training, an action is sampled from the masked
distribution. During validation and final evaluation, the controller chooses the
highest-logit feasible action, breaking exact ties by the lowest candidate index.

## 3.9 Reward and learning objective

The hourly reward retains the project's established objective:

$$
r_t=\frac{5C_t+2U_t+2J_t-0.25W_t-3O_t}{9}
$$

All five components are dimensionless. Let $q_{i,k,t}$ be requested MW, let
$\mathcal K_c$ contain Hospital, Water and Bakery, and define total demand $D_t$,
total requested power $Q_t$, delivered power $Y_t$ and usable power $P_t^{u}$:

$$
D_t=\sum_{i,k}d_{i,k,t},\quad Q_t=\sum_{i,k}q_{i,k,t},\quad
Y_t=\sum_{i,k}y_{i,k,t},\quad P_t^{u}=\min(S_t,D_t).
$$

$$
C_t=\frac{\sum_i\sum_{k\in\mathcal K_c}y_{i,k,t}}
{\max(\sum_i\sum_{k\in\mathcal K_c}d_{i,k,t},\varepsilon)},\qquad
U_t=\frac{Y_t}{\max(P_t^{u},\varepsilon)}.
$$

For fairness, use cumulative energy **after** dispatch at hour $t$:

$$
z_{i,t}=\frac{E_{i,t+1}^{-}}{\max(D_{i,t+1}^{-},\varepsilon)},\qquad
J_t=\frac{(\sum_i z_{i,t})^2}{7\sum_i z_{i,t}^2+\varepsilon}.
$$

$$
W_0=0,\qquad W_t=\frac{1}{7}\sum_i\mathbf{1}[a_{i,t}\ne a_{i,t-1}]
\quad(t>0),\qquad
O_t=\frac{\max(0,Q_t-S_t)}{\max(D_t,\varepsilon)}.
$$

Thus $C_t$ measures the hourly critical-service fraction, $U_t$ measures usable-supply
delivery, $J_t$ measures equality of cumulative service ratios, $W_t$ measures tier
changes and $O_t$ measures normalized oversubscription. The numerical safeguard is
$\varepsilon=10^{-8}$, with the corresponding units for each denominator. The reward
normalizer 9 equals the sum of the positive weights; it does not convert the score to
a percentage or a monetary value. Daily reward is $R=\sum_{t=0}^{23}r_t$; its ideal
upper bound is approximately 24. Switching averages include the zero first-hour value,
so the reported daily rate uses 24 hours rather than 23 transition opportunities.

The weights express the study's chosen priorities. They are not estimates of the
monetary or social cost of interruption. For feasible masked actions, critical service
is protected and oversubscription is absent, leaving the learner to balance use of the
remaining supply, accumulated fairness and switching. A higher reward can still conceal
a tradeoff among these components. The thesis therefore treats reward as a diagnostic
of the training objective and presents public service outcomes separately.

PPO compares the probability of each sampled action under the updated policy with its
probability under the rollout policy. Its clipped surrogate limits the objective's
incentive for excessively large probability changes during an update. The probability
ratio and clipped surrogate are:

$$
\rho_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}
$$

$$
L^{\mathrm{CLIP}}(\theta)=\mathbb{E}_t\!\left[
\min\!\left(
\rho_t(\theta)\widehat A_t,\,
\operatorname{clip}\!\left(\rho_t(\theta),1-\epsilon_{\mathrm{clip}},
1+\epsilon_{\mathrm{clip}}\right)\widehat A_t
\right)\right]
$$

Here $s_t$ and $a_t$ denote the observed state and sampled feasible action,
$\pi_\theta$ is the masked policy, and $\theta_{\mathrm{old}}$ identifies the rollout
policy. The advantage $\widehat A_t$ is normalized over the rollout,
$\epsilon_{\mathrm{clip}}=0.2$, and $\mathbb{E}_t$ denotes an empirical minibatch mean.
The implemented loss is minimized:

$$
\mathcal{L}(\theta,\phi)=
-L^{\mathrm{CLIP}}(\theta)
+0.25\,\mathbb{E}_t\!\left[
\left(V_\phi(s_t)-\widehat R_t\right)^2\right]
-c_H\,\mathbb{E}_t\!\left[H\!\left(\pi_\theta(\cdot\mid s_t)\right)\right]
$$

$V_\phi$ is the critic, $\widehat R_t$ is the fixed value target computed by generalized
advantage estimation (GAE), and $H$ is the masked policy's entropy. The target uses the raw
advantage, $\widehat R_t=A_t^{\mathrm{raw}}+V_{\mathrm{old}}(s_t)$, before advantage normalization.
The squared-error coefficient is exactly $0.25$: the value
coefficient $0.5$ multiplies half the mean squared error. The entropy coefficient
$c_H$ follows the schedule below. Stored masks, value targets and rollout probabilities
remain fixed during these updates.

GAE uses complete 24-step episodes. The terminal value is
zero because the episode has ended. The discount factor is 1.0, matching the
undiscounted daily score; the GAE parameter is 0.95. Advantages are normalized over a
complete rollout before minibatch optimization. Episodes are not linked across their
boundaries.

## 3.10 Demonstrations and training procedure

Before reinforcement learning, each actor is initialized using demonstrations from a
deterministic Fairness Rule. Among feasible actions within 0.005 of the best current-hour
reward, it maximizes the minimum projected cumulative locality service ratio. Reward
and then the lowest candidate index resolve ties. The tolerance was chosen in a prior
study and is not retuned on the present experiment's validation or test outcomes.

Each run collects 455 demonstration episodes per shortage level, giving 1,365 episodes
and 32,760 supervised states. Behavior cloning lasts ten epochs with minibatches of
1,024, including the last partial minibatch. Its loss combines masked action
cross-entropy and 0.25 times the mean squared critic error against teacher return-to-go.
A fresh PPO optimizer starts after this initialization. The retained behavior-cloning
checkpoint makes the procedure's starting point inspectable.

Five training runs use seeds 1729, 2718, 3141, 5772 and 8111. Each run completes 3,000
iterations containing 256 full episodes, for 18,432,000 reinforcement-learning
transitions per run and 92,160,000 across the five runs. Demonstration collection and
evaluation are outside those totals. Independent scenario streams are used across
runs, and separate random streams control initialization, action sampling and minibatch
ordering. Table 3 collects the fixed configuration used by all five runs.

**Table 3. PPO training configuration.**

| Setting | Value |
| --- | ---: |
| Actor and critic hidden layers | 256, 256 |
| Actor input and output dimensions | 46 and 128 |
| Iterations per run | 3,000 |
| Episodes per iteration | 256 |
| Episode length | 24 hours |
| PPO epochs per rollout | 4 |
| Minibatch size | 1,024 |
| Initial Adam learning rate | 0.0003 |
| Discount factor | 1.0 |
| GAE parameter | 0.95 |
| PPO clipping coefficient | 0.2 |
| Maximum gradient norm | 0.5 |
| Entropy coefficient | 0.01 to 0.001 |

Learning rate and entropy coefficient decrease linearly to their final values. Each
rollout uses 85 episodes per shortage level plus one rotating extra episode, giving
equal severity exposure over 3,000 iterations. The training budget and checkpoint
schedule are fixed before final evaluation.

## 3.11 Learned demand prediction

Ridge AR(3) fits each of the 35 locality-sector series using only observed demand at
$t-3$, $t-2$ and $t-1$ to predict hour $t$. Seeds 42, 43 and 44 provide 256 training
traces each: 768 traces and 16,128 targets per regression over hours 3–23.

Training-only means and population standard deviations normalize inputs, with a
$10^{-8}\,\mathrm{MW}$ scale floor. Float64 ridge regression fits centered targets by
minimizing mean squared residual plus a 0.001 coefficient penalty; the intercept is
unpenalized. Predictions restore the target mean and clamp negatives to zero. Fitted
parameters are retained.

Baselines use the same three-hour mean or last-hour persistence on identical targets.
Evaluation uses 320 separate demand traces once each because shortage variants share
demand. This forecast remains separate from PPO observations and the display projection.

Mean absolute error (MAE) and root mean squared error (RMSE) are reported in MW for
locality-sector and system-total targets. System error sums the 35 signed residuals
before taking absolute values or squares. Saved counts and error sums reconstruct
pooled errors with the same target weighting.
# 4 Application implementation

## 4.1 Architecture and execution

GridGuard Khartoum separates scientific calculations from presentation. Domain objects
define locality identities, sectors, units and valid tensor shapes. Scenario generation
creates demand and supply. Controllers provide service tiers. The simulator applies
the shared dispatch rules, and metric functions calculate outcomes from the resulting
traces. A presentation layer then converts selected results into interface data.

The implementation uses Python 3.12, PyTorch 2.8 with CUDA 12.8, CuPy and PySide6. PyTorch
(Paszke et al., 2019) executes batched generation, policy training, inference, dispatch and metrics.
CuPy (Okuta et al., 2017) performs resampling and percentile calculations, sharing GPU arrays through the DLPack interchange interface.
The scientific arrays remain on the GPU until the artifact or presentation transfer.
This structure supports repeated matched experiments while keeping model calculations
out of interface widgets.

The primary presentation and benchmark path requires the validated NVIDIA CUDA runtime.
Startup checks the GPU environment and required model and evidence resources. When a
required resource cannot be verified, the software reports the problem instead of
presenting fabricated results. Small host-side checks support development and artifact
assembly; they do not substitute a different scientific pipeline for the published
GPU experiment.

## 4.2 The desktop workflow

The native application organizes the demonstration into Overview, Live Scenario and
Results. The Overview introduces the localities, scope and available shortage levels.
Live Scenario allows the presenter to choose a shortage level and seed, run a matched
episode and inspect its hourly allocation. It uses the validation-selected PPO policy
from training seed 8111 at iteration 2500, alongside Fixed Rotation and Constrained
Optimization. Results retains the frozen five-run study; one live policy and scenario
do not replace its matched-scenario averages.

The visual design uses an open pale canvas, restrained teal controls, clear typography
and a contrasting selection color. The map, locality selector and time slider support
inspection of one locality and hour without hiding the whole-day comparison. Textual
service labels accompany color, and keyboard navigation provides access to the main
pages and locality choices. The interface remains an explanatory tool: a polished map
does not add physical realism to the underlying model.

Figure 4.1 shows locality context and navigation in the application overview. It is a
software demonstration, not evidence of operation on the Khartoum electrical network.

{{SCREENSHOT_OVERVIEW}}

Figure 4.2 shows the selected locality, service tier and hourly PPO replay. The desktop
demonstrates one validation-selected policy; the main study averages five independent runs.

{{SCREENSHOT_LIVE}}

## 4.3 Offline geography and result handling

The locality display uses packaged administrative boundaries from the United Nations
Office for the Coordination of Humanitarian Affairs (n.d.), with Nile river context from
Natural Earth (n.d.). These assets allow the presentation to function
offline and provide recognizable geographic orientation. They do not identify feeders,
substations, supply routes or measured service territories. No geographic coordinate
enters the policy's state or the simulator's dispatch calculation.

Live simulation runs away from the main interface event loop so that the window remains
responsive. Once a complete episode is available, playback advances through already
calculated frames. Moving the time slider does not retrain a model or alter the
published benchmark. A live demonstration helps explain behavior on one episode; the
main results rely on the separately evaluated matched scenario set.

Models and evidence are stored with explicit identities and SHA-256 checks. The
constrained training study retains numeric weights rather than executable training
checkpoint objects. Verification checks array shapes, finite values, model identity,
scenario matching and metric consistency. These checks provide traceability, while
the credibility of the scientific conclusion still depends on the validity of the
model and study design.

## 4.4 Verification approach

Tests cover controller action rules, mask feasibility at the supply boundary, the
critical-only fallback, reward and metric identities, terminal advantage estimation,
stored action probabilities, checkpoint selection and independence of scenario splits.
GUI checks cover page navigation, map interaction, selection, replay states and the
minimum presentation layout. Real CUDA tests confirm the intended computation path.

Independent replay provides a further check between selected model files and recorded
results. It regenerates scenarios, executes saved controllers and compares actions and
outcomes. The final verification status for the current experiment is summarized in
Section 5.9; detailed hashes and reproduction references belong in Appendix A rather
than interrupting the scientific argument.

# 5 Experimental design and results

## 5.1 Matched control evaluation

The main comparison includes Fixed Rotation, Constrained Optimization and five selected PPO actors. Five iteration-600 actors from the same training runs provide a secondary training-budget reference. All 12 methods receive identical inputs for each condition and use common dispatch and metrics. Final evaluation contains 960 conditions and 11,520 complete trajectories.

Training and demonstration scenario streams occupy separate ranges beginning at 4,000,000 and 4,500,000, with run-specific offsets. Validation uses seeds 122101–122105; final testing uses 122201–122210. Each seed generates 32 base draws with three coupled shortage variants. None of the final scenarios informs gradient updates, hyperparameters or checkpoint selection. The exact seed formulas and source bindings are retained in the protocol.

Table 4 distinguishes the selection set from the final-test set and counts coupled
conditions separately from independent base draws.

**Table 4. Validation and final-test scenario splits in the original focused study.**

| Evaluation element | Validation | Final test |
| --- | ---: | ---: |
| Seed groups | 5 | 10 |
| Independent base draws | 160 | 320 |
| Shortage variants per draw | 3 | 3 |
| Matched conditions | 480 | 960 |
| Independently trained PPO runs | 5 | 5 |

PPO inference uses deterministic highest-logit feasible actions. Variation arises from training differences and scenario generation, rather than repeated stochastic decisions at test time. The equally weighted five-run mean describes the learning procedure. The actor with the best final-test result is not substituted for it.

## 5.2 Validation selection and longer training

Each run is evaluated at iteration zero and every 100 iterations through 3,000. A checkpoint is eligible only when every validation condition has zero critical-outage locality-hours, zero critical-interruption locality-hours and zero power-balance violations, and fairness and utilization are each at least Fixed Rotation overall and at every shortage level. The eligible checkpoint with highest mean daily reward is selected; exact ties favor the earlier iteration. If none qualifies, the highest-reward fallback is identified explicitly and a positive primary claim is forbidden.

The rule is applied independently to all five runs. Selected weights and validation records are frozen before final testing. Constrained Optimization is a fixed comparator; no final-test-best actor, checkpoint or shortage-specific policy is selected. An optional validation-selected actor for demonstration never replaces the five-run result.

The iteration-600 snapshots belong to the same planned 3,000-iteration learning-rate and entropy schedule. They describe continuing that schedule, not an independently tuned short-budget method. A larger training budget creates an opportunity for improvement, but is not itself evidence of benefit.

## 5.3 Outcome definitions

Critical interruption occurs when positive critical demand receives less than 99% service during an hour. Summing across seven localities and 24 hours gives zero to 168 locality-hours. One locality-hour means one affected locality for one hour, not a system-wide clock hour.

For locality $i$, cumulative service ratio $z_{i,t}$ equals cumulative served energy divided by cumulative demand. Jain fairness is calculated across the seven ratios at every hour, then averaged across all 24 hours:

$$
J_t=\frac{\left(\sum_{i=1}^{7}z_{i,t}\right)^2}
{7\sum_{i=1}^{7}z_{i,t}^2+\varepsilon}
$$

A high index indicates similar demand-normalized service, not necessarily Full Service. It is not only an end-of-day index. Utilization is the mean hourly ratio of delivered power to usable power, $\min(\mathrm{supply},\mathrm{total\ demand})$, not a ratio of daily energy totals. Public critical-service ratio pools critical energy within an episode, whereas the reward uses each hour's critical-service fraction.

Table 5 defines the units and interpretation of the public outcomes.

**Table 5. Evaluation outcomes, units and interpretation.**

| Outcome | Unit | Interpretation |
| --- | --- | --- |
| Critical interruption | Locality-hours/day | Lower burden is favorable |
| Jain service fairness | Percent | Higher equality of cumulative service ratios |
| Supply utilization | Percent | Higher delivery of usable supply |
| Zero-service blackout | Locality-hours/day | Positive demand with exactly zero delivery |
| Blackout duration range / pairwise gap | Hours | Disparity in zero-service duration |
| Critical-interruption range / pairwise gap | Hours | Disparity in duration below 99% critical service |
| Longest blackout / critical interruption | Hours | Separate maximum locality streaks within the episode |
| Served / curtailed energy | MWh | Direct energy allocation |
| Switching rate | Mean hourly fraction | Operational proxy penalized in reward |
| Power-balance violation | Fraction of hours | Delivery above supply plus 0.001 MW |
| Daily reward | Unitless | Canonical objective diagnostic |

Duration range is the maximum minus minimum locality duration. Pairwise gap averages absolute differences across all 21 unordered locality pairs. They are calculated within each episode before aggregation. Longest streaks stop at episode boundaries. Zero disparity can describe uniformly adequate or uniformly poor service, so it is read alongside burden and energy. These direct measures assess blackout-duration inequality without relabeling Jain fairness.

## 5.4 Uncertainty and decision criteria

Effects are PPO minus the named comparator on matched conditions. Negative critical-interruption effects favor PPO; positive fairness and utilization effects favor PPO. Percentage measures use percentage-point differences. Unfavorable results and ties remain visible.

A hierarchical paired bootstrap uses 10,000 resamples with seed 20261024. It resamples
the five training runs and the 32 base-draw clusters within each of the ten final-test
seed groups. The same sampled clusters are shared across methods and training runs;
each cluster retains its coupled Mild, Moderate and Severe variants. The two primary
comparison claims use Bonferroni-adjusted 97.5% two-sided intervals, with linearly
interpolated 1.25th and 98.75th percentile endpoints. Descriptive, per-run and
secondary-budget intervals use 95% coverage. Shortage-specific intervals are exploratory;
shortage mean guardrails remain part of the declared primary decisions.

Every reported PPO procedure mean weights the five selected runs equally, then weights
all 960 conditions equally. Within a condition, fairness, utilization and switching
are means over 24 hours, reward and locality-hour burdens are sums, and energy is
integrated in MWh. For metric $m$, the overall paired effect is:

$$
\widehat\Delta_m=\frac{1}{5\times960}
\sum_{j=1}^{5}\sum_{c=1}^{960}
\left(m_{j,c}^{\mathrm{PPO}}-m_{j,c}^{\mathrm{comparator}}\right).
$$

The fixed and optimizer values repeat across training-run index $j$; for the budget
comparison the comparator is the same run's iteration-600 actor. Severity means use
the corresponding 320 conditions with the same equal-run weighting. Write $L_m$ and
$U_m$ for interval endpoints, and $\Delta_{m,h}$ for the mean difference at shortage
level $h$. A composite comparison passes only if **every** requirement below holds.

For **PPO versus Fixed Rotation**, the adjusted 97.5% endpoints must satisfy
$U_{\mathrm{critical\ interruption}}<0$, $L_{\mathrm{fairness}}>0$ and
$L_{\mathrm{utilization}}>0$. At each of Mild, Moderate and Severe, the mean
critical-interruption difference must be at most zero and the fairness and utilization
differences must each be at least zero. Each of the five selected PPO actors must have
zero critical-outage locality-hours, zero critical-interruption locality-hours and zero
power-balance violations in **every** final-test condition. All five validation selections
must be eligible under Section 5.2. Fixed Rotation may have critical interruptions;
reward is diagnostic and cannot substitute for these service requirements. This is
a test of the complete constrained system, not the isolated effect of learning.

For **PPO versus Constrained Optimization**, the adjusted 97.5% endpoints must satisfy
$L_R>0$, $L_{\mathrm{fairness}}\geq-0.002$ and
$L_{\mathrm{utilization}}\geq-0.002$. The ratio margin is 0.2 percentage points.
At **each** shortage level, the reward mean difference must be nonnegative and the
fairness and utilization mean differences must each be at least $-0.002$. Each selected
PPO actor **and** Constrained Optimization must have zero critical-outage locality-hours,
zero critical-interruption locality-hours and zero power-balance violations in every
condition. All five validation selections must be eligible. The analysis separately
reports whether the reward mean and adjusted lower endpoint exceed $+0.01$ daily reward.
That practical reference does not waive any failed service requirement. Both the
noninferiority margin and the reward reference are research choices, not validated
operating standards.

For the **secondary additional-training comparison**, the paired 95% interval must
have $L_R>0$, $L_{\mathrm{fairness}}\geq-0.002$ and
$L_{\mathrm{utilization}}\geq-0.002$. Each shortage must have nonnegative mean reward
difference and fairness/utilization differences of at least $-0.002$. Both the selected
and iteration-600 actors must meet the same zero-interruption, zero-outage and
zero-power-violation checks in every condition. All five selections must be eligible
and each selected iteration must exceed 600. Otherwise the result is reported as a
tradeoff, inconclusive comparison or earlier-checkpoint selection, as appropriate.

These rules, the training configuration and scientific source hashes were frozen in the
23 October 2026 protocol **before training and validation**. All five runs and their
validation selections were frozen before generating or inspecting the final-test
scenarios. This is a recorded internal protocol, not external preregistration or
independent custody. The uncertainty describes this generator and five training runs;
960 conditions are not 960 independent policies. The later audit in Section 5.7 is
explicitly post-hoc and does not change these original decision rules.

## 5.5 Main controller results

Table 6 compares all three procedures on common conditions. PPO values weight the five runs equally. Critical interruption, fairness and utilization lead the interpretation; duration, energy, switching and reward explain the behavior behind those outcomes.

{{RESULTS_TABLE}}

Figure 5.1 displays the three main service outcomes with separate units. Critical
protection and distribution quality are not compressed into one visual score.

{{RESULTS_FIGURE}}

{{PRIMARY_INTERVALS}}

Figure 5.2 plots PPO minus Constrained Optimization effects with adjusted 97.5% intervals.
The two primary comparisons both appear in Tables 8 and 9; this figure shows only the
optimizer comparison and does not present validation learning curves.

{{PRIMARY_CI_FIGURE}}

{{RESULTS_INTERPRETATION}}

Critical protection must be read together with the action rules. PPO reserves critical demand and can select Critical Only; Fixed Rotation can turn a locality entirely Off. Shared protection in PPO and Constrained Optimization makes their comparison more informative about scheduling after that engineering rule is imposed. A tie in critical interruption is expected when both satisfy the same reservation. The remaining outcomes show whether learning improves noncritical allocation.

Zero blackout duration and zero duration disparity do not imply Full Service. Critical Only can remove zero-service blackouts while substantial Residential and Commercial demand remains unmet. Duration results are therefore interpreted with service fairness, utilization and delivered energy, rather than as a complete measure of distributional equality.

## 5.6 Shortage levels and learning progression

Mild conditions allow more Full Service combinations. The Severe shortage window restricts the number fitting within supply. Table 10 reports each level so a favorable overall mean cannot conceal a weaker condition.

{{RESULTS_BY_SHORTAGE}}

Validation trajectories and retained checkpoints show learning without repeatedly consulting the final test. Individual run outcomes reveal sensitivity to initialization. A single favorable seed is insufficient to characterize the procedure.

Figure 5.3 shows each run's mean daily validation reward, Jain fairness and utilization,
with selected checkpoints and the same-run iteration-600 reference. These are validation
learning trajectories, not final-test performance at every training iteration.

{{TRAINING_CURVE_FIGURE}}

The TensorBoard diagnostics use the same recorded training values shown in the live monitor.
Reward, policy loss, value loss, entropy, approximate KL divergence and clipping fraction
are sampled every 25 PPO iterations, from iteration 25 to 3,000. Thin curves show the
five runs; the darker mean and shaded band show the mean plus or minus one sample
standard deviation. The band describes variation between training runs, not a confidence
interval for final performance.

Each loss point is a minibatch-size-weighted mean over that update's four epochs,
not a moving average over 25 updates. The value-loss panel reports half the mean
squared error before its additional $0.5$ weight. Training reward is the pre-update
mean daily reward over the rollout's 256 episodes.

For a recorded metric $x_{k,j}$ from training run $k$ at iteration $j$, the mean and
sample variance are:

$$
\bar{x}_j=\frac{1}{5}\sum_{k=1}^{5}x_{k,j},\qquad
s_j^2=\frac{1}{4}\sum_{k=1}^{5}\left(x_{k,j}-\bar{x}_j\right)^2.
$$

Figure 5.4 displays these unsmoothed training diagnostics, including the mean and
sample-standard-deviation band. It is a training plot, distinct from the validation
curves in Figure 5.3 and the final-test intervals in Figure 5.2.

[[FIGURE:diagnostics]]

The validation-variance graph shows daily reward, Jain fairness and utilization over
all 31 checkpoint evaluations, including the
behavior-cloning starting point. It measures variability across independently trained
policies on the same validation conditions. Fairness and utilization are converted to
percentage points before calculating their variances; their axes therefore use squared
percentage points. Figure 5.5 displays those variances. These descriptive curves do not
replace the paired final-test intervals.

[[FIGURE:variance]]

{{BUDGET_COMPARISON_TABLE}}

The budget comparison is interpreted through its paired effects, service margins and selected iterations. A small score increase is not automatically a meaningful improvement in service. Unfavorable or inconclusive findings remain part of the result.

## 5.7 Post-hoc matched baseline and feasibility audit

The committee review prompted an additional comparison with Fairness Rule and a
critical-aware rotating baseline. It reuses the original 960 final-test conditions and
the five frozen selected PPO actors; no actor is retrained or selected using its results.
This audit is post-hoc, not a fresh confirmatory holdout, and it does not change the
original criteria in Section 5.4. Its purpose is to expose comparator dependence and
separate critical protection from discretionary allocation.

{{FOLLOWUP_AUDIT}}

## 5.8 Learned forecasting results

The prediction experiment uses 320 demand traces from seeds 91001–91010, separate from the forecasting fit. Each method predicts hours 3–23, giving 235,200 locality-sector targets and 6,720 system targets. Common targets and training-only normalization prevent leakage through different windows or preprocessing.

Table 16 reports pooled errors on the common held-out target set.

**Table 16. Forecast error on 320 held-out demand traces; 235,200 locality-sector and 6,720 system targets.**

| Forecast method | Cell MAE (MW) | Cell RMSE (MW) | System MAE (MW) | System RMSE (MW) |
| --- | ---: | ---: | ---: | ---: |
| Three-hour trailing mean | 2.3385 | 3.7436 | 42.7512 | 49.9759 |
| One-hour persistence | 1.3465 | 2.1776 | 24.3809 | 28.5917 |
| Ridge AR(3) | 0.6573 | 1.1882 | 9.3376 | 11.9242 |

Ridge AR(3) has the lowest error on all four measures. This demonstrates learned prediction benefit for the selected synthetic comparison, not real-demand calibration or superiority over every architecture. The figures are descriptive pooled errors; no forecast confidence interval is asserted.

Saved evidence retains fitted parameters and per-seed counts, absolute-error sums and squared-error sums, not the full runtime prediction arrays. Errors can be reconstructed from these sufficient statistics. Forecasts remain separate from the policy and current display projection, so no controller improvement is attributed to them.

Figure 5.6 visualizes the same MAE and RMSE values from Table 16; it concerns prediction
accuracy rather than the controller reward.

[[FIGURE:forecast]]

## 5.9 Computation and verification

{{TRAINING_SUMMARY}}

Verification checks retained weights against validation records, common input identities across methods, action constraints and metric summaries. Statistical reconstruction checks intervals and decisions. Replay connects saved actors to regenerated inputs, recorded actions and outcomes. These checks address different failure modes; an evidence hash alone does not validate the scientific model.

# 6 Discussion

## 6.1 What AI and optimization contribute

The forecast and control studies provide separate evidence. Ridge AR(3) improves synthetic prediction; PPO is judged against Fixed Rotation and Constrained Optimization on the declared outcomes.

The optimizer selects the best feasible current action:

$$
\operatorname{arg\,max}_{a}\,r_t(s_t,a)
$$

PPO learns a policy maximizing:

$$
\mathbb{E}\!\left[\sum_{t=0}^{23}r_t\right]
$$

PPO cannot exceed this one-step maximum at the same state. A daily gain can arise from
changed service history and switching; no energy is stored. Section 5.5 quantifies the
gain as 0.0499 reward units alongside about 19.88 fewer served MWh per synthetic day
and increased switching. The reward weights value the fairness change sufficiently to
raise the score, but the declared service criterion still rejects the comparison.
Optimization therefore remains a credible choice when utilization and transparent
decision rules matter. A composite score does not settle those priorities for operators.

Findings do not establish superiority to full-horizon optimization, model predictive
control or all adaptive rules. Section 5.7 directly examines the current frozen actors
against the Fairness Rule teacher and critical-aware rotation. The earlier negative
teacher comparison remains separately identified in Appendix A.

## 6.2 Critical protection and fair service

Critical protection comes from the mask and Critical Only. Default supply also permits a deterministic rule to protect these sectors. PPO allocates remaining supply; its Fixed Rotation advantage cannot be attributed solely to learning.

Equal partial service can produce high Jain fairness. Critical Only can remove blackouts while Residential and Commercial demand remains unmet. High utilization may repeatedly exclude a locality. Duration, burden and energy must be interpreted together.

No equipment wear, restoration delay or consumer switching cost is modeled. The 99% critical threshold and reward weights are research choices; operational priorities, facility requirements and tradeoffs need agreement.

## 6.3 Data and physical limitations

Synthetic profiles, fixed locality weights and demand-linked supply limit generalization. Fresh seeds prevent sample reuse but do not test changed demand patterns, prolonged damage or different shortage mechanisms.

Global supply omits network connectivity, feeder limits, losses, voltage and frequency constraints. Sector separation may be physically unavailable. Hourly, single-day decisions omit transients and multi-day disadvantage; administrative geometry supplies no electrical validation.

A more realistic simulation requires the operational data identified in Table 18.

**Table 18. Data required for future physical and operational validation.**

| Required information | Purpose in a more realistic model | Present treatment |
| --- | --- | --- |
| Metered locality and sector demand | Fit profiles and assess changing demand | Illustrative profiles and random perturbations |
| Independent generation and import availability | Represent actual supply limits and disruptions | Supply fraction derived from simulated demand |
| Facility inventory and essential operating loads | Define feasible critical protection | Aggregate Hospital, Water and Bakery categories |
| Recorded interruption timing and causes | Validate duration and service-history behavior | Modeled hourly allocations only |
| Feeders, capacities and switching permissions | Determine physically feasible service actions | Shared global supply and abstract tiers |
| Stakeholder priorities and vulnerability data | Assess burden and acceptable tradeoffs | Stated critical categories and locality metrics |

These are required future data, not collected sources.

## 6.4 Learning and evaluation limitations

Five runs remain a small sample, and one recipe cannot establish performance across
architectures, rewards or initializations. Conclusions concern this procedure and
generator. Section 5.6 introduces the additional-training utilization loss of about
0.24 percentage points before this interpretation: longer training improved its daily
objective but did not pass the stated service margins. More computation is therefore
not equivalent to uniformly better service.

The action count grows as $2^N$, requiring different representations and new feasibility tests at larger scales. Imitation plus PPO leaves component attribution unresolved. Comparing training from scratch, imitation alone and fine-tuning under equal budgets would clarify their contributions.

Protocols and hashes support reproduction, not independent custody or field validation. Prior exploratory and unfavorable findings remain disclosed.

## 6.5 Implications for further work

Representative data and feasible network actions should precede planning with forecasts. Give competing methods equivalent forecasts and evaluate service under prediction error. Longer-horizon optimization and strong transparent rules should remain comparators.

The offline application meanwhile supports research by making assumptions, choices and matched outcomes understandable.
# 7 Conclusion

GridGuard Khartoum integrates shortage simulation, history-only prediction,
critical-priority allocation, constrained optimization, PPO and offline communication.
Its contribution is a reproducible comparison that identifies both objective gains
and service costs. Ridge AR(3) separately improves synthetic prediction error over the
selected simple baselines; its forecasts are not controller inputs.

On the original 960 matched conditions, the complete constrained PPO system met the
declared comparison against Fixed Rotation. Critical protection follows from the action
mask and sufficient supply, so the reduction from 102.05 to zero critical-interruption
locality-hours is not a standalone demonstration that learning is necessary.

Against the constrained optimizer, PPO increased fairness by 2.24 percentage points and
daily reward by 0.0499, with a 97.5% reward interval of [0.0406, 0.0577]. Both protected
critical service. PPO used 0.40 percentage points less available supply and switched more
often, so it failed the full declared service-guarded improvement criterion. The evidence
therefore supports a measured benefit of this trained procedure in fairness and the
daily objective against this current-hour optimizer, with an efficiency tradeoff.
It does not establish universal superiority to optimization or an operationally
preferable allocation.

Completing the planned fivefold training budget improved daily reward by 0.0295 relative to the
same runs at iteration 600, but also reduced utilization by 0.24 percentage points. More
training strengthened the learned objective without satisfying every service requirement.

The post-hoc audit strengthens the comparator analysis: PPO improves discretionary
allocation over critical-aware rotation, but its 0.0102 reward advantage over Fairness
Rule is accompanied by 16.68 fewer served MWh per day, lower utilization and more
switching. Reward gains occur in Severe shortage while Mild and Moderate results favor
the teacher. These findings do not establish uniform superiority or isolate the effect
of PPO fine-tuning from imitation. Under inadequate critical supply, the fallback
distributes the shortfall without exceeding capacity, while recording the unavoidable
critical interruptions.

The resulting software and evidence provide an inspectable academic decision-support system. The work does not claim national-grid calibration, operational readiness or universal AI superiority. Advancing toward those goals requires measured data, physically feasible actions and external engineering assessment.

# Appendix A Supporting studies and reproducibility

## A.1 Earlier comparisons

A previous constrained-PPO study evaluated five 600-iteration runs on a separate 960-condition test. PPO's mean daily reward was 23.44892, compared with 23.48759 for Fairness Rule. The difference was −0.03867 with a pointwise 95% interval of [−0.04425, −0.03319]. Both protected critical service in those default conditions. PPO improved reward and fairness over its behavior-cloning starting point but did not meet the full improvement criterion against Fairness Rule.

An earlier frozen-policy hybrid improved on constrained greedy allocation without beating Fairness Rule overall. The previous 2,187-action optimizer comparison also showed a tradeoff: Optimization had lower critical interruption and higher utilization, while the then-frozen PPO had higher Jain fairness. Those values concern different policies and scenario sets and are not mixed into the current main table. The longer training study does not retroactively reverse those findings.

## A.2 Research objectives and evidence

The research questions in Section 1.2 connect model construction, prediction, protected
allocation, matched evaluation and interpretation. Chapters 3 and 5 explain the data
generation rules and the experiments; Section 6.3 identifies the operational data needed
for a more realistic simulation.

The seven objectives are addressed through a Sudan-focused literature context; a
seven-locality, five-sector simulator; learned forecasting and PPO decision support;
critical-service tiers and constraints; comparisons with prediction baselines, priority
rules and constrained optimization; matched evaluation against Fixed Rotation; and
separate measures of downtime, service fairness, utilization and energy. The offline
application makes these assumptions and results available for inspection.

This scope supports a synthetic software study. It does not turn illustrative
parameters into measurements or make a favorable learning result a condition for a
valid research contribution.

## A.3 Evidence and reproducibility

The project source and review package are available in the
[GitHub repository](https://github.com/boomout2012-afk/gridguard-khartoum).
The reviewer guide identifies the current thesis, presentation, GUI captures and
evidence. The original submission state preceding this revision is retained at commit
`10ecbb425ddb05e36e602505a160a46345a4e635`; historical release evidence and later
follow-up analyses remain separately labelled. Reproducing the scientific runs requires the documented CUDA
environment; checking the supplied file manifest requires only Python.

The source literature review is retained at `docs/source/literature-review-authors.docx`,
with bibliographic and claim checks recorded in `docs/LITERATURE_REVIEW_AUDIT.md`. The
current control protocol is `docs/FOCUSED_PPO_PROTOCOL.md`. The research archive also
retains the forecasting study, earlier optimization comparisons and previous teacher
comparison alongside their supporting evidence.

The separately hashed `data/evidence/committee-baseline-audit.json` records the post-hoc
teacher, critical-aware rotation and infeasibility comparisons, including paired,
per-run and shortage-specific results. It preserves the original model identities
and exact replay of unchanged methods. Its descriptive 95% intervals and later audit
recipe do not replace the original protocol or adjusted primary intervals.

Model weights, configuration, source identities and scenario/action hashes bind each experiment to its evidence. Current, historical and live results remain distinct. An interrupted initial training attempt was retained and restarted under the same frozen configuration; the training log records it separately, and completed-run timings exclude that attempt. No final-test results were consulted to restart or select training.

Training and evaluation elapsed times describe experimental cost, not single-decision
latency. Comparing controller speed would require common hardware, batch size and
synchronization rules; this study makes no speed-superiority claim.
# Appendix B Source code and review package

The complete project source and review documents are available in the
[GridGuard Khartoum GitHub repository](https://github.com/boomout2012-afk/gridguard-khartoum).
The repository README is the entry point for the thesis, presentation, current GUI
captures and reviewer guide. It also distinguishes current evidence from historical
release artifacts.

The `src/gridguard/` package contains the domain model, scenario generation, dispatch,
controllers, metrics, forecasting and PySide6 interface. The `tests/` directory contains
unit, integration, GPU, GUI and packaging checks. `pyproject.toml` and `uv.lock` define
the project and locked dependencies. `tools/` and `scripts/` provide reproducible study,
document and packaging commands; `data/evidence/` and `data/model/` retain the scientific
results and policy identities used by the application.

The core paths include `domain.py` for the canonical reward, `simulation.py` for shared
dispatch and transitions, `constrained_ppo.py` for protected PPO actions,
`focused_training.py` for the five-run procedure, `focused_results.py` for current-study
integrity, and `scenarios.py` for synthetic inputs. These modules are under
`src/gridguard/`; scientific calculations are separate from the UI.

From the repository root, `python tools/verify_review_package.py` checks the submitted
file manifest using only the Python standard library. To run the application, install
the locked Python 3.12 environment with `uv sync --locked --extra dev --extra docs`,
check the NVIDIA CUDA runtime with `uv run --no-sync gridguard-doctor`, and launch
`uv run --no-sync gridguard-desktop`. The initial installation needs a package cache or
internet access; the installed application runs offline. The README and
`SUBMISSION_README.md` document the full GPU tests and Windows build procedure.

# References

Afzal, S., Mokhlis, H., Illias, H.A., Mansor, N.N. and Shareef, H. (2020) ‘State-of-the-art review on power system resilience and assessment techniques’, *IET Generation, Transmission & Distribution*, 14(25), pp. 6107–6121. https://doi.org/10.1049/iet-gtd.2020.0531.

Agarwal, R., Schwarzer, M., Castro, P.S., Courville, A. and Bellemare, M.G. (2021) ‘Deep reinforcement learning at the edge of the statistical precipice’, *Advances in Neural Information Processing Systems*, 34. Available at: https://arxiv.org/abs/2108.13264.

Al-Rikabi, I.J., Omara, A.A.M., Abuelnour, M.A., Abuelnuor, A.A.A., Mohamed, A.E. and Ismail, N.A. (2025) ‘Renewable energy in Sudan: Current status and future prospects’, *Engineering Reports*, 7(1), e13116. https://doi.org/10.1002/eng2.13116.

Basheir, R.H. and Abdelrahman, M.S. (2022) ‘The electricity crisis in Sudan: Between quick-fixes and opportunities for a sustainable energy transition’, *Transnational Institute*, 14 October. Policy essay. Available at: https://www.tni.org/en/article/the-electricity-crisis-in-sudan.

Bedi, J. and Toshniwal, D. (2018) ‘Empirical mode decomposition based deep learning for electricity demand forecasting’, *IEEE Access*, 6, pp. 49144–49156. https://doi.org/10.1109/ACCESS.2018.2867681.

Chen, H., Zhuang, J., Zhou, G., Wang, Y., Sun, Z. and Levron, Y. (2023) ‘Emergency load shedding strategy for high renewable energy penetrated power systems based on deep reinforcement learning’, *Energy Reports*, 9, pp. 434–443. https://doi.org/10.1016/j.egyr.2023.03.027.

Efron, B. and Tibshirani, R.J. (1993) *An introduction to the bootstrap*. Chapman and Hall/CRC. https://doi.org/10.1201/9780429246593.

García, J. and Fernández, F. (2015) ‘A comprehensive survey on safe reinforcement learning’, *Journal of Machine Learning Research*, 16(42), pp. 1437–1480. Available at: https://jmlr.org/papers/v16/garcia15a.html.

Glavic, M. (2019) ‘(Deep) reinforcement learning for electric power system control and related problems: A short review and perspectives’, *Annual Reviews in Control*, 48, pp. 22–35. https://doi.org/10.1016/j.arcontrol.2019.09.008.

Heggie, A., Eager, D., McKinnon, K. and van der Weijde, A.H. (2018) ‘Power rationing in a long-term power shortage’, *Energy Policy*, 121, pp. 202–210. https://doi.org/10.1016/j.enpol.2018.06.025.

Huang, Q., Huang, R., Hao, W., Tan, J., Fan, R. and Huang, Z. (2020) ‘Adaptive power system emergency control using deep reinforcement learning’, *IEEE Transactions on Smart Grid*, 11(2), pp. 1171–1182. https://doi.org/10.1109/TSG.2019.2933191.

Huang, S. and Ontañón, S. (2022) ‘A closer look at invalid action masking in policy gradient algorithms’, *The International FLAIRS Conference Proceedings*, 35. https://doi.org/10.32473/flairs.v35i.130584.

Irwin, B.R., Hoxha, K. and Grépin, K.A. (2020) ‘Conceptualising the effect of access to electricity on health in low- and middle-income countries: A systematic review’, *Global Public Health*, 15(3), pp. 452–473. https://doi.org/10.1080/17441692.2019.1695873.

Jain, R., Chiu, D.-M. and Hawe, W.R. (1984) *A quantitative measure of fairness and discrimination for resource allocation in shared computer systems*. DEC Research Report TR-301. Available at: https://arxiv.org/abs/cs/9809099.

Jenkins, K., McCauley, D., Heffron, R., Stephan, H. and Rehner, R. (2016) ‘Energy justice: A conceptual review’, *Energy Research & Social Science*, 11, pp. 174–182. https://doi.org/10.1016/j.erss.2015.10.004.

Ji, Y., Wang, J., Xu, J., Fang, X. and Zhang, H. (2019) ‘Real-time energy management of a microgrid using deep reinforcement learning’, *Energies*, 12(12), 2291. https://doi.org/10.3390/en12122291.

Larik, R.M., Mustafa, M.W. and Aman, M.N. (2019) ‘A critical review of the state-of-art schemes for under voltage load shedding’, *International Transactions on Electrical Energy Systems*, 29(5), e2828. https://doi.org/10.1002/2050-7038.2828.

Natural Earth (n.d.) *1:10m rivers and lake centerlines*. Version 5.0.0. Public-domain geographic dataset; source provenance is retained with the map assets.

Nduhuura, P., Garschagen, M. and Zerga, A. (2021) ‘Impacts of electricity outages in urban households in developing countries: A case of Accra, Ghana’, *Energies*, 14(12), 3676. https://doi.org/10.3390/en14123676.

Nti, I.K., Teimeh, M., Nyarko-Boateng, O. and Adekoya, A.F. (2020) ‘Electricity load forecasting: A systematic review’, *Journal of Electrical Systems and Information Technology*, 7, article 13. https://doi.org/10.1186/s43067-020-00021-8.

Okuta, R., Unno, Y., Nishino, D., Hido, S. and Loomis, C. (2017) ‘CuPy: A NumPy-compatible library for NVIDIA GPU calculations’, *Workshop on Machine Learning Systems at NIPS 2017*.

Oluwasuji, O.I., Malik, O., Zhang, J. and Ramchurn, S.D. (2020) ‘Solving the fair electric load shedding problem in developing countries’, *Autonomous Agents and Multi-Agent Systems*, 34, article 12. https://doi.org/10.1007/s10458-019-09428-8.

Paszke, A., Gross, S., Massa, F. et al. (2019) ‘PyTorch: An imperative style, high-performance deep learning library’, *Advances in Neural Information Processing Systems*, 32.

Poudel, S. and Dubey, A. (2019) ‘Critical load restoration using distributed energy resources for resilient power distribution system’, *IEEE Transactions on Power Systems*, 34(1), pp. 52–63. https://doi.org/10.1109/TPWRS.2018.2860256.

Ramchurn, S.D., Vytelingum, P., Rogers, A. and Jennings, N.R. (2012) ‘Putting the smarts into the smart grid: A grand challenge for artificial intelligence’, *Communications of the ACM*, 55(4), pp. 86–97. https://doi.org/10.1145/2133806.2133825.

Rosales-Asensio, E., Elejalde, J.L., Pulido-Alonso, A. and Colmenar-Santos, A. (2022) ‘Resilience framework, methods, and metrics for the prioritization of critical electrical grid customers’, *Electronics*, 11(14), 2246. https://doi.org/10.3390/electronics11142246.

Sarwar, S., Mokhlis, H., Othman, M., Muhammad, M.A., Laghari, J.A., Mansor, N.N., Mohamad, H. and Pourdaryaei, A. (2020) ‘A mixed integer linear programming based load shedding technique for improving the sustainability of islanded distribution systems’, *Sustainability*, 12(15), 6234. https://doi.org/10.3390/su12156234.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A. and Klimov, O. (2017) *Proximal policy optimization algorithms*. arXiv:1707.06347. https://doi.org/10.48550/arXiv.1707.06347.

United Nations Office for the Coordination of Humanitarian Affairs (n.d.) *Sudan administrative boundaries (COD-AB)*. Revision v03. Humanitarian Data Exchange. Geographic dataset valid from 31 August 2020; CC BY 3.0 IGO. Source provenance is retained with the map assets.

Vázquez-Canteli, J.R. and Nagy, Z. (2019) ‘Reinforcement learning for demand response: A review of algorithms and modeling techniques’, *Applied Energy*, 235, pp. 1072–1089. https://doi.org/10.1016/j.apenergy.2018.11.002.

Wang, C., Yu, H., Chai, L., Liu, H. and Zhu, B. (2021) ‘Emergency load shedding strategy for microgrids based on dueling deep Q-learning’, *IEEE Access*, 9, pp. 19707–19715. https://doi.org/10.1109/ACCESS.2021.3055401.

World Bank (2025) *Accelerating sustainable and clean energy and digital access transformation in Sudan (P507788)*. Project Information Document, Appraisal Stage, Report PIDIA01286, 1 May. Available at: https://documents1.worldbank.org/curated/en/099050125112514628/pdf/P507788-6a843250-44f5-47da-9f07-c6b1269a091f.pdf.




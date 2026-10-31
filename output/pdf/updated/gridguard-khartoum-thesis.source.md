# Declaration

We declare that this project, entitled “AI-Based Smart Load Shedding and Energy
Management System for Developing Countries”, is our original work, carried out under
the supervision of Dr. Abubakr Siraj at the Faculty of Mathematical Sciences and
Informatics, University of Khartoum. This work has not previously been submitted for
another degree or professional qualification, and sources consulted are acknowledged.

Murtada Asim Ibrahim Bakhit (18-317) — Signature: ____________________

Suliman Elawad Suliman Babiker (18-310) — Signature: ____________________

Abdalhay Abass Mobarak Abass (15-306) — Signature: ____________________

Date: ____________________

# Dedication

To our families, whose patience and support made this work possible.

To the people of Sudan, whose resilience in the face of energy hardship inspired this study.

# Acknowledgements

We thank our supervisor, Dr. Abubakr Siraj, for guidance and encouragement throughout this project.

# Abstract

Electricity shortages create a distribution problem as well as a shortage of generation. Allocating the same supply differently can change interruption durations, protect essential services and distribute unequal burdens between communities. This study develops and evaluates GridGuard Khartoum, an AI-assisted load-shedding and energy-management system combining software simulation, history-only demand prediction, priority rules, constrained optimization, reinforcement learning and an offline application. Khartoum State provides the geographic setting; all demand and supply traces are synthetic.

The simulation represents seven localities, five sectors and 24 hourly decisions under Mild, Moderate and Severe shortages. Hospital, Water and Bakery demand receives explicit priority. The main experiment compares Fixed Rotation, Constrained Optimization and Proximal Policy Optimization (PPO) on identical fresh scenarios. PPO and Constrained Optimization share 128 Critical Only/Full Service assignments and the same supply-feasibility rule. PPO uses 46 features available at decision time and Fairness Rule demonstrations before reinforcement learning. Five independent runs each complete 3,000 iterations, followed by validation selection and evaluation on 960 fresh matched conditions. Critical interruption, service fairness, utilization and direct duration disparities are reported separately from the training score.

Across 960 fresh matched conditions, PPO averaged 0.00 critical-interruption locality-hours, 97.02% service fairness and 97.02% utilization. Fixed Rotation recorded 102.05, 87.76% and 71.23%, respectively; Constrained Optimization recorded 0.00, 94.78% and 97.41%. PPO's daily-reward difference from the optimizer was +0.0499 [+0.0406, +0.0577] (97.5% interval). Utilization falls by 0.40 percentage points and switching increases, so the full declared optimizer-comparison criterion is not met. A post-hoc matched audit finds improved allocation over a critical-aware rotating rule. Against the Fairness Rule teacher, PPO gains 0.0102 daily reward but loses 0.308 utilization percentage points and 16.68 served MWh per day; the pooled gain is concentrated in Severe shortage. The teacher achieves about 80% of PPO's reward advantage over the optimizer with a smaller utilization loss. The infeasible-supply test verifies the common fallback rather than distinguishing protected controllers. A later descriptive application of the original numerical service margins to Fairness Rule versus the optimizer passes those checks; this was not an original confirmatory comparison. A separate fresh 960-condition audit of the same five runs finds that fine-tuning and validation selection add 0.1801 daily reward over imitation alone, while serving 77.20 fewer MWh per day. The imitation-only actors start 0.1704 below the teacher, so most of the stage gain closes that gap; selected PPO ends only 0.0097 above it. Higher mean Jain fairness does not ensure higher final service to the worst-served locality.

A separate forecasting experiment finds lower error for Ridge AR(3) than for persistence and a three-hour trailing mean on unseen synthetic targets. However, a later hour-of-day mean baseline, fitted only on the same training traces, has lower error than Ridge AR(3) on all four measures. The fixed daily profiles therefore favor a simple time-based predictor; an advantage from autoregression is not established against this stronger baseline. Forecasts are not controller inputs. The contribution is an inspectable decision-support system and outcome-specific evidence, not a real-grid controller or a general proof that learning surpasses optimization.

**Keywords:** load shedding; energy management; PPO; constrained optimization; critical services; fairness; prediction; Khartoum.

# الملخص

تتناول هذه الدراسة توزيع الكهرباء أثناء نقص الإمداد من خلال تطوير نظام «جريد جارد الخرطوم» لدعم القرار. يجمع النظام المحاكاة والتنبؤ بالطلب وقواعد أولوية الخدمات الأساسية والتحسين المقيد والتعلم المعزز، مع تطبيق مكتبي يعمل دون اتصال بالإنترنت. تمثل ولاية الخرطوم الإطار الجغرافي للدراسة، لكن بيانات الطلب والإمداد جميعها اصطناعية وليست قياسات للشبكة الفعلية.

تمثل المحاكاة سبع محليات وخمسة قطاعات وقرارات ساعية لمدة أربع وعشرين ساعة، تحت ثلاثة مستويات للنقص. تقارن الدراسة خوارزمية التحسين القريب للسياسة «PPO» بالتناوب الثابت والتحسين المقيد على الظروف نفسها. يشترك PPO والتحسين المقيد في مئة وثمانية وعشرين تخصيصا ممكنا وقيد لحماية طلب المستشفيات والمياه والمخابز عندما يسمح الإمداد. تستخدم السياسة ستا وأربعين سمة متاحة عند اتخاذ القرار، ويبدأ التدريب بتقليد قاعدة للإنصاف ثم بالتعلم المعزز. أجريت خمس عمليات تدريب مستقلة، لكل منها ثلاثة آلاف تكرار، واختيرت السياسات باستخدام مجموعة تحقق منفصلة قبل اختبار تسعمئة وستين حالة متطابقة.

بلغ متوسط PPO صفرا من ساعات انقطاع الخدمات الأساسية على مستوى المحليات، و97.02% للإنصاف و97.02% لاستغلال الإمداد. حقق النظام المقيد معيار المقارنة المعلن مقابل التناوب الثابت، لكن حماية الخدمات الأساسية تعود إلى القيد الصريح وليست دليلا منفردا على فائدة التعلم. مقارنة بالتحسين المقيد الذي يعظم مكافأة الساعة الحالية، زاد PPO متوسط الإنصاف بمقدار 2.24 نقطة مئوية والمكافأة اليومية بمقدار 0.0499، بفاصل ثقة معدل بنسبة 97.5% من 0.0406 إلى 0.0577. في المقابل انخفض استغلال الإمداد بمقدار 0.40 نقطة مئوية وزاد تبديل مستويات الخدمة، ولذلك لم يتحقق معيار التحسن الكامل المشروط بمؤشرات الخدمة. هذه النتائج لا تثبت تفوق التعلم على التحسين لجميع الأهداف أو على تخطيط أمثل لليوم الكامل.

أضاف تحليل لاحق مقارنات على حالات الاختبار نفسها. تحسن تخصيص PPO مقارنة بقاعدة تناوب تحمي الطلب الأساسي. وتحقق قاعدة الإنصاف وحدها نحو 80% من زيادة المكافأة التي حققها PPO مقارنة بالتحسين المقيد، مع انخفاض أقل في استغلال الإمداد. كما استوفت القاعدة الهوامش العددية الأصلية للمكافأة والخدمة في مقارنة لاحقة بأثر رجعي، دون اعتبارها اختبارا تأكيديا معلنا مسبقا. وأمام قاعدة الإنصاف المستخدمة لتوليد أمثلة التدريب، زادت المكافأة اليومية بمقدار 0.0102، لكن انخفض استغلال الإمداد بمقدار 0.308 نقطة مئوية والطاقة المقدمة بمقدار 16.68 ميغاواط ساعة يوميا. تركزت زيادة المكافأة في النقص الحاد؛ ولا يمثل التحليل اختبارا تأكيديا مستقلا جديدا. وتتحقق اختبارات نقص الإمداد الأساسي من تنفيذ الإجراء الاحتياطي، ولا تميز بين قرارات الطرق المقيدة لأن الإجراء الاحتياطي واحد.

خفض نموذج الانحدار الذاتي المنتظم «Ridge AR(3)» خطأ التنبؤ بالطلب الاصطناعي مقارنة باستمرار آخر قيمة ومتوسط الساعات الثلاث السابقة. لكن خط أساس لاحقا يعتمد متوسط الطلب حسب ساعة اليوم، ويُقدّر من بيانات التدريب نفسها فقط، حقق خطأ أقل من Ridge AR(3) في المقاييس الأربعة. لذلك لم يثبت تفوق الانحدار الذاتي على هذا البديل الأبسط في مولد ذي أنماط يومية ثابتة. لا تدخل هذه التنبؤات في سياسة التحكم. تتمثل مساهمة المشروع في نظام قابل للفحص ونتائج محددة النطاق توضح المفاضلات، وليس في إثبات الجاهزية التشغيلية لشبكة كهربائية حقيقية.

وفي تدقيق منفصل على تسعمئة وستين حالة جديدة، زادت مرحلة التعلم المعزز واختيار السياسة بالتحقق المكافأة اليومية بمقدار 0.1801 مقارنة بتهيئة التقليد الخاصة بكل تشغيل، لكنها قدمت طاقة أقل بمقدار 77.20 ميغاواط ساعة يوميا. حسابيا، يعوض معظم هذا الفرق نقص مكافأة نموذج التقليد البالغ 0.1704 مقارنة بقاعدة الإنصاف؛ وتنتهي السياسة المختارة أعلى من القاعدة بمقدار 0.0097 فقط. ولا يضمن ارتفاع متوسط مؤشر جاين للإنصاف تحسن الخدمة النهائية للمحلية الأقل حصولا عليها. تقيس هذه المقارنة المرحلة كاملة، ولا تعزل أثر خوارزمية PPO عن الاختيار والحساب الإضافي.

**الكلمات المفتاحية:** فصل الأحمال؛ إدارة الطاقة؛ PPO؛ التحسين المقيد؛ الخدمات الأساسية؛ الإنصاف؛ التنبؤ؛ الخرطوم.

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

These questions evaluate AI without requiring a predetermined advantage over other methods. A result favoring a transparent optimizer is useful. An improvement on one outcome is also useful when its scope and costs are explicit. Direct duration measures assess blackout inequality alongside Jain fairness; the two are not treated as interchangeable.

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

The Fixed Rotation comparison evaluates the complete system. It cannot assign every improvement to learning because Fixed Rotation lacks Critical Only actions and a critical-feasibility mask. PPO and Constrained Optimization, by contrast, share the same feasible assignments and information available at decision time. That comparison is stronger for assessing learned scheduling, while still not establishing superiority over a full-day optimum or every optimization method. Section 5.7 adds a post-hoc matched comparison with the Fairness Rule teacher and Critical-aware Fixed Rotation; it does not reclassify that follow-up as part of the original confirmatory design.
## 1.4 Key terms and thesis organisation

A locality receives one of three service tiers: Off requests no power, Critical Only
requests Hospital, Water and Bakery demand, and Full Service requests all five sectors.
A locality-hour is one locality affected for one hour; seven localities over 24 hours
give 168 locality-hours. Critical interruption means less than 99% of critical demand
is served. Complete critical outage means no critical power is delivered. Jain service
fairness describes equality of cumulative served-to-demanded ratios; it is neither the
fraction of demand served nor a direct measure of interruption duration.

Constrained Optimization denotes the exact current-hour comparator. Critical-aware
Fixed Rotation adds critical protection to a rotating allocation order. Fairness Rule
is the deterministic teacher used for imitation before PPO. Nonanticipative observations
use only information available at decision time.

Chapter 2 reviews related work. Chapter 3 defines the simulator, controllers, reward and
training procedure. Chapter 4 describes the application. Chapter 5 presents the design
and results, separating the original comparisons from subsequent audits. Chapter 6
interprets the tradeoffs and limitations, and Chapter 7 answers the five research
questions. Appendix A provides the exact generator and reproduction details; Appendix B
identifies the source code and academic package.

# 2 Related work



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

Section 5.10 additionally compares each selected policy with its own imitation-only
initialization on new matched conditions, separating the measured learning-stage change
from the broader comparison with its teacher.

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

National statistics in those documents are tied to their dates and source definitions. They do not calibrate the experiment's 700 MW nominal demand scale, sector shares or locality weights. The Moderate and Severe shortage windows are controlled scenarios, not reproductions of a documented Khartoum day. Fresh synthetic traces permit repeatable comparison while remaining insufficient for operational calibration.

Familiar locality names and administrative geometry make the geographic setting understandable. They do not establish electricity-network topology. This separation supports an honest Sudan-inspired study without turning geographic context into an unsupported claim of physical fidelity.

## 2.8 Fairness, equity and evaluation

Jenkins et al. (2016) distinguish distributional, recognition and procedural dimensions of energy justice. This explains why total delivered energy is incomplete: who receives service, whose needs count and how decisions are governed all matter. GridGuard measures a narrow distributional component and names critical services explicitly. It does not implement participatory governance or claim to resolve energy justice.

Oluwasuji et al. (2020) provide a close comparison for fair allocation under recurring shortage. They formulate household connections as mixed-integer programs with utilitarian, egalitarian and envy-freeness objectives and constraints, using consumption and comfort to represent heterogeneous needs. Their day-ahead schedules cover 24 hours, and their developing-country demand data are adapted from measured US household consumption. GridGuard instead makes hourly locality decisions from current observations and accumulated history, using illustrative sector profiles. It measures Jain fairness (Jain et al., 1984) of cumulative demand-normalized service and direct duration disparities. These quantities do not reproduce household utility or envy-freeness; a fairness reward term is not a hard equality guarantee.

Heggie et al. (2018) examine the efficiency cost of regional proportional-allocation targets in a Nigerian power-system model with transmission constraints and varying generation. They compare fairness over shorter and longer allocation periods. This is a useful precedent for interpreting fairness alongside delivered energy. GridGuard omits that electrical network and measures equality of locality service ratios rather than deviation from prescribed regional shares. Its tradeoff between fairness and utilization addresses a related question in a simpler model; the numerical results are not directly comparable.

Agarwal et al. (2021) explain how uncertainty and limited training runs affect reinforcement-learning conclusions. This motivates reporting all five independently trained policies rather than selecting the best final-test actor. Bootstrap reasoning (Efron and Tibshirani, 1993) supports retaining controller pairing and the dependence between shortage variants of the same underlying scenario.

## 2.9 Position of this study

The preceding eight themes establish that priority-aware optimization and fair rationing
already have substantial precedents. This study asks a narrower question: under the same
current information and critical-service constraints, what does imitation followed by
PPO add to transparent allocation methods, and what service costs accompany any gain?
The contribution is the implemented comparison and its evidence within a Sudan-inspired
simulation, not a new fairness definition or a claim to have introduced fair load shedding.

The design follows from that question. Constrained Optimization and Fairness Rule provide
stronger references than an unprotected fixed schedule. Critical service is enforced
through admissible actions; fairness, duration and energy remain separate outcomes.
History-only forecasting is evaluated independently because its predictions do not enter
the controllers. The offline application makes the assumptions and comparisons inspectable.
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

Figure 1 connects the common controller inputs, shared dispatch and outcome calculations.
The forecasting branch is evaluated separately and supplies no future information to PPO.

[[FIGURE:workflow]]

## 3.2 Synthetic demand

The nominal demand scale is 700 MW. It is divided among localities using the weights in Table 1. This is a normalization parameter, not a measured Khartoum demand or the simultaneous system peak: the sector profiles reach their maxima at different hours. Without random perturbations, the resulting system demand reaches approximately 579.99 MW at hour 19. Within each locality, sector coefficients scale these normalized hourly profiles. Appendix A.3 specifies the generator exactly.

**Table 1. Independent illustrative locality and sector parameter groups.**

| Parameter group | Values |
| --- | --- |
| Locality demand weights | Khartoum 1.00; Omdurman 0.95; Bahri 0.85; East Nile 0.80; Jebel Awlia 0.70; Karari 0.75; Ombada 0.90 |
| Sector base-peak coefficients | Hospital 0.08; Water 0.07; Bakery 0.05; Residential 0.55; Commercial 0.25 |

Every locality contains all five sectors. The two parameter groups are independent; their entries do not pair a locality with a particular sector.

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

Constrained Optimization enumerates all 128 assignments defined above, filters by the same feasibility condition, applies shared dispatch and scores the canonical current-hour reward. It receives current demand, supply, cumulative demand and served energy, and previous tiers. Exact score ties select the lowest candidate index. It receives no future demand or supply and evaluates every candidate rather than sampling a subset.

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
and cosine, current supply relative to demand, and supply relative to the fixed nominal system
demand scale. Each locality contributes normalized current demand, its current critical-demand
share, cumulative service ratio and previous service tier. These components form the
original 32-value state representation. Seven cumulative-demand and seven cumulative-
served-energy features complete the 46-value input.

The additional magnitudes matter because the same cumulative ratio can arise from
different amounts of past demand. They are normalized using 24 times the fixed nominal demand scale of
the corresponding locality. Their normalization does not use future episode demand.
All features come from current observations and accumulated history; the policy receives
no future supply, future demand, target-hour value or final evaluation outcome.

With $P_i=700w_i/\sum_j w_j$ MW as the fixed locality demand scale and
$d_{i,t}=\sum_k d_{i,k,t}$, the four global inputs are
$\sin(2\pi t/24)$, $\cos(2\pi t/24)$, $S_t/\max(\sum_i d_{i,t},\varepsilon)$ and $S_t/700$.
Each locality contributes $d_{i,t}/P_i$, its critical-demand fraction,
$E_{i,t}^{-}/\max(D_{i,t}^{-},\varepsilon)$ and $a_{i,t-1}/2$.
Previous tiers are encoded as Off = 0, Critical Only = 1 and Full Service = 2;
the binary $x_i$ in Section 3.6 instead indicates an upgrade from Critical Only to Full Service.
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

Generalized advantage estimation follows Schulman et al. (2016). For the complete 24-step episode, let $v_t=V_{\mathrm{old}}(s_t)$ and set $v_{24}=0$. The raw advantages are calculated backwards:

$$
\delta_t=r_t+\gamma v_{t+1}-v_t,\qquad
A_{24}^{\mathrm{raw}}=0,\qquad
A_t^{\mathrm{raw}}=\delta_t+\gamma\lambda A_{t+1}^{\mathrm{raw}}.
$$

Here $\gamma=1$ matches the undiscounted daily score and $\lambda=0.95$ controls the estimator's bias-variance tradeoff. The fixed critic target is $\widehat R_t=A_t^{\mathrm{raw}}+v_t$. The actor uses the raw advantages centered and divided by their rollout population standard deviation plus $10^{-8}$, over all $24\times256=6,144$ transitions. Neither the target nor the recursion crosses an episode boundary. The value target is computed before advantage normalization.

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
The cloning stage uses Adam with a constant learning rate of $3\times10^{-4}$,
$\epsilon=10^{-5}$ and maximum gradient norm 0.5. A fresh PPO optimizer starts after
this initialization. The retained behavior-cloning
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
| Adam learning rate | 0.0003 linearly to 0 |
| Discount factor | 1.0 |
| GAE parameter | 0.95 |
| PPO clipping coefficient | 0.2 |
| Maximum gradient norm | 0.5 |
| Entropy coefficient | 0.01 to 0.001 |

For zero-based PPO update index $k=0,\ldots,2999$, the learning rate and entropy coefficient are

$$
\eta_k=0.0003\left(1-\frac{k}{2999}\right),\qquad
c_H(k)=0.01+(0.001-0.01)\frac{k}{2999}.
$$

Thus the last scheduled update has zero learning rate and entropy coefficient 0.001. The plots label this as completed iteration 3,000. Adam uses a numerical epsilon of $10^{-5}$.

Each rollout uses 85 episodes per shortage level plus one rotating extra episode, giving
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

A later post-hoc comparison adds an hour-of-day mean for each locality-sector series.
It estimates each hourly value from the same 768 training traces, without using test
targets or the generator's analytic profile:

$$
\widehat d_{i,k,h}=\frac{1}{768}\sum_{n=1}^{768}d_{i,k,h}^{(n)}.
$$

The target hour is known at forecast time. Evaluation retains the original held-out
hours 3–23 and the same error aggregation, so the additional baseline tests whether
recent demand history improves on a simple training-fitted daily template.

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

Startup checks the required model and evidence resources. When a required resource
cannot be verified, the software reports the problem instead of presenting results
from an unidentified model or a mismatched study.

## 4.2 The desktop workflow

The native application organizes the demonstration into Overview, Live Scenario and
Results. The Overview introduces the localities, scope and available shortage levels.
Live Scenario allows the presenter to choose a shortage level and seed, run a matched
episode and inspect its hourly allocation. It uses the validation-selected PPO policy
from training seed 8111 at iteration 2500, alongside Fixed Rotation and Constrained
Optimization. Results retains the frozen five-run study; one live policy and scenario
do not replace its matched-scenario averages.

The map, locality selector and time slider support
inspection of one locality and hour without hiding the whole-day comparison. Textual
service labels accompany color, and keyboard navigation provides access to the main
pages and locality choices. The interface remains an explanatory tool: a polished map
does not add physical realism to the underlying model.

Figure 2 shows locality context and navigation in the application overview. It is a
software demonstration, not evidence of operation on the Khartoum electrical network.

[[FIGURE:overview]]

Figure 3 shows the selected locality, service tier and hourly PPO replay. The desktop
demonstrates one validation-selected policy; the main study averages five independent runs.
Its display-only next-hour estimate is a trailing mean of the selected locality's
observed demand, using up to the latest three hours. It is separate from the evaluated
Ridge AR(3) model and does not influence controller decisions.

[[FIGURE:live]]

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
minimum presentation layout. Integration tests exercise the scientific computation path.

The fifth research question is addressed through interface design and technical checks.
No participant study measured comprehension or usability.

Independent replay provides a further check between selected model files and recorded
results. It regenerates scenarios, executes saved controllers and compares actions and
outcomes. The final verification status for the current experiment is summarized in
Section 5.11; detailed hashes and reproduction references belong in Appendix A rather
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

Critical interruption occurs when positive critical demand receives less than 99% service during an hour. Summing across seven localities and 24 hours gives zero to 168 locality-hours. One locality-hour means one affected locality for one hour, not a system-wide clock hour. The retained evidence field named `critical_outage_locality_hours` uses this same 99% threshold; it is not a count of complete loss of critical service. The post-hoc audit separately counts complete critical outage when critical demand is positive and delivered critical power is exactly zero. Both definitions treat demand above $10^{-8}\,\mathrm{MW}$ as positive.

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
| Jain fairness | Percent | Higher equality of cumulative service ratios |
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

The internal record dates the protocol and source freeze to 23 October 2026 and records
training and validation selection before final-test generation. However, the earliest
available Git commit containing the protocol is dated 31 October. The retained hashes
identify the material used; they do not independently prove that the criteria were set
before outcomes were seen. Appendix A.4 gives the identities and dates. This is an
internally recorded design, not external preregistration or independent custody.
The uncertainty describes this generator and five training runs; 960 conditions are not
960 independent policies. The later audit in Section 5.7 remains explicitly post-hoc.

## 5.5 Main controller results

Table 6 compares all three procedures on common conditions. PPO values weight the five runs equally. Critical interruption, fairness and utilization lead the interpretation; duration, energy, switching and reward explain the behavior behind those outcomes.

**Table 6. Final-test controller outcomes on 960 matched conditions; PPO averages five selected runs.**

| Outcome | Fixed Rotation | Constrained Optimization | PPO |
| --- | --- | --- | --- |
| Critical interruption (locality-hours) | 102.05 | 0.00 | 0.00 |
| Jain fairness (%) | 87.76 | 94.78 | 97.02 |
| Supply utilization (%) | 71.23 | 97.41 | 97.02 |
| Daily reward | 13.9688 | 23.4468 | 23.4967 |
| Switching rate (%) | 25.00 | 20.55 | 27.76 |
| Critical service (%) | 42.62 | 100.00 | 100.00 |
| Zero-service blackout (locality-hours) | 96.00 | 0.00 | 0.00 |
| Power-balance violation rate | 0.0000 | 0.0000 | 0.0000 |

Table 7 separates duration burdens and disparities from service-allocation fairness.

**Table 7. Final-test interruption durations and disparities, averaged after calculation within each episode.**

| Duration outcome | Fixed Rotation | Constrained Optimization | PPO |
| --- | --- | --- | --- |
| Blackout duration range (hours) | 3.000 | 0.000 | 0.000 |
| Blackout mean pairwise gap (hours) | 1.714 | 0.000 | 0.000 |
| Longest blackout (hours) | 6.000 | 0.000 | 0.000 |
| Critical-interruption duration range (hours) | 3.667 | 0.000 | 0.000 |
| Critical-interruption mean pairwise gap (hours) | 1.903 | 0.000 | 0.000 |
| Longest critical interruption (hours) | 8.000 | 0.000 | 0.000 |

Ranges and pairwise gaps are calculated within each 24-hour scenario before averaging. A zero duration disparity under shared critical protection is not evidence of equal Full Service. Mean available energy was 6709.05 MWh per condition. Mean served energy was 4551.33 MWh for Fixed Rotation, 6547.90 MWh for Constrained Optimization and 6528.02 MWh for PPO. These energy totals differ from the reported mean of hourly utilization ratios.

Figure 4 displays the three main service outcomes and daily reward in four panels. Critical
protection and distribution quality are not compressed into one visual score.

[[FIGURE:outcomes]]

### PPO - Fixed Rotation

Table 8 reports matched effects in the order shown above; the primary intervals are adjusted for the two-comparison family.

**Table 8. Final-test paired effects: PPO - Fixed Rotation; adjusted 97.5% intervals.**

| Outcome | Mean difference | 97.5% lower | 97.5% upper |
| --- | --- | --- | --- |
| Critical interruption (locality-hours) | -102.0500 | -102.0812 | -102.0250 |
| Jain fairness (pp) | +9.2600 | +9.1423 | +9.3712 |
| Supply utilization (pp) | +25.7815 | +25.7017 | +25.8594 |
| Daily reward | +9.5279 | +9.5184 | +9.5357 |
| Switching rate (pp) | +2.7639 | +2.0601 | +3.6639 |

### PPO - Constrained Optimization

Table 9 reports matched effects in the order shown above; the primary intervals are adjusted for the two-comparison family.

**Table 9. Final-test paired effects: PPO - Constrained Optimization; adjusted 97.5% intervals.**

| Outcome | Mean difference | 97.5% lower | 97.5% upper |
| --- | --- | --- | --- |
| Critical interruption (locality-hours) | +0.0000 | +0.0000 | +0.0000 |
| Jain fairness (pp) | +2.2368 | +2.0955 | +2.3731 |
| Supply utilization (pp) | -0.3989 | -0.4642 | -0.3408 |
| Daily reward | +0.0499 | +0.0406 | +0.0577 |
| Switching rate (pp) | +7.2183 | +6.5310 | +8.1124 |

Figure 5 plots PPO minus Constrained Optimization effects with adjusted 97.5% intervals.
The two primary comparisons both appear in Tables 8 and 9; this figure shows only the
optimizer comparison and does not present validation learning curves.

[[FIGURE:intervals]]

The complete constrained PPO system met the declared improvement criterion against Fixed Rotation. This comparison includes differences in action choices and critical-service constraints, so it does not isolate the neural policy as the sole cause of the observed difference.

Against Constrained Optimization, the mean daily-reward difference was +0.0499 [+0.0406, +0.0577]. The full comparison criterion did not pass. Both controllers use the same critical-feasible action choices. The optimizer maximizes the current-hour objective exactly; PPO can trade a current decision against its consequences for later cumulative fairness and switching. Any whole-day advantage therefore concerns this nonanticipative myopic comparator and the specified reward, rather than a superior solution to the same exact optimization problem.

The corresponding fairness effect was +2.237 [+2.096, +2.373] percentage points, and utilization changed by -0.399 [-0.464, -0.341] percentage points. These outcomes distinguish service and efficiency effects from the weighted reward score.

Switching increased by +7.218 [+6.531, +8.112] percentage points, an unfavorable change. The fairness contribution outweighed both the utilization loss and extra switching penalty in the specified reward. Thus the trained procedure achieves fairer service allocation and a higher daily objective, with an efficiency and switching tradeoff.

In practical scale, the 0.0499 daily-score gain is 0.213% of the optimizer's 23.4468 score, or 0.208% of the approximate ideal daily bound of 24. Using the reward equation, the measured mean difference decomposes into +0.11930 from cumulative fairness, -0.02127 from utilization and -0.04812 from switching. Critical service and oversubscription contribute zero difference because both methods achieve full critical service without oversubscription on these conditions. These are arithmetic components of the chosen score, not independently identified causal effects.

The corresponding direct energy result is -19.88 MWh served per synthetic day: 6528.02 for PPO versus 6547.90 for the optimizer, from the same 6709.05 MWh available. PPO therefore leaves about 19.88 additional MWh unmet on average. This energy effect is calculated from the saved energy metrics; it is not a conversion of reward or of mean hourly utilization into MWh. The experiment does not measure the social value of the fairness gain, so it cannot establish that the score gain justifies the energy and switching costs in practice.

The optimizer reward difference also exceeded the earlier +0.01 daily-reward reference throughout its adjusted interval. That reference is a research convention; this simulation does not establish the operational value of a small score difference.

The unmet optimizer-comparison conditions were: severity service guardrails; utilization noninferiority.

The shortage-level utilization guardrail fails under Moderate (-0.49 percentage points) and Severe (-0.73 percentage points), alongside the overall utilization interval.

The post-hoc audit in Section 5.7 qualifies the learning interpretation. Fairness Rule obtains +0.0397 daily reward over Constrained Optimization, approximately 80% of PPO's +0.0499 advantage. Its utilization difference is -0.09 percentage points. Thus most of the observed reward advantage is also available from a transparent rule. The ratio describes these mean differences; it is not a causal share attributed to an individual learning component.

All three shortage levels are reported in Section 5.6. Severity-specific intervals are exploratory, so a favorable subgroup does not replace the overall declared result. Zero blackout-duration disparities among critical-protected controllers also do not imply equal access to Full Service.

## 5.6 Shortage levels and learning progression

Mild conditions allow more Full Service combinations. The Severe shortage window restricts the number fitting within supply. Table 10 reports each level so a favorable overall mean cannot conceal a weaker condition.

**Table 10. Final-test outcomes by shortage; 320 matched conditions per level and equal PPO run weights.**

| Shortage | Controller | Critical interruption (locality-hours) | Fairness (%) | Utilization (%) | Reward |
| --- | --- | --- | --- | --- | --- |
| Mild | Fixed Rotation | 96.00 | 87.74 | 54.28 | 13.1911 |
| Mild | Constrained Optimization | 0.00 | 96.41 | 98.09 | 23.5881 |
| Mild | PPO | 0.00 | 97.99 | 98.12 | 23.6098 |
| Moderate | Fixed Rotation | 96.00 | 87.74 | 68.69 | 13.9600 |
| Moderate | Constrained Optimization | 0.00 | 95.00 | 98.02 | 23.4626 |
| Moderate | PPO | 0.00 | 97.21 | 97.52 | 23.5002 |
| Severe | Fixed Rotation | 114.15 | 87.80 | 90.73 | 14.7554 |
| Severe | Constrained Optimization | 0.00 | 92.94 | 96.13 | 23.2897 |
| Severe | PPO | 0.00 | 95.86 | 95.40 | 23.3801 |

Figure 6 shows each run's mean daily validation reward, Jain fairness and utilization,
with selected checkpoints and the same-run iteration-600 reference. The largest reward
and fairness gains occur early, followed by smaller improvements, while utilization
declines. Thus the learning curves already show the tradeoff between reward and service examined
on the final test. Differences between runs remain, supporting equal-run reporting
rather than selection of one favorable test result. These are validation trajectories,
not final-test performance at every training iteration.

[[FIGURE:training]]

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

Figure 7 displays these unsmoothed training diagnostics, including the mean and
sample-standard-deviation band. It is a training plot, distinct from the validation
curves in Figure 6 and the final-test intervals in Figure 5.

[[FIGURE:diagnostics]]

The validation-variance graph shows daily reward, Jain fairness and utilization over
all 31 checkpoint evaluations, including the
behavior-cloning starting point. It measures variability across independently trained
policies on the same validation conditions. Fairness and utilization are converted to
percentage points before calculating their variances; their axes therefore use squared
percentage points. Figure 8 displays those variances. These descriptive curves do not
replace the paired final-test intervals.

Iteration zero follows ten epochs of behavior cloning; it is not an untrained common
initialization. The five retained actors have distinct weights. Their initial sample
variances are small but nonzero: $9.11\times10^{-6}$ in daily reward,
$2.27\times10^{-3}$ in Jain fairness (squared percentage points), and
$7.96\times10^{-5}$ in utilization (squared percentage points). The shared teacher and
common validation conditions provide context for their similar performance, but the
curves alone do not identify the cause of that similarity.

[[FIGURE:variance]]

The curves show diminishing improvements in the specified objective. Across five runs, mean validation reward increased from 23.3196 after imitation to 23.4697 at iteration 600 and 23.4989 at iteration 3,000; 83.7% of that observed increase occurred within the first 600 iterations. The final 500 iterations added only 0.00144. Over the complete schedule, validation fairness rose from 93.12% to 97.00%, while utilization fell from 98.24% to 97.04%. These are common-iteration validation averages, not the selected actors' final-test results.

Between-run reward variance peaked at $7.31\times10^{-4}$ at iteration 100 and ended at $7.99\times10^{-5}$, so the later plateau still contains differences between runs. The mean half-squared critic error fell from 0.00156 at the first recorded update to 0.000207 at iteration 3,000; entropy fell from 1.881 to 0.295 nats. Approximate KL and clipping fraction approach zero at the end partly because the scheduled learning rate reaches zero. These diagnostics support a description of the training trajectory, not proof of convergence to an optimal policy or removal of the observed utilization tradeoff.

Table 11 gives final-test rewards for validation-selected checkpoints, not the validation rewards used to choose them.

**Table 11. Same-run final-test daily reward: iteration 600 versus the validation-selected actor.**

| Training seed | Selected iteration | Iteration-600 test reward | Selected-actor test reward | Difference |
| --- | --- | --- | --- | --- |
| 1729 | 2900 | 23.4497 | 23.4835 | +0.0338 |
| 2718 | 3000 | 23.4654 | 23.4913 | +0.0258 |
| 3141 | 2900 | 23.4699 | 23.5034 | +0.0335 |
| 5772 | 3000 | 23.4781 | 23.5014 | +0.0233 |
| 8111 | 2500 | 23.4732 | 23.5040 | +0.0309 |

Table 12 gives the paired uncertainty for continuing the same training schedule.

**Table 12. Final-test effects of selected actors minus same-run iteration-600 actors; descriptive 95% intervals.**

| Across-run outcome | Mean difference and 95% interval |
| --- | --- |
| Jain fairness (pp) | +0.5895 [+0.5066, +0.6795] |
| Supply utilization (pp) | -0.2417 [-0.3165, -0.1908] |
| Daily reward | +0.0295 [+0.0255, +0.0332] |
| Switching rate (pp) | -1.6363 [-2.0580, -1.2294] |

The full declared additional-training criterion did not pass. A larger budget alone therefore does not establish improvement on every declared requirement. Unmet conditions: severity service guardrails; utilization noninferiority. Daily reward changed by +0.0295 [+0.0255, +0.0332], while utilization changed by -0.242 [-0.317, -0.191] percentage points. The mean utilization loss is therefore about 0.24 percentage points, and its interval breaches the allowed 0.2-point loss. Shortage-level utilization changes are Mild -0.189 points, Moderate -0.204 points, Severe -0.332 points; Moderate and Severe breach the mean-loss guardrail of 0.2 points.

## 5.7 Post-hoc matched baseline and feasibility audit

To test dependence on the comparator, a post-hoc audit adds Fairness Rule and
Critical-aware Fixed Rotation. It reuses the original 960 final-test conditions and
the five frozen selected PPO actors; no actor is retrained or selected using its results.
This audit is post-hoc, not a fresh confirmatory holdout, and it does not change the
original criteria in Section 5.4. Its purpose is to expose comparator dependence and
separate critical protection from discretionary allocation.

Fairness Rule retains its teacher setting: among feasible actions within 0.005 of the highest immediate reward, it maximizes the minimum projected cumulative service ratio, then resolves ties by reward and candidate index. Critical-aware Fixed Rotation first requests the critical floor everywhere, then tries Full Service upgrades in fixed rotating locality order, retaining each upgrade only when it fits supply. The order advances by three localities every three hours. It uses no learned model and does not optimize reward. This comparator shares the critical floor, not the exact discretionary decisions of the original Off/Full Service schedule.

The follow-up evaluates nine methods: four deterministic rules/optimizers and the five frozen PPO actors, giving 8,640 nominal trajectories. It regenerates seeds 122201–122210 with 32 base draws and three coupled shortage levels. Input hashes, action hashes and per-condition metrics reproduce the seven unchanged methods from the original experiment exactly. A separate frozen audit recipe precedes this execution; it is still post-hoc relative to the original results. No new primary pass/fail claim is registered.

Table 13 compares the five procedures. All four critical-protected procedures have zero critical interruption, zero complete critical outage and zero power violations in every nominal condition. Critical-aware Fixed Rotation alone removes the original schedule's critical interruptions, showing that protection does not require learning. Differences in the remaining service outcomes concern how noncritical supply is allocated.

**Table 13. Post-hoc matched baseline audit on the original 960 conditions; PPO is the equal mean of five frozen actors.**

| Procedure | Reward | Fairness (%) | Utilization (%) | Served (MWh) |
| --- | --- | --- | --- | --- |
| Fixed Rotation | 13.9688 | 87.76 | 71.23 | 4551.33 |
| Critical-aware Fixed Rotation | 23.2257 | 94.95 | 93.38 | 6286.39 |
| Constrained Optimization | 23.4468 | 94.78 | 97.41 | 6547.90 |
| Fairness Rule | 23.4865 | 96.15 | 97.32 | 6544.70 |
| PPO | 23.4967 | 97.02 | 97.02 | 6528.02 |

Table 14 reports PPO minus each added comparator. The paired hierarchical bootstrap uses 10,000 resamples with seed 20261031, equal run/condition weights and coupled shortage clusters. Its pointwise 95% intervals are descriptive; they do not replace the original adjusted 97.5% primary intervals.

**Table 14. Post-hoc PPO effects against the two added baselines; mean differences and descriptive paired 95% intervals.**

| Outcome | PPO - Critical-aware Fixed Rotation | PPO - Fairness Rule |
| --- | --- | --- |
| Daily reward | +0.2710 [+0.2631, +0.2782] | +0.0102 [+0.0024, +0.0172] |
| Fairness (pp) | +2.067 [+1.958, +2.172] | +0.864 [+0.747, +0.976] |
| Utilization (pp) | +3.638 [+3.573, +3.697] | -0.308 [-0.364, -0.255] |
| Served energy (MWh) | +241.63 [+236.89, +245.91] | -16.68 [-20.84, -12.99] |
| Switching rate (pp) | +4.994 [+4.355, +5.755] | +2.923 [+2.297, +3.682] |

PPO improves fairness, utilization, served energy and reward over critical-aware rotation, although it switches more often. This is evidence of discretionary allocation improvement after sharing critical protection. It is not a pure learning ablation: action selection and the imitation-plus-PPO procedure also differ from the rotating rule.

Against its Fairness Rule teacher, PPO gains 0.0102 reward and 0.864 fairness percentage points, but serves 16.68 fewer MWh per day and loses 0.308 utilization percentage points. The reward interval's lower endpoint remains below the earlier +0.01 reference. The utilization interval is entirely below the earlier -0.2-point margin. These references provide context, not a newly declared confirmatory test. The pointwise intervals are not adjusted for all outcomes, shortage levels and training-run comparisons, so they do not establish a familywise significance claim. A positive pooled reward difference is insufficient to claim uniform superiority over the teacher or an operational improvement.

Exploratory reward differences against Fairness Rule vary by shortage: Mild -0.0207 [-0.0265, -0.0152]; Moderate -0.0133 [-0.0247, -0.0038]; Severe +0.0646 [+0.0556, +0.0737]. Thus the overall gain is concentrated in Severe conditions, with losses under Mild and Moderate shortage. Run-level results also vary: seed 1729 has a negative mean reward difference while the other four have positive means. All severity and per-run records remain available in the audit artifact; the favorable subgroup does not replace the overall comparison.

The feasibility stress test retains the 320 base demand draws and replaces hourly supply with 75% or 25% of total critical demand. Each family has 320 conditions, with no artificial replication over the original shortage labels, giving 5,760 additional trajectories across nine methods. These are constructed infeasibility tests, not estimates of their frequency in Khartoum. Table 15 reports the resulting critical shortfall.

**Table 15. Infeasible-critical-floor stress test; means over 320 demand conditions in each family.**

| Supply / critical demand | Procedure | Critical service (%) | Unmet critical (MWh) | Interruption (locality-h) | Complete outage (locality-h) |
| --- | --- | --- | --- | --- | --- |
| 75% | All protected procedures | 75.00 | 671.51 | 168 | 0 |
| 75% | Fixed Rotation | 19.46 | 2163.33 | 168 | 96 |
| 25% | All protected procedures | 25.00 | 2014.53 | 168 | 0 |
| 25% | Fixed Rotation | 6.49 | 2511.80 | 168 | 96 |

All protected procedures means Critical-aware Fixed Rotation, Constrained Optimization, Fairness Rule and each PPO actor: their actions and outcomes coincide under the all-Critical-Only fallback. They fall back for all 24 hours and incur 168 locality-hours below 99% critical service, with zero complete critical outage. Their 75% and 25% service fractions follow the physical supply limit; high fairness and zero total blackouts do not mean adequate critical service. Every method records zero power-balance violations within the stated 0.001 MW tolerance. This verifies a defined response to infeasibility, not successful critical protection when supply is insufficient. No protected assignment is physically feasible at these supply levels. The implementation therefore enables one all-Critical-Only fallback, which dispatch scales proportionally. Identical protected-controller outcomes are forced by that fallback; they do not demonstrate equal decision quality or a PPO advantage.

A further post-hoc calculation applies the numerical service requirements in Section 5.4 to Fairness Rule minus Constrained Optimization on the original 960 conditions. It resamples 320 base draws within generation-seed strata, keeping the three shortage variants together, with 10,000 paired resamples. The 97.5% pointwise intervals reuse the original interval level but do not create a new confirmatory family. Daily reward changes by +0.0397 [+0.0365, +0.0430]; utilization changes by -0.091 [-0.114, -0.067] percentage points. All numerical service requirements pass, including the shortage-level guardrails and zero critical interruption and power violations. Table 22 reports the intervals. This retrospective result supports the simple rule as a stronger practical comparator within this simulator; it does not change the original PPO verdict or establish general superiority.

The below-critical stress test is supplemented by an action-availability boundary test on the same 320 base demand draws. Supply equals total critical demand plus 0.99, 1.01 or 2.00 times the smallest locality's noncritical demand. This brackets the first Full Service upgrade: merely setting supply slightly above critical demand may leave only one feasible assignment. Tables 23 and 24 give the choice counts and outcomes. At 0.99 every protected method has one choice and behaves identically. At 1.01 there are two or three choices per hour; PPO's mean reward is 22.4999 versus 22.6728 for Constrained Optimization. At 2.00 there are eight choices; PPO reaches 22.6198 versus 22.5261 for the optimizer, while Critical-aware Fixed Rotation reaches 22.7567. All protected methods retain zero critical interruption and power violations. These constructed challenges expose changing rankings, not a uniform PPO advantage.

## 5.8 Learned forecasting results

The prediction experiment uses 320 demand traces from seeds 91001–91010, separate from the forecasting fit. Each method predicts hours 3–23, giving 235,200 locality-sector targets and 6,720 system targets. Common targets and training-only normalization prevent leakage through different windows or preprocessing.

Table 16 reports pooled errors on the common held-out target set.

**Table 16. Forecast error on 320 held-out traces; the hourly mean is a post-hoc baseline.**

| Forecast method | Cell MAE (MW) | Cell RMSE (MW) | System MAE (MW) | System RMSE (MW) |
| --- | --- | --- | --- | --- |
| Three-hour trailing mean | 2.3385 | 3.7436 | 42.7512 | 49.9759 |
| One-hour persistence | 1.3465 | 2.1776 | 24.3809 | 28.5917 |
| Ridge AR(3) | 0.6573 | 1.1882 | 9.3376 | 11.9242 |
| Training hour-of-day mean | 0.3344 | 0.5425 | 2.5478 | 3.2500 |

Ridge AR(3) has the lowest error among the three original methods on all four measures.
The additional training-fitted hour-of-day mean performs better: locality-sector MAE
and RMSE are 0.3344 and 0.5425 MW, while system MAE and RMSE are 2.5478 and 3.2500 MW.
It uses the same training traces and held-out targets, but was added after the original
results were known. The comparison is therefore descriptive and post-hoc.

The generator repeats fixed daily profiles with independent perturbations, so the
hourly mean captures predictable structure directly. Ridge AR(3)'s improvement over
persistence does not establish added predictive value over this stronger, simpler
baseline. Generalization to shifted profiles or measured demand remains untested.
Both models are fitted only on training data, but the hourly mean uses known clock
time while Ridge AR(3) uses lagged demand without a clock feature. A controlled
comparison adding clock features to Ridge was not performed, so this result does not
isolate model class from information representation. No forecast confidence interval
is asserted.

Saved evidence retains fitted parameters and per-seed counts, absolute-error sums and squared-error sums, not the full runtime prediction arrays. Errors can be reconstructed from these sufficient statistics. Forecasts remain separate from the policy and current display projection, so no controller improvement is attributed to them.

Figure 9 visualizes the same MAE and RMSE values from Table 16; it concerns prediction
accuracy rather than the controller reward.

[[FIGURE:forecast]]

## 5.9 Locality service at the end of the day

The original final-test evidence also retains each locality's end-of-day service ratio: its total served MWh divided by its total demanded MWh over 24 hours. Figure 10 averages these ratios across the 960 matched conditions, with equal weights for the five selected PPO actors. It is a mean of per-condition ratios, not a pooled energy ratio or the hourly-mean Jain index used in the main results. No new scenarios or policies are introduced for this descriptive analysis.

PPO's mean endpoint service ranges from 54.83% for Jebel Awlia to 66.59% for Khartoum. Relative to Constrained Optimization, the mean ratio rises in Bahri, East Nile and Ombada, but falls in the other four localities. For example, it increases by 1.97 percentage points in East Nile and decreases by 2.42 points in Khartoum. Higher aggregate fairness therefore describes redistribution, not a service gain for every locality or 97% of demand being supplied.

The mean service ratio of the worst-served locality within each condition is 36.02% for Fixed Rotation, 44.90% for Constrained Optimization and 48.82% for PPO. Here the minimum is taken across the seven localities separately for each run and condition, before averaging; taking the minimum of the displayed locality means would answer a different question. PPO's 3.92-point improvement over the optimizer in this endpoint measure coexists with the 19.88 MWh lower mean energy delivery already reported. This is an allocation tradeoff, not evidence about fairness within a locality or the social value of that tradeoff.

[[FIGURE:locality]]

## 5.10 Fresh audit of the learning stage

The teacher comparison does not measure improvement over the imperfect cloned actor. A separate follow-up therefore compares each selected PPO actor with its own retained behavior-cloning (BC) initialization. The protocol was frozen before outcome generation, followed by an implementation-and-model freeze. The original findings were already known, so the motivation is post-hoc; fresh execution does not retroactively preregister the earlier studies. No policy was retrained, tuned or selected again.

Seeds 132201–132210 passed checks against retained training, demonstration, validation and evaluation metadata. Ten seeds provide 32 base draws each, giving 320 independent draws and 960 coupled Mild, Moderate and Severe conditions. None of their input hashes matches the 4,960 previous scenario hashes checked. Four fixed comparators, five BC actors and five selected PPO actors produce 13,440 trajectories on identical matched inputs.

The central contrast pairs actors within the five training runs. The hierarchical bootstrap uses 10,000 resamples with seed 20261031, resampling paired runs and base-draw clusters within ten seed strata. Sampled scenarios are shared across runs and methods, retaining all three coupled shortage variants. Means weight runs and conditions equally. The intervals are descriptive, pointwise and nominally 95%; they do not establish simultaneous superiority across outcomes or comparators. Table 17 reports the learning-stage effects.

**Table 17. Fresh matched effects of selected PPO minus its own behavior-cloning initialization; mean differences and descriptive pointwise 95% intervals.**

| Outcome | Difference [95% interval] |
| --- | --- |
| Daily reward | +0.1801 [+0.1722, +0.1878] |
| Jain fairness (pp) | +3.914 [+3.775, +4.052] |
| Utilization (pp) | -1.213 [-1.261, -1.166] |
| Served energy (MWh/day) | -77.20 [-80.79, -73.85] |
| Switching rate (pp) | -5.404 [-6.084, -4.577] |
| Final minimum locality service (pp) | +4.887 [+4.410, +5.365] |

All five run-level reward intervals and all three shortage-level reward intervals are positive. Nevertheless, utilization and served energy decrease in every shortage, with their intervals entirely below zero. Switching decreases overall, while its Severe-only interval crosses zero. Thus fine-tuning and selection improve the specified objective and several distribution measures relative to BC, with an energy cost. This contrast includes extra computation and validation checkpoint selection; it does not isolate the PPO algorithm, compare equal training budgets or establish that learning from scratch is necessary.

**Table 18. Six procedures on the 960 fresh matched conditions; BC and PPO weight the five runs equally.**

| Procedure | Reward | Jain (%) | Util. (%) | Served (MWh) | Final min. (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed Rotation | 13.9705 | 87.76 | 71.25 | 4551.75 | 36.03 |
| Critical-aware Fixed Rotation | 23.2283 | 94.96 | 93.42 | 6287.17 | 51.72 |
| Constrained Optimization | 23.4430 | 94.72 | 97.41 | 6546.25 | 44.92 |
| Fairness Rule | 23.4858 | 96.16 | 97.32 | 6543.23 | 50.90 |
| Behavior Cloning (BC) | 23.3154 | 93.09 | 98.23 | 6603.25 | 43.93 |
| PPO | 23.4955 | 97.00 | 97.01 | 6526.04 | 48.81 |

Final minimum service in Table 18 is the lowest of seven end-of-day locality service ratios within each condition, averaged afterward. It is distinct from time-averaged Jain fairness and from the minimum of averaged locality ratios. **PPO's higher Jain fairness coexists with worse final service for the least-served locality than the teacher or Critical-aware Fixed Rotation.** The respective final-minimum effects are -2.086 [-2.517, -1.609] and -2.909 [-3.327, -2.448] percentage points. These adverse endpoints prevent a blanket equity claim.

Against Fairness Rule, the fresh reward difference is +0.0097 [+0.0016, +0.0170], with a mean below the earlier +0.01 reference. Served energy changes by -17.19 [-21.09, -13.73] MWh/day. Mild and Moderate reward effects favor the teacher; Severe favors PPO. The reference is context, not a new acceptance gate. The four fixed-comparator contrasts are replication diagnostics and replace none of the original primary decisions.

The size of the learning-stage gain also reflects its starting point. Behavior cloning scores 23.3154, compared with 23.4858 for the teacher. The resulting deficit is 0.1704 daily reward, while selected PPO ends only 0.0097 above the teacher. Most of the 0.1801 stage gain therefore closes the imitation gap arithmetically. This does not identify how much is caused by PPO rather than extra computation or checkpoint selection. Further imitation under an equal budget is an untested alternative, not an established substitute.

All protected methods, including BC, have zero critical interruption below 99% service, zero complete critical outage and zero power violations in every fresh condition. A separate validation record confirms exact replay of all 13,440 trajectories and reconstruction of means and intervals; Appendix A.4 identifies those records. These fresh seeds test the same synthetic profiles and perturbations, not changed demand mechanisms or real-grid operation. Five training runs still limit uncertainty estimation; the original evidence remains unchanged.


## 5.11 Computation and verification

Five fresh training seeds completed 3,000 iterations each: 18,432,000 transitions per run and 92,160,000 overall. Total recorded training time was 35.98 minutes (the rounded per-run entries sum to 35.97); final evaluation took 30.38 seconds. Validation selected iterations 2900, 3000, 2900, 3000, 2500 for seeds 1729, 2718, 3141, 5772, 8111, respectively. All five selections met the validation eligibility requirements. The final report uses the equal-run procedure mean rather than selecting the best performer on the final test.

Table 19 reports the validation scores used for selection; Table 11 separately reports those actors' final-test scores.

**Table 19. Training cost and validation-selected checkpoints; reward is from validation, not the final test.**

| Training seed | Selected iteration | Validation reward | Training minutes |
| --- | --- | --- | --- |
| 1729 | 2900 | 23.4855 | 7.23 |
| 2718 | 3000 | 23.4941 | 7.16 |
| 3141 | 2900 | 23.5051 | 7.19 |
| 5772 | 3000 | 23.5032 | 7.17 |
| 8111 | 2500 | 23.5070 | 7.22 |

These timings describe the recorded execution configuration in Appendix A.4. They
exclude the disclosed interrupted attempt and are not single-decision latency measurements.

# 6 Discussion

## 6.1 What AI and optimization contribute

The central finding is comparator dependence. PPO improves the chosen daily objective
over the current-hour optimizer, but most of that reward advantage is also available
from the transparent Fairness Rule. The forecasting study addresses a separate task:
lower prediction error does not explain the control results because controllers do not
receive those forecasts.

The stronger forecasting comparison reinforces the need for an informative baseline.
An hour-of-day mean fitted on training traces outperforms Ridge AR(3) on every reported
error measure. Under fixed daily profiles, knowing the hour can be more useful than
extrapolating from the last three observations. This result limits the contribution of
the chosen learned forecaster; it does not establish that autoregression is ineffective
for other demand-generating processes.

The optimizer selects the best feasible current action:

$$
\operatorname{arg\,max}_{a}\,r_t(s_t,a)
$$

PPO learns a policy maximizing:

$$
\mathbb{E}\!\left[\sum_{t=0}^{23}r_t\right]
$$

PPO cannot exceed this one-step maximum at the same state. A daily gain can arise
because actions change cumulative service and later switching costs. This distinguishes
sequential allocation from storage: foregoing a Full Service upgrade does not save its
electricity for tomorrow, but it changes which locality is disadvantaged at the next
decision. The experiment therefore tests a learned policy against a myopic objective,
not against the optimum of the same full-day planning problem.

Section 5.5 reports a 0.0499 daily-reward gain with 19.88 fewer served MWh and more
switching. The Fairness Rule obtains 0.0397 of that gain, approximately 80%, with a
smaller utilization loss. Its tolerance admits a slightly lower current reward when
this helps the currently worst-served locality. A simple rule can consequently create
useful service-history effects without learning a long-horizon value function. The
remaining PPO-minus-rule difference is a comparison of complete procedures, not an
isolated causal estimate of learning. The teacher is preferable on reward under Mild
and Moderate shortage; PPO's pooled increment comes from Severe shortage.

The later retrospective comparison also finds that Fairness Rule meets the numerical
reward and service margins used for the optimizer comparison, whereas PPO does not.
That result is descriptive because this additional comparison was not part of the
original primary family. Even with that qualification, the rule supplies a direct
counterexample to the suggestion that learning is necessary to obtain a favorable
reward-and-utilization tradeoff in this simulation.

These results do not establish that the neural policy's added complexity is justified
by operational benefits. PPO requires demonstrations, training, validation selection
and retained weights; the Fairness Rule specifies its decisions directly. The present
study neither prices those costs nor establishes a decision-latency advantage. A
decision maker who values utilization or auditability may prefer the optimizer or
teacher even when the reported reward favors PPO. A higher score alone cannot settle
that choice, because its weights encode research priorities rather than a validated
economic or social valuation.

Findings do not establish superiority to full-horizon optimization, model predictive
control or all adaptive rules. Section 5.7 directly examines the current frozen actors
against the Fairness Rule teacher and Critical-aware Fixed Rotation. The earlier negative
teacher comparison remains separately identified in Appendix A.

The fresh audit in Section 5.10 makes the next distinction explicit. Relative to each
run's own imitation-only actor, fine-tuning and validation selection increase daily reward
by 0.1801 and Jain fairness by 3.914 percentage points, but reduce energy delivery by
77.20 MWh/day. This is a measured stage benefit for the chosen objective, with a service
cost. It includes the additional training and checkpoint selection; it is not an
equal-budget test isolating the PPO algorithm.

The imitation-only actors begin 0.1704 reward below their teacher, while the selected
policies finish 0.0097 above it. Thus about 95% of the measured stage increase closes
the teacher gap arithmetically. This does not show that PPO merely memorizes the rule,
or that more imitation would necessarily reach the same result. It shows why the
starting point matters: a large improvement over an imperfect clone can coexist with
a small improvement over the rule that generated its labels. Equal-budget comparisons
with further imitation and training from scratch are needed to distinguish those
explanations.

## 6.2 Critical protection and fair service

Critical protection comes from the mask and Critical Only. Default supply also permits a deterministic rule to protect these sectors. PPO allocates remaining supply; its Fixed Rotation advantage cannot be attributed solely to learning.

Jain fairness measures dispersion across cumulative service ratios, not the amount of
service available. Equal partial service can produce a high index while substantial
demand remains unmet. Critical Only can eliminate complete blackouts while Residential
and Commercial demand receives nothing. Utilization answers a different question:
how much usable supply is delivered. A controller can score well on one measure and
poorly on another without a numerical inconsistency.

The fresh endpoint analysis supplies a concrete counterexample: PPO has higher
time-averaged Jain fairness than Fairness Rule and Critical-aware Fixed Rotation, but lower
mean final service to the worst-served locality. The latter is 48.81% for PPO, 50.90%
for the teacher and 51.72% for Critical-aware Fixed Rotation. Choosing an equity measure
therefore changes which procedure appears preferable; these measures cannot resolve
stakeholder priorities on their own.

Timing also matters. The reward averages Jain fairness after each hour, whereas the
final minimum considers the least-served locality after all 24 hours. A policy can
improve equality early in the day yet finish with a lower minimum ratio. Averaging
first across scenarios would conceal which locality is worst within each episode;
the reported endpoint therefore takes the minimum before averaging. The locality
plot further shows that improved aggregate equality need not improve every locality.
Assessing fairness requires specifying both the unit of comparison and the time
horizon, alongside a minimum level of acceptable service.

The infeasible-supply test addresses a separate issue. When supply is below critical
demand, no protected assignment fits. The implementation explicitly enables one
all-Critical-Only fallback and dispatches it proportionally. Agreement among protected
controllers is then forced by the fallback, so it validates the declared response to
infeasibility rather than the relative quality of PPO decisions. Testing discretionary
allocation near this boundary requires enough supply for at least one Full Service
upgrade, not merely a nominal percentage above critical demand.

The additional boundary experiment makes this distinction visible by scaling spare
supply relative to the smallest locality's noncritical upgrade. Below that threshold,
there is only one feasible assignment. Just above it, two or three choices become
available and PPO's mean reward is lower than Constrained Optimization's. With eight
choices, PPO exceeds that optimizer, but Critical-aware Fixed Rotation has the highest
mean reward. These constructed conditions reveal sensitivity to action granularity,
not a stable ranking that can be extended to other supply regimes.

No equipment wear, restoration delay or consumer switching cost is modeled. The 99% critical threshold and reward weights are research choices; operational priorities, facility requirements and tradeoffs need agreement.

## 6.3 Data and physical limitations

Synthetic profiles, fixed locality weights and demand-linked supply limit generalization. Fresh seeds prevent sample reuse but do not test changed demand patterns, prolonged damage or different shortage mechanisms.

Global supply omits network connectivity, feeder limits, losses, voltage and frequency constraints. Sector separation may be physically unavailable. Hourly, single-day decisions omit transients and multi-day disadvantage; administrative geometry supplies no electrical validation.

A more realistic simulation requires the operational data identified in Table 20.

**Table 20. Data required for future physical and operational validation.**

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
generator. Continuing training improved daily reward but reduced utilization by about
0.24 percentage points and failed the stated service margins (Section 5.6). More
computation therefore did not produce uniformly better service.

The action count grows as $2^N$, requiring different representations and new feasibility
tests at larger scales. The fresh comparison identifies the change from each saved
imitation-only actor to its selected fine-tuned actor, but component attribution remains
limited by checkpoint selection and unequal computation. Comparing training from scratch,
further imitation and fine-tuning under equal budgets would test these mechanisms more
directly. The new seeds replicate the same generator rather than testing a distribution
shift, and the descriptive intervals do not establish familywise significance across
all the additional outcomes.

Protocols and hashes support reproduction, not independent custody or field validation. Prior exploratory and unfavorable findings remain disclosed.

## 6.5 Implications for further work

The most informative next learning experiment would compare further imitation,
imitation followed by PPO, and PPO trained from scratch under explicit common budgets.
Training and validation effort should both count. Reporting reward, energy and
worst-locality service would test whether the teacher gap can be closed without the
observed service cost, rather than rewarding whichever method receives more tuning.

A longer-horizon comparator would test the remaining sequential claim. An exact or
certified relaxation could provide an upper bound on achievable full-day reward;
beam search would provide another feasible heuristic unless accompanied by a valid
bound. Any comparator using future demand must be identified as a perfect-information
reference, or supplied forecasts with the same information available to the other
methods. Otherwise an apparent gain would mix planning ability with privileged inputs.

The hour-of-day result should be followed by tests with shifted peaks, changed demand
scales and measured traces. Such a design separates exploiting a fixed daily template
from adapting to unfamiliar demand. Forecast-in-the-loop control
should then give every controller equivalent forecasts and measure how prediction
errors affect service, rather than treating a lower forecasting error as sufficient
evidence of better allocation.

The boundary experiment should be extended to changing demand profiles and different
locality sizes, recording admissible-action counts as well as outcomes. This would test
whether its controller ranking persists beyond the present action granularity.
Representative facility data, feasible network
actions and stakeholder priorities remain prerequisites for interpreting any of these
simulation improvements as operational benefits.

# 7 Conclusion

The results answer the five research questions as follows.

**RQ1 — transparent scenarios.** A seeded generator represents seven localities,
five sectors and coupled Mild, Moderate and Severe shortages using explicit hourly
profiles and bounded perturbations. The equations in Chapter 3 and Appendix A.3 make
the assumptions reproducible. These are illustrative scenarios, not estimated Khartoum
operating conditions.

**RQ2 — forecasting.** Ridge AR(3) lowers all four reported errors relative to
persistence and a three-hour trailing mean on held-out synthetic targets, but the later
training-fitted hour-of-day mean outperforms Ridge AR(3) on all four. Thus the initial
comparison favors autoregression only against the weaker baselines; added value over
the daily template is not demonstrated. Forecasts do not enter the control experiment.

**RQ3 — comparison with Fixed Rotation.** Both constrained controllers improve
critical interruption, Jain fairness, utilization and blackout-duration
disparities relative to the original schedule. PPO meets the declared composite
criterion. Critical-aware Fixed Rotation also eliminates the critical interruptions,
so the fall from 102.05 locality-hours to zero primarily demonstrates the benefit of
critical protection, rather than a need for learning.

**RQ4 — comparison with Constrained Optimization.** PPO raises Jain fairness by
2.24 percentage points and daily reward by 0.0499, with an adjusted 97.5% reward interval
of [0.0406, 0.0577]. It serves 19.88 fewer MWh per day, loses 0.40 utilization points
and switches more often. It therefore fails the complete declared criterion. Fairness
Rule obtains approximately 80% of that reward advantage with a smaller utilization
loss. PPO's additional 0.0102 reward over the teacher is concentrated in Severe shortage;
Mild and Moderate favor the teacher. These outcomes support a limited objective gain
with service costs, not overall superiority of learning. Fairness Rule satisfies the
same numerical service margins in a later retrospective comparison; that comparison
does not acquire the status of an original confirmatory test.

**RQ5 — explanation through an interface.** The offline application presents locality
context, hourly matched replay and saved comparative results. Technical tests support
its implementation, but no participant study establishes that the interface improves
understanding. That part of the answer concerns design and functioning software rather
than measured usability.

Fine-tuning and validation selection add 0.1801 reward over each run's imitation-only
actor, mostly closing its 0.1704 teacher deficit, while delivering 77.20 fewer MWh per day.
The selected procedure ends 0.0097 above the teacher. Additional training also fails the
utilization guardrails. These results support a bounded comparison of allocation
procedures, not a general advantage from additional learning.

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

## A.3 Exact synthetic generator

Let $h\in\{0,\ldots,23\}$ denote the hour, $w_i$ the locality weights and $b_k$ the sector coefficients in Table 1. Define a Gaussian-shaped component and the normalized sector profiles by

$$
G(h;\mu,\sigma)=\exp\left[-\frac{1}{2}\left(\frac{h-\mu}{\sigma}\right)^2\right],\qquad
p_k(h)=\frac{g_k(h)}{\max_{u\in\{0,\ldots,23\}}g_k(u)}.
$$

The normalization uses the 24 integer hours, not the maximum over continuous time. In Table 21, each peak is written as $(\mathrm{amplitude},\mu,\sigma)$, with $\mu$ and $\sigma$ in hours; $g_k(h)$ is the constant plus the stated Gaussian components.

**Table 21. Parameters of the unnormalized synthetic sector profiles.**

| Sector | Constant | First peak | Second peak |
| --- | ---: | --- | --- |
| Hospital | 0.95 | None | None |
| Water | 0.45 | (0.35, 6, 2) | (0.40, 18, 2.5) |
| Bakery | 0.35 | (0.60, 5, 1.5) | (0.45, 16, 2) |
| Residential | 0.40 | (0.30, 7, 2) | (0.65, 20, 3) |
| Commercial | 0.25 | (0.75, 13, 4) | None |

Hospital's normalized profile is therefore constant at one. With independent uniform draws $u_{i,k,h},v_h\in[0,1)$, the nominal locality scale and perturbed demand are

$$
P_i=700\frac{w_i}{\sum_jw_j}\;\mathrm{MW},\qquad
d_{i,k,h}=P_i b_k p_k(h)\left[1+0.05(2u_{i,k,h}-1)\right].
$$

Let $D_h=\sum_{i,k}d_{i,k,h}$. Supply for shortage level $\ell$ is

$$
S_{\ell,h}=\min\left\{D_h f_\ell(h)\left[1+0.05(2v_h-1)\right],\;0.95D_h\right\}.
$$

The fractions are $f_{\mathrm{Mild}}(h)=0.80$; $f_{\mathrm{Moderate}}(h)=0.65-0.10\mathbf{1}[17\le h<21]$; and $f_{\mathrm{Severe}}(h)=0.50-0.15\mathbf{1}[16\le h<22]$. The same demand and supply draws are reused for all three shortage variants of a base condition. Each call first draws the demand array and then the supply array from the seeded CUDA generator. The nominal study uses these defaults without demand or supply overrides.

The bounded noise also explains why the nominal generator can protect critical demand. Write $C_h^0,N_h^0$ for unperturbed critical and noncritical demand, and $q_{\ell,h}=0.95f_\ell(h)$. The 95%-of-demand cap cannot bind under these defaults because $0.80\times1.05<0.95$. The lowest real-arithmetic supply margin over critical demand is bounded by

$$
S_{\ell,h}-C_h\;\ge\;(q_{\ell,h}-1)1.05C_h^0+q_{\ell,h}0.95N_h^0.
$$

Evaluating all 24 hours gives minimum margins of approximately 114.16 MW, 74.02 MW and 32.57 MW for Mild, Moderate and Severe, respectively. This is an analytical envelope of the stated generator, evaluated using its frozen profiles, not a new sampled experiment or a guarantee for actual electricity supply. The separate infeasibility stress tests deliberately violate this nominal construction.

## A.4 Evidence and reproducibility

The project source and review package are available in the
[GitHub repository](https://github.com/boomout2012-afk/gridguard-khartoum).
The reviewer guide identifies the current thesis, presentation, GUI captures and
evidence. The original submission state preceding this revision is retained at commit
`10ecbb425ddb05e36e602505a160a46345a4e635`; historical release evidence and later
follow-up analyses remain separately labelled. Reproducing the scientific runs requires the documented CUDA
environment; checking the supplied file manifest requires only Python.

The source literature review is retained at `docs/source/literature-review-authors.docx`,
with bibliographic and claim checks recorded in `docs/LITERATURE_REVIEW_AUDIT.md`. The
current control protocol is `docs/FOCUSED_PPO_PROTOCOL.md`. Its SHA-256 is
`e3e04b018846714490db047ec519ed963a371b738957ef09368fd4732be535f8`.
The internal freeze record states 23 October 2026 at 09:28:05 UTC. The earliest available
Git commit containing that protocol is
[`10ecbb425ddb05e36e602505a160a46345a4e635`](https://github.com/boomout2012-afk/gridguard-khartoum/commit/10ecbb425ddb05e36e602505a160a46345a4e635),
dated 31 October 2026. Thus Git confirms the preserved protocol's later identity, not
independent custody before training; no earlier public commit is asserted.
The research archive also
retains the forecasting study, earlier optimization comparisons and previous teacher
comparison alongside their supporting evidence.

The separately hashed `data/evidence/committee-baseline-audit.json` records the post-hoc
teacher, Critical-aware Fixed Rotation and infeasibility comparisons, including paired,
per-run and shortage-specific results. It preserves the original model identities
and exact replay of unchanged methods. Its descriptive 95% intervals and later audit
recipe do not replace the original protocol or adjusted primary intervals.

The new `docs/THESIS_LEARNING_AUDIT_PROTOCOL.md` was fixed before the fresh execution,
after the original findings were known. `data/evidence/thesis-learning-audit.json`
contains all 14 methods on 960 new conditions; its separate validation receipt records
exact replay and independent CUDA metric reconstruction. No policy was retrained or
reselected. The endpoint plot and numerical generator/training interpretations are
derived from the original evidence in `docs/assets/focused-study/locality-allocation.json`
and `interpretation-details.json`, with source and figure hashes. These additions remain
separate from the original benchmark shown in the app and defense slides.

Model weights, configuration, source identities and scenario/action hashes bind each experiment to its evidence. Current, historical and live results remain distinct. An interrupted initial training attempt was retained and restarted under the same frozen configuration; the training log records it separately, and completed-run timings exclude that attempt. No final-test results were consulted to restart or select training.

Training and evaluation elapsed times describe experimental cost, not single-decision
latency. Comparing controller speed would require common hardware, batch size and
synchronization rules; this study makes no speed-superiority claim.

The recorded runs used an NVIDIA GeForce RTX 5060 Laptop GPU with approximately
7.93 GiB of reported device memory, PyTorch 2.8.0 with CUDA 12.8, and CuPy 14.1.1.
Training used float32, deterministic algorithms, disabled TF32, and the cuBLAS
workspace setting `:4096:8`.

## A.5 Additional baseline analyses

Table 22 applies the original numerical service margins retrospectively to Fairness Rule minus Constrained Optimization. Both are deterministic, so there is no training-run resampling or validation-selection requirement. The table contains pointwise 97.5% paired intervals, not a multiplicity-adjusted family for the additional comparisons. All shortage-level mean guardrails also pass.

Every protected procedure records zero critical interruption and zero power-balance violations. The unprotected Fixed Rotation outcomes, individual PPO actors, action hashes and per-condition records are retained alongside these summaries in `data/evidence/additional-baselines.json`. Reproduction and portable verification use `tools/evaluate_additional_baselines.py`. The same artifact records the training-fitted hourly profile and sufficient forecast error statistics for all four methods in Table 16. It first reproduces the three original forecasting methods on the original targets.

**Table 22. Post-hoc Fairness Rule effects against Constrained Optimization.**

| Outcome | Mean difference and 97.5% interval | Required endpoint |
| --- | --- | --- |
| Daily reward | +0.0397 [+0.0365, +0.0430] | Lower above zero |
| Jain fairness (pp) | +1.373 [+1.300, +1.446] | Lower at least -0.2 |
| Utilization (pp) | -0.091 [-0.114, -0.067] | Lower at least -0.2 |
| Served energy (MWh) | -3.20 [-4.88, -1.48] | Descriptive |
| Switching rate (pp) | +4.296 [+4.069, +4.519] | Descriptive |

For the boundary challenge, let the critical set of sectors be $\mathcal C$. For each hourly demand array, define the first discretionary-service quantum and supply by

$$
q_t=\min_i\sum_{s\notin\mathcal C}d_{t,i,s},\qquad S_t=\sum_i\sum_{s\in\mathcal C}d_{t,i,s}+\alpha q_t,\quad \alpha\in\{0.99,1.01,2.00\}.
$$

No training or model selection uses these post-hoc challenges. Each family contains 320 conditions and nine controllers, for 8,640 trajectories in total. Each input hour contributes once to Table 23; the five actors are averaged equally for the PPO procedure in Table 24. The first upgrade needs about 35% headroom above critical demand on average, reflecting locality-level action granularity. These are action-availability boundaries, not a study of supply within 1% of critical demand.

**Table 23. Admissible assignments in the constructed boundary families.**

| Multiplier | Mean supply / critical (%) | Choice count range | Hours with multiple choices |
| --- | --- | --- | --- |
| 0.99 | 134.83 | 1–1 | 0 / 7,680 |
| 1.01 | 135.54 | 2–3 | 7,680 / 7,680 |
| 2.0 | 170.37 | 8–8 | 7,680 / 7,680 |

**Table 24. Protected controller outcomes on the boundary challenges.**

| Multiplier | Procedure | Reward | Served (MWh) | Jain (%) | Utilization (%) |
| --- | --- | --- | --- | --- | --- |
| 0.99 | Critical-aware Fixed Rotation | 22.6362 | 2686.03 | 99.99 | 74.44 |
| 0.99 | Constrained Optimization | 22.6362 | 2686.03 | 99.99 | 74.44 |
| 0.99 | Fairness Rule | 22.6362 | 2686.03 | 99.99 | 74.44 |
| 0.99 | PPO | 22.6362 | 2686.03 | 99.99 | 74.44 |
| 1.01 | Critical-aware Fixed Rotation | 22.4725 | 3628.07 | 71.64 | 99.74 |
| 1.01 | Constrained Optimization | 22.6728 | 3605.58 | 76.40 | 98.95 |
| 1.01 | Fairness Rule | 22.6724 | 3605.58 | 76.39 | 98.95 |
| 1.01 | PPO | 22.4999 | 3628.11 | 72.26 | 99.74 |
| 2.0 | Critical-aware Fixed Rotation | 22.7567 | 3845.16 | 93.27 | 84.46 |
| 2.0 | Constrained Optimization | 22.5261 | 3993.83 | 85.65 | 87.38 |
| 2.0 | Fairness Rule | 22.5531 | 3996.43 | 86.15 | 87.47 |
| 2.0 | PPO | 22.6198 | 3988.06 | 87.92 | 87.39 |

# Appendix B Source code and review package

The complete project source and review documents are available in the
[GridGuard Khartoum GitHub repository](https://github.com/boomout2012-afk/gridguard-khartoum).
The repository README is the entry point for the thesis, presentation, current GUI
captures and research guide. Setup and experiment commands are collected separately
so that the repository overview remains concise.

The [public validation run](https://github.com/boomout2012-afk/gridguard-khartoum/actions/runs/41225175729)
at commit `a2b98ec` completed 1,086 portable tests without failures on 31 October 2026.
The additional baseline analysis also passes 19 focused tests, including analytic
hourly-profile examples, held-out isolation, action-boundary cases and evidence tampering.

Verification checks retained weights against validation records, common input identities
across methods, action constraints and metric summaries. Statistical reconstruction checks
intervals and decisions. Replay connects saved actors to regenerated inputs, recorded
actions and outcomes. These checks address different failure modes; an evidence hash
alone does not validate the scientific model.

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

Natural Earth (n.d.) *1:10m rivers and lake centerlines*. Version 5.0.0. Public-domain geographic dataset. Available at: https://www.naturalearthdata.com/downloads/10m-physical-vectors/10m-rivers-lake-centerlines/ (Accessed: 31 October 2026).

Nduhuura, P., Garschagen, M. and Zerga, A. (2021) ‘Impacts of electricity outages in urban households in developing countries: A case of Accra, Ghana’, *Energies*, 14(12), 3676. https://doi.org/10.3390/en14123676.

Nti, I.K., Teimeh, M., Nyarko-Boateng, O. and Adekoya, A.F. (2020) ‘Electricity load forecasting: A systematic review’, *Journal of Electrical Systems and Information Technology*, 7, article 13. https://doi.org/10.1186/s43067-020-00021-8.

Okuta, R., Unno, Y., Nishino, D., Hido, S. and Loomis, C. (2017) ‘CuPy: A NumPy-compatible library for NVIDIA GPU calculations’, *Workshop on Machine Learning Systems at NIPS 2017*. Available at: https://learningsys.org/nips17/assets/papers/paper_16.pdf.

Oluwasuji, O.I., Malik, O., Zhang, J. and Ramchurn, S.D. (2020) ‘Solving the fair electric load shedding problem in developing countries’, *Autonomous Agents and Multi-Agent Systems*, 34, article 12. https://doi.org/10.1007/s10458-019-09428-8.

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Köpf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J. and Chintala, S. (2019) ‘PyTorch: An imperative style, high-performance deep learning library’, *Advances in Neural Information Processing Systems*, 32. Available at: https://papers.neurips.cc/paper/9015-pytorch-an-imperative-stylehigh-performance-deep-learning-library.pdf.

Poudel, S. and Dubey, A. (2019) ‘Critical load restoration using distributed energy resources for resilient power distribution system’, *IEEE Transactions on Power Systems*, 34(1), pp. 52–63. https://doi.org/10.1109/TPWRS.2018.2860256.

Ramchurn, S.D., Vytelingum, P., Rogers, A. and Jennings, N.R. (2012) ‘Putting the smarts into the smart grid: A grand challenge for artificial intelligence’, *Communications of the ACM*, 55(4), pp. 86–97. https://doi.org/10.1145/2133806.2133825.

Rosales-Asensio, E., Elejalde, J.L., Pulido-Alonso, A. and Colmenar-Santos, A. (2022) ‘Resilience framework, methods, and metrics for the prioritization of critical electrical grid customers’, *Electronics*, 11(14), 2246. https://doi.org/10.3390/electronics11142246.

Sarwar, S., Mokhlis, H., Othman, M., Muhammad, M.A., Laghari, J.A., Mansor, N.N., Mohamad, H. and Pourdaryaei, A. (2020) ‘A mixed integer linear programming based load shedding technique for improving the sustainability of islanded distribution systems’, *Sustainability*, 12(15), 6234. https://doi.org/10.3390/su12156234.

Schulman, J., Moritz, P., Levine, S., Jordan, M.I. and Abbeel, P. (2016) ‘High-dimensional continuous control using generalized advantage estimation’, *International Conference on Learning Representations*. Available at: https://arxiv.org/abs/1506.02438.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A. and Klimov, O. (2017) *Proximal policy optimization algorithms*. arXiv:1707.06347. https://doi.org/10.48550/arXiv.1707.06347.

United Nations Office for the Coordination of Humanitarian Affairs (n.d.) *Sudan administrative boundaries (COD-AB)*. Revision v03. Humanitarian Data Exchange. Geographic dataset valid from 31 August 2020; CC BY 3.0 IGO. Available at: https://data.humdata.org/dataset/cod-ab-sdn. The archived download URL and SHA-256 identity are retained in `src/gridguard/assets/map/map-provenance.json`.

Vázquez-Canteli, J.R. and Nagy, Z. (2019) ‘Reinforcement learning for demand response: A review of algorithms and modeling techniques’, *Applied Energy*, 235, pp. 1072–1089. https://doi.org/10.1016/j.apenergy.2018.11.002.

Wang, C., Yu, H., Chai, L., Liu, H. and Zhu, B. (2021) ‘Emergency load shedding strategy for microgrids based on dueling deep Q-learning’, *IEEE Access*, 9, pp. 19707–19715. https://doi.org/10.1109/ACCESS.2021.3055401.

World Bank (2025) *Accelerating sustainable and clean energy and digital access transformation in Sudan (P507788)*. Project Information Document, Appraisal Stage, Report PIDIA01286, 1 May. Available at: https://documents1.worldbank.org/curated/en/099050125112514628/pdf/P507788-6a843250-44f5-47da-9f07-c6b1269a091f.pdf.




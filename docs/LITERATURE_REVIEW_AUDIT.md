# Literature review audit

Audit date: 23 October 2026. Input: `tmp/literature-review-user.txt`, extracted from `docs/source/literature-review-authors.docx`. The original Word document was not edited. Its SHA-256 at audit time is `d2d8f1c0a65e1271976335c314b4b1ec43ec6b4bd1805a0e0160934e027a3edd`.

This is a targeted audit of 15 central references and their use in the chapter, not a verification of every reference. Evidence below comes from the original papers, publishers, or author/institution repositories. Bibliographic identification, abstract verification, and full-text verification are distinguished. A valid reference does not establish every claim made around it.

## Changes needed before incorporation into the thesis

| Draft location in extracted text | Finding | Recommended treatment |
| --- | --- | --- |
| Lines 17, 19, 213 | The review describes all 52 items as peer-reviewed and makes an exhaustive novelty claim. Basheir is policy analysis; no reproducible systematic search establishes the absence of all prior work. | Describe a selected review of research and contextual sources. Say which combination the project investigates, or “among the studies reviewed.” Retain the broader literature; qualify its role. |
| Lines 39, 195 | Envy-freeness is presented as the implemented geographic fairness definition. | Define the actual Jain service-allocation index and the separately measured blackout-duration metrics. Explain that Oluwasuji motivates fair allocation but uses different utility definitions and constraints. Do not claim envy-freeness unless separately formulated and tested. |
| Lines 49–51, 105 | “Hard constraints in the reward function” conflates preference penalties with feasibility enforcement; unconditional hospital protection is physically impossible if supply cannot cover critical demand. | State that the critical-service action mask restricts choices to feasible protected actions; reward terms rank those choices. State the critical-feasibility assumption and the behavior/recorded shortfall when it fails. |
| Lines 27, 73, 207 | Universal claims that optimization is too slow, or cannot respond to chronic deficits, are unsupported. Emergency transient-control timing is transferred to hourly allocation. | Present computation as an empirical question tied to formulation, scale, hardware and horizon. Compare PPO with the implemented Constrained Optimization and Fixed Rotation on identical scenarios. Avoid a speed claim without a controller-level timing experiment. |
| Lines 37, 159 | The precise hospitals/water/bakeries triad is attributed to Rosales as a universal minimum critical-load definition. | Treat these categories as the proposal's application choice. Cite resilience literature for context-dependent critical-load weighting, not for an unverified exact list. |
| Lines 27, 185, 215 | Synthetic validation is treated as validation of Sudan's physical grid, and a 24-hour experiment is described in terms of days/weeks. | Separate national context from assumed locality/sector profiles. Describe synthetic decision-support experiments over the actual horizon. Power balance is not voltage, frequency or transient-stability validation. |
| Lines 81, 105, 109, 191, 199 | Undated citation counts and phrases such as “most comprehensive” are used as evidence of technical correctness. | Remove the counts and superlatives. Describe each study's method, data, comparison and limits instead. |
| Lines 139, 165, 207 | Several broad causal or comparative statements exceed the cited studies. | Avoid “closest” country-case claims without a search rule, “most comparable” national-context claims without criteria, and claims that conventional rotation itself caused observed household inequities. |

These corrections improve the strength of the argument; they do not remove the proposal's objectives or presume that PPO must win.

## Corrected central references and claim checks

### 1. Chen et al. (2023): directly relevant PPO precedent

Chen, **H.**, Zhuang, **J.**, Zhou, **G.**, Wang, **Y.**, Sun, **Z.**, and Levron, Y. (2023). *Emergency load shedding strategy for high renewable energy penetrated power systems based on deep reinforcement learning*. Energy Reports, 9, 434–443. DOI: [10.1016/j.egyr.2023.03.027](https://doi.org/10.1016/j.egyr.2023.03.027).

**Verified:** Author-institution record confirms the authors, PPO, a modified IEEE 39-bus system and frequency-restoration experiments. Replace the draft's incorrect initials. This supports PPO as a relevant candidate; it does not demonstrate geographic fairness, superiority in chronic daily rationing, or field deployment in Sudan. The complete reward formulation was not inspected, so avoid the categorical claim that it contains no load differentiation. [Technion record and abstract](https://cris.technion.ac.il/en/publications/emergency-load-shedding-strategy-for-high-renewable-energy-penetr/).

### 2. Oluwasuji et al. (2020): fairness through optimization

Oluwasuji, O. I., Malik, O., Zhang, J., and Ramchurn, S. D. (2020). *Solving the fair electric load shedding problem in developing countries*. Autonomous Agents and Multi-Agent Systems, **34, article 12**. DOI: [10.1007/s10458-019-09428-8](https://link.springer.com/article/10.1007/s10458-019-09428-8).

**Verified in full text:** The paper uses mixed-integer formulations and household utility, supply and duration constraints informed by utilitarian, egalitarian and envy-freeness considerations. It adapts household data to a developing-country setting. The issue year is 2020 although online publication was December 2019. Its utility-based fairness formulation is not interchangeable with Jain's index. The inspected text does not justify declaring its approach too slow for all chronic-deficit allocation problems. This is a constructive reason to include an optimization comparator, not dismiss one.

### 3. Sarwar et al. (2020): priority-aware MILP

Sarwar, S., Mokhlis, H., Othman, M., Muhammad, M. A., Laghari, J. A., Mansor, N. N., Mohamad, H., and Pourdaryaei, A. (2020). *A mixed integer linear programming based load shedding technique for improving the sustainability of islanded distribution systems*. Sustainability, 12(15), 6234. DOI: [10.3390/su12156234](https://www.mdpi.com/2071-1050/12/15/6234).

**Verified from publisher text:** MILP combines load priority and a voltage-stability ranking; the paper tests a Malaysian 28-bus network and IEEE 69-bus system in PSCAD/EMTDC. Its reported recovery behavior belongs to those dynamic simulations. Our allocation optimizer is a different formulation and should not be described as a reproduction of Sarwar. Do not generalize its operating-time requirements to every optimizer. The last author's full name is Alireza Pourdaryaei; the draft's initial is acceptable.

### 4. García and Fernández (2015): safe-RL taxonomy

García, J., and Fernández, F. (2015). *A comprehensive survey on safe reinforcement learning*. Journal of Machine Learning Research, **16(42)**, 1437–1480. [JMLR paper](https://jmlr.org/papers/v16/garcia15a.html).

**Verified:** The survey distinguishes changes to the optimality criterion from changes to exploration. It motivates attention to safety; it does not make a finite negative reward a proof of hard feasibility. In the thesis, identify the implemented action mask explicitly and limit guarantees to its modeled constraints and feasible supply conditions. Do not imply that this citation validates the project's synthetic environment or physical grid stability.

### 5. Al-Rikabi et al. (2025): Sudan context, not locality calibration

Al-Rikabi, **I. J.**, Omara, **A. A. M.**, Abuelnour, **M. A.**, Abuelnuor, **A. A. A.**, Mohamed, **A. E.**, and Ismail, **N. A.** (2025). *Renewable energy in Sudan: Current status and future prospects*. Engineering Reports, **7(1)**, e13116. [Publisher paper, DOI 10.1002/eng2.13116](https://onlinelibrary.wiley.com/doi/full/10.1002/eng2.13116).

**Verified:** The review reports 1,907 MW hydropower capacity and a 54.6% hydro generation share. Its abstract distinguishes a 10 MW solar installation from **5 MW on-grid**; the draft's “grid-connected solar amounts to only 10 MW” needs correction or a specifically identified alternative table/year. Figures and future targets are compiled from differently dated sources. Date them and avoid presenting announced projects as completed. National aggregates do not establish the model's seven locality/sector profiles or current wartime operating capacity.

### 6. Basheir and Abdelrahman (2022): contextual policy source

Basheir, **R. H.**, and Abdelrahman, **M. S.** (2022, 14 October). *The electricity crisis in Sudan: Between quick-fixes and opportunities for a sustainable energy transition*. Transnational Institute. [Original TNI essay](https://www.tni.org/en/article/the-electricity-crisis-in-sudan).

**Verified:** Authors are Razaz H. Basheir and Mohamed Salah Abdelrahman. This is a policy longread, not a peer-reviewed journal article. Cite this 2022 version directly; the draft mixes it with a later book-chapter publication. Its statement that available capacity covered about 60% of demand describes its historical context. It is not a measured current deficit, a calibration of every locality, or evidence that every part of the deficit is managed exclusively by rotation.

### 7. Jenkins et al. (2016): energy justice

Jenkins, K., McCauley, D., Heffron, R., Stephan, H., and Rehner, R. (2016). *Energy justice: A conceptual review*. Energy Research & Social Science, 11, 174–182. DOI: [10.1016/j.erss.2015.10.004](https://doi.org/10.1016/j.erss.2015.10.004). [Institutional accepted manuscript](https://www.storre.stir.ac.uk/bitstream/1893/23236/1/Energy%20Justice%20A%20Conceptual%20Review%20-%20final%20pre-pub%20version.pdf).

**Verified:** Distributional, recognition and procedural justice are the paper's three central dimensions. A locality-level service index addresses only part of the distributional question. It does not, by itself, establish stakeholder participation, recognition of all vulnerable groups, or complete energy justice. Keep the framework as motivation and explicitly delimit the dimensions evaluated.

### 8. Rosales-Asensio et al. (2022): critical-customer resilience

Rosales-Asensio, E., Elejalde, **J. L.**, Pulido-Alonso, A., and Colmenar-Santos, A. (2022). *Resilience framework, methods, and metrics for the prioritization of critical electrical grid customers*. Electronics, 11(14), 2246. DOI: [10.3390/electronics11142246](https://doi.org/10.3390/electronics11142246). [Author-institution record](https://scienceportal.tecnalia.com/es/publications/resilience-framework-methods-and-metrics-for-the-prioritization-o/).

**Verified from institutional abstract:** The paper weights critical customers in resilience metrics and explicitly makes weighting system- and operator-dependent. Examples include control centers, fire stations and hospitals. The exact hospitals/water/bakeries triad and the draft's particular minimum-load wording were not verified from full text. Attribute the project categories to the proposal, retain the broader resilience rationale, and correct Elejalde's initials.

### 9. Irwin et al. (2020): electricity and health

Irwin, **B. R.**, Hoxha, **K.**, and Grépin, K. A. (2020). *Conceptualising the effect of access to electricity on health in low- and middle-income countries: A systematic review*. Global Public Health, 15(3), 452–473. [Publisher paper, DOI 10.1080/17441692.2019.1695873](https://www.tandfonline.com/doi/full/10.1080/17441692.2019.1695873).

**Verified from publisher abstract:** There were 33 included studies: 23 on electrification and 10 on reliability. The review reports associations between reliability and health outcomes while explicitly describing overall evidence quality as weak. Correct the draft's first two author initials and preserve that qualification. This supports concern for healthcare reliability; it does not establish a quantitative benefit of our controller, a causal mortality estimate, or the specific bakery category.

### 10. Nduhuura et al. (2021): household outage impacts

Nduhuura, P., Garschagen, M., and Zerga, A. (2021). *Impacts of electricity outages in urban households in developing countries: A case of Accra, Ghana*. Energies, 14(12), 3676. DOI: [10.3390/en14123676](https://doi.org/10.3390/en14123676). [Author-institution full paper](https://repository.pauwes-cop.net/bitstream/handle/1/443/energies-14-03676.pdf?isAllowed=y&sequence=1).

**Verified:** The study used 496 household questionnaires and found associations between reported outage effects and household circumstances, including socioeconomic disadvantage. This is observational evidence about outages in Accra. It does not identify Fixed Rotation as the causal mechanism, prove Accra is the most comparable city to Khartoum, or establish how much a simulated fairness improvement changes human welfare.

### 11. Huang et al. (2020): emergency-control benchmark

Huang, Q., Huang, R., Hao, W., Tan, J., Fan, R., and Huang, Z. (2020). *Adaptive power system emergency control using deep reinforcement learning*. IEEE Transactions on Smart Grid, 11(2), 1171–1182. DOI: [10.1109/TSG.2019.2933191](https://doi.org/10.1109/TSG.2019.2933191). [Authors' preprint](https://arxiv.org/abs/1903.03712), [PNNL publication record](https://www.pnnl.gov/publications/adaptive-power-system-emergency-control-using-deep-reinforcement-learning).

**Verified in preprint:** DQN controls dynamic braking and under-voltage shedding, with two-area and modified IEEE 39-bus cases. The preprint calls its platform RLGC; the PNNL abstract uses DeepGrid. Specify the version if naming it. Comparisons against MPC and robustness cases are task-specific; neither their speed nor frequency/voltage behavior establishes our hourly allocation controller's performance.

### 12. Larik et al. (2019): conventional methods are not all fixed

Larik, R. M., Mustafa, M. W., and Aman, M. N. (2019). *A critical review of the state-of-art schemes for under voltage load shedding*. International Transactions on Electrical Energy Systems, 29(5), e2828. [Publisher paper, DOI 10.1002/2050-7038.2828](https://onlinelibrary.wiley.com/doi/10.1002/2050-7038.2828).

**Verified from publisher full text:** The review distinguishes static/dynamic, open/closed-loop and event/response-based schemes. The draft's claim that even sophisticated conventional schemes necessarily shed one fixed percentage contradicts these distinctions. Its examples also involve differing timescales. Use the review to classify approaches, not to assert uniform rigidity or millisecond requirements for every load-shedding task. The quoted blackout population counts were not separately checked against original incident reports.

### 13. Ramchurn et al. (2012): research agenda, not a dominance theorem

Ramchurn, S. D., Vytelingum, P., Rogers, A., and Jennings, N. R. (2012). *Putting the “smarts” into the smart grid: A grand challenge for artificial intelligence*. Communications of the ACM, 55(4), 86–97. DOI: [10.1145/2133806.2133825](https://doi.org/10.1145/2133806.2133825). [Authors' institutional manuscript](https://web-archive.southampton.ac.uk/www.orchid.ac.uk/eprints/88/1/ramchurn_etal_smart_grid.pdf).

**Verified:** This is an agenda for AI and decentralized smart-grid coordination. Do not translate that agenda into a proof that classical optimization cannot manage grids or that learning is always necessary. Its broader discussion does not decide which controller wins the present small, fully specified allocation problem.

### 14. Ji et al. (2019): microgrid scheduling

Ji, Y., Wang, J., Xu, J., Fang, X., and Zhang, H. (2019). *Real-time energy management of a microgrid using deep reinforcement learning*. Energies, 12(12), 2291. [Publisher paper, DOI 10.3390/en12122291](https://www.mdpi.com/1996-1073/12/12/2291).

**Verified from publisher abstract:** DQN minimizes daily microgrid operating cost, with CAISO data. The abstract emphasizes learning without an explicit uncertainty predictor. Do not claim this establishes one universally standard forecast-based state representation. The project's current causal observation design and any demand-forecasting demonstration must be described separately; a forecasting literature review is not evidence that forecasts are inputs to the trained PPO policy.

### 15. Wang et al. (2021): correct authors and comparator

Wang, **C.**, Yu, H., Chai, **L.**, Liu, **H.**, and Zhu, **B.** (2021). *Emergency load shedding strategy for microgrids based on dueling deep Q-learning*. IEEE Access, 9, 19707–19715. DOI: [10.1109/ACCESS.2021.3055401](https://doi.org/10.1109/ACCESS.2021.3055401).

**Verified from the original article text in its public full-text copy:** Authors are Can Wang, Hongliang Yu, Lin Chai, Huikang Liu and Binxin Zhu. The comparison is with **implicit enumeration strategy (IES)**, not “intelligent energy load shedding.” The modified IEEE 25-bus case and load-priority contribution indicator are supported. The exact 23.9/35.6 ms values were not independently recovered from tables in this audit; omit those numbers unless checked against the original table and its timing definition. [Original paper full-text copy](https://www.researchgate.net/publication/348851587_Emergency_Load_Shedding_Strategy_for_Microgrids_Based_on_Dueling_Deep_Q-Learning).

## Method references worth adding

These support the actual implementation more directly than general surveys. They do not imply our implementation or results are identical to the original experiments.

- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. (2017). *Proximal policy optimization algorithms*. arXiv:1707.06347. Cite for the clipped PPO objective and minibatch policy updates. [Original paper](https://arxiv.org/abs/1707.06347).
- Huang, S., and Ontañón, S. (2022). *A closer look at invalid action masking in policy gradient algorithms*. FLAIRS, 35. DOI: 10.32473/flairs.v35i.130584. Distinguish this proceedings year from its initial 2020 preprint. Cite for masking in policy-gradient methods; the correctness of this project's critical-service mask still needs its own tests. [Authors' paper and publication metadata](https://arxiv.org/abs/2006.14171).
- Jain, R., Chiu, D., and Hawe, W. (1984). *A quantitative measure of fairness and discrimination for resource allocation in shared computer systems*. DEC Research Report TR-301. Cite the 1984 report year, not the 1998 arXiv deposit year. Define precisely which project service ratios enter the index and how time aggregation works. [Original report record](https://arxiv.org/abs/cs/9809099).

## Remaining checks and defensible wording

The Bedi and Toshniwal (2018) entry identifies *Empirical mode decomposition based deep learning for electricity demand forecasting*, IEEE Access 6, 49144–49156, DOI [10.1109/ACCESS.2018.2867681](https://doi.org/10.1109/ACCESS.2018.2867681). The publisher page could not be inspected during this audit. Do not retain the precise Chandigarh dataset or superiority details as verified here. [Publisher record](https://ieeexplore.ieee.org/document/8449937). Hevner and Chatterjee (2010) is cited in the prose but has no corresponding entry in the supplied reference list; add a verified book entry if retaining Design Science Research.

An additional check resolved the Wang (2020) entry: Wang, **C.**, Mei, **S.**, Dong, **Q.**, Chen, **R.**, and Zhu, **B.** (2020), *Coordinated load shedding control scheme for recovering frequency in islanded microgrids*, IEEE Access 8, 215388–215398, DOI [10.1109/ACCESS.2020.3041273](https://doi.org/10.1109/ACCESS.2020.3041273). The original article uses **Double-Q learning with two Q tables**, not the Double DQN described in the supplied review, and evaluates a modified IEEE 13-bus system. Correct both the author initials and algorithm label. [Original article full-text copy](https://www.researchgate.net/publication/347274188_Coordinated_Load_Shedding_Control_Scheme_for_Recovering_Frequency_in_Islanded_Microgrids).

Exact numerical improvement claims elsewhere in the chapter (including 75%, 53%, 30%, 40%, and sub-10 ms) were outside this targeted audit. Keep them only after verifying the original metric, baseline, dataset, table and conditions. A summary table should distinguish “not evaluated in this study” from “the method cannot do this.”

Suggested synthesis: The reviewed studies motivate adaptive control, explicit protection of essential services, and fair allocation during shortages. This project tests those ideas in a bounded synthetic Khartoum locality model. It compares PPO, Fixed Rotation and Constrained Optimization under matched inputs, reports which objectives each method meets, and leaves broader physical-grid and social-impact claims to future validation. Any reported PPO advantage must follow the frozen evaluation results, not the literature narrative.

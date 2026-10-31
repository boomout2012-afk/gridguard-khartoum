# Bibliographic notes

These notes record distinctions used when incorporating the authors' supplied
[literature review](source/literature-review-authors.docx) into the thesis. The complete
Harvard bibliography appears in the thesis. The review is selective, rather than a
systematic claim that no related work exists.

| Source | Relevance and boundary |
| --- | --- |
| [Oluwasuji et al. (2020)](https://doi.org/10.1007/s10458-019-09428-8) | Fair load allocation through optimization. Its household utility and fairness formulation differ from GridGuard's locality service ratios. |
| [Heggie et al. (2018)](https://doi.org/10.1016/j.enpol.2018.06.025) | Rationing during prolonged shortage; supports comparing adaptive allocation with conventional scheduling. |
| [Sarwar et al. (2020)](https://doi.org/10.3390/su12156234) | Priority-aware mixed-integer optimization. Explicit protection is not exclusive to learning methods. |
| [Chen et al. (2023)](https://doi.org/10.1016/j.egyr.2023.03.027) | PPO for emergency load shedding. Physical frequency-restoration results do not establish performance in an hourly allocation model. |
| [Huang et al. (2020)](https://doi.org/10.1109/TSG.2019.2933191) | Learned emergency control; its voltage dynamics and benchmarks differ from this study. |
| [Wang et al. (2021)](https://doi.org/10.1109/ACCESS.2021.3055401) | Dueling deep Q-learning for emergency shedding. The comparator IES means implicit enumeration strategy. |
| [Ji et al. (2019)](https://doi.org/10.3390/en12122291) | Microgrid energy management. Storage and pricing introduce mechanisms absent from GridGuard. |
| [Bedi and Toshniwal (2018)](https://doi.org/10.1109/ACCESS.2018.2867681) | Electricity-demand forecasting. The thesis cites the method without transferring unverified dataset or superiority claims. |
| [García and Fernández (2015)](https://jmlr.org/papers/v16/garcia15a.html) | Distinguishes reward preferences from restrictions on unsafe exploration. |
| [Rosales-Asensio et al. (2022)](https://doi.org/10.3390/electronics11142246) | Context-dependent critical-customer priorities. Hospital, Water and Bakery are this project's modeling choices, not a universal prescribed list. |
| [Jenkins et al. (2016)](https://doi.org/10.1016/j.erss.2015.10.004) | Energy justice provides context; a numerical allocation index does not establish social justice. |
| [Afzal et al. (2020)](https://doi.org/10.1049/iet-gtd.2020.0531) | Power-system resilience review. The bibliography uses the journal issue year, distinguishing it from later online metadata. |

## Method references

- [Schulman et al. (2017)](https://arxiv.org/abs/1707.06347): PPO's clipped policy objective.
- [Schulman et al. (2016)](https://arxiv.org/abs/1506.02438): generalized advantage estimation; 2016 is the conference year.
- [Huang and Ontañón (2022)](https://doi.org/10.32473/flairs.v35i.130584): invalid-action masking in policy-gradient methods; 2022 is the proceedings year.
- [Jain, Chiu and Hawe (1984)](https://arxiv.org/abs/cs/9809099): the fairness index; 1984 is the report year, rather than the later repository deposit date.

Sudan-focused contextual and policy sources motivate the research setting. They do not
provide the synthetic hourly locality traces used in the experiments. Claims about
response time, physical-network performance or universal algorithm superiority are
not transferred from these references to GridGuard.

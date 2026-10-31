# Research guide

The [thesis](../output/pdf/updated/gridguard-khartoum-thesis.pdf) presents the model,
methods and interpretation. The [presentation](../output/pdf/updated/gridguard-khartoum-project-presentation.pdf)
summarizes the project, and the [application pages](../output/pdf/gridguard-khartoum-gui-pages.pdf)
show the interface.

## Experiments

| Study | Comparison and sample | Evidence |
| --- | --- | --- |
| Main control experiment | Fixed Rotation, Constrained Optimization and five independently trained PPO policies on 960 matched conditions | [Results](../data/evidence/focused-ppo-study.json), [training](../data/evidence/focused-ppo-training.json), [protocol](FOCUSED_PPO_PROTOCOL.md) |
| Additional baselines | Critical-aware Fixed Rotation and Fairness Rule on the original 960 conditions, plus critical-supply stress cases | [Results](../data/evidence/committee-baseline-audit.json), [protocol](COMMITTEE_BASELINE_PROTOCOL.md) |
| Learning-stage comparison | Each selected PPO actor against its own imitation-only initialization and four deterministic baselines on 960 new conditions | [Results](../data/evidence/thesis-learning-audit.json), [protocol](THESIS_LEARNING_AUDIT_PROTOCOL.md) |
| Demand forecasting | Ridge AR(3), persistence and a trailing mean on 320 held-out demand traces | [Results](../data/evidence/proposal-supplement.json) |

Each set of 960 conditions consists of 320 independent base draws with three coupled
shortage variants. Five policies are five independent training runs; conditions are
not additional independent policies. Training and validation histories are separate
from the final-test results.

## Reading the results

PPO and the constrained optimizer share critical protection and feasible choices.
Fixed Rotation has a different action design, so its critical-interruption disadvantage
cannot establish a benefit from learning alone. Critical-aware rotation addresses
that comparison directly.

The optimizer maximizes the current-hour objective. PPO improves the main study's
mean daily reward and Jain fairness, while delivering less energy and switching more
often. It does not meet the full declared improvement criterion against that optimizer.
The study does not compare it with a full-day optimum.

The fresh learning-stage study measures fine-tuning plus validation selection, including
additional computation. It does not isolate the PPO algorithm or compare equal training
budgets. Its reward improvement coexists with lower energy delivery. Higher time-averaged
Jain fairness also does not ensure higher final service to the worst-served locality.

Primary intervals use the original study's adjusted 97.5% paired hierarchical bootstrap.
The additional analyses report descriptive pointwise 95% intervals. Training-curve
bands show between-run standard deviation, not confidence intervals.

## Application and source

Live Scenario uses the validation-selected policy from training seed 8111, checkpoint
2500. Results displays the five-run study and the additional-baseline comparison.
The newer learning-stage analysis is reported separately in the thesis.

Scientific calculations are under `src/gridguard/`; presentation models and Qt views
are under `src/gridguard/ui/`. Saved results, policy identities and experiment protocols
allow a reviewer to trace each reported comparison to its inputs.

All demand and supply are synthetic. Administrative boundaries supply geographic
context rather than an electrical-network model. These experiments assess a bounded
decision-support simulation, not operation of a physical grid.

See [setup and reproduction](../SUBMISSION_README.md) for commands and
[bibliographic notes](LITERATURE_REVIEW_AUDIT.md) for reference distinctions used in the thesis.

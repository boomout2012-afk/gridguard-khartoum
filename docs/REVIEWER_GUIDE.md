# Reviewer guide

GridGuard Khartoum is a graduation project at the University of Khartoum, Faculty of
Mathematical Sciences and Informatics, for the Bachelor of Computer Science degree.
The supervisor is Mrs. Abubakr Siraj. The three project authors and index numbers are
recorded in the [shared cover metadata](academic-cover.json) used by the academic documents.

## Suggested reading order

1. Read the [current thesis](../output/pdf/updated/gridguard-khartoum-thesis.pdf) for the
   problem, methods, results and limitations. Its editable source is
   [FOCUSED_THESIS.md](FOCUSED_THESIS.md).
2. Use the [presentation](../output/pdf/updated/gridguard-khartoum-project-presentation.pdf)
   for the defense overview and the [GUI collection](../output/pdf/gridguard-khartoum-gui-pages.pdf)
   for all application pages and detail tabs.
3. Review the [current training/evaluation protocol](FOCUSED_PPO_PROTOCOL.md), then the
   evidence files below. The protocol states the outcome criteria before final evaluation.
4. Follow the implementation map and run the source checks. The
   [submission guide](../SUBMISSION_README.md) provides longer build and reproduction commands.

## What the current comparison establishes

The main comparison is constrained PPO against Fixed Rotation and Constrained Optimization.
Five independent PPO runs each completed 3,000 iterations, totaling 92,160,000 training
transitions. Final evaluation uses 320 independent synthetic base draws, each with matched
Mild, Moderate and Severe variants: 960 conditions. Five selected PPO policies, their five
iteration-600 references and the two primary baselines give 11,520 evaluated trajectories.

PPO passes the complete declared comparison against Fixed Rotation. Against the constrained
current-hour optimizer, PPO improves fairness and daily reward, but its utilization loss
means the overall comparison does not pass. A positive reward difference alone is not a
successful overall comparison. The optimizer evaluates the same critical-feasible actions
and causal state as PPO; it does not solve the full-day optimum.

The **Results** page reports the equal-weight five-run study. **Live Scenario** uses the
validation-selected policy, training seed 8111 at checkpoint 2500, for one user-selected
scenario. These quantities have different aggregation levels and should not be expected
to have identical values. Live results never replace frozen study values.

Fixed Rotation uses Off and Full Service and is supply-blind. The constrained methods
explicitly protect critical service, so their advantage over Fixed Rotation cannot be
attributed to learning alone. Fairness Rule demonstrations initialize PPO; that teacher
role is disclosed in the protocol rather than presented as a primary comparison result.
Training bands show mean plus/minus sample standard deviation; the between-run variance
uses denominator four. Those bands are not confidence intervals. The two primary comparison
claims use adjusted 97.5% paired hierarchical bootstrap intervals, as defined in the protocol.

All findings are limited to synthetic scenarios. There are five independent training runs,
not 960 independent policies. Forecast evaluation is separate from control. Administrative
geography supplies display context and does not model electrical-network topology.

## Evidence and document sources

| File | Role |
| --- | --- |
| [focused-ppo-training.json](../data/evidence/focused-ppo-training.json) | Training configuration, seeds, all checkpoint summaries, telemetry and selected model identities |
| [focused-ppo-study.json](../data/evidence/focused-ppo-study.json) | Current full-precision outcomes, per-condition results, intervals and criterion decisions |
| [focused-ppo-training-diagnostics.json](../data/evidence/focused-ppo-training-diagnostics.json) | Training diagnostic means, sample SD and between-run variance |
| [focused-ppo-validation.json](focused-ppo-validation.json) | Independent replay and evidence-validation record |
| [proposal-supplement.json](../data/evidence/proposal-supplement.json) | Additional forecast and earlier three-controller evaluation; separate from the current five-run study |
| [Deployed PPO weights](../data/model/focused-ppo-study/seed-8111-selected.npz) | The deployed actor; sibling files retain every run's selected, iteration-600 and imitation-only weights |
| [GUI/document refresh validation](gui-document-refresh-validation.json) | Final PDF, deck, screenshot, logo and executable identities from the academic refresh |
| [Cover metadata](academic-cover.json) | Shared names, index numbers, institution, degree, supervisor and submission date |

Hashes bind evidence to its declared models and scientific sources. They detect a changed
file; an unsigned checksum does not authenticate its author. Match any review or validation
record to its exact scope and recorded hashes. A historical passing record does not certify
later source edits.

## Implementation map

Scientific modules do not import Qt. The application presents validated results rather
than calculating scientific outcomes inside widgets.

```text
domain -> scenarios -> controllers/simulation -> metrics/evaluation/evidence
                                                    |
                                                    v
                                            presentation state -> Qt views
```

| Concern | Source |
| --- | --- |
| Scenario inputs and dispatch | [scenarios.py](../src/gridguard/scenarios.py), [simulation.py](../src/gridguard/simulation.py) |
| Constrained action selection | [constrained.py](../src/gridguard/constrained.py), [constrained_ppo.py](../src/gridguard/constrained_ppo.py) |
| PPO training and study | [ppo_training.py](../src/gridguard/ppo_training.py), [focused_training.py](../src/gridguard/focused_training.py) |
| Current result integrity | [focused_results.py](../src/gridguard/focused_results.py) |
| Deployed policy identity | [live_policy.py](../src/gridguard/live_policy.py) |
| Forecasting and interruption durations | [forecast_model.py](../src/gridguard/forecast_model.py), [duration_metrics.py](../src/gridguard/duration_metrics.py) |
| Live comparison and interface | [live_runner.py](../src/gridguard/ui/live_runner.py), [window.py](../src/gridguard/ui/window.py) |

The [implementation contract](../SPEC.md) and [scope traceability](PROPOSAL_TRACEABILITY.md)
retain the development history and accepted scope. The current study protocol above defines
the constrained optimizer and longer PPO comparison used in the submitted thesis.

## Verification and reproduction

Start with `python tools/verify_review_package.py`. It verifies the files listed in
[review-package-manifest.json](review-package-manifest.json), including the final documents,
model weights, evidence and scientific sources. It uses only the Python standard library.
This check establishes byte integrity; the tests and CUDA runs below check behavior.
The [repository review map](REPOSITORY_REVIEW.md) explains the manifest and artifact scope.

Run these commands from the repository root after the locked setup in the README:

```powershell
uv run --no-sync gridguard-doctor
uv run --no-sync gridguard-evaluate --verify-evidence
uv run --no-sync pytest -ra -m "not source_archive and not release_artifact"
```

The last command is the mandatory presentation-GPU gate and requires zero skipped selected
tests. The portable CI command in the README excludes GPU tests and does not establish
GPU readiness. The optional source-archive parity gate audits the separate read-only research
archive. The normal package build does not require the sibling research archive.

The strict post-build `release_artifact` gate belongs to the historical Git-attested release
layout and manifest. It is distinct from the current source/GPU gate and from a portable
submission's per-file inventory verifier. This distinction preserves the historical gate
rather than changing its expected inputs to fit a later package.

Verify the current focused study directly against its pinned training and result hashes:

```powershell
$trainingSha = (Get-Content data/evidence/focused-ppo-training.sha256).Split()[0]
$studySha = (Get-Content data/evidence/focused-ppo-study.sha256).Split()[0]
uv run --no-sync python tools/evaluate_focused_ppo.py --verify --training data/evidence/focused-ppo-training.json --training-sha256 $trainingSha --output data/evidence/focused-ppo-study.json --sha256 $studySha
```

The [submission guide](../SUBMISSION_README.md#reproduce-the-current-evidence) explains fresh
evaluation and retraining into new paths. It also describes the portable academic rebuild:
thesis formulas and graphs can be regenerated from the included sources; the editable
PowerPoint can be exported through Microsoft PowerPoint with the supplied fonts.

## Historical files

[PROJECT_HISTORY.md](PROJECT_HISTORY.md) preserves the earlier overview and detailed release
contracts. The original [report](../output/pdf/gridguard-khartoum-final-report.pdf),
[deck](../output/presentation/gridguard-khartoum-defense.pptx),
[training provenance](../data/model/training-provenance.yaml) and
[release manifest](../RELEASE_MANIFEST.json) are historical test and audit fixtures.
Their model, scenario counts and claims differ from the current focused study. Review the
current documents linked at the top of this guide for the submission conclusions.

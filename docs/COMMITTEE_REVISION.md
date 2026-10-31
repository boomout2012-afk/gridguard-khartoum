# Response to the committee review

Review received: 31 October 2026. This record maps the supplied technical review to
the revised manuscript, additional experiments and verification evidence.

The version available when this revision began is preserved in Git commit
`10ecbb425ddb05e36e602505a160a46345a4e635`. Its original study, training records,
models and protocols remain frozen. New experiments prompted by the review are
identified as follow-up analyses; they were not part of the original protocol.

## Requested corrections

| ID | Committee concern | Required resolution |
| --- | --- | --- |
| C01 | The declared improvement criterion is missing from the thesis | State every threshold, confidence interval, severity guard, eligibility and safety condition; distinguish the original freeze from external preregistration |
| C02 | Reward, usable power, switching, states, actions and transitions lack formal definitions | Give the complete implemented mathematical definitions, units and observation/history conventions |
| C03 | Fixed Rotation lacks Critical Only, confounding the attribution to AI | Add a critical-aware rotating baseline and distinguish mechanical critical protection from learned allocation |
| C04 | The Fairness Rule teacher is absent from the final comparison | Evaluate the teacher on the same final scenarios with the frozen final PPO policies, with paired uncertainty |
| C05 | The reward difference lacks practical context | Report its relative scale and the corresponding measured energy and service tradeoffs; do not convert a composite reward directly into MWh |
| C06 | The word causal suggests causal inference | Use history-only, available-at-decision-time or non-anticipative terminology where appropriate |
| C07 | Front matter and figure/table references are incomplete | Provide contents, figure/table lists, abbreviations and meaningful text references to every figure/table |
| C08 | Appendix labels a GitHub link as Google Drive | Check the current document and provide a correctly labelled reproducibility link |
| C09 | Bedi and Toshniwal is cited but missing; Afzal is uncited | Check the current bibliography and supplied literature review against primary sources, then reconcile Harvard citations |
| C10 | Captions of Figures 5.2–5.5 describe the wrong plots | Match each caption and in-text interpretation to the actual graph and split |
| C11 | Tables 11 and 13 ambiguously label different reward results | Explicitly distinguish validation selection from final-test evaluation |
| C12 | Discussion cites a utilization loss before results introduce it | Present the numerical result and uncertainty before interpreting it |
| C13 | Arabic and English abstracts disagree | Provide equivalent summaries of objectives, methods, comparisons, findings and limitations |
| C14 | Related work says eight themes but has nine subsections | Make the stated structure match the actual sections |
| C15 | Critical demand exceeding supply is not tested | Add a labelled stress test of the infeasible critical-floor fallback and report the actual shortfall |

## Revision principles

- Preserve the original models, scientific source snapshots and evidence.
- Freeze the follow-up baseline and stress-test design before computing its outcomes.
- Use matched scenarios, the same feasibility rule and explicit aggregation units.
- Report adverse and inconclusive findings alongside improvements.
- Keep the thesis, slides, application evidence and repository documentation consistent.
- Resolve each comment against the current file: do not introduce a reference or numerical
  claim solely because an earlier review mentions it.

## Resolution in the revised manuscript

The reviewed document was the supplied Word thesis, whose SHA-256 is
`d6f5cef5d56bdcb7f6c78989d68f8c586ba9b9289ecf5b765ebd34a8e26d3d59`.
Its content differs from the repository's earlier PDF. The revision reconciles both
against the frozen scientific evidence and preserves the Word declaration, dedication
and source-code appendix. The authors confirmed **Dr. Abubakr Siraj** as the supervisor.

| Concern | Resolution and evidence |
| --- | --- |
| C01 | Section 5.4 states all original thresholds and adjusted intervals, every-shortage guards, validation eligibility and trajectory-level safety conditions. It distinguishes the internal pre-training hash freeze from external preregistration. |
| C02 | Sections 3.4, 3.6 and 3.8–3.9 define dispatch, admissible actions, simulator state, the 46-feature observation, transitions and all five reward components with units, time convention and numerical floors. |
| C03 | Section 5.7 compares the original schedule with Critical-aware Fixed Rotation. Both the critical-aware baseline and PPO preserve normal-scenario critical service; PPO's additional served energy is measured separately. |
| C04 | Section 5.7 includes the unchanged teacher with its original tolerance, all five final PPO policies, severity variation and the adverse seed-level finding. The post-hoc analysis does not replace the primary experiment. |
| C05 | Section 5.5 relates the reward gain to its scale and components, and reports the directly measured −19.88 MWh/day difference against the optimizer. Section 5.7 gives the teacher energy tradeoff. |
| C06 | The methods use history-only and nonanticipative terminology and explicitly distinguish it from causal identification. |
| C07 | The document builder supplies contents, figure/table lists, abbreviations, stable destinations and page references. The narrative references every numbered table and figure. |
| C08 | Appendix B contains the actual GitHub URL and reproducibility instructions, replacing the reviewed Word placeholder. |
| C09 | Bedi and Toshniwal (2018) is restored to the Word bibliography. Afzal et al. is correctly identified as a 2020 publication and cited in the resilience discussion. The three corrupted Word DOI strings are reconciled with verified records. |
| C10 | Figures 5.2–5.5 explicitly describe paired optimizer effects, validation progress, training diagnostics with sample SD, and between-run validation variance respectively. |
| C11 | Final-test checkpoint results and validation-selection results have distinct captions, column labels and interpretation. |
| C12 | Section 5.6 reports the −0.2417 percentage-point utilization change and its interval before the discussion interprets it. |
| C13 | English and Arabic abstracts include the optimizer criterion, teacher tradeoffs and infeasible-supply result consistently. |
| C14 | Related work identifies nine connected subsections: eight themes followed by the study-position synthesis. |
| C15 | Section 5.7 separately reports the two critical-supply stress families, partial critical interruption, unmet energy, complete outages, fallback hours and power-balance checks. |

The additional experiment is specified in [COMMITTEE_BASELINE_PROTOCOL.md](COMMITTEE_BASELINE_PROTOCOL.md)
and recorded in [committee-baseline-audit.json](../data/evidence/committee-baseline-audit.json).
Original models and original primary decision outcomes remain unchanged. The revised
results support specific benefits and tradeoffs; they do not establish a universal
advantage of PPO over optimization or a field-validated operating policy.

## Verification of the revision

The revised PDF and editable Word document share the same resolved manuscript,
equations, nine figures and eighteen numbered tables. The Word version retains the
reviewed academic template and has editable equations and refreshed navigation fields.
The presentation and application display the same original and follow-up evidence.
All rendered document pages, slides and GUI views were inspected. The
[revision validation record](committee-revision-validation.json) records the exact
artifact identities, source/GPU checks and rebuilt-application verification.

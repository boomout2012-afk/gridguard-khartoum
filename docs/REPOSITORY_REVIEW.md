# Repository and artifact verification

The [reviewer guide](REVIEWER_GUIDE.md) explains the current scientific claims and
implementation. This page distinguishes the submission artifacts from the retained
historical release records.

## Verify a downloaded repository

From the repository root, with Python 3.12 installed:

```powershell
python tools/verify_review_package.py
```

This standard-library check requires no GPU, application executable, training run or
private audit directory. It verifies the byte size and SHA-256 of every file listed in
[review-package-manifest.json](review-package-manifest.json), including the current
academic documents, shared cover metadata, logos, screenshots, frozen evidence and
model weights. It fails for missing or altered files and unsafe manifest paths. It
does not run the application or reproduce the statistical analysis. Checksums establish
consistency with this repository's manifest; they do not independently authenticate it.

The current academic deliverables are:

| Artifact | Location |
| --- | --- |
| Thesis | [Current thesis PDF](../output/pdf/updated/gridguard-khartoum-thesis.pdf) |
| Presentation | [Current presentation PDF](../output/pdf/updated/gridguard-khartoum-project-presentation.pdf) |
| Editable presentation | [Current PowerPoint](../output/presentation/updated/gridguard-khartoum-defense.pptx) |
| Complete GUI capture | [GUI pages PDF](../output/pdf/gridguard-khartoum-gui-pages.pdf) |

The [focused protocol](FOCUSED_PPO_PROTOCOL.md), [training evidence](../data/evidence/focused-ppo-training.json),
[study evidence](../data/evidence/focused-ppo-study.json) and
[training diagnostics](../data/evidence/focused-ppo-training-diagnostics.json) support the
current results. The Results page represents all five independent runs; Live Scenario
uses the validation-selected seed-8111 checkpoint-2500 model for one matched scenario.

## Source and runtime verification

Follow the [reviewer guide](REVIEWER_GUIDE.md) for environment setup, tests, evidence
verification and the application launch. GitHub Quality checks exercise the portable
source checks. CUDA checks require the declared NVIDIA presentation environment;
historical release-artifact checks additionally require their original built package.
The [submission guide](../SUBMISSION_README.md) describes the separate portable Windows
package and the document-build commands. That large executable package is not part of
a Git clone.

The [31 October 2026 validation record](repository-validation.json) records 1,569 passing
source/GPU tests with zero failures or skips, Ruff checks, an isolated installed-wheel
smoke test and a successful integrity check of the exact staged Git checkout. Eleven
optional historical archive/release tests were explicitly deselected. The GitHub
**Quality** workflow provides the separate portable CI result for each published commit.

Git attributes preserve the raw bytes of SHA-bound evidence, models, manuscripts,
protocols and capture sources. A regression test stages the published files and checks
them out in a temporary Git repository with Windows line-ending conversion enabled,
then compares every file's SHA-256. Do not fix a provenance mismatch by replacing a
stored hash without investigating the changed bytes.

## Historical records

`RELEASE_MANIFEST.json`, the original report and presentation outputs, original accepted
benchmark evidence, and the corresponding release tests retain the earlier release
contract. They remain useful for provenance and regression checks; their pass status and
dates do not certify the current academic submission.

The manual **Historical presentation release** workflow runs that original packaging
contract on its self-hosted presentation GPU runner. It is deliberately separate from
current submission verification. Earlier development notes are collected in
[PROJECT_HISTORY.md](PROJECT_HISTORY.md).

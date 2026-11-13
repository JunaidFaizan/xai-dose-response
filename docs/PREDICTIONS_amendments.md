# A Dose-Response Evaluation of Explanation Metrics under Controlled Shortcut Reliance in CNN Image Classifiers

## Pre-registration amendments

Amendments to the design that are written and committed before the models they
describe are trained. The Pet predictions P1–P4 and their recorded verdicts in
`docs/PREDICTIONS_pet.md` are not changed by anything here.

## Amendment A1 (2026-11-13): extra doses to fill the MNIST and PneumoniaMNIST reliance gaps

Written and committed on 2026-11-13, before any of the models below was
trained. No model listed here existed when this was written.

### Purpose

The existing 192 models leave gaps in reliance where it changes fastest with
dose (seed means from `results/dose_response/models.csv`):

- MNIST patch: 8.5% at dose 0.005, 29.6% at 0.01, 55.6% at 0.02, 76.2% at 0.05.
  Seeds scatter widely at the steep doses (0.01: 9.1–62.2%).
- MNIST texture: 9.1% at dose 0, 48.9% at 0.005, 70.4% at 0.01, 92.9% at 0.02
  (0.005: 14.2–76.8%; 0.01: 38.9–92.0%).
- PneumoniaMNIST patch: 32.4% at dose −0.05, 64.9% at 0 (21.3% at −0.1).
- PneumoniaMNIST texture: 38.6% at dose −0.05, 92.7% at 0 (30.9% at −0.1).

The added models place more models in the 30–90% reliance range, so the
switch-point estimates and the low-to-mid reliance comparisons rest on more
than one or two doses. This was recommended by the statistics reviewer
(`docs/audit/01_statistics_rigour.md`, R1) and listed as MD8 in
`docs/audit/00_combined_audit.md`.

### Added models

Same pipeline, architecture, epochs, data and evaluation as the existing
MNIST and PneumoniaMNIST models (`dose_response.py`, 100 test images per
model, oracle, method, placebo and random-control scores).

| Base | Shortcuts | Doses | Seeds | Models |
|---|---|---|---|---|
| MNIST | patch, texture | 0.0075, 0.015, 0.03 | 42, 1, 2 | 18 |
| MNIST | patch, texture | 0.0075, 0.015 (steepest region) | 3, 4, 5 | 12 |
| PneumoniaMNIST | patch, texture | −0.075, −0.025 | 42, 1, 2 | 12 |

Total: 42 new models (234 with the existing 192). Model names follow
`model_name()`, e.g. `mnist_patch_dose+0p75_seed42`, `mnist_texture_dose+1p5_seed3`,
`mnist_patch_dose+003_seed1`, `pneumonia_mnist_texture_dose-7p5_seed2`,
`pneumonia_mnist_patch_dose-2p5_seed42`. Existing model names are unchanged.

### Commitments

- No new predictions about the metrics are made. The added models only add
  data points to the existing analyses.
- Every added model is reported and included, whatever its reliance. No model
  is dropped or re-run with another seed after its result is seen.
- The doses and seeds above are fixed. If any further doses or seeds are added
  later, they will be recorded in a new dated amendment before training.
- Analyses are reported both for the original 192 models (as before) and for
  all models including these, labelled clearly.

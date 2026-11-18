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

### Completion record (added 2026-11-13, after training)

This section records what happened; it does not change the design above.

- Training started at 2026-11-13 10:31:36 +05:30, after this amendment was
  committed (8738b32, 10:30:56 +05:30). Training, evaluation, placebo and
  random controls finished at 12:04 the same day (about 1.5 CPU hours).
- All 42 models listed above were trained, evaluated and controlled. None was
  dropped or re-run.
- Extra seeds: the six seeds (42, 1, 2, 3, 4, 5) at the steepest MNIST doses
  0.0075 and 0.015 are the ones listed above. That is the maximum of six seeds
  per dose allowed by the plan, so no further seeds or doses were added and no
  further amendment was needed.

## Amendment A2 (2026-11-13): extra natural-image doses (SD2) and the pretrained Pet setting (SD1)

Written and committed on 2026-11-13, before any of the models below was
trained. No model listed here existed when this was written. The predictions
for these models (P5–P9) are in `docs/PREDICTIONS_pretrained.md`, committed
together with this amendment.

**Third-party timestamp:** Zenodo, DOI
[10.5281/zenodo.24285532](https://doi.org/10.5281/zenodo.24285532), published
2026-11-16 16:41 UTC. The archived files are this file and
`docs/PREDICTIONS_pretrained.md` exactly as committed in `f2bff81`
(MD5 f1e3209f3c8b5f57f2508e9b6833ffab for this file).

**Deviation, recorded 2026-11-16.** This amendment asked for the upload
before the first Kaggle session that trains an SD1 or SD2 model. The two SD1
pilot models (`pet_resnet18_pretrained_patch_dose+000_seed7` and
`..._dose-050_seed7`) were trained in an interactive Kaggle run shortly
before the upload. Only their reliance and accuracies were read, as the pilot
rule allows. No confirmatory SD1 or SD2 model existed at upload time. The
repository was private then, so the Zenodo record is the first public
timestamp. The pilots are retrained in the committed session-1 run, and the
SD1 completion note uses the values in that run's `a2_pilot_report.json`.

### Purpose

The second audit (`docs/audit2/00_combined_audit2.md`, N1, M6, SD1, SD2)
found two problems that need new models:

- **Reliance gaps on natural images (SD2).** Corrected reliance r_c jumps
  across dose ranges with no models in between (`results/dose_response/models.csv`):
  - CIFAR-10 ResNet-18: r_c 31.8–36.5 at dose −0.1, 95.8–97.0 at dose 0.
  - CIFAR-10 small CNN: r_c 41.1–42.7 at dose −0.1, 96.0–96.3 at dose 0.
  - Pet (from scratch): r_c 22.9–41.1 at dose −0.2, 60.1–82.7 at dose −0.1.

  The behavioural switch lies inside these gaps. It is not identified on
  CIFAR-10 and not bracketed on Pet.
- **All models are trained from scratch (SD1).** The setting in which
  practitioners use these metrics is a pretrained network. SD1 repeats the
  Pet design with ImageNet-pretrained weights as a confirmatory test.

### SD2: added doses for existing settings

Same pipeline, architecture, epochs, data, shortcut and evaluation as the
existing models of each setting (`dose_response.py`, 100 test images per
model, oracle, method, placebo, random-control and behaviour scores).

**CIFAR-10 ResNet-18 (`cifar10_resnet18`, Kaggle GPU).**

1. Pilot: dose −0.05, seed 42. After training, only its reliance and
   accuracies are read. No metric, oracle, method or behaviour scores are
   computed for it until the final dose set is recorded (step 3).
2. Dose rule, applied to the pilot's r_c:

   | Pilot r_c at dose −0.05 | Final doses |
   |---|---|
   | below 50 | −0.05, −0.03, −0.02, −0.01 |
   | 50 to 85 | −0.08, −0.05, −0.03, −0.01 |
   | above 85 | −0.09, −0.075, −0.06, −0.05 |

3. The pilot's r_c and the resulting dose set are recorded in a dated
   completion note under this amendment, and committed, before any other SD2
   CIFAR-10 model is trained.
4. Seeds 42, 1, 2 at each of the 4 final doses: 12 models. The pilot is one
   of them and is kept, as with the first Pet model.

**CIFAR-10 small CNN (`cifar10`, laptop CPU).** The same 4 final doses as
ResNet-18, seeds 42 and 1 (the existing small-CNN seeds): 8 models.

**Pet from scratch (`pet_resnet18`, Kaggle GPU).** Dose −0.15, seeds 42, 1,
2: 3 models. This dose sits between −0.2 and −0.1, where the gap is. No pilot.
Dose −0.05 is not part of this amendment. Adding it would need a new dated
amendment before it is trained.

SD2 total: 23 models. Model names follow `model_name()`, for example
`cifar10_resnet18_patch_dose-005_seed42`, `cifar10_patch_dose-7p5_seed1` and
`pet_resnet18_patch_dose-015_seed2`.

### SD1: ImageNet-pretrained ResNet-18 on Pet

New base `pet_resnet18_pretrained`. It differs from `pet_resnet18` only in
the network and its training:

- torchvision ResNet-18 with `IMAGENET1K_V1` weights and the standard
  ImageNet stem (7×7 stride-2 convolution and max-pool), with the final layer
  replaced by a 2-class layer. All layers are fine-tuned.
- ImageNet mean/std normalisation inside the model's forward pass. Images,
  perturbation fills (black, blur, inpaint), oracle maps and the shortcut
  therefore stay exactly as for `pet_resnet18`.
- 15 training epochs. The optimiser, learning rate and schedule are set in
  the code and committed before the pilot. They are not changed after it.
- Same data, 128×128 resolution, splits, 12×12 white-patch shortcut (top-left
  for cats, top-right for dogs), dose definition, object oracle (pet mask
  with the shortcut pixels removed), compact size-matched object map,
  metrics, controls and 100 test images per model.

**Pilot (Kaggle GPU).** Doses 0 and −0.5, seed 7. Seed 7 is not one of the
confirmatory seeds, so the pilot models are separate from the confirmatory
set. They are reported as pilots and excluded from the tests of P5–P9. Only
their reliance and accuracies are read before the final doses are recorded.

**Rule: the SD1 doses are finalised after the pilot.** After the pilot, the
author records the pilot's r_c and accuracies and the final SD1 doses in a
dated completion note under this amendment. The note is committed, and added
to the timestamped record, before any confirmatory SD1 model is trained. The
final doses follow this rule unless the note gives a reason to deviate:

| Pilot r_c at dose 0 | Final SD1 doses |
|---|---|
| 60 or above | −0.5, −0.4, −0.3, −0.2, −0.1, 0 (as `pet_resnet18`) |
| below 60 | −0.5, −0.2, 0, +0.1, +0.25, +0.4 (positive doses swap that share of training images for the other class, as on MNIST) |

**Confirmatory set.** The 6 final doses × seeds 42, 1, 2: 18 models. If the
confirmatory set does not reach r_c above 50 on at least two models,
predictions that need models above 50 are reported as "not testable", and
SD1 is reported as a negative result. No doses or seeds are added except by a
new dated amendment written before those models are trained.

### Primary behavioural label for all new models

For every SD1 and SD2 model, the primary per-image behavioural label is
`follows_shortcut_causal`: the model follows the moved shortcut on the
conflicting image, with undetermined images dropped. This is the label
already marked primary in `xai/dose_analysis.py`. The check labels
`follows_shortcut` (moved shortcut followed, no undetermined filter) and
`shortcut_dominant` are always reported next to it, with the undetermined
share. The labels answer different questions. The primary label measures
whether the shortcut is sufficient under conflict. `shortcut_dominant` is
closer to whether it is necessary on aligned images, which is what deletion
measures. No label is treated as ground truth for every metric.

### Commitments

- Every model listed above is reported and included in its analyses,
  whatever its reliance. No model is dropped or re-run with another seed
  after its result is seen. The SD1 pilot models are reported separately as
  pilots, as stated above.
- The only design choices left open are the CIFAR-10 SD2 doses and the SD1
  doses. Each is fixed by the rules above before the models it affects are
  trained.
- Analyses are reported both without the A2 models (the 234 models of the
  original set and A1, as before) and with them, labelled clearly. The Pet
  predictions P1–P4 are judged only on the original 18 Pet models, as
  recorded in `docs/PREDICTIONS_pet.md`.
- SD1 results are reported as a separate setting. They are not pooled with
  `pet_resnet18`.
- SD3 (exact ROAD) and SD4 (watermark shortcut) are not covered by this
  amendment. Their predictions will be written in a new dated amendment
  before any of their runs.

### Completion notes (added 2026-11-17, after session 1, before session 2)

These notes record the pilot results and the final doses fixed by the rules
above. They do not change the design. All values are copied from
`a2_pilot_report.json`, written by the committed Kaggle session-1 run.
No metric, oracle, method or behaviour score of any pilot model had been
computed when these notes were written, and no confirmatory SD1 model or
further SD2 CIFAR-10 model had been trained.

**Third-party timestamp of these notes:** Zenodo version 2 of the record,
DOI [10.5281/zenodo.24333540](https://doi.org/10.5281/zenodo.24333540)
(concept DOI [10.5281/zenodo.24285531](https://doi.org/10.5281/zenodo.24285531)),
published 2026-11-18 07:08 UTC, before session 2. It archives this file as
committed in `3db4a0e` (MD5 6002dc67463714abc580aee3959255d1). The
predictions P5–P9 are unchanged and remain archived in version 1
([10.5281/zenodo.24285532](https://doi.org/10.5281/zenodo.24285532)).

**Note A2-C1 (2026-11-17): SD2 CIFAR-10 doses.** Pilot
`cifar10_resnet18_patch_dose-005_seed42`: reliance 50.61, test accuracy
98.52, test accuracy without the shortcut 57.82, chance floor 4.69,
corrected reliance r_c 48.18, training time 14.68 min. r_c 48.18 is below 50,
so the rule gives the final SD2 CIFAR-10 doses −0.05, −0.03, −0.02, −0.01.
`cifar10_resnet18` is trained at these 4 doses with seeds 42, 1, 2 (12
models). The pilot is one of the 12 and is kept. The small CNN (`cifar10`,
laptop CPU) uses the same 4 doses with seeds 42 and 1 (8 models), as stated
in `kaggle/README.md`.

**Note A2-C2 (2026-11-17): SD1 doses.** Pilot
`pet_resnet18_pretrained_patch_dose+000_seed7`: reliance 36.06, test
accuracy 99.81, test accuracy without the shortcut 93.98, chance floor 6.02,
r_c 31.96, training time 1.38 min. Pilot
`pet_resnet18_pretrained_patch_dose-050_seed7`: reliance 2.56, test accuracy
97.19, test accuracy without the shortcut 97.22, chance floor 2.78, r_c
−0.23. The r_c at dose 0 (31.96) is below 60, so the rule gives the final SD1
doses −0.5, −0.2, 0, +0.1, +0.25, +0.4. With seeds 42, 1, 2 this is 18
confirmatory models. There is no deviation from the rule. The two seed-7
pilots stay outside the confirmatory set, as stated above.

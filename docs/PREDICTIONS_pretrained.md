# A Dose-Response Evaluation of Explanation Metrics under Controlled Shortcut Reliance in CNN Image Classifiers

## Pre-registered predictions: pretrained Pet (SD1) and extra natural-image doses (SD2)

Author: Md Junaid Faizan Shaikh

Date: 2026-11-13. Written and committed before any SD1 or SD2 model was
trained. The design (models, doses, pilots and the primary behavioural label)
is amendment A2 in `docs/PREDICTIONS_amendments.md`.

**Third-party timestamp:** Zenodo, DOI
[10.5281/zenodo.24285532](https://doi.org/10.5281/zenodo.24285532), published
2026-11-16 16:41 UTC. The archived files are this file and
`docs/PREDICTIONS_amendments.md` exactly as committed in `f2bff81`
(MD5 3b7f5a019289c69b2f76f951c12423ff for this file). The two SD1 pilot models
(seed 7, doses 0 and −0.5) were trained on Kaggle shortly before the upload;
see the note under A2. No confirmatory model existed at upload time.

### Settings

- **SD1:** `pet_resnet18_pretrained`, an ImageNet-pretrained ResNet-18
  fine-tuned on the Oxford-IIIT Pet cat-vs-dog task with the 12×12 patch
  shortcut. The confirmatory set is 18 models (6 doses × seeds 42, 1, 2). The
  doses are finalised after the pilot by the rule in A2. The pilot models
  (seed 7) are excluded from every test below.
- **SD2:** extra doses in the reliance gaps of `cifar10_resnet18` (12 models),
  `cifar10` (8 models) and `pet_resnet18` (3 models at dose −0.15), as listed
  in A2. Each setting is analysed with its existing models plus the A2 models.

### Definitions

These are the statistics the predictions use. They are written out here
because the second audit asks for changes to the interval methods (M3).
Whatever code computes them is committed before any SD1 or SD2 result is
analysed.

- **Reliance:** chance-corrected reliance r_c, as in the main study.
- **Metrics:** Deletion (black), Deletion (blur), Deletion (inpaint) and
  Insertion (blur).
- **Preference** for map A over map B: the share of a model's 100 test
  images on which the metric rates A as the better explanation.
- **S, O, R:** the shortcut oracle map, the full object oracle map (pet mask
  with the shortcut pixels removed) and the random map.
- **Over-detection rate (FPR)** of a metric: among models with r_c ≤ 50, the
  share with preference(S over O) > 0.5. It is reported with a Clopper–Pearson
  95% interval and the number of models. Two pre-specified secondary versions
  are reported but do not decide any verdict: the margin FPR (the model's own
  95% preference interval lies above 0.5), and the FPR split into r_c ≤ 10
  and 10 < r_c ≤ 50.
- **Excess shortcut preference** per model: preference(S over O) −
  preference(R over O).
- **Switch:** the r_c where an isotonic fit of preference on r_c crosses 0.5,
  with the ±10-point identification rule, the bracketing check and the
  lowest-reliance edge status (within 5 points of the lowest-r_c model).
- **Behavioural switch:** the r_c where an isotonic fit of the per-model
  follow rate under the primary label (`follows_shortcut_causal`) crosses
  0.5. The same identification and bracketing rules apply.
- **Significance** of a correlation comes from the model-level Spearman
  test. Intervals come from the bootstrap.

### Predictions

**SD1 (pretrained Pet, 18 confirmatory models)**

- **P5. Over-detection persists after pretraining.** The over-detection rate
  of Deletion (black) and of Deletion (blur), comparison S vs O, is above 0.5
  for each of the two metrics.
- **P6. Inpainting and insertion over-detect less.** The over-detection rate
  of Deletion (inpaint) and of Insertion (blur) are each lower than the
  over-detection rates of both Deletion (black) and Deletion (blur).
- **P7. The size-matched comparison switches at the low-reliance edge.** In
  the comparison "shortcut vs size-matched object", the switch lies within 5
  r_c points of the lowest-r_c model, for at least three of the four
  metrics. A metric whose preference is already at or above 0.5 at the
  lowest r_c counts as switching at the edge. A metric whose preference never
  reaches 0.5 does not.
- **P8. The metrics respond to reliance.** Both parts must hold:
  - (a) Spearman ρ between r_c and preference(S over O) is > 0.8 for at least
    three of the four metrics.
  - (b) Spearman ρ between r_c and excess shortcut preference is positive,
    with a one-sided BH-adjusted p < 0.05 over the four metrics, for at least
    three of the four metrics.

**SD2 (extra doses; CIFAR-10 ResNet-18, CIFAR-10 small CNN, Pet from scratch)**

- **P9. Metrics switch before behaviour.** With the A2 models added, in each
  of the three settings:
  - (a) Design check: the behavioural switch is identified. It is reported
    with its bracketing status.
  - (b) For each of the four metrics with an identified metric switch in the
    S vs O comparison, the metric switch lies below the behavioural switch.
    The paired bootstrap interval of (behavioural switch − metric switch)
    must lie entirely above 0.

  P9 holds if (b) holds for every such metric in every setting. If (a) fails
  in a setting, P9 is reported as "not testable" in that setting, and the
  behavioural switch there is reported as a bound (for example "> 43%, not
  identified").

### Pre-specified secondary outcomes (reported, no verdict)

- SD1 class-stratified behavioural AUROCs (pooled and within-model) under all
  three labels, with the undetermined share and its correlation with r_c.
- SD1 `confound.csv` row: the correlation of r_c with accuracy without the
  shortcut, and the partial correlation of preference with r_c given that
  accuracy.
- S–O, R–O and S–R preferences side by side at the lowest SD1 dose.
- An exploratory comparison of SD1 with `pet_resnet18` (from scratch).
  Grad-CAM maps are 4×4 at 128 px on the standard stem. This is reported,
  because Grad-CAM is a method under study, not a metric.

### Commitments

- All results are reported whether or not these predictions hold, with the
  commit of the code that computed them.
- A prediction that cannot be computed is reported as "not testable", with
  the reason. An example is P5 when no SD1 model has r_c ≤ 50.
- The SD1 dose rule, pilot handling and primary label are those in A2. Any
  change needs a new dated amendment before the models it affects are
  trained.
- Predictions for SD3 (exact ROAD) and SD4 (watermark shortcut) are not made
  here. They will be numbered from P10 and written in a new dated amendment
  before any of their runs.

## Outcome

Not yet run. Results are added here after the analysis, without changing the
text above.

### Outcome recorded 2026-11-21 (after the full A2 analysis)

Appended after the analysis; nothing above was changed. The line "Not yet
run" above was written before the analysis and is kept as committed.

Data: the 18 SD1 confirmatory models (Kaggle session 2), the 12 CIFAR-10
ResNet-18 SD2 models (pilot from session 1, 11 from session 2), the 3 Pet
−0.15 models (session 1) and the 8 CIFAR-10 small-CNN SD2 models (laptop).
`dose_response.py analyze` over all 275 registered models (code as of
2dbef8d, unchanged since). The two SD1 pilots (seed 7) are listed in
`results/dose_response/pilots.csv` and are not used. SD1 is analysed as its
own setting and is not pooled with `pet_resnet18`. All numbers are from
`results/dose_response/overdetection.csv`, `responsiveness_corrected.csv`,
`behaviour_switch.csv` and `switch_differences.csv`. Verdicts are judged on
point estimates, as for P1–P4.

**SD1 testability.** Of the 18 SD1 models, 9 have r_c ≤ 50 (r_c −0.08 to
31.21, doses −0.5, −0.2, 0) and 9 have r_c > 50 (96.43 to 98.83, doses
+0.1, +0.25, +0.4). The A2 rule (at least two models above 50) is met, so
P5–P8 are testable.

| Prediction | Result | Verdict |
|---|---|---|
| P5: over-detection rate (S vs O) of black and of blur deletion > 0.5 | black 0.89 (8/9, CP 95% 0.52–1.00), blur 0.56 (5/9, CP 0.21–0.86) | **Holds** (blur's interval reaches well below 0.5; margin FPR 0.44 for both) |
| P6: inpaint and insertion each over-detect less than both black and blur | inpaint 0.33 (3/9), insertion 0.44 (4/9), against black 0.89 and blur 0.56 | **Holds** |
| P7: shortcut vs size-matched object switches within 5 r_c points of the lowest-r_c model (−0.08) for ≥ 3 of 4 metrics | black 1.67, blur 1.66, inpaint −0.04 (all at the lowest-reliance edge); insertion already above 0.5 at the lowest reliance (preference 0.56), which counts as the edge | **Holds** (4 of 4) |
| P8 (a): ρ(r_c, preference S over O) > 0.8 for ≥ 3 of 4 | black 0.86, blur 0.86, inpaint 0.97, insertion 0.90 | (a) holds (4 of 4) |
| P8 (b): ρ(r_c, excess preference) > 0, one-sided BH p < 0.05 over the four metrics, for ≥ 3 of 4 | black 0.11 (p_bh 0.44), blur −0.33 (0.91), inpaint 0.71 (0.0018), insertion 0.37 (0.13) | (b) fails (1 of 4) |
| **P8 overall** (both parts must hold) | | **Fails** |
| P9 (a): behavioural switch identified in each SD2 setting | Pet 54.1 (bracketed), CIFAR-10 ResNet-18 55.2 (bracketed), CIFAR-10 small CNN 60.3 (bracketed) | (a) met in all three settings |
| P9 (b): every identified S vs O metric switch below the behavioural switch, paired bootstrap interval of the difference above 0 | Pet: black 52.8 [38.0, 62.9], inpaint 49.4 [34.2, 59.8], insertion 48.7 [34.0, 58.4]; CIFAR ResNet-18: inpaint 46.5 [32.9, 52.5], insertion 46.5 [38.7, 51.6]; CIFAR small CNN: inpaint 55.9 [46.2, 63.5], insertion 42.6 [33.1, 51.8] | **Holds** (7 of 7 cells) |

Notes on the table:

- P8 (b): the BH adjustment over the four SD1 metrics is applied to
  `p_excess_one_sided` of `overdetection.csv`, as the prediction states. The
  column `p_bh_excess_one_sided` in that file adjusts over all settings and
  metrics instead (0.58, 1.00, 0.0014, 0.14); it gives the same count (1 of
  4).
- P9 (b): metric switches against the full object map: Pet black 1.3 (edge),
  inpaint 4.6, insertion 5.4; CIFAR ResNet-18 inpaint 8.7, insertion 8.6;
  CIFAR small CNN inpaint 4.3 (edge), insertion 17.7. Differences are
  behavioural − metric switch in r_c points with the paired two-level 95%
  interval. The crossed-bootstrap intervals also lie above 0 (lowest end
  29.4, small-CNN insertion). Black and blur deletion on both CIFAR-10
  settings and blur deletion on Pet already prefer the shortcut map at the
  lowest reliance, so they have no metric switch and are not part of (b).
- Behavioural switch per SD2 setting (primary label, r_c, bracketing):
  Pet 54.1 (9.5 below, 6.0 above); CIFAR ResNet-18 55.2 (3.1 below, 9.6
  above); CIFAR small CNN 60.3 (5.4 below, 9.1 above). Without the A2 models
  (`pre_a2_models/`) they were 54.5 not bracketed (Pet), > 36.5 and > 42.7
  not identified (CIFAR-10). The check labels: CIFAR ResNet-18 54.0 (not
  bracketed) and 58.4; CIFAR small CNN 58.2 and 57.4 (both not bracketed);
  Pet 29.4 and 48.3 (both not bracketed).

Summary: four of five predictions hold (P5, P6, P7, P9). P8 fails on part
(b): on the pretrained network the metrics rank models by reliance, but the
shortcut's preference beyond a random ordering rises significantly only for
inpainting deletion.

### Pre-specified secondary outcomes (no verdict), recorded 2026-11-21

- **SD1 behavioural switch: not identified.** The isotonic crossing under
  the primary label is at r_c 49.1, but no model lies between r_c 31.2
  (dose 0) and 96.4 (dose +0.1), so it is reported as a bound: > 31.2 (not
  identified). Check labels: 43.9 and 64.2, also not identified.
- **SD1 class-stratified AUROC** (S vs O, primary label): pooled 0.79
  (black), 0.83 (blur), 0.97 (inpaint), 0.83 (insertion); within-model median
  0.84, 0.96, 0.995, 0.59. Under the check labels, pooled 0.78–0.97
  (moved-only) and 0.78–0.95 (dominance). Undetermined share (primary label):
  mean 0.25, max 0.48, Spearman ρ with r_c 0.91.
- **SD1 `confound.csv` / `confound_rc.csv`:** ρ(r_c, accuracy without the
  shortcut) = −0.96 (raw reliance −0.98). Partial ρ of preference with r_c
  given that accuracy: black 0.16, blur 0.41, inpaint 0.78 (p_bh 0.001),
  insertion 0.16.
- **S–O, R–O, S–R at the lowest SD1 dose (−0.5):** black 0.50 / 0.53 /
  0.44; blur 0.45 / 0.44 / 0.52; inpaint 0.24 / 0.28 / 0.48; insertion
  0.20 / 0.15 / 0.64.
- **Exploratory comparison with `pet_resnet18` (from scratch, 21 models):**
  S vs O over-detection black 0.79, blur 0.93, inpaint 0.64, insertion 0.64
  (SD1: 0.89, 0.56, 0.33, 0.44); ρ with r_c 0.86–0.97 on SD1. Grad-CAM maps
  are 4×4 on SD1 against 16×16 from scratch, so method-level comparisons mix
  pretraining with map resolution (`method_responsiveness_corrected.csv`:
  SD1 Grad-CAM shortcut mass 0.005 → 0.017).
- **SD2 CIFAR-10 small CNN** (not an SD1 outcome, recorded for P9): the 8
  laptop models reach r_c 50.9–94.4 (doses −0.05 to −0.01), filling the gap
  between 42.7 (dose −0.1) and 96.0 (dose 0).

# A Dose-Response Evaluation of Explanation Metrics under Controlled Shortcut Reliance in CNN Image Classifiers

## Pre-registered predictions: pretrained Pet (SD1) and extra natural-image doses (SD2)

Author: Md Junaid Faizan Shaikh

Date: 2026-11-13. Written and committed before any SD1 or SD2 model was
trained. The design (models, doses, pilots and the primary behavioural label)
is amendment A2 in `docs/PREDICTIONS_amendments.md`.

**Third-party timestamp:** `[PLACEHOLDER: OSF or Zenodo DOI/URL and upload
timestamp, to be added by the author after upload, before the first Kaggle
session that trains an SD1 or SD2 model.]`

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

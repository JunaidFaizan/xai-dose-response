# A Dose-Response Evaluation of Explanation Metrics under Controlled Shortcut Reliance in CNN Image Classifiers

## Pre-registered predictions: exact ROAD (SD3) and watermark shortcut (SD4)

Author: Md Junaid Faizan Shaikh

Date: 2026-11-18. Written and committed before any SD3 or SD4 model or score
existed. The design (ROAD operator, removal orders and grid, images, model
sets, timing gate, watermark, doses, pilots and stopping rule) is amendment
A3 in `docs/PREDICTIONS_amendments.md`.

**Third-party timestamp:** Zenodo version 3, DOI
[10.5281/zenodo.24416017](https://doi.org/10.5281/zenodo.24416017), published
2026-11-21 09:56 UTC (MD5 of this file as uploaded:
0ccbc01534200f30ca82e9076e1baec0, commit `6cef0e8`). This file
and `docs/PREDICTIONS_amendments.md` were uploaded together as a new version
of the record [10.5281/zenodo.24285532](https://doi.org/10.5281/zenodo.24285532)
(concept DOI [10.5281/zenodo.24285531](https://doi.org/10.5281/zenodo.24285531))
before any SD3 or SD4 model or score is produced.

### Settings

- **SD3:** ROAD with Noisy Linear Imputation (Rong et al., ICML 2022,
  https://arxiv.org/abs/2202.00449; reference code
  https://github.com/tleemann/road_evaluation, MIT licence), in two orders:
  ROAD MoRF (`road_morf`, low area is better) and ROAD LeRF (`road_lerf`,
  high area is better). Removal shares 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9
  (share removed, in both orders; for LeRF this differs from the reference
  LeRF grid, see A3).
  Scored on every registered model except pilots, on the first 100 (28/32 px
  small CNNs) or 50 (CIFAR-10 ResNet-18, Pet, SD1, SD4) of the existing
  scoring images; 25 for the Pet-sized settings if the A3 timing gate applies.
- **SD4:** `pet_resnet18_pretrained` with the watermark shortcut of A3 (partly position-coded within the corner)
  (cat: two vertical 4 px bars, box rows 0–11, columns 0–3 and 20–23; dog:
  two horizontal 4 px bars, box rows 0–3 and 8–11, columns 6–17; 96 stroke
  pixels each, in the 24×12 px box at image rows 116–127, columns 104–127,
  opacity 0.4). The confirmatory set is 18 models (6 doses × seeds 42, 1, 2),
  with doses fixed by the A3 pilot rule. The seed-7 pilots are excluded from
  every test below.

### Definitions

As in `docs/PREDICTIONS_pretrained.md` (reliance r_c, preference, S, O, R,
over-detection rate (FPR) with Clopper–Pearson 95% interval and n, the
margin FPR and r_c bands as secondary versions, switch rules, model-level
Spearman tests, bootstrap intervals). Additions:

- **ROAD** in a prediction means ROAD MoRF unless stated otherwise. ROAD
  LeRF is always reported next to it but decides no verdict.
- **Same images.** Whenever ROAD is compared with an existing metric, the
  existing metric's preference is recomputed from its stored per-image
  scores on the same images ROAD used (the first N of the existing 100).
  Its 100-image value is reported as well but decides nothing.
- **Natural-image settings:** `cifar10` (small CNN), `cifar10_resnet18`,
  `pet_resnet18` and SD1 (`pet_resnet18_pretrained`, patch).
- **Lowest-dose models of a setting:** all its registered models at its
  lowest dose: `cifar10` dose −0.8 (seeds 42, 1), `cifar10_resnet18` −0.8
  (seeds 42, 1, 2), `pet_resnet18` −0.5 (seeds 42, 1, 2), SD1 −0.5 (seeds 42,
  1, 2).
- **Paired image bootstrap:** within each model, images are resampled with
  replacement and the same resampled images are used for both metrics; the
  setting value is the mean over its models; 2000 replicates, seed 0
  (`BOOTSTRAP_REPS`, `BOOTSTRAP_SEED` in `dose_response.py`); percentile 95%
  interval.

### Predictions

**SD3 (exact ROAD)**

- **P10. ROAD lowers the random-map bias at low reliance.** In each of the
  four natural-image settings, at its lowest-dose models, the mean
  preference(R over O) under ROAD is lower than under Deletion (black), on
  the same images.
  - Test: the difference Deletion (black) − ROAD in mean preference(R over O),
    with its paired image-bootstrap 95% interval.
  - Holds in a setting if the difference is above 0 and its interval lies
    entirely above 0. Refuted in a setting if the difference is at or below
    0. Otherwise (a positive difference whose interval includes 0) it is
    "not confirmed" in that setting.
  - Overall: P10 holds if it holds in every testable setting. It is refuted
    if it is refuted in any testable setting. Otherwise it is "not
    confirmed". If fewer than three settings are testable, P10 as a whole is
    "not testable", and the per-setting results are still reported.
  - Not testable in a setting: Deletion (black) mean preference(R over O)
    at its lowest-dose models is at most 0.05 on the same images (floor:
    nothing left to lower), or the setting's lowest-dose models have no ROAD scores.

- **P11. ROAD over-detects less than black deletion on Pet.** On
  `pet_resnet18` (from scratch), the over-detection rate of ROAD in the S vs
  O comparison is lower than that of Deletion (black), on the same models
  and images.
  - Model set: all registered `pet_resnet18` models (the 18 original models
    and the 3 A2 models at dose −0.15) with r_c ≤ 50.
  - Test: the two FPRs (each with Clopper–Pearson 95% interval and n) and
    the difference FPR(black) − FPR(ROAD).
  - Holds if FPR(ROAD) < FPR(black). Refuted if FPR(ROAD) ≥ FPR(black).
    The decision uses the point values; the number of discordant models (one
    metric over-detects, the other not) and an exact two-sided sign test on
    them are reported but do not decide the verdict.
  - Not testable: no `pet_resnet18` model has r_c ≤ 50, or FPR(black) on the
    same images is 0 (it cannot be lowered).

**SD4 (watermark, 18 confirmatory models)**

- **P12. The metrics respond to watermark reliance.** Spearman ρ between r_c
  and preference(S over O) is above 0.8 for at least three of the four
  existing metrics: Deletion (black), Deletion (blur), Deletion (inpaint),
  Insertion (blur).
  - Test: model-level Spearman ρ over the 18 confirmatory models, on the
    existing 100 scoring images, with the two-level (models and images)
    bootstrap 95% interval. The decision uses the point estimate, as P8(a).
  - Refuted if ρ ≤ 0.8 for two or more of the four metrics.
  - Not testable: SD4 stopped under the A3 stopping rule, or fewer than two
    confirmatory models have r_c above 50 (the A3 rule).

- **P13. Black and blur deletion over-detect the watermark.** The
  over-detection rate in the S vs O comparison is above 0.5 for Deletion
  (black) and for Deletion (blur), each.
  - Model set: SD4 confirmatory models with r_c ≤ 50, on the existing 100
    scoring images.
  - Holds if both FPRs are above 0.5. Refuted if either is 0.5 or below.
    Each is reported with its Clopper–Pearson 95% interval, n, the margin
    FPR and the r_c bands; these do not decide the verdict.
  - Not testable: SD4 stopped under the A3 stopping rule, or no
    confirmatory model has r_c ≤ 50.

### Pre-specified secondary outcomes (reported, no verdict)

- ROAD MoRF and LeRF for every comparison already in the paper (S–O,
  S–size-matched, placebo–O, S–placebo, R–O, placebo–R, S–R), with tie
  shares, FPR, TPR, switch points and Spearman ρ, per setting.
- P10 and P11 repeated with ROAD LeRF, and P11 repeated on SD1 and SD4.
- P12 and P13 for ROAD MoRF and LeRF on the SD4 models.
- SD4 behaviour: class-stratified behavioural AUROCs under all three labels,
  the undetermined share, the behavioural switch with its status, and the
  `confound.csv` row.
- SD4 against SD1, setting by setting, as an exploratory comparison of
  shortcut form (no pooling).
- The timing of ROAD per model and setting.

### Commitments

- All results are reported whether or not these predictions hold, with the
  commit of the code that computed them.
- A prediction that cannot be computed is reported as "not testable", with
  the reason.
- The design is that of A3. Any change needs a new dated amendment before
  the models or scores it affects exist.
- P1–P9 and their verdicts are not changed by SD3 or SD4 results.

## Outcome

Not yet run. Results are added here after the analysis, without changing the
text above.

### Outcome recorded 2026-11-25 (after the full A3 analysis)

Appended after the analysis; nothing above was changed. The line "Not yet
run" above was written before the analysis and is kept as committed.

Data: the 18 SD4 confirmatory models (Kaggle session 4, 2026-11-25: doses
−0.5, −0.2, 0, +0.1, +0.25, +0.4 × seeds 42, 1, 2), exact ROAD scores for all
293 pooled models (206 on the laptop, T3.3; 69 GPU patch models and the 18
SD4 models in Kaggle session 4), and `dose_response.py analyze` over all 293
registered models. The SD4 pilots (seed 7) and the SD1 pilots are listed in
`results/dose_response/pilots.csv` and are not used. SD4 is its own setting
and is not pooled with SD1 or `pet_resnet18`. Numbers come from
`road_p10.csv` and `road_p11.csv` (P10, P11; code `road_p10` and `road_p11`
in `xai/dose_analysis.py`, added in the T5.1 commit that also adds these
files), `responsiveness_corrected.csv` and `overdetection.csv` (P12, P13),
and `responsiveness_corrected_road.csv`, `overdetection_road.csv`,
`behaviour_switch.csv`, `behaviour_summary.csv` and `confound_rc.csv`
(secondary outcomes), all in `results/dose_response/`. Values are rounded
from the CSVs.

**Images and timing gate.** ROAD used the first 100 images per model on the
28/32 px small CNNs (`mnist`, `pneumonia_mnist`, `cifar10`) and the first 50
on `cifar10_resnet18`, `pet_resnet18`, SD1 and SD4 (`road_models.csv`). The
timing gate was not reached (2.51 and 2.28 min per Pet model on a Kaggle T4,
note A3-C1), so N stayed 50.

**SD4 testability.** The +0.25 pilot had r_c 97.02 (stopping rule not
triggered). Of the 18 confirmatory models, 9 have r_c ≤ 50 (−0.11 to 1.09,
doses −0.5, −0.2, 0; all ≤ 10) and 9 have r_c > 50 (91.32 to 97.59, doses
+0.1, +0.25, +0.4). The A3 rule (at least two models above 50) is met, so
P12 and P13 are testable.

| Prediction | Result | Verdict |
|---|---|---|
| P10: Deletion (black) − ROAD (MoRF) in mean preference(R over O) at the lowest-dose models, same images, paired image bootstrap 95% CI (2000 replicates, seed 0) | `pet_resnet18` (dose −0.5, 3 models, 50 images): black 0.533, ROAD 0.337, difference 0.197 [0.107, 0.283]; `cifar10_resnet18` (−0.8, 3, 50): 0.747, 0.133, 0.613 [0.527, 0.693]; `cifar10` (−0.8, 2, 100): 0.815, 0.275, 0.540 [0.465, 0.610]; SD1 (−0.5, 3, 50): 0.500, 0.180, 0.320 [0.247, 0.400] | **Holds** (holds in all 4 testable settings; none at the 0.05 floor) |
| P11: on `pet_resnet18` (r_c ≤ 50), FPR(ROAD) < FPR(black), S vs O, same models and images | 14 models (of 21) with r_c ≤ 50, 50 images: black 0.71 (10/14, CP 95% 0.42–0.92), ROAD (MoRF) 0.64 (9/14, CP 0.35–0.87); difference 0.07; discordant models 1 (black only) and 0 (ROAD only), exact two-sided sign test p = 1.00 | **Holds** (on point values, by one model) |
| P12: ρ(r_c, preference S over O) > 0.8 for ≥ 3 of the 4 existing metrics, 18 SD4 models, 100 images, two-level bootstrap 95% CI | black 0.82 [0.69, 0.91], blur 0.88 [0.73, 0.93], inpaint 0.88 [0.74, 0.93], insertion 0.90 [0.78, 0.96] | **Holds** (4 of 4; the decision uses the point estimate) |
| P13: over-detection rate (S vs O, r_c ≤ 50) > 0.5 for black and for blur deletion, SD4, 100 images | black 1.00 (9/9, CP 0.66–1.00), blur 0.22 (2/9, CP 0.03–0.60); margin FPR 0.22 (2/9) and 0.00 (0/9); band r_c ≤ 10: the same 9 models, band 10 < r_c ≤ 50: no model | **Refuted** (blur is at or below 0.5) |

Notes on the table:

- P10: the 100-image Deletion (black) preference(R over O) at the same
  models, reported only, is 0.528 (`pet_resnet18`), 0.723
  (`cifar10_resnet18`), 0.815 (`cifar10`, the same 100 images) and 0.530
  (SD1). The bootstrap resamples the images within each lowest-dose model
  (models fixed), uses the same resampled images for both metrics, and
  starts from seed 0 for every setting and order.
- P11: the decision uses the point values as specified. The intervals
  overlap almost completely; ROAD (MoRF) still over-detects for most
  low-reliance Pet models (FPR above 0.5).
- P12: the crossed-bootstrap BCa intervals (`intervals_crossed.csv`) are
  0.69–0.91 (black), 0.80–0.97 (blur), 0.78–0.96 (inpaint) and 0.81–0.98
  (insertion). No model lies between r_c 1.09 and 91.32, so ρ mostly
  reflects the separation of the two groups of models.
- No test in this section was adjusted for multiplicity; the verdict rules
  are the pre-registered point-value and interval rules. The exact-ROAD
  tables have their own Benjamini–Hochberg families (each comparison over
  the 9 settings × 6 metrics of `responsiveness_corrected_road.csv`; the
  excess-preference p-values of `overdetection_road.csv` over its 54 rows).
  The SD4 setting is part of the families of the main analysis of all 293
  models.

Summary: three of four predictions hold (P10, P11, P12). P13 is refuted:
black deletion over-detects the watermark for every low-reliance model, but
blur deletion only for 2 of 9.

### Pre-specified secondary outcomes (no verdict), recorded 2026-11-25

- **P10 with ROAD (LeRF):** difference black − LeRF 0.213 [0.117, 0.310]
  (`pet_resnet18`), 0.287 [0.193, 0.377] (`cifar10_resnet18`), 0.430
  [0.335, 0.520] (`cifar10`), 0.397 [0.307, 0.487] (SD1); all positive with
  intervals above 0.
- **P11 with ROAD (LeRF)** on `pet_resnet18`: 0.71 (10/14), equal to black
  deletion (the rule would read it as not lower). **P11 on SD1** (50 images,
  9 models with r_c ≤ 50): black 0.67 (6/9), MoRF 0.33 (3/9), LeRF 0.33
  (3/9); discordant 3/0, sign test p = 0.25. **P11 on SD4** (50 images,
  9 models): black 0.78 (7/9), MoRF 0.00 (0/9), LeRF 0.00 (0/9); discordant
  7/0, sign test p = 0.016.
- **P12 and P13 with ROAD on SD4** (50 images): ρ(r_c, S over O) 0.87
  [0.74, 0.93] (MoRF) and 0.91 [0.78, 0.95] (LeRF); over-detection 0/9 for
  both.
- **ROAD in every comparison** (S–O, S–size-matched, placebo–O, S–placebo,
  R–O, placebo–R, S–R; ties, FPR, TPR, switches, ρ per setting):
  `responsiveness_corrected_road.csv`, `overdetection_road.csv`,
  `preference_road.csv` and the "Exact ROAD" section of `summary.md`. For S
  vs O, ROAD (MoRF) ρ with r_c is 0.60–0.98 over the nine settings, all
  positive with p_bh < 0.05; its over-detection rate is 0.64 (Pet), 0.47
  (CIFAR ResNet-18), 0.80 (CIFAR small CNN), 0.33 (SD1) and 0.00 (SD4).
  ROAD (LeRF) over-detects for every low-reliance MNIST patch (32/32) and
  PneumoniaMNIST patch (22/22) model.
- **SD4 behaviour:** behavioural switch not identified under all three
  labels (primary: isotonic crossing at r_c 51.1, reported as > 1.1, no model
  between 1.1 and 91.3; checks 46.7 and 70.2). Class-stratified AUROC (S vs
  O, primary label): pooled 0.79 (black), 0.91 (blur), 0.94 (inpaint), 0.93
  (insertion); within-model median 0.83, 0.78, 0.82, 0.61. Undetermined
  share: mean 0.21, max 0.47, Spearman ρ with r_c 0.79.
- **SD4 `confound_rc.csv`:** ρ(r_c, accuracy without the shortcut) = −0.85.
  Partial ρ of preference with r_c given that accuracy: black 0.29 (p_bh
  0.27), blur 0.67 (0.008), inpaint 0.70 (0.005), insertion 0.77 (0.001).
- **S–O / R–O / S–R at the lowest SD4 dose (−0.5):** black 0.54 / 0.57 /
  0.45; blur 0.44 / 0.44 / 0.53; inpaint 0.27 / 0.27 / 0.52; insertion
  0.12 / 0.11 / 0.60.
- **SD4 against SD1** (same network; SD4 changes the shortcut's shape,
  contrast and within-corner layout and is partly position-coded within the
  corner, so this is not a test of content coding alone): S vs O
  over-detection black 1.00 / 0.89, blur 0.22 / 0.56, inpaint 0.00 / 0.33,
  insertion 0.00 / 0.44 (SD4 / SD1); ρ with r_c 0.82–0.90 (SD4) and
  0.86–0.97 (SD1). Both settings have a reliance gap between doses 0 and
  +0.1 and an unidentified behavioural switch.
- **ROAD timing:** 2.51 min (`pet_resnet18`) and 2.28 min (SD1) per model
  on 50 images on a Kaggle T4 (`road_timing_kaggle_s3.csv`); 5.4–5.6 min
  (Pet) and 0.4 min (CIFAR ResNet-18) on the laptop CPU
  (`road_timing.csv`). Per-model times of the session-4 ROAD runs were not
  saved in the results.

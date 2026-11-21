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

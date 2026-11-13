# Audit 01: Statistics and experimental rigour

Project: *A Dose-Response Evaluation of Explanation Metrics under Controlled Shortcut Reliance in CNN Image Classifiers*

Reviewer lens: statistics and experimental rigour (seeds, bootstrap intervals, multiple comparisons, reliance gaps, ties, the 50% switch-point assumption, whether claims are supported).

All numbers below come from the project's own files (`results/dose_response/*.csv`, `results/analysis/summary.md`, `dose_response.py`) or from read-only re-analyses of those CSVs. The scripts and their full output are in `docs/audit/scratch/01/` (`checks.py` to `checks4.py` and the matching `*_out.txt`). Nothing outside `docs/audit/` was changed.

## 1. Summary verdict

**Q1 readiness (statistics lens): 5 / 10.**

The design is better than most XAI meta-evaluations: 192 independently trained models, a paired image design (the same 100 test images per model, `IMAGE_SEED = 0`), behavioural reliance, placebo and random controls, a model-level bootstrap, and a pre-registered extension whose failed prediction is reported. The monotone result ("preference rises with reliance") is solid. It survives multiplicity correction and image-level resampling in almost every setting.

The weak point is the second headline, that **the metrics switch too early**. That claim rests on a switch-point estimator whose value is not identified by the data:

- On MNIST patch there is no model between 23.0% and 45.2% reliance. Every reported switch point (30–42%) lies inside that gap.
- A different estimator moves the switch from 30–42% to 49–52%.
- The 50% ideal it is compared with is an assumption, not a derived quantity.
- On the binary tasks (Pet, PneumoniaMNIST) "reliance" includes ordinary classification errors. The Pet switch points of 19–25% correspond to roughly 1–7% reliance once that floor is removed.

Further gaps:

- No multiple-comparison correction across about 231 reported tests.
- The bootstrap ignores image-level noise.
- "Decision accuracy" is not balanced across classes.
- The claim that inpainting deletion is "the most robust variant" is not backed by a paired test. On one setting, black deletion is significantly better.
- Reliance is almost perfectly collinear with accuracy without the shortcut (Spearman −0.96 to −0.99 in five of seven settings). The design cannot separate "the metric tracks shortcut reliance" from "the metric tracks lost object competence".

A careful statistical reviewer at a Q1 journal would ask for major revisions on these points. Most of them are analysis fixes on existing CSVs (zero GPU). Two need modest new runs.

## 2. Strengths

1. **The unit of analysis is right.** `bootstrap_intervals` in `dose_response.py` (lines 504–542) resamples trained models (seeds) within each dose and keeps the dose design fixed. That is the correct primary level for between-model claims.
2. **Paired design.** Every model is scored on the same 100 test images, so differences between models are not confounded with image sampling.
3. **Controls for the form of the explanation map.** The placebo, random and size-matched maps make the "form bias" finding testable. Example: on CIFAR-10, black-fill deletion prefers a random map over the object map with preference 0.815 (95% seed+image CI 0.755–0.870) at 4% reliance, against 0.185 (0.130–0.245) for insertion (`checks2_out.txt`, section E). That gap is large and robust.
4. **The monotone dose-response holds up.** Of the 196 metric tests, 166 have p < 0.05 when recomputed exactly, 164 survive Benjamini–Hochberg and 154 survive Holm (`checks_out.txt`, section C). For "shortcut vs full object", Spearman ρ on the dose-level means (6 or 12 points per setting) is 0.85–1.00 on MNIST, CIFAR-10 and Pet (`checks_out.txt`, section C2).
5. **The pre-registration is honest.**
   - `docs/PREDICTIONS_pet.md` was committed in `a962b6c` (2026-11-10), before the Pet code commit `20694f2`.
   - The dose amendment `d823737` was committed before the results commit `1e29918`.
   - The P3 failure is reported.
   - P2 holds with a clear margin. The contrast mean(black, blur) − mean(inpaint, insertion) for "random vs full object" at dose −0.5 is 0.142 (95% CI 0.088–0.198). All four pairwise differences exclude 0 (`checks2_out.txt`, section E2).
6. **Ties are handled and reported.** Ties count as half (`TIE_TOLERANCE = 1e-4`) and the tie rate is in the CSV. Mean tie rates are 1–3%. They exceed 5% only on PneumoniaMNIST, where they reach 13% (`checks_out.txt`, section I).
7. **Seed instability is disclosed.** `summary.md` shows reliance std up to 31.8 points at one dose. The project uses measured reliance, not dose, as the x-axis, which is the right response to that instability.

## 3. Weaknesses, ranked by severity

### B1 (blocker): the switch point is not identified where the "too early" claim is made

**Evidence.**
- Coverage of the reliance range is sparse exactly where the switch point lives (`checks_out.txt`, section B):

  | setting | models | models with reliance 35–65% | largest gap in measured reliance |
  |---|---|---|---|
  | cifar10 patch | 12 | 2 | 45.1 → 96.3% |
  | cifar10_resnet18 patch | 18 | 2 | 39.0 → 96.2% |
  | mnist patch | 36 | 3 | 23.0 → 45.2% |
  | mnist texture | 36 | 2 | 55.6 → 76.8% |
  | pet_resnet18 patch | 18 | 3 | 57.1 → 72.3% |
  | pneumonia_mnist texture | 36 | 3 | 40.7 → 77.9% |

- On MNIST patch, the model-level data around the crossing are (`checks3_out.txt`). For black deletion, preference is 0.42 at 23.0% reliance and 0.63 at 45.2%. The reported switch points (black 35.3, blur 30.3, inpaint 41.7, insertion 38.8) all fall inside the empty interval.
- The estimate depends on the estimator. All of the following use the same data (`checks2_out.txt` K, `checks3_out.txt`):

  | MNIST patch, shortcut vs full object | black | blur | inpaint | insertion |
  |---|---|---|---|---|
  | paper: linear interpolation of per-dose means | 35.3 | 30.3 | 41.7 | 38.8 |
  | isotonic fit on the 36 models | 33.1 | 36.0 | 39.3 | 50.7 |
  | binomial logistic fit on model-level wins | 48.8 | 48.6 | 52.1 | 51.1 |

  The logistic model is misspecified, because preference saturates near 0.75–0.9 rather than 1. Treat it as a sensitivity check, not the truth. Still, three estimators give three answers, and the paper reports one with a narrow CI.
- Averaging seeds before interpolating mixes seeds that are far apart in reliance. At dose 0.01 on MNIST patch, the three seeds sit at 9.1, 17.5 and 62.2% reliance. Because the preference curve is nonlinear, averaging reliance and preference across those seeds and then interpolating gives a biased estimate (Jensen's inequality).
- The bootstrap CI is computed only over resamples that cross 0.5 (`switch_found_share`). It is therefore a conditional interval, and it cannot reflect the data gap: the CI for MNIST patch blur is 30.2–40.9 while there are no data between 23 and 45. The paper describes black and blur as "clearly before 50%" (`paper/build_paper.py`, lines 899–903).

**What the data do support.** Black and blur deletion already prefer the shortcut map (0.57–0.63) for the two MNIST patch models at 45–53% reliance, so they are early by a margin that cannot yet be measured. The data are consistent with insertion switching at about 50%.

### B2 (blocker): the 50% ideal and the reliance definition are not derived, and on binary tasks reliance includes the error rate

**Evidence.**
- Reliance is the share of *conflicting* test images on which the model predicts the shortcut's class (`train_one`, `reliance = accuracy(conf_preds, conflicting.patch_class)`). The metrics, however, are scored on *aligned* images, where shortcut and object agree (`evaluate_one` uses `shortcut_split(base, shortcut, "test")`). Nothing ties "preference crosses 0.5 on aligned images" to "the model follows the shortcut on 50% of conflicting images". `PAPER_OUTLINE.md` lists the 50% ideal as an assumption, which is correct, but the headline "switch too early" depends on it.
- On binary tasks every error on a conflicting image counts as following the shortcut, so reliance has a floor equal to the error rate. A chance-corrected reliance is (R − F) / (100 − F). Here F is the share of no-shortcut test images assigned to the shortcut's class: 100 − accuracy without the shortcut for 2 classes, and that divided by 9 for 10 classes. This is an approximation: images without the shortcut are out of distribution for a model trained with one. On that basis (`checks_out.txt` A, `checks2_out.txt` K):

  | Pet dose | raw reliance | floor | excess (R − F) |
  |---|---|---|---|
  | −0.5 | 17.5 | 17.6 | −0.2 |
  | −0.4 | 22.3 | 18.6 | 3.7 |
  | −0.3 | 33.2 | 21.4 | 11.8 |

  On the chest X-rays, PneumoniaMNIST patch at dose −0.4 has raw reliance 14.8 and a floor of 14.3. Switch points re-expressed on normalised reliance are:
  - Pet: black 1.4, inpaint 4.4, insertion 6.8 (paper: 18.9, 22.1, 24.7)
  - PneumoniaMNIST patch: 1.4–16.4 (paper: 16.8–31.9)

  The "lowest reliance reached was 15%" on the X-rays (`PAPER_OUTLINE.md`, Discussion) is therefore about 0% shortcut use. Under this correction the Pet finding becomes stronger: metrics switch while the model barely uses the shortcut. The current text understates the effect and makes the reliance axis hard to compare across 2-class and 10-class tasks.

### M1 (major): no multiple-comparison control across about 231 tests

**Evidence.**
- `responsiveness.csv` has 196 metric tests (7 comparisons × 7 settings × 4 metrics) and `method_responsiveness.csv` has 35 more. None is corrected.
- The p-values are rounded to 3 decimals and printed as `0.000`. They should be reported as "< 0.001".
- Results that lose significance under Holm (exact p recomputed; `checks_out.txt`, section C):
  - PneumoniaMNIST patch, Insertion vs full object: p = 0.0024 → Holm 0.095.
  - All three Grad-CAM trends in method responsiveness that were significant before correction: the two negative texture trends that README finding 8 and the abstract cite (MNIST texture ρ = −0.49, p = 0.003 → Holm 0.12; PneumoniaMNIST texture ρ = −0.49 → Holm 0.088), and the positive CIFAR-10 trend (ρ = 0.74 → Holm 0.23).
  - Pet placebo vs random, Insertion, ρ = −0.67: Holm 0.093.
  - CIFAR-10 random vs full object, black and blur: p = 0.049 and 0.021 → Holm 1.0 and 0.73.
- Under Benjamini–Hochberg (FDR 5%) nearly all of these survive (164 of 166). The project needs to choose and state a correction family.
- The negative method trends have seed-bootstrap CIs that exclude 0 (Grad-CAM, MNIST texture: −0.67 to −0.31; `checks2_out.txt`, section H). The finding is probably real, but the paper should present it with an FDR correction and a CI. `method_responsiveness.csv` has no CIs at all.

### M2 (major): the bootstrap ignores image-level variance, and three seeds make it coarse

**Evidence.**
- Each preference is the mean over 100 images (binomial SE about 0.05 per model). The bootstrap resamples only seeds. A two-level bootstrap (seeds within dose, then images within model) widens the intervals (`checks2_out.txt`, section D):
  - Pet black ρ: 0.71–0.95 → 0.60–0.93.
  - MNIST patch blur switch: 30.2–40.9 → 25.7–42.1.
  - PneumoniaMNIST patch inpaint switch: 28.7–50.0 → 27.6–57.0, so the interval now contains 50.
- With 3 seeds per dose there are only 10 distinct resamples per dose, and the bootstrap variance of a mean is (n−1)/n = 2/3 of the true variance. Both effects make the percentile intervals too narrow. CIFAR-10 small CNN uses 2 seeds, which gives 3 distinct resamples per dose.
- Pre-registered P1 for black deletion "holds" on the point estimate (0.85). Its 95% interval with image noise is 0.60–0.93, so ρ > 0.8 is not established at 95%. The pre-registration did not say whether the point estimate or the interval decides the outcome. State that the point estimate was used.

### M3 (major): "decision accuracy" is not balanced, so it mostly measures class balance

**Evidence (`checks_out.txt`, section G).** In CIFAR-10 only 1 of 6 doses has reliance above 50% (share 0.167). Black and blur deletion say "shortcut" for every model, which gives decision accuracy 0.167 and balanced accuracy 0.50, i.e. chance. The pattern across settings is uniform: the true-positive rate is 1.00 in 23 of 28 metric-setting cells, and all errors are false positives at low reliance.

| setting, metric | decision acc. (paper) | balanced | TNR (says "object" when reliance < 50%) |
|---|---|---|---|
| CIFAR-10 black / blur | 0.167 / 0.167 | 0.50 / 0.50 | 0.00 / 0.00 |
| Pet blur | 0.444 | 0.545 | 0.09 |
| PneumoniaMNIST texture blur / insertion | 0.667 / 0.667 | 0.50 / 0.50 | 0.00 / 0.00 |

Reporting balanced accuracy, or the false-positive rate at low reliance, states the real finding directly: the metrics over-detect the shortcut.

### M4 (major): "inpainting deletion is the most robust variant" is based on vote counting, not on a paired test

**Evidence.** The claim (`README.md` finding 5; `build_paper.py` lines 1018–1026, 1103, 1136) counts settings with p < 0.05. A paired seed bootstrap of the difference in ρ between metrics, using the same models in each resample, gives the following (`checks2_out.txt`, section F, shortcut vs full object):
- In 5 of 7 settings, inpaint is not distinguishable from black, blur or insertion (all CIs include 0; on the CIFAR-10 small CNN the interval against blur starts exactly at +0.000).
- On PneumoniaMNIST texture, inpaint is significantly **worse** than black deletion: Δρ = −0.197 (CI −0.385 to −0.039).
- Inpaint is significantly better than black only on Pet (+0.100, CI +0.007 to +0.235), and better than insertion only on PneumoniaMNIST texture (+0.347).

The defensible claims are narrower. Inpainting deletion is never non-significant. Inpainting deletion and insertion avoid the low-reliance form bias of black and blur deletion on natural images (strength 3 above). "Most robust" overall is not supported.

### M5 (major): reliance is almost collinear with object competence

**Evidence (`checks4_out.txt`).**
- Spearman(reliance, accuracy without the shortcut) is −0.99 (MNIST patch and texture), −0.98 (CIFAR ResNet-18), −0.96 (Pet), −0.88 and −0.81 (PneumoniaMNIST), and −0.72 (CIFAR small CNN).
- Partial Spearman of "shortcut vs full object" preference on reliance, controlling for accuracy without the shortcut, falls on MNIST patch to 0.11–0.51 (raw 0.96–0.97). On MNIST texture it falls to −0.28 to +0.36.
- Partial correlations under near-collinearity are unstable, so this does not show the metrics ignore reliance. It shows that the design cannot tell the two explanations apart. The swap dose (d > 0) relabels images, which degrades object learning directly. A reviewer will ask whether the metric responds to "the model uses the shortcut" or to "the model no longer uses the object" (speculative interpretation; the collinearity itself is measured).
- The CIFAR small CNN, the least collinear setting (−0.72), keeps partial ρ at 0.78–0.98, which is mildly reassuring.

### m1 (minor): the supporting study's claim that the sanity check "does not agree" with ground truth rests on 5-item Kendall τ

`results/analysis/summary.md` reports τ over 5 methods: −0.20 ± 0.20 (Decoy MNIST) and 0.067 ± 0.23 (Decoy Pneumonia). With n = 5 the exact one-sided null probability of τ ≥ 0.2 is 14/120 = 0.117. Only τ ≥ 0.6 reaches p ≈ 0.04. So "no agreement" is absence of evidence, not evidence of disagreement. On Decoy Pneumonia the deletion metrics are no better (black −0.33 ± 0.64, blur −0.33 ± 0.31). The per-image sanity agreement there is ρ = 0.61. The claim should be stated as "not shown to agree", with the exact τ null distribution.

### m2 (minor): the pre-registration is not independently timestamped

Git commit dates are set by the author. A reviewer can only trust the order if the GitHub push history, or a third-party registry, corroborates it. The order of commits `a962b6c` → `20694f2` → `d823737` → `1e29918` is consistent with the claims.

### m3 (minor): stale or inconsistent statements

- `README.md` still says "174 models" and that Pet is "pending", while `models.csv` has 192 models.
- "All eight settings" (README finding 5, `PAPER_OUTLINE.md` contribution 5) and "all six settings" (`build_paper.py` line 1018) do not match the 7 base × shortcut settings in `responsiveness.csv`.
- `PAPER_OUTLINE.md` says the MNIST inpaint switch interval "reaches about 51%", but `responsiveness.csv` gives 49.954. Under the two-level bootstrap it is 52.7.

### m4 (minor): the switch-point rule takes the first crossing only

`switch_point` returns the first crossing of 0.5. Several curves hover around 0.5 (placebo vs random, PneumoniaMNIST), and their reported "switch points" (for example 0.18% or 99.9%) are artefacts. Report switch points only for comparisons whose ideal curve crosses 0.5, which excludes placebo vs random.

## 4. Recommendations (ranked; cost = person-days and GPU hours)

Costs assume one student with AI help. CPU means the laptop. Training times come from `models.csv`: mean minutes per model are MNIST 1.7, PneumoniaMNIST 0.9, CIFAR ResNet-18 10.7 and Pet 25.4 (T4). Evaluation times are estimates (speculative).

| # | Fixes | Recommendation | Cost |
|---|---|---|---|
| R1 | B1 | **Fill the transition region with targeted doses and more seeds, then estimate the switch from model-level data.** Add doses where the gaps are: MNIST patch and texture (doses 0.0075, 0.015, 0.03), PneumoniaMNIST (−0.075, −0.025), CIFAR ResNet-18 (−0.075, −0.05, −0.025), Pet (−0.15, −0.05). Use 5–8 seeds at the steep doses, since seeds scatter widely there. Replace "interpolate per-dose means" with a model-level monotone fit: isotonic regression, or a 4-parameter logistic with free asymptotes. Report the switch with a two-level bootstrap CI, and declare it "not identified" when no model lies within ±10 points. | 3–4 days. CPU about 4–6 h for about 60 MNIST/PneumoniaMNIST models. GPU: 9 ResNet-18 (≈1.6 h train + ≈1 h eval) + 6 Pet (≈2.5 h + ≈1 h) ≈ **6 GPU h** |
| R2 | B2 | **Add a per-image ground truth that removes the 50% assumption.** Score the oracle maps on *conflicting* test images, with the model's predicted class as the target. If the prediction equals the shortcut's class, the shortcut is by construction the evidence; otherwise the object is. The code's own docstring states this in `xai/data.py` (lines 18–21). Then report each metric's per-image AUROC or accuracy in choosing the right map, as a function of reliance. Keep the aligned-image curves as the secondary analysis. | 2–3 days. CPU for MNIST/PneumoniaMNIST/CIFAR small CNN (≈1–2 h). **≈3–4 GPU h** for ResNet-18 and Pet |
| R3 | M1, M2, m4 | **Redo inference on the existing CSVs.** (a) Two-level bootstrap (seeds within dose, images within model) for ρ, switch points and the lowest-dose preferences. Use `scores_oracle.csv`; the code is sketched in `scratch/01/checks2.py`. (b) Benjamini–Hochberg within each family (metric × comparison; methods), plus Holm for the handful of headline claims. (c) Print p < 0.001 instead of 0.000. (d) Add CIs to `method_responsiveness`. (e) Report switch points only where the ideal curve crosses 0.5. | 1–1.5 days, 0 GPU |
| R4 | B2 | **Report chance-corrected reliance** (R − F)/(100 − F) next to raw reliance in every table, and re-express the Pet and PneumoniaMNIST switch points with it. Describe the correction as an approximation and give the formula. | 0.5 day, 0 GPU |
| R5 | M3 | **Replace decision accuracy with balanced accuracy, plus the false-positive rate at low reliance**, and add one calibration-style summary per metric (for example the mean absolute gap between the fitted preference curve and the ideal curve). Lead the narrative with "over-detection": TPR is 1.00 in 23 of 28 cells. | 0.5 day, 0 GPU |
| R6 | M4 | **Test metric-versus-metric claims with a paired bootstrap** of Δρ and Δ(lowest-dose preference), as in `scratch/01/checks2.py` section F. Rewrite "most robust" as the two claims the data support: inpainting deletion never fails to respond, and inpainting deletion and insertion avoid the form bias. Report the PneumoniaMNIST texture case where black deletion beats inpaint. | 0.5 day, 0 GPU |
| R7 | M5 | **Address the reliance–competence collinearity.** Minimum: report Spearman(reliance, accuracy without shortcut) per setting and the partial correlations in a limitations paragraph. Stronger, optional: a small decorrelation experiment on MNIST, for example varying shortcut salience (patch intensity) at a fixed swap dose of 0, so reliance changes while object accuracy stays high. Then show that preference still tracks reliance. | Minimum 0.5 day. Experiment: 2–3 days, CPU only (≈36 MNIST models ≈ 1 h train), 0 GPU |
| R8 | m1 | **Supporting study: give the exact Kendall τ null for n = 5** (or a permutation test over methods × seeds) and reword "does not agree" as "is not shown to agree". Note that deletion fails the same way on Decoy Pneumonia. | 0.25 day, 0 GPU |
| R9 | M2, m2 | **Pre-registration reporting.** Cite the commit hashes and link the GitHub commits, so readers can check the order of events. State that P1–P4 were judged on point estimates, and give the two-level CIs, for example black P1 0.60–0.93. For any further pre-registered run (such as R1 or R2), use a third-party timestamp (OSF or a Zenodo snapshot). | 0.25 day, 0 GPU |
| R10 | m3 | **Fix stale numbers and counts** (README 174 → 192 models and Pet results; "eight"/"six settings" → 7; the inpaint CI upper bound) in README, SYNOPSIS, PAPER_OUTLINE and `build_paper.py` text. | 0.25 day, 0 GPU |

**Suggested order with 3 weeks left.** Start with R3–R6, R8, R9 and R10 (about 3.5 days, no GPU). They change the wording of claims but need no new models. In parallel, run R1 and R2 on Kaggle (about 10 GPU h in total, within one week's quota). R7 (minimum version) fits in the writing week.

## 5. Claim-by-claim check

| Claim (source) | Verdict | Basis |
|---|---|---|
| Preference rises with reliance in every setting (README 1; abstract) | **Supported**, except PneumoniaMNIST texture insertion (ρ = 0.25, p = 0.14) and PneumoniaMNIST patch insertion after Holm (p = 0.095) | `checks_out.txt` C, C2 |
| Black and blur deletion switch "too early, at 30–35%" on MNIST patch (README 2; paper lines 899–903) | **Direction supported, number not.** Preference is above 0.5 at 45% reliance, but there are no models in 23–45%, and the estimate is estimator-dependent (30–36 interpolation or isotonic, 49 logistic) | B1 |
| All metrics switch at 6–15% against size-matched maps | **Plausible, but intervals too narrow.** Same gap problem; the CIs (1–20%) come from a seed-only bootstrap | B1, M2 |
| CIFAR-10 black and blur deletion prefer a random ordering over the object map 75–82% at 4% reliance | **Strongly supported** (black 0.815 [0.755, 0.870]; blur 0.750 [0.690, 0.810]) | `checks2_out.txt` E |
| Pet P2 (black and blur bias larger than inpaint and insertion) | **Supported** (contrast 0.142 [0.088, 0.198]) | E2 |
| Pet P1 for black (ρ > 0.8) | **Holds on the point estimate only** (two-level CI 0.60–0.93) | M2 |
| Inpainting deletion is the most robust variant | **Not supported as stated** | M4 |
| Saliency, SmoothGrad and Grad-CAM move away from the texture | **Supported for Saliency and SmoothGrad.** Grad-CAM fails Holm but passes BH and has a CI that excludes 0 | M1, `checks2_out.txt` H |
| The sanity check does not rank methods in line with ground truth (τ = −0.20 to 0.07) | **Weak:** with n = 5, |τ| ≤ 0.2 is indistinguishable from zero agreement in either direction | m1 |
| The X-ray setting is hard because the lowest reliance reached was 15% | **Misleading:** 15% is the error floor; chance-corrected reliance is about 0.6–4% | B2 |

## 6. External references (each URL opened during this audit)

- Bouthillier et al., *Accounting for Variance in Machine Learning Benchmarks*, MLSys 2021: <https://arxiv.org/abs/2103.03098>. Data sampling and weight initialisation both add substantial variance, which supports R1 (more seeds) and R3 (resampling images as well as seeds).
- Agarwal et al., *Deep Reinforcement Learning at the Edge of the Statistical Precipice*, NeurIPS 2021: <https://arxiv.org/abs/2108.13264>. Argues for interval estimates and stratified bootstrap when only a few runs exist, which matches the 2–3 seeds here.
- Saravanan, Berman and Sober, *Application of the hierarchical bootstrap to multi-level data in neuroscience*, NBDT 2020: <https://pmc.ncbi.nlm.nih.gov/articles/PMC7906290/>. Shows that treating nested samples as independent inflates false positives, and that resampling level by level fixes it. This is the basis for the two-level bootstrap in R3.
- Tomsett et al., *Sanity Checks for Saliency Metrics*, AAAI 2020: <https://arxiv.org/abs/1912.01451>. Already cited by the project; its reliability framing supports reporting per-image agreement (R2).

Holm and Benjamini–Hochberg are standard procedures. The adjusted p-values in `checks.py` are computed with a short in-script implementation, because `statsmodels` is not installed in the venv. No URL is cited for them.

# Audit report 03: Methodology and threats to validity

Project: *A Dose-Response Evaluation of Explanation Metrics under Controlled Shortcut Reliance in CNN Image Classifiers*

Reviewer lens: methodology and threats to validity (oracle-map design, size and shape confounds, shortcut realism, inpainting vs ROAD, metric implementation correctness, bugs and leakage, hostile-reviewer attacks).

Method of this audit: I read `dose_response.py`, `xai/data.py`, `xai/metrics.py`, `xai/explain.py`, `xai/models.py`, `paper/build_paper.py`, `README.md`, `SYNOPSIS.md`, `docs/PAPER_OUTLINE.md`, `docs/PREDICTIONS_pet.md`, `results/dose_response/summary.md` and the result CSVs. I ran two read-only scripts over the existing CSVs and test data: `docs/audit/scratch/03/analysis.py` (output in `analysis_out.txt`) and `docs/audit/scratch/03/ties.py`. Nothing outside `docs/audit/` was changed. Numbers below come from those files unless marked **[speculation]**.

---

## 1. Summary verdict

**Q1 readiness: 5 / 10** (from the validity angle only).

The core idea holds up: controlled reliance, behavioural measurement, fixed oracle maps, placebo and random controls, pre-registered Pet predictions, and 192 models. The code is clean and the computations I checked are correct. A hostile Q1 reviewer would still find three problems that go to the heart of the claims, and none of them needs new training to fix:

1. **The reliance measure has a floor equal to the model's own error rate.** On the binary tasks, "17–99% reliance" on Pet and "15–100%" on PneumoniaMNIST are really about 0–99% once corrected for chance. Several written claims and switch points are read on the wrong scale (Section 4, B1).
2. **The "ideal switch at 50% reliance" has no derivation**, and the design cannot separate "relies on the shortcut" from "has lost object competence": the two correlate at ρ = −0.81 to −0.99 across models. On MNIST, the partial correlation of preference with reliance (controlling for accuracy without the shortcut) falls from about 0.96 to between −0.28 and 0.51 (Section 3, W2).
3. **The "object" oracle is not an object on two of the five bases.** On PneumoniaMNIST it is the pixel intensity of the whole radiograph (97.5% of pixels > 0.1). On CIFAR-10 it is a centred Gaussian. Only MNIST and Pet have object maps that mean what the text says.

Cheap fixes exist: a re-analysis of the CSVs, an intervention-based ground truth that needs only forward passes, hierarchical statistics, and document corrections. With those, I would rate the methodology 7/10. Adding one realistic shortcut would move it to 8/10.

---

## 2. Strengths

- **Behavioural reliance, not assumed reliance.** `train_one` measures reliance on conflicting test images, and the analysis runs against measured reliance, not the dose. This handles the large seed spread near d = 0 (MNIST patch d = 0.01: 9–62%; MNIST texture d = 0.005: 14–77%).
- **Oracle maps separate metric evaluation from method quality.** That is the right design for a meta-evaluation.
- **Strong controls.** The size-matched object, placebo (same shape at a wrong class's position) and random maps are the right family. The paper correctly reads "placebo vs random ≈ 0.5" as evidence that the placebo carries no information. That holds for deletion: 0 of 21 deletion cells are significant positive except black deletion on PneumoniaMNIST texture (ρ = 0.77).
- **Pre-registration with an honest amendment and a reported failure (P3)** (`docs/PREDICTIONS_pet.md`). Few XAI papers do this, and it should be advertised.
- **Paired design.** All 192 models are scored on the same 100 test indices per base (`IMAGE_SEED = 0`, checked: 100 unique indices per base/shortcut, 100 rows per model).
- **The implementation is correct where I checked it.** Deletion and insertion masks (`_step_masks`), the trapezoid AUC, the direction of "better" (`higher_is_better`), tie handling (counted as half and reported), and the seed-within-dose bootstrap all do what the documents say. I found no train/test leakage: test images are never swapped or noised, the validation split is fixed by `SPLIT_SEED`, and Pet uses the official test split.
- **The Pet extension partly answers the Gaussian-object critique.** With real masks, black and blur deletion's preference for a random map over the object at the lowest dose drops to 0.53/0.49, against 0.82/0.75 on CIFAR-10.

---

## 3. Weaknesses ranked by severity

### Blockers (a hostile Q1 reviewer would reject on these)

**W1. Reliance is not chance-corrected; on binary tasks its floor equals the model's error rate.**
`reliance = accuracy(conf_preds, conflicting.patch_class)` (`dose_response.py`, `train_one`). With two classes, every object-based error on a conflicting image counts as "following the shortcut". At the lowest dose (from `analysis_out.txt`, section 1):

| base | reliance (lowest dose) | error without shortcut / (C−1) |
|---|---|---|
| pet_resnet18 | 17.45% | 17.64% |
| pneumonia_mnist patch | 14.85% | 14.32% |
| pneumonia_mnist texture | 18.38% | 15.06% |
| cifar10 | 3.98% | 3.18% |
| cifar10_resnet18 | 2.99% | 2.01% |

A chance-corrected reliance, r_c = (R − f)/(100 − f) with f = error/(C−1), gives a minimum of −0.8 to 3.0% in every setting. Consequences:
- "The lowest reliance reached on PneumoniaMNIST was about 15%, not 0" (README L178; `build_paper.py` around L874–877, L909, L1123) is wrong. The models reached about 0%, and the 15% is the measure's floor.
- Pet switch points such as black deletion at 18.9% (CI 17.4–20.4) sit just above a floor of 17.6%. On the corrected scale that is about 1–3% reliance, so the metric switches almost at once. The paper's "switch too early" story is stronger than written for binary tasks but weaker in the 10-class cases. Both need recomputing.
- `decision_accuracy` thresholds raw reliance at 50%, which is a different point on the corrected scale for binary and 10-class tasks.

**W2. The criterion (switch at 50%) is assumed, and reliance is confounded with object competence.**
- No argument shows that a faithful metric should switch at 50% behavioural reliance. Preference is measured on *aligned* images, where both cues agree. Reliance is measured on *conflicting* images. Under redundancy (either cue alone is enough), deleting either cue may barely move the probability, so the "correct" preference depends on the model's sufficiency and necessity structure, not on a 50% conflict rate. The outline lists this as a limitation; a Q1 reviewer will treat it as the main threat.
- The dose moves two things at once. Positive doses swap images, which degrades object features (MNIST no-shortcut accuracy falls from 98.8% to 13.3%). Spearman ρ(reliance, no-shortcut accuracy) is −0.985 (MNIST patch), −0.986 (MNIST texture), −0.963 (Pet), −0.981 (CIFAR ResNet-18) and −0.72 (CIFAR CNN). Partial Spearman of preference (shortcut vs full object) with reliance, controlling for no-shortcut accuracy:
  - MNIST patch: 0.11–0.51 (raw 0.96–0.97)
  - MNIST texture: −0.28 to 0.36 (raw 0.94–0.96)
  - Pet: 0.57–0.70 (raw 0.85–0.96)
  - CIFAR CNN: 0.78–0.98 (the only setting where the two separate)

  So on MNIST the data cannot tell "the metric tracks shortcut reliance" apart from "the metric tracks loss of object competence". The CIFAR small CNN gives the cleanest evidence and should be emphasised.

**W3. The "full object" oracle is not an object on PneumoniaMNIST and CIFAR-10.**
- `DecoyDataset.object_map` uses the image's own intensities for every grayscale dataset; its docstring assumes a dark background. On PneumoniaMNIST, 97.5% of pixels exceed 0.1 (`ties.py`), so the "object" map is a brightness ranking of the whole radiograph: bones, mediastinum and borders first. It is not where the model's object evidence lies. This alone may explain the weak PneumoniaMNIST results (ρ 0.25–0.80) that the paper reads as "X-rays are harder".
- On CIFAR-10 the map is a centred Gaussian, which the paper admits. Yet the headline finding 4 (black/blur prefer random over object 75–82%) rests on it. Pet now gives the real-mask test and should become the primary natural-image evidence.

### Major

**W4. Most of the AUC difference comes from the random tail.**
The shortcut map has 9 of 784 pixels on MNIST (1.1%) and 144 of 16,384 on Pet (0.9%). The first of the 20 steps removes 5%, so from step 1 onwards the shortcut map's ordering is a random tie-break (`_ranking`). "Shortcut vs full object" therefore largely compares "remove the cue, then random noise" with "remove the object in a structured order". The project's own random control shows this: random vs full object at low reliance is 0.815 for black deletion on CIFAR, the same as shortcut vs full object (0.815). A matched-remainder design would answer the critique directly: the two maps share the same tail ordering and differ only in which k pixels come first. A truncated AUC over the first k pixels would also work.

**W5. The size-matched control matches pixel count, not evidence, and its MNIST rule is a deterministic positional pick.**
- The shortcut map removes 100% of the shortcut. The size-matched map removes 9 of about 130 digit pixels (16.7% of 784). The two maps are equal in size, not in share of their cue removed, so the 6–15% switch points are biased towards the shortcut by construction. The full-object comparison is biased the other way. Present the two as a bracket, not as "the metric rewards size".
- `size_matched` takes `np.argsort(-flat, kind="stable")[:k]`. On MNIST a median of 29.5 pixels (max 115) tie at or above the 9th-brightest value, so the stable sort picks the first saturated pixels in row-major order. That is a run at the top of the digit (mean row 9.7 of 28), not "the brightest pixels". It is harmless but has to be described correctly, or ties have to be broken randomly.

**W6. The statistics understate uncertainty and ignore multiplicity.**
- `bootstrap_intervals` resamples only seeds within dose (2 or 3 per dose) and keeps the 100 images fixed. With 2–3 units per stratum the percentile bootstrap is known to be too narrow, and image-sampling variance is ignored. Implausibly tight intervals follow, for example CIFAR CNN inpaint ρ CI [0.986, 1.000] from 12 models and 2 seeds.
- About 196 Spearman tests (7 comparisons × 4 metrics × 7 settings) are reported with no correction.
- Significance is used where the effect is negligible. Grad-CAM on MNIST texture is "significant" (ρ = −0.49, p = 0.003), but its mean shortcut mass moves from 0.100 to 0.098. That is the texture's area share (10%), so the change is at chance level.
- `decision_accuracy` is unbalanced. On CIFAR only 2 of 12 (CNN) or 3 of 18 (ResNet) models exceed 50% reliance, so 0.167 just means "always says shortcut". Use balanced accuracy.

**W7. The perturbation fills do not remove the shortcut cleanly, and the inpainting is not ROAD.**
- Blur fill leaves 20.2% (28/32 px) and 24.5% (Pet) of the patch's intensity in place (`analysis.py` section 5). On MNIST that is a visible 0.2 square on a black background, at the class position. Blur deletion of the shortcut is therefore incomplete, and the insertion baseline already contains a faint shortcut. This is not documented.
- The texture shortcut survives blur almost not at all (0.0325 on pattern pixels vs 0.0285 off). This fits the insertion and blur failures on texture, but the paper explains those failures by "restoring scattered pixels barely changes it", which is a different mechanism.
- On MNIST, black fill equals exact counterfactual removal of the patch (the background is 0), but not of the texture (it also zeroes digit pixels). Each metric's validity differs per shortcut, and this should be stated.
- `inpaint` fills with the per-channel mean, runs 4-neighbour Jacobi averaging (100 or 1600 iterations) and adds N(0, 0.05) noise. [ROAD](https://arxiv.org/abs/2202.00449) proposes a "Remove and Debias" framework (its imputation is, from my background knowledge, a noisy linear imputation: **unverified** beyond the abstract), and its abstract places the confound in information leakage through the shape of the removed pixels. The project correctly says "ROAD-inspired", but a reviewer will ask for real ROAD (MoRF and LeRF orderings, the authors' imputation). A small quirk: at the last step all pixels are masked, `count` clamps to 1 and `mean` is 0, so the inpaint curve ends on a noisy black image, the same as black deletion.

**W8. The shortcuts are unrealistic, and only predictivity is varied.**
White squares at fixed corners and a fixed random-pixel texture are maximally available, constant-colour cues. [Hermann et al. (ICLR 2024)](https://arxiv.org/abs/2310.16228) show that feature use depends on both predictivity and availability. The dose here varies only predictivity, so the results may not carry over to subtle, natural shortcuts such as hospital-specific acquisition artefacts in chest X-rays ([DeGrave et al., Nat. Mach. Intell. 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC7523163/)). A Q1 reviewer in medical imaging or XAI will ask for at least one realistic shortcut (text tag, ruler, watermark, scanner-style intensity shift).

**W9. Pet is missing from the paper, and the paper argues against a hypothesis that Pet has already tested.**
`paper/build_paper.py` never mentions `pet_resnet18`. It still says the Gaussian-object cause "is a hypothesis to be tested with real object masks" (around L1087–1090 and L1120). The same-form tables (`SAME_FORM`, `PR_DEL`) exclude Pet, and the abstract counts `N_MODELS` without it, unless that count is computed from the CSV; I did not run the builder.

### Minor

- **W10.** Preference is computed for the true label on every aligned image, including misclassified ones (about 18% at Pet's lowest dose). Filter to correctly classified images or explain the predicted class, and report both.
- **W11.** Tie-break randomness is not shared between maps on the same image. `rng` is consumed in sequence in `evaluate_one`, so the shortcut and object maps get different random tails. This adds variance, not bias. Common random numbers would tighten the comparisons.
- **W12.** `switch_point` takes the first 0.5 crossing of the dose-mean curve. On non-monotone curves (for example PneumoniaMNIST patch insertion, which starts at 0.497) the first crossing is unstable. Report a fitted logistic midpoint as well.
- **W13.** Pet uses `ResNet18Cifar` (stride-1 3×3 stem, no max-pool) at 128×128, so Grad-CAM works on 16×16 maps, not the "final 4×4 feature maps" stated in the `ResNet18Cifar` docstring. This is legitimate but non-standard, and a reviewer may ask why the standard ImageNet stem was not used.
- **W14.** The `dose_response.py` docstring says "|d| is capped at 0.4", but `PET_DOSES` contains −0.5. The amendment documents this, but the code comment is stale.

---

## 4. Suspected bugs and incorrect claims in current documents

Each item gives the file and location, and why it is wrong. Severity is in brackets.

| # | File / location | What is wrong | Evidence |
|---|---|---|---|
| B1 [blocker, interpretation] | `dose_response.py` `train_one` (`"reliance"`); README L178; `build_paper.py` ~L874–877, L909, L1123 | Reliance is not chance-corrected, so "lowest reliance on PneumoniaMNIST was about 15%, not 0" misreads the floor of the measure | Floor f = 14.3% (patch) vs measured 14.85%; corrected minimum about 0% (`analysis_out.txt` §1) |
| B2 [major, stale] | README L18 and L29 | Says 174 models and that Pet is "prepared but not trained yet" | 192 models and Pet results are committed (`1e29918`); `responsiveness.csv` contains `pet_resnet18` |
| B3 [major, false count] | README L23 ("only metric that responds significantly in all eight settings (ρ ≥ 0.58, p < 0.001)") | There are 7 settings, not 8. For "shortcut vs full object", black and blur deletion are also significant in 7/7. Inpainting is unique only across all four "shortcut vs X" comparisons, and its minimum ρ there is 0.569 (shortcut vs placebo, PneumoniaMNIST patch), below 0.58 | `analysis_out.txt` §3–4 |
| B4 [major, false] | README L25, SYNOPSIS L108, PAPER_OUTLINE contribution 3 ("rises to 0.93–0.99 for every metric on the patch shortcut") | False for PneumoniaMNIST patch (0.727–0.785 vs random, 0.758–0.785 vs placebo) and Pet black deletion (0.763 / 0.773). `build_paper.py` handles PneumoniaMNIST separately (correct) but excludes Pet | `responsiveness.csv` |
| B5 [major, over-generalised] | SYNOPSIS L112; PAPER_OUTLINE contribution 5 and abstract ("Saliency, SmoothGrad and Grad-CAM move away from a diffuse shortcut") | True only on MNIST texture. On PneumoniaMNIST texture, Saliency ρ = +0.874 and SmoothGrad ρ = +0.696 (they move towards it). Grad-CAM's MNIST change (0.100 → 0.098) is negligible. README L26 is correctly qualified | `method_responsiveness.csv` |
| B6 [minor, range] | README L26 ("Integrated Gradients … ρ = 0.86–0.97") | The actual range is 0.856–0.986 (CIFAR CNN 0.986; Pet 0.936) | `method_responsiveness.csv` |
| B7 [major, description] | `dose_response.py` `size_matched`; summary text "the object's brightest pixels" | On MNIST, stable argsort over many tied saturated pixels picks the topmost pixels in row-major order (median 29.5 ties; mean row 9.7). This is a positional rule, not "brightest" | `scratch/03/ties.py` output |
| B8 [major, validity] | `xai/data.py` `DecoyDataset.object_map` docstring ("Grayscale datasets (dark background)") | PneumoniaMNIST has no dark background: 97.5% of pixels > 0.1, so the "object" map is an intensity ranking of the whole X-ray | `scratch/03/ties.py` output |
| B9 [minor, undocumented] | `xai/metrics.py` `blur` used as a deletion fill | Blur leaves 20–25% of the patch intensity, so blur deletion does not fully remove the shortcut. Not mentioned in any document | `analysis_out.txt` §5 |
| B10 [minor, statistics] | `dose_response.py` `bootstrap_intervals` | Resamples only 2–3 seeds per dose with fixed images, so the intervals are too narrow (e.g. CIFAR inpaint ρ CI [0.986, 1.000]) | `responsiveness.csv` |
| B11 [minor] | `dose_response.py` `responsiveness` (`decision_accuracy`) | Unbalanced; 0.167 on CIFAR is "always shortcut" | 2/12 and 3/18 models above 50% |
| B12 [minor] | `xai/models.py` `ResNet18Cifar` docstring ("final 4x4 feature maps"); `xai/explain.py` `grad_cam` comment ("6x6 to the 28x28") | For Pet the maps are 16×16 at 128 px; for CIFAR models 8×8 / 4×4 | Architecture code |
| B13 [minor] | `dose_response.py` module docstring ("|d| is capped at 0.4") | `PET_DOSES` includes −0.5 | `PET_DOSES` |
| B14 [major, omission] | `paper/build_paper.py` (whole file) | No Pet results; still calls the real-mask test future work (~L1087–1090, L1120) | grep for `pet` finds only the Petsiuk citation |

I found no code bug that invalidates a computed number. The `perturbation_curve` masks, AUC, preference direction, tie handling and the placebo/random builders behave as documented. The problems are in measurement design, interpretation and text.

---

## 5. What a hostile Q1 reviewer would write (condensed)

1. "Reliance on binary tasks includes the model's ordinary errors; the authors' own 'lowest reliance 15–17%' is simply the error rate. All switch points on PneumoniaMNIST and Pet are uninterpretable."
2. "The 50% switch criterion is asserted, not derived. Why should a deletion AUC on aligned images flip exactly when conflicting-image behaviour reaches 50%?"
3. "The dose moves shortcut reliance and object competence together (ρ ≈ −0.98). The metric may just be tracking the model getting worse at the object."
4. "On X-rays the 'object' map is image brightness; on CIFAR it is a Gaussian. What does 'prefers the object explanation' mean then?"
5. "The shortcut is a white square in a corner. Real shortcuts are subtle. Generalisation is unshown."
6. "Intervals from 2–3 seeds, 196 uncorrected tests, and 'significant' effects of 0.002 in mass."
7. "The inpainting is not ROAD, and blur does not remove the patch."

The project already has the data or the code paths (`mode="none"`, `mode="conflicting"`) to answer points 1–4 and 6–7 without new training.

---

## 6. Recommendations (ranked by value per cost)

Costs assume one student with AI help. "CPU" means the laptop; GPU hours are Kaggle T4. Training and evaluation costs are estimates from the timings in the repository (Pet about 23 min per model to train); evaluation times are **[speculation]**, since no evaluation timings are logged.

| Rank | Recommendation | Fixes | Cost |
|---|---|---|---|
| **R1** | **Chance-correct reliance.** Add r_c = (R − f)/(100 − f), f = error without shortcut/(C−1), to `models.csv` in analysis only. Recompute ρ, switch points, CIs and (balanced) decision accuracy on r_c, report both scales, and rewrite the "lowest reliance 15%" text. | W1, B1, B11 | 0.5–1 day, 0 GPU h (CSV re-analysis) |
| **R2** | **Replace the 50% ideal with an intervention-based ground truth for each image.** For each aligned test image, compute Δ_S = p(x) − p(x without shortcut) using the exact clean image (`mode="none"`), and Δ_O = p(x) − p(the shortcut on a different-class image) (the `conflicting` construction). The true preference is 1[Δ_S > Δ_O]. Score each metric by how well its preference agrees with it (accuracy or AUROC per model, then ρ with reliance). This gives a defensible target on every base without a normative 50% assumption. | W2 (main), W5 | 2–3 days; forward passes only: about 1–2 h CPU for the 174 small models, about 1 GPU h for 18 Pet [speculation] |
| **R3** | **Report the competence confound and separate the two factors.** (a) Report partial correlations (already computed in `scratch/03/analysis_out.txt`) and lead with the CIFAR CNN, where reliance and competence separate (partial ρ 0.78–0.98). (b) Optionally add a small 2-D grid on MNIST: fixed shortcut dose × image degradation (Gaussian noise or reduced contrast on the object only), so competence varies at constant reliance. | W2 | (a) 0.5 day, 0 GPU h. (b) 2–3 days, about 3–5 h CPU (MNIST models train in minutes) [speculation], 0 GPU h |
| **R4** | **Fix the oracle object maps and the AUC tail.** (a) Promote Pet to the main natural-image result and reframe CIFAR as a "Gaussian prior" sensitivity check. (b) For PneumoniaMNIST, either use a lung-field prior or mask, or drop the "full object" interpretation and keep only same-form comparisons. (c) Add a matched-remainder variant (shortcut-first vs object-first, with the same tail ordering) and a truncated AUC over the first k pixels. (d) Break `size_matched` ties randomly, and add an "evidence-matched" control (object pixels whose removal causes the same probability drop as removing the shortcut). | W3, W4, W5, B7, B8 | 3–4 days; re-evaluation about 6–10 h CPU for small models and 3–5 GPU h for Pet [speculation] |
| **R5** | **Use hierarchical statistics.** Fit a mixed-effects logistic regression of per-image preference on reliance, with random intercepts for model and image (or a two-level bootstrap over models and images, as in [Saravanan et al.](https://arxiv.org/abs/2007.07797)). Apply Holm correction across the ~196 tests, report effect sizes (change in preference, change in mass) next to ρ, and add a logistic-fit midpoint for the switch point. | W6, W12, B10 | 1–2 days, 0 GPU h (CSV only; `scores_oracle.csv` has per-image rows) |
| **R6** | **Add one realistic shortcut on Pet or PneumoniaMNIST**, for example a small text tag or an "L/R" marker with jittered position, size and contrast, or a semi-transparent watermark. Vary availability (size or contrast) at one dose, following Hermann et al. One shortcut × 6 doses × 3 seeds = 18 models. | W8 | 3–4 days; about 7 GPU h to train (18 × 23 min) plus about 3–4 GPU h to evaluate [speculation]; fits in one Kaggle week |
| **R7** | **Add real ROAD and document the blur residual.** Run the authors' ROAD imputation with both MoRF and LeRF on CIFAR and Pet, at least for the oracle maps. Report that blur fill keeps 20–25% of the patch, and either add a "blur + exact shortcut removal" fill or state the limitation. | W7, B9 | 2 days; about 2–4 GPU h for Pet, a few CPU hours for CIFAR [speculation] |
| **R8** | **Correct the documents** (B2–B6, B12–B14): update README to 192 models and Pet, fix "eight settings", the "0.93–0.99" claim, the Saliency/SmoothGrad claim and the IG range, add Pet to `build_paper.py`, and retire the "to be tested with real masks" text. | B2–B6, B12–B14, W9 | 1 day, 0 GPU h |
| **R9** | **Smaller hygiene items:** filter to correctly classified images (or explain the predicted class) as a robustness table; use common random numbers for tie-breaks across maps; justify the CIFAR-style ResNet stem at 128 px or note it. | W10, W11, W13 | 1 day plus re-evaluation (can share the R4 run), 0–2 GPU h |

Suggested order for the three remaining weeks: R1 → R8 → R5 → R2 in week 1 (no GPU), R4 and R7 re-evaluations in week 2 (one Kaggle session for Pet), and R6 in weeks 2–3 if GPU quota allows. R3(b) is optional.

---

## 7. External sources opened for this report

- Rong et al., ROAD, ICML 2022: <https://arxiv.org/abs/2202.00449>
- Hermann et al., *On the Foundations of Shortcut Learning*, ICLR 2024: <https://arxiv.org/abs/2310.16228>
- DeGrave, Janizek, Lee, *AI for radiographic COVID-19 detection selects shortcuts over signal* (Nature Machine Intelligence 2021; open PMC version): <https://pmc.ncbi.nlm.nih.gov/articles/PMC7523163/>
- Saravanan, Berman, Sober, *Application of the hierarchical bootstrap to multi-level data in neuroscience*: <https://arxiv.org/abs/2007.07797>
- Kim et al., *Sanity Simulations for Saliency Methods* (SMERF), ICML 2022: <https://arxiv.org/abs/2105.06506>. Related work: synthetic ground truth with controlled model-reasoning complexity, aimed at methods, not metrics. Worth citing and contrasting.
- Hooker et al., ROAR, NeurIPS 2019: <https://arxiv.org/abs/1806.10758>. The retraining-based alternative that ROAD avoids; a reviewer may ask why ROAR was not used.

I read only the abstract and landing pages. Details of ROAD's imputation beyond its abstract (MoRF/LeRF, linear imputation) come from my background knowledge and are **unverified** here; check them against the paper's method section before citing. Content was rephrased for compliance with licensing restrictions.

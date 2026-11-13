# Audit 2, report 03: Methodology and threats to validity

Project: *A Dose-Response Evaluation of Explanation Metrics under Controlled Shortcut Reliance in CNN Image Classifiers*

Reviewer lens: methodology and threats to validity. I checked the MD4 behavioural labels, the chance correction, the oracle maps, the metric and statistics implementation, and the attacks a hostile reviewer would make.

State audited: `main` at 80cd09f, 234 models. I read `dose_response.py` (training, evaluation, `behaviour_one`, analysis), `xai/data.py`, `xai/stats.py`, `xai/dose_analysis.py`, the text and tables of `paper/paper_draft.docx` (extracted to `scratch/03_methodology/paper_text.txt`), README, SYNOPSIS, `docs/PAPER_OUTLINE.md`, `docs/audit/03_methodology_validity.md` and `docs/audit/06_must_do_resolution.md`. I also read the result CSVs in `results/dose_response/`, including the per-model behaviour files.

I ran two read-only scripts over the existing CSVs: `docs/audit2/scratch/03_methodology/analysis.py` (output `analysis_out.txt`) and `analysis2.py` (output `analysis2_out.txt`). Section references such as "§A1" point to those outputs. No existing file was changed, and nothing was committed.

---

## 1. Score

**Q1 readiness (methodology lens): 6.5 / 10.** The first-audit score was 5 / 10.

The first-audit blockers on measurement and statistics are fixed, and fixed correctly:
- Reliance is now chance-corrected.
- Balanced accuracy replaces decision accuracy.
- A two-level bootstrap, BH/Holm correction, effect sizes and isotonic switches with an identification rule are in place.
- Pet is in the paper.
- The validity limits are stated openly.

I re-derived r_c, the switch CIs and the docx numbers I spot-checked, and they match the CSVs (Section 4).

The score does not reach 7 because the new primary analyses carry four validity problems. A careful reviewer who opens the repository will find them:
- The primary behavioural label measures sufficiency under conflict, not what deletion measures. The check label, which the paper defines but never reports, reverses the PneumoniaMNIST patch conclusion.
- The headline over-detection at low reliance is the same number as "random ordering beats the object map". It says little about the shortcut.
- The behavioural switch is almost the same quantity as r_c itself. On CIFAR-10 it fails the paper's own ±10-point identification rule.
- The bootstrap resamples images independently per model, although every model is scored on the same 100 images. The intervals on levels (switch points, low-dose preference) are therefore too narrow.

All four can be fixed or reframed from the existing CSVs in about 3–4 days without GPU time. That would bring the methodology to 7. With the planned GPU work (SD1, SD2, SD5-style re-evaluation, SD4), it can reach 8.

---

## 2. First-audit weaknesses: status

| First-audit item | Status | Evidence |
|---|---|---|
| **W1** reliance not chance-corrected | **Resolved** | `xai/stats.py` `chance_floor`/`corrected_reliance` implement f = (100 − acc_none)/(C − 1) and r_c = 100(R − f)/(100 − f). This is the correct inversion of R = r + (1 − r)f. The "15%, not 0" text is corrected (docx 5.1). Independent support: the per-image follow rate agrees with r_c to within 1.7–4.4 points MAE in every setting, including the 10-class ones (§1 of `analysis_out.txt`). So the uniform-error assumption does no visible harm. |
| **W2** 50% criterion assumed; reliance confounded with competence | **Partly** | `confound.csv` and docx 5.9 report the collinearity honestly (4 of 7 settings ≤ −0.95) and lead with the CIFAR CNN (partial ρ 0.78–0.98, which I reproduced). The 50% ideal is "replaced" by a behavioural switch, but that switch is close to a re-measurement of r_c (new problem N3). The core concern, that preference is scored on aligned images while the label comes from conflict, is not resolved by the primary label (N1). The 2-D grid (reliance × competence) was not done. |
| **W3** object oracle is not an object on PneumoniaMNIST and CIFAR | **Partly** (reported as a limitation) | Pet is promoted to primary, and the docstring and the paper (3.4, Limitations) state the problem. No lung-field prior was added. The CIFAR Gaussian still drives an abstract number ("81% of the time") and the "100% of CIFAR-10 models" over-detection figure (N2). |
| **W4** AUC dominated by the random tail | **Not resolved**, and now load-bearing | SD5 was not done. At the lowest dose the "shortcut vs object" preference equals "random vs object" to within 0.07 in all 28 cells, and exactly on CIFAR CNN black deletion (0.815 / 0.815) (§C of `analysis2_out.txt`). The over-detection headline therefore rests on this mechanism (N2). |
| **W5** size-matched control matches size only; tie rule is positional | **Partly** | The description is corrected (docx 3.4, `size_matched` docstring). Ties are still not broken at random, and there is no evidence-matched control. Pet uses `compact_matched`, which is sensible. |
| **W6** narrow intervals, no multiplicity control, effect sizes ignored, unbalanced accuracy | **Resolved, with a new caveat** | Two-level bootstrap, BH within families, Holm on the headline tests, median AUC gaps and balanced accuracy/FPR are all present. The caveat is that the bootstrap is nested while the design is crossed (N4). |
| **W7** blur residual; inpainting is not ROAD | **Partly** (documented) | The blur residual and the black end of the inpainting curve are documented (docx 3.5, Limitations). Exact ROAD (SD3) was not run. |
| **W8** unrealistic shortcut; only predictivity varied | **Not resolved** (stated as a limitation) | SD4 was not done. Limitations cite Hermann et al. |
| **W9** Pet missing from the paper | **Resolved** | Pet comes first in every table, and docx 5.3 holds the pre-registered test. |
| **W10** preference uses the true label on misclassified images | **Not resolved, not disclosed** | 20.7% of the aligned images in low-reliance Pet models (r_c ≤ 20) are misclassified; for PneumoniaMNIST the figure is 12.4%, and for the CIFAR CNN 12.8% (§E). These are exactly the models that carry the over-detection claim. Neither the docx nor the Limitations mention it. |
| **W11** no common random numbers for tie-breaks | **Not resolved** (minor) | `evaluate_one` still draws from one `rng` stream across maps. |
| **W12** first-crossing switch unstable | **Resolved** | Isotonic fit (PAVA), 4PL sensitivity, identification within ±10 points, bracketing and edge flags (`xai/stats.py`). |
| **W13** CIFAR-style stem on Pet | **Disclosed** | Docx 4 and Limitations. |
| **W14** stale "|d| ≤ 0.4" docstring | **Resolved** | |
| **B1–B14** (document bugs) | **Resolved** | I spot-checked B1, B3–B6 and B14 in the docx and found them fixed. |

---

## 3. New problems introduced by the fixes

### N1 [major]. The primary behavioural label measures sufficiency under conflict; deletion measures necessity on aligned images. The unreported check label reverses a conclusion.

How the label works: `add_causal_label` sets `follows_shortcut_causal = 1` when the moved shortcut wins on the conflicting image. Images whose shortcut-free prediction equals the conflict class are dropped as undetermined.

On a binary task, a determined image therefore always has pred_none = true label. Every label-1 image is one where **the object alone is enough for the argmax and the shortcut alone also wins**: both cues are sufficient (§A of `analysis2_out.txt`: share pred_none == label = 1.000 on Pet and on both PneumoniaMNIST settings).

On such redundant images, deleting the shortcut on the aligned image may barely move p_y. In that case a faithful deletion metric *should not* rate the shortcut map above the object map. This is the redundancy case the first audit raised in W2, and the data show it:
- On PneumoniaMNIST patch, the label-1 images have median p_label_none = 0.996, and 95.6% have Δ_S < 0.1.
- The two labels the project computes agree on only 50.3% of determined PneumoniaMNIST patch images, i.e. at chance. Only 14.6% of the "followed" images there are shortcut-dominant (§2 of `analysis_out.txt`).

The consequence is that the verdict depends on the label:

| PneumoniaMNIST patch, shortcut vs full object, pooled class-stratified AUROC | `follows_shortcut_causal` (primary) | `shortcut_dominant` (check, defined in docx 3.6) |
|---|---|---|
| Deletion (black) | 0.46 | 0.90 |
| Deletion (blur) | 0.50 | 0.92 |
| Deletion (inpaint) | 0.65 | 0.94 |
| Insertion (blur) | 0.41 | 0.91 |

Source: `behaviour_summary.csv` (§B of `analysis2_out.txt`).

The docx reports "PneumoniaMNIST patch is the weak case … black deletion 0.46, Holm p = 1.000", and `06_must_do_resolution.md` §3 lists this as a result that weakens earlier claims. The check label's numbers appear only in `summary.md`. A reviewer who opens the repository will read this as selective reporting.

The Limitations also concede that the causal label and the stratified AUROC were chosen after the original-set results had been inspected. The commits confirm the order: be03bac (labels), then 64bab8f (causal label), then b9b3a89 (stratified AUROC). That raises the stakes further.

A caveat runs the other way too. On MNIST patch, black fill equals exact removal of the patch, so the `shortcut_dominant` label (Δ_S = p(aligned) − p(none)) and the first deletion step use nearly the same intervention. Its pooled AUROC of 0.98–0.99 is therefore partly circular. Neither label is a clean ground truth for every metric. The paper has to show both and say which question each one answers: sufficiency under conflict versus necessity in agreement, in the sense of [Watson et al., UAI 2021](https://arxiv.org/abs/2103.14651).

### N2 [major]. At low reliance, "over-detection" equals "the object map does not beat a random ordering".

At the lowest dose, the preference for the shortcut over the object (S–O) equals the preference for random over the object (R–O) to within ±0.07 in all 28 cells:

| Cell | S–O | R–O |
|---|---|---|
| CIFAR CNN, black deletion | 0.815 | 0.815 |
| CIFAR ResNet-18, black deletion | 0.703 | 0.723 |
| Pet, blur deletion | 0.503 | 0.490 |
| Pet, insertion | 0.350 | 0.350 |

Over the same cells, shortcut vs random (S–R) is 0.45–0.62 (§C).

So when the model barely uses the shortcut, the metrics rate the 0.9% shortcut pixels like random pixels, which is correct. The preference above 0.5 comes from rating the *object map* no better than a random ordering. On CIFAR-10 the object map is a Gaussian prior, so "100% of CIFAR-10 models" over-detect (abstract, README finding 2, SYNOPSIS finding 2, highlight 2) mainly measures how the metric treats a centred Gaussian. It is weak evidence about shortcuts.

The docx gives the right mechanism in 5.2 ("Where does the over-detection come from?"). The abstract and highlight still frame the result as "over-detect shortcuts models barely use".

The real-mask Pet setting is the stronger case. Black deletion R–O is 0.53 [0.46, 0.60], so the metric cannot tell the pet mask from random pixels. That is a finding about object recognition by the metric, and it should be stated that way.

### N3 [moderate]. The behavioural switch is close to a re-measurement of r_c, and on CIFAR-10 it is not identified by the paper's own rule.

- **Follow rate ≈ r_c.** Per model, 100 × follow rate tracks r_c with Spearman 0.94–0.99 and a mean absolute difference of 1.7–4.4 points (§1). This is expected: both remove the same error floor. The behavioural switches (36.9–55.0%) are therefore mostly "where r_c ≈ 50%, measured on the 100 scored images". The README, SYNOPSIS, docx and highlight say the labels "replace the assumed 50% switch point". That is literally true, but it adds less independent evidence than it suggests. The first audit's objection (aligned vs conflicting images) is untouched (N1).
- **Not identified on CIFAR-10.** The CIFAR models leave large r_c gaps: 42.7 → 96.0 for the CNN and 36.5 → 95.8 for ResNet-18 (§G). The behavioural switches are 55.0 and 52.9, so the nearest model is 12.3 and 16.4 points away. Under the paper's ±10-point rule, which it applies to the metric switches, these are "not identified". The docx says the 50% ideal is "about right for Pet, CIFAR-10 and MNIST patch". For CIFAR-10 that is an interpolation across a gap of about 55 points.
- **Pet.** The Pet switch (54.5) lies between models at 41.1 and 60.1. It is identified but not bracketed.
- **No interval.** The behavioural switch has no CI, and the gap between the metric switch and the behavioural switch has no CI either.

### N4 [moderate]. The bootstrap is nested, but the design is crossed.

`two_level_picks` resamples images independently for each model slot. Every model of a setting is scored on the same 100 images; I asserted this in `analysis.py` and it holds. Image effects are therefore shared across models, and per-model resampling averages them away.

I re-implemented the project's bootstrap and compared it with a crossed version that uses one image resample shared by all models. The nested version reproduces the CSV intervals, e.g. the Pet inpaint switch is [1.9, 11.5] against the CSV's [2.1, 10.6]. The crossed version widens the level quantities (§6):

| Quantity | Nested width | Crossed width |
|---|---|---|
| Switch, PneumoniaMNIST patch, black deletion | 13.5 | 61.9 (CI [3.0, 65.0]) |
| Switch, PneumoniaMNIST patch, inpaint deletion | 35.2 | 58.0 |
| Switch, Pet, inpaint deletion | 9.6 | 15.5 |
| Low-dose preference (all checked settings) | — | 20–50% wider |

The Spearman ρ intervals barely change, so the responsiveness claims stand. The MNIST patch claim (metric switch about 39% against the behavioural 53.9%) also survives: the crossed upper bound is 48.3.

From my background knowledge (unverified), resampling both factors is mildly conservative. The [pigeonhole bootstrap](https://arxiv.org/abs/0712.1111) (Owen) resamples rows and columns separately and is the standard reference for crossed designs. Report it, or both versions, for switch points and preferences. The docx description ("images within model") is accurate; the problem is the choice of scheme.

### N5 [moderate]. The over-detection rate thresholds point estimates that are near coin flips.

The FPR counts a model as "says shortcut" when its preference exceeds 0.5. That preference is estimated from 100 images. On Pet, 6 of the 12 models with r_c ≤ 50 have preferences between 0.43 and 0.61, and their per-model image CIs include 0.5 (§5).

If a model only counts as over-detecting when its image-bootstrap lower bound exceeds 0.5, the Pet FPR changes as follows:

| Pet metric | FPR on point estimates | FPR with CI lower bound > 0.5 |
|---|---|---|
| Black deletion | 0.75 | 0.50 |
| Blur deletion | 0.92 | 0.58 |
| Inpaint deletion | 0.58 | 0.25 |
| Insertion | 0.58 | 0.50 |

The over-detection survives, mainly in the dose −0.3 and −0.2 models (r_c 15–41, preferences 0.55–0.88). The abstract's "75–92% of Pet models" is the upper end. With a small preference margin, the finding is roughly half of the Pet models.

### N6 [minor]. Within-model AUROC has little power.

Per model, the smaller label class has a median of 6 (CIFAR ResNet-18), 10 (CIFAR CNN), 11 (Pet) and 25 (MNIST patch) images (§7). On binary tasks with high reliance, 43–49% of the images are undetermined, and 79–95% of the determined ones come from a single true class (§4).

"Within a model only 0.43–0.80" is therefore partly a power statement. With 100 images and about 10 minority labels, a per-model AUROC has a standard error of roughly 0.1. Say this next to the number, or score more images per model.

### N7 [minor]. W10 is now attached to the headline population.

The models with r_c ≤ 50 that define the FPR are also the ones with the most misclassified aligned images (Pet 13–21%). Report a robustness row restricted to correctly classified images. `pred_aligned` is already in the behaviour CSVs, and the per-image AUCs are in `scores_oracle.csv`, so no new runs are needed.

### Numbers checked and found consistent

All of the following match `responsiveness_corrected.csv`, `behaviour_summary.csv` and `confound.csv` (§8 and §F):
- Abstract and 5.2: 27/28 significant cells, ρ 0.49–0.99; natural-image ρ 0.88–0.99; FPR 75–92% (Pet) and 100% (CIFAR); 40–70% for inpaint and insertion; Pet switches 1.3/4.6/5.4 with blur "> 0.5 at lowest".
- Section 5.4: pooled AUROC 0.75–0.90; within-model 0.43–0.80 (natural) and 0.45–0.66 (MNIST); calibration gap 0.27–0.63; all seven behavioural switches.
- Section 5.5: size-matched 0.007–0.022 → 0.275–0.373; switches 38.7–39.3 → 0.9–8.6; FPR 0.03–0.12 → 0.34–0.50.
- Table 4 text: CIFAR R–O 75–81% / 56–72% against 13–23%; Pet 49–53% against 35–38%.
- Section 5.9: CIFAR CNN partial 0.78–0.98; MNIST texture −0.26 to 0.53; 4 of 7 settings collinear.
- Table 6 entries: Pet, MNIST, PneumoniaMNIST.

I found no number in the docx that contradicts a CSV. The problems are in what the numbers are taken to mean (N1–N3, N5).

---

## 4. Hostile-reviewer attacks on the current draft (condensed)

1. "Your behavioural ground truth is chosen post hoc, and your own check label flips the PneumoniaMNIST verdict from 0.46 to 0.90. Which one is the truth, and why is the other not in the paper?" (N1)
2. "Over-detection at low reliance is just the random-ordering preference over your object map. On CIFAR that map is a Gaussian. The shortcut plays no role." (N2)
3. "The 'behavioural switch' is reliance measured again. On CIFAR it sits in a 55-point gap that your own rule would call unidentified." (N3)
4. "All models share the same 100 images, yet you resample images per model. Your switch intervals are too narrow." (N4)
5. "Reliance and object competence are collinear in 4 of 7 settings, the shortcut is a white square, and every network is trained from scratch at modest accuracy (Pet 82% without the shortcut)." These are known and stated (W2, W8, W13), but they remain the main external-validity attack.

Points 1–4 can be answered from existing files. Point 5 needs the GPU work.

---

## 5. Ranked recommendations

Costs assume one student with AI help. CPU means the laptop and GPU means a Kaggle T4. Training costs come from the logged timings (about 25 min per Pet model, about 11 min per CIFAR ResNet-18 model); evaluation costs are estimates **[speculation]**.

| Rank | Recommendation | Fixes | Cost |
|---|---|---|---|
| **R1** | **Report both behavioural labels and say what each measures.** Add the `shortcut_dominant` (necessity) AUROCs to Table 6, or a twin table. Explain that on binary tasks every label-1 image of the primary label is a redundancy image. Reword "PneumoniaMNIST patch is the weak case" as label-dependent. Before any new GPU result, record in a dated amendment (A2) which label is primary for the new models. Optionally add a graded necessity score (Δ_S − Δ_O) and its Spearman with the per-image AUC gap. Flag the circularity of `shortcut_dominant` for black deletion on MNIST patch. | N1, W2 | 1 day, 0 GPU h (CSV only) |
| **R2** | **Decompose over-detection and add a shortcut-specific statistic.** Report the excess shortcut preference, pref(S > O) − pref(R > O), per model, and its ρ with r_c. This removes the object-vs-random component. Put S–O, R–O and S–R side by side in Table 3. In the abstract, highlight, README and SYNOPSIS, either drop "100% of CIFAR-10 models" or qualify it as a Gaussian-prior effect, and state the Pet R–O ≈ 0.5 finding as "metrics do not rank the real object mask above random pixels". | N2, W3, W4 (reframe) | 1 day, 0 GPU h |
| **R3** | **Crossed bootstrap and intervals for the behavioural switch.** Resample seeds within dose and one image index vector shared by all models (pigeonhole), and report it for switches and preferences, or both schemes. Add a CI for the behavioural switch and for the metric-minus-behavioural switch difference. Apply the ±10 identification rule to the behavioural switch and mark CIFAR-10 "not identified". Rephrase "replaces the assumed 50%" as "re-measures it per image; it agrees with r_c to within 2–4 points". | N3, N4 | 1 day, 0 GPU h (`xai/stats.py` + `xai/dose_analysis.py`; about 1 h CPU to re-run) |
| **R4** | **Uncertainty-aware over-detection, plus a correctly-classified robustness row.** Report the FPR with a margin (preference CI lower bound > 0.5, or preference > 0.6) next to the point FPR. Add a row restricted to aligned images the model classifies correctly (W10). | N5, N7, W10 | 0.5 day, 0 GPU h |
| **R5** | **Fill the reliance gaps (SD2), pre-registered as A2.** CIFAR ResNet-18: doses between −0.25 and 0 (for example −0.2, −0.15, −0.05 × 3 seeds = 9 models). Pet: −0.25, −0.15, −0.05 × 3 seeds = 9 models. This makes the CIFAR behavioural switch identifiable and brackets the Pet one. | N3, W2 | about 2 GPU h CIFAR + about 4 GPU h Pet training, plus about 2 GPU h evaluation and behaviour; 1 day of work |
| **R6** | **Pretrained ImageNet ResNet-18 on Pet (SD1) as a confirmatory replication.** Pre-register predictions on the now-fixed statistics: excess shortcut preference ρ > 0, over-detection with margin, AUROC under both labels, switch identification. This turns the exploratory headlines (FPR, behavioural AUROC, chosen after the original results were inspected) into a confirmatory test, which a Q1 reviewer will value more than any further re-analysis. It also lifts Pet's 82% no-shortcut accuracy and answers W13. | W13, N1 (post hoc), W2 | 1.5–2 days; 18 models ≈ 6–8 GPU h training + 3 GPU h evaluation |
| **R7** | **Matched-remainder and truncated-AUC re-evaluation (SD5), storing full curves.** Score shortcut-first vs object-first maps that share the same tail ordering, use common random numbers across maps (W11), and store the 21-point curves so that truncated AUCs (first k pixels) can be computed. Score 300 images instead of 100 on Pet and CIFAR to give the within-model AUROC usable power. | W4, W11, N6 | 2–3 days; about 8–10 h CPU (small models) + 4–6 GPU h (Pet and CIFAR ResNet) |
| **R8** | **Realistic shortcut on Pet (SD4).** A semi-transparent watermark or text tag with jittered position and contrast, 6 doses × 3 seeds. Vary availability (contrast) at one dose, after Hermann et al. Pre-register it together with R6. | W8 | 2–3 days; about 7 GPU h training + 3–4 GPU h evaluation |
| **R9** | **Exact ROAD (SD3)** on CIFAR ResNet-18 and Pet for the oracle maps, MoRF and LeRF. | W7 | 2 days; 2–4 GPU h |
| **R10** | **Hygiene.** Break `size_matched` ties at random (W5) inside the R7 run, and add one sentence that within-model AUROC is computed on about 6–25 minority images per model. | W5, N6 | 0.5 day, 0 GPU h (with R7) |

Suggested order:
- **Week 1 (CPU, no GPU):** R1–R4 and R10 text. Commit amendment A2, with the label choice and predictions for R5, R6 and R8, before any GPU run.
- **Week 2 (one or two Kaggle sessions, about 20 GPU h):** R5 and R6.
- **Week 3 (about 15 GPU h):** R8, then R7 if quota is left. R9 last.

The total is about 30–35 GPU h, which fits the weekly Kaggle quota over two weeks.

Value of the planned GPU work under this lens:
- **SD1 (R6):** highest. It is the only item that converts post-hoc headlines into confirmatory ones.
- **SD2 (R5):** high, because it fixes the CIFAR identification.
- **SD4 (R8):** medium-high for external validity.
- **SD3 (R9):** lowest for methodology, since inpainting is already disclosed as not being ROAD.

---

## 6. Projected score

- **R1–R4 done (CSV only, about 3.5 days): 7 / 10.** The behavioural and over-detection claims would then be stated at the strength the data support, and the intervals would be honest.
- **Plus R5 and R6 (pre-registered): 7.5 / 10.**
- **Plus R7 and R8: 8 / 10.** The remaining ceiling comes from the single metric family and the competence collinearity on MNIST and Pet, both of which are already stated as limitations.

---

## 7. External sources opened for this report

- Owen, *The pigeonhole bootstrap*, arXiv:0712.1111: <https://arxiv.org/abs/0712.1111>. The page opened, but its rendered text showed only navigation. The summary used here (separate resampling of rows and columns, consistent for crossed random-effects data) comes from the search-result snippet for this same URL. The claim that resampling both factors is mildly conservative is my background knowledge and is **unverified**.
- Watson, Gultchin, Taly, Floridi, *Local Explanations via Necessity and Sufficiency: Unifying Theory and Practice*, UAI 2021, arXiv:2103.14651: <https://arxiv.org/abs/2103.14651>. Abstract page opened; it argues that necessity and sufficiency are the basic building blocks of explanations and are applied inconsistently in XAI.

Content was rephrased for compliance with licensing restrictions.

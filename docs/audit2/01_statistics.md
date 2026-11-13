# Audit 2, reviewer 01: statistics and rigour

Project: *A Dose-Response Evaluation of Explanation Metrics under Controlled Shortcut Reliance in CNN Image Classifiers*

Lens: statistics and rigour. Are the new statistics (chance-corrected reliance r_c, balanced accuracy and over-detection rate, two-level bootstrap, BH/Holm, effect sizes, isotonic switch points, class-stratified behavioural AUROC) computed correctly, and are they reported correctly in `paper/paper_draft.docx`, `README.md`, `SYNOPSIS.md` and `docs/PAPER_OUTLINE.md`?

State audited: `main` at 80cd09f, 234 models. Everything was read-only. My scripts and their full outputs are in `docs/audit2/scratch/01_statistics/`:

- `extract_docx.py` produces `paper_text.txt`, the docx text including tables.
- `checks1.py` produces `checks1_out.txt`: recomputation, counts, CI position, FPR, gaps, metric pairs, behaviour and confound checks.
- `checks2.py` produces `checks2_out.txt`: identification of the behavioural switch, label sensitivity, undetermined rates, and a paired two-level bootstrap of the behavioural switch minus the metric switch (1000 replicates, seed 123).
- `checks3.py` produces `checks3_out.txt`: the PneumoniaMNIST band claim, and the calibration gap computed on determined images only.
- `checks4_out.txt`: the cells whose ρ estimate lies outside its own CI.

My first-audit counterpart is `docs/audit/01_statistics_rigour.md`.

## 1. Verdict

**Q1 readiness, statistics lens: 7 / 10 (first audit: 5 / 10).**

The fixes are real and mostly correct. I recomputed or cross-checked the following, and they match the CSVs:

- the main ρ values;
- every headline count (27/28, 28/28, 25/26/25 of 28, 18/126);
- the over-detection and true-positive rates;
- the switch identification on MNIST;
- the Pet pre-registration intervals;
- the band AUROCs;
- the tie rates;
- the confound table.

I found no number in the docx that contradicts its source CSV. The implementation is careful in several places:

- the bootstrap is paired across metrics;
- switch CIs are suppressed when fewer than half the replicates cross;
- BH and Holm are applied within declared families;
- the Holm adjustment is correct (checked by hand on the Grad-CAM case, §3 item M1);
- the post-hoc analysis choices are disclosed in the Limitations section.

It is not yet an 8, for four reasons:

1. **The new reference quantity has the first audit's main problem.** The behavioural switch replaces the assumed 50% ideal and appears in the abstract. On both CIFAR-10 settings it lies in a 41–43-point reliance gap with no model within ±10 points, so it fails the paper's own identification rule. On Pet it is identified from one side only. The paper prints these values without a status (N1).
2. **The behavioural switch depends on the label, and that is not reported.** The Pet value is 54.5% under the primary label, 29.4% under the moved-only check label and 49.6% under the dominance label. The label is undetermined for 33% of the Pet images, and that share rises with reliance (N2).
3. **The over-detection rate depends on the design and has no uncertainty.** It mixes models from r_c ≈ 0 to 50, so where the doses are placed changes it, as Appendix C itself shows. It is reported for 10–15 models without an interval. On MNIST every metric has FPR = 0 at r_c ≤ 10 (N3).
4. **The reliance–competence collinearity (first-audit M5) is disclosed but not addressed.** It remains the largest threat to inference in 4 of 7 settings.

These are fixable in about one week, and only one of them needs GPU time (SD2).

## 2. First-audit weaknesses: status

| First-audit item | Status | Evidence now |
|---|---|---|
| **B1** Switch not identified (MNIST gap 23–45%) | **Partly resolved** | A1 filled the MNIST gap. Every MNIST-patch switch now has a model 0.5–1.03 points away and is bracketed (`checks1_out.txt` H), and the widest raw gap in 5–95% is 53.1→62.2%, as the paper states. The isotonic estimator, the ±10 rule, bracketing, edge status and the 4PL sensitivity check are implemented. CIFAR and Pet gaps remain: r_c 42.7→96.0 (small CNN), 36.5→95.8 (ResNet-18) and 41.1→60.1 (Pet). The behavioural switch now sits in those gaps (N1). |
| **B2** 50% ideal assumed; error floor in reliance | **Resolved** | r_c is implemented as specified (`xai/stats.py`). Per-image causal labels and the behavioural switch replace the 50% assumption. The 15% PneumoniaMNIST floor is now reported as r_c 0.6% / 3.9%. Remaining caveat: the label sensitivity in N2. |
| **M1** No multiplicity control | **Resolved** | BH within families, Holm over 112 headline tests, p < 0.001 printing. Grad-CAM, MNIST texture: p = 0.0200 and Holm 0.0255 within the 35-test methods family. It is the largest p in the family, so Holm carries the previous step's value. This is correct, and the 06 note on it is accurate. |
| **M2** Seed-only bootstrap | **Resolved, with a new caveat** | Two-level bootstrap with paired resamples. The percentile CIs for ρ are biased downward (N4). |
| **M3** Unbalanced decision accuracy | **Resolved** | Balanced accuracy, TPR and FPR are reported. TPR = 1.00 in 23/28 cells and FPR > 0 in 26/28, both verified. New issue: the FPR depends on the design and has no CIs (N3). |
| **M4** "Inpainting most robust" by vote count | **Resolved** | `metric_pairs.csv` has a paired bootstrap, and "most robust" is gone. 18/126 pairs have p_bh < 0.05 and 10 have Holm p < 0.05 (verified). |
| **M5** Reliance collinear with competence | **Not resolved (disclosed only)** | `confound.csv` and Table 10 are correct for raw reliance. On r_c, the primary scale, PneumoniaMNIST-patch inpaint has partial ρ = 0.24 (p = 0.124), which Table 10 does not show (N6). No decorrelation experiment was run. |
| **m1** Kendall τ, n = 5 | **Resolved** | Exact null. The 06 correction of my first-audit tail probabilities is right: P(T ≥ 0.6) = 14/120, so only τ ≥ 0.8 gives p < 0.05. I accept the correction. |
| **m2** No third-party timestamp | **Partly resolved** | Commit hashes are cited in §5.3. There is still no OSF or Zenodo timestamp, and that matters for SD1 and SD4. |
| **m3** Stale counts | **Resolved** | 234 / 192 / 42, seven settings, inpaint upper bound. One leftover overclaim is in the README (N7). |
| **m4** First-crossing artefacts | **Resolved** | Switches are computed only for the two comparisons whose ideal crosses 0.5. Edge cases are flagged (e). |

## 3. New problems introduced by the fixes

Ordered by severity. "Abstract", "§" and "Table" refer to `paper_draft.docx`.

### N1 (major): the behavioural switch fails the paper's own identification rule on CIFAR-10 and is not bracketed on Pet

`checks2_out.txt` A applies the ±10-point rule from `stats.switch_identification` to the follow-rate curve:

| Setting | Behavioural switch (r_c %) | Nearest model | Below / above | Status under the paper's rule |
|---|---|---|---|---|
| CIFAR small CNN | 55.0 | 12.3 | 12.3 / 41.0 | **not identified** |
| CIFAR ResNet-18 | 52.9 | 16.4 | 16.4 / 42.9 | **not identified** |
| Pet | 54.5 | 5.7 | 13.4 / 5.7 | identified, **not bracketed** |
| MNIST patch / texture, PneumoniaMNIST patch / texture | 53.9 / 45.7 / 36.9 / 41.9 | 1.7–7.1 | ≤ 7.5 on both sides | identified and bracketed |

`behaviour_summary.csv` gives `observed_switch_rc_status` = "crossed" for every row. §5.4, Table 6, the abstract (54.5%), README finding 3 and SYNOPSIS finding 3 print these values without a status. The two-level bootstrap interval cannot show the gap. For CIFAR ResNet-18 it is 45.0–57.3 (`checks2_out.txt` D), although no model lies between r_c 36.5 and 95.8. This is the same artefact as first-audit B1, now in the reference quantity.

**What the data do support.** Every natural-image model with r_c ≤ 42.7 has a follow rate of at most 0.43 (`checks2_out.txt` B; Pet's highest is 0.43 at r_c 41.1). The ordering "metric switch ≪ behavioural switch" therefore holds on natural images even under the most conservative reading: the behavioural switch lies above about 41–43%, and the metric switches are 1.3–17.7%. The paired bootstrap agrees. The difference (behavioural − metric) excludes 0 for all 12 natural-image cells, with lower bounds of 30–48 points (`checks2_out.txt` D). Report the CIFAR behavioural switch as "> 43% (not identified)" and Pet as "54.5% (not bracketed)", or fill the gaps (R1).

### N2 (major): the behavioural switch depends on the label, and the label is undetermined for a third of the binary-task images

`checks2_out.txt` A gives the same isotonic crossing for the three labels that the code already computes:

| Setting | Primary (causal) | Moved-only check | Dominance check |
|---|---|---|---|
| Pet | 54.5 | **29.4** | 49.6 |
| PneumoniaMNIST texture | 41.9 | 34.1 | **62.3** |
| MNIST texture | 45.7 | 38.6 | 53.2 |
| CIFAR ResNet-18 / CNN, MNIST patch | 52.4–56.3 | | |

The causal label leaves an image undetermined when the model already predicts the shortcut's class without a shortcut. That happens for 32.7% of the Pet images (up to 49% in one model), 32.7% on PneumoniaMNIST patch and 25.0% on texture. The share rises with reliance: Spearman 0.75–0.85 on the binary tasks and 0.88–0.91 on MNIST, where the share itself is small (`checks2_out.txt` C). The determined subset is therefore selected in a way that depends on reliance.

The paper describes the rule (§3.6) but does not report these rates or the label sensitivity. The direction of the headline survives: even the lowest Pet value, 29.4%, is far above the Pet metric switches of 1.3–5.4%. The number "54.5%" in the abstract does not survive. It should be given with the label range.

### N3 (major): the over-detection rate depends on where the doses are placed and has no uncertainty

The FPR is the share of all models with r_c ≤ 50 for which preference > 0.5. Its value therefore depends on how many models sit near 0 and how many near 50, which is a design choice. Appendix C shows this: adding A1 models at steep doses raised the MNIST-patch black FPR from 0.048 to 0.125. Splitting the population (`checks1_out.txt` G):

| Shortcut vs full object | FPR, r_c ≤ 50 (paper) | FPR, r_c ≤ 10 | FPR, 10 < r_c ≤ 50 |
|---|---|---|---|
| Pet, black / blur | 0.75 / 0.92 (n = 12) | 0.50 / 0.83 (n = 6) | 1.00 / 1.00 (n = 6) |
| Pet, inpaint / insertion | 0.58 / 0.58 | **0.17 / 0.17** | 1.00 / 1.00 |
| CIFAR ResNet-18, black / blur | 1.00 / 1.00 (n = 15) | 1.00 / 1.00 (n = 9) | 1.00 / 1.00 |
| CIFAR ResNet-18, inpaint / insertion | 0.40 / 0.47 | 0.11 / 0.11 | 0.83 / 1.00 |
| MNIST patch, all four | 0.03–0.12 (n = 32) | **0.00** (n = 21) | 0.09–0.36 |

Consequences:

1. **The CIFAR black/blur over-detection is robust.** Those metrics prefer the shortcut map for every model, including the 9 ResNet-18 models at r_c ≤ 10.
2. **The Pet inpaint/insertion over-detection (0.58) comes from models at r_c 15–41.** Those models follow the shortcut on 14–43% of determined images (`checks2_out.txt` B), so they do not "barely use" it. The Highlight "over-detect shortcuts models barely use" is accurate for black/blur on natural images, and not for inpaint/insertion.
3. **There is no interval.** Clopper–Pearson intervals, treating models as independent and so optimistic, are wide. Examples: Pet black 0.75 [0.43, 0.95], Pet inpaint 0.58 [0.28, 0.85] and CIFAR CNN inpaint 0.70 [0.35, 0.93] (`checks1_out.txt` F).
4. **TPR and balanced accuracy rest on very few models in places.** On the CIFAR CNN only 2 models have r_c > 50, and on ResNet-18 only 3. The TPR of 1.00 there, and half of each balanced accuracy, therefore rest on 2–3 models.

### N4 (moderate): the percentile CIs for ρ are biased downward, and the p-values and CIs are not consistent

- **The point estimate sits high in its own interval.** Over the 196 metric cells, the median position of the estimate inside its CI is 0.73 (0 = lower end, 1 = upper end).
- **In 20 of 196 cells the estimate lies above the upper bound** (`checks4_out.txt`). Examples: MNIST patch, all four metrics against the size-matched map (ρ = 0.970–0.974, upper bounds 0.969–0.972), and Pet random vs full object, inpaint (0.974 vs [0.815, 0.968]). No headline (shortcut vs full object) cell is outside its interval, but in 12 of the 28 the estimate is within 0.02 of the upper bound.
- **Two mechanisms make this expected.**
  - Resampling models with replacement creates duplicate models, and so tied ranks.
  - Resampling images adds noise to each model's preference, which pulls the bootstrap correlations toward 0.

  The intervals are therefore wider below than they should be. This weakens one sentence in particular. "The two-level interval [0.58, 0.93] reaches below the 0.8 threshold" (§5.3, P1 black) is partly an artefact of this bias.
- **The p-values and CIs come from different procedures, and they disagree in 3 cells** (`checks1_out.txt` E). The p-values use Spearman's t over models; the CIs use the two-level bootstrap. Example: PneumoniaMNIST texture, insertion is marked † (p_bh = 0.085) while its CI [0.06, 0.49] excludes 0. The paper should say which one governs significance and why.

### N5 (moderate): the calibration gap uses mismatched denominators

The calibration gap is mean(preference − follow rate). Preference is averaged over all 100 images. The follow rate is averaged over the determined images only. If preference is restricted to the same determined images (`checks3_out.txt`):

- Pet: 0.32–0.37 → **0.22–0.32**.
- PneumoniaMNIST patch: black 0.03 → −0.14, and inpaint −0.03 → −0.23.
- CIFAR: essentially unchanged.

The sentence "on the natural images the calibration gap is positive in every cell (0.27–0.63)" stays true, but the range becomes 0.22–0.63.

### N6 (minor): Table 10 uses raw reliance, while "all primary analyses use r_c"

Table 10 and §5.9 print `partial_rel_given_acc` (raw R). On r_c (`partial_rc_given_acc`, same CSV):

- PneumoniaMNIST patch: black 0.31 (p = 0.045), blur 0.31 (p = 0.049) and inpaint 0.24 (p = 0.124, not significant). The raw values are 0.40, 0.42 and 0.33 (p = 0.034).
- MNIST patch black: 0.34 against 0.48.

The partial p-values in Table 10 are also not adjusted for multiplicity (28 tests).

### N7 (minor): wording and small items

- **§5.5, "every switch has a model within 1.0 points".** The black and blur distances are 1.03 and 1.02. This is fine at one decimal, but "within about 1 point" is safer.
- **The MNIST-patch switch is called early without a test.** §5.5 sets the MNIST-patch switches (38.7–39.3) "against a behavioural switch of 53.9%", which reads as "too early". My paired bootstrap (`checks2_out.txt` D) gives:
  - black [3.5, 28.9] and blur [4.0, 28.3]: earlier;
  - inpaint [0.1, 23.2]: borderline;
  - insertion [−2.7, 23.4]: not distinguishable;
  - PneumoniaMNIST-patch inpaint [−9.6, 31.9]: not distinguishable.

  The text should say which switches are earlier.
- **The one-sided test was a post-hoc choice.** The one-sided alternative for the stratified AUROC was chosen alongside the stratification, after the original results had been seen. The Limitations section names the stratification but not the one-sidedness.
- **The README contradicts itself.** README supporting-study finding 2 is headed "The sanity check rewards the wrong methods", but its body correctly says agreement is "not shown" and that "the study cannot show disagreement either". The heading is still the old overclaim.
- **§5.2 mixes ranges.** It gives "ρ = 0.27–0.99" for the 27 rising cells. The abstract and README correctly give 0.49–0.99 for the 27 significant cells, so §5.2 includes the non-significant cell in the range.
- **The PneumoniaMNIST band claim checks out.** It is 0.005–0.078, with 18–89 non-followed images per band. It rests on only 18 negative images in the ≥ 80 band. The paper says this, which is good.

## 4. Numbers verified as correct

These all match their CSVs (`checks1_out.txt` A–M, `checks3_out.txt`):

- **Model counts:** 234 = 192 + 42 per setting.
- **ρ recomputation:** recomputed from `preference.csv`, ρ_rc matches the CSV to within 0.0004, the gap coming from 4-dp rounding in `preference.csv`.
- **Shortcut vs full object counts:** 27/28 BH-significant with significant ρ 0.488–0.986, and 27/28 under headline Holm.
- **Metric-variant counts:** inpaint 28/28 (lowest 0.505); black 25, blur 26 and insertion 25 of 28.
- **Metric pairs:** 18/126 at p_bh < 0.05.
- **FPR and TPR:** FPR 0.75–0.92 (Pet black/blur) and 1.00 (CIFAR); TPR 1.00 in 23/28.
- **Pet pre-registration:** P1 0.850 [0.582, 0.934] and P2 contrast 0.142 [0.086, 0.202].
- **Behavioural AUROC ranges:** pooled stratified 0.75–0.90 and within-model 0.43–0.80 on natural images.
- **Behavioural switches:** 54.5, 52.9, 55.0, 53.9, 45.7, 36.9 and 41.9.
- **Tie rates:** maximum 14.3% (PneumoniaMNIST) and 0.27% (MNIST).
- **Confound table:** Table 10 values for raw reliance.
- **Grad-CAM, MNIST texture:** mass 0.100 → 0.098.
- **Within-model significance:** 12 of 28 within-model stratified AUROCs are significant after headline Holm, including all 4 on Pet and none on MNIST patch. This is consistent with the paper's "only weakly".

## 5. Recommendations (ranked)

Costs assume one student with AI help: days of work, laptop CPU hours, and Kaggle T4 GPU hours. GPU training times come from `models.csv`: about 11 min per CIFAR ResNet-18 model and 25 per Pet model. Evaluation times, including the behaviour step, are my estimates.

| # | Fixes | Recommendation | Cost |
|---|---|---|---|
| R1 | N1, B1 remainder | **Apply the identification and bracketing rule to the behavioural switch, and fill the gaps that hold it (this is SD2).** Report a status column in Table 6. Until the gaps are filled, write CIFAR as "> 43% (not identified)" and Pet as "54.5% (not bracketed)". Then add models in the gaps: CIFAR ResNet-18 doses −0.2, −0.15, −0.05 (3 seeds, 9 models), Pet −0.25, −0.15, −0.05 (3 seeds, 9 models), and the CIFAR small CNN at −0.2, −0.05 (2 seeds, on CPU). Commit a dated amendment A2 first, as with A1. | Reporting: 0.5 day, 0 GPU. Runs: about 1.5 days of work; ≈ 1.7 h (CIFAR) + 3.8 h (Pet) training + ≈ 2 h evaluation ≈ **7–8 GPU h**; CPU ≈ 1.5 h for 4 small-CNN models |
| R2 | N3 | **Report over-detection in a form that does not depend on the design, with intervals.** (a) Give the isotonic-fitted preference at fixed r_c (for example 5, 10 and 25%) with two-level CIs, computed from the existing `boot` arrays. (b) Split the FPR at r_c ≤ 10 and 10–50, with two-level bootstrap or Clopper–Pearson CIs. (c) Flag balanced accuracy and TPR wherever n_above < 5. (d) Restate the Highlight as "black and blur deletion over-detect shortcuts on natural images". | 1 day, 0 GPU |
| R3 | N2 | **Label sensitivity.** Add a table with the behavioural switch and the within-model AUROC under all three labels (already in `behaviour_models.csv` and `behaviour_summary.csv`), and the undetermined share per setting with its correlation with r_c. In the abstract, give the Pet behavioural switch with its label range (29–55%) or state that it is "under the causal label". | 0.5 day, 0 GPU |
| R4 | M5 (open) | **Decorrelation experiment.** On MNIST, vary the shortcut's salience (patch intensity, for example 0.25–1.0) at dose 0 and at one small positive dose, so that reliance changes while accuracy without the shortcut stays high. Show that preference still follows r_c at fixed competence. This is the only way to turn Table 10 from a limitation into evidence. Pre-register the prediction (ρ > 0.8, partial ρ > 0.5). | 2–3 days; CPU ≈ 1 h training (≈ 36 models at 1.7 min) + ≈ 2 h evaluation; 0 GPU |
| R5 | N4 | **Fix the CI method for ρ and align the tests with the intervals.** Use BCa or bias-corrected percentile intervals, or report the bootstrap bias next to each ρ. Break the ties from duplicated models with a tiny jitter on reliance in each replicate. State that significance comes from the model-level Spearman test, and explain the 3 disagreements. Reword P1 black ("the interval includes 0.8") once the corrected CI is known. | 1 day, 0 GPU (about 30 min CPU to rerun `dose_response.py analyze`) |
| R6 | N7, N1 | **Test the switch differences directly.** Add the paired two-level bootstrap of (behavioural − metric) switch, as in `scratch/01_statistics/checks2.py` D, to `behaviour_summary.csv`. Say "earlier" only where the CI excludes 0: all natural-image cells, MNIST black and blur, but not MNIST insertion or PneumoniaMNIST-patch inpaint. | 0.5 day, 0 GPU |
| R7 | N5, N6 | **Make the denominators consistent.** Compute the calibration gap on determined images only (natural-image range 0.22–0.63). Put Table 10 on r_c, with raw values in parentheses, and add BH within the table. | 0.5 day, 0 GPU |
| R8 | N1, N2, within-model AUROC | **More images per model on the natural-image settings** (100 → 300 for Pet and both CIFAR networks). At 100 images, a Pet model at r_c 15–23% has about 70 determined images and only about 10–15 followed ones (follow rate 0.14–0.22), so each within-model AUROC is very noisy and the median-of-models summary is weak. Evaluation only, no retraining. | 1 day; ≈ **3–4 GPU h** (Pet + ResNet-18, estimate); small CNN on CPU ≈ 1 h |
| R9 | m2 | **Third-party timestamp.** Before any SD1, SD4 or R4 run, post its predictions on OSF or a Zenodo snapshot, and cite the DOI next to the commit hash. | 0.25 day, 0 GPU |
| R10 | N7 | **Fix the wording:** the README sanity-check heading, the §5.2 ρ range, "within 1.0 points", and the one-sided AUROC choice in the Limitations. | 0.25 day, 0 GPU |

**The planned GPU work (SD1–SD4), from this lens:**

- **SD2 is the most valuable.** It is R1 and directly fixes N1, the remaining part of B1. Do it first.
- **SD1 (pretrained ResNet-18 on Pet) and SD4 (realistic watermark) add external validity.** They do not fix any statistical flaw. They are valuable for the venue, but each adds a new setting with 18 models and the same gap risks. Use doses chosen to give reliance across 30–70%, and pre-register them (R9).
- **SD3 (exact ROAD) adds a metric family.** It is neutral for statistics.

Total for R1–R10: about 8 working days, ≈ 11 GPU h (R1 + R8) and ≈ 5 CPU h, within one week's Kaggle quota.

## 6. Projected score

- **R1–R3, R5–R7 and R10** (about 5 days, ≈ 8 GPU h): **8 / 10**. The headline quantities would then be identified, label-robust and design-independent, with consistent inference.
- **Adding R4** (the decorrelation experiment) and **R8**: **8.5 / 10**. The remaining limits would be 2–3 seeds per dose and one metric family.

## 7. External references (opened during this audit)

- Saravanan, Berman and Sober, *Application of the hierarchical bootstrap to multi-level data in neuroscience* (author manuscript, PMC): <https://pmc.ncbi.nlm.nih.gov/articles/PMC7906290/>. It supports resampling level by level for nested data, which the project now does. The authors also test the concern that the bootstrap is conservative, and report that it is not conservative for hierarchical data. That is consistent with my view that the two-level design is right and that the remaining interval problem in N4 is specific to rank correlations under resampling with replacement.

The tie-and-attenuation explanation in N4 rests on the measured CI positions in `checks4_out.txt` and on the mechanics of resampling with replacement. I tried to open a Stack Exchange discussion of this effect, but the page returned HTTP 403, so it is not cited. Holm, Benjamini–Hochberg, Clopper–Pearson and the isotonic (PAVA) fit are standard procedures, and no URL is cited for them.

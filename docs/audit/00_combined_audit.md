# Combined audit: Q1 readiness

Project: **A Dose-Response Evaluation of Explanation Metrics under Controlled Shortcut Reliance in CNN Image Classifiers**

This report merges five independent read-only reviews. Nothing outside `docs/audit/` was changed by the aggregation. No reviewer reported changing project files; their read-only analysis scripts are under `docs/audit/scratch/01`, `03`, `04` and `05`.

**Aggregate verdict: CHANGES_REQUESTED.** The design is sound, but the paper is not ready to submit to a Q1 journal. The main blockers can be fixed in 3 weeks, and most of the fixes need no GPU.

---

## 1. Reviewer status and scores

| # | Report | Lens | Present | Score | Ranked recs with costs |
|---|---|---|---|---|---|
| 01 | `01_statistics_rigour.md` | Statistics and experimental rigour | yes | 5/10 | 10 |
| 02 | `02_literature_novelty.md` | XAI literature and novelty | yes | 5/10 | 8 |
| 03 | `03_methodology_validity.md` | Methodology and threats to validity | yes | 5/10 (7/10 after the cheap fixes, 8/10 with a realistic shortcut) | 9 |
| 04 | `04_venue_framing.md` | Venue strategy and framing | yes | 5/10 for Neurocomputing / Machine Learning; 4/10 for Pattern Recognition / TNNLS | 9 |
| 05 | `05_high_leverage_additions.md` | High-leverage additions | yes | 5/10 (7/10 after its top 6 items) | 11 |

No reviewer failed. All five reports contain a numeric score and at least 5 ranked recommendations with cost estimates.

**Consensus Q1-readiness score: 5 / 10.** All five reviewers gave 5/10 independently. Reviewers 03 and 05 both estimate about **7 / 10** once the must-do items below are done. That projection is their judgement, not a measurement.

## 2. What every reviewer agreed is strong

- 192 independently trained models, with reliance **measured behaviourally** on conflicting images rather than assumed from the dose.
- Fixed oracle maps (shortcut, object, size-matched, placebo, random), so metrics are evaluated separately from attribution methods.
- A paired image design: the same 100 test images per base (`IMAGE_SEED = 0`).
- A pre-registered Pet study. The commit order `a962b6c` (predictions) → `20694f2` (code) → `d823737` (amendment) → `1e29918` (results) matches the claims. The P3 failure for black deletion (ρ = 0.75 vs random, 0.73 vs placebo) is reported honestly.
- The monotone dose-response holds. Of 196 metric tests, 166 have p < 0.05, 164 survive Benjamini–Hochberg and 154 survive Holm (01).
- The CIFAR-10 form bias is large and robust. Black deletion prefers a random map over the object map 0.815 of the time (95% seed+image CI 0.755–0.870) at 4% reliance (01).
- With real Pet masks, the black/blur random-vs-object bias drops to 0.53/0.49. This supports the hypothesis that the centred-Gaussian object map inflated the CIFAR-10 effect.
- Reviewer 03 found **no code bug that invalidates a computed number**. Masks, AUC, preference direction, tie handling, the placebo/random builders and the train/test split behave as documented.

## 3. Consensus weaknesses (de-duplicated)

| ID | Weakness | Raised by | Severity |
|---|---|---|---|
| W-A | The paper draft has no Pet results and no pre-registration section. `build_paper.py` line 141 asserts `N_MODELS == 174`, but `models.csv` has 192 rows, so the builder stops if re-run unchanged. | 02, 03, 04, 05 | Blocker |
| W-B | Reliance on binary tasks has a floor equal to the error rate. "Lowest reliance 15%" on PneumoniaMNIST is about 0% after chance correction. Pet switch points of 19–25% become about 1–7%. | 01, 03 | Blocker |
| W-C | The 50% switch ideal is assumed, not derived. Preference is scored on aligned images, while reliance is measured on conflicting images. | 01, 02, 03, 05 | Blocker |
| W-D | The switch point is not identified in the gaps. MNIST patch has no model between 23.0% and 45.2% reliance, and every reported switch (30–42%) lies in that gap. Three estimators give 30–42, 33–51 and 49–52%. | 01, 05 | Blocker (01), minor (05) |
| W-E | No multiple-comparison correction across about 231 tests. The bootstrap resamples only 2–3 seeds per dose and ignores image noise, so intervals are too narrow (e.g. CIFAR CNN inpaint ρ CI [0.986, 1.000]). No effect sizes are reported. | 01, 03, 05 | Major |
| W-F | Decision accuracy is unbalanced. 0.167 on CIFAR-10 means "always says shortcut" (balanced accuracy 0.50). | 01, 03, 05 | Major |
| W-G | Reliance is almost collinear with object competence (Spearman −0.96 to −0.99 in five of seven settings). On MNIST, partial ρ falls from about 0.96 to between −0.28 and 0.51. Only the CIFAR small CNN separates them (partial ρ 0.78–0.98). | 01, 03 | Major (01), blocker (03) |
| W-H | The closest prior work is not cited: BAM, SMERF, OSCAR, Bastings et al., plus confound papers (Blücher, ROAR pitfalls, Gomez, Gevaert, Quantus, FunnyBirds). The novelty sentence is too broad. | 02, 03, 04, 05 | Blocker (02), major (04, 05) |
| W-I | Current target is SN Computer Science, which SCImago rates **Q2**. The paper has no single thesis, and it leads with MNIST. | 04 | Blocker / major |
| W-J | All metrics come from one family (ordered removal), and the inpainting is "ROAD-inspired", not ROAD. | 02, 03, 05 | Major |
| W-K | All models are trained from scratch, and the ResNet uses a CIFAR stem at 128 px. | 02, 03, 04, 05 | Major |
| W-L | The only shortcut on natural images is a synthetic white square, and only predictivity is varied, not availability. | 03, 05 (02 optional) | Blocker (05), major (03) |
| W-M | The "object" oracle is not an object on PneumoniaMNIST (an intensity ranking of the whole X-ray; 97.5% of pixels > 0.1) or on CIFAR-10 (a centred Gaussian). | 03 | Blocker |
| W-N | Most of the AUC comes from the random tail: the shortcut is about 1% of pixels, and each of the 20 steps removes 5%. Blur fill leaves 20–25% of the patch intensity in place. | 03, 05 | Major / minor |
| W-O | "Inpainting deletion is the most robust variant" rests on vote counting. A paired bootstrap shows black deletion is significantly *better* on PneumoniaMNIST texture (Δρ = −0.197, CI −0.385 to −0.039). | 01, 03 | Major |

## 4. Ranked action plan

Costs are student days with AI help. GPU hours are Kaggle T4 hours. Where reviewers gave different estimates, the range covers all of them.

### Must-do (needed before any Q1 submission)

| # | Action | Fixes | Source recs | Days | GPU h |
|---|---|---|---|---|---|
| MD1 | **Bring the paper and docs up to the 192-model state.** Add Pet to `SETTINGS`, fix the `N_MODELS` assert, add the Pet panel and table rows, and add a "Pre-registered test" subsection with the P1–P4 table and commit links. State that P1–P4 were judged on point estimates, and give the two-level CIs (black P1: 0.60–0.93). Correct every item in Section 8. | W-A, Section 8 | 01-R9/R10, 02-#2, 03-R8, 04-R1 | 2–3 | 0 |
| MD2 | **Chance-correct reliance and replace decision accuracy.** Compute r_c = (R − f)/(100 − f), with f = error without shortcut/(C − 1). Report it next to raw reliance, recompute switch points on it, and rewrite the "lowest reliance 15%" text. Replace decision accuracy with balanced accuracy plus the false-positive rate at low reliance, and lead with "over-detection" (TPR is 1.00 in 23 of 28 cells). | W-B, W-F | 01-R4/R5, 03-R1 | 1 | 0 |
| MD3 | **Redo inference on the existing CSVs.** (a) Two-level bootstrap over seeds within dose and images within model. (b) Benjamini–Hochberg within each family, plus Holm for the headline claims. (c) Print p < 0.001 instead of 0.000. (d) Add CIs to `method_responsiveness`. (e) Report effect sizes (median per-image AUC gap) next to preferences. (f) Use a model-level monotone switch estimator: isotonic primary, 4-parameter logistic with free asymptotes as a sensitivity check. Mark a switch "not identified" when no model lies within ±10 points. Report switches only where the ideal curve crosses 0.5. (g) Paired bootstrap for metric-vs-metric claims; rewrite "most robust" as the two supported claims. (h) Give the exact Kendall τ null for n = 5 in the supporting study and reword "does not agree" as "is not shown to agree". | W-D (partly), W-E, W-O | 01-R3/R6/R8, 02-#4, 03-R5, 05-R3 | 2–3 | 0 |
| MD4 | **Per-image behavioural ground truth, replacing the 50% assumption.** For each scored image, use forward passes with the shortcut removed or moved to a wrong class's position to label whether the model follows the shortcut on that image. Report each metric's AUROC (or accuracy) against that label as a function of reliance. Keep the aligned-image curves as secondary. Pilot on 6 MNIST models first (05). | W-C | 01-R2, 03-R2, 05-R2 | 2–3 | 0.3–4 (CPU for the small models) |
| MD5 | **Rewrite related work and narrow the novelty claim.** Cite and differentiate BAM, SMERF, OSCAR, Bastings, Lin, Blücher, ROAR pitfalls, Gomez, Gevaert, Quantus, Hesse 2024, FunnyBirds, Lapuschkin, Suhail, XAI-Units and Miró-Nicolau 2026. Use the positioning table in 02 §2. Link each finding to its known mechanism (ROAD, ROAR pitfalls, Gomez, SaCo, Bilodeau). Fix the citation errors in Section 8. Record the search date. | W-H | 02-#1/#7, 04-R4, 05-R1 | 2 | 0 |
| MD6 | **Reframe and retarget to Q1.** Organise the paper around one thesis: "metrics respond to reliance but are not calibrated to it; their verdicts depend on explanation size and form". Lead with the natural-image results, and present MNIST and PneumoniaMNIST as the mechanism settings. Switch to the target journal's template, move the sanity-check study to an appendix, and confirm the no-fee route in a normal browser. | W-I | 04-R2/R3/R7/R8 | 2.5–3 | 0 |
| MD7 | **Report the validity limits that cannot be fixed by new runs.** Report Spearman(reliance, accuracy without shortcut) and the partial correlations per setting, and lead with the CIFAR small CNN. State that the PneumoniaMNIST "object" map is an intensity ranking, that CIFAR uses a Gaussian prior (with Pet as the primary natural-image evidence), that blur leaves 20–25% of the patch in place, and that `size_matched` picks positions among ties. | W-G, W-M, W-N | 01-R7 (min), 03-R3a/R4a/R7 | 1 | 0 |
| MD8 | **Fill the MNIST and PneumoniaMNIST reliance gaps (CPU only).** Add MNIST doses 0.0075, 0.015 and 0.03 and PneumoniaMNIST doses −0.075 and −0.025, with 5–8 seeds at the steep doses. Pre-register the added doses as an amendment first. | W-D | 01-R1 (CPU part) | 1.5 | 0 (about 4–6 CPU h) |

Must-do total: about **14–17 days**, about **0.3–4 GPU h**.

### Should-do (raises acceptance odds; do in the order listed, as GPU time allows)

| # | Action | Fixes | Source recs | Days | GPU h |
|---|---|---|---|---|---|
| SD1 | **Pretrained ImageNet ResNet-18 on Pet** (standard stem, about 10 fine-tuning epochs, 6 doses × 3 seeds). Pilot at d = 0 and d = −0.5 first, because ImageNet contains cat and dog breeds. Pre-register as an amendment. Reviewer 04 says this is required before submitting to Pattern Recognition. | W-K | 02-#6, 04-R5, 05-R4 | 3 | 3–8 |
| SD2 | **Fill the CIFAR-10 ResNet-18 and Pet reliance gaps** (ResNet-18 doses −0.075 to −0.02; Pet −0.15 and −0.05). | W-D | 01-R1 (GPU part), 05-R7 | 1 | 3–6 |
| SD3 | **Exact ROAD (official noisy linear imputation, MoRF and LeRF) on the oracle maps.** Measure agreement with the project's inpainting, and test whether ROAD removes the black/blur form bias. Run Pet ROAD on a Kaggle CPU notebook. | W-J, W-N | 02-#3 (part), 03-R7, 05-R5 | 2–3 | 1–4 |
| SD4 | **Realistic shortcut on Pet**: a semi-transparent class-coded watermark or text tag with jittered position, motivated by Lapuschkin et al. Run 18 models, preferably on the SD1 pretrained network. | W-L | 03-R6, 05-R6, 02-#8 (optional) | 3.5–4 | 3–4 pretrained; 10–11 from scratch |
| SD5 | **Oracle and AUC-tail sensitivity**: a matched-remainder variant, a truncated first-k AUC, 100 or log-spaced steps on a subset, random tie-breaks in `size_matched`, and a lung-field prior or an explicit drop of the "full object" reading on PneumoniaMNIST. One shared re-evaluation run. | W-M, W-N | 03-R4, 05-R8 | 2–3 | 1–5 |
| SD6 | **Release a reusable artifact**: a Zenodo DOI for code, CSVs and checkpoints, a `validate_metric()` entry point and a metric-validation checklist in the paper. | Contribution clarity | 04-R6, 05-R9 | 1–2.5 | 0 |

### Optional (only if time remains)

| # | Action | Source recs | Days | GPU h |
|---|---|---|---|---|
| OP1 | Extra faithfulness metrics from other families (Faithfulness Correlation, SRG = LIF − MIF, IROF, Region Perturbation), on oracle maps only. Reviewer 05 warns that binary maps create many ties for correlation-based metrics. | 02-#3, 05-R11 | 2.5–4 | 2–6 |
| OP2 | MetaQuantus versus dose-response ranking (MNIST patch + Pet, 3 doses). | 02-#5 | 3 | ~3 |
| OP3 | Decorrelation experiment on MNIST: vary shortcut salience at a fixed dose, or a dose × object-degradation grid, so reliance changes while competence stays fixed. | 01-R7, 03-R3b | 2–3 | 0 (CPU) |
| OP4 | Texture shortcut on Pet. Overlaps with SD4. | 05-R10 | 2.5 | ~3 pretrained; ~11 from scratch |
| OP5 | Hygiene: a robustness table on correctly classified images only, common random numbers for tie-breaks, and a note on the CIFAR-style stem. | 03-R9 | 1 | 0–2 |
| OP6 | Third-party timestamp (OSF or Zenodo) for any new pre-registered amendment. | 01-R9 | 0.25 | 0 |

**Not recommended within 3 weeks:** a ViT (the fixed title says "CNN"; 05), Waterbirds-style backgrounds (more than a week of data work; 05), and more attribution methods or blanket extra seeds (low gain per GPU hour; 05). Targeted extra seeds at the steep doses (MD8, SD2) are the exception.

## 5. Three-week schedule

The schedule assumes about 6 working days per week. GPU jobs run on Kaggle in the background while analysis and writing continue on the laptop. Each Kaggle session stays under 12 h.

| Week | Laptop / CPU work | Kaggle GPU work | GPU h |
|---|---|---|---|
| 1 | MD2, MD3 and MD7 on existing CSVs. Start the MD8 CPU training runs. Write and commit the dose and pretrained amendments before running anything new. Code the MD4 forward passes and pilot them on 6 MNIST models. Code SD1 and run its pilot. | SD1 pilot (~0.5), then the full SD1 run | 3–8 |
| 2 | MD4 for all small models. Recompute everything with the MD8 models. MD5 related work. Code SD3, and run Pet ROAD on a Kaggle CPU notebook. Code SD4 and pilot it if SD1 worked. | MD4 for ResNet and Pet models, SD2, SD4 pilot and full run | 7–14 |
| 3 | MD1 and MD6: rebuild the paper from CSVs, write the thesis-led abstract and introduction, apply the journal template, update figures, write the cover letter and declarations. SD6 release. Check every number in the paper against the CSVs. Confirm the no-fee route in a browser and submit. | SD3 / SD5 re-evaluation if not done (≤5) | 0–5 |

Total GPU use is about 10–27 h over 3 weeks, below the quota of about 30 h per week. If week 2 slips, drop SD4 and SD5 first and keep MD1–MD8 and SD1.

## 6. Points where reviewers disagreed

1. **How to replace the 50% ideal (MD4).** Reviewer 01 scores oracle maps on conflicting images with the predicted class as target. Reviewer 03 compares per-image probability drops Δ_S (shortcut removed) and Δ_O (shortcut placed on a different-class image). Reviewer 05 labels each aligned image by whether the model follows the shortcut when it is moved. All three agree on the aim. Reviewer 05's version is the cheapest to pilot. Pick one as primary and report another as a check.
2. **Switch-point estimator.** Reviewer 02 proposes a 2-parameter logistic fit (EC50 and slope). Reviewer 01 shows that this fit is misspecified because preference saturates at 0.75–0.9, and that it moves the MNIST switch to 49–52%. Reviewers 03 and 05 prefer mixed-effects logistic models on per-image data. The plan uses isotonic regression as primary and a 4-parameter logistic as the sensitivity check.
3. **Multiplicity family.** Reviewer 03 asks for Holm across about 196 tests. Reviewer 01 notes that Holm removes the Grad-CAM texture trends and some PneumoniaMNIST results, while BH keeps 164 of 166, and asks the authors to choose and state a family. The plan uses BH within families plus Holm for headline claims.
4. **The inpainting claim.** Reviewer 04's draft abstract keeps "the only variant that responded significantly in every setting". Reviewers 01 and 03 show that "most robust" fails a paired test, and that the README's "all eight settings, ρ ≥ 0.58" is wrong. Use reviewer 01's narrower wording, and re-check the "every setting" claim with Pet included.
5. **Pretrained backbone choice.** Reviewer 02 suggests ResNet-50 or ViT-S. Reviewers 04 and 05 recommend ResNet-18, and reviewer 05 notes that a ViT falls outside "CNN Image Classifiers" in the fixed title. The plan uses ResNet-18.
6. **Priority of a realistic shortcut.** Reviewer 05 calls it a blocker and reviewer 03 calls it major. Reviewer 02 lists it as optional, and reviewer 04 does not raise it. It is SD4 here: high value, but GPU-dependent.
7. **Metric breadth.** Reviewer 02 calls the single metric family a major gap and adds MetaQuantus. Reviewer 05 ranks extra Quantus metrics low because binary maps create ties. All reviewers who discuss it agree on exact ROAD (SD3).
8. **Pet at low reliance.** Reviewers 01 and 03 say chance correction makes the Pet "switches too early" finding *stronger* (corrected switches about 1–7%). Reviewer 05 shows that the Pet low-reliance AUC gaps are only 0.01–0.04, so those preferences are near-ties. Report both facts: the effect-size table limits how strongly the corrected switch points can be read.
9. **GPU estimates.** Per-model Pet cost is about 34 min train+eval (05, from timestamps) against about 30–35 min (01, 03). Pretrained ResNet-18 estimates range from about 3 GPU h (05) through 5–6 (04) to 6–8 (02). The plan uses ranges, and the pilot model settles the number.
10. **Reliance-gap figures.** Reviewer 01 gives model-level gaps (e.g. Pet 57.1→72.3%, CIFAR ResNet-18 39.0→96.2%). Reviewer 05 gives dose-mean gaps from `summary.md` (Pet 47→82%, ResNet-18 36→97%). Both are correct for their level. Use the model-level figures to choose new doses.
11. **Title.** Reviewer 04 notes that "Faithfulness Metrics" would match the field's search terms better, and recommends against a subtitle. The project rules fix the title, so it stays unchanged unless the user decides otherwise.

## 7. Candidate Q1 journals

The quartile and SJR 2025 values come from SCImago's public widget images, which reviewer 04 opened (copies in `docs/audit/scratch/04/`). The SCImago journal pages, ScienceDirect and SpringerLink pages were blocked (HTTP 403, bot check, or a login loop). **The free subscription route is verified only for TNNLS.** For the others it is unverified and must be checked in a normal browser before submitting (about 10 minutes per journal).

| Rank | Journal (publisher) | Quartile, SJR 2025 (verified via widget) | Fee route | Scope fit / odds (reviewer 04, speculative) | Verified links |
|---|---|---|---|---|---|
| 1 | **Neurocomputing** (Elsevier) | Q1 Artificial Intelligence, 1.47 | Elsevier hybrid; subscription route with no fee **UNVERIFIED** for this journal | Good: empirical NN/ML studies, XAI common. Odds moderate–good | [Wikipedia](https://en.wikipedia.org/wiki/Neurocomputing_(journal)); [SCImago widget](https://www.scimagojr.com/journal_img.php?id=24807) |
| 2 | **Machine Learning** (Springer) | Q1 AI, 1.03 | Hybrid; non-OA option implied by a search snippet, **UNVERIFIED** | Good for a measurement-validity framing; values pre-registration. Odds moderate | [Wikipedia](https://en.wikipedia.org/wiki/Machine_Learning_(journal)); [SCImago widget](https://www.scimagojr.com/journal_img.php?id=24775) |
| 3 | **Pattern Recognition** (Elsevier) | Q1 AI, 2.15 | Search snippet says "No open access publication fee" for the subscription option, **UNVERIFIED** | Very good scope, highest prestige among realistic options. Odds low–moderate; submit here first only if SD1 is done | [Wikipedia](https://en.wikipedia.org/wiki/Pattern_Recognition_(journal)); [SCImago widget](https://www.scimagojr.com/journal_img.php?id=24823) |
| 4 | **Neural Networks** (Elsevier) | Q1 AI, 1.55 | Elsevier hybrid, **UNVERIFIED** | Good. Odds moderate–low | [Wikipedia](https://en.wikipedia.org/wiki/Neural_Networks_(journal)); [SCImago widget](https://www.scimagojr.com/journal_img.php?id=24804) |
| 5 | **IEEE Trans. Neural Networks and Learning Systems** | Q1 AI, 3.15 | **Verified:** the $110/page charge is voluntary, but a mandatory $200/page applies beyond 10 pages. Free only if the paper fits in 10 IEEE pages | Good scope. Odds low (high bar) | [Wikipedia](https://en.wikipedia.org/wiki/IEEE_Transactions_on_Neural_Networks_and_Learning_Systems); [page charges](https://cis.ieee.org/publications/t-neural-networks-and-learning-systems/tnnls-page-charges); [SCImago widget](https://www.scimagojr.com/journal_img.php?id=21100235616) |

Other journals reviewer 04 checked:
- **Backups:** Data Mining and Knowledge Discovery (Q1, 1.2; [widget](https://www.scimagojr.com/journal_img.php?id=13579); [Wikipedia](https://en.wikipedia.org/wiki/Data_Mining_and_Knowledge_Discovery)), Pattern Recognition Letters (Q1 CV&PR, 0.79; [widget](https://www.scimagojr.com/journal_img.php?id=24825); [Wikipedia](https://en.wikipedia.org/wiki/Pattern_Recognition_Letters); its short format would force cutting controls) and CVIU (Q1 CV&PR, 0.82; [widget](https://www.scimagojr.com/journal_img.php?id=24161)).
- **Not recommended as first choice (scope):** ESWA ([widget](https://www.scimagojr.com/journal_img.php?id=24201)), Knowledge-Based Systems, EAAI and Information Sciences (widgets [24772](https://www.scimagojr.com/journal_img.php?id=24772), [24182](https://www.scimagojr.com/journal_img.php?id=24182), [15134](https://www.scimagojr.com/journal_img.php?id=15134)).
- **Current target, SN Computer Science**, is **Q2** (0.45; [widget](https://www.scimagojr.com/journal_img.php?id=21101083109)) and does not meet the Q1 goal.

Submit to one journal at a time. Q1 first decisions commonly take 1–4 months (04, not checked per journal), so the 3 weeks cover the submission package, not acceptance.

## 8. Bugs and incorrect claims in current documents

Reviewer 03 found no code bug that invalidates a computed number. The issues below are in the documents, code comments, interpretation and statistics.

**Stale or omitted content**
1. `paper/build_paper.py` line 141: `assert N_MODELS == 174`, but `models.csv` has 192 rows. The builder stops if re-run unchanged (02, 03, 04).
2. `paper/build_paper.py` has no Pet results. It still calls the real-mask test future work (around L1087–1090, L1120), and the `SAME_FORM` and `PR_DEL` tables exclude Pet (03, 04).
3. `README.md` L18 and L29 say 174 models and that Pet is "prepared but not trained yet". `SYNOPSIS.md` is stale in the same way (01, 03, 04).
4. `paper/build_paper.py` line 624 targets *SN Computer Science*, which is Q2 (04).
5. `paper/build_paper.py` around line 1125 flags the novelty search as unfinished (02).

**Incorrect or over-general claims**
6. README L23 says inpainting is "the only metric that responds significantly in all eight settings (ρ ≥ 0.58, p < 0.001)". There are 7 settings. Black and blur deletion are also significant in 7/7 for shortcut vs full object, and inpainting's minimum ρ across the shortcut comparisons is 0.569 (01, 03). `build_paper.py` line 1018 says "all six settings" and `PAPER_OUTLINE.md` contribution 5 says "eight" (01).
7. "Inpainting deletion is the most robust variant" (README finding 5; `build_paper.py` lines 1018–1026, 1103, 1136) is not supported by a paired test. Black deletion is significantly better on PneumoniaMNIST texture (01).
8. README L25, SYNOPSIS L108 and PAPER_OUTLINE contribution 3 say the preference "rises to 0.93–0.99 for every metric on the patch shortcut". This is false for PneumoniaMNIST patch (0.727–0.785) and for Pet black deletion (0.763 / 0.773) (03).
9. SYNOPSIS L112 and the PAPER_OUTLINE abstract say "Saliency, SmoothGrad and Grad-CAM move away from a diffuse shortcut". That holds only on MNIST texture. On PneumoniaMNIST texture, Saliency (ρ = +0.874) and SmoothGrad (+0.696) move *towards* it. Grad-CAM's MNIST change (0.100 → 0.098 mass) is negligible and fails Holm (01, 03).
10. README L26 gives the Integrated Gradients range as ρ = 0.86–0.97. The actual range is 0.856–0.986 (03).
11. README L178 and `build_paper.py` around L874–877, L909 and L1123 say the lowest reliance on PneumoniaMNIST was about 15%, not 0. The 15% is the error floor; chance-corrected reliance is about 0% (01, 03).
12. `PAPER_OUTLINE.md` says the MNIST inpaint switch interval "reaches about 51%". `responsiveness.csv` gives 49.954, and the two-level bootstrap gives 52.7 (01).
13. `build_paper.py` lines 899–903 call the black and blur switches "clearly before 50%". No model lies between 23% and 45% reliance on MNIST patch, so the number is not identified (01).
14. Pet P1 for black deletion is reported as "holds". It holds on the point estimate (0.85) only; the two-level CI is 0.60–0.93 (01).
15. The supporting study's statement that sanity-check rankings "do not agree" with ground truth rests on Kendall τ over 5 methods, where |τ| ≤ 0.2 cannot be told apart from zero (01).

**Citation errors**
16. `PAPER_OUTLINE.md` describes arXiv:2401.10640 as "Tabular/transparent models". It uses 52,000 synthetic 128×128 images with a decision tree (02).
17. The LATEC entry at `build_paper.py` line 248 has no title. The title is "Navigating the Maze of Explainable AI: A Systematic Approach to Evaluating Methods and Metrics", Klein et al., NeurIPS 2024 (02).
18. MetaQuantus should be cited as TMLR 2023, not arXiv. "Hedström 2024 (arXiv:2405.02383)" and "Sanity Checks Revisited" (arXiv:2401.06465) need to be disambiguated (02).

**Code comments and descriptions that misstate behaviour**
19. `dose_response.py` `size_matched` and the summary text say "the object's brightest pixels". On MNIST, stable argsort over tied saturated pixels (median 29.5 ties) picks the topmost pixels in row-major order (03).
20. `xai/data.py` `DecoyDataset.object_map` docstring assumes a dark background. On PneumoniaMNIST, 97.5% of pixels exceed 0.1 (03).
21. The blur deletion fill leaves 20–25% of the patch intensity in place, and no document mentions this (03).
22. At the last inpainting step all pixels are masked, the count clamps to 1 and the mean is 0, so the inpaint curve ends on a noisy black image (03).
23. `xai/models.py` `ResNet18Cifar` docstring says "final 4x4 feature maps", and `xai/explain.py` `grad_cam` says "6x6 to the 28x28". For Pet the maps are 16×16; for the CIFAR models they are 8×8 or 4×4 (03).
24. The `dose_response.py` module docstring says "|d| is capped at 0.4", but `PET_DOSES` includes −0.5 (03).

**Statistical reporting**
25. `bootstrap_intervals` resamples only 2–3 seeds per dose with fixed images. The intervals are too narrow and conditional on the curve crossing 0.5 (01, 03).
26. `decision_accuracy` is unbalanced (01, 03, 05).
27. p-values are printed as `0.000`; they should read "< 0.001". About 231 tests are reported with no correction (01).
28. `switch_point` takes the first 0.5 crossing, which gives artefact values (for example 0.18% or 99.9%) on flat curves such as placebo vs random (01, 03).

## 9. Verified references (URLs the reviewers marked as opened)

**Statistics**
- Bouthillier et al., MLSys 2021: https://arxiv.org/abs/2103.03098
- Agarwal et al., NeurIPS 2021: https://arxiv.org/abs/2108.13264
- Saravanan, Berman & Sober, hierarchical bootstrap: https://pmc.ncbi.nlm.nih.gov/articles/PMC7906290/ (also https://arxiv.org/abs/2007.07797)

**Closest prior work and positioning**
- BAM (Yang & Kim 2019): https://arxiv.org/abs/1907.09701; graded models: https://github.com/google-research-datasets/bam
- SMERF (Kim, Plumb & Talwalkar, ICML 2022): https://proceedings.mlr.press/v162/kim22h.html; https://arxiv.org/abs/2105.06506
- Bastings et al., EMNLP 2022: https://aclanthology.org/2022.emnlp-main.64/ (also https://arxiv.org/abs/2111.07367)
- OSCAR (2025): https://arxiv.org/abs/2512.18888
- Lin, Lee & Celik, KDD 2021: https://arxiv.org/abs/2009.10639
- Miró-Nicolau, Spinnato & Guidotti (2026): https://arxiv.org/abs/2609.30397
- FunnyBirds (ICCV 2023): https://arxiv.org/abs/2308.06248
- Hesse, Schaub-Meyer & Roth, NeurIPS 2024 D&B: https://arxiv.org/abs/2407.11910
- XAI-Units, FAccT 2025: https://arxiv.org/abs/2506.01059
- Lapuschkin et al., Clever Hans (Nat. Commun. 2019): https://arxiv.org/abs/1902.10178
- Suhail et al., shortcut susceptibility: https://arxiv.org/abs/2502.09150
- Hermann et al., ICLR 2024: https://arxiv.org/abs/2310.16228
- DeGrave et al., Nat. Mach. Intell. 2021: https://pmc.ncbi.nlm.nih.gov/articles/PMC7523163/

**Metric confounds and meta-evaluation**
- ROAD (Rong et al., ICML 2022): https://arxiv.org/abs/2202.00449; code: https://github.com/tleemann/road_evaluation
- ROAR (Hooker et al., NeurIPS 2019): https://arxiv.org/abs/1806.10758
- ROAR pitfalls: https://arxiv.org/abs/2304.13836
- Blücher et al., TMLR 2024: https://arxiv.org/abs/2401.06654
- Gomez et al. 2022: https://arxiv.org/abs/2201.13291
- Gevaert et al.: https://arxiv.org/abs/2202.12270
- Quantus (JMLR 2023): https://jmlr.org/papers/v24/22-0142.html; code: https://github.com/understandable-machine-intelligence-lab/Quantus
- MetaQuantus (TMLR 2023): https://arxiv.org/abs/2302.07265
- LATEC (NeurIPS 2024): https://arxiv.org/abs/2409.16756
- M4 (NeurIPS 2023 D&B): https://proceedings.neurips.cc/paper_files/paper/2023/hash/05957c194f4c77ac9d91e1374d2def6b-Abstract-Datasets_and_Benchmarks.html
- Tomsett et al., AAAI 2020: https://arxiv.org/abs/1912.01451
- SaCo (CVPR 2024): https://arxiv.org/abs/2404.01415
- From Flexibility to Manipulation (ECCV 2024 workshop): https://arxiv.org/abs/2412.05592
- Bilodeau et al., PNAS 2024: https://arxiv.org/abs/2212.11870
- Hase, Xie & Bansal, NeurIPS 2021: https://arxiv.org/abs/2106.00786
- Brocki & Chung: https://arxiv.org/abs/2303.01538
- F-Fidelity (ICLR 2025): https://arxiv.org/abs/2410.02970
- Explainer Divergence Scores: https://arxiv.org/abs/2211.07650
- Wilming et al. 2022: https://arxiv.org/abs/2111.07473
- XAI-TRIS: https://arxiv.org/abs/2306.12816
- CLEVR-XAI: https://arxiv.org/abs/2003.07258
- Faithfulness metrics meta-evaluation for LLM chains of thought (2026): https://arxiv.org/abs/2605.25052

**Already cited by the project (re-checked by 02)**
- BackX: https://arxiv.org/abs/2405.02344
- AttributionLab: https://arxiv.org/abs/2310.06514
- Back to the Baseline: https://arxiv.org/abs/2512.11433
- Miró-Nicolau et al., fidelity metrics: https://arxiv.org/abs/2401.10640
- Zhou et al., AAAI 2022: https://arxiv.org/abs/2104.14403
- Sun et al., MICCAI 2023: https://arxiv.org/abs/2307.12344
- Adebayo et al., ICLR 2022: https://arxiv.org/abs/2212.04629
- Binder et al., CVPR 2023: https://openaccess.thecvf.com/content/CVPR2023/html/Binder_Shortcomings_of_Top-Down_Randomization-Based_Sanity_Checks_for_Evaluations_of_Deep_CVPR_2023_paper.html
- Sanity Checks Revisited: https://arxiv.org/abs/2401.06465

**Tools and infrastructure**
- torchvision ResNet-18: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
- timm DeiT-tiny: https://huggingface.co/timm/deit_tiny_patch16_224.fb_in1k
- Group DRO / Waterbirds: https://arxiv.org/abs/1911.08731
- Zenodo upload limits: https://help.zenodo.org/docs/deposit/create-new-upload/

## 10. Unverified items (do not cite or rely on until checked)

- OpenReview submission `r1iCWpYhx4` (power-law onset of shortcut reliance). The page was behind a browser check: https://openreview.net/forum?id=r1iCWpYhx4
- arXiv:2405.02383 (README's "Hedström 2024"). It appeared in search results, but its page was not opened.
- ROAD imputation details beyond the abstract (noisy linear imputation, MoRF/LeRF). Reviewer 03 took these from background knowledge. Reviewer 05 reports that the ROAD repository README describes them, but check the paper's method section before citing.
- The LATEC track (main conference or D&B).
- The no-fee subscription route for Neurocomputing, Machine Learning, Pattern Recognition, Neural Networks, DAMI, PRL, CVIU and the Elsevier backups.
- Journal pages that could not be opened: the Neurocomputing home (sciencedirect.com/journal/neurocomputing), the Elsevier OA-options pages for Pattern Recognition and Pattern Recognition Letters, the Springer "how to publish" pages for Machine Learning and DAMI, the CVIU home, the SN Computer Science home, every SCImago `journalsearch.php` page and every Scopus source page.
- SCImago quartiles of any journal not listed in Section 7.
- Current Kaggle quota terms (taken from the brief and `kaggle/README.md`), the FLOP figure for the CIFAR-stem ResNet-18, ROAD solve times, and whether the ROAD repository has a GPU imputer.
- Every GPU-hour and evaluation-time estimate marked speculative in the source reports.

## 11. Notes on the completion checks

- Six report files are expected under `docs/audit/`: `00_combined_audit.md` (this file) and `01` to `05`, which are all present. The reviewers' scratch scripts and outputs are also under `docs/audit/scratch/`.
- Every reviewer report has a numeric score and at least 5 ranked recommendations with cost estimates (Section 1).
- Not every URL in the reviewer reports was opened. Reviewer 04's journal homepage, OA-options and SCImago `journalsearch.php` links, and reviewer 02's OpenReview link, are explicitly marked UNVERIFIED in their reports. This report carries them only in Section 10.
- The aggregation wrote only this file. It did not stage, commit or push. `git status` was not run in this step, because no shell tool was available.

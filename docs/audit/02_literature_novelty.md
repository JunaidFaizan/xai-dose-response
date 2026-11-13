# Reviewer 2: XAI evaluation literature and novelty

Project: *A Dose-Response Evaluation of Explanation Metrics under Controlled Shortcut Reliance in CNN Image Classifiers*

Lens: closest prior work, novelty, and missing related work. Web access was available. I opened every external page cited below; the one page I could not open is marked UNVERIFIED. Paper summaries are paraphrased from the abstracts or pages I opened. Content was rephrased for compliance with licensing restrictions.

## 1. Summary verdict

**Q1 readiness: 5/10 from this lens.**

The core idea holds up and, as far as my searches (2019–Oct 2026) found, nobody has published it yet. The idea is to use a graded, behaviourally measured reliance on an injected shortcut as a continuous ground-truth axis, and to test whether perturbation faithfulness metrics (not attribution methods) follow it, using fixed oracle maps and form-matched controls. The ingredients already exist separately. BAM and SMERF control what a model relies on, and BAM even trains a graded series of models. Zhou et al., Adebayo et al., Lin et al. and BackX use injected features or backdoors as ground truth. Tomsett, MetaQuantus, Miró-Nicolau et al., Gevaert et al. and LATEC meta-evaluate metrics. ROAD, the ROAR-pitfalls paper and Blücher et al. document confounds from removal shape and fill. The contribution is therefore a new protocol plus a set of careful findings, not a new phenomenon. That is enough for a Q1 journal only if the positioning is exact and the evidence is broad enough to back "explanation metrics" in the title.

Three things hold the score down:

1. Two of the closest prior works (BAM, SMERF) are not cited, and several directly relevant confound papers are missing.
2. Only one metric family (deletion/insertion with 3 fills) is tested. Every comparable benchmark covers more.
3. The paper draft does not yet contain the strongest evidence. `paper/build_paper.py` line 141 asserts `N_MODELS == 174`, so the 18 pre-registered Oxford-IIIT Pet models with real masks are excluded.

All three can be fixed in the 3 weeks available.

## 2. Novelty judgement

| Ingredient | Prior work that already has it | What this project adds |
|---|---|---|
| Inject a shortcut so the ground truth is known | Zhou et al. AAAI 2022; Adebayo et al. ICLR 2022; Lin et al. KDD 2021; BackX; Sun et al. MICCAI 2023; Bastings et al. EMNLP 2022 (text) | The target of evaluation is the **metric**, not the method |
| Graded or controlled model reliance | **BAM** (models `scene1`…`scene10` with the dog added to *i* scene classes, giving graded relative importance); **SMERF** (no-reliance, full-reliance and conditional-reliance settings); OSCAR (detector sensitive to the strength of the shortcut association) | A **continuous signed dose** with reliance *measured behaviourally* from 0–100%, 12 levels × 3 seeds |
| Meta-evaluation of faithfulness metrics | Tomsett et al. AAAI 2020; MetaQuantus (TMLR 2023, no ground truth); Miró-Nicolau et al. 2024 (transparent decision-tree ground truth) and 2026; Gevaert et al.; LATEC; M4 | Ground truth that *changes with the model*, and a responsiveness / switch-point readout |
| Shape, size and fill confounds | ROAD (class information leaks through the removal shape); ROAR pitfalls (bias towards blurry masks, also seen in ROAD); Blücher et al. TMLR 2024 (the occlusion strategy decides rankings); Back to the Baseline; Gomez et al. 2022 | A **size-matched oracle** and placebo/random controls that show the confound *as reliance varies*; real-mask replication on Pet |
| Pre-registration | I found no XAI-evaluation benchmark in this search that pre-registers predictions | `docs/PREDICTIONS_pet.md` with a documented amendment |

**Novelty in one sentence:** The protocol is new: graded, measured reliance used as ground truth to meta-evaluate perturbation metrics, with size and form controls. The individual pieces are not new, and the paper must say plainly that BAM and SMERF already controlled reliance for *methods*.

**Scoop risk (speculation, based on recent arXiv activity):** Miró-Nicolau et al. posted a synthetic ground-truth framework in September 2026, and OSCAR (December 2025) measures sensitivity to shortcut-association strength. Both groups work close to this idea. Submitting soon matters.

## 3. Strengths

- **The controls go further than in comparable papers.** Seven comparisons are run per setting (`responsiveness.csv`, 196 rows, 7 `comparison` values), including placebo and random maps. Most benchmarks I read compare methods against a random baseline. Few compare *oracle maps of different form* against one another.
- **Reliance is measured, not assumed.** `models.csv` (192 models) gives the reliance range per setting: MNIST 0.1–100%, CIFAR-10 ResNet-18 2.7–97%, Pet 15.5–99%. AttributionLab criticises the assumption that a trained model uses the designed features. Measured reliance answers that criticism directly.
- **Pre-registration with honest reporting.** P3 failed for black-fill deletion (ρ 0.75 vs random, 0.73 vs placebo, threshold 0.8), and the failure is reported. This is rare in the field and should be a headline point.
- **The real-mask replication tests its own earlier hypothesis.** On Pet, black/blur random-vs-object preference at the lowest dose is 0.53/0.49. On CIFAR-10 ResNet-18 it was 0.72/0.56 (`responsiveness.csv`). This supports the claim that the centred-Gaussian object map inflated the effect on CIFAR-10.
- **The supporting study agrees with Binder et al. (CVPR 2023)** and adds ground truth: sanity-check ranking vs ground truth gives Kendall τ = −0.20 on Decoy MNIST (`results/analysis/summary.md`).

## 4. Weaknesses, ranked

### Blocker

**B1. The two closest prior works are missing, and the novelty claim is not bounded.**
- BAM trained a series of models with graded relative feature importance. SMERF defines no-, full- and conditional-reliance settings. A reviewer from either line of work will ask why they are not cited.
- The related-work section in `paper/build_paper.py` (lines 727–739) and `docs/PAPER_OUTLINE.md` cite neither.
- Also missing: Bastings et al. (a shortcut-injection protocol), Lin et al. (backdoor-trigger ground truth), Blücher et al. (fill strategy decides pixel-flipping rankings), the ROAR-pitfalls paper (bias towards blurry masks, the closest known mechanism for finding 6/7), Gomez et al. (DAUC/IAUC use only the ranking, and the perturbed images are out of distribution), Gevaert et al. (metric results don't transfer across datasets, which matches finding 1), and Quantus.

### Major

**M1. One metric family is tested, but the title says "Explanation Metrics".** The study covers deletion with 3 fills and insertion with 1 fill (`xai/metrics.py` lines 46–53). LATEC uses 20 metrics, M4 and Quantus implement many more, and MetaQuantus ranks metrics within categories. Reviewers will ask whether the findings hold for other faithfulness estimators: Faithfulness Correlation, exact ROAD with MoRF+LeRF, the MIF/LIF symmetric relevance gain (SRG) from Blücher et al., and region-based variants. Without them, the conclusion "inpainting deletion is the most robust" is a ranking among 4 closely related variants.

**M2. The paper draft excludes the Pet results.** `build_paper.py:141` asserts 174 models, and the README's "Pending: Oxford-IIIT Pet" paragraph is out of date (the pre-registration outcome table is filled in). The Pet study is the best answer to the expected "MNIST toy" objection, because it has real masks, 128×128 images and pre-registration. It needs to be in the paper.

**M3. No comparison with ground-truth-free meta-evaluation.** MetaQuantus is the established way to rank metrics without ground truth. The project has ground truth, so it can test whether MetaQuantus's noise-resilience and adversarial-reactivity scores agree with the dose-response responsiveness. This mirrors the supporting study's "sanity check vs ground truth" result, and the literature does not answer the question yet (MetaQuantus explicitly works around the lack of labels).

**M4. "Dose-response" is used as a label, not as a model.** The analysis reports Spearman ρ and a linear-interpolation switch point. It does not fit a dose-response curve. A logistic fit of preference vs reliance would give an EC50 (the switch point) and a slope (sharpness) with CIs, and would justify the title term. CIs are already computed (`switch_ci_*`, `rho_ci_*`), so most of the work is done.

**M5. Architecture coverage is narrow for natural images.** All natural-image models are ResNet-18s trained from scratch. Hesse et al. (NeurIPS 2024 D&B) report that attribution quality changes with backbone design, and LATEC varies architectures. A reviewer will probably ask about a pretrained or transformer backbone (speculation about reviewer behaviour, but a common request in this literature).

### Minor

- **m1. Citation errors and gaps in the current drafts:**
  - `PAPER_OUTLINE.md` describes arXiv:2401.10640 as "Tabular/transparent models". The abstract says it uses 52,000 synthetic 128×128 **images** with a decision tree.
  - The LATEC entry in `build_paper.py:248` lacks its title. The title is "Navigating the Maze of Explainable AI: A Systematic Approach to Evaluating Methods and Metrics", Klein et al., accepted at NeurIPS 2024. I did not verify the track.
  - MetaQuantus should be cited as TMLR 2023, not arXiv.
  - "Hedström 2024 (arXiv:2405.02383)" and "Sanity Checks Revisited" (arXiv:2401.06465, NeurIPS XAIA 2023) are two versions of related work; cite the one you mean.
- **m2. Findings are not linked to the literature that explains them.**
  - "Random beats object" (CIFAR-10) matches ROAD's shape leakage and the ROAR-pitfalls blurriness bias.
  - SaCo (CVPR 2024) reports that standard metrics struggle to separate methods from random attribution, which supports finding 7 and the Pneumonia result in the supporting study.
  - The "From Flexibility to Manipulation" paper shows that metric hyperparameters can be tuned to manipulate outcomes, which supports reporting several fills.
- **m3. The 50% switch-point ideal is an assumption** (already disclosed in the limitations). Bilodeau et al. (PNAS 2024) argue that a defined end-task plus repeated model evaluations is the proper reference. Citing it frames the 50% convention as one possible end-task definition.
- **m4. The low end of reliance is missing on PneumoniaMNIST and Pet.** Minimum reliance is 12.8% (patch) and 16.4% (texture) on PneumoniaMNIST, and 15.5% on Pet (`models.csv`). "Preference at low reliance" there means about 15%, not 0%. State this next to every low-reliance number.
- **m5. Shortcuts are all synthetic.** Sun et al. (MICCAI 2023) used artificial confounders on chest X-rays that look more realistic, such as text tags. That is an optional extension, not required.

## 5. Recommendations, ranked (cost = person-days and GPU hours)

| # | Recommendation | Addresses | Cost |
|---|---|---|---|
| 1 | **Rewrite related work and the novelty claim.** Add BAM, SMERF, Bastings, Lin, Blücher, ROAR pitfalls, Gomez, Gevaert, Quantus, Hesse 2024, FunnyBirds, XAI-Units, Miró-Nicolau 2026 and OSCAR. Use the table in §2 as the positioning table. Claim novelty only for "graded measured reliance as ground truth for meta-evaluating perturbation metrics, with size/form controls". | B1, m1, m2 | 2 days, 0 GPU h |
| 2 | **Add the Pet study to the paper** and lead with the pre-registration (P1, P2, P4 held; P3 failed for black deletion). Change the `N_MODELS` assertion to 192, update the README "Pending" paragraph, and report Pet minimum reliance (~15%) beside low-reliance numbers. | M2, m4 | 1.5 days, 0 GPU h |
| 3 | **Add three or more faithfulness metrics from other families** to the same oracle-map pipeline: Faithfulness Correlation, exact ROAD with NLI imputation in both MoRF and LeRF, and the SRG = LIF − MIF combination from Blücher et al. Quantus implements several of these, but I have not checked its API against this code. Re-score existing checkpoints only; no retraining. MNIST/PneumoniaMNIST/CIFAR small CNN on the laptop CPU; CIFAR ResNet-18 and Pet on Kaggle. | M1 | 4 days, ~6 GPU h (estimate; Pet inpainting already uses 1600 steps) |
| 4 | **Fit logistic dose-response curves** (preference = logistic(slope·(reliance − EC50))) per metric and setting from `preference.csv`. Report EC50 and slope with seed-level bootstrap CIs, and add a mixed-effects check (random intercept per seed). This makes "dose-response" a real method and gives a one-number metric summary that other studies can reuse. | M4 | 1.5 days, 0 GPU h (CPU, existing CSVs) |
| 5 | **Run MetaQuantus on the same metrics and models** (a subset: MNIST patch + Pet, 3 doses). Report Kendall τ between the MetaQuantus ranking and the dose-response responsiveness ranking. Either result can be published: agreement validates MetaQuantus with ground truth, and disagreement repeats the sanity-check finding at the metric level. | M3 | 3 days, ~3 GPU h |
| 6 | **Add one modern backbone on Pet:** an ImageNet-pretrained ResNet-50 or ViT-S fine-tuned at 6 doses × 2 seeds (12 models), scored with the existing oracles and metrics. Pre-register it as an amendment in the same style as `PREDICTIONS_pet.md`. | M5 | 2.5 days, ~6–8 GPU h (speculative: about 20–30 min per fine-tune plus evaluation on a T4) |
| 7 | **Link each finding to its known mechanism in the discussion:** ROAD and ROAR pitfalls for the form and size bias, Gomez for ranking-only scoring, Gevaert for data dependence, SaCo for random-attribution parity, Bilodeau for the end-task framing of the 50% switch point, and "From Flexibility to Manipulation" for fill and hyperparameter sensitivity. | m2, m3 | 0.5 day, 0 GPU h |
| 8 | *(Optional)* **Add a semi-realistic shortcut on chest X-rays** (a text tag like those in Sun et al.) at 6 doses × 2 seeds. | m5 | 2 days, ~1 GPU h or CPU |

Suggested order within the 3 weeks: 1 → 2 → 4 (all CPU, about 5 days), then 3 and 5 in parallel with Kaggle runs, and 6 only if the GPU quota allows.

## 6. Verified references (all pages opened for this audit)

**Missing from the project, should be added**
- Yang & Kim (2019), *Benchmarking Attribution Methods with Relative Feature Importance* (BAM). https://arxiv.org/abs/1907.09701; graded model series confirmed at https://github.com/google-research-datasets/bam
- Kim, Plumb & Talwalkar (2022), *Sanity Simulations for Saliency Methods* (SMERF), ICML 2022. https://proceedings.mlr.press/v162/kim22h.html (reliance settings read at https://arxiv.org/html/2105.06506v3)
- Bastings et al. (2022), *"Will You Find These Shortcuts?"*, EMNLP 2022. https://aclanthology.org/2022.emnlp-main.64/
- Lin, Lee & Celik (2021), *What Do You See? Evaluation of XAI Interpretability through Neural Backdoors*, KDD 2021. https://arxiv.org/abs/2009.10639
- Blücher et al. (2024), *Decoupling Pixel Flipping and Occlusion Strategy for Consistent XAI Benchmarks*, TMLR. https://arxiv.org/abs/2401.06654
- *On Pitfalls of RemOve-And-Retrain: Data Processing Inequality Perspective* (v5, ICML 2026 MI workshop). https://arxiv.org/abs/2304.13836
- Gomez et al. (2022), *Metrics for saliency map evaluation of deep learning explanation methods*. https://arxiv.org/abs/2201.13291
- Gevaert et al., *Evaluating Feature Attribution Methods in the Image Domain*. https://arxiv.org/abs/2202.12270
- Hedström et al. (2023), *Quantus*, JMLR 24(34). https://jmlr.org/papers/v24/22-0142.html
- Hesse, Schaub-Meyer & Roth (2024), *Benchmarking the Attribution Quality of Vision Models*, NeurIPS 2024 D&B. https://arxiv.org/abs/2407.11910
- Hesse et al. (2023), *FunnyBirds*, ICCV 2023. https://arxiv.org/abs/2308.06248
- *XAI-Units: Benchmarking Explainability Methods with Unit Tests* (2025), FAccT 2025. https://arxiv.org/abs/2506.01059
- Miró-Nicolau, Spinnato & Guidotti (2026), *A Synthetic Ground-Truth Framework for the Evaluation of XAI Methods*. https://arxiv.org/abs/2609.30397
- *OSCAR: Localising Shortcut Learning in Pixel Space…* (2025). https://arxiv.org/abs/2512.18888
- *Explainer Divergence Scores (EDS)* (2022), CIKM 2022 workshop. https://arxiv.org/abs/2211.07650
- Wu et al. (2024), *On the Faithfulness of Vision Transformer Explanations* (SaCo), CVPR 2024. https://arxiv.org/abs/2404.01415
- *From Flexibility to Manipulation: The Slippery Slope of XAI Evaluation*, ECCV 2024 workshop. https://arxiv.org/abs/2412.05592
- Bilodeau et al. (2024), *Impossibility Theorems for Feature Attribution*, PNAS 121(2). https://arxiv.org/abs/2212.11870
- Hase, Xie & Bansal (2021), *The Out-of-Distribution Problem in Explainability…*, NeurIPS 2021. https://arxiv.org/abs/2106.00786
- Brocki & Chung, *Feature Perturbation Augmentation…*, Pattern Recognition Letters. https://arxiv.org/abs/2303.01538
- *F-Fidelity: A Robust Framework for Faithfulness Evaluation of XAI*, ICLR 2025. https://arxiv.org/abs/2410.02970 (take the author list from the PDF)
- Wilming et al. (2022), *Scrutinizing XAI using linear ground-truth data with suppressor variables*, Machine Learning. https://arxiv.org/abs/2111.07473
- Clark et al., *XAI-TRIS*. https://arxiv.org/abs/2306.12816
- Li et al. (2023), *M4*, NeurIPS 2023 D&B. https://proceedings.neurips.cc/paper_files/paper/2023/hash/05957c194f4c77ac9d91e1374d2def6b-Abstract-Datasets_and_Benchmarks.html
- Arras, Osman & Samek, *CLEVR-XAI*, Information Fusion. https://arxiv.org/abs/2003.07258
- Cross-domain parallel (LLM chains of thought, optional): *Faithfulness Metrics Don't Measure Faithfulness: A Meta-Evaluation with Ground Truth* (2026). https://arxiv.org/abs/2605.25052

**Already cited by the project, details re-checked**
- LATEC: Klein et al., *Navigating the Maze of Explainable AI…*, NeurIPS 2024. https://arxiv.org/abs/2409.16756
- MetaQuantus: Hedström et al., TMLR 2023. https://arxiv.org/abs/2302.07265
- BackX. https://arxiv.org/abs/2405.02344
- AttributionLab, NeurIPS 2023 XAIA. https://arxiv.org/abs/2310.06514
- Back to the Baseline (2025). https://arxiv.org/abs/2512.11433
- Miró-Nicolau et al., *A comprehensive study on fidelity metrics for XAI* (synthetic 128×128 images, decision tree). https://arxiv.org/abs/2401.10640
- Zhou et al., AAAI 2022. https://arxiv.org/abs/2104.14403
- Sun et al., MICCAI 2023. https://arxiv.org/abs/2307.12344
- Tomsett et al., AAAI 2020. https://arxiv.org/abs/1912.01451
- Adebayo et al., ICLR 2022. https://arxiv.org/abs/2212.04629
- Rong et al., ROAD, ICML 2022. https://arxiv.org/abs/2202.00449
- Binder et al., CVPR 2023. https://openaccess.thecvf.com/content/CVPR2023/html/Binder_Shortcomings_of_Top-Down_Randomization-Based_Sanity_Checks_for_Evaluations_of_Deep_CVPR_2023_paper.html
- Hedström et al., *Sanity Checks Revisited*. https://arxiv.org/abs/2401.06465 (the README's arXiv:2405.02383 also showed up in my search results, but I did not open its abstract page)

**UNVERIFIED**
- OpenReview submission `r1iCWpYhx4`. A search snippet says it studies how data scale, correlation strength and optimiser change gradient sensitivity to a spurious feature, with a power-law onset of shortcut reliance. The page sits behind a browser check and I could not open it. Do not cite it until someone opens it: https://openreview.net/forum?id=r1iCWpYhx4
- The LATEC track (main conference or D&B), and the SCImago quartiles of any journal named above. I did not check either.

## 7. Evidence from the project used in this report

- `results/dose_response/models.csv`: 192 rows. Per-setting reliance ranges as quoted in §3 and m4.
- `results/dose_response/responsiveness.csv`, shortcut vs full object:
  - MNIST: ρ = 0.94–0.97.
  - PneumoniaMNIST: ρ = 0.25 (texture insertion, p = 0.14) to 0.80.
  - Pet: black 0.85, blur 0.96, inpaint 0.95, insertion 0.93.
- `responsiveness.csv`, random vs full object at low reliance: CIFAR-10 small CNN black 0.82 / blur 0.75; ResNet-18 0.72 / 0.56; Pet 0.53 / 0.49; inpaint and insertion 0.13–0.38 in the same settings.
- `responsiveness.csv`, shortcut vs size-matched object on PneumoniaMNIST texture, black deletion: ρ = 0.07, p = 0.69.
- `xai/metrics.py` lines 46–53: the 4 metric variants are the whole metric set.
- `paper/build_paper.py` line 141 (`assert N_MODELS == 174`); lines 248 (LATEC placeholder) and 1125 (novelty search flagged as unfinished).
- `docs/PREDICTIONS_pet.md`: outcome table (P3 fails for black deletion).

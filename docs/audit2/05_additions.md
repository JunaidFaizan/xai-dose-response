# Audit 2, report 05: Remaining high-leverage additions

Project: **A Dose-Response Evaluation of Explanation Metrics under Controlled Shortcut Reliance in CNN Image Classifiers**

Reviewer lens 5: which further additions raise the chance of acceptance at a Q1 journal the most per day of work and per Kaggle GPU hour. The constraints are about 3 weeks, one student with AI help, a laptop CPU, and Kaggle T4 (about 30 GPU h per week, sessions of at most 12 h). The planned items SD1–SD4 are ranked together with other candidates.

This audit is read-only and was done at commit 80cd09f. Nothing outside `docs/audit2/` was changed. Two read-only scripts were run over existing CSVs and the docx. Their outputs are in `docs/audit2/scratch/05_additions/`:

- `coverage.py` writes `coverage.txt`, with reliance coverage, gaps and training time per setting from `models.csv`. It also writes `paper_text.txt`, the text of `paper/paper_draft.docx` extracted with python-docx.
- `behaviour_switch_support.py` writes `behaviour_switch_support.txt`. It applies the project's own ±10-point identification and bracketing rule to the behavioural switch in `behaviour_summary.csv` and `behaviour_models.csv`.

---

## 1. Score

**Q1 readiness now: 6 / 10** (first audit, this lens: 5 / 10).

The must-do round fixed the problems my first report rated most important among those that need no new training:

- The 50% ideal has been replaced by per-image behavioural labels, and a behavioural switch is reported.
- Effect sizes and a two-level bootstrap are now reported.
- The related work has been rewritten.
- The MNIST and PneumoniaMNIST reliance gaps have been filled under a dated amendment.

The paper now has one clear thesis ("respond but are not calibrated") and it is stated honestly, including the results that go against it. It reads like a careful empirical paper.

The score rises by one point, not two, for two reasons:

1. **Every first-audit weakness that needs new training is still open.** Natural images still use only a synthetic white square, all networks are trained from scratch, the metrics come from one family, and "inspired by ROAD but is not the ROAD procedure" is still the paper's wording (docx §3.5). These are the first objections a Neurocomputing reviewer in XAI will raise. The paper lists them as limitations, which is honest but does not answer them.
2. **One headline number is not supported where the models are thin (Section 3).** The CIFAR-10 behavioural switch is interpolated across a gap of 54–59 points in reliance with no models in it. The paper still calls the assumed 50% "about right" for CIFAR-10.

**Projected score: 7.5 / 10** if recommendations A0, A1, A2, SD2, SD1, SD3 and SD4 (Section 4) are done and the pilots work. It would be **7 / 10** if only one of SD1 or SD4 reaches a usable reliance range. These are judgments, not measurements.

## 2. First-audit weaknesses: status

| First-audit item (docs/audit/05) | Status | Evidence |
|---|---|---|
| B1: the only natural-image shortcut is a synthetic white square | **Not resolved** | `models.csv`: every `cifar10*` and `pet_resnet18` row has `shortcut = patch`. Docx §3.1 says texture is used only on the two 28×28 datasets. |
| B2: the 50% switch is unjustified and decision accuracy depends on it | **Resolved** (CIFAR-10 only partly, see N1) | MD4: `follows_shortcut_causal` labels for 234 × 100 images. Class-stratified AUROC and `observed_switch_rc` are in `behaviour_summary.csv`. Balanced accuracy and FPR are in `responsiveness_corrected.csv`. Docx §5.4. |
| M1: all metrics are from one family; ROAD is only approximated | **Not resolved** | Docx §3.5: "inspired by ROAD … but is not the ROAD procedure". No MoRF/LeRF or noisy linear imputation in `xai/metrics.py`. |
| M2: only from-scratch models, with a CIFAR stem at 128 px | **Not resolved** | `dose_response.py` L60: Pet ResNet-18 "trained from scratch for 30 epochs". SD1 is not done (`06_must_do_resolution.md` §4). |
| M3: the binary preference discards effect size | **Resolved** | `effect_sizes.csv`. Median AUC gaps are in the docx tables. The per-image AUROC uses the continuous gap. A mixed model was not fitted, but the two-level bootstrap covers the clustering. |
| M4: closest related work missing | **Resolved** | Docx §2.1–2.4 cite SMERF, FunnyBirds, BAM, OSCAR, Lapuschkin, Suhail and more, with Table 1 for positioning. |
| m1: reliance gaps where the switch is estimated | **Partly** | MNIST and PneumoniaMNIST are filled (A1). The remaining gaps are CIFAR CNN 42.7 → 96.0, CIFAR ResNet-18 36.5 → 95.8 and Pet 41.1 → 60.1 r_c (`coverage.txt`). The docx (§7) states this. |
| m2: step resolution versus map size (Pet shortcut = 0.9% of pixels, 5% steps) | **Not resolved** | `STEPS = 20` (`dose_response.py` L158). SD5 is not done. |
| m3: 100 images per model | **Partly** | The bootstrap now resamples images. Still 100 images per model. On Pet, 33% of labels are undetermined on average and 33–49% on the models near the switch (`behaviour_switch_support.txt`), so about 50–67 labelled images remain per model. |
| R9: released benchmark and `validate_metric()` | **Not done** | Data availability is still a placeholder (`06_must_do_resolution.md` §4). |

## 3. New problems introduced by the fixes

I checked the abstract's numbers against the CSVs, and they match: 75–92% and 100% FPR (`summary.md` FPR table), Pet switches 1.3–5.4, behavioural switch 54.5, pooled AUROC 0.75–0.90, within-model AUROC 0.43–0.80, CIFAR-10 random-vs-object 0.815 → 81%, and MNIST switches 38.7–39.3 → 0.9–8.6. The problems are in interpretation, not transcription.

**N1. The CIFAR-10 behavioural switch is not identified, but the paper treats it as located (major).** Docx §5.4 reports the switches at 52.9% (ResNet-18) and 55.0% (small CNN) and concludes: "The assumed 50% is therefore about right for Pet, CIFAR-10 and MNIST patch". No CIFAR model lies near these values (`behaviour_switch_support.txt`):

| Setting | Behavioural switch r_c | Nearest model below | Nearest model above | Identified (±10)? | Bracketed? |
|---|---|---|---|---|---|
| CIFAR-10 small CNN | 55.0 | 42.7 (follow rate 0.36), 12.3 pts away | 96.0 (0.97), 41.0 pts away | No | No |
| CIFAR-10 ResNet-18 | 52.9 | 36.5 (0.30), 16.4 pts away | 95.8 (0.97), 42.9 pts away | No | No |
| Pet ResNet-18 | 54.5 | 41.1 (0.43), 13.4 pts away | 60.1 (0.53), 5.7 pts away | Yes | No |
| MNIST patch | 53.9 | 3.1 pts | 7.4 pts | Yes | Yes |

On CIFAR-10 the "switch" is a straight line drawn between the dose −0.1 models and the dose 0 models. Any value from about 40 to 95 fits the data equally well. `behaviour_summary.csv` gives `observed_switch_rc_status = crossed` for every setting, because the identification and bracketing rule used for the metric switches (MD3f) was never applied to the behavioural switch.

- The over-detection result itself survives: the metrics prefer the shortcut map for models at r_c ≤ 43, where the follow rate is ≤ 0.36.
- The location claim and the "about right" sentence do not survive.
- The abstract's Pet figure (54.5%) is identified from above but not bracketed.
- Fix: apply the same status rule to the behavioural switch, change "about right for … CIFAR-10" to "not located on CIFAR-10 (no models between 43% and 96%)", and run SD2.

**N2. TPR on CIFAR-10 rests on a single dose (minor).** `n_above` is 2 (small CNN) and 3 (ResNet-18), all at dose 0 with r_c ≈ 96. The statements "TPR = 1.00 in 23 of 28 cells" and "the metrics rarely miss a shortcut the model uses" (docx §5.2) therefore rest, for CIFAR-10, on a few models at near-total reliance. The tables show n for the FPR but not for the TPR. Add `n_above` to the tables. SD2 also fixes this.

**N3. The undetermined-label rate is defined but not reported (minor).** The paper (§3.6) says undetermined images are excluded, but never gives the rate. It is 0.327 on Pet and PneumoniaMNIST patch and 0.250 on PneumoniaMNIST texture (`behaviour_summary.csv`), and up to 0.49 per Pet model. Readers need it to judge the within-model AUROCs. Add one sentence and a column.

I found no other number in the docx that disagrees with the CSVs, but I only checked the abstract, §5.2 and §5.4. Other reviewers cover the full cross-check.

## 4. Ranked recommendations with costs

### 4.1 How costs were estimated

These figures come from the project's own records:

- **Training time:** `training_minutes` in `models.csv`, averaged per setting in `coverage.txt`. Pet ResNet-18 (CIFAR stem, 30 epochs, T4) takes 25.4 min. CIFAR-10 ResNet-18 takes 10.7 min. The CIFAR-10 small CNN takes 22.5 min (laptop CPU).
- **Evaluation time:** about 8.6 min per Pet model and about 0.6 min per CIFAR ResNet-18 model. This is the first audit's estimate from file timestamps. The behaviour step runs on the CPU in about 42 s per Pet model (`06_must_do_resolution.md` MD4).
- **Pretrained ResNet-18:** the standard 7×7 stem at 128 px needs about 0.59 GFLOPs per image (1.81 GFLOPs at 224 px from the [torchvision model card](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html), scaled by (128/224)²). The CIFAR stem at 128 px used now needs roughly 15 times more. I budget about 6–8 min per pretrained Pet model for 10–15 fine-tuning epochs plus evaluation. Data loading on Kaggle may set the pace, so pilot first.
- **Gains:** expected score gains are my judgment in points on the 10-point scale. They are only for comparing items. They are not additive past about +1.5 because of overlap and execution risk.

"Sessions" means Kaggle GPU sessions of at most 12 h. Days include coding, the dated amendment, analysis, and the paragraph, table and figure for the paper.

### 4.2 Ranking

| Rank | Item | Fixes | Gain | Days | GPU h | Sessions | Gain / day | Gain / GPU h |
|---|---|---|---|---|---|---|---|---|
| 1 | **A0** Text fixes: status rule for the behavioural switch, n_above, undetermined rate (N1–N3) | N1–N3 | +0.10 | 0.25 | 0 | 0 | 0.40 | no GPU |
| 2 | **A1** Pre-register amendment A2 (SD2 doses) and predictions P5–P9 for SD1, SD3 and SD4 before any of their models is trained | turns new runs into confirmatory tests | +0.20 | 0.5 | 0 | 0 | 0.40 | no GPU |
| 3 | **A2** Minimal release: Zenodo DOI for code, per-image CSVs and checkpoints; fill the data-availability statement (SD6-lite) | R9, submission requirement | +0.15 | 0.5 | 0 | 0 | 0.30 | no GPU |
| 4 | **SD2** Fill reliance gaps: CIFAR-10 ResNet-18 and small CNN, and Pet | m1, N1, N2 | +0.30 | 1.5 | ≈4.5 | 1 | 0.20 | 0.067 |
| 5 | **SD1** Pretrained ImageNet ResNet-18 on Pet, patch, 18 models | M2 | +0.40 | 3 | ≈3.5 incl. pilot | 1 + short pilot | 0.13 | 0.11 |
| 6 | **SD3** Exact ROAD (noisy linear imputation, MoRF and LeRF) on the oracle maps | M1, tests the paper's leakage explanation | +0.35 | 3 | ≈3 + laptop CPU | 1 | 0.12 | 0.12 |
| 7 | **SD4** Realistic watermark shortcut on Pet, on the pretrained network from SD1 | B1 | +0.40 | 3.5 | ≈4 incl. pilot | 1 + short pilot | 0.11 | 0.10 |
| 8 | **N4** Metric verdicts on real attribution maps across reliance ("which method does each metric call most faithful?") | practitioner consequence | +0.20 | 2 | ≈2 | 1 (can share) | 0.10 | 0.10 |
| 9 | **SD5** Step-resolution and AUC-tail sensitivity (100 steps, or truncated AUC at shortcut size) | m2, Pet near-ties | +0.10 | 1 | ≈1.5 | 0–1 (can share) | 0.10 | 0.067 |
| 10 | **N5** 500 images per model for the within-model AUROC (existing checkpoints) | m3, headline "within a model only 0.43–0.80" | +0.10 | 1 | ≈3 | 1 | 0.10 | 0.033 |
| 11 | **N6** Competence-decoupled sub-experiment (second factor: share of shortcut-free training images) | MD7 collinearity (Pet −0.96, CIFAR ResNet-18 −0.98) | +0.20 | 2.5 | ≈3 (CIFAR ResNet-18) or 0 (MNIST on CPU) | 1 | 0.08 | 0.067 |
| 12 | **SD6-full** `validate_metric()` entry point and checklist | reuse, citations | +0.10 | 1.5 | 0 | 0 | 0.07 | no GPU |
| – | Not recommended: ViT (outside the title's CNN scope), Waterbirds-style backgrounds, from-scratch texture or watermark on Pet (about 10–11 GPU h each), more seeds or methods | – | – | – | – | – | – | – |

The ranking by gain per GPU hour differs from the ranking by gain per day. SD3, SD1 and SD4 give the most per GPU hour. The GPU budget is not the binding constraint, though: the plan below uses about 17–22 of the 90 h available. The binding constraint is days, so the order follows gain per day, subject to dependencies.

### 4.3 Details

**A0. Text fixes (0.25 day, 0 GPU h).**
- Apply the identification and bracketing rule to `observed_switch_rc` and report its status.
- Rewrite the §5.4 sentence about CIFAR-10 as in N1.
- Add `n_above` to the two FPR tables (shortcut vs full object and vs size-matched object), and the undetermined rate to Table 6.
- This is a change to the analysis code and the docx, so it is for the implementer, not for this audit.

**A1. Pre-registration of the new runs (0.5 day, 0 GPU h).** The Pet pre-registration is one of the strongest parts of the paper (docx §5.3). Write amendment A2 for the SD2 doses, and a new `PREDICTIONS_pretrained.md` with predictions derived from the thesis, committed before the full runs:

- P5: the over-detection rate of black and blur deletion stays above 0.5 on the pretrained Pet models.
- P6: inpainting deletion and insertion have a lower FPR than black and blur deletion.
- P7: the size-matched switch lies within 5 points of the lowest-reliance model.
- P8: exact ROAD lowers the random-vs-object preference at the lowest dose compared with black deletion.
- P9: on the watermark, the preference still rises with r_c (ρ > 0.8) for at least three of four metrics.

Pilot models should be declared as pilots and excluded from the confirmatory analysis, following the Pet dose amendment. A second pre-registered test on new settings is much harder for a reviewer to dismiss than post-hoc extensions, and it costs half a day.

**A2. Zenodo release (0.5 day, 0 GPU h).** Neurocomputing asks for a data-availability statement, and the docx still has a placeholder. Archive `checkpoint/dose/` (the first audit measured it at 1.65 GB), `results/dose_response/` and the code at the submission commit. Leave `validate_metric()` (SD6-full) for later.

**SD2. Fill the reliance gaps (1.5 days, about 4.5 GPU h, 1 session).** This ranks highest of the four planned items because it repairs a claim the paper already makes (N1, N2).
- **Doses:** reliance jumps from about 34–43% at dose −0.1 to 96% at dose 0 on CIFAR-10. Pilot dose −0.05 first, then choose 3–4 doses between −0.08 and −0.01.
- **CIFAR-10 ResNet-18:** 4 doses × 3 seeds = 12 models × about 11.3 min ≈ 2.3 h.
- **CIFAR-10 small CNN:** 4 doses × 2 seeds = 8 models. That is about 3 h on the laptop CPU (0 GPU h), or under 1 h on the T4.
- **Pet:** dose −0.15 × 3 seeds ≈ 3 × 34 min ≈ 1.7 h. This brackets the 54.5% switch, which currently has its nearest model 13.4 points below.
- **Outcome:** CIFAR-10 TPR, balanced accuracy and the behavioural switch then rest on more than one dose.

**SD1. Pretrained ResNet-18 on Pet (3 days, about 3.5 GPU h).**
- **Setup:** use the torchvision ResNet-18 with `IMAGENET1K_V1` weights and ImageNet normalisation (model card above), at 128 px so that the masks, the patch and the pipeline stay the same. Use 6 doses × 3 seeds.
- **What it shows:** this is the setting in which practitioners use these metrics. Changing only the initialisation makes any difference attributable to pretraining.
- **Main risk:** ImageNet features separate cats from dogs from the start, so at d ≤ 0 reliance may stay low. Pilot d = 0 and d = −0.5 (about 0.3 h). If reliance at d = 0 is below about 60%, add positive doses (image swaps, as on MNIST), and record the change in A1 before the full run.
- **Second risk:** the Grad-CAM map shrinks to 4×4 at 128 px. Report this, because Grad-CAM is a method under study, not a metric.
- **Decoupling:** a pretrained backbone may also partly break the reliance–competence collinearity (MD7), because object accuracy is high from the start. Report `confound.csv` for the new setting.

**SD3. Exact ROAD (3 days, about 3 GPU h plus laptop CPU).**
- **Why:** ROAD's own abstract attributes perturbation results to information leaking through the shape of the removed pixels ([Rong et al., ICML 2022](https://arxiv.org/abs/2202.00449)). That is the mechanism this paper uses to explain its form dependence (docx §2.3, §6).
- **How:** the [official repository](https://github.com/tleemann/road_evaluation) provides the Noisy Linear Imputation operator and supports MoRF and LeRF orders. [Quantus](https://github.com/understandable-machine-intelligence-lab/Quantus) also lists ROAD. Add a fill `road` with MoRF and LeRF.
- **Scope:** score the shortcut, object, size-matched and random maps on the existing checkpoints. Use 10 steps, and 50 images for Pet.
- **Compute:** run the 28 px and 32 px models on the laptop (cheap). Run CIFAR ResNet-18 and Pet in one Kaggle session. Imputation is a sparse solve per image and step: 128×128 Pet images give about 16k unknowns per solve, so time one Pet model first. The solve time is unverified.
- **Results:** either ROAD removes the over-detection and the random-map bias, which gives a positive recommendation, or it does not, which shows the problem is not leakage alone. Either result is publishable, and the paper can then name a standard metric.

**SD4. Watermark shortcut on Pet (3.5 days, about 4 GPU h, after SD1).**
- **Motivation:** Lapuschkin et al. found Clever-Hans strategies in standard models ([arXiv:1902.10178](https://arxiv.org/abs/1902.10178)). The abstract page I opened does not describe the PASCAL VOC copyright-tag example, which the first audit cites from the paper body. I did not re-verify it.
- **Design suggestion:** make the watermark **content-coded, not position-coded**. Use two different semi-transparent text or logo marks (opacity about 0.3–0.5, about 24×12 px) in the same corner, one per class. Reliance is then measured by swapping in the other class's mark. This differs from the patch in shape, contrast and coding mechanism. A position-coded watermark would mostly repeat the patch.
- **Oracle and controls:** the oracle shortcut map is the glyph stroke mask. The placebo is the other class's mark mask.
- **Network:** run it on the pretrained network from SD1. That gives a clean comparison: SD1 against the existing models isolates pretraining, and SD4 against SD1 isolates shortcut form. From scratch, it would cost about 10–11 GPU h and mix both changes.
- **Pilot:** 2 models (about 0.5 h), to check that the watermark is learned at d = 0.

**N4. Metric verdicts on real method maps (2 days, about 2 GPU h).** `scores_methods.csv` stores only `shortcut_mass`, so the metrics were never applied to the five attribution maps. Scoring them gives, per model, the method each metric rates most faithful. That can be compared with the method whose shortcut mass actually tracks reliance. It shows the practical cost of over-detection: on a low-reliance model, an over-detecting metric rewards a method that highlights the shortcut. Compute it on existing checkpoints. For SD1 and SD4 it adds only minutes if done during their runs.

**SD5. Step resolution (1 day, about 1.5 GPU h).** Only AUCs are stored (`scores_oracle.csv`), so this needs re-scoring. Use a subset (2 seeds × all doses per setting) with 100 steps, or log-spaced steps starting at 0.5% of pixels. It tests whether the Pet low-reliance near-ties (median gap −0.060 to 0.000) come from the 5% step size, since the 12×12 patch is 0.9% of the image.

**N5. More images per model (1 day, about 3 GPU h).** Re-score the oracle maps and labels on 500 images for the CIFAR ResNet-18 and Pet models, and on the CPU for the small models. The within-model AUROC is now a headline result ("within a model only 0.43–0.80"). Its CIs are wide (for example CIFAR CNN black deletion 0.427 [0.339, 0.555]), and on Pet only about 50–67 images per model are labelled.

**N6. Competence-decoupled design (2.5 days, 3 GPU h or CPU).** `confound.csv` shows reliance and object accuracy collinear at ≤ −0.95 in 4 of 7 settings, including Pet (−0.963). Add a second factor, a share q ∈ {0, 0.3} of training images without any shortcut. This keeps object competence high at a given reliance, so the response to reliance can be separated from the response to degraded object features. It is useful but needs a new data option. It is lower than SD1, which may partly decouple the two at no extra design cost.

## 5. Execution order within about 30 GPU h per week and 3 weeks

| Week | Laptop / CPU work | Kaggle GPU | GPU h |
|---|---|---|---|
| 1 | A0 (day 1); A1 written and committed before any run (day 1–2); code SD1 and SD2; CIFAR small-CNN SD2 models on the laptop; SD3 coding and ROAD on the 28/32 px models (laptop, overnight) | Session 1: SD2 pilot (dose −0.05), then CIFAR ResNet-18 (12) + Pet −0.15 (3), ≈4.5 h. Session 2 (short): SD1 pilot, 2 models, ≈0.5 h | ≈5 |
| 2 | Finalise SD1 doses in A1 if the pilot needs it; code SD4 (glyph rendering, masks, swap); analyse SD2 and SD1 | Session 3: SD1 full, 18 models, ≈3 h. Session 4: SD3 ROAD for CIFAR ResNet-18 and Pet, ≈3 h. Session 5 (short): SD4 pilot, ≈0.5 h | ≈6.5 |
| 3 | A2 Zenodo release; re-run `dose_response.py analyze`; rebuild the paper (new §5 subsections, updated abstract and limitations), ≈2 days | Session 6: SD4 full, 18 models, ≈3.5 h. Optional: N4 and SD5 in the same or a separate session, ≈3.5 h | ≈3.5–7 |

- **Totals:** about 15–19 GPU h out of about 90 available. That leaves room to rerun any failed session, or to add N5 (3 h).
- **Days:** A0 + A1 + A2 + SD2 + SD1 + SD3 + SD4 ≈ 12.25 days, plus about 2 days for rewriting, ≈ 14–15 working days. This fits 3 weeks with about one day of slack.
- **What to drop if time slips:** first N4 and SD5, then SD6-full. Don't drop A1: if SD1 and SD4 are run without pre-registered predictions, they lose most of their value as confirmatory tests.
- **Dependencies:** SD4 depends on the SD1 pilot. If the SD1 pilot cannot reach high reliance even with positive doses, run SD4 from scratch instead (about 10–11 GPU h, one session), and report SD1 as a negative pilot.

## 6. Sources opened for this report

- Rong et al., ROAD, ICML 2022 (abstract page): https://arxiv.org/abs/2202.00449
- ROAD official repository (Noisy Linear Imputation, MoRF/LeRF): https://github.com/tleemann/road_evaluation
- Quantus repository (metric list including ROAD, IROF, Faithfulness Correlation): https://github.com/understandable-machine-intelligence-lab/Quantus
- torchvision ResNet-18 model card (IMAGENET1K_V1, 1.81 GFLOPs, 44.7 MB): https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
- Lapuschkin et al., Unmasking Clever Hans predictors (arXiv abstract page): https://arxiv.org/abs/1902.10178

**Not verified:**
- The Neurocomputing open-access and subscription page redirected to an Elsevier login, so I could not open it. The no-fee route remains for the user to confirm, as `06_must_do_resolution.md` §4 already says.
- The Kaggle quota terms. The documentation page did not render, so the 30 h per week and 12 h per session figures come from the brief.
- The ROAD solve times, the pretrained fine-tuning time per model, and the doses needed for mid-range reliance. Pilots are planned for each.

All statements about external sources are paraphrased from the pages listed above, to comply with licensing restrictions.

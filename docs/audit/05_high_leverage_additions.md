# Audit report 05: High-leverage additions within constraints

Project: **A Dose-Response Evaluation of Explanation Metrics under Controlled Shortcut Reliance in CNN Image Classifiers**

Reviewer lens 5: which additional experiments and analyses raise the chance of acceptance at a Q1 journal the most per day of work and per GPU hour, given about 3 weeks, one student with AI help, a laptop CPU and Kaggle (about 30 GPU h per week, sessions of at most 12 h).

Read-only audit. Nothing outside `docs/audit/` was changed. The only analysis run was `docs/audit/scratch/05/auc_gap.py`, which reads `results/dose_response/scores_oracle.csv` and `models.csv`.

---

## 1. Summary verdict

**Q1 readiness: 5 / 10.**

The core design is sound and unusual: 192 trained models with measured reliance, fixed oracle maps, placebo and random controls, seed bootstrap intervals, and a pre-registered Pet study whose failed prediction (P3 for black deletion) is reported honestly. That puts the work above a typical workshop paper.

What holds it back at a Q1 journal is that most of the evidence comes from settings a reviewer can dismiss as toy, and the main evaluation choices are narrow:

- Only one shortcut type (a white square) reaches natural images. The diffuse texture exists only on 28×28 MNIST and PneumoniaMNIST.
- All models are trained from scratch. No pretrained network, the setting in which practitioners actually use these metrics, is tested.
- The four metrics are all ordered-removal curves from one family, and the inpainting variant is "ROAD-inspired", not ROAD (README, Limitations).
- The main outcome is a binary per-image preference, aggregated per model. The 50% switch-point ideal is an assumption the project itself flags.

None of these needs new ideas to fix. Most need re-analysis of existing CSVs or re-evaluation of existing checkpoints, and the two new training runs I recommend fit in about 10 Kaggle GPU hours in total. With the top 6 items below done, I would score the project 7/10. That estimate is my judgment, not a measurement.

## 2. Strengths (what the additions should build on)

1. **Reliance is measured, not assumed, and spans the range.** Pet reaches 17–99% reliance, ResNet-18 on CIFAR-10 3–97%, MNIST 0–100% (`summary.md`, reliance table).
2. **Pre-registration with an honest amendment and a reported failure.** `docs/PREDICTIONS_pet.md` records the dose change after the first model, before any metric results were seen, and P3 failing for black deletion (ρ = 0.75 vs random, 0.73 vs placebo).
3. **Form-matched controls.** The placebo and random maps separate "the metric responds to the model" from "the metric prefers a map of a certain shape". On CIFAR-10 ResNet-18 at 3% reliance, black deletion prefers a random ordering over the object map 72% of the time (`responsiveness.csv`, "random vs full object", `preference_low_reliance` 0.723). That finding is publishable.
4. **Real object masks on Pet** removed the centred-Gaussian caveat. The black/blur bias towards random maps falls from 0.56–0.82 on CIFAR-10 to 0.49–0.53 on Pet (`PREDICTIONS_pet.md`, exploratory notes).
5. **Infrastructure is in place.** The Kaggle bundle and notebook pipeline (`kaggle/README.md`) already ran 36 GPU models without problems, and evaluation resumes per model. So each extra run below costs mostly GPU time, not engineering.

## 3. Weaknesses, ranked by severity

### Blockers (likely to cause rejection at a Q1 journal if not addressed)

**B1. The only shortcut on natural images is a synthetic white square.** CIFAR-10 and Pet both use a corner or border patch (`dose_response.py`, Pet design notes; `models.csv` shows `shortcut = patch` for every `cifar10*` and `pet_resnet18` row). The headline claim "insertion fails on a diffuse shortcut" (README finding 4) rests only on 28×28 MNIST. A reviewer will ask whether any finding transfers to a realistic shortcut such as the copyright watermark found in PASCAL VOC models by [Lapuschkin et al. (Nature Communications 2019; arXiv:1902.10178)](https://arxiv.org/abs/1902.10178).

**B2. The 50% switch point is not justified, and the decision accuracy depends on it.** `responsiveness.csv` shows `decision_accuracy` of at most 0.25 for 12 of the 16 CIFAR-10 "shortcut vs full object/size-matched" rows (exactly 0.167 for 9 of them). That is because 5 of the 6 doses sit below 50% reliance (`summary.md`: 4–44% and then 96%), not necessarily because the metrics are wrong. Without a per-image ground truth, the reader cannot tell a "too early" switch from a wrong ideal.

### Major

**M1. All metrics come from one family, and the ROAD variant is approximate.** `xai/metrics.py` implements deletion with three fills and insertion with one fill, all as ordered-removal AUCs over 20 steps. The inpaint fill is iterative 4-neighbour averaging plus noise (lines 77–93), with no LeRF order and no comparison to ROAD's noisy linear imputation. ROAD is the obvious baseline here: its abstract says the shape of the removed pixels leaks information ([Rong et al., ICML 2022](https://arxiv.org/abs/2202.00449)), and that is the mechanism the CIFAR-10 finding is attributed to.

**M2. Only from-scratch models, and a nonstandard stem at 128 px.** `xai/models.py` `ResNet18Cifar` uses a 3×3 stride-1 first convolution and no max-pool, which is designed for 32×32 inputs and is also used for Pet at 128×128. A reviewer can argue that neither the network nor its training resembles practice, where an ImageNet-pretrained backbone is fine-tuned.

**M3. The binary preference discards effect size.** My scratch analysis (`docs/audit/scratch/05/auc_gap.py`) shows how small the AUC differences behind many preferences are. Values are medians of the per-image AUC gap (positive means the shortcut map scores better), over the 20% of images from the lowest-reliance and the highest-reliance models:

| Setting | Metric | Shortcut vs object, low reliance | Shortcut vs object, high reliance | Shortcut vs random, high reliance |
|---|---|---|---|---|
| Pet ResNet-18 | Deletion (black) | −0.009 | 0.226 | 0.106 |
| Pet ResNet-18 | Insertion (blur) | −0.039 | 0.211 | 0.105 |
| CIFAR-10 ResNet-18 | Deletion (black) | +0.060 | 0.886 | 0.329 |
| MNIST patch | Deletion (inpaint) | −0.676 | 0.829 | 0.800 |
| MNIST texture | Insertion (blur) | −0.165 | 0.017 | −0.025 |

On Pet, at low reliance, the real object mask beats a 144-pixel corner patch by only about 0.01–0.04 AUC. A preference of 0.35–0.46 there reflects a near-tie, not a confident verdict. The paper should report effect sizes with intervals next to preferences.

**M4. Some closely related work is missing from the related-work section.** SMERF ([Kim et al., ICML 2022](https://arxiv.org/abs/2105.06506)) builds a synthetic benchmark that controls the complexity of the model's reasoning, to give saliency methods a ground truth. FunnyBirds ([Hesse et al., ICCV 2023](https://arxiv.org/abs/2308.06248)) estimates ground-truth part importances by intervention. [Suhail et al. (arXiv:2502.09150)](https://arxiv.org/abs/2502.09150) inject positional and intensity shortcuts to compare CNN, MLP and ViT susceptibility. None of these appears in `README.md`, `SYNOPSIS.md`, `docs/PAPER_OUTLINE.md` or `paper/build_paper.py` (grep for SMERF, FunnyBirds, Clever Hans, Lapuschkin: no matches). They don't seem to vary reliance in steps to test metrics, judging from their abstracts (I didn't read the full papers), but a reviewer who knows them will expect a comparison.

### Minor

**m1. Gaps in reliance coverage.** CIFAR-10 has no model between 44% and 96% reliance, ResNet-18 none between 36% and 97%, and Pet none between 47% and 82% (`summary.md`). These are exactly the ranges where the switch point is estimated.

**m2. Step resolution versus map size.** `dose_response.py` uses `STEPS = 20`, so each step removes 5% of pixels. The Pet shortcut is 144 of 16,384 pixels (0.9%), so the shortcut map differs from a random ordering only in the first step. The remaining 95% of the curve is random tie order. This doesn't invalidate the results, but it inflates AUC noise and should be tested.

**m3. 100 images per model** (`scores_oracle.csv`: exactly 100 per model, 192 models). The per-model preference therefore has a standard error of about 0.05. Seed bootstrap intervals partly cover this, but image-level uncertainty isn't modelled.

**m4. Title constraint.** The fixed title says "CNN Image Classifiers" (`.kiro/steering/project-rules.md`), so a ViT experiment would sit outside the declared scope unless the user changes the title.

## 4. How GPU hours were estimated

These are based on the project's own records.

- **Training:** `models.csv` `training_minutes`: Pet ResNet-18 mean 25.4 min per model (30 epochs, T4); CIFAR-10 ResNet-18 mean 10.75 min (15 epochs); MNIST CNN 1.7 min and PneumoniaMNIST CNN 0.9 min (laptop CPU).
- **Evaluation (derived from file timestamps in `results/dose_response/per_model/`):** for Pet, the oracle and methods files of consecutive models are written about 6.0–6.3 min apart, and the placebo and random files about 1.2 min apart each. That makes evaluation about 8.6 min per Pet model. For CIFAR-10 ResNet-18 it is about 0.4 + 0.15 min per model.
- **Totals:** a full 18-model Pet run is about 18 × (25.4 + 8.6) ≈ **10.2 GPU h**. This matches the logs: seeds 42 and 1 (12 models) ran from about 10:40 to 17:21 in one session. A CIFAR-10 ResNet-18 run of 18 models takes about **3.4 GPU h**.
- **Pretrained ResNet-18 (standard stem) at 128 px, speculative:** about 0.59 GFLOPs per image. That figure is 1.81 GFLOPs at 224 px from the [torchvision model card](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html), scaled by (128/224)². The CIFAR-stem ResNet-18 at 128 px needs about 16 × 0.56 ≈ 9 GFLOPs, where 0.56 GFLOPs at 32 px is the usual figure for that stem; I didn't open a source for it. So one epoch should take roughly 1/10 to 1/15 of the from-scratch time. I budget about 6 min per model for training plus evaluation, so **about 2 h** for 18 models, plus a 1 h margin. Run one model first to check, as was done for Pet.

Kaggle CPU-only notebooks don't use the GPU quota. CPU-bound work, such as exact ROAD imputation, can run there in parallel with the laptop. I haven't checked current Kaggle quota terms; the 30 h figure comes from the brief and `kaggle/README.md`.

## 5. Ranked recommendations

Ranking is by (expected gain in acceptance chance) / (days of student work). Gains are my judgment, in percentage points of acceptance probability. They are speculative and only meant to compare items. GPU hours use the estimates in section 4. Days assume AI-assisted coding and include writing the paragraph and figure for the paper.

| Rank | Addition | Fixes | Expected gain | Days | GPU h | Gain per day |
|---|---|---|---|---|---|---|
| 1 | Related-work additions (SMERF, FunnyBirds, Clever Hans, shortcut susceptibility) | M4 | +2 pp | 0.5 | 0 | 4.0 |
| 2 | Per-image behavioural ground truth, metric AUROC | B2 | +6 pp | 2 | ~0.3 (or CPU) | 3.0 |
| 3 | Effect sizes and a mixed-effects model on existing CSVs | M3, m3 | +4 pp | 1.5 | 0 | 2.7 |
| 4 | Pretrained ImageNet ResNet-18 on Pet, patch, 18 models | M2 | +6 pp | 3 | ~3 | 2.0 |
| 5 | Exact ROAD (noisy linear imputation, MoRF and LeRF) on oracle maps | M1 | +6 pp | 3 | ~1 (+ CPU) | 2.0 |
| 6 | Realistic watermark/text shortcut on Pet, pretrained ResNet-18, 18 models | B1 | +7 pp | 3.5 | ~3–4 | 2.0 |
| 7 | Fill the reliance gaps (CIFAR-10 ResNet-18 and Pet) | m1, B2 | +2 pp | 1 | ~3 | 2.0 |
| 8 | Step-resolution sensitivity check | m2 | +2 pp | 1 | ~1 | 2.0 |
| 9 | Released benchmark with a "metric validation checklist" script | reproducibility | +4 pp | 2.5 | 0 | 1.6 |
| 10 | Texture shortcut on Pet | B1 (partly) | +4 pp | 2.5 | ~3 (pretrained) or ~11 (from scratch) | 1.6 |
| 11 | Extra Quantus faithfulness metrics on oracle maps | M1 | +3 pp | 2.5 | ~2 | 1.2 |
| – | Not recommended: ViT-tiny, Waterbirds-style graded background, more methods, more seeds | – | – | – | – | – |

Details for each:

### R1. Close the related-work gaps (0.5 day, 0 GPU h)

Add SMERF ([arXiv:2105.06506](https://arxiv.org/abs/2105.06506)), FunnyBirds ([arXiv:2308.06248](https://arxiv.org/abs/2308.06248)), Lapuschkin et al. ([arXiv:1902.10178](https://arxiv.org/abs/1902.10178)) as motivation for real shortcuts, and Suhail et al. ([arXiv:2502.09150](https://arxiv.org/abs/2502.09150)), each with one sentence on how this work differs. SMERF and FunnyBirds evaluate *methods* against a designed ground truth; this project evaluates *metrics* across graded, measured reliance. Read each PDF before citing, as `PAPER_OUTLINE.md` already requires. The gain is avoiding a "missed closest work" verdict, which costs almost nothing.

### R2. Per-image behavioural ground truth and metric AUROC (2 days, about 0.3 GPU h or about 1 h on CPU)

This is the most valuable analysis. For each of the 100 scored aligned images per model (`scores_oracle.csv` has `index`), also run the model on the *same* image with the shortcut moved to a wrong class's position. That gives a per-image label: the model follows the shortcut on this image, or it doesn't. It is fill-free, so it is not circular with the metrics under test. Then report, per metric and setting, the AUROC of the per-image AUC gap (shortcut map minus object map) for predicting that label, plus the switch point implied by the per-image labels.

This replaces the assumed 50% ideal with an observed one (B2) and gives an effect-size-aware headline number per metric. The cost is 2 forward passes per image and no explanations: under a minute per small model on CPU, and a few minutes for all 36 ResNet models on Kaggle. Whether the image-level label is informative depends on how many images fall near the decision boundary. Check that first on 6 MNIST models (speculative).

### R3. Effect sizes and a mixed-effects model (1.5 days, 0 GPU h)

The existing CSVs are enough. Report the median per-image AUC gap with a seed bootstrap interval next to every preference value (table above, M3). Fit a logistic mixed model, preference ~ reliance, with random intercepts for model and seed (for example `statsmodels` `BinomialBayesMixedGLM`, or `glmer` in R). Its slope gives one responsiveness number per metric that uses all 100 × N images and accounts for clustering. This answers the predictable statistical reviewer comment about Spearman ρ over only 12–36 models, and makes "Pet low-reliance preferences ≈ 0.5 are near-ties" explicit.

### R4. Pretrained ImageNet ResNet-18 on Pet (3 days, about 3 GPU h)

Use the standard torchvision ResNet-18 with `ResNet18_Weights.IMAGENET1K_V1` (a 44.7 MB download, per the [torchvision model card](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)), its standard 7×7 stem and ImageNet normalisation. Fine-tune about 10 epochs at 128 px, with the same 12×12 patch, 6 doses × 3 seeds.

- It addresses M2 directly, and with the same data and shortcut, any change is attributable to pretraining.
- **Risks:**
  - ImageNet contains cat and dog breeds, so the object features are strong from the start. High reliance may need positive doses (image swaps), so pilot one model at d = 0 and one at d = −0.5 first.
  - Grad-CAM drops to a 4×4 map at 128 px.
  - The Kaggle notebook needs Internet on to download the weights (it is already on for Captum).
- The code change is small: a new entry in `ARCHITECTURES` and `BASES`. Captum methods are unaffected.

### R5. Exact ROAD on the oracle maps (3 days, about 1 GPU h plus CPU)

Use the official noisy linear imputation from the [ROAD repository](https://github.com/tleemann/road_evaluation), whose README describes the imputation operator and the two orders, MoRF and LeRF. Alternatively use the ROAD metric in [Quantus](https://github.com/understandable-machine-intelligence-lab/Quantus) ([JMLR 2023](https://jmlr.org/papers/v24/22-0142.html)). Score the shortcut, object, size-matched, placebo and random maps under MoRF and LeRF on all 192 models.

Two results would come out of it:
- whether the project's iterative inpainting agrees with real ROAD, via the per-image AUC correlation, which turns "ROAD-inspired" into a measured statement;
- whether ROAD removes the black/blur bias against compact object maps, which the paper currently explains by ROAD's leakage argument without testing it.

The cost is dominated by the sparse linear solves on 128×128 Pet images, which run on CPU (speculative: about 0.05–0.2 s per solve). Use 50 images and 10 steps for Pet, on a Kaggle CPU notebook. The 28×28 and 32×32 models are cheap on the laptop.

### R6. A realistic watermark/text shortcut on Pet (3.5 days, about 3–4 GPU h on the pretrained network)

Add a third shortcut type: a semi-transparent class-coded text watermark (for example "©" plus a short string at about 30–40% opacity, bottom-left for cats and bottom-right for dogs). It is motivated by the copyright tag that PASCAL VOC models exploited, as reported in [Lapuschkin et al.](https://arxiv.org/abs/1902.10178). The oracle shortcut map is the rendered glyph mask. Reliance is measured as now, with the watermark moved to the wrong side.

- This is the most direct answer to B1. It changes shape (thin strokes, not a square), contrast (semi-transparent) and realism at once.
- Run it on the pretrained ResNet-18 from R4. That roughly halves the GPU cost relative to from-scratch (about 11 GPU h) and matches practice. If R4 isn't done, budget about 10–11 GPU h (two Kaggle sessions).
- **Risk:** dose calibration. Pilot two models (about 0.3 GPU h).

### R7. Fill the reliance gaps (1 day, about 3 GPU h)

Add CIFAR-10 ResNet-18 doses −0.05 and −0.02 (3 seeds each, 6 models × 11.3 min ≈ 1.1 GPU h) and Pet dose −0.15 (3 seeds, 3 × 34 min ≈ 1.7 GPU h). This puts models in the 45–95% reliance range, so `decision_accuracy` and the switch-point intervals on natural images stop resting on a single high-reliance dose. Pre-register the added doses in a short amendment, following the practice in `PREDICTIONS_pet.md`.

### R8. Step-resolution sensitivity (1 day, about 1 GPU h)

Re-score the oracle maps on a subset (for example 2 seeds × 6 doses per base) with 100 steps, or with log-spaced steps that start at 0.5% of pixels. Also report a "first-k%" AUC truncated at the shortcut size. If the preferences barely change, add one sentence and a supplementary table. If they change, that is a finding about how AUC resolution interacts with explanation size, which ties into the size confound (README finding 3).

### R9. Released benchmark with a "metric validation checklist" (2.5 days, 0 GPU h)

Package the 192 checkpoints (1.65 GB, under `checkpoint/dose/`), the per-image CSVs and the oracle-map generator on [Zenodo](https://help.zenodo.org/docs/deposit/create-new-upload/), which allows up to 50 GB per record and issues a DOI. Add one entry point: `validate_metric(metric_fn)`. It scores any user metric on the oracle maps of all models and prints the paper's diagnostics: responsiveness ρ, form-matched start near 0.5, random-vs-object bias at the lowest reliance, and switch point with interval.

Turn the findings into a 5-item checklist in the paper ("before using a faithfulness metric, check that…"). This turns the paper from "we found these problems" into "here is a tool to test your metric", which journals tend to value and which helps citations. It is pure engineering and writing.

### R10. Texture shortcut on Pet (2.5 days, about 3 GPU h pretrained or about 11 GPU h from scratch)

This extends the diffuse-shortcut findings (insertion failure, methods moving away from the texture) to natural images. Its value overlaps with R6, so do it only if R4 and R6 finish early. The 10% density and +0.3 brightness texture may be too weak or too strong on colour photos, so pilot 2 models first.

### R11. Extra Quantus faithfulness metrics (2.5 days, about 2 GPU h)

From [Quantus](https://github.com/understandable-machine-intelligence-lab/Quantus), add Faithfulness Correlation (random-subset removal, a different principle from ordered removal), IROF (superpixel removal) and Region Perturbation, on the oracle maps only. This answers "only one metric family" (M1). The gain is lower than R5, because binary oracle maps create many ties that correlation-based metrics handle poorly (speculative), so the results may need careful caveats.

### Not recommended within 3 weeks

- **ViT-tiny.** The fixed title restricts the scope to CNNs (m4). The [timm DeiT-tiny weights](https://huggingface.co/timm/deit_tiny_patch16_224.fb_in1k) expect 224×224 input, so the Pet pipeline and the scaled blur, inpainting and occlusion settings would have to be redone. Grad-CAM is not well defined for ViTs. Cost about 4–5 days and about 15+ GPU h (speculative), with a high risk of not finishing.
- **Waterbirds-style graded background correlation** ([Sagawa et al., arXiv:1911.08731](https://arxiv.org/abs/1911.08731)). It is scientifically attractive: the background is a large, realistic shortcut with masks. But it needs CUB and Places data preparation and new oracle definitions. That is more than a week of work.
- **More attribution methods or more seeds.** Methods are secondary to the paper's claim, and the seed bootstrap intervals are already reported. Both have low gain per GPU hour.

## 6. Suggested 3-week plan

The GPU runs go in the background while CPU and analysis work proceeds.

| Week | Laptop / CPU work | Kaggle GPU (h) |
|---|---|---|
| 1 | R1, R3, R2 (MNIST and Pneumonia on CPU); code R4 and run its 1-model pilot | R4 pilot 0.3, full R4 run ~3 |
| 2 | R5 (28 px models on the laptop, Pet on a Kaggle CPU notebook); code R6 and run its pilot; R2 for ResNet models | R6 pilot 0.3, full R6 run ~4, R2 ResNet forward passes ~0.3, R7 ~3 |
| 3 | R9 release and checklist; R8 if time allows; rewrite the results, discussion and limitations sections | R8 ~1 |

The total is about 12 GPU h, well inside the quota. That leaves room to rerun one failed session, or for R10 from scratch (about 11 h).

## 7. Sources opened for this report

- Rong et al., ROAD, ICML 2022: https://arxiv.org/abs/2202.00449
- ROAD official code: https://github.com/tleemann/road_evaluation
- Quantus repository (metric list including ROAD, Faithfulness Correlation, IROF, Region Perturbation): https://github.com/understandable-machine-intelligence-lab/Quantus
- Quantus, JMLR 24(34), 2023: https://jmlr.org/papers/v24/22-0142.html
- Kim et al., SMERF, ICML 2022: https://arxiv.org/abs/2105.06506
- Hesse et al., FunnyBirds, ICCV 2023: https://arxiv.org/abs/2308.06248
- Lapuschkin et al., Unmasking Clever Hans predictors (Nature Communications 2019, arXiv version opened): https://arxiv.org/abs/1902.10178
- Suhail et al., Shortcut Learning Susceptibility in Vision Classifiers: https://arxiv.org/abs/2502.09150
- Sagawa et al., Group DRO / Waterbirds: https://arxiv.org/abs/1911.08731
- torchvision ResNet-18 weights and GFLOPs: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
- timm DeiT-tiny model card: https://huggingface.co/timm/deit_tiny_patch16_224.fb_in1k
- Zenodo upload limits (100 files, 50 GB): https://help.zenodo.org/docs/deposit/create-new-upload/

Unverified: current Kaggle quota terms (taken from the brief and `kaggle/README.md`); the CIFAR-stem ResNet-18 FLOP figure; ROAD solve times; whether the ROAD repository includes a GPU imputer.

Paraphrased content: all statements about external sources were rephrased from the pages listed above, for compliance with licensing restrictions.

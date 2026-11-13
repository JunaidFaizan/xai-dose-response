# A Dose-Response Evaluation of Explanation Metrics under Controlled Shortcut Reliance in CNN Image Classifiers

M.Tech final year project in Explainable AI. It extends my B.Tech project, a handwritten digit recognition app built with PyTorch and Flask.

## The question

Explanation methods (heatmaps) are judged with evaluation metrics such as deletion and insertion curves, but those metrics are rarely checked themselves, because the true explanation is normally unknown.

This project controls the truth. It trains models on data with an injected shortcut, a local patch or a diffuse texture that gives away the label, and **varies in steps how much the model relies on it**: from ignoring it completely to depending on it almost entirely. Reliance is measured, not assumed. Then it asks whether the metrics notice: when the model's reasoning shifts from the object to the shortcut, does a metric's preference shift from an object explanation to a shortcut explanation?

To test the metrics independently of any explanation method, every model is scored with fixed "oracle" explanations (a shortcut-only map and an object-only map, plus a size-matched object map as a control). The five explanation methods are also run on every model.

In our literature searches (audit covering 2019 to October 2026, re-checked on 2026-11-13) we did not find a study that varies a model's measured reliance on a shortcut in steps to test the evaluation metrics themselves. The pieces exist separately: BAM and SMERF control what a model relies on to test *attribution methods*, and ROAD, the ROAR-pitfalls paper and Blücher et al. describe the removal-shape and fill confounds. See [Related work](#related-work) and `docs/audit/02_literature_novelty.md`.

**Thesis.** The metrics respond to reliance but are not calibrated to it, and their verdicts depend on the size and form of the explanations being compared.

**Paper draft:** `paper/paper_draft.docx`, built from the result files by `paper/build_paper.py`. Provisional target journal: Neurocomputing (Elsevier, Q1). The free subscription (no-fee) route is **not verified** and must be checked in a normal browser before submission.

## Main findings: dose-response study

234 models (192 original + 42 from the pre-registered amendment A1, `docs/PREDICTIONS_amendments.md`): Oxford-IIIT Pet with real segmentation masks (ResNet-18, 18 models), CIFAR-10 with the patch shortcut (ResNet-18, 18; small CNN, 12), MNIST (102) and PneumoniaMNIST (84) with patch and texture shortcuts. Each model is scored on 100 test images. Reliance is chance-corrected: r_c = 100 (R − f)/(100 − f), with f = (100 − accuracy without shortcut)/(C − 1).

Primary results are in `results/dose_response/summary.md` ("Corrected analysis (primary)") and the CSVs listed under [Outputs](#outputs). The numbers below come from them.

1. **The metrics respond to reliance.** The preference for the shortcut map over the object map rises with r_c in 27 of 28 metric–setting cells after Benjamini–Hochberg (Spearman ρ = 0.27–0.99; natural images 0.88–0.99). The exception is insertion on the PneumoniaMNIST texture (ρ = 0.27, p_bh = 0.085).
2. **They are not calibrated: they over-detect.** Among models with r_c ≤ 50, black and blur deletion still prefer the shortcut map for 75–92% of the Pet models and 100% of the CIFAR-10 models; inpainting deletion and insertion for 40–70%. The true-positive rate is 1.00 in 23 of 28 cells, and the over-detection rate is above 0 in 26 of 28.
3. **The 50% ideal is replaced by behaviour.** Per-image forward passes (shortcut moved to a wrong class's position, and removed) label whether the model follows the shortcut on each scored image. The models follow it on half of the images at r_c = 54.5% (Pet), 52.9% (CIFAR ResNet-18), 55.0% (CIFAR CNN), 53.9% (MNIST patch), 45.7% (MNIST texture), 36.9% (PneumoniaMNIST patch) and 41.9% (PneumoniaMNIST texture). On Pet the metrics switch at 1.3–5.4% r_c (blur deletion already prefers the shortcut at the lowest reliance).
4. **Per image, detection is weak inside a model.** The class-stratified AUROC of the per-image AUC gap against the behavioural label is 0.75–0.90 on natural images when all models are pooled, but the within-model median is only 0.43–0.80. PneumoniaMNIST patch is weak even pooled (0.41–0.65).
5. **Explanation size is a confound.** Against a size-matched object map the MNIST patch switch moves from 38.7–39.3% to 0.9–8.6% r_c, and the low-reliance preference for the shortcut rises from 0.007–0.022 to 0.275–0.373.
6. **Explanation form is a confound.** At the lowest CIFAR-10 dose, black and blur deletion prefer a random pixel ordering over the centred object map 75–82% of the time (small CNN) and 56–72% (ResNet-18), against 13–22% for inpainting and insertion. With real Pet masks the bias is smaller (49–53% against 35–38%).
7. **Same form, mostly correct.** Against a random ordering or a placebo, the patch-shortcut preference starts at 0.39–0.62 and rises to 0.76–1.00 (Pet), 0.96–1.00 (CIFAR ResNet-18), 0.91–0.99 (CIFAR CNN), 0.93–0.96 (MNIST patch) and 0.73–0.79 (PneumoniaMNIST patch). On diffuse textures some metrics move the wrong way: on MNIST texture blur deletion does not respond (ρ = 0.09) and insertion falls (ρ = −0.54); on PneumoniaMNIST texture black deletion falls (ρ = −0.48 vs random, −0.58 vs placebo).
8. **No metric variant is best everywhere.** Inpainting deletion is the only variant with a positive, BH-significant ρ in all 28 shortcut-map comparisons (4 comparisons × 7 settings, lowest ρ = 0.50). Black, blur and inpainting deletion are all significant in 7 of 7 settings against the full object map. A paired bootstrap shows black deletion *better* than inpainting on PneumoniaMNIST texture (Δρ = 0.293 [0.162, 0.530], Holm p = 0.021); 18 of 126 metric pairs differ at p_bh < 0.05.
9. **Explanation methods.** Integrated Gradients follows the shortcut in every setting (ρ = 0.86–1.00 on r_c). On MNIST texture, Saliency and SmoothGrad move away from the shortcut (ρ = −0.75, −0.68); Grad-CAM's trend is negative too (ρ = −0.32) but its shortcut mass changes only from 0.100 to 0.098. On PneumoniaMNIST texture, Saliency and SmoothGrad move *towards* it (ρ = 0.89, 0.78).
10. **Pre-registered Pet test.** P1, P2 and P4 hold and P3 fails for black deletion (ρ = 0.75 vs random, 0.73 vs placebo), as recorded in `docs/PREDICTIONS_pet.md`. The two-level interval for P1 black deletion (0.85) is [0.58, 0.93]; verdicts were judged on point estimates, as pre-registered.
11. **Reliance and object competence are collinear** (Spearman ≤ −0.95 in 4 of 7 settings). Only the CIFAR small CNN separates them well (ρ = −0.72; partial ρ 0.78–0.98).

The lowest PneumoniaMNIST reliance (per-dose mean 14.8% patch, 18.4% texture) is the error floor: chance-corrected it is r_c = 0.6% and 3.9%.
## Supporting study: fixed-shortcut ground truth

Measured with 5 explanation methods, 5 datasets and 3 training seeds each. The exact numbers are in `results/analysis/summary.md`.

1. **Methods differ widely in finding the true evidence.** On Decoy MNIST, Integrated Gradients puts 71% of its positive attribution on the 9-pixel shortcut patch and Occlusion 54%. Saliency, SmoothGrad and Grad-CAM stay at or below 10%. A random map scores 1%.
2. **The sanity check rewards the wrong methods.** Grad-CAM passes it best but localizes the shortcut worst. Integrated Gradients localizes it best but changes least under randomization. The sanity check's method ranking is not shown to agree with the ground truth (mean Kendall τ = −0.20 on Decoy MNIST, exact one-sided p = 0.84; 0.07 on Decoy Pneumonia, p = 0.44). With 5 methods a single seed reaches p < 0.05 only at τ ≥ 0.8, so the study cannot show disagreement either (`results/analysis/kendall_exact.csv`). The per-image evidence is mixed (Spearman ρ = −0.08 and 0.61).
3. **Perturbation metrics track the ground truth, but not everywhere.** Per image, deletion scores correlate with correct localization (Spearman ρ ≈ 0.65–0.76, and 0.36–0.39 for insertion). Method-level rankings match the ground truth on Decoy MNIST (τ = 0.60–0.87) but not on Decoy Pneumonia (τ = −0.33 to 0.33).
4. **The replacement value decides the winner.** With black pixels, Integrated Gradients and Occlusion rank 1st or 2nd on all three standard datasets. Both methods use black as their own reference image. With blurred pixels, Integrated Gradients drops to 3rd–4th and Occlusion to 4th–5th. With inpainted pixels, Integrated Gradients drops to 3rd–5th. Black-fill deletion agrees only weakly with the other variants (τ ≈ 0.17–0.21).
5. **On the chest X-rays, the metrics barely separate methods from random.** With blur replacement, all methods score close to the random baseline, and rankings are unstable across seeds (τ = 0.07). Be cautious when using these metrics on medical images.

Model accuracy (mean of 3 seeds):

| Dataset | Test accuracy |
|---|---|
| MNIST | 99.35% |
| Fashion-MNIST | 90.73% |
| PneumoniaMNIST | 83.97% (balanced accuracy 78.98%) |
| Decoy MNIST | 99.90%, but 12.64% with the patch removed. Follows the patch 96% of the time. |
| Decoy Pneumonia | 99.73%, but 37.71% with the patch removed. Follows the patch 99% of the time. |

## Datasets

| Name | Content | Classes | Source |
|---|---|---|---|
| `mnist` | Handwritten digits | 10 | torchvision |
| `fashion_mnist` | Clothing | 10 | torchvision |
| `pneumonia_mnist` | Paediatric chest X-rays | 2 | [MedMNIST v2](https://medmnist.com/) (CC BY 4.0) |
| `decoy_mnist` | MNIST plus a 3×3 white patch whose position encodes the class (10 positions along the borders) | 10 | Built from MNIST |
| `decoy_pneumonia` | PneumoniaMNIST plus a corner marker encoding the class | 2 | Built from PneumoniaMNIST |
| `pet_resnet18` | Oxford-IIIT Pet photos, cat vs dog, 128×128, with real segmentation masks (dose-response only) | 2 | torchvision `OxfordIIITPet`, preprocessed by `prepare_pet.py` |

How the decoy datasets work:

- **Training:** half the training images are swapped for an image of a different class, keeping the original label and patch. Only the patch then predicts the label reliably, which forces the model to rely on it.
- **Evaluation:** test images have the patch at a *wrong* class's position. Only images where the model follows the patch are kept, so for those images the patch is by construction the evidence for the prediction.

## Methods and metrics

| Method | Family |
|---|---|
| Saliency | Gradient |
| SmoothGrad (25 samples, noise 15%) | Gradient |
| Integrated Gradients (50 steps, black baseline) | Gradient |
| Grad-CAM (last conv block, 6×6) | Activation |
| Occlusion (4×4 patch, stride 2) | Perturbation |

| Metric | Better |
|---|---|
| Deletion AUC, black / blur (σ = 4) / inpaint (neighbour averaging plus noise, ROAD-inspired) | Lower |
| Insertion AUC from a blurred image | Higher |
| Sanity check: Spearman correlation after all layers are randomized (object pixels) | Lower |
| Ground truth (decoy only): relevance mass accuracy and rank accuracy (Arras et al., 2022) | Higher |

The statistics use 200 test images per model (50 for the sanity check), paired Wilcoxon tests with Holm correction, and Kendall τ to compare rankings between metrics, against the ground truth and between seeds.

## Dose-response design (`dose_response.py`)

- **Shortcuts:** a 3×3 white patch whose position encodes the class, or a diffuse texture: a class-specific random 10% of pixels brightened by 0.3.
- **Signed dose d:**
  - d > 0: a share d of training images is swapped for an image of another class, keeping the label and shortcut, so the model relies on the shortcut more.
  - d < 0: a share |d| of training shortcuts point to a wrong class, so it relies on the shortcut less.
  - MNIST and PneumoniaMNIST doses run from −0.4 to 0.4, with extra steps near 0, where reliance changes fastest (amendment A1 added 0.0075, 0.015, 0.03 on MNIST and −0.075, −0.025 on PneumoniaMNIST). CIFAR-10 doses reach −0.8 and Pet doses −0.5.
- **Reliance:** the share of test predictions that follow a shortcut placed at a wrong class's position, also reported chance-corrected (r_c).
- **Oracle maps,** scored on aligned test images while explaining the true label:
  - shortcut: the shortcut pixels
  - object: the image's own pixel values, with shortcut pixels removed
  - size-matched object: as many object pixels as the shortcut has, the first k of a stable descending sort (on MNIST the topmost tied saturated pixels; on Pet the mask pixels nearest the centroid)
  - placebo: the shortcut's shape at a wrong class's position; random: a random pixel ordering
- **Scores per metric:**
  - responsiveness: Spearman ρ between reliance and preference for the shortcut map
  - decision accuracy: the share of models where the preferred map matches whether reliance is above 50%
  - switch point: the reliance at which preference crosses 0.5. The 50% ideal is a modelling assumption.
- The sanity check isn't applied to the oracle maps, because they don't depend on the model.
- **Oxford-IIIT Pet (`pet_resnet18`, run on Kaggle):** a ResNet-18 trained from scratch for 30 epochs on 128×128 cat and dog photos, with a 12×12 corner patch (the same 3/32 of the side as CIFAR-10). The object map is the real segmentation mask, with the border band counted as pet. The size-matched map is the 144 mask pixels nearest the mask's centre. Image-size settings are scaled by 4: blur σ = 16 (kernel 97), 1600 inpainting steps, occlusion 16×16 with stride 8. 6 doses (−0.4 to 0) × 3 seeds = 18 models.

## Outputs

`results/dose_response/` (all 234 models; written by `dose_response.py analyze`):

| File | Content |
|---|---|
| `responsiveness_corrected.csv` | **Primary** metric table: ρ on r_c and raw reliance with two-level CIs, BH/Holm p, preferences with CIs, median AUC gaps, over-detection rate and balanced accuracy, isotonic and 4PL switch points with identification status |
| `method_responsiveness_corrected.csv` | **Primary** method table: ρ with CIs, BH/Holm p, shortcut mass at the lowest and highest dose |
| `behaviour_summary.csv` | **Primary** per-image behavioural test: class-stratified AUROC (pooled and within-model), calibration gap, behavioural switch |
| `behaviour_models.csv`, `behaviour_by_reliance.csv`, `behaviour_auroc.png` | Per-model and per-band behavioural results |
| `effect_sizes.csv`, `metric_pairs.csv` | Median per-image AUC gaps; paired metric-vs-metric tests |
| `pet_preregistered.csv` | P1–P4 statistics with two-level CIs and the original verdicts |
| `confound.csv` | Reliance vs accuracy without the shortcut, partial correlations |
| `models.csv`, `preference.csv` | Per-model data, with `classes`, `chance_floor`, `reliance_corrected`, `model_set` |
| `responsiveness.csv`, `method_responsiveness.csv`, `metric_dose_response*.png` | Legacy analysis (raw reliance, seed-only bootstrap), kept for continuity |
| `original_models/` | The same analysis on the 192 original models (the check promised by amendment A1) |

## Project structure

```
xai/
  data.py          dataset registry, decoy datasets and ground-truth masks
  models.py        CNN and checkpoint loading
  explain.py       the 5 explanation methods (Captum)
  metrics.py       perturbation curves, sanity check, localization metrics
  preprocess.py    turns dashboard drawings and uploads into model inputs
  stats.py         chance correction, isotonic/4PL switch, BH/Holm, AUROC, exact Kendall null, bootstrap picks
  dose_analysis.py corrected dose-response tables (two-level bootstrap, behaviour, confound)
  paths.py
dose_response.py           dose-response study: train, evaluate, analyze (resumable)
prepare_pet.py             downloads Oxford-IIIT Pet once and saves data/pet128/*.npz
train_models.py            trains models for any datasets and seeds
evaluate_explanations.py   runs every metric for every model
analyze_results.py         statistics, tables and figures
generate_explanations.py   example explanation grids
app.py, templates/, static/   interactive dashboard
checkpoint/                models (<dataset>.pt = seed 42, <dataset>_seed<N>.pt) and metrics JSON
results/metrics/<dataset>/ per-image results and plots for each dataset
results/analysis/          paper tables and figures (summary.md, *.csv, *.png)
results/explanations/      example grids
checkpoint/dose/           the 234 dose-response models and their reliance JSON (the .pt files are not in git)
results/dose_response/     dose-response tables and figures (summary.md, *.csv, *.png)
docs/                      paper outline, report outline, viva notes, pre-registrations
docs/audit/                expert audit reports and the resolution record (06_must_do_resolution.md)
paper/                     build_paper.py, paper_draft.docx, figures/
tests/                     unit tests for xai/stats.py and xai/dose_analysis.py
```

## Setup

Tested on Windows with Python 3.12 and a CPU only. No GPU is needed.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Reproduce everything

Run from the project root with the virtual environment active. The datasets download to `data/` automatically.

```powershell
python train_models.py --dataset all --seeds 42 1 2 --force   # about 70 min on a laptop CPU
python evaluate_explanations.py                                # about 20 min
python analyze_results.py                                      # under 1 min
foreach ($d in 'mnist','fashion_mnist','pneumonia_mnist','decoy_mnist','decoy_pneumonia') { python generate_explanations.py --dataset $d }
python dose_response.py all                                    # about 3 hours; resumes if interrupted
pwsh -File results\run_cifar.ps1                               # CIFAR-10 part, about 5 hours; resumes if interrupted
# ResNet-18 part: run on a free Kaggle GPU, about 4 hours (see kaggle/README.md)
python prepare_pet.py                                          # Pet data, about 800 MB download once
# Pet part: run on a free Kaggle GPU (see "Oxford-IIIT Pet" in kaggle/README.md)
python dose_response.py behaviour                              # MD4 per-image labels (forward passes only, minutes on CPU)
python dose_response.py analyze                                # all models (primary), about 20 min
python dose_response.py analyze --model-set original           # the 192-model check set -> results/dose_response/original_models/
python analyze_results.py --kendall-only                       # exact Kendall null for the supporting study
python paper\build_paper.py                                    # rebuild paper/paper_draft.docx
```

Each script's `--help` lists its options.

## Dashboard

```powershell
python app.py
```

Open http://127.0.0.1:5000/, then pick a dataset, and draw, upload or choose a random test image. The dashboard shows all five explanations, per-image faithfulness scores, a live sanity check and the benchmark table. On the decoy datasets you can watch the model follow the shortcut.

The server has no authentication and only listens on 127.0.0.1.

Tests: `python -m unittest test_app.py` and `python -m unittest discover -s tests -t .`

## Limitations

- Small CNNs on 28×28 grayscale images
- A synthetic shortcut, which is simpler than real-world shortcuts
- 3 seeds and 5 methods, so Kendall τ over 5 methods is coarse
- The inpainting variant is inspired by ROAD but is not the exact ROAD procedure
- The blur strength was chosen so a fully blurred image gives near-chance confidence
- PneumoniaMNIST results are a research study, not a diagnostic tool
- Dose-response: on PneumoniaMNIST and Pet the lowest raw reliance (about 15%) is the error floor; chance-corrected it is about 0%
- Dose-response: near d = 0 reliance varies a lot between seeds (for example 14–77% on MNIST texture at d = 0.005), so results are analysed against measured reliance, not the dose
- Dose-response, CIFAR-10 (both networks): one shortcut type, no augmentation, and a gap between about 45% and 96% reliance. The small CNN uses 2 seeds, ResNet-18 3 seeds.
- Dose-response, CIFAR-10 small CNN: 2 seeds and no augmentation (it slowed learning too much in a 10-epoch budget). The object map is a centred Gaussian prior, and no model fell between 45% and 96% reliance.
- Dose-response: the PneumoniaMNIST "object" map is an intensity ranking of the whole X-ray; the CIFAR-10 object map is a Gaussian prior (Pet is the main natural-image evidence); blur fill leaves 20–25% of the patch intensity; the size-matched map takes tied pixels in row-major order
- Dose-response: one metric family (deletion/insertion), models trained from scratch (the Pet ResNet-18 uses a CIFAR stem at 128 px), a synthetic shortcut, and the dose varies predictivity, not availability
- Dose-response: reliance and accuracy without the shortcut are almost collinear in 4 of 7 settings (`confound.csv`)

## Related work

Verified papers that overlap with parts of this project, and how this project differs:

- **Ground truth from injected features:** Zhou et al. (AAAI 2022) and BackX (arXiv:2405.02344) inject features or backdoors to evaluate *explanation methods* at a fixed strength. Sun et al. (MICCAI 2023) do this on chest X-rays.
- **Testing the metrics themselves:** Tomsett et al. (AAAI 2020) found saliency metrics statistically unreliable. MetaQuantus (Hedström et al., TMLR 2023) meta-evaluates metrics without ground truth. "A comprehensive study on fidelity metrics for XAI" (arXiv:2401.10640) checks fidelity metrics against a decision tree on 52,000 synthetic 128×128 images. LATEC ("Navigating the Maze of Explainable AI", Klein et al., NeurIPS 2024) shows that 20 metrics conflict across 17 methods.
- **Controlled environments:** AttributionLab (arXiv:2310.06514, NeurIPS 2023 XAIA workshop) sets network weights by hand so the true explanation is known.
- **Baseline effects:** "Back to the Baseline" (arXiv:2512.11433) shows the deletion/insertion baseline favours some methods, which matches the supporting study's finding 4.
- **Sanity check critiques:** Binder et al. (CVPR 2023) and Hedström et al. (2024).

None of these, as far as my searches found, varies the model's reliance on a shortcut in steps and checks whether the metrics' preference follows it.

## Files kept from the B.Tech version

`train.py`, `torch_serve/`, `app.yaml`, `digit_demo.gif`, `static/screenshot.PNG` and `checkpoint/mnist_btech.pt` are not used by the current code.

## References

1. Adebayo, J., et al. (2018). Sanity Checks for Saliency Maps. NeurIPS.
2. Adebayo, J., et al. (2022). Post hoc Explanations may be Ineffective for Detecting Unknown Spurious Correlation. ICLR.
3. Arras, L., Osman, A., and Samek, W. (2022). CLEVR-XAI: A Benchmark Dataset for the Ground Truth Evaluation of Neural Network Explanations. Information Fusion.
4. Binder, A., et al. (2023). Shortcomings of Top-Down Randomization-Based Sanity Checks for Evaluations of Deep Neural Network Explanations. CVPR.
5. Hedström, A., Weber, L., Lapuschkin, S., and Höhne, M. (2024). A Fresh Look at Sanity Checks for Saliency Maps. arXiv:2405.02383 (overlaps with "Sanity Checks Revisited", arXiv:2401.06465, NeurIPS XAIA 2023).
6. Kokhlikyan, N., et al. (2020). Captum: A Unified and Generic Model Interpretability Library for PyTorch. arXiv:2009.07896.
7. Petsiuk, V., Das, A., and Saenko, K. (2018). RISE: Randomized Input Sampling for Explanation of Black-box Models. BMVC.
8. Rong, Y., et al. (2022). A Consistent and Efficient Evaluation Strategy for Attribution Methods (ROAD). ICML.
9. Ross, A. S., Hughes, M. C., and Doshi-Velez, F. (2017). Right for the Right Reasons: Training Differentiable Models by Constraining their Explanations. IJCAI.
10. Samek, W., et al. (2017). Evaluating the Visualization of What a Deep Neural Network Has Learned. IEEE TNNLS.
11. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. ICCV.
12. Simonyan, K., Vedaldi, A., and Zisserman, A. (2014). Deep Inside Convolutional Networks. ICLR Workshop.
13. Smilkov, D., et al. (2017). SmoothGrad: Removing Noise by Adding Noise. arXiv:1706.03825.
14. Sundararajan, M., Taly, A., and Yan, Q. (2017). Axiomatic Attribution for Deep Networks. ICML.
15. Yang, J., et al. (2023). MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification. Scientific Data.
16. Zeiler, M. D., and Fergus, R. (2014). Visualizing and Understanding Convolutional Networks. ECCV.
17. Kermany, D. S., et al. (2018). Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning. Cell.

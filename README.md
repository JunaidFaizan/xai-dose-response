# Do XAI Metrics Notice When a Model Starts Cheating? A Dose-Response Evaluation of Explanation Metrics

M.Tech final year project in Explainable AI. It extends my B.Tech project, a handwritten digit recognition app built with PyTorch and Flask.

## The question

Explanation methods (heatmaps) are judged with evaluation metrics such as deletion and insertion curves, but those metrics are rarely checked themselves, because the true explanation is normally unknown.

This project controls the truth. It trains models on data with an injected shortcut, a local patch or a diffuse texture that gives away the label, and **varies in steps how much the model relies on it**: from ignoring it completely to depending on it almost entirely. Reliance is measured, not assumed. Then it asks whether the metrics notice: when the model's reasoning shifts from the object to the shortcut, does a metric's preference shift from an object explanation to a shortcut explanation?

To test the metrics independently of any explanation method, every model is scored with fixed "oracle" explanations (a shortcut-only map and an object-only map, plus a size-matched object map as a control). The five explanation methods are also run on every model.

In my literature searches (September 2026) I didn't find published work that varies shortcut reliance in steps to test the evaluation metrics. The closest work is listed under [Related work](#related-work). Confirm this with a full literature search before claiming novelty.

## Main findings: dose-response study

144 models: MNIST and PneumoniaMNIST × patch and texture shortcuts × 12 reliance levels × 3 seeds, plus 12 CIFAR-10 models (patch shortcut, 6 doses × 2 seeds), each scored on 100 test images. Exact numbers are in `results/dose_response/summary.md`, and the main figures are `metric_dose_response.png`, `metric_dose_response_size_matched.png` and `method_dose_response.png`.

1. **Responsiveness depends on the data.** On MNIST, every metric's preference tracks the model's reliance closely (Spearman ρ = 0.91–0.98 in every setting). On the chest X-rays it ranges from ρ = 0.07 to 0.84, and 3 of the 16 metric-setting combinations aren't significant. The worst case is black-fill deletion on the X-ray texture shortcut under the size-matched control, which doesn't respond at all (ρ = 0.07, p = 0.68).
2. **The metrics switch too early.** An ideal metric would switch to preferring the shortcut explanation when the model follows the shortcut half the time. On MNIST with a patch, they switch at 30–42% reliance against the full object map, and at only 6–15% against the size-matched map.
3. **Explanation size is a confound.** At low reliance, the metrics strongly prefer the full object map (preference for the shortcut map 0.01–0.02 on MNIST patch). Against an object map of the same size as the shortcut, that preference rises to 0.27–0.37. Part of what looks like "the metric recognises the object" is the object map simply being bigger.
4. **Insertion fails on a diffuse shortcut.** On MNIST texture, insertion (blur) only switches at about 98–99% reliance and never goes above 0.6. Deletion with black or inpainted pixels switches at 28–35%.
5. **Inpainting deletion is the most robust variant.** It is the only metric that responds significantly in all eight settings (ρ ≥ 0.58, p < 0.001).
6. **On CIFAR-10 (32×32 colour, a 0.77M-parameter CNN), the metrics rank correctly but are badly calibrated.** 12 models (6 doses × 2 seeds, patch shortcut) span 4–96% reliance, with 63–74% accuracy when the shortcut is removed. Every metric's preference rises with reliance (ρ = 0.88–0.99). But even at 4% reliance, black and blur deletion prefer the shortcut map 74–82% of the time, so they would wrongly conclude that a model is using a shortcut it ignores (decision accuracy 0.17). Inpainting deletion and insertion are much less biased (0.23–0.27 at low reliance). One possible cause is that removing scattered pixels damages natural images more than removing a compact region, which is the shape confound that ROAD describes. The CIFAR-10 object map is a centred Gaussian, not a real object mask, so treat this as a hypothesis.
7. **Some explanation methods move the wrong way.** As reliance on the MNIST texture shortcut grows, Saliency, SmoothGrad and Grad-CAM put *less* weight on it (ρ = −0.49 to −0.80). Integrated Gradients follows the shortcut in every setting (ρ = 0.86–0.97).

## Supporting study: fixed-shortcut ground truth

Measured with 5 explanation methods, 5 datasets and 3 training seeds each. The exact numbers are in `results/analysis/summary.md`.

1. **Methods differ widely in finding the true evidence.** On Decoy MNIST, Integrated Gradients puts 71% of its positive attribution on the 9-pixel shortcut patch and Occlusion 54%. Saliency, SmoothGrad and Grad-CAM stay at or below 10%. A random map scores 1%.
2. **The sanity check rewards the wrong methods.** Grad-CAM passes it best but localizes the shortcut worst. Integrated Gradients localizes it best but changes least under randomization. The sanity check's method ranking does not agree with the ground truth (Kendall τ = −0.20 on Decoy MNIST, 0.07 on Decoy Pneumonia). The per-image evidence is mixed (Spearman ρ = −0.08 and 0.61).
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
  - Doses run from −0.4 to 0.4, with extra steps near 0, where reliance changes fastest.
- **Reliance:** the share of test predictions that follow a shortcut placed at a wrong class's position.
- **Oracle maps,** scored on aligned test images while explaining the true label:
  - shortcut: the shortcut pixels
  - object: the image's own pixel values, with shortcut pixels removed
  - size-matched object: the brightest object pixels, as many as the shortcut has
- **Scores per metric:**
  - responsiveness: Spearman ρ between reliance and preference for the shortcut map
  - decision accuracy: the share of models where the preferred map matches whether reliance is above 50%
  - switch point: the reliance at which preference crosses 0.5. The 50% ideal is a modelling assumption.
- The sanity check isn't applied to the oracle maps, because they don't depend on the model.

## Project structure

```
xai/
  data.py          dataset registry, decoy datasets and ground-truth masks
  models.py        CNN and checkpoint loading
  explain.py       the 5 explanation methods (Captum)
  metrics.py       perturbation curves, sanity check, localization metrics
  preprocess.py    turns dashboard drawings and uploads into model inputs
  paths.py
dose_response.py           dose-response study: train, evaluate, analyze (resumable)
train_models.py            trains models for any datasets and seeds
evaluate_explanations.py   runs every metric for every model
analyze_results.py         statistics, tables and figures
generate_explanations.py   example explanation grids
app.py, templates/, static/   interactive dashboard
checkpoint/                models (<dataset>.pt = seed 42, <dataset>_seed<N>.pt) and metrics JSON
results/metrics/<dataset>/ per-image results and plots for each dataset
results/analysis/          paper tables and figures (summary.md, *.csv, *.png)
results/explanations/      example grids
checkpoint/dose/           the 144 dose-response models and their reliance JSON
results/dose_response/     dose-response tables and figures (summary.md, *.csv, *.png)
docs/                      paper outline, report outline, viva notes
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
```

Each script's `--help` lists its options.

## Dashboard

```powershell
python app.py
```

Open http://127.0.0.1:5000/, then pick a dataset, and draw, upload or choose a random test image. The dashboard shows all five explanations, per-image faithfulness scores, a live sanity check and the benchmark table. On the decoy datasets you can watch the model follow the shortcut.

The server has no authentication and only listens on 127.0.0.1.

Tests: `python -m unittest test_app.py`

## Limitations

- Small CNNs on 28×28 grayscale images
- A synthetic shortcut, which is simpler than real-world shortcuts
- 3 seeds and 5 methods, so Kendall τ over 5 methods is coarse
- The inpainting variant is inspired by ROAD but is not the exact ROAD procedure
- The blur strength was chosen so a fully blurred image gives near-chance confidence
- PneumoniaMNIST results are a research study, not a diagnostic tool
- Dose-response: the lowest reliance reached on PneumoniaMNIST was about 15%, not 0
- Dose-response: near d = 0 reliance varies a lot between seeds (for example 14–77% on MNIST texture at d = 0.005), so results are analysed against measured reliance, not the dose
- Dose-response, CIFAR-10: one shortcut type, 2 seeds and no augmentation (it slowed learning too much in a 10-epoch budget). The object map is a centred Gaussian prior, and no model fell between 45% and 96% reliance.
- Dose-response: the 50% switch point is a modelling assumption, and the size-matched control uses the brightest pixels, which is only one way to match size

## Related work

Verified papers that overlap with parts of this project, and how this project differs:

- **Ground truth from injected features:** Zhou et al. (AAAI 2022) and BackX (arXiv:2405.02344) inject features or backdoors to evaluate *explanation methods* at a fixed strength. Sun et al. (MICCAI 2023) do this on chest X-rays.
- **Testing the metrics themselves:** Tomsett et al. (AAAI 2020) found saliency metrics statistically unreliable. MetaQuantus (Hedström et al., 2023) meta-evaluates metrics without ground truth. "A comprehensive study on fidelity metrics for XAI" (arXiv:2401.10640) checks fidelity metrics against a transparent model. LATEC (Klein et al., NeurIPS 2024) shows that 20 metrics conflict across 17 methods.
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
5. Hedström, A., et al. (2024). A Fresh Look at Sanity Checks for Saliency Maps. arXiv:2405.02383.
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

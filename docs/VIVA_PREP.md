# Viva Preparation

Project title: A Dose-Response Evaluation of Explanation Metrics under Controlled Shortcut Reliance in CNN Image Classifiers

## One-minute pitch

> Explanation methods are judged with metrics such as deletion and insertion curves, but the metrics themselves are rarely checked, because the true explanation is normally unknown. I control the truth: a training dose sets how much each model relies on an injected shortcut, and I measure that reliance on conflicting images and correct it for chance. I trained 275 models, on Oxford-IIIT Pet with real masks (from scratch and, as a separate pre-registered setting, ImageNet-pretrained), CIFAR-10 with two networks, MNIST and chest X-rays. Then I scored fixed shortcut and object explanations with four metrics. The metrics respond to reliance, but they are not calibrated to it. On natural images, black and blur deletion report the shortcut for at least half of the Pet models and nearly all CIFAR-10 models that barely use it. Their verdicts also change with the size and form of the explanations being compared. A pre-registered test on Pet confirmed three of four predictions, and black deletion failed the form-matched control. A pre-registered amendment confirmed four of five more: the over-detection persists on the pretrained network, and the metrics switch before behaviour once the reliance gaps are filled; only the rise of the excess shortcut preference failed.
## Demo script (about 3 minutes)

1. Start `python app.py` and open http://127.0.0.1:5000/. Test it the day before.
2. **MNIST:** draw a digit and show the five heatmaps, the deletion scores and the chart.
3. **Decoy MNIST:** click **Random test image**. The patch sits at the wrong class's position and the model predicts the patch's class. Show that Integrated Gradients and Occlusion highlight the patch, while Grad-CAM and Saliency mostly don't.
4. Click **Run sanity check**. Grad-CAM "passes" and Integrated Gradients "fails", which is the main finding, live.
5. **PneumoniaMNIST:** show a test X-ray. Present it as a research demo, not a diagnosis.
6. Scroll to the benchmark table, which holds the test-set averages over 3 seeds.

Keep backup screenshots in your slides.

## Formulas to know

S_c is the log-probability of class c.

- **Saliency:** R_i = |∂S_c / ∂x_i|
- **SmoothGrad:** R = (1/n) Σ_k |∇S_c(x + ε_k)|, with ε_k ~ N(0, 0.15²) and n = 25
- **Integrated Gradients:** R_i = (x_i − x'_i) × ∫₀¹ ∂S_c(x' + α(x − x')) / ∂x_i dα, with a black x' and 50 steps
  - Completeness: Σ R_i = S_c(x) − S_c(x')
- **Grad-CAM:** α_k = mean over i,j of ∂S_c / ∂A^k_ij, then L = ReLU(Σ_k α_k A^k), upsampled from 6×6 (28×28 CNN), 16×16 (CIFAR CNN, Pet ResNet-18) or 4×4 (CIFAR ResNet-18)
- **Occlusion:** R_i = the mean of S_c(x) − S_c(x with a 4×4 patch set to 0), over the patches covering pixel i
- **Deletion AUC:** the area under p(c) as the top 0, 5, …, 100% of pixels are replaced. Lower is better.
- **Insertion AUC:** start from a blurred image and restore pixels. Higher is better.
- **Mass accuracy:** Σ_{i in patch} R_i⁺ / Σ_i R_i⁺
- **Rank accuracy:** the share of the top-9 pixels that fall inside the 9-pixel patch
- **Kendall τ:** (concordant pairs − discordant pairs) / total pairs, which compares two rankings from −1 to 1. With 5 methods the exact null has 120 orderings; one seed needs τ ≥ 0.8 for one-sided p < 0.05 (P(T ≥ 0.8) = 5/120, P(T ≥ 0.6) = 14/120, P(T ≥ 0.2) = 49/120)
- **Chance floor and corrected reliance:** f = (100 − accuracy without shortcut)/(C − 1); r_c = 100 (R − f)/(100 − f)
- **Over-detection rate:** the share of models with r_c ≤ 50 whose preference for the shortcut map is above 0.5; balanced accuracy = (TPR + TNR)/2
- **Switch point:** the first r_c where the increasing isotonic fit of preference reaches 0.5; "identified" only if a model lies within ±10 points
- **Two-level bootstrap:** resample models within each dose, then the 100 images within each picked model; 2000 replicates; percentile intervals
- **Behavioural label (primary):** 1 if the model predicts the shortcut's class when the shortcut is moved to a wrong class's position and not when it is removed; images already predicted as the shortcut's class without a shortcut are undetermined

## Likely questions

**Why do you trust the patch as ground truth?**
Three reasons. The model reaches 99.9% accuracy with the patch but only about 13% without it on Decoy MNIST. It follows a misleading patch in 96% or more of cases. And I only evaluate images where it followed the misleading patch: the image shows one class, the model predicts another, and the patch is the only thing that points to that class.

**Why swap half the training images?**
Without it, the digit alone predicts the label perfectly, so the model mostly ignored the patch. Shortcut reliance was only 1–17% in my first attempt. Swapping images makes only the patch fully predictive, so the model has to use it.

**Isn't a synthetic patch unrealistic?**
It's simpler than real shortcuts, such as the hospital text markers in X-rays. The point is a known ground truth, and the decoy pneumonia setup mimics a corner marker. Realistic shortcuts are future work.

**Why does Integrated Gradients fail the sanity check but localize best?**
IG multiplies the gradient by the input. That keeps the input's structure even for a random model, so the correlation stays high. But for the trained model, its gradients do point to the patch. Sensitivity to randomization and correctness are different properties. Binder et al. (2023) also argued that the sanity check has conceptual problems.

**Why does Grad-CAM do so badly?**
Its map is 6×6, so a 3×3 patch at the border is blurred over a large area. It also depends mostly on the top-layer weights, which is why it changes as soon as the output layer is randomized. That's why it "passes" the sanity check.

**Why three deletion variants?**
The replacement value is a known confound (ROAD, Rong et al., 2022). My results confirm it: with black replacement, IG and Occlusion rank at the top, and both use black as their own reference image. With blur or inpainting they drop.

**How did you choose the blur strength?**
I checked the model's confidence on fully blurred images. With σ = 2 an MNIST model still had 63% confidence, so the blur still showed the digit. With σ = 4 confidence falls to about chance level, so blurred pixels carry little information.

**Is the inpainting the same as ROAD?**
No. It's inspired by ROAD: neighbour averaging plus noise instead of ROAD's exact noisy linear imputation. I state this as a limitation.

**Why only 5 methods? Isn't Kendall τ coarse?**
Yes, with 5 methods τ moves in steps of 0.2. That's why I also report per-image Spearman correlations over about 3,000 (image, method) pairs, and repeat everything over 3 seeds.

**Why is Decoy Pneumonia different from Decoy MNIST?**
The patch sits on textured X-ray background rather than black, and with 2 classes the evidence is spread out more. Rankings are noisier there, which is itself a warning for medical use.

**Which metric would you recommend?**
Deletion with a neutral replacement (blur or inpaint), reported alongside other variants, and never the sanity check alone as proof of correctness. Where possible, check localization on a known shortcut first.

**What is your contribution compared with existing work?**
BAM and SMERF already control what a model relies on, but they use that to test attribution methods. Other work meta-evaluates metrics without ground truth, or documents removal-shape and fill confounds (ROAD, ROAR pitfalls, Blücher et al.). I combine graded, measured reliance as ground truth for the metrics themselves with size- and form-matched controls, per-image behavioural labels and a pre-registered test. In our searches we did not find this combination.

**How much did AI help?**
Answer honestly, following your university's policy. For example: you defined the research question and design, ran and checked every experiment, interpreted the results, and wrote the text. AI tools helped write and debug code. Be ready to explain every file in `xai/`.

## Questions on the dose-response study

**Why correct reliance for chance?**
A model that ignores the shortcut still lands on the shortcut's class whenever it makes an error. On a binary task every error does. So raw reliance has a floor equal to the error rate. On PneumoniaMNIST the lowest per-dose mean is 14.8% raw, which is 0.6% after correction. On Pet the lowest model is 15.5% raw and −0.1% corrected. The correction assumes errors spread evenly over the wrong classes. That is an approximation, because images without the shortcut are out of distribution.

**Why not just say the metric should switch at 50%?**
Preference is scored on aligned images, while reliance is measured on conflicting images, so 50% is an assumption. The per-image forward passes give a derived ideal: the reliance at which the models follow the shortcut on half of the scored images, judged by the same ±10-point rule as the metric switches. Under the primary label it is 53.9% r_c on MNIST patch, 45.7% on MNIST texture and 36.9% and 41.9% on PneumoniaMNIST. On Pet it is 54.1%, bracketed since the three amendment-A2 models at dose −0.15 were added (54.5% and not bracketed without them), and 29.4–54.1% depending on the label. On CIFAR-10 it used to lie in a reliance gap (above 42.7% for the small CNN and 36.5% for ResNet-18); the amendment-A2 models fill the gap, and it is now identified and bracketed at 55.2% (ResNet-18) and 60.3% (small CNN). On the pretrained Pet network (SD1) it is not identified: no model lies between 31.2% and 96.4%, so I can only say it is above 31.2%.

**What does "identified" mean for a switch point?**
The isotonic switch counts as identified only if a model lies within ±10 reliance points of it. Before amendment A1 there was no MNIST patch model between 23% and 45% raw reliance, so the early switches were interpolated across a gap. A1 added models there, and the nearest model is now about 1 point away. The switches moved to 38.7–39.3% r_c.

**Isn't reliance just the inverse of object accuracy?**
Largely, yes. Spearman(r_c, accuracy without the shortcut) is −0.95 or lower in 3 of the 7 settings. They are least collinear on PneumoniaMNIST (−0.78 and −0.82). The CIFAR small CNN was at −0.64 before its amendment-A2 models filled its reliance gap and is now at −0.92; there the partial correlation of preference with r_c, given accuracy, stays at 0.59–0.84, significant for all four metrics. After Benjamini–Hochberg over the 32 tests (including SD1), 19 of the 28 partial correlations of the main settings remain significant. I report this as a limitation.

**Why does the pooled AUROC look good but the within-model AUROC weak?**
Pooled over all models, high-reliance models have both more followed images and larger AUC gaps. Most of the signal therefore separates models, not images within one model. Within a model, the median AUROC on natural images is 0.50–0.78.

**Did the pre-registered predictions hold?**
P1, P2 and P4 held. P3 failed for black deletion (ρ = 0.75 against random, 0.73 against placebo). Verdicts were judged on point estimates, as registered. The later two-level interval for P1 black deletion, [0.58, 0.93], reaches below the threshold, and I report it as added information without changing the verdict. Of the amendment-A2 predictions, P5, P6, P7 and P9 held and P8 failed: on the pretrained network all four metrics respond (ρ = 0.86–0.97), but the excess preference over a random ordering rises significantly only for inpainting deletion (1 of 4, 3 required). The SD1 over-detection of blur deletion (5 of 9 models) has a Clopper–Pearson interval of 0.21–0.86, so P5 rests on a point estimate.

**Is inpainting deletion the best metric?**
Not shown. It is the only variant that responds significantly in all 28 shortcut-map comparisons. But a paired test finds black deletion better than inpainting on PneumoniaMNIST texture (Δρ = 0.293 [0.162, 0.530], Holm p = 0.024). Black and blur deletion carry the largest form bias on natural images.

**How many tests, and how did you correct?**
Benjamini–Hochberg within each family (each comparison, the methods, the metric pairs, each behavioural label), plus Holm over the headline tests. p-values below 0.001 are printed as p < 0.001.

**Which journal and why?**
The provisional target is Neurocomputing (Elsevier, Q1), because it fits empirical neural-network studies. I still have to confirm the no-fee route in a browser.
## Weak points to raise before the examiners do

- One metric family; one pretrained network only (SD1, whose Grad-CAM maps are 4×4); the Pet ResNet-18 from scratch has a CIFAR stem at 128 px; a synthetic shortcut whose predictivity, not availability, is varied
- Reliance and object competence are collinear in most settings
- The PneumoniaMNIST "object" map is an intensity ranking; the CIFAR-10 object map is a Gaussian prior; blur fill leaves 20–25% of the patch intensity
- 3 seeds and 5 methods
- The ROAD-inspired inpainting approximation
- The blur strength was chosen by a simple check
- The dashboard's 0.5 "passes/fails" cut-off is only a visual guide

## Before the viva

- [ ] `python -m unittest test_app.py` and `python -m unittest discover -s tests -t .` pass, and the dashboard runs from a fresh terminal
- [ ] You can explain `xai/stats.py` (chance correction, isotonic switch, BH/Holm) and the `behaviour` step in `dose_response.py`
- [ ] You can explain `xai/data.py` (decoy construction), `xai/explain.py` and `xai/metrics.py`
- [ ] You've read Adebayo et al. (2018), Binder et al. (2023) and the ROAD paper, at least the abstracts, figures and conclusions
- [ ] The slides include backup screenshots of the demo

# Viva Preparation

## One-minute pitch

> Explanation methods are judged with metrics, but nobody checks whether the metrics themselves are right, because the true explanation is normally unknown. I created cases where it is known: I trained models on data where a small patch gives away the label, so the models rely on the patch, and a correct explanation must point to it. Then I tested five explanation methods and the common metrics against this ground truth, on three seeds and on standard and medical data. The main findings: the popular sanity check rewards the wrong methods, since Grad-CAM passes it best but misses the evidence. Deletion metrics track the truth better, but which method wins depends on the replacement value. And on chest X-rays, the metrics hardly separate methods from random.

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
- **Grad-CAM:** α_k = mean over i,j of ∂S_c / ∂A^k_ij, then L = ReLU(Σ_k α_k A^k), upsampled from 6×6
- **Occlusion:** R_i = the mean of S_c(x) − S_c(x with a 4×4 patch set to 0), over the patches covering pixel i
- **Deletion AUC:** the area under p(c) as the top 0, 5, …, 100% of pixels are replaced. Lower is better.
- **Insertion AUC:** start from a blurred image and restore pixels. Higher is better.
- **Mass accuracy:** Σ_{i in patch} R_i⁺ / Σ_i R_i⁺
- **Rank accuracy:** the share of the top-9 pixels that fall inside the 9-pixel patch
- **Kendall τ:** (concordant pairs − discordant pairs) / total pairs, which compares two rankings from −1 to 1

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
Existing work either criticises the metrics in theory or evaluates methods against ground truth. I evaluate the metrics themselves against ground truth, on natural and medical data, with seeds and significance tests.

**How much did AI help?**
Answer honestly, following your university's policy. For example: you defined the research question and design, ran and checked every experiment, interpreted the results, and wrote the text. AI tools helped write and debug code. Be ready to explain every file in `xai/`.

## Weak points to raise before the examiners do

- Small models on 28×28 images, and a synthetic shortcut
- 3 seeds and 5 methods
- The ROAD-inspired inpainting approximation
- The blur strength was chosen by a simple check
- The dashboard's 0.5 "passes/fails" cut-off is only a visual guide

## Before the viva

- [ ] `python -m unittest test_app.py` passes, and the dashboard runs from a fresh terminal
- [ ] You can explain `xai/data.py` (decoy construction), `xai/explain.py` and `xai/metrics.py`
- [ ] You've read Adebayo et al. (2018), Binder et al. (2023) and the ROAD paper, at least the abstracts, figures and conclusions
- [ ] The slides include backup screenshots of the demo

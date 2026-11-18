# Synopsis

**Project Title:** A Dose-Response Evaluation of Explanation Metrics under Controlled Shortcut Reliance in CNN Image Classifiers

**Area:** Explainable Neural Networks (Explainable AI)

**Programme:** M.Tech, Final Year Project

**Student:** [Name], [Roll Number]

**Guide:** [Guide Name], [Department], [College Name]

---

## Background

Convolutional Neural Networks (CNNs) reach high accuracy on image classification, but they work as black boxes: they give a prediction without saying why. This limits their use where decisions must be justified, such as healthcare and finance, and regulations such as the EU AI Act increasingly require transparency.

Explainable AI (XAI) methods produce attribution maps (heatmaps) that highlight the input regions a model relied on. Because many methods exist and they often disagree, researchers compare them with **evaluation metrics**, such as deletion and insertion curves [6, 8] and the model-randomization sanity check [5]. These metrics decide which explanation methods are trusted and published.

The metrics themselves, however, are rarely validated, because the true explanation of a trained network is normally unknown. Recent work shows that metrics disagree with each other [10, 13], depend on hidden choices such as the replacement value for removed pixels [9, 14], and that the sanity check has conceptual flaws [11, 12]. A metric that cannot tell when a model's reasoning changes cannot be relied on to judge explanations.

A common and dangerous failure of neural networks is **shortcut learning**: the model relies on an irrelevant cue, such as a hospital marker on an X-ray, instead of the real object [15, 16]. Detecting this is one of the main reasons explanations are used in practice.

## Problem Statement

Earlier work controlled what a model relies on in order to test attribution methods (BAM [22], SMERF [23]), or meta-evaluated the evaluation metrics without ground truth [10, 13] or against the known explanation of a transparent model, a decision tree [25]. In our searches we did not find a study that varies a model's measured reliance on a shortcut in steps to test the evaluation metrics themselves. This project addresses the question:

> When a model's reliance on a shortcut increases in controlled steps, do XAI evaluation metrics shift their preference from object explanations to shortcut explanations, and at what level of reliance?

**Thesis:** the metrics respond to reliance but are not calibrated to it, and their verdicts depend on the size and form of the explanations being compared.

**Target:** provisional target journal Neurocomputing (Elsevier, Q1 in Artificial Intelligence). The free (subscription, no-fee) route must still be confirmed in a normal browser before submission.

## Novelty

In our literature searches (an audit covering 2019 to October 2026, re-checked on 2026-11-13) we did not find published work that:

1. **varies a model's reliance on a shortcut in controlled steps** (a "dose-response" design) to test the evaluation metrics, rather than the explanation methods;
2. scores **fixed oracle explanations** (shortcut-only, object-only, size-matched, placebo and random maps), so that the metrics are tested independently of any explanation method;
3. reports when metrics **switch their verdict** relative to the model's measured reliance, and separates genuine responses to the model from biases caused by the size and form of the explanation being scored.

The individual pieces are not new. BAM [22] and SMERF [23] already controlled what a model relies on, to evaluate *attribution methods*; shortcut or backdoor ground truth is used for methods at a fixed strength [16, 17, 18]; metrics are meta-evaluated without ground truth [10, 13]; hand-designed networks with known explanations test attribution methods [19]; fidelity metrics are checked against a decision tree's known explanation [25]; and removal shape, fill and baseline confounds are documented [9, 14, 24]. What is new is the combination: graded, measured reliance as ground truth for the metrics, with size and form controls.

## Objectives

1. Build datasets with injected shortcuts (a local patch and a diffuse texture) whose position or pattern encodes the class.
2. Control and measure how much each trained model relies on the shortcut, from about 0% to about 100%.
3. Test whether four faithfulness metrics follow this change, using fixed oracle explanations and controls for explanation size and shape.
4. Measure how five explanation methods respond to the same change in reliance.
5. Repeat on natural images with real object masks (Oxford-IIIT Pet), a colour natural-image dataset (CIFAR-10, including a standard ResNet-18), a handwriting dataset and a medical dataset, with several random seeds, and test pre-registered predictions on Pet.
6. Provide an interactive dashboard for generating and evaluating explanations live.

## Methodology

**Datasets**
- Oxford-IIIT Pet: cat vs dog photos, 128×128 colour, with real segmentation masks
- MNIST: handwritten digits, 10 classes, 28×28 grayscale
- PneumoniaMNIST [20]: paediatric chest X-rays, normal vs pneumonia, 28×28 grayscale
- CIFAR-10: natural images, 10 classes, 32×32 colour
- Supporting study also uses Fashion-MNIST

**Shortcuts**
- Patch: a 3×3 white square whose position encodes the class
- Texture: a class-specific set of about 10% of pixels, slightly brightened, spread over the whole image

**Controlling reliance (the "dose")**
- Positive dose: a share of training images is replaced by an image of another class while keeping the label and shortcut, so the model relies on the shortcut more.
- Negative dose: a share of training shortcuts points to a wrong class, so the model relies on it less.
- Reliance is measured, not assumed: the share of test predictions that follow a shortcut placed at a wrong class's position. It is chance-corrected, r_c = 100 (R − f)/(100 − f) with f = (100 − accuracy without shortcut)/(C − 1), because a model that ignores the shortcut still lands on its class when it errs.
- Per-image behavioural labels from forward passes (shortcut moved and removed) say whether the model follows the shortcut on each scored image.

**Models**
- A custom CNN with four convolutional layers for 28×28 images; for CIFAR-10, a larger CNN with batch normalisation (0.77M parameters) and a standard ResNet-18 (11.2M parameters)
- 237 trained models in total: 192 original (MNIST and PneumoniaMNIST 144, CIFAR-10 small CNN 12, CIFAR-10 ResNet-18 18, Pet ResNet-18 18), 42 from the pre-registered amendment A1 (extra MNIST and PneumoniaMNIST doses and seeds to fill reliance gaps) and 3 from amendment A2 (Pet ResNet-18 at dose −0.15). The A2 pilots are reported separately; the ImageNet-pretrained Pet models (SD1) and the extra CIFAR-10 doses (SD2) are pending
- The ResNet-18 models were trained on a free Kaggle GPU (NVIDIA T4); everything else on a laptop CPU

**Oracle explanations** (scored by every metric)
- Shortcut map: the shortcut pixels
- Object map: the object's own pixels
- Size-matched object map: the same number of pixels as the shortcut, to control for size
- Placebo map: the shortcut's shape at a position the model does not use, to control for shape
- Random map: a fully random pixel ordering, to check what the placebo's effect comes from

**Evaluation metrics under test**
- Deletion curves [6] with three replacement values: black, blurred and inpainted (ROAD-inspired [9])
- Insertion curve [8] from a blurred image
- Model-randomization sanity check [5] (in the supporting study)

**Scores for each metric**
- Responsiveness: correlation between the model's chance-corrected reliance and the metric's preference for the shortcut map
- Over-detection rate and balanced accuracy: how often the metric reports a shortcut the model does not use
- Switch point: the reliance at which the metric changes its verdict (isotonic fit), compared with the behavioural switch
- Per-image AUROC against the behavioural labels

**Explanation methods**

| Method | Family | Idea |
|---|---|---|
| Saliency [3] | Gradient-based | Gradient of the output with respect to each input pixel |
| SmoothGrad [21] | Gradient-based | Saliency averaged over noisy copies of the input |
| Integrated Gradients [2] | Gradient-based | Gradients accumulated along a path from a blank baseline to the input |
| Grad-CAM [1] | Activation-based | Weights the last convolutional feature maps by their gradients |
| Occlusion [4] | Perturbation-based | Hides image patches and measures the change in prediction |

**Statistics**
- Spearman and Kendall rank correlations with a two-level bootstrap (seeds within dose, images within model), Benjamini–Hochberg and Holm corrections, effect sizes, paired metric-vs-metric tests and an exact Kendall null for five methods

## Preliminary Findings

All 237 pooled models have been trained and evaluated, including the placebo, random-map and per-image behavioural controls (numbers from `results/dose_response/*_corrected.csv` and `behaviour_summary.csv`):

1. **The metrics respond to reliance.** Preference for the shortcut map rises with chance-corrected reliance in 27 of 28 metric–setting cells (ρ = 0.49–0.99 in those cells; natural images 0.83–0.99).
2. **Black and blur deletion over-detect on natural images.** Among models with chance-corrected reliance r_c ≤ 50%, black and blur deletion still prefer the shortcut map for 79–93% of Pet models (50–64% if each model's own 95% interval must exceed 0.5) and, against a centred Gaussian object prior, 100% of CIFAR-10 models (93–100% with that margin). Much of this comes from the object map: at the lowest dose the metrics prefer the shortcut map over the object map about as often as a random pixel ordering (within 0.07), and on Pet black and blur deletion do not rank the real object mask above random pixels. The shortcut's excess preference over random still rises with r_c for all four Pet metrics (ρ = 0.47–0.81).
3. **The 50% ideal is replaced by behaviour where the models identify it.** Under the primary label the models follow the shortcut on half of the images at r_c = 53.9% on MNIST patch and 36.9–45.7% on MNIST texture and PneumoniaMNIST. On Pet the crossing (54.1%) is now bracketed by the A2 models at dose −0.15 (54.5% and not bracketed without them) and depends on the label (29.4–54.1%); on CIFAR-10 it is not identified (above 42.7% and 36.5%, in a reliance gap). On Pet the metrics switch already at 1.3–5.4%, earlier than behaviour in every testable natural-image cell (paired bootstrap).
4. **Within a model, per-image detection is weak** (median AUROC 0.43–0.78 on natural images, against 0.74–0.89 pooled over models), and each within-model AUROC rests on only 5–11 minority-label images.
5. **Explanation size distorts the metrics.** With a size-matched object map, the MNIST patch switch moves from 38.7–39.3% to 0.9–8.6% r_c.
6. **Explanation form distorts the metrics.** On CIFAR-10 at the lowest dose, black and blur deletion prefer a random pixel ordering over a centred object map 75–81% (small CNN) and 56–72% (ResNet-18) of the time, against 13–23% for inpainting and insertion. With real Pet masks this drops to 49–53%.
7. **Same-form comparisons mostly behave correctly,** but the rise depends on the setting: the preference over a random or placebo map reaches 0.76–1.00 on Pet, 0.91–1.00 on CIFAR-10, 0.93–0.96 on MNIST patch and only 0.73–0.79 on PneumoniaMNIST patch. On diffuse textures, some metrics move the wrong way.
8. **No variant is best everywhere.** Inpainting deletion responds in all 28 shortcut-map comparisons, but a paired test shows black deletion better than inpainting on PneumoniaMNIST texture.
9. **Explanation methods differ:** Integrated Gradients follows the shortcut in every setting. Saliency and SmoothGrad move away from the diffuse shortcut on MNIST texture, but towards it on PneumoniaMNIST texture; Grad-CAM's change on MNIST texture is negligible (shortcut mass 0.100 → 0.098).
10. **Pre-registered Pet test:** P1, P2 and P4 hold; P3 fails for black deletion.
11. **Supporting study:** the model-randomization sanity check is not shown to agree with the ground truth (mean Kendall τ = −0.20, exact p = 0.84; 0.07, p = 0.44); with five methods only τ ≥ 0.8 is significant for a single seed.

## Tools and Technologies

| Tool | How it is used in the project |
|---|---|
| Python | Main language for training, explanation, evaluation and the web backend |
| PyTorch, torchvision | Building and training the CNNs; loading MNIST, Fashion-MNIST and CIFAR-10 |
| MedMNIST | Loading the PneumoniaMNIST chest X-ray dataset |
| Captum [7] | Computing the five attribution methods on the trained networks |
| NumPy, pandas, SciPy | Metrics, statistics and result tables |
| Matplotlib | Heatmaps, perturbation curves and dose-response figures |
| Flask, HTML, JavaScript | The interactive dashboard |

All tools are free and open source. Experiments run on a laptop CPU, except the ResNet-18 models, which were trained on a free Kaggle GPU.

## Expected Outcomes

- A dose-response protocol for testing XAI evaluation metrics against a controlled, measured change in model behaviour
- Evidence of when common metrics can and cannot be trusted, on natural, medical and colour images
- Controls for explanation size and shape that other evaluation studies can reuse
- 237 trained models, result tables and figures, with code for full reproduction
- An interactive explainability dashboard
- A research paper for a Q1 journal (provisional target: Neurocomputing), and the M.Tech thesis

## Limitations

- One metric family (deletion and insertion); models trained from scratch; the largest network is ResNet-18
- Synthetic shortcuts, which are simpler than real-world shortcuts; the dose varies predictivity, not availability
- The PneumoniaMNIST "object" map is an intensity ranking of the whole X-ray, and the CIFAR-10 object map is a Gaussian prior (Pet is the main natural-image evidence); blur fill leaves 20–25% of the patch intensity
- Reliance and object competence are strongly collinear in most settings
- The inpainting variant approximates ROAD rather than reproducing it exactly

## References

1. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. ICCV.
2. Sundararajan, M., Taly, A., and Yan, Q. (2017). Axiomatic Attribution for Deep Networks. ICML.
3. Simonyan, K., Vedaldi, A., and Zisserman, A. (2014). Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps. ICLR Workshop.
4. Zeiler, M. D., and Fergus, R. (2014). Visualizing and Understanding Convolutional Networks. ECCV.
5. Adebayo, J., et al. (2018). Sanity Checks for Saliency Maps. NeurIPS.
6. Samek, W., et al. (2017). Evaluating the Visualization of What a Deep Neural Network Has Learned. IEEE Transactions on Neural Networks and Learning Systems.
7. Kokhlikyan, N., et al. (2020). Captum: A Unified and Generic Model Interpretability Library for PyTorch. arXiv:2009.07896.
8. Petsiuk, V., Das, A., and Saenko, K. (2018). RISE: Randomized Input Sampling for Explanation of Black-box Models. BMVC.
9. Rong, Y., et al. (2022). A Consistent and Efficient Evaluation Strategy for Attribution Methods (ROAD). ICML.
10. Tomsett, R., et al. (2020). Sanity Checks for Saliency Metrics. AAAI.
11. Binder, A., et al. (2023). Shortcomings of Top-Down Randomization-Based Sanity Checks for Evaluations of Deep Neural Network Explanations. CVPR.
12. Hedström, A., Weber, L., Lapuschkin, S., and Höhne, M. (2024). A Fresh Look at Sanity Checks for Saliency Maps. In: Explainable Artificial Intelligence (xAI 2024), CCIS, Springer, pp. 403–420. doi:10.1007/978-3-031-63787-2_21 (extends "Sanity Checks Revisited", NeurIPS XAIA 2023, arXiv:2401.06465).
13. Hedström, A., et al. (2023). The Meta-Evaluation Problem in Explainable AI: Identifying Reliable Estimators with MetaQuantus. Transactions on Machine Learning Research.
14. Picard, A. M., Boissin, T., Subhash, V., Cadène, R., and Fel, T. (2025). Back to the Baseline: Examining Baseline Effects on Explainability Metrics. arXiv:2512.11433.
15. Ross, A. S., Hughes, M. C., and Doshi-Velez, F. (2017). Right for the Right Reasons: Training Differentiable Models by Constraining their Explanations. IJCAI.
16. Sun, S., et al. (2023). Right for the Wrong Reason: Can Interpretable ML Techniques Detect Spurious Correlations? MICCAI. arXiv:2307.12344.
17. Zhou, Y., et al. (2022). Do Feature Attribution Methods Correctly Attribute Features? AAAI. arXiv:2104.14403.
18. Yang, P., Akhtar, N., Jiang, J., and Mian, A. (2024). A Backdoor-based Explainable AI Benchmark for High Fidelity Evaluation of Attributions (BackX). arXiv:2405.02344.
19. Zhang, Y., Li, Y., Brown, H., Rezaei, M., Bischl, B., Torr, P., Khakzar, A., and Kawaguchi, K. (2023). AttributionLab: Faithfulness of Feature Attribution Under Controllable Environments. NeurIPS XAIA Workshop. arXiv:2310.06514.
20. Yang, J., et al. (2023). MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification. Scientific Data.
21. Smilkov, D., et al. (2017). SmoothGrad: Removing Noise by Adding Noise. arXiv:1706.03825.
22. Yang, M., and Kim, B. (2019). Benchmarking Attribution Methods with Relative Feature Importance (BAM). arXiv:1907.09701.
23. Kim, J. S., Plumb, G., and Talwalkar, A. (2022). Sanity Simulations for Saliency Methods (SMERF). ICML.
24. Blücher, S., Vielhaben, J., and Strodthoff, N. (2024). Decoupling Pixel Flipping and Occlusion Strategy for Consistent XAI Benchmarks. TMLR.
25. Miró-Nicolau, M., Jaume-i-Capó, A., and Moyà-Alcover, G. (2024). A Comprehensive Study on Fidelity Metrics for XAI. arXiv:2401.10640.

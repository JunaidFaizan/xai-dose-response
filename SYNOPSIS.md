# Synopsis

**Project Title:** Do XAI Metrics Notice When a Model Starts Cheating? A Dose-Response Evaluation of Explanation Metrics under Controlled Shortcut Reliance

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

Existing studies test explanation methods against a shortcut of one fixed strength, or test evaluation metrics without any ground truth. It is not known whether evaluation metrics respond correctly as a model's reliance on a shortcut gradually increases. This project addresses the question:

> When a model's reliance on a shortcut increases in controlled steps, do XAI evaluation metrics shift their preference from object explanations to shortcut explanations, and at what level of reliance?

## Novelty

In two literature searches (September–October 2026) we did not find published work that:

1. **varies a model's reliance on a shortcut in controlled steps** (a "dose-response" design) to test the evaluation metrics, rather than the explanation methods;
2. scores **fixed oracle explanations** (shortcut-only, object-only, size-matched and placebo maps), so that the metrics are tested independently of any explanation method;
3. reports when metrics **switch their verdict** relative to the model's measured reliance, and whether a correct ranking also means a correct verdict (calibration).

The closest related work covers parts of this: shortcut or backdoor ground truth for evaluating explanation methods at a fixed strength [16, 17, 18], meta-evaluation of metrics without ground truth [10, 13], hand-designed networks with known explanations [19], and baseline effects in deletion metrics [14]. This claim of novelty is based on our searches and will be confirmed by a full literature search with the guide before submission.

## Objectives

1. Build datasets with injected shortcuts (a local patch and a diffuse texture) whose position or pattern encodes the class.
2. Control and measure how much each trained model relies on the shortcut, from about 0% to about 100%.
3. Test whether four faithfulness metrics follow this change, using fixed oracle explanations and controls for explanation size and shape.
4. Measure how five explanation methods respond to the same change in reliance.
5. Repeat on a natural-image dataset, a medical dataset and a colour dataset with a larger network, with several random seeds.
6. Provide an interactive dashboard for generating and evaluating explanations live.

## Methodology

**Datasets**
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
- Reliance is measured, not assumed: the share of test predictions that follow a shortcut placed at a wrong class's position.

**Models**
- A custom CNN with four convolutional layers for 28×28 images, and a larger CNN with batch normalisation (0.77M parameters) for CIFAR-10
- 156 trained models in total: 12 reliance levels × 3 seeds × 2 shortcuts × 2 datasets, plus 6 levels × 2 seeds on CIFAR-10

**Oracle explanations** (scored by every metric)
- Shortcut map: the shortcut pixels
- Object map: the object's own pixels
- Size-matched object map: the same number of pixels as the shortcut, to control for size
- Placebo map: the shortcut's shape at a position the model does not use, to control for shape

**Evaluation metrics under test**
- Deletion curves [6] with three replacement values: black, blurred and inpainted (ROAD-inspired [9])
- Insertion curve [8] from a blurred image
- Model-randomization sanity check [5] (in the supporting study)

**Scores for each metric**
- Responsiveness: correlation between the model's reliance and the metric's preference for the shortcut map
- Decision accuracy: whether the metric's verdict matches whether the model actually relies on the shortcut
- Switch point: the reliance at which the metric changes its verdict

**Explanation methods**

| Method | Family | Idea |
|---|---|---|
| Saliency [3] | Gradient-based | Gradient of the output with respect to each input pixel |
| SmoothGrad [21] | Gradient-based | Saliency averaged over noisy copies of the input |
| Integrated Gradients [2] | Gradient-based | Gradients accumulated along a path from a blank baseline to the input |
| Grad-CAM [1] | Activation-based | Weights the last convolutional feature maps by their gradients |
| Occlusion [4] | Perturbation-based | Hides image patches and measures the change in prediction |

**Statistics**
- Spearman and Kendall rank correlations, paired Wilcoxon tests with Holm correction, and results over several random seeds

## Preliminary Findings

From the experiments completed so far (the placebo control is still running):

1. **Metric reliability depends on the data.** On MNIST, every metric tracks the model's reliance closely (ρ = 0.91–0.98). On chest X-rays, responsiveness falls to ρ = 0.07–0.84, and some metrics do not respond significantly.
2. **Explanation size distorts the metrics.** When the object explanation is matched in size to the shortcut, the metrics switch their verdict at only 6–15% reliance, instead of 30–42%.
3. **A correct ranking is not a correct verdict.** On CIFAR-10, black and blur deletion rank models correctly (ρ ≈ 0.9), yet prefer the shortcut explanation 74–82% of the time even when the model relies on the shortcut only 4% of the time.
4. **Insertion fails on a diffuse shortcut,** switching only at about 98% reliance.
5. **Inpainting-based deletion is the most robust variant tested.**
6. **Explanation methods differ:** Integrated Gradients follows the shortcut in every setting, while Saliency, SmoothGrad and Grad-CAM move away from a diffuse shortcut as the model relies on it more.
7. **Supporting study:** the model-randomization sanity check ranks explanation methods against the ground truth (Kendall τ = −0.20 to 0.07).

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

All tools are free and open source, and all experiments run on a laptop CPU.

## Expected Outcomes

- A dose-response protocol for testing XAI evaluation metrics against a controlled, measured change in model behaviour
- Evidence of when common metrics can and cannot be trusted, on natural, medical and colour images
- Controls for explanation size and shape that other evaluation studies can reuse
- 156 trained models, result tables and figures, with code for full reproduction
- An interactive explainability dashboard
- A research paper submitted to a peer-reviewed venue, and the M.Tech thesis

## Limitations

- Small images (up to 32×32) and small CNNs
- Synthetic shortcuts, which are simpler than real-world shortcuts
- The 50% switch point and the size-matching rule are modelling choices
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
12. Hedström, A., et al. (2024). A Fresh Look at Sanity Checks for Saliency Maps. arXiv:2405.02383.
13. Hedström, A., et al. (2023). The Meta-Evaluation Problem in Explainable AI: Identifying Reliable Estimators with MetaQuantus. arXiv:2302.07265.
14. Back to the Baseline: Examining Baseline Effects on Explainability Metrics (2025). arXiv:2512.11433.
15. Ross, A. S., Hughes, M. C., and Doshi-Velez, F. (2017). Right for the Right Reasons: Training Differentiable Models by Constraining their Explanations. IJCAI.
16. Sun, S., et al. (2023). Right for the Wrong Reason: Can Interpretable ML Techniques Detect Spurious Correlations? MICCAI. arXiv:2307.12344.
17. Zhou, Y., et al. (2022). Do Feature Attribution Methods Correctly Attribute Features? AAAI. arXiv:2104.14403.
18. A Backdoor-based Explainable AI Benchmark for High Fidelity Evaluation of Attributions (BackX). arXiv:2405.02344.
19. AttributionLab: Faithfulness of Feature Attribution Under Controllable Environments (2023). NeurIPS XAIA Workshop. arXiv:2310.06514.
20. Yang, J., et al. (2023). MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification. Scientific Data.
21. Smilkov, D., et al. (2017). SmoothGrad: Removing Noise by Adding Noise. arXiv:1706.03825.

# Seminar Guide: Do Explanation Metrics Catch Shortcuts?

A Ground-Truth Evaluation of XAI Metrics for CNN Image Classifiers

M.Tech Final Year Project, Explainable AI. This guide covers the whole project: the idea, how every part works, why each design choice was made, the key code, the results, and material for slides, a live demo and questions.

All numbers below come from `results/analysis/summary.md` (5 datasets × 3 seeds, 200 test images per model, 50 for the sanity check).

---

## Contents

1. [The project in one minute](#1-the-project-in-one-minute)
2. [Motivation and research question](#2-motivation-and-research-question)
3. [Contributions](#3-contributions)
4. [End-to-end pipeline](#4-end-to-end-pipeline)
5. [Code map](#5-code-map)
6. [Datasets and the decoy (shortcut) ground truth](#6-datasets-and-the-decoy-shortcut-ground-truth)
7. [The CNN model and training](#7-the-cnn-model-and-training)
8. [The five explanation methods](#8-the-five-explanation-methods)
9. [The evaluation metrics](#9-the-evaluation-metrics)
10. [Evaluation protocol](#10-evaluation-protocol)
11. [Statistical analysis](#11-statistical-analysis)
12. [Results and findings](#12-results-and-findings)
13. [The interactive dashboard](#13-the-interactive-dashboard)
14. [Design decisions and their significance](#14-design-decisions-and-their-significance)
15. [Limitations and future work](#15-limitations-and-future-work)
16. [Suggested slide deck](#16-suggested-slide-deck)
17. [Live demo script](#17-live-demo-script)
18. [Likely questions and answers](#18-likely-questions-and-answers)
19. [How to reproduce](#19-how-to-reproduce)
20. [Glossary](#20-glossary)
21. [References](#21-references)

---

## 1. The project in one minute

Explanation methods (heatmaps such as Saliency or Grad-CAM) are judged using evaluation metrics, for example deletion curves and the model-randomization sanity check. Those metrics are rarely checked themselves, because the true explanation of a real model is unknown.

This project builds cases where the truth is known. Models are trained on "decoy" datasets in which a small 3×3 white patch gives away the label. The models learn to rely on the patch, so a correct explanation must point to it. With this ground truth, five explanation methods and the common metrics are tested on standard data (MNIST, Fashion-MNIST), medical data (PneumoniaMNIST chest X-rays) and two decoy datasets, across 3 training seeds.

Headline findings:

- The popular sanity check rewards the wrong methods. Grad-CAM passes it best but localizes the true evidence worst. Integrated Gradients localizes best but changes least under randomization.
- Deletion metrics track the ground truth better, but which method "wins" depends on how deleted pixels are filled.
- On chest X-rays, the metrics barely separate real methods from a random map.

---

## 2. Motivation and research question

### Why explainability

CNNs are accurate but opaque. In healthcare, finance and safety-critical systems, a prediction must be justified. Regulations such as the EU AI Act push towards transparency. XAI methods answer "which pixels made the model decide this?" with an attribution map (heatmap).

### The problem with heatmaps

- Different methods give different heatmaps for the same prediction.
- A heatmap that looks plausible is not necessarily faithful. Adebayo et al. (2018) showed that some methods produce nearly the same map for a trained and a random model, so they cannot be showing what the model learned.
- So the field uses quantitative metrics. But a metric is only useful if it rewards explanations that are actually correct. That is rarely tested, because the correct explanation is normally unknown.

### Research question

> Do widely used XAI evaluation metrics (perturbation faithfulness and the model-randomization sanity check) agree with a known ground truth about what the model uses?

Sub-questions:

1. Which explanation methods find the true evidence (the shortcut patch)?
2. Does the sanity check rank methods the same way the ground truth does?
3. Do deletion/insertion metrics track the ground truth, and does the pixel replacement value change the verdict?
4. Do these conclusions carry over to medical images?

### How it evolved from the synopsis

The synopsis planned 4 methods on 2 datasets with a method comparison. The final project adds SmoothGrad, PneumoniaMNIST, two decoy datasets with ground truth, three deletion variants plus insertion, 3 seeds and significance tests. The focus moved from "which method is best?" to "can we trust the metrics that decide which method is best?".

---

## 3. Contributions

1. **Ground-truth testbed:** decoy versions of MNIST and PneumoniaMNIST where the evidence the model uses is known by construction and verified (the model follows the patch in 96–99% of conflicting cases).
2. **Metric-level evaluation:** the metrics themselves (not just the methods) are compared against ground-truth localization, at the method level (Kendall τ) and the image level (Spearman ρ).
3. **Replacement-value study:** deletion with black, blurred and inpainted pixels, plus insertion, showing the replacement value changes method rankings.
4. **Medical-data check:** the same analysis on chest X-rays, showing weak separation from random.
5. **Rigor:** 3 seeds, a random-map baseline, paired Wilcoxon tests with Holm correction, and seed-stability analysis.
6. **Interactive dashboard:** a Flask app that computes all five explanations, per-image deletion scores and a live sanity check for drawn, uploaded or test images.

---

## 4. End-to-end pipeline

```
                  ┌────────────────────────────┐
                  │ xai/data.py                │
                  │ MNIST, Fashion-MNIST,      │
                  │ PneumoniaMNIST, Decoy sets │
                  └─────────────┬──────────────┘
                                │ train / val / test splits
                                ▼
┌──────────────────────────────────────────────────────────────┐
│ 1. train_models.py                                           │
│    CNN (xai/models.py) per dataset × seed {42, 1, 2}         │
│    → checkpoint/<dataset>[_seedN].pt + *_metrics.json        │
│    Decoy: also accuracy without patch / with wrong patch     │
└─────────────────────────────┬────────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────┐
│ 2. evaluate_explanations.py                                  │
│    For 200 test images per model:                            │
│      5 methods + random map  (xai/explain.py)                │
│      deletion black/blur/inpaint, insertion blur             │
│      ground-truth mass / rank accuracy (decoy only)          │
│    For 50 images: cascading randomization sanity check       │
│    → results/metrics/<dataset>/seedN_*.csv + plots           │
└─────────────────────────────┬────────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────┐
│ 3. analyze_results.py                                        │
│    Means ± std over seeds, Wilcoxon + Holm,                  │
│    Kendall τ (metric vs metric, metric vs GT, seed vs seed), │
│    Spearman ρ per image vs GT                                │
│    → results/analysis/*.csv, *.png, summary.md               │
└─────────────────────────────┬────────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────┐
│ 4. app.py + templates/ + static/   (dashboard)               │
│    Live explanations, deletion curves, sanity check,         │
│    benchmark table read from results/analysis                │
└──────────────────────────────────────────────────────────────┘

 generate_explanations.py → results/explanations/<dataset>_examples.png
```

---

## 5. Code map

| File | Role |
|---|---|
| `xai/data.py` | Dataset registry, loading, the `DecoyDataset` class and ground-truth masks |
| `xai/models.py` | The CNN (`MnistModel`), checkpoint paths, `load_model` |
| `xai/explain.py` | The 5 explanation methods (Captum) behind a single `explain()` function |
| `xai/metrics.py` | Perturbation curves, inpainting, cascading randomization, rank correlation, localization metrics |
| `xai/preprocess.py` | Converts canvas drawings and uploads into 28×28 model inputs |
| `xai/paths.py` | Project folder paths |
| `train_models.py` | Trains any dataset × seed, saves the best-validation checkpoint and metrics |
| `evaluate_explanations.py` | Runs every metric for every model |
| `analyze_results.py` | Statistics, tables and figures |
| `generate_explanations.py` | Example heatmap grids |
| `app.py`, `templates/index.html`, `static/dashboard.js`, `static/dashboard.css` | Dashboard server and front end |
| `config.py` | Host (127.0.0.1), port (5000), debug flag |
| `test_app.py` | Unit tests for the app |

Legacy files from the B.Tech version (not used): `train.py`, `torch_serve/`, `app.yaml`, `digit_demo.gif`, `checkpoint/mnist_btech.pt`.

---

## 6. Datasets and the decoy (shortcut) ground truth

### 6.1 The five datasets

| Name | Content | Classes | Epochs | Source |
|---|---|---|---|---|
| `mnist` | Handwritten digits | 10 | 10 | torchvision |
| `fashion_mnist` | Clothing | 10 | 15 | torchvision |
| `pneumonia_mnist` | Paediatric chest X-rays (Normal / Pneumonia) | 2 | 20 | MedMNIST v2 |
| `decoy_mnist` | MNIST + class-coded 3×3 white patch | 10 | 5 | built from MNIST |
| `decoy_pneumonia` | PneumoniaMNIST + corner marker | 2 | 20 | built from PneumoniaMNIST |

All images are 1×28×28 with values in [0, 1]. For the torchvision datasets, 5,000 training images are held out as validation with a fixed split seed (42). MedMNIST has official splits.

### 6.2 Why a decoy dataset gives ground truth

In a normal model we do not know which pixels it truly uses, so we cannot say whether a heatmap is correct. A decoy dataset (after Ross et al., 2017) plants a shortcut: a patch whose position encodes the class. If the model is shown to depend on the patch, then the patch pixels are the correct explanation.

Patch positions:

```python
# xai/data.py
# MNIST: along the top and bottom borders, where digits never are.
# Binary tasks: the two top corners, like hospital markers on real X-rays.
DECOY_POSITIONS_10 = [(0, 0), (0, 6), (0, 12), (0, 18), (0, 25),
                      (25, 0), (25, 6), (25, 12), (25, 18), (25, 25)]
DECOY_POSITIONS_2 = [(0, 0), (0, 25)]
```

### 6.3 Forcing the model to use the shortcut

A first attempt simply added the patch. The model mostly ignored it (shortcut reliance 1–17%), because the digit alone already predicts the label perfectly. The fix: in training, swap half the images for an image of a different class while keeping the original label and patch. Now only the patch predicts the label reliably.

```python
# xai/data.py, DecoyDataset.__init__
# Index of the image actually shown for each item.
self.source = np.arange(len(self.labels))
if swap_fraction > 0:
    by_class = [np.flatnonzero(self.labels == c) for c in range(classes)]
    for i in np.flatnonzero(rng.random(len(self.labels)) < swap_fraction):
        other = (self.labels[i] + rng.integers(1, classes)) % classes
        self.source[i] = rng.choice(by_class[other])
```

### 6.4 Test modes and the ground-truth mask

| Mode | Patch position | Used for |
|---|---|---|
| `aligned` | the true class's position | normal accuracy |
| `conflicting` | a different class's position | the ground-truth evaluation |
| `none` | no patch | measuring dependence on the patch |

In `conflicting` mode the image shows one class and the patch points to another. If the model predicts the patch's class, the patch is by construction the evidence for that prediction.

```python
# xai/data.py
if mode == "conflicting":
    # Shift every label by 1..C-1 so the patch always points elsewhere.
    shift = rng.integers(1, classes, size=len(self.labels))
    self.patch_class = (self.labels + shift) % classes
...
def __getitem__(self, i):
    image, _ = self.base[int(self.source[i])]
    label = self.labels[i]
    if self.mode != "none":
        image = image.clone()
        y, x = self.positions[self.patch_class[i]]
        image[0, y:y + PATCH_SIZE, x:x + PATCH_SIZE] = PATCH_VALUE
    return image, _to_int(label)

def mask(self, i) -> np.ndarray:
    """Boolean 28x28 map of the patch pixels (the ground-truth evidence)."""
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
    if self.mode != "none":
        y, x = self.positions[self.patch_class[i]]
        mask[y:y + PATCH_SIZE, x:x + PATCH_SIZE] = True
    return mask
```

### 6.5 Evidence the shortcut is real

| Dataset | Accuracy with patch | Accuracy, patch removed | Shortcut reliance (follows wrong patch) |
|---|---|---|---|
| Decoy MNIST | 99.90% | 12.64% | 96.17% |
| Decoy Pneumonia | 99.73% | 37.71% | 99.47% |

Without the patch, Decoy MNIST falls to near chance (10%). With a misleading patch, the model follows the patch almost always. The patch covers 9 of 784 pixels (1.15%), so a random map puts about 1% of its mass there, which matches the random baseline (0.011).

---

## 7. The CNN model and training

### 7.1 Architecture

The same CNN is used for every dataset (only the output size changes). It is kept from the B.Tech project so old checkpoints still load.

```python
# xai/models.py
class MnistModel(nn.Module):
    def __init__(self, classes: int = 10) -> None:
        super().__init__()
        self.conv1A = nn.Conv2d(1, 16, kernel_size=5)
        self.conv1B = nn.Conv2d(16, 32, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.act = nn.ReLU()
        self.do = nn.Dropout(0.25)
        self.conv2A = nn.Conv2d(32, 64, kernel_size=3)
        self.conv2B = nn.Conv2d(64, 128, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.dense3 = nn.Linear(128 * 3 * 3, 32)
        self.dense4 = nn.Linear(32, classes)

    def forward(self, x):
        x = self.do(self.pool1(self.act(self.conv1B(self.act(self.conv1A(x))))))
        x = self.do(self.pool2(self.act(self.conv2B(self.act(self.conv2A(x))))))
        x = x.view(x.size(0), -1)
        x = self.do(self.act(self.dense3(x)))
        return nn.functional.log_softmax(self.dense4(x), dim=1)
```

| Layer | Output shape | Parameters |
|---|---|---|
| Input | 1×28×28 | – |
| conv1A 5×5, ReLU | 16×24×24 | 416 |
| conv1B 5×5, ReLU | 32×20×20 | 12,832 |
| MaxPool 2, Dropout 0.25 | 32×10×10 | – |
| conv2A 3×3, ReLU | 64×8×8 | 18,496 |
| conv2B 3×3, ReLU | 128×6×6 | 73,856 |
| MaxPool 2, Dropout 0.25 | 128×3×3 | – |
| dense3, ReLU, Dropout | 32 | 36,896 |
| dense4, log-softmax | 10 (or 2) | 330 (or 66) |
| **Total** | | **≈142.8k** (10 classes) |

The 128×6×6 activation before `pool2` is the layer Grad-CAM uses, which is why its maps are 6×6.

### 7.2 Training

- Optimizer Adam, learning rate 1e-3, batch size 128, loss `NLLLoss` (the model outputs log-probabilities)
- The epoch with the best validation accuracy is kept; test accuracy is measured once at the end
- Seeds 42, 1 and 2. Seed 42 is saved as `checkpoint/<dataset>.pt` and used by the dashboard
- Balanced accuracy is also reported, because PneumoniaMNIST is imbalanced

```python
# train_models.py
for epoch in range(1, epochs + 1):
    model.train()
    for images, labels in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()
    val_acc = accuracy(*predict_all(model, val_set, device))
    if val_acc > best_val:
        best_val = val_acc
        torch.save(model.state_dict(), output)
```

For decoy datasets, training also scores the test set with no patch and with a conflicting patch:

```python
"test_accuracy_no_patch": round(accuracy(none_preds, none_labels), 2),
"test_accuracy_conflicting_patch": round(accuracy(conf_preds, conf_labels), 2),
# Share of predictions that follow the patch instead of the image.
"shortcut_reliance": round(accuracy(conf_preds, conflicting.patch_class), 2),
```

### 7.3 Accuracy (mean ± std over 3 seeds)

| Dataset | Test accuracy | Balanced accuracy |
|---|---|---|
| MNIST | 99.35 ± 0.04% | 99.34% |
| Fashion-MNIST | 90.73 ± 0.22% | 90.73% |
| PneumoniaMNIST | 83.97 ± 2.11% | 78.98% |
| Decoy MNIST | 99.90 ± 0.09% | 99.90% |
| Decoy Pneumonia | 99.73 ± 0.27% | 99.65% |

---

## 8. The five explanation methods

All methods are implemented with Captum and exposed through one function. Each returns a 28×28 map where higher means more evidence for the target class. S_c is the log-probability of class c.

```python
# xai/explain.py
METHODS = {
    "saliency": ("Saliency", saliency),
    "smoothgrad": ("SmoothGrad", smoothgrad),
    "integrated_gradients": ("Integrated Gradients", integrated_gradients),
    "grad_cam": ("Grad-CAM", grad_cam),
    "occlusion": ("Occlusion", occlusion),
}

def explain(model, x, target, method):
    model.eval()
    x = x.detach().clone().requires_grad_()
    attribution = METHODS[method][1](model, x, int(target))
    return attribution.detach().cpu().numpy().reshape(x.shape[-2:])
```

| Method | Family | Formula | Settings |
|---|---|---|---|
| Saliency (Simonyan 2014) | Gradient | R_i = \|∂S_c/∂x_i\| | absolute gradient |
| SmoothGrad (Smilkov 2017) | Gradient | R = (1/n) Σ_k \|∇S_c(x + ε_k)\|, ε_k ~ N(0, σ²) | n = 25, σ = 0.15, seeded noise |
| Integrated Gradients (Sundararajan 2017) | Gradient | R_i = (x_i − x'_i) · ∫₀¹ ∂S_c(x' + α(x − x'))/∂x_i dα | black baseline x', 50 steps |
| Grad-CAM (Selvaraju 2017) | Activation | α_k = mean_ij ∂S_c/∂A^k_ij; L = ReLU(Σ_k α_k A^k) | last conv block (6×6), bilinear upsampling |
| Occlusion (Zeiler 2014) | Perturbation | R_i = mean drop in S_c when a patch covering i is set to 0 | 4×4 window, stride 2 |

Integrated Gradients satisfies completeness: Σ R_i = S_c(x) − S_c(x'). Its black baseline means it gives zero attribution to black pixels, which matters for the sanity check and the black-fill deletion metric.

Key implementation details:

```python
def smoothgrad(model, x, target):
    # Noise is seeded so the same input always gets the same map.
    state = torch.get_rng_state()
    torch.manual_seed(0)
    try:
        return NoiseTunnel(Saliency(model)).attribute(
            x, nt_type="smoothgrad", nt_samples=25, nt_samples_batch_size=25,
            stdevs=0.15, target=target, abs=True)
    finally:
        torch.set_rng_state(state)

def integrated_gradients(model, x, target):
    baseline = torch.zeros_like(x)
    return IntegratedGradients(model).attribute(
        x, baselines=baseline, target=target, n_steps=50)

def grad_cam(model, x, target):
    # The model reuses one ReLU module, so it cannot be hooked directly.
    # The input of pool2 is the post-ReLU output of the last conv layer.
    cam = LayerGradCam(model, model.pool2).attribute(
        x, target=target, relu_attributions=True, attribute_to_layer_input=True)
    return LayerAttribution.interpolate(cam, x.shape[-2:], interpolate_mode="bilinear")

def occlusion(model, x, target):
    return Occlusion(model).attribute(
        x, target=target, sliding_window_shapes=(1, 4, 4),
        strides=(1, 2, 2), baselines=0, perturbations_per_eval=64)
```

A subtle point worth mentioning: the model applies one shared `nn.ReLU` module in several places, so hooking "the last ReLU" would hook all of them. Hooking the input of `pool2` gets exactly the post-ReLU activation of `conv2B`.

For display, maps are normalized to their positive evidence:

```python
def normalize(attribution):
    positive = np.clip(attribution, 0, None)
    peak = positive.max()
    return positive / peak if peak > 0 else positive
```

---

## 9. The evaluation metrics

### 9.1 Perturbation faithfulness (deletion and insertion)

Idea (Samek 2017, Petsiuk 2018): if an explanation is faithful, removing its most important pixels first should make the model's confidence drop quickly.

- **Deletion:** rank pixels by attribution, replace the top 0%, 5%, …, 100% (20 steps), record p(c) each time. Area under the curve (AUC): lower is better.
- **Insertion:** start from a blurred image and restore pixels most-important-first. AUC: higher is better.

```
p(c)                          Deletion
1.0 ┤●
    │ ●  ●
    │       ●  ●  ●  ●                      ← random order: slow drop, large AUC
    │ ●                 ●  ●  ●
    │   ●                         ●  ●
    │     ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●   ← faithful map: fast drop, small AUC
0.0 ┼──────────────────────────────────────
    0%          fraction removed        100%
```

How removed pixels are filled is a known confound. Three variants:

| Variant | Replacement | Why |
|---|---|---|
| `zero` | black | the classic test; but black is also the IG and Occlusion baseline, which may favor them |
| `blur` | Gaussian-blurred copy (kernel 25, σ = 4) | a more neutral value; σ = 4 makes a fully blurred image give near-chance confidence (σ = 2 still left 63% on MNIST) |
| `inpaint` | neighbour averaging + noise (ROAD-inspired) | avoids leaking the shape of the removed region (Rong 2022) |

Core code:

```python
# xai/metrics.py
def _ranking(attribution, rng):
    """Pixel indices from most to least important, ties broken randomly."""
    flat = np.asarray(attribution).reshape(-1)
    perm = rng.permutation(flat.size)
    return perm[np.argsort(-flat[perm], kind="stable")]

def perturbation_curve(model, x, target, attribution, mode="deletion", fill="zero",
                       steps=20, rng=None):
    rng = rng or np.random.default_rng(0)
    masks = _step_masks(_ranking(attribution, rng), x.shape, steps)
    original = x.detach().expand(steps + 1, -1, -1, -1)
    if fill == "inpaint":
        images = inpaint(original.clone(), masks)
    else:
        reference = torch.zeros_like(original) if fill == "zero" \
                    else blur(x.detach()).expand_as(original)
        images = torch.where(masks, reference, original) if mode == "deletion" \
                 else torch.where(masks, original, reference)
    with torch.no_grad():
        probs = torch.exp(model(images))[:, target]
    return probs.cpu().numpy()

def area_under_curve(curve):
    return float(np.trapezoid(curve, dx=1.0 / (len(curve) - 1)))
```

All 21 perturbed images are built at once and scored in a single batched forward pass. Random tie-breaking matters: Grad-CAM and IG produce many exactly-zero pixels, and a fixed tie order would bias the curve.

Inpainting:

```python
def inpaint(images, masks, iterations=100, noise=0.05, seed=0):
    """Fill masked pixels by repeatedly averaging their 4 neighbours, then add noise."""
    kernel = torch.tensor([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]]).view(1, 1, 3, 3) / 4
    known = ~masks
    count = known.sum(dim=(1, 2, 3), keepdim=True).clamp(min=1)
    mean = (images * known).sum(dim=(1, 2, 3), keepdim=True) / count
    filled = torch.where(masks, mean.expand_as(images), images)
    for _ in range(iterations):
        average = F.conv2d(F.pad(filled, (1, 1, 1, 1), mode="replicate"), kernel)
        filled = torch.where(masks, average, images)
    jitter = torch.randn(images.shape, generator=torch.Generator().manual_seed(seed)) * noise
    return torch.where(masks, (filled + jitter).clamp(0, 1), images)
```

This is a heat-diffusion style fill: known pixels stay fixed and masked pixels converge to a smooth interpolation of their surroundings.

### 9.2 Sanity check: cascading model randomization

Idea (Adebayo 2018): an explanation that reflects what the model learned should change when the learned weights are destroyed. Layers are re-initialized one by one from the output towards the input:

```
original → dense4 → dense3 → conv2B → conv2A → conv1B → conv1A (fully random)
```

At each stage the explanation for the original prediction is recomputed and compared with the original map using Spearman rank correlation. Low correlation means the method "passes".

```python
RANDOMIZATION_ORDER = ["dense4", "dense3", "conv2B", "conv2A", "conv1B", "conv1A"]

def cascading_randomization(model, seed=0):
    randomized = copy.deepcopy(model)
    randomized.eval()
    generator_state = torch.get_rng_state()
    torch.manual_seed(seed)
    try:
        yield "original", randomized
        for name in RANDOMIZATION_ORDER:
            getattr(randomized, name).reset_parameters()
            yield name, randomized
    finally:
        torch.set_rng_state(generator_state)

def rank_correlation(a, b, mask=None):
    a, b = np.abs(a), np.abs(b)
    if mask is not None:
        a, b = a[mask], b[mask]
    ra = pd.Series(a.ravel()).rank().to_numpy()
    rb = pd.Series(b.ravel()).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])
```

Correlation is reported over all pixels and over object pixels only (`image > 0`). The object-pixel version matters: IG with a black baseline is zero on black background for any model, which would inflate the all-pixel correlation regardless of what the model learned.

### 9.3 Ground-truth localization (decoy datasets only)

Following Arras et al. (2022):

- **Relevance mass accuracy:** share of positive attribution that falls inside the patch. Σ_{i∈patch} R_i⁺ / Σ_i R_i⁺
- **Relevance rank accuracy:** share of the top-k pixels that lie inside the patch, with k = patch size (9)

```python
def relevance_mass_accuracy(attribution, mask):
    positive = np.clip(attribution, 0, None)
    total = positive.sum()
    return float(positive[mask].sum() / total) if total > 0 else float("nan")

def relevance_rank_accuracy(attribution, mask, rng=None):
    rng = rng or np.random.default_rng(0)
    k = int(mask.sum())
    top = _ranking(attribution, rng)[:k]
    return float(mask.reshape(-1)[top].mean())
```

These are the reference against which every other metric is judged.

---

## 10. Evaluation protocol

For each dataset × seed (`evaluate_explanations.py`):

1. Load the model and test set. Decoy test sets use `conflicting` mode.
2. Pick 200 random test images with a fixed seed, the same images for every model of a dataset. For decoy datasets, keep only images where the model follows the patch, so the patch is the true evidence.
3. For each image: predict class c, compute the 5 maps plus a random map, then 4 faithfulness AUCs and (decoy) mass/rank accuracy.
4. For the first 50 images: run the cascading randomization and record correlations at every stage. The explanation always targets the original prediction, so only the model changes.
5. Save per-image CSVs; for seed 42 also plots (`perturbation_curves.png`, `sanity_curve.png`, `sanity_examples.png`).

```python
# evaluate_explanations.py
def test_indices(test_set, samples, model=None):
    generator = torch.Generator().manual_seed(IMAGE_SEED)
    order = torch.randperm(len(test_set), generator=generator).tolist()
    if model is None:
        return order[:samples]
    with torch.no_grad():
        preds = torch.cat([model(images).argmax(dim=1)
                           for images, _ in DataLoader(test_set, batch_size=512)]).numpy()
    follows = preds == test_set.patch_class
    return [i for i in order if follows[i]][:samples]
```

The random map is a key control. A metric that cannot separate real methods from random noise is not informative.

---

## 11. Statistical analysis

`analyze_results.py` turns per-image results into claims:

| Analysis | Question answered | Method |
|---|---|---|
| Summary tables | How does each method score? | mean over seeds of per-seed means, std over seeds |
| Significance | Are two methods really different? | paired Wilcoxon signed-rank test per method pair, Holm–Bonferroni correction |
| Metric agreement | Do metrics rank methods the same way? | Kendall τ between method rankings, averaged over datasets and seeds |
| Ground-truth agreement (method level) | Does a metric rank methods like the ground truth? | Kendall τ between metric ranking and mass-accuracy ranking |
| Ground-truth agreement (image level) | Does a better score go with better localization per image? | Spearman ρ over all (image, method) pairs |
| Seed stability | Is a metric's ranking reproducible? | Kendall τ between rankings from different seeds |

All criteria are oriented so higher means better before ranking (deletion and sanity scores are negated):

```python
def goodness(values, higher):
    """Orient scores so that higher always means better."""
    return values if higher else -values
```

Holm correction:

```python
def holm(p_values):
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adjusted[i] = running
    return adjusted
```

Kendall τ = (concordant − discordant pairs) / total pairs, from −1 (reversed) to +1 (identical). With 5 methods there are 10 pairs, so τ moves in steps of 0.2. That coarseness is why per-image Spearman ρ (about 3,000 pairs) is reported too.

---

## 12. Results and findings

### 12.1 Ground truth on Decoy MNIST

| Method | Mass accuracy ↑ | Rank accuracy ↑ | Deletion black ↓ | Deletion blur ↓ | Sanity corr. ↓ |
|---|---|---|---|---|---|
| Saliency | 0.096 | 0.275 | 0.159 | 0.205 | −0.043 |
| SmoothGrad | 0.095 | 0.314 | 0.149 | 0.192 | −0.053 |
| **Integrated Gradients** | **0.709** | **0.800** | 0.103 | 0.146 | 0.265 |
| Grad-CAM | 0.043 | 0.101 | 0.448 | 0.489 | 0.008 |
| **Occlusion** | 0.539 | 0.650 | **0.091** | **0.126** | −0.208 |
| Random | 0.011 | 0.013 | 0.591 | 0.686 | – |

### Finding 1: methods differ widely in finding the true evidence

Integrated Gradients puts 71% of its positive attribution on a 9-pixel patch (1% of the image) and Occlusion 54%. Saliency, SmoothGrad and Grad-CAM stay at or below 10%. Popular, good-looking methods can miss the evidence the model actually uses.

Why: Saliency and SmoothGrad measure local sensitivity, which spreads over the digit's strokes. Grad-CAM's 6×6 resolution blurs a 3×3 border patch over a large area. IG and Occlusion measure the actual contribution relative to a black reference, and the patch is the brightest, most decisive region.

### Finding 2: the sanity check rewards the wrong methods

Grad-CAM changes most under randomization ("passes" best) but localizes the shortcut worst. IG changes least ("fails") but localizes best.

| Metric vs ground truth | Decoy MNIST τ | Decoy Pneumonia τ |
|---|---|---|
| Sanity (object px) | −0.20 | 0.07 |
| Deletion (black) | 0.80 | −0.33 |
| Deletion (blur) | 0.87 | −0.33 |
| Deletion (inpaint) | 0.67 | 0.33 |
| Insertion (blur) | 0.60 | 0.00 |

Passing the sanity check is a necessary property (the map depends on the weights), not proof of correctness. IG multiplies gradients by the input, so its map keeps the input's structure even for a random model, but for the trained model its gradients do point at the patch. This agrees with the critique by Binder et al. (2023). Per image the evidence is mixed (ρ = −0.08 on Decoy MNIST, 0.61 on Decoy Pneumonia).

### Finding 3: perturbation metrics track the ground truth, but not everywhere

Per image, deletion scores correlate strongly with correct localization:

| Per-image Spearman ρ with ground truth | Decoy MNIST | Decoy Pneumonia |
|---|---|---|
| Deletion (black) | 0.765 | 0.648 |
| Deletion (blur) | 0.732 | 0.743 |
| Deletion (inpaint) | 0.668 | 0.762 |
| Insertion (blur) | 0.394 | 0.357 |
| Sanity (object px) | −0.079 | 0.614 |

At the method level, rankings match the ground truth on Decoy MNIST (τ = 0.60–0.87) but not on Decoy Pneumonia (τ = −0.33 to 0.33). The patch there sits on textured X-ray background, and with 2 classes the evidence is more spread out.

### Finding 4: the replacement value decides the winner

MNIST deletion AUC (lower is better, rank in brackets):

| Method | Black | Blur | Inpaint |
|---|---|---|---|
| Saliency | 0.274 (4) | 0.341 (2) | 0.416 (4) |
| SmoothGrad | 0.247 (3) | 0.306 (1) | 0.328 (2) |
| Integrated Gradients | **0.088 (1)** | 0.364 (3) | 0.333 (3) |
| Grad-CAM | 0.436 (5) | 0.580 (5) | 0.569 (5) |
| Occlusion | 0.145 (2) | 0.403 (4) | **0.266 (1)** |

With black pixels, IG and Occlusion rank 1st or 2nd on all three standard datasets. Both use black as their own reference image, so black-fill deletion partly measures agreement with the method's own baseline. With blur, IG drops to 3rd–4th and Occlusion to 4th–5th. Black-fill deletion agrees only weakly with the other variants (mean τ ≈ 0.17–0.21), while blur, inpaint and insertion agree more with each other (τ ≈ 0.43–0.49).

Agreement between criteria (mean Kendall τ):

| | Del. black | Del. blur | Del. inpaint | Ins. blur | Sanity |
|---|---|---|---|---|---|
| Del. black | 1.00 | 0.17 | 0.21 | 0.20 | −0.08 |
| Del. blur | 0.17 | 1.00 | 0.43 | 0.49 | −0.21 |
| Del. inpaint | 0.21 | 0.43 | 1.00 | 0.48 | −0.07 |
| Ins. blur | 0.20 | 0.49 | 0.48 | 1.00 | −0.35 |
| Sanity | −0.08 | −0.21 | −0.07 | −0.35 | 1.00 |

The sanity check disagrees with every faithfulness metric.

### Finding 5: on chest X-rays the metrics barely separate methods from random

PneumoniaMNIST, deletion with blur: methods score 0.774–0.847 against 0.828 for random. Grad-CAM (0.847) and Occlusion (0.826) are no better than random. Rankings are unstable across seeds (τ = 0.07 for deletion blur). Conclusions drawn from these metrics on medical images need extra caution.

### Supporting figures

| Figure | File |
|---|---|
| Example explanations per dataset | `results/explanations/<dataset>_examples.png` |
| Mean perturbation curves | `results/metrics/<dataset>/perturbation_curves.png` |
| Sanity-check correlation vs randomized layers | `results/metrics/<dataset>/sanity_curve.png` |
| Heatmaps as layers are randomized | `results/metrics/<dataset>/sanity_examples.png` |
| Method rank under each criterion | `results/analysis/method_ranks.png` |
| Criterion agreement matrix | `results/analysis/metric_agreement.png` |
| Metrics vs ground truth | `results/analysis/gt_agreement.png` |

### Practical recommendation

- Don't use the sanity check alone as proof that an explanation is correct.
- Prefer deletion with a neutral replacement (blur or inpaint), and report several variants.
- Always include a random-map baseline and multiple seeds.
- Where possible, validate on a known shortcut before trusting a metric on a new domain.

---

## 13. The interactive dashboard

### 13.1 Architecture

```
Browser (templates/index.html + static/dashboard.js)
  │  draw on canvas / upload / pick random test image
  │  POST /api/explain  {"dataset": "...", "image": "data:image/png;base64,..."}
  │                  or {"dataset": "...", "sample_index": 123}
  ▼
Flask (app.py, 127.0.0.1:5000)
  read_input()  → xai/preprocess.py (uploads) or test set (sample_index)
  get_model()   → cached per dataset
  LOCK          → one explanation job at a time (Captum adds hooks)
  predict → 5 × explain() → 5 × deletion_curve() + random baseline
  ▼
JSON: input grid, prediction, probabilities, heatmaps, deletion AUCs and curves
  ▼
Browser renders heatmap overlays (jet colormap), probability bars, deletion chart

POST /api/sanity           → maps from a fully randomized model + correlation
GET  /api/benchmark/<ds>   → results/analysis/<ds>_summary.csv as a table
```

### 13.2 Preprocessing user images

Models were trained on a bright object on black, centered, values in [0, 1]. User input is converted to match:

1. Take the object as bright pixels: the alpha channel for the transparent drawing canvas, otherwise grayscale, inverted if the border is light (a dark digit on paper).
2. Crop to the object's bounding box.
3. Scale the longest side to the training object size (20 px for MNIST, which fits digits in a 20×20 box; 28 px for others) and center it in 28×28.
4. Divide by 255.

```python
# xai/preprocess.py
def image_to_tensor(image, dataset):
    frame = fit_to_frame(foreground(image), info(dataset)["fit_size"])
    return torch.from_numpy(frame / 255.0).float().reshape(1, 1, 28, 28)
```

Uploads are validated: base64 decoding, a 4096×4096 pixel cap (decompression-bomb protection), a 5 MB request limit, and a clear error for empty images.

### 13.3 The explain endpoint

```python
# app.py
@app.post("/api/explain")
def api_explain():
    dataset, x, true_label = read_input()
    with LOCK:
        model = get_model(dataset)
        pred, probs = predict(model, x)
        rng = np.random.default_rng(0)
        explanations = []
        for name, (title, _) in METHODS.items():
            attribution = explain(model, x, pred, name)
            curve = deletion_curve(model, x, pred, attribution, steps=20, rng=rng)
            explanations.append({
                "method": name, "title": title,
                "heatmap": rounded(normalize(attribution)),
                "deletion_auc": round(area_under_curve(curve), 3),
                "deletion_curve": rounded(curve),
            })
        random_curve = np.mean([
            deletion_curve(model, x, pred, rng.random(x.shape[-2:]), steps=20, rng=rng)
            for _ in range(5)], axis=0)
    ...
```

The dashboard reuses exactly the same `explain()` and `deletion_curve()` functions as the offline evaluation, so live scores and paper numbers come from the same code.

### 13.4 The live sanity check

```python
@app.post("/api/sanity")
def api_sanity():
    dataset, x, _ = read_input()
    mask = x[0, 0].numpy() > 0  # object pixels
    with LOCK:
        model = get_model(dataset)
        randomized = get_random_model(dataset)   # all layers re-initialized, cached
        pred, _ = predict(model, x)
        for name, (title, _) in METHODS.items():
            trained_map = explain(model, x, pred, name)
            random_map = explain(randomized, x, pred, name)
            correlation = rank_correlation(trained_map, random_map, mask)
            ...
```

The front end labels |ρ| < 0.5 as "changed: passes" and otherwise "similar: fails". The 0.5 cut-off is only a visual guide; the thesis reports raw values.

### 13.5 Security notes

- Binds to 127.0.0.1, so it is reachable only from the local machine. There is no authentication, so do not expose it on a network as-is.
- Debug mode is off by default (`DEBUG_MODE=0`).
- Benchmark HTML is escaped in `dashboard.js` (`escapeHtml`).

---

## 14. Design decisions and their significance

| Decision | Why it matters |
|---|---|
| Decoy datasets with a verified shortcut | Turns "is this heatmap right?" from opinion into a measurable fact |
| Swapping 50% of training images | Without it the model ignored the patch (1–17% reliance); with it, 96–99% |
| Evaluating only images where the model follows a conflicting patch | Guarantees the patch is the evidence for that specific prediction |
| Three deletion fills plus insertion | Exposes the replacement-value confound instead of hiding it |
| Blur σ = 4 chosen by a confidence check | Blurred pixels carry little class information, so blur is a near-neutral fill |
| Object-pixel sanity correlation | Removes the trivial zero-on-black agreement of IG |
| Random tie-breaking in pixel ranking | Prevents methods with many zero pixels from getting an arbitrary advantage |
| Random-map baseline | Shows whether a metric separates real methods from noise at all |
| 3 seeds, Wilcoxon + Holm, Kendall τ, Spearman ρ | Claims are about reproducible, significant differences, not one lucky run |
| Same code for dashboard and evaluation | The demo shows exactly what the paper measures |
| Seeded SmoothGrad noise and fixed image seeds | Every run reproduces the same numbers |
| Small CNN on 28×28 images, CPU-only | Full pipeline reproducible on a laptop (about 90 minutes) |

---

## 15. Limitations and future work

Limitations:

- Small CNNs on 28×28 grayscale images
- A synthetic 3×3 shortcut, simpler than real-world shortcuts such as hospital text markers
- 3 seeds and 5 methods, so method-level Kendall τ is coarse (steps of 0.2)
- The inpainting is ROAD-inspired, not the exact ROAD procedure
- Blur strength was chosen by a simple confidence check
- PneumoniaMNIST results are a research study, not a diagnostic tool

Future work:

- Larger models (ResNet, ViT) and higher-resolution images
- Realistic shortcuts (text markers, watermarks, background bias) and naturally occurring spurious features
- More methods (LRP, DeepLIFT, SHAP, RISE) and more metrics (ROAD proper, infidelity, sensitivity)
- Multiple shortcuts of different strength, to test whether metrics detect partial reliance
- Using the ground-truth testbed to design or calibrate better metrics

---

## 16. Suggested slide deck

About 15–18 slides for a 15–20 minute talk.

1. **Title** – project title, name, guide, institution
2. **Motivation** – black-box CNNs, need for explanations, EU AI Act
3. **What an explanation looks like** – one image with 5 heatmaps (`results/explanations/mnist_examples.png`)
4. **The problem** – methods disagree; plausible ≠ faithful; metrics are unverified
5. **Research question** – "Do XAI metrics agree with a known ground truth?"
6. **Key idea: decoy datasets** – patch positions, swapping, conflicting test mode (diagram from Section 6)
7. **Proof the shortcut is real** – table from 6.5
8. **Pipeline** – diagram from Section 4
9. **Model and methods** – CNN table and the 5-method table
10. **Metrics** – deletion curve sketch, three fills, sanity check, mass/rank accuracy
11. **Protocol and statistics** – 5 datasets × 3 seeds × 200 images, Wilcoxon, Holm, τ, ρ
12. **Finding 1** – Decoy MNIST ground-truth table (`results/explanations/decoy_mnist_examples.png`)
13. **Finding 2** – sanity check vs ground truth (`results/analysis/gt_agreement.png`)
14. **Finding 3 and 4** – replacement value changes the winner (`results/analysis/method_ranks.png`, `metric_agreement.png`)
15. **Finding 5** – medical images close to random
16. **Dashboard** – screenshot or live demo
17. **Recommendations, limitations, future work**
18. **Conclusion and references**

Closing line suggestion: "Before trusting a heatmap, check the metric that says the heatmap is good."

---

## 17. Live demo script

Start the dashboard from the project root:

```powershell
.\.venv\Scripts\Activate.ps1
python app.py
```

Open http://127.0.0.1:5000/. Test it the day before and keep screenshots as a backup.

1. **MNIST:** draw a digit. Show the prediction, the five heatmaps, the per-method deletion AUC and the deletion chart against the random baseline.
2. **Decoy MNIST:** click "Random test image". Point out that the patch sits at the wrong class's position and the model predicts the patch's class. IG and Occlusion highlight the patch; Saliency and Grad-CAM mostly don't.
3. Click **Run sanity check**. Grad-CAM "passes" and IG "fails": Finding 2, live.
4. **PneumoniaMNIST:** show a test X-ray. Present it as a research demo, not a diagnosis.
5. Scroll to the benchmark table: the averages over 3 seeds.

---

## 18. Likely questions and answers

More detail is in `docs/VIVA_PREP.md`.

**Why trust the patch as ground truth?**
The Decoy MNIST model gets 99.9% with the patch and about 13% without it, follows a misleading patch 96% of the time, and only images where it followed the misleading patch are evaluated. For those, nothing but the patch points to the predicted class.

**Isn't a synthetic patch unrealistic?**
It is simpler than real shortcuts. The goal is a known ground truth. The Decoy Pneumonia corner marker mimics hospital markers on X-rays. Realistic shortcuts are future work.

**Why does IG fail the sanity check but localize best?**
IG multiplies gradients by the input, so even for a random model its map keeps the input's structure and correlates with the original. Sensitivity to randomization and correctness are different properties.

**Why does Grad-CAM do so badly?**
Its map is 6×6, so a 3×3 border patch gets smeared. It depends mostly on top-layer weights, so it changes as soon as the output layer is randomized, which is why it "passes".

**Why three deletion variants?**
The replacement value is a known confound (ROAD). With black, IG and Occlusion win because black is their own baseline. With blur or inpainting they drop.

**Why is Decoy Pneumonia different?**
The patch sits on textured background rather than black, and with 2 classes evidence is more spread out. Rankings are noisier, which is itself a warning for medical use.

**Which metric do you recommend?**
Deletion with a neutral fill (blur or inpaint), reported with other variants, a random baseline and several seeds. Never the sanity check alone.

**What is new compared with existing work?**
Existing work either criticises metrics theoretically or evaluates methods against ground truth. This project evaluates the metrics themselves against ground truth, on natural and medical data, with seeds and significance tests.

**Why not a bigger model?**
To make a full, multi-seed, statistically tested study reproducible on a CPU. The methodology transfers to larger models.

---

## 19. How to reproduce

Windows, Python 3.12, CPU only.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

python train_models.py --dataset all --seeds 42 1 2 --force   # about 70 min
python evaluate_explanations.py                                # about 20 min
python analyze_results.py                                      # under 1 min
foreach ($d in 'mnist','fashion_mnist','pneumonia_mnist','decoy_mnist','decoy_pneumonia') { python generate_explanations.py --dataset $d }

python app.py                                                  # dashboard
python -m unittest test_app.py                                 # tests
```

Datasets download to `data/` on first use.

---

## 20. Glossary

| Term | Meaning |
|---|---|
| Attribution map / heatmap | Per-pixel score of how much each pixel supports the predicted class |
| Faithfulness | Whether an explanation reflects what the model actually uses |
| Shortcut / spurious feature | A feature correlated with the label that the model uses instead of the real signal |
| Decoy dataset | A dataset with a planted shortcut, giving a known ground-truth explanation |
| Deletion / insertion AUC | Area under the confidence curve as important pixels are removed / restored |
| Baseline (IG, Occlusion) | The reference input that represents "no information" (black here) |
| Cascading randomization | Re-initializing layers from output to input to test explanation sensitivity |
| Relevance mass / rank accuracy | Share of attribution / of top-k pixels inside the ground-truth region |
| Kendall τ | Agreement between two rankings, from −1 to 1 |
| Spearman ρ | Rank correlation between two sets of values |
| Wilcoxon signed-rank test | Non-parametric paired test for differences between two methods on the same images |
| Holm correction | Adjusts p-values for many simultaneous comparisons |
| Captum | PyTorch library for model interpretability |

---

## 21. References

1. Adebayo, J., et al. (2018). Sanity Checks for Saliency Maps. NeurIPS.
2. Adebayo, J., et al. (2022). Post hoc Explanations may be Ineffective for Detecting Unknown Spurious Correlation. ICLR.
3. Arras, L., Osman, A., and Samek, W. (2022). CLEVR-XAI: A Benchmark Dataset for the Ground Truth Evaluation of Neural Network Explanations. Information Fusion.
4. Binder, A., et al. (2023). Shortcomings of Top-Down Randomization-Based Sanity Checks for Evaluations of Deep Neural Network Explanations. CVPR.
5. Hedström, A., et al. (2024). A Fresh Look at Sanity Checks for Saliency Maps. arXiv:2405.02383.
6. Kermany, D. S., et al. (2018). Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning. Cell.
7. Kokhlikyan, N., et al. (2020). Captum: A Unified and Generic Model Interpretability Library for PyTorch. arXiv:2009.07896.
8. Petsiuk, V., Das, A., and Saenko, K. (2018). RISE: Randomized Input Sampling for Explanation of Black-box Models. BMVC.
9. Rong, Y., et al. (2022). A Consistent and Efficient Evaluation Strategy for Attribution Methods (ROAD). ICML.
10. Ross, A. S., Hughes, M. C., and Doshi-Velez, F. (2017). Right for the Right Reasons. IJCAI.
11. Samek, W., et al. (2017). Evaluating the Visualization of What a Deep Neural Network Has Learned. IEEE TNNLS.
12. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. ICCV.
13. Simonyan, K., Vedaldi, A., and Zisserman, A. (2014). Deep Inside Convolutional Networks. ICLR Workshop.
14. Smilkov, D., et al. (2017). SmoothGrad: Removing Noise by Adding Noise. arXiv:1706.03825.
15. Sundararajan, M., Taly, A., and Yan, Q. (2017). Axiomatic Attribution for Deep Networks. ICML.
16. Yang, J., et al. (2023). MedMNIST v2. Scientific Data.
17. Zeiler, M. D., and Fergus, R. (2014). Visualizing and Understanding Convolutional Networks. ECCV.

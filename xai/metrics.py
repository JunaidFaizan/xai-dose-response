"""Quantitative evaluation of explanations.

1. Faithfulness: perturbation curves (Samek et al., 2017; Petsiuk et al., 2018)
   Deletion: pixels are replaced, most important first, and the model's
   probability for the explained class is recorded after each step. A fast
   drop (LOW area under the curve) means a faithful explanation.
   Insertion: start from a blurred image and restore pixels, most important
   first. A fast rise (HIGH area) means a faithful explanation.

   How removed pixels are filled matters, so three variants are provided:
     zero     black pixels (the classic deletion test; also the Integrated
              Gradients baseline, which may favour that method)
     blur     pixels from a Gaussian-blurred copy of the image
     inpaint  pixels estimated from their neighbours plus a little noise,
              inspired by ROAD (Rong et al., 2022), which showed that the
              shape of the removed region can leak class information

2. Sanity check: cascading model randomization (Adebayo et al., 2018)
   Layers are re-initialised from the output layer down. An explanation that
   depends on what the model learned should change, so its Spearman rank
   correlation with the original explanation should FALL towards 0. It is
   computed over all pixels and over object pixels only.

3. Localization against ground truth (for decoy datasets)
   The model is known to rely on a patch, so a correct explanation puts its
   evidence there (Arras et al., 2022):
     mass accuracy  share of positive attribution inside the patch
     rank accuracy  share of the top-k pixels inside the patch (k = patch size)
"""

import copy

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur

# Layers randomized in cascade, from the output layer to the input layer.
RANDOMIZATION_ORDER = ["dense4", "dense3", "conv2B", "conv2A", "conv1B", "conv1A"]

# name: (title, mode, fill, higher_is_better)
FAITHFULNESS_METRICS = {
    "deletion_zero": ("Deletion (black)", "deletion", "zero", False),
    "deletion_blur": ("Deletion (blur)", "deletion", "blur", False),
    "deletion_inpaint": ("Deletion (inpaint)", "deletion", "inpaint", False),
    "insertion_blur": ("Insertion (blur)", "insertion", "blur", True),
}

# sigma 4 was chosen so a fully blurred image gives near-chance confidence
# (checked on MNIST and Fashion-MNIST); sigma 2 still kept ~63% on MNIST.
BLUR_KERNEL, BLUR_SIGMA = 25, 4.0
INPAINT_ITERATIONS, INPAINT_NOISE = 100, 0.05


def _ranking(attribution, rng):
    """Pixel indices from most to least important, ties broken randomly."""
    flat = np.asarray(attribution).reshape(-1)
    perm = rng.permutation(flat.size)
    return perm[np.argsort(-flat[perm], kind="stable")]


def _step_masks(order, shape, steps):
    """(steps+1, 1, H, W) masks; step i covers the top i/steps of pixels."""
    n = int(np.prod(shape[-2:]))
    masks = torch.zeros(steps + 1, n, dtype=torch.bool)
    order = torch.from_numpy(order.copy())
    for i in range(1, steps + 1):
        masks[i, order[:round(n * i / steps)]] = True
    return masks.view(steps + 1, 1, *shape[-2:])


def blur(x, kernel=BLUR_KERNEL, sigma=BLUR_SIGMA):
    return gaussian_blur(x, kernel_size=[kernel] * 2, sigma=[sigma] * 2)


def inpaint(images, masks, iterations=INPAINT_ITERATIONS, noise=INPAINT_NOISE, seed=0):
    """Fill masked pixels by repeatedly averaging their 4 neighbours, then add noise."""
    channels = images.shape[1]
    kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3) / 4
    kernel = kernel.repeat(channels, 1, 1, 1).to(images.device)  # one kernel per channel
    known = ~masks
    count = known.sum(dim=(2, 3), keepdim=True).clamp(min=1)
    mean = (images * known).sum(dim=(2, 3), keepdim=True) / count  # per-channel mean
    filled = torch.where(masks, mean.expand_as(images), images)
    for _ in range(iterations):
        average = F.conv2d(F.pad(filled, (1, 1, 1, 1), mode="replicate"), kernel, groups=channels)
        filled = torch.where(masks, average, images)
    # Noise is drawn on the CPU so the same seed gives the same noise on any device.
    generator = torch.Generator().manual_seed(seed)
    jitter = (torch.randn(images.shape, generator=generator) * noise).to(images.device)
    return torch.where(masks, (filled + jitter).clamp(0, 1), images)


def perturbation_curve(model, x, target, attribution, mode="deletion", fill="zero",
                       steps=20, rng=None, blur_kernel=BLUR_KERNEL, blur_sigma=BLUR_SIGMA,
                       inpaint_iterations=INPAINT_ITERATIONS):
    """Probability of `target` as pixels are removed (or inserted), most important first.

    Args:
        model: classifier returning log-probabilities.
        x: input of shape (1, C, H, W) with values in [0, 1].
        target: class whose probability is tracked.
        attribution: (H, W) map; higher means more important.
        mode: "deletion" or "insertion".
        fill: "zero", "blur" or "inpaint" (inpaint is deletion only).
        steps: number of steps (each changes 1/steps of the pixels).
        rng: NumPy generator used to break ties between equal scores.
        blur_kernel, blur_sigma: Gaussian blur for fill "blur". The defaults
            suit 28x28 and 32x32 images; larger images scale them.
        inpaint_iterations: neighbour-averaging steps for fill "inpaint".

    Returns:
        Array of steps + 1 probabilities.
    """
    rng = rng or np.random.default_rng(0)
    masks = _step_masks(_ranking(attribution, rng), x.shape, steps).to(x.device)
    original = x.detach().expand(steps + 1, -1, -1, -1)

    if fill == "inpaint":
        if mode != "deletion":
            raise ValueError("inpaint is only defined for deletion")
        images = inpaint(original.clone(), masks, iterations=inpaint_iterations)
    else:
        reference = (torch.zeros_like(original) if fill == "zero"
                     else blur(x.detach(), blur_kernel, blur_sigma).expand_as(original))
        if mode == "deletion":
            images = torch.where(masks, reference, original)
        elif mode == "insertion":
            images = torch.where(masks, original, reference)
        else:
            raise ValueError("mode must be 'deletion' or 'insertion'")

    with torch.no_grad():
        probs = torch.exp(model(images))[:, target]
    return probs.cpu().numpy()


def deletion_curve(model, x, target, attribution, steps=20, rng=None):
    """Classic deletion curve with black replacement pixels."""
    return perturbation_curve(model, x, target, attribution, "deletion", "zero", steps, rng)


def area_under_curve(curve):
    """Area under a curve sampled evenly over [0, 1]."""
    return float(np.trapezoid(curve, dx=1.0 / (len(curve) - 1)))


def cascading_randomization(model, seed=0):
    """Yield (stage, model) pairs with progressively more layers randomized.

    The first stage is "original" (a copy of the trained model). Each next
    stage additionally re-initialises one layer, following
    RANDOMIZATION_ORDER. The same model object is modified in place, so use
    each stage before requesting the next.
    """
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
    """Spearman rank correlation between the absolute values of two maps.

    Args:
        mask: optional boolean (H, W) array. If given, only those pixels are
            compared. Using the object (non-background) pixels removes the
            effect of methods that are zero on a black background for any
            model, such as Integrated Gradients with a black baseline.

    Returns NaN if either map is constant (for example an all-zero Grad-CAM),
    since correlation is undefined in that case.
    """
    a, b = np.abs(a), np.abs(b)
    if mask is not None:
        a, b = a[mask], b[mask]
    ra = pd.Series(a.ravel()).rank().to_numpy()
    rb = pd.Series(b.ravel()).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def relevance_mass_accuracy(attribution, mask):
    """Share of positive attribution that falls inside the ground-truth mask."""
    positive = np.clip(attribution, 0, None)
    total = positive.sum()
    return float(positive[mask].sum() / total) if total > 0 else float("nan")


def relevance_rank_accuracy(attribution, mask, rng=None):
    """Share of the top-k pixels inside the mask, where k is the mask size."""
    rng = rng or np.random.default_rng(0)
    k = int(mask.sum())
    top = _ranking(attribution, rng)[:k]
    return float(mask.reshape(-1)[top].mean())

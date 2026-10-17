"""Explanation (attribution) methods built on Captum.

Every method takes a model, an input of shape (1, 1, 28, 28) and a target
class, and returns a 28x28 NumPy attribution map. Higher values mean the
pixel contributed more evidence for the target class.
"""

import numpy as np
import torch
from captum.attr import (
    IntegratedGradients,
    LayerAttribution,
    LayerGradCam,
    NoiseTunnel,
    Occlusion,
    Saliency,
)


def saliency(model, x, target):
    # Absolute gradient of the class score with respect to each pixel.
    return Saliency(model).attribute(x, target=target, abs=True)


def smoothgrad(model, x, target):
    # Saliency averaged over 25 noisy copies of the input (noise std = 15% of
    # the pixel range, Smilkov et al., 2017). The noise is seeded so the same
    # input always gets the same map, which keeps comparisons fair.
    state = torch.get_rng_state()
    torch.manual_seed(0)
    try:
        return NoiseTunnel(Saliency(model)).attribute(
            x, nt_type="smoothgrad", nt_samples=25, nt_samples_batch_size=25,
            stdevs=0.15, target=target, abs=True,
        )
    finally:
        torch.set_rng_state(state)


def integrated_gradients(model, x, target):
    # Gradients averaged along a straight path from a black image (baseline)
    # to the input, multiplied by (input - baseline).
    baseline = torch.zeros_like(x)
    return IntegratedGradients(model).attribute(
        x, baselines=baseline, target=target, n_steps=50
    )


def grad_cam(model, x, target):
    # Feature maps of the last conv layer weighted by their average gradient,
    # then upsampled from 6x6 to the 28x28 input size. Grad-CAM needs the
    # activations after ReLU, which is the input of pool2 (the model reuses
    # one ReLU module, so it cannot be hooked directly).
    cam = LayerGradCam(model, model.pool2).attribute(
        x, target=target, relu_attributions=True, attribute_to_layer_input=True
    )
    return LayerAttribution.interpolate(cam, x.shape[-2:], interpolate_mode="bilinear")


def occlusion(model, x, target):
    # Slide a 4x4 black patch over the image and record the drop in the
    # class score at each position.
    return Occlusion(model).attribute(
        x,
        target=target,
        sliding_window_shapes=(x.shape[1], 4, 4),  # cover all colour channels together
        strides=(x.shape[1], 2, 2),
        baselines=0,
        perturbations_per_eval=64,  # batch patches for speed; same result
    )


METHODS = {
    "saliency": ("Saliency", saliency),
    "smoothgrad": ("SmoothGrad", smoothgrad),
    "integrated_gradients": ("Integrated Gradients", integrated_gradients),
    "grad_cam": ("Grad-CAM", grad_cam),
    "occlusion": ("Occlusion", occlusion),
}


def explain(model, x: torch.Tensor, target: int, method: str) -> np.ndarray:
    """Compute an HxW attribution map for one image.

    For colour images the per-channel attributions are summed into one map.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method '{method}'. Choose from {list(METHODS)}")
    model.eval()
    x = x.detach().clone().requires_grad_()
    attribution = METHODS[method][1](model, x, int(target))
    return attribution.detach().cpu().numpy()[0].sum(axis=0)


def normalize(attribution: np.ndarray) -> np.ndarray:
    """Scale positive evidence to [0, 1] for display."""
    positive = np.clip(attribution, 0, None)
    peak = positive.max()
    return positive / peak if peak > 0 else positive

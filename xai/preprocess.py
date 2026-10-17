"""Convert user images (canvas drawings or uploads) into model inputs.

The models were trained on 28x28 images with a bright object on a black
background and values in [0, 1]. User images are converted to match:

1. Take the object as bright pixels: the alpha channel for transparent
   images (the drawing canvas), otherwise grayscale, inverted if the
   background is light (for example a dark digit photographed on paper).
2. Crop to the object's bounding box.
3. Scale it to the size objects have in the training data (MNIST digits
   fit a 20x20 box; Fashion-MNIST items fill the 28x28 frame) and center it.
4. Scale pixel values to [0, 1].
"""

import base64
import binascii
from io import BytesIO

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from .data import info

IMAGE_SIZE = 28
MAX_PIXELS = 4096 * 4096


class InvalidImage(ValueError):
    """Raised when an image cannot be decoded or contains no object."""


def decode_data_url(data_url: str) -> Image.Image:
    """Decode a 'data:image/...;base64,...' string into a PIL image."""
    try:
        _, encoded = data_url.split(",", 1)
        raw = base64.b64decode(encoded, validate=True)
        image = Image.open(BytesIO(raw))
        if image.width * image.height > MAX_PIXELS:
            raise InvalidImage("The image is too large.")
        image.load()
    except (AttributeError, ValueError, binascii.Error,
            UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        if isinstance(error, InvalidImage):
            raise
        raise InvalidImage("Could not read the image. Use a PNG or JPEG file.") from error
    return image


def foreground(image: Image.Image) -> np.ndarray:
    """Grayscale array (0-255) with the object bright and the background dark."""
    has_alpha = image.mode in ("RGBA", "LA") or (
        image.mode == "P" and "transparency" in image.info
    )
    if has_alpha:
        alpha = np.asarray(image.convert("RGBA").getchannel("A"), dtype=np.float32)
        if alpha.min() < 255:  # real transparency, e.g. the drawing canvas
            return alpha

    gray = np.asarray(image.convert("L"), dtype=np.float32)
    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    if border.mean() > 127:  # light background: invert so the object is bright
        gray = 255.0 - gray
    return gray


def fit_to_frame(pixels: np.ndarray, fit_size: int) -> np.ndarray:
    """Crop the object, scale its longest side to fit_size, center in 28x28."""
    peak = pixels.max()
    mask = pixels > max(20.0, 0.2 * peak)
    if peak <= 0 or not mask.any():
        raise InvalidImage("The image is empty. Draw or upload something first.")

    rows, cols = np.where(mask)
    cropped = pixels[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    height, width = cropped.shape
    scale = fit_size / max(height, width)
    new_w = max(1, round(width * scale))
    new_h = max(1, round(height * scale))

    resized = Image.fromarray(np.clip(cropped, 0, 255).astype(np.uint8), mode="L")
    resized = np.asarray(resized.resize((new_w, new_h), Image.Resampling.LANCZOS))

    frame = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    top = (IMAGE_SIZE - new_h) // 2
    left = (IMAGE_SIZE - new_w) // 2
    frame[top:top + new_h, left:left + new_w] = resized
    return frame


def image_to_tensor(image: Image.Image, dataset: str) -> torch.Tensor:
    """Full pipeline: PIL image -> model input of shape (1, 1, 28, 28)."""
    frame = fit_to_frame(foreground(image), info(dataset)["fit_size"])
    return torch.from_numpy(frame / 255.0).float().reshape(1, 1, IMAGE_SIZE, IMAGE_SIZE)

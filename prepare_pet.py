"""Preprocess the Oxford-IIIT Pet dataset (cat vs dog) for the dose-response study.

Usage:
    python prepare_pet.py           # download (about 800 MB, once) and preprocess
    python prepare_pet.py --force   # rebuild the .npz files

Output: data/pet128/trainval.npz and data/pet128/test.npz (the official
splits), each with
    images  uint8 (N, 3, 128, 128)   RGB photo
    masks   uint8 (N, 128, 128)      trimap: 1 = pet, 2 = background, 3 = border
    labels  int64 (N,)               0 = cat, 1 = dog (torchvision "binary-category")
Images are resized so the shorter side is 128 px (bilinear, antialiased) and
centre-cropped to 128x128; masks get the same geometry with nearest-neighbour
resizing, so they stay aligned with the images. The object map used by the
study counts trimap 1 AND 3 as pet (xai.data.PET_OBJECT_VALUES): the border
band is the pet's uncertain outline. 500 trainval images are held out for
validation with seed 42 (xai/data.py). Kaggle needs no download: the .npz
files are included in the bundle (kaggle/make_bundle.ps1 -Dataset pet).

Also saves results/pet_sanity.png: 4 test images with the shortcut patch, the
object mask overlay and the patch outline, to check that masks align.

If the download fails (e.g. an expired certificate), download by hand:
    mkdir data\\oxford-iiit-pet
    curl.exe -L -o data\\oxford-iiit-pet\\images.tar.gz https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
    curl.exe -L -o data\\oxford-iiit-pet\\annotations.tar.gz https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
Check the MD5 with Get-FileHash -Algorithm MD5:
    images.tar.gz       5c4f3ee8e5d25df40f4fd59a7f44e54c
    annotations.tar.gz  95a8c909bbe2e81eed6a22bccdf3f68f
If curl reports a certificate error, add --ssl-no-revoke or, as a last resort,
-k (the MD5 check still guards the files). Extract both:
    tar -xzf data\\oxford-iiit-pet\\images.tar.gz -C data\\oxford-iiit-pet
    tar -xzf data\\oxford-iiit-pet\\annotations.tar.gz -C data\\oxford-iiit-pet
then run python prepare_pet.py again (torchvision finds the extracted files).
"""

import argparse
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as TF
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import InterpolationMode

from xai.data import PET_OBJECT_VALUES, shortcut_split
from xai.paths import DATA_DIR, RESULTS_DIR

SIZE = 128
OUT_DIR = DATA_DIR / "pet128"
SPLITS = ("trainval", "test")
MANUAL = __doc__[__doc__.index("If the download fails"):]


def load_pet(split):
    try:
        return OxfordIIITPet(root=DATA_DIR, split=split,
                             target_types=["binary-category", "segmentation"], download=True)
    except Exception as error:  # noqa: BLE001 (any download problem gets the same advice)
        print(f"Could not download or open the Oxford-IIIT Pet dataset: {error}\n\n{MANUAL}")
        sys.exit(1)


def resize_crop(image, interpolation):
    image = TF.resize(image, SIZE, interpolation=interpolation, antialias=True)
    return TF.center_crop(image, [SIZE, SIZE])


def preprocess(split):
    ds = load_pet(split)
    images, masks, labels = [], [], []
    for image, (label, trimap) in ds:
        image = resize_crop(image.convert("RGB"), InterpolationMode.BILINEAR)
        trimap = resize_crop(trimap, InterpolationMode.NEAREST)
        images.append(np.asarray(image, dtype=np.uint8).transpose(2, 0, 1))
        masks.append(np.asarray(trimap, dtype=np.uint8))
        labels.append(int(label))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_DIR / f"{split}.npz", images=np.stack(images), masks=np.stack(masks),
             labels=np.array(labels, dtype=np.int64))
    return ds.bin_classes


def report(split, names):
    with np.load(OUT_DIR / f"{split}.npz") as d:
        images, masks, labels = d["images"], d["masks"], d["labels"]
    counts = ", ".join(f"{name} {int((labels == c).sum())} ({(labels == c).mean():.1%})"
                       for c, name in enumerate(names))
    print(f"{split}: {len(labels)} images, image shape {images.shape[1:]} {images.dtype}, "
          f"mask shape {masks.shape[1:]}, mask values {np.unique(masks).tolist()}, "
          f"pet pixels {np.isin(masks, PET_OBJECT_VALUES).mean():.1%} | {counts}")


def sanity_figure():
    test = shortcut_split("pet_resnet18", "patch", "test")
    picks = [i for c in (0, 1) for i in np.flatnonzero(test.labels == c)[:2]]
    names = ["cat", "dog"]
    fig, axes = plt.subplots(1, len(picks), figsize=(3.2 * len(picks), 3.5))
    for ax, i in zip(axes, picks):
        image, label = test[i]
        ax.imshow(image.permute(1, 2, 0).numpy())
        overlay = np.zeros((test.size, test.size, 4))
        overlay[test.object_map(i) > 0] = [0.0, 0.6, 1.0, 0.35]
        ax.imshow(overlay)
        rows, cols = np.nonzero(test.mask(i))
        ax.add_patch(plt.Rectangle((cols.min() - 0.5, rows.min() - 0.5), np.ptp(cols) + 1, np.ptp(rows) + 1,
                                   fill=False, edgecolor="red", linewidth=1.5))
        ax.set_xlim(-0.5, test.size - 0.5)
        ax.set_ylim(test.size - 0.5, -0.5)
        ax.set_title(f"#{i} {names[label]} (patch: {names[test.patch_class[i]]})", fontsize=9)
        ax.axis("off")
    fig.suptitle("Pet 128x128: object mask (blue) and shortcut patch (red outline)", fontsize=10)
    fig.tight_layout()
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / "pet_sanity.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"saved {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="rebuild existing .npz files")
    args = parser.parse_args()
    names = ["Cat", "Dog"]
    for split in SPLITS:
        if args.force or not (OUT_DIR / f"{split}.npz").exists():
            names = preprocess(split)
        report(split, names)
    sanity_figure()


if __name__ == "__main__":
    main()

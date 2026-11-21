"""Datasets: MNIST, Fashion-MNIST, PneumoniaMNIST and shortcut ("decoy") variants.

Images of the original datasets are 1x28x28 with values in [0, 1]. The
dose-response study also uses colour images: 3x32x32 (CIFAR-10) and
3x128x128 (Oxford-IIIT Pet, preprocessed by prepare_pet.py).

Decoy datasets (after Ross et al., 2017) add a small white patch whose
position encodes the class. A model trained on them learns to rely on the
patch, so the correct explanation is known: the patch pixels. This gives a
ground truth for evaluating explanation methods and metrics.

In training, half of the images are swapped for an image of another class
(label and patch kept), so only the patch is fully predictive and the model
must rely on it.

Decoy test modes (test images are never swapped):
    aligned      patch at the true class's position
    conflicting  patch at a different class's position. When the model
                 predicts the patch's class, the patch is by construction the
                 evidence for that prediction: the ground truth used here.
    none         no patch
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, random_split
from torchvision import datasets, transforms

from .paths import DATA_DIR

VALIDATION_SIZE = 5000  # held out from the torchvision training sets
SPLIT_SEED = 42         # fixed so every run uses the same validation images
PATCH_SIZE = 3
PATCH_VALUE = 1.0
IMAGE_SIZE = 28

# Oxford-IIIT Pet trimap values: 1 = pet, 2 = background, 3 = border band
# (the uncertain outline around the pet). The border band is counted as pet,
# so the object map covers the whole animal including its outline.
PET_OBJECT_VALUES = (1, 3)

# Patch positions (top-left corner). MNIST: along the top and bottom borders,
# where digits never are. Binary tasks: the two top corners, like the
# hospital markers often burned into real X-rays.
DECOY_POSITIONS_10 = [(0, 0), (0, 6), (0, 12), (0, 18), (0, 25),
                      (25, 0), (25, 6), (25, 12), (25, 18), (25, 25)]
DECOY_POSITIONS_2 = [(0, 0), (0, 25)]

# Diffuse ("texture") shortcut: each class gets a fixed random set of about
# 10% of all pixels, spread over the whole image, which are brightened by 0.3.
SHORTCUT_TYPES = ("patch", "texture")
TEXTURE_DENSITY = 0.10
TEXTURE_VALUE = 0.3
TEXTURE_SEED = 123

# Amendment A3, SD4 (docs/PREDICTIONS_amendments.md): a content-coded
# watermark, used only with pet_resnet18_pretrained. Two white semi-
# transparent marks (0 = cat, 1 = dog, the Pet class order) in a 24 px wide x
# 12 px high box flush with the bottom-right corner (rows 116-127, columns
# 104-127). Strokes are half-open (row start, row stop, col start, col stop)
# ranges in box coordinates: cat = two vertical bars (rows 0-11, columns 0-3
# and 20-23), dog = two horizontal bars (rows 0-3 and 8-11, columns 6-17);
# 96 pixels each, disjoint. Drawn from these coordinates (no font) and saved
# as xai/watermark_masks.npy. SHORTCUT_TYPES stays (patch, texture), since it
# sets the CLI defaults and ORIGINAL_DESIGN.
A3_SHORTCUT_TYPES = ("watermark",)
ALL_SHORTCUT_TYPES = SHORTCUT_TYPES + A3_SHORTCUT_TYPES
WATERMARK_BASES = ("pet_resnet18_pretrained",)
WATERMARK_SIZE = 128
WATERMARK_BOX = (116, 104)       # top-left (row, col) of the box in the image
WATERMARK_BOX_SHAPE = (12, 24)   # (height, width)
WATERMARK_STROKES = {0: ((0, 12, 0, 4), (0, 12, 20, 24)),
                     1: ((0, 4, 6, 18), (8, 12, 6, 18))}
WATERMARK_ALPHA = 0.4
WATERMARK_MASK_FILE = Path(__file__).with_name("watermark_masks.npy")


def watermark_masks(size: int = WATERMARK_SIZE) -> np.ndarray:
    """(2, size, size) boolean stroke masks of the cat (0) and dog (1) marks."""
    masks = np.zeros((2, size, size), dtype=bool)
    top, left = WATERMARK_BOX
    for cls, strokes in WATERMARK_STROKES.items():
        for r0, r1, c0, c1 in strokes:
            masks[cls, top + r0:top + r1, left + c0:left + c1] = True
    return masks


def save_watermark_masks(path=WATERMARK_MASK_FILE) -> None:
    np.save(path, watermark_masks())


def apply_watermark(image, mask):
    """(1 - alpha) * pixel + alpha * 1 on the stroke pixels, in every channel;
    other pixels unchanged. mask: boolean (H,W) tensor."""
    return torch.where(mask, image * (1 - WATERMARK_ALPHA) + WATERMARK_ALPHA * 1.0, image)


def texture_patterns(classes: int, size: int = IMAGE_SIZE) -> np.ndarray:
    """(classes, size, size) boolean masks of the class-coded texture pixels."""
    rng = np.random.default_rng(TEXTURE_SEED)
    return rng.random((classes, size, size)) < TEXTURE_DENSITY


def border_positions(classes: int, size: int, patch_size: int = PATCH_SIZE) -> list[tuple[int, int]]:
    """Patch positions along the top and bottom borders (10 classes) or the two
    top corners (2 classes), for any image and patch size. For 28x28 with the
    default 3x3 patch this gives exactly DECOY_POSITIONS_10 / DECOY_POSITIONS_2."""
    last = size - patch_size
    if classes == 2:
        return [(0, 0), (0, last)]
    xs = [round(k * last / 4) for k in range(5)]
    if size == IMAGE_SIZE:
        xs = [0, 6, 12, 18, 25]  # the original MNIST positions
    return [(0, x) for x in xs] + [(last, x) for x in xs]

DATASETS = {
    "mnist": {
        "title": "MNIST (digits)",
        "source": "torchvision", "cls": datasets.MNIST,
        "classes": [str(i) for i in range(10)],
        "fit_size": 20, "epochs": 10,
    },
    "fashion_mnist": {
        "title": "Fashion-MNIST (clothing)",
        "source": "torchvision", "cls": datasets.FashionMNIST,
        "classes": ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"],
        "fit_size": 28, "epochs": 15,
    },
    "pneumonia_mnist": {
        "title": "PneumoniaMNIST (chest X-ray)",
        "source": "medmnist", "flag": "pneumoniamnist",
        "classes": ["Normal", "Pneumonia"],
        "fit_size": 28, "epochs": 20,
    },
    # Used only by the dose-response study (32x32 colour, a larger CNN).
    "cifar10": {
        "title": "CIFAR-10 (natural images)",
        "source": "torchvision", "cls": datasets.CIFAR10,
        "classes": ["airplane", "automobile", "bird", "cat", "deer",
                    "dog", "frog", "horse", "ship", "truck"],
        # Augmentation is off: with a 10-epoch budget, random crop and flip cut
        # validation accuracy from 43% to 21% in a 2-epoch test on 10k images.
        "size": 32, "arch": "cifar", "augment": False, "dose_only": True,
        "fit_size": 32, "epochs": 20,
    },
    # The same CIFAR-10 data with a standard ResNet-18 (run on a GPU, e.g. Kaggle).
    "cifar10_resnet18": {
        "title": "CIFAR-10 with ResNet-18",
        "source": "torchvision", "cls": datasets.CIFAR10,
        "classes": ["airplane", "automobile", "bird", "cat", "deer",
                    "dog", "frog", "horse", "ship", "truck"],
        "size": 32, "arch": "resnet18", "augment": False, "dose_only": True,
        "fit_size": 32, "epochs": 15,
    },
    # Oxford-IIIT Pet, cat vs dog, 128x128 photos with real segmentation masks
    # (preprocessed once by prepare_pet.py into data/pet128/*.npz; run on a GPU,
    # e.g. Kaggle). Settings that depend on image size are scaled from the
    # 32px values by 128/32 = 4; datasets without these keys use the defaults
    # (3x3 patch, blur 25/4.0, 100 inpainting steps, occlusion 4/2), so their
    # behaviour is unchanged:
    #   patch_size 12          3/32 of the side, the same fraction as CIFAR-10
    #   blur_sigma 16.0        4 x 128/32
    #   blur_kernel 97         6 x sigma + 1 (odd)
    #   inpaint_iterations     1600 = 100 x (128/32)^2: averaging neighbours is
    #                          a diffusion, whose reach grows with the square
    #                          root of the steps, so 4x the distance needs 16x
    #   occlusion 16 / 8       window size/8 and stride size/16 (4 and 2 at 32px)
    #   matched "compact"      size-matched map = the k mask pixels nearest the
    #                          mask centroid (a binary mask has no "brightest")
    "pet_resnet18": {
        "title": "Oxford-IIIT Pet (cat vs dog) with ResNet-18",
        "source": "npz", "folder": "pet128", "val_size": 500,
        "classes": ["cat", "dog"],
        "size": 128, "arch": "resnet18", "augment": False, "dose_only": True,
        "fit_size": 128, "epochs": 30,
        "patch_size": 12, "matched": "compact",
        "blur_kernel": 97, "blur_sigma": 16.0, "inpaint_iterations": 1600,
        "occlusion_window": 16, "occlusion_stride": 8,
    },
    "decoy_mnist": {
        "title": "Decoy MNIST (shortcut)",
        "base": "mnist", "positions": DECOY_POSITIONS_10, "swap_fraction": 0.5,
        "fit_size": 20, "epochs": 5,
    },
    "decoy_pneumonia": {
        "title": "Decoy PneumoniaMNIST (shortcut)",
        "base": "pneumonia_mnist", "positions": DECOY_POSITIONS_2, "swap_fraction": 0.5,
        "fit_size": 28, "epochs": 20,
    },
}
# Amendment A2, SD1 (docs/PREDICTIONS_amendments.md): the same data, patch,
# fills and oracles as pet_resnet18, with an ImageNet-pretrained ResNet-18
# fine-tuned for 15 epochs at Adam lr 1e-4. pet_resnet18 itself is unchanged.
DATASETS["pet_resnet18_pretrained"] = dict(
    DATASETS["pet_resnet18"],
    title="Oxford-IIIT Pet (cat vs dog) with ImageNet-pretrained ResNet-18",
    arch="resnet18_pretrained", epochs=15, lr=1e-4)


def info(dataset: str) -> dict:
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from {list(DATASETS)}")
    return DATASETS[dataset]


def is_decoy(dataset: str) -> bool:
    return "base" in info(dataset)


def class_names(dataset: str) -> list[str]:
    entry = info(dataset)
    return class_names(entry["base"]) if "base" in entry else entry["classes"]


def num_classes(dataset: str) -> int:
    return len(class_names(dataset))


def _to_int(label) -> int:
    return int(np.asarray(label).reshape(-1)[0])


def _labels(ds) -> np.ndarray:
    """All labels of a dataset without loading its images."""
    if isinstance(ds, Subset):
        return _labels(ds.dataset)[np.asarray(ds.indices)]
    if hasattr(ds, "targets"):
        return np.asarray(ds.targets).reshape(-1)
    if hasattr(ds, "labels"):
        return np.asarray(ds.labels).reshape(-1)
    return np.array([_to_int(ds[i][1]) for i in range(len(ds))])


class NpzImages(Dataset):
    """Preprocessed images from an .npz file (keys images uint8 (N,C,H,W),
    masks uint8 (N,H,W), labels), loaded into memory once."""

    def __init__(self, path):
        with np.load(path) as d:
            self.images = torch.from_numpy(d["images"])
            self.masks = d["masks"]
            self.targets = d["labels"].astype(np.int64)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return self.images[i].float() / 255, int(self.targets[i])


def _object_masks(ds):
    """Boolean (N,H,W) real object masks of a dataset, or None if it has none."""
    if isinstance(ds, Subset):
        masks = _object_masks(ds.dataset)
        return None if masks is None else masks[np.asarray(ds.indices)]
    if isinstance(ds, NpzImages):
        return np.isin(ds.masks, PET_OBJECT_VALUES)
    return None


def _base_split(dataset: str, split: str):
    entry = info(dataset)
    to_tensor = transforms.ToTensor()
    DATA_DIR.mkdir(exist_ok=True)

    if entry["source"] == "npz":
        path = DATA_DIR / entry["folder"] / ("test.npz" if split == "test" else "trainval.npz")
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Create it with: python prepare_pet.py")
        full = NpzImages(path)
        if split == "test":
            return full
        val = entry.get("val_size", VALIDATION_SIZE)
        train, valid = random_split(
            range(len(full)), [len(full) - val, val],
            generator=torch.Generator().manual_seed(SPLIT_SEED),
        )
        return Subset(full, list(train.indices) if split == "train" else list(valid.indices))

    if entry["source"] == "torchvision":
        if split == "test":
            return entry["cls"](root=DATA_DIR, train=False, download=True, transform=to_tensor)
        # Training images may be augmented (random crop and flip). The shortcut
        # is added AFTER this, so it always stays at its class-coded position.
        transform = to_tensor
        if split == "train" and entry.get("augment"):
            transform = transforms.Compose([transforms.RandomCrop(entry["size"], padding=4),
                                            transforms.RandomHorizontalFlip(), to_tensor])
        full = entry["cls"](root=DATA_DIR, train=True, download=True, transform=transform)
        # Split indices only; the same seed gives the same validation images for
        # either transform.
        train, val = random_split(
            range(len(full)), [len(full) - VALIDATION_SIZE, VALIDATION_SIZE],
            generator=torch.Generator().manual_seed(SPLIT_SEED),
        )
        return Subset(full, list(train.indices) if split == "train" else list(val.indices))

    # MedMNIST provides official train/val/test splits.
    import medmnist
    cls = getattr(medmnist, medmnist.INFO[entry["flag"]]["python_class"])
    return cls(split=split, transform=to_tensor, target_transform=_to_int,
               download=True, root=str(DATA_DIR))


class DecoyDataset(Dataset):
    """Adds a class-coded shortcut to every image of a base dataset.

    shortcut: "patch" (a white square whose position encodes the class; 3x3
    by default, patch_size per dataset, e.g. 12x12 for 128x128 Pet images)
    or "texture" (a class-specific set of scattered pixels brightened by 0.3)
    or "watermark" (amendment A3: a class-specific semi-transparent mark in the
    bottom-right corner; 128x128 images, 2 classes).

    object_masks: optional boolean (N,H,W) real object masks of the base
    images (Pet segmentation masks), used by object_map.

    swap_fraction: share of images replaced by an image of a DIFFERENT class
    while keeping the label and shortcut. The shortcut then predicts the label
    better than the image does, so a higher fraction makes the model rely on
    the shortcut more. Used for training and validation only.
    """

    MODES = ("aligned", "conflicting", "none")

    def __init__(self, base, positions, mode="aligned", swap_fraction=0.0, seed=0,
                 shortcut="patch", shortcut_noise=0.0, patch_size=PATCH_SIZE, object_masks=None):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        if shortcut not in ALL_SHORTCUT_TYPES:
            raise ValueError(f"shortcut must be one of {ALL_SHORTCUT_TYPES}")
        self.base, self.positions, self.mode = base, positions, mode
        self.shortcut = shortcut
        self.patch_size, self.object_masks = patch_size, object_masks
        sample, _ = base[0]
        self.channels, self.size = int(sample.shape[0]), int(sample.shape[-1])
        self.patterns = texture_patterns(len(positions), self.size) if shortcut == "texture" else None
        if shortcut == "watermark":
            if self.size != WATERMARK_SIZE or len(positions) != 2:
                raise ValueError(f"the watermark needs {WATERMARK_SIZE}x{WATERMARK_SIZE} images and 2 classes")
            self.watermark_np = watermark_masks(self.size)
            self.watermarks = torch.from_numpy(self.watermark_np)
        grid = np.arange(self.size) - (self.size - 1) / 2
        sigma = self.size / 4
        self.center_prior = np.exp(-(grid[:, None] ** 2 + grid[None, :] ** 2) / (2 * sigma ** 2))
        self.labels = _labels(base).astype(int)
        classes = len(positions)
        rng = np.random.default_rng(seed)
        if mode == "conflicting":
            # Shift every label by 1..C-1 so the patch always points elsewhere.
            shift = rng.integers(1, classes, size=len(self.labels))
            self.patch_class = (self.labels + shift) % classes
        else:
            self.patch_class = self.labels

        # Index of the image actually shown for each item.
        self.source = np.arange(len(self.labels))
        if swap_fraction > 0:
            by_class = [np.flatnonzero(self.labels == c) for c in range(classes)]
            for i in np.flatnonzero(rng.random(len(self.labels)) < swap_fraction):
                other = (self.labels[i] + rng.integers(1, classes)) % classes
                self.source[i] = rng.choice(by_class[other])

        # shortcut_noise: share of items whose shortcut points to a random
        # WRONG class, making the shortcut less reliable than the image.
        if shortcut_noise > 0 and mode == "aligned":
            self.patch_class = self.patch_class.copy()
            noisy = np.flatnonzero(rng.random(len(self.labels)) < shortcut_noise)
            shift = rng.integers(1, classes, size=len(noisy))
            self.patch_class[noisy] = (self.labels[noisy] + shift) % classes

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        image, _ = self.base[int(self.source[i])]
        label = self.labels[i]
        if self.mode != "none":
            image = image.clone()
            if self.shortcut == "patch":
                y, x = self.positions[self.patch_class[i]]
                p = self.patch_size
                image[:, y:y + p, x:x + p] = PATCH_VALUE  # white in every channel
            elif self.shortcut == "watermark":
                image = apply_watermark(image, self.watermarks[self.patch_class[i]])
            else:
                pattern = torch.from_numpy(self.patterns[self.patch_class[i]])
                image = torch.where(pattern, (image + TEXTURE_VALUE).clamp(max=1.0), image)
        return image, _to_int(label)

    def with_shortcut(self, image, cls):
        """A copy of `image` with class cls's patch or texture added (the same
        two branches as __getitem__). Used by the per-image behaviour labels."""
        image = image.clone()
        if self.shortcut == "patch":
            y, x = self.positions[int(cls)]
            p = self.patch_size
            image[:, y:y + p, x:x + p] = PATCH_VALUE
        elif self.shortcut == "watermark":
            image = apply_watermark(image, self.watermarks[int(cls)])
        else:
            pattern = torch.from_numpy(self.patterns[int(cls)])
            image = torch.where(pattern, (image + TEXTURE_VALUE).clamp(max=1.0), image)
        return image

    def mask(self, i) -> np.ndarray:
        """Boolean HxW map of the shortcut pixels (the ground-truth evidence)."""
        if self.mode == "none":
            return np.zeros((self.size, self.size), dtype=bool)
        return self.mask_for_class(self.patch_class[i])

    def mask_for_class(self, cls) -> np.ndarray:
        """Where the shortcut WOULD be for class cls: a patch at that class's
        position, that class's texture pattern, or its watermark stroke mask.
        Used for placebo maps."""
        if self.shortcut == "texture":
            return self.patterns[int(cls)].copy()
        if self.shortcut == "watermark":
            return self.watermark_np[int(cls)].copy()
        mask = np.zeros((self.size, self.size), dtype=bool)
        y, x = self.positions[int(cls)]
        mask[y:y + self.patch_size, x:x + self.patch_size] = True
        return mask

    def object_map(self, i) -> np.ndarray:
        """Oracle map of the object, with shortcut pixels set to 0.

        Grayscale datasets: the image's own pixel values. On MNIST the
        background is black, so this is the digit. PneumoniaMNIST has no dark
        background (97.5% of pixels exceed 0.1), so there the map is an
        intensity ranking of the whole radiograph, not an object region.
        Colour natural images (CIFAR-10) have no clean background, so the map
        is a centred Gaussian (sigma = size/4), since CIFAR-10 objects are
        mostly centred. This is a modelling choice and is stated as such.
        Datasets with real object masks (Oxford-IIIT Pet) use the binary
        segmentation mask of the shown image (pet and border band = 1).
        """
        if self.object_masks is not None:
            values = self.object_masks[int(self.source[i])].astype(np.float64)
        elif self.channels == 1:
            image, _ = self.base[int(self.source[i])]
            values = image[0].numpy().astype(np.float64).copy()
        else:
            values = self.center_prior.copy()
        values[self.mask(i)] = 0.0
        return values


class _IntLabels(Dataset):
    """Wraps a dataset so labels are always plain ints."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        image, label = self.base[i]
        return image, _to_int(label)


def load_split(dataset: str, split: str, decoy_mode: str = "aligned"):
    """Load 'train', 'val' or 'test'. Data downloads to data/ on first use.

    decoy_mode only applies to decoy datasets and the test split; training
    and validation images always use the aligned patch.
    """
    if split not in ("train", "val", "test"):
        raise ValueError("split must be 'train', 'val' or 'test'")
    entry = info(dataset)
    if "base" in entry:
        base = _base_split(entry["base"], split)
        if split == "test":
            return DecoyDataset(base, entry["positions"], mode=decoy_mode)
        return DecoyDataset(base, entry["positions"], mode="aligned",
                            swap_fraction=entry["swap_fraction"])
    return _IntLabels(_base_split(dataset, split))


def shortcut_split(base: str, shortcut: str, split: str, swap_fraction: float = 0.0,
                   shortcut_noise: float = 0.0, mode: str = "aligned") -> DecoyDataset:
    """A base dataset with a patch, texture or watermark shortcut, for the
    dose-response study. The watermark (amendment A3, SD4) is allowed only
    with the bases in WATERMARK_BASES.

    swap_fraction and shortcut_noise only apply to 'train' and 'val'; test
    images are never altered that way. mode ('aligned', 'conflicting',
    'none') only applies to 'test'.
    """
    if shortcut == "watermark" and base not in WATERMARK_BASES:
        raise ValueError(f"the watermark shortcut is used only with {', '.join(WATERMARK_BASES)}, not {base}")
    entry = info(base)
    patch = entry.get("patch_size", PATCH_SIZE)
    positions = border_positions(num_classes(base), entry.get("size", IMAGE_SIZE), patch)
    base_ds = _base_split(base, split)
    extra = {"patch_size": patch, "object_masks": _object_masks(base_ds)}
    if split == "test":
        return DecoyDataset(base_ds, positions, mode=mode, shortcut=shortcut, **extra)
    return DecoyDataset(base_ds, positions, mode="aligned", swap_fraction=swap_fraction,
                        shortcut=shortcut, shortcut_noise=shortcut_noise, **extra)

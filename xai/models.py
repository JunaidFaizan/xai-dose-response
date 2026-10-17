"""CNN model definition and checkpoint loading."""

from pathlib import Path

import torch
from torch import nn

from .data import num_classes
from .paths import CHECKPOINT_DIR, PROJECT_ROOT  # noqa: F401 (re-exported)

DEFAULT_SEED = 42  # the seed of the main model used by the dashboard


class MnistModel(nn.Module):
    """CNN for 28x28 grayscale images (used for every dataset).

    Architecture (unchanged from the B.Tech project so old checkpoints load):
        (CONV => RELU) * 2 => POOL => DROPOUT
        (CONV => RELU) * 2 => POOL => DROPOUT
        FC => RELU => DROPOUT => FC => LOG_SOFTMAX
    """

    def __init__(self, classes: int = 10) -> None:
        super().__init__()
        self.classes = classes

        # (N,1,28,28) -> (N,16,24,24) -> (N,32,20,20) -> pool (N,32,10,10)
        self.conv1A = nn.Conv2d(1, 16, kernel_size=5)
        self.conv1B = nn.Conv2d(16, 32, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.act = nn.ReLU()
        self.do = nn.Dropout(0.25)

        # (N,32,10,10) -> (N,64,8,8) -> (N,128,6,6) -> pool (N,128,3,3)
        self.conv2A = nn.Conv2d(32, 64, kernel_size=3)
        self.conv2B = nn.Conv2d(64, 128, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.dense3 = nn.Linear(128 * 3 * 3, 32)
        self.dense4 = nn.Linear(32, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.do(self.pool1(self.act(self.conv1B(self.act(self.conv1A(x))))))
        x = self.do(self.pool2(self.act(self.conv2B(self.act(self.conv2A(x))))))
        x = x.view(x.size(0), -1)
        x = self.do(self.act(self.dense3(x)))
        # Log-probabilities; use torch.exp() to get probabilities.
        return nn.functional.log_softmax(self.dense4(x), dim=1)


class CifarModel(nn.Module):
    """Larger CNN for 32x32 colour images (CIFAR-10), about 0.6M parameters.

    Same layer names as MnistModel, so Grad-CAM (input of pool2) and the
    cascading randomization (dense4 ... conv1A) work unchanged:
        (CONV 3x3 => BN => RELU) * 2 => POOL => DROPOUT   32x32 -> 16x16
        (CONV 3x3 => BN => RELU) * 2 => POOL => DROPOUT   16x16 -> 8x8
        FC => RELU => DROPOUT => FC => LOG_SOFTMAX
    Batch normalisation is used so training converges in about 20 epochs.
    """

    def __init__(self, classes: int = 10) -> None:
        super().__init__()
        self.classes = classes
        self.conv1A = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1A = nn.BatchNorm2d(32)
        self.conv1B = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn1B = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.act = nn.ReLU()
        self.do = nn.Dropout(0.25)
        self.conv2A = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2A = nn.BatchNorm2d(128)
        self.conv2B = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2B = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.dense3 = nn.Linear(128 * 8 * 8, 64)
        self.dense4 = nn.Linear(64, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1A(self.conv1A(x)))
        x = self.do(self.pool1(self.act(self.bn1B(self.conv1B(x)))))
        x = self.act(self.bn2A(self.conv2A(x)))
        x = self.do(self.pool2(self.act(self.bn2B(self.conv2B(x)))))
        x = x.view(x.size(0), -1)
        x = self.do(self.act(self.dense3(x)))
        return nn.functional.log_softmax(self.dense4(x), dim=1)


def build_model(dataset: str) -> nn.Module:
    """An untrained model of the right architecture for a dataset."""
    from .data import info
    cls = CifarModel if info(dataset).get("arch") == "cifar" else MnistModel
    return cls(classes=num_classes(dataset))


def checkpoint_path(dataset: str, seed: int = DEFAULT_SEED) -> Path:
    suffix = "" if seed == DEFAULT_SEED else f"_seed{seed}"
    return CHECKPOINT_DIR / f"{dataset}{suffix}.pt"


def metrics_path(dataset: str, seed: int = DEFAULT_SEED) -> Path:
    suffix = "" if seed == DEFAULT_SEED else f"_seed{seed}"
    return CHECKPOINT_DIR / f"{dataset}{suffix}_metrics.json"


def load_model(dataset: str, device: torch.device | str = "cpu",
               seed: int = DEFAULT_SEED) -> MnistModel:
    """Load a trained model for a dataset in evaluation mode."""
    path = checkpoint_path(dataset, seed)
    if not path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {path}. Train it with: "
            f"python train_models.py --dataset {dataset} --seeds {seed}"
        )
    model = build_model(dataset)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

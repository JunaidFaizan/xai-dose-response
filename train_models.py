"""Train the CNN on one or more datasets and seeds.

Usage:
    python train_models.py --dataset mnist
    python train_models.py --dataset all --seeds 42 1 2
    python train_models.py --dataset decoy_mnist --epochs 3 --force

Each run keeps the epoch with the best validation accuracy and reports test
accuracy once at the end. Seed 42 is saved as checkpoint/<dataset>.pt (the
model used by the dashboard); other seeds as checkpoint/<dataset>_seed<N>.pt.
Metrics go to the matching *_metrics.json file.

For decoy (shortcut) datasets, the test set is also scored with the patch
removed and with the patch moved to another class's position. If the model
relies on the shortcut, accuracy collapses and predictions follow the patch.
"""

import argparse
import json
import time

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from utils import random_seed
from xai.data import DATASETS, info, is_decoy, load_split, num_classes
from xai.models import CHECKPOINT_DIR, build_model, checkpoint_path, metrics_path


def predict_all(model, dataset, device, batch_size=512):
    """Predicted and true labels for a whole dataset."""
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for images, targets in DataLoader(dataset, batch_size=batch_size):
            preds.append(model(images.to(device)).argmax(dim=1).cpu())
            labels.append(torch.as_tensor(targets))
    return torch.cat(preds).numpy(), torch.cat(labels).numpy()


def accuracy(preds, labels):
    return 100.0 * float((preds == labels).mean())


def balanced_accuracy(preds, labels, classes):
    """Mean per-class recall; unlike accuracy it is not inflated by class imbalance."""
    recalls = [(preds[labels == c] == c).mean() for c in range(classes) if (labels == c).any()]
    return 100.0 * float(np.mean(recalls))


def train(dataset, seed, epochs, batch_size, lr, force):
    output = checkpoint_path(dataset, seed)
    if output.exists() and not force:
        print(f"{output.name} exists, skipping (use --force to retrain)")
        return

    random_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classes = num_classes(dataset)
    print(f"\n=== {dataset} | seed {seed} | {epochs} epochs | device {device} ===")

    train_set = load_split(dataset, "train")
    val_set = load_split(dataset, "val")
    test_set = load_split(dataset, "test")
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))

    model = build_model(dataset).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.NLLLoss()  # the model outputs log-probabilities

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    best_val = -1.0
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for images, labels in loader:
            images, labels = images.to(device), torch.as_tensor(labels).to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * labels.size(0)
        val_acc = accuracy(*predict_all(model, val_set, device))
        print(f"epoch {epoch}/{epochs} | loss {running / len(train_set):.4f} | val acc {val_acc:.2f}%")
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), output)
    minutes = (time.perf_counter() - start) / 60

    model.load_state_dict(torch.load(output, map_location=device))
    preds, labels = predict_all(model, test_set, device)
    metrics = {
        "dataset": dataset, "seed": seed, "epochs": epochs,
        "batch_size": batch_size, "learning_rate": lr,
        "best_val_accuracy": round(best_val, 2),
        "test_accuracy": round(accuracy(preds, labels), 2),
        "test_balanced_accuracy": round(balanced_accuracy(preds, labels, classes), 2),
        "training_minutes": round(minutes, 1),
    }

    if is_decoy(dataset):
        none_preds, none_labels = predict_all(model, load_split(dataset, "test", "none"), device)
        conflicting = load_split(dataset, "test", "conflicting")
        conf_preds, conf_labels = predict_all(model, conflicting, device)
        metrics.update({
            "test_accuracy_no_patch": round(accuracy(none_preds, none_labels), 2),
            "test_accuracy_conflicting_patch": round(accuracy(conf_preds, conf_labels), 2),
            # Share of predictions that follow the patch instead of the image.
            "shortcut_reliance": round(accuracy(conf_preds, conflicting.patch_class), 2),
        })

    metrics_path(dataset, seed).write_text(json.dumps(metrics, indent=2))
    print("  " + ", ".join(f"{k}={v}" for k, v in metrics.items()
                           if k.startswith(("test", "shortcut", "training"))))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=list(DATASETS) + ["all"], required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--epochs", type=int, default=None,
                        help="default: the per-dataset value in xai/data.py")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--force", action="store_true", help="overwrite existing checkpoints")
    args = parser.parse_args()

    # "all" skips datasets used only by the dose-response study (cifar10).
    names = [d for d in DATASETS if not info(d).get("dose_only")] if args.dataset == "all" else [args.dataset]
    for name in names:
        for seed in args.seeds:
            train(name, seed, args.epochs or info(name)["epochs"], args.batch_size, args.lr, args.force)


if __name__ == "__main__":
    main()

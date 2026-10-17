"""Generate side-by-side explanation grids for test images.

Usage:
    python generate_explanations.py --dataset mnist
    python generate_explanations.py --dataset fashion_mnist --samples 8

Each row shows one test image followed by the attribution map of every
method. Saved to results/explanations/<dataset>_examples.png.
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from utils import random_seed
from xai.data import DATASETS, class_names, is_decoy, load_split
from xai.explain import METHODS, explain, normalize
from xai.models import load_model
from xai.paths import RESULTS_DIR

OUTPUT_DIR = RESULTS_DIR / "explanations"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=list(DATASETS), required=True)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random_seed(args.seed)
    model = load_model(args.dataset)
    # Decoy datasets: patch pointing to a wrong class, as in the evaluation,
    # so it is visible whether a method finds the patch or the object.
    test_set = load_split(args.dataset, "test", "conflicting" if is_decoy(args.dataset) else "aligned")
    names = class_names(args.dataset)
    indices = torch.randperm(len(test_set))[: args.samples].tolist()

    columns = 1 + len(METHODS)
    fig, axes = plt.subplots(args.samples, columns,
                             figsize=(2.2 * columns, 2.4 * args.samples), squeeze=False)

    for row, index in enumerate(indices):
        image, label = test_set[index]
        x = image.unsqueeze(0)
        with torch.no_grad():
            probs = torch.exp(model(x))[0]
        pred = int(probs.argmax())

        ax = axes[row, 0]
        ax.imshow(image[0], cmap="gray")
        color = "green" if pred == label else "red"
        ax.set_title(f"true: {names[label]}\npred: {names[pred]} ({probs[pred]:.0%})",
                     fontsize=8, color=color)

        # Explain the predicted class: "why did the model say this?"
        for col, (method, (title, _)) in enumerate(METHODS.items(), start=1):
            heatmap = normalize(explain(model, x, pred, method))
            ax = axes[row, col]
            ax.imshow(image[0], cmap="gray")
            ax.imshow(heatmap, cmap="jet", alpha=0.5)
            if row == 0:
                ax.set_title(title, fontsize=9)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"Explanations for the predicted class ({args.dataset})")
    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / f"{args.dataset}_examples.png"
    fig.savefig(output, dpi=120)
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()

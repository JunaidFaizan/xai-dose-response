"""Evaluate every explanation method with every metric, for every trained model.

Usage:
    python evaluate_explanations.py                        # all datasets, seeds 42 1 2
    python evaluate_explanations.py --dataset decoy_mnist --seeds 42
    python evaluate_explanations.py --samples 500 --sanity-samples 100

Per model (dataset + seed) it measures, on the same random test images:
  faithfulness  deletion AUC with black / blurred / inpainted pixels, and
                insertion AUC from a blurred image (see xai/metrics.py)
  localization  (decoy datasets only) relevance mass and rank accuracy of the
                known shortcut patch. Decoy models are evaluated on test images
                whose patch points to a wrong class and that the model
                classified by the patch, so the patch is the true evidence.
  sanity        rank correlation with the original explanation as layers are
                randomized (all pixels and object pixels)

A random attribution map is scored alongside the methods as a baseline.
Results go to results/metrics/<dataset>/seed<N>_{faithfulness,sanity}.csv.
For seed 42 it also saves plots. Run analyze_results.py afterwards for the
statistics and paper tables.
"""

import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from utils import random_seed
from xai.data import DATASETS, info, is_decoy, load_split
from xai.explain import METHODS, explain
from xai.metrics import (
    FAITHFULNESS_METRICS,
    RANDOMIZATION_ORDER,
    area_under_curve,
    cascading_randomization,
    perturbation_curve,
    rank_correlation,
    relevance_mass_accuracy,
    relevance_rank_accuracy,
)
from xai.models import DEFAULT_SEED, checkpoint_path, load_model
from xai.paths import RESULTS_DIR

METRICS_DIR = RESULTS_DIR / "metrics"
STEPS = 20
IMAGE_SEED = 0  # the same test images are used for every model
STAGES = ["original"] + RANDOMIZATION_ORDER
TITLES = {name: title for name, (title, _) in METHODS.items()}
TITLES["random"] = "Random (baseline)"
COLORS = {"saliency": "#1f77b4", "smoothgrad": "#9467bd", "integrated_gradients": "#ff7f0e",
          "grad_cam": "#2ca02c", "occlusion": "#d62728", "random": "#7f7f7f"}


def test_indices(test_set, samples, model=None):
    """Random test images; the same ones for every model of a dataset.

    For decoy datasets (conflicting patch) only images where the model follows
    the patch are kept: for those, the patch is known to be the evidence.
    """
    generator = torch.Generator().manual_seed(IMAGE_SEED)
    order = torch.randperm(len(test_set), generator=generator).tolist()
    if model is None:
        return order[:samples]
    with torch.no_grad():
        preds = torch.cat([model(images).argmax(dim=1)
                           for images, _ in torch.utils.data.DataLoader(test_set, batch_size=512)]).numpy()
    follows = preds == test_set.patch_class
    return [i for i in order if follows[i]][:samples]


def predicted_class(model, x):
    with torch.no_grad():
        return int(model(x).argmax(dim=1))


def run_faithfulness(model, test_set, indices, decoy, keep_curves):
    rows = []
    curves = {m: {name: [] for name in list(METHODS) + ["random"]} for m in FAITHFULNESS_METRICS}
    rng = np.random.default_rng(0)
    for count, index in enumerate(indices, start=1):
        image, label = test_set[index]
        x = image.unsqueeze(0)
        pred = predicted_class(model, x)
        maps = {name: explain(model, x, pred, name) for name in METHODS}
        maps["random"] = rng.random(image.shape[-2:])
        mask = test_set.mask(index) if decoy else None

        for name, attribution in maps.items():
            row = {"index": index, "label": label, "prediction": pred,
                   "correct": pred == label, "method": name}
            for metric, (_, mode, fill, _) in FAITHFULNESS_METRICS.items():
                curve = perturbation_curve(model, x, pred, attribution, mode, fill, STEPS, rng)
                row[metric] = area_under_curve(curve)
                if keep_curves:
                    curves[metric][name].append(curve)
            if decoy:
                row["mass_accuracy"] = relevance_mass_accuracy(attribution, mask)
                row["rank_accuracy"] = relevance_rank_accuracy(attribution, mask, rng)
            rows.append(row)
        if count % 50 == 0 or count == len(indices):
            print(f"  faithfulness: {count}/{len(indices)} images")
    mean_curves = {m: {n: np.mean(c, axis=0) for n, c in d.items() if c} for m, d in curves.items()}
    return pd.DataFrame(rows), mean_curves


def run_sanity(model, test_set, indices):
    inputs = []
    for index in indices:
        image, _ = test_set[index]
        x = image.unsqueeze(0)
        inputs.append((index, x, predicted_class(model, x), image[0].numpy() > 0))

    rows, originals, example = [], {}, {}
    for stage, stage_model in cascading_randomization(model, seed=0):
        for index, x, pred, mask in inputs:
            for name in METHODS:
                # Always explain the ORIGINAL prediction, so only the model changes.
                attribution = explain(stage_model, x, pred, name)
                if stage == "original":
                    originals[(index, name)] = attribution
                if index == indices[0]:
                    example[(stage, name)] = attribution
                original = originals[(index, name)]
                rows.append({
                    "index": index, "method": name, "stage": stage,
                    "corr_all": rank_correlation(original, attribution),
                    "corr_object": rank_correlation(original, attribution, mask),
                })
    print("  sanity: done")
    return pd.DataFrame(rows), example


def plot_curves(mean_curves, dataset, path):
    fig, axes = plt.subplots(1, len(FAITHFULNESS_METRICS), figsize=(4.2 * len(FAITHFULNESS_METRICS), 3.8),
                             sharey=True)
    fractions = np.linspace(0, 1, STEPS + 1)
    for ax, (metric, (title, mode, _, _)) in zip(axes, FAITHFULNESS_METRICS.items()):
        for name, curve in mean_curves[metric].items():
            ax.plot(fractions, curve, "--" if name == "random" else "-", color=COLORS[name],
                    label=f"{TITLES[name]} ({area_under_curve(curve):.3f})")
        ax.set_title(f"{title}\n({'lower' if mode == 'deletion' else 'higher'} AUC is better)", fontsize=10)
        ax.set_xlabel(f"Fraction of pixels {'removed' if mode == 'deletion' else 'inserted'}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Mean probability of predicted class")
    fig.suptitle(f"Perturbation curves ({dataset}, seed {DEFAULT_SEED})")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_sanity_curve(sanity, dataset, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, (column, label) in zip(axes, [("corr_all", "all pixels"), ("corr_object", "object pixels only")]):
        for name in METHODS:
            means = sanity[sanity.method == name].groupby("stage")[column].mean()
            ax.plot(STAGES, [means[s] for s in STAGES], marker="o", color=COLORS[name], label=TITLES[name])
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_title(label)
        ax.set_xlabel("Layers randomized (cumulative, output layer first)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Rank correlation with original")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Sanity check ({dataset}, seed {DEFAULT_SEED}): correlation should fall towards 0")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_sanity_examples(example, image, dataset, path):
    fig, axes = plt.subplots(len(METHODS), len(STAGES) + 1,
                             figsize=(1.7 * (len(STAGES) + 1), 1.8 * len(METHODS)))
    for row, name in enumerate(METHODS):
        axes[row, 0].imshow(image[0], cmap="gray")
        axes[row, 0].set_ylabel(TITLES[name], fontsize=8)
        for col, stage in enumerate(STAGES, start=1):
            heatmap = np.abs(example[(stage, name)])
            peak = heatmap.max()
            axes[row, col].imshow(heatmap / peak if peak > 0 else heatmap, cmap="hot")
            if row == 0:
                axes[row, col].set_title(stage, fontsize=8)
    axes[0, 0].set_title("input", fontsize=8)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Explanations as layers are randomized ({dataset})")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def evaluate(dataset, seed, samples, sanity_samples):
    if not checkpoint_path(dataset, seed).exists():
        print(f"\n--- {dataset} seed {seed}: no checkpoint, skipped ---")
        return
    print(f"\n=== {dataset} | seed {seed} ===")
    start = time.perf_counter()
    random_seed(0)
    model = load_model(dataset, seed=seed)
    decoy = is_decoy(dataset)
    test_set = load_split(dataset, "test", "conflicting" if decoy else "aligned")
    indices = test_indices(test_set, samples, model if decoy else None)
    if decoy:
        print(f"  using {len(indices)} conflicting-patch images the model classified by the patch")
    primary = seed == DEFAULT_SEED

    faithfulness, mean_curves = run_faithfulness(model, test_set, indices, decoy, primary)
    sanity, example = run_sanity(model, test_set, indices[:sanity_samples])
    for table in (faithfulness, sanity):
        table.insert(0, "seed", seed)
        table.insert(0, "dataset", dataset)

    out = METRICS_DIR / dataset
    out.mkdir(parents=True, exist_ok=True)
    faithfulness.to_csv(out / f"seed{seed}_faithfulness.csv", index=False)
    sanity.to_csv(out / f"seed{seed}_sanity.csv", index=False)
    if primary:
        plot_curves(mean_curves, dataset, out / "perturbation_curves.png")
        plot_sanity_curve(sanity, dataset, out / "sanity_curve.png")
        plot_sanity_examples(example, test_set[indices[0]][0], dataset, out / "sanity_examples.png")

    means = faithfulness.groupby("method")[list(FAITHFULNESS_METRICS)].mean().round(3)
    print(means.reindex(list(METHODS) + ["random"]).to_string())
    print(f"  saved to {out} ({(time.perf_counter() - start) / 60:.1f} min)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=list(DATASETS) + ["all"], default="all")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    parser.add_argument("--samples", type=int, default=200, help="test images per model")
    parser.add_argument("--sanity-samples", type=int, default=50,
                        help="test images for the sanity check (the first of --samples)")
    args = parser.parse_args()

    names = [d for d in DATASETS if not info(d).get("dose_only")] if args.dataset == "all" else [args.dataset]
    for name in names:
        for seed in args.seeds:
            evaluate(name, seed, args.samples, min(args.sanity_samples, args.samples))


if __name__ == "__main__":
    main()

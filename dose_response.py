"""Dose-response meta-evaluation: do XAI metrics notice how much a model relies on a shortcut?

Usage:
    python dose_response.py all              # train, evaluate, analyze (about 2 hours on a CPU)
    python dose_response.py train
    python dose_response.py evaluate --samples 100
    python dose_response.py analyze

Design
    For each base dataset (MNIST, PneumoniaMNIST) and shortcut type (a local
    3x3 patch, or a diffuse texture of scattered bright pixels), models are
    trained at a signed "dose" d in DOSES:
        d > 0   a share d of training images is replaced by an image of
                another class (label and shortcut kept): the image becomes
                less reliable, so the model relies on the shortcut MORE
        d < 0   a share |d| of training shortcuts points to a wrong class:
                the shortcut becomes less reliable, so the model relies on it LESS
    Both are needed because the shortcut is easier to learn than the image,
    so with d = 0 models already rely on it heavily. |d| is capped at 0.4 so
    neither cue ever points away from the label on the binary task.

    Reliance is measured behaviourally: on test images whose shortcut points
    to a wrong class, the share of predictions that follow the shortcut.

    Oracle explanations: for each aligned test image (shortcut and image agree)
    two fixed maps are scored by every metric, explaining the true label:
        shortcut map   the shortcut pixels
        object map     the image's own pixel values (shortcut pixels removed)
    Preference = share of images where the metric rates the shortcut map as
    the better explanation. A metric that responds to the model's reasoning
    should prefer the object map when reliance is low and the shortcut map
    when reliance is high, switching near 50% reliance.

    Scores per metric (over all models of one base and shortcut type):
        responsiveness   Spearman correlation between reliance and preference
        decision acc.    share of models where (preference > 0.5) equals (reliance > 50%)
        switch point     reliance at which preference crosses 0.5 (ideal: 50%)

    The five attribution methods are also run, and the share of their positive
    attribution on the shortcut is correlated with reliance.

    The model-randomization sanity check is not applied to oracle maps: they do
    not depend on the model, so the check cannot score them at all.

Oxford-IIIT Pet (pet_resnet18, run on a Kaggle GPU via kaggle/dose_response_pet.ipynb)
    Cat vs dog photos at 128x128 (data/pet128/*.npz, made by prepare_pet.py),
    a ResNet-18 trained from scratch for 30 epochs, a 12x12 white corner patch.
    The object map is the real segmentation mask (border band counted as pet);
    the size-matched map is the k mask pixels nearest the mask centroid. Blur
    (sigma 16, kernel 97), inpainting (1600 steps) and occlusion (16x16, stride
    8) are scaled to the image size. PET_DOSES x PET_SEEDS = 18 models:
        python dose_response.py all --bases pet_resnet18 --shortcuts patch \\
            --doses -0.5 -0.4 -0.3 -0.2 -0.1 0 --seeds 42 1 2

Output
    checkpoint/dose/<base>_<shortcut>_dose<+NNN>_seed<N>.pt (+ _metrics.json)
    results/dose_response/scores_oracle.csv, scores_methods.csv  per-image scores
    results/dose_response/models.csv, preference.csv, responsiveness.csv,
        method_responsiveness.csv, summary.md, *.png
"""

import argparse
import itertools
import json
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata, spearmanr
from torch import nn, optim
from torch.utils.data import DataLoader

from train_models import accuracy, predict_all
from utils import random_seed
from xai.data import SHORTCUT_TYPES, info, num_classes, shortcut_split
from xai.explain import METHODS, explain
from xai.metrics import (BLUR_KERNEL, BLUR_SIGMA, FAITHFULNESS_METRICS, INPAINT_ITERATIONS,
                         area_under_curve, perturbation_curve, relevance_mass_accuracy)
from xai.models import build_model
from xai.paths import CHECKPOINT_DIR, RESULTS_DIR

# Base dataset: training epochs. cifar10 is opt-in (--bases cifar10): its models
# take 20-40 min each on a CPU, so it is not part of the default run.
BASES = {"mnist": 3, "pneumonia_mnist": 20, "cifar10": 10, "cifar10_resnet18": 15,
         "pet_resnet18": 30}
# CIFAR-10 learns the patch almost at once (96.7% validation accuracy after one
# epoch at d = 0), so its range reaches further down. With 10 classes a
# shortcut noise of 0.8 still points to the true class most often (20% vs
# about 9% for each wrong class), so the doses stay valid.
CIFAR_DOSES = [-0.8, -0.6, -0.4, -0.25, -0.1, 0.0]
CIFAR_SEEDS = [42, 1]
# Oxford-IIIT Pet (cat vs dog): a binary task, so |d| <= 0.4. CIFAR-10 learned
# the patch strongly at d = 0, so the doses are negative. They may be adjusted
# after the first Kaggle model (kaggle/README.md).
# Adjusted after the first Kaggle model (dose -0.4, seed 42: reliance 24.8%):
# -0.05 dropped and -0.5 added. At -0.5 half the training shortcuts point to
# the wrong class, so the shortcut carries no information about the label,
# which gives the low-reliance end of the range.
PET_DOSES = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0]
PET_SEEDS = [42, 1, 2]
DEFAULT_BASES = ["mnist", "pneumonia_mnist"]
# Signed dose: negative = that share of training shortcuts point to a wrong
# class (less reliance); positive = that share of training images are swapped
# for another class (more reliance). Capped at 0.4 so that neither cue ever
# points away from the label on the binary task.
# The small doses around 0 fill the range where reliance changes fastest
# (the texture shortcut on MNIST is learned almost all-or-nothing).
DOSES = [-0.4, -0.25, -0.1, -0.05, 0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.4]
SEEDS = [42, 1, 2]


def dose_params(dose):
    """(swap_fraction, shortcut_noise) for a signed dose."""
    return max(dose, 0.0), max(-dose, 0.0)


def dose_label(dose):
    """'+010' for 0.1, '+0p5' for 0.005 (whole percentages keep the original format)."""
    percent = dose * 100
    if abs(percent - round(percent)) < 1e-9:
        return f"{round(percent):+04d}"
    return f"{percent:+.1f}".replace(".", "p")
MODEL_DIR = CHECKPOINT_DIR / "dose"
# A GPU is used when available (e.g. on Kaggle); otherwise the CPU, as before.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = RESULTS_DIR / "dose_response"
STEPS = 20
IMAGE_SEED = 0
METRIC_TITLES = {name: title for name, (title, *_) in FAITHFULNESS_METRICS.items()}
METHOD_TITLES = {name: title for name, (title, _) in METHODS.items()}


def model_name(base, shortcut, dose, seed):
    return f"{base}_{shortcut}_dose{dose_label(dose)}_seed{seed}"


def configurations(args):
    return list(itertools.product(args.bases, args.shortcuts, args.doses, args.seeds))


def metric_options(base):
    """Blur and inpainting settings for a base dataset. Datasets without their
    own values (all 28px and 32px ones) get the module defaults, unchanged."""
    entry = info(base)
    return {"blur_kernel": entry.get("blur_kernel", BLUR_KERNEL),
            "blur_sigma": entry.get("blur_sigma", BLUR_SIGMA),
            "inpaint_iterations": entry.get("inpaint_iterations", INPAINT_ITERATIONS)}


def method_options(base, method):
    """Extra settings for an attribution method (occlusion window and stride;
    4 and 2 pixels unless the dataset sets its own)."""
    if method != "occlusion":
        return {}
    entry = info(base)
    return {"window": entry.get("occlusion_window", 4), "stride": entry.get("occlusion_stride", 2)}


# ---------------------------------------------------------------- training

def train_one(base, shortcut, dose, seed, force):
    name = model_name(base, shortcut, dose, seed)
    swap, noise = dose_params(dose)
    path = MODEL_DIR / f"{name}.pt"
    # The metrics JSON is written only after training finishes, so it marks a
    # complete model; a .pt without it is a partial run and is retrained.
    if (MODEL_DIR / f"{name}_metrics.json").exists() and not force:
        return
    random_seed(seed)
    epochs = BASES[base]
    train_set = shortcut_split(base, shortcut, "train", swap, noise)
    val_set = shortcut_split(base, shortcut, "val", swap, noise)
    # On a GPU, load batches in background workers so the GPU isn't left waiting.
    workers = {"num_workers": 2, "pin_memory": True, "persistent_workers": True} if DEVICE.type == "cuda" else {}
    loader = DataLoader(train_set, batch_size=128, shuffle=True,
                        generator=torch.Generator().manual_seed(seed), **workers)
    model = build_model(base).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.NLLLoss()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best, start = -1.0, time.perf_counter()
    for _ in range(epochs):
        model.train()
        for images, labels in loader:
            images, labels = images.to(DEVICE), torch.as_tensor(labels).to(DEVICE)
            optimizer.zero_grad()
            loss_fn(model(images), labels).backward()
            optimizer.step()
        val_acc = accuracy(*predict_all(model, val_set, DEVICE))
        if val_acc > best:
            best = val_acc
            torch.save(model.state_dict(), path)
        if base.startswith(("cifar10", "pet")):  # slow models: show progress per epoch
            print(f"  {name} epoch {_ + 1}/{epochs}: val acc {val_acc:.1f}%", flush=True)

    model.load_state_dict(torch.load(path, map_location=DEVICE))
    conflicting = shortcut_split(base, shortcut, "test", mode="conflicting")
    conf_preds, conf_labels = predict_all(model, conflicting, DEVICE)
    metrics = {
        "base": base, "shortcut": shortcut, "dose": dose, "swap_fraction": swap,
        "shortcut_noise": noise, "seed": seed, "epochs": epochs,
        "best_val_accuracy": round(best, 2),
        "test_accuracy": round(accuracy(*predict_all(model, shortcut_split(base, shortcut, "test"), DEVICE)), 2),
        "test_accuracy_no_shortcut": round(accuracy(*predict_all(
            model, shortcut_split(base, shortcut, "test", mode="none"), DEVICE)), 2),
        "test_accuracy_conflicting": round(accuracy(conf_preds, conf_labels), 2),
        "reliance": round(accuracy(conf_preds, conflicting.patch_class), 2),
        "training_minutes": round((time.perf_counter() - start) / 60, 2),
    }
    (MODEL_DIR / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"{name}: reliance {metrics['reliance']}% | acc {metrics['test_accuracy']}% | "
          f"no shortcut {metrics['test_accuracy_no_shortcut']}% | {metrics['training_minutes']} min", flush=True)


def train(args):
    for config in configurations(args):
        train_one(*config, args.force)


# -------------------------------------------------------------- evaluation

def load(base, shortcut, dose, seed):
    model = build_model(base)
    model.load_state_dict(torch.load(MODEL_DIR / f"{model_name(base, shortcut, dose, seed)}.pt",
                                     map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def size_matched(object_values, k):
    """Control oracle: a binary map of the k brightest object pixels, where k is
    the number of shortcut pixels. It has the same size and form as the shortcut
    map, so a preference between them cannot come from size alone."""
    flat = object_values.reshape(-1)
    top = np.argsort(-flat, kind="stable")[:k]
    matched = np.zeros_like(flat)
    matched[top] = 1.0
    return matched.reshape(object_values.shape)


def compact_matched(object_values, k):
    """Size-matched control for binary object masks (Oxford-IIIT Pet): the k
    object pixels nearest the object's centroid, a compact region like the
    shortcut patch. A binary mask has no "brightest" pixels, so size_matched
    would pick k pixels in an arbitrary tie order. Ties in distance are broken
    in row-major order. Fewer than k object pixels: all of them are used."""
    rows, cols = np.nonzero(object_values > 0)
    matched = np.zeros(object_values.shape, dtype=np.float64)
    if len(rows) == 0:
        return matched
    distance = (rows - rows.mean()) ** 2 + (cols - cols.mean()) ** 2
    nearest = np.argsort(distance, kind="stable")[:k]
    matched[rows[nearest], cols[nearest]] = 1.0
    return matched


def evaluate_one(base, shortcut, dose, seed, samples):
    model = load(base, shortcut, dose, seed)
    test_set = shortcut_split(base, shortcut, "test")  # aligned: shortcut and image agree
    generator = torch.Generator().manual_seed(IMAGE_SEED)
    indices = torch.randperm(len(test_set), generator=generator)[:samples].tolist()
    rng = np.random.default_rng(0)
    oracle_rows, method_rows = [], []
    config = {"base": base, "shortcut": shortcut, "dose": dose, "seed": seed}
    matcher = compact_matched if info(base).get("matched") == "compact" else size_matched
    options = metric_options(base)

    for index in indices:
        image, label = test_set[index]
        x = image.unsqueeze(0).to(DEVICE)
        mask = test_set.mask(index)
        object_values = test_set.object_map(index)
        maps = {"shortcut": mask.astype(np.float64), "object": object_values,
                "object_matched": matcher(object_values, int(mask.sum()))}
        row = dict(config, index=index, label=label)
        for metric, (_, mode, fill, _) in FAITHFULNESS_METRICS.items():
            for which, attribution in maps.items():
                curve = perturbation_curve(model, x, label, attribution, mode, fill, STEPS, rng, **options)
                row[f"{metric}_{which}"] = area_under_curve(curve)
        oracle_rows.append(row)

        for method in METHODS:
            attribution = explain(model, x, label, method, **method_options(base, method))
            method_rows.append(dict(config, index=index, method=method,
                                    shortcut_mass=relevance_mass_accuracy(attribution, mask)))
    return oracle_rows, method_rows


def evaluate(args):
    """Scores every model. Each model's scores are cached in results/dose_response/
    per_model/, so an interrupted run resumes where it stopped (--force redoes all)."""
    cache = OUT / "per_model"
    cache.mkdir(parents=True, exist_ok=True)
    for config in configurations(args):
        name = model_name(*config)
        oracle_path, methods_path = cache / f"{name}_oracle.csv", cache / f"{name}_methods.csv"
        if oracle_path.exists() and methods_path.exists() and not args.force:
            continue
        start = time.perf_counter()
        rows_o, rows_m = evaluate_one(*config, args.samples)
        pd.DataFrame(rows_m).to_csv(methods_path, index=False)
        pd.DataFrame(rows_o).to_csv(oracle_path, index=False)  # written last: marks the model done
        print(f"evaluated {name} ({time.perf_counter() - start:.0f} s)", flush=True)
    combine(args)


def placebo_map(test_set, index, label):
    """Placebo oracle: the same shape as the shortcut map, at a place the image
    has no shortcut (a wrong class's patch position or texture pattern).
    It never carries evidence for the true label, at any reliance."""
    classes = len(test_set.positions)
    rng = np.random.default_rng(index)
    wrong = (label + int(rng.integers(1, classes))) % classes
    return test_set.mask_for_class(wrong).astype(np.float64)


def random_map(test_set, index, label):
    """Random control: independent uniform scores for every pixel, i.e. a random
    pixel ordering with no ties. If the placebo behaves like this map, the
    placebo's effect comes from the random ordering of its tied pixels."""
    rng = np.random.default_rng(10_000 + index)
    return rng.random((test_set.size, test_set.size))


# Control maps scored by `dose_response.py placebo`: name -> (builder, rng seed).
CONTROL_MAPS = {"placebo": (placebo_map, 1), "random": (random_map, 2)}


def evaluate_control_one(base, shortcut, dose, seed, samples, control):
    """Score one control map with every metric on the same images as evaluate_one."""
    builder, rng_seed = CONTROL_MAPS[control]
    model = load(base, shortcut, dose, seed)
    test_set = shortcut_split(base, shortcut, "test")
    generator = torch.Generator().manual_seed(IMAGE_SEED)
    indices = torch.randperm(len(test_set), generator=generator)[:samples].tolist()
    rng = np.random.default_rng(rng_seed)
    rows = []
    for index in indices:
        image, label = test_set[index]
        x = image.unsqueeze(0).to(DEVICE)
        attribution = builder(test_set, index, label)
        row = {"base": base, "shortcut": shortcut, "dose": dose, "seed": seed, "index": index}
        for metric, (_, mode, fill, _) in FAITHFULNESS_METRICS.items():
            curve = perturbation_curve(model, x, label, attribution, mode, fill, STEPS, rng,
                                       **metric_options(base))
            row[f"{metric}_{control}"] = area_under_curve(curve)
        rows.append(row)
    return rows


def evaluate_placebo_one(base, shortcut, dose, seed, samples):
    return evaluate_control_one(base, shortcut, dose, seed, samples, "placebo")


def all_cached_models():
    """Every model that has a finished checkpoint (metrics JSON) and oracle scores."""
    cache = OUT / "per_model"
    names = []
    for path in sorted(MODEL_DIR.glob("*_metrics.json")):
        run = json.loads(path.read_text())
        config = (run["base"], run["shortcut"], run["dose"], run["seed"])
        if (cache / f"{model_name(*config)}_oracle.csv").exists():
            names.append(config)
    return names


def placebo(args):
    """Placebo and random controls for every evaluated model (all datasets),
    cached per model. Already computed controls are skipped (--force redoes them)."""
    cache = OUT / "per_model"
    for config in all_cached_models():
        name = model_name(*config)
        for control in CONTROL_MAPS:
            path = cache / f"{name}_{control}.csv"
            if path.exists() and not args.force:
                continue
            start = time.perf_counter()
            pd.DataFrame(evaluate_control_one(*config, args.samples, control)).to_csv(path, index=False)
            print(f"{control} {name} ({time.perf_counter() - start:.0f} s)", flush=True)
    combine(args)


def combine(args):
    """Check the selected configurations are evaluated, then merge EVERY cached
    model (so evaluating cifar10 separately keeps the MNIST/PneumoniaMNIST results)."""
    cache = OUT / "per_model"
    selected = [model_name(*config) for config in configurations(args)] if args is not None else []
    missing = [n for n in selected if not (cache / f"{n}_oracle.csv").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} models not evaluated yet, e.g. {missing[0]}. Run evaluate again.")
    names = sorted(p.name[:-len("_oracle.csv")] for p in cache.glob("*_oracle.csv")
                   if (MODEL_DIR / f"{p.name[:-len('_oracle.csv')]}.pt").exists())
    stale = [n for n in names if "deletion_zero_object_matched" not in pd.read_csv(cache / f"{n}_oracle.csv", nrows=0)]
    if stale:
        raise SystemExit(f"{len(stale)} cached results predate the size-matched control. Run: evaluate --force")
    oracle = pd.concat([pd.read_csv(cache / f"{n}_oracle.csv") for n in names])
    # Control scores are added when every model has them (dose_response.py placebo).
    for control in CONTROL_MAPS:
        if all((cache / f"{n}_{control}.csv").exists() for n in names):
            scores = pd.concat([pd.read_csv(cache / f"{n}_{control}.csv") for n in names])
            oracle = oracle.merge(scores, on=["base", "shortcut", "dose", "seed", "index"], how="left")
    oracle.to_csv(OUT / "scores_oracle.csv", index=False)
    pd.concat([pd.read_csv(cache / f"{n}_methods.csv") for n in names]).to_csv(OUT / "scores_methods.csv", index=False)


# ---------------------------------------------------------------- analysis

def load_models():
    runs = [json.loads(p.read_text()) for p in sorted(MODEL_DIR.glob("*_metrics.json"))]
    return pd.DataFrame(runs)


TIE_TOLERANCE = 1e-4  # AUC difference treated as a tie
# Pairwise comparisons: (map A, map B, label, figure suffix, ideal behaviour).
# Preference = share of images where the metric rates map A above map B.
COMPARISONS = [
    ("shortcut", "object", "shortcut vs full object", "",
     "rise from 0 to 1, crossing 0.5 at 50% reliance"),
    ("shortcut", "object_matched", "shortcut vs size-matched object", "_size_matched",
     "rise from 0 to 1, crossing 0.5 at 50% reliance"),
    ("placebo", "object", "placebo vs full object", "_placebo_vs_object",
     "low at low reliance; a rise at high reliance can be correct, since the object map then explains nothing"),
    ("shortcut", "placebo", "shortcut vs placebo", "_shortcut_vs_placebo",
     "about 0.5 at 0% reliance, rising to 1"),
    # Random-map controls: does the placebo act like a random pixel ordering?
    ("random", "object", "random vs full object", "_random_vs_object",
     "low at low reliance (the object explains the model better than random)"),
    ("placebo", "random", "placebo vs random", "_placebo_vs_random",
     "about 0.5 at every reliance if the placebo carries no information"),
    ("shortcut", "random", "shortcut vs random", "_shortcut_vs_random",
     "about 0.5 at 0% reliance, rising to 1"),
]


def preference_table(oracle, first="shortcut", second="object"):
    """Share of images where each metric rates map `first` above map `second`."""
    rows = []
    keys = ["base", "shortcut", "dose", "seed"]
    for key, group in oracle.groupby(keys):
        row = dict(zip(keys, key))
        for metric, (_, _, _, higher) in FAITHFULNESS_METRICS.items():
            shortcut, obj = group[f"{metric}_{first}"], group[f"{metric}_{second}"]
            better = shortcut > obj if higher else shortcut < obj
            ties = np.isclose(shortcut, obj, atol=TIE_TOLERANCE)
            better = better & ~ties
            # Ties (both maps change the probability equally, usually because the
            # model is saturated) count as half; their share is reported.
            row[metric] = float(better.mean() + 0.5 * ties.mean())
            row[f"{metric}_ties"] = float(ties.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def switch_point(reliance, preference):
    """Reliance at which preference crosses 0.5, by linear interpolation of the
    mean preference per reliance level (NaN if it never crosses)."""
    frame = pd.DataFrame({"r": reliance, "p": preference}).sort_values("r")
    r, p = frame.r.to_numpy(), frame.p.to_numpy()
    for i in range(len(r) - 1):
        if (p[i] - 0.5) * (p[i + 1] - 0.5) <= 0 and p[i] != p[i + 1]:
            return float(r[i] + (0.5 - p[i]) * (r[i + 1] - r[i]) / (p[i + 1] - p[i]))
    return float("nan")


def responsiveness(pref):
    rows = []
    for (base, shortcut), group in pref.groupby(["base", "shortcut"]):
        levels = group.groupby("dose")[["reliance"] + list(FAITHFULNESS_METRICS)].mean()
        for metric in FAITHFULNESS_METRICS:
            rho, p = spearmanr(group.reliance, group[metric])
            decisions = (group[metric] > 0.5) == (group.reliance > 50)
            rows.append({
                "base": base, "shortcut": shortcut, "metric": METRIC_TITLES[metric],
                "responsiveness_rho": rho, "p_value": p, "models": len(group),
                "decision_accuracy": float(decisions.mean()),
                "tie_rate": float(group[f"{metric}_ties"].mean()),
                "switch_point_reliance": switch_point(levels.reliance, levels[metric]),
                "preference_low_reliance": float(levels[metric].iloc[0]),
                "preference_high_reliance": float(levels[metric].iloc[-1]),
                "reliance_range": f"{levels.reliance.min():.0f}-{levels.reliance.max():.0f}%",
            })
    return pd.DataFrame(rows).round(3)


BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 0


def _spearman(x, y):
    """Spearman correlation without p-value (fast enough for the bootstrap)."""
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def bootstrap_intervals(pref, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED):
    """95% bootstrap intervals for the switch point and responsiveness rho.

    The unit of resampling is the trained model: within each dose, the seeds
    are drawn with replacement, so the dose design stays fixed and only the
    seed-to-seed variation (the main source of noise near d = 0) is resampled.

    switch_found_share is the share of resamples in which preference crosses
    0.5 at all; the interval is computed over those resamples only, and is
    left empty if fewer than half of them cross.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for (base, shortcut), group in pref.groupby(["base", "shortcut"]):
        by_dose = [g for _, g in group.groupby("dose")]
        for metric in FAITHFULNESS_METRICS:
            reliance = [g.reliance.to_numpy() for g in by_dose]
            values = [g[metric].to_numpy() for g in by_dose]
            switches, rhos = [], []
            for _ in range(reps):
                picks = [rng.integers(0, len(r), size=len(r)) for r in reliance]
                r_all = np.concatenate([r[i] for r, i in zip(reliance, picks)])
                v_all = np.concatenate([v[i] for v, i in zip(values, picks)])
                level_r = [r[i].mean() for r, i in zip(reliance, picks)]
                level_v = [v[i].mean() for v, i in zip(values, picks)]
                switches.append(switch_point(level_r, level_v))
                rhos.append(_spearman(r_all, v_all))
            switches, rhos = np.array(switches), np.array(rhos)
            found = ~np.isnan(switches)
            share = float(found.mean())
            low, high = (np.percentile(switches[found], [2.5, 97.5]) if share >= 0.5
                         else (np.nan, np.nan))
            valid = rhos[~np.isnan(rhos)]
            rho_low, rho_high = np.percentile(valid, [2.5, 97.5]) if len(valid) else (np.nan, np.nan)
            rows.append({"base": base, "shortcut": shortcut, "metric": METRIC_TITLES[metric],
                         "switch_ci_low": low, "switch_ci_high": high,
                         "switch_found_share": share,
                         "rho_ci_low": rho_low, "rho_ci_high": rho_high})
    return pd.DataFrame(rows).round(3)


def method_responsiveness(methods, models):
    mass = methods.groupby(["base", "shortcut", "dose", "seed", "method"]).shortcut_mass.mean()
    mass = mass.reset_index().merge(models, on=["base", "shortcut", "dose", "seed"])
    rows = []
    for (base, shortcut, method), group in mass.groupby(["base", "shortcut", "method"]):
        rho, p = spearmanr(group.reliance, group.shortcut_mass)
        rows.append({"base": base, "shortcut": shortcut, "method": METHOD_TITLES[method],
                     "responsiveness_rho": rho, "p_value": p,
                     "mass_low_reliance": group[group.dose == group.dose.min()].shortcut_mass.mean(),
                     "mass_high_reliance": group[group.dose == group.dose.max()].shortcut_mass.mean()})
    return pd.DataFrame(rows).round(3), mass


def plot_panels(frame, value_columns, titles, ylabel, path, title, line=0.5):
    groups = list(frame.groupby(["base", "shortcut"]))
    fig, axes = plt.subplots(1, len(groups), figsize=(4.3 * len(groups), 3.9), squeeze=False, sharey=True)
    for ax, ((base, shortcut), group) in zip(axes[0], groups):
        for column in value_columns:
            levels = group.groupby("dose")[["reliance", column]].mean().sort_values("reliance")
            ax.scatter(group.reliance, group[column], s=10, alpha=0.35)
            ax.plot(levels.reliance, levels[column], marker="o", label=titles[column])
        if line is not None:
            ax.axhline(line, color="gray", linewidth=0.8)
            ax.axvline(50, color="gray", linewidth=0.8, linestyle=":")
        ax.set_title(f"{base}, {shortcut} shortcut", fontsize=10)
        ax.set_xlabel("Measured reliance on shortcut (%)")
        ax.set_xlim(-2, 102)
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel(ylabel)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def markdown(frame):
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "–" if np.isnan(v) else f"{v:.3f}"
        return str(v)
    lines = ["| " + " | ".join(frame.columns) + " |", "|" + "|".join("---" for _ in frame.columns) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join(lines)


def analyze(_args=None):
    # Rebuild the combined score files from every cached model first, so models
    # added from elsewhere (e.g. the Kaggle ResNet-18 results) are included.
    combine(None)
    models = load_models()
    oracle = pd.read_csv(OUT / "scores_oracle.csv")
    methods = pd.read_csv(OUT / "scores_methods.csv")
    keys = ["base", "shortcut", "dose", "seed"]

    prefs, resps = [], []
    for first, second, label, suffix, ideal in COMPARISONS:
        if f"deletion_zero_{first}" not in oracle or f"deletion_zero_{second}" not in oracle:
            continue  # placebo not computed yet
        pref = preference_table(oracle, first, second).merge(models[keys + ["reliance"]], on=keys)
        resp = responsiveness(pref).merge(bootstrap_intervals(pref), on=["base", "shortcut", "metric"])
        prefs.append(pref.assign(comparison=label))
        resps.append(resp.assign(comparison=label))
        plot_panels(pref, list(FAITHFULNESS_METRICS), METRIC_TITLES,
                    f"Preference for {first} map", OUT / f"metric_dose_response{suffix}.png",
                    f"{label}: preference as reliance grows (ideal: {ideal})")
    pref, resp = pd.concat(prefs), pd.concat(resps)
    method_resp, mass = method_responsiveness(methods, models[keys + ["reliance"]])

    models.to_csv(OUT / "models.csv", index=False)
    pref.round(4).to_csv(OUT / "preference.csv", index=False)
    resp.to_csv(OUT / "responsiveness.csv", index=False)
    method_resp.to_csv(OUT / "method_responsiveness.csv", index=False)
    mass = mass.pivot_table(index=keys + ["reliance"], columns="method", values="shortcut_mass").reset_index()
    plot_panels(mass, list(METHODS), METHOD_TITLES, "Share of positive attribution on shortcut",
                OUT / "method_dose_response.png", "Do explanation methods follow the shortcut as reliance grows?",
                line=None)

    level = models.groupby(["base", "shortcut", "dose"])[
        ["reliance", "test_accuracy", "test_accuracy_no_shortcut"]].agg(["mean", "std"]).round(2)
    level.columns = [f"{a}_{b}" for a, b in level.columns]
    lines = ["# Dose-response results", "", "Generated by `dose_response.py analyze`.", "",
             "## Reliance per dose (mean and std over seeds, %)", "", markdown(level.reset_index()), "",
             "## Metric responsiveness to reliance (oracle maps)", "",
             "responsiveness_rho: Spearman correlation between reliance and preference for the shortcut map "
             "(ideal: close to 1). decision_accuracy: share of models where the preferred map matches whether "
             "reliance is above 50%. switch_point_reliance: reliance where preference crosses 0.5 (ideal: 50). "
             "comparison: which two oracle maps are compared (preference = share of images where the metric "
             "rates the first above the second). Size-matched object: a binary map of the object's brightest "
             "pixels, as many as the shortcut. Placebo: the shortcut's shape at a wrong class's position, which "
             "never carries evidence. Random: independent uniform scores per pixel (a fully random ordering). "
             "The placebo's non-patch pixels are tied and ordered randomly, so it behaves almost like the random "
             "map ('placebo vs random' stays near 0.5). 'Placebo vs full object' and 'random vs full object' "
             "should be LOW at low reliance; a rise at high reliance is correct, because the object map then "
             "explains the model worse than a random ordering. "
             f"switch_ci_low/high and rho_ci_low/high: 95% bootstrap intervals ({BOOTSTRAP_REPS} resamples of "
             "the seeds within each dose). switch_found_share: share of resamples in which preference crosses "
             "0.5 at all (the switch interval is left empty below 0.5).", "",
             markdown(resp[["comparison"] + [c for c in resp.columns if c != "comparison"]]), "", "## Method responsiveness (share of attribution on the shortcut vs reliance)", "",
             markdown(method_resp), ""]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def keep_awake():
    """On Windows, ask the system not to sleep while this process runs.
    This doesn't change power settings and ends automatically when the process
    exits. Closing the laptop lid can still cause sleep, depending on settings."""
    if sys.platform == "win32":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["train", "evaluate", "placebo", "analyze", "all"])
    parser.add_argument("--bases", nargs="+", default=DEFAULT_BASES, choices=list(BASES))
    parser.add_argument("--shortcuts", nargs="+", default=list(SHORTCUT_TYPES), choices=list(SHORTCUT_TYPES))
    parser.add_argument("--doses", nargs="+", type=float, default=DOSES)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--samples", type=int, default=100, help="test images per model")
    parser.add_argument("--force", action="store_true", help="retrain existing models")
    args = parser.parse_args()
    for base in args.bases:
        info(base)  # validates the name
    keep_awake()
    if args.step in ("train", "all"):
        train(args)
    if args.step in ("evaluate", "all"):
        evaluate(args)
    if args.step in ("placebo", "all"):
        placebo(args)
    if args.step in ("analyze", "all"):
        analyze(args)


if __name__ == "__main__":
    main()

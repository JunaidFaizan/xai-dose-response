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
    so with d = 0 models already rely on it heavily. For MNIST and
    PneumoniaMNIST (DOSES) |d| is capped at 0.4 so neither cue ever points
    away from the label on the binary task. CIFAR_DOSES reach -0.8 (with 10
    classes the shortcut still points to the true class most often), and
    PET_DOSES include -0.5, where the shortcut carries no label information.

    Reliance is measured behaviourally: on test images whose shortcut points
    to a wrong class, the share of predictions that follow the shortcut.

    Oracle explanations: for each aligned test image (shortcut and image agree)
    two fixed maps are scored by every metric, explaining the true label:
        shortcut map   the shortcut pixels
        object map     the image's own pixel values (shortcut pixels removed)
    Preference = share of images where the metric rates the shortcut map as
    the better explanation. A metric that responds to the model's reasoning
    should prefer the object map when reliance is low and the shortcut map
    when reliance is high. The legacy analysis assumed the switch at 50%
    reliance; the per-image behaviour labels (`behaviour` step,
    behaviour_summary.csv) test this directly instead.

    Legacy scores per metric (responsiveness.csv, kept for continuity):
        responsiveness   Spearman correlation between reliance and preference
        decision acc.    share of models where (preference > 0.5) equals (reliance > 50%)
                         (unbalanced: superseded by balanced accuracy and the
                         over-detection rate)
        switch point     first reliance at which the dose means cross 0.5

    Corrected scores (PRIMARY: responsiveness_corrected.csv,
    method_responsiveness_corrected.csv; see xai/dose_analysis.py): chance-
    corrected reliance r_c, two-level bootstrap intervals, BH/Holm-adjusted
    p-values, balanced accuracy and over-detection rate, effect sizes, and
    isotonic switch points marked "not identified" without a model nearby.

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

ImageNet-pretrained Pet (pet_resnet18_pretrained, amendment A2 SD1; Kaggle GPU via
kaggle/dose_response_a2_session1.ipynb and _session2.ipynb)
    The pet_resnet18 data, patch, fills and oracles with torchvision's
    ImageNet-pretrained ResNet-18 (IMAGENET1K_V1, standard 7x7 stride-2 stem
    and max-pool, new 2-class final layer, all layers fine-tuned). ImageNet
    mean/std normalisation is inside the model, so inputs stay in [0,1].
    Optimiser fixed before the pilot: Adam, lr 1e-4, no schedule, batch 128,
    15 epochs. Pilot: doses 0 and -0.5, seed 7 (model set "A2_pilot", never
    pooled with the confirmatory models); the final doses follow
    a2_dose_set() applied to the pilot's r_c at dose 0.

Output
    checkpoint/dose/<base>_<shortcut>_dose<+NNN>_seed<N>.pt (+ _metrics.json)
    results/dose_response/scores_oracle.csv, scores_methods.csv  per-image scores
    results/dose_response/per_model/<name>_behaviour.csv  per-image behaviour labels
        (python dose_response.py behaviour; not part of "all")
    results/dose_response/models.csv, preference.csv, responsiveness.csv,
        method_responsiveness.csv, summary.md, *.png   (legacy tables)
    results/dose_response/responsiveness_corrected.csv,
        method_responsiveness_corrected.csv (PRIMARY), effect_sizes.csv,
        metric_pairs.csv, pet_preregistered.csv, confound.csv,
        behaviour_models.csv, behaviour_summary.csv, behaviour_by_reliance.csv,
        behaviour_auroc.png
    analyze --model-set original writes the same files for the 192 models
    from before amendment A1 to results/dose_response/original_models/, and
    analyze --model-set pre_a2 the 234 models from before amendment A2 to
    results/dose_response/pre_a2_models/.
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
from xai import dose_analysis
from xai.stats import chance_floor, corrected_reliance, format_p

# Base dataset: training epochs. cifar10 is opt-in (--bases cifar10): its models
# take 20-40 min each on a CPU, so it is not part of the default run.
BASES = {"mnist": 3, "pneumonia_mnist": 20, "cifar10": 10, "cifar10_resnet18": 15,
         "pet_resnet18": 30, "pet_resnet18_pretrained": 15}
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
    """'+010' for 0.1, '+0p5' for 0.005, '+0p75' for 0.0075 (whole percentages keep
    the original format; fractional ones keep every significant decimal, so
    nearby doses such as 0.0075 and 0.008 never share a name)."""
    percent = dose * 100
    if abs(percent - round(percent)) < 1e-9:
        return f"{round(percent):+04d}"
    return f"{round(percent, 6):+g}".replace(".", "p")
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
    optimizer = optim.Adam(model.parameters(), lr=info(base).get("lr", 1e-3))
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
    model = build_model(base, pretrained=False)  # weights come from the checkpoint
    model.load_state_dict(torch.load(MODEL_DIR / f"{model_name(base, shortcut, dose, seed)}.pt",
                                     map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def size_matched(object_values, k):
    """Control oracle: a binary map of k object pixels, where k is the number of
    shortcut pixels: the first k of a stable descending sort of the object
    values. It has the same size as the shortcut map, so a preference between
    them cannot come from size alone. On MNIST many saturated pixels tie
    (median 29.5 tied pixels at the top value), and the stable sort then picks
    the topmost saturated pixels in row-major order, not the k "brightest"."""
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


BEHAVIOUR_BATCH = 25
SWAP_SEED = 20_000


def behaviour_sets(base, shortcut):
    """(aligned, conflicting, none) test sets; the conflicting shift uses seed 0,
    as in the reliance measurement of train_one."""
    return tuple(shortcut_split(base, shortcut, "test", mode=mode) for mode in ("aligned", "conflicting", "none"))


def swap_partner(labels, i):
    """Index of a different-class test image for item i (fixed per item)."""
    for j in np.random.default_rng(SWAP_SEED + i).permutation(len(labels)):
        if labels[j] != labels[i]:
            return int(j)
    raise ValueError("no image of another class")


def behaviour_one(base, shortcut, dose, seed, samples, sets=None):
    """Per-image behavioural labels on the scored aligned images (forward passes only).

    follows_shortcut (moved-shortcut pass, reviewer 05): the model predicts the
        shortcut's class when the shortcut of the same image is moved to a wrong
        class's position (the conflicting test image). The PRIMARY label,
        follows_shortcut_causal, is derived from this and pred_none in the
        analysis (xai.dose_analysis.add_causal_label): images the model already
        assigns to the shortcut's class without any shortcut are undetermined.
    shortcut_dominant (check, reviewer 03): delta_s > delta_o, where
        delta_s = p_y(aligned) - p_y(same image without shortcut) and
        delta_o = p_y(aligned) - p_y(image of another class with the label-y shortcut).
    """
    model = load(base, shortcut, dose, seed)
    aligned, conflicting, none = sets or behaviour_sets(base, shortcut)
    generator = torch.Generator().manual_seed(IMAGE_SEED)
    indices = torch.randperm(len(aligned), generator=generator)[:samples].tolist()
    rows = []
    for start in range(0, len(indices), BEHAVIOUR_BATCH):
        batch = indices[start:start + BEHAVIOUR_BATCH]
        labels = [int(aligned.labels[i]) for i in batch]
        partners = [swap_partner(aligned.labels, i) for i in batch]
        images = {
            "aligned": torch.stack([aligned[i][0] for i in batch]),
            "conflicting": torch.stack([conflicting[i][0] for i in batch]),
            "none": torch.stack([none[i][0] for i in batch]),
            "swapped": torch.stack([aligned.with_shortcut(none[j][0], y) for j, y in zip(partners, labels)]),
        }
        with torch.no_grad():
            probs = {k: torch.exp(model(v.to(DEVICE))).cpu().numpy() for k, v in images.items()}
        for k, (i, y, j) in enumerate(zip(batch, labels, partners)):
            conflict = int(conflicting.patch_class[i])
            p_aligned, p_none, p_swapped = probs["aligned"][k, y], probs["none"][k, y], probs["swapped"][k, y]
            pred_conflicting = int(probs["conflicting"][k].argmax())
            delta_s, delta_o = float(p_aligned - p_none), float(p_aligned - p_swapped)
            rows.append({"base": base, "shortcut": shortcut, "dose": dose, "seed": seed, "index": i,
                         "label": y, "conflict_class": conflict,
                         "pred_aligned": int(probs["aligned"][k].argmax()), "p_label_aligned": float(p_aligned),
                         "pred_conflicting": pred_conflicting,
                         "p_conflict_class": float(probs["conflicting"][k, conflict]),
                         "p_label_conflicting": float(probs["conflicting"][k, y]),
                         "pred_none": int(probs["none"][k].argmax()), "p_label_none": float(p_none),
                         "swap_index": j, "p_label_swapped": float(p_swapped),
                         "follows_shortcut": int(pred_conflicting == conflict),
                         "delta_s": delta_s, "delta_o": delta_o, "shortcut_dominant": int(delta_s > delta_o)})
    return rows


def behaviour(args):
    """MD4: per-image behaviour labels for every cached model (or only the
    selected configurations with --selection-only), cached per model in
    per_model/<name>_behaviour.csv. Existing files are skipped (--force redoes them)."""
    cache = OUT / "per_model"
    configs = all_cached_models()
    if getattr(args, "selection_only", False):
        selected = {model_name(*c) for c in configurations(args)}
        configs = [c for c in configs if model_name(*c) in selected]
    sets = {}
    for config in configs:
        name = model_name(*config)
        path = cache / f"{name}_behaviour.csv"
        if path.exists() and not args.force:
            continue
        key = config[:2]
        if key not in sets:
            sets = {key: behaviour_sets(*key)}  # keep one dataset in memory at a time
        start = time.perf_counter()
        pd.DataFrame(behaviour_one(*config, args.samples, sets[key])).to_csv(path, index=False)
        print(f"behaviour {name} ({time.perf_counter() - start:.1f} s)", flush=True)


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
    oracle, methods = _merge_scores(names)
    oracle.to_csv(OUT / "scores_oracle.csv", index=False)
    methods.to_csv(OUT / "scores_methods.csv", index=False)


def _merge_scores(names):
    """(oracle, methods) per-image scores of the named models. Control scores
    are added when every model has them (dose_response.py placebo)."""
    cache = OUT / "per_model"
    oracle = pd.concat([pd.read_csv(cache / f"{n}_oracle.csv") for n in names])
    for control in CONTROL_MAPS:
        if all((cache / f"{n}_{control}.csv").exists() for n in names):
            scores = pd.concat([pd.read_csv(cache / f"{n}_{control}.csv") for n in names])
            oracle = oracle.merge(scores, on=["base", "shortcut", "dose", "seed", "index"], how="left")
    methods = pd.concat([pd.read_csv(cache / f"{n}_methods.csv") for n in names])
    return oracle, methods


# Pre-registered amendment A1 (docs/PREDICTIONS_amendments.md): the 42 models
# added to fill reliance gaps. Every other model belongs to the "original" set.
AMENDMENT_A1 = {"mnist": {"doses": (0.0075, 0.015, 0.03), "seeds": (3, 4, 5)},
                "pneumonia_mnist": {"doses": (-0.075, -0.025), "seeds": ()}}


# Pre-registered amendment A2 (docs/PREDICTIONS_amendments.md) dose rules,
# keyed by the pilot's corrected reliance r_c (%) as worded in A2.
# CIFAR-10 (SD2): pilot cifar10_resnet18 dose -0.05 seed 42; the small CNN
# (cifar10) uses the same 4 final doses.
A2_CIFAR_DOSE_SETS = {"below 50": [-0.05, -0.03, -0.02, -0.01],
                      "50 to 85": [-0.08, -0.05, -0.03, -0.01],
                      "above 85": [-0.09, -0.075, -0.06, -0.05]}
# Pretrained Pet (SD1): pilot pet_resnet18_pretrained dose 0 seed 7.
A2_SD1_DOSE_SETS = {"60 or above": [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0],
                    "below 60": [-0.5, -0.2, 0.0, 0.1, 0.25, 0.4]}


def a2_dose_set(base, pilot_rc):
    """Final A2 doses for a base given its pilot's r_c (%). "50 to 85"
    includes both ends; "above 85" and "below 50" exclude them."""
    if base in ("cifar10", "cifar10_resnet18"):
        key = "below 50" if pilot_rc < 50 else "50 to 85" if pilot_rc <= 85 else "above 85"
        return list(A2_CIFAR_DOSE_SETS[key])
    if base == "pet_resnet18_pretrained":
        return list(A2_SD1_DOSE_SETS["60 or above" if pilot_rc >= 60 else "below 60"])
    raise ValueError(f"amendment A2 has no dose rule for {base}")


def model_set_of(row):
    """'A2_pilot' for the SD1 pilot models (pet_resnet18_pretrained, seed 7),
    'A2' for the other models added by amendment A2 (SD1; CIFAR-10 doses
    strictly between -0.1 and 0; Pet dose -0.15), 'A1' for a model added by
    amendment A1, otherwise 'original'."""
    base, dose = row["base"], float(row["dose"])
    if base == "pet_resnet18_pretrained":
        return "A2_pilot" if int(row["seed"]) == 7 else "A2"
    if base in ("cifar10", "cifar10_resnet18") and -0.1 + 1e-9 < dose < -1e-9:
        return "A2"
    if base == "pet_resnet18" and np.isclose(dose, -0.15):
        return "A2"
    rule = AMENDMENT_A1.get(base)
    if rule and (np.isclose(row["dose"], rule["doses"]).any() or int(row["seed"]) in rule["seeds"]):
        return "A1"
    return "original"


def in_model_set(run, model_set):
    """Whether a model (metrics JSON or models.csv row) belongs to a model set:
    'all' = every model except the SD1 pilots, 'pre_a2' = original + A1 (the
    234 models before amendment A2), otherwise the exact model_set_of value."""
    kind = model_set_of(run)
    if model_set == "all":
        return kind != "A2_pilot"
    if model_set == "pre_a2":
        return kind in ("original", "A1")
    return kind == model_set


def complete_model_names(model_set="all"):
    """Names of models with a metrics JSON, a .pt and all four per-model score
    files (oracle, methods, placebo, random), restricted to a model set
    (see in_model_set)."""
    cache = OUT / "per_model"
    names = []
    for path in sorted(MODEL_DIR.glob("*_metrics.json")):
        run = json.loads(path.read_text())
        name = model_name(run["base"], run["shortcut"], run["dose"], run["seed"])
        files = [MODEL_DIR / f"{name}.pt"] + [cache / f"{name}_{kind}.csv"
                                             for kind in ("oracle", "methods", *CONTROL_MAPS)]
        if all(f.exists() for f in files) and in_model_set(run, model_set):
            names.append(name)
    return sorted(names)


# ---------------------------------------------------------------- analysis

def load_models():
    runs = [json.loads(p.read_text()) for p in sorted(MODEL_DIR.glob("*_metrics.json"))]
    return pd.DataFrame(runs)


def add_corrected_reliance(models):
    """Append classes, chance_floor, reliance_corrected and model_set.

    chance_floor = (100 - test_accuracy_no_shortcut) / (C - 1): the reliance
    expected from errors alone (on a binary task every error on a conflicting
    image lands on the shortcut's class). reliance_corrected =
    100 (reliance - floor) / (100 - floor), in percent and not clipped.
    """
    models = models.copy()
    models["classes"] = [num_classes(b) for b in models.base]
    models["chance_floor"] = chance_floor(models.test_accuracy_no_shortcut, models.classes)
    models["reliance_corrected"] = corrected_reliance(models.reliance, models.chance_floor)
    models["model_set"] = [model_set_of(row) for _, row in models.iterrows()]
    return models


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


def _ci_text(frame, column, low=None, high=None, digits=3):
    """'v [low, high]' strings for a value column and its interval columns."""
    low, high = low or f"{column}_ci_low", high or f"{column}_ci_high"

    def fmt(v):
        return "–" if pd.isna(v) else f"{v:.{digits}f}"
    return [f"{fmt(v)} [{fmt(a)}, {fmt(b)}]" for v, a, b in zip(frame[column], frame[low], frame[high])]


def _p_text(frame, columns):
    frame = frame.copy()
    for column in columns:
        frame[column] = [format_p(v) for v in frame[column]]
    return frame


def load_behaviour(oracle):
    """Per-image behaviour labels of every scored model, or (None, missing names)."""
    keys = ["base", "shortcut", "dose", "seed"]
    names = [model_name(*k) for k in oracle[keys].drop_duplicates().itertuples(index=False)]
    paths = [OUT / "per_model" / f"{n}_behaviour.csv" for n in names]
    missing = [n for n, p in zip(names, paths) if not p.exists()]
    if missing:
        return None, missing
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True), []


def plot_behaviour_auroc(bands, path):
    data = bands[(bands.comparison == "shortcut vs full object") & (bands.label == dose_analysis.PRIMARY_LABEL)]
    groups = list(data.groupby(["base", "shortcut"]))
    fig, axes = plt.subplots(1, len(groups), figsize=(4.3 * len(groups), 3.9), squeeze=False, sharey=True)
    for ax, ((base, shortcut), group) in zip(axes[0], groups):
        for metric, rows in group.groupby("metric", sort=False):
            ax.plot(rows.band_centre, rows.auroc_strat, marker="o", label=metric)
        ax.axhline(0.5, color="gray", linewidth=0.8)
        ax.set_title(f"{base}, {shortcut} shortcut", fontsize=10)
        ax.set_xlabel("Chance-corrected reliance band (%)")
        ax.set_xlim(-2, 102)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("Class-stratified AUROC vs per-image label\n(follows shortcut, causal)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Shortcut vs full object: does the per-image AUC gap detect images where the model follows the shortcut?")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def corrected_analysis(oracle, methods, models, reps, out):
    """MD2/MD3/MD4/MD7 outputs (written to `out`); returns the summary.md lines."""
    settings = dose_analysis.build_settings(oracle, methods, models, reps, BOOTSTRAP_SEED, TIE_TOLERANCE)
    resp = dose_analysis.responsiveness_corrected(settings, COMPARISONS, METRIC_TITLES)
    effects = dose_analysis.effect_sizes(settings, COMPARISONS, METRIC_TITLES)
    pairs = dose_analysis.metric_pairs(settings, COMPARISONS, METRIC_TITLES)
    method_resp = dose_analysis.method_responsiveness_corrected(settings, METHOD_TITLES)
    pet = dose_analysis.pet_preregistered(settings, COMPARISONS, METRIC_TITLES, METHOD_TITLES)
    confound = dose_analysis.confound_table(settings, COMPARISONS, METRIC_TITLES)
    behaviour, missing = load_behaviour(oracle)
    beh_models = beh_summary = beh_bands = None
    if behaviour is not None:
        beh_models, beh_summary, beh_bands = dose_analysis.behaviour_tables(
            settings, behaviour, COMPARISONS, METRIC_TITLES)
    else:
        print(f"warning: {len(missing)} models have no behaviour labels (e.g. {missing[0]}); "
              "run `dose_response.py behaviour`. MD4 tables skipped.")
    resp, beh_summary = dose_analysis.headline_holm(resp, beh_summary)

    resp.to_csv(out / "responsiveness_corrected.csv", index=False)
    effects.to_csv(out / "effect_sizes.csv", index=False)
    pairs.to_csv(out / "metric_pairs.csv", index=False)
    method_resp.to_csv(out / "method_responsiveness_corrected.csv", index=False)
    pet.to_csv(out / "pet_preregistered.csv", index=False)
    confound.to_csv(out / "confound.csv", index=False)
    if beh_summary is not None:
        beh_models.to_csv(out / "behaviour_models.csv", index=False)
        beh_summary.to_csv(out / "behaviour_summary.csv", index=False)
        beh_bands.to_csv(out / "behaviour_by_reliance.csv", index=False)
        plot_behaviour_auroc(beh_bands, out / "behaviour_auroc.png")

    setting = lambda f: f.base + " / " + f.shortcut  # noqa: E731
    lines = [
        "## Corrected analysis (primary)", "",
        "Primary outputs: `responsiveness_corrected.csv` (metrics) and `method_responsiveness_corrected.csv` "
        "(attribution methods), with `behaviour_summary.csv` as the per-image behavioural test (MD4). "
        "Supporting outputs: `effect_sizes.csv`, `metric_pairs.csv`, `pet_preregistered.csv`, `confound.csv`, "
        "`behaviour_models.csv`, `behaviour_by_reliance.csv`, `behaviour_auroc.png`; `models.csv` and "
        "`preference.csv` gain the columns below. The legacy tables follow after this section.", "",
        "### Definitions", "",
        "- Chance floor f = (100 − accuracy without shortcut) / (C − 1), C = number of classes: the reliance "
        "expected from errors alone (on a binary task every error on a conflicting image lands on the shortcut's "
        "class; with C classes 1/(C − 1) of errors if they spread uniformly). An approximation, since images "
        "without the shortcut are out of distribution.",
        "- Chance-corrected reliance r_c = 100 (R − f) / (100 − f), in percent, not clipped (it can be slightly "
        "negative). Columns `classes`, `chance_floor`, `reliance_corrected` and `model_set` in `models.csv`.",
        "- Decisions on r_c: positive class r_c > 50, 'metric says shortcut' = preference > 0.5. "
        "TPR = P(says shortcut | r_c > 50), TNR = P(says object | r_c ≤ 50), balanced accuracy = (TPR + TNR)/2, "
        "**over-detection rate** fpr = 1 − TNR. `_raw` columns: the same on raw reliance.",
        f"- Two-level bootstrap ({reps} replicates, seed {BOOTSTRAP_SEED}): models resampled within each dose, "
        "and each picked model's 100 images resampled independently; the same picks are reused for every "
        "comparison, metric and method (paired). 95% percentile intervals.",
        "- Switch point (only for comparisons whose ideal crosses 0.5 at 50%: shortcut vs full / size-matched "
        "object): increasing isotonic fit (PAVA) of model preference on r_c, first r_c where the fit reaches 0.5 "
        f"(linear interpolation). 'not identified' when no model lies within ±{dose_analysis.IDENTIFY_WINDOW} "
        f"points. `switch_bracketed` (sensitivity): models within ±{dose_analysis.IDENTIFY_WINDOW} points on both "
        "sides; a switch inside a wider gap between models can be 'identified' from one edge but is not bracketed, "
        "and should not be quoted as located. 'identified at lowest-reliance edge': the switch lies within "
        f"{dose_analysis.EDGE_WINDOW} points of the lowest-reliance model, i.e. the metric already prefers the "
        "shortcut near the bottom of the reliance range (over-detection), not a mid-range switch. The switch CI "
        "is empty when the full-data switch is undefined or fewer than half of the replicates cross. "
        "Sensitivity: 4-parameter logistic with free asymptotes (`switch_4pl_rc`). `switch_interp_rc`: "
        "the legacy first-crossing rule applied to r_c dose means.",
        "- Effect size: per-image AUC gap g (positive = shortcut map rated better; AUC_B − AUC_A for deletion, "
        "AUC_A − AUC_B for insertion); medians at the lowest and highest dose in `effect_sizes.csv`.",
        "- p-values come from two sources. Correlations (responsiveness, methods): Spearman's t approximation "
        "over the models of a setting, not from the bootstrap (the bootstrap gives only their intervals). "
        "Paired metric differences: two-sided percentile bootstrap, floored at 1/(reps + 1). Behaviour AUROC vs "
        "0.5: z-test with the two-level bootstrap SE (not floored; the floored percentile value is "
        "`p_auroc_boot`); one-sided (H1: AUROC > 0.5) for the class-stratified AUROCs (`*_strat`, primary), "
        "two-sided with a `direction` column for the unstratified ones. BH and Holm within families: "
        "responsiveness:<comparison>, methods, metric_pairs:<comparison>, behaviour:<label> (each AUROC column "
        "adjusted separately). Headline Holm: one family over the union of the headline rows (shortcut vs full "
        "object and shortcut vs random responsiveness, and the pooled and within-model class-stratified "
        "one-sided primary behaviour AUROC for shortcut vs full object); `p_holm_headline` in the "
        "responsiveness table, `p_holm_headline_strat` and `p_holm_headline_within_strat` in the behaviour "
        "table. p < 0.001 is printed as '< 0.001'.", ""]

    main = resp[resp.comparison.isin(dose_analysis.SWITCH_COMPARISONS)].copy()
    main["setting"] = setting(main)
    full = main[main.comparison == "shortcut vs full object"]
    lines += ["### Over-detection at low reliance (decisions on r_c)", "",
              f"TPR = 1.00 in {int((full.tpr == 1).sum())} of {int(full.tpr.notna().sum())} cells "
              "(shortcut vs full object, r_c axis); over-detection rate fpr > 0 in "
              f"{int((full.fpr > 0).sum())} of {int(full.fpr.notna().sum())} cells.", "",
              markdown(main[["comparison", "setting", "metric", "fpr", "balanced_accuracy", "tpr",
                             "n_below", "n_above", "fpr_raw", "balanced_accuracy_raw"]]), ""]

    shown = resp[resp.comparison.isin(["shortcut vs full object", "shortcut vs size-matched object",
                                       "shortcut vs random", "shortcut vs placebo"])].copy()
    shown["setting"] = setting(shown)
    shown["rho_rc [95% CI]"] = _ci_text(shown, "rho_rc")
    shown["pref_low [CI]"] = _ci_text(shown, "pref_low")
    shown["pref_high [CI]"] = _ci_text(shown, "pref_high")
    shown = _p_text(shown, ["p_rho_rc", "p_bh", "p_holm_headline"])
    significant = resp[(resp.p_bh < 0.05) & (resp.rho_rc > 0)].groupby("comparison").size()
    lines += ["### Responsiveness on r_c (two-level intervals)", "",
              "Positive and BH-significant (p_bh < 0.05) rows per comparison: "
              + ", ".join(f"{k}: {v} of {int((resp.comparison == k).sum())}" for k, v in significant.items()) + ".", "",
              markdown(shown[["comparison", "setting", "metric", "rho_rc [95% CI]", "p_rho_rc", "p_bh",
                              "p_holm_headline", "pref_low [CI]", "pref_high [CI]", "median_gap_low",
                              "median_gap_high"]]), ""]

    main["switch_iso_rc [CI]"] = _ci_text(main, "switch_iso_rc", "switch_iso_rc_ci_low", "switch_iso_rc_ci_high", 1)
    lines += ["### Switch points on r_c (isotonic primary, 4PL sensitivity)", "",
              markdown(main[["comparison", "setting", "metric", "switch_iso_rc [CI]", "switch_found_share",
                             "switch_status", "nearest_model_distance", "switch_bracketed",
                             "switch_distance_below", "switch_distance_above", "switch_4pl_rc", "fit_4pl_ok",
                             "switch_iso_raw", "switch_interp_rc"]]), ""]

    sig_pairs = pairs[pairs.p_bh < 0.05].copy()
    sig_pairs["setting"] = setting(sig_pairs)
    sig_pairs["delta_rho_rc [CI]"] = _ci_text(sig_pairs, "delta_rho_rc")
    sig_pairs["delta_pref_low [CI]"] = _ci_text(sig_pairs, "delta_pref_low")
    lines += ["### Paired metric-vs-metric differences (BH-significant rows; all rows in metric_pairs.csv)", "",
              f"{len(sig_pairs)} of {len(pairs)} metric pairs differ in rho_rc at p_bh < 0.05.", "",
              markdown(_p_text(sig_pairs, ["p", "p_bh", "p_holm"])[
                  ["comparison", "setting", "metric_a", "metric_b", "delta_rho_rc [CI]", "p", "p_bh", "p_holm",
                   "delta_pref_low [CI]"]]) if len(sig_pairs) else "(none)", ""]

    shown = method_resp.copy()
    shown["setting"] = setting(shown)
    shown["rho_rc [95% CI]"] = _ci_text(shown, "rho_rc")
    shown["mass_low [CI]"] = _ci_text(shown, "mass_low")
    shown["mass_high [CI]"] = _ci_text(shown, "mass_high")
    lines += ["### Method responsiveness on r_c (two-level intervals)", "",
              markdown(_p_text(shown, ["p", "p_bh", "p_holm"])[
                  ["setting", "method", "rho_rc [95% CI]", "p", "p_bh", "p_holm", "mass_low [CI]",
                   "mass_high [CI]"]]), ""]

    if len(pet):
        shown = pet.copy()
        shown["estimate [95% CI]"] = _ci_text(shown, "estimate", "ci_low", "ci_high")
        lines += ["### Pre-registered Pet predictions (original verdicts kept; intervals added)", "",
                  dose_analysis.PET_NOTE, "",
                  markdown(shown[["prediction", "comparison", "metric_or_method", "statistic", "estimate [95% CI]",
                                  "rho_rc", "threshold", "original_verdict"]]), ""]

    shown = confound.copy()
    shown["setting"] = setting(shown)
    lines += ["### Reliance vs object competence (MD7)", "",
              "rho_reliance_competence: Spearman(reliance, accuracy without shortcut) over the models of a setting. "
              "partial_rel_given_acc: partial Spearman of reliance and preference (shortcut vs full object) "
              "controlling for accuracy without shortcut (rank residuals, t test with n − 3 df). Sorted from the "
              "least collinear setting.", "",
              markdown(_p_text(shown, ["p_partial_rel_given_acc", "p_partial_acc_given_rel"])[
                  ["setting", "models", "rho_reliance_competence", "rho_rc_competence", "metric", "rho_rel",
                   "partial_rel_given_acc", "p_partial_rel_given_acc", "partial_acc_given_rel",
                   "p_partial_acc_given_rel"]]), "",
              "Validity limits (not fixable by new runs):",
              "- PneumoniaMNIST: the 'object' map is the image's own intensities, and 97.5% of pixels exceed 0.1, "
              "so it is an intensity ranking of the whole radiograph, not an object segmentation.",
              "- CIFAR-10 (both architectures): the 'object' map is a centred Gaussian prior, so Pet (real "
              "segmentation masks) is the primary natural-image evidence.",
              "- Blur fill leaves about 20% (28 and 32 px) to 25% (Pet) of the patch intensity in place, so blur "
              "deletion does not fully remove a patch shortcut.",
              "- `size_matched` takes the first k pixels of a stable descending sort of the object values; on MNIST "
              "many saturated pixels tie, so it picks the topmost saturated pixels in row-major order.", ""]

    lines += ["### MD4 per-image behavioural ground truth", ""]
    if beh_summary is None:
        lines += [f"Skipped: {len(missing)} models have no behaviour labels. Run `dose_response.py behaviour`.", ""]
    else:
        lines += [
            "Each scored aligned image gets a label from forward passes only. Primary "
            "(`follows_shortcut_causal`, reviewer 05's moved-shortcut design plus the shortcut-removed pass): 1 if "
            "the model predicts the shortcut's class when the same image's shortcut is moved to a wrong class's "
            "position and does not predict that class with the shortcut removed; 0 if the moved shortcut does not "
            "win; undetermined (excluded, `undetermined_rate`) if the model already predicts the shortcut's class "
            "without any shortcut (this takes precedence over the moved pass; the rare case moved → other class "
            "and removed → shortcut's class is also undetermined). The removed-pass rule keeps ordinary errors (on "
            "a binary task every error lands on the shortcut's class) from counting as shortcut use, the same floor "
            "r_c removes from reliance. "
            "Checks: `follows_shortcut` (moved-shortcut pass only) and `shortcut_dominant` (reviewer 03): "
            "Δ_S > Δ_O, with Δ_S = p_y(aligned) − p_y(no shortcut) and Δ_O = p_y(aligned) − p_y(another-class "
            "image with the label-y shortcut). The score is the per-image AUC gap (shortcut vs object). "
            "**Class-stratified AUROC (primary):** on the binary settings (PneumoniaMNIST, Pet) which images are "
            "undetermined and which are followed depends strongly on the image's true class (a model that fails "
            "without its shortcut predicts one class), and the AUC gap also differs by class, so an AUROC over "
            "images of both classes can fall far below 0.5 from class differences alone (Simpson's paradox). "
            "The primary statistics therefore compute the AUROC within each true class and average over the "
            "classes that have both labels: `auroc_pooled_strat` (images of all models pooled; still mixes "
            "between-model and within-model signal) and `median_model_auroc_strat` (per model, then median over "
            "models; detection inside one model; `models_with_both_labels_in_a_class` models contribute). Both "
            "have two-level CIs and one-sided z-test p-values (H1: AUROC > 0.5), so an AUROC below 0.5 is never "
            "significant. The unstratified `auroc_pooled` and `median_model_auroc` are kept as checks with "
            "two-sided p-values and a `direction` column; an unstratified value significantly below 0.5 means "
            "inverted ranking (on the binary settings, mostly a class effect), not detection. "
            "accuracy = mean(1 − |w − y|). calibration_gap = mean(preference − follow rate) (positive = "
            "over-detection). observed_switch_rc: r_c where the follow rate crosses 0.5 (isotonic), the derived "
            "ideal replacing the assumed 50%. **The per-image test is now primary; the aligned-image preference "
            "curves are secondary.**", ""]
        for name in dose_analysis.BEHAVIOUR_LABELS:
            shown = beh_summary[beh_summary.label == name].copy()
            shown["setting"] = setting(shown)
            shown["auroc_pooled_strat [95% CI]"] = _ci_text(
                shown, "auroc_pooled_strat", "auroc_strat_ci_low", "auroc_strat_ci_high")
            shown["median_model_auroc_strat [95% CI]"] = _ci_text(
                shown, "median_model_auroc_strat", "median_model_auroc_strat_ci_low",
                "median_model_auroc_strat_ci_high")
            shown["auroc_pooled [95% CI]"] = _ci_text(shown, "auroc_pooled", "auroc_ci_low", "auroc_ci_high")
            shown["median_model_auroc [95% CI]"] = _ci_text(
                shown, "median_model_auroc", "median_model_auroc_ci_low", "median_model_auroc_ci_high")
            columns = ["comparison", "setting", "metric", "label_rate", "undetermined_rate",
                       "auroc_pooled_strat [95% CI]", "p_bh_strat", "median_model_auroc_strat [95% CI]",
                       "p_bh_within_strat", "models_with_both_labels_in_a_class"]
            if name == dose_analysis.PRIMARY_LABEL:
                columns += ["p_holm_headline_strat", "p_holm_headline_within_strat"]
            columns += ["auroc_pooled [95% CI]", "direction", "p_bh", "median_model_auroc [95% CI]",
                        "direction_within", "p_bh_within",
                        "accuracy", "balanced_accuracy", "calibration_gap", "rho_pref_follow", "observed_switch_rc"]
            lines += [f"Label `{name}`" + (" (primary)" if name == dose_analysis.PRIMARY_LABEL else " (check)") + ":", "",
                      markdown(_p_text(shown, ["p_auroc", "p_bh", "p_bh_within", "p_bh_strat", "p_bh_within_strat",
                                               "p_holm_headline_strat", "p_holm_headline_within_strat"])[columns]),
                      ""]
    return lines


def analyze(args=None):
    """Legacy tables (unchanged computation) plus the corrected analysis.

    --model-set all (default, PRIMARY): every cached model; the combined score
    files are rebuilt first, so models added from elsewhere (e.g. the Kaggle
    ResNet-18 results) are included; outputs go to results/dose_response/.
    --model-set original: the 192 models from before amendment A1, merged in
    memory (scores_*.csv are not rewritten); outputs go to original_models/.
    """
    model_set = getattr(args, "model_set", "all")
    reps = getattr(args, "reps", BOOTSTRAP_REPS)
    keys = ["base", "shortcut", "dose", "seed"]
    models = add_corrected_reliance(load_models())
    if model_set == "all":
        # The SD1 pilots (A2) are reported separately, never pooled (no-op before A2).
        models = models[models.model_set != "A2_pilot"].reset_index(drop=True)
        combine(None)
        out = OUT
        oracle = pd.read_csv(OUT / "scores_oracle.csv")
        methods = pd.read_csv(OUT / "scores_methods.csv")
    else:
        names = set(complete_model_names(model_set))
        oracle, methods = _merge_scores(sorted(names))
        out = OUT / f"{model_set}_models"
        out.mkdir(parents=True, exist_ok=True)
        models = models[[model_name(*k) in names for k in models[keys].itertuples(index=False)]]
        models = models.reset_index(drop=True)

    prefs, resps = [], []
    for first, second, label, suffix, ideal in COMPARISONS:
        if f"deletion_zero_{first}" not in oracle or f"deletion_zero_{second}" not in oracle:
            continue  # placebo not computed yet
        pref = preference_table(oracle, first, second).merge(models[keys + ["reliance"]], on=keys)
        resp = responsiveness(pref).merge(bootstrap_intervals(pref), on=["base", "shortcut", "metric"])
        prefs.append(pref.assign(comparison=label))
        resps.append(resp.assign(comparison=label))
        plot_panels(pref, list(FAITHFULNESS_METRICS), METRIC_TITLES,
                    f"Preference for {first} map", out / f"metric_dose_response{suffix}.png",
                    f"{label}: preference as reliance grows (ideal: {ideal})")
    pref, resp = pd.concat(prefs), pd.concat(resps)
    method_resp, mass = method_responsiveness(methods, models[keys + ["reliance"]])

    models.to_csv(out / "models.csv", index=False)
    pref = pref.merge(models[keys + ["reliance_corrected"]], on=keys, how="left").round(4)
    pref.to_csv(out / "preference.csv", index=False)
    resp.to_csv(out / "responsiveness.csv", index=False)
    method_resp.to_csv(out / "method_responsiveness.csv", index=False)
    mass = mass.pivot_table(index=keys + ["reliance"], columns="method", values="shortcut_mass").reset_index()
    plot_panels(mass, list(METHODS), METHOD_TITLES, "Share of positive attribution on shortcut",
                out / "method_dose_response.png", "Do explanation methods follow the shortcut as reliance grows?",
                line=None)
    corrected = corrected_analysis(oracle, methods, models, reps, out)

    level = models.groupby(["base", "shortcut", "dose"])[
        ["reliance", "test_accuracy", "test_accuracy_no_shortcut"]].agg(["mean", "std"]).round(2)
    level.columns = [f"{a}_{b}" for a, b in level.columns]
    lines = ["## Legacy analysis (raw reliance, seed-only bootstrap; kept for continuity)", "",
             "These tables are computed exactly as before amendment A1 and the audit corrections "
             "(responsiveness.csv, method_responsiveness.csv, preference.csv). The corrected analysis above "
             "is primary.", "",
             "### Reliance per dose (mean and std over seeds, %)", "", markdown(level.reset_index()), "",
             "### Metric responsiveness to reliance (oracle maps)", "",
             "responsiveness_rho: Spearman correlation between reliance and preference for the shortcut map "
             "(ideal: close to 1). decision_accuracy: share of models where the preferred map matches whether "
             "reliance is above 50%. switch_point_reliance: reliance where preference crosses 0.5 (ideal: 50). "
             "comparison: which two oracle maps are compared (preference = share of images where the metric "
             "rates the first above the second). Size-matched object: a binary map of as many object pixels "
             "as the shortcut has, the first k of a stable descending sort of the object values (on MNIST many "
             "saturated pixels tie, so it takes the topmost saturated pixels in row-major order; on Pet, the k "
             "mask pixels nearest the mask centroid). Placebo: the shortcut's shape at a wrong class's position, which "
             "never carries evidence. Random: independent uniform scores per pixel (a fully random ordering). "
             "The placebo's non-patch pixels are tied and ordered randomly, so it behaves almost like the random "
             "map ('placebo vs random' stays near 0.5). 'Placebo vs full object' and 'random vs full object' "
             "should be LOW at low reliance; a rise at high reliance is correct, because the object map then "
             "explains the model worse than a random ordering. "
             f"switch_ci_low/high and rho_ci_low/high: 95% bootstrap intervals ({BOOTSTRAP_REPS} resamples of "
             "the seeds within each dose). switch_found_share: share of resamples in which preference crosses "
             "0.5 at all (the switch interval is left empty below 0.5).", "",
             markdown(resp[["comparison"] + [c for c in resp.columns if c != "comparison"]]), "", "### Method responsiveness (share of attribution on the shortcut vs reliance)", "",
             markdown(method_resp), ""]
    header = ["# Dose-response results", "", "Generated by `dose_response.py analyze"
              + ("`." if model_set == "all" else f" --model-set {model_set}`."), "",
              f"Model set: **{model_set}** ({len(oracle[keys].drop_duplicates())} scored models"
              + (", primary)." if model_set == "all" else
                 "; the 234 models before amendment A2)." if model_set == "pre_a2" else
                 "; the pre-amendment check promised in docs/PREDICTIONS_amendments.md)."), ""]
    lines = header + corrected + lines
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # e.g. a cp1252 Windows console
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
    parser.add_argument("step", choices=["train", "evaluate", "placebo", "behaviour", "analyze", "all"])
    parser.add_argument("--bases", nargs="+", default=DEFAULT_BASES, choices=list(BASES))
    parser.add_argument("--shortcuts", nargs="+", default=list(SHORTCUT_TYPES), choices=list(SHORTCUT_TYPES))
    parser.add_argument("--doses", nargs="+", type=float, default=DOSES)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--samples", type=int, default=100, help="test images per model")
    parser.add_argument("--force", action="store_true", help="retrain existing models")
    parser.add_argument("--model-set", choices=["all", "original", "pre_a2"], default="all",
                        help="analyze: all models except the SD1 pilots (primary, results/dose_response/), "
                             "the 192 models from before amendment A1 (results/dose_response/original_models/) "
                             "or the 234 models before amendment A2 (results/dose_response/pre_a2_models/)")
    parser.add_argument("--reps", type=int, default=BOOTSTRAP_REPS,
                        help="analyze: two-level bootstrap replicates of the corrected analysis")
    parser.add_argument("--selection-only", action="store_true",
                        help="behaviour: only the models selected by --bases/--shortcuts/--doses/--seeds")
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
    if args.step == "behaviour":  # not part of "all"
        behaviour(args)
    if args.step in ("analyze", "all"):
        analyze(args)


if __name__ == "__main__":
    main()

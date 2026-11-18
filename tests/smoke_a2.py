"""CPU smoke test of the amendment-A2 code path (not a unittest module).

    .\\.venv\\Scripts\\python.exe -W ignore tests/smoke_a2.py

Runs the real dose_response step functions (train, evaluate, placebo,
behaviour) in a temporary directory, with 1 epoch, a 512-image training
subset, a 128-image validation subset and 2 scored test images, for:
    (a) SD1 pilot: pet_resnet18_pretrained, dose 0, seed 7 (downloads the
        ImageNet ResNet-18 weights once, about 45 MB);
    (b) SD1 confirmatory, positive dose: pet_resnet18_pretrained, dose +0.1,
        seed 42 (the swap-fraction path: swap 0.1, shortcut noise 0);
    (c) SD2: cifar10, dose -0.03, seed 42 (a final A2 CIFAR-10 dose).
It also checks the fractional name dose-7p5 for -0.075.
The temporary directory is always removed, and the script checks that
nothing was written under checkpoint/ or results/ of the repository.
"""

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import Subset  # noqa: E402

import dose_response  # noqa: E402
from xai.models import build_model  # noqa: E402

SUBSETS = {"train": 512, "val": 128}
SAMPLES = 2
PARTS = [("SD1 pilot", "pet_resnet18_pretrained", 0.0, 7, "A2_pilot"),
         ("SD1 +0.1", "pet_resnet18_pretrained", 0.1, 42, "A2"),
         ("SD2", "cifar10", -0.03, 42, "A2")]
REPO_DIRS = [ROOT / "checkpoint" / "dose", ROOT / "results"]


def repo_files():
    """{path: (size, mtime)} of every file under checkpoint/dose and results/."""
    return {p: (p.stat().st_size, p.stat().st_mtime_ns)
            for folder in REPO_DIRS for p in folder.rglob("*") if p.is_file()}


def subset_split(original):
    """shortcut_split with the train and val splits cut to fixed small subsets
    (test splits are untouched: they need .patch_class, .mask and so on)."""
    def wrapped(base, shortcut, split, *args, **kwargs):
        data = original(base, shortcut, split, *args, **kwargs)
        if split in SUBSETS:
            return Subset(data, list(range(min(SUBSETS[split], len(data)))))
        return data
    return wrapped


def run_part(label, base, dose, seed, expected_set):
    start = time.perf_counter()
    args = argparse.Namespace(bases=[base], shortcuts=["patch"], doses=[dose], seeds=[seed],
                              samples=SAMPLES, force=True, selection_only=True)
    name = dose_response.model_name(base, "patch", dose, seed)
    dose_response.train(args)
    dose_response.evaluate(args)
    dose_response.placebo(args)
    dose_response.behaviour(args)

    metrics = json.loads((dose_response.MODEL_DIR / f"{name}_metrics.json").read_text())
    for key in ("reliance", "test_accuracy_no_shortcut", "training_minutes"):
        assert key in metrics, f"{name}: metrics JSON lacks {key}"
    swap, noise = dose_response.dose_params(dose)
    assert (metrics["swap_fraction"], metrics["shortcut_noise"]) == (swap, noise), f"{name}: dose params"
    cache = dose_response.OUT / "per_model"
    for kind in ("oracle", "methods", "placebo", "random", "behaviour"):
        path = cache / f"{name}_{kind}.csv"
        assert path.exists(), f"missing {path.name}"
        frame = pd.read_csv(path)
        assert frame["index"].nunique() == SAMPLES, f"{path.name}: {frame['index'].nunique()} images"
    assert "deletion_zero_object_matched" in pd.read_csv(cache / f"{name}_oracle.csv", nrows=0)
    model = build_model(base, pretrained=False)
    model.load_state_dict(torch.load(dose_response.MODEL_DIR / f"{name}.pt", map_location="cpu"), strict=True)
    got = dose_response.model_set_of(metrics)
    assert got == expected_set, f"{name}: model set {got}, expected {expected_set}"
    print(f"{label} {name}: reliance {metrics['reliance']}%, no shortcut "
          f"{metrics['test_accuracy_no_shortcut']}%, {(time.perf_counter() - start) / 60:.1f} min", flush=True)
    return name


def main():
    before = repo_files()
    tmp = Path(tempfile.mkdtemp(prefix="xai_smoke_"))
    names = [dose_response.model_name(base, "patch", dose, seed) for _, base, dose, seed, _ in PARTS]
    try:
        dose_response.MODEL_DIR = tmp / "checkpoint" / "dose"
        dose_response.OUT = tmp / "results" / "dose_response"
        for _, base, *_ in PARTS:
            dose_response.BASES[base] = 1
        dose_response.shortcut_split = subset_split(dose_response.shortcut_split)
        for part in PARTS:
            run_part(*part)
        assert names[1] == "pet_resnet18_pretrained_patch_dose+010_seed42", names[1]
        fractional = dose_response.model_name("cifar10", "patch", -0.075, 42)
        assert fractional == "cifar10_patch_dose-7p5_seed42", fractional
        # -0.075 belongs to another A2 dose rule, not the final CIFAR-10 doses
        assert dose_response.model_set_of({"base": "cifar10", "shortcut": "patch", "dose": -0.075,
                                           "seed": 1}) == "unregistered"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert not tmp.exists(), f"temporary directory left behind: {tmp}"
    after = repo_files()
    new = sorted(str(p) for p in after.keys() - before.keys())
    assert not new, f"files written into the repository: {new[:5]}"
    # The real Kaggle SD1 pilot shares a name with part (a), so check by content
    # stamp rather than by name: no existing repository file may change.
    changed = sorted(str(p) for p in before.keys() & after.keys() if before[p] != after[p])
    assert not changed, f"repository files changed: {changed[:5]}"
    print("smoke_a2: OK")


if __name__ == "__main__":
    main()

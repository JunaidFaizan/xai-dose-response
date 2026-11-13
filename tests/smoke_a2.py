"""CPU smoke test of the amendment-A2 code path (not a unittest module).

    .\\.venv\\Scripts\\python.exe -W ignore tests/smoke_a2.py

Runs the real dose_response step functions (train, evaluate, placebo,
behaviour) in a temporary directory, with 1 epoch, a 512-image training
subset, a 128-image validation subset and 2 scored test images, for:
    (a) SD1: pet_resnet18_pretrained, dose 0, seed 7 (downloads the ImageNet
        ResNet-18 weights once, about 45 MB);
    (b) SD2: cifar10, dose -0.075, seed 42 (the fractional name dose-7p5).
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
PARTS = [("SD1", "pet_resnet18_pretrained", 0.0, 7, "A2_pilot"),
         ("SD2", "cifar10", -0.075, 42, "A2")]
REPO_DIRS = [ROOT / "checkpoint" / "dose", ROOT / "results"]


def repo_files():
    return {p for folder in REPO_DIRS for p in folder.rglob("*") if p.is_file()}


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
        assert names[1] == "cifar10_patch_dose-7p5_seed42", names[1]
        assert dose_response.model_set_of({"base": "cifar10", "dose": -0.075, "seed": 1}) == "A2"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert not tmp.exists(), f"temporary directory left behind: {tmp}"
    new = repo_files() - before
    assert not new, f"files written into the repository: {sorted(map(str, new))[:5]}"
    leaked = [p for p in before | repo_files() if "smoke" in p.name or any(n in p.name for n in names)]
    assert not leaked, f"smoke files in the repository: {leaked[:5]}"
    print("smoke_a2: OK")


if __name__ == "__main__":
    main()

"""CPU smoke test of the amendment-A3 code path (not a unittest module).

    $env:XAI_DATA_DIR = "<project>\\data"
    .\\.venv\\Scripts\\python.exe -W ignore tests/smoke_a3.py --checkpoint-dir <project>\\checkpoint\\dose

Runs the real dose_response step functions in a temporary directory, with
1 epoch, a 512-image training subset and a 128-image validation subset:
    (a) SD4 pilot: pet_resnet18_pretrained with the watermark, dose 0, seed 7
        (training only, as in Kaggle session 3);
    (b) SD4 swap path: dose +0.25, seed 7 (swap fraction 0.25, shortcut
        noise 0): train, evaluate, placebo and behaviour with 2 test images,
        the watermark oracle and placebo maps, and road_one on 2 images
        (rough CPU seconds per Pet-sized image);
    (c) exact ROAD (SD3) on 3 images of mnist_patch_dose+010_seed42, copied
        from --checkpoint-dir: the road command, a --force rerun (identical
        bytes), --time-only (road_timing.csv only) and the refusal of pilots.
These temporary models and scores are not results: the directory is always
removed. The script checks that nothing changed under checkpoint/ or results/
of the repository, under XAI_DATA_DIR or under --checkpoint-dir.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from torch.utils.data import Subset  # noqa: E402

import dose_response  # noqa: E402
from xai.paths import DATA_DIR  # noqa: E402

SUBSETS = {"train": 512, "val": 128}
SAMPLES = 2
ROAD_SMOKE_IMAGES = 3
SD4 = "pet_resnet18_pretrained"
MNIST = ("mnist", "patch", 0.1, 42)
STROKE_PIXELS = 96


def snapshot(folders):
    """{path: (size, mtime)} of every file under the folders."""
    return {p: (p.stat().st_size, p.stat().st_mtime_ns)
            for folder in folders if folder.exists() for p in folder.rglob("*") if p.is_file()}


def subset_split(original):
    """shortcut_split with the train and val splits cut to fixed small subsets
    (test splits are untouched: they need .patch_class, .mask and so on)."""
    def wrapped(base, shortcut, split, *args, **kwargs):
        data = original(base, shortcut, split, *args, **kwargs)
        if split in SUBSETS:
            return Subset(data, list(range(min(SUBSETS[split], len(data)))))
        return data
    return wrapped


def selection(base, shortcut, dose, seed, **extra):
    return argparse.Namespace(bases=[base], shortcuts=[shortcut], doses=[dose], seeds=[seed],
                              samples=SAMPLES, force=True, selection_only=True, models=None,
                              time_only=False, **extra)


def part_a():
    start = time.perf_counter()
    name = dose_response.model_name(SD4, "watermark", 0.0, 7)
    assert name == "pet_resnet18_pretrained_watermark_dose+000_seed7", name
    dose_response.train(selection(SD4, "watermark", 0.0, 7))
    metrics = json.loads((dose_response.MODEL_DIR / f"{name}_metrics.json").read_text())
    assert dose_response.model_set_of(metrics) == "A3_pilot", dose_response.model_set_of(metrics)
    assert not dose_response.in_model_set(metrics, "all"), "SD4 pilot pooled"
    print(f"(a) {name}: reliance {metrics['reliance']}%, no shortcut {metrics['test_accuracy_no_shortcut']}%, "
          f"{(time.perf_counter() - start) / 60:.1f} min", flush=True)
    return name


def part_b():
    start = time.perf_counter()
    args = selection(SD4, "watermark", 0.25, 7)
    name = dose_response.model_name(SD4, "watermark", 0.25, 7)
    assert name == "pet_resnet18_pretrained_watermark_dose+025_seed7", name
    dose_response.train(args)
    dose_response.evaluate(args)
    dose_response.placebo(args)
    dose_response.behaviour(args)

    metrics = json.loads((dose_response.MODEL_DIR / f"{name}_metrics.json").read_text())
    assert (metrics["swap_fraction"], metrics["shortcut_noise"]) == (0.25, 0.0), "dose params"
    assert metrics["shortcut"] == "watermark", metrics["shortcut"]
    assert dose_response.model_set_of(metrics) == "A3_pilot", dose_response.model_set_of(metrics)
    cache = dose_response.OUT / "per_model"
    for kind in ("oracle", "methods", "placebo", "random", "behaviour"):
        path = cache / f"{name}_{kind}.csv"
        assert path.exists(), f"missing {path.name}"
        assert pd.read_csv(path)["index"].nunique() == SAMPLES, f"{path.name}: wrong image count"

    test_set = dose_response.shortcut_split(SD4, "watermark", "test")
    scored = pd.read_csv(cache / f"{name}_oracle.csv")
    for index, label in zip(scored["index"], scored["label"]):
        mask = test_set.mask(index)
        assert mask.sum() == STROKE_PIXELS, f"image {index}: {mask.sum()} stroke pixels"
        assert np.array_equal(mask, test_set.mask_for_class(label)), f"image {index}: not the label's mark"
        placebo = dose_response.placebo_map(test_set, index, label)
        assert np.array_equal(placebo, test_set.mask_for_class(1 - label).astype(np.float64)), "placebo map"
        assert not (placebo.astype(bool) & mask).any(), "placebo overlaps the shown mark"
        assert (test_set.object_map(index)[mask] == 0).all(), f"image {index}: object map on the strokes"

    road_start = time.perf_counter()
    rows = dose_response.road_one(SD4, "watermark", 0.25, 7, SAMPLES)
    per_image = (time.perf_counter() - road_start) / SAMPLES
    assert len(rows) == SAMPLES, len(rows)
    assert sum(k.startswith("road_") for k in rows[0]) == 10, sorted(rows[0])
    print(f"(b) {name}: reliance {metrics['reliance']}%, road_one {per_image:.1f} s per image (CPU), "
          f"{(time.perf_counter() - start) / 60:.1f} min", flush=True)
    return name, per_image


def part_c(checkpoint_dir, pilot):
    start = time.perf_counter()
    name = dose_response.model_name(*MNIST)
    for suffix in (".pt", "_metrics.json"):
        shutil.copy2(checkpoint_dir / f"{name}{suffix}", dose_response.MODEL_DIR / f"{name}{suffix}")
    dose_response.ROAD_IMAGES["mnist"] = ROAD_SMOKE_IMAGES
    path = dose_response.OUT / "per_model" / f"{name}_road.csv"

    dose_response.road(argparse.Namespace(**{**vars(selection(*MNIST)), "force": False}))
    frame = pd.read_csv(path)
    assert len(frame) == ROAD_SMOKE_IMAGES, len(frame)
    road_columns = [c for c in frame.columns if c.startswith("road_")]
    assert len(road_columns) == 10, road_columns
    committed = pd.read_csv(ROOT / "results" / "dose_response" / "per_model" / f"{name}_oracle.csv")
    assert frame["index"].tolist() == committed["index"].tolist()[:ROAD_SMOKE_IMAGES], "ROAD images"
    first = path.read_bytes()

    dose_response.road(selection(*MNIST))  # force=True
    assert path.read_bytes() == first, "--force rerun changed the ROAD scores"

    stamp = path.stat().st_mtime_ns
    dose_response.road(argparse.Namespace(**{**vars(selection(*MNIST)), "force": False, "time_only": True}))
    timing = pd.read_csv(dose_response.OUT / "road_timing.csv")
    assert len(timing) == 1 and timing["model"].tolist() == [name], timing
    assert path.read_bytes() == first and path.stat().st_mtime_ns == stamp, "--time-only wrote _road.csv"

    try:
        dose_response.road(argparse.Namespace(**{**vars(selection(*MNIST)), "models": [pilot]}))
    except SystemExit as error:
        print(f"(c) road --models {pilot} refused: {error}", flush=True)
    else:
        raise AssertionError("road scored an SD4 pilot")
    # Without --models the pilots are simply not candidates.
    dose_response.road(argparse.Namespace(**{**vars(selection(SD4, "watermark", 0.0, 7)),
                                             "selection_only": False, "force": False}))
    assert not list((dose_response.OUT / "per_model").glob(f"{SD4}_watermark_*_road.csv")), "pilot ROAD file"
    print(f"(c) {name}: {ROAD_SMOKE_IMAGES} ROAD images, timing row {timing['minutes'].iloc[0]} min, "
          f"{(time.perf_counter() - start) / 60:.1f} min", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoint" / "dose")
    checkpoint_dir = parser.parse_args().checkpoint_dir.resolve()
    assert os.environ.get("XAI_DATA_DIR"), "set XAI_DATA_DIR to the data folder (read only)"
    watched = [ROOT / "checkpoint" / "dose", ROOT / "results", DATA_DIR, checkpoint_dir]
    before = snapshot(watched)
    tmp = Path(tempfile.mkdtemp(prefix="xai_smoke_a3_"))
    start = time.perf_counter()
    try:
        dose_response.MODEL_DIR = tmp / "checkpoint" / "dose"
        dose_response.OUT = tmp / "results" / "dose_response"
        dose_response.MODEL_DIR.mkdir(parents=True)
        dose_response.BASES[SD4] = 1
        dose_response.shortcut_split = subset_split(dose_response.shortcut_split)
        pilot = part_a()
        part_b()
        part_c(checkpoint_dir, pilot)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert not tmp.exists(), f"temporary directory left behind: {tmp}"
    after = snapshot(watched)
    new = sorted(str(p) for p in after.keys() - before.keys())
    assert not new, f"files written outside the temporary directory: {new[:5]}"
    changed = sorted(str(p) for p in before.keys() & after.keys() if before[p] != after[p])
    assert not changed, f"files changed outside the temporary directory: {changed[:5]}"
    print(f"total {(time.perf_counter() - start) / 60:.1f} min")
    print("smoke_a3: OK")


if __name__ == "__main__":
    main()

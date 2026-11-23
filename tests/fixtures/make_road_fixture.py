"""Make tests/fixtures/road_nli_fixture.npz from the reference ROAD code (run once).

    git clone https://github.com/tleemann/road_evaluation %TEMP%\\road_evaluation_ref
    .\\.venv\\Scripts\\python.exe tests/fixtures/make_road_fixture.py --reference %TEMP%\\road_evaluation_ref

Runs the reference NoisyLinearImputer (road/imputations.py, Rong et al., ICML
2022) without noise on two small float32 images in [0, 1] (3x9x11 and 1x8x8),
each with two keep-masks in the reference convention (1 = keep, 0 = impute):
a random one with about half the pixels removed, and one with the whole top
row, the whole left column and the bottom-right corner removed. Saves the
images, masks, outputs and the reference commit. tests/test_road.py checks
xai/road.py against these outputs.

The reference package __init__ imports modules that need the removed `imp`
module, so road/imputations.py is loaded on its own, with a stub for the GAIN
network import it does not use here.
"""
import argparse
import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import torch

OUT = Path(__file__).resolve().parent / "road_nli_fixture.npz"
SHAPES = [(3, 9, 11), (1, 8, 8)]


def load_imputer(reference):
    gsmv = types.ModuleType("road.gisp.models.gsmv")
    gsmv.GNet_ResNet = None  # only used by GAINImputer
    for name in ("road", "road.gisp", "road.gisp.models"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["road.gisp.models.gsmv"] = gsmv
    spec = importlib.util.spec_from_file_location("road_reference_imputations",
                                                  Path(reference) / "road" / "imputations.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.NoisyLinearImputer


def border_mask(shape):
    keep = np.ones(shape, dtype=np.float32)
    keep[0, :] = 0
    keep[:, 0] = 0
    keep[-1, -1] = 0
    return keep


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", required=True, help="clone of github.com/tleemann/road_evaluation")
    args = parser.parse_args()
    imputer = load_imputer(args.reference)(noise=0.0)
    commit = subprocess.run(["git", "-C", args.reference, "rev-parse", "HEAD"], check=True,
                            capture_output=True, text=True).stdout.strip()
    rng = np.random.default_rng(0)
    arrays = {"reference_commit": np.array(commit)}
    case = 0
    for shape in SHAPES:
        image = rng.random(shape, dtype=np.float32)
        random_keep = (rng.random(shape[1:]) >= 0.5).astype(np.float32)
        for keep in (random_keep, border_mask(shape[1:])):
            output = imputer(torch.tensor(image), torch.tensor(keep))
            arrays[f"image_{case}"] = image
            arrays[f"mask_{case}"] = keep
            arrays[f"output_{case}"] = output.numpy().astype(np.float32)
            case += 1
    arrays["cases"] = np.array(case)
    np.savez_compressed(OUT, **arrays)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes, {case} cases, reference {commit})")


if __name__ == "__main__":
    main()

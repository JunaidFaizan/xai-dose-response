"""Bit-identity check for the amendment-A3 code (not a unittest module).
    .\\.venv\\Scripts\\python.exe -W ignore tests/check_identity_a3.py --checkpoint-dir PATH

(a) Per-model bytes: scores mnist_patch_dose+010_seed42 with the working code
    (evaluate_one, evaluate_control_one for placebo and random, behaviour_one;
    100 images) and compares the CSV bytes with the committed
    results/dose_response/per_model files (or with --reference DIR, made once
    with --save DIR if the committed files cannot be reproduced on this machine).
(b) Analyze identity: runs `analyze --model-set all` and `--model-set pre_a2`
    (--model-sets) with --reps R on the base revision's code (git archive) and
    on the working code, each in its own parallel process on temporary copies of the committed
    checkpoint/dose/*_metrics.json (with zero-byte .pt files; analyze never
    loads weights) and results/dose_response/per_model/, and compares every
    output file byte for byte (PNG-only differences are reported separately).

Needs XAI_DATA_DIR (or data/ in this checkout) for part (a). Temporary
directories are always removed. Prints PASS or FAIL; exits 1 on FAIL.
"""
import argparse
import filecmp
import io
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL = ("mnist", "patch", 0.1, 42)
SAMPLES = 100
KINDS = ("oracle", "methods", "placebo", "random", "behaviour")
BASE_FILES = ["dose_response.py", "train_models.py", "utils.py", "xai"]

# Runs analyze for one model set with the code in the current directory.
RUNNER = """
import sys
from argparse import Namespace
from pathlib import Path
sys.path.insert(0, ".")
import dose_response
dose_response.MODEL_DIR = Path(sys.argv[1])
dose_response.OUT = Path(sys.argv[2])
dose_response.analyze(Namespace(model_set=sys.argv[4], reps=int(sys.argv[3])))
"""


def part_a(checkpoint_dir, reference, save):
    import pandas as pd
    import dose_response

    dose_response.MODEL_DIR = Path(checkpoint_dir)
    name = dose_response.model_name(*MODEL)
    rows = {}
    rows["oracle"], rows["methods"] = dose_response.evaluate_one(*MODEL, SAMPLES)
    for control in dose_response.CONTROL_MAPS:
        rows[control] = dose_response.evaluate_control_one(*MODEL, SAMPLES, control)
    rows["behaviour"] = dose_response.behaviour_one(*MODEL, SAMPLES)
    expected_dir = Path(reference) if reference else ROOT / "results" / "dose_response" / "per_model"
    failures = []
    with tempfile.TemporaryDirectory(prefix="xai_identity_a_") as tmp:
        for kind in KINDS:
            path = Path(tmp) / f"{name}_{kind}.csv"
            pd.DataFrame(rows[kind]).to_csv(path, index=False)
            if save:
                Path(save).mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, Path(save) / path.name)
            if path.read_bytes() != (expected_dir / path.name).read_bytes():
                failures.append(path.name)
    return failures


def archive_base(rev, target):
    data = subprocess.run(["git", "-C", str(ROOT), "archive", "--format=tar", rev, *BASE_FILES],
                          check=True, capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        tar.extractall(target, filter="data")


def output_files(out):
    return {p.relative_to(out).as_posix() for p in out.rglob("*")
            if p.is_file() and p.relative_to(out).parts[0] != "per_model"}


def part_b(rev, reps, model_sets):
    """One process per (code, model set), all in parallel, each with its own OUT."""
    tmp = Path(tempfile.mkdtemp(prefix="xai_identity_b_"))
    try:
        base_code, model_dir = tmp / "base_code", tmp / "models"
        base_code.mkdir()
        model_dir.mkdir()
        archive_base(rev, base_code)
        for path in sorted((ROOT / "checkpoint" / "dose").glob("*_metrics.json")):
            shutil.copyfile(path, model_dir / path.name)
            (model_dir / f"{path.name[:-len('_metrics.json')]}.pt").touch()
        runs = {}
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        for model_set in model_sets:
            for label, code in (("base", base_code), ("working", ROOT)):
                out = tmp / f"out_{label}_{model_set}"
                shutil.copytree(ROOT / "results" / "dose_response" / "per_model", out / "per_model")
                log = open(tmp / f"{label}_{model_set}.log", "w", encoding="utf-8", errors="replace")
                runs[label, model_set] = (out, log, subprocess.Popen(
                    [sys.executable, "-W", "ignore", "-c", RUNNER, str(model_dir), str(out), str(reps), model_set],
                    cwd=code, stdout=log, stderr=subprocess.STDOUT, env=env))
        for (label, model_set), (out, log, process) in runs.items():
            process.wait()
            log.close()
            if process.returncode:
                tail = (tmp / f"{label}_{model_set}.log").read_text(encoding="utf-8", errors="replace")[-3000:]
                raise SystemExit(f"analyze --model-set {model_set} failed on the {label} code:\n{tail}")
        count, only, differ = 0, [], []
        for model_set in model_sets:
            base_out, work_out = runs["base", model_set][0], runs["working", model_set][0]
            base_files, work_files = output_files(base_out), output_files(work_out)
            count += len(base_files)
            only += [f"{model_set}: {f}" for f in sorted(base_files ^ work_files)]
            differ += [f"{model_set}: {f}" for f in sorted(base_files & work_files)
                       if not filecmp.cmp(base_out / f, work_out / f, shallow=False)]
        return count, only, [f for f in differ if not f.endswith(".png")], [f for f in differ if f.endswith(".png")]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-dir", default=str(ROOT / "checkpoint" / "dose"))
    parser.add_argument("--base-rev", default="284a980")
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--model-sets", nargs="+", default=["all", "pre_a2"], choices=["all", "original", "pre_a2"])
    parser.add_argument("--reference", help="compare part (a) with the CSVs in this folder")
    parser.add_argument("--save", help="also save the part (a) CSVs to this folder")
    args = parser.parse_args()
    ok = True

    start = time.perf_counter()
    failures = part_a(args.checkpoint_dir, args.reference, args.save)
    print(f"(a) per-model bytes of {'_'.join(map(str, MODEL))}: "
          + ("equal" if not failures else f"DIFFER: {', '.join(failures)}")
          + f" ({time.perf_counter() - start:.0f} s)", flush=True)
    ok &= not failures

    start = time.perf_counter()
    count, only, differ, png = part_b(args.base_rev, args.reps, args.model_sets)
    print(f"(b) analyze {' + '.join(args.model_sets)} (--reps {args.reps}), base {args.base_rev} vs working code: "
          f"{count} output files, {len(only)} only on one side, {len(differ)} differ, "
          f"{len(png)} PNGs differ ({time.perf_counter() - start:.0f} s)", flush=True)
    for label, files in (("only on one side", only), ("differ", differ), ("PNG differs", png)):
        for f in files:
            print(f"    {label}: {f}")
    ok &= not (only or differ or png)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

"""Builds the Zenodo research-data release package in release/zenodo/ (git-ignored; nothing is uploaded).

    .venv/Scripts/python.exe paper/make_release_package.py

Run it after the last commit (or give the commit as an argument): the code zip is `git archive <rev>` (tracked files only) and is named with the
short commit hash. Writes:
  xai-dose-response_code_<sha>.zip          tracked files of HEAD
  xai-dose-response_results_<sha>.zip       results/dose_response (CSVs, per_model files, figures)
  xai-dose-response_checkpoints_partNofM.zip  checkpoint/dose/*.pt and *_metrics.json, each part under 1.3 GB
  README.md                                 copy of docs/submission/RELEASE_README.md
  MANIFEST.md                               SHA-256 and size of every file, and a file inventory
"""
import hashlib
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "release" / "zenodo"
PART_LIMIT = 1.3 * 1024 ** 3  # far below Zenodo's 50 GB per record; several parts keep each upload small


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def main(rev="HEAD"):
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*"):
        old.unlink()
    sha = git("rev-parse", "--short", rev)
    dirty = git("status", "--porcelain", "--untracked-files=no") if rev == "HEAD" else ""
    code = OUT / f"xai-dose-response_code_{sha}.zip"
    subprocess.run(["git", "archive", "--format=zip", "-o", str(code), rev], cwd=ROOT, check=True)

    results = OUT / f"xai-dose-response_results_{sha}.zip"
    files = sorted(p for p in (ROOT / "results" / "dose_response").rglob("*") if p.is_file())
    with zipfile.ZipFile(results, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(ROOT).as_posix())

    models = sorted((ROOT / "checkpoint" / "dose").glob("*.pt"))
    parts, current, size = [], [], 0
    for pt in models:
        extra = pt.stat().st_size
        if current and size + extra > PART_LIMIT:
            parts.append(current)
            current, size = [], 0
        current.append(pt)
        size += extra
    parts.append(current)
    ckpt = []
    for i, part in enumerate(parts, 1):
        name = OUT / f"xai-dose-response_checkpoints_part{i}of{len(parts)}.zip"
        with zipfile.ZipFile(name, "w", zipfile.ZIP_STORED) as z:  # .pt files do not compress
            for pt in part:
                for f in (pt, pt.with_name(pt.stem + "_metrics.json")):
                    if f.exists():
                        z.write(f, f.relative_to(ROOT).as_posix())
        ckpt.append((name, part))

    shutil.copyfile(ROOT / "docs" / "submission" / "RELEASE_README.md", OUT / "README.md")
    zips = [code, results] + [n for n, _ in ckpt]
    lines = ["# Manifest of the Zenodo release package", "",
             f"Code revision: `{sha}` (`git archive {rev}`, tracked files only)."
             + (" WARNING: tracked files had uncommitted changes when this package was built." if dirty else ""), "",
             "| File | Size (bytes) | SHA-256 |", "|---|---|---|"]
    for z in [OUT / "README.md"] + zips:
        lines.append(f"| {z.name} | {z.stat().st_size} | {sha256(z)} |")
    lines += ["", "## Inventory", ""]
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            names = zf.namelist()
        lines.append(f"- `{z.name}`: {len(names)} files")
    lines += ["", f"results/dose_response: {len(files)} files, listed in the results zip.", "",
              f"checkpoint/dose: {len(models)} `.pt` files and their `_metrics.json` files in "
              f"{len(parts)} parts:", ""]
    for name, part in ckpt:
        lines.append(f"- `{name.name}`: {len(part)} models, {sum(p.stat().st_size for p in part) / 1024 ** 3:.2f} GB")
    lines += ["", "## Checkpoint files", ""] + [f"- {p.name} ({p.stat().st_size} bytes)" for p in models]
    (OUT / "MANIFEST.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    for z in [OUT / "README.md", OUT / "MANIFEST.md"] + zips:
        print(f"{z.stat().st_size / 1024 ** 2:10.1f} MB  {z}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "HEAD")  # optional: a commit to archive, for example 0e128c4

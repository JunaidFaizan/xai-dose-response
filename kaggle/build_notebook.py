"""Writes the Kaggle notebooks (run: python kaggle/build_notebook.py):
    kaggle/dose_response_resnet18.ipynb   CIFAR-10 with ResNet-18
    kaggle/dose_response_pet.ipynb        Oxford-IIIT Pet (cat vs dog) with ResNet-18
"""
import json
from pathlib import Path

SELECTION = ("--bases cifar10_resnet18 --shortcuts patch "
             "--doses -0.8 -0.6 -0.4 -0.25 -0.1 0 --seeds 42 1 2")
# Matches PET_DOSES and PET_SEEDS in dose_response.py.
PET_SELECTION = ("--bases pet_resnet18 --shortcuts patch "
                 "--doses -0.4 -0.3 -0.2 -0.1 -0.05 0 --seeds 42 1 2")

CIFAR_INTRO = """# Dose-response study: CIFAR-10 with ResNet-18

Trains 18 ResNet-18 models (6 reliance doses x 3 seeds, patch shortcut), scores
them with every metric and control map, and packs the results into
`resnet18_results.zip` (in the notebook's Output) to copy back to the laptop.

Before running: **Settings → Accelerator → GPU T4** (or P100), and **Internet on**
(needed once, to install Captum). Then use **Save Version → Save & Run All (Commit)**
so it keeps running with the browser closed."""

PET_INTRO = """# Dose-response study: Oxford-IIIT Pet with ResNet-18

Trains 18 ResNet-18 models from scratch on cat vs dog photos (6 reliance doses x
3 seeds, 12x12 patch shortcut, 30 epochs, 128x128 images), scores them with every
metric and control map, and packs the results into `pet_results.zip` (in the
notebook's Output) to copy back to the laptop. The images come preprocessed in
the bundle (`data/pet128/*.npz`), so nothing is downloaded except Captum.

Before running: **Settings → Accelerator → GPU T4** (or P100), and **Internet on**
(needed once, to install Captum). Then use **Save Version → Save & Run All (Commit)**
so it keeps running with the browser closed."""

PET_DATA_CHECK = """
for name in ('trainval.npz', 'test.npz'):
    assert os.path.exists(f'data/pet128/{name}'), f'data/pet128/{name} missing: rebuild the bundle with -Dataset pet.'
print('Pet data found')"""


def cells_for(intro, selection, prefix, zip_name, data_check=""):
    return [
        ("markdown", intro),
        ("code", """# 1. Copy the uploaded project into the writable working folder
import glob, os, shutil, subprocess, sys
found = glob.glob('/kaggle/input/**/dose_response.py', recursive=True)
assert found, 'Project files not found: add your uploaded dataset with "+ Add Input".'
src = os.path.dirname(found[0])
WORK = '/kaggle/working/project'
if not os.path.exists(WORK):
    shutil.copytree(src, WORK)
os.chdir(WORK)
print('project copied from', src)
print(sorted(os.listdir(WORK)))"""),
        ("code", """# 2. Install the one missing package, and check the GPU
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'captum==0.9.0'], check=True)
import torch
print('torch', torch.__version__, '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
assert torch.cuda.is_available(), 'No GPU: turn on Settings → Accelerator → GPU, then restart.'""" + data_check),
        ("code", f"""# 3. Train (prints validation accuracy per epoch and minutes per model).
#    Re-running skips models that already finished.
!python -u -W ignore dose_response.py train {selection}"""),
        ("code", f"""# 4. Score every model with all metrics, oracle maps and controls
!python -u -W ignore dose_response.py evaluate {selection}
!python -u -W ignore dose_response.py placebo {selection}"""),
        ("code", f"""# 5. Pack the results to download
import zipfile
out = '/kaggle/working/{zip_name}'
files = glob.glob('checkpoint/dose/{prefix}*') + glob.glob('results/dose_response/per_model/{prefix}*')
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(f, f)  # keeps checkpoint/... and results/... paths
print(len(files), 'files ->', out)
shutil.rmtree(WORK)  # keep only the zip in the Output"""),
    ]


def cell(kind, text):
    lines = text.splitlines(keepends=True)
    base = {"cell_type": kind, "metadata": {}, "source": lines}
    return base if kind == "markdown" else {**base, "execution_count": None, "outputs": []}


def write(name, cells):
    notebook = {
        "cells": [cell(k, t) for k, t in cells],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    path = Path(__file__).with_name(name)
    path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print("wrote", path)


write("dose_response_resnet18.ipynb",
      cells_for(CIFAR_INTRO, SELECTION, "cifar10_resnet18_", "resnet18_results.zip"))
write("dose_response_pet.ipynb",
      cells_for(PET_INTRO, PET_SELECTION, "pet_resnet18_", "pet_results.zip", PET_DATA_CHECK))

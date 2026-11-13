"""Writes the Kaggle notebooks (run: python kaggle/build_notebook.py):
    kaggle/dose_response_resnet18.ipynb   CIFAR-10 with ResNet-18
    kaggle/dose_response_pet.ipynb        Oxford-IIIT Pet (cat vs dog) with ResNet-18
    kaggle/dose_response_a2_session1.ipynb  amendment A2: SD2 pilot, Pet dose -0.15, SD1 pilot
    kaggle/dose_response_a2_session2.ipynb  amendment A2: final SD2 CIFAR-10 and SD1 doses
"""
import json
from pathlib import Path

SELECTION = ("--bases cifar10_resnet18 --shortcuts patch "
             "--doses -0.8 -0.6 -0.4 -0.25 -0.1 0 --seeds 42 1 2")
# Matches PET_DOSES and PET_SEEDS in dose_response.py.
PET_SELECTION = ("--bases pet_resnet18 --shortcuts patch "
                 "--doses -0.5 -0.4 -0.3 -0.2 -0.1 0 --seeds 42 1 2")

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


# ------------------------------------------------ amendment A2 (SD1, SD2)

TITLE = ("A Dose-Response Evaluation of Explanation Metrics under Controlled Shortcut "
         "Reliance in CNN Image Classifiers")
SD1_PILOT = "--bases pet_resnet18_pretrained --shortcuts patch --doses 0 -0.5 --seeds 7"
CIFAR_PILOT = "--bases cifar10_resnet18 --shortcuts patch --doses -0.05 --seeds 42"
PET_A2 = "--bases pet_resnet18 --shortcuts patch --doses -0.15 --seeds 42 1 2"

A2_SESSION1_INTRO = f"""# {TITLE}

## Amendment A2, session 1: SD2 pilot, Pet dose -0.15 and the SD1 pilot

Pre-registered in `docs/PREDICTIONS_amendments.md` (A2). Run only after both
prediction files are uploaded to OSF or Zenodo and the placeholders are filled.

- Cell 3: SD1 pilot, ImageNet-pretrained ResNet-18 on Pet, doses 0 and -0.5,
  seed 7. The first printed model line gives the minutes per pretrained model.
- Cell 4: SD2 CIFAR-10 ResNet-18 pilot, dose -0.05, seed 42 (training only).
- Cells 5-6: Pet from scratch at dose -0.15, seeds 42 1 2, trained and scored.
- Cell 7: pilot report (reliance, accuracies, r_c and the A2 dose set) in
  `a2_pilot_report.json`. Pilots are not scored: A2 lets only their reliance
  and accuracies be read before the dated completion notes are committed.
- Cell 8: `a2_session1_results.zip` in the notebook's Output.

Before running: **Settings → Accelerator → GPU T4 x2**, and **Internet on**
(Captum and the ImageNet weights). Then **Save Version → Save & Run All (Commit)**."""

A2_DATA_CHECK = """
for name in ('trainval.npz', 'test.npz'):
    assert os.path.exists(f'data/pet128/{name}'), f'data/pet128/{name} missing: rebuild the bundle with -Dataset a2.'
assert os.path.isdir('data/cifar-10-batches-py'), 'data/cifar-10-batches-py missing: rebuild the bundle with -Dataset a2.'
print('Pet and CIFAR-10 data found')"""

WEIGHTS_DOWNLOAD = """
from torchvision.models import ResNet18_Weights, resnet18
try:
    resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # cached for the training cells
except Exception as error:
    raise RuntimeError('ImageNet weights download failed: turn Internet on (Settings → Internet).') from error
print('ImageNet ResNet-18 weights ready')"""

COPY_PROJECT = """# 1. Copy the uploaded project into the writable working folder
import glob, os, shutil, subprocess, sys
found = glob.glob('/kaggle/input/**/dose_response.py', recursive=True)
assert found, 'Project files not found: add your uploaded dataset with "+ Add Input".'
src = os.path.dirname(found[0])
WORK = '/kaggle/working/project'
if not os.path.exists(WORK):
    shutil.copytree(src, WORK)
os.chdir(WORK)
print('project copied from', src)
print(sorted(os.listdir(WORK)))"""

SETUP = """# 2. Install the one missing package, check the GPU and the data, get the ImageNet weights
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'captum==0.9.0'], check=True)
import torch
print('torch', torch.__version__, '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
assert torch.cuda.is_available(), 'No GPU: turn on Settings → Accelerator → GPU, then restart.'""" + A2_DATA_CHECK

PILOT_REPORT = """# 7. Pilot report: reliance, accuracies, r_c and the A2 dose set (written to a2_pilot_report.json)
import json
sys.path.insert(0, WORK)
import dose_response
from xai.data import num_classes
from xai.stats import chance_floor, corrected_reliance
PILOTS = [('pet_resnet18_pretrained', 0.0, 7), ('pet_resnet18_pretrained', -0.5, 7), ('cifar10_resnet18', -0.05, 42)]
report = []
for base, dose, seed in PILOTS:
    name = dose_response.model_name(base, 'patch', dose, seed)
    m = json.load(open(f'checkpoint/dose/{name}_metrics.json'))
    floor = float(chance_floor(m['test_accuracy_no_shortcut'], num_classes(base)))
    rc = float(corrected_reliance(m['reliance'], floor))
    entry = {'model': name, 'reliance': m['reliance'], 'test_accuracy': m['test_accuracy'],
             'test_accuracy_no_shortcut': m['test_accuracy_no_shortcut'], 'chance_floor': round(floor, 2),
             'reliance_corrected': round(rc, 2), 'training_minutes': m['training_minutes']}
    # The A2 rules use the CIFAR pilot (dose -0.05) and the SD1 pilot at dose 0.
    if dose != -0.5:
        entry['a2_dose_set'] = dose_response.a2_dose_set(base, rc)
    report.append(entry)
    print(entry)
json.dump(report, open('a2_pilot_report.json', 'w'), indent=2)"""

SESSION1_ZIP = """# 8. Pack the results to download (explicit name patterns: the prefix
#    pet_resnet18_ alone would also match pet_resnet18_pretrained_)
import zipfile
out = '/kaggle/working/a2_session1_results.zip'
patterns = ['checkpoint/dose/pet_resnet18_pretrained_patch_*',
            'checkpoint/dose/cifar10_resnet18_patch_dose-005_seed42*',
            'checkpoint/dose/pet_resnet18_patch_dose-015_*',
            'results/dose_response/per_model/pet_resnet18_patch_dose-015_*',
            'a2_pilot_report.json']
files = sorted({f for p in patterns for f in glob.glob(p)})
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(f, f)  # keeps checkpoint/... and results/... paths
print(len(files), 'files ->', out)
shutil.rmtree(WORK)  # keep only the zip in the Output"""

A2_SESSION1 = [
    ("markdown", A2_SESSION1_INTRO),
    ("code", COPY_PROJECT),
    ("code", SETUP + WEIGHTS_DOWNLOAD),
    ("code", f"""# 3. SD1 pilot: ImageNet-pretrained ResNet-18 on Pet, doses 0 and -0.5, seed 7.
#    The first model line (pet_resnet18_pretrained_patch_dose+000_seed7) gives the minutes per model.
!python -u -W ignore dose_response.py train {SD1_PILOT}"""),
    ("code", f"""# 4. SD2 pilot: CIFAR-10 ResNet-18, dose -0.05, seed 42 (training only; not scored until the A2 note)
!python -u -W ignore dose_response.py train {CIFAR_PILOT}"""),
    ("code", f"""# 5. SD2: Pet from scratch at dose -0.15, seeds 42 1 2
!python -u -W ignore dose_response.py train {PET_A2}"""),
    ("code", f"""# 6. Score the Pet dose -0.15 models (metrics, oracle maps, controls, behaviour labels).
#    The pilots have no scores, so placebo skips them.
!python -u -W ignore dose_response.py evaluate {PET_A2}
!python -u -W ignore dose_response.py placebo {PET_A2}
!python -u -W ignore dose_response.py behaviour --selection-only {PET_A2}"""),
    ("code", PILOT_REPORT),
    ("code", SESSION1_ZIP),
]

A2_SESSION2_INTRO = f"""# {TITLE}

## Amendment A2, session 2: final SD2 CIFAR-10 ResNet-18 doses and the SD1 confirmatory set

Run only after the two dated A2 completion notes (CIFAR pilot r_c → dose set;
SD1 pilot r_c → dose set) are committed in `docs/PREDICTIONS_amendments.md`.

1. Fill `CIFAR_DOSES` and `SD1_DOSES` in the parameters cell from those notes.
2. **+ Add Input**: the bundle dataset and the Output of the session-1 notebook
   (it holds `a2_session1_results.zip`, so the CIFAR pilot checkpoint is reused).
3. **Settings → Accelerator → GPU T4 x2**, **Internet on**, then
   **Save Version → Save & Run All (Commit)**.

Each non-empty list is trained with seeds 42 1 2 and scored; the results are
packed into `a2_session2_results.zip` in the notebook's Output."""

A2_PARAMETERS = """# Parameters: copy the final doses from the dated A2 completion notes
# (docs/PREDICTIONS_amendments.md), e.g. CIFAR_DOSES = [-0.09, -0.075, -0.06, -0.05].
CIFAR_DOSES = []  # cifar10_resnet18, seeds 42 1 2
SD1_DOSES = []    # pet_resnet18_pretrained, seeds 42 1 2
SEEDS = [42, 1, 2]
assert CIFAR_DOSES or SD1_DOSES, 'Fill CIFAR_DOSES and/or SD1_DOSES from the A2 completion notes first.'"""

EXTRACT_SESSION1 = """
# Reuse the session-1 checkpoints (train skips models whose metrics JSON exists)
import zipfile
for archive in glob.glob('/kaggle/input/**/a2_session1_results.zip', recursive=True):
    with zipfile.ZipFile(archive) as z:
        z.extractall(WORK)
    print('extracted', archive)"""

SESSION2_RUN = """# 3. Train and score every model of the filled lists
def run(*args):
    subprocess.run([sys.executable, '-u', '-W', 'ignore', 'dose_response.py', *args], check=True)

JOBS = [(base, doses) for base, doses in (('cifar10_resnet18', CIFAR_DOSES), ('pet_resnet18_pretrained', SD1_DOSES))
        if doses]
for base, doses in JOBS:
    selection = ['--bases', base, '--shortcuts', 'patch', '--doses', *map(str, doses),
                 '--seeds', *map(str, SEEDS)]
    run('train', *selection)
    run('evaluate', *selection)
    run('placebo', *selection)
    run('behaviour', '--selection-only', *selection)"""

SESSION2_ZIP = """# 4. Pack the results to download (only the models of this session's lists)
sys.path.insert(0, WORK)
import dose_response
names = [dose_response.model_name(base, 'patch', dose, seed) for base, doses in JOBS for dose in doses for seed in SEEDS]
files = sorted({f for n in names for f in glob.glob(f'checkpoint/dose/{n}.pt') + glob.glob(f'checkpoint/dose/{n}_metrics.json')
                + glob.glob(f'results/dose_response/per_model/{n}_*.csv')})
out = '/kaggle/working/a2_session2_results.zip'
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(f, f)  # keeps checkpoint/... and results/... paths
print(len(files), 'files for', len(names), 'models ->', out)
shutil.rmtree(WORK)  # keep only the zip in the Output"""

A2_SESSION2 = [
    ("markdown", A2_SESSION2_INTRO),
    ("code", A2_PARAMETERS),
    ("code", COPY_PROJECT + "\n" + EXTRACT_SESSION1),
    ("code", SETUP + "\nif SD1_DOSES:" + WEIGHTS_DOWNLOAD.replace("\n", "\n    ")),
    ("code", SESSION2_RUN),
    ("code", SESSION2_ZIP),
]


write("dose_response_resnet18.ipynb",
      cells_for(CIFAR_INTRO, SELECTION, "cifar10_resnet18_", "resnet18_results.zip"))
write("dose_response_pet.ipynb",
      cells_for(PET_INTRO, PET_SELECTION, "pet_resnet18_", "pet_results.zip", PET_DATA_CHECK))
write("dose_response_a2_session1.ipynb", A2_SESSION1)
write("dose_response_a2_session2.ipynb", A2_SESSION2)

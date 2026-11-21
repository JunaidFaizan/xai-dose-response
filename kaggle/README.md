# A Dose-Response Evaluation of Explanation Metrics under Controlled Shortcut Reliance in CNN Image Classifiers

## Running the ResNet-18 experiment on Kaggle (free GPU)

Trains 18 ResNet-18 models on CIFAR-10 (6 reliance doses × 3 seeds, patch shortcut) and scores them exactly like the other dose-response models. The results are then copied back into this project and analysed together with the existing 156 models.

## One-time setup

1. Create a free account at https://www.kaggle.com.
2. Verify your phone number: **Settings → Phone verification**. Without it, notebooks can't use a GPU or the internet.

## 1. Make the upload file (on the laptop)

From the project folder:

```powershell
pwsh -File kaggle\make_bundle.ps1
```

This creates `kaggle\xai_dose_bundle.zip` (about 164 MB): the code plus the CIFAR-10 data. Re-run it whenever the code changes.

## 2. Upload it as a private Kaggle dataset

1. On Kaggle: **Create → New Dataset**.
2. Drag in `xai_dose_bundle.zip`. Kaggle unpacks it automatically.
3. Name it, for example, `xai-dose-bundle`, keep it **Private**, and click **Create**.

## 3. Create the notebook

1. **Create → New Notebook → File → Import Notebook**, and choose `kaggle\dose_response_resnet18.ipynb`.
2. In the right-hand panel: **+ Add Input** → **Your Datasets** → `xai-dose-bundle`.
3. **Settings → Accelerator → GPU T4 x2** (or **GPU P100**).
4. **Settings → Internet → On** (needed once, to install Captum).

## 4. Test, then run it fully

1. Run cells 1 and 2 (the play button on each cell). Cell 2 must print a GPU name.
2. Run cell 3 and watch the first model. It prints validation accuracy per epoch, then a line like `cifar10_resnet18_patch_dose-080_seed42: reliance ... | 9.5 min`. Stop the cell (■) after that line.
   - Multiply the minutes by 18 for the total training time.
   - The free GPU quota is about 30 hours per week and one run is limited to about 12 hours, so the total must stay under about 11 hours. If it doesn't, tell me and we'll cut seeds or doses.
3. Click **Save Version → Save & Run All (Commit) → Save**. This runs the whole notebook on Kaggle's servers, so you can close the browser. It skips nothing it hasn't finished, so the first model is simply retrained.
4. Check progress under **View Active Events** (bottom left) or the notebook's **Versions** list.

## 5. Bring the results back (on the laptop)

1. When the version finishes, open it → **Output** → download `resnet18_results.zip`.
2. Unzip it **into the project folder** (it contains `checkpoint\dose\...` and `results\dose_response\per_model\...`, so the files land in the right place):

```powershell
Expand-Archive $HOME\Downloads\resnet18_results.zip -DestinationPath . -Force
```

3. Re-run the analysis. It rebuilds the combined score files and merges the ResNet-18 models with the existing ones:

```powershell
.\.venv\Scripts\python.exe -W ignore dose_response.py analyze
```

ResNet-18 then appears as `cifar10_resnet18` in `results\dose_response\summary.md` and as an extra panel in every figure.

## Oxford-IIIT Pet (cat vs dog, 128×128)

The same steps, with a second bundle and notebook. It trains 18 ResNet-18 models from scratch (6 doses × 3 seeds, 30 epochs each).

1. On the laptop, preprocess the data once and build the bundle:

   ```powershell
   .\.venv\Scripts\python.exe prepare_pet.py
   pwsh -File kaggle\make_bundle.ps1 -Dataset pet
   ```

   This creates `kaggle\xai_pet_bundle.zip` (the code plus `data\pet128\*.npz`, so Kaggle downloads no images). If the Pet download fails, `prepare_pet.py` prints manual download steps with `curl.exe`.
2. Upload `xai_pet_bundle.zip` as a new **Private** dataset, for example `xai-pet-bundle`.
3. **Import Notebook** → `kaggle\dose_response_pet.ipynb`; **+ Add Input** → `xai-pet-bundle`; GPU on; Internet on.
4. Run cells 1 and 2 (cell 2 must print a GPU name and "Pet data found"). Run cell 3 and stop it (■) after the first line like `pet_resnet18_patch_dose-040_seed42: reliance ... | N min`.
   - Multiply N by 18 for the training time. Evaluation (cell 4) comes on top and could not be timed on the laptop, so leave a margin: training should stay under about 8 hours, the whole run under about 11. If it doesn't fit, tell me and we'll cut seeds or doses.
   - Also tell me the reliance of that first model: the doses may need adjusting.
   - If the cell stops with **CUDA out of memory**, tell me (the batch size would need lowering).
5. **Save Version → Save & Run All (Commit)**.
6. When it finishes, download `pet_results.zip` from **Output** and unzip it into the project folder, then re-run the analysis (it merges all cached models; Pet appears as `pet_resnet18`):

   ```powershell
   Expand-Archive $HOME\Downloads\pet_results.zip -DestinationPath . -Force
   .\.venv\Scripts\python.exe -W ignore dose_response.py analyze
   ```

## Amendment A2: session 1 (SD2 pilot + Pet −0.15, SD1 pilot)

Pre-registered in `docs/PREDICTIONS_amendments.md` (A2) with the predictions P5–P9 in `docs/PREDICTIONS_pretrained.md`. Session 1 trains the SD1 pilot (ImageNet-pretrained ResNet-18 on Pet, doses 0 and −0.5, seed 7), the SD2 CIFAR-10 ResNet-18 pilot (dose −0.05, seed 42) and the 3 Pet models at dose −0.15 (seeds 42, 1, 2), and scores only the Pet −0.15 models. A2 lets only the pilots' reliance and accuracies be read before the dated completion notes are committed.

0. **Before the session (A2 requires it):** upload `docs/PREDICTIONS_amendments.md` and `docs/PREDICTIONS_pretrained.md` to OSF or Zenodo, fill both `[PLACEHOLDER: ...]` lines with the DOI/URL and the upload timestamp, and commit.
1. Build the bundle on the laptop (code, `data\pet128\*.npz` and the CIFAR-10 data):

   ```powershell
   pwsh -File kaggle\make_bundle.ps1 -Dataset a2
   ```

   This creates `kaggle\xai_a2_bundle.zip` (not tracked by git).
2. On Kaggle, upload it as a new **Private** dataset, for example `xai-a2-bundle`.
3. **Create → New Notebook → File → Import Notebook** → `kaggle\dose_response_a2_session1.ipynb`.
4. **+ Add Input** → **Your Datasets** → `xai-a2-bundle`.
5. **Settings → Accelerator → GPU T4 x2**, and **Settings → Internet → On** (Captum and the ImageNet ResNet-18 weights are downloaded).
6. Run cells 1–3. Cell 2 must print a GPU name, "Pet and CIFAR-10 data found" and "ImageNet ResNet-18 weights ready". In cell 3, read the first line like `pet_resnet18_pretrained_patch_dose+000_seed7: reliance ... | N min` (the minutes per pretrained model and the pilot's reliance), then stop the cell (■).
7. **Save Version → Save & Run All (Commit) → Save**. The whole notebook then runs on Kaggle's servers; the stopped pilot model is simply retrained.
8. When the version finishes, open it → **Output** → download `a2_session1_results.zip`.
9. On the laptop, unzip it into the project folder and re-run the analyses:

   ```powershell
   Expand-Archive $HOME\Downloads\a2_session1_results.zip -DestinationPath . -Force
   .\.venv\Scripts\python.exe -W ignore dose_response.py behaviour
   .\.venv\Scripts\python.exe -W ignore dose_response.py analyze
   .\.venv\Scripts\python.exe -W ignore dose_response.py analyze --model-set original
   .\.venv\Scripts\python.exe -W ignore dose_response.py analyze --model-set pre_a2
   ```

   `behaviour` does nothing for the Kaggle-scored models (their behaviour files are in the zip). `analyze` includes the 3 Pet −0.15 models; the pilots have no scores and are not pooled; `analyze` lists them in `results\dose_response\pilots.csv` and in a separate section of `summary.md`. `--model-set pre_a2` writes the 234 models from before A2 to `results\dose_response\pre_a2_models\`.
10. Read `a2_pilot_report.json` (unzipped into the project folder): reliance, accuracies, r_c and the A2 dose set for the CIFAR pilot and the SD1 pilot at dose 0. Write the two dated A2 completion notes in `docs/PREDICTIONS_amendments.md` (CIFAR pilot r_c → dose set; SD1 pilot r_c → dose set), add them to the timestamped record, and commit them **before session 2**.

Dose −0.05 on Pet is **not** part of A2. Adding it needs a new dated amendment before it is trained. `analyze` labels any model outside the registered designs `unregistered`, prints a warning and leaves it out of every model set.

**Session 1 result (2026-11-17).** Completion notes A2-C1 and A2-C2 in `docs/PREDICTIONS_amendments.md`; the pilot report is kept as `docs/kaggle_runs/a2_session1_pilot_report.json` (the copy in the project root is ignored by git). CIFAR pilot r_c 48.18 → CIFAR doses −0.05, −0.03, −0.02, −0.01. SD1 pilot r_c 31.96 at dose 0 → SD1 doses −0.5, −0.2, 0, +0.1, +0.25, +0.4. The 3 Pet −0.15 models are pooled in `analyze`; the pilots are listed in `results\dose_response\pilots.csv`, not pooled.

## Amendment A2: session 2 (final SD2 CIFAR-10 doses, SD1 confirmatory set)

Run only after the two completion notes are committed (done: A2-C1 and A2-C2) **and** added as a new version of the Zenodo record (step 0).

0. **Timestamp the completion notes.** On Zenodo, open record [10.5281/zenodo.24285532](https://doi.org/10.5281/zenodo.24285532) → **New version** → replace `PREDICTIONS_amendments.md` with the committed file (keep `PREDICTIONS_pretrained.md`) → **Publish**. Then replace the `TODO (author)` line in the A2 completion notes with the new version's DOI and publication time, and commit.
1. **Update the bundle dataset.** The code in the bundle changed after session 1 (`dose_response.py`: model-set rules and `pilots.csv`; `xai/models.py`: docstring). Training and scoring are unchanged, but the bundle should match the committed code. Rebuild it on the laptop and upload it to the existing dataset: Kaggle → **Your Work → Datasets → `xai-a2-bundle` → New Version** → drop in `kaggle\xai_a2_bundle.zip` → **Create**.

   ```powershell
   pwsh -File kaggle\make_bundle.ps1 -Dataset a2
   ```

2. **Create → New Notebook → File → Import Notebook** → `kaggle\dose_response_a2_session2.ipynb`.
3. **+ Add Input → Your Datasets** → `xai-a2-bundle` (latest version).
4. **+ Add Input → Your Work → Notebooks** → the session-1 notebook (the committed version whose Output holds `a2_session1_results.zip`). Cell 2 unpacks it (or copies the files if Kaggle shows them unpacked) and stops with an error if `cifar10_resnet18_patch_dose-005_seed42.pt` or its metrics JSON is missing. The pilot is one of the 12 CIFAR models: `train` skips it because its metrics JSON exists, and it is scored in this session.
5. The parameters cell is pre-filled from the notes: `CIFAR_DOSES = [-0.05, -0.03, -0.02, -0.01]`, `SD1_DOSES = [-0.5, -0.2, 0, 0.1, 0.25, 0.4]`, `SEEDS = [42, 1, 2]` (seed 7 is refused). Don't change them.
6. **Settings → Accelerator → GPU T4 x2**, **Settings → Internet → On**, then run the first three code cells once to check: the Parameters cell, cell 1 (Copy) and cell 2 (Install). The Copy cell prints "CIFAR pilot found: it is reused, not retrained"; the Install cell prints a GPU name, "Pet and CIFAR-10 data found" and "ImageNet ResNet-18 weights ready".
7. **Save Version → Save & Run All (Commit) → Save**. It trains 11 CIFAR-10 ResNet-18 models and 18 SD1 models, and scores all 30 (metrics, oracle maps, controls, behaviour labels).
8. When it finishes, download `a2_session2_results.zip` from **Output**, unzip it into the project folder and re-run the analyses:

   ```powershell
   Expand-Archive $HOME\Downloads\a2_session2_results.zip -DestinationPath . -Force
   .\.venv\Scripts\python.exe -W ignore dose_response.py behaviour
   .\.venv\Scripts\python.exe -W ignore dose_response.py analyze
   .\.venv\Scripts\python.exe -W ignore dose_response.py analyze --model-set original
   .\.venv\Scripts\python.exe -W ignore dose_response.py analyze --model-set pre_a2
   ```

**Session 2 time estimate** (measured in session 1, `docs/kaggle_runs/a2_session1_pilot_report.json` and the 2 h 05 min session):

- CIFAR-10 ResNet-18: 11 models to train × 14.7 min = 162 min; scoring 12 models × about 1 min (0.6 min evaluation, audit-2 §4.1, plus controls and behaviour) = 12 min.
- SD1: 18 × 1.4 min training = 25 min. Scoring was not timed separately. Session 1 spent 125 − 97 = 28 min outside training (setup plus scoring the 3 Pet −0.15 models), so scoring a `pet_resnet18` model takes at most 9.3 min. The pretrained network is cheaper per image, so 3–9.3 min per SD1 model = 54–167 min.
- Setup about 5 min.
- **Total ≈ 4.3–6.2 h**, under the 11-hour margin of Kaggle's 12-hour limit, so no split is needed.

**SD2 small CNN on the laptop (CPU), you run it in your own terminal.** The same 4 final doses (A2-C1) and seeds 42 and 1, 8 models:

```powershell
.\.venv\Scripts\python.exe -W ignore dose_response.py train --bases cifar10 --shortcuts patch --doses -0.05 -0.03 -0.02 -0.01 --seeds 42 1
.\.venv\Scripts\python.exe -W ignore dose_response.py evaluate --bases cifar10 --shortcuts patch --doses -0.05 -0.03 -0.02 -0.01 --seeds 42 1
.\.venv\Scripts\python.exe -W ignore dose_response.py placebo --bases cifar10 --shortcuts patch --doses -0.05 -0.03 -0.02 -0.01 --seeds 42 1
.\.venv\Scripts\python.exe -W ignore dose_response.py behaviour --selection-only --bases cifar10 --shortcuts patch --doses -0.05 -0.03 -0.02 -0.01 --seeds 42 1
```

Training takes about 8 × 22.5 min ≈ 3 h on the CPU (mean `training_minutes` of the `cifar10` models in `results\dose_response\models.csv`). Then re-run the three `analyze` commands above.

## A2 time estimates

Session 1 measured values (`docs/kaggle_runs/a2_session1_pilot_report.json`, Pet −0.15 `training_minutes`):

| Model | Training per model | Scoring per model |
|---|---|---|
| Pet ResNet-18 from scratch (`pet_resnet18`, 30 epochs) | 26.6 min | at most 9.3 min (session total minus training) |
| CIFAR-10 ResNet-18 (`cifar10_resnet18`, 15 epochs) | 14.7 min | about 0.6 min evaluation (audit-2 §4.1 estimate) |
| Pet ImageNet-pretrained ResNet-18 (`pet_resnet18_pretrained`, 15 epochs) | 1.4 min | not measured; 3–9.3 min assumed |

- **Session 1** took 2 h 05 min (estimate was 2.0–2.3 h).
- **Session 2** ≈ 4.3–6.2 h (above). Both stay under Kaggle's 12-hour limit per run.

## If something goes wrong

- **"Project files not found"**: the dataset isn't attached. Use **+ Add Input** again.
- **"No GPU"**: set the accelerator, then **Run → Restart & clear cell outputs**.
- **pip install fails**: Internet is off, or the phone isn't verified.
- **The run hits the 12-hour limit**: the zip isn't created. Tell me, and I'll split the run into two notebooks.

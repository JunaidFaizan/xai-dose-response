# Running the ResNet-18 experiment on Kaggle (free GPU)

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

## If something goes wrong

- **"Project files not found"**: the dataset isn't attached. Use **+ Add Input** again.
- **"No GPU"**: set the accelerator, then **Run → Restart & clear cell outputs**.
- **pip install fails**: Internet is off, or the phone isn't verified.
- **The run hits the 12-hour limit**: the zip isn't created. Tell me, and I'll split the run into two notebooks.

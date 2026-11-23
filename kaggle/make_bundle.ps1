# Packs only the files a Kaggle run needs into a zip:
#   pwsh -File kaggle\make_bundle.ps1                 -> kaggle\xai_dose_bundle.zip
#       the code, plus the extracted CIFAR-10 data (the official CIFAR-10 mirror
#       currently has an expired security certificate, so downloading it there can fail)
#   pwsh -File kaggle\make_bundle.ps1 -Dataset pet    -> kaggle\xai_pet_bundle.zip
#       the code, plus the preprocessed Oxford-IIIT Pet files data\pet128\*.npz
#       (create them first with: python prepare_pet.py)
#   pwsh -File kaggle\make_bundle.ps1 -Dataset a2     -> kaggle\xai_a2_bundle.zip
#       amendment A2 (both A2 notebooks): the code, data\pet128\*.npz and the
#       extracted CIFAR-10 data
#   pwsh -File kaggle\make_bundle.ps1 -Dataset a3     -> kaggle\xai_a3_bundle.zip
#       amendment A3 (sessions 3, 4, 4a, 4b): the code, data\pet128\*.npz, the
#       extracted CIFAR-10 data and the two ROAD timing checkpoints
#       (pet_resnet18_patch_dose-050_seed42, pet_resnet18_pretrained_patch_dose+000_seed42)
#   pwsh -File kaggle\make_bundle.ps1 -Dataset a3models -> kaggle\xai_a3models_bundle.zip
#       amendment A3 (sessions 4, 4b): only the .pt and _metrics.json files of the
#       69 registered cifar10_resnet18, pet_resnet18 and pet_resnet18_pretrained
#       patch models (no seed-7 pilots), for exact ROAD on the GPU
# -DataDir and -CheckpointDir read the data and checkpoints from another folder
# (defaults: data\ and checkpoint\dose\ of this project). Run from anywhere.
param([ValidateSet('cifar', 'pet', 'a2', 'a3', 'a3models')][string]$Dataset = 'cifar',
      [string]$DataDir = '',
      [string]$CheckpointDir = '')

$root = Split-Path $PSScriptRoot -Parent
if (-not $DataDir) { $DataDir = "$root\data" }
if (-not $CheckpointDir) { $CheckpointDir = "$root\checkpoint\dose" }
$petFiles = "$DataDir\pet128\trainval.npz", "$DataDir\pet128\test.npz"
$timingModels = 'pet_resnet18_patch_dose-050_seed42', 'pet_resnet18_pretrained_patch_dose+000_seed42'
$withPet = $Dataset -in 'pet', 'a2', 'a3'
$withCode = $Dataset -ne 'a3models'
if ($Dataset -eq 'cifar') {
    $stage = Join-Path $env:TEMP "xai_dose_bundle"
    $zip = Join-Path $PSScriptRoot 'xai_dose_bundle.zip'
} else {
    $stage = Join-Path $env:TEMP "xai_${Dataset}_bundle"
    $zip = Join-Path $PSScriptRoot "xai_${Dataset}_bundle.zip"
}
if ($withPet) {
    foreach ($file in $petFiles) {
        if (-not (Test-Path $file)) { throw "$file not found. Run first: python prepare_pet.py" }
    }
}

Remove-Item $stage -Recurse -Force -ErrorAction SilentlyContinue
if ($withCode) {
    New-Item -ItemType Directory -Path "$stage\xai", "$stage\data" | Out-Null
    Copy-Item "$root\dose_response.py", "$root\train_models.py", "$root\utils.py" $stage
    Copy-Item "$root\xai\*.py" "$stage\xai"
}
if ($Dataset -eq 'a3') {
    # the committed watermark masks (xai/data.py draws the same masks; the tests compare them)
    Copy-Item "$root\xai\watermark_masks.npy" "$stage\xai"
}
if ($withPet) {
    New-Item -ItemType Directory -Path "$stage\data\pet128" | Out-Null
    Copy-Item $petFiles "$stage\data\pet128"
}
if ($Dataset -in 'cifar', 'a2', 'a3') {
    Copy-Item "$DataDir\cifar-10-batches-py" "$stage\data" -Recurse
}
if ($Dataset -eq 'a3') {
    New-Item -ItemType Directory -Path "$stage\checkpoint\dose" | Out-Null
    foreach ($name in $timingModels) {
        foreach ($suffix in '.pt', '_metrics.json') {
            $file = "$CheckpointDir\$name$suffix"
            if (-not (Test-Path -LiteralPath $file)) { throw "$file not found." }
            Copy-Item -LiteralPath $file "$stage\checkpoint\dose"
        }
    }
}
if ($Dataset -eq 'a3models') {
    New-Item -ItemType Directory -Path "$stage\checkpoint\dose" | Out-Null
    $models = Get-ChildItem -LiteralPath $CheckpointDir -Filter '*.pt' |
        Where-Object { $_.Name -match '^(cifar10_resnet18|pet_resnet18|pet_resnet18_pretrained)_patch_' -and
                       $_.Name -notmatch '_seed7\.pt$' }
    if ($models.Count -ne 69) { throw "expected 69 ROAD GPU checkpoints in $CheckpointDir, found $($models.Count)" }
    foreach ($model in $models) {
        $json = Join-Path $CheckpointDir ($model.BaseName + '_metrics.json')
        if (-not (Test-Path -LiteralPath $json)) { throw "$json not found." }
        Copy-Item -LiteralPath $model.FullName, $json "$stage\checkpoint\dose"
    }
}

Remove-Item $zip -ErrorAction SilentlyContinue
Compress-Archive -Path "$stage\*" -DestinationPath $zip
Remove-Item $stage -Recurse -Force
Write-Host ("Created {0} ({1:N0} MB)" -f $zip, ((Get-Item $zip).Length / 1MB)) -ForegroundColor Green

# Packs only the files a Kaggle run needs into a zip:
#   pwsh -File kaggle\make_bundle.ps1                 -> kaggle\xai_dose_bundle.zip
#       the code, plus the extracted CIFAR-10 data (the official CIFAR-10 mirror
#       currently has an expired security certificate, so downloading it there can fail)
#   pwsh -File kaggle\make_bundle.ps1 -Dataset pet    -> kaggle\xai_pet_bundle.zip
#       the code, plus the preprocessed Oxford-IIIT Pet files data\pet128\*.npz
#       (create them first with: python prepare_pet.py)
# Run from anywhere.
param([ValidateSet('cifar', 'pet')][string]$Dataset = 'cifar')

$root = Split-Path $PSScriptRoot -Parent
if ($Dataset -eq 'pet') {
    $stage = Join-Path $env:TEMP "xai_pet_bundle"
    $zip = Join-Path $PSScriptRoot 'xai_pet_bundle.zip'
    $petFiles = "$root\data\pet128\trainval.npz", "$root\data\pet128\test.npz"
    foreach ($file in $petFiles) {
        if (-not (Test-Path $file)) { throw "$file not found. Run first: python prepare_pet.py" }
    }
} else {
    $stage = Join-Path $env:TEMP "xai_dose_bundle"
    $zip = Join-Path $PSScriptRoot 'xai_dose_bundle.zip'
}

Remove-Item $stage -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path "$stage\xai", "$stage\data" | Out-Null
Copy-Item "$root\dose_response.py", "$root\train_models.py", "$root\utils.py" $stage
Copy-Item "$root\xai\*.py" "$stage\xai"
if ($Dataset -eq 'pet') {
    New-Item -ItemType Directory -Path "$stage\data\pet128" | Out-Null
    Copy-Item $petFiles "$stage\data\pet128"
} else {
    Copy-Item "$root\data\cifar-10-batches-py" "$stage\data" -Recurse
}

Remove-Item $zip -ErrorAction SilentlyContinue
Compress-Archive -Path "$stage\*" -DestinationPath $zip
Remove-Item $stage -Recurse -Force
Write-Host ("Created {0} ({1:N0} MB)" -f $zip, ((Get-Item $zip).Length / 1MB)) -ForegroundColor Green

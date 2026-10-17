# CIFAR-10 dose-response run. Shows progress in this terminal and also saves it
# to results\cifar_log.txt. Safe to re-run: finished models are skipped.
Set-Location (Join-Path $PSScriptRoot '..')
$py = '.\.venv\Scripts\python.exe'
$sel = @('--bases', 'cifar10', '--shortcuts', 'patch', '--doses', '-0.8', '-0.6', '-0.4', '-0.25', '-0.1', '0', '--seeds', '42', '1')
$log = 'results\cifar_log.txt'

foreach ($step in 'train', 'evaluate') {
    Write-Host "`n=== $step ===" -ForegroundColor Cyan
    & $py -u -W ignore dose_response.py $step @sel 2>&1 | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) { Write-Host "Step '$step' failed. Fix the error above, then run this script again." -ForegroundColor Red; exit 1 }
}
Write-Host "`n=== analyze ===" -ForegroundColor Cyan
& $py -u -W ignore dose_response.py analyze 2>&1 | Tee-Object -FilePath $log -Append
'done' | Set-Content results\cifar_done.txt
Write-Host "`nAll done. Results: results\dose_response\summary.md" -ForegroundColor Green

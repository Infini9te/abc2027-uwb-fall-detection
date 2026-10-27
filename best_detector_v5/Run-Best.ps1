param(
    [string]$Data = "C:\Users\shubh\Downloads\abc2027_uwb_train_v1 (1)\abc2027_uwb_train_v1",
    [ValidateSet("val","test")][string]$Split = "val",
    [string]$Python = "C:\Users\shubh\Documents\Codex\2026-10-22\how\work\venv\Scripts\python.exe"
)
$ErrorActionPreference = "Stop"
if (-not (Test-Path -LiteralPath $Python)) { $Python = "python" }
& $Python (Join-Path $PSScriptRoot "predict.py") --data $Data --split $Split
if ($LASTEXITCODE -ne 0) { throw "Prediction failed. Send the complete error output." }

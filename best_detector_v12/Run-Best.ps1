param(
 [string]$Data = (Split-Path $PSScriptRoot -Parent),
 [ValidateSet('val','test')][string]$Split = 'val',
 [string]$EvaluatorPackage = '',
 [string]$Output = (Join-Path $PSScriptRoot 'results')
)
$ErrorActionPreference = 'Stop'
$Python = 'C:\Users\shubh\Documents\Codex\2026-10-22\how\work\venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) { $Python = 'python' }
$RunArgs = @((Join-Path $PSScriptRoot 'predict.py'), '--data', $Data, '--split', $Split, '--out', $Output)
if ($EvaluatorPackage) { $RunArgs += @('--evaluator-package', $EvaluatorPackage) }
& $Python @RunArgs
if ($LASTEXITCODE -ne 0) { throw 'V12 inference or validation failed.' }

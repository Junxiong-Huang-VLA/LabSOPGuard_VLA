param(
    [string]$VenvDir = ".venv-e2e",
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPath = Join-Path $repoRoot $VenvDir
$requirementsPath = Join-Path $repoRoot "requirements.local-e2e.txt"

if (-not (Test-Path $requirementsPath)) {
    throw "requirements.local-e2e.txt not found: $requirementsPath"
}

if (-not (Test-Path $venvPath)) {
    & $PythonExe -m venv $venvPath
}

$venvPython = Join-Path $venvPath "Scripts\\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "venv python not found: $venvPython"
}

& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r $requirementsPath

Write-Host ""
Write-Host "Local E2E environment is ready."
Write-Host "Use it with:"
Write-Host "  $venvPython backend/main.py"

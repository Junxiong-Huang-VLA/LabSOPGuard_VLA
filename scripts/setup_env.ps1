param(
    [string]$ProjectName = "LabSOPGuard",
    [string]$PythonVersion = "3.10"
)

$ErrorActionPreference = "Stop"
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    conda run -n base python 00_setup_environment.py --project-name $ProjectName --python-version $PythonVersion
    if ($LASTEXITCODE -ne 0) { throw "Environment setup failed." }
    Write-Host "[DONE] Environment ready: $ProjectName"
}
finally {
    Pop-Location
}

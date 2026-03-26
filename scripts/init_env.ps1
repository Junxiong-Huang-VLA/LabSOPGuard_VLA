param(
    [string]$ProjectName = "LabSOPGuard",
    [string]$PythonVersion = "3.10"
)

$ErrorActionPreference = "Stop"
& (Join-Path $PSScriptRoot "setup_env.ps1") -ProjectName $ProjectName -PythonVersion $PythonVersion

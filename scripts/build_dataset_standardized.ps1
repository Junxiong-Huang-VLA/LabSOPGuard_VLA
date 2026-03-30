param(
    [string]$PythonExe = "C:\Users\Win10\miniconda3\envs\LabSOPGuard\python.exe",
    [string]$SrcRoot = "data\interim\frames",
    [string]$ClassYaml = "configs\data\class_schema.yaml",
    [string]$OutRoot = "data\dataset",
    [int]$Seed = 42,
    [double]$TrainRatio = 0.8,
    [double]$ValRatio = 0.0,
    [double]$TestRatio = 0.2,
    [switch]$CopyImages,
    [switch]$Clean,
    [switch]$DropInvalidLabel
)

$ErrorActionPreference = "Stop"

$argsList = @(
    ".\scripts\build_dataset_standardized.py",
    "--src-root", $SrcRoot,
    "--class-yaml", $ClassYaml,
    "--out-root", $OutRoot,
    "--seed", "$Seed",
    "--train-ratio", "$TrainRatio",
    "--val-ratio", "$ValRatio",
    "--test-ratio", "$TestRatio"
)

if ($CopyImages) { $argsList += "--copy-images" }
if ($Clean) { $argsList += "--clean" }
if ($DropInvalidLabel) { $argsList += "--drop-invalid-label" }

Write-Host "[INFO] Building standardized dataset..."
& $PythonExe @argsList

Write-Host "[DONE] Dataset root: $OutRoot"
Write-Host "[DONE] Report: $OutRoot\build_report.json"
Write-Host "[DONE] Mapping: $OutRoot\source_to_dataset_mapping.csv"

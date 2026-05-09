param(
    [string]$WorkspaceRoot = "D:\LabEmbodiedVLA",
    [string]$ProjectRoot = "D:\LabEmbodiedVLA\LabSOPGuard"
)

$ErrorActionPreference = "Stop"

$memorySource = Join-Path $WorkspaceRoot "memory"
$memoryTarget = Join-Path $ProjectRoot "memory"
$ultralyticsSource = Join-Path $WorkspaceRoot ".ultralytics"
$ultralyticsTarget = Join-Path $ProjectRoot ".ultralytics"
$runtimeTarget = Join-Path $ProjectRoot ".runtime"

New-Item -ItemType Directory -Force -Path $memoryTarget, $ultralyticsTarget, $runtimeTarget | Out-Null

if (Test-Path $memorySource) {
    Copy-Item -LiteralPath (Join-Path $memorySource "*") -Destination $memoryTarget -Recurse -Force
}

if (Test-Path $ultralyticsSource) {
    Copy-Item -LiteralPath (Join-Path $ultralyticsSource "*") -Destination $ultralyticsTarget -Recurse -Force
}

Write-Host "Workspace runtime folders consolidated under $ProjectRoot"
Write-Host "Legacy folders were copied, not deleted. Remove them manually only after verifying no external tools reference them."

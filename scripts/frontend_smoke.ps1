param(
  [string]$SessionDir = "",
  [string]$ExperimentId = "",
  [string]$FrontendUrl = "http://127.0.0.1:5173",
  [string]$Output = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot

function Resolve-KeyActionSession {
  param([string]$InputSessionDir)
  if ($InputSessionDir) {
    return (Resolve-Path $InputSessionDir).Path
  }
  $experimentsRoot = Join-Path $repoRoot "LabSOPGuard\outputs\experiments"
  $candidates = Get-ChildItem -Path $experimentsRoot -Directory -ErrorAction SilentlyContinue |
    ForEach-Object {
      $candidate = Join-Path $_.FullName "key_action_index"
      if (Test-Path $candidate) { Get-Item $candidate }
    } |
    Sort-Object LastWriteTime -Descending
  if (-not $candidates -or $candidates.Count -eq 0) {
    throw "No key_action_index session found under $experimentsRoot. Pass -SessionDir or -ExperimentId."
  }
  return $candidates[0].FullName
}

$resolvedSession = Resolve-KeyActionSession $SessionDir
if (-not $ExperimentId) {
  $ExperimentId = Split-Path (Split-Path $resolvedSession -Parent) -Leaf
}
if (-not $Output) {
  $Output = Join-Path $resolvedSession "reports\frontend_smoke_report.json"
}

$frontendApp = Join-Path $repoRoot "LabSOPGuard\frontend-app"
$scriptPath = Join-Path $PSScriptRoot "frontend_smoke_check.js"

$env:FRONTEND_URL = $FrontendUrl.TrimEnd("/")
$env:EXPERIMENT_ID = $ExperimentId
$env:FRONTEND_SMOKE_OUTPUT = $Output
$env:FRONTEND_SMOKE_SCREENSHOTS = Join-Path $resolvedSession "reports\frontend_smoke_screenshots"

Push-Location $frontendApp
try {
  node $scriptPath
  if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
  }
} finally {
  Pop-Location
}

Write-Host "Frontend smoke report written to $Output"

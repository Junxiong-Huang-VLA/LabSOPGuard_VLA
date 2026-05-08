param(
  [string]$SessionDir = "",
  [string[]]$Query = @("balance weighing"),
  [switch]$FailOnWarning
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$srcPath = Join-Path $repoRoot "src"

if ($env:PYTHONPATH) {
  $env:PYTHONPATH = "$srcPath;$env:PYTHONPATH"
} else {
  $env:PYTHONPATH = $srcPath
}

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
    throw "No key_action_index session found under $experimentsRoot. Pass -SessionDir explicitly."
  }
  return $candidates[0].FullName
}

$resolvedSession = Resolve-KeyActionSession $SessionDir
$reportDir = Join-Path $resolvedSession "reports"
New-Item -ItemType Directory -Force -Path $reportDir | Out-Null

$argsList = @(
  "-m", "key_action_indexer.cli",
  "health",
  "--session-dir", $resolvedSession,
  "--output-json", (Join-Path $reportDir "run_health_report.json"),
  "--output-md", (Join-Path $reportDir "run_health_report.md"),
  "--fail-on", ($(if ($FailOnWarning) { "warning" } else { "error" }))
)

foreach ($item in $Query) {
  if ($item) {
    $argsList += @("--query", $item)
  }
}

python @argsList
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

Write-Host "Health reports written to $reportDir"

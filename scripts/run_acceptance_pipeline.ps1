param(
  [Parameter(Mandatory = $true)]
  [string]$SessionDir,

  [string]$DecisionsFile = "",

  [switch]$ApplyDecisions,
  [switch]$Strict,
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$srcPath = Join-Path $repoRoot "src"

if ($env:PYTHONPATH) {
  $env:PYTHONPATH = "$srcPath;$env:PYTHONPATH"
} else {
  $env:PYTHONPATH = $srcPath
}

$argsList = @(
  "-m", "key_action_indexer.cli",
  "acceptance-pipeline",
  "--session-dir", $SessionDir
)

if ($DecisionsFile) {
  $argsList += @("--decisions-file", $DecisionsFile)
}
if ($ApplyDecisions) {
  $argsList += "--apply-decisions"
}
if ($Strict) {
  $argsList += "--strict"
}
if ($DryRun) {
  $argsList += "--dry-run"
}

python @argsList

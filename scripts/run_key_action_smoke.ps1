param(
  [string]$SessionDir = "",
  [switch]$SkipPytest,
  [switch]$SkipFrontend,
  [switch]$RunBrowserSmoke,
  [string]$FrontendUrl = "http://127.0.0.1:5173",
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

function Invoke-Step {
  param(
    [string]$Name,
    [scriptblock]$Body
  )
  Write-Host ""
  Write-Host "== $Name =="
  & $Body
  if ($LASTEXITCODE -ne 0) {
    throw "$Name failed with exit code $LASTEXITCODE"
  }
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
      $required = @(
        (Join-Path $candidate "metadata\key_action_segments.jsonl"),
        (Join-Path $candidate "metadata\micro_segments.jsonl"),
        (Join-Path $candidate "metadata\vector_metadata.jsonl"),
        (Join-Path $candidate "index\fallback_index.pkl")
      )
      if ((Test-Path $candidate) -and (($required | Where-Object { -not (Test-Path $_) }).Count -eq 0)) {
        Get-Item $candidate
      }
    } |
    Sort-Object LastWriteTime -Descending
  if (-not $candidates -or $candidates.Count -eq 0) {
    throw "No complete key_action_index session found under $experimentsRoot. Pass -SessionDir explicitly."
  }
  return $candidates[0].FullName
}

$resolvedSession = Resolve-KeyActionSession $SessionDir
Write-Host "Using session: $resolvedSession"

Push-Location $repoRoot
try {
  Invoke-Step "python compile" {
    python -m compileall -q src LabSOPGuard\backend tests
  }

  if (-not $SkipPytest) {
    Invoke-Step "pytest" {
      python -m pytest -q
    }
  }

  Invoke-Step "key-action health" {
    if ($FailOnWarning) {
      & (Join-Path $PSScriptRoot "check_key_action_outputs.ps1") -SessionDir $resolvedSession -FailOnWarning
    } else {
      & (Join-Path $PSScriptRoot "check_key_action_outputs.ps1") -SessionDir $resolvedSession
    }
  }

  Invoke-Step "project scope guard" {
    & (Join-Path $PSScriptRoot "check_project_scope.ps1")
  }

  Invoke-Step "cli query smoke" {
    python -m key_action_indexer.cli query --session-dir $resolvedSession --query "balance weighing" --top-k 2
  }

  if (-not $SkipFrontend) {
    Push-Location (Join-Path $repoRoot "LabSOPGuard\frontend-app")
    try {
      Invoke-Step "frontend build" {
        npm run build
      }
      Invoke-Step "frontend tests" {
        npm run test
      }
    } finally {
      Pop-Location
    }
  }

  if ($RunBrowserSmoke) {
    Invoke-Step "browser smoke" {
      & (Join-Path $PSScriptRoot "frontend_smoke.ps1") -SessionDir $resolvedSession -FrontendUrl $FrontendUrl
    }
  }
} finally {
  Pop-Location
}

Write-Host ""
Write-Host "Key-action smoke completed."

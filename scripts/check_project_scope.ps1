param(
  [string]$RepoRoot = "",
  [switch]$Json
)

$ErrorActionPreference = "Stop"
if (-not $RepoRoot) {
  $RepoRoot = Split-Path -Parent $PSScriptRoot
}
$RepoRoot = (Resolve-Path $RepoRoot).Path

$scanRoots = @(
  "src/key_action_indexer",
  "LabSOPGuard/backend",
  "LabSOPGuard/frontend-app/src",
  "scripts",
  "tests"
)

$forbidden = @(
  @{ Pattern = "camera_api_stub"; Reason = "PTZ/camera stubs belong in D:\PtzTracker or D:\MultiCameraMonitor" },
  @{ Pattern = "ptz-tracker"; Reason = "PTZ tracker API belongs in D:\PtzTracker" },
  @{ Pattern = "/api/v1/cameras"; Reason = "camera control APIs are out of LabCapability scope" },
  @{ Pattern = "wireless_video"; Reason = "wireless video transport belongs in D:\MultiCameraMonitor" },
  @{ Pattern = "wvd_sdk"; Reason = "wireless video SDK belongs in D:\MultiCameraMonitor" },
  @{ Pattern = "camera_proxy"; Reason = "camera proxy belongs in D:\MultiCameraMonitor" },
  @{ Pattern = "camera_streaming"; Reason = "camera streaming belongs in D:\MultiCameraMonitor" },
  @{ Pattern = "usb_camera_worker"; Reason = "USB camera worker belongs in D:\MultiCameraMonitor" },
  @{ Pattern = "multi-monitor/recording"; Reason = "multi-monitor recording belongs in D:\MultiCameraMonitor" },
  @{ Pattern = "multi_monitor"; Reason = "five-camera/multi-monitor orchestration is out of LabCapability scope" },
  @{ Pattern = "wireless_1"; Reason = "fixed wireless camera topology is out of LabCapability scope" },
  @{ Pattern = "wireless_2"; Reason = "fixed wireless camera topology is out of LabCapability scope" },
  @{ Pattern = "wireless_3"; Reason = "fixed wireless camera topology is out of LabCapability scope" }
)

$matches = @()
foreach ($root in $scanRoots) {
  $path = Join-Path $RepoRoot $root
  if (-not (Test-Path $path)) {
    continue
  }
  foreach ($rule in $forbidden) {
    $rgOutput = & rg -n --fixed-strings --glob "!**/__pycache__/**" --glob "!**/.pytest_cache/**" --glob "!**/node_modules/**" --glob "!scripts/check_project_scope.ps1" $rule.Pattern $path 2>$null
    if ($LASTEXITCODE -gt 1) {
      throw "rg failed while scanning $root for $($rule.Pattern)"
    }
    foreach ($line in $rgOutput) {
      if (-not $line) { continue }
      $matches += [pscustomobject]@{
        pattern = $rule.Pattern
        reason = $rule.Reason
        match = $line
      }
    }
  }
}

$payload = [pscustomobject]@{
  schema_version = "labcapability_scope_guard.v1"
  repo_root = $RepoRoot
  scanned_roots = $scanRoots
  forbidden_count = $matches.Count
  matches = $matches
}

if ($Json) {
  $payload | ConvertTo-Json -Depth 6
} else {
  if ($matches.Count -eq 0) {
    Write-Host "Scope guard passed: no PTZ/camera/wireless/multi-monitor core code found."
  } else {
    Write-Host "Scope guard failed: PTZ/camera/wireless/multi-monitor code leaked into LabCapability core." -ForegroundColor Red
    $matches | Format-Table -AutoSize
  }
}

if ($matches.Count -gt 0) {
  exit 1
}

exit 0

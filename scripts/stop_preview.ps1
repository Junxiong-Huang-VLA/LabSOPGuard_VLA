$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$pidFile = Join-Path $projectRoot "outputs\logs\preview_server.pid"

if (-not (Test-Path $pidFile)) {
    Write-Host "[INFO] No preview PID file found."
    exit 0
}

$pidText = (Get-Content $pidFile -Raw).Trim()
if (-not $pidText) {
    Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
    Write-Host "[INFO] Empty PID file removed."
    exit 0
}

$proc = Get-Process -Id ([int]$pidText) -ErrorAction SilentlyContinue
if ($proc) {
    Stop-Process -Id $proc.Id -Force
    Write-Host "[DONE] Preview server stopped. PID=$($proc.Id)"
} else {
    Write-Host "[INFO] Process not running. PID=$pidText"
}

Remove-Item $pidFile -Force -ErrorAction SilentlyContinue

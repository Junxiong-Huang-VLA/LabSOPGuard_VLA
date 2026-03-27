param(
    [string]$EnvName = "LabSOPGuard",
    [string]$VersionSpec = "mediapipe>=0.10.14,<0.10.20"
)

$ErrorActionPreference = "Stop"

function Write-Info([string]$msg) {
    Write-Host "[INFO] $msg" -ForegroundColor Cyan
}

function Write-Ok([string]$msg) {
    Write-Host "[OK] $msg" -ForegroundColor Green
}

function Write-WarnLine([string]$msg) {
    Write-Host "[WARN] $msg" -ForegroundColor Yellow
}

function Run-CondaPython([string]$code) {
    $args = @("run", "-n", $EnvName, "python", "-c", $code)
    Write-Info ("conda " + ($args -join " "))
    & conda @args
    if ($LASTEXITCODE -ne 0) {
        throw "failed: conda run -n $EnvName python -c <code>"
    }
}

function Run-CondaPip([string[]]$pipArgs) {
    $args = @("run", "-n", $EnvName, "python", "-m", "pip") + $pipArgs
    Write-Info ("conda " + ($args -join " "))
    & conda @args
    if ($LASTEXITCODE -ne 0) {
        throw "failed: conda run -n $EnvName python -m pip $($pipArgs -join ' ')"
    }
}

try {
    Write-Info "Checking current mediapipe API availability..."
    Run-CondaPython "import mediapipe as mp;print('version=',getattr(mp,'__version__','unknown'));print('has_solutions=',hasattr(mp,'solutions'));print('has_tasks=',hasattr(mp,'tasks'))"

    $tmp = Join-Path (Resolve-Path ".").Path "outputs\tmp\pip_fix"
    New-Item -ItemType Directory -Force -Path $tmp | Out-Null
    $env:TEMP = $tmp
    $env:TMP = $tmp
    Write-Info "TEMP/TMP set to: $tmp"

    Write-Info "Uninstalling current mediapipe (if exists)..."
    try {
        Run-CondaPip @("uninstall", "-y", "mediapipe")
    } catch {
        Write-WarnLine "mediapipe uninstall skipped (not installed or already removed)."
    }

    Write-Info "Installing compatible mediapipe range: $VersionSpec"
    Run-CondaPip @("install", "--no-cache-dir", $VersionSpec)

    Write-Info "Validating mediapipe API..."
    Run-CondaPython "import mediapipe as mp;print('version=',getattr(mp,'__version__','unknown'));print('has_solutions=',hasattr(mp,'solutions'));print('has_tasks=',hasattr(mp,'tasks'));import sys;sys.exit(0 if hasattr(mp,'solutions') else 2)"

    Write-Ok "mediapipe compatibility fix completed."
    Write-Host "[NEXT] Restart preview server:"
    Write-Host "powershell -ExecutionPolicy Bypass -File .\scripts\stop_preview.ps1"
    Write-Host "powershell -ExecutionPolicy Bypass -File .\scripts\start_preview.ps1"
}
catch {
    Write-Host "[ERROR] $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "[SUGGEST] If network is unstable, retry with mobile hotspot and rerun this script."
    exit 1
}

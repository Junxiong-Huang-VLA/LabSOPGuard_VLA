param(
    [string]$PreviewHost = "127.0.0.1",
    [int]$Port = 5001,
    [string]$PythonExe = "C:\Users\Win10\miniconda3\envs\LabSOPGuard\python.exe",
    [switch]$NoOpen
)

$ErrorActionPreference = "Stop"

function Write-Info([string]$msg) {
    Write-Host "[INFO] $msg" -ForegroundColor Cyan
}

$projectRoot = Split-Path -Parent $PSScriptRoot
$pidFile = Join-Path $projectRoot "outputs\logs\preview_server.pid"
$outLog = Join-Path $projectRoot "outputs\logs\preview_stdout.log"
$errLog = Join-Path $projectRoot "outputs\logs\preview_stderr.log"
$mplDir = Join-Path $projectRoot ".matplotlib"

New-Item -ItemType Directory -Force -Path (Join-Path $projectRoot "outputs\logs") | Out-Null
New-Item -ItemType Directory -Force -Path $mplDir | Out-Null

if (Test-Path $pidFile) {
    $oldPid = (Get-Content $pidFile -Raw).Trim()
    if ($oldPid) {
        $oldProc = Get-Process -Id ([int]$oldPid) -ErrorAction SilentlyContinue
        if ($oldProc) {
            Write-Host "[WARN] Preview server already running. PID=$oldPid" -ForegroundColor Yellow
            Write-Host "[INFO] URL: http://$PreviewHost`:$Port"
            exit 0
        }
    }
}

Write-Info "Starting preview server on http://$PreviewHost`:$Port ..."
$env:MPLCONFIGDIR = $mplDir
$env:INTEGRATED_HOST = $PreviewHost
$env:INTEGRATED_PORT = "$Port"

$proc = Start-Process -FilePath $PythonExe `
    -ArgumentList @(".\integrated_system\app_integrated.py") `
    -WorkingDirectory $projectRoot `
    -RedirectStandardOutput $outLog `
    -RedirectStandardError $errLog `
    -PassThru

Start-Sleep -Seconds 2
$alive = Get-Process -Id $proc.Id -ErrorAction SilentlyContinue
if (-not $alive) {
    Write-Host "[ERROR] Preview process exited immediately." -ForegroundColor Red
    Write-Host "[HINT] Check logs:"
    Write-Host "  $outLog"
    Write-Host "  $errLog"
    exit 1
}

Set-Content -Path $pidFile -Value $proc.Id -Encoding UTF8
Write-Info "Preview PID: $($proc.Id)"
Write-Info "stdout: $outLog"
Write-Info "stderr: $errLog"

if (-not $NoOpen) {
    Start-Sleep -Seconds 2
    Start-Process "http://$PreviewHost`:$Port"
}

Write-Host "[DONE] Preview started."

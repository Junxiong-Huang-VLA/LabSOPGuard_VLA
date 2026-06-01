param(
    [switch]$SkipRedis,
    [switch]$NoBrowser,
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 5173
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
$FrontendDir = Join-Path $ProjectRoot "frontend"
$RuntimeDir = Join-Path $ProjectRoot ".runtime"
$RunLogDir = Join-Path $ProjectRoot "outputs\run_logs"

function Stop-PortOwner {
    param([int]$Port)
    $connections = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue |
        Where-Object { $_.State -eq "Listen" -and $_.OwningProcess -gt 0 }
    $processIds = $connections | Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($processId in $processIds) {
        try {
            $process = Get-Process -Id $processId -ErrorAction Stop
            Write-Host "Stopping process $($process.Id) on port $Port ($($process.ProcessName))"
            Stop-Process -Id $process.Id -Force
        } catch {
            Write-Warning "Could not stop process $processId on port ${Port}: $($_.Exception.Message)"
        }
    }
}

function Wait-HttpOk {
    param(
        [string]$Url,
        [int]$TimeoutSec = 60
    )
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    do {
        try {
            $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 3
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
                return
            }
        } catch {
            Start-Sleep -Milliseconds 700
        }
    } while ((Get-Date) -lt $deadline)
    throw "Timed out waiting for $Url"
}

Set-Location -LiteralPath $ProjectRoot
New-Item -ItemType Directory -Force -Path $RuntimeDir, $RunLogDir | Out-Null

$env:PYTHONPATH = "src"
if (-not $env:LAB_MODELS_DIR) { $env:LAB_MODELS_DIR = "D:\LabModels" }
if (-not $env:LAB_VIDEO_STORE_ROOT) { $env:LAB_VIDEO_STORE_ROOT = "D:\LabVideo\raw_uploads" }
if (-not $env:LAB_MATERIAL_LIBRARY_ROOT) { $env:LAB_MATERIAL_LIBRARY_ROOT = "D:\LabMaterialLibrary" }
if (-not $env:YOLO_CONFIG_DIR) { $env:YOLO_CONFIG_DIR = Join-Path $ProjectRoot ".ultralytics" }

$distPath = Join-Path $FrontendDir "dist"
$resolvedDistParent = Resolve-Path -LiteralPath $FrontendDir
if (Test-Path -LiteralPath $distPath) {
    $resolvedDist = Resolve-Path -LiteralPath $distPath
    if ($resolvedDist.Path.StartsWith($resolvedDistParent.Path, [StringComparison]::OrdinalIgnoreCase)) {
        Remove-Item -LiteralPath $resolvedDist.Path -Recurse -Force
    }
}

Stop-PortOwner -Port $BackendPort
Stop-PortOwner -Port $FrontendPort

if (-not $SkipRedis) {
    $redis = Get-Command redis-server -ErrorAction SilentlyContinue
    if ($redis) {
        Start-Process -FilePath $redis.Source -WorkingDirectory $ProjectRoot -WindowStyle Hidden | Out-Null
    } else {
        Write-Warning "redis-server was not found; continuing without starting Redis."
    }
}

$python = (Get-Command python -ErrorAction Stop).Source
$npm = (Get-Command npm.cmd -ErrorAction SilentlyContinue)
if (-not $npm) { $npm = Get-Command npm -ErrorAction Stop }

$backendOutLog = Join-Path $RunLogDir "backend.out.log"
$backendErrLog = Join-Path $RunLogDir "backend.err.log"
$frontendOutLog = Join-Path $RunLogDir "frontend.out.log"
$frontendErrLog = Join-Path $RunLogDir "frontend.err.log"

$backendArgs = @(
    "-m", "uvicorn", "backend.main:app",
    "--host", "127.0.0.1",
    "--port", [string]$BackendPort
)
$backend = Start-Process -FilePath $python -ArgumentList $backendArgs -WorkingDirectory $ProjectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput $backendOutLog -RedirectStandardError $backendErrLog

$frontendArgs = @("run", "dev", "--", "--host", "127.0.0.1", "--port", [string]$FrontendPort, "--strictPort")
$frontend = Start-Process -FilePath $npm.Source -ArgumentList $frontendArgs -WorkingDirectory $FrontendDir -WindowStyle Hidden -PassThru -RedirectStandardOutput $frontendOutLog -RedirectStandardError $frontendErrLog

$state = [ordered]@{
    schema_version = "labembodied.full_stack_run.v1"
    started_at = (Get-Date).ToString("o")
    project_root = [string]$ProjectRoot
    backend = @{ port = $BackendPort; pid = $backend.Id; url = "http://127.0.0.1:$BackendPort" }
    frontend = @{ port = $FrontendPort; pid = $frontend.Id; url = "http://127.0.0.1:$FrontendPort" }
    logs = @{
        backend_stdout = $backendOutLog
        backend_stderr = $backendErrLog
        frontend_stdout = $frontendOutLog
        frontend_stderr = $frontendErrLog
    }
}
$state | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $RuntimeDir "start_full_stack.json") -Encoding UTF8

Wait-HttpOk -Url "http://127.0.0.1:$BackendPort/" -TimeoutSec 90
Wait-HttpOk -Url "http://127.0.0.1:$FrontendPort/" -TimeoutSec 90

Write-Host "Backend:  http://127.0.0.1:$BackendPort"
Write-Host "Frontend: http://127.0.0.1:$FrontendPort"
Write-Host "Run state: $RuntimeDir\start_full_stack.json"

if (-not $NoBrowser) {
    Start-Process "http://127.0.0.1:$FrontendPort"
}

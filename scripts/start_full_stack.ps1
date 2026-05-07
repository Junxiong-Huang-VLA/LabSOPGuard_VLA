param(
    [string]$HostName = "127.0.0.1",
    [string]$BackendHost = "0.0.0.0",
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 5173,
    [int]$RedisPort = 6379,
    [string]$PythonExe = "E:\conda_envs\labsopguard\python.exe",
    [switch]$NoOpen,
    [switch]$SkipRedis
)

$ErrorActionPreference = "Stop"

function Write-Info([string]$Message)  { Write-Host "[INFO] $Message" -ForegroundColor Cyan }
function Write-Ok([string]$Message)    { Write-Host "[OK]   $Message" -ForegroundColor Green }
function Write-Warn([string]$Message)  { Write-Host "[WARN] $Message" -ForegroundColor Yellow }
function Write-Fail([string]$Message)  { Write-Host "[FAIL] $Message" -ForegroundColor Red }

function Get-PortOwners([int]$Port) {
    @(Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue |
        Where-Object { $_.State -eq "Listen" } |
        Select-Object -ExpandProperty OwningProcess -Unique)
}

function Stop-ProcessTree([int]$ParentId) {
    Get-CimInstance Win32_Process -Filter "ParentProcessId=$ParentId" -ErrorAction SilentlyContinue |
        ForEach-Object { Stop-ProcessTree $_.ProcessId }
    Stop-Process -Id $ParentId -Force -ErrorAction SilentlyContinue
}

function Stop-Port([int]$Port) {
    $owners = Get-PortOwners $Port
    foreach ($owner in $owners) {
        if ($owner -and $owner -ne $PID) {
            $proc = Get-Process -Id $owner -ErrorAction SilentlyContinue
            if ($proc) {
                Write-Warn "Killing PID=$owner process tree on port $Port ($($proc.ProcessName))"
                Stop-ProcessTree $owner
            }
        }
    }
    if ($owners.Count -gt 0) {
        $deadline = (Get-Date).AddSeconds(10)
        while ((Get-Date) -lt $deadline -and (Get-PortOwners $Port).Count -gt 0) {
            Start-Sleep -Milliseconds 300
        }
    }
}

function Wait-Port([int]$Port, [int]$TimeoutSec = 30) {
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        if ((Get-PortOwners $Port).Count -gt 0) { return $true }
        Start-Sleep -Milliseconds 500
    }
    return $false
}

function Test-Http([string]$Url, [int]$TimeoutSec = 20) {
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        try {
            $r = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 5
            if ($r.StatusCode -ge 200 -and $r.StatusCode -lt 500) { return $r.StatusCode }
        } catch { Start-Sleep -Milliseconds 700 }
    }
    return $null
}

# ── paths ──────────────────────────────────────────────────────────
$projectRoot   = Split-Path -Parent $PSScriptRoot
$frontendRoot  = Join-Path $projectRoot "frontend-app"
$runLogDir     = Join-Path $projectRoot "outputs\run_logs"
$redisDataDir  = Join-Path $projectRoot "outputs\redis"
$redisExe      = Join-Path $projectRoot "tools\redis\Redis-8.6.2-Windows-x64-msys2\redis-server.exe"

New-Item -ItemType Directory -Force -Path $runLogDir, $redisDataDir | Out-Null

Write-Host ""
Write-Host "========================================" -ForegroundColor White
Write-Host "  LabSOPGuard Full-Stack Launcher" -ForegroundColor White
Write-Host "========================================" -ForegroundColor White
Write-Host ""
Write-Info "Project root : $projectRoot"
Write-Info "Python       : $PythonExe"
Write-Info "Logs         : $runLogDir"
Write-Host ""

# ── validate python ────────────────────────────────────────────────
if (-not (Test-Path $PythonExe)) {
    Write-Fail "Python not found: $PythonExe"
    Write-Warn "Install the labsopguard conda env or pass -PythonExe <path>"
    exit 1
}
$pyVer = & $PythonExe --version 2>&1
Write-Ok "Python: $pyVer"

# ── step 1: kill ALL old processes ─────────────────────────────────
Write-Info "Cleaning up old processes on ports $RedisPort, $BackendPort, $FrontendPort, 3000 ..."
Stop-Port $BackendPort
Stop-Port $FrontendPort
Stop-Port 3000
if (-not $SkipRedis) { Stop-Port $RedisPort }

# ── step 2: clean stale frontend build cache ───────────────────────
$distDir = Join-Path $frontendRoot "dist"
if (Test-Path $distDir) {
    Remove-Item -Recurse -Force $distDir
    Write-Warn "Removed stale frontend-app/dist/ to prevent serving old build"
}

# ── step 3: redis ──────────────────────────────────────────────────
if (-not $SkipRedis) {
    if (-not (Test-Path $redisExe)) {
        Write-Fail "redis-server.exe not found: $redisExe"
        Write-Warn "Use -SkipRedis if Redis is managed separately."
        exit 1
    }
    Write-Info "Starting Redis on port $RedisPort"
    Start-Process -FilePath $redisExe `
        -ArgumentList @("--port", "$RedisPort", "--appendonly", "yes", "--dir", $redisDataDir) `
        -WorkingDirectory (Split-Path $redisExe) `
        -RedirectStandardOutput (Join-Path $runLogDir "redis_$RedisPort.out.log") `
        -RedirectStandardError  (Join-Path $runLogDir "redis_$RedisPort.err.log") `
        -WindowStyle Minimized | Out-Null
    if (-not (Wait-Port $RedisPort 20)) {
        Write-Fail "Redis did not start on port $RedisPort"
        exit 1
    }
    Write-Ok "Redis listening on port $RedisPort"
} else {
    Write-Info "Skipping Redis (managed separately)"
}

# ── step 4: backend ───────────────────────────────────────────────
Write-Info "Starting backend on http://${BackendHost}:${BackendPort}"
$oldPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = if ($oldPythonPath) { "src;$oldPythonPath" } else { "src" }

Start-Process -FilePath $PythonExe `
    -ArgumentList @("-m", "uvicorn", "backend.main:app", "--host", $BackendHost, "--port", "$BackendPort") `
    -WorkingDirectory $projectRoot `
    -RedirectStandardOutput (Join-Path $runLogDir "backend_$BackendPort.out.log") `
    -RedirectStandardError  (Join-Path $runLogDir "backend_$BackendPort.err.log") `
    -WindowStyle Minimized | Out-Null

$env:PYTHONPATH = $oldPythonPath

if (-not (Wait-Port $BackendPort 90)) {
    Write-Fail "Backend did not start on port $BackendPort"
    Write-Warn "Check: $(Join-Path $runLogDir "backend_$BackendPort.err.log")"
    exit 1
}
Write-Ok "Backend listening on port $BackendPort"

# ── step 5: frontend ──────────────────────────────────────────────
if (-not (Test-Path (Join-Path $frontendRoot "node_modules"))) {
    Write-Warn "node_modules not found, running npm install ..."
    Push-Location $frontendRoot
    try { npm install } finally { Pop-Location }
}

Write-Info "Starting frontend on http://${HostName}:${FrontendPort}"
Start-Process -FilePath "npm.cmd" `
    -ArgumentList @("run", "dev", "--", "--host", $HostName, "--port", "$FrontendPort") `
    -WorkingDirectory $frontendRoot `
    -RedirectStandardOutput (Join-Path $runLogDir "frontend_$FrontendPort.out.log") `
    -RedirectStandardError  (Join-Path $runLogDir "frontend_$FrontendPort.err.log") `
    -WindowStyle Minimized | Out-Null

if (-not (Wait-Port $FrontendPort 30)) {
    Write-Fail "Frontend did not start on port $FrontendPort"
    Write-Warn "Check: $(Join-Path $runLogDir "frontend_$FrontendPort.err.log")"
    exit 1
}
Write-Ok "Frontend listening on port $FrontendPort"

# ── step 6: health checks ─────────────────────────────────────────
Write-Host ""
Write-Info "Running health checks ..."

$frontendUrl = "http://${HostName}:${FrontendPort}/"
$backendHealthUrl = "http://${HostName}:${BackendPort}/api/v1/experiments?limit=1"

$feStatus = Test-Http $frontendUrl 15
if ($feStatus) { Write-Ok "Frontend  : HTTP $feStatus" } else { Write-Warn "Frontend  : no response within timeout" }

$beStatus = Test-Http $backendHealthUrl 15
if ($beStatus) { Write-Ok "Backend   : HTTP $beStatus" } else { Write-Warn "Backend   : no response within timeout" }

$proxyStatus = Test-Http "http://${HostName}:${FrontendPort}/api/v1/experiments?limit=1" 10
if ($proxyStatus) { Write-Ok "Vite proxy: HTTP $proxyStatus" } else { Write-Warn "Vite proxy: no response (check vite.config.ts proxy)" }

$camStatus = Test-Http "http://${HostName}:${BackendPort}/api/v1/cameras" 10
if ($camStatus) { Write-Ok "Cameras   : HTTP $camStatus" } else { Write-Warn "Cameras   : camera module not loaded" }

# ── step 7: find latest experiment for quick-open links ────────────
$latestExpId = $null
try {
    $resp = Invoke-WebRequest -Uri "http://${HostName}:${BackendPort}/api/v1/experiments?limit=1" -UseBasicParsing -TimeoutSec 5
    $expData = $resp.Content | ConvertFrom-Json
    if ($expData.experiments -and $expData.experiments.Count -gt 0) {
        $latestExpId = $expData.experiments[0].experiment_id
    }
} catch {}

# ── summary ────────────────────────────────────────────────────────
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  All services running" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Frontend        : $frontendUrl" -ForegroundColor White
if ($latestExpId) {
    Write-Host "  Latest workspace: http://${HostName}:${FrontendPort}/experiments/${latestExpId}/workspace" -ForegroundColor White
    Write-Host "  Latest materials: http://${HostName}:${FrontendPort}/experiments/${latestExpId}/materials" -ForegroundColor White
}
Write-Host "  Camera monitor  : http://${HostName}:${FrontendPort}/cameras" -ForegroundColor White
Write-Host ""
Write-Host "  Backend stderr  : $(Join-Path $runLogDir "backend_$BackendPort.err.log")" -ForegroundColor DarkGray
Write-Host "  Frontend stderr : $(Join-Path $runLogDir "frontend_$FrontendPort.err.log")" -ForegroundColor DarkGray
Write-Host ""

if (-not $NoOpen) {
    if ($latestExpId) {
        Start-Process "http://${HostName}:${FrontendPort}/experiments/${latestExpId}/workspace"
    } else {
        Start-Process $frontendUrl
    }
}

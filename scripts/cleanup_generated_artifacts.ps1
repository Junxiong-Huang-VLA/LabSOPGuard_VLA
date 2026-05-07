param(
    [switch]$PurgeOutputs,
    [switch]$PurgeUploads,
    [switch]$PurgeLocalEnv
)

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

function Remove-RepoPath {
    param([string]$RelativePath)
    $target = Join-Path $repoRoot $RelativePath
    if (Test-Path -LiteralPath $target) {
        Remove-Item -LiteralPath $target -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "Removed $RelativePath"
    }
}

function Clear-RepoDirectory {
    param([string]$RelativePath)
    $target = Join-Path $repoRoot $RelativePath
    if (-not (Test-Path -LiteralPath $target)) {
        return
    }

    Get-ChildItem -LiteralPath $target -Force -ErrorAction SilentlyContinue | ForEach-Object {
        if ($_.Name -eq ".gitkeep") {
            return
        }
        Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
    }
    Write-Host "Cleared $RelativePath"
}

$fixedDirectories = @(
    ".pytest_cache",
    ".ultralytics",
    "Ultralytics",
    "frontend-app\\node_modules",
    "frontend-app\\dist",
    "logs"
)

if ($PurgeLocalEnv) {
    $fixedDirectories += ".venv-e2e"
}

foreach ($path in $fixedDirectories) {
    Remove-RepoPath -RelativePath $path
}

Get-ChildItem -LiteralPath $repoRoot -Recurse -Directory -Force -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -eq "__pycache__" } |
    ForEach-Object {
        Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "Removed $($_.FullName.Substring($repoRoot.Length + 1))"
    }

$rootLogFiles = @(
    "tmp_backend.err",
    "tmp_backend.log",
    "backend_api_probe.err.log",
    "backend_api_probe.out.log",
    "backend_probe.err.log",
    "backend_probe.out.log",
    "backend_start.err.log",
    "backend_start.out.log",
    "nicegui_frontend.err.log",
    "nicegui_frontend.out.log"
)

foreach ($file in $rootLogFiles) {
    $target = Join-Path $repoRoot $file
    if (Test-Path -LiteralPath $target) {
        Remove-Item -LiteralPath $target -Force -ErrorAction SilentlyContinue
        Write-Host "Removed $file"
    }
}

if ($PurgeOutputs) {
    Clear-RepoDirectory -RelativePath "outputs"
    Clear-RepoDirectory -RelativePath "integrated_system\\outputs"
}

if ($PurgeUploads) {
    Clear-RepoDirectory -RelativePath "uploads"
}

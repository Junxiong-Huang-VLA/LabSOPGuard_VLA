param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
)

$ErrorActionPreference = "Stop"

$logDir = Join-Path $ProjectRoot "outputs\run_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $logDir "docker_install_validate_$stamp.log"
$resultPath = Join-Path $logDir "docker_install_validate_latest.json"

function Write-Log {
    param([string]$Message)
    $line = "$(Get-Date -Format o) $Message"
    Write-Host $line
    Add-Content -LiteralPath $logPath -Value $line -Encoding UTF8
}

function Write-Result {
    param(
        [string]$Status,
        [string]$Message,
        [hashtable]$Extra = @{}
    )
    $payload = @{
        status = $Status
        message = $Message
        log_path = $logPath
        written_at = (Get-Date).ToString("o")
    }
    foreach ($key in $Extra.Keys) {
        $payload[$key] = $Extra[$key]
    }
    $payload | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $resultPath -Encoding UTF8
}

function Assert-Admin {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw "This script must run in an elevated Administrator PowerShell."
    }
}

function Find-Winget {
    $cmd = Get-Command winget -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }
    $candidate = Join-Path $env:LOCALAPPDATA "Microsoft\WindowsApps\winget.exe"
    if (Test-Path -LiteralPath $candidate) {
        return $candidate
    }
    throw "winget.exe was not found."
}

function Find-Docker {
    $candidates = @(
        (Join-Path $env:USERPROFILE ".docker\bin\docker.exe"),
        (Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Links\docker.exe"),
        (Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages\Docker.DockerCLI_Microsoft.Winget.Source_8wekyb3d8bbwe\docker\docker.exe"),
        (Join-Path $env:ProgramFiles "Docker\Docker\resources\bin\docker.exe"),
        (Join-Path ${env:ProgramFiles(x86)} "Docker\Docker\resources\bin\docker.exe")
    )
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate)) {
            return $candidate
        }
    }
    $cmd = Get-Command docker -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }
    return $null
}

function Test-RestartPending {
    $keys = @(
        "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Component Based Servicing\RebootPending",
        "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\WindowsUpdate\Auto Update\RebootRequired",
        "HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager"
    )

    foreach ($key in $keys) {
        if (-not (Test-Path -LiteralPath $key)) {
            continue
        }
        if ($key -like "*Session Manager") {
            $value = Get-ItemProperty -LiteralPath $key -Name PendingFileRenameOperations -ErrorAction SilentlyContinue
            if ($value.PendingFileRenameOperations) {
                return $true
            }
            continue
        }
        return $true
    }
    return $false
}

function Ensure-DockerCli {
    $docker = Find-Docker
    if ($docker) {
        Write-Log "Docker CLI already available: $docker"
        return $docker
    }

    $winget = Find-Winget
    Write-Log "Installing Docker CLI with winget: $winget"
    & $winget install --id Docker.DockerCLI -e --accept-source-agreements --accept-package-agreements --disable-interactivity | Tee-Object -FilePath $logPath -Append
    if ($LASTEXITCODE -ne 0) {
        throw "Docker CLI installation failed. Exit code: $LASTEXITCODE"
    }

    $docker = Find-Docker
    if (-not $docker) {
        throw "docker.exe was not found after Docker CLI installation."
    }
    return $docker
}

function Ensure-ComposePlugin {
    param([string]$DockerExe)

    & $DockerExe compose version *> $null
    if ($LASTEXITCODE -eq 0) {
        Write-Log "Docker Compose plugin already available."
        return
    }

    $compose = Get-Command docker-compose -ErrorAction SilentlyContinue
    if (-not $compose) {
        $winget = Find-Winget
        Write-Log "Installing Docker Compose with winget: $winget"
        & $winget install --id Docker.DockerCompose -e --accept-source-agreements --accept-package-agreements --disable-interactivity | Tee-Object -FilePath $logPath -Append
        if ($LASTEXITCODE -ne 0) {
            throw "Docker Compose installation failed. Exit code: $LASTEXITCODE"
        }
        $compose = Get-Command docker-compose -ErrorAction SilentlyContinue
    }

    if (-not $compose) {
        throw "docker-compose.exe was not found after Docker Compose installation."
    }

    $pluginDir = Join-Path $env:USERPROFILE ".docker\cli-plugins"
    New-Item -ItemType Directory -Force -Path $pluginDir | Out-Null
    Copy-Item -LiteralPath $compose.Source -Destination (Join-Path $pluginDir "docker-compose.exe") -Force
    Write-Log "Installed Docker Compose CLI plugin from: $($compose.Source)"
}

function Enable-RequiredFeatures {
    Write-Log "Enabling WSL and VirtualMachinePlatform optional features."
    $features = @("Microsoft-Windows-Subsystem-Linux", "VirtualMachinePlatform")
    foreach ($feature in $features) {
        & dism.exe /online /enable-feature /featurename:$feature /all /norestart | Tee-Object -FilePath $logPath -Append
        $code = $LASTEXITCODE
        if ($code -ne 0 -and $code -ne 3010) {
            throw "Failed to enable feature $feature. Exit code: $code"
        }
        if ($code -eq 3010) {
            $script:restartRequired = $true
        }
    }
}

function Ensure-HypervisorLaunch {
    Write-Log "Ensuring hypervisor launch type is auto."
    & bcdedit.exe /set hypervisorlaunchtype auto | Tee-Object -FilePath $logPath -Append
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to set hypervisorlaunchtype auto. Exit code: $LASTEXITCODE"
    }
}

function Install-DockerDesktop {
    $desktop = Join-Path $env:ProgramFiles "Docker\Docker\Docker Desktop.exe"
    if (Test-Path -LiteralPath $desktop) {
        Write-Log "Docker Desktop already installed: $desktop"
        return
    }

    $winget = Find-Winget
    $downloadDir = Join-Path $ProjectRoot "outputs\downloads"
    New-Item -ItemType Directory -Force -Path $downloadDir | Out-Null

    Write-Log "Downloading Docker Desktop installer with winget: $winget"
    & $winget download --id Docker.DockerDesktop -e --download-directory $downloadDir --accept-source-agreements --accept-package-agreements | Tee-Object -FilePath $logPath -Append
    if ($LASTEXITCODE -ne 0) {
        throw "Docker Desktop installer download failed. Exit code: $LASTEXITCODE"
    }

    $installer = Get-ChildItem -LiteralPath $downloadDir -Filter "Docker Desktop*.exe" |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if (-not $installer) {
        throw "Docker Desktop installer was not found in $downloadDir"
    }

    Write-Log "Installing Docker Desktop from: $($installer.FullName)"
    & $installer.FullName install --quiet --accept-license --backend=wsl-2 | Tee-Object -FilePath $logPath -Append
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        throw "Docker Desktop installation failed. Exit code: $code"
    }
}

function Start-DockerDesktop {
    $desktop = Join-Path $env:ProgramFiles "Docker\Docker\Docker Desktop.exe"
    if (-not (Test-Path -LiteralPath $desktop)) {
        throw "Docker Desktop executable not found: $desktop"
    }
    Write-Log "Starting Docker Desktop."
    Start-Process -FilePath $desktop | Out-Null
}

function Wait-DockerReady {
    param(
        [string]$DockerExe,
        [int]$TimeoutSec = 900
    )
    Write-Log "Waiting up to $TimeoutSec seconds for Docker daemon."
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    do {
        & $DockerExe info *> $null
        if ($LASTEXITCODE -eq 0) {
            Write-Log "Docker daemon is ready."
            return
        }
        Start-Sleep -Seconds 5
    } while ((Get-Date) -lt $deadline)
    throw "Docker daemon did not become ready within $TimeoutSec seconds."
}

function Run-Validation {
    param([string]$DockerExe)
    Push-Location $ProjectRoot
    try {
        Write-Log "Running docker compose config validation."
        & $DockerExe compose -f docker-compose.yml --profile wireless-video --profile monitoring config --quiet | Tee-Object -FilePath $logPath -Append
        if ($LASTEXITCODE -ne 0) {
            throw "docker compose config validation failed. Exit code: $LASTEXITCODE"
        }

        Write-Log "Running promtool validation inside prom/prometheus:latest."
        $mount = "${ProjectRoot}\monitoring:/etc/prometheus:ro"
        & $DockerExe run --rm -v $mount prom/prometheus:latest promtool check config /etc/prometheus/prometheus.yml | Tee-Object -FilePath $logPath -Append
        if ($LASTEXITCODE -ne 0) {
            throw "promtool container validation failed. Exit code: $LASTEXITCODE"
        }
    }
    finally {
        Pop-Location
    }
}

$script:restartRequired = $false

try {
    Assert-Admin
    Write-Log "Starting Docker install and validation for project: $ProjectRoot"
    $docker = Ensure-DockerCli
    Ensure-ComposePlugin -DockerExe $docker
    Enable-RequiredFeatures
    Ensure-HypervisorLaunch
    if (Test-RestartPending) {
        $script:restartRequired = $true
    }

    if ($script:restartRequired) {
        Write-Log "A Windows restart is required before Docker can start."
        Write-Result -Status "restart_required" -Message "Windows optional features were enabled and require restart before Docker validation." -Extra @{restart_required = $true}
        exit 3010
    }

    Install-DockerDesktop
    Start-DockerDesktop
    $docker = Find-Docker
    if (-not $docker) {
        throw "docker.exe was not found after Docker Desktop installation."
    }
    Write-Log "Using Docker CLI: $docker"
    Wait-DockerReady -DockerExe $docker
    Run-Validation -DockerExe $docker
    Write-Result -Status "ok" -Message "Docker Compose and promtool container validation passed." -Extra @{docker = $docker}
    Write-Log "Docker validation completed successfully."
}
catch {
    Write-Log "ERROR: $($_.Exception.Message)"
    Write-Result -Status "failed" -Message $_.Exception.Message -Extra @{restart_required = $script:restartRequired}
    exit 1
}

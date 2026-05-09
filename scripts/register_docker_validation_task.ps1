param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$TaskName = "LabSOPGuard-DockerInstallValidate"
)

$ErrorActionPreference = "Stop"

function Assert-Admin {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw "This script must run in an elevated Administrator PowerShell."
    }
}

Assert-Admin

$validationScript = Join-Path $ProjectRoot "scripts\install_docker_and_validate.ps1"
if (-not (Test-Path -LiteralPath $validationScript)) {
    throw "Validation script not found: $validationScript"
}

$quotedScript = '"' + $validationScript + '"'
$quotedProject = '"' + $ProjectRoot + '"'
$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File $quotedScript -ProjectRoot $quotedProject"
$trigger = New-ScheduledTaskTrigger -AtLogOn
$principal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType Interactive `
    -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2)

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Principal $principal `
    -Settings $settings `
    -Description "Install Docker Desktop and run LabSOPGuard Docker Compose/promtool validation after restart." `
    -Force | Out-Null

Write-Host "Registered scheduled task: $TaskName"

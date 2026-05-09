param(
  [string]$TaskName = "LabSOPGuardCaptureDaemon",
  [string]$PythonExe = "python",
  [string]$ConfigPath = "$PSScriptRoot\..\configs\runtime\multicam_soak.yaml"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path "$PSScriptRoot\.."
$ScriptPath = Join-Path $RepoRoot "scripts\run_capture_daemon.py"
$Action = New-ScheduledTaskAction -Execute $PythonExe -Argument "`"$ScriptPath`" --config `"$ConfigPath`"" -WorkingDirectory $RepoRoot
$Trigger = New-ScheduledTaskTrigger -AtStartup
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)

Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Description "LabSOPGuard supervised multi-camera capture daemon" -Force | Out-Null
Write-Host "Installed scheduled task: $TaskName"

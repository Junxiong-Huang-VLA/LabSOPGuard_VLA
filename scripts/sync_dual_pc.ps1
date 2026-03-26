param(
    [ValidateSet("status", "pull", "save", "sync")]
    [string]$Action = "status",
    [string]$Message = "chore: sync update",
    [string]$Remote = "origin",
    [string]$Branch = "main",
    [switch]$PushMirror
)

$ErrorActionPreference = "Stop"

function Write-Info([string]$text) {
    Write-Host "[INFO] $text" -ForegroundColor Cyan
}

function Write-WarnLine([string]$text) {
    Write-Host "[WARN] $text" -ForegroundColor Yellow
}

function Run-Git([string[]]$gitArgs) {
    Write-Info ("git " + ($gitArgs -join " "))
    & git @gitArgs
    if ($LASTEXITCODE -ne 0) {
        throw "git $($gitArgs -join ' ') failed with exit code $LASTEXITCODE"
    }
}

function Ensure-GitRepo() {
    $inside = (& git rev-parse --is-inside-work-tree 2>$null)
    if ($LASTEXITCODE -ne 0 -or $inside -ne "true") {
        throw "Current directory is not a git repository."
    }
}

function Show-Status() {
    Run-Git @("status", "-sb")
    Run-Git @("remote", "-v")
    $branch = (& git branch --show-current)
    Write-Info "current branch: $branch"
}

function Pull-Latest() {
    Run-Git @("fetch", $Remote)
    Run-Git @("pull", "--rebase", $Remote, $Branch)
}

function Save-Local() {
    Run-Git @("add", "-A")
    $hasChanges = (& git status --porcelain)
    if ([string]::IsNullOrWhiteSpace($hasChanges)) {
        Write-WarnLine "No local changes to commit."
        return
    }
    Run-Git @("commit", "-m", $Message)
}

function Push-OriginAndMaybeMirror() {
    Run-Git @("push", $Remote, $Branch)
    if ($PushMirror) {
        $mirrorExists = (& git remote) -contains "mirror"
        if (-not $mirrorExists) {
            Write-WarnLine "Remote 'mirror' not found. Skip mirror push."
            return
        }
        Run-Git @("push", "mirror", $Branch)
    }
}

try {
    Ensure-GitRepo
    switch ($Action) {
        "status" {
            Show-Status
        }
        "pull" {
            Pull-Latest
            Show-Status
        }
        "save" {
            Save-Local
            Push-OriginAndMaybeMirror
            Show-Status
        }
        "sync" {
            Pull-Latest
            Save-Local
            Push-OriginAndMaybeMirror
            Show-Status
        }
    }
    Write-Host "[DONE] sync_dual_pc completed." -ForegroundColor Green
}
catch {
    Write-Host "[ERROR] $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

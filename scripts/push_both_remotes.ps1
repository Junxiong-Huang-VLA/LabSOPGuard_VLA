param(
    [string]$Branch = "main",
    [int]$Retries = 5,
    [int]$RetryDelaySec = 8,
    [switch]$PushTags,
    [switch]$ForceMirror,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Write-Info([string]$text) {
    Write-Host "[INFO] $text" -ForegroundColor Cyan
}

function Write-WarnLine([string]$text) {
    Write-Host "[WARN] $text" -ForegroundColor Yellow
}

function Write-Ok([string]$text) {
    Write-Host "[OK] $text" -ForegroundColor Green
}

function Ensure-GitRepo() {
    $inside = (& git rev-parse --is-inside-work-tree 2>$null)
    if ($LASTEXITCODE -ne 0 -or $inside -ne "true") {
        throw "Current directory is not a git repository."
    }
}

function Invoke-GitWithRetry(
    [string[]]$GitArgs,
    [int]$AttemptCount,
    [int]$DelaySec
) {
    $cmdText = "git " + ($GitArgs -join " ")
    for ($i = 1; $i -le $AttemptCount; $i++) {
        if ($DryRun) {
            Write-Info "[DRY-RUN] $cmdText"
            return $true
        }

        Write-Info "$cmdText (attempt $i/$AttemptCount)"
        & git @GitArgs
        if ($LASTEXITCODE -eq 0) {
            Write-Ok "$cmdText succeeded."
            return $true
        }
        if ($i -lt $AttemptCount) {
            Write-WarnLine "$cmdText failed. retry after ${DelaySec}s..."
            Start-Sleep -Seconds $DelaySec
        }
    }

    Write-WarnLine "$cmdText failed after $AttemptCount attempts."
    return $false
}

try {
    Ensure-GitRepo

    $status = (& git status --porcelain)
    if (-not [string]::IsNullOrWhiteSpace($status)) {
        Write-WarnLine "Working tree is not clean. Please commit before pushing."
        exit 1
    }

    $originExists = (& git remote) -contains "origin"
    $mirrorExists = (& git remote) -contains "mirror"
    if (-not $originExists) {
        throw "Remote 'origin' not found."
    }
    if (-not $mirrorExists) {
        throw "Remote 'mirror' not found."
    }

    $okOrigin = Invoke-GitWithRetry -GitArgs @("push", "origin", $Branch) -AttemptCount $Retries -DelaySec $RetryDelaySec
    if (-not $okOrigin) {
        throw "Push to origin failed."
    }

    if ($PushTags) {
        [void](Invoke-GitWithRetry -GitArgs @("push", "origin", "--tags") -AttemptCount $Retries -DelaySec $RetryDelaySec)
    }

    $mirrorArgs = @("push", "mirror", $Branch)
    if ($ForceMirror) {
        $mirrorArgs += "--force-with-lease"
    }
    $okMirror = Invoke-GitWithRetry -GitArgs $mirrorArgs -AttemptCount $Retries -DelaySec $RetryDelaySec
    if (-not $okMirror -and -not $ForceMirror) {
        Write-WarnLine "Mirror push failed without force. Re-run with -ForceMirror if you want overwrite-safe force-with-lease."
        exit 1
    }
    if (-not $okMirror -and $ForceMirror) {
        throw "Push to mirror failed even with -ForceMirror."
    }

    if ($PushTags) {
        [void](Invoke-GitWithRetry -GitArgs @("push", "mirror", "--tags") -AttemptCount $Retries -DelaySec $RetryDelaySec)
    }

    Write-Ok "Both remotes are synced for branch '$Branch'."
}
catch {
    Write-Host "[ERROR] $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

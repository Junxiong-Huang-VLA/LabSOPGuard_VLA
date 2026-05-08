param(
    [string]$SourceRoot = 'D:\LabEmbodiedVLA',
    [string]$TargetRoot = (Resolve-Path '.').Path,
    [int64]$LargeFileBytes = 50MB
)

$ErrorActionPreference = 'Stop'

function Get-RelativePath([string]$base, [string]$path) {
    return $path.Substring($base.Length).TrimStart('\')
}

function Test-Excluded([string]$relativePath) {
    return $relativePath -match '(^|\\)(\.git|__pycache__|\.pytest_cache)(\\|$)'
}

function Test-UnderLinkedRoot([string]$relativePath, [string[]]$linkedRoots) {
    foreach ($root in $linkedRoots) {
        if ($relativePath.Equals($root, [StringComparison]::OrdinalIgnoreCase) -or
            $relativePath.StartsWith($root + '\', [StringComparison]::OrdinalIgnoreCase)) {
            return $true
        }
    }
    return $false
}

function Convert-ToLongPath([string]$path) {
    if ($path.StartsWith('\\?\')) {
        return $path
    }
    if ($path -match '^[A-Za-z]:\\') {
        return '\\?\' + $path
    }
    if ($path.StartsWith('\\')) {
        return '\\?\UNC\' + $path.TrimStart('\')
    }
    return $path
}

function Test-PathCompat([string]$path) {
    return (Test-Path -LiteralPath $path) -or (Test-Path -LiteralPath (Convert-ToLongPath $path))
}

function Get-ItemCompat([string]$path) {
    if (Test-Path -LiteralPath $path) {
        return Get-Item -LiteralPath $path -Force
    }
    return Get-Item -LiteralPath (Convert-ToLongPath $path) -Force
}

function Ensure-ParentDirectory([string]$path) {
    $parent = Split-Path -Parent $path
    if ($parent -and -not (Test-PathCompat $parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
}

function Copy-FileCompat([string]$sourcePath, [string]$targetPath) {
    Ensure-ParentDirectory $targetPath
    try {
        Copy-Item -LiteralPath $sourcePath -Destination $targetPath -Force -ErrorAction Stop
    } catch {
        $sourceDir = Split-Path -Parent $sourcePath
        $targetDir = Split-Path -Parent $targetPath
        $fileName = Split-Path -Leaf $sourcePath
        robocopy $sourceDir $targetDir $fileName /R:1 /W:1 /NFL /NDL /NJH /NJS /NP | Out-Null
        if ($LASTEXITCODE -ge 8) {
            throw
        }
    }
    if (-not (Test-PathCompat $targetPath)) {
        throw "Copy did not produce target: $targetPath"
    }
}

function New-DirectoryJunction([string]$targetPath, [string]$sourcePath) {
    Ensure-ParentDirectory $targetPath
    if (Test-PathCompat $targetPath) {
        return $false
    }
    New-Item -ItemType Junction -Path $targetPath -Target $sourcePath | Out-Null
    return $true
}

function New-FileSymlink([string]$targetPath, [string]$sourcePath) {
    Ensure-ParentDirectory $targetPath
    if (Test-PathCompat $targetPath) {
        return $false
    }
    New-Item -ItemType SymbolicLink -Path $targetPath -Target $sourcePath | Out-Null
    return $true
}

if (-not (Test-Path -LiteralPath $SourceRoot)) {
    throw "SourceRoot not found: $SourceRoot"
}
if (-not (Test-Path -LiteralPath $TargetRoot)) {
    throw "TargetRoot not found: $TargetRoot"
}

$SourceRoot = (Resolve-Path -LiteralPath $SourceRoot).Path.TrimEnd('\')
$TargetRoot = (Resolve-Path -LiteralPath $TargetRoot).Path.TrimEnd('\')

$stats = [ordered]@{
    DirectoryJunctionsCreated = 0
    FileSymlinksCreated       = 0
    FilesCopied               = 0
    FilesUpdated              = 0
    FilesSkippedExisting      = 0
    FilesSkippedTargetNewer   = 0
    FilesSkippedExcluded      = 0
    CopyErrors                = 0
    LinkErrors                = 0
}

# Known large artifact/data roots. These keep the target path usable without duplicating tens of GB.
$missingDirectoryLinkRoots = @(
    'LabSOPGuard\outputs\captures',
    'LabSOPGuard\integrated_system\outputs',
    'LabSOPGuard\runs\detect',
    'LabSOPGuard\outputs\predictions',
    'LabSOPGuard\data\dataset',
    'LabSOPGuard\data\processed',
    'LabSOPGuard\data\interim',
    'LabSOPGuard\outputs\inference',
    'lab_preprocessing\data\outputs'
)

# Existing aggregate directories must be preserved, so link only missing children under them.
$childLinkParents = @(
    'LabSOPGuard\uploads\experiments',
    'LabSOPGuard\outputs\experiments'
)

$linkedRoots = New-Object System.Collections.Generic.List[string]

foreach ($rel in $missingDirectoryLinkRoots) {
    $srcPath = Join-Path $SourceRoot $rel
    $dstPath = Join-Path $TargetRoot $rel
    if (Test-PathCompat $srcPath) {
        try {
            if (New-DirectoryJunction $dstPath $srcPath) {
                $stats.DirectoryJunctionsCreated++
                $linkedRoots.Add($rel)
                Write-Host "JUNCTION $rel -> $srcPath"
            } elseif (Test-PathCompat $dstPath) {
                $item = Get-ItemCompat $dstPath
                if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                    $linkedRoots.Add($rel)
                }
            }
        } catch {
            $stats.LinkErrors++
            Write-Warning "Failed to create junction: $rel :: $($_.Exception.Message)"
        }
    }
}

foreach ($parentRel in $childLinkParents) {
    $srcParent = Join-Path $SourceRoot $parentRel
    $dstParent = Join-Path $TargetRoot $parentRel
    if (-not (Test-PathCompat $srcParent)) { continue }
    if (-not (Test-PathCompat $dstParent)) {
        New-Item -ItemType Directory -Path $dstParent -Force | Out-Null
    }
    Get-ChildItem -LiteralPath $srcParent -Directory -Force | ForEach-Object {
        $childRel = Join-Path $parentRel $_.Name
        $dstChild = Join-Path $TargetRoot $childRel
        if (-not (Test-PathCompat $dstChild)) {
            try {
                if (New-DirectoryJunction $dstChild $_.FullName) {
                    $stats.DirectoryJunctionsCreated++
                    $linkedRoots.Add($childRel)
                    Write-Host "JUNCTION $childRel -> $($_.FullName)"
                }
            } catch {
                $stats.LinkErrors++
                Write-Warning "Failed to create child junction: $childRel :: $($_.Exception.Message)"
            }
        }
    }
}

# Copy small source files and symlink large missing files. Existing target files are preserved unless source is newer and small.
Get-ChildItem -LiteralPath $SourceRoot -Recurse -File -Force | ForEach-Object {
    $rel = Get-RelativePath $SourceRoot $_.FullName
    if (Test-Excluded $rel) {
        $stats.FilesSkippedExcluded++
        return
    }
    if (Test-UnderLinkedRoot $rel $linkedRoots.ToArray()) {
        return
    }

    $dstFile = Join-Path $TargetRoot $rel
    if (-not (Test-PathCompat $dstFile)) {
        try {
            if ($_.Length -ge $LargeFileBytes) {
                if (New-FileSymlink $dstFile $_.FullName) {
                    $stats.FileSymlinksCreated++
                    Write-Host "SYMLINK $rel -> $($_.FullName)"
                }
            } else {
                Copy-FileCompat $_.FullName $dstFile
                $stats.FilesCopied++
            }
        } catch {
            $stats.CopyErrors++
            Write-Warning "Failed to add: $rel :: $($_.Exception.Message)"
        }
        return
    }

    try {
        $dstItem = Get-ItemCompat $dstFile
        if (($dstItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            $stats.FilesSkippedExisting++
            return
        }
        if ($_.LastWriteTimeUtc -gt $dstItem.LastWriteTimeUtc -and $_.Length -lt $LargeFileBytes) {
            Copy-FileCompat $_.FullName $dstFile
            $stats.FilesUpdated++
        } elseif ($_.LastWriteTimeUtc -gt $dstItem.LastWriteTimeUtc -and $_.Length -ge $LargeFileBytes) {
            # Avoid replacing existing local large artifacts; target may contain newer local runs nearby.
            $stats.FilesSkippedExisting++
        } elseif ($_.LastWriteTimeUtc -le $dstItem.LastWriteTimeUtc) {
            $stats.FilesSkippedTargetNewer++
        } else {
            $stats.FilesSkippedExisting++
        }
    } catch {
        $stats.CopyErrors++
        Write-Warning "Failed to compare/update: $rel :: $($_.Exception.Message)"
    }
}

Write-Host '--- Integration summary ---'
$stats.GetEnumerator() | ForEach-Object { '{0}: {1}' -f $_.Key, $_.Value }

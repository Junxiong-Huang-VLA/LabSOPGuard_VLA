# Dual-PC Sync Guide

Use this guide when you switch between company PC and home PC.

## 1) Recommended workflow

- Single source of truth: `origin/main`
- Start work: always pull first
- End work: commit and push

## 2) Script

Script path:

`scripts/sync_dual_pc.ps1`

## 3) Commands

Run in project root (`D:\LabEmbodiedVLA\LabSOPGuard`):

```powershell
# Check repository state
powershell -ExecutionPolicy Bypass -File .\scripts\sync_dual_pc.ps1 -Action status

# Pull latest from origin/main
powershell -ExecutionPolicy Bypass -File .\scripts\sync_dual_pc.ps1 -Action pull

# Commit+push local changes to origin/main
powershell -ExecutionPolicy Bypass -File .\scripts\sync_dual_pc.ps1 -Action save -Message "feat: your message"

# Pull + commit + push in one command
powershell -ExecutionPolicy Bypass -File .\scripts\sync_dual_pc.ps1 -Action sync -Message "chore: sync update"
```

If you also want to push to `mirror`:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\sync_dual_pc.ps1 -Action save -Message "feat: update" -PushMirror
```

## 4) Typical daily usage

- Company PC before coding:
  `-Action pull`
- Company PC after coding:
  `-Action save -Message "..."`
- Home PC before coding:
  `-Action pull`
- Home PC after coding:
  `-Action save -Message "..."`

## 5) Notes

- The script does not use destructive commands like `reset --hard`.
- If there is no change, `save` will skip commit.
- If mirror push fails (network/auth), origin push is still kept.

## 6) One-Command Push To Both Remotes

Script path:

`scripts/push_both_remotes.ps1`

Use after local commit is ready:

```powershell
# Normal mode (push origin then mirror)
powershell -ExecutionPolicy Bypass -File .\scripts\push_both_remotes.ps1

# With retry tuning
powershell -ExecutionPolicy Bypass -File .\scripts\push_both_remotes.ps1 -Retries 6 -RetryDelaySec 10

# If mirror branch diverged and you want safe force overwrite
powershell -ExecutionPolicy Bypass -File .\scripts\push_both_remotes.ps1 -ForceMirror
```

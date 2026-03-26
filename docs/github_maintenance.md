# GitHub Maintenance Guide

## Branch Strategy

- `main`: stable release branch
- `dev`: integration branch
- `feature/*`: feature branches

## Commit Style

Use Conventional Commits:

- `feat:`
- `fix:`
- `docs:`
- `refactor:`
- `test:`
- `chore:`

## Daily Workflow

```bash
git status
git pull --rebase origin main
git add -A
git commit -m "feat(monitor): add event structurer"
git push origin main
```

## Safe Sync When Remote Has Updates

```bash
git fetch origin
git pull --rebase origin main
# solve conflicts if any
git push origin main
```

## Large Files Policy

Do not commit:

- raw videos
- model weights/checkpoints
- runtime outputs/logs/cache

Keep placeholders only and manage large assets in external storage.

# Contributing Guide

Thanks for contributing to LabSOPGuard.

## Branch Workflow

- `main`: stable
- `dev`: integration
- `feature/*`: feature implementation

## Commit Convention

Use Conventional Commits:

- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation
- `refactor:` refactor
- `test:` tests
- `chore:` maintenance

## Local Validation

Before opening a PR, run:

```bash
conda run -n LabSOPGuard python scripts/check_env.py --project-name LabSOPGuard
conda run -n LabSOPGuard python scripts/data_check.py --config configs/data/dataset.yaml
conda run -n LabSOPGuard python scripts/infer.py --video data/raw/videos/session_001.mp4 --sample-id smoke
```

## Data and Privacy

- Do not commit raw private videos.
- Keep only placeholders in `data/raw`.
- Do not commit runtime outputs/logs/weights.

## PR Requirements

- Fill PR template completely.
- Explain SOP impact and validation evidence.
- Update docs if behavior changed.

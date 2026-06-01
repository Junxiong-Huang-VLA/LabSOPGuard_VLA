# Contributing Guide

Thanks for contributing to LabEmbodied.

## Branch Workflow

- `main`: stable
- `dev`: integration
- `feature/*`: feature implementation

## Commit Convention

- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation
- `refactor:` refactor
- `test:` tests
- `chore:` maintenance

## Local Validation

```powershell
python scripts/check_env.py --project-name LabEmbodied
python scripts/startup_runtime_preflight.py
python scripts/check_project_scope.py
python -m pytest -q
cd frontend
npm run build
```

## Data and Privacy

- Do not commit raw private videos.
- Keep YOLO/model weights in `D:\LabModels` or another external `LAB_MODELS_DIR`.
- Keep raw videos in `LAB_VIDEO_STORE_ROOT`.
- Keep real secrets in local `.env` only.
- Do not commit runtime outputs, logs, caches, or generated reports unless they are tiny intentional fixtures.

## PR Requirements

- Explain evidence-pipeline impact and validation evidence.
- Update docs when behavior or runtime setup changes.
- Keep PTZ, camera orchestration, and wireless-video transport out of this repo.

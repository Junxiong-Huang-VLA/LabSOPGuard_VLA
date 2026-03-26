# lab_titration_vla

This folder is the new primary project structure for titration-task VLA + robotic execution.

- Existing modules in `src/project_name` are retained as compatibility backend.
- New modules here wrap and reuse validated code to avoid breaking current progress.
- Migration strategy: train/infer in current pipeline, then switch runtime entry to `deploy/runtime_pipeline/run_runtime_pipeline.py`.


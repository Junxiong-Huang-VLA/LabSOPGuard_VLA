# LabTitrationVLA Migration Map

This project now includes the new primary structure at:

- `lab_titration_vla/`

To avoid losing progress, validated modules are retained and reused from existing code:

- detection core: `src/project_name/detection/multi_level_detector.py`
- perception core: `src/project_name/perception/*`
- instruction parser: `src/project_name/language/instruction_parser.py`
- action policy: `src/project_name/action/policy.py`
- robot command bridge: `src/project_name/action/robot_command_builder.py`
- vla pipeline: `src/project_name/pipelines/vla_pipeline.py`

New runtime entry:

- `lab_titration_vla/deploy/runtime_pipeline/run_runtime_pipeline.py`

Compatibility note:

- Existing scripts under `scripts/` are intentionally preserved.
- New structure is ready for progressive migration without breaking current runs.


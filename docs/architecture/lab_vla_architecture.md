# Lab VLA Unified Architecture

## Goal

Build a single runtime backbone for lab embodied VLA with strict boundaries:

1. camera/input
2. perception output
3. scene state
4. task parse + skill planning
5. safety gate + recovery
6. execution adapter
7. verify + logs + metrics

## Mainline Directories

- `src/lab_vla/core`
- `src/lab_vla/adapters`
- `src/lab_vla/perception`
- `src/lab_vla/memory`
- `src/lab_vla/skills`
- `src/lab_vla/planner`
- `src/lab_vla/safety`
- `src/lab_vla/execution`
- `src/lab_vla/evaluation`
- `src/lab_vla/apps`

## Runtime Chain

`CameraInput -> DetectorService -> SceneState -> TaskParser/SkillPlanner -> SafetyGate -> Executor -> RuntimeStore + Metrics`

## Contract Boundaries

- Planner returns `SkillPlan` and `SkillStep`, never invokes robot SDK.
- Adapter layer executes commands only, no planning logic.
- Perception only emits detections and scene state, no SOP business policy.
- Safety runs rule checks and recovery planning only.

## Configuration Layout

- `configs/devices/default_devices.yaml`
- `configs/models/perception.yaml`
- `configs/calibration/default_calibration.yaml`
- `configs/tasks/default_task.yaml`
- `configs/runtime/lab_vla_runtime.yaml`

All path/threshold/runtime parameters are externally configured.

## Real Hardware Integration Points

- camera source switch: `configs/devices/default_devices.yaml` (`camera.mode`, `camera.source`)
- detector backend and model: `configs/model/detection_runtime.yaml`
- robot execution backend: `configs/devices/default_devices.yaml` (`robot.adapter=moveit`)
- camera intrinsics and hand-eye: `configs/calibration/default_calibration.yaml`

## Compatibility

- Existing `integrated_system` and `lab_titration_vla` directories remain unchanged.
- Existing scripts remain; unified entry added as `scripts/run_lab_vla_runtime.py`.

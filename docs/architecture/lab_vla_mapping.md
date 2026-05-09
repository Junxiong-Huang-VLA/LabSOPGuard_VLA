# Lab VLA Mapping

This document maps existing modules into the new `src/lab_vla` mainline.

## Old to New Mapping

| Old Path | Category | New Mainline Module | Strategy |
|---|---|---|---|
| `src/project_name/common/config.py` | core | `src/lab_vla/core/config.py` | migrated utility + normalized runtime loader |
| `src/project_name/common/schemas.py` | core/contract | `src/lab_vla/core/contracts.py` | expanded protocol schema |
| `src/project_name/common/logging_utils.py` | core | `src/lab_vla/core/logger.py` | wrapped logger setup |
| `src/project_name/video/capture.py` | adapters/perception | `src/lab_vla/perception/camera_input.py` | wrapper + mock camera fallback |
| `src/project_name/detection/multi_level_detector.py` | perception | `src/lab_vla/perception/detector_service.py` | wrapped detector service |
| `src/project_name/perception/perception_engine.py` | perception | `src/lab_vla/perception/detector_service.py` | scene-state oriented adaptation |
| `src/project_name/language/instruction_parser.py` | planner | `src/lab_vla/planner/task_parser.py` | via qwen interface wrapper |
| `src/project_name/action/policy.py` | skills/planner | `src/lab_vla/skills/library.py` + `src/lab_vla/planner/skill_planner.py` | reused for action payload generation |
| `src/project_name/action/robot_command_builder.py` | execution/adapters | `src/lab_vla/adapters/robot_adapter.py` | wrapped robot command generation |
| `src/project_name/monitoring/sop_engine.py` | safety | `src/lab_vla/safety/safety_gate.py` | reused engine for rule checks |
| `src/project_name/alerting/notifier.py` | memory/log | `src/lab_vla/memory/runtime_store.py` | unified trace + summary store |
| `src/project_name/pipelines/vla_pipeline.py` | app/task | `src/lab_vla/core/runtime.py` | replaced by unified closed-loop runtime |
| `scripts/run_vla_dynamic_closed_loop.py` | app/task | `scripts/run_lab_vla_runtime.py` | new unified runtime entry kept alongside old |
| `scripts/run_vla_robot_bridge.py` | adapters/execution | `src/lab_vla/adapters/robot_adapter.py` | behavior absorbed by adapter layer |
| `integrated_system/app_integrated.py` | app/task | `src/lab_vla/apps/main.py` | retained old app; new runtime app added |
| `integrated_system/hand_detection.py` | perception | `src/lab_vla/perception/*` | keep old app-specific path unchanged |
| `integrated_system/step_checker.py` | safety | `src/lab_vla/safety/safety_gate.py` | canonical safety path in new mainline |
| `lab_titration_vla/control/moveit_interface/command_bridge.py` | adapters | `src/lab_vla/adapters/robot_adapter.py` | direct wrapper |
| `lab_titration_vla/control/grasp_planner/planner.py` | skills/execution | `src/lab_vla/skills/library.py` | conceptual reuse |
| `lab_titration_vla/control/recovery_controller/controller.py` | safety | `src/lab_vla/safety/safety_gate.py` | direct wrapper |
| `lab_titration_vla/vlm/qwen3_vl_parser/parser.py` | planner | `src/lab_vla/planner/task_parser.py` | direct wrapper |
| `lab_titration_vla/vlm/task_planner/planner.py` | planner | `src/lab_vla/planner/skill_planner.py` | normalized into SkillPlan |
| `lab_titration_vla/vlm/safety_checker/checker.py` | safety | `src/lab_vla/safety/safety_gate.py` | direct wrapper |

## Classification Summary

- core: runtime/config/logger/contracts/device context
- adapters: camera + robot adapter boundary
- perception: detector + scene state
- memory: runtime trace and summary
- skills: reusable step library
- planner: task parser + skill planner + backend protocol
- safety: SOP and VLM safety fusion + recovery
- execution: command dispatch and verify
- evaluation: metrics collector
- app/task: `lab_vla.apps.main` and script entry

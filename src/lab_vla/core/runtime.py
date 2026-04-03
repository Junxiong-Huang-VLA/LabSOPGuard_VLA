from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

from lab_vla.core.config import RuntimeConfig, load_yaml
from lab_vla.core.device_context import DeviceContext
from lab_vla.core.logger import setup_logger
from lab_vla.evaluation.metrics import MetricsCollector
from lab_vla.execution.executor import Executor
from lab_vla.memory.runtime_store import RuntimeStore
from lab_vla.perception.camera_input import CameraInput, CameraInputConfig
from lab_vla.perception.detector_service import DetectorService
from lab_vla.planner.skill_planner import SkillPlanner
from lab_vla.planner.task_parser import TaskParser
from lab_vla.safety.safety_gate import SafetyGate


def _load_component_configs(cfg: RuntimeConfig) -> Dict[str, Any]:
    devices = load_yaml(cfg.resolve_path(cfg.get("devices_config", "configs/devices/default_devices.yaml")))
    models = load_yaml(cfg.resolve_path(cfg.get("models_config", "configs/models/perception.yaml")))
    calibration = load_yaml(cfg.resolve_path(cfg.get("calibration_config", "configs/calibration/default_calibration.yaml")))
    task = load_yaml(cfg.resolve_path(cfg.get("task_config", "configs/tasks/default_task.yaml")))
    safety_rules = load_yaml(cfg.resolve_path(cfg.get("safety_rules_config", "configs/sop/rules.yaml")))
    robot_bridge = load_yaml(cfg.resolve_path(cfg.get("robot_bridge_config", "configs/robot/bridge.yaml")))
    return {
        "devices": devices,
        "models": models,
        "calibration": calibration,
        "task": task,
        "safety_rules": safety_rules,
        "robot_bridge": robot_bridge,
    }


def run_lab_vla(runtime_config_path: str) -> Dict[str, Any]:
    runtime_cfg = RuntimeConfig.from_file(runtime_config_path)
    loaded = _load_component_configs(runtime_cfg)

    log_file = runtime_cfg.resolve_path(runtime_cfg.get("log_file", "logs/lab_vla/runtime.log"))
    logger = setup_logger("lab_vla.runtime", log_file=log_file, level=str(runtime_cfg.get("log_level", "INFO")))

    devices = DeviceContext.from_configs(
        devices_cfg=loaded["devices"],
        calibration_cfg=loaded["calibration"],
    )
    task_cfg = loaded["task"]
    task_id = str(task_cfg.get("task_id", "lab_vla_task"))
    instruction = str(task_cfg.get("instruction", "Pick the sample container and place it to target zone."))
    max_frames = int(runtime_cfg.get("max_frames", 10))

    parser = TaskParser()
    planner = SkillPlanner()
    detector = DetectorService(
        confidence_threshold=float(loaded["models"].get("confidence_threshold", 0.45)),
        runtime_config_path=runtime_cfg.resolve_path(
            loaded["models"].get("detection_runtime_config", "configs/model/detection_runtime.yaml")
        ),
    )
    safety = SafetyGate(
        rules=loaded["safety_rules"],
        cooldown_seconds=float(runtime_cfg.get("safety_cooldown_seconds", 1.0)),
    )
    executor = Executor(
        bridge_cfg=loaded["robot_bridge"].get("bridge", {}),
        adapter_name=str(devices.robot.adapter),
    )
    store = RuntimeStore(
        trace_jsonl=runtime_cfg.resolve_path(runtime_cfg.get("trace_jsonl", "outputs/runtime/lab_vla_trace.jsonl")),
        summary_json=runtime_cfg.resolve_path(runtime_cfg.get("summary_json", "outputs/runtime/lab_vla_summary.json")),
    )
    metrics = MetricsCollector()

    logger.info("runtime start task_id=%s camera_mode=%s robot_adapter=%s", task_id, devices.camera.mode, devices.robot.adapter)
    command = parser.parse(task_id=task_id, instruction=instruction)
    cam = CameraInput(
        CameraInputConfig(
            mode=devices.camera.mode,
            source=devices.camera.source,
            target_fps=devices.camera.target_fps,
        )
    )

    last_result: Dict[str, Any] = {}
    for frame in cam.frames(max_frames=max_frames):
        perception = detector.infer(frame)
        scene = detector.build_scene_state(sample_id=task_id, target_object=command.target_object, perception=perception)
        metrics.on_scene(scene)

        plan = planner.plan(command=command, scene=scene)
        metrics.on_plan()

        plan_payload = {"plan_id": plan.plan_id, "steps": [asdict(x) for x in plan.steps]}
        safety_decision = safety.evaluate(perception=perception, scene=scene, plan_payload=plan_payload)
        if not safety_decision.safe_to_execute:
            metrics.on_safety_block()

        execution = executor.run(
            task=command,
            scene=scene,
            plan=plan,
            safety=safety_decision,
            calibration=devices.calibration,
        )
        metrics.on_execution(execution)

        row = {
            "task_command": asdict(command),
            "scene_state": asdict(scene),
            "skill_plan": asdict(plan),
            "safety_decision": asdict(safety_decision),
            "execution_result": asdict(execution),
        }
        store.append(row)
        last_result = row

    summary = {
        "task_id": task_id,
        "instruction": instruction,
        "device_context": {
            "camera": asdict(devices.camera),
            "robot": asdict(devices.robot),
        },
        "safety_status": safety.status(),
        "metrics": metrics.metrics.as_dict(),
        "last_result": last_result,
    }
    store.write_summary(summary)
    logger.info("runtime end metrics=%s", summary["metrics"])
    return summary


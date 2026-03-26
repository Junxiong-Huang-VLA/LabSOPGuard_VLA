from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
import sys

from fastapi import FastAPI
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from project_name.common.config import load_yaml
from project_name.pipelines.vla_pipeline import VLAPipeline

app = FastAPI(title="LabSOPGuard Runtime API", version="0.1.0")


class RunRequest(BaseModel):
    sample_id: str
    instruction: str
    rgb_path: str
    depth_path: Optional[str] = None
    robot_config: str = "configs/robot/bridge.yaml"


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/v1/runtime/run")
def run_runtime(req: RunRequest) -> dict:
    cfg = load_yaml(req.robot_config)
    cam = cfg.get("camera", {}) if isinstance(cfg, dict) else {}
    intrinsics = cam.get("intrinsics")
    extrinsics = cam.get("hand_eye_extrinsics")

    pipe = VLAPipeline(robot_config_path=req.robot_config)
    result = pipe.run(
        sample_id=req.sample_id,
        instruction=req.instruction,
        rgb_path=req.rgb_path,
        depth_path=req.depth_path,
        camera_intrinsics=intrinsics,
        hand_eye_extrinsics=extrinsics,
    )

    out_path = Path("outputs/predictions") / f"{req.sample_id}_api_runtime.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sample_id": result.sample_id,
        "instruction": result.instruction,
        "target_name": result.perception.target_name,
        "center_point": result.perception.center_point,
        "depth_info": result.perception.depth_info,
        "action_sequence": result.action.action_sequence,
        "robot_command": result.action.robot_command,
        "robot_command_status": result.action.metadata.get("robot_command_status", "unknown"),
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from project_name.common.config import load_yaml
from project_name.pipelines.vla_pipeline import VLAPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runtime pipeline for lab_titration_vla.")
    parser.add_argument("--sample-id", default="runtime_demo")
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--rgb-path", required=True)
    parser.add_argument("--depth-path", default=None)
    parser.add_argument("--robot-config", default="configs/robot/bridge.yaml")
    parser.add_argument("--out-json", default="outputs/predictions/runtime_vla_result.json")
    parser.add_argument("--out-command", default="outputs/predictions/runtime_robot_command.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.robot_config)
    cam = cfg.get("camera", {}) if isinstance(cfg, dict) else {}
    intrinsics = cam.get("intrinsics")
    extrinsics = cam.get("hand_eye_extrinsics")

    pipe = VLAPipeline(robot_config_path=args.robot_config)
    result = pipe.run(
        sample_id=args.sample_id,
        instruction=args.instruction,
        rgb_path=args.rgb_path,
        depth_path=args.depth_path,
        camera_intrinsics=intrinsics,
        hand_eye_extrinsics=extrinsics,
    )

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")

    cmd_json = Path(args.out_command)
    cmd_json.parent.mkdir(parents=True, exist_ok=True)
    cmd_json.write_text(
        json.dumps(result.action.robot_command or {"status": "missing"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"runtime result: {out_json}")
    print(f"runtime command: {cmd_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


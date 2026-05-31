from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from project_name.common.config import load_yaml
from project_name.pipelines.vla_pipeline import VLAPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VLA perception->action->robot bridge and export robot command JSON."
    )
    parser.add_argument("--sample-id", default="robot_bridge_demo")
    parser.add_argument(
        "--instruction",
        default="pick the sample container and place it to the target zone carefully",
    )
    parser.add_argument("--rgb-path", required=True)
    parser.add_argument("--depth-path", default=None)
    parser.add_argument(
        "--robot-config",
        default="configs/robot/bridge.yaml",
        help="Contains bridge defaults + camera intrinsics/extrinsics.",
    )
    parser.add_argument(
        "--raw-target-json",
        default=None,
        help='Optional raw target JSON string, e.g. {"target_name":"sample_container","bbox":[100,100,240,260]}',
    )
    parser.add_argument("--out-json", default="outputs/predictions/vla_robot_bridge.json")
    parser.add_argument("--out-command-json", default="outputs/predictions/robot_action_command.json")
    return parser.parse_args()


def _parse_raw_target(raw_target_json: Optional[str]) -> Optional[Dict[str, Any]]:
    if not raw_target_json:
        return None
    return json.loads(raw_target_json)


def main() -> int:
    args = parse_args()
    rgb_path = Path(args.rgb_path)
    if not rgb_path.exists():
        raise FileNotFoundError(f"rgb image not found: {rgb_path}")
    if args.depth_path and not Path(args.depth_path).exists():
        raise FileNotFoundError(f"depth image not found: {args.depth_path}")

    robot_cfg = load_yaml(args.robot_config)
    camera_cfg = robot_cfg.get("camera", {}) if isinstance(robot_cfg, dict) else {}
    intrinsics = camera_cfg.get("intrinsics")
    extrinsics = camera_cfg.get("hand_eye_extrinsics")

    pipeline = VLAPipeline(robot_config_path=args.robot_config)
    result = pipeline.run(
        sample_id=args.sample_id,
        instruction=args.instruction,
        rgb_path=str(rgb_path),
        depth_path=args.depth_path,
        raw_target=_parse_raw_target(args.raw_target_json),
        camera_intrinsics=intrinsics,
        hand_eye_extrinsics=extrinsics,
    )

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")

    cmd_out = Path(args.out_command_json)
    cmd_out.parent.mkdir(parents=True, exist_ok=True)
    command = result.action.robot_command or {"status": "missing"}
    cmd_out.write_text(json.dumps(command, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"vla result: {out}")
    print(f"robot command: {cmd_out}")
    print(f"robot command status: {command.get('status', 'unknown')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

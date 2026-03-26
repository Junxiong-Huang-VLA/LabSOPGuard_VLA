from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from project_name.common.config import load_yaml
from project_name.utils.spatial import camera_to_robot_xyz, pixel_to_camera_xyz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check RealSense-to-robot calibration by mapping pixel/depth to robot XYZ."
    )
    parser.add_argument("--robot-config", default="configs/robot/bridge.yaml")
    parser.add_argument("--u", type=float, default=None, help="pixel x")
    parser.add_argument("--v", type=float, default=None, help="pixel y")
    parser.add_argument("--depth-m", type=float, default=None, help="depth in meters")
    parser.add_argument(
        "--bbox",
        default=None,
        help="bbox in xyxy format: x1,y1,x2,y2. If provided, center will be used as (u,v).",
    )
    parser.add_argument(
        "--samples-json",
        default=None,
        help='Optional samples json list, e.g. [{"name":"p1","u":320,"v":240,"depth_m":0.5}]',
    )
    parser.add_argument("--out-json", default="outputs/reports/calibration_check.json")
    return parser.parse_args()


def _parse_bbox_center(bbox_text: str) -> List[float]:
    parts = [float(x.strip()) for x in bbox_text.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be x1,y1,x2,y2")
    x1, y1, x2, y2 = parts
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]


def _run_one(
    name: str,
    u: float,
    v: float,
    depth_m: float,
    intrinsics: Dict[str, float],
    extrinsics: Dict[str, Any],
) -> Dict[str, Any]:
    camera_xyz = pixel_to_camera_xyz(u=u, v=v, z_m=depth_m, intrinsics=intrinsics)
    robot_xyz = camera_to_robot_xyz(camera_xyz, extrinsics)
    return {
        "name": name,
        "pixel_uv": [u, v],
        "depth_m": depth_m,
        "camera_xyz_m": camera_xyz,
        "robot_xyz_m": robot_xyz,
    }


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.robot_config)
    cam = cfg.get("camera", {}) if isinstance(cfg, dict) else {}
    intrinsics = cam.get("intrinsics", {})
    extrinsics = cam.get("hand_eye_extrinsics", {})

    required_k = ("fx", "fy", "cx", "cy")
    if any(k not in intrinsics for k in required_k):
        raise ValueError(f"intrinsics missing keys: {required_k}")

    rows: List[Dict[str, Any]] = []

    if args.samples_json:
        samples = json.loads(args.samples_json)
        if not isinstance(samples, list):
            raise ValueError("--samples-json must be a json list")
        for i, s in enumerate(samples):
            row = _run_one(
                name=str(s.get("name", f"sample_{i:03d}")),
                u=float(s["u"]),
                v=float(s["v"]),
                depth_m=float(s["depth_m"]),
                intrinsics=intrinsics,
                extrinsics=extrinsics,
            )
            rows.append(row)
    else:
        if args.depth_m is None:
            raise ValueError("either --samples-json or --depth-m must be provided")
        if args.bbox:
            center = _parse_bbox_center(args.bbox)
            u, v = float(center[0]), float(center[1])
        else:
            if args.u is None or args.v is None:
                raise ValueError("provide --u --v or --bbox")
            u, v = float(args.u), float(args.v)
        rows.append(
            _run_one(
                name="single",
                u=u,
                v=v,
                depth_m=float(args.depth_m),
                intrinsics=intrinsics,
                extrinsics=extrinsics,
            )
        )

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "robot_config": args.robot_config,
        "intrinsics": intrinsics,
        "hand_eye_extrinsics": extrinsics,
        "results": rows,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved: {out}")
    for r in rows:
        print(
            f"[{r['name']}] uv={r['pixel_uv']} depth={r['depth_m']:.4f}m "
            f"camera_xyz={r['camera_xyz_m']} robot_xyz={r['robot_xyz_m']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


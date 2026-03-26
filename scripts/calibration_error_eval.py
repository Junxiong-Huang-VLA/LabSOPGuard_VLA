from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from project_name.common.config import load_yaml
from project_name.utils.spatial import camera_to_robot_xyz, pixel_to_camera_xyz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate hand-eye calibration error with measured robot points."
    )
    parser.add_argument("--robot-config", default="configs/robot/bridge.yaml")
    parser.add_argument(
        "--samples-csv",
        default=None,
        help="CSV columns: name,u,v,depth_m,measured_x,measured_y,measured_z",
    )
    parser.add_argument(
        "--samples-json",
        default=None,
        help='JSON list. Example: [{"name":"p1","u":640,"v":360,"depth_m":0.55,"measured_xyz":[-0.46,-0.13,0.58]}]',
    )
    parser.add_argument("--out-json", default="outputs/reports/calibration_error_report.json")
    parser.add_argument("--out-csv", default="outputs/reports/calibration_error_points.csv")
    return parser.parse_args()


def _read_samples_from_csv(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"samples csv not found: {p}")
    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "name": r.get("name", f"sample_{len(rows):03d}"),
                    "u": float(r["u"]),
                    "v": float(r["v"]),
                    "depth_m": float(r["depth_m"]),
                    "measured_xyz": [
                        float(r["measured_x"]),
                        float(r["measured_y"]),
                        float(r["measured_z"]),
                    ],
                }
            )
    return rows


def _read_samples(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.samples_csv:
        return _read_samples_from_csv(args.samples_csv)
    if args.samples_json:
        data = json.loads(args.samples_json)
        if not isinstance(data, list):
            raise ValueError("--samples-json must be a list")
        rows = []
        for i, s in enumerate(data):
            m = s.get("measured_xyz")
            if not isinstance(m, list) or len(m) != 3:
                raise ValueError(f"samples_json[{i}].measured_xyz must be [x,y,z]")
            rows.append(
                {
                    "name": str(s.get("name", f"sample_{i:03d}")),
                    "u": float(s["u"]),
                    "v": float(s["v"]),
                    "depth_m": float(s["depth_m"]),
                    "measured_xyz": [float(m[0]), float(m[1]), float(m[2])],
                }
            )
        return rows
    raise ValueError("provide --samples-csv or --samples-json")


def _vec_sub(a: List[float], b: List[float]) -> List[float]:
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def _vec_abs(a: List[float]) -> List[float]:
    return [abs(a[0]), abs(a[1]), abs(a[2])]


def _norm(a: List[float]) -> float:
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def _mean(vals: List[float]) -> float:
    return sum(vals) / max(1, len(vals))


def _rmse(vals: List[float]) -> float:
    return math.sqrt(sum(v * v for v in vals) / max(1, len(vals)))


def _axis_stats(points: List[Dict[str, Any]], key: str) -> Tuple[float, float]:
    vals = [float(p[key]) for p in points]
    return _mean([abs(v) for v in vals]), _rmse(vals)


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.robot_config)
    cam = cfg.get("camera", {}) if isinstance(cfg, dict) else {}
    intrinsics = cam.get("intrinsics", {})
    extrinsics = cam.get("hand_eye_extrinsics", {})

    req = ("fx", "fy", "cx", "cy")
    if any(k not in intrinsics for k in req):
        raise ValueError(f"intrinsics missing keys: {req}")

    samples = _read_samples(args)
    point_rows: List[Dict[str, Any]] = []
    err_xyz_abs: List[float] = []
    err_norm: List[float] = []

    for s in samples:
        cam_xyz = pixel_to_camera_xyz(
            u=float(s["u"]),
            v=float(s["v"]),
            z_m=float(s["depth_m"]),
            intrinsics=intrinsics,
        )
        if cam_xyz is None:
            continue
        pred_xyz = camera_to_robot_xyz(cam_xyz, extrinsics)
        if pred_xyz is None:
            continue
        measured = s["measured_xyz"]
        err = _vec_sub(pred_xyz, measured)
        abs_err = _vec_abs(err)
        dist_err = _norm(err)
        point_rows.append(
            {
                "name": s["name"],
                "u": s["u"],
                "v": s["v"],
                "depth_m": s["depth_m"],
                "pred_x": pred_xyz[0],
                "pred_y": pred_xyz[1],
                "pred_z": pred_xyz[2],
                "measured_x": measured[0],
                "measured_y": measured[1],
                "measured_z": measured[2],
                "err_x": err[0],
                "err_y": err[1],
                "err_z": err[2],
                "abs_err_x": abs_err[0],
                "abs_err_y": abs_err[1],
                "abs_err_z": abs_err[2],
                "err_dist": dist_err,
            }
        )
        err_xyz_abs.extend(abs_err)
        err_norm.append(dist_err)

    mae_x, rmse_x = _axis_stats(point_rows, "err_x")
    mae_y, rmse_y = _axis_stats(point_rows, "err_y")
    mae_z, rmse_z = _axis_stats(point_rows, "err_z")
    report = {
        "robot_config": args.robot_config,
        "points_count": len(point_rows),
        "metrics": {
            "mae_xyz_avg_m": _mean(err_xyz_abs),
            "rmse_xyz_dist_m": _rmse(err_norm),
            "mae_x_m": mae_x,
            "mae_y_m": mae_y,
            "mae_z_m": mae_z,
            "rmse_x_m": rmse_x,
            "rmse_y_m": rmse_y,
            "rmse_z_m": rmse_z,
        },
        "points": point_rows,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "name",
        "u",
        "v",
        "depth_m",
        "pred_x",
        "pred_y",
        "pred_z",
        "measured_x",
        "measured_y",
        "measured_z",
        "err_x",
        "err_y",
        "err_z",
        "abs_err_x",
        "abs_err_y",
        "abs_err_z",
        "err_dist",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(point_rows)

    print(f"saved report: {out_json}")
    print(f"saved points: {out_csv}")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standardized pipeline: build pose dataset -> formal train -> boxed compare video."
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--run-name", default="yolo26s_pose_lab_v1")
    parser.add_argument("--model", default="yolo26s-pose.pt")
    parser.add_argument(
        "--demo-video",
        default=r"D:\labdata\discription_pdf\first_person_复杂长操作_normal_correct_001_rgb.mp4",
    )
    parser.add_argument("--demo-fps", type=float, default=10.0)
    parser.add_argument("--demo-duration-sec", type=float, default=60.0)
    parser.add_argument("--demo-conf", type=float, default=0.25)
    return parser.parse_args()


def _detect_src_root(project_root: Path) -> Dict[str, str]:
    labeling_root = project_root / "data" / "interim" / "labeling" / "frames"
    if (labeling_root / "images" / "train").exists() and (labeling_root / "labels" / "train").exists():
        return {
            "src_root": str(labeling_root).replace("\\", "/"),
            "class_yaml": "data/processed/yolo_dataset/dataset.yaml",
        }
    return {
        "src_root": "data/processed/yolo_dataset",
        "class_yaml": "data/processed/yolo_dataset/dataset.yaml",
    }


def _run(cmd: List[str]) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    pose_out_root = project_root / "data" / "processed" / "yolo_pose_dataset"
    source = _detect_src_root(project_root)

    build_cmd = [
        sys.executable,
        "scripts/build_pose_dataset.py",
        "--src-root",
        source["src_root"],
        "--class-yaml",
        source["class_yaml"],
        "--out-root",
        "data/processed/yolo_pose_dataset",
        "--schema",
        "configs/data/pose_keypoints_schema.yaml",
        "--clean",
    ]
    _run(build_cmd)

    train_cmd = [
        sys.executable,
        "scripts/train_yolo_lab.py",
        "--dataset-yaml",
        "data/processed/yolo_pose_dataset/dataset.yaml",
        "--model",
        args.model,
        "--epochs",
        str(args.epochs),
        "--imgsz",
        str(args.imgsz),
        "--batch",
        str(args.batch),
        "--device",
        str(args.device),
        "--workers",
        str(args.workers),
        "--name",
        args.run_name,
        "--demo-video",
        str(args.demo_video),
        "--demo-fps",
        str(args.demo_fps),
        "--demo-duration-sec",
        str(args.demo_duration_sec),
        "--demo-conf",
        str(args.demo_conf),
    ]
    _run(train_cmd)

    out_dir = project_root / "outputs" / "training" / args.run_name
    summary = {
        "pipeline": "pose_training",
        "source_dataset": source,
        "pose_dataset_yaml": str((pose_out_root / "dataset.yaml")).replace("\\", "/"),
        "train_run_dir": str(out_dir).replace("\\", "/"),
        "best_weight": str((out_dir / "weights" / "best.pt")).replace("\\", "/"),
        "compare_video": str((out_dir / "detection_compare_60s.mp4")).replace("\\", "/"),
    }
    summary_path = project_root / "outputs" / "reports" / f"{args.run_name}_pipeline_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


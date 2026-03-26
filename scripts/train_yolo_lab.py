from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO on lab annotation dataset.")
    parser.add_argument("--dataset-yaml", default="data/processed/yolo_dataset/dataset.yaml")
    parser.add_argument("--model", default="yolo26s-pose.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="outputs/training")
    parser.add_argument("--name", default="yolo_lab")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--demo-video", default=None, help="Optional demo RGB video for post-train visualization.")
    parser.add_argument("--baseline-model", default="yolo26s-pose.pt", help="Baseline model for compare video.")
    parser.add_argument("--demo-fps", type=float, default=10.0)
    parser.add_argument("--demo-duration-sec", type=float, default=60.0)
    parser.add_argument("--demo-conf", type=float, default=0.25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ds_yaml = Path(args.dataset_yaml)
    if not ds_yaml.exists():
        raise FileNotFoundError(f"dataset yaml not found: {ds_yaml}")

    local_cfg = Path(".ultralytics")
    local_cfg.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(local_cfg.resolve()))
    local_mpl = Path(".matplotlib")
    local_mpl.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(local_mpl.resolve()))

    try:
        from ultralytics import YOLO  # type: ignore
        from ultralytics import settings as yolo_settings  # type: ignore
    except Exception as exc:
        raise RuntimeError("ultralytics is required. pip install ultralytics") from exc

    local_runs = (Path("outputs") / "training").resolve()
    local_runs.mkdir(parents=True, exist_ok=True)
    try:
        yolo_settings.update({"runs_dir": str(local_runs).replace("\\", "/")})
    except Exception:
        pass

    model = YOLO(args.model)
    project_abs = str(Path(args.project).resolve())
    Path(project_abs).mkdir(parents=True, exist_ok=True)

    result = model.train(
        data=str(ds_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=project_abs,
        name=args.name,
        workers=args.workers,
        pretrained=True,
        verbose=True,
        exist_ok=True,
    )

    out_dir = Path(project_abs) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_run_meta.json").write_text(
        json.dumps(
            {
                "dataset_yaml": str(ds_yaml),
                "model": args.model,
                "epochs": args.epochs,
                "imgsz": args.imgsz,
                "batch": args.batch,
                "device": args.device,
                "project": args.project,
                "name": args.name,
                "result_repr": str(result),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    best_weight = out_dir / "weights" / "best.pt"
    if args.demo_video and best_weight.exists():
        compare_video = out_dir / "detection_compare_60s.mp4"
        cmd = [
            sys.executable,
            "scripts/render_detection_compare_video.py",
            "--video",
            args.demo_video,
            "--model-before",
            args.baseline_model,
            "--model-after",
            str(best_weight),
            "--conf",
            str(args.demo_conf),
            "--target-fps",
            str(args.demo_fps),
            "--duration-sec",
            str(args.demo_duration_sec),
            "--out-video",
            str(compare_video),
        ]
        subprocess.run(cmd, check=True)
        print(f"demo compare video: {compare_video}")

    print(f"training output dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

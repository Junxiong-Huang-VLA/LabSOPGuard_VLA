from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO on lab annotation dataset.")
    parser.add_argument("--dataset-yaml", default="data/processed/yolo_dataset/dataset.yaml")
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Optional dataset root override. Useful on Linux when dataset.yaml contains a Windows path.",
    )
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


def _looks_like_windows_absolute_path(value: str) -> bool:
    return len(value) >= 3 and value[1] == ":" and value[2] in ("\\", "/")


def _resolve_path_value(path_value: str, dataset_root: Path) -> Path:
    path_text = str(path_value).strip()
    if _looks_like_windows_absolute_path(path_text):
        return Path(path_text)
    path_obj = Path(path_text)
    if path_obj.is_absolute():
        return path_obj
    return (dataset_root / path_obj).resolve()


def _resolve_dataset_root(ds_yaml: Path, configured_root: object, explicit_root: str | None) -> Path:
    if explicit_root:
        override_root = Path(explicit_root).expanduser()
        if not override_root.is_absolute():
            override_root = (Path.cwd() / override_root).resolve()
        return override_root

    fallback_root = ds_yaml.parent.resolve()
    if not isinstance(configured_root, str) or not configured_root.strip():
        return fallback_root

    configured_text = configured_root.strip()
    if _looks_like_windows_absolute_path(configured_text):
        candidate = Path(configured_text)
        if os.name == "nt" and candidate.exists():
            return candidate.resolve()
        return fallback_root

    candidate = Path(configured_text).expanduser()
    if not candidate.is_absolute():
        candidate = (ds_yaml.parent / candidate).resolve()
    if candidate.exists():
        return candidate.resolve()
    return fallback_root


def _prepare_dataset_yaml(ds_yaml: Path, out_dir: Path, explicit_root: str | None) -> tuple[Path, Path]:
    config = yaml.safe_load(ds_yaml.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError(f"dataset yaml must be a mapping: {ds_yaml}")

    dataset_root = _resolve_dataset_root(ds_yaml, config.get("path"), explicit_root)
    config["path"] = str(dataset_root).replace("\\", "/")

    required_splits = ("train", "val")
    missing_entries: list[str] = []
    for split_name in required_splits:
        split_value = config.get(split_name)
        if not isinstance(split_value, str) or not split_value.strip():
            missing_entries.append(f"{split_name}=<missing>")
            continue
        split_path = _resolve_path_value(split_value, dataset_root)
        if not split_path.exists():
            missing_entries.append(f"{split_name}={split_path}")
    if missing_entries:
        raise FileNotFoundError(
            "dataset layout is incomplete after path resolution: "
            + ", ".join(missing_entries)
        )

    resolved_yaml = out_dir / "dataset_resolved.yaml"
    resolved_yaml.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return resolved_yaml, dataset_root


def main() -> int:
    args = parse_args()
    ds_yaml = Path(args.dataset_yaml).resolve()
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

    project_abs = str(Path(args.project).resolve())
    Path(project_abs).mkdir(parents=True, exist_ok=True)
    out_dir = Path(project_abs) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    resolved_ds_yaml, dataset_root = _prepare_dataset_yaml(ds_yaml, out_dir, args.dataset_root)
    print(f"dataset yaml: {resolved_ds_yaml}")
    print(f"dataset root: {dataset_root}")

    model = YOLO(args.model)

    result = model.train(
        data=str(resolved_ds_yaml),
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

    (out_dir / "train_run_meta.json").write_text(
        json.dumps(
            {
                "dataset_yaml_original": str(ds_yaml),
                "dataset_yaml_resolved": str(resolved_ds_yaml),
                "dataset_root_resolved": str(dataset_root),
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

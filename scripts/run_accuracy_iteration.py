from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one accuracy iteration: focused dataset -> train(optional) -> conf sweep -> best conf report."
    )
    parser.add_argument("--src-dataset-yaml", default="data/processed/yolo_pose_dataset_std_80_20/dataset.yaml")
    parser.add_argument("--focused-out-root", default="data/processed/yolo_pose_dataset_focus_auto")
    parser.add_argument(
        "--focus-class-multiplier",
        nargs="*",
        default=["lab_coat:2", "gloved_hand:3", "spatula:10"],
    )
    parser.add_argument("--run-name", default="yolo26s_pose_lab_v4_focus_auto")
    parser.add_argument("--base-model", default="yolo26s-pose.pt", help="Used when not skipping train.")
    parser.add_argument("--resume-weights", default="", help="When skip-train=true, evaluate this weights path.")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--conf-list", nargs="*", type=float, default=[0.25, 0.30, 0.35, 0.40, 0.45])
    parser.add_argument("--target-classes", nargs="*", default=["lab_coat", "gloved_hand", "spatula"])
    parser.add_argument("--demo-video", default="", help="Optional demo video for rendering best-conf output.")
    parser.add_argument("--clean-focused", action="store_true")
    return parser.parse_args()


def _run(cmd: List[str]) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _load_report(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _pick_best(
    reports: Dict[float, Path],
    target_classes: List[str],
) -> Dict:
    scored: List[Dict] = []
    for conf, p in reports.items():
        data = _load_report(p)
        rows = data.get("per_class", [])
        by_name = {r.get("class_name"): r for r in rows}
        avg_f1 = 0.0
        fp_sum = 0
        fn_sum = 0
        for c in target_classes:
            r = by_name.get(c, {})
            avg_f1 += float(r.get("f1", 0.0))
            fp_sum += int(r.get("fp", 0))
            fn_sum += int(r.get("fn", 0))
        avg_f1 = avg_f1 / max(1, len(target_classes))
        scored.append(
            {
                "conf": conf,
                "avg_f1_target": round(avg_f1, 6),
                "fp_target_sum": fp_sum,
                "fn_target_sum": fn_sum,
                "report_json": str(p).replace("\\", "/"),
            }
        )
    scored.sort(key=lambda x: (-x["avg_f1_target"], x["fp_target_sum"], x["fn_target_sum"]))
    return {"best": scored[0] if scored else None, "all": scored}


def main() -> int:
    args = parse_args()
    py = sys.executable
    project_root = Path(__file__).resolve().parents[1]

    focused_cmd = [
        py,
        "scripts/build_focused_pose_dataset.py",
        "--src-dataset-yaml",
        args.src_dataset_yaml,
        "--out-root",
        args.focused_out_root,
        "--focus-class-multiplier",
        *args.focus_class_multiplier,
    ]
    if args.clean_focused:
        focused_cmd.append("--clean")
    _run(focused_cmd)

    focused_yaml = Path(args.focused_out_root) / "dataset.yaml"
    if not focused_yaml.exists():
        raise FileNotFoundError(f"focused dataset yaml not found: {focused_yaml}")

    if args.skip_train:
        if not args.resume_weights:
            raise ValueError("--resume-weights is required when --skip-train is used.")
        best_weights = Path(args.resume_weights)
    else:
        train_cmd = [
            py,
            "scripts/train_yolo_lab.py",
            "--dataset-yaml",
            str(focused_yaml).replace("\\", "/"),
            "--model",
            args.base_model,
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
        ]
        _run(train_cmd)
        best_weights = project_root / "outputs" / "training" / args.run_name / "weights" / "best.pt"

    if not best_weights.exists():
        raise FileNotFoundError(f"best weights not found: {best_weights}")

    reports: Dict[float, Path] = {}
    for conf in args.conf_list:
        conf_tag = str(conf).replace(".", "p")
        out_json = project_root / "outputs" / "reports" / f"{args.run_name}_conf{conf_tag}.json"
        out_csv = project_root / "outputs" / "reports" / f"{args.run_name}_conf{conf_tag}.csv"
        eval_cmd = [
            py,
            "scripts/analyze_detection_errors.py",
            "--dataset-yaml",
            args.src_dataset_yaml,
            "--weights",
            str(best_weights).replace("\\", "/"),
            "--conf",
            str(conf),
            "--iou-thr",
            "0.5",
            "--imgsz",
            str(args.imgsz),
            "--out-json",
            str(out_json).replace("\\", "/"),
            "--out-csv",
            str(out_csv).replace("\\", "/"),
        ]
        _run(eval_cmd)
        reports[conf] = out_json

    score = _pick_best(reports, args.target_classes)
    best = score["best"]

    summary = {
        "run_name": args.run_name,
        "focused_dataset_yaml": str(focused_yaml).replace("\\", "/"),
        "weights": str(best_weights).replace("\\", "/"),
        "target_classes": args.target_classes,
        "score": score,
    }

    if best and args.demo_video:
        out_video = project_root / "outputs" / "predictions" / f"{args.run_name}_best_conf_demo.mp4"
        out_meta = project_root / "outputs" / "predictions" / f"{args.run_name}_best_conf_demo.json"
        render_cmd = [
            py,
            "scripts/render_detection_video.py",
            "--video",
            args.demo_video,
            "--weights",
            str(best_weights).replace("\\", "/"),
            "--conf",
            str(best["conf"]),
            "--imgsz",
            str(args.imgsz),
            "--target-fps",
            "10",
            "--duration-sec",
            "60",
            "--out-video",
            str(out_video).replace("\\", "/"),
            "--out-json",
            str(out_meta).replace("\\", "/"),
        ]
        _run(render_cmd)
        summary["best_demo_video"] = str(out_video).replace("\\", "/")
        summary["best_demo_meta"] = str(out_meta).replace("\\", "/")

    summary_path = project_root / "outputs" / "reports" / f"{args.run_name}_iteration_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[DONE] iteration summary:", summary_path)
    if best:
        print(
            "[BEST] conf=",
            best["conf"],
            "avg_f1_target=",
            best["avg_f1_target"],
            "fp_target_sum=",
            best["fp_target_sum"],
            "fn_target_sum=",
            best["fn_target_sum"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

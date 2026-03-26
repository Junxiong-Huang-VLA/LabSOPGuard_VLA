from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List


def _run(cmd: List[str], cwd: Path) -> None:
    print(f"[RUN] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(cwd), check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="0-to-1 pipeline: scan -> infer -> monitor -> export"
    )
    parser.add_argument("--dataset-root", default=r"D:\labdata")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--valid-only", action="store_true", default=True)

    parser.add_argument("--scan-max-frames-per-video", type=int, default=5)
    parser.add_argument("--scan-interval-sec", type=float, default=1.0)
    parser.add_argument("--scan-frames-root", default="data/interim/frames")
    parser.add_argument("--scan-manifest-csv", default="data/interim/video_manifest.csv")
    parser.add_argument("--scan-report-json", default="outputs/reports/video_scan_report.json")

    parser.add_argument("--infer-max-frames", type=int, default=120)
    parser.add_argument("--infer-target-fps", type=float, default=10.0)
    parser.add_argument("--infer-out-dir", default="outputs/predictions/batch_infer")

    parser.add_argument("--monitor-max-frames", type=int, default=120)
    parser.add_argument("--monitor-target-fps", type=float, default=10.0)
    parser.add_argument("--monitor-out-dir", default="outputs/predictions/batch_monitor")

    parser.add_argument("--export-max-frames", type=int, default=120)
    parser.add_argument("--export-target-fps", type=float, default=10.0)
    parser.add_argument("--export-out-json", default="outputs/predictions/export_summary.json")
    parser.add_argument("--export-out-csv", default="outputs/reports/export_summary.csv")
    parser.add_argument("--export-events-jsonl", default="outputs/predictions/export_events.jsonl")
    parser.add_argument("--export-events-csv", default="outputs/reports/export_events.csv")

    parser.add_argument("--skip-scan", action="store_true")
    parser.add_argument("--skip-infer", action="store_true")
    parser.add_argument("--skip-monitor", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument("--audit-batch-monitor-dir", default="outputs/predictions/batch_monitor")
    parser.add_argument("--audit-out-dir", default="outputs/reports/audit_assets")
    parser.add_argument("--audit-max-snaps-per-sample", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    python = sys.executable
    start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[INFO] 0-to-1 pipeline started at {start}")

    valid_only_flag = ["--valid-only"] if args.valid_only else []
    recursive_flag = ["--recursive"] if args.recursive else []

    if not args.skip_scan:
        _run(
            [
                python,
                "scripts/scan_and_extract_frames.py",
                "--dataset-root",
                args.dataset_root,
                *recursive_flag,
                "--max-frames-per-video",
                str(args.scan_max_frames_per_video),
                "--interval-sec",
                str(args.scan_interval_sec),
                "--manifest-csv",
                args.scan_manifest_csv,
                "--report-json",
                args.scan_report_json,
                "--frames-root",
                args.scan_frames_root,
                "--verbose",
            ],
            cwd=project_root,
        )

    if not args.skip_infer:
        _run(
            [
                python,
                "scripts/infer.py",
                *valid_only_flag,
                "--max-frames",
                str(args.infer_max_frames),
                "--target-fps",
                str(args.infer_target_fps),
                "--batch-output-dir",
                args.infer_out_dir,
            ],
            cwd=project_root,
        )

    if not args.skip_monitor:
        _run(
            [
                python,
                "scripts/run_monitor.py",
                *valid_only_flag,
                "--max-frames",
                str(args.monitor_max_frames),
                "--target-fps",
                str(args.monitor_target_fps),
                "--batch-output-dir",
                args.monitor_out_dir,
            ],
            cwd=project_root,
        )

    if not args.skip_export:
        _run(
            [
                python,
                "scripts/export_results.py",
                *valid_only_flag,
                "--max-frames",
                str(args.export_max_frames),
                "--target-fps",
                str(args.export_target_fps),
                "--out-json",
                args.export_out_json,
                "--out-csv",
                args.export_out_csv,
                "--out-events-jsonl",
                args.export_events_jsonl,
                "--out-events-csv",
                args.export_events_csv,
            ],
            cwd=project_root,
        )

    if not args.skip_audit:
        _run(
            [
                python,
                "scripts/build_audit_assets.py",
                "--batch-monitor-dir",
                args.audit_batch_monitor_dir,
                "--out-dir",
                args.audit_out_dir,
                "--max-snaps-per-sample",
                str(args.audit_max_snaps_per_sample),
            ],
            cwd=project_root,
        )

    end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[INFO] 0-to-1 pipeline finished at {end}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

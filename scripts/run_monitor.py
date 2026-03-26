from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from project_name.common.config import load_yaml
from project_name.common.logging_utils import setup_logger
from project_name.pipelines.sop_monitor_pipeline import SOPMonitorPipeline
from project_name.reporting.pdf_report import generate_compliance_report

CSV_FIELDS = [
    "sample_id",
    "camera_id",
    "frame_id",
    "timestamp",
    "class_name",
    "confidence",
    "event_type",
    "sop_step",
    "violation_flag",
    "severity_level",
]


def export_events_jsonl(path: str | Path, events: List[Dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def export_events_csv(path: str | Path, events: List[Dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for e in events:
            writer.writerow({k: e.get(k) for k in CSV_FIELDS})


def _load_offsets(path: str | None) -> dict | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"camera offsets json not found: {path}")
    return json.loads(p.read_text(encoding="utf-8"))


def _read_manifest(path: str) -> List[Dict[str, str]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _resolve_manifest_csv(manifest_csv: str | None, data_config: str) -> str:
    if manifest_csv:
        return manifest_csv
    cfg = load_yaml(data_config)
    dataset_cfg = cfg.get("dataset", {}) if isinstance(cfg, dict) else {}
    path = dataset_cfg.get("manifest_csv")
    if not path:
        raise ValueError(
            f"No --manifest-csv provided and no dataset.manifest_csv in {data_config}."
        )
    return str(path)


def _is_row_valid(row: Dict[str, str]) -> bool:
    valid_status = str(row.get("valid_status", "")).strip().lower()
    if valid_status:
        return valid_status == "valid"
    return str(row.get("pair_status", "")).strip().lower() == "paired"


def _run_one(
    pipeline: SOPMonitorPipeline,
    video: str,
    session_id: str,
    camera_id: str,
    max_frames: int,
    target_fps: float,
    offsets: dict | None,
) -> Dict[str, Any]:
    return pipeline.run(
        video_source=video,
        max_frames=max_frames,
        target_fps=target_fps,
        sample_id=session_id,
        camera_id=camera_id,
        camera_offsets_ms=offsets,
    )


def _write_single_outputs(result: Dict[str, Any], report_cfg: Dict[str, Any]) -> str:
    out_json = Path("outputs/predictions/runtime_result.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    export_events_jsonl("outputs/predictions/runtime_events.jsonl", result["events"])
    export_events_csv("outputs/reports/runtime_events.csv", result["events"])

    report_input = dict(result["report_input"])
    report_input["title"] = report_cfg.get("report", {}).get(
        "title", report_input.get("title", "Lab SOP Compliance Report")
    )
    return generate_compliance_report(report_input, "outputs/reports/runtime_report.pdf")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SOP monitor in single or batch mode")
    parser.add_argument("--video", default=None, help="Single video path")
    parser.add_argument("--manifest-csv", default=None, help="Batch manifest csv path")
    parser.add_argument("--valid-only", action="store_true", help="Batch mode: only valid_status=valid")
    parser.add_argument("--data-config", default="configs/data/dataset.yaml")
    parser.add_argument("--rules", default="configs/sop/rules.yaml")
    parser.add_argument("--report-config", default="configs/report/report.yaml")
    parser.add_argument("--session-id", default="session_runtime")
    parser.add_argument("--camera-id", default="cam0")
    parser.add_argument("--camera-offsets-json", default=None)
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--target-fps", type=float, default=10.0)
    parser.add_argument("--batch-output-dir", default="outputs/predictions/batch_monitor")
    args = parser.parse_args()

    if not args.video and not args.manifest_csv:
        args.manifest_csv = _resolve_manifest_csv(args.manifest_csv, args.data_config)

    logger = setup_logger("run_monitor")
    rules = load_yaml(args.rules)
    report_cfg = load_yaml(args.report_config)
    pipeline = SOPMonitorPipeline(rules=rules)
    offsets = _load_offsets(args.camera_offsets_json)

    if args.video:
        result = _run_one(
            pipeline=pipeline,
            video=args.video,
            session_id=args.session_id,
            camera_id=args.camera_id,
            max_frames=args.max_frames,
            target_fps=args.target_fps,
            offsets=offsets,
        )
        pdf_path = _write_single_outputs(result, report_cfg)
        logger.info("runtime result: outputs/predictions/runtime_result.json")
        logger.info("events jsonl: outputs/predictions/runtime_events.jsonl")
        logger.info("events csv: outputs/reports/runtime_events.csv")
        logger.info("report path: %s", pdf_path)
        return 0

    rows = _read_manifest(args.manifest_csv)
    if args.valid_only:
        rows = [r for r in rows if _is_row_valid(r)]

    batch_dir = Path(args.batch_output_dir)
    batch_dir.mkdir(parents=True, exist_ok=True)
    all_events: List[Dict[str, Any]] = []
    summary: List[Dict[str, Any]] = []

    for idx, row in enumerate(rows):
        sample_id = row.get("sample_id") or f"sample_{idx:04d}"
        video = row.get("rgb_path") or row.get("video_path")
        if not video:
            logger.warning("skip sample without video path: %s", sample_id)
            continue

        result = _run_one(
            pipeline=pipeline,
            video=video,
            session_id=sample_id,
            camera_id=args.camera_id,
            max_frames=args.max_frames,
            target_fps=args.target_fps,
            offsets=offsets,
        )

        sample_json = batch_dir / f"{sample_id}.json"
        sample_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        export_events_jsonl(batch_dir / f"{sample_id}.events.jsonl", result["events"])
        export_events_csv(batch_dir / f"{sample_id}.events.csv", result["events"])

        report_input = dict(result["report_input"])
        report_input["title"] = report_cfg.get("report", {}).get(
            "title", report_input.get("title", "Lab SOP Compliance Report")
        )
        report_file = generate_compliance_report(report_input, str(batch_dir / f"{sample_id}.report.pdf"))

        all_events.extend(result["events"])
        summary.append(
            {
                "sample_id": sample_id,
                "video": video,
                "events": len(result["events"]),
                "violations": len(result.get("violations", [])),
                "result_json": str(sample_json).replace("\\", "/"),
                "report": str(report_file).replace("\\", "/"),
            }
        )
        logger.info("batch [%d/%d] done: %s", idx + 1, len(rows), sample_id)

    summary_path = batch_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    export_events_jsonl(batch_dir / "all_events.jsonl", all_events)
    export_events_csv(batch_dir / "all_events.csv", all_events)

    logger.info("saved summary: %s", summary_path)
    logger.info("saved merged events: %s", batch_dir / "all_events.jsonl")
    logger.info("saved merged events csv: %s", batch_dir / "all_events.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

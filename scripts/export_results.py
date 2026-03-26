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


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch monitor export for manifest rows")
    parser.add_argument("--manifest-csv", default=None)
    parser.add_argument("--data-config", default="configs/data/dataset.yaml")
    parser.add_argument("--valid-only", action="store_true")
    parser.add_argument("--rules", default="configs/sop/rules.yaml")
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--target-fps", type=float, default=10.0)
    parser.add_argument("--out-json", default="outputs/predictions/export_summary.json")
    parser.add_argument("--out-csv", default="outputs/reports/export_summary.csv")
    parser.add_argument("--out-events-jsonl", default="outputs/predictions/export_events.jsonl")
    parser.add_argument("--out-events-csv", default="outputs/reports/export_events.csv")
    args = parser.parse_args()

    manifest_csv = _resolve_manifest_csv(args.manifest_csv, args.data_config)
    rows = _read_manifest(manifest_csv)
    if args.valid_only:
        rows = [r for r in rows if _is_row_valid(r)]

    rules = load_yaml(args.rules)
    pipeline = SOPMonitorPipeline(rules=rules)
    logger = setup_logger("export_results")

    outputs: List[Dict[str, Any]] = []
    all_events: List[Dict[str, Any]] = []
    summary: List[Dict[str, Any]] = []

    for idx, row in enumerate(rows):
        sample_id = row.get("sample_id") or f"sample_{idx:04d}"
        video_path = row.get("rgb_path") or row.get("video_path")
        if not video_path:
            logger.warning("skip sample without video path: %s", sample_id)
            continue

        result = pipeline.run(
            video_source=video_path,
            max_frames=args.max_frames,
            target_fps=args.target_fps,
            sample_id=sample_id,
            camera_id=str(row.get("camera_id", "cam0")),
        )
        outputs.append(
            {
                "sample_id": sample_id,
                "video_path": video_path,
                "status": result["status"],
                "violations": result["violations"],
            }
        )
        all_events.extend(result["events"])
        summary.append(
            {
                "sample_id": sample_id,
                "video_path": video_path,
                "compliance_ratio": result["status"]["compliance_ratio"],
                "violation_count": len(result["violations"]),
                "event_count": len(result["events"]),
            }
        )

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["sample_id", "video_path", "compliance_ratio", "violation_count", "event_count"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summary:
            writer.writerow({k: row.get(k) for k in fields})

    events_jsonl = Path(args.out_events_jsonl)
    events_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with events_jsonl.open("w", encoding="utf-8") as f:
        for e in all_events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    events_csv = Path(args.out_events_csv)
    events_csv.parent.mkdir(parents=True, exist_ok=True)
    if all_events:
        fields = sorted({k for e in all_events for k in e.keys()})
        with events_csv.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in all_events:
                writer.writerow(row)
    else:
        with events_csv.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["no_events"])

    logger.info("exported sessions=%d", len(outputs))
    logger.info("summary json: %s", out_json)
    logger.info("summary csv: %s", out_csv)
    logger.info("events jsonl: %s", events_jsonl)
    logger.info("events csv: %s", events_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

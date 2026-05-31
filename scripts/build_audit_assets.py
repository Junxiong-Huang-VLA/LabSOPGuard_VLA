from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_target_events(events: List[Dict[str, Any]], max_per_sample: int) -> List[Dict[str, Any]]:
    violations = [e for e in events if bool(e.get("violation_flag")) or str(e.get("event_type")) == "violation"]
    if violations:
        base = violations
    else:
        base = [e for e in events if str(e.get("event_type")) == "detection"]
    seen = set()
    out: List[Dict[str, Any]] = []
    for e in base:
        frame_id = int(e.get("frame_id", -1))
        if frame_id < 0 or frame_id in seen:
            continue
        seen.add(frame_id)
        out.append(e)
        if len(out) >= max_per_sample:
            break
    return out


def _extract_frame(video_path: Path, frame_id: int) -> Tuple[bool, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, None
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_id))
    ok, frame = cap.read()
    cap.release()
    return ok, frame


def _safe_write_jpg(path: Path, frame) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), frame)
    if ok:
        return True
    ok2, buf = cv2.imencode(".jpg", frame)
    if not ok2:
        return False
    try:
        buf.tofile(str(path))
        return True
    except Exception:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build audit assets from batch monitor outputs")
    parser.add_argument(
        "--batch-monitor-dir",
        default="outputs/predictions/batch_monitor",
        help="Directory containing summary.json and per-sample results.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/reports/audit_assets",
        help="Output directory for audit timeline and snapshots.",
    )
    parser.add_argument("--max-snaps-per-sample", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    batch_dir = Path(args.batch_monitor_dir)
    summary_path = batch_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found under {batch_dir}")

    summary = _read_json(summary_path)
    out_dir = Path(args.out_dir)
    snap_dir = out_dir / "snapshots"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for item in summary:
        sample_id = item.get("sample_id", "unknown")
        video = item.get("video")
        result_json_rel = item.get("result_json")
        if not video or not result_json_rel:
            continue

        result_json = Path(result_json_rel)
        if not result_json.is_absolute():
            result_json = Path.cwd() / result_json_rel
        if not result_json.exists():
            continue

        result = _read_json(result_json)
        events = result.get("events", [])
        target_events = _collect_target_events(events, args.max_snaps_per_sample)
        video_path = Path(video)

        for idx, e in enumerate(target_events):
            frame_id = int(e.get("frame_id", -1))
            ok, frame = _extract_frame(video_path, frame_id)
            if not ok:
                continue

            event_type = str(e.get("event_type", "event"))
            severity = str(e.get("severity_level", "none"))
            snap_name = f"{sample_id}__f{frame_id:06d}__{event_type}__{severity}__{idx:02d}.jpg"
            snap_path = snap_dir / sample_id / snap_name
            if not _safe_write_jpg(snap_path, frame):
                continue

            rows.append(
                {
                    "sample_id": sample_id,
                    "video": str(video_path),
                    "frame_id": frame_id,
                    "timestamp": e.get("timestamp"),
                    "timestamp_sec": e.get("timestamp_sec"),
                    "event_type": event_type,
                    "sop_step": e.get("sop_step"),
                    "violation_flag": bool(e.get("violation_flag")),
                    "severity_level": severity,
                    "class_name": e.get("class_name"),
                    "confidence": e.get("confidence"),
                    "snapshot_path": str(snap_path).replace("\\", "/"),
                }
            )

    timeline_json = out_dir / "audit_timeline.json"
    timeline_csv = out_dir / "audit_timeline.csv"
    timeline_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    fields = [
        "sample_id",
        "video",
        "frame_id",
        "timestamp",
        "timestamp_sec",
        "event_type",
        "sop_step",
        "violation_flag",
        "severity_level",
        "class_name",
        "confidence",
        "snapshot_path",
    ]
    with timeline_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"audit timeline json: {timeline_json}")
    print(f"audit timeline csv: {timeline_csv}")
    print(f"snapshot count: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

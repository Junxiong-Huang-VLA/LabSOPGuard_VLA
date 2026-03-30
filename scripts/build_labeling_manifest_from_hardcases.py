from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build labeling_manifest.csv template from hardcase manifest."
    )
    parser.add_argument(
        "--hardcase-csv",
        default="data/interim/hardcases/ppe_missing_manifest.csv",
        help="Input hardcase manifest csv path.",
    )
    parser.add_argument(
        "--out-csv",
        default="data/interim/labeling/ppe_hardcases_labeling_manifest.csv",
        help="Output labeling manifest csv path.",
    )
    parser.add_argument(
        "--dedup-frame-path",
        action="store_true",
        help="When enabled, drop duplicated frame_path rows.",
    )
    return parser.parse_args()


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _to_float_str(value: str) -> str:
    try:
        return f"{float(value):.3f}"
    except Exception:
        return ""


def main() -> int:
    args = parse_args()
    in_csv = Path(args.hardcase_csv)
    out_csv = Path(args.out_csv)

    if not in_csv.exists():
        raise FileNotFoundError(f"hardcase manifest not found: {in_csv}")

    rows = _read_rows(in_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    out_rows: List[Dict[str, str]] = []
    for row in rows:
        frame_path = str(row.get("image_path", "")).replace("\\", "/").strip()
        if not frame_path:
            continue
        if args.dedup_frame_path and frame_path in seen:
            continue
        seen.add(frame_path)

        out_rows.append(
            {
                "sample_id": str(row.get("sample_id", "")).strip(),
                "video_path": str(row.get("video_path", "")).strip(),
                "frame_id": str(row.get("frame_id", "")).strip(),
                "timestamp_sec": _to_float_str(str(row.get("timestamp_sec", "")).strip()),
                "frame_path": frame_path,
                "annotation_status": "pending",
                "labels_json": "[]",
                "source_rule_id": str(row.get("rule_id", "")).strip(),
                "source_center_frame_id": str(row.get("center_frame_id", "")).strip(),
                "source_is_center": str(row.get("is_center", "")).strip(),
                "source_missing_ppe_items": str(row.get("missing_ppe_items", "")).strip(),
            }
        )

    fields = [
        "sample_id",
        "video_path",
        "frame_id",
        "timestamp_sec",
        "frame_path",
        "annotation_status",
        "labels_json",
        "source_rule_id",
        "source_center_frame_id",
        "source_is_center",
        "source_missing_ppe_items",
    ]
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)

    summary = {
        "source_hardcase_csv": str(in_csv).replace("\\", "/"),
        "output_labeling_csv": str(out_csv).replace("\\", "/"),
        "total_input_rows": len(rows),
        "total_output_rows": len(out_rows),
        "dedup_frame_path": bool(args.dedup_frame_path),
    }
    summary_path = out_csv.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"labeling manifest: {out_csv}")
    print(f"summary json: {summary_path}")
    print(f"rows: {len(out_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

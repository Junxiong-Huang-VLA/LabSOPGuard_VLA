from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate labeling manifest before YOLO conversion.")
    parser.add_argument(
        "--manifest-csv",
        default="data/interim/labeling/ppe_hardcases_labeling_manifest.csv",
        help="Labeling manifest csv path.",
    )
    parser.add_argument(
        "--class-schema",
        default="configs/data/class_schema.yaml",
        help="Class schema yaml path.",
    )
    parser.add_argument(
        "--status-filter",
        default="done,verified",
        help="Comma-separated statuses considered as labeled.",
    )
    parser.add_argument(
        "--out-json",
        default="outputs/reports/labeling_manifest_validation.json",
        help="Output validation report json.",
    )
    return parser.parse_args()


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _load_class_names(path: Path) -> set[str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    classes = data.get("classes", [])
    return {str(c.get("name", "")).strip() for c in classes if str(c.get("name", "")).strip()}


def _validate_bbox(bbox) -> Tuple[bool, str]:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False, "bbox must be list[4]"
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except Exception:
        return False, "bbox values must be numeric"
    if x2 <= x1 or y2 <= y1:
        return False, "bbox must satisfy x2>x1 and y2>y1"
    return True, ""


def main() -> int:
    args = parse_args()
    manifest_csv = Path(args.manifest_csv)
    schema_path = Path(args.class_schema)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    if not manifest_csv.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_csv}")
    if not schema_path.exists():
        raise FileNotFoundError(f"class schema not found: {schema_path}")

    rows = _read_rows(manifest_csv)
    class_names = _load_class_names(schema_path)
    valid_status = {s.strip().lower() for s in args.status_filter.split(",") if s.strip()}

    status_counter: Counter[str] = Counter()
    problems: List[Dict[str, str]] = []
    labeled_rows = 0
    valid_labeled_rows = 0
    total_boxes = 0
    valid_boxes = 0

    for idx, row in enumerate(rows, 1):
        status = str(row.get("annotation_status", "")).strip().lower()
        status_counter[status or "<empty>"] += 1
        if status not in valid_status:
            continue
        labeled_rows += 1

        labels_text = row.get("labels_json", "[]") or "[]"
        try:
            labels = json.loads(labels_text)
        except Exception:
            problems.append(
                {
                    "row": str(idx),
                    "sample_id": str(row.get("sample_id", "")),
                    "frame_id": str(row.get("frame_id", "")),
                    "issue": "labels_json is not valid json",
                }
            )
            continue

        if not isinstance(labels, list):
            problems.append(
                {
                    "row": str(idx),
                    "sample_id": str(row.get("sample_id", "")),
                    "frame_id": str(row.get("frame_id", "")),
                    "issue": "labels_json must be a list",
                }
            )
            continue

        row_ok = True
        for ann_idx, ann in enumerate(labels):
            total_boxes += 1
            if not isinstance(ann, dict):
                row_ok = False
                problems.append(
                    {
                        "row": str(idx),
                        "sample_id": str(row.get("sample_id", "")),
                        "frame_id": str(row.get("frame_id", "")),
                        "issue": f"annotation[{ann_idx}] must be object",
                    }
                )
                continue
            class_name = str(ann.get("class_name", "")).strip()
            if class_name not in class_names:
                row_ok = False
                problems.append(
                    {
                        "row": str(idx),
                        "sample_id": str(row.get("sample_id", "")),
                        "frame_id": str(row.get("frame_id", "")),
                        "issue": f"annotation[{ann_idx}] unknown class_name: {class_name}",
                    }
                )
                continue
            ok_bbox, msg = _validate_bbox(ann.get("bbox"))
            if not ok_bbox:
                row_ok = False
                problems.append(
                    {
                        "row": str(idx),
                        "sample_id": str(row.get("sample_id", "")),
                        "frame_id": str(row.get("frame_id", "")),
                        "issue": f"annotation[{ann_idx}] invalid bbox: {msg}",
                    }
                )
                continue
            valid_boxes += 1
        if row_ok:
            valid_labeled_rows += 1

    report = {
        "manifest_csv": str(manifest_csv).replace("\\", "/"),
        "class_schema": str(schema_path).replace("\\", "/"),
        "total_rows": len(rows),
        "status_counts": dict(status_counter),
        "labeled_rows": labeled_rows,
        "valid_labeled_rows": valid_labeled_rows,
        "total_boxes_in_labeled_rows": total_boxes,
        "valid_boxes": valid_boxes,
        "problem_count": len(problems),
        "problems_preview": problems[:100],
        "ready_for_conversion": labeled_rows > 0 and len(problems) == 0,
    }
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

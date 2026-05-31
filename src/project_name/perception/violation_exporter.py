from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


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


def export_events_jsonl(path: str | Path, events: Iterable[Dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def export_events_csv(path: str | Path, events: Iterable[Dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for e in events:
            row = {k: e.get(k) for k in CSV_FIELDS}
            writer.writerow(row)

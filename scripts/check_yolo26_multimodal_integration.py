#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

from multimodal_eval_common import PROJECT_ROOT, ensure_reports_dir, read_json


def check(exp_id: str, run_demo: bool = False) -> dict:
    if run_demo:
        subprocess.check_call(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "demo_multisource_run.py"),
                "--exp-id",
                exp_id,
            ],
            cwd=str(PROJECT_ROOT),
        )

    exp_dir = PROJECT_ROOT / "outputs" / "experiments" / exp_id
    experiment = read_json(exp_dir / "experiment.json", {})
    preprocessing = read_json(exp_dir / "preprocessing.json", {})
    material_stream = read_json(exp_dir / "material_stream.json", [])
    physical_events = read_json(exp_dir / "physical_events.json", [])
    index_path = exp_dir / "material_index.sqlite"

    detected_items = [item for item in material_stream if item.get("detected_objects")]
    object_label_items = [item for item in material_stream if item.get("object_labels")]
    detector_labels = sorted({label for item in material_stream for label in (item.get("object_labels") or [])})
    physical_with_detector_metadata = [
        event for event in physical_events if (event.get("metadata") or {}).get("material_item_id")
    ]
    sqlite_hits = 0
    sqlite_detector_label_hits = []
    if index_path.exists():
        conn = sqlite3.connect(str(index_path))
        try:
            sqlite_hits = conn.execute("SELECT COUNT(*) FROM material_items").fetchone()[0]
            sqlite_detector_label_hits = conn.execute(
                "SELECT item_id, object_labels_json FROM material_items WHERE object_labels_json LIKE ? LIMIT 10",
                ("%reagent_bottle%",),
            ).fetchall()
        finally:
            conn.close()

    result = {
        "experiment_id": exp_id,
        "weights_path_configured": bool((experiment.get("metadata") or {}).get("detector_status") or detector_labels),
        "detector_adapter_exists": (PROJECT_ROOT / "src" / "labsopguard" / "detectors.py").exists(),
        "material_stream_count": len(material_stream),
        "items_with_detected_objects": len(detected_items),
        "items_with_object_labels": len(object_label_items),
        "detector_labels": detector_labels,
        "physical_event_count": len(physical_events),
        "physical_events_linked_to_material_items": len(physical_with_detector_metadata),
        "detected_changes_count": len(preprocessing.get("detected_changes", []) or []),
        "material_index_exists": index_path.exists(),
        "material_index_row_count": sqlite_hits,
        "material_index_reagent_bottle_hits": len(sqlite_detector_label_hits),
        "integration_status": "integrated"
        if detected_items and physical_with_detector_metadata and sqlite_detector_label_hits
        else "partial_or_missing",
    }
    out = ensure_reports_dir() / "yolo26_multimodal_integration_check.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Check YOLO26 detector integration into material/event/search chain.")
    parser.add_argument("--exp-id", default="final_acceptance_e2e")
    parser.add_argument("--run-demo", action="store_true")
    args = parser.parse_args()
    result = check(args.exp_id, run_demo=args.run_demo)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["integration_status"] == "integrated" else 2


if __name__ == "__main__":
    raise SystemExit(main())

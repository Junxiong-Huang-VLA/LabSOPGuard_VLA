#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate(exp_id: str) -> dict:
    exp_dir = PROJECT_ROOT / "outputs" / "experiments" / exp_id
    paths = {
        "experiment_json": exp_dir / "experiment.json",
        "preprocessing_json": exp_dir / "preprocessing.json",
        "material_stream_json": exp_dir / "material_stream.json",
        "physical_events_json": exp_dir / "physical_events.json",
        "material_index_sqlite": exp_dir / "material_index.sqlite",
    }
    checks = {}
    for name, path in paths.items():
        checks[name] = {"path": str(path), "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else 0}

    experiment = _load_json(paths["experiment_json"]) if paths["experiment_json"].exists() else {}
    preprocessing = _load_json(paths["preprocessing_json"]) if paths["preprocessing_json"].exists() else {}
    material_stream = _load_json(paths["material_stream_json"]) if paths["material_stream_json"].exists() else []
    physical_events = _load_json(paths["physical_events_json"]) if paths["physical_events_json"].exists() else []

    timestamp_ok = any("timestamp_sec" in item and "local_timestamp_sec" in item for item in material_stream)
    context_link_ok = any(item.get("linked_context_event_ids") or item.get("transcript_segment") for item in material_stream)
    key_frame_ok = len(preprocessing.get("key_frames", [])) > 0
    key_clip_ok = len(preprocessing.get("key_clips", [])) > 0
    detected_changes_ok = len(preprocessing.get("detected_changes", [])) > 0
    video_inputs_ok = len(experiment.get("video_inputs", [])) >= 2 or len(preprocessing.get("video_streams", [])) >= 2

    sqlite_count = 0
    sqlite_query_ok = False
    if paths["material_index_sqlite"].exists():
        conn = sqlite3.connect(str(paths["material_index_sqlite"]))
        try:
            sqlite_count = int(conn.execute("SELECT COUNT(*) FROM material_items").fetchone()[0])
            sqlite_query_ok = sqlite_count > 0
        finally:
            conn.close()

    result = {
        "experiment_id": exp_id,
        "output_dir": str(exp_dir),
        "file_checks": checks,
        "video_inputs_ok": video_inputs_ok,
        "timestamp_and_local_timestamp_ok": timestamp_ok,
        "context_backlink_ok": context_link_ok,
        "key_frame_ok": key_frame_ok,
        "key_clip_ok": key_clip_ok,
        "physical_events_count": len(physical_events),
        "detected_changes_ok": detected_changes_ok,
        "material_index_row_count": sqlite_count,
        "material_index_query_ok": sqlite_query_ok,
        "material_stream_count": len(material_stream),
        "passed": all(
            [
                all(item["exists"] and item["size_bytes"] > 0 for item in checks.values()),
                video_inputs_ok,
                timestamp_ok,
                context_link_ok,
                key_frame_ok or key_clip_ok,
                sqlite_query_ok,
            ]
        ),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the generated end-to-end acceptance output.")
    parser.add_argument("--exp-id", default="final_acceptance_e2e")
    parser.add_argument("--run-demo", action="store_true", help="Run demo_multisource_run.py before validation.")
    args = parser.parse_args()

    if args.run_demo:
        subprocess.check_call([sys.executable, str(PROJECT_ROOT / "scripts" / "demo_multisource_run.py"), "--exp-id", args.exp_id])

    result = validate(args.exp_id)
    out_path = PROJECT_ROOT / "outputs" / "experiments" / args.exp_id / "pipeline_e2e_check.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

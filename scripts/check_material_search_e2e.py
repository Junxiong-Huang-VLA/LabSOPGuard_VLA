#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from multimodal_eval_common import PROJECT_ROOT, compact, ensure_reports_dir, read_json, write_matrix_rows

from labsopguard.retrieval import MaterialQuery, MaterialRetrievalIndex


def top3(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    compacted = []
    for item in items[:3]:
        compacted.append(
            {
                "item_id": item.get("item_id"),
                "timestamp_sec": item.get("timestamp_sec"),
                "camera_id": item.get("camera_id"),
                "object_labels": item.get("object_labels"),
                "event_types": item.get("event_types"),
                "clip_exists": item.get("clip_exists"),
                "clip_file_path": item.get("clip_file_path"),
                "embedding_score": item.get("embedding_score"),
            }
        )
    return compacted


def run_queries(exp_id: str, matrix_path: Path) -> Dict[str, Any]:
    exp_dir = PROJECT_ROOT / "outputs" / "experiments" / exp_id
    index = MaterialRetrievalIndex(exp_dir / "material_index.sqlite")
    preprocessing = read_json(exp_dir / "preprocessing.json", {})
    rows = []
    checks = []
    queries = [
        ("objects", MaterialQuery(objects=["reagent_bottle"], limit=20), "reagent_bottle"),
        ("objects", MaterialQuery(objects=["gloved_hand"], limit=20), "gloved_hand"),
        ("actions", MaterialQuery(actions=["transfer"], limit=20), "action/transfer text"),
        ("time_range", MaterialQuery(start_time_sec=0.0, end_time_sec=1.0, limit=20), "early window"),
        ("camera_id", MaterialQuery(camera_id="cam_local", limit=20), "cam_local"),
        ("camera_id", MaterialQuery(camera_id="cam_rtsp_sim", limit=20), "cam_rtsp_sim"),
        ("clip_exists", MaterialQuery(clip_exists=True, limit=20), "materialized clips"),
        ("text", MaterialQuery(text="reagent", limit=20), "reagent text"),
        ("embedding_text", MaterialQuery(embedding_text="operator transfers reagent into sample bottle", limit=20), "semantic reagent transfer"),
        ("transcript", MaterialQuery(text="operator", limit=20), "transcript operator"),
    ]
    try:
        for query_type, query, expected in queries:
            items = index.query(query)
            broken = [
                item for item in items
                if item.get("clip_file_path") and not Path(str(item.get("clip_file_path"))).exists()
            ]
            pass_fail = "pass" if items and not broken else "fail"
            check = {
                "query_type": query_type,
                "query_params": query.__dict__,
                "total_hits": len(items),
                "top_3_results": top3(items),
                "expected_hit": expected,
                "pass_fail": pass_fail,
                "broken_clip_refs": len(broken),
                "embedding_mode": index.embedding_provider.mode if query.embedding_text else None,
            }
            checks.append(check)
            rows.append(
                {
                    "task_type": f"material_search_{query_type}",
                    "sample_id": f"search_{len(rows) + 1:03d}",
                    "model_name": index.embedding_provider.mode if query.embedding_text else "sqlite_fts_filters",
                    "input_path": str(exp_dir / "material_index.sqlite"),
                    "expected": expected,
                    "actual": {"total_hits": len(items), "top_3": top3(items)},
                    "pass_fail": pass_fail,
                    "response_time_ms": 0,
                    "notes": f"broken_clip_refs={len(broken)}",
                }
            )

        event_types = sorted({event.get("event_type") for event in preprocessing.get("detected_changes", []) or [] if event.get("event_type")})
        event_check = {
            "query_type": "event_types_association",
            "query_params": {"source": "preprocessing.detected_changes"},
            "total_hits": len(event_types),
            "top_3_results": event_types[:3],
            "expected_hit": "non-empty detected_changes event types",
            "pass_fail": "pass" if event_types else "fail",
            "broken_clip_refs": 0,
        }
        checks.append(event_check)
        rows.append(
            {
                "task_type": "material_search_event_types",
                "sample_id": f"search_{len(rows) + 1:03d}",
                "model_name": "preprocessing_index",
                "input_path": str(exp_dir / "preprocessing.json"),
                "expected": "non-empty event_types",
                "actual": event_types,
                "pass_fail": event_check["pass_fail"],
                "response_time_ms": 0,
                "notes": "event types linked from detected_changes",
            }
        )

        clip_refs = preprocessing.get("key_clips", []) or []
        valid_clips = [clip for clip in clip_refs if clip.get("file_path") and Path(str(clip.get("file_path"))).exists()]
        clip_check = {
            "query_type": "key_frame_clip_references",
            "query_params": {"source": "preprocessing.key_clips"},
            "total_hits": len(valid_clips),
            "top_3_results": valid_clips[:3],
            "expected_hit": "valid clip files",
            "pass_fail": "pass" if valid_clips else "fail",
            "broken_clip_refs": len(clip_refs) - len(valid_clips),
        }
        checks.append(clip_check)
        rows.append(
            {
                "task_type": "material_search_clip_replay",
                "sample_id": f"search_{len(rows) + 1:03d}",
                "model_name": "filesystem_reference_check",
                "input_path": str(exp_dir / "preprocessing.json"),
                "expected": "valid clip files",
                "actual": {"valid_clips": len(valid_clips), "total_clips": len(clip_refs)},
                "pass_fail": clip_check["pass_fail"],
                "response_time_ms": 0,
                "notes": f"broken_clip_refs={clip_check['broken_clip_refs']}",
            }
        )
    finally:
        index.close()

    write_matrix_rows(matrix_path, rows, append=matrix_path.exists())
    summary = {
        "experiment_id": exp_id,
        "query_count": len(checks),
        "pass_count": sum(1 for item in checks if item["pass_fail"] == "pass"),
        "checks": checks,
    }
    out = ensure_reports_dir() / "material_search_e2e_check.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run material search and evidence replay checks.")
    parser.add_argument("--exp-id", default="final_acceptance_e2e")
    parser.add_argument("--matrix", default=str(ensure_reports_dir() / "multimodal_eval_matrix.csv"))
    args = parser.parse_args()
    summary = run_queries(args.exp_id, Path(args.matrix))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["pass_count"] >= 8 else 2


if __name__ == "__main__":
    raise SystemExit(main())

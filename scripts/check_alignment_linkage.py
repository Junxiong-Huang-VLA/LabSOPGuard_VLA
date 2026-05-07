#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_eval_common import PROJECT_ROOT, ensure_reports_dir, read_json


def ratio(count: int, total: int) -> float:
    return round(count / total, 4) if total else 0.0


def check(exp_id: str) -> dict:
    exp_dir = PROJECT_ROOT / "outputs" / "experiments" / exp_id
    experiment = read_json(exp_dir / "experiment.json", {})
    preprocessing = read_json(exp_dir / "preprocessing.json", {})
    material_stream = read_json(exp_dir / "material_stream.json", [])
    timeline = read_json(exp_dir / "timeline.json", {})
    total = len(material_stream)
    timestamp_count = sum(1 for item in material_stream if item.get("timestamp_sec") is not None)
    local_count = sum(1 for item in material_stream if item.get("local_timestamp_sec") is not None)
    linked_count = sum(1 for item in material_stream if item.get("linked_context_event_ids"))
    transcript_count = sum(1 for item in material_stream if item.get("transcript_segment"))
    key_frame_count = sum(1 for item in material_stream if item.get("is_key_frame"))
    clip_count = sum(1 for item in material_stream if item.get("clip_id"))
    context_inputs = experiment.get("context_inputs", []) or []
    exp_context_events = experiment.get("context_events", []) or []
    timeline_context_events = timeline.get("context_events", []) or []
    result = {
        "experiment_id": exp_id,
        "output_dir": str(exp_dir),
        "material_item_total": total,
        "timestamp_sec_ratio": ratio(timestamp_count, total),
        "local_timestamp_sec_ratio": ratio(local_count, total),
        "linked_context_event_ids_ratio": ratio(linked_count, total),
        "transcript_segment_ratio": ratio(transcript_count, total),
        "key_frame_coverage": ratio(key_frame_count, total),
        "key_clip_coverage": ratio(clip_count, total),
        "context_input_count": len(context_inputs),
        "experiment_context_event_count": len(exp_context_events),
        "timeline_context_event_count": len(timeline_context_events),
        "preprocessing_aligned_text_count": len(preprocessing.get("aligned_text", []) or []),
        "preprocessing_key_frame_count": len(preprocessing.get("key_frames", []) or []),
        "preprocessing_key_clip_count": len(preprocessing.get("key_clips", []) or []),
        "alignment_summary": preprocessing.get("alignment_summary", {}),
        "breakpoint_analysis": {
            "transcript_exists": any((item.get("kind") == "transcript" or item.get("source_type") == "asr") for item in context_inputs),
            "context_event_generated": bool(exp_context_events or timeline_context_events),
            "material_stream_backlinked": linked_count > 0 or transcript_count > 0,
            "code_landed_not_docs_only": total > 0 and (linked_count > 0 or transcript_count > 0),
        },
    }
    out = ensure_reports_dir() / "alignment_linkage_check.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Check time alignment and transcript/ContextEvent backlinking.")
    parser.add_argument("--exp-id", default="final_acceptance_e2e")
    args = parser.parse_args()
    result = check(args.exp_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    ok = (
        result["timestamp_sec_ratio"] == 1.0
        and result["local_timestamp_sec_ratio"] == 1.0
        and result["breakpoint_analysis"]["material_stream_backlinked"]
    )
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

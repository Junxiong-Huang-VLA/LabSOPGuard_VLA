from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .schemas import read_jsonl, write_jsonl


SCHEMA_VERSION = "key_action_micro_coverage_backfill.v1"

_NON_OBJECT_LABELS = {
    "",
    "background",
    "gloved_hand",
    "hand",
    "lab_coat",
    "person",
}

_OBJECT_PRIORITY = {
    "pipette": 90,
    "pipette_tip": 88,
    "balance": 84,
    "spatula": 82,
    "beaker": 80,
    "container": 78,
    "reagent_bottle": 76,
    "sample_bottle": 76,
    "paper": 60,
}


def backfill_micro_coverage(
    session_dir: str | Path,
    *,
    output_report: str | Path | None = None,
) -> dict[str, Any]:
    session = Path(session_dir)
    metadata = session / "metadata"
    segment_path = metadata / "key_action_segments.jsonl"
    micro_path = metadata / "micro_segments.jsonl"
    if not segment_path.exists():
        raise FileNotFoundError(f"Missing key action segments: {segment_path}")

    segments = read_jsonl(segment_path)
    micros = read_jsonl(micro_path) if micro_path.exists() else []
    existing_parent_ids = {str(row.get("parent_segment_id") or row.get("segment_id") or "") for row in micros}
    existing_micro_ids = {str(row.get("micro_segment_id") or "") for row in micros}

    added: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for segment in segments:
        segment_id = str(segment.get("segment_id") or "")
        if not segment_id:
            skipped.append({"segment_id": "", "reason": "missing_segment_id"})
            continue
        if segment_id in existing_parent_ids:
            skipped.append({"segment_id": segment_id, "reason": "already_has_micro"})
            continue
        if not _has_segment_level_visual_evidence(segment):
            skipped.append({"segment_id": segment_id, "reason": "no_segment_level_visual_evidence"})
            continue
        micro = _build_backfill_micro(segment, existing_micro_ids)
        micros.append(micro)
        added.append(
            {
                "segment_id": segment_id,
                "micro_segment_id": micro["micro_segment_id"],
                "primary_object": micro["interaction"]["primary_object"],
                "evidence_level": micro["evidence"]["evidence_level"],
                "reason": micro["evidence"]["coverage_backfill_reason"],
            }
        )
        existing_micro_ids.add(str(micro["micro_segment_id"]))

    if added:
        micros.sort(key=lambda row: (str(row.get("parent_segment_id") or ""), float(row.get("start_sec") or 0.0), str(row.get("micro_segment_id") or "")))
        _renumber_display_order(micros)
        write_jsonl(micro_path, micros)
        _write_segments_with_backfill_refs(segment_path, segments, micros)

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "session_dir": str(session),
        "segment_count": len(segments),
        "input_micro_count": len(micros) - len(added),
        "added_micro_count": len(added),
        "output_micro_count": len(micros),
        "added": added,
        "skipped_counts": dict(sorted(Counter(item["reason"] for item in skipped).items())),
        "artifacts": {
            "micro_segments": str(micro_path),
            "key_action_segments": str(segment_path),
        },
    }
    target = Path(output_report) if output_report else session / "evaluation" / "micro_coverage_backfill_summary.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _build_backfill_micro(segment: Mapping[str, Any], existing_micro_ids: set[str]) -> dict[str, Any]:
    segment_id = str(segment.get("segment_id") or "segment")
    micro_id = _unique_micro_id(segment_id, existing_micro_ids)
    global_start = str(segment.get("global_start_time") or "")
    global_end = str(segment.get("global_end_time") or global_start)
    start_sec, end_sec = _segment_local_bounds(segment)
    duration = _duration(segment, start_sec, end_sec)
    primary = _primary_object(segment)
    interactions = _as_list(segment.get("interaction_events")) or _as_list(segment.get("yolo_interactions"))
    score_values = [_float(item.get("confidence"), None) for item in interactions if isinstance(item, Mapping)]
    score_values = [value for value in score_values if value is not None]
    max_score = max(score_values or [0.0])
    avg_score = sum(score_values) / len(score_values) if score_values else 0.0
    yolo_evidence = _yolo_evidence(segment, primary)
    keyframes = _keyframes(segment)
    detected_objects = _detected_objects(segment, primary)
    evidence_level = "weak_visual_evidence" if interactions else "insufficient_evidence"
    text = _index_text(segment, micro_id, primary, global_start, global_end, evidence_level)
    warnings = {
        "coverage_backfill_candidate",
        "no_dialogue_context",
        "retrieval_candidate_only",
        "segment_level_retrieval_backfill",
    }
    if not interactions:
        warnings.add("very_low_signal_yolo_candidate")
    elif len(interactions) <= 1:
        warnings.add("single_frame_coverage_candidate")
        warnings.add("low_signal_yolo_candidate")
    elif max_score < 0.5:
        warnings.add("low_signal_yolo_candidate")

    return {
        "session_id": segment.get("session_id"),
        "parent_segment_id": segment_id,
        "micro_segment_id": micro_id,
        "display_order": 1,
        "display_id": "micro_001",
        "start_sec": round(start_sec, 6),
        "end_sec": round(end_sec, 6),
        "duration_sec": round(duration, 6),
        "global_start_time": global_start,
        "global_end_time": global_end,
        "first_person": _view_ref(segment, "first_person"),
        "third_person": _view_ref(segment, "third_person"),
        "interaction": {
            "interaction_type": f"segment_level_{primary}_candidate",
            "primary_object": primary,
            "secondary_objects": ["hand"] if "gloved_hand" in detected_objects or "hand" in detected_objects else [],
            "detected_objects": detected_objects,
            "avg_interaction_score": round(avg_score, 6),
            "max_interaction_score": round(max_score, 6),
            "contact_start_sec": round(start_sec, 6),
            "peak_interaction_sec": round((start_sec + end_sec) / 2.0, 6),
            "contact_end_sec": round(end_sec, 6),
            "evidence_frame_indices": [int(item.get("frame_index") or 0) for item in yolo_evidence if isinstance(item, Mapping)],
            "avg_hand_object_distance": None,
            "max_bbox_overlap": 0.0,
            "primary_object_family": _object_family(primary),
            "primary_object_arbitration": "segment_level_backfill",
            "primary_object_vote_score": round(max_score, 6),
            "primary_object_vote_margin": 0.0,
            "primary_object_vote_counts": {primary: len(interactions)} if primary else {},
            "primary_object_vote_scores": {primary: round(sum(score_values), 6)} if primary else {},
            "peak_primary_object": primary,
        },
        "keyframes": keyframes,
        "dialogue_context": [],
        "text_description": {
            "action_type": _action_type(segment, primary),
            "summary": f"Segment-level retrieval backfill for {primary}; requires confirmation before process claims.",
            "index_text": text,
        },
        "index": {"index_level": "micro_segment", "embedding_id": f"emb_{micro_id}"},
        "quality": {"confidence": "low" if max_score < 0.5 else "medium", "warnings": sorted(warnings)},
        "class_threshold": {"interaction_threshold": 1.0, "min_duration_sec": 0.0, "query_boost": 1.0},
        "dialogue_context_available": False,
        "dialogue_match_window_sec": 2.0,
        "dialogue_keywords": [],
        "asset_bindings": _asset_bindings(segment, micro_id, segment_id, global_start, global_end, keyframes, max_score),
        "yolo_evidence": yolo_evidence,
        "evidence": {
            "evidence_level": evidence_level,
            "evidence_reasons": _evidence_reasons(segment, primary, interactions, keyframes),
            "limitations": [
                "segment-level coverage backfill; not a validated hand-object micro-segment",
                "retrieval candidate only; not eligible for strong process claims",
            ],
            "coverage_backfill": True,
            "segment_level_coverage_backfill": True,
            "force_retrieval_candidate": True,
            "coverage_backfill_reason": "parent segment had segment-level visual evidence but no micro-segment",
            "coverage_signal_grade": "segment_level_retrieval_candidate",
            "coverage_evidence_frame_count": len(yolo_evidence),
            "coverage_bbox_frame_count": _bbox_frame_count(yolo_evidence, primary),
            "coverage_max_interaction_score": round(max_score, 6),
            "coverage_avg_interaction_score": round(avg_score, 6),
        },
        "evidence_level": evidence_level,
        "evidence_reasons": _evidence_reasons(segment, primary, interactions, keyframes),
        "limitations": [
            "segment-level coverage backfill; not a validated hand-object micro-segment",
            "retrieval candidate only; not eligible for strong process claims",
        ],
        "manual_corrected": True,
        "manual_correction_note": "auto_segment_level_micro_coverage_backfill",
    }


def _has_segment_level_visual_evidence(segment: Mapping[str, Any]) -> bool:
    if _as_list(segment.get("interaction_events")) or _as_list(segment.get("interaction_keyframes")):
        return True
    if _as_list(segment.get("yolo_interactions")):
        return True
    labels = _as_dict(segment.get("yolo_label_counts"))
    if any(_norm(label) not in _NON_OBJECT_LABELS and int(count or 0) > 0 for label, count in labels.items()):
        return True
    return bool(_as_list(segment.get("asset_bindings")))


def _primary_object(segment: Mapping[str, Any]) -> str:
    interactions = [item for item in _as_list(segment.get("interaction_events")) + _as_list(segment.get("yolo_interactions")) if isinstance(item, Mapping)]
    if interactions:
        best = max(interactions, key=lambda item: (_float(item.get("confidence"), 0.0), _OBJECT_PRIORITY.get(_norm(item.get("object_label")), 0)))
        label = _norm(best.get("object_label"))
        if label and label not in _NON_OBJECT_LABELS:
            return label
    counts = _as_dict(segment.get("yolo_label_counts"))
    candidates = [
        (_norm(label), int(count or 0))
        for label, count in counts.items()
        if _norm(label) not in _NON_OBJECT_LABELS
    ]
    if candidates:
        return max(candidates, key=lambda item: (_OBJECT_PRIORITY.get(item[0], 0), item[1], item[0]))[0]
    text = _as_dict(segment.get("text_description"))
    for value in _as_list(text.get("tools")) + _as_list(text.get("objects")):
        label = _norm(value).strip("[]'\"")
        if label and label not in _NON_OBJECT_LABELS:
            return label
    return "unknown_object"


def _detected_objects(segment: Mapping[str, Any], primary: str) -> list[str]:
    labels = {_norm(label) for label in _as_list(segment.get("yolo_labels"))}
    labels.update(_norm(label) for label in _as_dict(segment.get("yolo_label_counts")))
    for item in _as_list(segment.get("interaction_events")) + _as_list(segment.get("yolo_interactions")):
        if isinstance(item, Mapping):
            labels.add(_norm(item.get("hand_label")))
            labels.add(_norm(item.get("object_label")))
    labels.add(primary)
    return sorted(label for label in labels if label)


def _yolo_evidence(segment: Mapping[str, Any], primary: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in _as_list(segment.get("yolo_interactions")) + _as_list(segment.get("interaction_events")):
        if not isinstance(item, Mapping):
            continue
        local = _float(item.get("local_time_sec"), 0.0)
        frame_index = int(local * 30.0)
        detections = [dict(det) for det in _as_list(item.get("detections")) if isinstance(det, Mapping)]
        rows.append(
            {
                "frame_index": frame_index,
                "view": item.get("view"),
                "local_time_sec": local,
                "global_time": item.get("global_time"),
                "primary_object": _norm(item.get("object_label")) or primary,
                "interaction_score": _float(item.get("confidence"), 0.0),
                "bbox_overlap": 0.0,
                "hand_object_distance": None,
                "source": item.get("source") or "segment_level_backfill",
                "detections": detections,
                "hand_object_interactions": [
                    {
                        "hand_label": item.get("hand_label") or "gloved_hand",
                        "object_label": item.get("object_label") or primary,
                        "score": _float(item.get("confidence"), 0.0),
                        "hand_bbox": _bbox_for_label(detections, item.get("hand_label") or "gloved_hand"),
                        "object_bbox": _bbox_for_label(detections, item.get("object_label") or primary),
                    }
                ]
                if detections
                else [],
            }
        )
    return rows


def _keyframes(segment: Mapping[str, Any]) -> dict[str, str | None]:
    interaction_frames = [item for item in _as_list(segment.get("interaction_keyframes")) if isinstance(item, Mapping) and item.get("path")]
    if interaction_frames:
        paths = [str(item.get("path")) for item in interaction_frames]
        return {
            "contact_frame": paths[0],
            "peak_frame": paths[len(paths) // 2],
            "release_frame": paths[-1],
        }
    bindings = [item for item in _as_list(segment.get("asset_bindings")) if isinstance(item, Mapping)]
    for binding in bindings:
        keyframes = _as_dict(binding.get("keyframes"))
        if keyframes:
            return {
                "contact_frame": keyframes.get("start") or keyframes.get("middle") or keyframes.get("end"),
                "peak_frame": keyframes.get("middle") or keyframes.get("start") or keyframes.get("end"),
                "release_frame": keyframes.get("end") or keyframes.get("middle") or keyframes.get("start"),
            }
    for binding in bindings:
        if binding.get("keyframe_path"):
            path = str(binding.get("keyframe_path"))
            return {"contact_frame": path, "peak_frame": path, "release_frame": path}
    return {"contact_frame": None, "peak_frame": None, "release_frame": None}


def _asset_bindings(
    segment: Mapping[str, Any],
    micro_id: str,
    segment_id: str,
    global_start: str,
    global_end: str,
    keyframes: Mapping[str, Any],
    confidence: float,
) -> list[dict[str, Any]]:
    output = []
    for binding in _as_list(segment.get("asset_bindings")):
        if not isinstance(binding, Mapping):
            continue
        item = dict(binding)
        item["level"] = "micro_segment"
        item["micro_segment_id"] = micro_id
        item["parent_segment_id"] = segment_id
        item["global_start_time"] = global_start
        item["global_end_time"] = global_end
        item["confidence"] = round(confidence, 6)
        item["evidence_source"] = "segment_level_micro_coverage_backfill"
        output.append(item)
    for rel, path in keyframes.items():
        if path:
            output.append(
                {
                    "level": "micro_segment",
                    "micro_segment_id": micro_id,
                    "parent_segment_id": segment_id,
                    "asset_type": "keyframe",
                    "rel": rel,
                    "path": path,
                    "global_start_time": global_start,
                    "global_end_time": global_end,
                    "confidence": round(confidence, 6),
                    "evidence_source": "segment_level_micro_coverage_backfill",
                }
            )
    return output


def _write_segments_with_backfill_refs(segment_path: Path, segments: list[dict[str, Any]], micros: list[dict[str, Any]]) -> None:
    by_parent: dict[str, list[dict[str, Any]]] = {}
    for micro in micros:
        by_parent.setdefault(str(micro.get("parent_segment_id") or ""), []).append(micro)
    updated = []
    for segment in segments:
        row = dict(segment)
        parent_id = str(row.get("segment_id") or "")
        parent_micros = sorted(by_parent.get(parent_id, []), key=lambda item: (int(item.get("display_order") or 0), str(item.get("micro_segment_id") or "")))
        row["micro_segments"] = [_parent_micro_ref(item) for item in parent_micros]
        updated.append(row)
    write_jsonl(segment_path, updated)


def _parent_micro_ref(micro: Mapping[str, Any]) -> dict[str, Any]:
    interaction = _as_dict(micro.get("interaction"))
    evidence = _as_dict(micro.get("evidence"))
    keyframes = _as_dict(micro.get("keyframes"))
    return {
        "micro_segment_id": micro.get("micro_segment_id"),
        "display_order": micro.get("display_order"),
        "display_id": micro.get("display_id"),
        "primary_object": interaction.get("primary_object"),
        "primary_object_family": interaction.get("primary_object_family"),
        "primary_object_arbitration": interaction.get("primary_object_arbitration"),
        "interaction_type": interaction.get("interaction_type"),
        "global_start_time": micro.get("global_start_time"),
        "global_end_time": micro.get("global_end_time"),
        "duration_sec": micro.get("duration_sec"),
        "max_interaction_score": interaction.get("max_interaction_score"),
        "confidence": _as_dict(micro.get("quality")).get("confidence"),
        "peak_keyframe": keyframes.get("peak_frame"),
        "first_person_clip": _as_dict(micro.get("first_person")).get("clip_path"),
        "third_person_clip": _as_dict(micro.get("third_person")).get("clip_path"),
        "manual_corrected": micro.get("manual_corrected"),
        "dialogue_context_available": micro.get("dialogue_context_available"),
        "dialogue_match_window_sec": micro.get("dialogue_match_window_sec"),
        "dialogue_keywords": micro.get("dialogue_keywords") or [],
        "evidence_level": evidence.get("evidence_level"),
        "evidence": evidence,
        "asset_bindings": micro.get("asset_bindings") or [],
        "yolo_evidence": micro.get("yolo_evidence") or [],
        "class_threshold": micro.get("class_threshold") or {},
    }


def _renumber_display_order(micros: list[dict[str, Any]]) -> None:
    by_parent: dict[str, list[dict[str, Any]]] = {}
    for row in micros:
        by_parent.setdefault(str(row.get("parent_segment_id") or ""), []).append(row)
    for rows in by_parent.values():
        rows.sort(key=lambda row: (float(row.get("start_sec") or 0.0), str(row.get("micro_segment_id") or "")))
        for index, row in enumerate(rows, start=1):
            row["display_order"] = index
            row["display_id"] = f"micro_{index:03d}"


def _unique_micro_id(segment_id: str, existing: set[str]) -> str:
    index = 1
    while True:
        candidate = f"{segment_id}_micro_{index:03d}"
        if candidate not in existing:
            return candidate
        index += 1


def _segment_local_bounds(segment: Mapping[str, Any]) -> tuple[float, float]:
    cv = _as_dict(segment.get("cv_detection"))
    start = _float(cv.get("start_sec"), None)
    end = _float(cv.get("end_sec"), None)
    if start is not None and end is not None and end > start:
        return start, end
    for view in ("third_person", "first_person"):
        ref = _as_dict(segment.get(view))
        start = _float(ref.get("local_start_sec"), None)
        end = _float(ref.get("local_end_sec"), None)
        if start is not None and end is not None and end > start:
            return start, end
    duration = _float(segment.get("duration_sec"), 0.0)
    return 0.0, max(0.0, duration)


def _duration(segment: Mapping[str, Any], start_sec: float, end_sec: float) -> float:
    value = _float(segment.get("duration_sec"), None)
    if value is not None and value > 0:
        return value
    return max(0.0, end_sec - start_sec)


def _view_ref(segment: Mapping[str, Any], view: str) -> dict[str, Any] | None:
    ref = _as_dict(segment.get(view))
    if not ref:
        return None
    return {
        "clip_path": ref.get("clip_path"),
        "local_start_sec": ref.get("local_start_sec"),
        "local_end_sec": ref.get("local_end_sec"),
        "annotated_clip_path": ref.get("annotated_clip_path"),
    }


def _action_type(segment: Mapping[str, Any], primary: str) -> str:
    text = _as_dict(segment.get("text_description"))
    if text.get("action_type"):
        return str(text.get("action_type"))
    return f"{primary}_segment_candidate"


def _index_text(segment: Mapping[str, Any], micro_id: str, primary: str, start: str, end: str, evidence_level: str) -> str:
    text = _as_dict(segment.get("text_description"))
    summary = str(text.get("summary") or "")
    return "\n".join(
        [
            f"session: {segment.get('session_id')}",
            f"parent segment: {segment.get('segment_id')}",
            f"micro segment: {micro_id}",
            "index_level: micro_segment",
            "source: segment_level_micro_coverage_backfill",
            f"time: {start} to {end}",
            f"primary object: {primary}",
            f"action type: {_action_type(segment, primary)}",
            f"evidence level: {evidence_level}",
            "process evidence role: retrieval_candidate",
            "strong process evidence: false",
            summary,
        ]
    )


def _evidence_reasons(segment: Mapping[str, Any], primary: str, interactions: list[Any], keyframes: Mapping[str, Any]) -> list[str]:
    reasons = [f"segment-level fallback primary object: {primary}"]
    if interactions:
        reasons.append(f"segment interaction candidates: {len(interactions)}")
    counts = _as_dict(segment.get("yolo_label_counts"))
    if counts:
        labels = ", ".join(f"{label}({count})" for label, count in sorted(counts.items()))
        reasons.append(f"segment yolo labels: {labels}")
    if any(keyframes.values()):
        reasons.append("segment keyframes available")
    return reasons


def _bbox_frame_count(rows: list[Mapping[str, Any]], primary: str) -> int:
    count = 0
    for row in rows:
        for item in _as_list(row.get("hand_object_interactions")):
            if not isinstance(item, Mapping):
                continue
            if _norm(item.get("object_label")) not in {"", primary}:
                continue
            if item.get("hand_bbox") and item.get("object_bbox"):
                count += 1
                break
    return count


def _bbox_for_label(detections: list[Mapping[str, Any]], label: Any) -> Any:
    target = _norm(label)
    for det in detections:
        if _norm(det.get("label") or det.get("raw_label")) == target and det.get("bbox"):
            return det.get("bbox")
    return None


def _object_family(label: str) -> str | None:
    if not label or label == "unknown_object":
        return None
    if label in {"pipette", "pipette_tip"}:
        return "pipette_family"
    if label in {"sample_bottle", "reagent_bottle", "bottle"}:
        return "bottle_family"
    return f"{label}_family"


def _norm(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "_")


def _float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


__all__ = ["SCHEMA_VERSION", "backfill_micro_coverage"]

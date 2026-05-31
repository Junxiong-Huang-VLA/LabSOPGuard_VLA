from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .evidence import attach_evidence
from .schemas import MicroSegmentConfig, read_jsonl, write_jsonl


def _interaction(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("interaction") if isinstance(row.get("interaction"), dict) else {}


def _text_description(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("text_description") if isinstance(row.get("text_description"), dict) else {}


def _quality(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("quality") if isinstance(row.get("quality"), dict) else {}


def _primary_object(row: dict[str, Any]) -> str:
    return str(row.get("primary_object") or _interaction(row).get("primary_object") or "")


def _action_type(row: dict[str, Any]) -> str:
    return str(row.get("action_type") or _text_description(row).get("action_type") or "")


def _evidence_level(row: dict[str, Any]) -> str:
    evidence = row.get("evidence") if isinstance(row.get("evidence"), dict) else {}
    return str(row.get("evidence_level") or evidence.get("evidence_level") or "")


def _confidence(row: dict[str, Any]) -> str:
    return str(row.get("confidence") or _quality(row).get("confidence") or "")


def _is_low_confidence(row: dict[str, Any]) -> bool:
    return _confidence(row).lower() == "low"


def _compatible_action(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_action = _action_type(left)
    right_action = _action_type(right)
    if not left_action or not right_action:
        return True
    if left_action == right_action:
        return True
    compatible = {
        frozenset({"reagent_bottle_interaction", "bottle_interaction"}),
        frozenset({"sample_adding_candidate", "possible_sample_handling"}),
    }
    return frozenset({left_action, right_action}) in compatible


def _compatible_evidence(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_level = _evidence_level(left)
    right_level = _evidence_level(right)
    if not left_level or not right_level:
        return True
    if "insufficient" in {left_level, right_level}:
        return left_level == right_level
    return True


def _gap_sec(left: dict[str, Any], right: dict[str, Any]) -> float:
    return float(right.get("start_sec", 0.0) or 0.0) - float(left.get("end_sec", 0.0) or 0.0)


def _can_merge(left: dict[str, Any], right: dict[str, Any], config: MicroSegmentConfig) -> tuple[bool, str]:
    if str(left.get("parent_segment_id") or "") != str(right.get("parent_segment_id") or ""):
        return False, ""
    if _primary_object(left) != _primary_object(right):
        return False, ""
    if not _compatible_action(left, right):
        return False, ""
    if not _compatible_evidence(left, right):
        return False, ""
    if not config.merge_low_confidence_adjacent and (_is_low_confidence(left) or _is_low_confidence(right)):
        return False, ""
    gap = _gap_sec(left, right)
    if gap < -0.05 or gap > config.same_object_merge_gap_sec:
        return False, ""
    merged_duration = float(right.get("end_sec", 0.0) or 0.0) - float(left.get("start_sec", 0.0) or 0.0)
    if merged_duration > config.max_merged_micro_duration_sec:
        return False, ""
    return True, f"same_primary_object_adjacent_gap_{gap:.3f}s"


def _merge_keyframes(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = rows[0].get("keyframes") if isinstance(rows[0].get("keyframes"), dict) else {}
    last = rows[-1].get("keyframes") if isinstance(rows[-1].get("keyframes"), dict) else {}
    peak_row = max(rows, key=lambda row: float(_interaction(row).get("max_interaction_score", 0.0) or 0.0))
    peak = peak_row.get("keyframes") if isinstance(peak_row.get("keyframes"), dict) else {}
    return {
        "contact_frame": first.get("contact_frame") or peak.get("contact_frame"),
        "peak_frame": peak.get("peak_frame") or first.get("peak_frame"),
        "release_frame": last.get("release_frame") or peak.get("release_frame"),
    }


def _merge_interaction(rows: list[dict[str, Any]]) -> dict[str, Any]:
    interactions = [_interaction(row) for row in rows]
    primary = _primary_object(rows[0])
    max_score = max(float(item.get("max_interaction_score", 0.0) or 0.0) for item in interactions)
    total_duration = sum(float(row.get("duration_sec", 0.0) or 0.0) for row in rows) or float(len(rows))
    avg_score = sum(float(item.get("avg_interaction_score", 0.0) or 0.0) * float(row.get("duration_sec", 1.0) or 1.0) for row, item in zip(rows, interactions)) / total_duration
    detected = sorted({str(label) for item in interactions for label in item.get("detected_objects", []) if label})
    frame_indices = [idx for item in interactions for idx in item.get("evidence_frame_indices", [])]
    return {
        **interactions[0],
        "primary_object": primary,
        "interaction_type": interactions[0].get("interaction_type") or f"hand_{primary}_contact",
        "detected_objects": detected or interactions[0].get("detected_objects", []),
        "avg_interaction_score": round(avg_score, 6),
        "max_interaction_score": round(max_score, 6),
        "contact_start_sec": min(float(item.get("contact_start_sec", rows[0].get("start_sec", 0.0)) or 0.0) for item in interactions),
        "contact_end_sec": max(float(item.get("contact_end_sec", rows[-1].get("end_sec", 0.0)) or 0.0) for item in interactions),
        "peak_interaction_sec": _interaction(max(rows, key=lambda row: float(_interaction(row).get("max_interaction_score", 0.0) or 0.0))).get("peak_interaction_sec"),
        "evidence_frame_indices": frame_indices,
        "max_bbox_overlap": max(float(item.get("max_bbox_overlap", 0.0) or 0.0) for item in interactions),
    }


def _merge_quality(rows: list[dict[str, Any]]) -> dict[str, Any]:
    confidence_order = {"high": 3, "medium": 2, "low": 1, "unknown": 0}
    confidence = max((_confidence(row) or "unknown" for row in rows), key=lambda value: confidence_order.get(value, 0))
    warnings = sorted({str(warning) for row in rows for warning in _quality(row).get("warnings", [])})
    return {"confidence": confidence, "warnings": warnings}


def _merge_dialogue(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    context: list[dict[str, Any]] = []
    keywords: list[str] = []
    seen_context: set[str] = set()
    seen_keywords: set[str] = set()
    for row in rows:
        for item in row.get("dialogue_context") or []:
            key = str(item.get("utterance_id") or item.get("text") or item)
            if key not in seen_context:
                seen_context.add(key)
                context.append(item)
        for keyword in row.get("dialogue_keywords") or []:
            key = str(keyword)
            if key not in seen_keywords:
                seen_keywords.add(key)
                keywords.append(key)
    return context, keywords


def _replace_text_identity(text: str, source_id: str, target_id: str, display_id: str) -> str:
    text = str(text or "")
    if source_id and target_id and source_id != target_id:
        text = text.replace(source_id, target_id)
    if display_id:
        lines = []
        replaced = False
        for line in text.splitlines():
            if line.startswith("display_id:"):
                lines.append(f"display_id: {display_id}")
                replaced = True
            else:
                lines.append(line)
        text = "\n".join(lines)
        if not replaced:
            text += f"\ndisplay_id: {display_id}"
    return text


def _view_data(row: dict[str, Any], view: str) -> dict[str, Any]:
    value = row.get(view)
    return value if isinstance(value, dict) else {}


def _merged_view_data(rows: list[dict[str, Any]], view: str, start_sec: float, end_sec: float) -> dict[str, Any]:
    merged: dict[str, Any] = {"local_start_sec": round(start_sec, 6), "local_end_sec": round(end_sec, 6)}
    for row in rows:
        view_data = _view_data(row, view)
        if not view_data:
            continue
        for key in ("clip_path", "annotated_clip_path", "keyframe_path", "keyframe_paths", "keyframes"):
            value = view_data.get(key)
            if value:
                merged.setdefault(key, deepcopy(value))
    return merged


def _row_asset_bindings(row: dict[str, Any]) -> list[dict[str, Any]]:
    raw_bindings = row.get("asset_bindings")
    if isinstance(raw_bindings, list):
        return [deepcopy(item) for item in raw_bindings if isinstance(item, dict)]
    bindings: list[dict[str, Any]] = []
    for view in ("third_person", "first_person"):
        view_data = _view_data(row, view)
        if not view_data:
            continue
        bindings.append(
            {
                "level": "micro_segment",
                "micro_segment_id": row.get("micro_segment_id"),
                "parent_segment_id": row.get("parent_segment_id") or row.get("segment_id"),
                "view": view,
                "global_start_time": row.get("global_start_time"),
                "global_end_time": row.get("global_end_time"),
                "local_start_sec": view_data.get("local_start_sec", row.get("start_sec")),
                "local_end_sec": view_data.get("local_end_sec", row.get("end_sec")),
                "clip_path": view_data.get("clip_path"),
                "annotated_clip_path": view_data.get("annotated_clip_path"),
                "keyframe_path": view_data.get("keyframe_path"),
                "keyframe_paths": deepcopy(view_data.get("keyframe_paths") or []),
                "keyframes": deepcopy(view_data.get("keyframes") or {}),
            }
        )
    return bindings


def _merge_asset_bindings(rows: list[dict[str, Any]], merged_id: str, parent_id: str) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        source_micro_id = str(row.get("micro_segment_id") or "")
        for binding in _row_asset_bindings(row):
            view = str(binding.get("view") or binding.get("camera_view") or "").strip()
            clip_path = str(binding.get("clip_path") or binding.get("annotated_clip_path") or "")
            keyframe_path = str(binding.get("keyframe_path") or "")
            key = (view, clip_path, keyframe_path)
            if key in seen:
                continue
            seen.add(key)
            binding["source_micro_segment_id"] = source_micro_id
            binding["micro_segment_id"] = merged_id
            binding["parent_segment_id"] = parent_id
            merged.append(binding)
    return merged


def _merged_row(rows: list[dict[str, Any]], sequence: int) -> dict[str, Any]:
    if len(rows) == 1:
        row = deepcopy(rows[0])
        row["display_order"] = sequence
        row["display_id"] = f"micro_{sequence:03d}"
        return attach_evidence(row)

    first = deepcopy(rows[0])
    merged_ids = [str(row.get("micro_segment_id")) for row in rows if row.get("micro_segment_id")]
    source_id = str(first.get("micro_segment_id") or "")
    merged_id = f"{source_id}_merged" if source_id else f"{first.get('parent_segment_id', 'segment')}_micro_merged_{sequence:03d}"
    start_sec = min(float(row.get("start_sec", 0.0) or 0.0) for row in rows)
    end_sec = max(float(row.get("end_sec", 0.0) or 0.0) for row in rows)
    first["micro_segment_id"] = merged_id
    first["display_order"] = sequence
    first["display_id"] = f"micro_{sequence:03d}"
    first["start_sec"] = round(start_sec, 6)
    first["end_sec"] = round(end_sec, 6)
    first["duration_sec"] = round(max(0.0, end_sec - start_sec), 6)
    first["global_start_time"] = rows[0].get("global_start_time")
    first["global_end_time"] = rows[-1].get("global_end_time")
    parent_id = str(first.get("parent_segment_id") or first.get("segment_id") or "")
    first["first_person"] = _merged_view_data(rows, "first_person", start_sec, end_sec)
    first["third_person"] = _merged_view_data(rows, "third_person", start_sec, end_sec)
    first["asset_bindings"] = _merge_asset_bindings(rows, merged_id, parent_id)
    first["interaction"] = _merge_interaction(rows)
    first["keyframes"] = _merge_keyframes(rows)
    first["quality"] = _merge_quality(rows)
    context, keywords = _merge_dialogue(rows)
    first["dialogue_context"] = context
    first["dialogue_context_available"] = bool(context)
    first["dialogue_keywords"] = keywords
    first["merged_from_micro_segment_ids"] = merged_ids
    first["merge_reason"] = f"same_primary_object_adjacent_gap_{max(_gap_sec(left, right) for left, right in zip(rows, rows[1:])):.3f}s"
    text_description = first.get("text_description") if isinstance(first.get("text_description"), dict) else {}
    text_description["index_text"] = _replace_text_identity(
        str(text_description.get("index_text", "")),
        source_id,
        merged_id,
        first["display_id"],
    )
    first["text_description"] = text_description
    index = first.get("index") if isinstance(first.get("index"), dict) else {}
    index["embedding_id"] = f"emb_{merged_id}"
    index["index_level"] = "micro_segment"
    first["index"] = index
    return attach_evidence(first)


def merge_same_object_adjacent_micro_segments(
    micro_segments: list[dict[str, Any]],
    config: MicroSegmentConfig | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg = config or MicroSegmentConfig()
    ordered = sorted(
        [deepcopy(row) for row in micro_segments],
        key=lambda row: (str(row.get("parent_segment_id") or ""), float(row.get("start_sec", 0.0) or 0.0), str(row.get("micro_segment_id") or "")),
    )
    if not cfg.same_object_merge_enabled:
        merged = [_merged_row([row], sequence=index) for index, row in enumerate(ordered, start=1)]
        return merged, _merge_stats(micro_segments, merged, enabled=False)

    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for row in ordered:
        if not current:
            current = [row]
            continue
        ok, _reason = _can_merge(current[-1], row, cfg)
        if ok:
            current.append(row)
        else:
            groups.append(current)
            current = [row]
    if current:
        groups.append(current)
    merged = [_merged_row(group, sequence=index) for index, group in enumerate(groups, start=1)]
    return merged, _merge_stats(micro_segments, merged, enabled=True)


def _duration_span(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    start = min(float(row.get("start_sec", 0.0) or 0.0) for row in rows)
    end = max(float(row.get("end_sec", row.get("start_sec", 0.0)) or 0.0) for row in rows)
    return max(0.0, end - start)


def _merge_stats(raw: list[dict[str, Any]], merged: list[dict[str, Any]], *, enabled: bool) -> dict[str, Any]:
    raw_span = _duration_span(raw)
    merged_span = _duration_span(merged)
    raw_per_minute = len(raw) / raw_span * 60.0 if raw_span > 0 else 0.0
    merged_per_minute = len(merged) / merged_span * 60.0 if merged_span > 0 else 0.0
    return {
        "same_object_merge_enabled": bool(enabled),
        "raw_micro_count": len(raw),
        "merged_micro_count": len(merged),
        "merge_count": max(0, len(raw) - len(merged)),
        "micro_per_minute_before": raw_per_minute,
        "micro_per_minute_after": merged_per_minute,
    }


def merge_micro_segments_file(
    source_path: str | Path,
    output_path: str | Path,
    config: MicroSegmentConfig | None = None,
    stats_path: str | Path | None = None,
) -> dict[str, Any]:
    rows = read_jsonl(source_path) if Path(source_path).exists() else []
    merged, stats = merge_same_object_adjacent_micro_segments(rows, config=config)
    write_jsonl(output_path, merged)
    if stats_path is not None:
        Path(stats_path).parent.mkdir(parents=True, exist_ok=True)
        Path(stats_path).write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats

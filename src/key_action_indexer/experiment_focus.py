from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .clip_extractor import extract_clip_ffmpeg
from .frame_time_map import capture_sec_to_video_sec
from .schemas import SessionManifest, VideoSource, read_jsonl
from .time_alignment import global_time_to_local_sec, local_sec_to_global_time, parse_time
from .video_utils import get_video_duration_sec


FOCUS_WINDOW_SCHEMA = "experiment_focus_window.v1"
FOCUS_CLIPS_SCHEMA = "experiment_focus_clips.v1"
CORE_START_OBJECTS = {
    "balance",
    "beaker",
    "container",
    "magnetic_stir_bar",
    "magnetic_stirrer",
    "reagent_bottle",
    "sample_bottle",
    "sample_bottle_blue",
    "tube",
}
PREP_OBJECTS = {"gloved_hand", "hand", "lab_coat", "paper", "ppe_storage", "spatula"}
PPE_ACTIVE_LABELS = {"gloved_hand", "lab_coat"}
PPE_PREP_LABELS = {"gloved_hand", "hand", "lab_coat", "ppe_storage"}
PPE_EXIT_LABELS = {"hand", "ppe_storage"}
LIFECYCLE_RULE_SCHEMA = "experiment_lifecycle_boundary_rules.v1"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _session_sec(manifest: SessionManifest, global_time: str) -> float:
    return (parse_time(global_time) - parse_time(manifest.session_start_time)).total_seconds()


def _source_for_view(manifest: SessionManifest, view: str | None) -> VideoSource:
    normalized = str(view or "").strip().lower()
    if normalized == "first_person" and manifest.videos.first_person is not None:
        return manifest.videos.first_person
    if normalized in manifest.videos.extra_views:
        return manifest.videos.extra_views[normalized]
    return manifest.videos.third_person


def _global_time_to_video_sec(source: VideoSource, global_time: str) -> float:
    capture_sec = max(0.0, global_time_to_local_sec(source, global_time))
    return max(0.0, capture_sec_to_video_sec(source, capture_sec, use_frame_time_map="auto"))


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_rows(path: Path) -> list[dict[str, Any]]:
    try:
        if path.exists():
            return [row for row in read_jsonl(path) if isinstance(row, dict)]
    except Exception:
        return []
    return []


def _norm_label(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in text:
        text = text.replace("__", "_")
    if text == "ppe_storage":
        return "ppe_storage"
    if text == "ppe":
        return "ppe_storage"
    if text == "tube-cap":
        return "tube_cap"
    if text == "weighing_paper":
        return "paper"
    return text


def _row_labels(row: dict[str, Any]) -> set[str]:
    labels: set[str] = set()
    counts = row.get("label_counts") if isinstance(row.get("label_counts"), dict) else {}
    for label, count in counts.items():
        try:
            if int(count) <= 0:
                continue
        except (TypeError, ValueError):
            continue
        labels.add(_norm_label(label))
    for detection in row.get("detections") or []:
        if not isinstance(detection, dict):
            continue
        labels.add(_norm_label(detection.get("label") or detection.get("raw_label") or detection.get("class_name")))
    for interaction in row.get("hand_object_interactions") or []:
        if not isinstance(interaction, dict):
            continue
        for key in ("hand_label", "object_label", "object_name", "label"):
            labels.add(_norm_label(interaction.get(key)))
    labels.discard("")
    return labels


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, str(default))))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)


def _focus_clip_preview_options() -> dict[str, Any]:
    preview_enabled = _env_flag("KEY_ACTION_FOCUS_CLIP_PREVIEW_H264", True)
    if not preview_enabled:
        return {"copy_first": True}
    return {
        "copy_first": False,
        "video_codec": os.environ.get("KEY_ACTION_FOCUS_CLIP_VIDEO_CODEC") or os.environ.get("KEY_ACTION_FFMPEG_ENCODER") or "auto",
        "video_bitrate": os.environ.get("KEY_ACTION_FOCUS_CLIP_VIDEO_BITRATE") or "3M",
        "crf": _env_int("KEY_ACTION_FOCUS_CLIP_CRF", 28),
        "preset": os.environ.get("KEY_ACTION_FOCUS_CLIP_PRESET") or None,
        "max_width": _env_int("KEY_ACTION_FOCUS_CLIP_MAX_WIDTH", 1280),
        "include_audio": _env_flag("KEY_ACTION_FOCUS_CLIP_INCLUDE_AUDIO", False),
        "faststart": True,
    }


def _focus_clip_max_extract_sec() -> float:
    return max(0.0, _env_float("KEY_ACTION_EXPERIMENT_FOCUS_CLIP_MAX_EXTRACT_SEC", 900.0))


def _focus_clip_force_extract(default: bool = False) -> bool:
    return _env_flag("KEY_ACTION_EXPERIMENT_FOCUS_CLIP_FORCE_EXTRACT", default)


def _candidate_index_paths(session_dir: Path) -> list[Path]:
    roots = [
        session_dir.parent / "_material_review_queue",
        session_dir.parent / "material_candidates",
        session_dir.parent / "material_references",
        session_dir / "material_candidates",
        session_dir / "material_references",
    ]
    paths: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.glob("*.jsonl"):
            if path.name.lower() == "review_log.jsonl":
                continue
            paths.append(path)
    return paths


def _load_candidate_rows(session_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in _candidate_index_paths(session_dir):
        for row in _read_rows(path):
            if not (row.get("candidate_id") or row.get("candidate_group_id") or row.get("pipeline_flow")):
                continue
            key = str(row.get("candidate_id") or "") + "|" + str(row.get("stored_file") or row.get("source_file") or "")
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return rows


def _candidate_is_start_anchor(row: dict[str, Any]) -> bool:
    if str(row.get("role") or "").lower() not in {"micro_clip", "peak_clip", "segment_clip", ""}:
        return False
    if _safe_float(row.get("start_sec")) is None:
        return False
    status = str(row.get("candidate_status") or row.get("review_status") or "").lower()
    recommended = bool(row.get("recommended"))
    yolo_status = str(((row.get("yolo_recheck") or {}).get("status") if isinstance(row.get("yolo_recheck"), dict) else "") or "").lower()
    vlm_status = str(((row.get("vlm_semantics") or {}).get("status") if isinstance(row.get("vlm_semantics"), dict) else "") or "").lower()
    evidence_ok = yolo_status in {"passed", ""} and vlm_status in {"aligned", "partial", ""}
    return evidence_ok and (recommended or status in {"approved", "accepted"})


def _candidate_anchor_session_sec(
    manifest: SessionManifest,
    row: dict[str, Any],
    segments_by_id: dict[str, dict[str, Any]],
) -> float | None:
    local_start = _safe_float(row.get("start_sec"))
    if local_start is None:
        return None
    view = str(row.get("view") or "third_person")
    try:
        global_time = local_sec_to_global_time(_source_for_view(manifest, view), local_start)
        return (global_time - parse_time(manifest.session_start_time)).total_seconds()
    except Exception:
        pass

    segment = segments_by_id.get(str(row.get("segment_id") or row.get("parent_segment_id") or ""))
    if segment:
        ref = segment.get(view)
        if isinstance(ref, dict):
            ref_local_start = _safe_float(ref.get("local_start_sec"))
            segment_global_start = segment.get("global_start_time")
            if ref_local_start is not None and segment_global_start:
                return _session_sec(manifest, str(segment_global_start)) + max(0.0, local_start - ref_local_start)
    return local_start


def _earliest_candidate_anchor(
    manifest: SessionManifest,
    session_dir: Path,
    segments_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    anchors: list[dict[str, Any]] = []
    for row in _load_candidate_rows(session_dir):
        if not _candidate_is_start_anchor(row):
            continue
        start_sec = _candidate_anchor_session_sec(manifest, row, segments_by_id)
        if start_sec is None:
            continue
        anchors.append(
            {
                "source": "approved_material_candidate",
                "start_sec": float(start_sec),
                "segment_id": row.get("segment_id") or row.get("parent_segment_id"),
                "micro_segment_id": row.get("micro_segment_id"),
                "primary_object": row.get("primary_object"),
                "candidate_id": row.get("candidate_id"),
                "candidate_group_id": row.get("candidate_group_id"),
                "status": row.get("candidate_status") or row.get("review_status"),
                "recommended": bool(row.get("recommended")),
            }
        )
    return min(anchors, key=lambda item: float(item["start_sec"])) if anchors else None


def _labels_for_segment(segment: dict[str, Any]) -> set[str]:
    labels = {str(label) for label in (segment.get("yolo_label_counts") or {}) if str(label)}
    for key in ("first_person", "third_person"):
        ref = segment.get(key)
        if isinstance(ref, dict):
            labels.update(str(label) for label in (ref.get("yolo_label_counts") or {}) if str(label))
    for micro in segment.get("micro_segments") or []:
        if isinstance(micro, dict) and micro.get("primary_object"):
            labels.add(str(micro.get("primary_object")))
    return labels


def _fallback_anchor(manifest: SessionManifest, segments: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not segments:
        return None
    scored: list[tuple[int, float, dict[str, Any], set[str]]] = []
    for segment in segments:
        start = _safe_float((segment.get("cv_detection") or {}).get("start_sec"))
        if start is None and segment.get("global_start_time"):
            start = _session_sec(manifest, str(segment.get("global_start_time")))
        if start is None:
            continue
        labels = _labels_for_segment(segment)
        action_type = str((segment.get("text_description") or {}).get("action_type") or "").lower()
        score = 0
        if labels & CORE_START_OBJECTS:
            score += 5
        if any(isinstance(micro, dict) and str(micro.get("primary_object") or "") in CORE_START_OBJECTS for micro in segment.get("micro_segments") or []):
            score += 6
        if action_type in {"weighing", "sample_handling", "container_operation", "balance_operation"}:
            score += 4
        if labels and labels <= PREP_OBJECTS:
            score -= 4
        scored.append((score, float(start), segment, labels))
    if not scored:
        first = segments[0]
        return {
            "source": "first_detected_segment",
            "start_sec": _safe_float((first.get("cv_detection") or {}).get("start_sec"), 0.0) or 0.0,
            "segment_id": first.get("segment_id"),
            "primary_object": None,
        }
    scored.sort(key=lambda item: (-item[0], item[1]))
    score, start, segment, labels = scored[0]
    return {
        "source": "core_key_action_segment" if score > 0 else "first_detected_segment",
        "start_sec": start,
        "segment_id": segment.get("segment_id"),
        "primary_object": sorted(labels & CORE_START_OBJECTS)[0] if labels & CORE_START_OBJECTS else None,
        "score": score,
    }


def _row_session_time(manifest: SessionManifest, row: dict[str, Any]) -> float | None:
    if row.get("global_time"):
        try:
            return _session_sec(manifest, str(row.get("global_time")))
        except Exception:
            pass
    value = _safe_float(row.get("alignment_time_sec"))
    if value is not None:
        return value
    local = _safe_float(row.get("local_time_sec"), _safe_float(row.get("time_sec")))
    if local is None:
        return None
    view = str(row.get("source_view") or row.get("view") or "third_person")
    try:
        global_time = local_sec_to_global_time(_source_for_view(manifest, view), local)
        return (global_time - parse_time(manifest.session_start_time)).total_seconds()
    except Exception:
        return local


def _row_has_yolo_lab_evidence(row: dict[str, Any]) -> bool:
    if row.get("hand_object_interactions"):
        return True
    labels = _row_labels(row)
    if labels & (CORE_START_OBJECTS | PREP_OBJECTS):
        return True
    for detection in row.get("detections") or []:
        if _norm_label((detection or {}).get("label") or "") in (CORE_START_OBJECTS | PREP_OBJECTS):
            return True
    return False


def _row_has_yolo_hand_object_activity(row: dict[str, Any]) -> bool:
    if row.get("hand_object_interactions"):
        return True
    labels = _row_labels(row)
    return bool(("gloved_hand" in labels) and (labels & CORE_START_OBJECTS) and _safe_float(row.get("active_score"), 0.0) >= 0.45)


def _row_has_ppe_prep_evidence(row: dict[str, Any]) -> bool:
    labels = _row_labels(row)
    if "ppe_storage" in labels and labels & {"hand", "gloved_hand"}:
        return True
    if "lab_coat" in labels and labels & {"hand", "gloved_hand"}:
        return True
    return False


def _row_has_ppe_active_evidence(row: dict[str, Any]) -> bool:
    return bool(_row_labels(row) & PPE_ACTIVE_LABELS)


def _row_has_ppe_exit_evidence(row: dict[str, Any]) -> bool:
    labels = _row_labels(row)
    return bool((labels & PPE_EXIT_LABELS) and not (labels & PPE_ACTIVE_LABELS))


def _source_duration_sec(manifest: SessionManifest, session_root: Path, rows: list[dict[str, Any]] | None = None) -> float | None:
    candidates: list[float] = []
    for _view, source in [("third_person", manifest.videos.third_person), ("first_person", manifest.videos.first_person)]:
        if source is None:
            continue
        try:
            duration = get_video_duration_sec(source.path)
            end_global = local_sec_to_global_time(source, duration)
            candidates.append(max(0.0, (end_global - parse_time(manifest.session_start_time)).total_seconds()))
        except Exception:
            continue
    for row in rows or []:
        for key in ("source_duration_sec", "video_duration_sec"):
            value = _safe_float(row.get(key))
            if value is not None and value > 0:
                time_sec = _row_session_time(manifest, row)
                if time_sec is not None:
                    local_sec = _safe_float(row.get("local_time_sec"), _safe_float(row.get("time_sec")))
                    if local_sec is not None:
                        candidates.append(max(float(time_sec), float(time_sec) - float(local_sec) + float(value)))
                    else:
                        candidates.append(float(value))
                else:
                    candidates.append(float(value))
    return max(candidates) if candidates else None


def _experiment_lifecycle_bounds(
    manifest: SessionManifest,
    session_root: Path,
    *,
    min_duration_sec: float,
) -> dict[str, Any] | None:
    rows = _read_rows(session_root / "cv_outputs" / "frame_scores.jsonl")
    rows.extend(_read_rows(session_root / "cv_outputs" / "yolo_frame_rows.jsonl"))
    timed_rows: list[tuple[float, dict[str, Any], set[str]]] = []
    for row in rows:
        time_sec = _row_session_time(manifest, row)
        if time_sec is None:
            continue
        labels = _row_labels(row)
        if not labels:
            continue
        timed_rows.append((float(time_sec), row, labels))
    if not timed_rows:
        return None
    timed_rows.sort(key=lambda item: item[0])
    sample_period = _infer_sample_period([time_sec for time_sec, _row, _labels in timed_rows])
    ppe_prep_times = [
        time_sec
        for time_sec, row, _labels in timed_rows
        if _row_has_ppe_prep_evidence(row)
    ]
    ppe_active_times = [
        time_sec
        for time_sec, row, _labels in timed_rows
        if _row_has_ppe_active_evidence(row)
    ]
    action_times = [
        time_sec
        for time_sec, row, _labels in timed_rows
        if _row_has_yolo_hand_object_activity(row)
    ]
    lab_times = [
        time_sec
        for time_sec, row, _labels in timed_rows
        if _row_has_yolo_lab_evidence(row)
    ]
    if not (ppe_prep_times or ppe_active_times or action_times):
        return None

    first_action = min(action_times) if action_times else None
    ppe_start_candidates = [
        value
        for value in ppe_prep_times
        if first_action is None or value <= first_action + _env_float("KEY_ACTION_EXPERIMENT_START_PREP_AFTER_ACTION_TOLERANCE_SEC", 1.0)
    ]
    start_confirmed = bool(ppe_start_candidates)
    if ppe_start_candidates:
        start_sec = min(ppe_start_candidates)
        start_reason = "ppe_prep_before_first_core_action"
    elif ppe_active_times:
        start_sec = min(ppe_active_times)
        start_reason = "ppe_already_active_left_censored"
    else:
        start_sec = min(action_times)
        start_reason = "core_action_without_ppe_entry_left_censored"

    source_duration = _source_duration_sec(manifest, session_root, rows)
    last_action = max(action_times) if action_times else max(lab_times or [start_sec])
    last_ppe_active = max(ppe_active_times) if ppe_active_times else None
    exit_times = [
        time_sec
        for time_sec, row, _labels in timed_rows
        if time_sec >= last_action and _row_has_ppe_exit_evidence(row)
    ]
    end_confirmed = False
    if exit_times and last_ppe_active is not None and min(exit_times) >= last_ppe_active - sample_period:
        end_sec = min(exit_times) + sample_period
        end_reason = "ppe_exit_after_last_core_action"
        end_confirmed = True
    elif last_ppe_active is not None:
        if source_duration and _env_flag("KEY_ACTION_EXPERIMENT_EXTEND_TO_VIDEO_END_WITHOUT_PPE_EXIT", True):
            end_sec = source_duration
            end_reason = "no_ppe_exit_seen_extend_to_video_end"
        else:
            end_sec = last_ppe_active + sample_period
            end_reason = "last_ppe_active_without_exit"
    else:
        end_sec = max(lab_times or action_times or [start_sec]) + sample_period
        end_reason = "last_lab_activity_without_ppe_exit"

    if source_duration:
        end_sec = min(float(source_duration), float(end_sec))
    end_sec = max(float(end_sec), float(start_sec) + float(min_duration_sec))
    if source_duration:
        end_sec = min(float(source_duration), float(end_sec))
    return {
        "schema_version": LIFECYCLE_RULE_SCHEMA,
        "available": True,
        "rule_summary": [
            "start requires PPE preparation/entry evidence: lab_coat or PPE storage together with hand/gloved_hand before core action",
            "balance/glove/object presence alone is context, not a confirmed experiment start",
            "end requires PPE exit/removal evidence after the last core action",
            "if PPE exit is not observed, the experiment remains open and the focus window extends to the last observable PPE state or video end",
            "first_person evidence may extend the end when third_person fixed view misses off-bench work",
        ],
        "start_sec": round(float(start_sec), 6),
        "end_sec": round(float(end_sec), 6),
        "duration_sec": round(max(0.0, float(end_sec) - float(start_sec)), 6),
        "start_boundary_confirmed": start_confirmed,
        "end_boundary_confirmed": end_confirmed,
        "start_reason": start_reason,
        "end_reason": end_reason,
        "first_action_sec": round(float(first_action), 6) if first_action is not None else None,
        "last_action_sec": round(float(last_action), 6),
        "last_ppe_active_sec": round(float(last_ppe_active), 6) if last_ppe_active is not None else None,
        "sample_period_sec": round(float(sample_period), 6),
        "source_duration_sec": round(float(source_duration), 6) if source_duration else None,
        "ppe_prep_evidence_count": len(ppe_prep_times),
        "ppe_active_evidence_count": len(ppe_active_times),
        "core_action_evidence_count": len(action_times),
        "exit_evidence_count": len(exit_times),
    }


def _apply_lifecycle_bounds_to_focus_window(
    manifest: SessionManifest,
    session_root: Path,
    summary: dict[str, Any],
    *,
    min_duration_sec: float,
) -> dict[str, Any]:
    lifecycle = _experiment_lifecycle_bounds(manifest, session_root, min_duration_sec=min_duration_sec)
    if lifecycle is None:
        summary["boundary_rule_schema"] = LIFECYCLE_RULE_SCHEMA
        summary["lifecycle_boundary"] = {"available": False, "reason": "no_ppe_lifecycle_evidence"}
        return summary

    current_start = float(summary.get("start_sec") or 0.0)
    current_end = float(summary.get("end_sec") or current_start)
    lifecycle_start = float(lifecycle["start_sec"])
    lifecycle_end = float(lifecycle["end_sec"])
    first_action = _safe_float(lifecycle.get("first_action_sec"))
    source = str(summary.get("source") or "")

    use_lifecycle_start = False
    if bool(lifecycle.get("start_boundary_confirmed")):
        if lifecycle_start <= current_start:
            use_lifecycle_start = True
        elif source in {"yolo_scan_coverage_fallback", "approved_material_candidate", "core_key_action_segment", "first_detected_segment"}:
            use_lifecycle_start = first_action is None or lifecycle_start <= float(first_action) + 1.0
        else:
            use_lifecycle_start = first_action is not None and lifecycle_start <= float(first_action) + 1.0
    elif current_start <= float(lifecycle.get("sample_period_sec") or 1.0) * 1.5:
        summary["start_boundary_status"] = "left_censored_no_confirmed_ppe_entry"

    if use_lifecycle_start:
        current_start = lifecycle_start
    current_end = max(current_end, lifecycle_end)
    if current_end <= current_start:
        current_end = current_start + max(0.1, float(min_duration_sec))

    pseudo = VideoSource("session", "session", manifest.session_start_time)
    summary["start_sec"] = round(current_start, 6)
    summary["true_start_sec"] = round(current_start, 6)
    summary["end_sec"] = round(current_end, 6)
    summary["true_end_sec"] = round(current_end, 6)
    summary["duration_sec"] = round(current_end - current_start, 6)
    summary["global_start_time"] = local_sec_to_global_time(pseudo, current_start).isoformat()
    summary["global_end_time"] = local_sec_to_global_time(pseudo, current_end).isoformat()
    summary["boundary_rule_schema"] = LIFECYCLE_RULE_SCHEMA
    summary["lifecycle_boundary"] = lifecycle
    summary["start_boundary_status"] = (
        "confirmed_ppe_preparation"
        if lifecycle.get("start_boundary_confirmed") and use_lifecycle_start
        else summary.get("start_boundary_status", "unconfirmed_or_action_left_censored")
    )
    summary["end_boundary_status"] = (
        "confirmed_ppe_exit"
        if lifecycle.get("end_boundary_confirmed")
        else "open_until_video_end_or_last_ppe_evidence"
    )
    return summary


def _infer_sample_period(times: list[float]) -> float:
    ordered = sorted(set(round(value, 6) for value in times))
    gaps = [
        ordered[index + 1] - ordered[index]
        for index in range(len(ordered) - 1)
        if ordered[index + 1] > ordered[index]
    ]
    if not gaps:
        return 1.0
    return max(0.1, min(5.0, sorted(gaps)[len(gaps) // 2]))


def _yolo_scan_coverage_window(
    manifest: SessionManifest,
    session_root: Path,
    *,
    min_duration_sec: float,
) -> dict[str, Any] | None:
    rows = _read_rows(session_root / "cv_outputs" / "frame_scores.jsonl")
    yolo_rows = _read_rows(session_root / "cv_outputs" / "yolo_frame_rows.jsonl")
    if yolo_rows:
        rows.extend(yolo_rows)
    times = [
        float(t)
        for row in rows
        if _row_has_yolo_lab_evidence(row)
        for t in [_row_session_time(manifest, row)]
        if t is not None
    ]
    if not times:
        return None
    sample_period = _infer_sample_period(times)
    start_sec = max(0.0, min(times))
    if start_sec <= sample_period * 1.5:
        start_sec = 0.0
    end_sec = max(times) + sample_period
    end_sec = max(end_sec, start_sec + min_duration_sec)
    global_start = local_sec_to_global_time(VideoSource("session", "session", manifest.session_start_time), start_sec)
    global_end = local_sec_to_global_time(VideoSource("session", "session", manifest.session_start_time), end_sec)
    summary = {
        "schema_version": FOCUS_WINDOW_SCHEMA,
        "detected": True,
        "source": "yolo_scan_coverage_fallback",
        "start_sec": round(start_sec, 6),
        "true_start_sec": round(start_sec, 6),
        "end_sec": round(end_sec, 6),
        "duration_sec": round(end_sec - start_sec, 6),
        "global_start_time": global_start.isoformat(),
        "global_end_time": global_end.isoformat(),
        "anchor": {
            "source": "yolo_scan_coverage",
            "start_sec": round(start_sec, 6),
            "segment_id": None,
            "primary_object": None,
            "evidence_row_count": len(times),
        },
        "included_segment_ids": [],
        "segment_count": 0,
    }
    summary = _apply_lifecycle_bounds_to_focus_window(
        manifest,
        session_root,
        summary,
        min_duration_sec=min_duration_sec,
    )
    _write_json(session_root / "metadata" / "experiment_focus_window.json", summary)
    return summary


def _yolo_hand_object_activity_bounds(manifest: SessionManifest, session_root: Path) -> tuple[float, float] | None:
    rows = _read_rows(session_root / "cv_outputs" / "frame_scores.jsonl")
    rows.extend(_read_rows(session_root / "cv_outputs" / "yolo_frame_rows.jsonl"))
    times = [
        float(t)
        for row in rows
        if _row_has_yolo_hand_object_activity(row)
        for t in [_row_session_time(manifest, row)]
        if t is not None
    ]
    if not times:
        return None
    sample_period = _infer_sample_period(times)
    start_sec = max(0.0, min(times))
    end_sec = max(times) + sample_period
    return start_sec, end_sec


def _episode_focus_window(
    manifest: SessionManifest,
    session_root: Path,
    *,
    min_duration_sec: float,
) -> dict[str, Any] | None:
    episodes = _read_rows(session_root / "metadata" / "experiment_episodes.jsonl")
    candidates = []
    for row in episodes:
        start = _safe_float(row.get("session_start_sec"))
        end = _safe_float(row.get("session_end_sec"))
        true_start = _safe_float(row.get("true_start_sec"), start)
        true_end = _safe_float(row.get("true_end_sec"), end)
        if start is None and row.get("global_start_time"):
            start = _session_sec(manifest, str(row.get("global_start_time")))
        if end is None and row.get("global_end_time"):
            end = _session_sec(manifest, str(row.get("global_end_time")))
        if true_start is None:
            true_start = start
        if true_end is None:
            true_end = end
        if true_start is None or true_end is None or true_end <= true_start:
            continue
        candidates.append((float(true_start), float(true_end), row))
    if not candidates:
        return None
    ordered = sorted(candidates, key=lambda item: item[0])
    start_sec = min(item[0] for item in ordered)
    end_sec = max(item[1] for item in ordered)
    yolo_activity_bounds = _yolo_hand_object_activity_bounds(manifest, session_root)
    if yolo_activity_bounds is not None:
        start_sec = min(start_sec, yolo_activity_bounds[0])
        end_sec = max(end_sec, yolo_activity_bounds[1])
    episode = ordered[0][2]
    global_start = local_sec_to_global_time(VideoSource("session", "session", manifest.session_start_time), start_sec)
    global_end = local_sec_to_global_time(VideoSource("session", "session", manifest.session_start_time), end_sec)
    segment_id = episode.get("segment_id") or episode.get("episode_id")
    included_segment_ids = [
        str(row.get("segment_id") or row.get("episode_id"))
        for _, _, row in ordered
        if row.get("segment_id") or row.get("episode_id")
    ]
    primary_objects: dict[str, Any] = {}
    for _, _, row in ordered:
        for label, count in dict(row.get("primary_objects") or {}).items():
            try:
                primary_objects[str(label)] = primary_objects.get(str(label), 0) + int(count)
            except (TypeError, ValueError):
                primary_objects.setdefault(str(label), count)
    summary = {
        "schema_version": FOCUS_WINDOW_SCHEMA,
        "detected": True,
        "source": "all_true_experiment_episodes",
        "episode_count": len(candidates),
        "episode_id": episode.get("episode_id"),
        "segment_id": segment_id,
        "start_sec": round(start_sec, 6),
        "true_start_sec": round(start_sec, 6),
        "end_sec": round(end_sec, 6),
        "true_end_sec": round(end_sec, 6),
        "duration_sec": round(end_sec - start_sec, 6),
        "global_start_time": global_start.isoformat(),
        "global_end_time": global_end.isoformat(),
        "anchor": {
            "source": "experiment_episode",
            "episode_id": episode.get("episode_id"),
            "segment_id": segment_id,
            "primary_objects": primary_objects,
        },
        "included_segment_ids": included_segment_ids,
        "segment_count": len(included_segment_ids),
    }
    summary = _apply_lifecycle_bounds_to_focus_window(
        manifest,
        session_root,
        summary,
        min_duration_sec=min_duration_sec,
    )
    _write_json(session_root / "metadata" / "experiment_focus_window.json", summary)
    return summary


def select_experiment_focus_window(
    session_dir: str | Path,
    *,
    buffer_before_sec: float = 0.0,
    buffer_after_sec: float = 2.0,
    min_duration_sec: float = 8.0,
    max_gap_after_anchor_sec: float = 1800.0,
) -> dict[str, Any]:
    session_root = Path(session_dir)
    manifest = SessionManifest.load(session_root / "manifest.json")
    episode_window = _episode_focus_window(manifest, session_root, min_duration_sec=min_duration_sec)
    if episode_window is not None:
        return episode_window

    segments = _read_rows(session_root / "metadata" / "key_action_segments.jsonl")
    segments_by_id = {str(row.get("segment_id")): row for row in segments if row.get("segment_id")}
    anchor = _earliest_candidate_anchor(manifest, session_root, segments_by_id) or _fallback_anchor(manifest, segments)

    all_starts: list[float] = []
    all_ends: list[float] = []
    for segment in segments:
        start = _safe_float((segment.get("cv_detection") or {}).get("start_sec"))
        end = _safe_float((segment.get("cv_detection") or {}).get("end_sec"))
        if start is None and segment.get("global_start_time"):
            start = _session_sec(manifest, str(segment.get("global_start_time")))
        if end is None and segment.get("global_end_time"):
            end = _session_sec(manifest, str(segment.get("global_end_time")))
        if start is None or end is None:
            continue
        all_starts.append(float(start))
        all_ends.append(float(end))

    if anchor is None:
        yolo_window = _yolo_scan_coverage_window(manifest, session_root, min_duration_sec=min_duration_sec)
        if yolo_window is not None:
            return yolo_window
        start_sec = min(all_starts, default=0.0)
        anchor = {"source": "no_anchor_fallback", "start_sec": start_sec, "segment_id": None, "primary_object": None}
    anchor_start = max(0.0, float(anchor.get("start_sec") or 0.0))
    window_start = max(0.0, anchor_start - max(0.0, buffer_before_sec))
    candidate_ends = [
        end
        for start, end in zip(all_starts, all_ends)
        if end >= anchor_start and (start - anchor_start) <= max_gap_after_anchor_sec
    ]
    raw_end = max(candidate_ends, default=max(all_ends, default=anchor_start + min_duration_sec))
    window_end = max(window_start + min_duration_sec, raw_end + max(0.0, buffer_after_sec))
    global_start = local_sec_to_global_time(VideoSource("session", "session", manifest.session_start_time), window_start)
    global_end = local_sec_to_global_time(VideoSource("session", "session", manifest.session_start_time), window_end)
    included = [
        str(row.get("segment_id"))
        for row in segments
        if (_safe_float((row.get("cv_detection") or {}).get("end_sec"), -1.0) or -1.0) >= window_start
        and (_safe_float((row.get("cv_detection") or {}).get("start_sec"), window_end + 1.0) or window_end + 1.0) <= window_end
    ]
    summary = {
        "schema_version": FOCUS_WINDOW_SCHEMA,
        "detected": True,
        "source": anchor.get("source"),
        "start_sec": round(window_start, 6),
        "true_start_sec": round(anchor_start, 6),
        "end_sec": round(window_end, 6),
        "duration_sec": round(window_end - window_start, 6),
        "global_start_time": global_start.isoformat(),
        "global_end_time": global_end.isoformat(),
        "anchor": anchor,
        "included_segment_ids": included,
        "segment_count": len(included),
    }
    summary = _apply_lifecycle_bounds_to_focus_window(
        manifest,
        session_root,
        summary,
        min_duration_sec=min_duration_sec,
    )
    _write_json(session_root / "metadata" / "experiment_focus_window.json", summary)
    return summary


def extract_experiment_focus_clips(
    session_dir: str | Path,
    *,
    dry_run: bool = False,
    buffer_before_sec: float = 0.0,
    buffer_after_sec: float = 2.0,
    force_extract: bool | None = None,
) -> dict[str, Any]:
    session_root = Path(session_dir)
    manifest = SessionManifest.load(session_root / "manifest.json")
    window = select_experiment_focus_window(
        session_root,
        buffer_before_sec=buffer_before_sec,
        buffer_after_sec=buffer_after_sec,
    )
    clips_dir = session_root / "clips" / "experiment_focus"
    clips_dir.mkdir(parents=True, exist_ok=True)
    views: list[tuple[str, VideoSource]] = [("third_person", manifest.videos.third_person)]
    if manifest.videos.first_person is not None:
        views.append(("first_person", manifest.videos.first_person))

    clip_rows: list[dict[str, Any]] = []
    force_clip_extract = _focus_clip_force_extract() if force_extract is None else bool(force_extract)
    max_extract_sec = _focus_clip_max_extract_sec()
    for view, source in views:
        local_start = _global_time_to_video_sec(source, str(window["global_start_time"]))
        local_end = _global_time_to_video_sec(source, str(window["global_end_time"]))
        local_start = max(0.0, float(local_start))
        try:
            local_end = min(float(local_end), get_video_duration_sec(source.path))
        except Exception:
            local_end = float(local_end)
        if local_end <= local_start:
            continue
        clip_path = clips_dir / f"{view}.mp4"
        duration_sec = max(0.0, float(local_end) - float(local_start))
        should_extract = dry_run or force_clip_extract or max_extract_sec <= 0.0 or duration_sec <= max_extract_sec
        if should_extract:
            extract_clip_ffmpeg(
                source.path,
                local_start,
                local_end,
                clip_path,
                dry_run=dry_run,
                **_focus_clip_preview_options(),
            )
            clip_file_status = "extracted"
            resolved_clip_path: str | None = str(clip_path)
        else:
            clip_file_status = "source_reference_only"
            resolved_clip_path = None
        clip_rows.append(
            {
                "view": view,
                "video_path": source.path,
                "clip_path": resolved_clip_path,
                "source_video_path": source.path,
                "source_reference": clip_file_status == "source_reference_only",
                "clip_file_status": clip_file_status,
                "extract_skipped_reason": (
                    f"duration_sec>{max_extract_sec:.3f}" if clip_file_status == "source_reference_only" else None
                ),
                "local_start_sec": round(local_start, 6),
                "local_end_sec": round(float(local_end), 6),
                "local_duration_sec": round(duration_sec, 6),
                "time_start_sec": float(window["start_sec"]),
                "time_end_sec": float(window["end_sec"]),
                "global_start_time": window["global_start_time"],
                "global_end_time": window["global_end_time"],
            }
        )

    summary = {
        "schema_version": FOCUS_CLIPS_SCHEMA,
        "available": bool(clip_rows),
        "window": window,
        "clips": clip_rows,
        "clips_by_view": {row["view"]: row for row in clip_rows},
        "extract_policy": {
            "force_extract": force_clip_extract,
            "max_extract_sec": max_extract_sec,
        },
        "extracted_clip_count": sum(1 for row in clip_rows if row.get("clip_file_status") == "extracted"),
        "source_reference_count": sum(1 for row in clip_rows if row.get("source_reference")),
    }
    _write_json(session_root / "metadata" / "experiment_focus_clips.json", summary)
    return summary


__all__ = ["extract_experiment_focus_clips", "select_experiment_focus_window"]

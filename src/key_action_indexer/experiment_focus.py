from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .clip_extractor import extract_clip_ffmpeg
from .schemas import SessionManifest, VideoSource, read_jsonl
from .time_alignment import global_time_to_local_sec, local_sec_to_global_time, parse_time
from .video_utils import get_video_duration_sec


FOCUS_WINDOW_SCHEMA = "experiment_focus_window.v1"
FOCUS_CLIPS_SCHEMA = "experiment_focus_clips.v1"
CORE_START_OBJECTS = {
    "balance",
    "beaker",
    "container",
    "reagent_bottle",
    "sample_bottle",
    "sample_bottle_blue",
    "tube",
}
PREP_OBJECTS = {"gloved_hand", "hand", "lab_coat", "paper", "ppe_storage", "spatula"}


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
    _write_json(session_root / "metadata" / "experiment_focus_window.json", summary)
    return summary


def extract_experiment_focus_clips(
    session_dir: str | Path,
    *,
    dry_run: bool = False,
    buffer_before_sec: float = 0.0,
    buffer_after_sec: float = 2.0,
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
    for view, source in views:
        local_start = global_time_to_local_sec(source, str(window["global_start_time"]))
        local_end = global_time_to_local_sec(source, str(window["global_end_time"]))
        local_start = max(0.0, float(local_start))
        try:
            local_end = min(float(local_end), get_video_duration_sec(source.path))
        except Exception:
            local_end = float(local_end)
        if local_end <= local_start:
            continue
        clip_path = clips_dir / f"{view}.mp4"
        extract_clip_ffmpeg(source.path, local_start, local_end, clip_path, dry_run=dry_run)
        clip_rows.append(
            {
                "view": view,
                "video_path": source.path,
                "clip_path": str(clip_path),
                "local_start_sec": round(local_start, 6),
                "local_end_sec": round(float(local_end), 6),
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
    }
    _write_json(session_root / "metadata" / "experiment_focus_clips.json", summary)
    return summary


__all__ = ["extract_experiment_focus_clips", "select_experiment_focus_window"]

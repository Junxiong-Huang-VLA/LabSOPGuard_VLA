from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from .chinese_index import refresh_micro_row_chinese_index, refresh_segment_chinese_index
from .clip_extractor import extract_multiview_clips
from .description_builder import build_segment_description
from .evidence import apply_segment_evidence
from .micro_segmenter import micro_row_to_vector_metadata
from .schemas import (
    DetectedSegment,
    KeyActionSegment,
    SessionManifest,
    VideoSource,
    read_jsonl,
    to_json_dict,
    write_jsonl,
)
from .time_alignment import find_dialogue_for_segment, local_sec_to_global_time, parse_time
from .transcript import TranscriptUtterance
from .vector_index import VectorIndex
from .video_utils import get_video_duration_sec


EPISODE_SCHEMA_VERSION = "key_action_experiment_episode.v2"
SUMMARY_SCHEMA_VERSION = "key_action_episode_segmentation_summary.v1"
ACTION_WINDOW_SCHEMA_VERSION = "key_action_candidate_action_window.v1"
_FOCUS_HAND_LABELS = {"gloved_hand", "hand"}
_FOCUS_CORE_OBJECTS = {
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
_DEFAULT_SHORT_EPISODE_MERGE_GAP_SEC = 30.0
_DEFAULT_INTERNAL_EPISODE_MERGE_GAP_SEC = 120.0
_DEFAULT_EPISODE_GAP_OUTLIER_MULTIPLIER = 3.0
_DEFAULT_DENSE_EPISODE_MERGE_MIN_FRAGMENTS = 6
_STRONG_EVIDENCE_LEVELS = {"trusted", "visual_confirmed", "visual_and_transcript_confirmed"}
_WEAK_EVIDENCE_LEVELS = {"weak_visual_evidence", "insufficient", "insufficient_evidence", "low"}
_PARTIAL_EVIDENCE_LEVELS = {"transcript_supported"}
_PARTIAL_PROCESS_ROLES = {"retrieval_candidate", "supporting_process_candidate"}
_DEFAULT_MIN_OFFICIAL_EPISODE_DURATION_SEC = 120.0
_DEFAULT_EXPERIMENT_WINDOW_SILENCE_GAP_SEC = 30.0
_DEFAULT_EXPERIMENT_WINDOW_ATTACH_GAP_SEC = 10.0
_DEFAULT_EXPERIMENT_WINDOW_ACTIVITY_MIN_SCORE = 0.05
_DEFAULT_EXPERIMENT_LIFECYCLE_GAP_SEC = 120.0
_DEFAULT_EXPERIMENT_LIFECYCLE_ATTACH_GAP_SEC = 90.0
_REQUIRED_FORMAL_EPISODE_VIEWS = {"first_person", "third_person"}
_LIFECYCLE_STATE_LABELS = {"gloved_hand", "lab_coat"}
_LIFECYCLE_PREP_CONTEXT_LABELS = {"ppe_storage", "lab_coat"}
_LIFECYCLE_EXIT_LABELS = {"hand", "ppe_storage"}


def rebuild_episode_segments_from_micro_evidence(
    *,
    manifest: SessionManifest,
    session_dir: str | Path,
    key_segments: list[KeyActionSegment],
    micro_rows: list[dict[str, Any]],
    yolo_frame_rows: list[dict[str, Any]] | None,
    utterances: list[TranscriptUtterance],
    detector_summary: dict[str, Any] | None = None,
    dry_run: bool = False,
    gap_sec: float = 7.0,
    pre_roll_sec: float = 2.0,
    post_roll_sec: float = 3.0,
    min_episode_duration_sec: float = _DEFAULT_MIN_OFFICIAL_EPISODE_DURATION_SEC,
    min_micro_evidence_count: int = 2,
    expected_experiment_count: int | None = None,
) -> dict[str, Any]:
    """Rebuild parent key-action segments as true experiment episodes.

    The detector may return one coarse segment covering most of a long video.
    This pass treats YOLO-backed micro windows as the source of truth, clusters
    them into operation episodes, and rewrites all parent/micro/index artifacts.
    """

    session_root = Path(session_dir)
    metadata_dir = session_root / "metadata"
    min_episode_duration_sec = _official_experiment_window_min_sec(min_episode_duration_sec)
    all_episode_specs = _episode_specs_from_micros(
        manifest,
        micro_rows,
        activity_rows=yolo_frame_rows or [],
        coarse_key_segments=key_segments,
        gap_sec=gap_sec,
        pre_roll_sec=pre_roll_sec,
        post_roll_sec=post_roll_sec,
        min_episode_duration_sec=min_episode_duration_sec,
        min_micro_evidence_count=min_micro_evidence_count,
        include_candidates=True,
    )
    episode_specs = [spec for spec in all_episode_specs if _is_official_episode_spec(spec)]
    candidate_specs = [spec for spec in all_episode_specs if not _is_official_episode_spec(spec)]
    candidate_rows = _candidate_action_window_rows(manifest, candidate_specs)
    write_jsonl(metadata_dir / "candidate_action_windows.jsonl", candidate_rows)
    write_jsonl(metadata_dir / "experiment_episode_candidates.jsonl", candidate_rows)
    if not episode_specs:
        write_jsonl(metadata_dir / "experiment_episodes.jsonl", [])
        write_jsonl(session_root / "cv_outputs" / "detected_segments.jsonl", [])
        summary = {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "rebuilt": False,
            "reason": "no_official_episode_from_activity_layer",
            "episode_count": 0,
            "official_episode_count": 0,
            "candidate_action_window_count": len(candidate_rows),
            "micro_segment_count": len(micro_rows),
            "min_micro_evidence_count": min_micro_evidence_count,
            "min_official_episode_duration_sec": min_episode_duration_sec,
            "episode_window_expansion": _episode_window_expansion_stats(all_episode_specs),
            "candidate_reasons": dict(
                Counter(
                    reason
                    for spec in candidate_specs
                    for reason in list(spec.get("candidate_reasons") or [])
                    if reason
                )
            ),
        }
        _write_json(metadata_dir / "episode_segmentation_summary.json", summary)
        return summary

    source_duration_sec = _source_duration_sec(manifest, episode_specs, key_segments, micro_rows)
    episode_segments = [
        _detected_segment_from_spec(manifest, spec, index, source_duration_sec=source_duration_sec)
        for index, spec in enumerate(episode_specs, start=1)
    ]
    episode_key_segments: list[KeyActionSegment] = []
    for segment in episode_segments:
        key_segment = extract_multiview_clips(
            manifest=manifest,
            segment=segment,
            clips_dir=session_root / "clips" / "episodes",
            keyframes_dir=session_root / "keyframes" / "episodes",
            yolo_frame_rows=yolo_frame_rows,
            dry_run=dry_run,
        )
        dialogue = find_dialogue_for_segment(segment.global_start_time, segment.global_end_time, utterances)
        key_segment = build_segment_description(key_segment, dialogue)
        key_segment = apply_segment_evidence(key_segment)
        refresh_segment_chinese_index(key_segment)
        episode_key_segments.append(key_segment)

    remapped_micros, micro_map = _remap_micro_rows_to_episodes(manifest, micro_rows, episode_specs)
    _attach_micro_refs_to_parent_dicts(episode_key_segments, remapped_micros)
    for key_segment in episode_key_segments:
        refresh_segment_chinese_index(key_segment)

    segment_rows = _segment_dicts(episode_key_segments, episode_specs, source_duration_sec)
    remapped_micros = [refresh_micro_row_chinese_index(row) for row in remapped_micros]
    _attach_micro_refs_to_segment_rows(segment_rows, remapped_micros)
    episode_rows = _episode_rows(manifest, segment_rows, episode_specs, detector_summary or {}, source_duration_sec)

    write_jsonl(metadata_dir / "key_action_segments.jsonl", segment_rows)
    write_jsonl(metadata_dir / "micro_segments.jsonl", remapped_micros)
    write_jsonl(metadata_dir / "experiment_episodes.jsonl", episode_rows)
    write_jsonl(session_root / "cv_outputs" / "detected_segments.jsonl", episode_segments)

    segment_vectors = [_segment_vector_metadata_from_row(row) for row in segment_rows]
    micro_vectors = [micro_row_to_vector_metadata(row) for row in remapped_micros]
    combined_vectors = [*segment_vectors, *micro_vectors]
    write_jsonl(metadata_dir / "vector_metadata.jsonl", combined_vectors)
    write_jsonl(metadata_dir / "micro_vector_metadata.jsonl", micro_vectors)
    _rebuild_indexes(session_root, segment_vectors, micro_vectors, combined_vectors)
    _rewrite_parent_segment_artifacts(session_root, micro_map)
    focus_summary = _write_first_episode_focus(
        manifest,
        session_root,
        episode_rows,
        yolo_frame_rows=yolo_frame_rows or [],
        dry_run=dry_run,
    )

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "rebuilt": True,
        "source": "activity_episode_from_yolo_action_windows",
        "episode_count": len(episode_rows),
        "official_episode_count": len(episode_rows),
        "candidate_action_window_count": len(candidate_rows),
        "micro_segment_count": len(remapped_micros),
        "source_video_duration_sec": source_duration_sec,
        "gap_sec": gap_sec,
        "pre_roll_sec": pre_roll_sec,
        "post_roll_sec": post_roll_sec,
        "min_official_episode_duration_sec": min_episode_duration_sec,
        "micro_parent_remap_count": len(micro_map),
        "expected_experiment_count": expected_experiment_count,
        "expected_experiment_count_applied": False,
        "episode_merge_stats": _episode_merge_stats(episode_specs),
        "episode_window_expansion": _episode_window_expansion_stats(episode_specs),
        "episode_ids": [row.get("episode_id") for row in episode_rows],
        "focus": focus_summary,
    }
    _write_json(metadata_dir / "episode_segmentation_summary.json", summary)
    return summary


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _float_env(name: str, default: float, *, minimum: float | None = None) -> float:
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = float(default)
    if minimum is not None:
        value = max(float(minimum), value)
    return value


def _float_env_any(names: tuple[str, ...], default: float, *, minimum: float | None = None) -> float:
    value = float(default)
    for name in names:
        raw = os.environ.get(name)
        if raw is None:
            continue
        try:
            value = float(raw)
            break
        except (TypeError, ValueError):
            value = float(default)
            break
    if minimum is not None:
        value = max(float(minimum), value)
    return value


def _official_experiment_window_min_sec(default: float = _DEFAULT_MIN_OFFICIAL_EPISODE_DURATION_SEC) -> float:
    return _float_env_any(
        (
            "KEY_ACTION_EXPERIMENT_WINDOW_MIN_SEC",
            "KEY_ACTION_MIN_OFFICIAL_EXPERIMENT_DURATION_SEC",
            "KEY_ACTION_FAST_LOCATE_MIN_EXPERIMENT_EPISODE_SEC",
            "KEY_ACTION_EPISODE_MIN_OFFICIAL_DURATION_SEC",
        ),
        default,
        minimum=0.0,
    )


def _experiment_window_silence_gap_sec() -> float:
    return _float_env_any(
        (
            "KEY_ACTION_EXPERIMENT_WINDOW_SILENCE_GAP_SEC",
            "KEY_ACTION_ACTIVITY_VALLEY_GAP_SEC",
            "KEY_ACTION_EPISODE_ACTIVITY_VALLEY_GAP_SEC",
        ),
        _DEFAULT_EXPERIMENT_WINDOW_SILENCE_GAP_SEC,
        minimum=0.5,
    )


def _experiment_window_attach_gap_sec() -> float:
    return _float_env_any(
        (
            "KEY_ACTION_EXPERIMENT_WINDOW_ATTACH_GAP_SEC",
            "KEY_ACTION_ACTIVITY_WINDOW_ATTACH_GAP_SEC",
        ),
        _DEFAULT_EXPERIMENT_WINDOW_ATTACH_GAP_SEC,
        minimum=0.0,
    )


def _experiment_window_activity_min_score() -> float:
    return _float_env_any(
        (
            "KEY_ACTION_EXPERIMENT_WINDOW_ACTIVITY_MIN_SCORE",
            "KEY_ACTION_ACTIVITY_WINDOW_MIN_SCORE",
        ),
        _DEFAULT_EXPERIMENT_WINDOW_ACTIVITY_MIN_SCORE,
        minimum=0.0,
    )


def _experiment_lifecycle_gap_sec() -> float:
    return _float_env_any(
        (
            "KEY_ACTION_EXPERIMENT_LIFECYCLE_GAP_SEC",
            "KEY_ACTION_EXPERIMENT_STATE_GAP_SEC",
        ),
        _DEFAULT_EXPERIMENT_LIFECYCLE_GAP_SEC,
        minimum=5.0,
    )


def _experiment_lifecycle_attach_gap_sec() -> float:
    return _float_env_any(
        (
            "KEY_ACTION_EXPERIMENT_LIFECYCLE_ATTACH_GAP_SEC",
            "KEY_ACTION_EXPERIMENT_STATE_ATTACH_GAP_SEC",
        ),
        _DEFAULT_EXPERIMENT_LIFECYCLE_ATTACH_GAP_SEC,
        minimum=0.0,
    )


def _session_sec(manifest: SessionManifest, global_time: str | None) -> float | None:
    if not global_time:
        return None
    try:
        return (parse_time(str(global_time)) - parse_time(manifest.session_start_time)).total_seconds()
    except Exception:
        return None


def _valid_interval(start: Any, end: Any) -> tuple[float, float] | None:
    start_value = _safe_float(start)
    end_value = _safe_float(end)
    if start_value is None or end_value is None or end_value <= start_value:
        return None
    return max(0.0, float(start_value)), max(0.0, float(end_value))


def _norm_activity_label(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in text:
        text = text.replace("__", "_")
    if text == "ppe":
        return "ppe_storage"
    if text == "weighing_paper":
        return "paper"
    return text


def _micro_interval_details(manifest: SessionManifest, row: Mapping[str, Any]) -> dict[str, Any] | None:
    """Resolve micro evidence onto the shared session timeline."""
    global_interval = _valid_interval(
        _session_sec(manifest, str(row.get("global_start_time") or "")),
        _session_sec(manifest, str(row.get("global_end_time") or "")),
    )
    if global_interval is not None:
        return {
            "start": global_interval[0],
            "end": global_interval[1],
            "timeline": "session",
            "source": "global_time",
            "source_fields": ["global_start_time", "global_end_time"],
        }

    for source, start_key, end_key in (
        ("session_fields", "session_start_sec", "session_end_sec"),
        ("alignment_fields", "alignment_start_sec", "alignment_end_sec"),
        ("session_fields", "session_time_start_sec", "session_time_end_sec"),
    ):
        interval = _valid_interval(row.get(start_key), row.get(end_key))
        if interval is not None:
            return {
                "start": interval[0],
                "end": interval[1],
                "timeline": "session",
                "source": source,
                "source_fields": [start_key, end_key],
            }

    interaction = row.get("interaction") if isinstance(row.get("interaction"), Mapping) else {}
    contact_interval = _valid_interval(interaction.get("contact_start_sec"), interaction.get("contact_end_sec"))
    if contact_interval is not None:
        return {
            "start": contact_interval[0],
            "end": contact_interval[1],
            "timeline": "session",
            "source": "interaction_contact_fields",
            "source_fields": ["interaction.contact_start_sec", "interaction.contact_end_sec"],
        }

    local_interval = _valid_interval(row.get("start_sec"), row.get("end_sec"))
    if local_interval is not None:
        return {
            "start": local_interval[0],
            "end": local_interval[1],
            "timeline": "session_or_local_fallback",
            "source": "start_end_fields",
            "source_fields": ["start_sec", "end_sec"],
        }
    return None


def _micro_interval(manifest: SessionManifest, row: Mapping[str, Any]) -> tuple[float, float] | None:
    details = _micro_interval_details(manifest, row)
    if details is None:
        return None
    return float(details["start"]), float(details["end"])


def _micro_primary_object(row: Mapping[str, Any]) -> str:
    interaction = row.get("interaction") if isinstance(row.get("interaction"), Mapping) else {}
    return str(row.get("primary_object") or interaction.get("primary_object") or "").strip()


def _micro_has_physical_evidence(row: Mapping[str, Any]) -> bool:
    if _micro_primary_object(row):
        return True
    interaction = row.get("interaction") if isinstance(row.get("interaction"), Mapping) else {}
    if interaction.get("detected_objects"):
        return True
    return bool(row.get("yolo_evidence"))


def _activity_row_time_sec(manifest: SessionManifest, row: Mapping[str, Any]) -> float | None:
    for key in ("alignment_time_sec", "session_time_sec", "session_start_sec", "time_sec", "local_time_sec"):
        value = _safe_float(row.get(key))
        if value is not None:
            return max(0.0, float(value))
    for key in ("global_time", "global_start_time"):
        value = _session_sec(manifest, str(row.get(key) or ""))
        if value is not None:
            return max(0.0, float(value))
    return None


def _activity_row_labels(row: Mapping[str, Any]) -> set[str]:
    labels: set[str] = set()
    counts = row.get("label_counts") if isinstance(row.get("label_counts"), Mapping) else {}
    for label, count in dict(counts).items():
        try:
            if int(count) <= 0:
                continue
        except (TypeError, ValueError):
            continue
        labels.add(_norm_activity_label(label))
    for detection in row.get("detections") or []:
        if not isinstance(detection, Mapping):
            continue
        label = str(detection.get("label") or detection.get("class_name") or detection.get("raw_label") or "").strip()
        if label:
            labels.add(_norm_activity_label(label))
    for interaction in row.get("hand_object_interactions") or []:
        if not isinstance(interaction, Mapping):
            continue
        for key in ("object_label", "object_name", "label", "hand_label"):
            label = str(interaction.get(key) or "").strip()
            if label:
                labels.add(_norm_activity_label(label))
    return labels


def _activity_row_score(row: Mapping[str, Any]) -> float:
    score = 0.0
    for key in ("active_score", "interaction_score", "parent_activity_score", "motion_score", "raw_yolo_active_score", "probability"):
        value = _safe_float(row.get(key))
        if value is not None:
            score = max(score, float(value))
    return score


def _row_has_activity_evidence(row: Mapping[str, Any]) -> bool:
    interactions = row.get("hand_object_interactions")
    if isinstance(interactions, list) and interactions:
        return True
    if bool(row.get("is_active") or row.get("is_experiment_active")):
        return True
    labels = _activity_row_labels(row)
    if (labels & _FOCUS_HAND_LABELS) and (labels & _FOCUS_CORE_OBJECTS):
        return True
    return _activity_row_score(row) >= _experiment_window_activity_min_score()


def _activity_row_view(row: Mapping[str, Any]) -> str:
    for key in ("source_view", "view", "camera_view", "camera_role"):
        value = str(row.get(key) or "").strip().lower()
        if value in {"first", "operator", "operator_view", "first_person"}:
            return "first_person"
        if value in {"third", "top", "top_view", "third_person"}:
            return "third_person"
    return ""


def _row_has_lifecycle_state_evidence(row: Mapping[str, Any]) -> bool:
    labels = _activity_row_labels(row)
    if labels & _LIFECYCLE_STATE_LABELS:
        return True
    if "ppe_storage" in labels and labels & _FOCUS_HAND_LABELS:
        return True
    interactions = row.get("hand_object_interactions")
    if isinstance(interactions, list) and interactions:
        return True
    if bool(row.get("is_experiment_active")):
        return True
    return False


def _row_has_lifecycle_prep_evidence(row: Mapping[str, Any]) -> bool:
    labels = _activity_row_labels(row)
    if labels & _LIFECYCLE_PREP_CONTEXT_LABELS and labels & _FOCUS_HAND_LABELS:
        return True
    if "lab_coat" in labels:
        return True
    return False


def _row_has_lifecycle_exit_evidence(row: Mapping[str, Any]) -> bool:
    labels = _activity_row_labels(row)
    return bool(labels & _LIFECYCLE_EXIT_LABELS) and not bool(labels & _LIFECYCLE_STATE_LABELS)


def _lifecycle_windows_from_rows(
    manifest: SessionManifest,
    rows: list[Mapping[str, Any]],
    source_duration: float | None,
) -> list[dict[str, Any]]:
    timed_rows: list[tuple[float, Mapping[str, Any]]] = []
    for row in rows:
        if not isinstance(row, Mapping) or not _row_has_lifecycle_state_evidence(row):
            continue
        time_sec = _activity_row_time_sec(manifest, row)
        if time_sec is not None:
            timed_rows.append((float(time_sec), row))
    if not timed_rows:
        return []

    timed_rows.sort(key=lambda item: item[0])
    sample_period = _median_sample_period([time_sec for time_sec, _row in timed_rows])
    gap_sec = _experiment_lifecycle_gap_sec()
    groups: list[list[tuple[float, Mapping[str, Any]]]] = []
    current: list[tuple[float, Mapping[str, Any]]] = []
    last_time: float | None = None
    for item in timed_rows:
        time_sec = item[0]
        if current and last_time is not None and time_sec - last_time > gap_sec:
            groups.append(current)
            current = []
        current.append(item)
        last_time = time_sec
    if current:
        groups.append(current)

    windows: list[dict[str, Any]] = []
    for index, group in enumerate(groups, start=1):
        start_sec = max(0.0, min(time_sec for time_sec, _row in group) - sample_period)
        end_sec = max(time_sec for time_sec, _row in group) + sample_period
        if source_duration:
            end_sec = min(float(source_duration), end_sec)
        prep_times = [time_sec for time_sec, row in group if _row_has_lifecycle_prep_evidence(row)]
        exit_times = [time_sec for time_sec, row in group if _row_has_lifecycle_exit_evidence(row)]
        view_counts = Counter(_activity_row_view(row) for _time_sec, row in group)
        view_counts.pop("", None)
        windows.append(
            {
                "window_id": f"lifecycle_window_{index:06d}",
                "start_sec": round(start_sec, 6),
                "end_sec": round(max(end_sec, start_sec + 0.1), 6),
                "row_count": len(group),
                "sample_period_sec": round(sample_period, 6),
                "state_gap_sec": round(gap_sec, 6),
                "prep_evidence_count": len(prep_times),
                "exit_evidence_count": len(exit_times),
                "prep_start_sec": round(min(prep_times), 6) if prep_times else None,
                "exit_start_sec": round(min(exit_times), 6) if exit_times else None,
                "source_views": sorted(view_counts),
                "source_view_counts": dict(sorted(view_counts.items())),
                "source": "experiment_lifecycle_state_rows",
            }
        )
    return windows


def _lifecycle_window_for_action(
    lifecycle_windows: list[dict[str, Any]],
    *,
    action_start: float,
    action_end: float,
) -> dict[str, Any] | None:
    if not lifecycle_windows:
        return None
    attach_gap_sec = _experiment_lifecycle_attach_gap_sec()
    matches = [
        window
        for window in lifecycle_windows
        if action_start <= float(window.get("end_sec") or 0.0) + attach_gap_sec
        and action_end >= float(window.get("start_sec") or 0.0) - attach_gap_sec
    ]
    if not matches:
        return None
    matches.sort(
        key=lambda window: (
            0
            if float(window.get("start_sec") or 0.0) <= action_start <= float(window.get("end_sec") or 0.0)
            else 1,
            abs(((float(window.get("start_sec") or 0.0) + float(window.get("end_sec") or 0.0)) / 2.0) - action_start),
        )
    )
    window = dict(matches[0])
    start_sec = float(window.get("start_sec") or 0.0)
    prep_start = _safe_float(window.get("prep_start_sec"))
    if prep_start is not None and prep_start <= action_start + 1.0:
        start_sec = min(start_sec, float(prep_start))
        start_reason = "ppe_or_lab_state_preparation_before_core_action"
        start_confirmed = True
    else:
        start_reason = "earliest_experiment_state_before_or_near_core_action"
        start_confirmed = False

    end_sec = float(window.get("end_sec") or start_sec)
    exit_start = _safe_float(window.get("exit_start_sec"))
    if exit_start is not None and exit_start >= action_end:
        end_sec = min(end_sec, float(exit_start) + float(window.get("sample_period_sec") or 0.0))
        end_reason = "ppe_exit_after_last_core_action"
        end_confirmed = True
    else:
        end_reason = "last_experiment_state_without_visible_ppe_exit"
        end_confirmed = False

    return {
        **window,
        "selected_start_sec": round(start_sec, 6),
        "selected_end_sec": round(max(end_sec, start_sec + 0.1), 6),
        "start_reason": start_reason,
        "end_reason": end_reason,
        "start_boundary_confirmed": start_confirmed,
        "end_boundary_confirmed": end_confirmed,
        "action_start_sec": round(float(action_start), 6),
        "action_end_sec": round(float(action_end), 6),
        "attach_gap_sec": round(attach_gap_sec, 6),
    }


def _median_sample_period(times: list[float]) -> float:
    ordered = sorted(set(round(value, 6) for value in times))
    gaps = [
        ordered[index + 1] - ordered[index]
        for index in range(len(ordered) - 1)
        if ordered[index + 1] > ordered[index]
    ]
    if not gaps:
        return 1.0
    return max(0.1, min(10.0, sorted(gaps)[len(gaps) // 2]))


def _activity_windows_from_rows(
    manifest: SessionManifest,
    rows: list[Mapping[str, Any]],
    source_duration: float | None,
) -> list[dict[str, Any]]:
    timed_rows: list[tuple[float, Mapping[str, Any]]] = []
    for row in rows:
        if not isinstance(row, Mapping) or not _row_has_activity_evidence(row):
            continue
        time_sec = _activity_row_time_sec(manifest, row)
        if time_sec is not None:
            timed_rows.append((float(time_sec), row))
    if not timed_rows:
        return []

    timed_rows.sort(key=lambda item: item[0])
    sample_period = _median_sample_period([time_sec for time_sec, _row in timed_rows])
    silence_gap_sec = _experiment_window_silence_gap_sec()
    groups: list[list[tuple[float, Mapping[str, Any]]]] = []
    current: list[tuple[float, Mapping[str, Any]]] = []
    last_time: float | None = None
    for item in timed_rows:
        time_sec = item[0]
        if current and last_time is not None and time_sec - last_time > silence_gap_sec:
            groups.append(current)
            current = []
        current.append(item)
        last_time = time_sec
    if current:
        groups.append(current)

    windows: list[dict[str, Any]] = []
    for index, group in enumerate(groups, start=1):
        start_sec = max(0.0, min(time_sec for time_sec, _row in group) - sample_period)
        end_sec = max(time_sec for time_sec, _row in group) + sample_period
        if source_duration:
            end_sec = min(float(source_duration), end_sec)
        windows.append(
            {
                "window_id": f"activity_window_{index:06d}",
                "start_sec": round(start_sec, 6),
                "end_sec": round(max(end_sec, start_sec + 0.1), 6),
                "row_count": len(group),
                "sample_period_sec": round(sample_period, 6),
                "silence_gap_sec": round(silence_gap_sec, 6),
                "source": "yolo_activity_rows",
            }
        )
    return windows


def _coarse_windows_from_key_segments(key_segments: list[KeyActionSegment]) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    for index, segment in enumerate(key_segments, start=1):
        cv = getattr(segment, "cv_detection", None)
        start = _safe_float(getattr(cv, "start_sec", None))
        end = _safe_float(getattr(cv, "end_sec", None))
        if start is None:
            start = _safe_float(getattr(segment, "start_sec", None))
        if end is None:
            end = _safe_float(getattr(segment, "end_sec", None))
        if start is None or end is None or end <= start:
            continue
        windows.append(
            {
                "window_id": f"coarse_segment_{index:06d}",
                "start_sec": round(max(0.0, float(start)), 6),
                "end_sec": round(max(float(end), float(start) + 0.1), 6),
                "source": "coarse_key_action_segment",
                "segment_id": str(getattr(segment, "segment_id", "") or ""),
            }
        )
    return windows


def _intervals_touch(start: float, end: float, window: Mapping[str, Any], *, attach_gap_sec: float) -> bool:
    window_start = float(window.get("start_sec") or 0.0)
    window_end = float(window.get("end_sec") or window_start)
    return start <= window_end + attach_gap_sec and end >= window_start - attach_gap_sec


def _duration_limit_for_expansion(
    source_duration: float | None,
    _specs: list[dict[str, Any]],
    activity_windows: list[dict[str, Any]],
    coarse_windows: list[dict[str, Any]],
) -> float | None:
    if source_duration and source_duration > 0:
        return float(source_duration)
    ends: list[float] = []
    for collection in (activity_windows, coarse_windows):
        for item in collection:
            end = _safe_float(item.get("end_sec"))
            if end is not None:
                ends.append(float(end))
    return max(ends) if ends else None


def _expand_interval_to_min_duration(
    start_sec: float,
    end_sec: float,
    *,
    min_duration_sec: float,
    duration_limit_sec: float | None,
) -> tuple[float, float]:
    start_sec = max(0.0, float(start_sec))
    end_sec = max(start_sec + 0.1, float(end_sec))
    current_duration = end_sec - start_sec
    if current_duration >= min_duration_sec:
        return start_sec, end_sec
    target_duration = float(min_duration_sec)
    if duration_limit_sec is not None and duration_limit_sec > 0:
        target_duration = min(target_duration, float(duration_limit_sec))
    if target_duration <= current_duration:
        return start_sec, end_sec

    missing = target_duration - current_duration
    expanded_start = max(0.0, start_sec - missing / 2.0)
    expanded_end = end_sec + missing / 2.0
    if duration_limit_sec is not None and duration_limit_sec > 0 and expanded_end > duration_limit_sec:
        overflow = expanded_end - float(duration_limit_sec)
        expanded_end = float(duration_limit_sec)
        expanded_start = max(0.0, expanded_start - overflow)
    if expanded_start <= 0.0:
        expanded_end = max(expanded_end, min(target_duration, float(duration_limit_sec or target_duration)))
    return expanded_start, max(expanded_end, expanded_start + 0.1)


def _compact_window_refs(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "window_id": item.get("window_id"),
            "source": item.get("source"),
            "start_sec": item.get("start_sec"),
            "end_sec": item.get("end_sec"),
            "row_count": item.get("row_count"),
            "segment_id": item.get("segment_id"),
        }
        for item in windows[:10]
    ]


def _expand_episode_specs_to_activity_windows(
    manifest: SessionManifest,
    specs: list[dict[str, Any]],
    *,
    activity_rows: list[Mapping[str, Any]],
    coarse_key_segments: list[KeyActionSegment],
    source_duration: float | None,
    min_window_sec: float,
) -> list[dict[str, Any]]:
    if not specs:
        return specs
    activity_windows = _activity_windows_from_rows(manifest, activity_rows, source_duration)
    lifecycle_windows = _lifecycle_windows_from_rows(manifest, activity_rows, source_duration)
    coarse_windows = _coarse_windows_from_key_segments(coarse_key_segments)
    duration_limit = _duration_limit_for_expansion(source_duration, specs, activity_windows, coarse_windows)
    if not source_duration and lifecycle_windows:
        lifecycle_limit = max(float(item.get("end_sec") or 0.0) for item in lifecycle_windows)
        duration_limit = max(float(duration_limit or 0.0), lifecycle_limit)
    attach_gap_sec = _experiment_window_attach_gap_sec()
    max_coarse_fallback_sec = _float_env_any(
        ("KEY_ACTION_EXPERIMENT_WINDOW_MAX_COARSE_FALLBACK_SEC",),
        max(float(min_window_sec) * 2.0, 180.0),
        minimum=max(float(min_window_sec), 1.0),
    )
    max_activity_window_sec = _float_env_any(
        ("KEY_ACTION_EXPERIMENT_WINDOW_MAX_ACTIVITY_CONTEXT_SEC",),
        max(float(min_window_sec) * 4.0, 300.0),
        minimum=max(float(min_window_sec), 1.0),
    )
    expanded_specs: list[dict[str, Any]] = []
    for spec in specs:
        item = dict(spec)
        original_start = float(item.get("start_sec") or 0.0)
        original_end = max(original_start + 0.1, float(item.get("end_sec") or original_start))
        action_start = float(item.get("true_start_sec") or original_start)
        action_end = max(action_start + 0.1, float(item.get("true_end_sec") or original_end))
        expanded_start = original_start
        expanded_end = original_end

        matched_activity = [
            window
            for window in activity_windows
            if _intervals_touch(action_start, action_end, window, attach_gap_sec=attach_gap_sec)
            and float(window.get("end_sec") or 0.0) - float(window.get("start_sec") or 0.0) <= max_activity_window_sec
        ]
        matched_coarse = [
            window
            for window in coarse_windows
            if _intervals_touch(action_start, action_end, window, attach_gap_sec=attach_gap_sec)
            and float(window.get("end_sec") or 0.0) - float(window.get("start_sec") or 0.0) <= max_coarse_fallback_sec
        ]
        for window in [*matched_activity, *matched_coarse]:
            expanded_start = min(expanded_start, float(window.get("start_sec") or expanded_start))
            expanded_end = max(expanded_end, float(window.get("end_sec") or expanded_end))
        matched_lifecycle = _lifecycle_window_for_action(
            lifecycle_windows,
            action_start=action_start,
            action_end=action_end,
        )
        if matched_lifecycle:
            expanded_start = min(expanded_start, float(matched_lifecycle.get("selected_start_sec") or expanded_start))
            expanded_end = max(expanded_end, float(matched_lifecycle.get("selected_end_sec") or expanded_end))
        has_external_window_context = bool(matched_activity or matched_coarse or (source_duration and source_duration > 0))
        if matched_lifecycle:
            has_external_window_context = True
        if has_external_window_context:
            expanded_start, expanded_end = _expand_interval_to_min_duration(
                expanded_start,
                expanded_end,
                min_duration_sec=float(min_window_sec),
                duration_limit_sec=duration_limit,
            )
        if duration_limit is not None and duration_limit > 0:
            expanded_end = min(expanded_end, float(duration_limit))
        changed = abs(expanded_start - original_start) > 1e-6 or abs(expanded_end - original_end) > 1e-6
        if changed:
            item["action_start_sec"] = round(action_start, 6)
            item["action_end_sec"] = round(action_end, 6)
            item["action_duration_sec"] = round(max(0.0, action_end - action_start), 6)
            item["start_sec"] = round(expanded_start, 6)
            item["end_sec"] = round(max(expanded_end, expanded_start + 0.1), 6)
            item["true_start_sec"] = item["start_sec"]
            item["true_end_sec"] = item["end_sec"]
            item["duration_sec"] = round(float(item["end_sec"]) - float(item["start_sec"]), 6)
            reasons = _ordered_unique([*_as_strings(item.get("episode_merge_reasons")), "official_episode_window_expanded_to_activity_boundary"])
            item["episode_merge_reasons"] = reasons
            boundary_evidence = dict(_as_mapping(item.get("boundary_evidence")))
            boundary_evidence["action_start_sec"] = round(action_start, 6)
            boundary_evidence["action_end_sec"] = round(action_end, 6)
            boundary_evidence["official_start_sec"] = item["start_sec"]
            boundary_evidence["official_end_sec"] = item["end_sec"]
            boundary_evidence["official_boundary_source"] = (
                "experiment_lifecycle_state_window"
                if matched_lifecycle
                else "coarse_activity_or_min_duration_window"
            )
            if matched_lifecycle:
                boundary_evidence["lifecycle_boundary"] = {
                    "window_id": matched_lifecycle.get("window_id"),
                    "start_reason": matched_lifecycle.get("start_reason"),
                    "end_reason": matched_lifecycle.get("end_reason"),
                    "start_boundary_confirmed": matched_lifecycle.get("start_boundary_confirmed"),
                    "end_boundary_confirmed": matched_lifecycle.get("end_boundary_confirmed"),
                    "source_views": matched_lifecycle.get("source_views"),
                    "source_view_counts": matched_lifecycle.get("source_view_counts"),
                }
            item["boundary_evidence"] = boundary_evidence
        item["episode_window_expansion"] = {
            "schema_version": "official_experiment_window_expansion.v1",
            "expanded": bool(changed),
            "source": (
                "experiment_lifecycle_state_and_action_evidence"
                if matched_lifecycle
                else "coarse_activity_evidence_and_min_duration"
            ),
            "original_start_sec": round(original_start, 6),
            "original_end_sec": round(original_end, 6),
            "action_start_sec": round(action_start, 6),
            "action_end_sec": round(action_end, 6),
            "expanded_start_sec": round(float(item.get("start_sec") or expanded_start), 6),
            "expanded_end_sec": round(float(item.get("end_sec") or expanded_end), 6),
            "min_window_sec": round(float(min_window_sec), 6),
            "silence_gap_sec": round(_experiment_window_silence_gap_sec(), 6),
            "attach_gap_sec": round(attach_gap_sec, 6),
            "max_activity_context_sec": round(float(max_activity_window_sec), 6),
            "max_coarse_fallback_sec": round(float(max_coarse_fallback_sec), 6),
            "activity_window_count": len(matched_activity),
            "coarse_window_count": len(matched_coarse),
            "lifecycle_window_count": 1 if matched_lifecycle else 0,
            "activity_windows": _compact_window_refs(matched_activity),
            "coarse_windows": _compact_window_refs(matched_coarse),
            "lifecycle_window": (
                {
                    "window_id": matched_lifecycle.get("window_id"),
                    "start_sec": matched_lifecycle.get("selected_start_sec"),
                    "end_sec": matched_lifecycle.get("selected_end_sec"),
                    "start_reason": matched_lifecycle.get("start_reason"),
                    "end_reason": matched_lifecycle.get("end_reason"),
                    "start_boundary_confirmed": matched_lifecycle.get("start_boundary_confirmed"),
                    "end_boundary_confirmed": matched_lifecycle.get("end_boundary_confirmed"),
                    "source_views": matched_lifecycle.get("source_views"),
                }
                if matched_lifecycle
                else None
            ),
        }
        expanded_specs.append(item)
    return expanded_specs


def _episode_window_expansion_stats(specs: list[Mapping[str, Any]]) -> dict[str, Any]:
    expansions = [dict(_as_mapping(spec.get("episode_window_expansion"))) for spec in specs]
    expansions = [item for item in expansions if item]
    return {
        "schema_version": "official_experiment_window_expansion_summary.v1",
        "enabled": True,
        "episode_count": len(specs),
        "expanded_count": sum(1 for item in expansions if item.get("expanded")),
        "min_window_sec": expansions[0].get("min_window_sec") if expansions else _official_experiment_window_min_sec(),
        "silence_gap_sec": expansions[0].get("silence_gap_sec") if expansions else _experiment_window_silence_gap_sec(),
    }


def _micro_source_views(row: Mapping[str, Any]) -> list[str]:
    views: list[str] = []
    for key in ("source_view", "view", "camera_view"):
        value = str(row.get(key) or "").strip()
        if value:
            views.append(value)
    for view_key in ("first_person", "third_person"):
        if isinstance(row.get(view_key), Mapping):
            views.append(view_key)
    for collection_key in ("yolo_evidence", "asset_bindings"):
        for item in row.get(collection_key) or []:
            if not isinstance(item, Mapping):
                continue
            value = str(item.get("source_view") or item.get("view") or item.get("camera_view") or "").strip()
            if value:
                views.append(value)
    return _ordered_unique(views)


def _micro_evidence_source_views(row: Mapping[str, Any]) -> list[str]:
    views: list[str] = []
    for key in ("source_view", "view", "camera_view"):
        value = str(row.get(key) or "").strip()
        if value:
            views.append(value)
    for item in row.get("yolo_evidence") or []:
        if not isinstance(item, Mapping):
            continue
        value = str(item.get("source_view") or item.get("view") or item.get("camera_view") or "").strip()
        if value:
            views.append(value)
    return _ordered_unique(views)


def _cluster_boundary_items(cluster: list[dict[str, Any]], target_sec: float, edge: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for item in cluster:
        start = float(item.get("start") or 0.0)
        end = float(item.get("end") or start)
        if (edge == "start" and abs(start - target_sec) > 1e-6) or (edge == "end" and abs(end - target_sec) > 1e-6):
            continue
        items.append(
            {
                "micro_segment_id": item.get("micro_segment_id"),
                "time_sec": round(start if edge == "start" else end, 6),
                "source_views": list(item.get("source_views") or []),
                "evidence_source_views": list(item.get("evidence_source_views") or []),
                "primary_object": item.get("primary_object"),
                "timeline_source": item.get("interval_source"),
                "timeline_fields": list(item.get("interval_source_fields") or []),
            }
        )
    return items


def _cluster_boundary_evidence(cluster: list[dict[str, Any]], raw_start: float, raw_end: float) -> dict[str, Any]:
    source_view_counts = Counter(
        view
        for item in cluster
        for view in list(item.get("source_views") or [])
        if view
    )
    evidence_source_view_counts = Counter(
        view
        for item in cluster
        for view in list(item.get("evidence_source_views") or [])
        if view
    )
    timeline_sources = Counter(str(item.get("interval_source") or "unknown") for item in cluster)
    return {
        "schema_version": "key_action_episode_boundary_evidence.v1",
        "timeline": "session",
        "start_sec": round(raw_start, 6),
        "end_sec": round(raw_end, 6),
        "start_reason": "earliest_micro_physical_evidence_on_session_timeline",
        "end_reason": "latest_micro_physical_evidence_on_session_timeline",
        "start_evidence": _cluster_boundary_items(cluster, raw_start, "start"),
        "end_evidence": _cluster_boundary_items(cluster, raw_end, "end"),
        "micro_segment_count": len(cluster),
        "source_views": sorted(source_view_counts),
        "source_view_counts": dict(sorted(source_view_counts.items())),
        "evidence_source_views": sorted(evidence_source_view_counts),
        "evidence_source_view_counts": dict(sorted(evidence_source_view_counts.items())),
        "timeline_source_counts": dict(sorted(timeline_sources.items())),
    }


def _merge_boundary_evidence(
    left: Mapping[str, Any] | None,
    right: Mapping[str, Any] | None,
) -> dict[str, Any]:
    left = _as_mapping(left)
    right = _as_mapping(right)
    if not left:
        return dict(right)
    if not right:
        return dict(left)
    start_left = _safe_float(left.get("start_sec"))
    start_right = _safe_float(right.get("start_sec"))
    end_left = _safe_float(left.get("end_sec"))
    end_right = _safe_float(right.get("end_sec"))
    if start_left is None:
        start_left = start_right if start_right is not None else 0.0
    if start_right is None:
        start_right = start_left
    if end_left is None:
        end_left = end_right if end_right is not None else start_left
    if end_right is None:
        end_right = end_left
    start_source = left if float(start_left) <= float(start_right) else right
    end_source = left if float(end_left) >= float(end_right) else right
    view_counts: Counter[str] = Counter(left.get("source_view_counts") or {})
    view_counts.update(right.get("source_view_counts") or {})
    evidence_view_counts: Counter[str] = Counter(left.get("evidence_source_view_counts") or {})
    evidence_view_counts.update(right.get("evidence_source_view_counts") or {})
    timeline_counts: Counter[str] = Counter(left.get("timeline_source_counts") or {})
    timeline_counts.update(right.get("timeline_source_counts") or {})
    return {
        "schema_version": "key_action_episode_boundary_evidence.v1",
        "timeline": "session",
        "start_sec": round(min(float(start_left), float(start_right)), 6),
        "end_sec": round(max(float(end_left), float(end_right)), 6),
        "start_reason": start_source.get("start_reason") or "earliest_micro_physical_evidence_on_session_timeline",
        "end_reason": end_source.get("end_reason") or "latest_micro_physical_evidence_on_session_timeline",
        "start_evidence": list(start_source.get("start_evidence") or []),
        "end_evidence": list(end_source.get("end_evidence") or []),
        "micro_segment_count": int(left.get("micro_segment_count") or 0) + int(right.get("micro_segment_count") or 0),
        "source_views": sorted(view_counts),
        "source_view_counts": dict(sorted(view_counts.items())),
        "evidence_source_views": sorted(evidence_view_counts),
        "evidence_source_view_counts": dict(sorted(evidence_view_counts.items())),
        "timeline_source_counts": dict(sorted(timeline_counts.items())),
    }


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None and str(item)]
    return [str(value)] if str(value) else []


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def _normalized_text(value: Any) -> str:
    return str(value or "").strip().casefold().replace("-", "_").replace(" ", "_")


def _micro_evidence_level(row: Mapping[str, Any]) -> str:
    evidence = _as_mapping(row.get("evidence"))
    return _normalized_text(evidence.get("evidence_level") or row.get("evidence_level"))


def _micro_process_role(row: Mapping[str, Any]) -> str:
    evidence = _as_mapping(row.get("evidence"))
    return _normalized_text(evidence.get("process_evidence_role") or row.get("process_evidence_role"))


def _micro_quality_warnings(row: Mapping[str, Any]) -> list[str]:
    evidence = _as_mapping(row.get("evidence"))
    quality = _as_mapping(row.get("quality"))
    return _ordered_unique(
        [
            *_as_strings(quality.get("warnings")),
            *_as_strings(row.get("quality_warnings")),
            *_as_strings(evidence.get("limitations")),
            *_as_strings(row.get("limitations")),
        ]
    )


def _micro_has_partial_marker(row: Mapping[str, Any]) -> bool:
    evidence = _as_mapping(row.get("evidence"))
    warnings_text = " ".join(_micro_quality_warnings(row)).casefold()
    return bool(
        evidence.get("coverage_backfill")
        or evidence.get("segment_level_coverage_backfill")
        or evidence.get("force_retrieval_candidate")
        or "backfill" in warnings_text
        or "retrieval candidate" in warnings_text
        or "supporting process candidate" in warnings_text
        or "requires human" in warnings_text
    )


def _micro_is_weak_only(row: Mapping[str, Any]) -> bool:
    return (
        _micro_evidence_level(row) in _WEAK_EVIDENCE_LEVELS
        and _micro_process_role(row) not in _PARTIAL_PROCESS_ROLES
        and not _micro_has_partial_marker(row)
    )


def _support_quality_from_components(
    *,
    evidence_level_counts: Counter[str],
    process_role_counts: Counter[str],
    strong_ids: list[str],
    partial_ids: list[str],
    weak_ids: list[str],
    quality_warnings: list[str],
) -> dict[str, Any]:
    total = sum(evidence_level_counts.values())
    strong_ids = _ordered_unique(strong_ids)
    partial_ids = _ordered_unique(partial_ids)
    weak_ids = _ordered_unique(weak_ids)
    strong_count = len(strong_ids)
    partial_count = len(partial_ids)
    weak_count = len(weak_ids)
    if total <= 0 or (strong_count == 0 and partial_count == 0):
        fact_strength = "weak"
    elif strong_count == total and partial_count == 0 and weak_count == 0:
        fact_strength = "strong"
    else:
        fact_strength = "partial"

    if fact_strength == "strong":
        recommended_level = "visual_confirmed"
    elif evidence_level_counts.get("transcript_supported") and not weak_count:
        recommended_level = "transcript_supported"
    else:
        recommended_level = "weak_visual_evidence"

    reasons: list[str] = []
    if weak_count:
        reasons.append("weak_child_micro_evidence")
    if partial_count:
        reasons.append("partial_child_micro_evidence")
    if evidence_level_counts.get("unknown"):
        reasons.append("missing_child_micro_evidence_level")
    if fact_strength != "strong":
        reasons.append("not_eligible_for_strong_fact_claim")

    return {
        "schema_version": "key_action_episode_support_quality.v1",
        "fact_strength": fact_strength,
        "strong_fact_allowed": fact_strength == "strong",
        "recommended_evidence_level": recommended_level,
        "micro_segment_count": total,
        "strong_micro_segment_count": strong_count,
        "partial_micro_segment_count": partial_count,
        "weak_micro_segment_count": weak_count,
        "strong_micro_segment_ids": strong_ids,
        "partial_micro_segment_ids": partial_ids,
        "weak_micro_segment_ids": weak_ids,
        "evidence_level_counts": dict(sorted(evidence_level_counts.items())),
        "process_role_counts": dict(sorted((key, value) for key, value in process_role_counts.items() if key)),
        "quality_warnings": _ordered_unique(quality_warnings),
        "reasons": _ordered_unique(reasons),
    }


def _episode_support_quality_from_micro_rows(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    evidence_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    strong_ids: list[str] = []
    partial_ids: list[str] = []
    weak_ids: list[str] = []
    quality_warnings: list[str] = []
    for index, row in enumerate(rows, start=1):
        micro_id = str(row.get("micro_segment_id") or f"micro_{index:06d}")
        evidence = _as_mapping(row.get("evidence"))
        level = _micro_evidence_level(row) or "unknown"
        role = _micro_process_role(row)
        evidence_counts[level] += 1
        if role:
            role_counts[role] += 1
        quality_warnings.extend(_micro_quality_warnings(row))

        is_weak = level in _WEAK_EVIDENCE_LEVELS
        is_partial = level in _PARTIAL_EVIDENCE_LEVELS or role in _PARTIAL_PROCESS_ROLES or _micro_has_partial_marker(row)
        is_strong = bool(evidence.get("strong_process_evidence")) or role == "strong_process_evidence" or level in _STRONG_EVIDENCE_LEVELS
        if is_weak:
            weak_ids.append(micro_id)
        elif is_partial or not is_strong:
            partial_ids.append(micro_id)
        else:
            strong_ids.append(micro_id)

    return _support_quality_from_components(
        evidence_level_counts=evidence_counts,
        process_role_counts=role_counts,
        strong_ids=strong_ids,
        partial_ids=partial_ids,
        weak_ids=weak_ids,
        quality_warnings=quality_warnings,
    )


def _merge_episode_support_quality(left: Mapping[str, Any] | None, right: Mapping[str, Any] | None) -> dict[str, Any]:
    left = _as_mapping(left)
    right = _as_mapping(right)
    evidence_counts: Counter[str] = Counter(left.get("evidence_level_counts") or {})
    evidence_counts.update(right.get("evidence_level_counts") or {})
    role_counts: Counter[str] = Counter(left.get("process_role_counts") or {})
    role_counts.update(right.get("process_role_counts") or {})
    return _support_quality_from_components(
        evidence_level_counts=evidence_counts,
        process_role_counts=role_counts,
        strong_ids=[*_as_strings(left.get("strong_micro_segment_ids")), *_as_strings(right.get("strong_micro_segment_ids"))],
        partial_ids=[*_as_strings(left.get("partial_micro_segment_ids")), *_as_strings(right.get("partial_micro_segment_ids"))],
        weak_ids=[*_as_strings(left.get("weak_micro_segment_ids")), *_as_strings(right.get("weak_micro_segment_ids"))],
        quality_warnings=[*_as_strings(left.get("quality_warnings")), *_as_strings(right.get("quality_warnings"))],
    )


def _episode_specs_from_micros(
    manifest: SessionManifest,
    micro_rows: list[dict[str, Any]],
    *,
    activity_rows: list[Mapping[str, Any]] | None = None,
    coarse_key_segments: list[KeyActionSegment] | None = None,
    gap_sec: float,
    pre_roll_sec: float,
    post_roll_sec: float,
    min_episode_duration_sec: float,
    min_micro_evidence_count: int,
    include_candidates: bool = False,
) -> list[dict[str, Any]]:
    candidates = []
    for row in micro_rows:
        if not isinstance(row, dict) or not _micro_has_physical_evidence(row):
            continue
        interval_details = _micro_interval_details(manifest, row)
        if interval_details is None:
            continue
        candidates.append(
            {
                "row": row,
                "start": interval_details["start"],
                "end": interval_details["end"],
                "interval_source": interval_details["source"],
                "interval_source_fields": interval_details["source_fields"],
                "timeline": interval_details["timeline"],
                "primary_object": _micro_primary_object(row),
                "micro_segment_id": row.get("micro_segment_id"),
                "source_views": _micro_source_views(row),
                "evidence_source_views": _micro_evidence_source_views(row),
            }
        )
    candidates.sort(key=lambda item: (float(item["start"]), float(item["end"])))
    strong_or_partial_candidates = [
        item
        for item in candidates
        if isinstance(item.get("row"), Mapping) and not _micro_is_weak_only(item["row"])
    ]
    if len(strong_or_partial_candidates) >= max(1, min_micro_evidence_count):
        candidates = strong_or_partial_candidates
    if len(candidates) < 1:
        return []
    if len(candidates) < max(1, min_micro_evidence_count) and not include_candidates:
        return []

    clusters: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_end = 0.0
    for item in candidates:
        start = float(item["start"])
        end = float(item["end"])
        if current and start - current_end > gap_sec:
            clusters.append(current)
            current = []
        current.append(item)
        current_end = max(current_end, end)
    if current:
        clusters.append(current)

    specs: list[dict[str, Any]] = []
    source_duration = _duration_from_videos(manifest)
    for cluster in clusters:
        raw_start = min(float(item["start"]) for item in cluster)
        raw_end = max(float(item["end"]) for item in cluster)
        start = max(0.0, raw_start - max(0.0, pre_roll_sec))
        end = raw_end + max(0.0, post_roll_sec)
        if source_duration:
            end = min(float(source_duration), end)
        objects = Counter(str(item.get("primary_object") or "") for item in cluster if item.get("primary_object"))
        source_view_counts = Counter(
            view
            for item in cluster
            for view in list(item.get("source_views") or [])
            if view
        )
        evidence_source_view_counts = Counter(
            view
            for item in cluster
            for view in list(item.get("evidence_source_views") or [])
            if view
        )
        boundary_evidence = _cluster_boundary_evidence(cluster, raw_start, raw_end)
        support_quality = _episode_support_quality_from_micro_rows([item["row"] for item in cluster if isinstance(item.get("row"), Mapping)])
        specs.append(
            {
                "start_sec": round(start, 6),
                "end_sec": round(max(end, start + 0.1), 6),
                "true_start_sec": round(raw_start, 6),
                "true_end_sec": round(raw_end, 6),
                "duration_sec": round(max(end - start, 0.1), 6),
                "micro_segment_ids": [str(item.get("micro_segment_id")) for item in cluster if item.get("micro_segment_id")],
                "primary_objects": dict(objects),
                "anchor_micro_segment_id": cluster[0].get("micro_segment_id"),
                "anchor_primary_object": cluster[0].get("primary_object"),
                "support_quality": support_quality,
                "understanding_fact_strength": support_quality.get("fact_strength"),
                "strong_fact_allowed": support_quality.get("strong_fact_allowed"),
                "timeline": "session",
                "source_views": sorted(source_view_counts),
                "source_view_counts": dict(sorted(source_view_counts.items())),
                "evidence_source_views": sorted(evidence_source_view_counts),
                "evidence_source_view_counts": dict(sorted(evidence_source_view_counts.items())),
                "boundary_evidence": boundary_evidence,
                "episode_merge_strategy": "initial_micro_cluster",
                "episode_merge_reasons": ["initial_yolo_micro_cluster"],
                "episode_merge_events": [],
            }
        )

    coalesced = _coalesce_dense_episode_specs(
        _merge_overlapping_episode_specs(specs, source_duration),
        source_duration,
        base_gap_sec=gap_sec,
    )
    expanded = _expand_episode_specs_to_activity_windows(
        manifest,
        coalesced,
        activity_rows=activity_rows or [],
        coarse_key_segments=coarse_key_segments or [],
        source_duration=source_duration,
        min_window_sec=min_episode_duration_sec,
    )
    annotated = _annotate_episode_specs(expanded, min_episode_duration_sec=min_episode_duration_sec)
    if include_candidates:
        return annotated
    return [spec for spec in annotated if _is_official_episode_spec(spec)]


def _merge_overlapping_episode_specs(specs: list[dict[str, Any]], source_duration: float | None) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for spec in sorted(specs, key=lambda item: float(item["start_sec"])):
        if not merged or float(spec["start_sec"]) > float(merged[-1]["end_sec"]) + 0.25:
            merged.append(dict(spec))
            continue
        _merge_episode_spec(merged[-1], spec, source_duration, strategy="overlap_or_padding")
    return merged


def _spec_true_gap_sec(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    left_end = float(left.get("true_end_sec") or left.get("end_sec") or 0.0)
    right_start = float(right.get("true_start_sec") or right.get("start_sec") or 0.0)
    return max(0.0, right_start - left_end)


def _positive_spec_gaps(specs: list[dict[str, Any]]) -> list[float]:
    gaps: list[float] = []
    ordered = sorted(specs, key=lambda item: float(item.get("start_sec") or 0.0))
    for index in range(1, len(ordered)):
        gap = _spec_true_gap_sec(ordered[index - 1], ordered[index])
        if gap > 0.0:
            gaps.append(gap)
    return gaps


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * max(0.0, min(1.0, percentile))
    lower = int(rank)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = rank - lower
    return float(ordered[lower]) * (1.0 - fraction) + float(ordered[upper]) * fraction


def _episode_internal_merge_gap_sec(specs: list[dict[str, Any]], *, base_gap_sec: float) -> float:
    short_gap = _float_env(
        "KEY_ACTION_EPISODE_SHORT_MERGE_GAP_SEC",
        _DEFAULT_SHORT_EPISODE_MERGE_GAP_SEC,
        minimum=max(0.0, base_gap_sec),
    )
    default_internal_gap = _float_env(
        "KEY_ACTION_FAST_LOCATE_EXPERIMENT_MACRO_MERGE_GAP_SEC",
        _DEFAULT_INTERNAL_EPISODE_MERGE_GAP_SEC,
        minimum=short_gap,
    )
    internal_gap_cap = _float_env(
        "KEY_ACTION_EPISODE_INTERNAL_MERGE_GAP_SEC",
        default_internal_gap,
        minimum=short_gap,
    )
    gaps = _positive_spec_gaps(specs)
    if not gaps:
        return short_gap
    dense_gap = _percentile(gaps, 0.75)
    multiplier = _float_env(
        "KEY_ACTION_EPISODE_GAP_OUTLIER_MULTIPLIER",
        _DEFAULT_EPISODE_GAP_OUTLIER_MULTIPLIER,
        minimum=1.0,
    )
    return round(min(internal_gap_cap, max(short_gap, dense_gap * multiplier)), 6)


def _merge_episode_spec(
    target: dict[str, Any],
    spec: Mapping[str, Any],
    source_duration: float | None,
    *,
    strategy: str,
    gap_sec: float | None = None,
    threshold_sec: float | None = None,
) -> None:
    target["end_sec"] = round(max(float(target["end_sec"]), float(spec["end_sec"])), 6)
    if source_duration:
        target["end_sec"] = round(min(float(source_duration), float(target["end_sec"])), 6)
    target["true_start_sec"] = round(
        min(float(target.get("true_start_sec") or target["start_sec"]), float(spec.get("true_start_sec") or spec["start_sec"])),
        6,
    )
    target["true_end_sec"] = round(
        max(float(target.get("true_end_sec") or target["end_sec"]), float(spec.get("true_end_sec") or spec["end_sec"])),
        6,
    )
    target["duration_sec"] = round(float(target["end_sec"]) - float(target["start_sec"]), 6)
    target["micro_segment_ids"] = [*target.get("micro_segment_ids", []), *spec.get("micro_segment_ids", [])]
    objects = Counter(target.get("primary_objects") or {})
    objects.update(spec.get("primary_objects") or {})
    target["primary_objects"] = dict(objects)
    merged_from = list(target.get("merged_from_anchor_micro_segment_ids") or [])
    for anchor in (target.get("anchor_micro_segment_id"), spec.get("anchor_micro_segment_id")):
        if anchor and str(anchor) not in merged_from:
            merged_from.append(str(anchor))
    if merged_from:
        target["merged_from_anchor_micro_segment_ids"] = merged_from
    strategies = list(target.get("episode_merge_strategies") or [])
    if strategy not in strategies:
        strategies.append(strategy)
    target["episode_merge_strategies"] = strategies
    target["episode_merge_strategy"] = strategy
    reasons = _ordered_unique(
        [
            *_as_strings(target.get("episode_merge_reasons")),
            *_as_strings(spec.get("episode_merge_reasons")),
            _merge_reason_for_strategy(strategy, gap_sec=gap_sec, threshold_sec=threshold_sec),
        ]
    )
    target["episode_merge_reasons"] = reasons
    events = list(target.get("episode_merge_events") or [])
    events.extend(item for item in list(spec.get("episode_merge_events") or []) if isinstance(item, Mapping))
    events.append(
        {
            "decision": "merge",
            "strategy": strategy,
            "reason": _merge_reason_for_strategy(strategy, gap_sec=gap_sec, threshold_sec=threshold_sec),
            "gap_sec": round(float(gap_sec), 6) if gap_sec is not None else None,
            "threshold_sec": round(float(threshold_sec), 6) if threshold_sec is not None else None,
            "left_anchor_micro_segment_id": target.get("anchor_micro_segment_id"),
            "right_anchor_micro_segment_id": spec.get("anchor_micro_segment_id"),
        }
    )
    target["episode_merge_events"] = events
    if gap_sec is not None:
        target["last_merge_gap_sec"] = round(float(gap_sec), 6)
    if threshold_sec is not None:
        target["macro_merge_gap_sec"] = round(float(threshold_sec), 6)
    view_counts = Counter(target.get("source_view_counts") or {})
    view_counts.update(spec.get("source_view_counts") or {})
    target["source_view_counts"] = dict(sorted(view_counts.items()))
    target["source_views"] = sorted(view_counts)
    evidence_view_counts = Counter(target.get("evidence_source_view_counts") or {})
    evidence_view_counts.update(spec.get("evidence_source_view_counts") or {})
    target["evidence_source_view_counts"] = dict(sorted(evidence_view_counts.items()))
    target["evidence_source_views"] = sorted(evidence_view_counts)
    target["boundary_evidence"] = _merge_boundary_evidence(target.get("boundary_evidence"), spec.get("boundary_evidence"))
    support_quality = _merge_episode_support_quality(target.get("support_quality"), spec.get("support_quality"))
    target["support_quality"] = support_quality
    target["understanding_fact_strength"] = support_quality.get("fact_strength")
    target["strong_fact_allowed"] = support_quality.get("strong_fact_allowed")


def _merge_reason_for_strategy(
    strategy: str,
    *,
    gap_sec: float | None = None,
    threshold_sec: float | None = None,
) -> str:
    if strategy == "density_gap_macro_merge":
        if gap_sec is not None and threshold_sec is not None:
            return "small_silence_gap_within_macro_merge_gap"
        return "dense_micro_fragments_coalesced"
    if strategy == "overlap_or_padding":
        return "overlapping_or_padded_micro_episode_windows"
    if strategy == "short_gap_merge_without_dense_fragment_count":
        return "small_silence_gap_within_short_merge_gap"
    return strategy


def _has_complementary_dual_view_evidence(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    views = {
        str(view)
        for spec in (left, right)
        for view in list(spec.get("evidence_source_views") or [])
        if view
    }
    return {"first_person", "third_person"}.issubset(views)


def _episode_true_duration_sec(spec: Mapping[str, Any]) -> float:
    try:
        start = float(spec.get("true_start_sec") or spec.get("start_sec") or 0.0)
        end = float(spec.get("true_end_sec") or spec.get("end_sec") or start)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, end - start)


def _episode_formal_views(spec: Mapping[str, Any]) -> set[str]:
    views = {
        str(view).strip()
        for key in ("evidence_source_views", "source_views")
        for view in list(spec.get(key) or [])
        if str(view).strip()
    }
    if "dual_view" in views:
        views.update(_REQUIRED_FORMAL_EPISODE_VIEWS)
    return views


def _episode_candidate_reasons(spec: Mapping[str, Any], *, min_episode_duration_sec: float) -> list[str]:
    reasons: list[str] = []
    true_duration = _episode_true_duration_sec(spec)
    if true_duration < float(min_episode_duration_sec):
        reasons.append("too_short_action_window")
    views = _episode_formal_views(spec)
    if not _REQUIRED_FORMAL_EPISODE_VIEWS.issubset(views):
        reasons.append("single_view_candidate" if views else "missing_dual_view_action_evidence")
    return _ordered_unique(reasons)


def _annotate_episode_specs(
    specs: list[dict[str, Any]],
    *,
    min_episode_duration_sec: float,
) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for spec in specs:
        item = dict(spec)
        reasons = _episode_candidate_reasons(item, min_episode_duration_sec=min_episode_duration_sec)
        official = not reasons
        item["true_duration_sec"] = round(_episode_true_duration_sec(item), 6)
        item["min_official_episode_duration_sec"] = float(min_episode_duration_sec)
        item["required_episode_views"] = sorted(_REQUIRED_FORMAL_EPISODE_VIEWS)
        item["episode_status"] = "official" if official else "candidate"
        item["candidate_status"] = "official_episode" if official else "candidate_action_window"
        item["official_episode"] = bool(official)
        item["formal_results_allowed"] = bool(official)
        item["single_view_candidate"] = bool(
            not _REQUIRED_FORMAL_EPISODE_VIEWS.issubset(_episode_formal_views(item))
        )
        item["candidate_reasons"] = reasons
        item["source_layer"] = "episode_activity" if official else "action_window"
        annotated.append(item)
    return annotated


def _is_official_episode_spec(spec: Mapping[str, Any]) -> bool:
    return bool(spec.get("official_episode") or spec.get("episode_status") == "official")


def _candidate_action_window_rows(manifest: SessionManifest, specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pseudo = VideoSource("session", "session", manifest.session_start_time)
    for index, spec in enumerate(sorted(specs, key=lambda item: float(item.get("start_sec") or 0.0)), start=1):
        start_sec = float(spec.get("start_sec") or 0.0)
        end_sec = max(start_sec + 0.1, float(spec.get("end_sec") or start_sec))
        true_start_sec = float(spec.get("true_start_sec") or start_sec)
        true_end_sec = max(true_start_sec, float(spec.get("true_end_sec") or end_sec))
        rows.append(
            {
                "schema_version": ACTION_WINDOW_SCHEMA_VERSION,
                "session_id": manifest.session_id,
                "action_window_id": f"action_window_{index:06d}",
                "candidate_id": f"candidate_action_window_{index:06d}",
                "candidate_status": "candidate_action_window",
                "official_episode": False,
                "formal_results_allowed": False,
                "single_view_candidate": bool(spec.get("single_view_candidate")),
                "candidate_reasons": list(spec.get("candidate_reasons") or []),
                "source_layer": "action_window",
                "session_start_sec": round(start_sec, 6),
                "session_end_sec": round(end_sec, 6),
                "duration_sec": round(end_sec - start_sec, 6),
                "true_start_sec": round(true_start_sec, 6),
                "true_end_sec": round(true_end_sec, 6),
                "true_duration_sec": round(max(0.0, true_end_sec - true_start_sec), 6),
                "global_start_time": local_sec_to_global_time(pseudo, start_sec).isoformat(),
                "global_end_time": local_sec_to_global_time(pseudo, end_sec).isoformat(),
                "micro_segment_ids": list(spec.get("micro_segment_ids") or []),
                "primary_objects": dict(spec.get("primary_objects") or {}),
                "source_views": list(spec.get("source_views") or []),
                "source_view_counts": dict(spec.get("source_view_counts") or {}),
                "evidence_source_views": list(spec.get("evidence_source_views") or []),
                "evidence_source_view_counts": dict(spec.get("evidence_source_view_counts") or {}),
                "boundary_evidence": dict(spec.get("boundary_evidence") or {}),
                "episode_window_expansion": dict(spec.get("episode_window_expansion") or {}),
                "episode_merge_strategy": spec.get("episode_merge_strategy"),
                "episode_merge_reasons": list(spec.get("episode_merge_reasons") or []),
                "interpretation": "candidate_action_window_not_official_experiment_episode",
            }
        )
    return rows


def _coalesce_dense_episode_specs(
    specs: list[dict[str, Any]],
    source_duration: float | None,
    *,
    base_gap_sec: float,
) -> list[dict[str, Any]]:
    if len(specs) <= 1:
        return specs
    short_gap = _float_env(
        "KEY_ACTION_EPISODE_SHORT_MERGE_GAP_SEC",
        _DEFAULT_SHORT_EPISODE_MERGE_GAP_SEC,
        minimum=max(0.0, base_gap_sec),
    )
    min_dense_fragments = int(
        _float_env(
            "KEY_ACTION_EPISODE_DENSE_MERGE_MIN_FRAGMENTS",
            float(_DEFAULT_DENSE_EPISODE_MERGE_MIN_FRAGMENTS),
            minimum=2.0,
        )
    )
    if len(specs) < min_dense_fragments:
        merged: list[dict[str, Any]] = []
        for spec in sorted(specs, key=lambda item: float(item.get("start_sec") or 0.0)):
            if not merged:
                item = dict(spec)
                item["macro_merge_gap_sec"] = short_gap
                merged.append(item)
                continue
            gap = _spec_true_gap_sec(merged[-1], spec)
            padded_gap = max(0.0, float(spec.get("start_sec") or 0.0) - float(merged[-1].get("end_sec") or 0.0))
            if padded_gap <= 0.25 or (gap <= short_gap and _has_complementary_dual_view_evidence(merged[-1], spec)):
                _merge_episode_spec(
                    merged[-1],
                    spec,
                    source_duration,
                    strategy="short_gap_merge_without_dense_fragment_count",
                    gap_sec=gap,
                    threshold_sec=short_gap,
                )
                continue
            item = dict(spec)
            item["episode_merge_strategy"] = "density_gap_not_enough_fragments"
            item["macro_merge_gap_sec"] = short_gap
            item["previous_episode_gap_sec"] = round(gap, 6)
            item["previous_boundary_decision"] = {
                "decision": "split",
                "reason": "silence_gap_exceeds_short_merge_gap_and_not_enough_dense_fragments",
                "gap_sec": round(gap, 6),
                "threshold_sec": round(short_gap, 6),
            }
            merged.append(item)
        return merged
    threshold = _episode_internal_merge_gap_sec(specs, base_gap_sec=base_gap_sec)
    merged: list[dict[str, Any]] = []
    for spec in sorted(specs, key=lambda item: float(item.get("start_sec") or 0.0)):
        if not merged:
            item = dict(spec)
            item["episode_merge_strategy"] = item.get("episode_merge_strategy") or "density_gap_seed"
            item["macro_merge_gap_sec"] = threshold
            merged.append(item)
            continue
        gap = _spec_true_gap_sec(merged[-1], spec)
        padded_gap = max(0.0, float(spec.get("start_sec") or 0.0) - float(merged[-1].get("end_sec") or 0.0))
        if padded_gap <= 0.25 or gap <= threshold:
            _merge_episode_spec(
                merged[-1],
                spec,
                source_duration,
                strategy="density_gap_macro_merge",
                gap_sec=gap,
                threshold_sec=threshold,
            )
            continue
        item = dict(spec)
        item["episode_merge_strategy"] = item.get("episode_merge_strategy") or "density_gap_boundary"
        item["macro_merge_gap_sec"] = threshold
        item["previous_episode_gap_sec"] = round(gap, 6)
        item["previous_boundary_decision"] = {
            "decision": "split",
            "reason": "large_silence_gap_exceeds_macro_merge_gap",
            "gap_sec": round(gap, 6),
            "threshold_sec": round(threshold, 6),
        }
        merged.append(item)
    return merged


def _episode_merge_stats(specs: list[dict[str, Any]]) -> dict[str, Any]:
    gaps = _positive_spec_gaps(specs)
    strategies = Counter()
    reasons = Counter()
    source_views = Counter()
    evidence_source_views = Counter()
    split_count = 0
    for spec in specs:
        for strategy in spec.get("episode_merge_strategies") or [spec.get("episode_merge_strategy")]:
            if strategy:
                strategies[str(strategy)] += 1
        for reason in spec.get("episode_merge_reasons") or []:
            if reason:
                reasons[str(reason)] += 1
        source_views.update(spec.get("source_view_counts") or {})
        evidence_source_views.update(spec.get("evidence_source_view_counts") or {})
        if isinstance(spec.get("previous_boundary_decision"), Mapping):
            split_count += 1
    thresholds = sorted({float(spec.get("macro_merge_gap_sec")) for spec in specs if spec.get("macro_merge_gap_sec") is not None})
    return {
        "strategy_counts": dict(strategies),
        "reason_counts": dict(reasons),
        "positive_gap_count": len(gaps),
        "split_boundary_count": split_count,
        "max_remaining_gap_sec": round(max(gaps), 6) if gaps else 0.0,
        "macro_merge_gap_sec": thresholds[-1] if thresholds else None,
        "source_view_counts": dict(sorted(source_views.items())),
        "evidence_source_view_counts": dict(sorted(evidence_source_views.items())),
        "timeline": "session",
    }


def _coalesce_episode_specs_to_expected_count(
    specs: list[dict[str, Any]],
    expected_count: int | None,
    source_duration: float | None,
) -> list[dict[str, Any]]:
    if expected_count is None or expected_count <= 0 or len(specs) <= expected_count:
        return specs
    ordered = sorted(specs, key=lambda item: (float(item.get("start_sec") or 0.0), float(item.get("end_sec") or 0.0)))
    gaps: list[tuple[float, int]] = []
    for index in range(1, len(ordered)):
        previous_end = float(ordered[index - 1].get("end_sec") or 0.0)
        current_start = float(ordered[index].get("start_sec") or 0.0)
        gaps.append((current_start - previous_end, index))
    boundary_indexes = {
        index
        for _gap, index in sorted(gaps, key=lambda item: (item[0], item[1]), reverse=True)[: max(0, expected_count - 1)]
    }
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for index, spec in enumerate(ordered):
        if current and index in boundary_indexes:
            groups.append(current)
            current = []
        current.append(spec)
    if current:
        groups.append(current)
    merged: list[dict[str, Any]] = []
    for group in groups:
        start_sec = min(float(item.get("start_sec") or 0.0) for item in group)
        end_sec = max(float(item.get("end_sec") or start_sec) for item in group)
        if source_duration:
            end_sec = min(float(source_duration), end_sec)
        true_start_sec = min(float(item.get("true_start_sec") or item.get("start_sec") or start_sec) for item in group)
        true_end_sec = max(float(item.get("true_end_sec") or item.get("end_sec") or end_sec) for item in group)
        objects: Counter[str] = Counter()
        micro_segment_ids: list[str] = []
        merged_from: list[str] = []
        support_quality: dict[str, Any] | None = None
        boundary_evidence: dict[str, Any] | None = None
        for item in group:
            objects.update(item.get("primary_objects") or {})
            micro_segment_ids.extend(str(value) for value in item.get("micro_segment_ids") or [] if value)
            anchor = item.get("anchor_micro_segment_id")
            if anchor:
                merged_from.append(str(anchor))
            support_quality = _merge_episode_support_quality(support_quality, item.get("support_quality"))
            boundary_evidence = _merge_boundary_evidence(boundary_evidence, item.get("boundary_evidence"))
        support_quality = support_quality or _support_quality_from_components(
            evidence_level_counts=Counter(),
            process_role_counts=Counter(),
            strong_ids=[],
            partial_ids=[],
            weak_ids=[],
            quality_warnings=[],
        )
        merged.append(
            {
                **dict(group[0]),
                "start_sec": round(start_sec, 6),
                "end_sec": round(max(end_sec, start_sec + 0.1), 6),
                "true_start_sec": round(true_start_sec, 6),
                "true_end_sec": round(true_end_sec, 6),
                "duration_sec": round(max(end_sec - start_sec, 0.1), 6),
                "micro_segment_ids": micro_segment_ids,
                "primary_objects": dict(objects),
                "merged_from_anchor_micro_segment_ids": merged_from,
                "episode_merge_strategy": "expected_count_largest_gap_partition",
                "episode_merge_reasons": _ordered_unique(
                    [
                        *[
                            reason
                            for item in group
                            for reason in _as_strings(item.get("episode_merge_reasons"))
                        ],
                        "expected_count_largest_gap_partition",
                    ]
                ),
                "expected_experiment_count": expected_count,
                "support_quality": support_quality,
                "understanding_fact_strength": support_quality.get("fact_strength"),
                "strong_fact_allowed": support_quality.get("strong_fact_allowed"),
                "boundary_evidence": boundary_evidence or {},
            }
        )
    return merged


def _duration_from_videos(manifest: SessionManifest) -> float | None:
    durations: list[float] = []
    for source in manifest.videos.all_sources().values():
        try:
            manifest_duration = float(getattr(source, "duration_sec", 0.0) or 0.0)
        except (TypeError, ValueError):
            manifest_duration = 0.0
        if manifest_duration > 0:
            durations.append(manifest_duration)
            continue
        try:
            durations.append(float(get_video_duration_sec(source.path)))
        except Exception:
            continue
    return max(durations) if durations else None


def _source_duration_sec(
    manifest: SessionManifest,
    episode_specs: list[dict[str, Any]],
    key_segments: list[KeyActionSegment],
    micro_rows: list[dict[str, Any]],
) -> float:
    duration = _duration_from_videos(manifest)
    if duration and duration > 0:
        return round(float(duration), 6)
    ends: list[float] = []
    ends.extend(float(spec.get("end_sec") or 0.0) for spec in episode_specs)
    for segment in key_segments:
        cv = getattr(segment, "cv_detection", None)
        ends.append(float(getattr(cv, "end_sec", 0.0) or 0.0))
    for row in micro_rows:
        interval = _micro_interval(manifest, row)
        if interval:
            ends.append(interval[1])
    return round(max(ends, default=0.0), 6)


def _detected_segment_from_spec(
    manifest: SessionManifest,
    spec: dict[str, Any],
    index: int,
    *,
    source_duration_sec: float,
) -> DetectedSegment:
    start_sec = float(spec["start_sec"])
    end_sec = min(float(source_duration_sec), float(spec["end_sec"])) if source_duration_sec else float(spec["end_sec"])
    global_start = local_sec_to_global_time(VideoSource("session", "session", manifest.session_start_time), start_sec)
    global_end = local_sec_to_global_time(VideoSource("session", "session", manifest.session_start_time), end_sec)
    object_counts = dict(spec.get("primary_objects") or {})
    support_count = len(spec.get("micro_segment_ids") or [])
    merge_reasons = [str(reason) for reason in spec.get("episode_merge_reasons") or [] if reason]
    source_views = [str(view) for view in spec.get("source_views") or [] if view]
    expansion = _as_mapping(spec.get("episode_window_expansion"))
    return DetectedSegment(
        segment_id=f"episode_{index:06d}",
        start_sec=start_sec,
        end_sec=end_sec,
        duration_sec=round(end_sec - start_sec, 6),
        global_start_time=global_start.isoformat(),
        global_end_time=global_end.isoformat(),
        avg_motion_score=0.0,
        avg_active_score=1.0 if support_count else 0.0,
        start_reason=(
            "coarse_activity_experiment_window_start"
            if expansion.get("expanded")
            else "yolo_micro_physical_evidence_start"
        ),
        end_reason=(
            "coarse_activity_experiment_window_end"
            if expansion.get("expanded")
            else "yolo_micro_physical_evidence_end"
        ),
        review_required=False,
        detector_backend="yolo_episode",
        detector_source_view="multiview",
        yolo_label_counts=object_counts,
        yolo_interaction_count=support_count,
        boundary_confidence=1.0 if support_count else 0.0,
        boundary_support_count=support_count,
        boundary_source="coarse_activity_experiment_window" if expansion.get("expanded") else "activity_episode_window",
        decision_path="episode_activity_layer",
        decision_trace=[
            "backend=yolo_episode",
            "timeline=session",
            "source_layer=episode_activity",
            "boundary_source=activity_episode_window",
            f"merge_strategy={spec.get('episode_merge_strategy') or ''}",
            f"merge_reasons={','.join(merge_reasons) if merge_reasons else 'none'}",
            f"source_views={','.join(source_views) if source_views else 'unknown'}",
            f"official_window_expanded={bool(expansion.get('expanded'))}",
            f"official_window_min_sec={expansion.get('min_window_sec') if expansion else ''}",
        ],
        source="experiment_episode",
    )


def _segment_interval(row: Mapping[str, Any]) -> tuple[float, float] | None:
    cv = row.get("cv_detection") if isinstance(row.get("cv_detection"), Mapping) else {}
    start = _safe_float(cv.get("start_sec"), _safe_float(row.get("start_sec")))
    end = _safe_float(cv.get("end_sec"), _safe_float(row.get("end_sec")))
    if start is None or end is None or end <= start:
        return None
    return float(start), float(end)


def _best_episode_for_micro(interval: tuple[float, float], specs: list[dict[str, Any]]) -> int | None:
    start, end = interval
    best_index: int | None = None
    best_overlap = 0.0
    for index, spec in enumerate(specs):
        ep_start = float(spec["start_sec"])
        ep_end = float(spec["end_sec"])
        overlap = max(0.0, min(end, ep_end) - max(start, ep_start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_index = index
    if best_index is not None and best_overlap > 0.0:
        return best_index
    for index, spec in enumerate(specs):
        if float(spec["start_sec"]) <= start <= float(spec["end_sec"]):
            return index
    return None


def _remap_micro_rows_to_episodes(
    manifest: SessionManifest,
    micro_rows: list[dict[str, Any]],
    specs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    remapped: list[dict[str, Any]] = []
    micro_map: dict[str, str] = {}
    display_order_by_episode: Counter[str] = Counter()
    for row in sorted(micro_rows, key=lambda item: _micro_interval(manifest, item) or (0.0, 0.0)):
        interval = _micro_interval(manifest, row)
        if interval is None:
            continue
        episode_index = _best_episode_for_micro(interval, specs)
        if episode_index is None:
            continue
        episode_id = f"episode_{episode_index + 1:06d}"
        display_order_by_episode[episode_id] += 1
        item = dict(row)
        old_parent = str(item.get("parent_segment_id") or item.get("segment_id") or "")
        if item.get("micro_segment_id"):
            micro_map[str(item["micro_segment_id"])] = episode_id
        item["parent_segment_id"] = episode_id
        item["segment_id"] = episode_id
        item["episode_id"] = episode_id
        item["source_parent_segment_id"] = old_parent
        item["display_order"] = int(display_order_by_episode[episode_id])
        item["display_id"] = f"micro_{display_order_by_episode[episode_id]:03d}"
        _remap_micro_asset_bindings(item, episode_id)
        remapped.append(item)
    return remapped, micro_map


def _remap_micro_asset_bindings(row: dict[str, Any], episode_id: str) -> None:
    bindings = []
    for binding in row.get("asset_bindings") or []:
        if isinstance(binding, dict):
            item = dict(binding)
            item["parent_segment_id"] = episode_id
            item["segment_id"] = episode_id
            item["episode_id"] = episode_id
            bindings.append(item)
    row["asset_bindings"] = bindings


def _micro_ref(row: Mapping[str, Any]) -> dict[str, Any]:
    interaction = row.get("interaction") if isinstance(row.get("interaction"), Mapping) else {}
    keyframes = row.get("keyframes") if isinstance(row.get("keyframes"), Mapping) else {}
    first_person = row.get("first_person") if isinstance(row.get("first_person"), Mapping) else {}
    third_person = row.get("third_person") if isinstance(row.get("third_person"), Mapping) else {}
    quality = row.get("quality") if isinstance(row.get("quality"), Mapping) else {}
    evidence = row.get("evidence") if isinstance(row.get("evidence"), Mapping) else {}
    return {
        "micro_segment_id": row.get("micro_segment_id"),
        "display_order": row.get("display_order"),
        "display_id": row.get("display_id"),
        "primary_object": interaction.get("primary_object") or row.get("primary_object"),
        "interaction_type": interaction.get("interaction_type") or row.get("interaction_type"),
        "global_start_time": row.get("global_start_time"),
        "global_end_time": row.get("global_end_time"),
        "duration_sec": row.get("duration_sec"),
        "max_interaction_score": interaction.get("max_interaction_score"),
        "confidence": quality.get("confidence") if isinstance(quality, Mapping) else None,
        "peak_keyframe": keyframes.get("peak_frame"),
        "first_person_clip": first_person.get("clip_path"),
        "third_person_clip": third_person.get("clip_path"),
        "manual_corrected": row.get("manual_corrected", False),
        "dialogue_context_available": row.get("dialogue_context_available", False),
        "dialogue_match_window_sec": row.get("dialogue_match_window_sec"),
        "dialogue_keywords": row.get("dialogue_keywords", []),
        "evidence_level": evidence.get("evidence_level") or row.get("evidence_level"),
        "evidence": evidence,
        "asset_bindings": row.get("asset_bindings", []),
        "yolo_evidence": row.get("yolo_evidence", []),
        "class_threshold": row.get("class_threshold", {}),
        "merged_from_micro_segment_ids": row.get("merged_from_micro_segment_ids", []),
        "merge_reason": row.get("merge_reason"),
    }


def _attach_micro_refs_to_parent_dicts(key_segments: list[KeyActionSegment], micro_rows: list[dict[str, Any]]) -> None:
    refs_by_parent: dict[str, list[dict[str, Any]]] = {}
    for row in micro_rows:
        refs_by_parent.setdefault(str(row.get("parent_segment_id") or ""), []).append(_micro_ref(row))
    for segment in key_segments:
        segment.micro_segments = sorted(
            refs_by_parent.get(segment.segment_id, []),
            key=lambda item: int(item.get("display_order") or 0),
        )


def _attach_micro_refs_to_segment_rows(segment_rows: list[dict[str, Any]], micro_rows: list[dict[str, Any]]) -> None:
    refs_by_parent: dict[str, list[dict[str, Any]]] = {}
    for row in micro_rows:
        refs_by_parent.setdefault(str(row.get("parent_segment_id") or ""), []).append(_micro_ref(row))
    for row in segment_rows:
        row["micro_segments"] = sorted(
            refs_by_parent.get(str(row.get("segment_id") or ""), []),
            key=lambda item: int(item.get("display_order") or 0),
        )


def _apply_episode_support_quality_to_segment_row(row: dict[str, Any], support_quality: Mapping[str, Any]) -> None:
    quality = dict(support_quality)
    row["segment_support_quality"] = quality
    row["understanding_fact_strength"] = quality.get("fact_strength")
    row["strong_fact_allowed"] = bool(quality.get("strong_fact_allowed"))
    evidence = dict(_as_mapping(row.get("evidence")))
    original_level = str(evidence.get("evidence_level") or row.get("evidence_level") or "")
    evidence["segment_support_quality"] = quality
    if original_level:
        evidence.setdefault("original_segment_evidence_level", original_level)
    if not quality.get("strong_fact_allowed"):
        recommended = str(quality.get("recommended_evidence_level") or "weak_visual_evidence")
        evidence["evidence_level"] = recommended
        evidence["evidence_reasons"] = _ordered_unique(
            [
                *_as_strings(evidence.get("evidence_reasons")),
                *_as_strings(quality.get("reasons")),
                "parent_episode_downgraded_by_child_support_quality",
            ]
        )
        evidence["limitations"] = _ordered_unique(
            [
                *_as_strings(evidence.get("limitations")),
                "episode support is partial or weak; do not use as a strong fact without confirmation",
            ]
        )
        row["evidence_level"] = recommended
        row["evidence_reasons"] = evidence["evidence_reasons"]
        row["limitations"] = evidence["limitations"]
    elif original_level:
        row["evidence_level"] = original_level
    row["evidence"] = evidence


def _segment_dicts(
    key_segments: list[KeyActionSegment],
    specs: list[dict[str, Any]],
    source_duration_sec: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, key_segment in enumerate(key_segments):
        row = to_json_dict(key_segment)
        spec = specs[index]
        row["episode_id"] = key_segment.segment_id
        row["parent_kind"] = "experiment_episode"
        row["source_video_duration_sec"] = source_duration_sec
        row["episode_segmentation"] = {
            "schema_version": EPISODE_SCHEMA_VERSION,
            "source": "activity_episode_from_yolo_action_windows",
            "source_layer": spec.get("source_layer") or "episode_activity",
            "episode_status": spec.get("episode_status") or "official",
            "candidate_status": spec.get("candidate_status") or "official_episode",
            "official_episode": bool(spec.get("official_episode", True)),
            "formal_results_allowed": bool(spec.get("formal_results_allowed", True)),
            "single_view_candidate": bool(spec.get("single_view_candidate", False)),
            "candidate_reasons": spec.get("candidate_reasons", []),
            "min_official_episode_duration_sec": spec.get("min_official_episode_duration_sec"),
            "true_duration_sec": spec.get("true_duration_sec"),
            "true_start_sec": spec.get("true_start_sec"),
            "true_end_sec": spec.get("true_end_sec"),
            "anchor_micro_segment_id": spec.get("anchor_micro_segment_id"),
            "anchor_primary_object": spec.get("anchor_primary_object"),
            "micro_segment_ids": spec.get("micro_segment_ids", []),
            "primary_objects": spec.get("primary_objects", {}),
            "source_views": spec.get("source_views", []),
            "source_view_counts": spec.get("source_view_counts", {}),
            "evidence_source_views": spec.get("evidence_source_views", []),
            "evidence_source_view_counts": spec.get("evidence_source_view_counts", {}),
            "boundary_evidence": spec.get("boundary_evidence", {}),
            "episode_window_expansion": spec.get("episode_window_expansion", {}),
            "episode_merge_strategy": spec.get("episode_merge_strategy"),
            "episode_merge_strategies": spec.get("episode_merge_strategies", []),
            "episode_merge_reasons": spec.get("episode_merge_reasons", []),
            "episode_merge_events": spec.get("episode_merge_events", []),
            "macro_merge_gap_sec": spec.get("macro_merge_gap_sec"),
            "previous_episode_gap_sec": spec.get("previous_episode_gap_sec"),
            "previous_boundary_decision": spec.get("previous_boundary_decision"),
            "merged_from_anchor_micro_segment_ids": spec.get("merged_from_anchor_micro_segment_ids", []),
            "support_quality": spec.get("support_quality", {}),
            "understanding_fact_strength": spec.get("understanding_fact_strength"),
            "strong_fact_allowed": bool(spec.get("strong_fact_allowed")),
        }
        row["source_layer"] = spec.get("source_layer") or "episode_activity"
        row["episode_status"] = spec.get("episode_status") or "official"
        row["candidate_status"] = spec.get("candidate_status") or "official_episode"
        row["official_episode"] = bool(spec.get("official_episode", True))
        row["formal_results_allowed"] = bool(spec.get("formal_results_allowed", True))
        row["single_view_candidate"] = bool(spec.get("single_view_candidate", False))
        row["candidate_reasons"] = spec.get("candidate_reasons", [])
        row["episode_window_expansion"] = spec.get("episode_window_expansion", {})
        _apply_episode_support_quality_to_segment_row(row, spec.get("support_quality") or {})
        rows.append(row)
    return rows


def _episode_rows(
    manifest: SessionManifest,
    segment_rows: list[dict[str, Any]],
    specs: list[dict[str, Any]],
    detector_summary: Mapping[str, Any],
    source_duration_sec: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, segment in enumerate(segment_rows, start=1):
        cv = segment.get("cv_detection") if isinstance(segment.get("cv_detection"), Mapping) else {}
        spec = specs[index - 1]
        support_quality = dict(_as_mapping(spec.get("support_quality")))
        strong_fact_allowed = bool(support_quality.get("strong_fact_allowed"))
        rows.append(
            {
                "schema_version": EPISODE_SCHEMA_VERSION,
                "session_id": manifest.session_id,
                "episode_id": f"episode_{index:06d}",
                "segment_id": segment.get("segment_id"),
                "global_start_time": segment.get("global_start_time"),
                "global_end_time": segment.get("global_end_time"),
                "session_start_sec": cv.get("start_sec"),
                "session_end_sec": cv.get("end_sec"),
                "true_start_sec": spec.get("true_start_sec"),
                "true_end_sec": spec.get("true_end_sec"),
                "duration_sec": segment.get("duration_sec"),
                "source_video_duration_sec": source_duration_sec,
                "detector_backend": "yolo_episode",
                "detector_source_view": "multiview",
                "start_reason": cv.get("start_reason"),
                "end_reason": cv.get("end_reason"),
                "boundary_source": "activity_episode_window",
                "source_layer": spec.get("source_layer") or "episode_activity",
                "episode_status": spec.get("episode_status") or "official",
                "candidate_status": spec.get("candidate_status") or "official_episode",
                "official_episode": bool(spec.get("official_episode", True)),
                "formal_results_allowed": bool(spec.get("formal_results_allowed", True)),
                "single_view_candidate": bool(spec.get("single_view_candidate", False)),
                "candidate_reasons": spec.get("candidate_reasons", []),
                "min_official_episode_duration_sec": spec.get("min_official_episode_duration_sec"),
                "true_duration_sec": spec.get("true_duration_sec"),
                "boundary_evidence": spec.get("boundary_evidence", {}),
                "episode_window_expansion": spec.get("episode_window_expansion", {}),
                "episode_merge_strategy": spec.get("episode_merge_strategy"),
                "episode_merge_strategies": spec.get("episode_merge_strategies", []),
                "episode_merge_reasons": spec.get("episode_merge_reasons", []),
                "episode_merge_events": spec.get("episode_merge_events", []),
                "macro_merge_gap_sec": spec.get("macro_merge_gap_sec"),
                "previous_episode_gap_sec": spec.get("previous_episode_gap_sec"),
                "previous_boundary_decision": spec.get("previous_boundary_decision"),
                "view_alignment": dict(detector_summary.get("view_alignment") or {}),
                "micro_segment_ids": spec.get("micro_segment_ids", []),
                "merged_from_anchor_micro_segment_ids": spec.get("merged_from_anchor_micro_segment_ids", []),
                "primary_objects": spec.get("primary_objects", {}),
                "source_views": spec.get("source_views", []),
                "source_view_counts": spec.get("source_view_counts", {}),
                "evidence_source_views": spec.get("evidence_source_views", []),
                "evidence_source_view_counts": spec.get("evidence_source_view_counts", {}),
                "support_quality": support_quality,
                "understanding_fact_strength": support_quality.get("fact_strength"),
                "strong_fact_allowed": strong_fact_allowed,
                "recommended_evidence_level": support_quality.get("recommended_evidence_level"),
                "clips_by_view": {
                    "third_person": segment.get("third_person"),
                    "first_person": segment.get("first_person"),
                },
                "interpretation": (
                    "official_activity_episode_from_yolo_evidence"
                    if strong_fact_allowed
                    else "official_activity_episode_with_partial_or_weak_child_evidence"
                ),
            }
        )
    return rows


def _segment_vector_metadata_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    text = row.get("text_description") if isinstance(row.get("text_description"), Mapping) else {}
    index_info = row.get("index") if isinstance(row.get("index"), Mapping) else {}
    third = row.get("third_person") if isinstance(row.get("third_person"), Mapping) else {}
    first = row.get("first_person") if isinstance(row.get("first_person"), Mapping) else {}
    evidence = row.get("evidence") if isinstance(row.get("evidence"), Mapping) else {}
    episode_segmentation = row.get("episode_segmentation") if isinstance(row.get("episode_segmentation"), Mapping) else {}
    support_quality = dict(_as_mapping(row.get("segment_support_quality") or episode_segmentation.get("support_quality")))
    micro_refs = [item for item in row.get("micro_segments") or [] if isinstance(item, Mapping)]
    primary_object = next((str(item.get("primary_object")) for item in micro_refs if item.get("primary_object")), None)
    interaction_type = next((str(item.get("interaction_type")) for item in micro_refs if item.get("interaction_type")), None)
    detected_objects = sorted(
        {
            str(value)
            for item in micro_refs
            for value in [item.get("primary_object"), *(item.get("evidence", {}).get("detected_objects", []) if isinstance(item.get("evidence"), Mapping) else [])]
            if value
        }
    )
    return {
        "index_level": "segment",
        "embedding_id": index_info.get("embedding_id") or f"emb_{row.get('segment_id')}",
        "segment_id": row.get("segment_id"),
        "session_id": row.get("session_id"),
        "index_text": index_info.get("index_text") or text.get("summary") or "",
        "global_start_time": row.get("global_start_time"),
        "global_end_time": row.get("global_end_time"),
        "third_person_clip": third.get("clip_path"),
        "first_person_clip": first.get("clip_path"),
        "related_dialogue": row.get("dialogue_context", []),
        "action_type": text.get("action_type"),
        "interaction_keyframes": row.get("interaction_keyframes", []),
        "interaction_events": row.get("interaction_events", []),
        "yolo_interactions": row.get("yolo_interactions", []),
        "asset_bindings": row.get("asset_bindings", []),
        "primary_object": primary_object,
        "interaction_type": interaction_type,
        "detected_objects": detected_objects,
        "evidence": evidence,
        "evidence_level": evidence.get("evidence_level"),
        "evidence_reasons": evidence.get("evidence_reasons", []),
        "limitations": evidence.get("limitations", []),
        "segment_support_quality": support_quality,
        "understanding_fact_strength": row.get("understanding_fact_strength") or support_quality.get("fact_strength"),
        "strong_fact_allowed": bool(row.get("strong_fact_allowed")) if row.get("strong_fact_allowed") is not None else bool(support_quality.get("strong_fact_allowed")),
        "dialogue_context_available": bool(row.get("dialogue_context")),
        "dialogue_match_window_sec": row.get("dialogue_match_window_sec"),
        "dialogue_keywords": row.get("dialogue_keywords", []),
        "source_video_duration_sec": row.get("source_video_duration_sec"),
        "episode_window_expansion": episode_segmentation.get("episode_window_expansion") or row.get("episode_window_expansion") or {},
    }


def _rebuild_indexes(
    session_root: Path,
    segment_vectors: list[dict[str, Any]],
    micro_vectors: list[dict[str, Any]],
    combined_vectors: list[dict[str, Any]],
) -> None:
    index_root = session_root / "index"
    index = VectorIndex()
    index.build([str(item.get("index_text") or "") for item in combined_vectors], combined_vectors)
    index.save(index_root)
    write_jsonl(index_root / "docstore.jsonl", combined_vectors)
    segment_index = VectorIndex()
    segment_index.build([str(item.get("index_text") or "") for item in segment_vectors], segment_vectors)
    segment_index.save(index_root / "segments")
    micro_index = VectorIndex()
    micro_index.build([str(item.get("index_text") or "") for item in micro_vectors], micro_vectors)
    micro_index.save(index_root / "micro_segments")


def _rewrite_parent_segment_artifacts(session_root: Path, micro_map: dict[str, str]) -> None:
    for relative_path in (
        "metadata/model_observation_events.jsonl",
        "metadata/advanced_vision_evidence.jsonl",
        "metadata/video_understanding.jsonl",
        "metadata/unified_multimodal_timeline.jsonl",
    ):
        path = session_root / relative_path
        if not path.exists():
            continue
        try:
            rows = read_jsonl(path)
        except Exception:
            continue
        changed = False
        for row in rows:
            if not isinstance(row, dict):
                continue
            micro_id = str(row.get("micro_segment_id") or "")
            episode_id = micro_map.get(micro_id)
            if not episode_id:
                continue
            old_segment_id = str(row.get("segment_id") or row.get("parent_segment_id") or "")
            row["source_parent_segment_id"] = old_segment_id
            row["segment_id"] = episode_id
            row["parent_segment_id"] = episode_id
            row["episode_id"] = episode_id
            changed = True
        if changed:
            write_jsonl(path, rows)


def _focus_row_session_time(row: Mapping[str, Any]) -> float | None:
    try:
        if row.get("alignment_time_sec") is not None:
            return float(row.get("alignment_time_sec") or 0.0)
        return float(row.get("local_time_sec", row.get("time_sec", 0.0)) or 0.0)
    except (TypeError, ValueError):
        return None


def _focus_row_labels(row: Mapping[str, Any]) -> set[str]:
    labels: set[str] = set()
    counts = row.get("label_counts") if isinstance(row.get("label_counts"), Mapping) else {}
    for label, count in dict(counts).items():
        try:
            if int(count) <= 0:
                continue
        except (TypeError, ValueError):
            continue
        labels.add(_norm_activity_label(label))
    return labels


def _focus_activity_bounds_from_yolo_rows(rows: list[dict[str, Any]]) -> tuple[float, float] | None:
    times: list[float] = []
    for row in rows:
        labels = _focus_row_labels(row)
        if not row.get("hand_object_interactions") and not ((labels & _FOCUS_HAND_LABELS) and (labels & _FOCUS_CORE_OBJECTS)):
            continue
        time_sec = _focus_row_session_time(row)
        if time_sec is not None:
            times.append(float(time_sec))
    if not times:
        return None
    ordered = sorted(set(round(value, 6) for value in times))
    gaps = [
        ordered[index + 1] - ordered[index]
        for index in range(len(ordered) - 1)
        if ordered[index + 1] > ordered[index]
    ]
    sample_period = max(0.1, min(5.0, sorted(gaps)[len(gaps) // 2])) if gaps else 1.0
    start_sec = max(0.0, min(times))
    return start_sec, max(times) + sample_period


def _write_first_episode_focus(
    manifest: SessionManifest,
    session_root: Path,
    episode_rows: list[dict[str, Any]],
    *,
    yolo_frame_rows: list[dict[str, Any]] | None = None,
    dry_run: bool,
) -> dict[str, Any]:
    if not episode_rows:
        summary = {"available": False, "reason": "no_episode"}
        _write_json(session_root / "metadata" / "experiment_focus_clips.json", summary)
        return summary
    try:
        from .experiment_focus import extract_experiment_focus_clips

        return extract_experiment_focus_clips(session_root, dry_run=dry_run)
    except Exception:
        pass
    ordered = sorted(
        episode_rows,
        key=lambda row: float(row.get("true_start_sec") or row.get("session_start_sec") or 0.0),
    )
    first = ordered[0]
    start_sec = min(float(row.get("true_start_sec") or row.get("session_start_sec") or 0.0) for row in ordered)
    end_sec = max(float(row.get("true_end_sec") or row.get("session_end_sec") or start_sec) for row in ordered)
    yolo_activity_bounds = _focus_activity_bounds_from_yolo_rows(yolo_frame_rows or [])
    if yolo_activity_bounds is not None:
        start_sec = min(start_sec, yolo_activity_bounds[0])
        end_sec = max(end_sec, yolo_activity_bounds[1])
    included_segment_ids = [
        str(row.get("segment_id") or row.get("episode_id"))
        for row in ordered
        if row.get("segment_id") or row.get("episode_id")
    ]
    primary_objects: dict[str, Any] = {}
    for row in ordered:
        for label, count in dict(row.get("primary_objects") or {}).items():
            try:
                primary_objects[str(label)] = primary_objects.get(str(label), 0) + int(count)
            except (TypeError, ValueError):
                primary_objects.setdefault(str(label), count)
    global_start = local_sec_to_global_time(
        VideoSource("session", "session", manifest.session_start_time),
        start_sec,
    ).isoformat()
    global_end = local_sec_to_global_time(
        VideoSource("session", "session", manifest.session_start_time),
        end_sec,
    ).isoformat()
    window = {
        "schema_version": "experiment_focus_window.v1",
        "detected": True,
        "source": "all_true_experiment_episodes",
        "episode_count": len(ordered),
        "episode_id": first.get("episode_id"),
        "segment_id": first.get("segment_id"),
        "start_sec": round(start_sec, 6),
        "true_start_sec": round(start_sec, 6),
        "end_sec": round(end_sec, 6),
        "true_end_sec": round(end_sec, 6),
        "duration_sec": round(end_sec - start_sec, 6),
        "global_start_time": global_start,
        "global_end_time": global_end,
        "anchor": {
            "source": "yolo_micro_episode",
            "episode_id": first.get("episode_id"),
            "segment_id": first.get("segment_id"),
            "primary_objects": primary_objects,
        },
        "included_segment_ids": included_segment_ids,
        "segment_count": len(included_segment_ids),
    }
    _write_json(session_root / "metadata" / "experiment_focus_window.json", window)
    clip_rows = []
    for view, source in manifest.videos.all_sources().items():
        ref = None
        for row in ordered:
            clips_by_view = row.get("clips_by_view") if isinstance(row.get("clips_by_view"), Mapping) else {}
            ref = clips_by_view.get(view)
            if isinstance(ref, Mapping):
                break
        if not isinstance(ref, Mapping):
            continue
        clip_rows.append(
            {
                "view": view,
                "video_path": source.path,
                "clip_path": ref.get("clip_path"),
                "local_start_sec": start_sec,
                "local_end_sec": end_sec,
                "time_start_sec": start_sec,
                "time_end_sec": end_sec,
                "global_start_time": global_start,
                "global_end_time": global_end,
                "episode_id": first.get("episode_id"),
                "segment_id": first.get("segment_id"),
            }
        )
    summary = {
        "schema_version": "experiment_focus_clips.v1",
        "available": bool(clip_rows),
        "source": "all_true_experiment_episodes",
        "window": window,
        "clips": clip_rows,
        "clips_by_view": {row["view"]: row for row in clip_rows},
    }
    _write_json(session_root / "metadata" / "experiment_focus_clips.json", summary)
    return summary


__all__ = ["rebuild_episode_segments_from_micro_evidence"]

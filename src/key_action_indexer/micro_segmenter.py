from __future__ import annotations

import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .clip_extractor import extract_clip_ffmpeg
from .description_builder import build_micro_segment_description
from .evidence import apply_micro_evidence
from .evidence import attach_evidence
from .family_merge import object_family
from .schemas import (
    KeyActionSegment,
    MicroSegment,
    MicroSegmentConfig,
    MicroSegmentIndexInfo,
    MicroSegmentInteraction,
    MicroSegmentKeyframes,
    MicroSegmentQuality,
    MicroSegmentTextDescription,
    MicroSegmentView,
    SessionManifest,
    TranscriptUtterance,
    VideoSource,
)
from .physical_evidence import PHYSICAL_EVIDENCE_MIN_FRAMES, valid_yolo_physical_evidence, validate_yolo_physical_evidence
from .time_alignment import find_dialogue_for_segment, global_time_to_local_sec, local_sec_to_global_time, parse_time
from .yolo_detector import find_hand_object_interactions


HAND_LABELS = {"hand", "hands", "gloved_hand", "glove", "gloves"}
OBJECT_PRIORITY = {
    "balance": 78,
    "reagent_bottle": 90,
    "sample_bottle_blue": 84,
    "sample_bottle": 82,
    "bottle": 80,
    "spatula": 96,
    "pipette": 98,
    "pipette_tip": 99,
    "tube": 60,
    "beaker": 55,
    "paper": 88,
}

SMALL_TOOL_BOOST = {
    "spatula": 0.16,
    "pipette": 0.18,
    "pipette_tip": 0.20,
    "tube": 0.08,
    "paper": 0.12,
}

BACKGROUND_OBJECT_PENALTY = {
    "balance": 0.10,
    "paper": 0.0,
}

COVERAGE_BACKFILL_MAX_RUNS_PER_PARENT = 12


def _env_truthy(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() not in {"", "0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, str(default))))
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)


def _micro_asset_worker_count() -> int:
    if (
        _fast_locate_asset_mode()
        and os.environ.get("KEY_ACTION_MICRO_ASSET_WORKERS") is None
    ):
        return max(1, _env_int("KEY_ACTION_FAST_LOCATE_MICRO_ASSET_WORKERS", 4))
    return max(1, _env_int("KEY_ACTION_MICRO_ASSET_WORKERS", 4))


def _micro_parent_worker_count() -> int:
    if (
        _fast_locate_asset_mode()
        and os.environ.get("KEY_ACTION_MICRO_PARENT_WORKERS") is None
    ):
        return max(1, _env_int("KEY_ACTION_FAST_LOCATE_MICRO_PARENT_WORKERS", 1))
    return max(1, _env_int("KEY_ACTION_MICRO_PARENT_WORKERS", 1))


def _fast_locate_asset_mode() -> bool:
    return _env_truthy("KEY_ACTION_FAST_LOCATE_ONLY", False) or _env_truthy("KEY_ACTION_DEFER_SEGMENT_ASSETS", False)


def _micro_clip_views() -> set[str]:
    raw = os.environ.get("KEY_ACTION_MICRO_CLIP_VIEWS")
    if raw is None and _fast_locate_asset_mode():
        raw = os.environ.get("KEY_ACTION_FAST_LOCATE_MICRO_CLIP_VIEWS", "all")
    raw = "all" if raw is None else str(raw).strip().lower()
    if raw in {"", "0", "false", "no", "none", "off"}:
        return set()
    if raw in {"all", "both", "*"}:
        return {"third_person", "first_person"}
    if raw in {"evidence", "best", "primary", "source"}:
        return {"__evidence__"}
    aliases = {
        "third": "third_person",
        "top": "third_person",
        "top_view": "third_person",
        "operator": "first_person",
        "first": "first_person",
        "operator_view": "first_person",
    }
    views = set()
    for item in raw.replace(";", ",").split(","):
        key = item.strip().lower()
        if not key:
            continue
        views.add(aliases.get(key, key))
    return {view for view in views if view in {"third_person", "first_person"}}


def _micro_keyframe_roles() -> set[str]:
    raw = os.environ.get("KEY_ACTION_MICRO_KEYFRAME_ROLES")
    if raw is None and _fast_locate_asset_mode():
        raw = os.environ.get("KEY_ACTION_FAST_LOCATE_KEYFRAME_ROLES", "peak")
    raw = "contact,peak,release" if raw is None else str(raw).strip().lower()
    if raw in {"", "0", "false", "no", "none", "off"}:
        return set()
    if raw in {"all", "*"}:
        return {"contact", "peak", "release"}
    aliases = {"best": "peak", "mid": "peak", "middle": "peak"}
    roles = set()
    for item in raw.replace(";", ",").split(","):
        key = item.strip().lower()
        if not key:
            continue
        roles.add(aliases.get(key, key))
    return {role for role in roles if role in {"contact", "peak", "release"}}


def _micro_keyframe_box_mode() -> str:
    raw = os.environ.get("KEY_ACTION_MICRO_KEYFRAME_BOX_MODE")
    if raw is None and _fast_locate_asset_mode():
        raw = os.environ.get("KEY_ACTION_FAST_LOCATE_KEYFRAME_BOX_MODE", "strict")
    if raw is not None:
        value = str(raw).strip().lower()
        if value in {"", "0", "false", "no", "off", "none", "clean", "hide"}:
            return "clean"
        if value in {"strict", "trusted", "safe", "accurate"}:
            return "strict"
        if value in {"1", "true", "yes", "on", "always", "all", "audit"}:
            return "always"
    legacy = os.environ.get("KEY_ACTION_MICRO_KEYFRAME_DRAW_BOXES")
    if legacy is None and _fast_locate_asset_mode():
        legacy = os.environ.get("KEY_ACTION_FAST_LOCATE_KEYFRAME_DRAW_BOXES")
    if legacy is not None:
        return "always" if str(legacy).strip().lower() not in {"", "0", "false", "no", "off"} else "clean"
    return "strict" if _fast_locate_asset_mode() else "always"


def _micro_keyframe_draw_boxes_enabled() -> bool:
    return _micro_keyframe_box_mode() != "clean"


def _micro_clip_stream_copy_enabled() -> bool:
    raw = os.environ.get("KEY_ACTION_MICRO_CLIP_STREAM_COPY")
    if raw is None and _fast_locate_asset_mode():
        raw = os.environ.get("KEY_ACTION_FAST_LOCATE_MICRO_CLIP_STREAM_COPY")
    if raw is None:
        return _fast_locate_asset_mode()
    return str(raw).strip().lower() not in {"", "0", "false", "no", "off"}


def _max_micro_segments_per_parent() -> int:
    raw = os.environ.get("KEY_ACTION_MAX_MICROS_PER_SEGMENT")
    if raw is None and _fast_locate_asset_mode():
        raw = os.environ.get("KEY_ACTION_FAST_LOCATE_MAX_MICROS_PER_SEGMENT", "6")
    if raw is None:
        return 0
    try:
        return max(0, int(float(raw)))
    except (TypeError, ValueError):
        return 0


def _class_config(config: MicroSegmentConfig, primary_object: str) -> dict[str, float]:
    label = _label(primary_object)
    raw = dict(config.class_thresholds.get(label) or {})
    legacy_thresholds = getattr(config, "micro_object_thresholds", {}) or {}
    default_interaction_threshold = float(getattr(config, "micro_interaction_threshold", config.default_interaction_threshold))
    if label in legacy_thresholds:
        raw["interaction_threshold"] = float(legacy_thresholds[label])
    return {
        "interaction_threshold": float(raw.get("interaction_threshold", default_interaction_threshold)),
        "min_duration_sec": float(raw.get("min_duration_sec", getattr(config, "micro_min_duration_sec", config.default_min_duration_sec))),
        "query_boost": float(raw.get("query_boost", 1.0)),
    }


@dataclass
class InteractionState:
    frame_index: int
    local_time_sec: float
    global_time: str
    source_view: str
    hand_detected: bool
    objects_near_hand: list[str]
    primary_object: str
    interaction_type: str
    interaction_score: float
    hand_object_distance: float | None
    bbox_overlap: float
    detected_objects: list[str]
    raw_row: dict[str, Any]


@dataclass
class _InteractionRun:
    parent_segment: KeyActionSegment
    states: list[InteractionState]
    sequence: int


def _label(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _row_view(row: dict[str, Any]) -> str:
    view = _label(row.get("source_view") or row.get("view") or row.get("camera") or "third_person")
    if view in {"first", "first_person", "fpv", "egocentric", "bottom", "bottom_view", "operator", "head", "wrist"}:
        return "first_person"
    if view in {"third", "third_person", "top", "top_view", "overview", "external", "scene"}:
        return "third_person"
    return "third_person"


def _source_for_view(manifest: SessionManifest, view: str) -> VideoSource:
    if view == "first_person" and manifest.videos.first_person is not None:
        return manifest.videos.first_person
    return manifest.videos.third_person


def _parse_global(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return parse_time(str(value))
    except Exception:
        return None


def _session_sec(manifest: SessionManifest, global_time: str | datetime) -> float:
    return (parse_time(global_time) - parse_time(manifest.session_start_time)).total_seconds()


def _row_local_time(row: dict[str, Any]) -> float:
    for key in ("local_time_sec", "time_sec", "timestamp_sec", "sec", "t"):
        value = row.get(key)
        if value is not None:
            return float(value)
    frame_index = row.get("frame_index")
    fps = row.get("source_fps") or row.get("fps")
    if frame_index is not None and fps:
        return float(frame_index) / max(float(fps), 1e-6)
    return 0.0


def _row_global_time(manifest: SessionManifest, row: dict[str, Any]) -> str:
    global_value = row.get("global_time") or row.get("global_timestamp") or row.get("timestamp_global")
    if global_value:
        return str(global_value)
    view = _row_view(row)
    source = _source_for_view(manifest, view)
    return local_sec_to_global_time(source, _row_local_time(row)).isoformat()


def _row_in_parent(manifest: SessionManifest, row: dict[str, Any], parent: KeyActionSegment) -> bool:
    row_global = _parse_global(_row_global_time(manifest, row))
    if row_global is not None:
        return parse_time(parent.global_start_time) <= row_global <= parse_time(parent.global_end_time)
    view = _row_view(row)
    ref = parent.first_person if view == "first_person" and parent.first_person is not None else parent.third_person
    local_time = _row_local_time(row)
    return ref.local_start_sec - 0.25 <= local_time <= ref.local_end_sec + 0.25


def _detected_objects(row: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    counts = row.get("label_counts") or {}
    if isinstance(counts, dict):
        labels.extend(str(key) for key, value in counts.items() if int(value or 0) > 0)
    detections = row.get("detections") or []
    if isinstance(detections, list):
        for det in detections:
            if isinstance(det, dict):
                labels.append(str(det.get("label") or det.get("class_name") or det.get("name") or ""))
    seen: set[str] = set()
    ordered: list[str] = []
    for label in labels:
        normalized = _label(label)
        if normalized and normalized not in seen:
            seen.add(normalized)
            ordered.append(normalized)
    return ordered


def _hand_detected(row: dict[str, Any], detected_objects: list[str]) -> bool:
    if any(label in HAND_LABELS or "hand" in label for label in detected_objects):
        return True
    for item in row.get("hand_object_interactions") or []:
        if isinstance(item, dict) and item.get("hand_label"):
            return True
    return False


def _interaction_candidates(row: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = [dict(item) for item in (row.get("hand_object_interactions") or []) if isinstance(item, dict)]
    if candidates:
        return candidates
    detections = row.get("detections") or []
    if isinstance(detections, list) and detections:
        try:
            return find_hand_object_interactions(
                [item for item in detections if isinstance(item, dict)],
                frame_width=row.get("frame_width"),
                frame_height=row.get("frame_height"),
                min_interaction_score=0.05,
            )
        except Exception:
            return []
    return []


def _candidate_object(candidate: dict[str, Any]) -> str:
    return _label(candidate.get("object_label") or candidate.get("target_label") or candidate.get("object") or candidate.get("label"))


def _object_enabled(config: MicroSegmentConfig, primary_object: str) -> bool:
    label = _label(primary_object)
    if not label:
        return False
    disabled = {_label(item) for item in getattr(config, "disabled_primary_objects", []) or []}
    if label in disabled:
        return False
    allowed = {_label(item) for item in getattr(config, "allowed_primary_objects", []) or []}
    return not allowed or label in allowed


def _filter_candidates_for_config(candidates: list[dict[str, Any]], config: MicroSegmentConfig) -> list[dict[str, Any]]:
    return [candidate for candidate in candidates if _object_enabled(config, _candidate_object(candidate))]


def _sop_action_backcheck_objects(config: MicroSegmentConfig) -> list[str]:
    configured = list(getattr(config, "sop_action_backcheck_objects", []) or [])
    if not configured:
        configured = list(getattr(config, "allowed_primary_objects", []) or [])
    seen: set[str] = set()
    objects: list[str] = []
    for item in configured:
        label = _label(item)
        if label and label not in seen and _object_enabled(config, label):
            seen.add(label)
            objects.append(label)
    return objects


def _candidate_score(candidate: dict[str, Any]) -> float:
    value = candidate.get("score", candidate.get("confidence", candidate.get("interaction_score", 0.0)))
    return max(0.0, min(1.0, float(value or 0.0)))


def _candidate_overlap(candidate: dict[str, Any]) -> float:
    return max(0.0, min(1.0, float(candidate.get("iou", candidate.get("bbox_overlap", 0.0)) or 0.0)))


def _candidate_distance_score(candidate: dict[str, Any]) -> float:
    distance = candidate.get("normalized_distance")
    if distance is None:
        distance = candidate.get("distance_norm")
    if distance is None:
        return 0.5
    try:
        return max(0.0, min(1.0, 1.0 - float(distance)))
    except Exception:
        return 0.5


def _primary_rank(candidate: dict[str, Any]) -> float:
    label = _candidate_object(candidate)
    raw_score = _candidate_score(candidate)
    overlap = _candidate_overlap(candidate)
    distance_score = _candidate_distance_score(candidate)
    class_priority = OBJECT_PRIORITY.get(label, 40) / 100.0
    score = (
        raw_score * 0.55
        + overlap * 0.20
        + distance_score * 0.10
        + class_priority * 0.05
        + SMALL_TOOL_BOOST.get(label, 0.0)
        - BACKGROUND_OBJECT_PENALTY.get(label, 0.0)
    )
    if label == "balance" and overlap < 0.05 and raw_score < 0.7:
        score -= 0.08
    return score


def choose_primary_interaction(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None

    def sort_key(candidate: dict[str, Any]) -> tuple[float, float, int]:
        label = _candidate_object(candidate)
        return (_primary_rank(candidate), _candidate_score(candidate), OBJECT_PRIORITY.get(label, 0))

    best = max(candidates, key=sort_key)
    best_score = _primary_rank(best)
    surface_candidate = _contextual_surface_candidate(candidates, best_score)
    if surface_candidate is not None:
        return surface_candidate
    near = [
        item
        for item in candidates
        if best_score - _primary_rank(item) <= 0.03
    ]
    return max(near, key=lambda item: OBJECT_PRIORITY.get(_candidate_object(item), 0))


def _contextual_surface_candidate(candidates: list[dict[str, Any]], best_score: float) -> dict[str, Any] | None:
    paper_candidates = [item for item in candidates if _candidate_object(item) == "paper"]
    if not paper_candidates:
        return None
    best_paper = max(paper_candidates, key=lambda item: (_primary_rank(item), _candidate_score(item), _candidate_overlap(item)))
    paper_score = _primary_rank(best_paper)
    raw_score = _candidate_score(best_paper)
    overlap = _candidate_overlap(best_paper)
    best_label = _candidate_object(max(candidates, key=lambda item: _primary_rank(item)))
    has_weighing_surface_context = any(
        _candidate_object(item) in {"balance", "bottle", "sample_bottle", "sample_bottle_blue", "reagent_bottle"}
        for item in candidates
    )
    if has_weighing_surface_context and best_label != "spatula":
        if raw_score >= 0.45 and overlap >= 0.04 and best_score - paper_score <= 0.34:
            return best_paper
        if raw_score >= 0.68 and best_score - paper_score <= 0.38:
            return best_paper
    if raw_score >= 0.55 and overlap >= 0.07 and best_score - paper_score <= 0.20:
        return best_paper
    if raw_score >= 0.75 and best_score - paper_score <= 0.26:
        return best_paper
    return None


def _candidate_for_object(candidates: list[dict[str, Any]], primary_object: str) -> dict[str, Any] | None:
    matches = [candidate for candidate in candidates if _candidate_object(candidate) == primary_object]
    if not matches:
        return None
    return max(matches, key=lambda candidate: (_candidate_score(candidate), _primary_rank(candidate)))


def _state_from_candidate(
    manifest: SessionManifest,
    row: dict[str, Any],
    candidate: dict[str, Any],
    candidates: list[dict[str, Any]],
    detected: list[str],
    *,
    secondary_candidate: bool = False,
) -> InteractionState | None:
    primary_object = _candidate_object(candidate)
    if not primary_object:
        return None
    score = _candidate_score(candidate)
    view = _row_view(row)
    local_time = _row_local_time(row)
    global_time = _row_global_time(manifest, row)
    distance = candidate.get("distance_px")
    if distance is not None and row.get("frame_width") and row.get("frame_height"):
        diag = (float(row["frame_width"]) ** 2 + float(row["frame_height"]) ** 2) ** 0.5
        distance = float(distance) / max(diag, 1e-6)
    raw_row = row
    if secondary_candidate or not raw_row.get("hand_object_interactions"):
        raw_row = dict(row)
        if not raw_row.get("hand_object_interactions"):
            raw_row["hand_object_interactions"] = [candidate]
        if secondary_candidate:
            raw_row["_secondary_interaction_candidate"] = True
    return InteractionState(
        frame_index=int(row.get("frame_index", 0) or 0),
        local_time_sec=float(local_time),
        global_time=global_time,
        source_view=view,
        hand_detected=_hand_detected(row, detected),
        objects_near_hand=sorted({_candidate_object(item) for item in candidates if _candidate_object(item)}),
        primary_object=primary_object,
        interaction_type=f"hand_{primary_object}_contact",
        interaction_score=score,
        hand_object_distance=float(distance) if distance is not None else None,
        bbox_overlap=float(candidate.get("iou", candidate.get("bbox_overlap", 0.0)) or 0.0),
        detected_objects=detected,
        raw_row=raw_row,
    )


def _state_for_backchecked_object(
    manifest: SessionManifest,
    row: dict[str, Any],
    config: MicroSegmentConfig,
    primary_object: str,
) -> InteractionState | None:
    candidates = _filter_candidates_for_config(_interaction_candidates(row), config)
    candidate = _candidate_for_object(candidates, primary_object)
    if candidate is None:
        return None
    if _candidate_score(candidate) < _class_config(config, primary_object)["interaction_threshold"]:
        return None
    state = _state_from_candidate(manifest, row, candidate, candidates, _detected_objects(row))
    if state is None:
        return None
    state.raw_row = dict(state.raw_row)
    state.raw_row["_sop_action_backcheck"] = True
    state.raw_row["_sop_action_backcheck_object"] = primary_object
    return state


def compute_interaction_state(manifest: SessionManifest, row: dict[str, Any]) -> InteractionState | None:
    candidates = _interaction_candidates(row)
    primary = choose_primary_interaction(candidates)
    if primary is None:
        return None
    return _state_from_candidate(manifest, row, primary, candidates, _detected_objects(row))


def compute_interaction_states(
    manifest: SessionManifest,
    row: dict[str, Any],
    config: MicroSegmentConfig | None = None,
) -> list[InteractionState]:
    cfg = config or MicroSegmentConfig()
    candidates = _filter_candidates_for_config(_interaction_candidates(row), cfg)
    detected = _detected_objects(row)
    primary = choose_primary_interaction(candidates)
    states: list[InteractionState] = []
    primary_object = _candidate_object(primary) if primary is not None else None
    if primary is not None:
        state = _state_from_candidate(manifest, row, primary, candidates, detected)
        if state is not None:
            states.append(state)

    if not cfg.allow_secondary_interaction_candidates:
        return states

    for candidate in candidates:
        obj = _candidate_object(candidate)
        object_thresholds = getattr(cfg, "micro_object_thresholds", {}) or {}
        if not obj or obj == primary_object or (obj not in cfg.class_thresholds and obj not in object_thresholds):
            continue
        if not _object_enabled(cfg, obj):
            continue
        if _candidate_score(candidate) < _class_config(cfg, obj)["interaction_threshold"]:
            continue
        state = _state_from_candidate(
            manifest,
            row,
            candidate,
            candidates,
            detected,
            secondary_candidate=True,
        )
        if state is not None:
            states.append(state)
    return states


def compute_primary_interaction_timeline(
    manifest: SessionManifest,
    rows: list[dict[str, Any]],
    config: MicroSegmentConfig | None = None,
) -> list[InteractionState]:
    cfg = config or MicroSegmentConfig()
    ordered_rows = sorted(
        [row for row in rows if isinstance(row, dict)],
        key=lambda row: parse_time(_row_global_time(manifest, row)),
    )
    states: list[InteractionState] = []
    current_object = ""
    pending_object = ""
    pending_since: datetime | None = None
    for row in ordered_rows:
        candidates = _filter_candidates_for_config(_interaction_candidates(row), cfg)
        detected = _detected_objects(row)
        primary = choose_primary_interaction(candidates)
        if primary is None:
            continue
        proposed = _state_from_candidate(manifest, row, primary, candidates, detected)
        if proposed is None:
            continue
        if not cfg.single_primary_object_timeline or not current_object or proposed.primary_object == current_object:
            current_object = proposed.primary_object
            pending_object = ""
            pending_since = None
            states.append(proposed)
            continue

        current_candidate = _candidate_for_object(candidates, current_object)
        current_state = (
            _state_from_candidate(manifest, row, current_candidate, candidates, detected)
            if current_candidate is not None
            else None
        )
        if current_state is None:
            current_object = proposed.primary_object
            pending_object = ""
            pending_since = None
            states.append(proposed)
            continue

        current_threshold = _class_config(cfg, current_object)["interaction_threshold"]
        current_active = current_state.interaction_score >= current_threshold
        proposed_advantage = _primary_rank(primary) - _primary_rank(current_candidate)
        row_time = parse_time(proposed.global_time)
        if proposed.primary_object != pending_object:
            pending_object = proposed.primary_object
            pending_since = row_time
        stable_for = (row_time - pending_since).total_seconds() if pending_since is not None else 0.0
        should_switch = (not current_active) or (
            proposed_advantage >= cfg.primary_switch_margin and stable_for >= cfg.primary_min_stable_sec
        )
        if should_switch:
            current_object = proposed.primary_object
            pending_object = ""
            pending_since = None
            states.append(proposed)
        else:
            current_state.raw_row = dict(current_state.raw_row)
            current_state.raw_row["_primary_hysteresis_kept"] = True
            current_state.raw_row["_suppressed_primary_object"] = proposed.primary_object
            states.append(current_state)
    states = _apply_tracklet_primary_voting(manifest, states, cfg)
    return _collapse_multiview_primary_states(states, cfg)


def _run_primary_object(run: _InteractionRun, config: MicroSegmentConfig) -> str:
    return _label((_run_primary_vote_summary(run.states, config) or {}).get("primary_object") or (run.states[0].primary_object if run.states else ""))


def _run_bounds(run: _InteractionRun) -> tuple[datetime, datetime]:
    ordered = sorted(run.states, key=lambda item: parse_time(item.global_time))
    return parse_time(ordered[0].global_time), parse_time(ordered[-1].global_time)


def _run_overlap_seconds(left: _InteractionRun, right: _InteractionRun) -> float:
    left_start, left_end = _run_bounds(left)
    right_start, right_end = _run_bounds(right)
    overlap_start = max(left_start, right_start)
    overlap_end = min(left_end, right_end)
    return max(0.0, (overlap_end - overlap_start).total_seconds())


def _same_object_overlap(left: _InteractionRun, right: _InteractionRun, config: MicroSegmentConfig) -> bool:
    if _run_primary_object(left, config) != _run_primary_object(right, config):
        return False
    overlap = _run_overlap_seconds(left, right)
    if overlap <= 0.0:
        return False
    left_start, left_end = _run_bounds(left)
    right_start, right_end = _run_bounds(right)
    shortest = min(
        max(0.0, (left_end - left_start).total_seconds()),
        max(0.0, (right_end - right_start).total_seconds()),
    )
    return overlap >= max(0.20, shortest * 0.50)


def _run_avg_score(run: _InteractionRun) -> float:
    if not run.states:
        return 0.0
    return sum(float(state.interaction_score) for state in run.states) / len(run.states)


def _overlaps_stronger_confirmed_action(run: _InteractionRun, existing: _InteractionRun, config: MicroSegmentConfig) -> bool:
    left_object = _run_primary_object(run, config)
    right_object = _run_primary_object(existing, config)
    if not left_object or not right_object or left_object == right_object:
        return False
    overlap = _run_overlap_seconds(run, existing)
    if overlap <= 0.0:
        return False
    left_start, left_end = _run_bounds(run)
    right_start, right_end = _run_bounds(existing)
    shortest = min(
        max(0.0, (left_end - left_start).total_seconds()),
        max(0.0, (right_end - right_start).total_seconds()),
    )
    if overlap < max(0.35, shortest * 0.60):
        return False
    same_family = bool(_primary_family(left_object) and _primary_family(left_object) == _primary_family(right_object))
    lower_priority = OBJECT_PRIORITY.get(left_object, 0) < OBJECT_PRIORITY.get(right_object, 0)
    if not same_family and not lower_priority:
        return False
    return _run_avg_score(run) <= _run_avg_score(existing) + 0.10


def _sop_action_backcheck_runs(
    manifest: SessionManifest,
    parent: KeyActionSegment,
    parent_rows: list[dict[str, Any]],
    config: MicroSegmentConfig,
    existing_runs: list[_InteractionRun],
) -> list[_InteractionRun]:
    if not getattr(config, "sop_action_backcheck_enabled", True):
        return []
    objects = _sop_action_backcheck_objects(config)
    if not objects:
        return []
    min_valid_frames = max(1, int(getattr(config, "sop_action_min_valid_frames", 4) or 4))
    confirmed: list[_InteractionRun] = []
    all_known_runs = list(existing_runs)
    for primary_object in objects:
        states = [
            state
            for row in parent_rows
            if (state := _state_for_backchecked_object(manifest, row, config, primary_object)) is not None
        ]
        if not states:
            continue
        object_runs = build_interaction_runs(parent, states, config)
        for run in object_runs:
            if len(run.states) < min_valid_frames:
                continue
            if any(_same_object_overlap(run, existing, config) for existing in all_known_runs):
                continue
            if any(_overlaps_stronger_confirmed_action(run, existing, config) for existing in all_known_runs):
                continue
            for state in run.states:
                state.raw_row = dict(state.raw_row)
                state.raw_row["_sop_action_backcheck_confirmed"] = True
            confirmed.append(run)
            all_known_runs.append(run)
    return confirmed


def _renumber_runs(runs: list[_InteractionRun]) -> list[_InteractionRun]:
    ordered = sorted(runs, key=lambda run: parse_time(run.states[0].global_time) if run.states else datetime.max)
    for index, run in enumerate(ordered, start=1):
        run.sequence = index
    return ordered


def _final_micro_physical_evidence_required(config: MicroSegmentConfig) -> bool:
    return bool(_sop_action_backcheck_objects(config))


def _micro_has_final_physical_evidence(micro: MicroSegment) -> bool:
    valid = valid_yolo_physical_evidence(micro.yolo_evidence, micro.interaction.primary_object)
    return len(valid) >= PHYSICAL_EVIDENCE_MIN_FRAMES


def _tracklet_view_weight(config: MicroSegmentConfig, source_view: str) -> float:
    weights = getattr(config, "tracklet_view_weights", {}) or {}
    try:
        return float(weights.get(str(source_view or ""), 1.0))
    except Exception:
        return 1.0


def _primary_family(primary_object: str) -> str | None:
    family = object_family(_label(primary_object))
    return str(family) if family else None


def _state_vote_weight(state: InteractionState, config: MicroSegmentConfig) -> float:
    overlap_boost = 1.0 + min(0.25, max(0.0, state.bbox_overlap) * 0.35)
    distance_boost = 1.0
    if state.hand_object_distance is not None:
        distance_boost += min(0.20, max(0.0, 1.0 - state.hand_object_distance) * 0.12)
    return float(state.interaction_score) * _tracklet_view_weight(config, state.source_view) * overlap_boost * distance_boost


def _run_primary_vote_summary(states: list[InteractionState], config: MicroSegmentConfig) -> dict[str, Any]:
    object_scores: dict[str, float] = {}
    object_counts: dict[str, int] = {}
    family_scores: dict[str, float] = {}
    family_counts: dict[str, int] = {}
    for state in states:
        primary = _label(state.primary_object)
        if not primary:
            continue
        weight = _state_vote_weight(state, config)
        object_scores[primary] = object_scores.get(primary, 0.0) + weight
        object_counts[primary] = object_counts.get(primary, 0) + 1
        family = _primary_family(primary)
        if family:
            family_scores[family] = family_scores.get(family, 0.0) + weight
            family_counts[family] = family_counts.get(family, 0) + 1
    if not object_scores:
        return {
            "primary_object": "",
            "primary_object_family": None,
            "vote_score": 0.0,
            "vote_margin": 0.0,
            "vote_counts": {},
            "vote_scores": {},
            "arbitration": "tracklet_window_vote_empty",
        }

    family_weight = float(getattr(config, "tracklet_family_vote_weight", 0.0) or 0.0) if getattr(config, "tracklet_family_vote_enabled", True) else 0.0
    adjusted_scores: dict[str, float] = {}
    for primary, score in object_scores.items():
        family = _primary_family(primary)
        adjusted_scores[primary] = score + family_weight * family_scores.get(family or "", 0.0)

    winner = max(
        adjusted_scores,
        key=lambda item: (
            adjusted_scores[item],
            object_counts.get(item, 0),
            OBJECT_PRIORITY.get(item, 0),
        ),
    )
    if getattr(config, "tracklet_family_vote_enabled", True) and family_scores:
        dominant_family = max(family_scores, key=lambda item: (family_scores[item], family_counts.get(item, 0)))
        winner_family = _primary_family(winner)
        if dominant_family and dominant_family != winner_family:
            family_candidates = [item for item in adjusted_scores if _primary_family(item) == dominant_family]
            if family_candidates:
                family_winner = max(
                    family_candidates,
                    key=lambda item: (
                        adjusted_scores[item],
                        object_counts.get(item, 0),
                        OBJECT_PRIORITY.get(item, 0),
                    ),
                )
                margin = float(getattr(config, "tracklet_family_vote_margin", 0.0) or 0.0)
                if adjusted_scores[family_winner] + margin >= adjusted_scores[winner]:
                    winner = family_winner

    if "paper" in adjusted_scores and winner in {"balance", "bottle", "sample_bottle", "sample_bottle_blue", "reagent_bottle"}:
        paper_count = object_counts.get("paper", 0)
        winner_count = object_counts.get(winner, 0)
        continuity_bonus = 0.10 if paper_count >= 2 else 0.0
        surface_margin = 0.18 + continuity_bonus
        if paper_count >= max(1, min(winner_count, 2)) and adjusted_scores["paper"] + surface_margin >= adjusted_scores[winner]:
            winner = "paper"

    ranked_scores = sorted(adjusted_scores.values(), reverse=True)
    vote_score = adjusted_scores[winner]
    vote_margin = vote_score - (ranked_scores[1] if len(ranked_scores) > 1 else 0.0)
    return {
        "primary_object": winner,
        "primary_object_family": _primary_family(winner),
        "vote_score": round(float(vote_score), 6),
        "vote_margin": round(float(vote_margin), 6),
        "vote_counts": {key: int(value) for key, value in sorted(object_counts.items())},
        "vote_scores": {key: round(float(adjusted_scores[key]), 6) for key in sorted(adjusted_scores)},
        "arbitration": "tracklet_family_window_vote" if family_weight > 0 else "tracklet_window_vote",
    }


def _same_object_or_family(left: str, right: str, config: MicroSegmentConfig) -> bool:
    left_label = _label(left)
    right_label = _label(right)
    if left_label == right_label:
        return True
    if not getattr(config, "tracklet_family_run_merge_enabled", True):
        return False
    left_family = _primary_family(left_label)
    right_family = _primary_family(right_label)
    return bool(left_family and right_family and left_family == right_family)


def _apply_tracklet_primary_voting(
    manifest: SessionManifest,
    states: list[InteractionState],
    config: MicroSegmentConfig,
) -> list[InteractionState]:
    if not getattr(config, "tracklet_primary_vote_enabled", True):
        return states
    window_sec = max(0.0, float(getattr(config, "tracklet_vote_window_sec", 0.0) or 0.0))
    if window_sec <= 0.0 or len(states) < 2:
        return states
    min_count = max(1, int(getattr(config, "tracklet_vote_min_count", 1) or 1))
    margin = max(0.0, float(getattr(config, "tracklet_vote_margin", 0.0) or 0.0))
    ordered = sorted(states, key=lambda item: parse_time(item.global_time))
    voted: list[InteractionState] = []
    for state in ordered:
        center = parse_time(state.global_time)
        votes: dict[str, float] = {}
        counts: dict[str, int] = {}
        for neighbor in ordered:
            delta = abs((parse_time(neighbor.global_time) - center).total_seconds())
            if delta > window_sec:
                continue
            decay = 1.0 - 0.5 * (delta / max(window_sec, 1e-6))
            weight = neighbor.interaction_score * _tracklet_view_weight(config, neighbor.source_view) * decay
            votes[neighbor.primary_object] = votes.get(neighbor.primary_object, 0.0) + weight
            counts[neighbor.primary_object] = counts.get(neighbor.primary_object, 0) + 1
        if not votes:
            voted.append(state)
            continue
        winner = max(votes, key=lambda key: votes[key])
        current_vote = votes.get(state.primary_object, 0.0)
        if winner == state.primary_object or counts.get(winner, 0) < min_count or votes[winner] < current_vote + margin:
            voted.append(state)
            continue
        candidate = _candidate_for_object(_interaction_candidates(state.raw_row), winner)
        if candidate is None:
            kept = state
            kept.raw_row = dict(kept.raw_row)
            kept.raw_row["_tracklet_vote_winner_unavailable"] = winner
            voted.append(kept)
            continue
        reassigned = _state_from_candidate(manifest, state.raw_row, candidate, _interaction_candidates(state.raw_row), _detected_objects(state.raw_row))
        if reassigned is None:
            voted.append(state)
            continue
        reassigned.raw_row = dict(reassigned.raw_row)
        reassigned.raw_row["_tracklet_vote_reassigned"] = True
        reassigned.raw_row["_tracklet_vote_original_primary_object"] = state.primary_object
        reassigned.raw_row["_tracklet_vote_score"] = round(float(votes[winner]), 6)
        reassigned.raw_row["_tracklet_vote_current_score"] = round(float(current_vote), 6)
        voted.append(reassigned)
    return voted


def _collapse_multiview_primary_states(states: list[InteractionState], config: MicroSegmentConfig) -> list[InteractionState]:
    if not getattr(config, "single_primary_object_timeline", True):
        return states
    buckets: dict[float, list[InteractionState]] = {}
    for state in states:
        key = round(parse_time(state.global_time).timestamp(), 3)
        buckets.setdefault(key, []).append(state)
    collapsed: list[InteractionState] = []
    for bucket_states in buckets.values():
        if len(bucket_states) == 1:
            collapsed.append(bucket_states[0])
            continue
        collapsed.append(
            max(
                bucket_states,
                key=lambda item: (
                    item.interaction_score * _tracklet_view_weight(config, item.source_view),
                    OBJECT_PRIORITY.get(item.primary_object, 0),
                ),
            )
        )
    return sorted(collapsed, key=lambda item: parse_time(item.global_time))


def _sample_period(states: list[InteractionState], default: float = 0.5) -> float:
    times = sorted(state.local_time_sec for state in states)
    deltas = [b - a for a, b in zip(times, times[1:]) if b > a]
    if not deltas:
        return default
    return max(0.01, min(deltas))


def build_interaction_runs(
    parent_segment: KeyActionSegment,
    states: list[InteractionState],
    config: MicroSegmentConfig | None = None,
) -> list[_InteractionRun]:
    cfg = config or MicroSegmentConfig()

    def has_yolo_detection_input(state: InteractionState) -> bool:
        row = state.raw_row or {}
        return bool(row.get("detections"))

    def is_active_micro_state(state: InteractionState) -> bool:
        threshold = _class_config(cfg, state.primary_object)["interaction_threshold"]
        active = (
            state.hand_detected
            and bool(state.primary_object)
            and _object_enabled(cfg, state.primary_object)
            and state.interaction_score >= threshold
        )
        if not active:
            return False
        if has_yolo_detection_input(state):
            valid, _reasons = validate_yolo_physical_evidence(state.raw_row or {}, state.primary_object)
            return valid
        return True

    ordered = sorted(
        [state for state in states if is_active_micro_state(state)],
        key=lambda state: parse_time(state.global_time),
    )
    if cfg.single_primary_object_timeline and cfg.micro_object_change_split:
        runs: list[_InteractionRun] = []
        current: list[InteractionState] = []
        for state in ordered:
            if not current:
                current = [state]
                continue
            prev = current[-1]
            gap = (parse_time(state.global_time) - parse_time(prev.global_time)).total_seconds()
            object_changed = not _same_object_or_family(state.primary_object, prev.primary_object, cfg)
            if object_changed or gap > cfg.micro_merge_gap_sec:
                runs.append(_InteractionRun(parent_segment=parent_segment, states=current, sequence=0))
                current = [state]
            else:
                current.append(state)
        if current:
            runs.append(_InteractionRun(parent_segment=parent_segment, states=current, sequence=0))

        for index, run in enumerate(runs, start=1):
            run.sequence = index
        return runs

    grouped: dict[str, list[InteractionState]] = {}
    if cfg.micro_object_change_split:
        for state in ordered:
            key = state.primary_object
            if getattr(cfg, "tracklet_family_run_merge_enabled", True):
                key = _primary_family(state.primary_object) or state.primary_object
            grouped.setdefault(key, []).append(state)
    else:
        grouped["__all__"] = ordered

    runs: list[_InteractionRun] = []
    for object_states in grouped.values():
        current: list[InteractionState] = []
        for state in sorted(object_states, key=lambda item: parse_time(item.global_time)):
            if not current:
                current = [state]
                continue
            prev = current[-1]
            gap = (parse_time(state.global_time) - parse_time(prev.global_time)).total_seconds()
            if gap > cfg.micro_merge_gap_sec:
                runs.append(_InteractionRun(parent_segment=parent_segment, states=current, sequence=0))
                current = [state]
            else:
                current.append(state)
        if current:
            runs.append(_InteractionRun(parent_segment=parent_segment, states=current, sequence=0))

    runs.sort(key=lambda run: parse_time(run.states[0].global_time))
    for index, run in enumerate(runs, start=1):
        run.sequence = index
    return runs


def _coverage_backfill_config(config: MicroSegmentConfig) -> MicroSegmentConfig:
    class_thresholds = deepcopy(config.class_thresholds)
    for raw in class_thresholds.values():
        raw["interaction_threshold"] = min(float(raw.get("interaction_threshold", config.default_interaction_threshold)), 0.45)
        raw["min_duration_sec"] = min(float(raw.get("min_duration_sec", config.default_min_duration_sec)), 0.25)
    return replace(
        config,
        default_interaction_threshold=min(float(config.default_interaction_threshold), 0.45),
        default_min_duration_sec=min(float(config.default_min_duration_sec), 0.25),
        micro_interaction_threshold=min(float(config.micro_interaction_threshold), 0.45),
        micro_min_duration_sec=min(float(config.micro_min_duration_sec), 0.25),
        class_thresholds=class_thresholds,
        micro_object_thresholds={
            str(key): min(float(value), 0.45)
            for key, value in dict(config.micro_object_thresholds or {}).items()
        },
    )


def _coverage_candidate_threshold(config: MicroSegmentConfig, primary_object: str) -> float:
    return max(0.18, float(_class_config(config, primary_object)["interaction_threshold"]) - 0.40)


def _coverage_candidate_states(states: list[InteractionState], config: MicroSegmentConfig) -> list[InteractionState]:
    candidates: list[InteractionState] = []
    for state in states:
        if not state.hand_detected or not state.primary_object:
            continue
        if not _object_enabled(config, state.primary_object):
            continue
        if not state.detected_objects and not (state.raw_row or {}).get("detections"):
            continue
        if float(state.interaction_score) < _coverage_candidate_threshold(config, state.primary_object):
            continue
        patched = state
        patched.raw_row = dict(patched.raw_row or {})
        patched.raw_row["_coverage_backfill_candidate"] = True
        candidates.append(patched)
    return candidates


def _coverage_backfill_runs(
    parent_segment: KeyActionSegment,
    states: list[InteractionState],
    config: MicroSegmentConfig,
) -> list[_InteractionRun]:
    ordered = sorted(_coverage_candidate_states(states, config), key=lambda state: parse_time(state.global_time))
    runs: list[_InteractionRun] = []
    current: list[InteractionState] = []
    for state in ordered:
        if not current:
            current = [state]
            continue
        prev = current[-1]
        gap = (parse_time(state.global_time) - parse_time(prev.global_time)).total_seconds()
        object_changed = not _same_object_or_family(state.primary_object, prev.primary_object, config)
        if object_changed or gap > config.micro_merge_gap_sec:
            runs.append(_InteractionRun(parent_segment=parent_segment, states=current, sequence=0))
            current = [state]
        else:
            current.append(state)
    if current:
        runs.append(_InteractionRun(parent_segment=parent_segment, states=current, sequence=0))
    for index, run in enumerate(runs, start=1):
        run.sequence = index
    return runs


def _coverage_interaction_states(
    manifest: SessionManifest,
    rows: list[dict[str, Any]],
    config: MicroSegmentConfig,
) -> list[InteractionState]:
    states: list[InteractionState] = []
    ordered_rows = sorted(
        [row for row in rows if isinstance(row, dict)],
        key=lambda row: parse_time(_row_global_time(manifest, row)),
    )
    for row in ordered_rows:
        candidates = _filter_candidates_for_config(_interaction_candidates(row), config)
        detected = _detected_objects(row)
        for candidate in candidates:
            obj = _candidate_object(candidate)
            if not obj or not _object_enabled(config, obj):
                continue
            if _candidate_score(candidate) < _coverage_candidate_threshold(config, obj):
                continue
            state = _state_from_candidate(
                manifest,
                row,
                candidate,
                candidates,
                detected,
                secondary_candidate=True,
            )
            if state is not None:
                states.append(state)
    return states


def _coverage_run_overlaps_existing_micro(
    run: _InteractionRun,
    existing_micros: list[MicroSegment],
    config: MicroSegmentConfig,
) -> bool:
    if not existing_micros:
        return False
    run_object = _run_primary_object(run, config)
    if not run_object:
        return False
    run_start, run_end = _run_bounds(run)
    for micro in existing_micros:
        micro_objects = []
        if micro.interaction:
            micro_objects.append(_label(micro.interaction.primary_object))
            micro_objects.extend(_label(item) for item in micro.interaction.secondary_objects or [])
        if not any(_same_object_or_family(run_object, micro_object, config) for micro_object in micro_objects if micro_object):
            continue
        micro_start = parse_time(micro.global_start_time)
        micro_end = parse_time(micro.global_end_time)
        overlap_start = max(run_start, micro_start)
        overlap_end = min(run_end, micro_end)
        if (overlap_end - overlap_start).total_seconds() > 0.0:
            return True
    return False


def _coverage_run_rank(run: _InteractionRun, config: MicroSegmentConfig) -> tuple[int, float, int, datetime]:
    primary = _run_primary_object(run, config)
    bbox_frame_count = sum(1 for state in run.states if _state_has_physical_evidence(state, primary))
    max_score = max((float(state.interaction_score) for state in run.states), default=0.0)
    start, _end = _run_bounds(run)
    return (bbox_frame_count, max_score, len(run.states), start)


def _limit_interaction_runs(
    runs: list[_InteractionRun],
    config: MicroSegmentConfig,
    *,
    max_count: int,
) -> list[_InteractionRun]:
    if max_count <= 0 or len(runs) <= max_count:
        return runs
    ranked = sorted(runs, key=lambda run: _coverage_run_rank(run, config), reverse=True)[:max_count]
    ranked.sort(key=lambda run: parse_time(run.states[0].global_time) if run.states else datetime.max)
    for index, run in enumerate(ranked, start=1):
        run.sequence = index
    return ranked


def _mark_coverage_backfill_micro(micro: MicroSegment, run: _InteractionRun) -> MicroSegment:
    warnings = set(micro.quality.warnings or [])
    warnings.update({"coverage_backfill_candidate", "physical_evidence_validation_relaxed"})
    primary = micro.interaction.primary_object if micro.interaction else None
    frame_count = len(run.states)
    bbox_frame_count = sum(1 for state in run.states if _state_has_physical_evidence(state, primary))
    max_score = max((float(state.interaction_score) for state in run.states), default=0.0)
    avg_score = sum(float(state.interaction_score) for state in run.states) / frame_count if frame_count else 0.0
    if frame_count <= 1:
        warnings.add("single_frame_coverage_candidate")
        warnings.add("low_signal_yolo_candidate")
    if bbox_frame_count <= 1:
        warnings.add("weak_bbox_continuity")
    if max_score < 0.35:
        warnings.add("very_low_signal_yolo_candidate")
    elif max_score < 0.5:
        warnings.add("low_signal_yolo_candidate")
    micro.quality.warnings = sorted(warnings)
    micro.manual_corrected = True
    micro.manual_correction_note = "auto_coverage_backfill_from_yolo_parent_rows"
    evidence = dict(micro.evidence or {})
    limitations = [str(item) for item in evidence.get("limitations") or []]
    limitations.append("coverage backfill candidate; requires human/model confirmation before strong process claims")
    evidence["limitations"] = sorted(set(limitations))
    evidence["coverage_backfill"] = True
    evidence["coverage_backfill_reason"] = "parent segment had YOLO hand-object candidates but no validated micro-segment"
    evidence["coverage_signal_grade"] = _coverage_signal_grade(frame_count, bbox_frame_count, max_score)
    evidence["coverage_evidence_frame_count"] = frame_count
    evidence["coverage_bbox_frame_count"] = bbox_frame_count
    evidence["coverage_max_interaction_score"] = round(max_score, 6)
    evidence["coverage_avg_interaction_score"] = round(avg_score, 6)
    micro.evidence = evidence
    return micro


def _coverage_signal_grade(frame_count: int, bbox_frame_count: int, max_score: float) -> str:
    if max_score < 0.35:
        return "very_low_signal_yolo_candidate"
    if frame_count <= 1 or bbox_frame_count <= 1:
        return "single_frame_yolo_candidate"
    if bbox_frame_count < frame_count:
        return "continuous_yolo_candidate"
    return "physical_continuity_candidate"


def _quality(states: list[InteractionState], duration_sec: float, warnings: list[str]) -> MicroSegmentQuality:
    max_score = max((state.interaction_score for state in states), default=0.0)
    if max_score >= 0.75 and duration_sec >= 2.0:
        confidence = "high"
    elif max_score >= 0.5:
        confidence = "medium"
    else:
        confidence = "low"
    if duration_sec < 1.0:
        warnings.append("very_short_micro_segment")
    if max_score < 0.5:
        warnings.append("low_interaction_score")
    if len(states) <= 1:
        warnings.append("only_single_frame_evidence")
    return MicroSegmentQuality(confidence=confidence, warnings=sorted(set(warnings)))


def _evidence_interaction_for_state(state: InteractionState, primary_object: str | None) -> dict[str, Any] | None:
    interactions = [item for item in (state.raw_row.get("hand_object_interactions") or []) if isinstance(item, dict)]
    if not interactions:
        return None
    primary = _label(primary_object or state.primary_object)
    matching = [item for item in interactions if _candidate_object(item) == primary]
    candidates = matching or interactions
    return max(candidates, key=lambda item: _candidate_score(item))


def _has_evidence_boxes(evidence: dict[str, Any] | None) -> bool:
    if not evidence:
        return False
    hand_bbox = evidence.get("hand_bbox")
    object_bbox = evidence.get("object_bbox")
    return isinstance(hand_bbox, list) and len(hand_bbox) >= 4 and isinstance(object_bbox, list) and len(object_bbox) >= 4


def _state_has_physical_evidence(state: InteractionState, primary_object: str | None) -> bool:
    return _has_evidence_boxes(_evidence_interaction_for_state(state, primary_object))


def _state_keyframe_rank(state: InteractionState, primary_object: str | None) -> tuple[float, float, float, float]:
    evidence = _evidence_interaction_for_state(state, primary_object)
    evidence_score = _candidate_score(evidence or {}) if evidence else 0.0
    box_bonus = 1.0 if _has_evidence_boxes(evidence) else 0.0
    distance_bonus = 0.0
    if state.hand_object_distance is not None:
        distance_bonus = max(0.0, 1.0 - float(state.hand_object_distance))
    return (
        box_bonus,
        max(float(state.interaction_score), evidence_score),
        float(state.bbox_overlap or 0.0),
        distance_bonus,
    )


def _select_physical_keyframe_states(
    states: list[InteractionState],
    primary_object: str | None,
) -> tuple[InteractionState, InteractionState, InteractionState, list[str]]:
    primary = _label(primary_object)
    primary_states = [state for state in states if _label(state.primary_object) == primary] if primary else []
    scoped_states = primary_states or states
    physical_states = [state for state in scoped_states if _state_has_physical_evidence(state, primary_object)]
    keyframe_states = physical_states or scoped_states
    warnings: list[str] = []
    if not physical_states:
        warnings.append("missing_hand_object_bbox_keyframe_evidence")
    contact_state = keyframe_states[0]
    peak_state = max(keyframe_states, key=lambda state: _state_keyframe_rank(state, primary_object))
    release_state = keyframe_states[-1]
    return contact_state, peak_state, release_state, warnings


def _nearest_view_evidence_state(
    states: list[InteractionState],
    *,
    view: str,
    target_global_time: str,
    primary_object: str | None,
    max_delta_sec: float = 0.75,
) -> InteractionState | None:
    target_view = _label(view)
    target_time = parse_time(target_global_time)
    primary = _label(primary_object)
    candidates: list[tuple[float, tuple[float, float, float, float], InteractionState]] = []
    for state in states:
        if _label(state.source_view) != target_view:
            continue
        if primary and _label(state.primary_object) != primary:
            continue
        delta = abs((parse_time(state.global_time) - target_time).total_seconds())
        if delta > max_delta_sec:
            continue
        candidates.append((delta, _state_keyframe_rank(state, primary_object), state))
    if not candidates:
        return None
    return max(candidates, key=lambda item: (-item[0], item[1]))[2]


def _scaled_box_for_frame(
    bbox: Any,
    *,
    row: dict[str, Any],
    frame: Any,
    strict: bool,
) -> list[int] | None:
    if not bbox or len(bbox) < 4:
        return None
    try:
        height, width = frame.shape[:2]
        raw = [float(value) for value in bbox[:4]]
    except Exception:
        return None
    src_width = 0.0
    src_height = 0.0
    try:
        src_width = float(row.get("frame_width") or row.get("width") or 0.0)
        src_height = float(row.get("frame_height") or row.get("height") or 0.0)
    except Exception:
        src_width, src_height = 0.0, 0.0
    if src_width <= 0 or src_height <= 0:
        if strict:
            return None
        src_width, src_height = float(width), float(height)
    scale_x = float(width) / max(src_width, 1.0)
    scale_y = float(height) / max(src_height, 1.0)
    if strict and not (0.25 <= scale_x <= 4.0 and 0.25 <= scale_y <= 4.0):
        return None
    x1, y1, x2, y2 = [
        int(round(raw[0] * scale_x)),
        int(round(raw[1] * scale_y)),
        int(round(raw[2] * scale_x)),
        int(round(raw[3] * scale_y)),
    ]
    x1, x2 = max(0, min(x1, width - 1)), max(0, min(x2, width - 1))
    y1, y2 = max(0, min(y1, height - 1)), max(0, min(y2, height - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _draw_evidence_boxes(
    frame: Any,
    state: InteractionState,
    primary_object: str | None,
    *,
    source_view: str | None = None,
    target_local_time_sec: float | None = None,
    strict: bool = False,
) -> Any:
    import cv2

    row = state.raw_row or {}
    if strict:
        if source_view and _label(source_view) != _label(state.source_view):
            return frame
        if target_local_time_sec is not None:
            delta = abs(float(state.local_time_sec) - float(target_local_time_sec))
            max_delta = max(0.0, _env_float("KEY_ACTION_KEYFRAME_BOX_MAX_DELTA_SEC", 0.18))
            if delta > max_delta:
                return frame
        if not row.get("frame_width") or not row.get("frame_height"):
            return frame
    evidence = _evidence_interaction_for_state(state, primary_object)
    if not evidence:
        return frame
    hand_bbox = evidence.get("hand_bbox")
    object_bbox = evidence.get("object_bbox")
    hand_label = str(evidence.get("hand_label") or "hand")
    object_label = str(evidence.get("object_label") or primary_object or state.primary_object or "object")
    score = float(evidence.get("score", state.interaction_score) or 0.0)

    def _draw_box(bbox: Any, label: str, color: tuple[int, int, int]) -> None:
        scaled = _scaled_box_for_frame(bbox, row=row, frame=frame, strict=strict)
        if scaled is None:
            return
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = scaled
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        label_text = f"{label} {score:.2f}" if label == object_label else label
        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        text_y = max(0, y1 - text_h - 8)
        cv2.rectangle(frame, (x1, text_y), (min(width - 1, x1 + text_w + 8), y1), color, -1)
        cv2.putText(frame, label_text, (x1 + 4, max(text_h + 1, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    _draw_box(object_bbox, object_label, (0, 128, 255))
    _draw_box(hand_bbox, hand_label, (0, 220, 0))
    footer = f"physical evidence: {hand_label} -> {object_label} | score={score:.2f} | t={state.local_time_sec:.2f}s"
    cv2.rectangle(frame, (0, 0), (min(frame.shape[1] - 1, 760), 30), (20, 20, 20), -1)
    cv2.putText(frame, footer, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return frame


def _write_keyframe(
    source: VideoSource,
    local_time_sec: float,
    output_path: Path,
    *,
    dry_run: bool,
    evidence_state: InteractionState | None = None,
    primary_object: str | None = None,
) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        output_path.write_bytes(f"DRY RUN MICRO KEYFRAME {local_time_sec:.3f}\n".encode("utf-8"))
        return str(output_path)
    source_image = None
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("opencv-python is required for micro keyframe extraction") from exc
    if output_path.exists() and evidence_state is None:
        box_mode = _micro_keyframe_box_mode()
        if not _fast_locate_asset_mode() or box_mode == "always":
            return str(output_path)
    cap = cv2.VideoCapture(source.path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for micro keyframe extraction: {source.path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        duration = frame_count / fps if fps > 0 and frame_count > 0 else None
        seek = max(0.0, float(local_time_sec))
        if duration is not None:
            seek = min(seek, max(0.0, duration - 0.08))
        cap.set(cv2.CAP_PROP_POS_MSEC, seek * 1000.0)
        ok, frame = cap.read()
        if not ok and frame_count > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_count) - 1))
            ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Cannot read micro keyframe at {local_time_sec:.3f}s from {source.path}")
        box_mode = _micro_keyframe_box_mode()
        if evidence_state is not None and box_mode != "clean":
            frame = _draw_evidence_boxes(
                frame,
                evidence_state,
                primary_object,
                source_view=source.role or source.name,
                target_local_time_sec=local_time_sec,
                strict=box_mode == "strict",
            )
        cv2.imwrite(str(output_path), frame)
    finally:
        cap.release()
    return str(source_image or output_path)


def _keyframe_cache_key(
    source: VideoSource,
    local_time_sec: float,
    evidence_state: InteractionState | None,
    primary_object: str | None,
) -> tuple[str, str, float, str, str, str | None]:
    evidence_key = None
    if evidence_state is not None:
        evidence_key = "|".join(
            [
                str(id(evidence_state)),
                _label(evidence_state.source_view),
                str(evidence_state.frame_index),
                str(evidence_state.global_time),
            ]
        )
    return (
        str(Path(source.path)),
        _label(source.role or source.name),
        round(float(local_time_sec), 3),
        _label(primary_object),
        _micro_keyframe_box_mode(),
        evidence_key,
    )


def _copy_cached_keyframe(cached_path: str, output_path: Path) -> str | None:
    source_path = Path(cached_path)
    if not source_path.exists():
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if source_path.resolve() != output_path.resolve():
            shutil.copyfile(source_path, output_path)
    except Exception:
        return None
    return str(output_path)


def _write_keyframe_cached(
    source: VideoSource,
    local_time_sec: float,
    output_path: Path,
    *,
    dry_run: bool,
    evidence_state: InteractionState | None,
    primary_object: str | None,
    keyframe_cache: dict[tuple[str, str, float, str, str, str | None], str] | None,
) -> str:
    if keyframe_cache is None:
        return _write_keyframe(
            source,
            local_time_sec,
            output_path,
            dry_run=dry_run,
            evidence_state=evidence_state,
            primary_object=primary_object,
        )
    cache_key = _keyframe_cache_key(source, local_time_sec, evidence_state, primary_object)
    cached_path = keyframe_cache.get(cache_key)
    if cached_path:
        copied = _copy_cached_keyframe(cached_path, output_path)
        if copied is not None:
            return copied
    saved_path = _write_keyframe(
        source,
        local_time_sec,
        output_path,
        dry_run=dry_run,
        evidence_state=evidence_state,
        primary_object=primary_object,
    )
    if Path(saved_path).exists():
        keyframe_cache[cache_key] = saved_path
    return saved_path


def _write_per_view_keyframes(
    manifest: SessionManifest,
    states: list[InteractionState],
    role_states: dict[str, InteractionState],
    keyframe_roles: set[str],
    keyframe_views: set[str],
    micro_keyframe_dir: Path,
    *,
    dry_run: bool,
    primary_object: str | None,
    keyframe_cache: dict[tuple[str, str, float, str, str, str | None], str] | None = None,
) -> dict[str, dict[str, str]]:
    view_keyframes: dict[str, dict[str, str]] = {}
    for view in sorted(keyframe_views):
        if view == "first_person" and manifest.videos.first_person is None:
            continue
        if view not in {"third_person", "first_person"}:
            continue
        source = _source_for_view(manifest, view)
        for role in sorted(keyframe_roles):
            role_state = role_states.get(role)
            if role_state is None:
                continue
            target_state = _nearest_view_evidence_state(
                states,
                view=view,
                target_global_time=role_state.global_time,
                primary_object=primary_object,
            )
            try:
                local_time = global_time_to_local_sec(source, target_state.global_time if target_state else role_state.global_time)
                output_path = micro_keyframe_dir / view / f"{role}.jpg"
                saved = _write_keyframe_cached(
                    source,
                    local_time,
                    output_path,
                    dry_run=dry_run,
                    evidence_state=target_state,
                    primary_object=primary_object,
                    keyframe_cache=keyframe_cache,
                )
            except Exception:
                continue
            view_keyframes.setdefault(view, {})[role] = saved
    return view_keyframes


def _clip_has_visible_frame(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        import cv2
        import numpy as np
    except Exception:
        return True
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return False
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        probe_positions = [0]
        if frame_count > 2:
            probe_positions.extend([max(0, frame_count // 2), max(0, frame_count - 2)])
        for frame_index in probe_positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean = float(np.mean(gray))
            std = float(np.std(gray))
            if 8.0 <= mean <= 247.0 and std >= 2.0:
                return True
    finally:
        cap.release()
    return False


def _extract_micro_clip(
    source: VideoSource,
    local_start: float,
    local_end: float,
    clip_path: Path,
    *,
    dry_run: bool,
) -> str:
    if _micro_clip_stream_copy_enabled():
        try:
            extract_clip_ffmpeg(
                source.path,
                local_start,
                local_end,
                clip_path,
                dry_run=dry_run,
                copy_first=True,
            )
            if dry_run or _clip_has_visible_frame(clip_path):
                return str(clip_path)
            clip_path.unlink(missing_ok=True)
        except Exception:
            try:
                clip_path.unlink(missing_ok=True)
            except Exception:
                pass
    extract_clip_ffmpeg(
        source.path,
        local_start,
        local_end,
        clip_path,
        dry_run=dry_run,
        copy_first=False,
        video_codec=os.environ.get("KEY_ACTION_PREVIEW_CLIP_ENCODER", "auto"),
        video_bitrate=os.environ.get("KEY_ACTION_PREVIEW_CLIP_VIDEO_BITRATE", "2M"),
        crf=int(float(os.environ.get("KEY_ACTION_PREVIEW_CLIP_CRF", "30"))),
        preset=os.environ.get("KEY_ACTION_PREVIEW_CLIP_PRESET"),
        max_width=int(float(os.environ.get("KEY_ACTION_PREVIEW_CLIP_MAX_WIDTH", "960"))),
        include_audio=str(os.environ.get("KEY_ACTION_PREVIEW_CLIP_INCLUDE_AUDIO", "0")).strip().lower()
        not in {"", "0", "false", "no", "off"},
        faststart=True,
    )
    return str(clip_path)


def _view_clip(
    source: VideoSource,
    global_start: str,
    global_end: str,
    clip_path: Path | None,
    *,
    dry_run: bool,
) -> MicroSegmentView:
    local_start = global_time_to_local_sec(source, global_start)
    local_end = global_time_to_local_sec(source, global_end)
    saved_clip: str | None = None
    if clip_path is not None:
        saved_clip = _extract_micro_clip(
            source,
            local_start,
            local_end,
            clip_path,
            dry_run=dry_run,
        )
    return MicroSegmentView(
        clip_path=saved_clip,
        local_start_sec=float(local_start),
        local_end_sec=float(local_end),
    )


def _compact_detection(detection: dict[str, Any]) -> dict[str, Any]:
    compact = {
        "label": detection.get("label") or detection.get("class_name") or detection.get("name"),
        "confidence": detection.get("confidence", detection.get("conf", detection.get("score"))),
    }
    bbox = detection.get("bbox") or detection.get("box") or detection.get("xyxy")
    if isinstance(bbox, list) and len(bbox) >= 4:
        compact["bbox"] = [round(float(value), 3) for value in bbox[:4]]
    return {key: value for key, value in compact.items() if value is not None}


def _compact_interaction(interaction: dict[str, Any]) -> dict[str, Any]:
    compact = {
        "hand_label": interaction.get("hand_label") or interaction.get("hand"),
        "object_label": interaction.get("object_label") or interaction.get("target_label") or interaction.get("object"),
        "score": interaction.get("score", interaction.get("confidence", interaction.get("interaction_score"))),
        "distance_px": interaction.get("distance_px"),
        "normalized_distance": interaction.get("normalized_distance") or interaction.get("distance_norm"),
        "iou": interaction.get("iou", interaction.get("bbox_overlap")),
    }
    for key in ("hand_bbox", "object_bbox"):
        bbox = interaction.get(key)
        if isinstance(bbox, list) and len(bbox) >= 4:
            compact[key] = [round(float(value), 3) for value in bbox[:4]]
    return {key: value for key, value in compact.items() if value is not None and value != ""}


def _prioritized_compact_detections(
    detections: list[dict[str, Any]],
    interactions: list[dict[str, Any]],
    primary_object: str | None,
    *,
    max_items: int = 16,
) -> list[dict[str, Any]]:
    priority_labels = {_label(primary_object)}
    for interaction in interactions:
        priority_labels.add(_label(interaction.get("hand_label")))
        priority_labels.add(_label(interaction.get("object_label")))
    priority_labels.update(HAND_LABELS)
    priority_labels.discard("")

    compacted: list[tuple[int, dict[str, Any]]] = []
    for index, detection in enumerate(detections):
        compact = _compact_detection(detection)
        if compact:
            compacted.append((index, compact))

    def rank(item: tuple[int, dict[str, Any]]) -> tuple[int, float, int]:
        index, detection = item
        label = _label(detection.get("label"))
        try:
            confidence = float(detection.get("confidence") or 0.0)
        except Exception:
            confidence = 0.0
        return (0 if label in priority_labels else 1, -confidence, index)

    return [item for _index, item in sorted(compacted, key=rank)[:max_items]]


def _state_yolo_evidence(states: list[InteractionState], *, max_frames: int = 12) -> list[dict[str, Any]]:
    if not states:
        return []
    if len(states) <= max_frames:
        sampled = states
    else:
        step = max(1, len(states) // max_frames)
        sampled = states[::step][:max_frames]
        if states[-1] not in sampled:
            sampled[-1] = states[-1]
    evidence: list[dict[str, Any]] = []
    for state in sampled:
        row = state.raw_row or {}
        interactions = [_compact_interaction(item) for item in (row.get("hand_object_interactions") or []) if isinstance(item, dict)]
        detections = _prioritized_compact_detections(
            [item for item in (row.get("detections") or []) if isinstance(item, dict)],
            interactions,
            state.primary_object,
        )
        evidence.append(
            {
                "frame_index": state.frame_index,
                "view": state.source_view,
                "source_view": state.source_view,
                "time_sec": row.get("time_sec", row.get("local_time_sec", state.local_time_sec)),
                "local_time_sec": state.local_time_sec,
                "global_time": state.global_time,
                "frame_width": row.get("frame_width"),
                "frame_height": row.get("frame_height"),
                "primary_object": state.primary_object,
                "interaction_score": round(float(state.interaction_score), 6),
                "bbox_overlap": round(float(state.bbox_overlap or 0.0), 6),
                "hand_object_distance": state.hand_object_distance,
                "source": row.get("source") or row.get("source_file") or "yolo_frame_rows",
                "detections": detections,
                "hand_object_interactions": interactions[:6],
            }
        )
    return evidence


ACTION_OBJECT_BY_LABEL = {
    "balance": "balance",
    "scale": "balance",
    "paper": "paper",
    "weighing_paper": "paper",
    "spatula": "spatula",
    "pipette": "pipette",
    "pipette_tip": "pipette_tip",
    "reagent_bottle": "bottle",
    "sample_bottle": "bottle",
    "sample_bottle_blue": "bottle",
    "bottle": "bottle",
    "vial": "bottle",
    "beaker": "container",
    "container": "container",
    "tube": "container",
    "flask": "container",
}

NON_MATERIAL_LABELS = HAND_LABELS | {"", "lab_coat", "ppe_storage"}


def _ordered_unique_labels(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        label = _label(value)
        if label and label not in seen:
            seen.add(label)
            ordered.append(label)
    return ordered


def _ordered_unique_text(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        text = str(value or "").strip().lower()
        if text and text not in seen:
            seen.add(text)
            ordered.append(text)
    return ordered


def _state_interactions(state: InteractionState) -> list[dict[str, Any]]:
    return [item for item in (state.raw_row or {}).get("hand_object_interactions") or [] if isinstance(item, dict)]


def _state_interaction_score_for_object(state: InteractionState, object_label: str) -> float:
    target = _label(object_label)
    scores = [
        _candidate_score(item)
        for item in _state_interactions(state)
        if _candidate_object(item) == target
    ]
    if _label(state.primary_object) == target:
        scores.append(float(state.interaction_score or 0.0))
    return max(scores, default=0.0)


def _state_has_detection_for_object(state: InteractionState, object_label: str) -> bool:
    target = _label(object_label)
    return target in {_label(item) for item in [*state.detected_objects, *state.objects_near_hand]}


def _support_for_object(states: list[InteractionState], object_label: str) -> dict[str, Any]:
    target = _label(object_label)
    detection_frames = 0
    interaction_frames = 0
    frame_indices: list[int] = []
    max_score = 0.0
    views: set[str] = set()
    for state in states:
        detected = _state_has_detection_for_object(state, target)
        score = _state_interaction_score_for_object(state, target)
        if detected:
            detection_frames += 1
        if score > 0.0:
            interaction_frames += 1
            frame_indices.append(state.frame_index)
            max_score = max(max_score, score)
            views.add(state.source_view)
    return {
        "object": target,
        "frame_count": interaction_frames,
        "interaction_frame_count": interaction_frames,
        "detection_frame_count": detection_frames,
        "max_interaction_score": round(max_score, 6),
        "views": sorted(views),
        "frame_indices": frame_indices[:24],
    }


def _secondary_objects_for_window(states: list[InteractionState], primary_object: str) -> list[str]:
    primary = _label(primary_object)
    values: list[Any] = []
    for state in states:
        values.extend(state.objects_near_hand)
        values.extend(state.detected_objects)
        values.extend(_candidate_object(item) for item in _state_interactions(state))
    return [
        label
        for label in _ordered_unique_labels(values)
        if label not in NON_MATERIAL_LABELS and label != primary
    ]


def _canonical_action_object(object_label: str) -> str:
    label = _label(object_label)
    return ACTION_OBJECT_BY_LABEL.get(label, label)


def _canonical_hand_action(object_label: str) -> str:
    canonical = _canonical_action_object(object_label).replace("_", "-")
    return f"hand-{canonical}" if canonical else ""


def _secondary_actions_for_window(primary_object: str, secondary_objects: list[str]) -> list[str]:
    base_action = _canonical_hand_action(primary_object)
    actions = [_canonical_hand_action(item) for item in secondary_objects]
    secondary_action_objects = [_canonical_action_object(item).replace("_", "-") for item in secondary_objects]
    if base_action and secondary_action_objects:
        actions.append(f"{base_action}+{'+'.join(secondary_action_objects)}")
    return _ordered_unique_text(actions)


def _window_audit_from_states(
    states: list[InteractionState],
    *,
    primary_object: str,
    secondary_objects: list[str],
    warnings: list[str],
    vote_summary: dict[str, Any],
) -> dict[str, Any]:
    target_support = _support_for_object(states, primary_object)
    secondary_support = [_support_for_object(states, item) for item in secondary_objects]
    reasons = _ordered_unique_labels(list(warnings))
    if target_support["interaction_frame_count"] < PHYSICAL_EVIDENCE_MIN_FRAMES:
        reasons.append("target_object_support_below_physical_evidence_min_frames")
    for item in secondary_support:
        if item["detection_frame_count"] > item["interaction_frame_count"]:
            reasons.append(f"secondary_{item['object']}_detection_exceeds_interaction_support")
        if item["interaction_frame_count"] == 0:
            reasons.append(f"secondary_{item['object']}_has_no_hand_interaction_frames")
    if float(vote_summary.get("vote_margin") or 0.0) < 0.05 and len(states) > 1:
        reasons.append("low_primary_object_vote_margin")
    reasons = _ordered_unique_labels(reasons)
    if target_support["interaction_frame_count"] >= PHYSICAL_EVIDENCE_MIN_FRAMES and not reasons:
        uncertainty = "none"
    elif target_support["interaction_frame_count"] >= PHYSICAL_EVIDENCE_MIN_FRAMES:
        uncertainty = "low"
    else:
        uncertainty = "medium"
    return {
        "schema_version": "key_action_window_audit.v1",
        "source": "yolo_micro_evidence",
        "interaction_frame_count": len(states),
        "target_object_support": target_support,
        "secondary_object_support": secondary_support,
        "uncertainty": uncertainty,
        "uncertainty_reasons": reasons,
        "reasons": reasons,
    }


def _micro_asset_bindings(
    *,
    micro_id: str,
    parent_segment_id: str,
    global_start: str,
    global_end: str,
    first_view: MicroSegmentView | None,
    third_view: MicroSegmentView,
    keyframes: MicroSegmentKeyframes,
    confidence: float,
    evidence_source: str,
    view_keyframes: dict[str, dict[str, str]] | None = None,
    evidence_views: set[str] | None = None,
) -> list[dict[str, Any]]:
    frames = {
        "contact": keyframes.contact_frame,
        "peak": keyframes.peak_frame,
        "release": keyframes.release_frame,
    }
    bindings: list[dict[str, Any]] = []
    evidence_view_set = {_label(view) for view in (evidence_views or set()) if view}
    for view, ref in (("third_person", third_view), ("first_person", first_view)):
        if ref is None:
            continue
        view_frames = {
            key: value
            for key, value in (view_keyframes or {}).get(view, {}).items()
            if value
        }
        if not view_frames:
            view_frames = {key: value for key, value in frames.items() if value}
        keyframe_path = view_frames.get("peak") or view_frames.get("contact") or view_frames.get("release")
        bindings.append(
            {
                "level": "micro_segment",
                "micro_segment_id": micro_id,
                "parent_segment_id": parent_segment_id,
                "view": view,
                "global_start_time": global_start,
                "global_end_time": global_end,
                "local_start_sec": ref.local_start_sec,
                "local_end_sec": ref.local_end_sec,
                "clip_path": ref.clip_path,
                "keyframe_path": keyframe_path,
                "keyframe_paths": [path for path in view_frames.values() if path],
                "keyframes": view_frames,
                "keyframe_view": view,
                "evidence_role": "yolo_evidence" if view in evidence_view_set else "synchronized_context",
                "confidence": round(max(0.0, min(1.0, confidence)), 6),
                "evidence_source": evidence_source,
            }
        )
    return bindings


def _micro_from_run(
    manifest: SessionManifest,
    run: _InteractionRun,
    clips_dir: Path,
    keyframes_dir: Path,
    config: MicroSegmentConfig,
    utterances: list[TranscriptUtterance],
    *,
    dry_run: bool,
) -> MicroSegment | None:
    states = sorted(run.states, key=lambda state: parse_time(state.global_time))
    if not states:
        return None
    period = _sample_period(states, default=0.5)
    contact_duration = max(0.0, (parse_time(states[-1].global_time) - parse_time(states[0].global_time)).total_seconds() + period)
    overall_peak = max(states, key=lambda state: state.interaction_score)
    vote_summary = _run_primary_vote_summary(states, config)
    primary = str(vote_summary.get("primary_object") or overall_peak.primary_object)
    primary_states = [state for state in states if _label(state.primary_object) == _label(primary)]
    peak_candidates = [state for state in (primary_states or states) if _state_has_physical_evidence(state, primary)]
    peak = max(peak_candidates or primary_states or states, key=lambda state: _state_keyframe_rank(state, primary))
    class_threshold = _class_config(config, primary)
    if contact_duration < class_threshold["min_duration_sec"]:
        return None

    parent_start = parse_time(run.parent_segment.global_start_time)
    parent_end = parse_time(run.parent_segment.global_end_time)
    contact_start = parse_time(states[0].global_time)
    contact_end = parse_time(states[-1].global_time)
    micro_start = max(parent_start, contact_start)
    micro_end = min(parent_end, contact_end)
    micro_start = max(parent_start, micro_start - timedelta(seconds=config.micro_pre_buffer_sec))
    micro_end = min(parent_end, micro_end + timedelta(seconds=config.micro_post_buffer_sec + period))
    if micro_end <= micro_start:
        return None

    micro_id = f"{run.parent_segment.segment_id}_micro_{run.sequence:03d}"
    micro_clip_dir = clips_dir / "micro"
    micro_keyframe_dir = keyframes_dir / "micro" / micro_id
    global_start = micro_start.isoformat()
    global_end = micro_end.isoformat()
    clip_views = _micro_clip_views() if config.extract_micro_clips else set()
    if "__evidence__" in clip_views:
        evidence_views = [
            _label(state.source_view)
            for state in states
            if _label(state.source_view) in {"third_person", "first_person"}
        ]
        preferred_view = _label(peak.source_view)
        clip_views = {preferred_view} if preferred_view in {"third_person", "first_person"} else set()
        if not clip_views and evidence_views:
            clip_views = {evidence_views[0]}

    first_view = None
    if manifest.videos.first_person is not None:
        first_clip = micro_clip_dir / f"{micro_id}_first_person.mp4" if "first_person" in clip_views else None
        first_view = _view_clip(manifest.videos.first_person, global_start, global_end, first_clip, dry_run=dry_run)
    third_clip = micro_clip_dir / f"{micro_id}_third_person.mp4" if "third_person" in clip_views else None
    third_view = _view_clip(manifest.videos.third_person, global_start, global_end, third_clip, dry_run=dry_run)

    keyframes = MicroSegmentKeyframes()
    keyframe_cache: dict[tuple[str, str, float, str, str, str | None], str] | None = {} if _fast_locate_asset_mode() else None
    warnings: list[str] = []
    if any(state.raw_row.get("_secondary_interaction_candidate") for state in states):
        warnings.append("ambiguous_primary_object")
    if any(state.raw_row.get("_primary_hysteresis_kept") for state in states):
        warnings.append("primary_object_hysteresis_applied")
    if any(state.raw_row.get("_tracklet_vote_reassigned") for state in states) or _label(primary) != _label(overall_peak.primary_object):
        warnings.append("tracklet_primary_window_vote_applied")
    if float(vote_summary.get("vote_margin") or 0.0) < float(getattr(config, "tracklet_vote_margin", 0.0) or 0.0) and len(states) > 1:
        warnings.append("ambiguous_primary_object_vote")
    if config.extract_micro_keyframes:
        contact_state, peak_state, release_state, keyframe_warnings = _select_physical_keyframe_states(states, primary)
        warnings.extend(keyframe_warnings)
        keyframe_roles = _micro_keyframe_roles()
        role_states = {
            "contact": contact_state,
            "peak": peak_state,
            "release": release_state,
        }

        def _state_source_and_local(state: InteractionState) -> tuple[VideoSource, float]:
            source = _source_for_view(manifest, state.source_view)
            return source, global_time_to_local_sec(source, state.global_time)

        try:
            if "contact" in keyframe_roles:
                contact_source, contact_local = _state_source_and_local(contact_state)
                keyframes.contact_frame = _write_keyframe_cached(
                    contact_source,
                    contact_local,
                    micro_keyframe_dir / "contact.jpg",
                    dry_run=dry_run,
                    evidence_state=contact_state,
                    primary_object=primary,
                    keyframe_cache=keyframe_cache,
                )
            if "peak" in keyframe_roles:
                peak_source, peak_local = _state_source_and_local(peak_state)
                keyframes.peak_frame = _write_keyframe_cached(
                    peak_source,
                    peak_local,
                    micro_keyframe_dir / "peak.jpg",
                    dry_run=dry_run,
                    evidence_state=peak_state,
                    primary_object=primary,
                    keyframe_cache=keyframe_cache,
                )
            if "release" in keyframe_roles:
                release_source, release_local = _state_source_and_local(release_state)
                keyframes.release_frame = _write_keyframe_cached(
                    release_source,
                    release_local,
                    micro_keyframe_dir / "release.jpg",
                    dry_run=dry_run,
                    evidence_state=release_state,
                    primary_object=primary,
                    keyframe_cache=keyframe_cache,
                )
        except Exception:
            warnings.append("missing_keyframe")
    elif config.micro_peak_keyframe_required:
        warnings.append("missing_keyframe")
        keyframe_roles = set()
        role_states = {}
    else:
        keyframe_roles = set()
        role_states = {}

    keyframe_views = {
        view
        for view in clip_views
        if view in {"third_person", "first_person"}
    }
    evidence_views = {
        _label(state.source_view)
        for state in states
        if _label(state.source_view) in {"third_person", "first_person"}
    }
    if not keyframe_views:
        keyframe_views = set(evidence_views)
    view_keyframes = (
        _write_per_view_keyframes(
            manifest,
            states,
            role_states,
            keyframe_roles,
            keyframe_views,
            micro_keyframe_dir,
            dry_run=dry_run,
            primary_object=primary,
            keyframe_cache=keyframe_cache,
        )
        if config.extract_micro_keyframes and keyframe_roles
        else {}
    )

    detected_objects = sorted({label for state in states for label in state.detected_objects if label})
    secondary_objects = _secondary_objects_for_window(states, primary)
    secondary_actions = _secondary_actions_for_window(primary, secondary_objects)
    distances = [state.hand_object_distance for state in states if state.hand_object_distance is not None]
    dialogue_window_sec = 2.0
    dialogue = find_dialogue_for_segment(global_start, global_end, utterances, window_sec=dialogue_window_sec)
    if not dialogue:
        warnings.append("no_dialogue_context")
    duration_sec = (micro_end - micro_start).total_seconds()
    start_sec = _session_sec(manifest, global_start)
    end_sec = _session_sec(manifest, global_end)
    window_audit = _window_audit_from_states(
        states,
        primary_object=primary,
        secondary_objects=secondary_objects,
        warnings=warnings,
        vote_summary=dict(vote_summary),
    )
    interaction = MicroSegmentInteraction(
        interaction_type=f"hand_{primary}_contact",
        primary_object=primary,
        secondary_objects=secondary_objects,
        detected_objects=detected_objects or [primary, "hand"],
        avg_interaction_score=round(sum(state.interaction_score for state in states) / len(states), 6),
        max_interaction_score=round(peak.interaction_score, 6),
        contact_start_sec=round(_session_sec(manifest, states[0].global_time), 6),
        peak_interaction_sec=round(_session_sec(manifest, peak.global_time), 6),
        contact_end_sec=round(_session_sec(manifest, states[-1].global_time), 6),
        evidence_frame_indices=[state.frame_index for state in states],
        avg_hand_object_distance=round(sum(distances) / len(distances), 6) if distances else None,
        max_bbox_overlap=round(max((state.bbox_overlap for state in states), default=0.0), 6),
        primary_object_family=vote_summary.get("primary_object_family"),
        primary_object_arbitration=str(vote_summary.get("arbitration") or "tracklet_window_vote"),
        primary_object_vote_score=float(vote_summary.get("vote_score") or 0.0),
        primary_object_vote_margin=float(vote_summary.get("vote_margin") or 0.0),
        primary_object_vote_counts=dict(vote_summary.get("vote_counts") or {}),
        primary_object_vote_scores=dict(vote_summary.get("vote_scores") or {}),
        peak_primary_object=overall_peak.primary_object,
        secondary_actions=secondary_actions,
    )
    quality = _quality(states, duration_sec, warnings)
    yolo_evidence = _state_yolo_evidence(states)
    evidence_source = str((states[0].raw_row or {}).get("source") or (states[0].raw_row or {}).get("source_file") or "yolo_frame_rows")
    asset_bindings = _micro_asset_bindings(
        micro_id=micro_id,
        parent_segment_id=run.parent_segment.segment_id,
        global_start=global_start,
        global_end=global_end,
        first_view=first_view,
        third_view=third_view,
        keyframes=keyframes,
        view_keyframes=view_keyframes,
        evidence_views=evidence_views,
        confidence=float(interaction.max_interaction_score),
        evidence_source=evidence_source,
    )
    micro = MicroSegment(
        micro_segment_id=micro_id,
        parent_segment_id=run.parent_segment.segment_id,
        session_id=manifest.session_id,
        display_order=0,
        display_id="",
        start_sec=round(start_sec, 6),
        end_sec=round(end_sec, 6),
        duration_sec=round(duration_sec, 6),
        global_start_time=global_start,
        global_end_time=global_end,
        first_person=first_view,
        third_person=third_view,
        interaction=interaction,
        keyframes=keyframes,
        dialogue_context=[],
        text_description=MicroSegmentTextDescription(action_type="", summary="", index_text=""),
        index=MicroSegmentIndexInfo(index_level="micro_segment", embedding_id=f"emb_{micro_id}"),
        quality=quality,
        class_threshold=class_threshold,
        dialogue_context_available=bool(dialogue),
        dialogue_match_window_sec=dialogue_window_sec,
        window_audit=window_audit,
        asset_bindings=asset_bindings,
        yolo_evidence=yolo_evidence,
    )
    return apply_micro_evidence(build_micro_segment_description(micro, dialogue))


def _fallback_states_from_parent_events(manifest: SessionManifest, parent: KeyActionSegment) -> list[InteractionState]:
    states: list[InteractionState] = []
    for event in parent.interaction_events:
        object_label = _label(getattr(event, "object_label", None) if not isinstance(event, dict) else event.get("object_label"))
        if not object_label:
            continue
        global_time = getattr(event, "global_time", None) if not isinstance(event, dict) else event.get("global_time")
        local_time = getattr(event, "local_time_sec", 0.0) if not isinstance(event, dict) else event.get("local_time_sec", 0.0)
        view = getattr(event, "view", "first_person") if not isinstance(event, dict) else event.get("view", "first_person")
        if not global_time:
            source = _source_for_view(manifest, _label(view))
            global_time = local_sec_to_global_time(source, float(local_time or 0.0)).isoformat()
        confidence = getattr(event, "confidence", 0.6) if not isinstance(event, dict) else event.get("confidence", 0.6)
        states.append(
            InteractionState(
                frame_index=int(float(local_time or 0.0) * 30.0),
                local_time_sec=float(local_time or 0.0),
                global_time=str(global_time),
                source_view=_row_view({"view": view}),
                hand_detected=True,
                objects_near_hand=[object_label],
                primary_object=object_label,
                interaction_type=f"hand_{object_label}_contact",
                interaction_score=float(confidence or 0.6),
                hand_object_distance=None,
                bbox_overlap=0.0,
                detected_objects=["hand", object_label],
                raw_row={},
            )
        )
    return states


def _build_micro_for_run(
    manifest: SessionManifest,
    run: _InteractionRun,
    clips_root: Path,
    keyframes_root: Path,
    config: MicroSegmentConfig,
    utterances: list[TranscriptUtterance],
    *,
    dry_run: bool,
    final_physical_evidence_required: bool,
) -> MicroSegment | None:
    micro = _micro_from_run(
        manifest,
        run,
        clips_root,
        keyframes_root,
        config,
        utterances,
        dry_run=dry_run,
    )
    if micro is None:
        return None
    if final_physical_evidence_required and not _micro_has_final_physical_evidence(micro):
        return None
    return micro


def _build_micro_runs_parallel(
    manifest: SessionManifest,
    runs: list[_InteractionRun],
    clips_root: Path,
    keyframes_root: Path,
    config: MicroSegmentConfig,
    utterances: list[TranscriptUtterance],
    *,
    dry_run: bool,
    final_physical_evidence_required: bool,
    worker_count: int,
) -> list[tuple[_InteractionRun, MicroSegment]]:
    if not runs:
        return []

    def _build(run: _InteractionRun) -> MicroSegment | None:
        return _build_micro_for_run(
            manifest,
            run,
            clips_root,
            keyframes_root,
            config,
            utterances,
            dry_run=dry_run,
            final_physical_evidence_required=final_physical_evidence_required,
        )

    max_workers = min(max(1, int(worker_count)), len(runs))
    if max_workers <= 1:
        micros = [_build(run) for run in runs]
    else:
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="micro-assets") as executor:
            micros = list(executor.map(_build, runs))
    return [(run, micro) for run, micro in zip(runs, micros) if micro is not None]


def _next_micro_sequence(parent_micros: list[MicroSegment]) -> int:
    max_sequence = 0
    for micro in parent_micros:
        marker = "_micro_"
        if marker not in micro.micro_segment_id:
            continue
        try:
            max_sequence = max(max_sequence, int(micro.micro_segment_id.rsplit(marker, 1)[1]))
        except (TypeError, ValueError):
            continue
    return max_sequence + 1


def _build_micro_segments_for_parent(
    manifest: SessionManifest,
    parent: KeyActionSegment,
    rows: list[dict[str, Any]],
    clips_root: Path,
    keyframes_root: Path,
    cfg: MicroSegmentConfig,
    utterances: list[TranscriptUtterance],
    *,
    dry_run: bool,
    max_micros_per_parent: int,
    micro_asset_workers: int,
) -> tuple[KeyActionSegment, list[MicroSegment], list[dict[str, Any]]]:
    parent_rows = [
        row
        for row in rows
        if isinstance(row, dict) and _row_in_parent(manifest, row, parent)
    ]
    states = compute_primary_interaction_timeline(manifest, parent_rows, cfg)
    runs = build_interaction_runs(parent, states, cfg)
    backcheck_runs = _sop_action_backcheck_runs(manifest, parent, parent_rows, cfg, runs)
    if backcheck_runs:
        runs = _renumber_runs([*runs, *backcheck_runs])
    if not runs:
        fallback_states = _fallback_states_from_parent_events(manifest, parent)
        runs = build_interaction_runs(parent, fallback_states, cfg)
    runs = _limit_interaction_runs(runs, cfg, max_count=max_micros_per_parent)
    parent_micros: list[MicroSegment] = []
    final_physical_evidence_required = _final_micro_physical_evidence_required(cfg)
    if micro_asset_workers <= 1:
        for run in runs:
            run.sequence = len(parent_micros) + 1
            micro = _build_micro_for_run(
                manifest,
                run,
                clips_root,
                keyframes_root,
                cfg,
                utterances,
                dry_run=dry_run,
                final_physical_evidence_required=final_physical_evidence_required,
            )
            if micro is None:
                continue
            parent_micros.append(micro)
    else:
        for sequence, run in enumerate(runs, start=1):
            run.sequence = sequence
        parent_micros.extend(
            micro
            for _run, micro in _build_micro_runs_parallel(
                manifest,
                runs,
                clips_root,
                keyframes_root,
                cfg,
                utterances,
                dry_run=dry_run,
                final_physical_evidence_required=final_physical_evidence_required,
                worker_count=micro_asset_workers,
            )
        )
    coverage_states = _coverage_interaction_states(manifest, parent_rows, cfg)
    remaining_coverage_slots = max(0, max_micros_per_parent - len(parent_micros)) if max_micros_per_parent > 0 else 0
    if coverage_states and (max_micros_per_parent <= 0 or remaining_coverage_slots > 0):
        coverage_cfg = _coverage_backfill_config(cfg)
        coverage_runs = [
            run
            for run in _coverage_backfill_runs(parent, coverage_states, cfg)
            if not _coverage_run_overlaps_existing_micro(run, parent_micros, cfg)
        ]
        coverage_limit = remaining_coverage_slots if max_micros_per_parent > 0 else COVERAGE_BACKFILL_MAX_RUNS_PER_PARENT
        coverage_runs = sorted(coverage_runs, key=lambda run: _coverage_run_rank(run, cfg), reverse=True)[
            : min(COVERAGE_BACKFILL_MAX_RUNS_PER_PARENT, coverage_limit)
        ]
        coverage_runs.sort(key=lambda run: parse_time(run.states[0].global_time) if run.states else datetime.max)
        next_sequence = _next_micro_sequence(parent_micros)
        if micro_asset_workers <= 1:
            for run in coverage_runs:
                run.sequence = next_sequence
                next_sequence += 1
                micro = _micro_from_run(
                    manifest,
                    run,
                    clips_root,
                    keyframes_root,
                    coverage_cfg,
                    utterances,
                    dry_run=dry_run,
                )
                if micro is None:
                    continue
                parent_micros.append(_mark_coverage_backfill_micro(micro, run))
        else:
            for offset, run in enumerate(coverage_runs):
                run.sequence = next_sequence + offset
            parent_micros.extend(
                _mark_coverage_backfill_micro(micro, run)
                for run, micro in _build_micro_runs_parallel(
                    manifest,
                    coverage_runs,
                    clips_root,
                    keyframes_root,
                    coverage_cfg,
                    utterances,
                    dry_run=dry_run,
                    final_physical_evidence_required=False,
                    worker_count=micro_asset_workers,
                )
            )
    parent_micros.sort(key=lambda item: parse_time(item.global_start_time))
    parent_refs: list[dict[str, Any]] = []
    for display_order, micro in enumerate(parent_micros, start=1):
        micro.display_order = display_order
        micro.display_id = f"micro_{display_order:03d}"
        parent_refs.append(
            {
                "micro_segment_id": micro.micro_segment_id,
                "display_order": micro.display_order,
                "display_id": micro.display_id,
                "primary_object": micro.interaction.primary_object,
                "secondary_objects": micro.interaction.secondary_objects,
                "secondary_actions": micro.interaction.secondary_actions,
                "primary_object_family": micro.interaction.primary_object_family,
                "primary_object_arbitration": micro.interaction.primary_object_arbitration,
                "interaction_type": micro.interaction.interaction_type,
                "global_start_time": micro.global_start_time,
                "global_end_time": micro.global_end_time,
                "duration_sec": micro.duration_sec,
                "max_interaction_score": micro.interaction.max_interaction_score,
                "confidence": micro.quality.confidence,
                "peak_keyframe": micro.keyframes.peak_frame,
                "first_person_clip": micro.first_person.clip_path if micro.first_person else None,
                "third_person_clip": micro.third_person.clip_path,
                "manual_corrected": micro.manual_corrected,
                "dialogue_context_available": micro.dialogue_context_available,
                "dialogue_match_window_sec": micro.dialogue_match_window_sec,
                "dialogue_keywords": micro.dialogue_keywords,
                "evidence_level": micro.evidence.get("evidence_level"),
                "evidence": micro.evidence,
                "window_audit": micro.window_audit,
                "asset_bindings": micro.asset_bindings,
                "yolo_evidence": micro.yolo_evidence,
                "class_threshold": micro.class_threshold,
            }
        )
    return parent, parent_micros, parent_refs


def generate_micro_segments(
    *,
    manifest: SessionManifest,
    key_segments: list[KeyActionSegment],
    yolo_frame_rows: list[dict[str, Any]] | None,
    utterances: list[TranscriptUtterance],
    clips_dir: str | Path,
    keyframes_dir: str | Path,
    config: MicroSegmentConfig | None = None,
    dry_run: bool = False,
) -> list[MicroSegment]:
    cfg = config or MicroSegmentConfig()
    clips_root = Path(clips_dir)
    keyframes_root = Path(keyframes_dir)
    micro_segments: list[MicroSegment] = []
    rows = yolo_frame_rows or []
    max_micros_per_parent = _max_micro_segments_per_parent()
    micro_asset_workers = _micro_asset_worker_count()
    parent_worker_count = min(_micro_parent_worker_count(), len(key_segments))
    parent_results: list[tuple[KeyActionSegment, list[MicroSegment], list[dict[str, Any]]]]
    if parent_worker_count > 1 and micro_asset_workers > 1:
        per_parent_asset_workers = max(1, micro_asset_workers // parent_worker_count)
        with ThreadPoolExecutor(max_workers=parent_worker_count, thread_name_prefix="micro-parent-assets") as executor:
            parent_results = list(
                executor.map(
                    lambda parent: _build_micro_segments_for_parent(
                        manifest,
                        parent,
                        rows,
                        clips_root,
                        keyframes_root,
                        cfg,
                        utterances,
                        dry_run=dry_run,
                        max_micros_per_parent=max_micros_per_parent,
                        micro_asset_workers=per_parent_asset_workers,
                    ),
                    key_segments,
                )
            )
    else:
        parent_results = [
            _build_micro_segments_for_parent(
                manifest,
                parent,
                rows,
                clips_root,
                keyframes_root,
                cfg,
                utterances,
                dry_run=dry_run,
                max_micros_per_parent=max_micros_per_parent,
                micro_asset_workers=micro_asset_workers,
            )
            for parent in key_segments
        ]
    for parent, parent_micros, parent_refs in parent_results:
        micro_segments.extend(parent_micros)
        parent.micro_segments = parent_refs
    micro_segments, dedup_log = deduplicate_micro_segments(micro_segments)
    for display_order, micro in enumerate(micro_segments, start=1):
        micro.display_order = display_order
        micro.display_id = f"micro_{display_order:03d}"
    return micro_segments, dedup_log


def micro_segment_to_vector_metadata(micro: MicroSegment) -> dict[str, Any]:
    keyframes = [
        value
        for value in [
            micro.keyframes.contact_frame,
            micro.keyframes.peak_frame,
            micro.keyframes.release_frame,
        ]
        if value
    ]
    metadata = {
        "index_level": "micro_segment",
        "embedding_id": micro.index.embedding_id,
        "segment_id": micro.parent_segment_id,
        "micro_segment_id": micro.micro_segment_id,
        "parent_segment_id": micro.parent_segment_id,
        "display_order": micro.display_order,
        "display_id": micro.display_id,
        "session_id": micro.session_id,
        "index_text": micro.text_description.index_text,
        "global_start_time": micro.global_start_time,
        "global_end_time": micro.global_end_time,
        "run_manifest_id": getattr(micro, "run_manifest_id", None),
        "source_view": _row_view(micro.first_person.__dict__ if micro.first_person else micro.third_person.__dict__) if hasattr(micro, "_source_view") is False else getattr(micro, "_source_view", None),
        "third_person_clip": micro.third_person.clip_path,
        "first_person_clip": micro.first_person.clip_path if micro.first_person else None,
        "related_dialogue": [item.get("text", "") for item in micro.dialogue_context],
        "action_type": micro.text_description.action_type,
        "interaction_object": micro.interaction.primary_object,
        "action_trigger_point": micro.interaction.contact_start_sec,
        "interaction_type": micro.interaction.interaction_type,
        "primary_object": micro.interaction.primary_object,
        "secondary_objects": micro.interaction.secondary_objects,
        "secondary_actions": micro.interaction.secondary_actions,
        "primary_object_family": micro.interaction.primary_object_family,
        "primary_object_arbitration": micro.interaction.primary_object_arbitration,
        "primary_object_vote_score": micro.interaction.primary_object_vote_score,
        "primary_object_vote_margin": micro.interaction.primary_object_vote_margin,
        "primary_object_vote_counts": micro.interaction.primary_object_vote_counts,
        "primary_object_vote_scores": micro.interaction.primary_object_vote_scores,
        "detected_objects": micro.interaction.detected_objects,
        "keyframes": keyframes,
        "yolo_evidence": micro.yolo_evidence,
        "asset_bindings": micro.asset_bindings,
        "quality": micro.quality.confidence,
        "class_threshold": micro.class_threshold,
        "dialogue_context_available": micro.dialogue_context_available,
        "dialogue_match_window_sec": micro.dialogue_match_window_sec,
        "dialogue_keywords": micro.dialogue_keywords,
        "evidence": micro.evidence,
        "window_audit": micro.window_audit,
        "evidence_level": micro.evidence.get("evidence_level"),
        "evidence_reasons": micro.evidence.get("evidence_reasons", []),
        "limitations": micro.evidence.get("limitations", []),
        "evidence_link": micro.evidence.get("evidence_link"),
        "retrieval_boost_factors": micro.interaction_keyframes if isinstance(micro.interaction_keyframes, dict) else {},
        "manual_corrected": micro.manual_corrected,
        "manual_correction_note": micro.manual_correction_note,
        "interaction": {
            "avg_interaction_score": micro.interaction.avg_interaction_score,
            "max_interaction_score": micro.interaction.max_interaction_score,
            "evidence_frame_indices": micro.interaction.evidence_frame_indices,
            "primary_object_family": micro.interaction.primary_object_family,
            "primary_object_arbitration": micro.interaction.primary_object_arbitration,
            "primary_object_vote_score": micro.interaction.primary_object_vote_score,
            "primary_object_vote_margin": micro.interaction.primary_object_vote_margin,
            "primary_object_vote_counts": micro.interaction.primary_object_vote_counts,
            "primary_object_vote_scores": micro.interaction.primary_object_vote_scores,
            "peak_primary_object": micro.interaction.peak_primary_object,
            "secondary_objects": micro.interaction.secondary_objects,
            "secondary_actions": micro.interaction.secondary_actions,
        },
    }
    return attach_evidence(metadata)


def micro_row_to_vector_metadata(row: dict[str, Any]) -> dict[str, Any]:
    interaction = row.get("interaction") if isinstance(row.get("interaction"), dict) else {}
    text_description = row.get("text_description") if isinstance(row.get("text_description"), dict) else {}
    index_info = row.get("index") if isinstance(row.get("index"), dict) else {}
    keyframes = row.get("keyframes") if isinstance(row.get("keyframes"), dict) else {}
    first_person = row.get("first_person") if isinstance(row.get("first_person"), dict) else {}
    third_person = row.get("third_person") if isinstance(row.get("third_person"), dict) else {}
    quality = row.get("quality") if isinstance(row.get("quality"), dict) else {}
    evidence = row.get("evidence") if isinstance(row.get("evidence"), dict) else {}
    keyframe_paths = [value for value in keyframes.values() if value]
    metadata = {
        "index_level": "micro_segment",
        "embedding_id": index_info.get("embedding_id") or f"emb_{row.get('micro_segment_id')}",
        "segment_id": row.get("parent_segment_id"),
        "micro_segment_id": row.get("micro_segment_id"),
        "parent_segment_id": row.get("parent_segment_id"),
        "display_order": row.get("display_order"),
        "display_id": row.get("display_id"),
        "session_id": row.get("session_id"),
        "index_text": text_description.get("index_text", ""),
        "global_start_time": row.get("global_start_time"),
        "global_end_time": row.get("global_end_time"),
        "run_manifest_id": row.get("run_manifest_id"),
        "source_view": row.get("source_view"),
        "interaction_object": interaction.get("primary_object"),
        "action_trigger_point": row.get("action_trigger_point") or interaction.get("contact_start_sec"),
        "third_person_clip": third_person.get("clip_path"),
        "first_person_clip": first_person.get("clip_path"),
        "related_dialogue": [item.get("text", "") for item in row.get("dialogue_context", []) if isinstance(item, dict)],
        "action_type": text_description.get("action_type"),
        "interaction_type": interaction.get("interaction_type"),
        "primary_object": interaction.get("primary_object"),
        "secondary_objects": interaction.get("secondary_objects", row.get("secondary_objects", [])),
        "secondary_actions": interaction.get("secondary_actions", row.get("secondary_actions", [])),
        "primary_object_family": interaction.get("primary_object_family") or row.get("primary_object_family"),
        "primary_object_arbitration": interaction.get("primary_object_arbitration") or row.get("primary_object_arbitration"),
        "primary_object_vote_score": interaction.get("primary_object_vote_score"),
        "primary_object_vote_margin": interaction.get("primary_object_vote_margin"),
        "primary_object_vote_counts": interaction.get("primary_object_vote_counts", {}),
        "primary_object_vote_scores": interaction.get("primary_object_vote_scores", {}),
        "detected_objects": interaction.get("detected_objects", []),
        "keyframes": keyframe_paths,
        "yolo_evidence": row.get("yolo_evidence", []),
        "asset_bindings": row.get("asset_bindings", []),
        "quality": quality.get("confidence") if quality else row.get("quality"),
        "quality_warnings": quality.get("warnings", []) if quality else [],
        "class_threshold": row.get("class_threshold", {}),
        "dialogue_context_available": bool(row.get("dialogue_context_available") or row.get("dialogue_context")),
        "dialogue_match_window_sec": row.get("dialogue_match_window_sec"),
        "dialogue_keywords": row.get("dialogue_keywords", []),
        "evidence": evidence,
        "window_audit": row.get("window_audit", {}),
        "coverage_signal_grade": evidence.get("coverage_signal_grade"),
        "coverage_backfill": evidence.get("coverage_backfill", False),
        "process_evidence_role": evidence.get("process_evidence_role"),
        "process_eligible": evidence.get("process_eligible"),
        "strong_process_evidence": evidence.get("strong_process_evidence"),
        "retrieval_priority": evidence.get("retrieval_priority"),
        "retrieval_priority_bucket": evidence.get("retrieval_priority_bucket"),
        "evidence_level": row.get("evidence_level") or evidence.get("evidence_level"),
        "evidence_reasons": row.get("evidence_reasons") or evidence.get("evidence_reasons", []),
        "limitations": row.get("limitations") or evidence.get("limitations", []),
        "evidence_link": row.get("evidence_link") or evidence.get("evidence_link"),
        "retrieval_boost_factors": row.get("retrieval_boost_factors", {}),
        "manual_corrected": row.get("manual_corrected", False),
        "manual_correction_note": row.get("manual_correction_note"),
        "merged_from_micro_segment_ids": row.get("merged_from_micro_segment_ids", []),
        "merge_reason": row.get("merge_reason"),
        "interaction": {
            "avg_interaction_score": interaction.get("avg_interaction_score"),
            "max_interaction_score": interaction.get("max_interaction_score"),
            "evidence_frame_indices": interaction.get("evidence_frame_indices", []),
            "primary_object_family": interaction.get("primary_object_family"),
            "primary_object_arbitration": interaction.get("primary_object_arbitration"),
            "primary_object_vote_score": interaction.get("primary_object_vote_score"),
            "primary_object_vote_margin": interaction.get("primary_object_vote_margin"),
            "primary_object_vote_counts": interaction.get("primary_object_vote_counts", {}),
            "primary_object_vote_scores": interaction.get("primary_object_vote_scores", {}),
            "peak_primary_object": interaction.get("peak_primary_object"),
            "secondary_objects": interaction.get("secondary_objects", row.get("secondary_objects", [])),
            "secondary_actions": interaction.get("secondary_actions", row.get("secondary_actions", [])),
        },
    }
    return attach_evidence(metadata)


def deduplicate_micro_segments(
    micros: list[MicroSegment],
    *,
    overlap_threshold: float = 0.80,
) -> tuple[list[MicroSegment], list[dict[str, Any]]]:
    """Remove micro-segments with high time overlap, keeping higher-scored ones.

    Uses sorted sliding window: only compares each micro against recent kept items
    whose end_time could still overlap, achieving O(n log n) instead of O(n²).
    """
    if len(micros) <= 1:
        return micros, []

    ordered = sorted(micros, key=lambda m: parse_time(m.global_start_time))
    kept: list[MicroSegment] = []
    kept_ends: list[float] = []
    dedup_log: list[dict[str, Any]] = []

    for micro in ordered:
        m_start = parse_time(micro.global_start_time)
        m_end = parse_time(micro.global_end_time)
        m_start_ts = m_start.timestamp()
        m_end_ts = m_end.timestamp()
        m_duration = max(0.001, m_end_ts - m_start_ts)
        m_score = float(micro.interaction.max_interaction_score) if micro.interaction else 0.0

        duplicate_of = None
        best_overlap = 0.0
        for i in range(len(kept) - 1, -1, -1):
            if kept_ends[i] < m_start_ts:
                break
            e_start_ts = parse_time(kept[i].global_start_time).timestamp()
            e_end_ts = kept_ends[i]
            e_duration = max(0.001, e_end_ts - e_start_ts)

            overlap_sec = max(0.0, min(m_end_ts, e_end_ts) - max(m_start_ts, e_start_ts))
            overlap_ratio = overlap_sec / min(m_duration, e_duration)

            if overlap_ratio >= overlap_threshold:
                e_score = float(kept[i].interaction.max_interaction_score) if kept[i].interaction else 0.0
                if m_score <= e_score:
                    duplicate_of = kept[i]
                    best_overlap = overlap_ratio
                    break

        if duplicate_of is not None:
            dedup_log.append({
                "removed_micro_id": micro.micro_segment_id,
                "kept_micro_id": duplicate_of.micro_segment_id,
                "overlap_ratio": round(best_overlap, 4),
                "removed_score": round(m_score, 4),
                "kept_score": round(float(duplicate_of.interaction.max_interaction_score) if duplicate_of.interaction else 0.0, 4),
            })
        else:
            kept.append(micro)
            kept_ends.append(m_end_ts)

    return kept, dedup_log


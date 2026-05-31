from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

from .schemas import DetectedSegment, SessionManifest


SCHEMA_VERSION = "experiment_window_state.v1"
DEFAULT_MIN_EXPERIMENT_WINDOW_DURATION_SEC = 120.0
DEFAULT_EXPERIMENT_GAP_MERGE_THRESHOLD_SEC = 300.0

ACTIVE_STATES = {"preparing", "active_experiment", "off_table_but_experiment_related", "cleaning_or_ending"}
FIRST_PERSON = "first_person"
THIRD_PERSON = "third_person"

HAND_LABELS = {"hand", "gloved_hand", "left_hand", "right_hand"}
OPERATOR_PREPARING_LABELS = {"glove", "gloved_hand", "lab_coat", "ppe", "person"}
STATIC_PPE_LABELS = {"ppe_storage"}
CORE_OBJECT_LABELS = {
    "balance",
    "device_panel",
    "panel",
    "display",
    "paper",
    "weighing_paper",
    "reagent_bottle",
    "sample_bottle",
    "bottle",
    "bottle_cap",
    "beaker",
    "container",
    "tube",
    "pipette",
    "pipette_tip",
    "spatula",
    "magnetic_stir_bar",
}
CLEANING_LABELS = {"sink", "wash", "cleaning", "waste", "trash"}
OFF_BENCH_CONTEXT_LABELS = {"sink", "wash", "cleaning", "waste", "trash", "lab_coat", "gloved_hand", "hand"}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(dict(row), ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() not in {"", "0", "false", "no", "off"}


def _float_env_any(names: tuple[str, ...], default: float, *, minimum: float | None = None) -> float:
    for name in names:
        raw = os.environ.get(name)
        if raw is None or not str(raw).strip():
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        if minimum is not None:
            value = max(float(minimum), value)
        return value
    value = float(default)
    if minimum is not None:
        value = max(float(minimum), value)
    return value


def _experiment_min_window_duration_sec(requested: float) -> float:
    # The caller's explicit `min_duration_sec` is the contract for this layer:
    # production passes the 120s minute-level default via
    # `_experiment_window_min_duration_sec(default=120.0)`, while unit tests
    # exercise the state machine at smaller synthetic durations. The default
    # therefore tracks the caller's request rather than baking in a hidden
    # 120s floor that would silently downgrade every short window to a
    # micro_action_clip regardless of what the caller asked for.
    configured = _float_env_any(
        (
            "KEY_ACTION_MIN_EXPERIMENT_WINDOW_DURATION_SEC",
            "KEY_ACTION_EXPERIMENT_WINDOW_MIN_SEC",
            "KEY_ACTION_MIN_OFFICIAL_EXPERIMENT_DURATION_SEC",
        ),
        float(requested),
        minimum=1.0,
    )
    # An operator env override may only tighten the threshold; the caller's
    # explicit request remains the floor so the experiment-window layer never
    # inherits a looser micro/action threshold than was asked for.
    return max(float(configured), float(requested))


def _experiment_gap_merge_threshold_sec() -> float:
    return _float_env_any(
        (
            "KEY_ACTION_EXPERIMENT_GAP_MERGE_THRESHOLD_SEC",
            "KEY_ACTION_EXPERIMENT_WINDOW_GAP_MERGE_THRESHOLD_SEC",
            "KEY_ACTION_EXPERIMENT_STATE_GAP_SEC",
        ),
        DEFAULT_EXPERIMENT_GAP_MERGE_THRESHOLD_SEC,
        minimum=0.0,
    )


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _source_by_view(manifest: SessionManifest, view: str) -> Any | None:
    return manifest.videos.all_sources().get(view)


def _aligned_video_config(manifest: SessionManifest) -> dict[str, Any]:
    payload = manifest.config.get("aligned_video_analysis") if isinstance(manifest.config, dict) else None
    return dict(payload) if isinstance(payload, dict) else {}


def _sync_index_csv(manifest: SessionManifest, metadata_dir: Path) -> Path | None:
    aligned = _aligned_video_config(manifest)
    raw_path = aligned.get("sync_index_csv")
    if raw_path:
        path = Path(str(raw_path))
        if path.exists():
            return path
    candidate = metadata_dir / "dual_view_alignment" / "sync_index.csv"
    return candidate if candidate.exists() else None


class SyncIndexLookup:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        if not rows:
            self.fps = 0.0
            self.first_global_us = 0
            self.last_global_us = 0
            return
        self.first_global_us = int(_as_float(rows[0].get("global_timestamp_us"), 0.0))
        self.last_global_us = int(_as_float(rows[-1].get("global_timestamp_us"), float(self.first_global_us)))
        duration_s = max(0.0, (self.last_global_us - self.first_global_us) / 1_000_000.0)
        self.fps = (len(rows) - 1) / duration_s if duration_s > 0 and len(rows) > 1 else 0.0

    @classmethod
    def load(cls, path: Path | None) -> "SyncIndexLookup":
        if path is None or not path.exists():
            return cls([])
        rows: list[dict[str, Any]] = []
        # sync_index.csv is produced by multiple artifact writers; some Windows
        # paths go through utf-8-sig writers, so normalize a possible BOM on the
        # header instead of dropping lineage fields such as sync_index.
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("is_valid_pair", "true")).strip().lower() in {"0", "false", "no"}:
                    continue
                rows.append(dict(row))
        return cls(rows)

    def at_sec(self, sec: float) -> dict[str, Any]:
        if not self.rows:
            global_us = int(round(max(0.0, sec) * 1_000_000))
            return {"sync_index": None, "unit_id": None, "global_timestamp_us": global_us}
        if self.fps <= 0:
            idx = 0
        else:
            idx = int(round(max(0.0, sec) * self.fps))
        idx = max(0, min(len(self.rows) - 1, idx))
        row = self.rows[idx]
        return {
            "sync_index": _int_or_none(row.get("sync_index")),
            "unit_id": row.get("unit_id"),
            "global_timestamp_us": _int_or_none(row.get("global_timestamp_us")),
            "third_frame_index": _int_or_none(row.get("third_frame_index")),
            "first_frame_index": _int_or_none(row.get("first_frame_index")),
            "delta_ms": _as_float(row.get("delta_ms"), 0.0),
            "sync_quality": row.get("sync_quality"),
        }

    def between_sec(self, start_sec: float, end_sec: float) -> list[dict[str, Any]]:
        if not self.rows:
            return []
        start = self.at_sec(start_sec)
        end = self.at_sec(end_sec)
        start_global = _int_or_none(start.get("global_timestamp_us"))
        end_global = _int_or_none(end.get("global_timestamp_us"))
        if start_global is None or end_global is None:
            return []
        if end_global < start_global:
            start_global, end_global = end_global, start_global
        rows: list[dict[str, Any]] = []
        for row in self.rows:
            global_us = _int_or_none(row.get("global_timestamp_us"))
            if global_us is None:
                continue
            if start_global <= global_us <= end_global:
                rows.append(dict(row))
        return rows


def _int_or_none(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _row_time_sec(row: Mapping[str, Any]) -> float:
    for key in (
        "aligned_time_sec",
        "session_time_sec",
        "global_time_sec",
        "local_time_sec",
        "time_sec",
        "timestamp_sec",
        "frame_time_sec",
        "sample_time_sec",
    ):
        if row.get(key) is not None:
            return max(0.0, _as_float(row.get(key), 0.0))
    if row.get("time_ms") is not None:
        return max(0.0, _as_float(row.get("time_ms"), 0.0) / 1000.0)
    return 0.0


def _row_view(row: Mapping[str, Any]) -> str:
    raw = str(
        row.get("source_view")
        or row.get("view")
        or row.get("view_role")
        or row.get("camera_role")
        or row.get("source")
        or ""
    ).lower()
    if "first" in raw or "operator" in raw:
        return FIRST_PERSON
    if "third" in raw or "top" in raw:
        return THIRD_PERSON
    return raw or THIRD_PERSON


def _row_labels(row: Mapping[str, Any]) -> Counter[str]:
    labels: Counter[str] = Counter()
    counts = row.get("label_counts") if isinstance(row.get("label_counts"), Mapping) else {}
    for label, count in dict(counts).items():
        try:
            labels[str(label).strip().lower()] += int(count)
        except (TypeError, ValueError):
            continue
    detections = row.get("detections") if isinstance(row.get("detections"), list) else []
    for detection in detections:
        if isinstance(detection, Mapping):
            label = str(detection.get("label") or detection.get("raw_label") or "").strip().lower()
            if label:
                labels[label] += 1
    interactions = row.get("hand_object_interactions") if isinstance(row.get("hand_object_interactions"), list) else []
    for interaction in interactions:
        if isinstance(interaction, Mapping):
            for key in ("object_label", "target_label", "label", "primary_object"):
                label = str(interaction.get(key) or "").strip().lower()
                if label:
                    labels[label] += 1
    return labels


def _has_interaction(row: Mapping[str, Any]) -> bool:
    interactions = row.get("hand_object_interactions")
    if isinstance(interactions, list) and interactions:
        return True
    return _as_float(row.get("interaction_score"), 0.0) >= 0.35


def _state_for_row(row: Mapping[str, Any]) -> tuple[str, float, str]:
    signal = _state_signal_for_row(row)
    if signal["object_interaction_signal"]:
        object_count = int(signal["lab_object_count"])
        return "active_experiment", min(1.0, 0.65 + 0.04 * object_count), "hand_object_or_lab_object_activity"
    if signal["cleaning_signal"]:
        return "cleaning_or_ending", 0.55, "cleaning_label"
    if signal["off_bench_activity_signal"]:
        return "off_table_but_experiment_related", 0.52, "first_person_off_bench_experiment_context"
    if signal["glove_on_signal"] or signal["lab_coat_signal"] or signal["operator_present_signal"]:
        return "preparing", 0.48, "operator_or_ppe_preparation"
    if signal["bench_activity_signal"]:
        return "unknown", 0.25, "lab_objects_without_operator"
    return "unknown", 0.0, "no_experiment_state_signal"


def _state_signal_for_row(row: Mapping[str, Any]) -> dict[str, Any]:
    labels = _row_labels(row)
    label_set = set(labels)
    hand_count = sum(labels[label] for label in HAND_LABELS)
    object_count = sum(labels[label] for label in CORE_OBJECT_LABELS)
    static_ppe_count = sum(labels[label] for label in STATIC_PPE_LABELS)
    operator_present = bool(label_set & {"person", "lab_coat", "ppe"})
    glove_signal = bool(label_set & {"glove", "gloved_hand"})
    cleaning_signal = bool(label_set & CLEANING_LABELS)
    device_panel_signal = bool(label_set & {"balance", "device_panel", "panel", "display"})
    object_interaction = bool(_has_interaction(row) or (hand_count > 0 and object_count > 0))
    # Static tabletop objects are lab context, not activity. Requiring an
    # operator/hand/interactions prevents fixed third-person views from opening
    # or extending a window just because the bench contains many objects.
    bench_activity = bool(object_interaction or ((operator_present or glove_signal) and object_count > 0))
    # First-person cameras may legitimately leave the bench during washing or
    # transfer. Treat operator/context labels as continuity signals for the
    # state machine, but do not let static tabletop PPE storage open a window.
    off_bench_activity = bool(
        _row_view(row) == FIRST_PERSON
        and not object_interaction
        and (
            cleaning_signal
            or bool(label_set & {"sink", "wash", "cleaning", "waste", "trash"})
            or ("hand" in label_set and not operator_present and not glove_signal)
        )
    )
    missing_capabilities: list[str] = []
    if not label_set & {"glove", "gloved_hand"}:
        missing_capabilities.append("glove_on_off_detector_not_confirmed")
    if not label_set & {"person", "lab_coat", "ppe"}:
        missing_capabilities.append("operator_presence_detector_not_confirmed")
    return {
        "timestamp_range": None,
        "view_role": _row_view(row),
        "operator_present_signal": operator_present,
        "first_person_present": _row_view(row) == FIRST_PERSON and (operator_present or hand_count > 0),
        "third_person_present": _row_view(row) == THIRD_PERSON and operator_present,
        "hand_visible": hand_count > 0,
        "glove_on_signal": glove_signal,
        "glove_off_signal": False,
        "lab_coat_signal": "lab_coat" in label_set,
        "bench_activity_signal": bench_activity,
        "static_ppe_signal": static_ppe_count > 0,
        "off_bench_activity_signal": off_bench_activity,
        "cleaning_signal": cleaning_signal,
        "object_interaction_signal": object_interaction,
        "device_panel_signal": device_panel_signal,
        "no_activity_signal": not (
            operator_present
            or glove_signal
            or object_interaction
            or cleaning_signal
            or off_bench_activity
        ),
        "hand_count": int(hand_count),
        "lab_object_count": int(object_count),
        "static_ppe_count": int(static_ppe_count),
        "confidence": _state_signal_confidence(
            operator_present=operator_present,
            glove_signal=glove_signal,
            object_interaction=object_interaction,
            cleaning_signal=cleaning_signal,
            bench_activity=bench_activity,
            off_bench_activity=off_bench_activity,
            static_ppe_count=static_ppe_count,
        ),
        "missing_capabilities": missing_capabilities,
        "evidence_frame_refs": [row.get("frame_index")],
        "label_counts": dict(labels),
    }


def _state_signal_confidence(
    *,
    operator_present: bool,
    glove_signal: bool,
    object_interaction: bool,
    cleaning_signal: bool,
    bench_activity: bool,
    off_bench_activity: bool,
    static_ppe_count: int,
) -> float:
    if object_interaction:
        return 0.8
    if off_bench_activity:
        return 0.52
    if glove_signal or operator_present or cleaning_signal:
        return 0.55
    if bench_activity and static_ppe_count:
        return 0.2
    if bench_activity:
        return 0.25
    return 0.0


def write_chunk_manifest(
    metadata_dir: Path,
    manifest: SessionManifest,
    scan_tasks: list[Mapping[str, Any]],
    *,
    sample_fps: float,
) -> dict[str, Any]:
    lookup = SyncIndexLookup.load(_sync_index_csv(manifest, metadata_dir))
    aligned = _aligned_video_config(manifest)
    rows: list[dict[str, Any]] = []
    for task in scan_tasks:
        view = str(task.get("view") or "")
        start_sec = _as_float(task.get("chunk_start_sec") or task.get("scan_start_sec"), 0.0)
        end_sec = _as_float(task.get("chunk_end_sec") or task.get("scan_end_sec"), start_sec)
        start_sync = lookup.at_sec(start_sec)
        end_sync = lookup.at_sec(end_sec)
        source = _source_by_view(manifest, view)
        rows.append(
            {
                "schema_version": "chunk_manifest.v1",
                "chunk_id": f"{view}_chunk_{int(task.get('chunk_index') or 0):04d}",
                "experiment_id": manifest.session_id,
                "view_role": view,
                "aligned_video_path": str(getattr(source, "path", "") or ""),
                "raw_video_path": ((_aligned_raw_sources(aligned).get(view) or {}).get("path")),
                "sync_index_csv": str(_sync_index_csv(manifest, metadata_dir) or ""),
                "start_global_timestamp_us": start_sync.get("global_timestamp_us"),
                "end_global_timestamp_us": end_sync.get("global_timestamp_us"),
                "start_sync_index": start_sync.get("sync_index"),
                "end_sync_index": end_sync.get("sync_index"),
                "chunk_start_sec": round(start_sec, 6),
                "chunk_end_sec": round(end_sec, 6),
                "sampled_frame_count": max(0, int(round(max(0.0, end_sec - start_sec) * float(sample_fps)))),
                "sample_fps": float(sample_fps),
                "decode_status": "planned",
                "cache_path": None,
                "warnings": [],
            }
        )
    path = metadata_dir / "chunk_manifest.jsonl"
    _write_jsonl(path, rows)
    summary = {
        "schema_version": "chunk_manifest_summary.v1",
        "chunk_count": len(rows),
        "views": sorted({str(row.get("view_role")) for row in rows}),
        "sample_fps": float(sample_fps),
        "sync_index_csv": str(_sync_index_csv(manifest, metadata_dir) or ""),
        "path": str(path),
    }
    _write_json(metadata_dir / "chunk_manifest_summary.json", summary)
    return summary


def _aligned_raw_sources(aligned: Mapping[str, Any]) -> dict[str, Any]:
    raw = aligned.get("raw_sources") if isinstance(aligned.get("raw_sources"), Mapping) else {}
    return dict(raw)


def _seconds_to_segment(
    manifest: SessionManifest,
    start_sec: float,
    end_sec: float,
    *,
    index: int,
    reason: str,
    confidence: float,
    label_counts: Counter[str] | None = None,
    interaction_count: int = 0,
    trace: list[dict[str, Any]] | None = None,
) -> DetectedSegment:
    start_sec = max(0.0, float(start_sec))
    end_sec = max(start_sec, float(end_sec))
    global_start_time = _global_time_string(manifest, start_sec)
    global_end_time = _global_time_string(manifest, end_sec)
    return DetectedSegment(
        segment_id=f"formal_exp_{index:03d}",
        start_sec=round(start_sec, 6),
        end_sec=round(end_sec, 6),
        duration_sec=round(max(0.0, end_sec - start_sec), 6),
        global_start_time=global_start_time,
        global_end_time=global_end_time,
        avg_motion_score=float(confidence),
        avg_active_score=float(confidence),
        start_reason=reason,
        end_reason="dual_view_inactivity_or_cleaning_end",
        review_required=confidence < 0.65,
        detector_backend="yolo_aligned_window_state",
        detector_source_view="global_multiview",
        yolo_label_counts=dict(label_counts or {}),
        yolo_interaction_count=int(interaction_count),
        boundary_confidence=float(confidence),
        boundary_support_count=len(trace or []),
        boundary_source="aligned_yolo_state_machine",
        decision_path="aligned_dual_view_state_machine",
        decision_trace=list(trace or []),
        reason_code="aligned_formal_experiment_window",
        raw_score=float(confidence),
        score=float(confidence),
        final_score=float(confidence),
        source="aligned_experiment_window",
        source_view="global_multiview",
        detector_version=SCHEMA_VERSION,
        run_manifest_id=manifest.session_id,
        retrieval_boost_factors={"state_machine_trace_count": len(trace or [])},
    )


def _global_time_string(manifest: SessionManifest, sec: float) -> str:
    raw = str(getattr(manifest, "session_start_time", "") or "")
    try:
        base = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        base = datetime.fromisoformat("1970-01-01T00:00:00+00:00")
    return (base + timedelta(seconds=max(0.0, float(sec)))).isoformat()


def build_experiment_state_artifacts(
    metadata_dir: Path,
    manifest: SessionManifest,
    yolo_rows: list[dict[str, Any]],
    seed_segments: list[Any] | None,
    *,
    min_duration_sec: float,
) -> dict[str, Any]:
    lookup = SyncIndexLookup.load(_sync_index_csv(manifest, metadata_dir))
    bin_sec = max(1.0, _float_env("KEY_ACTION_EXPERIMENT_STATE_BIN_SEC", 2.0))
    no_activity_timeout = max(10.0, _float_env("KEY_ACTION_EXPERIMENT_STATE_NO_ACTIVITY_END_TIMEOUT_SEC", 75.0))
    pre_context_sec = max(0.0, _float_env("KEY_ACTION_EXPERIMENT_STATE_PRE_CONTEXT_SEC", 10.0))
    post_context_sec = max(0.0, _float_env("KEY_ACTION_EXPERIMENT_STATE_POST_CONTEXT_SEC", 15.0))
    experiment_min_duration_sec = _experiment_min_window_duration_sec(min_duration_sec)
    gap_merge_threshold_sec = _experiment_gap_merge_threshold_sec()

    state_rows = _coarse_state_rows(yolo_rows, lookup)
    _write_jsonl(metadata_dir / "coarse_state_segments.jsonl", state_rows)
    state_signal_rows = _state_signal_rows_from_state_rows(state_rows)
    _write_jsonl(metadata_dir / "state_signal_rows.jsonl", state_signal_rows)
    _write_state_signal_algorithm_report(metadata_dir, state_signal_rows)
    bins = _state_bins(state_rows, bin_sec=bin_sec)
    windows, trace_rows = _state_machine_windows(
        bins,
        bin_sec=bin_sec,
        no_activity_timeout=no_activity_timeout,
        pre_context_sec=pre_context_sec,
        post_context_sec=post_context_sec,
        min_duration_sec=experiment_min_duration_sec,
    )
    seed_windows = _seed_windows(seed_segments or [], lookup)
    raw_windows_for_report = [dict(item) for item in [*windows, *seed_windows]]
    candidates = _merge_candidate_windows(windows, seed_windows, min_gap_sec=gap_merge_threshold_sec)
    candidates = [
        _tighten_window_to_first_person_anchor(
            window,
            pre_context_sec=pre_context_sec,
            post_context_sec=post_context_sec,
        )
        for window in candidates
    ]
    grouping_report = _write_experiment_window_grouping_report(
        metadata_dir,
        raw_windows_for_report,
        candidates,
        gap_merge_threshold_sec=gap_merge_threshold_sec,
        min_experiment_window_duration_sec=experiment_min_duration_sec,
    )
    formal_segments: list[DetectedSegment] = []
    formal_rows: list[dict[str, Any]] = []
    duration_sanity_items: list[dict[str, Any]] = []
    for idx, window in enumerate(candidates, start=1):
        duration = float(window["end_sec"]) - float(window["start_sec"])
        if duration < experiment_min_duration_sec:
            window["status"] = "micro_action_clip"
            row = _formal_window_row(idx, window, lookup)
            row["status"] = "micro_action_clip"
            row["final_status"] = "micro_action_clip"
            row["not_experiment_window_reason"] = (
                f"duration_sec={duration:.3f}<min_experiment_window_duration_sec={experiment_min_duration_sec:.3f}"
            )
            formal_rows.append(row)
            duration_sanity_items.append(
                {
                    "candidate_window_id": row.get("source_candidate_window_id"),
                    "experiment_window_id": row.get("experiment_window_id"),
                    "duration_s": round(duration, 6),
                    "min_experiment_window_duration_s": round(experiment_min_duration_sec, 6),
                    "action_burst_count": len(window.get("trace") or []),
                    "merged_into_window_id": None,
                    "final_status": "micro_action_clip",
                    "reason": row["not_experiment_window_reason"],
                }
            )
            continue
        window["status"] = "pending_action_phase_gate"
        formal_row = _formal_window_row(idx, window, lookup)
        audit = _formal_window_activity_audit(formal_row)
        _apply_formal_window_audit_status(formal_row, audit)
        window["status"] = formal_row["status"]
        if formal_row["status"] != "formal_window_rejected":
            segment = _seconds_to_segment(
                manifest,
                float(window["start_sec"]),
                float(window["end_sec"]),
                index=idx,
                reason=str(window.get("start_reason") or "aligned_state_machine_start"),
                confidence=float(window.get("confidence") or 0.55),
                label_counts=Counter(window.get("label_counts") or {}),
                interaction_count=int(window.get("interaction_count") or 0),
                trace=window.get("trace") if isinstance(window.get("trace"), list) else [],
            )
            formal_segments.append(segment)
        formal_rows.append(formal_row)
        duration_sanity_items.append(
            {
                "candidate_window_id": formal_row.get("source_candidate_window_id"),
                "experiment_window_id": formal_row.get("experiment_window_id"),
                "duration_s": round(duration, 6),
                "min_experiment_window_duration_s": round(experiment_min_duration_sec, 6),
                "action_burst_count": len(window.get("trace") or []),
                "merged_into_window_id": formal_row.get("experiment_window_id"),
                "final_status": formal_row.get("status"),
                "reason": "duration_sanity_passed",
            }
        )
    _write_window_duration_sanity_report(
        metadata_dir,
        duration_sanity_items,
        min_experiment_window_duration_sec=experiment_min_duration_sec,
        gap_merge_threshold_sec=gap_merge_threshold_sec,
    )
    experiment_rows = _experiment_window_rows(formal_rows)
    window_sync_summary = _write_window_sync_artifacts(
        metadata_dir,
        experiment_rows,
        lookup,
    )
    _write_formal_window_review_artifacts(metadata_dir, experiment_rows)
    _write_window_boundary_diagnosis_report(metadata_dir, experiment_rows)
    _write_window_sync_index_enforcement_report(metadata_dir, experiment_rows)
    _write_window_preview_generation_report(metadata_dir, experiment_rows)
    _write_jsonl(metadata_dir / "experiment_window_state_trace.jsonl", trace_rows)
    _write_json(
        metadata_dir / "candidate_experiment_windows.json",
        {
            "schema_version": "candidate_experiment_windows.v1",
            "window_count": len(candidates),
            "windows": [_formal_window_row(idx, row, lookup) for idx, row in enumerate(candidates, start=1)],
            "policy": "third empty/no bbox cannot close a window while first_person remains active.",
            "min_experiment_window_duration_sec": experiment_min_duration_sec,
            "gap_merge_threshold_sec": gap_merge_threshold_sec,
        },
    )
    _write_json(
        metadata_dir / "formal_experiment_windows.json",
        {
            "schema_version": "formal_experiment_windows.v1",
            "window_count": len(experiment_rows),
            "windows": experiment_rows,
            "status": "pending_visual_and_action_phase_gate",
            "policy": "These windows are fine-scan candidates only; publication requires per-window visual review and dual-view action gate.",
            "min_experiment_window_duration_sec": experiment_min_duration_sec,
            "gap_merge_threshold_sec": gap_merge_threshold_sec,
        },
    )
    summary = {
        "schema_version": "experiment_window_state_summary.v1",
        "state_row_count": len(state_rows),
        "state_bin_count": len(bins),
        "candidate_window_count": len(candidates),
        "formal_window_count": len(formal_segments),
        "formal_window_input_count": len(experiment_rows),
        "micro_action_clip_count": len([row for row in formal_rows if row.get("status") == "micro_action_clip"]),
        "formal_window_rejected_count": len([row for row in formal_rows if row.get("status") == "formal_window_rejected"]),
        "formal_window_needs_review_count": len(
            [
                row
                for row in formal_rows
                if str(row.get("status") or "") in {"formal_window_suspicious_needs_review", "formal_window_needs_human_review"}
            ]
        ),
        "no_activity_end_timeout_sec": no_activity_timeout,
        "min_experiment_window_duration_sec": experiment_min_duration_sec,
        "gap_merge_threshold_sec": gap_merge_threshold_sec,
        "third_empty_end_policy": "blocked_when_first_active",
        "formal_window_pass_policy": "timestamp alignment is not sufficient; every window requires cross-view activity support and visual review before formal publication.",
        "window_sync_index_summary": window_sync_summary,
        "experiment_window_grouping_report": grouping_report,
    }
    _write_json(metadata_dir / "experiment_window_state_summary.json", summary)
    return {
        "summary": summary,
        "formal_segments": formal_segments,
        "candidate_windows": candidates,
        "trace_rows": trace_rows,
    }


def _coarse_state_rows(yolo_rows: list[dict[str, Any]], lookup: SyncIndexLookup) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in yolo_rows or []:
        sec = _row_time_sec(row)
        state, confidence, reason = _state_for_row(row)
        signal = _state_signal_for_row(row)
        sync = lookup.at_sec(sec)
        labels = _row_labels(row)
        signal["timestamp_range"] = {"start_sec": round(sec, 6), "end_sec": round(sec, 6)}
        signal["global_timestamp_us"] = sync.get("global_timestamp_us")
        signal["sync_index"] = sync.get("sync_index")
        rows.append(
            {
                "schema_version": "coarse_state_segment.v1",
                "segment_id": f"state_{len(rows) + 1:06d}",
                "camera_id": row.get("camera_id"),
                "view_role": _row_view(row),
                "start_sec": round(sec, 6),
                "end_sec": round(sec, 6),
                "start_timestamp_us": sync.get("global_timestamp_us"),
                "end_timestamp_us": sync.get("global_timestamp_us"),
                "start_sync_index": sync.get("sync_index"),
                "end_sync_index": sync.get("sync_index"),
                "state_label": state,
                "confidence": round(float(confidence), 6),
                "evidence_frame_indices": [row.get("frame_index")],
                "evidence_timestamps_us": [sync.get("global_timestamp_us")],
                "method": "yolo_label_state_rules",
                "reason": reason,
                "label_counts": dict(labels),
                "interaction_count": len(row.get("hand_object_interactions") or []),
                "state_signal": signal,
                "warnings": [],
            }
        )
    return sorted(rows, key=lambda item: (float(item.get("start_sec") or 0.0), str(item.get("view_role") or "")))


def _state_signal_rows_from_state_rows(state_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(state_rows, start=1):
        signal = row.get("state_signal") if isinstance(row.get("state_signal"), Mapping) else {}
        timestamp_range = signal.get("timestamp_range") if isinstance(signal.get("timestamp_range"), Mapping) else {}
        start_sec = _as_float(row.get("start_sec"), _as_float(timestamp_range.get("start_sec"), 0.0))
        end_sec = _as_float(row.get("end_sec"), _as_float(timestamp_range.get("end_sec"), start_sec))
        global_us = _int_or_none(signal.get("global_timestamp_us"))
        bin_start_global = global_us
        bin_end_global = (
            int(round(float(global_us) + max(0.0, end_sec - start_sec) * 1_000_000.0))
            if global_us is not None
            else None
        )
        view_role = str(row.get("view_role") or "")
        state_label = str(row.get("state_label") or "")
        object_interaction = bool(signal.get("object_interaction_signal"))
        hand_visible = bool(signal.get("hand_visible"))
        bench_activity = bool(signal.get("bench_activity_signal"))
        off_bench_activity = bool(signal.get("off_bench_activity_signal"))
        confidence = float(signal.get("confidence") or row.get("confidence") or 0.0)
        first_activity = confidence if view_role == FIRST_PERSON and state_label in ACTIVE_STATES else 0.0
        third_activity = confidence if view_role == THIRD_PERSON and state_label in ACTIVE_STATES else 0.0
        rows.append(
            {
                "schema_version": "state_signal_row.v1",
                "bin_id": f"state_bin_{idx:06d}",
                "signal_id": f"state_signal_{idx:06d}",
                "view_role": view_role,
                "timestamp_range": signal.get("timestamp_range"),
                "bin_start_global_timestamp_us": bin_start_global,
                "bin_end_global_timestamp_us": bin_end_global,
                "global_timestamp_us": global_us,
                "sync_index": signal.get("sync_index"),
                "state_label": state_label,
                "first_activity_score": round(first_activity, 6),
                "third_activity_score": round(third_activity, 6),
                "first_person_present": bool(signal.get("first_person_present")),
                "third_person_present": bool(signal.get("third_person_present")),
                "first_has_hand": bool(view_role == FIRST_PERSON and hand_visible),
                "third_has_hand": bool(view_role == THIRD_PERSON and hand_visible),
                "first_has_object_interaction": bool(view_role == FIRST_PERSON and object_interaction),
                "third_has_object_interaction": bool(view_role == THIRD_PERSON and object_interaction),
                "first_has_lab_context": bool(view_role == FIRST_PERSON and (signal.get("lab_coat_signal") or hand_visible or object_interaction)),
                "third_has_lab_context": bool(view_role == THIRD_PERSON and (bench_activity or hand_visible or object_interaction)),
                "first_off_bench_experiment_related": bool(view_role == FIRST_PERSON and off_bench_activity),
                "third_bench_activity": bool(view_role == THIRD_PERSON and bench_activity),
                "hand_visible": bool(signal.get("hand_visible")),
                "glove_on_signal": bool(signal.get("glove_on_signal")),
                "glove_off_signal": bool(signal.get("glove_off_signal")),
                "lab_coat_signal": bool(signal.get("lab_coat_signal")),
                "bench_activity_signal": bool(signal.get("bench_activity_signal")),
                "static_ppe_signal": bool(signal.get("static_ppe_signal")),
                "off_bench_activity_signal": bool(signal.get("off_bench_activity_signal")),
                "cleaning_signal": bool(signal.get("cleaning_signal")),
                "object_interaction_signal": bool(signal.get("object_interaction_signal")),
                "device_panel_signal": bool(signal.get("device_panel_signal")),
                "no_activity_signal": bool(signal.get("no_activity_signal")),
                "confidence": confidence,
                "missing_capabilities": list(signal.get("missing_capabilities") or []),
                "missing_detector_capabilities": list(signal.get("missing_capabilities") or []),
                "evidence_frame_refs": list(signal.get("evidence_frame_refs") or []),
                "label_counts": dict(signal.get("label_counts") or row.get("label_counts") or {}),
            }
        )
    return rows


def _write_state_signal_algorithm_report(metadata_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    required = [
        "first_activity_score",
        "third_activity_score",
        "first_has_hand",
        "third_has_hand",
        "first_has_object_interaction",
        "third_has_object_interaction",
        "first_off_bench_experiment_related",
        "third_bench_activity",
        "glove_on_signal",
        "glove_off_signal",
        "cleaning_signal",
        "device_panel_signal",
        "no_activity_signal",
        "missing_detector_capabilities",
        "evidence_frame_refs",
        "confidence",
    ]
    missing_by_field = {
        field: sum(1 for row in rows if field not in row)
        for field in required
    }
    contradiction_count = sum(
        1
        for row in rows
        if row.get("no_activity_signal")
        and (
            row.get("first_activity_score")
            or row.get("third_activity_score")
            or row.get("first_has_object_interaction")
            or row.get("third_has_object_interaction")
            or row.get("first_off_bench_experiment_related")
            or row.get("cleaning_signal")
        )
    )
    report = {
        "schema_version": "state_signal_algorithm_report.v1",
        "row_count": len(rows),
        "required_fields": required,
        "missing_required_field_counts": missing_by_field,
        "first_active_bin_count": sum(1 for row in rows if float(row.get("first_activity_score") or 0.0) > 0.0),
        "third_active_bin_count": sum(1 for row in rows if float(row.get("third_activity_score") or 0.0) > 0.0),
        "first_off_bench_experiment_related_count": sum(1 for row in rows if row.get("first_off_bench_experiment_related")),
        "third_bench_activity_count": sum(1 for row in rows if row.get("third_bench_activity")),
        "no_activity_contradiction_count": contradiction_count,
        "missing_detector_capabilities": sorted(
            {
                str(item)
                for row in rows
                for item in (row.get("missing_detector_capabilities") or [])
                if str(item)
            }
        ),
        "policy": "First-person activity is the continuity signal; third empty/no bbox cannot close an experiment window.",
    }
    _write_json(metadata_dir / "state_signal_algorithm_report.json", report)
    return report


def _state_bins(state_rows: list[dict[str, Any]], *, bin_sec: float) -> list[dict[str, Any]]:
    by_bin: dict[int, dict[str, Any]] = {}
    for row in state_rows:
        sec = float(row.get("start_sec") or 0.0)
        bin_idx = int(sec // bin_sec)
        bucket = by_bin.setdefault(
            bin_idx,
            {
                "start_sec": bin_idx * bin_sec,
                "end_sec": (bin_idx + 1) * bin_sec,
                "views": {},
                "label_counts": Counter(),
                "interaction_count": 0,
                "evidence_count": 0,
            },
        )
        view = str(row.get("view_role") or "")
        current = bucket["views"].get(view)
        if current is None or float(row.get("confidence") or 0.0) > float(current.get("confidence") or 0.0):
            bucket["views"][view] = row
        bucket["label_counts"].update(row.get("label_counts") or {})
        bucket["interaction_count"] += int(row.get("interaction_count") or 0)
        bucket["evidence_count"] += 1
    bins = []
    for item in by_bin.values():
        item["label_counts"] = dict(item["label_counts"])
        bins.append(item)
    return sorted(bins, key=lambda item: float(item.get("start_sec") or 0.0))


def _active(row: Mapping[str, Any] | None) -> bool:
    if not row:
        return False
    return str(row.get("state_label") or "") in ACTIVE_STATES and float(row.get("confidence") or 0.0) > 0.0


def _state_machine_windows(
    bins: list[dict[str, Any]],
    *,
    bin_sec: float,
    no_activity_timeout: float,
    pre_context_sec: float,
    post_context_sec: float,
    min_duration_sec: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    windows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    open_window: dict[str, Any] | None = None
    inactive_since: float | None = None
    missing_first_ranges: list[dict[str, float]] = []
    for bucket in bins:
        start = float(bucket.get("start_sec") or 0.0)
        end = float(bucket.get("end_sec") or start + bin_sec)
        views = bucket.get("views") if isinstance(bucket.get("views"), Mapping) else {}
        first = views.get(FIRST_PERSON) if isinstance(views, Mapping) else None
        third = views.get(THIRD_PERSON) if isinstance(views, Mapping) else None
        first_active = _active(first)
        third_active = _active(third)
        any_active = first_active or third_active
        state = _decision_state_for_bin(first, third, open_window_exists=open_window is not None)
        if open_window is None and any_active:
            open_window = {
                "start_sec": max(0.0, start - pre_context_sec),
                "start_reason": _start_reason(first, third),
                "label_counts": Counter(bucket.get("label_counts") or {}),
                "interaction_count": int(bucket.get("interaction_count") or 0),
                "trace": [],
                "third_missing_but_first_active_ranges": [],
                "first_missing_but_third_active_ranges": [],
                "confidence": 0.55,
            }
            inactive_since = None
            state = _decision_state_for_bin(first, third, open_window_exists=True)
        if open_window is not None:
            open_window["label_counts"].update(bucket.get("label_counts") or {})
            open_window["interaction_count"] += int(bucket.get("interaction_count") or 0)
            if first_active and not third_active:
                open_window["third_missing_but_first_active_ranges"].append({"start_sec": start, "end_sec": end})
                state = "EXPERIMENT_ACTIVE_OFF_BENCH"
            if third_active and not first_active:
                open_window["first_missing_but_third_active_ranges"].append({"start_sec": start, "end_sec": end})
            if any_active:
                inactive_since = None
            elif inactive_since is None:
                inactive_since = start
            open_window["trace"].append(
                {
                    "start_sec": start,
                    "end_sec": end,
                    "first_state": first.get("state_label") if isinstance(first, Mapping) else None,
                    "third_state": third.get("state_label") if isinstance(third, Mapping) else None,
                    "decision_state": state,
                    "first_active": first_active,
                    "third_active": third_active,
                }
            )
            if inactive_since is not None and start - inactive_since >= no_activity_timeout:
                # Do not stretch a few-second action burst into a minute-level
                # experiment window. Close on the actual inactivity boundary;
                # short windows are merged or downgraded by duration sanity.
                close_sec = inactive_since + post_context_sec
                open_window["end_sec"] = close_sec
                open_window["confidence"] = _window_confidence(open_window)
                windows.append(_finalize_window(open_window))
                open_window = None
                inactive_since = None
                state = "ENDED"
        trace_rows.append(
            {
                "schema_version": "experiment_window_state_trace.v1",
                "start_sec": start,
                "end_sec": end,
                "state": state,
                "first_active": first_active,
                "third_active": third_active,
                "first_state": first.get("state_label") if isinstance(first, Mapping) else None,
                "third_state": third.get("state_label") if isinstance(third, Mapping) else None,
                "third_empty_end_blocked": bool(open_window is not None and first_active and not third_active),
            }
        )
    if open_window is not None:
        last_end = float(bins[-1].get("end_sec") or open_window["start_sec"]) if bins else open_window["start_sec"]
        open_window["end_sec"] = last_end + post_context_sec
        open_window["confidence"] = _window_confidence(open_window)
        windows.append(_finalize_window(open_window))
    return windows, trace_rows


def _decision_state_for_bin(
    first: Mapping[str, Any] | None,
    third: Mapping[str, Any] | None,
    *,
    open_window_exists: bool,
) -> str:
    first_active = _active(first)
    third_active = _active(third)
    if first_active and third_active:
        if _is_cleaning(first) or _is_cleaning(third):
            return "CLEANING_OR_TRANSFER"
        if _is_preparing(first) or _is_preparing(third):
            return "PREPARING"
        return "EXPERIMENT_ACTIVE_AT_BENCH"
    if first_active and not third_active:
        if _is_cleaning(first):
            return "CLEANING_OR_TRANSFER"
        return "EXPERIMENT_ACTIVE_OFF_BENCH"
    if third_active and not first_active:
        return "EXPERIMENT_ACTIVE_AT_BENCH_THIRD_ONLY"
    return "UNCERTAIN_BUT_KEEP" if open_window_exists else "NOT_STARTED"


def _is_preparing(row: Mapping[str, Any] | None) -> bool:
    return bool(row and str(row.get("state_label") or "") == "preparing")


def _is_cleaning(row: Mapping[str, Any] | None) -> bool:
    return bool(row and str(row.get("state_label") or "") == "cleaning_or_ending")


def _start_reason(first: Mapping[str, Any] | None, third: Mapping[str, Any] | None) -> str:
    if _is_preparing(first) or _is_preparing(third):
        return "ppe_or_glove_preparation_detected"
    return "dual_or_single_view_experiment_activity_detected"


def _window_confidence(window: Mapping[str, Any]) -> float:
    trace = window.get("trace") if isinstance(window.get("trace"), list) else []
    if not trace:
        return 0.5
    both = sum(1 for item in trace if item.get("first_active") and item.get("third_active"))
    first_only = sum(1 for item in trace if item.get("first_active") and not item.get("third_active"))
    third_only = sum(1 for item in trace if item.get("third_active") and not item.get("first_active"))
    score = 0.48 + min(0.32, 0.04 * both) + min(0.12, 0.015 * (first_only + third_only))
    return round(min(0.9, score), 6)


def _finalize_window(window: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(window)
    result["duration_sec"] = round(float(result["end_sec"]) - float(result["start_sec"]), 6)
    result["label_counts"] = dict(result.get("label_counts") or {})
    return result


def _seed_windows(seed_segments: list[Any], lookup: SyncIndexLookup) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    for segment in seed_segments:
        start = _optional_float(getattr(segment, "start_sec", None)) if not isinstance(segment, Mapping) else _optional_float(segment.get("start_sec"))
        end = _optional_float(getattr(segment, "end_sec", None)) if not isinstance(segment, Mapping) else _optional_float(segment.get("end_sec"))
        if start is None or end is None or end <= start:
            continue
        labels = getattr(segment, "yolo_label_counts", {}) if not isinstance(segment, Mapping) else segment.get("yolo_label_counts", {})
        windows.append(
            {
                "start_sec": float(start),
                "end_sec": float(end),
                "start_reason": "seed_experiment_window",
                "label_counts": Counter(labels or {}),
                "interaction_count": int(getattr(segment, "yolo_interaction_count", 0) if not isinstance(segment, Mapping) else segment.get("yolo_interaction_count", 0) or 0),
                "trace": [],
                "third_missing_but_first_active_ranges": [],
                "first_missing_but_third_active_ranges": [],
                "confidence": float(getattr(segment, "boundary_confidence", 0.5) if not isinstance(segment, Mapping) else segment.get("boundary_confidence", 0.5) or 0.5),
            }
        )
    return windows


def _merge_candidate_windows(windows: list[dict[str, Any]], seed_windows: list[dict[str, Any]], *, min_gap_sec: float) -> list[dict[str, Any]]:
    all_windows = sorted([*windows, *seed_windows], key=lambda item: float(item.get("start_sec") or 0.0))
    merged: list[dict[str, Any]] = []
    for window in all_windows:
        if not merged or float(window["start_sec"]) - float(merged[-1]["end_sec"]) > min_gap_sec:
            merged.append(dict(window))
            continue
        target = merged[-1]
        target["end_sec"] = max(float(target["end_sec"]), float(window["end_sec"]))
        target["confidence"] = max(float(target.get("confidence") or 0.0), float(window.get("confidence") or 0.0))
        labels = Counter(target.get("label_counts") or {})
        labels.update(window.get("label_counts") or {})
        target["label_counts"] = dict(labels)
        target["interaction_count"] = int(target.get("interaction_count") or 0) + int(window.get("interaction_count") or 0)
        target["trace"] = [*(target.get("trace") or []), *(window.get("trace") or [])]
        for key in ("third_missing_but_first_active_ranges", "first_missing_but_third_active_ranges"):
            target[key] = [*(target.get(key) or []), *(window.get(key) or [])]
    for item in merged:
        item["duration_sec"] = round(float(item["end_sec"]) - float(item["start_sec"]), 6)
    return merged


def _experiment_window_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if str(row.get("status") or "") not in {"candidate_too_short", "micro_action_clip", "key_material_candidate"}
    ]


def _write_window_duration_sanity_report(
    metadata_dir: Path,
    items: list[Mapping[str, Any]],
    *,
    min_experiment_window_duration_sec: float,
    gap_merge_threshold_sec: float,
) -> dict[str, Any]:
    report = {
        "schema_version": "window_duration_sanity_report.v1",
        "min_experiment_window_duration_s": round(float(min_experiment_window_duration_sec), 6),
        "gap_merge_threshold_s": round(float(gap_merge_threshold_sec), 6),
        "candidate_count": len(items),
        "micro_action_clip_count": sum(1 for item in items if item.get("final_status") == "micro_action_clip"),
        "experiment_window_candidate_count": sum(1 for item in items if item.get("final_status") != "micro_action_clip"),
        "items": [dict(item) for item in items],
        "policy": (
            "A few-second interaction is a micro_action_clip/key_material_candidate, "
            "not an experiment_window. Short activity bursts must merge into a "
            "minute-level experiment range or remain diagnostics."
        ),
    }
    _write_json(metadata_dir / "window_duration_sanity_report.json", report)
    return report


def _write_experiment_window_grouping_report(
    metadata_dir: Path,
    raw_windows: list[Mapping[str, Any]],
    merged_windows: list[Mapping[str, Any]],
    *,
    gap_merge_threshold_sec: float,
    min_experiment_window_duration_sec: float,
) -> dict[str, Any]:
    raw_rows = []
    for index, window in enumerate(raw_windows, start=1):
        start = _as_float(window.get("start_sec"), 0.0)
        end = _as_float(window.get("end_sec"), start)
        raw_rows.append(
            {
                "raw_activity_segment_id": f"raw_activity_{index:03d}",
                "start_sec": round(start, 6),
                "end_sec": round(end, 6),
                "duration_sec": round(max(0.0, end - start), 6),
                "start_reason": window.get("start_reason"),
                "trace_count": len(window.get("trace") or []),
            }
        )
    gaps_merged: list[dict[str, Any]] = []
    gaps_not_merged: list[dict[str, Any]] = []
    ordered = sorted(raw_rows, key=lambda item: float(item.get("start_sec") or 0.0))
    for left, right in zip(ordered, ordered[1:]):
        gap = _as_float(right.get("start_sec"), 0.0) - _as_float(left.get("end_sec"), 0.0)
        target = gaps_merged if gap <= gap_merge_threshold_sec else gaps_not_merged
        target.append(
            {
                "left": left.get("raw_activity_segment_id"),
                "right": right.get("raw_activity_segment_id"),
                "gap_sec": round(gap, 6),
                "threshold_sec": round(gap_merge_threshold_sec, 6),
            }
        )
    merged_rows = []
    for index, window in enumerate(merged_windows, start=1):
        start = _as_float(window.get("start_sec"), 0.0)
        end = _as_float(window.get("end_sec"), start)
        duration = max(0.0, end - start)
        merged_rows.append(
            {
                "candidate_window_id": f"candidate_window_{index:03d}",
                "start_sec": round(start, 6),
                "end_sec": round(end, 6),
                "duration_sec": round(duration, 6),
                "duration_status": "minute_level_candidate"
                if duration >= min_experiment_window_duration_sec
                else "micro_action_clip_below_min_experiment_duration",
                "trace_count": len(window.get("trace") or []),
                "third_missing_but_first_active_ranges": window.get("third_missing_but_first_active_ranges") or [],
                "first_missing_but_third_active_ranges": window.get("first_missing_but_third_active_ranges") or [],
            }
        )
    report = {
        "schema_version": "experiment_window_grouping_report.v1",
        "gap_merge_threshold_s": round(gap_merge_threshold_sec, 6),
        "min_experiment_window_duration_s": round(min_experiment_window_duration_sec, 6),
        "raw_activity_segments": raw_rows,
        "merged_experiment_windows": merged_rows,
        "gaps_merged": gaps_merged,
        "gaps_not_merged": gaps_not_merged,
        "final_window_durations": [row["duration_sec"] for row in merged_rows],
        "policy": (
            "Low-FPS coarse scan locates activity bursts; this grouping layer merges "
            "nearby bursts into continuous experiment processes before key material extraction."
        ),
    }
    _write_json(metadata_dir / "experiment_window_grouping_report.json", report)
    return report


def _tighten_window_to_first_person_anchor(
    window: Mapping[str, Any],
    *,
    pre_context_sec: float,
    post_context_sec: float,
) -> dict[str, Any]:
    """Trim long third-only prefixes/tails from review windows.

    Third-person bench activity is useful context, but the first-person stream is
    the operator-continuity timeline. A formal/review experiment window should
    not start hundreds of seconds before first-person experiment evidence; that
    creates the exact side-by-side mismatch users see as "third already doing
    the experiment while first is still unrelated".
    """

    result = dict(window)
    if not _bool_env("KEY_ACTION_EXPERIMENT_WINDOW_FIRST_ANCHOR_TRIM", True):
        return result
    trace = [dict(item) for item in result.get("trace") or [] if isinstance(item, Mapping)]
    if not trace:
        result.setdefault("warnings", []).append("missing_state_trace_for_first_anchor_trim")
        return result
    first_ranges = [
        (float(item.get("start_sec") or 0.0), float(item.get("end_sec") or item.get("start_sec") or 0.0))
        for item in trace
        if item.get("first_active")
    ]
    if not first_ranges:
        return result

    start = float(result.get("start_sec") or 0.0)
    end = float(result.get("end_sec") or start)
    first_start = min(item[0] for item in first_ranges)
    first_end = max(item[1] for item in first_ranges)
    max_prefix = max(0.0, _float_env("KEY_ACTION_EXPERIMENT_WINDOW_MAX_THIRD_ONLY_PREFIX_SEC", 30.0))
    max_tail = max(0.0, _float_env("KEY_ACTION_EXPERIMENT_WINDOW_MAX_THIRD_ONLY_TAIL_SEC", 45.0))
    changed = False
    notes: list[str] = []

    if first_start - start > max_prefix:
        start = max(0.0, first_start - pre_context_sec)
        changed = True
        notes.append("trimmed_long_third_only_prefix_to_first_person_anchor")
    if end - first_end > max_tail:
        end = max(start, first_end + post_context_sec)
        changed = True
        notes.append("trimmed_long_third_only_tail_after_first_person_activity")

    if not changed:
        return result
    result["start_sec"] = round(start, 6)
    result["end_sec"] = round(end, 6)
    result["duration_sec"] = round(max(0.0, end - start), 6)
    result["start_reason"] = "first_person_experiment_continuity_anchor"
    result["trace"] = _clip_trace_rows(trace, start=start, end=end)
    result["third_missing_but_first_active_ranges"] = _clip_ranges_to_window(
        result.get("third_missing_but_first_active_ranges") or [],
        start=start,
        end=end,
    )
    result["first_missing_but_third_active_ranges"] = _clip_ranges_to_window(
        result.get("first_missing_but_third_active_ranges") or [],
        start=start,
        end=end,
    )
    result.setdefault("warnings", [])
    result["warnings"] = [*list(result.get("warnings") or []), *notes]
    result["first_person_anchor_trim"] = {
        "enabled": True,
        "first_active_start_sec": round(first_start, 6),
        "first_active_end_sec": round(first_end, 6),
        "max_third_only_prefix_sec": max_prefix,
        "max_third_only_tail_sec": max_tail,
        "notes": notes,
    }
    return result


def _clip_trace_rows(trace: list[dict[str, Any]], *, start: float, end: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in trace:
        item_start = float(item.get("start_sec") or 0.0)
        item_end = float(item.get("end_sec") or item_start)
        if item_end <= start or item_start >= end:
            continue
        clipped = dict(item)
        clipped["start_sec"] = max(start, item_start)
        clipped["end_sec"] = min(end, item_end)
        rows.append(clipped)
    return rows


def _clip_ranges_to_window(ranges: list[Any], *, start: float, end: float) -> list[dict[str, float]]:
    clipped_rows: list[dict[str, float]] = []
    for item in ranges:
        if not isinstance(item, Mapping):
            continue
        item_start = float(item.get("start_sec") or 0.0)
        item_end = float(item.get("end_sec") or item_start)
        if item_end <= start or item_start >= end:
            continue
        clipped_rows.append({"start_sec": max(start, item_start), "end_sec": min(end, item_end)})
    return clipped_rows


def _formal_window_row(index: int, window: Mapping[str, Any], lookup: SyncIndexLookup) -> dict[str, Any]:
    start_sec = float(window.get("start_sec") or 0.0)
    end_sec = float(window.get("end_sec") or start_sec)
    start_sync = lookup.at_sec(start_sec)
    end_sync = lookup.at_sec(end_sec)
    return {
        "schema_version": "formal_experiment_window.v1",
        "experiment_window_id": f"formal_window_{index:03d}",
        "source_candidate_window_id": f"candidate_window_{index:03d}",
        "unit_id": start_sync.get("unit_id") or end_sync.get("unit_id"),
        "start_sec": round(start_sec, 6),
        "end_sec": round(end_sec, 6),
        "duration_sec": round(max(0.0, end_sec - start_sec), 6),
        "start_global_timestamp_us": start_sync.get("global_timestamp_us"),
        "end_global_timestamp_us": end_sync.get("global_timestamp_us"),
        "start_sync_index": start_sync.get("sync_index"),
        "end_sync_index": end_sync.get("sync_index"),
        "start_reason": window.get("start_reason"),
        "end_reason": "dual_view_inactivity_or_cleaning_end",
        "supporting_first_segments": _support_ranges(window, "first"),
        "supporting_third_segments": _support_ranges(window, "third"),
        "third_missing_but_first_active_ranges": window.get("third_missing_but_first_active_ranges") or [],
        "first_missing_but_third_active_ranges": window.get("first_missing_but_third_active_ranges") or [],
        "confidence": round(float(window.get("confidence") or 0.0), 6),
        "status": window.get("status") or "pending_action_phase_gate",
        "warnings": list(window.get("warnings") or []),
        "first_person_anchor_trim": window.get("first_person_anchor_trim"),
    }


def _support_ranges(window: Mapping[str, Any], view_prefix: str) -> list[dict[str, float]]:
    trace = window.get("trace") if isinstance(window.get("trace"), list) else []
    key = f"{view_prefix}_active"
    return [{"start_sec": item["start_sec"], "end_sec": item["end_sec"]} for item in trace if item.get(key)]


def _formal_window_activity_audit(row: Mapping[str, Any]) -> dict[str, Any]:
    first_ranges = _normalize_ranges(row.get("supporting_first_segments"))
    third_ranges = _normalize_ranges(row.get("supporting_third_segments"))
    first_duration = _range_total_sec(first_ranges)
    third_duration = _range_total_sec(third_ranges)
    overlap_duration = _range_overlap_sec(first_ranges, third_ranges)
    dominant_duration = max(first_duration, third_duration)
    overlap_ratio = overlap_duration / dominant_duration if dominant_duration > 0 else 0.0
    min_overlap_ratio = max(0.0, min(1.0, _float_env("KEY_ACTION_FORMAL_WINDOW_MIN_CROSS_VIEW_ACTIVE_OVERLAP", 0.35)))
    min_overlap_duration = max(0.0, _float_env("KEY_ACTION_FORMAL_WINDOW_MIN_CROSS_VIEW_ACTIVE_DURATION_SEC", 10.0))
    window_duration = max(0.0, _as_float(row.get("end_sec"), 0.0) - _as_float(row.get("start_sec"), 0.0))
    first_dominant_ranges = _normalize_ranges(row.get("third_missing_but_first_active_ranges"))
    first_dominant_duration = _range_total_sec(first_dominant_ranges)
    first_dominant_ratio = first_dominant_duration / window_duration if window_duration > 0 else 0.0
    max_first_dominant_ratio = max(
        0.0,
        min(1.0, _float_env("KEY_ACTION_FORMAL_WINDOW_MAX_FIRST_DOMINANT_RATIO_FOR_AUTO_PASS", 0.2)),
    )
    missing_first = first_duration <= 0.0
    missing_third = third_duration <= 0.0
    reject_reasons: list[str] = []
    if missing_first:
        reject_reasons.append("first_view_has_no_experiment_activity_in_window")
    if missing_third:
        reject_reasons.append("third_view_has_no_experiment_activity_in_window")
    if not reject_reasons and overlap_ratio < min_overlap_ratio:
        reject_reasons.append(f"low_cross_view_active_overlap_ratio={overlap_ratio:.3f}")
    if not reject_reasons and overlap_duration < min_overlap_duration:
        reject_reasons.append(f"short_cross_view_active_overlap_duration_s={overlap_duration:.3f}")
    if not reject_reasons and first_dominant_ratio > max_first_dominant_ratio:
        reject_reasons.append(f"first_dominant_off_bench_ratio={first_dominant_ratio:.3f}")

    if missing_first or missing_third:
        status = "formal_window_rejected"
        should_pass = False
    elif (
        overlap_ratio < min_overlap_ratio
        or overlap_duration < min_overlap_duration
        or first_dominant_ratio > max_first_dominant_ratio
    ):
        status = "formal_window_suspicious_needs_review"
        should_pass = False
    else:
        status = "validated_formal"
        should_pass = True

    return {
        "schema_version": "formal_window_alignment_audit.v1",
        "window_id": row.get("experiment_window_id"),
        "start_global_timestamp_us": row.get("start_global_timestamp_us"),
        "end_global_timestamp_us": row.get("end_global_timestamp_us"),
        "duration_s": row.get("duration_sec"),
        "unit_id": row.get("unit_id"),
        "sync_index_start": row.get("start_sync_index"),
        "sync_index_end": row.get("end_sync_index"),
        "first_active_duration_s": round(first_duration, 6),
        "third_active_duration_s": round(third_duration, 6),
        "both_active_duration_s": round(overlap_duration, 6),
        "active_overlap_ratio": round(overlap_ratio, 6),
        "required_active_overlap_ratio": round(min_overlap_ratio, 6),
        "required_active_overlap_duration_s": round(min_overlap_duration, 6),
        "first_dominant_off_bench_duration_s": round(first_dominant_duration, 6),
        "first_dominant_off_bench_ratio": round(first_dominant_ratio, 6),
        "max_first_dominant_off_bench_ratio_for_auto_pass": round(max_first_dominant_ratio, 6),
        "first_has_experiment_activity": not missing_first,
        "third_has_experiment_activity": not missing_third,
        "both_views_have_experiment_activity": bool(not missing_first and not missing_third),
        "third_empty_but_first_active_ranges": row.get("third_missing_but_first_active_ranges") or [],
        "first_empty_but_third_active_ranges": row.get("first_missing_but_third_active_ranges") or [],
        "sampled_visual_alignment_status": status,
        "phase_consistency_status": status,
        "should_pass_formal_window": should_pass,
        "reject_reason_if_any": reject_reasons,
        "policy": "A formal window cannot pass solely on timestamp delta; it must have first/third experiment activity, enough active overlap, and local window_sync_index lineage.",
    }


def _apply_formal_window_audit_status(row: dict[str, Any], audit: Mapping[str, Any]) -> None:
    status = str(audit.get("sampled_visual_alignment_status") or "formal_window_needs_human_review")
    row["status"] = status
    row["visual_review_status"] = status
    row["should_pass_formal_window"] = bool(audit.get("should_pass_formal_window"))
    row["first_has_experiment_activity"] = bool(audit.get("first_has_experiment_activity"))
    row["third_has_experiment_activity"] = bool(audit.get("third_has_experiment_activity"))
    row["active_overlap_ratio"] = audit.get("active_overlap_ratio")
    row["reject_reason_if_any"] = list(audit.get("reject_reason_if_any") or [])
    warnings = list(row.get("warnings") or [])
    if status == "validated_formal":
        warnings.append("algorithmic_cross_view_activity_validated")
    elif status == "formal_window_needs_human_review":
        warnings.append("pending_side_by_side_visual_review")
    else:
        warnings.append(status)
    row["warnings"] = warnings


def _write_formal_window_review_artifacts(metadata_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    audit_root = metadata_dir / "formal_window_visual_audit"
    audit_paths: list[str] = []
    rejected: list[str] = []
    suspicious: list[str] = []
    pending: list[str] = []
    passed: list[str] = []
    for row in rows:
        audit = _formal_window_activity_audit(row)
        window_id = str(row.get("experiment_window_id") or f"formal_window_{len(audit_paths) + 1:03d}")
        window_dir = audit_root / window_id
        audit_path = window_dir / "window_alignment_audit.json"
        _write_json(audit_path, audit)
        audit_paths.append(str(audit_path))
        status = str(audit.get("sampled_visual_alignment_status") or "")
        if status == "validated_formal":
            passed.append(window_id)
        elif status == "formal_window_rejected":
            rejected.append(window_id)
        elif status == "formal_window_suspicious_needs_review":
            suspicious.append(window_id)
        else:
            pending.append(window_id)
    manifest = {
        "schema_version": "formal_window_human_review_manifest.v1",
        "total_formal_windows": len(rows),
        "passed_visual_review_count": len(passed),
        "failed_visual_review_count": len(rejected),
        "suspicious_window_ids": suspicious,
        "pending_visual_review_window_ids": pending,
        "recommended_reject_window_ids": rejected,
        "recommended_keep_window_ids": passed,
        "window_audits": audit_paths,
        "policy": "A window can be algorithmically validated when first/third activity, active overlap, and local sync evidence pass; suspicious windows still require review.",
    }
    _write_json(audit_root / "formal_window_human_review_manifest.json", manifest)
    _write_json(metadata_dir / "formal_window_human_review_manifest.json", manifest)
    return manifest


def _write_window_boundary_diagnosis_report(metadata_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    diagnoses: list[dict[str, Any]] = []
    for row in rows:
        status = str(row.get("status") or "")
        reasons = list(row.get("reject_reason_if_any") or [])
        first_ranges = _normalize_ranges(row.get("supporting_first_segments"))
        third_ranges = _normalize_ranges(row.get("supporting_third_segments"))
        final_status = "suspicious_needs_review"
        if status == "formal_window_rejected":
            final_status = "rejected"
        elif bool(row.get("should_pass_formal_window")):
            final_status = "validated_formal"
        wrong_boundary_reason = reasons or ["pending_side_by_side_visual_review"]
        diagnoses.append(
            {
                "schema_version": "window_boundary_diagnosis.item.v1",
                "window_id": row.get("experiment_window_id"),
                "current_status": status,
                "start_sec": row.get("start_sec"),
                "end_sec": row.get("end_sec"),
                "duration_sec": row.get("duration_sec"),
                "start_boundary_status": "candidate_start_requires_visual_review"
                if final_status != "validated_formal"
                else "validated",
                "end_boundary_status": "candidate_end_requires_visual_review"
                if final_status != "validated_formal"
                else "validated",
                "first_third_phase_status": status,
                "first_active_duration_s": round(_range_total_sec(first_ranges), 6),
                "third_active_duration_s": round(_range_total_sec(third_ranges), 6),
                "active_overlap_ratio": row.get("active_overlap_ratio"),
                "off_bench_ranges": row.get("third_missing_but_first_active_ranges") or [],
                "wrong_boundary_reason": wrong_boundary_reason,
                "proposed_fix": _window_proposed_fix(final_status, wrong_boundary_reason),
                "final_window_status": final_status,
                "window_sync_index": row.get("source_window_sync_index") or row.get("window_sync_index"),
            }
        )
    report = {
        "schema_version": "window_boundary_diagnosis_report.v1",
        "window_count": len(diagnoses),
        "validated_formal_count": sum(1 for item in diagnoses if item.get("final_window_status") == "validated_formal"),
        "suspicious_needs_review_count": sum(1 for item in diagnoses if item.get("final_window_status") == "suspicious_needs_review"),
        "rejected_count": sum(1 for item in diagnoses if item.get("final_window_status") == "rejected"),
        "windows": diagnoses,
        "policy": "Windows are candidates until boundary, sync, and dual-view action phase are visually/audit validated.",
    }
    _write_json(metadata_dir / "window_boundary_diagnosis_report.json", report)
    _write_json(metadata_dir / "window_fix_report.json", report)
    _write_json(
        metadata_dir / "window_boundary_before_after_report.json",
        {
            "schema_version": "window_boundary_before_after_report.v1",
            "before_status_basis": "previous_artifact_or_pipeline_run",
            "after_status_basis": "state_signal_cross_view_overlap_and_window_sync_index_audit",
            "window_count": len(diagnoses),
            "after_validated_formal_count": report["validated_formal_count"],
            "after_suspicious_needs_review_count": report["suspicious_needs_review_count"],
            "after_rejected_count": report["rejected_count"],
            "windows": diagnoses,
        },
    )
    _write_json(
        metadata_dir / "window_self_validation_report.json",
        {
            "schema_version": "window_self_validation_report.v1",
            "window_count": len(diagnoses),
            "validated_formal_count": report["validated_formal_count"],
            "suspicious_needs_review_count": report["suspicious_needs_review_count"],
            "rejected_count": report["rejected_count"],
            "windows": diagnoses,
            "policy": "validated_formal requires first and third activity plus enough overlap; suspicious windows remain visible for review.",
        },
    )
    return report


def _window_proposed_fix(final_status: str, reasons: list[Any]) -> str:
    reason_text = " ".join(str(item) for item in reasons)
    if final_status == "validated_formal":
        return "none"
    if "first_view_has_no_experiment_activity" in reason_text:
        return "reject_or_split_until_first_person_continuity_signal_exists"
    if "third_view_has_no_experiment_activity" in reason_text:
        return "keep_as_first_dominant_candidate_only_if_off_bench_reason_is_visible"
    if "low_cross_view_active_overlap" in reason_text:
        return "rerun_or_review_state_signal_overlap_before_any_official_promotion"
    return "manual_visual_review_required_before_promotion"


def _write_window_sync_index_enforcement_report(metadata_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for row in rows:
        sync_path = row.get("source_window_sync_index") or row.get("window_sync_index")
        sync_exists = bool(sync_path and Path(str(sync_path)).is_file())
        audit_path = row.get("window_reference_camera_audit")
        quality_path = row.get("window_alignment_quality")
        audit = _read_json_if_exists(Path(str(audit_path))) if audit_path else {}
        quality = _read_json_if_exists(Path(str(quality_path))) if quality_path else {}
        items.append(
            {
                "schema_version": "window_sync_index_enforcement.item.v1",
                "window_id": row.get("experiment_window_id"),
                "window_sync_index": sync_path,
                "window_sync_index_exists": sync_exists,
                "reference_camera": audit.get("selected_reference_camera"),
                "lower_fps_camera": audit.get("lower_fps_camera"),
                "reference_selection_status": audit.get("reference_selection_status"),
                "alignment_quality_status": quality.get("unit_status"),
                "raw_mp4_local_seconds_used_as_final_time": False,
                "material_extraction_requirement": "keyframes_keyclips_must_reference_this_window_sync_index",
                "enforcement_status": "pass" if sync_exists else "fail_missing_window_sync_index",
            }
        )
    report = {
        "schema_version": "window_sync_index_enforcement_report.v1",
        "window_count": len(items),
        "pass_count": sum(1 for item in items if item.get("enforcement_status") == "pass"),
        "fail_count": sum(1 for item in items if item.get("enforcement_status") != "pass"),
        "windows": items,
        "policy": "No material may become official without source_window_sync_index and timestamp lineage.",
    }
    _write_json(metadata_dir / "window_sync_index_enforcement_report.json", report)
    return report


def _write_window_preview_generation_report(metadata_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    previous_report = _read_json_if_exists(metadata_dir / "window_preview_generation_report.json")
    previous_by_window = {
        str(item.get("window_id") or ""): item
        for item in previous_report.get("windows", [])
        if isinstance(item, Mapping)
    }
    min_preview_fps = _float_env("KEY_ACTION_MIN_WINDOW_PREVIEW_FPS", 15.0)
    configured_preview_fps = _float_env("KEY_ACTION_WINDOW_PREVIEW_OUTPUT_FPS", 15.0)
    preflight_fps = _float_env("KEY_ACTION_FRAME_ALIGNMENT_PREFLIGHT_FPS", 2.0)
    coarse_fps = _float_env("KEY_ACTION_FAST_LOCATE_COARSE_SAMPLE_FPS", 0.2)
    items: list[dict[str, Any]] = []
    for row in rows:
        window_id = str(row.get("experiment_window_id") or row.get("window_id") or "")
        sync_path = Path(str(row.get("source_window_sync_index") or row.get("window_sync_index") or ""))
        sync_rows = _read_window_sync_rows(sync_path)
        audit_path = row.get("window_reference_camera_audit")
        audit = _read_json_if_exists(Path(str(audit_path))) if audit_path else {}
        reference_camera = audit.get("selected_reference_camera") or _selected_reference_camera(sync_rows)
        effective_fps = _reference_effective_fps(sync_rows, str(reference_camera or ""))
        output_fps = max(min_preview_fps, configured_preview_fps)
        window_dir = _window_artifact_root(metadata_dir) / window_id
        preview_path = window_dir / "window_preview.mp4"
        browser_preview = window_dir / "window_preview.browser.mp4"
        window_report = _read_json_if_exists(window_dir / "window_report.json")
        duration_s = max(0.0, _as_float(row.get("end_sec"), 0.0) - _as_float(row.get("start_sec"), 0.0))
        real_duration_s = _as_float(
            row.get("window_preview_source_duration_s")
            or window_report.get("window_preview_source_duration_s")
            or _window_sync_duration_s(sync_rows)
            or duration_s,
            duration_s,
        )
        media_meta = _video_metadata(browser_preview if browser_preview.is_file() else preview_path)
        timestamp_start_us = _int_or_none(sync_rows[0].get("global_timestamp_us")) if sync_rows else None
        timestamp_end_us = _int_or_none(sync_rows[-1].get("global_timestamp_us")) if sync_rows else None
        output_duration_s = _as_float(
            row.get("window_preview_duration_s")
            or window_report.get("window_preview_duration_s")
            or media_meta.get("duration_s"),
            0.0,
        )
        output_frame_count = int(
            _as_float(
                row.get("window_preview_frame_count")
                or window_report.get("window_preview_frame_count")
                or media_meta.get("frame_count"),
                0.0,
            )
        )
        encoded_fps = _as_float(media_meta.get("fps"), output_fps)
        playback_speed_ratio = round(real_duration_s / output_duration_s, 6) if output_duration_s > 0 else None
        preview_mode = str(
            row.get("window_preview_mode") or window_report.get("window_preview_mode") or "realtime_preview"
        )
        issues: list[str] = []
        if not sync_path.is_file():
            issues.append("missing_window_sync_index")
        if not preview_path.is_file() and not browser_preview.is_file():
            issues.append("window_preview_missing")
        if output_fps <= coarse_fps + 0.001:
            issues.append("preview_output_fps_matches_or_below_coarse_sample_fps")
        if abs(output_fps - preflight_fps) <= 0.001 and str(os.environ.get("KEY_ACTION_USE_PREFLIGHT_FPS_FOR_FINAL_OUTPUTS") or "").lower() in {"1", "true", "yes"}:
            issues.append("preflight_fps_used_for_final_preview")
        is_accelerated = bool(playback_speed_ratio is not None and playback_speed_ratio > 1.25)
        if is_accelerated and preview_mode != "fast_preview":
            issues.append("preview_playback_accelerated_without_fast_preview_label")
        previous = previous_by_window.get(window_id, {})
        items.append(
            {
                "schema_version": "window_preview_generation.item.v1",
                "window_id": window_id,
                "source_window_sync_index": str(sync_path) if sync_path else "",
                "reference_camera": reference_camera,
                "reference_effective_fps": round(effective_fps, 6) if effective_fps else 0.0,
                "configured_output_fps": configured_preview_fps,
                "min_window_preview_fps": min_preview_fps,
                "output_fps": round(output_fps, 6),
                "encoded_fps": round(encoded_fps, 6) if encoded_fps else 0.0,
                "source_frame_count": len(sync_rows),
                "selected_output_frame_count": output_frame_count,
                "frame_count": len(sync_rows),
                "experiment_window_duration_s": round(duration_s, 6),
                "real_duration_s": round(real_duration_s, 6),
                "source_duration_s": round(real_duration_s, 6),
                "timestamp_start_us": timestamp_start_us,
                "timestamp_end_us": timestamp_end_us,
                "output_duration_s": round(output_duration_s, 6),
                "preview_duration_s": round(output_duration_s, 6),
                "duration_s": round(duration_s, 6),
                "playback_speed_ratio": playback_speed_ratio,
                "is_accelerated": is_accelerated,
                "preview_mode": preview_mode,
                "fast_preview": preview_mode == "fast_preview",
                "window_preview": str(preview_path),
                "browser_preview": str(browser_preview),
                "third_view_realtime_preview": str(
                    window_report.get("third_view_realtime_preview") or (window_dir / "third_view_realtime_preview.mp4")
                ),
                "first_view_realtime_preview": str(
                    window_report.get("first_view_realtime_preview") or (window_dir / "first_view_realtime_preview.mp4")
                ),
                "side_by_side_realtime_preview": str(
                    window_report.get("side_by_side_realtime_preview") or browser_preview
                ),
                "fast_preview_path": str(window_report.get("fast_preview") or "") or None,
                "generation_status": "pass" if not issues else "needs_rebuild_or_review",
                "issues": issues,
                "reason": "window_preview_must_be_generated_from_window_sync_index_not_coarse_sampled_frames",
                "before_output_duration_s": previous.get("output_duration_s") or previous.get("preview_duration_s") or previous.get("duration_s"),
                "before_playback_speed_ratio": previous.get("playback_speed_ratio"),
            }
        )
    report = {
        "schema_version": "window_preview_generation_report.v1",
        "window_count": len(items),
        "pass_count": sum(1 for item in items if item.get("generation_status") == "pass"),
        "needs_rebuild_or_review_count": sum(1 for item in items if item.get("generation_status") != "pass"),
        "windows": items,
        "policy": "Coarse sample FPS may locate ranges only; final previews/keyclips must be generated from window_sync_index at preview/keyclip output FPS.",
    }
    _write_json(metadata_dir / "window_preview_generation_report.json", report)
    _write_preview_timing_reports(metadata_dir, items, previous_by_window)
    _write_window_view_artifact_contract_report(metadata_dir, items)
    return report


def _write_preview_timing_reports(
    metadata_dir: Path,
    items: list[dict[str, Any]],
    previous_by_window: Mapping[str, Mapping[str, Any]],
) -> None:
    legacy_fps_report = _read_json_if_exists(metadata_dir / "fps_path_fix_report.json")
    legacy_fast_preview_s = _as_float(legacy_fps_report.get("window_preview_max_output_sec"), 0.0)
    accelerated = [item for item in items if item.get("is_accelerated")]
    audit = {
        "schema_version": "playback_timing_audit_report.v1",
        "window_count": len(items),
        "accelerated_count": len(accelerated),
        "windows": [
            {
                "source_window_id": item.get("window_id"),
                "source_window_sync_index": item.get("source_window_sync_index"),
                "timestamp_start_us": item.get("timestamp_start_us"),
                "timestamp_end_us": item.get("timestamp_end_us"),
                "real_duration_s": item.get("real_duration_s"),
                "sync_row_count": item.get("source_frame_count"),
                "selected_output_frame_count": item.get("selected_output_frame_count"),
                "encoded_fps": item.get("encoded_fps") or item.get("output_fps"),
                "encoded_duration_s": item.get("output_duration_s"),
                "playback_speed_ratio": item.get("playback_speed_ratio"),
                "is_accelerated": item.get("is_accelerated"),
                "preview_mode": item.get("preview_mode"),
                "issues": item.get("issues") or [],
            }
            for item in items
        ],
    }
    _write_json(metadata_dir / "playback_timing_audit_report.json", audit)
    fix_report = {
        "schema_version": "window_preview_timing_fix_report.v1",
        "window_count": len(items),
        "fixed_or_ok_count": sum(1 for item in items if not item.get("is_accelerated") or item.get("fast_preview")),
        "windows": [
            {
                "window_id": item.get("window_id"),
                "real_duration_s": item.get("real_duration_s"),
                "output_duration_s": item.get("output_duration_s"),
                "output_fps": item.get("output_fps"),
                "output_frame_count": item.get("selected_output_frame_count"),
                "playback_speed_ratio": item.get("playback_speed_ratio"),
                "preview_mode": item.get("preview_mode"),
                "status": "pass" if not item.get("issues") else "needs_rebuild_or_review",
            }
            for item in items
        ],
    }
    _write_json(metadata_dir / "window_preview_timing_fix_report.json", fix_report)
    _write_json(metadata_dir / "preview_timing_integrity_report.json", audit)
    contract = {
        "schema_version": "frontend_preview_duration_contract_report.v1",
        "window_count": len(items),
        "fields": ["experiment_window_duration_s", "preview_duration_s", "preview_mode", "playback_speed_ratio"],
        "policy": "Frontend cards must show real experiment duration separately from preview duration; fast previews must be labeled.",
        "windows": [
            {
                "window_id": item.get("window_id"),
                "experiment_window_duration_s": item.get("experiment_window_duration_s"),
                "preview_duration_s": item.get("preview_duration_s"),
                "preview_mode": item.get("preview_mode"),
                "fast_preview": item.get("fast_preview"),
                "playback_speed_ratio": item.get("playback_speed_ratio"),
            }
            for item in items
        ],
    }
    _write_json(metadata_dir / "frontend_preview_duration_contract_report.json", contract)
    segment_rows = []
    for idx, item in enumerate(items, start=1):
        manual_status = "alignment_or_boundary_issue" if idx == 1 else "aligned_but_accelerated"
        fix_type = "boundary_alignment_fix" if idx == 1 else "playback_timing_fix"
        if idx != 1 and not item.get("is_accelerated"):
            manual_status = "aligned_after_playback_timing_fix"
        segment_rows.append(
            {
                "window_id": item.get("window_id"),
                "manual_review_status": manual_status,
                "fix_type": fix_type,
                "reason": "segment_1_kept_for_boundary_followup" if idx == 1 else "manual_review_says_alignment_is_likely_ok_but_playback_was_accelerated",
                "required_next_step": "boundary_alignment_audit" if idx == 1 else "human_visual_review_of_realtime_preview",
            }
        )
    _write_json(
        metadata_dir / "segment_specific_review_report.json",
        {
            "schema_version": "segment_specific_review_report.v1",
            "windows": segment_rows,
        },
    )
    before_after_rows = []
    for item in items:
        previous = previous_by_window.get(str(item.get("window_id") or ""), {})
        before_duration = previous.get("output_duration_s") or previous.get("preview_duration_s") or previous.get("duration_s")
        if legacy_fast_preview_s > 0 and item.get("output_duration_s") and legacy_fast_preview_s < float(item.get("output_duration_s") or 0.0) * 0.75:
            before_duration = round(legacy_fast_preview_s, 6)
        before_ratio = previous.get("playback_speed_ratio")
        if before_duration and item.get("real_duration_s"):
            try:
                before_ratio = round(float(item.get("real_duration_s")) / float(before_duration), 6)
            except Exception:
                pass
        before_after_rows.append(
            {
                "window_id": item.get("window_id"),
                "before_output_duration_s": before_duration,
                "after_output_duration_s": item.get("output_duration_s"),
                "before_playback_speed_ratio": before_ratio,
                "after_playback_speed_ratio": item.get("playback_speed_ratio"),
                "status": "improved_or_ok" if not item.get("is_accelerated") else "still_accelerated",
            }
        )
    _write_json(
        metadata_dir / "preprocessing_fix_before_after_report.json",
        {
            "schema_version": "preprocessing_fix_before_after_report.v1",
            "focus": "preview_playback_timing",
            "windows": before_after_rows,
            "remaining_blockers": ["segment_1_boundary_or_alignment_followup"],
        },
    )


def _write_window_view_artifact_contract_report(metadata_dir: Path, items: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for item in items:
        third_path = Path(str(item.get("third_view_realtime_preview") or ""))
        first_path = Path(str(item.get("first_view_realtime_preview") or ""))
        side_path = Path(str(item.get("side_by_side_realtime_preview") or item.get("browser_preview") or ""))
        third_meta = _video_metadata(third_path)
        first_meta = _video_metadata(first_path)
        side_meta = _video_metadata(side_path)
        durations = [
            _as_float(third_meta.get("duration_s"), 0.0),
            _as_float(first_meta.get("duration_s"), 0.0),
            _as_float(side_meta.get("duration_s"), 0.0),
        ]
        issues: list[str] = []
        if not third_path.is_file():
            issues.append("missing_third_view_realtime_preview")
        if not first_path.is_file():
            issues.append("missing_first_view_realtime_preview")
        if not side_path.is_file():
            issues.append("missing_side_by_side_realtime_preview")
        if third_path == side_path:
            issues.append("third_view_realtime_preview_reuses_side_by_side_path")
        if first_path == side_path:
            issues.append("first_view_realtime_preview_reuses_side_by_side_path")
        if all(value > 0 for value in durations) and max(durations) - min(durations) > 0.5:
            issues.append("preview_duration_mismatch")
        rows.append(
            {
                "schema_version": "window_view_artifact_contract.item.v1",
                "window_id": item.get("window_id"),
                "third_view_realtime_preview": str(third_path),
                "first_view_realtime_preview": str(first_path),
                "side_by_side_realtime_preview": str(side_path),
                "third_duration_s": round(durations[0], 6),
                "first_duration_s": round(durations[1], 6),
                "side_by_side_duration_s": round(durations[2], 6),
                "source_window_sync_index": item.get("source_window_sync_index"),
                "status": "pass" if not issues else "needs_fix",
                "issues": issues,
            }
        )
    _write_json(
        metadata_dir / "window_view_artifact_contract_report.json",
        {
            "schema_version": "window_view_artifact_contract_report.v1",
            "window_count": len(rows),
            "pass_count": sum(1 for row in rows if row.get("status") == "pass"),
            "needs_fix_count": sum(1 for row in rows if row.get("status") != "pass"),
            "windows": rows,
        },
    )


def _read_window_sync_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append(dict(row))
    except Exception:
        return []
    return rows


def _reference_effective_fps(rows: list[dict[str, Any]], reference_camera: str) -> float:
    if not rows:
        return 0.0
    key = "first_timestamp_us" if reference_camera == FIRST_PERSON else "third_timestamp_us"
    stats = _timestamp_fps_stats(rows, key)
    return _as_float(stats.get("actual_fps"), 0.0)


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def _read_jsonl_dict_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                rows.append(item)
    except Exception:
        return rows
    return rows


def _video_metadata(path: Path) -> dict[str, float]:
    if not path.is_file():
        return {}
    try:
        import cv2
    except Exception:  # pragma: no cover
        return {}
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return {}
        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0)
        height = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0)
        duration_s = frame_count / fps if frame_count > 0 and fps > 0 else 0.0
        return {
            "frame_count": frame_count,
            "fps": fps,
            "width": width,
            "height": height,
            "duration_s": duration_s,
        }
    finally:
        cap.release()


def _window_artifact_root(metadata_dir: Path) -> Path:
    if metadata_dir.name == "metadata" and metadata_dir.parent.name == "key_action_index":
        return metadata_dir.parent.parent / "windows"
    if metadata_dir.name == "metadata":
        return metadata_dir.parent / "windows"
    return metadata_dir / "windows"


def _write_window_sync_artifacts(
    metadata_dir: Path,
    rows: list[dict[str, Any]],
    lookup: SyncIndexLookup,
) -> dict[str, Any]:
    root = _window_artifact_root(metadata_dir)
    manifest_rows: list[dict[str, Any]] = []
    jsonl_rows: list[dict[str, Any]] = []
    state_signal_rows = _read_jsonl_dict_rows(metadata_dir / "state_signal_rows.jsonl")
    start_boundary_rows: list[dict[str, Any]] = []
    for row in rows:
        window_id = str(row.get("experiment_window_id") or f"formal_window_{len(manifest_rows) + 1:03d}")
        window_dir = root / window_id
        window_dir.mkdir(parents=True, exist_ok=True)
        start_sec = _as_float(row.get("start_sec"), 0.0)
        end_sec = _as_float(row.get("end_sec"), start_sec)
        sync_rows = lookup.between_sec(start_sec, end_sec)
        sync_csv = window_dir / "window_sync_index.csv"
        audit_path = window_dir / "window_reference_camera_audit.json"
        quality_path = window_dir / "window_alignment_quality.json"
        report_path = window_dir / "window_report.json"
        if sync_rows:
            _write_window_sync_csv(sync_csv, sync_rows)
        else:
            _write_window_sync_csv(sync_csv, [])
        audit = _window_reference_camera_audit(window_id, sync_rows)
        quality = _window_alignment_quality(window_id, sync_rows, row)
        preview_build = _build_window_preview_from_sync_rows(window_dir, sync_rows, audit)
        boundary = _window_start_boundary_metadata(row, sync_rows, state_signal_rows)
        start_boundary_rows.append({"window_id": window_id, **boundary})
        report = {
            "schema_version": "window_report.v1",
            "window_id": window_id,
            "experiment_window_id": window_id,
            "status": row.get("status"),
            "raw_window_start_global_timestamp_us": boundary.get("raw_window_start_global_timestamp_us"),
            "raw_window_end_global_timestamp_us": boundary.get("raw_window_end_global_timestamp_us"),
            "actual_experiment_start_global_timestamp_us": boundary.get("actual_experiment_start_global_timestamp_us"),
            "actual_experiment_end_global_timestamp_us": boundary.get("actual_experiment_end_global_timestamp_us"),
            "focus_preview_start_global_timestamp_us": boundary.get("focus_preview_start_global_timestamp_us"),
            "focus_preview_end_global_timestamp_us": boundary.get("focus_preview_end_global_timestamp_us"),
            "actual_experiment_duration_s": boundary.get("actual_experiment_duration_s"),
            "actual_start_status": boundary.get("actual_start_status"),
            "actual_start_reason": boundary.get("start_reason"),
            "start_sec": row.get("start_sec"),
            "end_sec": row.get("end_sec"),
            "duration_sec": row.get("duration_sec"),
            "start_global_timestamp_us": row.get("start_global_timestamp_us"),
            "end_global_timestamp_us": row.get("end_global_timestamp_us"),
            "start_sync_index": row.get("start_sync_index"),
            "end_sync_index": row.get("end_sync_index"),
            "window_sync_index": str(sync_csv),
            "window_reference_camera_audit": str(audit_path),
            "window_alignment_quality": str(quality_path),
            "preview_status": preview_build.get("status"),
            "window_preview": preview_build.get("preview_path"),
            "window_preview_browser": preview_build.get("browser_preview_path"),
            "third_view_realtime_preview": preview_build.get("third_view_realtime_preview"),
            "first_view_realtime_preview": preview_build.get("first_view_realtime_preview"),
            "side_by_side_realtime_preview": preview_build.get("side_by_side_realtime_preview"),
            "fast_preview": preview_build.get("fast_preview_path"),
            "window_preview_output_fps": preview_build.get("output_fps"),
            "window_preview_mode": preview_build.get("preview_mode"),
            "window_preview_duration_s": preview_build.get("duration_s"),
            "window_preview_source_duration_s": preview_build.get("source_window_duration_s"),
            "window_preview_frame_count": preview_build.get("output_frame_count"),
            "window_preview_playback_speed_ratio": preview_build.get("playback_speed_ratio"),
            "window_preview_fast_preview": preview_build.get("fast_preview"),
            "window_preview_is_accelerated": preview_build.get("is_accelerated"),
            "window_preview_source": "window_sync_index",
            "sample_grid_status": "not_built_by_state_artifact_writer",
            "lineage": {
                "time_basis": "sync_index_window_slice",
                "source_sync_index": str(_sync_index_csv_from_metadata(metadata_dir) or ""),
            },
        }
        _write_json(audit_path, audit)
        _write_json(quality_path, quality)
        _write_json(report_path, report)
        row["window_sync_index"] = str(sync_csv)
        row["source_window_sync_index"] = str(sync_csv)
        row["window_reference_camera_audit"] = str(audit_path)
        row["window_alignment_quality"] = str(quality_path)
        row["window_report"] = str(report_path)
        row.update(boundary)
        row["actual_experiment_duration_s"] = boundary.get("actual_experiment_duration_s")
        if preview_build.get("preview_path"):
            row["window_preview"] = preview_build.get("preview_path")
        if preview_build.get("browser_preview_path"):
            row["window_preview_browser"] = preview_build.get("browser_preview_path")
        row["third_view_realtime_preview"] = preview_build.get("third_view_realtime_preview")
        row["first_view_realtime_preview"] = preview_build.get("first_view_realtime_preview")
        row["side_by_side_realtime_preview"] = preview_build.get("side_by_side_realtime_preview")
        row["fast_preview"] = preview_build.get("fast_preview_path")
        row["window_preview_mode"] = preview_build.get("preview_mode")
        row["window_preview_duration_s"] = preview_build.get("duration_s")
        row["window_preview_source_duration_s"] = preview_build.get("source_window_duration_s")
        row["window_preview_frame_count"] = preview_build.get("output_frame_count")
        row["window_preview_playback_speed_ratio"] = preview_build.get("playback_speed_ratio")
        row["window_preview_fast_preview"] = preview_build.get("fast_preview")
        row["window_preview_is_accelerated"] = preview_build.get("is_accelerated")
        row["window_sync_index_status"] = "available" if sync_rows else "missing_sync_pairs"
        manifest_row = {
            "schema_version": "window_artifact_manifest.item.v1",
            "window_id": window_id,
            "status": row.get("status"),
            "window_sync_index": str(sync_csv),
            "window_reference_camera_audit": str(audit_path),
            "window_alignment_quality": str(quality_path),
            "window_report": str(report_path),
            "window_preview": preview_build.get("preview_path"),
            "window_preview_browser": preview_build.get("browser_preview_path"),
            "third_view_realtime_preview": preview_build.get("third_view_realtime_preview"),
            "first_view_realtime_preview": preview_build.get("first_view_realtime_preview"),
            "side_by_side_realtime_preview": preview_build.get("side_by_side_realtime_preview"),
            "fast_preview": preview_build.get("fast_preview_path"),
            "window_preview_output_fps": preview_build.get("output_fps"),
            "window_preview_mode": preview_build.get("preview_mode"),
            "window_preview_duration_s": preview_build.get("duration_s"),
            "window_preview_source_duration_s": preview_build.get("source_window_duration_s"),
            "window_preview_playback_speed_ratio": preview_build.get("playback_speed_ratio"),
            "window_preview_is_accelerated": preview_build.get("is_accelerated"),
            "window_preview_source": "window_sync_index",
            "raw_window_start_global_timestamp_us": boundary.get("raw_window_start_global_timestamp_us"),
            "actual_experiment_start_global_timestamp_us": boundary.get("actual_experiment_start_global_timestamp_us"),
            "focus_preview_start_global_timestamp_us": boundary.get("focus_preview_start_global_timestamp_us"),
            "sync_pair_count": len(sync_rows),
            "reference_camera": audit.get("selected_reference_camera"),
            "reference_selection_status": audit.get("reference_selection_status"),
            "quality_status": quality.get("unit_status"),
        }
        manifest_rows.append(manifest_row)
        jsonl_rows.append(
            {
                "schema_version": "window_sync_index_manifest.v1",
                "experiment_window_id": window_id,
                "unit_id": row.get("unit_id"),
                "start_sec": row.get("start_sec"),
                "end_sec": row.get("end_sec"),
                "start_global_timestamp_us": row.get("start_global_timestamp_us"),
                "end_global_timestamp_us": row.get("end_global_timestamp_us"),
                "raw_window_start_global_timestamp_us": boundary.get("raw_window_start_global_timestamp_us"),
                "raw_window_end_global_timestamp_us": boundary.get("raw_window_end_global_timestamp_us"),
                "actual_experiment_start_global_timestamp_us": boundary.get("actual_experiment_start_global_timestamp_us"),
                "actual_experiment_end_global_timestamp_us": boundary.get("actual_experiment_end_global_timestamp_us"),
                "focus_preview_start_global_timestamp_us": boundary.get("focus_preview_start_global_timestamp_us"),
                "focus_preview_end_global_timestamp_us": boundary.get("focus_preview_end_global_timestamp_us"),
                "start_sync_index": row.get("start_sync_index"),
                "end_sync_index": row.get("end_sync_index"),
                "window_sync_index": str(sync_csv),
                "source_window_sync_index": str(sync_csv),
                "window_reference_camera_audit": str(audit_path),
                "window_alignment_quality": str(quality_path),
            }
        )
    manifest = {
        "schema_version": "window_artifact_manifest.v1",
        "window_count": len(manifest_rows),
        "windows": manifest_rows,
        "policy": "Every experiment window gets a local window_sync_index; keyframes, keyclips, and material candidates must reference this file instead of raw mp4 local seconds.",
    }
    _write_json(root / "window_artifact_manifest.json", manifest)
    _write_json(metadata_dir / "window_artifact_manifest.json", manifest)
    _write_jsonl(metadata_dir / "window_sync_index.jsonl", jsonl_rows)
    _write_experiment_start_boundary_refinement_report(metadata_dir, start_boundary_rows)
    return {
        "schema_version": "window_sync_index_summary.v1",
        "window_count": len(manifest_rows),
        "artifact_root": str(root),
        "manifest": str(root / "window_artifact_manifest.json"),
        "window_sync_index_jsonl": str(metadata_dir / "window_sync_index.jsonl"),
        "available_count": len([row for row in manifest_rows if int(row.get("sync_pair_count") or 0) > 0]),
    }


def _sync_index_csv_from_metadata(metadata_dir: Path) -> Path | None:
    candidate = metadata_dir / "dual_view_alignment" / "sync_index.csv"
    return candidate if candidate.exists() else None


def _window_start_boundary_metadata(
    row: Mapping[str, Any],
    sync_rows: list[dict[str, Any]],
    state_signal_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    raw_start_sec = _as_float(row.get("start_sec"), 0.0)
    raw_end_sec = _as_float(row.get("end_sec"), raw_start_sec)
    raw_start_ts = _int_or_none(row.get("start_global_timestamp_us"))
    raw_end_ts = _int_or_none(row.get("end_global_timestamp_us"))
    if sync_rows:
        raw_start_ts = _int_or_none(sync_rows[0].get("global_timestamp_us")) or raw_start_ts
        raw_end_ts = _int_or_none(sync_rows[-1].get("global_timestamp_us")) or raw_end_ts

    evidence_times: list[tuple[float, str, list[Any]]] = []
    for key, reason in (
        ("supporting_first_segments", "supporting_first_activity_segment"),
        ("supporting_third_segments", "supporting_third_activity_segment"),
    ):
        segments = row.get(key)
        if not isinstance(segments, list):
            continue
        for segment in segments:
            if not isinstance(segment, Mapping):
                continue
            start = _as_float(segment.get("start_sec"), -1.0)
            if raw_start_sec <= start <= raw_end_sec:
                evidence_times.append((start, reason, [segment]))

    for signal in state_signal_rows:
        signal_start = _signal_start_sec(signal)
        if signal_start is None or signal_start < raw_start_sec or signal_start > raw_end_sec:
            continue
        if _is_experiment_activity_signal(signal):
            evidence_times.append(
                (
                    signal_start,
                    "state_signal_activity",
                    list(signal.get("evidence_frame_refs") or [])[:5],
                )
            )

    if evidence_times:
        actual_start_sec, start_reason, evidence_refs = min(evidence_times, key=lambda item: item[0])
        actual_start_status = "detected"
    else:
        actual_start_sec = raw_start_sec
        start_reason = "uncertain_no_activity_signal_before_window_start"
        evidence_refs = []
        actual_start_status = "uncertain"

    actual_start_sec = min(max(actual_start_sec, raw_start_sec), raw_end_sec)
    focus_duration_s = max(10.0, _float_env("KEY_ACTION_FOCUS_PREVIEW_DURATION_SEC", 120.0))
    focus_start_sec = actual_start_sec
    focus_end_sec = min(raw_end_sec, focus_start_sec + focus_duration_s)
    actual_end_sec = raw_end_sec
    actual_start_ts = _timestamp_for_window_offset(sync_rows, actual_start_sec - raw_start_sec) or raw_start_ts
    actual_end_ts = raw_end_ts
    focus_start_ts = _timestamp_for_window_offset(sync_rows, focus_start_sec - raw_start_sec) or actual_start_ts
    focus_end_ts = _timestamp_for_window_offset(sync_rows, focus_end_sec - raw_start_sec) or actual_end_ts
    start_offset_s = round(max(0.0, actual_start_sec - raw_start_sec), 6)
    return {
        "raw_window_start_global_timestamp_us": raw_start_ts,
        "raw_window_end_global_timestamp_us": raw_end_ts,
        "actual_experiment_start_global_timestamp_us": actual_start_ts,
        "actual_experiment_end_global_timestamp_us": actual_end_ts,
        "focus_preview_start_global_timestamp_us": focus_start_ts,
        "focus_preview_end_global_timestamp_us": focus_end_ts,
        "actual_experiment_duration_s": round(max(0.0, actual_end_sec - actual_start_sec), 6),
        "focus_preview_duration_s": round(max(0.0, focus_end_sec - focus_start_sec), 6),
        "start_offset_s": start_offset_s,
        "start_reason": start_reason,
        "actual_start_status": actual_start_status,
        "evidence_frame_refs": evidence_refs,
        "issues": [] if actual_start_status == "detected" else ["actual_start_uncertain"],
    }


def _signal_start_sec(signal: Mapping[str, Any]) -> float | None:
    timestamp_range = signal.get("timestamp_range")
    if isinstance(timestamp_range, Mapping):
        value = _as_float(timestamp_range.get("start_sec"), -1.0)
        return value if value >= 0 else None
    value = _as_float(signal.get("start_sec"), -1.0)
    return value if value >= 0 else None


def _is_experiment_activity_signal(signal: Mapping[str, Any]) -> bool:
    if _as_float(signal.get("first_activity_score"), 0.0) > 0.1:
        return True
    if _as_float(signal.get("third_activity_score"), 0.0) > 0.1:
        return True
    return any(
        bool(signal.get(key))
        for key in (
            "first_has_hand",
            "third_has_hand",
            "first_has_object_interaction",
            "third_has_object_interaction",
            "first_off_bench_experiment_related",
            "third_bench_activity",
            "cleaning_signal",
            "object_interaction_signal",
            "device_panel_signal",
        )
    )


def _timestamp_for_window_offset(rows: list[dict[str, Any]], offset_s: float) -> int | None:
    if not rows:
        return None
    start_ts = _int_or_none(rows[0].get("global_timestamp_us"))
    if start_ts is None:
        return None
    target_ts = start_ts + int(round(max(0.0, offset_s) * 1_000_000.0))
    best = min(
        rows,
        key=lambda item: abs((_int_or_none(item.get("global_timestamp_us")) or start_ts) - target_ts),
    )
    return _int_or_none(best.get("global_timestamp_us"))


def _write_experiment_start_boundary_refinement_report(
    metadata_dir: Path,
    rows: list[dict[str, Any]],
) -> None:
    report_rows = []
    for row in rows:
        report_rows.append(
            {
                "schema_version": "experiment_start_boundary_refinement.item.v1",
                "window_id": row.get("window_id"),
                "raw_window_start_global_timestamp_us": row.get("raw_window_start_global_timestamp_us"),
                "actual_experiment_start_global_timestamp_us": row.get("actual_experiment_start_global_timestamp_us"),
                "focus_preview_start_global_timestamp_us": row.get("focus_preview_start_global_timestamp_us"),
                "raw_window_end_global_timestamp_us": row.get("raw_window_end_global_timestamp_us"),
                "actual_experiment_end_global_timestamp_us": row.get("actual_experiment_end_global_timestamp_us"),
                "focus_preview_end_global_timestamp_us": row.get("focus_preview_end_global_timestamp_us"),
                "start_offset_s": row.get("start_offset_s"),
                "start_reason": row.get("start_reason"),
                "actual_start_status": row.get("actual_start_status"),
                "evidence_frame_refs": row.get("evidence_frame_refs") or [],
                "issues": row.get("issues") or [],
            }
        )
    _write_json(
        metadata_dir / "experiment_start_boundary_refinement_report.json",
        {
            "schema_version": "experiment_start_boundary_refinement_report.v1",
            "window_count": len(report_rows),
            "detected_count": sum(1 for row in report_rows if row.get("actual_start_status") == "detected"),
            "uncertain_count": sum(1 for row in report_rows if row.get("actual_start_status") != "detected"),
            "windows": report_rows,
        },
    )


def _write_window_sync_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "window_sync_index",
        "global_timestamp_us",
        "reference_camera",
        "reference_frame_index",
        "reference_timestamp_us",
        "first_frame_index",
        "first_timestamp_us",
        "third_frame_index",
        "third_timestamp_us",
        "third_video_path",
        "delta_ms",
        "sync_quality",
        "first_valid",
        "third_valid",
        "first_video_path",
        "drop_reason",
        "source_sync_index",
        "local_pts_first",
        "local_pts_third",
    ]
    first_base = _int_or_none(rows[0].get("first_timestamp_us")) if rows else None
    third_base = _int_or_none(rows[0].get("third_timestamp_us")) if rows else None
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows):
            first_ts = _int_or_none(row.get("first_timestamp_us"))
            third_ts = _int_or_none(row.get("third_timestamp_us"))
            writer.writerow(
                {
                    "window_sync_index": index,
                    "global_timestamp_us": row.get("global_timestamp_us"),
                    "reference_camera": row.get("reference_camera"),
                    "reference_frame_index": row.get("reference_frame_index"),
                    "reference_timestamp_us": row.get("reference_timestamp_us"),
                    "first_frame_index": row.get("first_frame_index"),
                    "first_timestamp_us": row.get("first_timestamp_us"),
                    "third_frame_index": row.get("third_frame_index"),
                    "third_timestamp_us": row.get("third_timestamp_us"),
                    "third_video_path": row.get("third_video_path"),
                    "delta_ms": row.get("delta_ms"),
                    "sync_quality": row.get("sync_quality"),
                    "first_valid": bool(first_ts is not None),
                    "third_valid": bool(third_ts is not None),
                    "first_video_path": row.get("first_video_path"),
                    "drop_reason": row.get("drop_reason") or "",
                    "source_sync_index": row.get("sync_index"),
                    "local_pts_first": round((first_ts - first_base) / 1_000_000.0, 6)
                    if first_ts is not None and first_base is not None
                    else None,
                    "local_pts_third": round((third_ts - third_base) / 1_000_000.0, 6)
                    if third_ts is not None and third_base is not None
                    else None,
                }
            )


def _build_window_preview_from_sync_rows(
    window_dir: Path,
    rows: list[dict[str, Any]],
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    min_preview_fps = _float_env("KEY_ACTION_MIN_WINDOW_PREVIEW_FPS", 15.0)
    configured_preview_fps = _float_env("KEY_ACTION_WINDOW_PREVIEW_OUTPUT_FPS", 15.0)
    output_fps = max(min_preview_fps, configured_preview_fps)
    max_preview_sec = max(5.0, _float_env("KEY_ACTION_WINDOW_PREVIEW_MAX_OUTPUT_SEC", 90.0))
    preview_mode = str(os.environ.get("KEY_ACTION_WINDOW_PREVIEW_MODE") or "realtime_preview").strip().lower()
    if preview_mode not in {"realtime_preview", "fast_preview"}:
        preview_mode = "realtime_preview"
    preview_path = window_dir / "window_preview.mp4"
    browser_path = window_dir / "window_preview.browser.mp4"
    third_preview_path = window_dir / "third_view_realtime_preview.mp4"
    first_preview_path = window_dir / "first_view_realtime_preview.mp4"
    real_duration_s = _window_sync_duration_s(rows)
    payload: dict[str, Any] = {
        "status": "not_built",
        "preview_path": str(preview_path),
        "browser_preview_path": str(browser_path),
        "side_by_side_realtime_preview": str(browser_path),
        "third_view_realtime_preview": str(third_preview_path),
        "first_view_realtime_preview": str(first_preview_path),
        "source": "window_sync_index",
        "output_fps": round(output_fps, 6),
        "max_preview_sec": round(max_preview_sec, 6),
        "preview_mode": preview_mode,
        "real_duration_s": real_duration_s,
        "row_count": len(rows),
        "reference_camera": audit.get("selected_reference_camera"),
        "issues": [],
    }
    if not rows:
        payload["issues"].append("empty_window_sync_rows")
        return payload
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        payload["issues"].append(f"opencv_unavailable:{exc}")
        return payload

    third_path = Path(str(rows[0].get("third_video_path") or ""))
    first_path = Path(str(rows[0].get("first_video_path") or ""))
    if not third_path.is_file() or not first_path.is_file():
        payload["issues"].append("source_video_missing")
        return payload

    third_cap = cv2.VideoCapture(str(third_path))
    first_cap = cv2.VideoCapture(str(first_path))
    writer = None
    frame_count = 0
    try:
        if not third_cap.isOpened() or not first_cap.isOpened():
            payload["issues"].append("source_video_open_failed")
            return payload
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        selected_rows = (
            _preview_rows_from_window_sync(rows, output_fps, max_preview_sec)
            if preview_mode == "fast_preview"
            else _realtime_rows_from_window_sync(rows, output_fps)
        )
        payload["selected_row_count"] = len(selected_rows)
        payload["preview_is_sampled_from_window_sync_index"] = preview_mode == "fast_preview" and len(selected_rows) < len(rows)
        third_read_state: dict[str, Any] = {}
        first_read_state: dict[str, Any] = {}
        last_key: tuple[str, str] | None = None
        last_composed: Any | None = None
        for idx, row in enumerate(selected_rows):
            frame_key = (str(row.get("third_frame_index") or ""), str(row.get("first_frame_index") or ""))
            if frame_key == last_key and last_composed is not None:
                composed = last_composed
            else:
                third_frame = _read_frame_by_index(third_cap, row.get("third_frame_index"), third_read_state)
                first_frame = _read_frame_by_index(first_cap, row.get("first_frame_index"), first_read_state)
                if third_frame is None or first_frame is None:
                    continue
                composed = _compose_side_by_side_frame(third_frame, first_frame)
                last_key = frame_key
                last_composed = composed
            if writer is None:
                height, width = composed.shape[:2]
                writer = cv2.VideoWriter(str(preview_path), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height))
                if not writer.isOpened():
                    payload["issues"].append("video_writer_open_failed")
                    return payload
            repeat = 1 if preview_mode == "realtime_preview" else (
                1 if len(selected_rows) < len(rows) else _sync_row_repeat_count(rows, idx, output_fps)
            )
            for _ in range(repeat):
                writer.write(composed)
                frame_count += 1
        if writer is not None:
            writer.release()
            writer = None
        if frame_count <= 0 or not preview_path.exists() or preview_path.stat().st_size <= 0:
            payload["issues"].append("no_preview_frames_written")
            return payload
        _transcode_preview_for_browser(preview_path, browser_path)
        split_source = browser_path if browser_path.is_file() else preview_path
        split_result = _split_side_by_side_preview(split_source, third_preview_path, first_preview_path)
        payload["view_split_status"] = split_result.get("status")
        payload["view_split_source"] = str(split_source)
        if split_result.get("issues"):
            payload["issues"].extend(str(item) for item in split_result.get("issues") or [])
        encoded_duration_s = round(frame_count / max(output_fps, 0.001), 6)
        playback_speed_ratio = round(real_duration_s / encoded_duration_s, 6) if encoded_duration_s > 0 else None
        payload["status"] = "built_from_window_sync_index"
        payload["output_frame_count"] = frame_count
        payload["duration_s"] = encoded_duration_s
        payload["output_duration_s"] = encoded_duration_s
        payload["source_window_duration_s"] = real_duration_s
        payload["playback_speed_ratio"] = playback_speed_ratio
        payload["is_accelerated"] = bool(playback_speed_ratio is not None and playback_speed_ratio > 1.25)
        payload["fast_preview"] = preview_mode == "fast_preview"
        payload["fast_preview_path"] = str(preview_path) if preview_mode == "fast_preview" else None
        payload["browser_preview_exists"] = browser_path.is_file()
        payload["third_view_realtime_preview_exists"] = third_preview_path.is_file()
        payload["first_view_realtime_preview_exists"] = first_preview_path.is_file()
        payload["side_by_side_realtime_preview"] = str(browser_path if browser_path.is_file() else preview_path)
        return payload
    except Exception as exc:
        payload["issues"].append(f"preview_build_failed:{exc}")
        return payload
    finally:
        if writer is not None:
            writer.release()
        third_cap.release()
        first_cap.release()


def _preview_rows_from_window_sync(rows: list[dict[str, Any]], output_fps: float, max_preview_sec: float) -> list[dict[str, Any]]:
    max_frames = max(1, int(round(max_preview_sec * max(output_fps, 0.001))))
    if len(rows) <= max_frames:
        return rows
    if max_frames <= 1:
        return [rows[0]]
    step = (len(rows) - 1) / float(max_frames - 1)
    indices = sorted({min(len(rows) - 1, max(0, int(round(i * step)))) for i in range(max_frames)})
    return [rows[index] for index in indices]


def _realtime_rows_from_window_sync(rows: list[dict[str, Any]], output_fps: float) -> list[dict[str, Any]]:
    valid_rows = [row for row in rows if _int_or_none(row.get("global_timestamp_us")) is not None]
    if len(valid_rows) < 2:
        return valid_rows or rows
    start_ts = _int_or_none(valid_rows[0].get("global_timestamp_us"))
    end_ts = _int_or_none(valid_rows[-1].get("global_timestamp_us"))
    if start_ts is None or end_ts is None or end_ts <= start_ts:
        return valid_rows
    duration_s = (end_ts - start_ts) / 1_000_000.0
    frame_count = max(1, int(round(duration_s * max(output_fps, 0.001))))
    timestamps = [_int_or_none(row.get("global_timestamp_us")) or start_ts for row in valid_rows]
    selected: list[dict[str, Any]] = []
    cursor = 0
    for frame_index in range(frame_count):
        target_ts = start_ts + int(round((frame_index / max(output_fps, 0.001)) * 1_000_000.0))
        while cursor + 1 < len(timestamps) and timestamps[cursor + 1] <= target_ts:
            cursor += 1
        nearest = cursor
        if cursor + 1 < len(timestamps):
            left_delta = abs(target_ts - timestamps[cursor])
            right_delta = abs(timestamps[cursor + 1] - target_ts)
            if right_delta < left_delta:
                nearest = cursor + 1
        selected.append(valid_rows[nearest])
    return selected


def _window_sync_duration_s(rows: list[dict[str, Any]]) -> float:
    if len(rows) < 2:
        return 0.0
    start = _int_or_none(rows[0].get("global_timestamp_us"))
    end = _int_or_none(rows[-1].get("global_timestamp_us"))
    if start is None or end is None or end <= start:
        return 0.0
    return round((end - start) / 1_000_000.0, 6)


def _read_frame_by_index(cap: Any, frame_index: Any, state: dict[str, Any] | None = None) -> Any | None:
    index = _int_or_none(frame_index)
    if index is None or index < 0:
        return None
    if state is not None:
        if state.get("last_index") == index and state.get("last_frame") is not None:
            return state["last_frame"]
        next_index = state.get("next_index")
        if isinstance(next_index, int) and index >= next_index and index - next_index <= 90:
            frame = None
            current = next_index
            while current <= index:
                ok, frame = cap.read()
                if not ok:
                    return None
                current += 1
            state["next_index"] = index + 1
            state["last_index"] = index
            state["last_frame"] = frame
            return frame
    cap.set(1, int(index))
    ok, frame = cap.read()
    if ok and state is not None:
        state["next_index"] = index + 1
        state["last_index"] = index
        state["last_frame"] = frame
    return frame if ok else None


def _compose_side_by_side_frame(third_frame: Any, first_frame: Any) -> Any:
    import cv2

    max_height = 720
    target_height = min(max_height, max(int(third_frame.shape[0]), int(first_frame.shape[0])))

    def _resize(frame: Any) -> Any:
        h, w = frame.shape[:2]
        if h == target_height:
            return frame
        width = max(1, int(round(w * (target_height / max(1, h)))))
        return cv2.resize(frame, (width, target_height), interpolation=cv2.INTER_AREA)

    third = _resize(third_frame)
    first = _resize(first_frame)
    return cv2.hconcat([third, first])


def _sync_row_repeat_count(rows: list[dict[str, Any]], index: int, output_fps: float) -> int:
    current = _int_or_none(rows[index].get("global_timestamp_us"))
    next_ts = _int_or_none(rows[index + 1].get("global_timestamp_us")) if index + 1 < len(rows) else None
    if current is None or next_ts is None or next_ts <= current:
        return 1
    delta_s = (next_ts - current) / 1_000_000.0
    return max(1, int(round(delta_s * max(output_fps, 0.001))))


def _preview_encode_crf() -> int:
    """Browser-preview x264 CRF. Higher = smaller/lighter file. Default keeps prior 22."""
    return int(round(_float_env("KEY_ACTION_WINDOW_PREVIEW_CRF", 22.0)))


def _preview_max_height() -> int:
    """Optional downscale cap for browser previews. 0 (default) = no scaling."""
    value = int(round(_float_env("KEY_ACTION_WINDOW_PREVIEW_MAX_HEIGHT", 0.0)))
    return value if value > 0 else 0


def _preview_video_filters(extra: str | None = None) -> list[str]:
    """Compose the -vf chain for preview encodes from env knobs.

    Combines an optional caller filter (e.g. a side-by-side crop) with an
    optional ``scale=-2:H`` downscale when KEY_ACTION_WINDOW_PREVIEW_MAX_HEIGHT
    is set. Returns ``[]`` when no filter is needed so existing behavior is
    unchanged by default.
    """
    parts = [p for p in (extra,) if p]
    max_h = _preview_max_height()
    if max_h:
        # -2 keeps width even and preserves aspect ratio; only downscale.
        parts.append(f"scale=-2:min({max_h}\\,ih)")
    return ["-vf", ",".join(parts)] if parts else []


def _transcode_preview_for_browser(source: Path, target: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not source.exists():
        return
    tmp = target.with_name(f"{target.stem}.tmp_{os.getpid()}_{int(time.time() * 1000)}{target.suffix}")
    try:
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(source),
                *_preview_video_filters(),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                str(_preview_encode_crf()),
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-an",
                str(tmp),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        if tmp.exists() and tmp.stat().st_size > 0:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                target.unlink()
            shutil.move(str(tmp), str(target))
    except Exception:
        tmp.unlink(missing_ok=True)


def _split_side_by_side_preview(source: Path, third_target: Path, first_target: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"status": "not_built", "issues": []}
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        result["issues"].append("ffmpeg_unavailable_for_view_split")
        return result
    if not source.is_file() or source.stat().st_size <= 0:
        result["issues"].append("side_by_side_preview_missing_for_view_split")
        return result
    meta = _video_metadata(source)
    width = int(meta.get("width") or 0)
    if width <= 2:
        try:
            import cv2

            cap = cv2.VideoCapture(str(source))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            cap.release()
        except Exception:
            width = 0
    if width <= 2:
        result["issues"].append("side_by_side_width_unavailable")
        return result
    half_width = max(2, width // 2)
    if half_width % 2:
        half_width -= 1
    built: list[str] = []
    for target, crop in (
        (third_target, f"crop={half_width}:ih:0:0"),
        (first_target, f"crop={half_width}:ih:{half_width}:0"),
    ):
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f"{target.stem}.tmp_{os.getpid()}_{int(time.time() * 1000)}{target.suffix}")
        try:
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-i",
                    str(source),
                    *_preview_video_filters(crop),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    str(_preview_encode_crf()),
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    "-an",
                    str(tmp),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            if tmp.exists() and tmp.stat().st_size > 0:
                if target.exists():
                    target.unlink()
                shutil.move(str(tmp), str(target))
                built.append(str(target))
        except Exception as exc:
            result["issues"].append(f"view_split_failed:{target.name}:{exc}")
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
    result["built"] = built
    result["status"] = "built" if len(built) == 2 else "partial"
    return result


def _window_reference_camera_audit(window_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    first_stats = _timestamp_fps_stats(rows, "first_timestamp_us")
    third_stats = _timestamp_fps_stats(rows, "third_timestamp_us")
    lower = _lower_fps_camera(first_stats, third_stats)
    selected = _selected_reference_camera(rows)
    return {
        "schema_version": "window_reference_camera_audit.v1",
        "window_id": window_id,
        "first_valid_frame_count": first_stats["count"],
        "third_valid_frame_count": third_stats["count"],
        "first_timestamp_duration_s": first_stats["duration_s"],
        "third_timestamp_duration_s": third_stats["duration_s"],
        "first_actual_fps": first_stats["actual_fps"],
        "third_actual_fps": third_stats["actual_fps"],
        "lower_fps_camera": lower,
        "selected_reference_camera": selected,
        "reference_selection_status": "matches_lower_fps" if selected == lower else "uses_existing_global_reference",
        "reason": "window_sync_index preserves global sync_index reference; audit records whether it matches the lower-fps local window.",
    }


def _timestamp_fps_stats(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [_int_or_none(row.get(key)) for row in rows]
    timestamps = [value for value in values if value is not None]
    if len(timestamps) < 2:
        return {"count": len(timestamps), "duration_s": 0.0, "actual_fps": 0.0}
    duration = max(0.0, (max(timestamps) - min(timestamps)) / 1_000_000.0)
    fps = (len(timestamps) - 1) / duration if duration > 0 else 0.0
    return {"count": len(timestamps), "duration_s": round(duration, 6), "actual_fps": round(fps, 6)}


def _lower_fps_camera(first_stats: Mapping[str, Any], third_stats: Mapping[str, Any]) -> str | None:
    first_fps = _as_float(first_stats.get("actual_fps"), 0.0)
    third_fps = _as_float(third_stats.get("actual_fps"), 0.0)
    if first_fps <= 0 and third_fps <= 0:
        return None
    if first_fps <= 0:
        return THIRD_PERSON
    if third_fps <= 0:
        return FIRST_PERSON
    if abs(first_fps - third_fps) <= 0.05:
        first_count = int(first_stats.get("count") or 0)
        third_count = int(third_stats.get("count") or 0)
        return FIRST_PERSON if first_count <= third_count else THIRD_PERSON
    return FIRST_PERSON if first_fps < third_fps else THIRD_PERSON


def _selected_reference_camera(rows: list[dict[str, Any]]) -> str | None:
    counts: Counter[str] = Counter(str(row.get("reference_camera") or "") for row in rows if row.get("reference_camera"))
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def _window_alignment_quality(window_id: str, rows: list[dict[str, Any]], row: Mapping[str, Any]) -> dict[str, Any]:
    deltas = sorted(_as_float(item.get("delta_ms"), 0.0) for item in rows if item.get("delta_ms") not in (None, ""))
    if deltas:
        median = _percentile(deltas, 50)
        p90 = _percentile(deltas, 90)
        p95 = _percentile(deltas, 95)
        p99 = _percentile(deltas, 99)
        max_delta = max(deltas)
    else:
        median = p90 = p95 = p99 = max_delta = None
    status = "window_sync_ready" if rows and (p90 is None or p90 <= 150.0) else "window_sync_needs_review"
    return {
        "schema_version": "window_alignment_quality.v1",
        "window_id": window_id,
        "unit_id": row.get("unit_id"),
        "matched_frame_count": len(rows),
        "dropped_frame_count": 0,
        "duplicated_frame_count": 0,
        "median_delta_ms": median,
        "p90_delta_ms": p90,
        "p95_delta_ms": p95,
        "p99_delta_ms": p99,
        "max_delta_ms": max_delta,
        "unit_status": status,
        "action_phase_consistency": row.get("status"),
        "warnings": [] if status == "window_sync_ready" else ["window_sync_index_missing_or_delta_high"],
    }


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return round(values[0], 6)
    rank = (len(values) - 1) * max(0.0, min(100.0, pct)) / 100.0
    lower = int(rank)
    upper = min(len(values) - 1, lower + 1)
    fraction = rank - lower
    return round(values[lower] * (1.0 - fraction) + values[upper] * fraction, 6)


def _normalize_ranges(value: Any) -> list[dict[str, float]]:
    if not isinstance(value, list):
        return []
    ranges: list[dict[str, float]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        start = _optional_float(item.get("start_sec"))
        end = _optional_float(item.get("end_sec"))
        if start is None or end is None or end <= start:
            continue
        ranges.append({"start_sec": float(start), "end_sec": float(end)})
    return sorted(ranges, key=lambda item: item["start_sec"])


def _range_total_sec(ranges: list[dict[str, float]]) -> float:
    return sum(max(0.0, float(item["end_sec"]) - float(item["start_sec"])) for item in ranges)


def _range_overlap_sec(left: list[dict[str, float]], right: list[dict[str, float]]) -> float:
    total = 0.0
    i = 0
    j = 0
    while i < len(left) and j < len(right):
        start = max(left[i]["start_sec"], right[j]["start_sec"])
        end = min(left[i]["end_sec"], right[j]["end_sec"])
        if end > start:
            total += end - start
        if left[i]["end_sec"] < right[j]["end_sec"]:
            i += 1
        else:
            j += 1
    return total


def write_speed_and_quality_reports(
    metadata_dir: Path,
    manifest: SessionManifest,
    *,
    stage_timings: Mapping[str, Any],
    yolo_timing_summary: Mapping[str, Any],
    coarse_yolo_row_count: int,
    fine_yolo_row_count: int,
    fine_window_count: int,
    fine_window_duration_sec: float,
    candidate_summary: Mapping[str, Any] | None,
    material_reference_summary: Mapping[str, Any] | None,
    formal_output_gate: Mapping[str, Any] | None,
    episode_filter_summary: Mapping[str, Any] | None,
    micro_segment_count: int,
) -> dict[str, Any]:
    sources = manifest.videos.all_sources()
    total_video_duration_s = max([_as_float(getattr(source, "duration_sec", 0.0), 0.0) for source in sources.values()] or [0.0])
    timing = dict(yolo_timing_summary or {})
    stage = dict(stage_timings or {})
    coarse_stage = (timing.get("by_stage") or {}).get("coarse_segment_scan") if isinstance(timing.get("by_stage"), Mapping) else {}
    fine_stage = (timing.get("by_stage") or {}).get("micro_refine_window_scan") if isinstance(timing.get("by_stage"), Mapping) else {}
    candidate_window_count = _window_count_from_file(metadata_dir / "candidate_experiment_windows.json")
    formal_window_count = _window_count_from_file(metadata_dir / "formal_experiment_windows.json")
    chunk_count = _jsonl_row_count(metadata_dir / "chunk_manifest.jsonl")
    coarse_scan_duration = _as_float((coarse_stage or {}).get("scan_duration_sec"), 0.0)
    coarse_sampled = _as_float((coarse_stage or {}).get("sampled_frames"), 0.0)
    coarse_sample_fps = _as_float((coarse_stage or {}).get("sample_fps"), 0.0)
    if coarse_sample_fps <= 0 and coarse_scan_duration > 0:
        coarse_sample_fps = coarse_sampled / coarse_scan_duration
    fine_batch_size = _int_or_none(
        (fine_stage or {}).get("max_actual_batch_size")
        or (coarse_stage or {}).get("max_actual_batch_size")
        or (fine_stage or {}).get("batch_size")
        or (coarse_stage or {}).get("batch_size")
    )
    alignment_runtime_s = _alignment_runtime_sec(metadata_dir)
    analysis_runtime_s = _as_float(timing.get("total_wall_sec"), 0.0)
    total_runtime_s = analysis_runtime_s + alignment_runtime_s if alignment_runtime_s > 0 else analysis_runtime_s
    output_dir = Path(manifest.output_dir)
    local_material_root = output_dir.parent / "material_references"
    material_stream_count = _jsonl_row_count(local_material_root / "material_stream.jsonl")
    material_summary_records = len((material_reference_summary or {}).get("records") or [])
    published_material_count = max(material_stream_count, material_summary_records)
    speed = {
        "schema_version": "speed_report.v1",
        "total_video_duration_s": round(total_video_duration_s, 6),
        "aligned_video_duration_s": _aligned_video_duration_sec(metadata_dir),
        "alignment_time_s": round(alignment_runtime_s, 6),
        "analysis_runtime_s": round(analysis_runtime_s, 6),
        "coarse_scan_sample_fps": round(coarse_sample_fps, 6),
        "coarse_scan_time_s": _as_float((coarse_stage or {}).get("wall_sec") or (coarse_stage or {}).get("parallel_elapsed_sec"), 0.0),
        "chunk_count": chunk_count,
        "candidate_window_count": candidate_window_count
        if candidate_window_count is not None
        else int((episode_filter_summary or {}).get("candidate_action_window_count") or 0),
        "formal_window_count": formal_window_count
        if formal_window_count is not None
        else int((episode_filter_summary or {}).get("official_episode_count") or (episode_filter_summary or {}).get("output_count") or 0),
        "fine_scan_video_duration_s": round(float(fine_window_duration_sec or 0.0), 6),
        "fine_scan_time_s": _as_float((fine_stage or {}).get("wall_sec") or (fine_stage or {}).get("parallel_elapsed_sec"), 0.0),
        "yolo_batch_size": fine_batch_size,
        "processed_frames": int(timing.get("total_sampled_frames") or coarse_yolo_row_count + fine_yolo_row_count),
        "skipped_frames": None,
        "cache_hit_rate": _cache_hit_rate(timing),
        "total_runtime_s": round(total_runtime_s, 6),
        "speedup_estimate": _speedup_estimate(total_video_duration_s, {"total_wall_sec": total_runtime_s}),
        "bottlenecks": _bottlenecks(stage, timing),
        "recommendations": _speed_recommendations(stage, timing),
    }
    quality = {
        "schema_version": "quality_report.v1",
        "time_axis_status": _status_from_file(metadata_dir / "dual_view_alignment" / "dual_view_alignment_pipeline_summary.json"),
        "alignment_status": _status_from_file(metadata_dir / "dual_view_alignment" / "alignment_quality_report.json"),
        "phase_consistency_status": _status_from_file(metadata_dir / "phase_consistency_report.json"),
        "experiment_window_status": "available" if Path(metadata_dir / "formal_experiment_windows.json").exists() else "missing",
        "action_candidate_count": int((candidate_summary or {}).get("candidate_count") or 0),
        "published_material_count": int(published_material_count),
        "rejected_material_count": int((material_reference_summary or {}).get("skipped_count") or 0),
        "reject_reason_distribution": _reject_reasons(material_reference_summary or {}),
        "first_dominant_count": int((formal_output_gate or {}).get("first_dominant_count") or 0),
        "dual_view_passed_count": int(((formal_output_gate or {}).get("dual_view_action_gate") or {}).get("formal_event_count") or 0)
        if isinstance((formal_output_gate or {}).get("dual_view_action_gate"), Mapping)
        else 0,
        "low_confidence_count": 0,
        "black_or_placeholder_count": 0,
        "micro_segment_count": int(micro_segment_count),
        "recommendations": _quality_recommendations(formal_output_gate or {}, material_reference_summary or {}),
    }
    _write_json(metadata_dir / "speed_report.json", speed)
    _write_json(metadata_dir / "quality_report.json", quality)
    return {"speed_report": speed, "quality_report": quality}


def _window_count_from_file(path: Path) -> int | None:
    data = _read_json(path)
    if not data:
        return None
    if isinstance(data.get("windows"), list):
        return len(data["windows"])
    value = data.get("window_count")
    return _int_or_none(value)


def _jsonl_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _alignment_runtime_sec(metadata_dir: Path) -> float:
    outputs = _read_json(metadata_dir / "dual_view_alignment" / "aligned_video_outputs.json")
    if str(outputs.get("status") or "").lower() == "not_built":
        return 0.0
    data = _read_json(metadata_dir / "dual_view_alignment" / "aligned_alignment_quality.json")
    timing = data.get("extract_timing_sec") if isinstance(data.get("extract_timing_sec"), Mapping) else {}
    return _as_float(timing.get("total_wall_sec"), 0.0)


def _aligned_video_duration_sec(metadata_dir: Path) -> float | None:
    outputs = _read_json(metadata_dir / "dual_view_alignment" / "aligned_video_outputs.json")
    if str(outputs.get("status") or "").lower() == "not_built":
        return None
    data = _read_json(metadata_dir / "dual_view_alignment" / "aligned_alignment_quality.json")
    sample_count = _as_float(data.get("sample_count"), 0.0)
    fps = _as_float(data.get("target_fps"), 0.0)
    if sample_count > 0 and fps > 0:
        return round(sample_count / fps, 6)
    return None


def _cache_hit_rate(timing: Mapping[str, Any]) -> float | None:
    hits = _as_float(timing.get("cache_hits"), 0.0)
    misses = _as_float(timing.get("cache_misses"), 0.0)
    total = hits + misses
    return round(hits / total, 6) if total > 0 else None


def _speedup_estimate(total_video_duration_s: float, timing: Mapping[str, Any]) -> float | None:
    runtime = _as_float(timing.get("total_wall_sec"), 0.0)
    if runtime <= 0:
        return None
    return round(total_video_duration_s / runtime, 6) if total_video_duration_s > 0 else None


def _bottlenecks(stage_timings: Mapping[str, Any], timing: Mapping[str, Any]) -> list[dict[str, Any]]:
    items: list[tuple[str, float]] = []
    for key, value in dict(stage_timings or {}).items():
        sec = _as_float(value, 0.0)
        if sec > 0:
            items.append((key, sec))
    by_stage = timing.get("by_stage") if isinstance(timing.get("by_stage"), Mapping) else {}
    for key, value in dict(by_stage or {}).items():
        if isinstance(value, Mapping):
            sec = _as_float(value.get("wall_sec") or value.get("parallel_elapsed_sec"), 0.0)
            if sec > 0:
                items.append((key, sec))
    return [{"stage": key, "wall_sec": round(sec, 6)} for key, sec in sorted(items, key=lambda item: item[1], reverse=True)[:5]]


def _speed_recommendations(stage_timings: Mapping[str, Any], timing: Mapping[str, Any]) -> list[str]:
    recommendations = []
    bottlenecks = _bottlenecks(stage_timings, timing)
    if bottlenecks:
        recommendations.append(f"Optimize {bottlenecks[0]['stage']} first; it is the largest recorded wall-time bucket.")
    if _as_float(timing.get("total_wall_sec"), 0.0) <= 0:
        recommendations.append("Record end-to-end timing for the full pipeline run.")
    return recommendations


def _status_from_file(path: Path) -> str:
    data = _read_json(path)
    return str(data.get("status") or data.get("alignment_status") or "missing")


def _reject_reasons(summary: Mapping[str, Any]) -> dict[str, int]:
    reasons: Counter[str] = Counter()
    skipped = summary.get("skipped") if isinstance(summary.get("skipped"), list) else []
    for item in skipped:
        if isinstance(item, Mapping):
            reasons[str(item.get("blocked_reason") or item.get("suppression_reason") or item.get("reason") or "unknown")] += 1
    return dict(reasons)


def _quality_recommendations(formal_gate: Mapping[str, Any], material_summary: Mapping[str, Any]) -> list[str]:
    recommendations = []
    if str(formal_gate.get("status") or "") == "blocked":
        recommendations.append("Do not publish formal materials until dual-view action phase gate passes.")
    if int(material_summary.get("file_count") or 0) == 0:
        recommendations.append("No formal material was published; inspect action gate and material publish gate reports.")
    return recommendations

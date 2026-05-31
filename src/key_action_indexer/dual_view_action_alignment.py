from __future__ import annotations

import os
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping

from .family_merge import object_family as canonical_object_family
from .schemas import DetectedSegment, SessionManifest, VideoSource, read_jsonl, write_jsonl
from .semantic_alias import infer_action_type_from_metadata
from .time_alignment import local_sec_to_global_time


SCHEMA_VERSION = "dual_view_action_alignment.v1"

HAND_LABELS = {"hand", "gloved_hand"}
CONTEXT_ONLY_LABELS = {"lab_coat", "ppe_storage", "tube_rack"}
DISQUALIFYING_VIEW_QUALITY_FLAG_MARKERS = ("single_frame", "no_explicit")
ACTION_PHASES = {"prep_glove_or_sleeve", "bench_operation", "off_bench_or_invalid", "unknown"}
TEMPORAL_FRAME_PHASES = {"start", "middle", "mid", "end"}
ACTION_PHASE_FIELDS = (
    "action_phase",
    "semantic_action_phase",
    "event_action_phase",
    "interaction_phase",
    "sop_phase",
    "phase_label",
    "phase",
)
ACTION_PHASE_TEXT_FIELDS = (
    "action_type",
    "semantic_action_type",
    "interaction_type",
    "activity_label",
    "event_type",
    "event_label",
    "description",
    "caption",
    "vlm_description",
    "text",
    "notes",
)
PREP_PHASE_LABELS = {"glove", "gloves", "sleeve", "lab_coat", "ppe_storage", "ppe", "gown", "cuff"}
OFF_BENCH_PHASE_LABELS = {"off_bench", "invalid", "background", "out_of_scope", "non_action", "no_action"}
BENCH_OBJECT_FAMILIES = {
    "reagent_bottle_family",
    "sample_bottle_family",
    "pipette_family",
    "balance_family",
    "paper_family",
    "spatula_family",
    "container_family",
    "equipment_family",
}

ACTION_DISPLAY_NAMES = {
    "reagent_bottle_operation": "\u624b\u90e8\u4e0e\u8bd5\u5242\u74f6\u64cd\u4f5c",
    "sample_bottle_operation": "\u624b\u90e8\u4e0e\u8bd5\u5242\u74f6\u64cd\u4f5c",
    "pipette_operation": "\u624b\u90e8\u4e0e\u79fb\u6db2\u67aa\u64cd\u4f5c",
    "balance_operation": "\u5929\u5e73\u8bbe\u5907\u9762\u677f\u64cd\u4f5c",
    "weighing_paper_operation": "\u624b\u90e8\u4e0e\u79f0\u91cf\u7eb8\u64cd\u4f5c",
    "spatula_operation": "\u624b\u90e8\u4e0e\u836f\u5319\u64cd\u4f5c",
    "container_operation": "\u624b\u90e8\u4e0e\u5bb9\u5668\u64cd\u4f5c",
    "equipment_operation": "\u8bbe\u5907\u64cd\u4f5c",
    "hand_object_operation": "\u624b\u90e8\u4e0e\u7269\u4f53\u64cd\u4f5c",
}

OBJECT_DISPLAY_NAMES = {
    "reagent_bottle_family": "\u8bd5\u5242\u74f6",
    "sample_bottle_family": "\u6837\u54c1\u74f6",
    "pipette_family": "\u79fb\u6db2\u67aa",
    "balance_family": "\u5929\u5e73",
    "paper_family": "\u79f0\u91cf\u7eb8",
    "spatula_family": "\u836f\u5319",
    "container_family": "\u5bb9\u5668",
    "equipment_family": "\u8bbe\u5907",
    "object_family": "\u7269\u4f53",
}


@dataclass(frozen=True)
class ViewActionEvidence:
    evidence_id: str
    session_id: str
    view: str
    session_start_sec: float
    session_end_sec: float
    duration_sec: float
    peak_session_sec: float
    action_family: str
    action_display_name: str
    object_family: str
    object_display_name: str
    raw_yolo_labels: list[str]
    hand_count: int
    object_count: int
    row_count: int
    interaction_row_count: int
    max_interaction_score: float
    avg_interaction_score: float
    evidence_density: float
    evidence_kind: str
    source_row_indices: list[int]
    source_frame_indices: list[int]
    quality_flags: list[str]
    action_phase: str = "bench_operation"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DualViewActionEvent:
    dual_event_id: str
    session_id: str
    session_start_sec: float
    session_end_sec: float
    duration_sec: float
    first_evidence_id: str
    third_evidence_id: str
    action_family: str
    action_display_name: str
    object_family: str
    object_display_name: str
    canonical_action_type: str
    canonical_object_family: str
    overlap_sec: float
    center_delta_sec: float
    peak_delta_sec: float
    temporal_overlap_score: float
    action_match_score: float
    object_match_score: float
    evidence_symmetry_score: float
    alignment_score: float
    status: str
    decision_trace: list[str]
    action_phase: str = "bench_operation"
    action_phase_match_score: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_dual_view_action_alignment(
    manifest: SessionManifest,
    rows: list[Mapping[str, Any]],
    *,
    output_dir: str | Path | None = None,
    min_alignment_score: float | None = None,
    max_center_delta_sec: float | None = None,
    group_gap_sec: float | None = None,
    episode_merge_gap_sec: float | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Build strict dual-view action events from frame-level YOLO evidence.

    A single-view row never becomes a formal action event by itself. It is
    either paired with compatible evidence from the other view or written as an
    unmatched debug record.
    """

    metadata_dir = Path(output_dir) if output_dir is not None else None
    if metadata_dir is not None and metadata_dir.name != "metadata":
        metadata_dir = metadata_dir / "metadata"
    threshold = float(min_alignment_score if min_alignment_score is not None else _env_float("KEY_ACTION_DUAL_VIEW_ALIGNMENT_SCORE", 0.72))
    max_delta = float(max_center_delta_sec if max_center_delta_sec is not None else _env_float("KEY_ACTION_DUAL_VIEW_MAX_CENTER_DELTA_SEC", 2.0))
    gap = float(group_gap_sec if group_gap_sec is not None else _env_float("KEY_ACTION_VIEW_ACTION_GROUP_GAP_SEC", 1.5))
    episode_gap = float(episode_merge_gap_sec if episode_merge_gap_sec is not None else _env_float("KEY_ACTION_DUAL_EVENT_EPISODE_MERGE_GAP_SEC", 45.0))

    view_evidence = build_view_action_evidence(manifest, rows, group_gap_sec=gap)
    first = [item for item in view_evidence if item.view == "first_person"]
    third = [item for item in view_evidence if item.view == "third_person"]
    dual_events, unmatched = match_dual_view_action_events(
        manifest,
        first,
        third,
        min_alignment_score=threshold,
        max_center_delta_sec=max_delta,
    )
    episodes = detected_segments_from_dual_events(manifest, dual_events, episode_merge_gap_sec=episode_gap)
    diagnostics = _alignment_diagnostics(view_evidence, dual_events, unmatched)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "session_id": manifest.session_id,
        "strict": bool(strict),
        "min_alignment_score": threshold,
        "max_center_delta_sec": max_delta,
        "min_view_action_row_count": _min_view_action_row_count(),
        "min_view_action_interaction_row_count": _min_view_action_interaction_row_count(),
        "min_view_action_strong_score": _min_view_action_strong_score(),
        "min_temporal_overlap_score": _min_temporal_overlap_score(),
        "min_temporal_overlap_sec": _min_temporal_overlap_sec(),
        "max_peak_delta_sec": _max_peak_delta_sec(max_delta),
        "require_peak_alignment": _require_peak_alignment(),
        "require_temporal_overlap": _require_temporal_overlap(),
        "require_action_phase_match": True,
        "allow_sparse_paired_view_evidence": _allow_sparse_paired_view_evidence(),
        "view_action_group_gap_sec": gap,
        "episode_merge_gap_sec": episode_gap,
        "view_action_evidence_count": len(view_evidence),
        "first_view_action_evidence_count": len(first),
        "third_view_action_evidence_count": len(third),
        "dual_view_action_event_count": len(dual_events),
        "unmatched_view_evidence_count": len(unmatched),
        "unmatched_reason_counts": diagnostics["unmatched_reason_counts"],
        "phase_mismatch": int(diagnostics["unmatched_reason_counts"].get("phase_mismatch", 0)),
        "phase_unknown": int(diagnostics["unmatched_reason_counts"].get("phase_unknown", 0)),
        "view_alignment_diagnostics": diagnostics,
        "episode_count": len(episodes),
        "formal_results_allowed": bool(dual_events) if strict else True,
        "decision": "dual_view_action_aligned" if dual_events else "no_dual_view_action_events",
    }
    if metadata_dir is not None:
        metadata_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(metadata_dir / "view_action_evidence.jsonl", [item.to_dict() for item in view_evidence])
        write_jsonl(metadata_dir / "dual_view_action_events.jsonl", [item.to_dict() for item in dual_events])
        write_jsonl(metadata_dir / "unmatched_view_evidence.jsonl", unmatched)
        write_jsonl(metadata_dir / "dual_view_aligned_experiment_episodes.jsonl", [_segment_to_row(item, dual_events) for item in episodes])
        (metadata_dir / "dual_view_action_alignment_summary.json").write_text(
            __import__("json").dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return {
        "summary": summary,
        "view_action_evidence": [item.to_dict() for item in view_evidence],
        "dual_view_action_events": [item.to_dict() for item in dual_events],
        "unmatched_view_evidence": unmatched,
        "episodes": episodes,
    }


def build_dual_view_action_events(
    output_dir: str | Path,
    *,
    yolo_frame_rows: list[Mapping[str, Any]] | None = None,
    yolo_micro_frame_rows: list[Mapping[str, Any]] | None = None,
    micro_segments: list[Mapping[str, Any]] | None = None,
    manifest: SessionManifest | None = None,
    min_alignment_score: float | None = None,
    max_center_delta_sec: float | None = None,
) -> dict[str, Any]:
    """Build and persist formal dual-view action event artifacts.

    This wrapper keeps the formal event contract explicit: a single-view
    evidence item is written as unmatched debug evidence and is not promoted.
    """

    session_dir = Path(output_dir)
    metadata_dir = session_dir / "metadata"
    cv_outputs_dir = session_dir / "cv_outputs"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    manifest = manifest or _default_manifest(session_dir)
    if yolo_frame_rows is None:
        yolo_frame_rows = _read_jsonl_if_exists(cv_outputs_dir / "yolo_frame_rows.jsonl")
    if yolo_micro_frame_rows is None:
        yolo_micro_frame_rows = _read_jsonl_if_exists(cv_outputs_dir / "yolo_micro_frame_rows.jsonl")
    if micro_segments is None:
        micro_segments = _read_jsonl_if_exists(metadata_dir / "micro_segments.jsonl")

    rows = [dict(row) for row in yolo_frame_rows or []]
    rows.extend(dict(row) for row in yolo_micro_frame_rows or [])
    rows.extend(_rows_from_micro_segments(micro_segments or []))
    alignment = build_dual_view_action_alignment(
        manifest,
        rows,
        output_dir=None,
        min_alignment_score=min_alignment_score,
        max_center_delta_sec=max_center_delta_sec,
    )
    evidence = [_compat_evidence_row(dict(row), micro_segments or []) for row in alignment["view_action_evidence"]]
    evidence_by_id = {str(row.get("evidence_id")): row for row in evidence}
    event_rows = [
        _formal_event_row(dict(event), evidence_by_id, micro_segments or [])
        for event in alignment["dual_view_action_events"]
    ]
    unmatched_rows = [
        _unmatched_event_row(dict(row), evidence_by_id)
        for row in alignment["unmatched_view_evidence"]
    ]
    episode_rows = [
        _segment_to_row(segment, [DualViewActionEvent(**event) for event in alignment["dual_view_action_events"]])
        for segment in alignment["episodes"]
    ]
    write_jsonl(metadata_dir / "view_action_evidence.jsonl", evidence)
    write_jsonl(metadata_dir / "dual_view_action_events.jsonl", event_rows)
    write_jsonl(metadata_dir / "unmatched_view_evidence.jsonl", unmatched_rows)
    write_jsonl(metadata_dir / "dual_view_aligned_experiment_episodes.jsonl", episode_rows)
    summary = dict(alignment["summary"])
    summary.update(
        {
            "schema_version": "key_action_dual_view_action_alignment_summary.v1",
            "available": True,
            "artifacts": {
                "view_action_evidence": str(metadata_dir / "view_action_evidence.jsonl"),
                "dual_view_action_events": str(metadata_dir / "dual_view_action_events.jsonl"),
                "unmatched_view_evidence": str(metadata_dir / "unmatched_view_evidence.jsonl"),
                "dual_view_aligned_experiment_episodes": str(metadata_dir / "dual_view_aligned_experiment_episodes.jsonl"),
            },
        }
    )
    (metadata_dir / "dual_view_action_alignment_summary.json").write_text(
        __import__("json").dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def build_view_action_evidence(
    manifest: SessionManifest,
    rows: list[Mapping[str, Any]],
    *,
    group_gap_sec: float = 1.5,
) -> list[ViewActionEvidence]:
    candidates: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        item = _candidate_from_row(manifest.session_id, row, index)
        if item is not None:
            candidates.append(item)
    candidates.sort(key=lambda item: (item["view"], item["action_family"], item["object_family"], item["action_phase"], item["time_sec"]))
    grouped: list[ViewActionEvidence] = []
    active: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for item in candidates:
        key = (item["view"], item["action_family"], item["object_family"], item["action_phase"])
        bucket = active.get(key)
        if bucket and float(item["time_sec"]) - float(bucket[-1]["time_sec"]) > group_gap_sec:
            grouped.append(_evidence_from_bucket(manifest.session_id, len(grouped) + 1, key, bucket))
            bucket = []
        if bucket is None:
            bucket = []
        bucket.append(item)
        active[key] = bucket
    for key, bucket in active.items():
        if bucket:
            grouped.append(_evidence_from_bucket(manifest.session_id, len(grouped) + 1, key, bucket))
    grouped.sort(key=lambda item: (item.session_start_sec, item.view, item.action_family, item.object_family, item.action_phase))
    return [
        ViewActionEvidence(**{**item.to_dict(), "evidence_id": f"vae_{idx:06d}"})
        for idx, item in enumerate(grouped, start=1)
    ]


def match_dual_view_action_events(
    manifest: SessionManifest,
    first: list[ViewActionEvidence],
    third: list[ViewActionEvidence],
    *,
    min_alignment_score: float = 0.72,
    max_center_delta_sec: float = 2.0,
) -> tuple[list[DualViewActionEvent], list[dict[str, Any]]]:
    eligible_first, weak_first = _split_pairable_view_evidence(first)
    eligible_third, weak_third = _split_pairable_view_evidence(third)
    pairs: list[tuple[float, ViewActionEvidence, ViewActionEvidence, dict[str, float]]] = []
    for left in eligible_first:
        for right in eligible_third:
            scores = _pair_scores(left, right, max_center_delta_sec=max_center_delta_sec)
            if _pair_gate_rejection_reason(left, right, scores, max_center_delta_sec=max_center_delta_sec):
                continue
            temporal_score = _temporal_score_for_pair(scores, max_center_delta_sec=max_center_delta_sec)
            score = (
                0.35 * temporal_score
                + 0.25 * scores["action_match_score"]
                + 0.20 * scores["object_match_score"]
                + 0.10 * min(left.evidence_density, right.evidence_density)
                + 0.10 * scores["evidence_symmetry_score"]
            )
            if score >= min_alignment_score:
                pairs.append((score, left, right, scores))
    pairs.sort(key=lambda item: item[0], reverse=True)
    used_first: set[str] = set()
    used_third: set[str] = set()
    events: list[DualViewActionEvent] = []
    for score, left, right, scores in pairs:
        if left.evidence_id in used_first or right.evidence_id in used_third:
            continue
        start = max(left.session_start_sec, right.session_start_sec)
        end = min(left.session_end_sec, right.session_end_sec)
        if end <= start:
            start = min(left.session_start_sec, right.session_start_sec)
            end = max(left.session_end_sec, right.session_end_sec)
        action_family = left.action_family if left.action_family == right.action_family else _prefer_action_family(left, right)
        object_family = left.object_family if left.object_family == right.object_family else _prefer_object_family(left, right)
        canonical_action = _canonical_action_type_for_family(action_family)
        canonical_object = _canonical_object_family_for_family(object_family)
        action_phase = _pair_action_phase(left, right)
        decision_trace = [
            "first_and_third_strong_view_evidence_present",
            "canonical_action_and_object_match",
            "action_phase_match",
            f"action_phase={action_phase}",
            _temporal_gate_trace(scores, max_center_delta_sec=max_center_delta_sec),
            f"canonical_action_type={canonical_action}",
            f"canonical_object_family={canonical_object}",
            f"overlap_score={scores['temporal_overlap_score']:.3f}",
            f"center_delta_sec={scores['center_delta_sec']:.3f}",
            f"overlap_sec={scores['overlap_sec']:.3f}",
            f"peak_delta_sec={scores['peak_delta_sec']:.3f}",
            f"min_alignment_score={min_alignment_score:.3f}",
            f"score={score:.3f}",
        ]
        if _is_sparse_pairable_view_evidence(left) or _is_sparse_pairable_view_evidence(right):
            decision_trace.append(
                "dual_view_sparse_pairing=allowed_only_because_both_views_have_explicit_temporally_aligned_action_evidence"
            )
        events.append(
            DualViewActionEvent(
                dual_event_id=f"dual_event_{len(events) + 1:06d}",
                session_id=manifest.session_id,
                session_start_sec=round(float(start), 6),
                session_end_sec=round(float(end), 6),
                duration_sec=round(max(0.1, float(end) - float(start)), 6),
                first_evidence_id=left.evidence_id,
                third_evidence_id=right.evidence_id,
                action_family=action_family,
                action_display_name=ACTION_DISPLAY_NAMES.get(action_family, ACTION_DISPLAY_NAMES["hand_object_operation"]),
                object_family=object_family,
                object_display_name=OBJECT_DISPLAY_NAMES.get(object_family, OBJECT_DISPLAY_NAMES["object_family"]),
                canonical_action_type=canonical_action,
                canonical_object_family=canonical_object,
                overlap_sec=round(scores["overlap_sec"], 6),
                center_delta_sec=round(scores["center_delta_sec"], 6),
                peak_delta_sec=round(scores["peak_delta_sec"], 6),
                temporal_overlap_score=round(scores["temporal_overlap_score"], 6),
                action_match_score=round(scores["action_match_score"], 6),
                object_match_score=round(scores["object_match_score"], 6),
                evidence_symmetry_score=round(scores["evidence_symmetry_score"], 6),
                alignment_score=round(float(score), 6),
                status="confirmed",
                decision_trace=decision_trace,
                action_phase=action_phase,
                action_phase_match_score=round(scores["action_phase_match_score"], 6),
            )
        )
        used_first.add(left.evidence_id)
        used_third.add(right.evidence_id)
    unmatched = []
    for item, reason in [*weak_first, *weak_third]:
        unmatched.append(_unmatched_view_evidence_row(item, reason))
    for item in [*eligible_first, *eligible_third]:
        if item.evidence_id in used_first or item.evidence_id in used_third:
            continue
        opposite_weak = weak_third if item.view == "first_person" else weak_first
        reason, details = _unmatched_reason_for_evidence(
            item,
            eligible_third if item.view == "first_person" else eligible_first,
            max_center_delta_sec=max_center_delta_sec,
            min_alignment_score=min_alignment_score,
        )
        if reason == "single_view_candidate" and opposite_weak:
            reason = "weak_view_evidence"
            details = {
                "opposite_view_weak_evidence_ids": [other.evidence_id for other, _reason in opposite_weak],
                "opposite_view_weak_reasons": {
                    other.evidence_id: _weak_view_evidence_reasons(other)
                    for other, _reason in opposite_weak
                },
            }
        unmatched.append(_unmatched_view_evidence_row(item, reason, **details))
    return events, unmatched


def detected_segments_from_dual_events(
    manifest: SessionManifest,
    events: list[DualViewActionEvent],
    *,
    episode_merge_gap_sec: float = 45.0,
    pre_roll_sec: float = 1.0,
    post_roll_sec: float = 1.0,
) -> list[DetectedSegment]:
    if not events:
        return []
    ordered = sorted(events, key=lambda item: item.session_start_sec)
    clusters: list[list[DualViewActionEvent]] = []
    current: list[DualViewActionEvent] = []
    current_end = 0.0
    for event in ordered:
        if current and event.session_start_sec - current_end > episode_merge_gap_sec:
            clusters.append(current)
            current = []
        current.append(event)
        current_end = max(current_end, event.session_end_sec)
    if current:
        clusters.append(current)
    pseudo = VideoSource(
        name="global_multiview",
        path="global_multiview",
        start_time=manifest.session_start_time,
        fps=1.0,
        offset_sec=0.0,
    )
    segments: list[DetectedSegment] = []
    for index, cluster in enumerate(clusters, start=1):
        start = max(0.0, min(item.session_start_sec for item in cluster) - pre_roll_sec)
        end = max(item.session_end_sec for item in cluster) + post_roll_sec
        label_counts = Counter(item.object_family for item in cluster)
        avg_score = sum(item.alignment_score for item in cluster) / max(1, len(cluster))
        segment = DetectedSegment(
            segment_id=f"seg_{index:06d}",
            start_sec=round(float(start), 6),
            end_sec=round(float(end), 6),
            duration_sec=round(max(0.1, float(end) - float(start)), 6),
            global_start_time=local_sec_to_global_time(pseudo, start).isoformat(),
            global_end_time=local_sec_to_global_time(pseudo, end).isoformat(),
            avg_motion_score=round(avg_score, 6),
            avg_active_score=round(avg_score, 6),
            start_reason="dual_view_action_event_cluster_start",
            end_reason="dual_view_action_event_cluster_end",
            review_required=False,
            detector_backend="dual_view_action_alignment",
            detector_source_view="dual_view",
            yolo_label_counts=dict(label_counts),
            yolo_interaction_count=len(cluster),
            boundary_confidence=round(avg_score, 6),
            boundary_support_count=len(cluster),
            boundary_source="dual_view_action_events",
            decision_path="dual_view_action_alignment",
            decision_trace=[
                "formal_episode_requires_confirmed_dual_view_action_events",
                f"dual_event_count={len(cluster)}",
            ],
            fallback_used=False,
            fallback_reason="",
            reason_code="dual_view_action_aligned",
            raw_score=round(avg_score, 6),
            score=round(avg_score, 6),
            source="dual_view_action_event",
            source_view="dual_view",
            detector_version=SCHEMA_VERSION,
            final_score=round(avg_score, 6),
            evidence_link=",".join(item.dual_event_id for item in cluster),
            retrieval_boost_factors={"dual_view_action_alignment": 1.0, "dual_event_count": len(cluster)},
        )
        segments.append(segment)
    return segments


def _evidence_range(items: list[ViewActionEvidence]) -> dict[str, Any]:
    if not items:
        return {
            "count": 0,
            "explicit_interaction_count": 0,
            "action_phase_counts": {},
            "first_start_sec": None,
            "last_end_sec": None,
            "first_explicit_interaction_sec": None,
            "last_explicit_interaction_sec": None,
        }
    explicit = [item for item in items if int(item.interaction_row_count or 0) > 0]
    phase_counts = Counter(str(item.action_phase or "unknown") for item in items)
    return {
        "count": len(items),
        "explicit_interaction_count": len(explicit),
        "action_phase_counts": dict(sorted(phase_counts.items())),
        "first_start_sec": round(min(float(item.session_start_sec) for item in items), 6),
        "last_end_sec": round(max(float(item.session_end_sec) for item in items), 6),
        "first_explicit_interaction_sec": (
            round(min(float(item.session_start_sec) for item in explicit), 6)
            if explicit
            else None
        ),
        "last_explicit_interaction_sec": (
            round(max(float(item.session_end_sec) for item in explicit), 6)
            if explicit
            else None
        ),
    }


def _coverage_by_family(items: list[ViewActionEvidence]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[ViewActionEvidence]] = defaultdict(list)
    for item in items:
        grouped[(item.action_family, item.object_family, item.action_phase)].append(item)
    rows: list[dict[str, Any]] = []
    for (action_family, object_family, action_phase), family_items in sorted(grouped.items()):
        first_items = [item for item in family_items if item.view == "first_person"]
        third_items = [item for item in family_items if item.view == "third_person"]
        rows.append(
            {
                "action_family": action_family,
                "object_family": object_family,
                "action_phase": action_phase,
                "canonical_action_type": _canonical_action_type_for_family(action_family),
                "canonical_object_family": _canonical_object_family_for_family(object_family),
                "first_person": _evidence_range(first_items),
                "third_person": _evidence_range(third_items),
                "both_views_present": bool(first_items and third_items),
                "explicit_both_views_present": bool(
                    any(int(item.interaction_row_count or 0) > 0 for item in first_items)
                    and any(int(item.interaction_row_count or 0) > 0 for item in third_items)
                ),
            }
        )
    return rows


def _alignment_diagnostics(
    view_evidence: list[ViewActionEvidence],
    events: list[DualViewActionEvent],
    unmatched: list[dict[str, Any]],
) -> dict[str, Any]:
    first = [item for item in view_evidence if item.view == "first_person"]
    third = [item for item in view_evidence if item.view == "third_person"]
    unmatched_reasons = Counter(str(row.get("reason") or row.get("unmatched_reason") or "unknown") for row in unmatched)
    confirmed_families = Counter(f"{event.action_family}/{event.object_family}" for event in events)
    evidence_phase_counts = Counter(str(item.action_phase or "unknown") for item in view_evidence)
    confirmed_phase_counts = Counter(str(event.action_phase or "unknown") for event in events)
    denominator = max(1, min(len(first), len(third)))
    return {
        "schema_version": "dual_view_action_alignment_diagnostics.v1",
        "formal_event_count": len(events),
        "view_evidence_count": len(view_evidence),
        "first_person": _evidence_range(first),
        "third_person": _evidence_range(third),
        "formal_event_to_smaller_view_evidence_ratio": round(len(events) / denominator, 6),
        "unmatched_reason_counts": dict(sorted(unmatched_reasons.items())),
        "confirmed_family_counts": dict(sorted(confirmed_families.items())),
        "evidence_phase_counts": dict(sorted(evidence_phase_counts.items())),
        "confirmed_phase_counts": dict(sorted(confirmed_phase_counts.items())),
        "coverage_by_action_family": _coverage_by_family(view_evidence),
    }


def _candidate_from_row(session_id: str, row: Mapping[str, Any], row_index: int) -> dict[str, Any] | None:
    row_session_id = str(row.get("session_id") or row.get("run_session_id") or "").strip()
    if row_session_id and row_session_id != session_id:
        return None
    view = _norm_view(row.get("source_view") or row.get("view"))
    if view not in {"first_person", "third_person"}:
        return None
    time_sec = _row_time(row)
    labels = _labels_from_row(row)
    hand_count = sum(int((row.get("label_counts") or {}).get(label, 0) or 0) for label in HAND_LABELS)
    interactions = [item for item in row.get("hand_object_interactions") or [] if isinstance(item, Mapping)]
    scored_interactions = []
    for item in interactions:
        obj = _norm_label(item.get("object_label") or item.get("object_name") or item.get("label"))
        if not obj or obj in CONTEXT_ONLY_LABELS:
            continue
        score = _float(item.get("score", item.get("confidence", item.get("interaction_score"))), 0.0)
        if score >= _env_float("KEY_ACTION_VIEW_ACTION_MIN_INTERACTION_SCORE", 0.25):
            scored_interactions.append((score, obj))
    if scored_interactions:
        score, obj = max(scored_interactions, key=lambda item: item[0])
        evidence_kind = "hand_object_interaction"
    else:
        active_score = _float(row.get("active_score"), 0.0)
        object_labels = sorted(label for label in labels if label not in HAND_LABELS and label not in CONTEXT_ONLY_LABELS)
        if hand_count <= 0 or not object_labels or active_score < _env_float("KEY_ACTION_VIEW_ACTION_MIN_COPRESENCE_SCORE", 0.45):
            return None
        obj = _prefer_object_label(object_labels)
        score = active_score
        evidence_kind = "hand_object_copresence"
    action_family = action_family_for_label(obj)
    object_family = object_family_for_label(obj)
    if action_family == "context_only":
        return None
    action_phase = _action_phase_from_row(
        row,
        labels=labels,
        object_label=obj,
        action_family=action_family,
        object_family=object_family,
        evidence_kind=evidence_kind,
    )
    return {
        "session_id": session_id,
        "view": view,
        "time_sec": time_sec,
        "action_family": action_family,
        "object_family": object_family,
        "action_phase": action_phase,
        "raw_yolo_labels": sorted(labels),
        "hand_count": hand_count,
        "object_count": len([label for label in labels if label not in HAND_LABELS and label not in CONTEXT_ONLY_LABELS]),
        "interaction_score": score,
        "evidence_kind": evidence_kind,
        "row_index": row_index,
        "frame_index": int(_float(row.get("frame_index"), row_index)),
    }


def _evidence_from_bucket(
    session_id: str,
    index: int,
    key: tuple[str, str, str, str],
    bucket: list[dict[str, Any]],
) -> ViewActionEvidence:
    view, action_family, object_family, action_phase = key
    start = min(float(item["time_sec"]) for item in bucket)
    end = max(float(item["time_sec"]) for item in bucket)
    duration = max(0.1, end - start)
    scores = [float(item.get("interaction_score") or 0.0) for item in bucket]
    peak_item = max(bucket, key=lambda item: (float(item.get("interaction_score") or 0.0), float(item["time_sec"])))
    raw_labels = sorted({label for item in bucket for label in item.get("raw_yolo_labels", [])})
    kinds = Counter(str(item.get("evidence_kind") or "") for item in bucket)
    interaction_count = int(kinds.get("hand_object_interaction", 0))
    quality_flags: list[str] = []
    if interaction_count <= 0:
        quality_flags.append("no_explicit_hand_object_interaction")
    if len(bucket) < 2:
        quality_flags.append("single_frame_action_evidence")
    return ViewActionEvidence(
        evidence_id=f"vae_{index:06d}",
        session_id=session_id,
        view=view,
        session_start_sec=round(start, 6),
        session_end_sec=round(end, 6),
        duration_sec=round(duration, 6),
        peak_session_sec=round(float(peak_item["time_sec"]), 6),
        action_family=action_family,
        action_display_name=ACTION_DISPLAY_NAMES.get(action_family, ACTION_DISPLAY_NAMES["hand_object_operation"]),
        object_family=object_family,
        object_display_name=OBJECT_DISPLAY_NAMES.get(object_family, OBJECT_DISPLAY_NAMES["object_family"]),
        raw_yolo_labels=raw_labels,
        hand_count=max(int(item.get("hand_count") or 0) for item in bucket),
        object_count=max(int(item.get("object_count") or 0) for item in bucket),
        row_count=len(bucket),
        interaction_row_count=interaction_count,
        max_interaction_score=round(max(scores or [0.0]), 6),
        avg_interaction_score=round(sum(scores) / max(1, len(scores)), 6),
        evidence_density=round(min(1.0, len(bucket) / max(1.0, duration)), 6),
        evidence_kind="hand_object_interaction" if interaction_count else "hand_object_copresence",
        source_row_indices=[int(item.get("row_index") or 0) for item in bucket],
        source_frame_indices=[int(item.get("frame_index") or 0) for item in bucket],
        quality_flags=quality_flags,
        action_phase=_bucket_action_phase(bucket, fallback=action_phase),
    )


def _bucket_action_phase(bucket: list[dict[str, Any]], *, fallback: str = "unknown") -> str:
    phases = [
        _canonical_action_phase(item.get("action_phase"))
        for item in bucket
        if _canonical_action_phase(item.get("action_phase"))
    ]
    if phases:
        return Counter(phases).most_common(1)[0][0]
    phase = _canonical_action_phase(fallback)
    return phase or "unknown"


def _action_phase_from_row(
    row: Mapping[str, Any],
    *,
    labels: set[str],
    object_label: str,
    action_family: str,
    object_family: str,
    evidence_kind: str,
) -> str:
    declared = _declared_action_phase_from_row(row)
    if declared:
        return declared
    text_phase = _text_action_phase_from_row(row)
    if text_phase:
        return text_phase
    norm_object = _norm_label(object_label)
    if norm_object in OFF_BENCH_PHASE_LABELS or labels & OFF_BENCH_PHASE_LABELS:
        return "off_bench_or_invalid"
    prep_labels = labels & PREP_PHASE_LABELS
    has_bench_label = _has_bench_object_label(labels | {norm_object})
    if (prep_labels or norm_object in PREP_PHASE_LABELS) and not has_bench_label:
        return "prep_glove_or_sleeve"
    if evidence_kind == "hand_object_interaction" and _is_bench_action_family(action_family, object_family):
        return "bench_operation"
    return "unknown"


def _declared_action_phase_from_row(row: Mapping[str, Any]) -> str:
    for key in ACTION_PHASE_FIELDS:
        if key not in row:
            continue
        raw = row.get(key)
        if raw is None or str(raw).strip() == "":
            continue
        phase = _canonical_action_phase(raw)
        if phase:
            return phase
        if key != "phase":
            return "unknown"
    return ""


def _text_action_phase_from_row(row: Mapping[str, Any]) -> str:
    text = " ".join(str(row.get(key) or "") for key in ACTION_PHASE_TEXT_FIELDS if row.get(key))
    return _canonical_action_phase(text)


def _canonical_action_phase(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    norm = text.replace("-", "_").replace(" ", "_").replace("/", "_")
    if norm in ACTION_PHASES:
        return norm
    if norm in TEMPORAL_FRAME_PHASES:
        return ""
    if norm in {"unknown", "uncertain", "unconfirmed", "unclear", "unsure"}:
        return "unknown"
    if any(marker in norm for marker in ("off_bench", "invalid", "out_of_scope", "background", "context_only", "non_action", "no_action")):
        return "off_bench_or_invalid"
    if any(marker in norm for marker in ("bench", "operation", "operate", "contact", "interaction", "manipulat", "transfer", "pipette", "weigh", "balance", "spatula", "bottle", "container", "pour", "mix", "measure", "sample", "reagent")):
        return "bench_operation"
    if any(marker in norm for marker in ("prep_glove", "glove_prep", "put_on_glove", "wear_glove", "wearing_glove", "don_glove", "adjust_glove", "sleeve", "ppe", "lab_coat", "glove")):
        return "prep_glove_or_sleeve"
    return ""


def _has_bench_object_label(labels: set[str]) -> bool:
    return bool(
        {
            "balance",
            "scale",
            "panel",
            "paper",
            "weighing_paper",
            "reagent_bottle",
            "reagent_bottle_open",
            "bottle_cap",
            "sample_bottle",
            "sample_bottle_blue",
            "beaker",
            "container",
            "tube",
            "tube_cap",
            "spatula",
            "pipette",
            "pipette_tip",
            "magnetic_stirrer",
            "magnetic_stir_bar",
        }
        & labels
    )


def _is_bench_action_family(action_family: str, object_family: str) -> bool:
    if str(action_family or "") in {"", "context_only"}:
        return False
    return str(object_family or "") in BENCH_OBJECT_FAMILIES


def _infer_action_phase(
    *,
    view: str,
    action_family: str,
    object_family: str,
    raw_labels: list[str],
    row_count: int,
    interaction_row_count: int,
    quality_flags: list[str],
) -> str:
    """Infer a coarse action phase for formal dual-view pairing.

    This is deliberately conservative: time alignment only proves that two
    frames share a clock; formal evidence also needs both views to describe the
    same physical phase. Preparation/glove handling is kept out of formal
    bench-operation matches even when a detector assigns a nearby lab object.
    """

    labels = {str(label or "").strip().lower() for label in raw_labels or []}
    if not labels:
        return "unknown"
    context_labels = {"lab_coat", "ppe_storage"}
    has_bare_hand = "hand" in labels
    has_gloved_hand = "gloved_hand" in labels
    has_lab_object = bool(labels & {
        "balance",
        "scale",
        "panel",
        "paper",
        "weighing_paper",
        "reagent_bottle",
        "reagent_bottle_open",
        "bottle_cap",
        "sample_bottle",
        "sample_bottle_blue",
        "beaker",
        "container",
        "tube",
        "spatula",
        "pipette",
        "pipette_tip",
        "magnetic_stirrer",
        "magnetic_stir_bar",
    })
    explicit = int(interaction_row_count or 0) > 0
    sparse_or_weak = row_count <= 3 or any("single_frame" in str(flag) for flag in quality_flags or [])

    if not has_lab_object and labels <= (HAND_LABELS | context_labels):
        return "prep_glove_or_sleeve"
    if view == "first_person" and has_bare_hand and has_gloved_hand and sparse_or_weak:
        return "prep_glove_or_sleeve"
    if not explicit and has_gloved_hand and labels <= (HAND_LABELS | context_labels | {"balance", "paper", "weighing_paper"}):
        return "prep_glove_or_sleeve"
    if explicit and object_family not in {"object_family", ""} and action_family not in {"context_only", ""}:
        return "bench_operation"
    if not explicit and has_lab_object and not sparse_or_weak:
        return "bench_operation"
    if has_lab_object:
        return "unknown"
    return "off_bench_or_invalid"


def _pair_action_phase(left: ViewActionEvidence, right: ViewActionEvidence) -> str:
    left_phase = _canonical_action_phase(left.action_phase) or "unknown"
    right_phase = _canonical_action_phase(right.action_phase) or "unknown"
    if left_phase == right_phase and left_phase not in {"", "unknown", "off_bench_or_invalid"}:
        return left_phase
    return ""


def _pair_phase_rejection_reason(left: ViewActionEvidence, right: ViewActionEvidence) -> str:
    left_phase = _canonical_action_phase(left.action_phase) or "unknown"
    right_phase = _canonical_action_phase(right.action_phase) or "unknown"
    if "off_bench_or_invalid" in {left_phase, right_phase}:
        return "phase_unknown"
    if left_phase == right_phase and left_phase not in {"", "unknown"}:
        return ""
    if "unknown" in {left_phase, right_phase, ""}:
        return "phase_unknown"
    return "phase_mismatch"


def _pair_scores(left: ViewActionEvidence, right: ViewActionEvidence, *, max_center_delta_sec: float) -> dict[str, float]:
    overlap = max(0.0, min(left.session_end_sec, right.session_end_sec) - max(left.session_start_sec, right.session_start_sec))
    min_duration = max(0.1, min(left.duration_sec, right.duration_sec))
    overlap_score = min(1.0, overlap / min_duration)
    left_center = (left.session_start_sec + left.session_end_sec) / 2.0
    right_center = (right.session_start_sec + right.session_end_sec) / 2.0
    center_delta = abs(left_center - right_center)
    center_score = max(0.0, 1.0 - center_delta / max(0.001, max_center_delta_sec))
    peak_delta = abs(float(left.peak_session_sec) - float(right.peak_session_sec))
    peak_score = max(0.0, 1.0 - peak_delta / max(0.001, max_center_delta_sec))
    left_action = _canonical_action_type_for_family(left.action_family)
    right_action = _canonical_action_type_for_family(right.action_family)
    if left_action == right_action:
        action = 1.0
    elif _allow_operation_context_action_match() and _operation_group_for_family(left.action_family) == _operation_group_for_family(right.action_family):
        action = 0.88
    elif _compatible_actions(left.action_family, right.action_family):
        action = 0.82
    else:
        action = 0.0
    if left.object_family == right.object_family:
        obj = 1.0
    elif _allow_operation_context_object_match() and _object_context_group_for_family(left.object_family) == _object_context_group_for_family(right.object_family):
        obj = 0.88
    elif _compatible_objects(left.object_family, right.object_family):
        obj = 0.82
    else:
        obj = 0.0
    density_delta = abs(left.evidence_density - right.evidence_density)
    evidence_symmetry = max(0.0, 1.0 - density_delta / max(left.evidence_density, right.evidence_density, 0.001))
    phase_match = 1.0 if _pair_action_phase(left, right) else 0.0
    return {
        "temporal_overlap_score": overlap_score,
        "raw_overlap_score": overlap_score,
        "center_score": center_score,
        "center_delta_sec": center_delta,
        "overlap_sec": overlap,
        "peak_delta_sec": peak_delta,
        "peak_score": peak_score,
        "action_match_score": action,
        "object_match_score": obj,
        "evidence_symmetry_score": evidence_symmetry,
        "action_phase_match_score": phase_match,
    }


def _split_pairable_view_evidence(
    items: list[ViewActionEvidence],
) -> tuple[list[ViewActionEvidence], list[tuple[ViewActionEvidence, str]]]:
    eligible: list[ViewActionEvidence] = []
    rejected: list[tuple[ViewActionEvidence, str]] = []
    for item in items:
        reason = _view_evidence_rejection_reason(item)
        if reason:
            rejected.append((item, reason))
        else:
            eligible.append(item)
    return eligible, rejected


def _view_evidence_rejection_reason(item: ViewActionEvidence) -> str:
    return "weak_view_evidence" if _weak_view_evidence_reasons(item) else ""


def _weak_view_evidence_reasons(item: ViewActionEvidence) -> list[str]:
    min_rows = _min_view_action_row_count()
    min_interaction_rows = _min_view_action_interaction_row_count()
    min_score = _min_view_action_strong_score()
    reasons: list[str] = []
    sparse_pairable = _is_sparse_pairable_view_evidence(item)
    if item.row_count < min_rows and not sparse_pairable:
        reasons.append(f"row_count<{min_rows}")
    if item.interaction_row_count < min_interaction_rows:
        reasons.append(f"interaction_row_count<{min_interaction_rows}")
    bad_flags = []
    for flag in item.quality_flags or []:
        text = str(flag)
        if "no_explicit" in text:
            bad_flags.append(text)
            continue
        if "single_frame" in text and not sparse_pairable:
            bad_flags.append(text)
    if bad_flags:
        reasons.append("quality_flags=" + ",".join(sorted(bad_flags)))
    if float(item.max_interaction_score or 0.0) < min_score:
        reasons.append(f"max_interaction_score<{min_score:.3f}")
    return reasons


def _allow_sparse_paired_view_evidence() -> bool:
    return _env_truthy("KEY_ACTION_ALLOW_DUAL_VIEW_SPARSE_PAIRING", False)


def _is_sparse_pairable_view_evidence(item: ViewActionEvidence) -> bool:
    """Allow low-fps single-frame evidence only as part of a dual-view pair.

    A sparse item is never enough by itself. It only enters the pair search when
    it has explicit hand-object interaction evidence; the final event still
    requires a matching first/third item in the same session time window.
    """

    if not _allow_sparse_paired_view_evidence():
        return False
    if int(item.row_count or 0) >= _min_view_action_row_count():
        return False
    if int(item.interaction_row_count or 0) < _min_view_action_interaction_row_count():
        return False
    if float(item.max_interaction_score or 0.0) < _min_view_action_strong_score():
        return False
    for flag in item.quality_flags or []:
        if "no_explicit" in str(flag):
            return False
    return True


def _pair_gate_rejection_reason(
    left: ViewActionEvidence,
    right: ViewActionEvidence,
    scores: Mapping[str, float],
    *,
    max_center_delta_sec: float,
) -> str:
    if left.session_id != right.session_id:
        return "session_mismatch"
    if float(scores.get("action_match_score") or 0.0) <= 0.0:
        return "action_mismatch"
    if float(scores.get("object_match_score") or 0.0) <= 0.0:
        return "object_mismatch"
    phase_reason = _pair_phase_rejection_reason(left, right)
    if phase_reason:
        return phase_reason
    has_overlap = (
        float(scores.get("overlap_sec") or 0.0) >= _min_temporal_overlap_sec()
        and float(scores.get("temporal_overlap_score") or 0.0) >= _min_temporal_overlap_score()
    )
    has_center = float(scores.get("center_delta_sec") or 0.0) <= max_center_delta_sec
    has_peak = float(scores.get("peak_delta_sec") or 0.0) <= _max_peak_delta_sec(max_center_delta_sec)
    if _require_temporal_overlap() and not has_overlap:
        return "no_temporal_overlap"
    if _require_peak_alignment() and not has_peak:
        if has_overlap or has_center:
            return "peak_mismatch"
        return "no_temporal_overlap"
    if not (has_overlap or has_center or has_peak):
        return "no_temporal_overlap"
    return ""


def _temporal_score_for_pair(scores: Mapping[str, float], *, max_center_delta_sec: float) -> float:
    peak_score = float(scores.get("peak_score") or 0.0)
    if _require_peak_alignment():
        return peak_score
    if float(scores.get("peak_delta_sec") or 0.0) <= _max_peak_delta_sec(max_center_delta_sec):
        return peak_score
    if (
        float(scores.get("overlap_sec") or 0.0) >= _min_temporal_overlap_sec()
        and float(scores.get("temporal_overlap_score") or 0.0) >= _min_temporal_overlap_score()
    ):
        return float(scores.get("temporal_overlap_score") or 0.0)
    if float(scores.get("center_delta_sec") or 0.0) <= max_center_delta_sec:
        return float(scores.get("peak_score") or scores.get("center_score") or 0.0)
    if float(scores.get("peak_delta_sec") or 0.0) <= _max_peak_delta_sec(max_center_delta_sec):
        return float(scores.get("peak_score") or 0.0)
    return 0.0


def _temporal_gate_trace(scores: Mapping[str, float], *, max_center_delta_sec: float) -> str:
    if float(scores.get("peak_delta_sec") or 0.0) <= _max_peak_delta_sec(max_center_delta_sec):
        return "temporal_gate=peak_delta"
    if (
        float(scores.get("overlap_sec") or 0.0) >= _min_temporal_overlap_sec()
        and float(scores.get("temporal_overlap_score") or 0.0) >= _min_temporal_overlap_score()
    ):
        return "temporal_gate=overlap_only"
    if float(scores.get("center_delta_sec") or 0.0) <= max_center_delta_sec:
        return "temporal_gate=center_delta"
    return "temporal_gate=failed"


def _unmatched_view_evidence_row(item: ViewActionEvidence, reason: str, **details: Any) -> dict[str, Any]:
    weak_reasons = _weak_view_evidence_reasons(item)
    return {
        "schema_version": SCHEMA_VERSION,
        "session_id": item.session_id,
        "evidence_id": item.evidence_id,
        "view": item.view,
        "session_start_sec": item.session_start_sec,
        "session_end_sec": item.session_end_sec,
        "peak_session_sec": item.peak_session_sec,
        "action_family": item.action_family,
        "action_display_name": item.action_display_name,
        "object_family": item.object_family,
        "object_display_name": item.object_display_name,
        "canonical_action_type": _canonical_action_type_for_family(item.action_family),
        "canonical_object_family": _canonical_object_family_for_family(item.object_family),
        "raw_yolo_labels": item.raw_yolo_labels,
        "raw_labels": item.raw_yolo_labels,
        "action_phase": item.action_phase,
        "status": "unmatched_view_evidence",
        "formal_action_event_allowed": False,
        "formal_material_allowed": False,
        "video_memory_allowed": False,
        "reason": reason or "single_view_candidate",
        "unmatched_reason": reason or "single_view_candidate",
        "weak_evidence_reasons": weak_reasons,
        "reason_details": details,
    }


def _unmatched_reason_for_evidence(
    item: ViewActionEvidence,
    opposite_items: list[ViewActionEvidence],
    *,
    max_center_delta_sec: float,
    min_alignment_score: float,
) -> tuple[str, dict[str, Any]]:
    if not opposite_items:
        return "single_view_candidate", {"opposite_view_candidate_count": 0}
    same_action = [
        other
        for other in opposite_items
        if _canonical_action_type_for_family(other.action_family) == _canonical_action_type_for_family(item.action_family)
    ]
    if not same_action:
        return (
            "action_mismatch",
            {
                "canonical_action_type": _canonical_action_type_for_family(item.action_family),
                "opposite_canonical_action_types": sorted(
                    {_canonical_action_type_for_family(other.action_family) for other in opposite_items}
                ),
            },
        )
    same_object = [
        other
        for other in same_action
        if other.object_family == item.object_family
    ]
    if not same_object:
        return (
            "object_mismatch",
            {
                "canonical_object_family": _canonical_object_family_for_family(item.object_family),
                "object_family": item.object_family,
                "opposite_canonical_object_families": sorted(
                    {_canonical_object_family_for_family(other.object_family) for other in same_action}
                ),
                "opposite_object_families": sorted({other.object_family for other in same_action}),
            },
        )
    temporal_candidates = []
    phase_failures = []
    temporal_failures = []
    for other in same_object:
        scores = _pair_scores(item, other, max_center_delta_sec=max_center_delta_sec)
        reason = _pair_gate_rejection_reason(item, other, scores, max_center_delta_sec=max_center_delta_sec)
        detail = {
            "opposite_evidence_id": other.evidence_id,
            "overlap_sec": round(float(scores.get("overlap_sec") or 0.0), 6),
            "overlap_score": round(float(scores.get("temporal_overlap_score") or 0.0), 6),
            "center_delta_sec": round(float(scores.get("center_delta_sec") or 0.0), 6),
            "peak_delta_sec": round(float(scores.get("peak_delta_sec") or 0.0), 6),
            "source_action_phase": item.action_phase,
            "opposite_action_phase": other.action_phase,
        }
        if reason in {"phase_mismatch", "phase_unknown"}:
            detail["phase_rejection_reason"] = reason
            phase_failures.append(detail)
        elif reason in {"no_temporal_overlap", "peak_mismatch"}:
            detail["temporal_rejection_reason"] = reason
            temporal_failures.append(detail)
        elif not reason:
            temporal_candidates.append((other, scores))
    if not temporal_candidates:
        if phase_failures:
            failure_reasons = sorted({str(item.get("phase_rejection_reason") or "phase_mismatch") for item in phase_failures})
            reason = "phase_unknown" if "phase_unknown" in failure_reasons else "phase_mismatch"
            return (
                reason,
                {
                    "policy": "formal dual-view evidence requires both views to be in the same confirmed action phase.",
                    "candidates": phase_failures,
                },
            )
        failure_reasons = sorted({str(item.get("temporal_rejection_reason") or "no_temporal_overlap") for item in temporal_failures})
        reason = "peak_mismatch" if "peak_mismatch" in failure_reasons else "no_temporal_overlap"
        return (
            reason,
            {
                "max_center_delta_sec": max_center_delta_sec,
                "max_peak_delta_sec": _max_peak_delta_sec(max_center_delta_sec),
                "require_peak_alignment": _require_peak_alignment(),
                "min_temporal_overlap_score": _min_temporal_overlap_score(),
                "candidates": temporal_failures,
            },
        )
    return (
        "no_confirmed_pair_above_threshold",
        {
            "min_alignment_score": min_alignment_score,
            "candidate_evidence_ids": [other.evidence_id for other, _scores in temporal_candidates],
        },
    )


def _min_view_action_row_count() -> int:
    return max(1, _env_int("KEY_ACTION_DUAL_VIEW_MIN_EVIDENCE_ROWS", 2))


def _min_view_action_interaction_row_count() -> int:
    return max(1, _env_int("KEY_ACTION_DUAL_VIEW_MIN_INTERACTION_ROWS", 1))


def _min_view_action_strong_score() -> float:
    return _env_float("KEY_ACTION_DUAL_VIEW_MIN_VIEW_INTERACTION_SCORE", 0.55)


def _min_temporal_overlap_score() -> float:
    return _env_float("KEY_ACTION_DUAL_VIEW_MIN_RAW_OVERLAP_SCORE", 0.2)


def _min_temporal_overlap_sec() -> float:
    return _env_float("KEY_ACTION_DUAL_VIEW_MIN_OVERLAP_SEC", 0.001)


def _max_peak_delta_sec(max_center_delta_sec: float) -> float:
    return _env_float("KEY_ACTION_DUAL_VIEW_MAX_PEAK_DELTA_SEC", max_center_delta_sec)


def _require_peak_alignment() -> bool:
    return _env_truthy("KEY_ACTION_REQUIRE_DUAL_VIEW_PEAK_ALIGNMENT", True)


def _require_temporal_overlap() -> bool:
    return _env_truthy("KEY_ACTION_REQUIRE_DUAL_VIEW_TEMPORAL_OVERLAP", True)


def _canonical_action_type_for_family(family: str) -> str:
    return {
        "reagent_bottle_operation": "hand-bottle",
        "sample_bottle_operation": "hand-bottle",
        "pipette_operation": "hand-pipette",
        "balance_operation": "hand-balance",
        "weighing_paper_operation": "hand-paper",
        "spatula_operation": "hand-spatula",
        "container_operation": "hand-container",
        "equipment_operation": "hand-equipment",
        "hand_object_operation": "hand-object",
    }.get(str(family or ""), str(family or ""))


def _operation_group_for_family(family: str) -> str:
    value = str(family or "")
    if value in {"balance_operation", "weighing_paper_operation", "spatula_operation"}:
        return "weighing_operation"
    if value in {"reagent_bottle_operation", "sample_bottle_operation", "container_operation"}:
        return "container_handling"
    if value == "pipette_operation":
        return "pipetting_operation"
    if value == "equipment_operation":
        return "equipment_operation"
    if value == "hand_object_operation":
        return "hand_object_operation"
    return value


def _allow_operation_context_action_match() -> bool:
    return _env_truthy("KEY_ACTION_DUAL_VIEW_ALLOW_OPERATION_CONTEXT_ACTION_MATCH", False)


def _allow_operation_context_object_match() -> bool:
    return _env_truthy("KEY_ACTION_DUAL_VIEW_ALLOW_OPERATION_CONTEXT_OBJECT_MATCH", False)


def _object_context_group_for_family(family: str) -> str:
    value = str(family or "")
    if value in {"balance_family", "paper_family", "spatula_family"}:
        return "weighing_workspace"
    if value in {"reagent_bottle_family", "sample_bottle_family", "container_family"}:
        return "container_workspace"
    if value == "pipette_family":
        return "pipette_workspace"
    if value == "equipment_family":
        return "equipment_workspace"
    return value


def _canonical_object_family_for_family(family: str) -> str:
    primary = _object_from_family(str(family or ""))
    return canonical_object_family(primary) or str(family or "")


def action_family_for_label(label: Any) -> str:
    value = _norm_label(label)
    if value in {"reagent_bottle", "reagent_bottle_open", "bottle_cap", "cap", "bottle"}:
        return "reagent_bottle_operation"
    if value in {"sample_bottle", "sample_bottle_blue"}:
        return "sample_bottle_operation"
    if value in {"pipette", "pipette_tip", "spearhead"}:
        return "pipette_operation"
    if value in {"balance", "scale", "panel"}:
        return "balance_operation"
    if value in {"paper", "weighing_paper"}:
        return "weighing_paper_operation"
    if value == "spatula":
        return "spatula_operation"
    if value in {"beaker", "container", "tube", "tube_cap", "tube_rack"}:
        return "container_operation"
    if value in {"magnetic_stirrer", "magnetic_stir_bar"}:
        return "equipment_operation"
    if value in CONTEXT_ONLY_LABELS:
        return "context_only"
    return "hand_object_operation"


def object_family_for_label(label: Any) -> str:
    value = _norm_label(label)
    if value in {"reagent_bottle", "reagent_bottle_open", "bottle_cap", "cap", "bottle"}:
        return "reagent_bottle_family"
    if value in {"sample_bottle", "sample_bottle_blue"}:
        return "sample_bottle_family"
    if value in {"pipette", "pipette_tip", "spearhead"}:
        return "pipette_family"
    if value in {"balance", "scale", "panel"}:
        return "balance_family"
    if value in {"paper", "weighing_paper"}:
        return "paper_family"
    if value == "spatula":
        return "spatula_family"
    if value in {"beaker", "container", "tube", "tube_cap", "tube_rack"}:
        return "container_family"
    if value in {"magnetic_stirrer", "magnetic_stir_bar"}:
        return "equipment_family"
    return "object_family"


def _segment_to_row(segment: DetectedSegment, events: list[DualViewActionEvent]) -> dict[str, Any]:
    event_ids = set(str(segment.evidence_link or "").split(","))
    matched = [event for event in events if event.dual_event_id in event_ids]
    return {
        "schema_version": "dual_view_aligned_experiment_episode.v1",
        "episode_id": segment.segment_id.replace("seg_", "episode_"),
        "segment_id": segment.segment_id,
        "session_start_sec": segment.start_sec,
        "session_end_sec": segment.end_sec,
        "duration_sec": segment.duration_sec,
        "global_start_time": segment.global_start_time,
        "global_end_time": segment.global_end_time,
        "dual_event_ids": [event.dual_event_id for event in matched],
        "main_action_families": sorted({event.action_family for event in matched}),
        "main_object_families": sorted({event.object_family for event in matched}),
        "alignment_confidence": segment.final_score,
        "detector_backend": segment.detector_backend,
        "decision_path": segment.decision_path,
        "decision_trace": segment.decision_trace,
        "reason_code": segment.reason_code,
    }


def _labels_from_row(row: Mapping[str, Any]) -> set[str]:
    labels: set[str] = set()
    counts = row.get("label_counts") if isinstance(row.get("label_counts"), Mapping) else {}
    for label, count in dict(counts).items():
        try:
            if int(count) <= 0:
                continue
        except (TypeError, ValueError):
            continue
        labels.add(_norm_label(label))
    for det in row.get("detections") or []:
        if isinstance(det, Mapping):
            labels.add(_norm_label(det.get("label") or det.get("raw_label")))
    labels.discard("")
    return labels


def _prefer_object_label(labels: list[str]) -> str:
    priority = [
        "balance",
        "pipette",
        "pipette_tip",
        "spatula",
        "reagent_bottle",
        "reagent_bottle_open",
        "bottle_cap",
        "sample_bottle",
        "sample_bottle_blue",
        "paper",
        "beaker",
        "container",
        "tube",
    ]
    available = set(labels)
    for item in priority:
        if item in available:
            return item
    return labels[0] if labels else "object"


def _prefer_action_family(left: ViewActionEvidence, right: ViewActionEvidence) -> str:
    if left.interaction_row_count >= right.interaction_row_count:
        return left.action_family
    return right.action_family


def _prefer_object_family(left: ViewActionEvidence, right: ViewActionEvidence) -> str:
    if left.interaction_row_count >= right.interaction_row_count:
        return left.object_family
    return right.object_family


def _compatible_actions(left: str, right: str) -> bool:
    if left == right:
        return True
    if _allow_operation_context_action_match() and _operation_group_for_family(left) == _operation_group_for_family(right):
        return True
    if _env_truthy("KEY_ACTION_STRICT_DUAL_VIEW_ACTION_FAMILY", True):
        return False
    if _env_truthy("KEY_ACTION_DUAL_VIEW_ALLOW_GENERIC_HAND_OBJECT_ACTION_MATCH", False):
        return _operation_group_for_family(left) not in {"", "context_only"} and _operation_group_for_family(right) not in {"", "context_only"}
    bottle = {"reagent_bottle_operation", "sample_bottle_operation"}
    container = {"container_operation", "reagent_bottle_operation", "sample_bottle_operation"}
    return (left in bottle and right in bottle) or (left in container and right in container)


def _compatible_objects(left: str, right: str) -> bool:
    if left == right:
        return True
    if _allow_operation_context_object_match() and _object_context_group_for_family(left) == _object_context_group_for_family(right):
        return True
    if _env_truthy("KEY_ACTION_STRICT_DUAL_VIEW_OBJECT_FAMILY", True):
        return False
    if _env_truthy("KEY_ACTION_DUAL_VIEW_ALLOW_GENERIC_HAND_OBJECT_OBJECT_MATCH", False):
        return _object_context_group_for_family(left) not in {"", "context_only"} and _object_context_group_for_family(right) not in {"", "context_only"}
    container = {"container_family", "reagent_bottle_family", "sample_bottle_family"}
    return left in container and right in container and "container_family" in {left, right}


def _row_time(row: Mapping[str, Any]) -> float:
    for key in ("alignment_time_sec", "session_time_sec", "local_time_sec", "time_sec", "start_sec"):
        value = row.get(key)
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _norm_view(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"first", "operator", "operator_view", "first_person"}:
        return "first_person"
    if text in {"third", "top", "top_view", "third_person"}:
        return "third_person"
    return text


def _norm_label(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return int(default)


def _env_truthy(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() not in {"", "0", "false", "no", "off"}


def _read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.exists() else []


def _default_manifest(session_dir: Path) -> SessionManifest:
    manifest_path = session_dir / "manifest.json"
    if manifest_path.exists():
        try:
            return SessionManifest.load(manifest_path)
        except Exception:
            pass
    return SessionManifest.from_dict(
        {
            "session_id": session_dir.name,
            "session_start_time": "1970-01-01T00:00:00+00:00",
            "videos": {
                "third_person": {
                    "path": str(session_dir / "third_person.mp4"),
                    "start_time": "1970-01-01T00:00:00+00:00",
                    "fps": 30,
                },
                "first_person": {
                    "path": str(session_dir / "first_person.mp4"),
                    "start_time": "1970-01-01T00:00:00+00:00",
                    "fps": 30,
                },
            },
            "output_dir": str(session_dir),
        }
    )


def _rows_from_micro_segments(micro_segments: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for micro_index, micro in enumerate(micro_segments, start=1):
        if not isinstance(micro, Mapping):
            continue
        interaction = micro.get("interaction") if isinstance(micro.get("interaction"), Mapping) else {}
        primary = _norm_label(micro.get("primary_object") or interaction.get("primary_object"))
        peak = _float(interaction.get("peak_interaction_sec"), _float(micro.get("start_sec"), 0.0))
        score = _float(interaction.get("max_interaction_score"), 0.65)
        for evidence in micro.get("yolo_evidence") or []:
            if not isinstance(evidence, Mapping):
                continue
            view = _norm_view(evidence.get("source_view") or evidence.get("view"))
            if view not in {"first_person", "third_person"}:
                continue
            obj = _norm_label(evidence.get("primary_object") or evidence.get("object_label") or primary)
            if not obj:
                continue
            time_sec = _row_time(evidence)
            if time_sec <= 0.0:
                time_sec = peak
            confidence = _float(evidence.get("interaction_score"), _float(evidence.get("confidence"), score))
            row = dict(evidence)
            row.update(
                {
                    "source_view": view,
                    "alignment_time_sec": time_sec,
                    "local_time_sec": row.get("local_time_sec", row.get("time_sec", time_sec)),
                    "frame_index": row.get("frame_index", int(time_sec * 30)),
                    "interaction_score": confidence,
                    "active_score": max(confidence, _float(row.get("active_score"), 0.0)),
                    "micro_segment_id": micro.get("micro_segment_id"),
                    "parent_segment_id": micro.get("parent_segment_id"),
                    "source_mode": "micro_segments.yolo_evidence",
                    "source_row_index": micro_index,
                }
            )
            row.setdefault("label_counts", {"gloved_hand": 1, obj: 1})
            row.setdefault(
                "hand_object_interactions",
                [{"hand_label": "gloved_hand", "object_label": obj, "score": confidence}],
            )
            rows.append(row)
        if _env_truthy("KEY_ACTION_DUAL_VIEW_ALLOW_ASSET_BINDING_EVIDENCE", False):
            evidence_views = {
                _norm_view(item.get("source_view") or item.get("view"))
                for item in micro.get("yolo_evidence") or []
                if isinstance(item, Mapping)
            }
            for binding_index, binding in enumerate(micro.get("asset_bindings") or [], start=1):
                if not isinstance(binding, Mapping):
                    continue
                view = _norm_view(binding.get("view"))
                if view not in {"first_person", "third_person"} or view in evidence_views:
                    continue
                if str(binding.get("evidence_role") or "") != "yolo_evidence" or not primary:
                    continue
                confidence = _float(binding.get("confidence"), score)
                rows.append(
                    {
                        "source_view": view,
                        "alignment_time_sec": peak,
                        "local_time_sec": binding.get("local_start_sec", peak),
                        "frame_index": int(peak * 30) + binding_index,
                        "interaction_score": confidence,
                        "active_score": confidence,
                        "micro_segment_id": micro.get("micro_segment_id"),
                        "parent_segment_id": micro.get("parent_segment_id"),
                        "source_mode": "micro_segments.asset_bindings",
                        "source_row_index": micro_index,
                        "label_counts": {"gloved_hand": 1, primary: 1},
                        "hand_object_interactions": [{"hand_label": "gloved_hand", "object_label": primary, "score": confidence}],
                    }
                )
    return rows


def _compat_evidence_row(row: dict[str, Any], micro_segments: list[Mapping[str, Any]]) -> dict[str, Any]:
    start = _float(row.get("session_start_sec"), 0.0)
    end = _float(row.get("session_end_sec"), start)
    primary = _object_from_family(str(row.get("object_family") or ""))
    interaction_type = "hand_object_contact"
    row.setdefault("schema_version", "key_action_view_action_evidence.v1")
    row["start_sec"] = start
    row["end_sec"] = end
    row["center_sec"] = round((start + end) / 2.0, 6)
    row["peak_sec"] = _float(row.get("peak_session_sec"), row["center_sec"])
    row["confidence"] = _float(row.get("max_interaction_score"), _float(row.get("avg_interaction_score"), 0.0))
    row["frame_count"] = int(row.get("row_count") or 0)
    row["interaction_frame_count"] = int(row.get("interaction_row_count") or 0)
    row["action_type"] = row.get("action_family")
    row["canonical_action_type"] = _canonical_action_type_for_family(str(row.get("action_family") or ""))
    row["semantic_action_type"] = infer_action_type_from_metadata(
        {"primary_object": primary, "interaction_type": interaction_type}
    )
    row["interaction_type"] = interaction_type
    row["primary_object"] = primary
    row["primary_object_family"] = row.get("object_family")
    row["canonical_object_family"] = canonical_object_family(primary)
    row["action_phase"] = _canonical_action_phase(row.get("action_phase")) or "unknown"
    row["micro_segment_ids"] = _micro_ids_for_interval(micro_segments, start, end)
    row["parent_segment_ids"] = _parent_ids_for_interval(micro_segments, start, end)
    row["source_modes"] = ["yolo_frame_rows", "yolo_micro_frame_rows", "micro_segments"]
    row["raw_labels"] = list(row.get("raw_labels") or row.get("raw_yolo_labels") or [])
    weak_flags = [
        flag
        for flag in row.get("quality_flags", []) or []
        if any(marker in str(flag) for marker in DISQUALIFYING_VIEW_QUALITY_FLAG_MARKERS)
    ]
    row["evidence_level"] = (
        "strong"
        if row["confidence"] >= _min_view_action_strong_score()
        and row["frame_count"] >= _min_view_action_row_count()
        and row["interaction_frame_count"] >= _min_view_action_interaction_row_count()
        and not weak_flags
        else "candidate"
    )
    return row


def _formal_event_row(event: dict[str, Any], evidence_by_id: dict[str, dict[str, Any]], micro_segments: list[Mapping[str, Any]]) -> dict[str, Any]:
    first = evidence_by_id.get(str(event.get("first_evidence_id"))) or {}
    third = evidence_by_id.get(str(event.get("third_evidence_id"))) or {}
    start = _float(event.get("session_start_sec"), min(_float(first.get("start_sec"), 0.0), _float(third.get("start_sec"), 0.0)))
    end = _float(event.get("session_end_sec"), max(_float(first.get("end_sec"), start), _float(third.get("end_sec"), start)))
    first_peak = _float(first.get("peak_sec"), start)
    third_peak = _float(third.get("peak_sec"), start)
    overlap_start = max(_float(first.get("start_sec"), start), _float(third.get("start_sec"), start))
    overlap_end = min(_float(first.get("end_sec"), end), _float(third.get("end_sec"), end))
    sync_session_sec = (
        (overlap_start + overlap_end) * 0.5
        if overlap_end >= overlap_start
        else (first_peak + third_peak) * 0.5
    )
    row = dict(event)
    row.update(
        {
            "schema_version": "key_action_dual_view_action_event.v1",
            "event_id": event.get("dual_event_id"),
            "first_person_evidence_id": event.get("first_evidence_id"),
            "third_person_evidence_id": event.get("third_evidence_id"),
            "action_type": event.get("action_family"),
            "action_phase": event.get("action_phase") or "unknown",
            "semantic_action_type": infer_action_type_from_metadata(
                {
                    "primary_object": _object_from_family(str(event.get("object_family") or "")),
                    "interaction_type": "hand_object_contact",
                }
            ),
            "interaction_type": "hand_object_contact",
            "primary_object": _object_from_family(str(event.get("object_family") or "")),
            "primary_object_family": event.get("object_family"),
            "canonical_object_family": canonical_object_family(_object_from_family(str(event.get("object_family") or ""))),
            "start_sec": start,
            "end_sec": end,
            "center_sec": round((start + end) / 2.0, 6),
            "confidence": event.get("alignment_score"),
            "match_score": event.get("alignment_score"),
            "time_delta_sec": round(abs(_float(first.get("center_sec"), start) - _float(third.get("center_sec"), start)), 6),
            "center_delta_sec": event.get("center_delta_sec"),
            "peak_delta_sec": event.get("peak_delta_sec"),
            "first_peak_sec": round(first_peak, 6),
            "third_peak_sec": round(third_peak, 6),
            "signed_peak_delta_sec": round(third_peak - first_peak, 6),
            "synchronized_session_sec": round(sync_session_sec, 6),
            "action_peak_alignment_required": _require_peak_alignment(),
            "overlap_sec": round(max(0.0, min(_float(first.get("end_sec"), end), _float(third.get("end_sec"), end)) - max(_float(first.get("start_sec"), start), _float(third.get("start_sec"), start))), 6),
            "status": "matched_dual_view",
            "formal_event_promoted": True,
            "formal_material_allowed": True,
            "required_views": ["first_person", "third_person"],
            "views": {
                "first_person": _event_view_payload(first),
                "third_person": _event_view_payload(third),
            },
            "micro_segment_ids": _micro_ids_for_interval(micro_segments, start, end),
            "parent_segment_ids": _parent_ids_for_interval(micro_segments, start, end),
            "source_modes": sorted(set([*first.get("source_modes", []), *third.get("source_modes", [])])),
            "match_reasons": event.get("decision_trace") or [],
        }
    )
    return row


def _unmatched_event_row(row: dict[str, Any], evidence_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    evidence = evidence_by_id.get(str(row.get("evidence_id"))) or {}
    merged = {**evidence, **row}
    merged.update(
        {
            "schema_version": "key_action_unmatched_view_evidence.v1",
            "unmatched_reason": row.get("reason") or "no_compatible_dual_view_action_evidence",
            "formal_event_promoted": False,
            "formal_event_policy": "single_view_action_evidence_is_debug_only",
            "primary_object": evidence.get("primary_object") or _object_from_family(str(row.get("object_family") or "")),
            "primary_object_family": evidence.get("primary_object_family") or row.get("object_family"),
            "canonical_object_family": evidence.get("canonical_object_family")
            or canonical_object_family(_object_from_family(str(row.get("object_family") or ""))),
            "semantic_action_type": evidence.get("semantic_action_type")
            or infer_action_type_from_metadata(
                {
                    "primary_object": _object_from_family(str(row.get("object_family") or "")),
                    "interaction_type": "hand_object_contact",
                }
            ),
        }
    )
    return merged


def _event_view_payload(evidence: Mapping[str, Any]) -> dict[str, Any]:
    raw_labels = list(evidence.get("raw_labels") or evidence.get("raw_yolo_labels") or [])
    return {
        "evidence_id": evidence.get("evidence_id"),
        "view": evidence.get("view"),
        "action_family": evidence.get("action_family"),
        "action_display_name": evidence.get("action_display_name"),
        "object_family": evidence.get("object_family"),
        "object_display_name": evidence.get("object_display_name"),
        "action_phase": evidence.get("action_phase") or "unknown",
        "start_sec": evidence.get("start_sec"),
        "end_sec": evidence.get("end_sec"),
        "center_sec": evidence.get("center_sec"),
        "peak_sec": evidence.get("peak_sec"),
        "confidence": evidence.get("confidence"),
        "frame_count": evidence.get("frame_count"),
        "interaction_frame_count": evidence.get("interaction_frame_count"),
        "quality_flags": evidence.get("quality_flags") or [],
        "primary_object": evidence.get("primary_object"),
        "primary_object_family": evidence.get("primary_object_family"),
        "canonical_object_family": evidence.get("canonical_object_family"),
        "raw_labels": raw_labels,
        "evidence": {"raw_labels": raw_labels},
        "micro_segment_ids": evidence.get("micro_segment_ids") or [],
    }


def _micro_ids_for_interval(micro_segments: list[Mapping[str, Any]], start: float, end: float) -> list[str]:
    ids = [
        str(micro.get("micro_segment_id"))
        for micro in micro_segments
        if isinstance(micro, Mapping)
        and micro.get("micro_segment_id")
        and _intervals_overlap(start, end, _float(micro.get("start_sec"), 0.0), _float(micro.get("end_sec"), 0.0))
    ]
    return sorted(set(ids))


def _parent_ids_for_interval(micro_segments: list[Mapping[str, Any]], start: float, end: float) -> list[str]:
    ids = [
        str(micro.get("parent_segment_id") or micro.get("segment_id"))
        for micro in micro_segments
        if isinstance(micro, Mapping)
        and (micro.get("parent_segment_id") or micro.get("segment_id"))
        and _intervals_overlap(start, end, _float(micro.get("start_sec"), 0.0), _float(micro.get("end_sec"), 0.0))
    ]
    return sorted(set(ids))


def _intervals_overlap(left_start: float, left_end: float, right_start: float, right_end: float) -> bool:
    return min(left_end, right_end) >= max(left_start, right_start)


def _object_from_family(family: str) -> str:
    return {
        "reagent_bottle_family": "reagent_bottle",
        "sample_bottle_family": "sample_bottle",
        "pipette_family": "pipette",
        "balance_family": "balance",
        "paper_family": "paper",
        "spatula_family": "spatula",
        "container_family": "container",
        "equipment_family": "equipment",
    }.get(family, "object")


__all__ = [
    "ACTION_DISPLAY_NAMES",
    "OBJECT_DISPLAY_NAMES",
    "ViewActionEvidence",
    "DualViewActionEvent",
    "action_family_for_label",
    "object_family_for_label",
    "build_view_action_evidence",
    "match_dual_view_action_events",
    "detected_segments_from_dual_events",
    "build_dual_view_action_alignment",
    "build_dual_view_action_events",
]

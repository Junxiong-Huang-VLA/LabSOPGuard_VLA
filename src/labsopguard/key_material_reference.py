"""Key material reference ledger and step-oriented retrieval helpers.

The canonical evidence asset remains the folder/file package.  This module
adds a machine-readable JSON/JSONL ledger and a rebuildable query layer on top
of existing LabSOPGuard preprocessing outputs.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SCHEMA_KEY_MATERIAL_REFERENCE = "key_material_reference.v1"
SCHEMA_TIME_ALIGNMENT = "time_alignment.v1"
SCHEMA_PHYSICAL_CHANGE = "physical_change.v1"

PHYSICAL_CHANGE_EVENT_TYPES = {
    "hand_object_interaction",
    "object_move",
    "liquid_transfer",
    "panel_operation",
    "container_state_change",
}

LAB_STEP_TOKEN_SYNONYMS = {
    "称量纸": ("paper", "weighing_paper", "hand-paper", "手与paper操作"),
    "天平": ("balance", "scale", "hand-balance", "balance-weighing"),
    "药匙": ("spatula", "hand-spatula", "solid-transfer"),
    "试剂瓶": ("reagent_bottle", "bottle", "hand-bottle"),
    "样品瓶": ("sample_bottle", "bottle", "hand-bottle"),
    "固体称量": ("solid-weighing", "balance-weighing", "hand-paper", "hand-balance", "paper", "balance"),
    "取样": ("spatula", "solid-transfer", "hand-spatula"),
}

LAB_OBJECT_QUERY_TARGETS = {
    "paper": ("称量纸", "纸片", "weighing_paper", "paper", "hand-paper"),
    "balance": ("天平", "电子天平", "balance", "scale", "hand-balance"),
    "spatula": ("药匙", "刮勺", "勺", "spatula", "scoop", "hand-spatula"),
    "reagent_bottle": ("试剂瓶", "试剂", "reagent_bottle", "hand-bottle"),
    "sample_bottle": ("样品瓶", "蓝盖样品瓶", "sample_bottle", "sample_bottle_blue", "hand-bottle"),
}


def write_experiment_reference_outputs(
    *,
    experiment_dir: str | Path,
    experiment_record: Dict[str, Any],
    material_stream: Sequence[Dict[str, Any]],
    preprocessing: Dict[str, Any],
    steps: Optional[Sequence[Dict[str, Any]]] = None,
    segmentation: Optional[Dict[str, Any]] = None,
    formal_library_root: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Write time alignment, key material references, and physical change logs."""
    exp_dir = Path(experiment_dir)
    artifacts_dir = exp_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    experiment_id = str(experiment_record.get("experiment_id") or exp_dir.name)

    time_alignment = build_time_alignment(
        experiment_id=experiment_id,
        experiment_record=experiment_record,
        preprocessing=preprocessing,
    )
    references = build_key_material_references(
        experiment_id=experiment_id,
        experiment_dir=exp_dir,
        experiment_record=experiment_record,
        material_stream=material_stream,
        preprocessing=preprocessing,
        steps=steps or [],
        segmentation=segmentation or _load_json(exp_dir / "artifacts" / "experiment_segmentation.json") or {},
    )
    changes = build_physical_change_log(references)

    time_alignment_path = artifacts_dir / "time_alignment.json"
    references_path = artifacts_dir / "key_material_references.jsonl"
    references_sqlite_path = artifacts_dir / "key_material_references.sqlite"
    manifest_path = artifacts_dir / "key_material_reference_manifest.json"
    changes_path = artifacts_dir / "physical_change_log.jsonl"

    _write_json(time_alignment_path, time_alignment)
    _write_jsonl(references_path, references)
    _write_reference_sqlite(references_sqlite_path, references)
    _write_jsonl(changes_path, changes)

    manifest = {
        "schema_version": "key_material_reference_manifest.v1",
        "experiment_id": experiment_id,
        "created_at": _now(),
        "reference_count": len(references),
        "physical_change_count": len(changes),
        "event_reference_count": sum(1 for item in references if item.get("asset_type") == "event_clip"),
        "key_frame_reference_count": sum(1 for item in references if item.get("asset_type") == "key_frame"),
        "key_clip_reference_count": sum(1 for item in references if item.get("asset_type") == "key_clip"),
        "paths": {
            "time_alignment": str(time_alignment_path),
            "key_material_references": str(references_path),
            "key_material_references_sqlite": str(references_sqlite_path),
            "physical_change_log": str(changes_path),
        },
    }

    formal_package = mirror_to_formal_material_package(
        experiment_dir=exp_dir,
        experiment_record=experiment_record,
        references=references,
        time_alignment=time_alignment,
        physical_changes=changes,
        library_root=formal_library_root,
    )
    if formal_package:
        manifest["formal_material_package"] = formal_package

    _write_json(manifest_path, manifest)
    return {
        "time_alignment": time_alignment,
        "references": references,
        "physical_changes": changes,
        "manifest": manifest,
        "paths": manifest["paths"],
    }


def build_time_alignment(
    *,
    experiment_id: str,
    experiment_record: Dict[str, Any],
    preprocessing: Dict[str, Any],
) -> Dict[str, Any]:
    video_streams = list(preprocessing.get("video_streams") or [])
    if not video_streams:
        video_streams = _video_streams_from_experiment_record(experiment_record)
    session_start_at = (
        experiment_record.get("session_start_time")
        or experiment_record.get("started_at")
        or (experiment_record.get("metadata") or {}).get("session_start_time")
    )
    return {
        "schema_version": SCHEMA_TIME_ALIGNMENT,
        "experiment_id": experiment_id,
        "created_at": _now(),
        "session_start_at": session_start_at,
        "timezone": (experiment_record.get("timezone") or (experiment_record.get("metadata") or {}).get("timezone") or "Asia/Shanghai"),
        "video_streams": [
            {
                "video_index": stream.get("video_index", index),
                "video_asset_id": stream.get("video_asset_id"),
                "camera_id": stream.get("camera_id"),
                "stream_id": stream.get("stream_id"),
                "sync_group": stream.get("sync_group"),
                "file_path": stream.get("recorded_file_path") or stream.get("file_path"),
                "duration_sec": _safe_float(stream.get("duration_sec"), 0.0),
                "global_start_sec": _safe_float(stream.get("start_offset_sec"), 0.0),
                "global_end_sec": _safe_float(stream.get("end_offset_sec"), stream.get("duration_sec") or 0.0),
                "offset_sec": _safe_float((stream.get("sync_profile") or {}).get("offset_sec"), stream.get("start_offset_sec") or 0.0),
                "clock_scale": _safe_float(stream.get("clock_scale"), 1.0),
                "alignment_confidence": _safe_float((stream.get("sync_profile") or {}).get("confidence"), 0.7),
                "offset_source": stream.get("offset_source", "sequential"),
            }
            for index, stream in enumerate(video_streams)
        ],
        "message_alignment_policy": {
            "default_window_before_sec": 90.0,
            "default_window_after_sec": 180.0,
            "fallback_to_segment_search": True,
        },
        "alignment_summary": preprocessing.get("alignment_summary") or {},
    }


def build_key_material_references(
    *,
    experiment_id: str,
    experiment_dir: str | Path,
    experiment_record: Dict[str, Any],
    material_stream: Sequence[Dict[str, Any]],
    preprocessing: Dict[str, Any],
    steps: Sequence[Dict[str, Any]],
    segmentation: Dict[str, Any],
) -> List[Dict[str, Any]]:
    exp_dir = Path(experiment_dir)
    events = list(preprocessing.get("physical_events") or _event_items_from_stream(material_stream))
    asset_packs = {
        str(pack.get("event_id") or ""): pack
        for pack in (preprocessing.get("event_asset_packs") or [])
        if pack.get("event_id")
    }
    event_stream_by_id = {
        str(item.get("event_id") or ""): item
        for item in material_stream
        if item.get("event_id")
    }
    refs: List[Dict[str, Any]] = []
    for event in events:
        event_id = str(event.get("event_id") or event.get("id") or "")
        if not event_id:
            continue
        stream_item = event_stream_by_id.get(event_id, {})
        asset = _event_asset(event, asset_packs.get(event_id), stream_item)
        start = _event_start(event, stream_item)
        end = _event_end(event, stream_item, start)
        step = _best_step_for_time(steps, start, end)
        segment = _segment_for_time(segmentation, start, end)
        objects = _dedupe([
            *(event.get("involved_objects") or []),
            *(event.get("related_detection_classes") or []),
            *(stream_item.get("object_labels") or []),
        ])
        event_type = str(event.get("event_type") or stream_item.get("event_type") or "")
        actions = _dedupe([event_type, *(stream_item.get("detected_activities") or [])])
        key_timestamps = [
            _safe_float(value, None)
            for value in (event.get("key_timestamps") or [start, (start + end) / 2.0, end])
        ]
        key_timestamps = [round(value, 3) for value in key_timestamps if value is not None]
        material_id = str(stream_item.get("item_id") or f"mat_{event_id}")
        source_container = event.get("source_container") or {}
        target_container = event.get("target_container") or {}
        ref = {
            "schema_version": SCHEMA_KEY_MATERIAL_REFERENCE,
            "material_id": material_id,
            "experiment_id": experiment_id,
            "experiment_title": experiment_record.get("title") or experiment_id,
            "experiment_segment_id": segment.get("segment_id"),
            "experiment_segment_index": segment.get("index"),
            "step_id": step.get("step_id"),
            "step_name": step.get("step_name"),
            "event_id": event_id,
            "event_type": event_type,
            "asset_type": "event_clip",
            "start_sec": round(start, 3),
            "end_sec": round(end, 3),
            "duration_sec": round(max(0.0, end - start), 3),
            "key_timestamps": key_timestamps,
            "objects": objects,
            "actions": actions,
            "before_state": _state_payload(event.get("state_before"), source_container),
            "after_state": _state_payload(event.get("state_after"), target_container),
            "source_container": source_container,
            "target_container": target_container,
            "actor_track_id": event.get("actor_track_id"),
            "tool_track_id": event.get("tool_track_id"),
            "primary_track_id": event.get("primary_track_id"),
            "related_tracks": event.get("related_tracks") or event.get("involved_track_ids") or [],
            "clip_path": _relative_path(asset.get("clip_path") or stream_item.get("clip_file_path"), exp_dir),
            "preview_path": _relative_path(asset.get("preview_path") or stream_item.get("preview_path"), exp_dir),
            "keyframe_paths": [_relative_path(path, exp_dir) for path in (asset.get("keyframe_paths") or stream_item.get("keyframe_paths") or [])],
            "evidence_grade": event.get("evidence_grade") or stream_item.get("evidence_grade"),
            "review_status": event.get("review_status") or stream_item.get("review_status") or "candidate",
            "judgement_status": "unreviewed",
            "judgement": None,
            "confidence": _safe_float(event.get("confidence"), stream_item.get("confidence") or 0.0),
            "evidence_summary": event.get("evidence_summary") or stream_item.get("evidence_summary") or "",
            "searchable_text": _text_blob(
                step.get("step_name"),
                event.get("display_name"),
                event.get("stable_name"),
                event_type,
                objects,
                actions,
                event.get("evidence_summary"),
                event.get("direction_status"),
                event.get("state_change_type"),
            ),
            "payload": {"event": event, "stream_item": stream_item},
        }
        refs.append(ref)

    refs.extend(_key_frame_references(experiment_id, exp_dir, experiment_record, preprocessing, steps, segmentation))
    refs.extend(_key_clip_references(experiment_id, exp_dir, experiment_record, preprocessing, steps, segmentation))
    refs = _dedupe_references(refs)
    refs.sort(key=lambda item: (_safe_float(item.get("start_sec"), 0.0), item.get("asset_type", ""), item.get("material_id", "")))
    return refs


def build_physical_change_log(references: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    changes: List[Dict[str, Any]] = []
    for ref in references:
        event_type = str(ref.get("event_type") or "")
        if event_type not in PHYSICAL_CHANGE_EVENT_TYPES:
            continue
        payload = {
            "schema_version": SCHEMA_PHYSICAL_CHANGE,
            "change_id": f"chg_{_stable_id(ref.get('material_id'), ref.get('event_id'), event_type)}",
            "experiment_id": ref.get("experiment_id"),
            "experiment_segment_id": ref.get("experiment_segment_id"),
            "material_id": ref.get("material_id"),
            "event_id": ref.get("event_id"),
            "event_type": event_type,
            "start_sec": ref.get("start_sec"),
            "end_sec": ref.get("end_sec"),
            "subject": _primary_subject(ref),
            "actor": ref.get("actor_track_id") or "gloved_hand",
            "before": ref.get("before_state") or {},
            "after": ref.get("after_state") or {},
            "evidence_material_ids": [ref.get("material_id")],
            "clip_path": ref.get("clip_path"),
            "preview_path": ref.get("preview_path"),
            "confidence": ref.get("confidence"),
        }
        changes.append(payload)
    changes.sort(key=lambda item: (_safe_float(item.get("start_sec"), 0.0), item.get("event_type", "")))
    return changes


def append_reference_items_to_preprocessing(preprocessing: Dict[str, Any], references: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    payload = dict(preprocessing or {})
    stream = list(payload.get("time_anchored_material_stream") or payload.get("video_index") or [])
    existing_ids = {str(item.get("item_id") or item.get("material_id") or "") for item in stream}
    for ref in references:
        material_id = str(ref.get("material_id") or "")
        if not material_id or material_id in existing_ids:
            continue
        stream.append(
            {
                "schema_version": "material_stream.key_reference.v1",
                "item_id": material_id,
                "experiment_id": ref.get("experiment_id"),
                "timestamp_sec": ref.get("start_sec"),
                "start_time_sec": ref.get("start_sec"),
                "end_time_sec": ref.get("end_sec"),
                "event_id": ref.get("event_id"),
                "event_type": ref.get("event_type"),
                "object_labels": ref.get("objects") or [],
                "detected_activities": ref.get("actions") or [],
                "scene_description": ref.get("evidence_summary") or ref.get("step_name") or ref.get("event_type"),
                "clip_id": ref.get("event_id") or material_id,
                "clip_file_path": ref.get("clip_path"),
                "preview_path": ref.get("preview_path"),
                "material_reference_id": material_id,
                "analysis": {"key_material_reference": ref},
            }
        )
        existing_ids.add(material_id)
    stream.sort(key=lambda item: _safe_float(item.get("timestamp_sec") or item.get("start_time_sec"), 0.0))
    payload["time_anchored_material_stream"] = stream
    payload["key_material_reference_count"] = len(references)
    return payload


def material_stream_items_from_references(
    references: Sequence[Dict[str, Any]],
    *,
    existing_item_ids: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    existing = {str(item) for item in (existing_item_ids or []) if str(item)}
    items: List[Dict[str, Any]] = []
    for ref in references:
        material_id = str(ref.get("material_id") or "")
        if not material_id or material_id in existing:
            continue
        items.append(
            {
                "schema_version": "material_stream.key_reference.v1",
                "item_id": material_id,
                "experiment_id": ref.get("experiment_id"),
                "timestamp_sec": ref.get("start_sec"),
                "local_timestamp_sec": ref.get("start_sec"),
                "global_timestamp_sec": ref.get("start_sec"),
                "start_time_sec": ref.get("start_sec"),
                "end_time_sec": ref.get("end_sec"),
                "event_id": ref.get("event_id"),
                "event_type": ref.get("event_type"),
                "material_reference_id": material_id,
                "object_labels": ref.get("objects") or [],
                "detected_objects": ref.get("objects") or [],
                "detected_activities": ref.get("actions") or [],
                "scene_description": ref.get("evidence_summary") or ref.get("step_name") or ref.get("event_type"),
                "clip_id": ref.get("event_id") or material_id,
                "clip_file_path": ref.get("clip_path"),
                "preview_path": ref.get("preview_path"),
                "keyframe_paths": ref.get("keyframe_paths") or [],
                "confidence": ref.get("confidence") or 0.0,
                "is_key_frame": ref.get("asset_type") == "key_frame",
                "key_frame_reason": "key_material_reference" if ref.get("asset_type") == "key_frame" else None,
                "analysis": {"key_material_reference": ref},
                "provenance": {"source": "key_material_reference", "schema_version": SCHEMA_KEY_MATERIAL_REFERENCE},
            }
        )
        existing.add(material_id)
    items.sort(key=lambda item: _safe_float(item.get("timestamp_sec"), 0.0))
    return items


@dataclass
class StepQueryResult:
    experiment_id: str
    step_text: str
    message_video_time_sec: Optional[float]
    search_window: Optional[Dict[str, float]]
    judgement: Dict[str, Any]
    candidates: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "step_text": self.step_text,
            "message_video_time_sec": self.message_video_time_sec,
            "search_window": self.search_window,
            "judgement": self.judgement,
            "candidates": self.candidates,
        }


def query_step_materials(
    *,
    experiment_dir: str | Path,
    step_text: str,
    message_sent_at: Optional[str] = None,
    window_before_sec: Optional[float] = None,
    window_after_sec: Optional[float] = None,
    limit: int = 8,
) -> StepQueryResult:
    exp_dir = Path(experiment_dir)
    references = load_key_material_references(exp_dir)
    experiment = _load_json(exp_dir / "experiment.json") or {}
    alignment = _load_time_alignment(exp_dir)
    message_video_time = map_message_time_to_video_sec(alignment, message_sent_at)
    policy = alignment.get("message_alignment_policy") or {}
    before = float(window_before_sec if window_before_sec is not None else policy.get("default_window_before_sec", 90.0))
    after = float(window_after_sec if window_after_sec is not None else policy.get("default_window_after_sec", 180.0))
    window = None
    if message_video_time is not None:
        window = {
            "start_sec": round(max(0.0, message_video_time - before), 3),
            "end_sec": round(message_video_time + after, 3),
        }
    candidates = _rank_references(references, step_text, window=window, limit=limit)
    if not candidates and window is not None and (policy.get("fallback_to_segment_search") is not False):
        candidates = _rank_references(references, step_text, window=None, limit=limit)
    judgement = judge_step_candidates(step_text, candidates)
    return StepQueryResult(
        experiment_id=str(experiment.get("experiment_id") or exp_dir.name),
        step_text=step_text,
        message_video_time_sec=message_video_time,
        search_window=window,
        judgement=judgement,
        candidates=candidates,
    )


def map_message_time_to_video_sec(alignment: Dict[str, Any], message_sent_at: Optional[str]) -> Optional[float]:
    if not message_sent_at:
        return None
    message_dt = _parse_datetime(message_sent_at)
    session_dt = _parse_datetime(alignment.get("session_start_at"))
    if message_dt is None or session_dt is None:
        return None
    delta = (message_dt - session_dt).total_seconds()
    streams = alignment.get("video_streams") or []
    if streams:
        stream = streams[0]
        offset = _safe_float(stream.get("offset_sec"), stream.get("global_start_sec") or 0.0)
        clock_scale = _safe_float(stream.get("clock_scale"), 1.0) or 1.0
        return round(max(0.0, (delta - offset) / clock_scale), 3)
    return round(max(0.0, delta), 3)


def load_key_material_references(experiment_dir: str | Path) -> List[Dict[str, Any]]:
    exp_dir = Path(experiment_dir)
    candidates = [
        exp_dir / "artifacts" / "key_material_references.jsonl",
        exp_dir / "key_material_references.jsonl",
        exp_dir / "material_references" / "key_material_references.jsonl",
    ]
    for path in candidates:
        if not path.exists() or path.stat().st_size <= 0:
            continue
        rows = _read_jsonl(path)
        if rows:
            return rows
    return []


def _load_time_alignment(experiment_dir: Path) -> Dict[str, Any]:
    for path in (
        experiment_dir / "artifacts" / "time_alignment.json",
        experiment_dir / "time_alignment.json",
        experiment_dir / "material_references" / "time_alignment.json",
    ):
        payload = _load_json(path)
        if isinstance(payload, dict):
            return payload
    return {}


def judge_step_candidates(step_text: str, candidates: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not candidates:
        return {
            "status": "insufficient",
            "label": "证据不足",
            "confidence": 0.0,
            "reason": "未检索到与该步骤匹配的关键素材片段。",
        }
    normalized = _norm_text(step_text)
    top = candidates[0]
    target_objects = _query_target_objects(step_text)
    if target_objects:
        covered = {
            target
            for target in target_objects
            if any(target in _reference_target_tokens(item) for item in candidates)
        }
        missing = [target for target in target_objects if target not in covered]
        if missing:
            return {
                "status": "insufficient",
                "label": "证据不足",
                "confidence": 0.25,
                "reason": f"已检索到部分相关素材，但缺少目标对象证据：{', '.join(missing)}。",
                "evidence_material_id": top.get("material_id"),
                "evidence_time_range": [top.get("start_sec"), top.get("end_sec")],
                "covered_objects": sorted(covered),
                "missing_objects": missing,
            }
    move_like = any(str(item.get("event_type")) == "object_move" for item in candidates)
    return_like = any(token in normalized for token in ("归位", "放回", "原位", "复位", "return", "restore"))
    for item in candidates:
        before = item.get("before_state") or {}
        after = item.get("after_state") or {}
        before_zone = _zone_from_state(before)
        after_zone = _zone_from_state(after)
        if return_like and before_zone and after_zone and before_zone != after_zone:
            return {
                "status": "incorrect",
                "label": "不符合要求",
                "confidence": max(0.55, _safe_float(item.get("confidence"), 0.55)),
                "reason": f"检索到目标对象操作前位于 {before_zone}，操作后位于 {after_zone}，不满足归位/原位保持要求。",
                "evidence_material_id": item.get("material_id"),
                "evidence_time_range": [item.get("start_sec"), item.get("end_sec")],
            }
    return {
        "status": "correct" if top else "insufficient",
        "label": "符合要求" if top else "证据不足",
        "confidence": max(0.45, _safe_float(top.get("confidence"), 0.45)),
        "reason": (
            "已检索到与步骤匹配的关键素材，未发现规则层面的显著异常。"
            if not (return_like and move_like)
            else "已检索到归位相关素材，未发现超出当前规则阈值的区域变化。"
        ),
        "evidence_material_id": top.get("material_id"),
        "evidence_time_range": [top.get("start_sec"), top.get("end_sec")],
    }


def mirror_to_formal_material_package(
    *,
    experiment_dir: Path,
    experiment_record: Dict[str, Any],
    references: Sequence[Dict[str, Any]],
    time_alignment: Optional[Dict[str, Any]] = None,
    physical_changes: Optional[Sequence[Dict[str, Any]]] = None,
    library_root: Optional[str | Path] = None,
) -> Optional[Dict[str, Any]]:
    if not references:
        return None
    root = Path(library_root) if library_root else experiment_dir.parents[1] / "material_references"
    title = str(experiment_record.get("title") or experiment_record.get("experiment_id") or experiment_dir.name)
    package_dir = root / _safe_filename(title)
    frames_dir = package_dir / "关键帧"
    clips_dir = package_dir / "关键片段"
    reports_dir = package_dir / "专业报告"
    frames_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    material_index_rows: List[Dict[str, Any]] = []
    package_refs: List[Dict[str, Any]] = []
    for ref in references:
        package_ref = dict(ref)
        copied_clip = _copy_asset(ref.get("clip_path"), experiment_dir, clips_dir, ref.get("material_id"), ".mp4")
        copied_preview = _copy_asset(ref.get("preview_path"), experiment_dir, frames_dir, ref.get("material_id"), ".jpg")
        if copied_clip:
            package_ref["formal_clip_path"] = _relative_path(copied_clip, package_dir)
        if copied_preview:
            package_ref["formal_preview_path"] = _relative_path(copied_preview, package_dir)
        package_refs.append(package_ref)
        if copied_clip or copied_preview:
            material_index_rows.append(
                {
                    "material_id": ref.get("material_id"),
                    "asset_type": ref.get("asset_type"),
                    "display_name": ref.get("step_name") or ref.get("event_type") or ref.get("material_id"),
                    "event_type": ref.get("event_type"),
                    "start_sec": ref.get("start_sec"),
                    "end_sec": ref.get("end_sec"),
                    "objects": _reference_objects(ref),
                    "actions": _reference_actions(ref),
                    "stored_file": package_ref.get("formal_clip_path") or package_ref.get("formal_preview_path"),
                    "clip_path": package_ref.get("formal_clip_path"),
                    "preview_path": package_ref.get("formal_preview_path"),
                    "judgement_status": ref.get("judgement_status"),
                    "payload": ref,
                }
            )

    _write_jsonl(package_dir / "key_material_references.jsonl", package_refs)
    _write_jsonl(package_dir / "physical_change_log.jsonl", list(physical_changes or []))
    _write_json(package_dir / "time_alignment.json", time_alignment or {})
    _write_reference_sqlite(package_dir / "key_material_references.sqlite", package_refs)
    _write_json(package_dir / "素材索引.json", {"schema_version": "material_index.v1", "records": material_index_rows})
    _write_jsonl(package_dir / "素材索引.jsonl", material_index_rows)
    portable_entrypoints = {
        "manifest": "manifest.json",
        "evidence_package_manifest": "evidence_package_manifest.json",
        "material_index_json": "素材索引.json",
        "material_index_jsonl": "素材索引.jsonl",
        "key_material_references_jsonl": "key_material_references.jsonl",
        "key_material_references_sqlite": "key_material_references.sqlite",
        "physical_change_log_jsonl": "physical_change_log.jsonl",
        "time_alignment_json": "time_alignment.json",
    }
    _write_json(
        package_dir / "manifest.json",
        {
            "schema_version": "formal_material_package.v1",
            "experiment_id": experiment_record.get("experiment_id"),
            "title": title,
            "created_at": _now(),
            "reference_count": len(package_refs),
            "stored_material_count": len(material_index_rows),
            "folders": ["关键片段", "关键帧", "专业报告"],
        },
    )
    package_id = _stable_id(experiment_record.get("experiment_id"), title, len(package_refs), len(material_index_rows))
    _write_json(
        package_dir / "evidence_package_manifest.json",
        {
            "schema_version": "evidence_package_manifest.v1",
            "package_id": package_id,
            "experiment_id": experiment_record.get("experiment_id"),
            "title": title,
            "created_at": _now(),
            "path_mode": "relative_to_package_root",
            "portable": True,
            "reference_count": len(package_refs),
            "stored_material_count": len(material_index_rows),
            "physical_change_count": len(physical_changes or []),
            "entrypoints": portable_entrypoints,
            "asset_roots": {
                "key_clips": clips_dir.name,
                "key_frames": frames_dir.name,
                "reports": reports_dir.name,
            },
            "consumer_contract": {
                "resolve_paths_by": "package_root_plus_relative_path",
                "do_not_require_original_source_paths": True,
                "preferred_query_files": [
                    "key_material_references.jsonl",
                    "physical_change_log.jsonl",
                    "time_alignment.json",
                    "素材索引.jsonl",
                ],
            },
        },
    )
    readme = package_dir / "README.md"
    if not readme.exists():
        readme.write_text(
            f"# {title}\n\n本目录保存实验关键帧、关键片段、专业报告及可检索素材引用总账。\n",
            encoding="utf-8",
        )
    return {
        "package_dir": str(package_dir),
        "reference_count": len(package_refs),
        "stored_material_count": len(material_index_rows),
        "path_mode": "relative_to_package_root",
        "entrypoints": portable_entrypoints,
    }


def _rank_references(
    references: Sequence[Dict[str, Any]],
    step_text: str,
    *,
    window: Optional[Dict[str, float]],
    limit: int,
) -> List[Dict[str, Any]]:
    tokens = _tokens(step_text)
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for ref in references:
        score = 0.0
        text_blob = _norm_text(ref.get("searchable_text") or _text_blob(ref))
        for token in tokens:
            if token and token in text_blob:
                score += 2.0
        if window is not None:
            start = _safe_float(ref.get("start_sec"), 0.0)
            end = _safe_float(ref.get("end_sec"), start)
            if end >= window["start_sec"] and start <= window["end_sec"]:
                score += 4.0
            else:
                center = (start + end) / 2.0
                distance = min(abs(center - window["start_sec"]), abs(center - window["end_sec"]))
                score += max(0.0, 1.5 - distance / 300.0)
        if ref.get("step_name") and str(ref.get("step_name")) in step_text:
            score += 3.0
        if ref.get("asset_type") == "event_clip":
            score += 0.8
        score += min(1.0, _safe_float(ref.get("confidence"), 0.0))
        if score > 0:
            enriched = dict(ref)
            enriched["retrieval_score"] = round(score, 4)
            scored.append((score, enriched))
    scored.sort(key=lambda pair: (pair[0], -_safe_float(pair[1].get("start_sec"), 0.0)), reverse=True)
    return [item for _, item in scored[: max(1, min(int(limit), 50))]]


def _query_target_objects(step_text: str) -> List[str]:
    compact = _norm_text(step_text)
    targets: List[str] = []
    for canonical, aliases in LAB_OBJECT_QUERY_TARGETS.items():
        if canonical in compact or any(_norm_text(alias) in compact for alias in aliases):
            targets.append(canonical)
    return _dedupe(targets)


def _reference_objects(ref: Dict[str, Any]) -> List[str]:
    return _dedupe(
        [
            *(ref.get("objects") or []),
            *(ref.get("object_labels") or []),
            ref.get("primary_object"),
            ref.get("canonical_object"),
            *(ref.get("secondary_objects") or []),
        ]
    )


def _reference_actions(ref: Dict[str, Any]) -> List[str]:
    return _dedupe(
        [
            *(ref.get("actions") or []),
            ref.get("action_name"),
            ref.get("canonical_action_type"),
            *(ref.get("secondary_actions") or []),
        ]
    )


def _reference_target_tokens(ref: Dict[str, Any]) -> set[str]:
    blob = _norm_text(
        _text_blob(
            _reference_objects(ref),
            _reference_actions(ref),
            ref.get("searchable_text"),
            ref.get("event_type"),
            ref.get("asset_kind"),
        )
    )
    tokens = set(_tokens(blob))
    for canonical, aliases in LAB_OBJECT_QUERY_TARGETS.items():
        if canonical in blob or any(_norm_text(alias) in blob for alias in aliases):
            tokens.add(canonical)
    return tokens


def _video_streams_from_experiment_record(experiment_record: Dict[str, Any]) -> List[Dict[str, Any]]:
    streams: List[Dict[str, Any]] = []
    descriptors = experiment_record.get("video_inputs") or []
    if not descriptors:
        paths = experiment_record.get("video_paths") or []
        descriptors = [{"video_path": path, "video_index": index} for index, path in enumerate(paths)]
    for index, item in enumerate(descriptors):
        path = item.get("video_path") if isinstance(item, dict) else str(item)
        streams.append(
            {
                "video_index": item.get("video_index", index) if isinstance(item, dict) else index,
                "video_asset_id": item.get("video_asset_id") if isinstance(item, dict) else None,
                "camera_id": item.get("camera_id") if isinstance(item, dict) else None,
                "stream_id": item.get("stream_id") if isinstance(item, dict) else None,
                "file_path": path,
                "duration_sec": item.get("duration_sec", 0.0) if isinstance(item, dict) else 0.0,
                "start_offset_sec": item.get("start_offset_sec", 0.0) if isinstance(item, dict) else 0.0,
                "end_offset_sec": item.get("end_offset_sec", item.get("duration_sec", 0.0)) if isinstance(item, dict) else 0.0,
                "clock_scale": item.get("clock_scale", 1.0) if isinstance(item, dict) else 1.0,
                "offset_source": item.get("offset_source", "explicit") if isinstance(item, dict) else "explicit",
            }
        )
    return streams


def _event_items_from_stream(material_stream: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    events = []
    for item in material_stream:
        if item.get("event_id") or item.get("schema_version") == "material_stream.event.v1":
            events.append((item.get("analysis") or {}).get("event_preprocessing") or item)
    return events


def _event_asset(event: Dict[str, Any], asset_pack: Optional[Dict[str, Any]], stream_item: Dict[str, Any]) -> Dict[str, Any]:
    asset = event.get("asset_pack") if isinstance(event.get("asset_pack"), dict) else {}
    merged = {}
    for source in (asset, asset_pack or {}, stream_item):
        for key in ("clip_path", "clip_file_path", "preview_path", "keyframe_paths"):
            value = source.get(key)
            if value and key not in merged:
                merged["clip_path" if key == "clip_file_path" else key] = value
    return merged


def _key_frame_references(
    experiment_id: str,
    exp_dir: Path,
    experiment_record: Dict[str, Any],
    preprocessing: Dict[str, Any],
    steps: Sequence[Dict[str, Any]],
    segmentation: Dict[str, Any],
) -> List[Dict[str, Any]]:
    refs = []
    for frame in preprocessing.get("key_frames") or []:
        ts = _safe_float(frame.get("timestamp_sec"), 0.0)
        step = _best_step_for_time(steps, ts, ts)
        segment = _segment_for_time(segmentation, ts, ts)
        material_id = f"kf_{_stable_id(frame.get('video_asset_id'), frame.get('frame_id'), ts)}"
        objects = _dedupe([*(frame.get("object_labels") or []), *(frame.get("detected_objects") or [])])
        actions = _dedupe(frame.get("detected_activities") or [])
        refs.append(
            {
                "schema_version": SCHEMA_KEY_MATERIAL_REFERENCE,
                "material_id": material_id,
                "experiment_id": experiment_id,
                "experiment_title": experiment_record.get("title") or experiment_id,
                "experiment_segment_id": segment.get("segment_id"),
                "experiment_segment_index": segment.get("index"),
                "step_id": step.get("step_id"),
                "step_name": step.get("step_name"),
                "event_id": None,
                "event_type": "key_frame",
                "asset_type": "key_frame",
                "start_sec": round(ts, 3),
                "end_sec": round(ts, 3),
                "duration_sec": 0.0,
                "key_timestamps": [round(ts, 3)],
                "objects": objects,
                "actions": actions,
                "before_state": {},
                "after_state": {},
                "clip_path": None,
                "preview_path": _relative_path(frame.get("frame_bgr_path"), exp_dir),
                "keyframe_paths": [_relative_path(frame.get("frame_bgr_path"), exp_dir)] if frame.get("frame_bgr_path") else [],
                "evidence_grade": None,
                "review_status": "candidate",
                "judgement_status": "unreviewed",
                "judgement": None,
                "confidence": _safe_float(frame.get("confidence"), 0.5),
                "evidence_summary": frame.get("scene_description") or frame.get("key_frame_reason") or "",
                "searchable_text": _text_blob(step.get("step_name"), objects, actions, frame.get("scene_description"), frame.get("key_frame_reason")),
                "payload": {"key_frame": frame},
            }
        )
    return refs


def _key_clip_references(
    experiment_id: str,
    exp_dir: Path,
    experiment_record: Dict[str, Any],
    preprocessing: Dict[str, Any],
    steps: Sequence[Dict[str, Any]],
    segmentation: Dict[str, Any],
) -> List[Dict[str, Any]]:
    refs = []
    for clip in preprocessing.get("key_clips") or []:
        start = _safe_float(clip.get("start_time_sec"), 0.0)
        end = _safe_float(clip.get("end_time_sec"), start)
        step = _best_step_for_time(steps, start, end)
        segment = _segment_for_time(segmentation, start, end)
        material_id = f"kc_{_stable_id(clip.get('clip_id'), start, end)}"
        refs.append(
            {
                "schema_version": SCHEMA_KEY_MATERIAL_REFERENCE,
                "material_id": material_id,
                "experiment_id": experiment_id,
                "experiment_title": experiment_record.get("title") or experiment_id,
                "experiment_segment_id": segment.get("segment_id"),
                "experiment_segment_index": segment.get("index"),
                "step_id": step.get("step_id"),
                "step_name": step.get("step_name"),
                "event_id": None,
                "event_type": "key_clip",
                "asset_type": "key_clip",
                "start_sec": round(start, 3),
                "end_sec": round(end, 3),
                "duration_sec": round(max(0.0, end - start), 3),
                "key_timestamps": [round(_safe_float(clip.get("anchor_timestamp_sec"), start), 3)],
                "objects": [],
                "actions": [str(clip.get("reason") or "key_clip")],
                "before_state": {},
                "after_state": {},
                "clip_path": _relative_path(clip.get("file_path"), exp_dir),
                "preview_path": _relative_path(clip.get("key_frame_path"), exp_dir),
                "keyframe_paths": [_relative_path(clip.get("key_frame_path"), exp_dir)] if clip.get("key_frame_path") else [],
                "evidence_grade": None,
                "review_status": "candidate",
                "judgement_status": "unreviewed",
                "judgement": None,
                "confidence": 0.45,
                "evidence_summary": str(clip.get("reason") or "key clip"),
                "searchable_text": _text_blob(step.get("step_name"), clip.get("reason"), clip.get("clip_id")),
                "payload": {"key_clip": clip},
            }
        )
    return refs


def _best_step_for_time(steps: Sequence[Dict[str, Any]], start: float, end: float) -> Dict[str, Any]:
    best: Tuple[float, Dict[str, Any]] = (0.0, {})
    for step in steps:
        s = _safe_float(step.get("start_time_sec"), 0.0)
        e = _safe_float(step.get("end_time_sec"), s)
        overlap = max(0.0, min(end, e) - max(start, s))
        if start <= e and end >= s:
            overlap = max(overlap, 0.001)
        if overlap > best[0]:
            best = (overlap, step)
    return best[1]


def _segment_for_time(segmentation: Dict[str, Any], start: float, end: float) -> Dict[str, Any]:
    for segment in segmentation.get("segments") or []:
        s = _safe_float(segment.get("start_sec"), 0.0)
        e = _safe_float(segment.get("end_sec"), s)
        if end >= s and start <= e:
            return segment
    return {"segment_id": "seg_0", "index": 0}


def _event_start(event: Dict[str, Any], stream_item: Dict[str, Any]) -> float:
    return _safe_float(event.get("start_time_sec"), event.get("timestamp_sec") or stream_item.get("time_start") or stream_item.get("timestamp_sec") or 0.0)


def _event_end(event: Dict[str, Any], stream_item: Dict[str, Any], start: float) -> float:
    value = event.get("end_time_sec")
    if value is None:
        value = event.get("end_timestamp_sec")
    if value is None:
        value = stream_item.get("time_end")
    if value is None:
        value = start + _safe_float(event.get("duration_sec"), stream_item.get("duration_sec") or 0.0)
    return max(start, _safe_float(value, start))


def _state_payload(state: Any, container: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    if isinstance(state, dict):
        payload.update(state)
    elif state:
        payload["state"] = state
    if container:
        payload.setdefault("object_name", container.get("object_name") or container.get("class_name"))
        payload.setdefault("track_id", container.get("track_id"))
        if container.get("bbox") is not None:
            payload.setdefault("bbox", container.get("bbox"))
        if container.get("zone") is not None:
            payload.setdefault("zone", container.get("zone"))
    return payload


def _primary_subject(ref: Dict[str, Any]) -> Optional[str]:
    for value in ref.get("objects") or []:
        if "hand" not in str(value).lower() and "glove" not in str(value).lower():
            return str(value)
    return (ref.get("objects") or [None])[0]


def _zone_from_state(state: Dict[str, Any]) -> Optional[str]:
    if not isinstance(state, dict):
        return None
    for key in ("zone", "region", "area", "location"):
        if state.get(key):
            return str(state[key])
    nested = state.get("position") if isinstance(state.get("position"), dict) else {}
    for key in ("zone", "region", "area"):
        if nested.get(key):
            return str(nested[key])
    return None


def _dedupe_references(refs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    output = []
    for ref in refs:
        key = (ref.get("material_id"), ref.get("asset_type"))
        if key in seen:
            continue
        seen.add(key)
        output.append(ref)
    return output


def _copy_asset(path_value: Any, experiment_dir: Path, target_dir: Path, material_id: Any, suffix: str) -> Optional[Path]:
    path = _resolve_path(path_value, experiment_dir)
    if path is None or not path.exists() or not path.is_file():
        return None
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{_safe_filename(str(material_id or path.stem))}{path.suffix or suffix}"
    shutil.copy2(path, target)
    return target


def _resolve_path(path_value: Any, experiment_dir: Path) -> Optional[Path]:
    if not path_value:
        return None
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return (experiment_dir / path).resolve()


def _relative_path(path_value: Any, root: Path) -> Optional[str]:
    path = _resolve_path(path_value, root)
    if path is None:
        return None
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return str(path)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_reference_sqlite(path: Path, refs: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS key_material_refs (
                material_id TEXT PRIMARY KEY,
                experiment_id TEXT,
                experiment_segment_id TEXT,
                step_id TEXT,
                step_name TEXT,
                event_id TEXT,
                event_type TEXT,
                asset_type TEXT,
                start_sec REAL,
                end_sec REAL,
                objects_json TEXT,
                actions_json TEXT,
                clip_path TEXT,
                preview_path TEXT,
                judgement_status TEXT,
                evidence_grade TEXT,
                review_status TEXT,
                searchable_text TEXT,
                payload_json TEXT
            )
            """
        )
        for statement in (
            "CREATE INDEX IF NOT EXISTS idx_key_refs_time ON key_material_refs(start_sec, end_sec)",
            "CREATE INDEX IF NOT EXISTS idx_key_refs_step ON key_material_refs(step_id, step_name)",
            "CREATE INDEX IF NOT EXISTS idx_key_refs_event ON key_material_refs(event_type, asset_type)",
        ):
            conn.execute(statement)
        try:
            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS key_material_refs_fts USING fts5(material_id UNINDEXED, searchable_text)"
            )
            has_fts = True
        except sqlite3.OperationalError:
            has_fts = False
        conn.execute("DELETE FROM key_material_refs")
        if has_fts:
            conn.execute("DELETE FROM key_material_refs_fts")
        for ref in refs:
            row = {
                "material_id": str(ref.get("material_id") or _stable_id(ref)),
                "experiment_id": ref.get("experiment_id"),
                "experiment_segment_id": ref.get("experiment_segment_id"),
                "step_id": ref.get("step_id"),
                "step_name": ref.get("step_name"),
                "event_id": ref.get("event_id"),
                "event_type": ref.get("event_type"),
                "asset_type": ref.get("asset_type"),
                "start_sec": _safe_float(ref.get("start_sec"), 0.0),
                "end_sec": _safe_float(ref.get("end_sec"), 0.0),
                "objects_json": json.dumps(_reference_objects(ref), ensure_ascii=False),
                "actions_json": json.dumps(_reference_actions(ref), ensure_ascii=False),
                "clip_path": ref.get("clip_path"),
                "preview_path": ref.get("preview_path"),
                "judgement_status": ref.get("judgement_status"),
                "evidence_grade": ref.get("evidence_grade"),
                "review_status": ref.get("review_status"),
                "searchable_text": ref.get("searchable_text") or _text_blob(ref),
                "payload_json": json.dumps(ref, ensure_ascii=False),
            }
            conn.execute(
                """
                INSERT OR REPLACE INTO key_material_refs
                (material_id, experiment_id, experiment_segment_id, step_id, step_name, event_id, event_type,
                 asset_type, start_sec, end_sec, objects_json, actions_json, clip_path, preview_path,
                 judgement_status, evidence_grade, review_status, searchable_text, payload_json)
                VALUES (:material_id, :experiment_id, :experiment_segment_id, :step_id, :step_name, :event_id,
                 :event_type, :asset_type, :start_sec, :end_sec, :objects_json, :actions_json, :clip_path,
                 :preview_path, :judgement_status, :evidence_grade, :review_status, :searchable_text, :payload_json)
                """,
                row,
            )
            if has_fts:
                conn.execute("INSERT INTO key_material_refs_fts(material_id, searchable_text) VALUES (?, ?)", (row["material_id"], row["searchable_text"]))
        conn.commit()
    finally:
        conn.close()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def _dedupe(values: Iterable[Any]) -> List[str]:
    output: List[str] = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def _text_blob(*parts: Any) -> str:
    values: List[str] = []
    for part in parts:
        if isinstance(part, dict):
            values.append(_text_blob(*part.values()))
        elif isinstance(part, (list, tuple, set)):
            values.append(_text_blob(*part))
        elif part is not None:
            text = str(part).strip()
            if text:
                values.append(text)
    return " ".join(values)


def _tokens(text: str) -> List[str]:
    original = str(text or "").strip().lower().replace("-", "_")
    tokens = [
        _norm_text(token)
        for token in re.split(r"[\s,，。:：;；/\\|]+", original)
        if _norm_text(token)
    ]
    compact = _norm_text(original)
    for trigger, synonyms in LAB_STEP_TOKEN_SYNONYMS.items():
        if _norm_text(trigger) in compact:
            for synonym in synonyms:
                normalized_synonym = _norm_text(synonym)
                if normalized_synonym and normalized_synonym not in tokens:
                    tokens.append(normalized_synonym)
    for token in re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]+", original):
        normalized = _norm_text(token)
        if normalized and normalized not in tokens:
            tokens.append(normalized)
    if compact and compact not in tokens:
        tokens.append(compact)
    return tokens


def _norm_text(text: Any) -> str:
    return str(text or "").strip().lower().replace("-", "_").replace(" ", "")


def _safe_float(value: Any, default: Any = 0.0) -> Any:
    try:
        return float(value)
    except Exception:
        return default


def _parse_datetime(value: Any) -> Optional[datetime]:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_id(*parts: Any) -> str:
    return hashlib.sha1("|".join(str(part) for part in parts if part is not None).encode("utf-8")).hexdigest()[:12]


def _safe_filename(value: str) -> str:
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", str(value or "").strip())
    text = text.strip(" ._")
    return text[:96] if text else "unnamed"


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 10000):
        candidate = path.with_name(f"{path.stem}_{index:02d}{path.suffix}")
        if not candidate.exists():
            return candidate
    return path.with_name(f"{path.stem}_{_stable_id(_now())}{path.suffix}")

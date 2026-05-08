from __future__ import annotations

import copy
import json
import shutil
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .schemas import read_jsonl, write_jsonl
from .vector_index import VectorIndex


REVIEW_STATE_FILENAME = "key_action_review_state.json"
REVIEWED_DATASET_SCHEMA_VERSION = "key_action_reviewed_dataset.v1"
REVIEWED_MANIFEST_FILENAME = "reviewed_dataset_manifest.json"
REVIEWED_EXPORT_FILENAME = "reviewed_export.json"
REVIEWED_SEGMENTS_FILENAME = "reviewed_segments.jsonl"
REVIEWED_MICROS_FILENAME = "reviewed_micro_segments.jsonl"
REVIEWED_EVIDENCE_FILENAME = "reviewed_evidence.jsonl"
REVIEWED_VECTOR_METADATA_FILENAME = "reviewed_vector_metadata.jsonl"
REVIEWED_RELEASES_DIRNAME = "reviewed_releases"
LATEST_REVIEWED_RELEASE_FILENAME = "latest_reviewed_release.json"
REVIEWED_RELEASE_MANIFEST_FILENAME = "reviewed_release_manifest.json"


def freeze_reviewed_dataset(session_dir: str | Path, *, create_release: bool = True) -> dict[str, Any]:
    """Materialize review decisions into a stable delivery dataset.

    Raw detector artifacts remain immutable. Approved, pending, and needs-review
    rows are copied into reviewed artifacts with explicit review metadata; rejected
    segment or micro rows are excluded from reviewed data products.
    """

    session = Path(session_dir)
    metadata = session / "metadata"
    state = _load_review_state(session)
    decisions = _latest_decisions(state)

    source_segments = _read_jsonl(metadata / "key_action_segments.jsonl")
    source_micros = _read_jsonl(metadata / "micro_segments.jsonl")
    source_segment_vectors = _read_jsonl(metadata / "vector_metadata.jsonl")
    source_micro_vectors = _read_jsonl(metadata / "micro_vector_metadata.jsonl")

    reviewed_micros = _review_rows(source_micros, decisions, item_type="micro", id_key="micro_segment_id")
    convergence = _auto_converge_segments(source_segments, reviewed_micros, decisions)
    reviewed_segments = convergence["segments"]
    reviewed_micros = convergence["micro_segments"]
    reviewed_vectors = (
        _rows_to_vector_metadata(reviewed_segments, reviewed_micros)
        if convergence["applied"]
        else _review_vector_rows([*source_segment_vectors, *source_micro_vectors], decisions)
    )
    if not reviewed_vectors:
        reviewed_vectors = _rows_to_vector_metadata(reviewed_segments, reviewed_micros)

    reviewed_evidence = _review_evidence_rows(
        session,
        decisions,
        rejected_segments=_rejected_ids(decisions, "segment:"),
        rejected_micros=_rejected_ids(decisions, "micro:"),
    )

    metadata.mkdir(parents=True, exist_ok=True)
    write_jsonl(metadata / REVIEWED_SEGMENTS_FILENAME, reviewed_segments)
    write_jsonl(metadata / REVIEWED_MICROS_FILENAME, reviewed_micros)
    write_jsonl(metadata / REVIEWED_EVIDENCE_FILENAME, reviewed_evidence)
    write_jsonl(metadata / REVIEWED_VECTOR_METADATA_FILENAME, reviewed_vectors)

    reviewed_index = _build_reviewed_index(session, reviewed_vectors)
    reviewed_metrics = _timeline_metrics(reviewed_segments, reviewed_micros, _source_extent(source_segments, reviewed_micros))
    decision_counts = Counter(str(row.get("decision") or "pending") for row in decisions.values())
    row_review_counts = Counter(str(row.get("review_status") or "pending") for row in [*reviewed_segments, *reviewed_micros])
    now = _now()
    manifest = {
        "schema_version": REVIEWED_DATASET_SCHEMA_VERSION,
        "generated_at": now,
        "session_dir": str(session),
        "state_path": str(metadata / REVIEW_STATE_FILENAME),
        "review_state_updated_at": state.get("updated_at"),
        "auto_convergence": {
            "applied": convergence["applied"],
            "reason": convergence["reason"],
            "source_segment_count": len(source_segments),
            "micro_window_segment_count": convergence["micro_window_segment_count"],
            "manual_review_gap_count": len(convergence["manual_review_gaps"]),
            "manual_review_gaps": convergence["manual_review_gaps"],
            "metrics": reviewed_metrics,
        },
        "source_counts": {
            "segments": len(source_segments),
            "micro_segments": len(source_micros),
            "segment_vectors": len(source_segment_vectors),
            "micro_vectors": len(source_micro_vectors),
        },
        "reviewed_counts": {
            "segments": len(reviewed_segments),
            "micro_segments": len(reviewed_micros),
            "evidence": len(reviewed_evidence),
            "vectors": len(reviewed_vectors),
        },
        "reviewed_metrics": reviewed_metrics,
        "decision_counts": dict(sorted(decision_counts.items())),
        "row_review_status_counts": dict(sorted(row_review_counts.items())),
        "rejected_item_ids": sorted(
            item_id
            for item_id, decision in decisions.items()
            if str(decision.get("decision") or "") == "rejected"
        ),
        "boundary_adjustment_count": sum(
            1
            for decision in decisions.values()
            if decision.get("boundary_start_sec") is not None or decision.get("boundary_end_sec") is not None
        ),
        "delivery_ready": bool(row_review_counts and row_review_counts.get("pending", 0) == 0),
        "artifacts": {
            "reviewed_segments": str(metadata / REVIEWED_SEGMENTS_FILENAME),
            "reviewed_micro_segments": str(metadata / REVIEWED_MICROS_FILENAME),
            "reviewed_evidence": str(metadata / REVIEWED_EVIDENCE_FILENAME),
            "reviewed_vector_metadata": str(metadata / REVIEWED_VECTOR_METADATA_FILENAME),
            "reviewed_index": str(reviewed_index),
            "reviewed_export": str(metadata / REVIEWED_EXPORT_FILENAME),
        },
    }
    (metadata / REVIEWED_MANIFEST_FILENAME).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    export_payload = {
        "schema_version": "key_action_reviewed_export.v1",
        "generated_at": now,
        "session_dir": str(session),
        "manifest": manifest,
        "segments": reviewed_segments,
        "micro_segments": reviewed_micros,
        "evidence": reviewed_evidence,
    }
    (metadata / REVIEWED_EXPORT_FILENAME).write_text(json.dumps(export_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    release = _create_reviewed_release(
        session,
        manifest=manifest,
        reviewed_segments=reviewed_segments,
        reviewed_micros=reviewed_micros,
        reviewed_evidence=reviewed_evidence,
        reviewed_vectors=reviewed_vectors,
        reviewed_index=reviewed_index,
    ) if create_release else None
    if release:
        manifest["release"] = release
        (metadata / REVIEWED_MANIFEST_FILENAME).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["export_path"] = str(metadata / REVIEWED_EXPORT_FILENAME)
    return manifest


def reviewed_index_dir(session_dir: str | Path) -> Path:
    session = Path(session_dir)
    latest = _latest_release_manifest(session)
    if latest:
        release_index = Path(str(latest.get("release_dir") or "")) / "reviewed_index"
        if (release_index / "fallback_index.pkl").exists():
            return release_index
    reviewed = session / "reviewed_index"
    if (reviewed / "fallback_index.pkl").exists():
        return reviewed
    return session / "index"


def reviewed_metadata_path(session_dir: str | Path, file_name: str) -> Path:
    session = Path(session_dir)
    metadata = session / "metadata"
    reviewed_name = {
        "key_action_segments.jsonl": REVIEWED_SEGMENTS_FILENAME,
        "micro_segments.jsonl": REVIEWED_MICROS_FILENAME,
        "vector_metadata.jsonl": REVIEWED_VECTOR_METADATA_FILENAME,
        "micro_vector_metadata.jsonl": REVIEWED_VECTOR_METADATA_FILENAME,
    }.get(file_name)
    latest = _latest_release_manifest(session)
    if reviewed_name and latest:
        candidate = Path(str(latest.get("release_dir") or "")) / "artifacts" / reviewed_name
        if candidate.exists():
            return candidate
    if reviewed_name and (metadata / reviewed_name).exists():
        return metadata / reviewed_name
    return metadata / file_name


def latest_reviewed_release(session_dir: str | Path) -> dict[str, Any] | None:
    return _latest_release_manifest(Path(session_dir))


def rollback_reviewed_release(session_dir: str | Path, version: str | None = None) -> dict[str, Any]:
    session = Path(session_dir)
    releases_dir = session / REVIEWED_RELEASES_DIRNAME
    if version:
        release_dir = releases_dir / version
    else:
        releases = sorted([path for path in releases_dir.glob("v*") if path.is_dir()])
        if len(releases) < 2:
            raise ValueError("no previous reviewed release is available")
        release_dir = releases[-2]
    manifest_path = release_dir / REVIEWED_RELEASE_MANIFEST_FILENAME
    if not manifest_path.exists():
        raise ValueError(f"reviewed release manifest not found: {release_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    artifacts = release_dir / "artifacts"
    metadata = session / "metadata"
    for file_name in (
        REVIEWED_MANIFEST_FILENAME,
        REVIEWED_EXPORT_FILENAME,
        REVIEWED_SEGMENTS_FILENAME,
        REVIEWED_MICROS_FILENAME,
        REVIEWED_EVIDENCE_FILENAME,
        REVIEWED_VECTOR_METADATA_FILENAME,
    ):
        source = artifacts / file_name
        if source.exists():
            metadata.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, metadata / file_name)
    source_index = release_dir / "reviewed_index"
    target_index = session / "reviewed_index"
    if source_index.exists():
        if target_index.exists():
            shutil.rmtree(target_index)
        shutil.copytree(source_index, target_index)
    latest_payload = {
        "schema_version": "key_action_reviewed_release_pointer.v1",
        "updated_at": _now(),
        "active_version": release_dir.name,
        "release_dir": str(release_dir),
        "rollback": True,
    }
    (releases_dir / LATEST_REVIEWED_RELEASE_FILENAME).write_text(json.dumps(latest_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "rolled_back", "active_version": release_dir.name, "release": manifest, "latest": latest_payload}


def load_reviewed_export(session_dir: str | Path) -> dict[str, Any]:
    session = Path(session_dir)
    path = session / "metadata" / REVIEWED_EXPORT_FILENAME
    if not path.exists():
        freeze_reviewed_dataset(session, create_release=False)
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {"schema_version": "key_action_reviewed_export.error", "session_dir": str(session), "error": "reviewed export is unreadable"}
    return data if isinstance(data, dict) else {"schema_version": "key_action_reviewed_export.error", "session_dir": str(session)}


def _auto_converge_segments(
    source_segments: list[Mapping[str, Any]],
    reviewed_micros: list[dict[str, Any]],
    decisions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    reviewed_segments = _review_rows(source_segments, decisions, item_type="segment", id_key="segment_id")
    extent = _source_extent(source_segments, reviewed_micros)
    source_metrics = _timeline_metrics(reviewed_segments, reviewed_micros, extent)
    should_apply = bool(reviewed_micros) and (
        _float(source_metrics.get("total_action_coverage_ratio")) is not None
        and float(source_metrics.get("total_action_coverage_ratio") or 0.0) > 0.65
        or _float(source_metrics.get("longest_segment_ratio")) is not None
        and float(source_metrics.get("longest_segment_ratio") or 0.0) > 0.5
        or any((_duration_seconds(row) or 0.0) > 45.0 for row in source_segments)
    )
    if not should_apply:
        return {
            "applied": False,
            "reason": "source_segments_within_gate",
            "segments": reviewed_segments,
            "micro_segments": reviewed_micros,
            "micro_window_segment_count": 0,
            "manual_review_gaps": _segments_without_micro(source_segments, reviewed_micros),
        }

    source_by_id = {str(row.get("segment_id") or f"segment_{index:06d}"): row for index, row in enumerate(source_segments, start=1)}
    micro_children_by_parent: dict[str, list[dict[str, Any]]] = {}
    for micro in reviewed_micros:
        parent_id = str(micro.get("parent_segment_id") or micro.get("segment_id") or "")
        if parent_id:
            micro_children_by_parent.setdefault(parent_id, []).append(micro)

    converged_segments: list[dict[str, Any]] = []
    converged_micros: list[dict[str, Any]] = []
    rejected_segments = _rejected_ids(decisions, "segment:")
    for parent_id, children in sorted(micro_children_by_parent.items()):
        if parent_id in rejected_segments:
            continue
        source_segment = source_by_id.get(parent_id, {})
        for micro in sorted(children, key=lambda row: (_float(row.get("start_sec"), 0.0) or 0.0, str(row.get("micro_segment_id") or ""))):
            micro_id = str(micro.get("micro_segment_id") or "")
            if not micro_id:
                continue
            reviewed_segment_id = f"reviewed_{_safe_id(parent_id)}_{_safe_id(micro_id)}"
            micro_row = copy.deepcopy(micro)
            _ensure_auto_review(micro_row, item_id=f"micro:{micro_id}")
            segment_row = _segment_from_micro_window(source_segment, micro_row, parent_id=parent_id, reviewed_segment_id=reviewed_segment_id)
            micro_row["source_parent_segment_id"] = parent_id
            micro_row["parent_segment_id"] = reviewed_segment_id
            micro_row["segment_id"] = reviewed_segment_id
            micro_row["convergence_source"] = "micro_window"
            _append_review_index_text(micro_row)
            converged_segments.append(segment_row)
            converged_micros.append(micro_row)

    for gap in _segments_without_micro(source_segments, reviewed_micros):
        source_id = str(gap.get("segment_id") or "")
        if source_id in rejected_segments:
            continue
        source_segment = source_by_id.get(source_id)
        if not source_segment:
            continue
        row = _review_row(source_segment, f"segment:{source_id}", decisions.get(f"segment:{source_id}"))
        if row is None:
            continue
        if row.get("review_status") == "pending":
            row["review_status"] = "needs_review"
            row["review"] = {
                "item_id": f"segment:{source_id}",
                "decision": "needs_review",
                "reviewer": "auto_convergence",
                "note": "No micro-window evidence was available for this coarse segment.",
                "reviewed_at": _now(),
            }
            _append_review_index_text(row)
        row["convergence_source"] = "no_micro_manual_blocker"
        converged_segments.append(row)

    if not converged_segments:
        return {
            "applied": False,
            "reason": "no_micro_windows_available",
            "segments": reviewed_segments,
            "micro_segments": reviewed_micros,
            "micro_window_segment_count": 0,
            "manual_review_gaps": _segments_without_micro(source_segments, reviewed_micros),
        }
    return {
        "applied": True,
        "reason": "coverage_or_long_segment_rebuilt_from_micro_windows",
        "segments": converged_segments,
        "micro_segments": converged_micros,
        "micro_window_segment_count": len([row for row in converged_segments if row.get("convergence_source") == "micro_window"]),
        "manual_review_gaps": _segments_without_micro(source_segments, reviewed_micros),
    }


def _segment_from_micro_window(
    source_segment: Mapping[str, Any],
    micro: Mapping[str, Any],
    *,
    parent_id: str,
    reviewed_segment_id: str,
) -> dict[str, Any]:
    segment = copy.deepcopy(dict(source_segment or {}))
    segment["segment_id"] = reviewed_segment_id
    segment["source_segment_id"] = parent_id
    segment["reviewed_from_micro_segment_id"] = micro.get("micro_segment_id")
    segment["convergence_source"] = "micro_window"
    start, end = _row_interval(micro)
    if start is not None:
        segment["start_sec"] = start
    if end is not None:
        segment["end_sec"] = end
    if start is not None and end is not None and end >= start:
        segment["duration_sec"] = round(end - start, 4)
    for key in (
        "primary_object",
        "primary_object_family",
        "detected_objects",
        "action_type",
        "interaction_type",
        "evidence_level",
        "evidence_reasons",
        "coverage_signal_grade",
        "quality",
        "quality_warnings",
        "keyframes",
        "interaction_keyframes",
        "asset_bindings",
        "first_person",
        "third_person",
        "first_person_clip",
        "third_person_clip",
        "clip_path",
        "preview_path",
    ):
        if micro.get(key) not in (None, "", []):
            segment[key] = copy.deepcopy(micro.get(key))
    confidence = _row_confidence(micro) or _row_confidence(source_segment) or 0.75
    segment["boundary_confidence"] = confidence
    segment["confidence"] = confidence
    segment["review_status"] = str(micro.get("review_status") or "approved")
    segment["review"] = copy.deepcopy(micro.get("review")) if isinstance(micro.get("review"), Mapping) else {}
    segment["review"].update(
        {
            "item_id": f"segment:{reviewed_segment_id}",
            "decision": segment["review_status"],
            "reviewer": segment["review"].get("reviewer") or "auto_convergence",
            "note": segment["review"].get("note") or "Auto rebuilt reviewed segment from micro-window evidence.",
            "reviewed_at": segment["review"].get("reviewed_at") or _now(),
        }
    )
    for view_key in ("third_person", "first_person"):
        view = segment.get(view_key)
        if isinstance(view, dict):
            if start is not None:
                view["local_start_sec"] = start
            if end is not None:
                view["local_end_sec"] = end
    _append_review_index_text(segment)
    return segment


def _ensure_auto_review(row: dict[str, Any], *, item_id: str) -> None:
    status = str(row.get("review_status") or "pending").lower()
    if status == "pending":
        row["review_status"] = "approved"
        row["review"] = {
            "item_id": item_id,
            "decision": "approved",
            "reviewer": "auto_convergence",
            "note": "Auto accepted from micro-window convergence.",
            "reviewed_at": _now(),
        }
        return
    review = row.get("review") if isinstance(row.get("review"), Mapping) else {}
    row["review"] = {
        "item_id": review.get("item_id") or item_id,
        "decision": row.get("review_status"),
        "reviewer": review.get("reviewer"),
        "note": review.get("note") or "",
        "reviewed_at": review.get("reviewed_at"),
    }


def _segments_without_micro(source_segments: list[Mapping[str, Any]], micros: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    parents = {str(row.get("parent_segment_id") or row.get("segment_id") or "") for row in micros}
    output = []
    for index, segment in enumerate(source_segments, start=1):
        segment_id = str(segment.get("segment_id") or f"segment_{index:06d}")
        if segment_id in parents:
            continue
        start, end = _row_interval(segment)
        output.append(
            {
                "segment_id": segment_id,
                "start_sec": start,
                "end_sec": end,
                "duration_sec": _duration_seconds(segment),
                "reason": "no_micro_window_evidence",
            }
        )
    return output


def _timeline_metrics(
    reviewed_segments: list[Mapping[str, Any]],
    reviewed_micros: list[Mapping[str, Any]],
    extent_sec: float | None,
) -> dict[str, Any]:
    segment_intervals = [_row_interval(row) for row in reviewed_segments]
    micro_intervals = [_row_interval(row) for row in reviewed_micros]
    segment_intervals = [(start, end) for start, end in segment_intervals if start is not None and end is not None and end >= start]
    micro_intervals = [(start, end) for start, end in micro_intervals if start is not None and end is not None and end >= start]
    coverage_intervals = segment_intervals or micro_intervals
    coverage_sec = _merged_duration(coverage_intervals)
    micro_coverage_sec = _merged_duration(micro_intervals)
    longest_sec = max((end - start for start, end in coverage_intervals), default=0.0)
    total_reviewed = [*reviewed_segments, *reviewed_micros]
    unreviewed_count = sum(1 for row in total_reviewed if str(row.get("review_status") or "pending") in {"pending", "needs_review"})
    return {
        "segment_count": len(reviewed_segments),
        "micro_segment_count": len(reviewed_micros),
        "extent_sec": round(extent_sec, 4) if extent_sec is not None else None,
        "total_action_coverage_sec": round(coverage_sec, 4),
        "total_action_coverage_ratio": round(coverage_sec / extent_sec, 6) if extent_sec and extent_sec > 0 else None,
        "micro_coverage_sec": round(micro_coverage_sec, 4),
        "micro_coverage_ratio": round(micro_coverage_sec / extent_sec, 6) if extent_sec and extent_sec > 0 else None,
        "longest_segment_sec": round(longest_sec, 4),
        "longest_segment_ratio": round(longest_sec / extent_sec, 6) if extent_sec and extent_sec > 0 else None,
        "unreviewed_count": unreviewed_count,
        "review_status_counts": dict(sorted(Counter(str(row.get("review_status") or "pending") for row in total_reviewed).items())),
    }


def _source_extent(source_segments: list[Mapping[str, Any]], reviewed_micros: list[Mapping[str, Any]]) -> float | None:
    intervals = []
    for row in [*source_segments, *reviewed_micros]:
        start, end = _row_interval(row)
        if start is not None and end is not None and end >= start:
            intervals.append((start, end))
    if not intervals:
        return None
    minimum = min(start for start, _ in intervals)
    maximum = max(end for _, end in intervals)
    return round(maximum - minimum, 4) if maximum >= minimum else None


def _create_reviewed_release(
    session: Path,
    *,
    manifest: Mapping[str, Any],
    reviewed_segments: list[Mapping[str, Any]],
    reviewed_micros: list[Mapping[str, Any]],
    reviewed_evidence: list[Mapping[str, Any]],
    reviewed_vectors: list[Mapping[str, Any]],
    reviewed_index: Path,
) -> dict[str, Any]:
    releases_dir = session / REVIEWED_RELEASES_DIRNAME
    releases_dir.mkdir(parents=True, exist_ok=True)
    version = _next_release_version(releases_dir)
    release_dir = releases_dir / version
    artifacts_dir = release_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    previous = _latest_release_manifest(session)
    previous_artifacts = Path(str(previous.get("release_dir") or "")) / "artifacts" if previous else None
    diff = _release_diff(previous_artifacts, reviewed_segments, reviewed_micros, reviewed_evidence)
    release_manifest = {
        "schema_version": "key_action_reviewed_release.v1",
        "generated_at": _now(),
        "version": version,
        "release_dir": str(release_dir),
        "previous_version": previous.get("version") if isinstance(previous, Mapping) else None,
        "review_stats": {
            "reviewed_counts": dict(manifest.get("reviewed_counts") or {}),
            "row_review_status_counts": dict(manifest.get("row_review_status_counts") or {}),
            "decision_counts": dict(manifest.get("decision_counts") or {}),
            "boundary_adjustment_count": manifest.get("boundary_adjustment_count"),
            "delivery_ready": manifest.get("delivery_ready"),
            "auto_convergence": dict(manifest.get("auto_convergence") or {}),
        },
        "diff": diff,
        "artifacts": {
            "reviewed_segments": str(artifacts_dir / REVIEWED_SEGMENTS_FILENAME),
            "reviewed_micro_segments": str(artifacts_dir / REVIEWED_MICROS_FILENAME),
            "reviewed_evidence": str(artifacts_dir / REVIEWED_EVIDENCE_FILENAME),
            "reviewed_vector_metadata": str(artifacts_dir / REVIEWED_VECTOR_METADATA_FILENAME),
            "reviewed_dataset_manifest": str(artifacts_dir / REVIEWED_MANIFEST_FILENAME),
            "reviewed_export": str(artifacts_dir / REVIEWED_EXPORT_FILENAME),
            "reviewed_index": str(release_dir / "reviewed_index"),
        },
    }
    manifest_copy = copy.deepcopy(dict(manifest))
    manifest_copy["release"] = release_manifest
    export_payload = {
        "schema_version": "key_action_reviewed_export.v1",
        "generated_at": release_manifest["generated_at"],
        "session_dir": str(session),
        "manifest": manifest_copy,
        "segments": [dict(row) for row in reviewed_segments],
        "micro_segments": [dict(row) for row in reviewed_micros],
        "evidence": [dict(row) for row in reviewed_evidence],
    }
    write_jsonl(artifacts_dir / REVIEWED_SEGMENTS_FILENAME, [dict(row) for row in reviewed_segments])
    write_jsonl(artifacts_dir / REVIEWED_MICROS_FILENAME, [dict(row) for row in reviewed_micros])
    write_jsonl(artifacts_dir / REVIEWED_EVIDENCE_FILENAME, [dict(row) for row in reviewed_evidence])
    write_jsonl(artifacts_dir / REVIEWED_VECTOR_METADATA_FILENAME, [dict(row) for row in reviewed_vectors])
    (artifacts_dir / REVIEWED_MANIFEST_FILENAME).write_text(json.dumps(manifest_copy, ensure_ascii=False, indent=2), encoding="utf-8")
    (artifacts_dir / REVIEWED_EXPORT_FILENAME).write_text(json.dumps(export_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    target_index = release_dir / "reviewed_index"
    if target_index.exists():
        shutil.rmtree(target_index)
    if reviewed_index.exists():
        shutil.copytree(reviewed_index, target_index)
    package_path = release_dir / "reviewed_release_export.zip"
    _zip_release(release_dir, package_path)
    release_manifest["export_package"] = str(package_path)
    (release_dir / REVIEWED_RELEASE_MANIFEST_FILENAME).write_text(json.dumps(release_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_payload = {
        "schema_version": "key_action_reviewed_release_pointer.v1",
        "updated_at": _now(),
        "active_version": version,
        "release_dir": str(release_dir),
        "rollback": False,
    }
    (releases_dir / LATEST_REVIEWED_RELEASE_FILENAME).write_text(json.dumps(latest_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return release_manifest


def _latest_release_manifest(session: Path) -> dict[str, Any] | None:
    releases_dir = session / REVIEWED_RELEASES_DIRNAME
    pointer = releases_dir / LATEST_REVIEWED_RELEASE_FILENAME
    release_dir: Path | None = None
    if pointer.exists():
        try:
            data = json.loads(pointer.read_text(encoding="utf-8-sig"))
            if isinstance(data, Mapping) and data.get("release_dir"):
                release_dir = Path(str(data["release_dir"]))
        except (OSError, json.JSONDecodeError):
            release_dir = None
    if release_dir is None:
        releases = sorted(path for path in releases_dir.glob("v*") if path.is_dir()) if releases_dir.exists() else []
        release_dir = releases[-1] if releases else None
    if release_dir is None:
        return None
    path = release_dir / REVIEWED_RELEASE_MANIFEST_FILENAME
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _next_release_version(releases_dir: Path) -> str:
    numbers = []
    for path in releases_dir.glob("v*"):
        if not path.is_dir():
            continue
        try:
            numbers.append(int(path.name[1:]))
        except ValueError:
            continue
    return f"v{(max(numbers) + 1) if numbers else 1:03d}"


def _release_diff(
    previous_artifacts: Path | None,
    reviewed_segments: list[Mapping[str, Any]],
    reviewed_micros: list[Mapping[str, Any]],
    reviewed_evidence: list[Mapping[str, Any]],
) -> dict[str, Any]:
    previous_segments = _read_jsonl(previous_artifacts / REVIEWED_SEGMENTS_FILENAME) if previous_artifacts and previous_artifacts.exists() else []
    previous_micros = _read_jsonl(previous_artifacts / REVIEWED_MICROS_FILENAME) if previous_artifacts and previous_artifacts.exists() else []
    previous_evidence = _read_jsonl(previous_artifacts / REVIEWED_EVIDENCE_FILENAME) if previous_artifacts and previous_artifacts.exists() else []
    return {
        "counts_delta": {
            "segments": len(reviewed_segments) - len(previous_segments),
            "micro_segments": len(reviewed_micros) - len(previous_micros),
            "evidence": len(reviewed_evidence) - len(previous_evidence),
        },
        "segments": _id_set_diff(previous_segments, reviewed_segments, "segment_id"),
        "micro_segments": _id_set_diff(previous_micros, reviewed_micros, "micro_segment_id"),
        "review_status_delta": _counter_delta(
            Counter(str(row.get("review_status") or "pending") for row in [*previous_segments, *previous_micros]),
            Counter(str(row.get("review_status") or "pending") for row in [*reviewed_segments, *reviewed_micros]),
        ),
    }


def _id_set_diff(previous: list[Mapping[str, Any]], current: list[Mapping[str, Any]], key: str) -> dict[str, Any]:
    old = {str(row.get(key) or "") for row in previous if row.get(key)}
    new = {str(row.get(key) or "") for row in current if row.get(key)}
    return {
        "added": sorted(new - old),
        "removed": sorted(old - new),
        "unchanged_count": len(old & new),
    }


def _counter_delta(previous: Counter[str], current: Counter[str]) -> dict[str, int]:
    keys = sorted(set(previous) | set(current))
    return {key: int(current.get(key, 0) - previous.get(key, 0)) for key in keys}


def _zip_release(release_dir: Path, package_path: Path) -> None:
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(release_dir.rglob("*")):
            if not path.is_file() or path == package_path:
                continue
            archive.write(path, path.relative_to(release_dir))


def _review_rows(
    rows: list[Mapping[str, Any]],
    decisions: Mapping[str, Mapping[str, Any]],
    *,
    item_type: str,
    id_key: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        source_id = str(row.get(id_key) or f"{item_type}_{index:06d}")
        item_id = f"{item_type}:{source_id}"
        reviewed = _review_row(row, item_id, decisions.get(item_id))
        if reviewed is not None:
            output.append(reviewed)
    return output


def _review_vector_rows(rows: list[Mapping[str, Any]], decisions: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        item_ids = []
        micro_id = row.get("micro_segment_id")
        segment_id = row.get("segment_id") or row.get("parent_segment_id")
        if micro_id:
            item_ids.append(f"micro:{micro_id}")
        if segment_id:
            item_ids.append(f"segment:{segment_id}")
        effective_decision = _first_decision(item_ids, decisions)
        reviewed = _review_row(row, item_ids[0] if item_ids else "vector:unknown", effective_decision)
        if reviewed is not None:
            reviewed.setdefault("index_level", "micro_segment" if micro_id else "segment")
            output.append(reviewed)
    return output


def _review_row(
    row: Mapping[str, Any],
    item_id: str,
    decision: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    raw_status = decision.get("decision") if isinstance(decision, Mapping) else None
    status = str(raw_status or "pending").lower()
    if status == "rejected":
        return None
    reviewed = copy.deepcopy(dict(row))
    reviewed["review_status"] = status
    reviewed["review"] = {
        "item_id": item_id,
        "decision": status,
        "reviewer": decision.get("reviewer") if isinstance(decision, Mapping) else None,
        "note": decision.get("note") if isinstance(decision, Mapping) else "",
        "reviewed_at": decision.get("updated_at") if isinstance(decision, Mapping) else None,
    }
    if isinstance(decision, Mapping):
        _apply_boundary_adjustment(reviewed, decision)
    _append_review_index_text(reviewed)
    return reviewed


def _first_decision(item_ids: list[str], decisions: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any] | None:
    for item_id in item_ids:
        decision = decisions.get(item_id)
        if isinstance(decision, Mapping):
            return decision
    return None


def _apply_boundary_adjustment(row: dict[str, Any], decision: Mapping[str, Any]) -> None:
    has_start = decision.get("boundary_start_sec") is not None
    has_end = decision.get("boundary_end_sec") is not None
    if not has_start and not has_end:
        return
    original = {
        "start_sec": row.get("start_sec"),
        "end_sec": row.get("end_sec"),
        "duration_sec": row.get("duration_sec"),
    }
    if has_start:
        row["start_sec"] = float(decision["boundary_start_sec"])
    if has_end:
        row["end_sec"] = float(decision["boundary_end_sec"])
    start = _float(row.get("start_sec"))
    end = _float(row.get("end_sec"))
    if start is not None and end is not None and end >= start:
        row["duration_sec"] = round(end - start, 4)
    for view_key in ("third_person", "first_person"):
        view = row.get(view_key)
        if not isinstance(view, dict):
            continue
        if has_start:
            view["local_start_sec"] = row.get("start_sec")
        if has_end:
            view["local_end_sec"] = row.get("end_sec")
    row["reviewed_boundary"] = {
        "original": original,
        "adjusted_start_sec": row.get("start_sec") if has_start else None,
        "adjusted_end_sec": row.get("end_sec") if has_end else None,
        "adjusted_duration_sec": row.get("duration_sec"),
    }
    row.setdefault("review", {})["boundary_adjusted"] = True
    row["boundary_source"] = "reviewed_boundary_adjustment"


def _append_review_index_text(row: dict[str, Any]) -> None:
    index = row.get("index")
    if not isinstance(index, dict):
        index = {}
        row["index"] = index
    text = str(index.get("index_text") or row.get("index_text") or row.get("text") or "")
    review = row.get("review") if isinstance(row.get("review"), Mapping) else {}
    suffix = f"\nreview_status: {row.get('review_status') or 'pending'}"
    if review.get("boundary_adjusted"):
        suffix += " reviewed_boundary_adjusted"
    if suffix.strip() not in text:
        text = f"{text.rstrip()}{suffix}\n" if text else suffix.strip()
    index["index_text"] = text
    row["index_text"] = text


def _rows_to_vector_metadata(segments: list[Mapping[str, Any]], micros: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for segment in segments:
        item = copy.deepcopy(dict(segment))
        item.setdefault("index_level", "segment")
        item.setdefault("index_text", _index_text(item))
        rows.append(item)
    for micro in micros:
        item = copy.deepcopy(dict(micro))
        item.setdefault("index_level", "micro_segment")
        item.setdefault("segment_id", item.get("parent_segment_id"))
        item.setdefault("index_text", _index_text(item))
        rows.append(item)
    return rows


def _review_evidence_rows(
    session: Path,
    decisions: Mapping[str, Mapping[str, Any]],
    *,
    rejected_segments: set[str],
    rejected_micros: set[str],
) -> list[dict[str, Any]]:
    metadata = session / "metadata"
    evidence_files = (
        "model_observation_events.jsonl",
        "advanced_vision_evidence.jsonl",
        "video_understanding.jsonl",
        "unified_multimodal_timeline.jsonl",
    )
    output: list[dict[str, Any]] = []
    for file_name in evidence_files:
        for row in _read_jsonl(metadata / file_name):
            segment_id = str(row.get("segment_id") or row.get("parent_segment_id") or "")
            micro_id = str(row.get("micro_segment_id") or "")
            if segment_id in rejected_segments or micro_id in rejected_micros:
                continue
            item_ids = []
            if micro_id:
                item_ids.append(f"micro:{micro_id}")
            if segment_id:
                item_ids.append(f"segment:{segment_id}")
            decision = _first_decision(item_ids, decisions)
            reviewed = _review_row(row, item_ids[0] if item_ids else f"evidence:{file_name}", decision)
            if reviewed is not None:
                reviewed["source_reviewed_file"] = file_name
                output.append(reviewed)
    return output


def _build_reviewed_index(session: Path, rows: list[Mapping[str, Any]]) -> Path:
    target = session / "reviewed_index"
    texts = [_index_text(row) for row in rows]
    index = VectorIndex()
    index.build(texts, [dict(row) for row in rows])
    index.save(target)
    write_jsonl(target / "docstore.jsonl", [dict(row) for row in rows])
    return target


def _index_text(row: Mapping[str, Any]) -> str:
    index = row.get("index") if isinstance(row.get("index"), Mapping) else {}
    text_description = row.get("text_description") if isinstance(row.get("text_description"), Mapping) else {}
    return str(
        row.get("index_text")
        or index.get("index_text")
        or text_description.get("index_text")
        or text_description.get("summary")
        or row.get("summary")
        or row.get("title")
        or row.get("segment_id")
        or row.get("micro_segment_id")
        or ""
    )


def _rejected_ids(decisions: Mapping[str, Mapping[str, Any]], prefix: str) -> set[str]:
    return {
        item_id[len(prefix) :]
        for item_id, decision in decisions.items()
        if item_id.startswith(prefix) and str(decision.get("decision") or "") == "rejected"
    }


def _row_interval(row: Mapping[str, Any]) -> tuple[float | None, float | None]:
    for view_key in ("third_person", "first_person"):
        view = row.get(view_key)
        if isinstance(view, Mapping):
            start = _float(view.get("local_start_sec"))
            end = _float(view.get("local_end_sec"))
            if start is not None or end is not None:
                if end is None and start is not None:
                    duration = _duration_seconds(row)
                    end = start + duration if duration is not None else start
                return start, end
    start = _float(row.get("start_sec"))
    end = _float(row.get("end_sec"))
    duration = _duration_seconds(row)
    if start is None and end is not None and duration is not None:
        start = end - duration
    if end is None and start is not None and duration is not None:
        end = start + duration
    if start is None and end is None and duration is not None:
        start = 0.0
        end = duration
    return start, end


def _duration_seconds(row: Mapping[str, Any]) -> float | None:
    duration = _float(row.get("duration_sec"))
    if duration is not None:
        return duration
    start = _float(row.get("start_sec"))
    end = _float(row.get("end_sec"))
    if start is not None and end is not None and end >= start:
        return end - start
    return None


def _row_confidence(row: Mapping[str, Any]) -> float | None:
    for key in ("boundary_confidence", "confidence_score", "confidence", "score", "max_interaction_score"):
        value = _float(row.get(key))
        if value is not None:
            return value
    interaction = row.get("interaction") if isinstance(row.get("interaction"), Mapping) else {}
    for key in ("max_interaction_score", "avg_interaction_score"):
        value = _float(interaction.get(key))
        if value is not None:
            return value
    text = str(row.get("confidence") or "").lower()
    if "high" in text:
        return 0.85
    if "medium" in text:
        return 0.6
    if "low" in text:
        return 0.35
    return None


def _merged_duration(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
            continue
        merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return sum(end - start for start, end in merged)


def _safe_id(value: Any) -> str:
    text = str(value or "unknown")
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in text)


def _load_review_state(session: Path) -> dict[str, Any]:
    path = session / "metadata" / REVIEW_STATE_FILENAME
    if not path.exists():
        return {"schema_version": "key_action_review_state.v1", "decisions": {}, "audit": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    data.setdefault("schema_version", "key_action_review_state.v1")
    data.setdefault("decisions", {})
    data.setdefault("audit", [])
    return data


def _latest_decisions(state: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw = state.get("decisions") if isinstance(state.get("decisions"), Mapping) else {}
    return {str(key): value for key, value in raw.items() if isinstance(value, Mapping)}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.exists() else []


def _float(value: Any, default: float | None = None) -> float | None:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "REVIEWED_DATASET_SCHEMA_VERSION",
    "REVIEWED_EVIDENCE_FILENAME",
    "REVIEWED_EXPORT_FILENAME",
    "REVIEWED_MANIFEST_FILENAME",
    "REVIEWED_MICROS_FILENAME",
    "REVIEWED_RELEASE_MANIFEST_FILENAME",
    "REVIEWED_RELEASES_DIRNAME",
    "REVIEWED_SEGMENTS_FILENAME",
    "REVIEWED_VECTOR_METADATA_FILENAME",
    "freeze_reviewed_dataset",
    "latest_reviewed_release",
    "load_reviewed_export",
    "rollback_reviewed_release",
    "reviewed_index_dir",
    "reviewed_metadata_path",
]

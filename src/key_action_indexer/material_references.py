from __future__ import annotations

import json
import hashlib
import re
import shutil
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from .physical_evidence import (
    PHYSICAL_EVIDENCE_MIN_FRAMES,
    evidence_view,
    physical_evidence_policy_summary,
    valid_yolo_physical_evidence,
    yolo_physical_evidence_diagnostics,
)
from .schemas import read_jsonl
from .yolo_detector import HAND_LABELS, canonical_yolo_label, filter_implausible_detections, find_hand_object_interactions
from .yolo_vlm_pipeline import apply_yolo_vlm_review_pipeline


KEYFRAME_DIR_NAME = "\u5173\u952e\u5e27"
KEY_CLIP_DIR_NAME = "\u5173\u952e\u7247\u6bb5"
REPORT_DIR_NAME = "\u4e13\u4e1a\u62a5\u544a"
MATERIAL_INDEX_BASENAME = "\u7d20\u6750\u7d22\u5f15"
MATERIAL_CANDIDATE_INDEX_BASENAME = "\u7d20\u6750\u5019\u9009\u7d22\u5f15"
MATERIAL_CANDIDATE_REVIEW_LOG = "review_log.jsonl"
MATERIAL_REVIEW_QUEUE_DIR_NAME = "_material_review_queue"
LEGACY_MATERIAL_CANDIDATE_DIR_NAME = "material_candidates"
YOLO_SUFFIX = "YOLO\u6807\u6ce8"
NAMING_RULE = "\u6b63\u5f0f\u4ea4\u4ed8\u76ee\u5f55=\u5b9e\u9a8c\u6807\u9898_\u65e5\u671f\uff1b\u5173\u952e\u5e27/\u5173\u952e\u7247\u6bb5=\u624b\u4e0e\u88ab\u4ea4\u4e92\u5bf9\u8c61\u64cd\u4f5c_\u65e5\u671f[_\u5e8f\u53f7].\u6269\u5c55\u540d"
README_TITLE = "\u5173\u952e\u7269\u7406\u52a8\u4f5c\u7d20\u6750\u5f15\u7528"

STALE_SPLIT_MARKERS = (
    "seg_000001_part02",
    "seg_000001_part03",
    "seg_000001_part04",
    "part02",
    "part03",
    "part04",
)

ACTION_NAME_BY_OBJECT = {
    "balance": "\u624b\u4e0e\u5929\u5e73\u64cd\u4f5c",
    "spatula": "\u624b\u4e0e\u836f\u5319\u64cd\u4f5c",
    "pipette": "\u624b\u4e0e\u79fb\u6db2\u67aa\u64cd\u4f5c",
    "pipette_tip": "\u624b\u4e0e\u79fb\u6db2\u67aa\u5934\u64cd\u4f5c",
    "reagent_bottle": "\u624b\u4e0e\u8bd5\u5242\u74f6\u64cd\u4f5c",
    "sample_bottle": "\u624b\u4e0e\u6837\u54c1\u74f6\u64cd\u4f5c",
    "sample_bottle_blue": "\u624b\u4e0e\u84dd\u76d6\u6837\u54c1\u74f6\u64cd\u4f5c",
    "beaker": "\u624b\u4e0e\u70e7\u676f\u64cd\u4f5c",
    "container": "\u624b\u4e0e\u5bb9\u5668\u64cd\u4f5c",
}

VIEW_LABELS = {
    "first_person": "\u7b2c\u4e00\u4eba\u79f0",
    "third_person": "\u7b2c\u4e09\u4eba\u79f0",
}

FRAME_LABELS = {
    "contact": "\u63a5\u89e6\u5e27",
    "peak": "\u5cf0\u503c\u5e27",
    "release": "\u91ca\u653e\u5e27",
}


def _read_jsonl_if_exists(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    return read_jsonl(source)


def material_references_root(session_dir: str | Path) -> Path:
    """Return the run-local material reference mirror for a key-action run."""

    session_root = Path(session_dir)
    if session_root.name == "key_action_index":
        return session_root.parent / "material_references"
    return session_root / "material_references"


def formal_material_references_root(session_dir: str | Path) -> Path:
    """Return the formal handoff folder under LabSOPGuard/outputs/material_references."""

    session_root = Path(session_dir)
    experiment = _experiment_metadata(session_root)
    return _formal_material_root(session_root, experiment["label"])


def material_candidates_root(session_dir: str | Path) -> Path:
    """Return the frontend-review candidate folder for a key-action run."""

    session_root = Path(session_dir)
    if session_root.name == "key_action_index":
        return session_root.parent / MATERIAL_REVIEW_QUEUE_DIR_NAME
    return session_root / MATERIAL_REVIEW_QUEUE_DIR_NAME


def legacy_material_candidates_root(session_dir: str | Path) -> Path:
    """Return the pre-review-queue candidate folder kept for read compatibility."""

    session_root = Path(session_dir)
    if session_root.name == "key_action_index":
        return session_root.parent / LEGACY_MATERIAL_CANDIDATE_DIR_NAME
    return session_root / LEGACY_MATERIAL_CANDIDATE_DIR_NAME


def material_candidate_roots(session_dir: str | Path) -> list[Path]:
    """Return candidate roots in preferred read order."""

    canonical = material_candidates_root(session_dir)
    legacy = legacy_material_candidates_root(session_dir)
    return [canonical] if canonical == legacy else [canonical, legacy]


def existing_material_candidates_root(session_dir: str | Path) -> Path:
    """Return the candidate root with an existing index, preferring the review queue."""

    for root in material_candidate_roots(session_dir):
        if (root / f"{MATERIAL_CANDIDATE_INDEX_BASENAME}.jsonl").exists():
            return root
    for root in material_candidate_roots(session_dir):
        if root.exists():
            return root
    return material_candidates_root(session_dir)


def build_yolo_material_references(
    session_dir: str | Path,
    *,
    dry_run: bool = False,
    ffmpeg_path: str | Path = "ffmpeg",
    archive_existing: bool = True,
) -> dict[str, Any]:
    """Build the human-readable YOLO physical-action material folder."""

    session_root = Path(session_dir)
    metadata_dir = session_root / "metadata"
    ref_root = material_references_root(session_root)
    keyframe_dir = ref_root / KEYFRAME_DIR_NAME
    clip_dir = ref_root / KEY_CLIP_DIR_NAME
    archive_root = session_root / "archive" / f"material_references_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    archived_items = _prepare_reference_root(ref_root, archive_root, archive_existing=archive_existing)

    micro_rows = _read_jsonl_if_exists(metadata_dir / "micro_segments.jsonl")
    segment_rows = _read_jsonl_if_exists(metadata_dir / "key_action_segments.jsonl")
    annotated_lookup = _annotated_clip_lookup(_read_jsonl_if_exists(metadata_dir / "annotated_clips.jsonl"))
    segment_by_id = {str(row.get("segment_id") or ""): row for row in segment_rows}
    experiment = _experiment_metadata(session_root)
    formal_root = formal_material_references_root(session_root)
    ffmpeg_ok = (not dry_run) and _ffmpeg_available(ffmpeg_path)

    records: list[dict[str, Any]] = []
    planned_records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    used_names: set[str] = set()

    for micro in micro_rows:
        micro_id = str(micro.get("micro_segment_id") or "")
        parent_id = str(micro.get("parent_segment_id") or micro.get("segment_id") or "")
        if _is_stale_identifier(micro_id) or _is_stale_identifier(parent_id):
            skipped.append({"micro_segment_id": micro_id, "reason": "stale_split_marker"})
            continue
        raw_evidence = [item for item in micro.get("yolo_evidence") or [] if isinstance(item, dict)]
        if not raw_evidence:
            skipped.append({"micro_segment_id": micro_id, "reason": "missing_yolo_evidence"})
            continue
        interaction = micro.get("interaction") if isinstance(micro.get("interaction"), dict) else {}
        primary = canonical_yolo_label(micro.get("primary_object") or interaction.get("primary_object") or "")
        action_name = _action_name(primary or "object")
        diagnostics = yolo_physical_evidence_diagnostics(raw_evidence, primary)
        valid_evidence = valid_yolo_physical_evidence(raw_evidence, primary)
        if len(valid_evidence) < PHYSICAL_EVIDENCE_MIN_FRAMES:
            skipped.append(
                {
                    "micro_segment_id": micro_id,
                    "reason": "no_valid_yolo_physical_evidence",
                    "primary_object": primary,
                    "diagnostics": diagnostics,
                }
            )
            continue

        start_sec = _safe_float(micro.get("start_sec", micro.get("session_start_sec")))
        end_sec = _safe_float(micro.get("end_sec", micro.get("session_end_sec")), start_sec)
        if end_sec <= start_sec:
            skipped.append({"micro_segment_id": micro_id, "reason": "invalid_time_range"})
            continue
        segment = segment_by_id.get(parent_id, {})
        file_date = _micro_date_label(micro, experiment["date"])
        duration = max(0.1, end_sec - start_sec)

        for view in ("third_person", "first_person"):
            view_evidence = [item for item in valid_evidence if evidence_view(item) == view]
            if len(view_evidence) < PHYSICAL_EVIDENCE_MIN_FRAMES:
                skipped.append(
                    {
                        "micro_segment_id": micro_id,
                        "view": view,
                        "reason": "no_valid_yolo_physical_evidence_for_view",
                        "primary_object": primary,
                        "valid_evidence_count": len(view_evidence),
                    }
                )
                continue
            source_clip = _source_clip_for_view(session_root, annotated_lookup, segment, micro, view)
            if source_clip is None:
                skipped.append({"micro_segment_id": micro_id, "view": view, "reason": "missing_source_clip"})
                continue
            if not dry_run and not source_clip.is_file():
                skipped.append({"micro_segment_id": micro_id, "view": view, "reason": "source_clip_missing_on_disk", "path": str(source_clip)})
                continue

            segment_start = _segment_view_start(segment, view)
            offset = max(0.0, start_sec - segment_start)
            frame_rows = _evidence_frame_rows(view_evidence, start_sec, end_sec)
            clip_windows = [
                ("micro_clip", offset, duration),
                ("peak_clip", max(0.0, _safe_float(frame_rows[1][1].get("local_time_sec"), start_sec) - segment_start - 0.8), min(1.6, duration)),
            ]

            for role, clip_offset, clip_duration in clip_windows:
                target = clip_dir / _unique_name(used_names, f"{action_name}_{file_date}", ".mp4")
                row = _record(
                    micro=micro,
                    segment=segment,
                    target=target,
                    source=source_clip,
                    material_type=KEY_CLIP_DIR_NAME,
                    view=view,
                    action_name=action_name,
                    generated=False,
                    dry_run=dry_run,
                    error=None,
                )
                row.update({"role": role, "source_offset_sec": clip_offset})
                planned_records.append(row)
                if dry_run:
                    continue
                error = None
                generated = False
                try:
                    _render_filtered_interaction_clip(
                        source_clip,
                        clip_offset,
                        clip_duration,
                        target,
                        view_evidence,
                        primary,
                        segment_start,
                    )
                    generated = target.exists()
                except Exception as exc:  # pragma: no cover
                    if ffmpeg_ok:
                        try:
                            _cut_video(ffmpeg_path, source_clip, clip_offset, clip_duration, target)
                            generated = target.exists()
                            error = f"annotation_fallback_unboxed:{exc}"
                        except Exception:
                            error = str(exc)
                    else:
                        error = f"annotation_failed_ffmpeg_unavailable:{exc}"
                records.append(
                    {
                        **row,
                        "exists": bool(generated),
                        "generated": bool(generated),
                        "error": error,
                        "size_bytes": target.stat().st_size if target.exists() else 0,
                        "yolo_annotation_rendered": bool(generated and not error),
                    }
                )

            for frame_type, evidence_row in frame_rows:
                local_time = _safe_float(evidence_row.get("local_time_sec"), start_sec)
                target = keyframe_dir / _unique_name(used_names, f"{action_name}_{file_date}", ".jpg")
                row = _record(
                    micro=micro,
                    segment=segment,
                    target=target,
                    source=source_clip,
                    material_type=KEYFRAME_DIR_NAME,
                    view=view,
                    action_name=action_name,
                    frame_type=frame_type,
                    generated=False,
                    dry_run=dry_run,
                    error=None,
                )
                row.update({"source_offset_sec": max(0.0, local_time - segment_start)})
                planned_records.append(row)
                if dry_run:
                    continue
                error = None
                generated = False
                try:
                    _extract_filtered_interaction_frame(
                        source_clip,
                        row["source_offset_sec"],
                        target,
                        evidence_row,
                        primary,
                    )
                    generated = target.exists()
                except Exception as exc:  # pragma: no cover
                    if ffmpeg_ok:
                        try:
                            _extract_frame(ffmpeg_path, source_clip, row["source_offset_sec"], target)
                            generated = target.exists()
                            error = f"annotation_fallback_unboxed:{exc}"
                        except Exception:
                            error = str(exc)
                    else:
                        error = f"annotation_failed_ffmpeg_unavailable:{exc}"
                records.append(
                    {
                        **row,
                        "exists": bool(generated),
                        "generated": bool(generated),
                        "error": error,
                        "size_bytes": target.stat().st_size if target.exists() else 0,
                        "yolo_annotation_rendered": bool(generated and not error),
                    }
                )

    index_json = ref_root / f"{MATERIAL_INDEX_BASENAME}.json"
    index_jsonl = ref_root / f"{MATERIAL_INDEX_BASENAME}.jsonl"
    output_rows = records if not dry_run else planned_records
    summary = {
        "schema_version": "yolo_physical_action_material_files.v1",
        "created_at": datetime.now().isoformat(),
        "experiment_id": experiment["id"],
        "experiment_title": experiment["title"],
        "experiment_date": experiment["date"],
        "experiment_label": experiment["label"],
        "session_dir": str(session_root),
        "material_references": str(ref_root),
        "source_material_references": str(ref_root),
        "formal_material_references": str(formal_root),
        "simplified_material_references": str(formal_root),
        "keyframe_folder": str(keyframe_dir),
        "key_clip_folder": str(clip_dir),
        "simplified_keyframe_folder": str(formal_root / KEYFRAME_DIR_NAME),
        "simplified_key_clip_folder": str(formal_root / KEY_CLIP_DIR_NAME),
        "index_json": str(index_json),
        "index_jsonl": str(index_jsonl),
        "dry_run": bool(dry_run),
        "ffmpeg_available": bool(ffmpeg_ok),
        "file_count": sum(1 for row in records if row.get("exists")),
        "planned_file_count": len(planned_records) if dry_run else 0,
        "keyframe_count": sum(1 for row in records if row.get("asset_kind") == KEYFRAME_DIR_NAME and row.get("exists")),
        "key_clip_count": sum(1 for row in records if row.get("asset_kind") == KEY_CLIP_DIR_NAME and row.get("exists")),
        "naming_rule": NAMING_RULE,
        "policy": "Validated YOLO hand-object keyframes and clips are staged here as review sources; only approved candidates are copied to the formal delivery folder.",
        "physical_evidence_policy": physical_evidence_policy_summary(),
        "excluded_stale_markers": list(STALE_SPLIT_MARKERS),
        "archive_root": str(archive_root) if archived_items else None,
        "archived_count": len(archived_items),
        "archived_items": archived_items,
        "skipped_count": len(skipped),
        "skipped": skipped,
        "planned_records": planned_records,
        "records": output_rows,
    }
    _write_json(index_json, summary)
    _write_jsonl(index_jsonl, output_rows)
    _write_json(ref_root / "manifest.json", _manifest(summary))
    _write_readme(ref_root / "README.md", summary)
    return summary


def _experiment_metadata(session_root: Path) -> dict[str, str]:
    payload: dict[str, Any] = {}
    for path in (session_root.parent / "experiment.json", session_root / "experiment.json"):
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                break
            except Exception:
                payload = {}
    experiment_id = str(payload.get("experiment_id") or payload.get("id") or session_root.parent.name)
    title = str(payload.get("title") or payload.get("experiment_title") or payload.get("name") or session_root.parent.name)
    date = _date_from_text(str(payload.get("created_at") or payload.get("experiment_date") or payload.get("date") or experiment_id))
    if not date:
        date = datetime.now().strftime("%Y%m%d")
    return {
        "id": experiment_id,
        "title": title,
        "date": date,
        "label": _safe_name(f"{title}_{date}"),
    }


def _date_from_text(value: str) -> str | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).strftime("%Y%m%d")
    except ValueError:
        pass
    match = re.search(r"(?<!\d)(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)(?!\d)", value)
    return "".join(match.groups()) if match else None


def _micro_date_label(micro: dict[str, Any], fallback: str) -> str:
    for key in ("global_start_time", "start_time", "created_at"):
        label = _date_from_text(str(micro.get(key) or ""))
        if label:
            return label
    return fallback or datetime.now().strftime("%Y%m%d")


def _formal_material_root(session_root: Path, experiment_label: str) -> Path:
    ref_root = material_references_root(session_root)
    if session_root.name == "key_action_index":
        experiment_dir = session_root.parent
        outputs_dir = experiment_dir.parent.parent if experiment_dir.parent.name == "experiments" else experiment_dir.parent
        return outputs_dir / "material_references" / experiment_label
    return ref_root.parent / "material_references" / experiment_label


def _simplified_material_root(session_root: Path, experiment_label: str) -> Path:
    return _formal_material_root(session_root, experiment_label)


def _source_clip_for_view(
    session_root: Path,
    lookup: dict[tuple[str, str], str],
    segment: dict[str, Any],
    micro: dict[str, Any],
    view: str,
) -> Path | None:
    segment_id = str(segment.get("segment_id") or micro.get("parent_segment_id") or micro.get("segment_id") or "")
    view_data = segment.get(view) if isinstance(segment.get(view), dict) else {}
    candidates = [
        view_data.get("clip_path"),
        view_data.get("raw_clip_path"),
        lookup.get((segment_id, view)),
        view_data.get("annotated_clip_path"),
        micro.get("annotated_clip"),
        segment.get(f"{view}_annotated_clip"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(str(candidate))
        if not path.is_absolute():
            path = session_root / path
        return path
    return None


def _evidence_frame_rows(
    yolo_evidence_rows: list[dict[str, Any]],
    start_sec: float,
    end_sec: float,
) -> list[tuple[str, dict[str, Any]]]:
    rows = [
        item
        for item in yolo_evidence_rows
        if isinstance(item, dict) and item.get("local_time_sec") is not None
    ]
    if not rows:
        return [
            ("contact", {"local_time_sec": start_sec}),
            ("release", {"local_time_sec": end_sec}),
        ]
    rows = sorted(rows, key=lambda item: _safe_float(item.get("local_time_sec"), start_sec))
    max_score = max(_safe_float(item.get("interaction_score"), 0.0) for item in rows)
    midpoint = (start_sec + end_sec) / 2.0
    peak_candidates = [
        item
        for item in rows
        if _safe_float(item.get("interaction_score"), 0.0) >= max_score - 1e-9
    ]
    peak = min(peak_candidates, key=lambda item: abs(_safe_float(item.get("local_time_sec"), midpoint) - midpoint))
    selected = [("contact", rows[0]), ("peak", peak), ("release", rows[-1])]
    deduped: list[tuple[str, dict[str, Any]]] = []
    seen: set[float] = set()
    for role, row in selected:
        ts = round(_safe_float(row.get("local_time_sec"), start_sec), 3)
        if ts in seen:
            continue
        seen.add(ts)
        deduped.append((role, row))
    return deduped


def _copy_simplified_materials(ref_root: Path, simplified_root: Path, summary: dict[str, Any]) -> None:
    if ref_root.resolve() == simplified_root.resolve():
        return
    if simplified_root.exists():
        for name in (KEYFRAME_DIR_NAME, KEY_CLIP_DIR_NAME, "manifest.json", "README.md", f"{MATERIAL_INDEX_BASENAME}.json", f"{MATERIAL_INDEX_BASENAME}.jsonl"):
            target = simplified_root / name
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
    for folder in (KEYFRAME_DIR_NAME, KEY_CLIP_DIR_NAME, REPORT_DIR_NAME):
        source_dir = ref_root / folder
        target_dir = simplified_root / folder
        target_dir.mkdir(parents=True, exist_ok=True)
        if source_dir.is_dir():
            for source in source_dir.iterdir():
                if source.is_file():
                    shutil.copy2(source, target_dir / source.name)
    simplified_summary = dict(summary)
    simplified_summary["material_references"] = str(simplified_root)
    simplified_summary["formal_material_references"] = str(simplified_root)
    simplified_summary["local_material_references_mirror"] = str(ref_root)
    simplified_summary["keyframe_folder"] = str(simplified_root / KEYFRAME_DIR_NAME)
    simplified_summary["key_clip_folder"] = str(simplified_root / KEY_CLIP_DIR_NAME)
    simplified_summary["index_json"] = str(simplified_root / f"{MATERIAL_INDEX_BASENAME}.json")
    simplified_summary["index_jsonl"] = str(simplified_root / f"{MATERIAL_INDEX_BASENAME}.jsonl")
    rebased_rows = []
    for row in summary.get("records", []):
        updated = dict(row)
        filename = str(updated.get("stored_filename") or updated.get("file_name") or "")
        asset_kind = str(updated.get("asset_kind") or "")
        if filename and asset_kind in {KEYFRAME_DIR_NAME, KEY_CLIP_DIR_NAME, REPORT_DIR_NAME}:
            target = simplified_root / asset_kind / filename
            if target.exists() or asset_kind in {KEYFRAME_DIR_NAME, KEY_CLIP_DIR_NAME}:
                updated["stored_file"] = str(target)
                updated["exists"] = target.exists()
        rebased_rows.append(updated)
    simplified_summary["records"] = rebased_rows
    _write_json(simplified_root / f"{MATERIAL_INDEX_BASENAME}.json", simplified_summary)
    _write_jsonl(simplified_root / f"{MATERIAL_INDEX_BASENAME}.jsonl", rebased_rows)
    _write_json(simplified_root / "manifest.json", _manifest(simplified_summary))
    _write_readme(simplified_root / "README.md", simplified_summary)


def sync_professional_report_material_references(
    session_dir: str | Path,
    *,
    report_summary: dict[str, Any],
    archive_existing: bool = False,
) -> dict[str, Any]:
    """Stage generated professional report artifacts in the review queue.

    Professional PDFs follow the same candidate-first policy as keyframes and
    clips. They are copied into the formal material reference folder only after
    an operator approves the candidate group.
    """

    session_root = Path(session_dir)
    candidate_root = material_candidates_root(session_root)
    report_dir = candidate_root / REPORT_DIR_NAME
    if archive_existing and report_dir.exists():
        archived = report_dir.with_name(f"{report_dir.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        shutil.move(str(report_dir), str(archived))
    report_dir.mkdir(parents=True, exist_ok=True)

    copied_rows: list[dict[str, Any]] = []
    for role, key in (
        ("professional_report_pdf", "pdf_path"),
        ("professional_report_html", "html_path"),
        ("professional_report_json", "sidecar_path"),
        ("professional_report_manifest", "manifest_path"),
    ):
        source_value = report_summary.get(key) or (report_summary.get("json_path") if role == "professional_report_json" else None)
        if not source_value:
            continue
        source = Path(str(source_value))
        if not source.exists():
            continue
        candidate_target = report_dir / source.name
        shutil.copy2(source, candidate_target)
        record = _professional_report_record(role=role, source=source, target=candidate_target)
        record["review_status"] = "pending"
        record["delivery_scope"] = "professional_report_candidate"
        copied_rows.append(_candidate_record_from_reference(record, source, candidate_target, exists=candidate_target.exists()))

    index_jsonl = candidate_root / f"{MATERIAL_CANDIDATE_INDEX_BASENAME}.jsonl"
    existing_rows = read_jsonl(index_jsonl) if index_jsonl.exists() else []
    existing_rows = [
        row
        for row in existing_rows
        if not (row.get("asset_kind") == REPORT_DIR_NAME and row.get("role") in {
            "professional_report_pdf",
            "professional_report_html",
            "professional_report_json",
            "professional_report_manifest",
        })
    ]
    rows = existing_rows + copied_rows
    _mark_recommended_candidates(rows)
    summary = {
        "schema_version": "material_references.report_candidate_sync.v1",
        "created_at": datetime.now().isoformat(),
        "session_dir": str(session_root),
        "candidate_folder": str(candidate_root),
        "keyframe_folder": str(candidate_root / KEYFRAME_DIR_NAME),
        "key_clip_folder": str(candidate_root / KEY_CLIP_DIR_NAME),
        "report_folder": str(report_dir),
        "report_count": len(copied_rows),
        "candidate_count": len(rows),
        "pending_total": sum(1 for row in rows if row.get("candidate_status") == "pending"),
        "recommended_total": sum(1 for row in rows if row.get("recommended") is True),
        "available": bool(copied_rows),
        "records": copied_rows,
        "pipeline_summary": None,
        "policy": "Professional reports require frontend approval before entering material_references.",
    }
    _write_jsonl(index_jsonl, rows)
    _write_json(candidate_root / f"{MATERIAL_CANDIDATE_INDEX_BASENAME}.json", {**summary, "records": rows})
    _write_json(candidate_root / "manifest.json", _candidate_manifest({**summary, "records": rows, "candidate_count": len(rows)}))
    _write_candidate_readme(candidate_root / "README.md", {**summary, "records": rows})
    return {
        "available": bool(copied_rows),
        "status": "candidate_staged",
        "path": str(report_dir),
        "candidate_index": str(index_jsonl),
        "report_count": len(copied_rows),
        "policy": summary["policy"],
    }


def _sync_professional_report_material_references_legacy(
    session_dir: str | Path,
    *,
    report_summary: dict[str, Any],
    archive_existing: bool = False,
) -> dict[str, Any]:
    """Legacy direct report publish path retained for migrations only."""

    session_root = Path(session_dir)
    ref_root = material_references_root(session_root)
    simplified_root = _simplified_delivery_root(ref_root, report_summary)
    report_dir = ref_root / REPORT_DIR_NAME
    simplified_report_dir = simplified_root / REPORT_DIR_NAME
    for target in (report_dir, simplified_report_dir):
        if archive_existing and target.exists():
            archived = target.with_name(f"{target.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            shutil.move(str(target), str(archived))
        target.mkdir(parents=True, exist_ok=True)

    copied_rows: list[dict[str, Any]] = []
    for role, key in (
        ("professional_report_pdf", "pdf_path"),
        ("professional_report_html", "html_path"),
        ("professional_report_json", "sidecar_path"),
        ("professional_report_manifest", "manifest_path"),
    ):
        source_value = report_summary.get(key) or (report_summary.get("json_path") if role == "professional_report_json" else None)
        if not source_value:
            continue
        source = Path(str(source_value))
        if not source.exists():
            continue
        formal_target = report_dir / source.name
        simplified_target = simplified_report_dir / source.name
        shutil.copy2(source, formal_target)
        shutil.copy2(source, simplified_target)
        copied_rows.append(_professional_report_record(role=role, source=source, target=formal_target))

    existing_rows = []
    index_jsonl = ref_root / f"{MATERIAL_INDEX_BASENAME}.jsonl"
    if index_jsonl.exists():
        existing_rows = [row for row in read_jsonl(index_jsonl) if row.get("asset_kind") != REPORT_DIR_NAME]
    rows = existing_rows + copied_rows
    summary = {
        "schema_version": "material_references.report_sync.v1",
        "created_at": datetime.now().isoformat(),
        "session_dir": str(session_root),
        "report_folder": str(report_dir),
        "simplified_report_folder": str(simplified_report_dir),
        "report_count": len(copied_rows),
        "available": bool(copied_rows),
        "records": copied_rows,
    }
    _write_jsonl(index_jsonl, rows)
    _write_json(ref_root / f"{MATERIAL_INDEX_BASENAME}.json", {"records": rows, **summary})
    _write_json(report_dir / "manifest.json", summary)
    _sync_simplified_metadata(ref_root, simplified_root, rows, summary)
    return {
        "available": bool(copied_rows),
        "path": str(report_dir),
        "simplified_path": str(simplified_report_dir),
        "report_count": len(copied_rows),
    }


def build_yolo_material_candidates(
    session_dir: str | Path,
    *,
    dry_run: bool = False,
    ffmpeg_path: str | Path = "ffmpeg",
    archive_existing: bool = True,
    rebuild_source: bool = False,
    vlm_client: Any | None = None,
    enable_vlm: bool = False,
    max_vlm_groups: int = 8,
    vlm_model_name: str | None = None,
) -> dict[str, Any]:
    """Build review-gated YOLO material candidates.

    These files are intentionally kept outside ``material_references`` until a
    reviewer calls ``approve_material_candidates``. The VLM step can only add
    advisory semantics within the YOLO evidence packet.
    """

    session_root = Path(session_dir)
    source_root = material_references_root(session_root)
    source_index = source_root / f"{MATERIAL_INDEX_BASENAME}.jsonl"
    if rebuild_source or not source_index.exists():
        build_yolo_material_references(
            session_root,
            dry_run=dry_run,
            ffmpeg_path=ffmpeg_path,
            archive_existing=archive_existing,
        )

    source_rows = read_jsonl(source_index) if source_index.exists() else []
    candidate_root = material_candidates_root(session_root)
    archive_root = session_root / "archive" / f"material_review_queue_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    _prepare_candidate_root(candidate_root, archive_root, archive_existing=archive_existing)
    keyframe_dir = candidate_root / KEYFRAME_DIR_NAME
    clip_dir = candidate_root / KEY_CLIP_DIR_NAME
    keyframe_dir.mkdir(parents=True, exist_ok=True)
    clip_dir.mkdir(parents=True, exist_ok=True)

    candidate_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    used_names: set[str] = set()
    for row in source_rows:
        if str(row.get("asset_kind") or row.get("material_type") or "") not in {KEYFRAME_DIR_NAME, KEY_CLIP_DIR_NAME}:
            continue
        source = _stored_path_from_row(row, source_root)
        if source is None or (not dry_run and not source.is_file()):
            skipped.append({"source": str(source) if source else "", "reason": "source_file_missing"})
            continue
        target_dir = keyframe_dir if str(row.get("asset_kind") or "") == KEYFRAME_DIR_NAME else clip_dir
        filename = str(row.get("stored_filename") or row.get("file_name") or (source.name if source else "candidate"))
        target = target_dir / _unique_name(used_names, Path(filename).stem, Path(filename).suffix or source.suffix)
        if not dry_run:
            shutil.copy2(source, target)
        candidate_rows.append(_candidate_record_from_reference(row, source, target, exists=target.exists()))

    _mark_recommended_candidates(candidate_rows)
    micro_path = session_root / "metadata" / "micro_segments.jsonl"
    micro_rows = read_jsonl(micro_path) if micro_path.exists() else []
    pipeline_summary = apply_yolo_vlm_review_pipeline(
        session_root,
        candidate_rows,
        micro_rows,
        vlm_client=vlm_client,
        enable_vlm=enable_vlm,
        max_vlm_groups=max_vlm_groups,
        vlm_model_name=vlm_model_name,
    )

    index_json = candidate_root / f"{MATERIAL_CANDIDATE_INDEX_BASENAME}.json"
    index_jsonl = candidate_root / f"{MATERIAL_CANDIDATE_INDEX_BASENAME}.jsonl"
    summary = {
        "schema_version": "yolo_material_candidates.v1",
        "created_at": datetime.now().isoformat(),
        "session_dir": str(session_root),
        "candidate_folder": str(candidate_root),
        "keyframe_folder": str(keyframe_dir),
        "key_clip_folder": str(clip_dir),
        "index_json": str(index_json),
        "index_jsonl": str(index_jsonl),
        "review_log": str(candidate_root / MATERIAL_CANDIDATE_REVIEW_LOG),
        "dry_run": bool(dry_run),
        "candidate_count": len(candidate_rows),
        "pending_total": sum(1 for row in candidate_rows if row.get("candidate_status") == "pending"),
        "recommended_total": sum(1 for row in candidate_rows if row.get("recommended") is True),
        "skipped": skipped,
        "pipeline_summary": pipeline_summary,
        "policy": "Candidates require frontend approval before entering material_references.",
        "records": candidate_rows,
    }
    _write_jsonl(index_jsonl, candidate_rows)
    _write_json(index_json, summary)
    _write_json(candidate_root / "pipeline_summary.json", pipeline_summary)
    _write_json(candidate_root / "manifest.json", _candidate_manifest(summary))
    _write_candidate_readme(candidate_root / "README.md", summary)
    return summary


def approve_material_candidates(
    session_dir: str | Path,
    *,
    candidate_group_id: str | None = None,
    candidate_ids: list[str] | None = None,
    reviewer: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Promote reviewed candidate files into the formal material reference folder."""

    session_root = Path(session_dir)
    candidate_root = existing_material_candidates_root(session_root)
    index_jsonl = candidate_root / f"{MATERIAL_CANDIDATE_INDEX_BASENAME}.jsonl"
    rows = read_jsonl(index_jsonl)
    if not rows:
        raise FileNotFoundError(f"Material candidate index is not ready: {index_jsonl}")

    explicit_ids = {str(item) for item in (candidate_ids or []) if str(item).strip()}
    if explicit_ids:
        selected = [row for row in rows if str(row.get("candidate_id") or "") in explicit_ids]
    elif candidate_group_id:
        group_rows = [row for row in rows if str(row.get("candidate_group_id") or "") == str(candidate_group_id)]
        selected = [row for row in group_rows if row.get("recommended") is True]
        if not selected:
            selected = _best_candidate_rows(group_rows)
    else:
        raise ValueError("candidate_group_id or candidate_ids is required")
    if not selected:
        raise ValueError("No material candidates matched the approval request")

    approved_at = datetime.now().astimezone().isoformat()
    selected_ids = {str(row.get("candidate_id") or "") for row in selected}
    selected_groups = {str(row.get("candidate_group_id") or "") for row in selected if row.get("candidate_group_id")}
    updated_rows: list[dict[str, Any]] = []
    for row in rows:
        updated = dict(row)
        if str(row.get("candidate_id") or "") in selected_ids:
            updated.update(
                {
                    "candidate_status": "approved",
                    "review_status": "accepted",
                    "approved_at": approved_at,
                    "approved_by": reviewer or "operator",
                    "review_notes": notes,
                }
            )
        elif str(row.get("candidate_group_id") or "") in selected_groups:
            updated["candidate_status"] = "not_selected"
            updated["review_status"] = "not_selected"
        updated_rows.append(updated)

    _write_jsonl(index_jsonl, updated_rows)
    _refresh_candidate_review_metadata(candidate_root, updated_rows)
    _append_review_log(
        candidate_root / MATERIAL_CANDIDATE_REVIEW_LOG,
        {
            "reviewed_at": approved_at,
            "reviewer": reviewer or "operator",
            "decision": "approved",
            "candidate_group_id": candidate_group_id,
            "candidate_ids": sorted(selected_ids),
            "notes": notes,
        },
    )
    sync_summary = reset_material_references_to_approved_candidates(
        session_root,
        approved_rows=[row for row in updated_rows if str(row.get("candidate_id") or "") in selected_ids],
        merge_existing=True,
    )
    return {
        "schema_version": "material_candidate_review.v1",
        "approved_candidate_ids": sorted(selected_ids),
        "approved_count": len(selected_ids),
        "material_references_summary": sync_summary,
        "candidate_index": str(index_jsonl),
    }


def reset_material_references_to_approved_candidates(
    session_dir: str | Path,
    *,
    approved_rows: list[dict[str, Any]],
    merge_existing: bool = True,
) -> dict[str, Any]:
    session_root = Path(session_dir)
    ref_root = material_references_root(session_root)
    candidate_root = existing_material_candidates_root(session_root)
    keyframe_dir = ref_root / KEYFRAME_DIR_NAME
    clip_dir = ref_root / KEY_CLIP_DIR_NAME
    report_dir = ref_root / REPORT_DIR_NAME
    ref_root.mkdir(parents=True, exist_ok=True)
    keyframe_dir.mkdir(parents=True, exist_ok=True)
    clip_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    experiment = _experiment_metadata(session_root)
    formal_root = formal_material_references_root(session_root)
    for folder in (keyframe_dir, clip_dir, report_dir):
        for stale_file in folder.iterdir():
            if stale_file.is_file():
                stale_file.unlink()

    existing_index = ref_root / f"{MATERIAL_INDEX_BASENAME}.jsonl"
    existing_rows = read_jsonl(existing_index) if merge_existing and existing_index.exists() else []
    kept_rows = [row for row in existing_rows if row.get("asset_kind") == REPORT_DIR_NAME]
    promoted: list[dict[str, Any]] = []
    for row in approved_rows:
        source = _stored_path_from_row(row, candidate_root)
        if source is None or not source.is_file():
            continue
        if row.get("asset_kind") == KEYFRAME_DIR_NAME:
            target_dir = keyframe_dir
        elif row.get("asset_kind") == REPORT_DIR_NAME:
            target_dir = report_dir
        else:
            target_dir = clip_dir
        target = target_dir / source.name
        shutil.copy2(source, target)
        promoted.append(_approved_reference_record_from_candidate(row, source, target))

    rows = kept_rows + promoted
    local_index_json = ref_root / f"{MATERIAL_INDEX_BASENAME}.json"
    local_summary = {
        "schema_version": "material_references.approved_candidates.v1",
        "created_at": datetime.now().isoformat(),
        "experiment_id": experiment["id"],
        "experiment_title": experiment["title"],
        "experiment_date": experiment["date"],
        "experiment_label": experiment["label"],
        "session_dir": str(session_root),
        "material_references": str(ref_root),
        "formal_material_references": str(formal_root),
        "simplified_material_references": str(formal_root),
        "local_material_references_mirror": str(ref_root),
        "keyframe_folder": str(keyframe_dir),
        "key_clip_folder": str(clip_dir),
        "report_folder": str(report_dir),
        "formal_keyframe_folder": str(formal_root / KEYFRAME_DIR_NAME),
        "formal_key_clip_folder": str(formal_root / KEY_CLIP_DIR_NAME),
        "formal_report_folder": str(formal_root / REPORT_DIR_NAME),
        "simplified_keyframe_folder": str(formal_root / KEYFRAME_DIR_NAME),
        "simplified_key_clip_folder": str(formal_root / KEY_CLIP_DIR_NAME),
        "simplified_report_folder": str(formal_root / REPORT_DIR_NAME),
        "index_json": str(local_index_json),
        "index_jsonl": str(existing_index),
        "local_index_json": str(local_index_json),
        "local_index_jsonl": str(existing_index),
        "file_count": len(rows),
        "planned_file_count": 0,
        "keyframe_count": sum(1 for row in rows if row.get("asset_kind") == KEYFRAME_DIR_NAME),
        "key_clip_count": sum(1 for row in rows if row.get("asset_kind") == KEY_CLIP_DIR_NAME),
        "report_count": sum(1 for row in rows if row.get("asset_kind") == REPORT_DIR_NAME),
        "naming_rule": NAMING_RULE,
        "policy": "Only frontend-approved candidates are stored in the formal material reference folders.",
        "archive_root": None,
        "excluded_stale_markers": list(STALE_SPLIT_MARKERS),
        "records": rows,
    }
    _write_jsonl(existing_index, rows)
    _write_json(local_index_json, local_summary)
    _write_json(ref_root / "manifest.json", _manifest(local_summary))
    _write_readme(ref_root / "README.md", local_summary)
    _copy_simplified_materials(ref_root, formal_root, local_summary)
    summary = dict(local_summary)
    summary.update(
        {
            "material_references": str(formal_root),
            "keyframe_folder": str(formal_root / KEYFRAME_DIR_NAME),
            "key_clip_folder": str(formal_root / KEY_CLIP_DIR_NAME),
            "index_json": str(formal_root / f"{MATERIAL_INDEX_BASENAME}.json"),
            "index_jsonl": str(formal_root / f"{MATERIAL_INDEX_BASENAME}.jsonl"),
        }
    )
    return summary


def _prepare_reference_root(ref_root: Path, archive_root: Path, *, archive_existing: bool) -> list[dict[str, str]]:
    ref_root.mkdir(parents=True, exist_ok=True)
    archived: list[dict[str, str]] = []
    for name in (KEYFRAME_DIR_NAME, KEY_CLIP_DIR_NAME, "manifest.json", "README.md", f"{MATERIAL_INDEX_BASENAME}.json", f"{MATERIAL_INDEX_BASENAME}.jsonl"):
        target = ref_root / name
        if not target.exists():
            continue
        if archive_existing:
            archived.append(_move_to_archive(target, ref_root, archive_root, "rebuild"))
        elif target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    (ref_root / KEYFRAME_DIR_NAME).mkdir(parents=True, exist_ok=True)
    (ref_root / KEY_CLIP_DIR_NAME).mkdir(parents=True, exist_ok=True)
    return archived


def _prepare_candidate_root(candidate_root: Path, archive_root: Path, *, archive_existing: bool) -> list[dict[str, str]]:
    candidate_root.mkdir(parents=True, exist_ok=True)
    archived: list[dict[str, str]] = []
    for name in (
        KEYFRAME_DIR_NAME,
        KEY_CLIP_DIR_NAME,
        "manifest.json",
        "README.md",
        f"{MATERIAL_CANDIDATE_INDEX_BASENAME}.json",
        f"{MATERIAL_CANDIDATE_INDEX_BASENAME}.jsonl",
        "pipeline_summary.json",
        MATERIAL_CANDIDATE_REVIEW_LOG,
    ):
        target = candidate_root / name
        if not target.exists():
            continue
        if archive_existing:
            archived.append(_move_to_archive(target, candidate_root, archive_root, "rebuild_candidates"))
        elif target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    return archived


def _stored_path_from_row(row: dict[str, Any], root: Path) -> Path | None:
    raw_path = row.get("stored_file") or row.get("source_file")
    if raw_path:
        path = Path(str(raw_path))
        return path if path.is_absolute() else (root / path)
    filename = row.get("stored_filename") or row.get("file_name")
    asset_kind = row.get("asset_kind") or row.get("material_type")
    if filename and asset_kind:
        return root / str(asset_kind) / str(filename)
    return None


def _candidate_record_from_reference(row: dict[str, Any], source_file: Path, target: Path, *, exists: bool) -> dict[str, Any]:
    identity = "|".join(
        str(row.get(key) or "")
        for key in ("micro_segment_id", "parent_segment_id", "asset_kind", "view", "frame_type", "file_name")
    )
    digest = hashlib.sha1(identity.encode("utf-8", errors="ignore")).hexdigest()[:12]
    candidate = dict(row)
    candidate.update(
        {
            "candidate_id": f"material_candidate_{digest}",
            "candidate_group_id": _candidate_group_id(row),
            "candidate_status": "pending",
            "review_status": "pending",
            "review_required": True,
            "recommended": False,
            "stored_file": str(target),
            "stored_filename": target.name,
            "source_reference_file": str(source_file),
            "exists": bool(exists),
            "size_bytes": target.stat().st_size if exists and target.is_file() else 0,
            "quality_score": _candidate_quality_score(row),
            "quality_reasons": [
                "yolo_physical_evidence",
                "frontend_review_required_before_publish",
            ],
        }
    )
    return candidate


def _candidate_group_id(row: dict[str, Any]) -> str:
    group_source = "|".join(
        str(row.get(key) or "")
        for key in ("micro_segment_id", "parent_segment_id", "primary_object", "view", "start_sec", "end_sec")
    )
    digest = hashlib.sha1(group_source.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"candidate_group_{digest}"


def _candidate_quality_score(row: dict[str, Any]) -> float:
    score = 0.55 + min(0.3, max(0, int(row.get("yolo_evidence_count") or 0)) * 0.03)
    if row.get("asset_kind") == KEYFRAME_DIR_NAME:
        score += {"peak": 0.12, "contact": 0.08, "release": 0.06}.get(str(row.get("frame_type") or row.get("frame_role") or ""), 0.03)
    elif row.get("asset_kind") == KEY_CLIP_DIR_NAME:
        score += 0.1
    elif row.get("asset_kind") == REPORT_DIR_NAME:
        score += 0.18 if str(row.get("role") or "").endswith("_pdf") else 0.08
    return round(min(1.0, score), 3)


def _mark_recommended_candidates(rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("candidate_group_id") or ""), []).append(row)
    for group_rows in grouped.values():
        for selected in _best_candidate_rows(group_rows):
            selected["recommended"] = True
            selected["recommendation_reason"] = "best_quality_per_asset_kind"


def _best_candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for asset_kind in (KEYFRAME_DIR_NAME, KEY_CLIP_DIR_NAME, REPORT_DIR_NAME):
        subset = [row for row in rows if row.get("asset_kind") == asset_kind]
        if subset:
            selected.append(max(subset, key=lambda item: (_safe_float(item.get("quality_score")), str(item.get("frame_type") or ""))))
    return selected


def _approved_reference_record_from_candidate(row: dict[str, Any], source_file: Path, target: Path) -> dict[str, Any]:
    approved = dict(row)
    approved.update(
        {
            "material_type": row.get("asset_kind"),
            "stored_file": str(target),
            "stored_filename": target.name,
            "file_name": target.name,
            "source_candidate_file": str(source_file),
            "exists": target.is_file(),
            "size_bytes": target.stat().st_size if target.is_file() else 0,
            "candidate_status": "approved",
            "review_status": "accepted",
            "formal_material_reference": True,
        }
    )
    return approved


def _candidate_manifest(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": summary["schema_version"],
        "created_at": summary["created_at"],
        "updated_at": summary.get("updated_at"),
        "session_dir": summary["session_dir"],
        "candidate_folder": summary["candidate_folder"],
        "keyframe_folder": summary["keyframe_folder"],
        "key_clip_folder": summary["key_clip_folder"],
        "candidate_count": summary["candidate_count"],
        "pending_total": summary["pending_total"],
        "approved_total": summary.get("approved_total", 0),
        "not_selected_total": summary.get("not_selected_total", 0),
        "rejected_total": summary.get("rejected_total", 0),
        "processed_total": summary.get("processed_total", 0),
        "recommended_total": summary["recommended_total"],
        "policy": summary["policy"],
        "pipeline_summary": summary.get("pipeline_summary"),
    }


def _candidate_status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "pending_total": 0,
        "approved_total": 0,
        "not_selected_total": 0,
        "rejected_total": 0,
        "processed_total": 0,
        "recommended_total": 0,
    }
    for row in rows:
        status = str(row.get("candidate_status") or row.get("review_status") or "pending").lower()
        if status == "approved" or str(row.get("review_status") or "").lower() == "accepted":
            counts["approved_total"] += 1
            counts["processed_total"] += 1
        elif status == "not_selected":
            counts["not_selected_total"] += 1
            counts["processed_total"] += 1
        elif status == "rejected":
            counts["rejected_total"] += 1
            counts["processed_total"] += 1
        else:
            counts["pending_total"] += 1
        if row.get("recommended") is True:
            counts["recommended_total"] += 1
    return counts


def _refresh_candidate_review_metadata(candidate_root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    index_json = candidate_root / f"{MATERIAL_CANDIDATE_INDEX_BASENAME}.json"
    existing = _load_json(index_json) if index_json.exists() else {}
    manifest = _load_json(candidate_root / "manifest.json") if (candidate_root / "manifest.json").exists() else {}
    counts = _candidate_status_counts(rows)
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get("candidate_group_id") or row.get("candidate_id") or "ungrouped"), []).append(row)
    pending_groups = 0
    approved_groups = 0
    for group_rows in groups.values():
        statuses = {str(row.get("candidate_status") or row.get("review_status") or "pending").lower() for row in group_rows}
        review_statuses = {str(row.get("review_status") or "").lower() for row in group_rows}
        if "pending" in statuses:
            pending_groups += 1
        if "approved" in statuses or "accepted" in review_statuses:
            approved_groups += 1
    pipeline_summary = dict(existing.get("pipeline_summary") or manifest.get("pipeline_summary") or {})
    pipeline_summary.update(
        {
            "candidate_count": len(rows),
            "group_count": len(groups),
            "groups_waiting_frontend_review": pending_groups,
            "groups_approved": approved_groups,
        }
    )
    summary = {
        **existing,
        "schema_version": existing.get("schema_version") or manifest.get("schema_version") or "yolo_material_candidates.v1",
        "created_at": existing.get("created_at") or manifest.get("created_at") or datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "session_dir": existing.get("session_dir") or manifest.get("session_dir") or "",
        "candidate_folder": str(candidate_root),
        "keyframe_folder": existing.get("keyframe_folder") or manifest.get("keyframe_folder") or str(candidate_root / KEYFRAME_DIR_NAME),
        "key_clip_folder": existing.get("key_clip_folder") or manifest.get("key_clip_folder") or str(candidate_root / KEY_CLIP_DIR_NAME),
        "candidate_count": len(rows),
        "policy": existing.get("policy") or manifest.get("policy") or "Candidates require frontend approval before entering material_references.",
        "pipeline_summary": pipeline_summary,
        "records": rows,
        **counts,
    }
    _write_json(index_json, summary)
    _write_json(candidate_root / "manifest.json", _candidate_manifest(summary))
    _write_candidate_readme(candidate_root / "README.md", summary)
    return summary


def _write_candidate_readme(path: Path, summary: dict[str, Any]) -> None:
    path.write_text(
        f"""# \u5173\u952e\u7d20\u6750\u5019\u9009\u5ba1\u6838

session_dir: {summary["session_dir"]}

- candidates: {summary["candidate_count"]}
- pending: {summary["pending_total"]}
- approved: {summary.get("approved_total", 0)}
- not_selected: {summary.get("not_selected_total", 0)}
- recommended: {summary["recommended_total"]}

Policy: {summary["policy"]}
""",
        encoding="utf-8",
    )


def _append_review_log(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _move_to_archive(src: Path, ref_root: Path, archive_root: Path, reason: str) -> dict[str, str]:
    dest = archive_root / src.relative_to(ref_root)
    dest = _numbered_path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.move(str(src), str(dest))
    else:
        shutil.move(str(src), str(dest))
    return {"source": str(src), "archived_to": str(dest), "reason": reason}


def _numbered_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(1, 1000):
        candidate = path.with_name(f"{path.stem}_{index:03d}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Unable to find free archive path for {path}")


def _annotated_clip_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str], str]:
    lookup: dict[tuple[str, str], str] = {}
    for row in rows:
        segment_id = str(row.get("segment_id") or row.get("parent_segment_id") or "")
        view = str(row.get("view") or row.get("source") or "")
        path = row.get("annotated_clip") or row.get("path") or row.get("file")
        if segment_id and view and path:
            lookup[(segment_id, view)] = str(path)
    return lookup


def _resolve_annotated_clip(
    session_root: Path,
    lookup: dict[tuple[str, str], str],
    segment: dict[str, Any],
    micro: dict[str, Any],
    view: str,
) -> Path | None:
    segment_id = str(segment.get("segment_id") or micro.get("parent_segment_id") or micro.get("segment_id") or "")
    candidates = [
        lookup.get((segment_id, view)),
        micro.get("annotated_clip"),
        segment.get(f"{view}_annotated_clip"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(str(candidate))
        if not path.is_absolute():
            path = session_root / path
        if path.exists():
            return path
    return None


def _segment_view_start(segment: dict[str, Any], view: str) -> float:
    view_data = segment.get(view) if isinstance(segment.get(view), dict) else {}
    if view_data:
        return _safe_float(view_data.get("local_start_sec", view_data.get("start_sec", segment.get("start_sec", 0.0))))
    starts = segment.get("view_start_sec") if isinstance(segment.get("view_start_sec"), dict) else {}
    return _safe_float(starts.get(view, segment.get("start_sec", 0.0)))


def _has_yolo_evidence(micro: dict[str, Any]) -> bool:
    return any(isinstance(row, dict) for row in micro.get("yolo_evidence", []))


def _action_name(primary_object: str) -> str:
    return ACTION_NAME_BY_OBJECT.get(str(primary_object), f"\u624b\u4e0e{primary_object}\u64cd\u4f5c")


def _evidence_times(micro: dict[str, Any], start_sec: float, end_sec: float) -> dict[str, float]:
    evidence = [row for row in micro.get("yolo_evidence", []) if isinstance(row, dict)]
    times = [_safe_float(row.get("time_sec", row.get("timestamp_sec")), start_sec) for row in evidence]
    times = [ts for ts in times if start_sec <= ts <= end_sec]
    if not times:
        midpoint = (start_sec + end_sec) / 2
        return {"peak": midpoint}
    return {
        "contact": min(times),
        "peak": times[len(times) // 2],
        "release": max(times),
    }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _date_label(value: str) -> str:
    parsed = _date_from_text(value)
    if parsed:
        return parsed
    match = re.search(r"(20\d{6})", value)
    return match.group(1) if match else datetime.now().strftime("%Y%m%d")


def _is_stale_identifier(value: Any) -> bool:
    text = str(value or "")
    return any(marker in text for marker in STALE_SPLIT_MARKERS)


def _safe_name(value: str) -> str:
    return re.sub(r'[<>:"/\\|?*\s]+', "_", value).strip("._") or "material"


def _unique_name(used_names: set[str], basename: str, suffix: str) -> str:
    base = _safe_name(basename)
    name = f"{base}{suffix}"
    index = 2
    while name in used_names:
        name = f"{base}_{index:02d}{suffix}"
        index += 1
    used_names.add(name)
    return name


def _record(
    *,
    micro: dict[str, Any],
    segment: dict[str, Any],
    target: Path,
    source: Path,
    material_type: str,
    view: str,
    action_name: str,
    generated: bool,
    dry_run: bool,
    error: str | None,
    frame_type: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "material_reference.item.v1",
        "material_type": material_type,
        "asset_kind": material_type,
        "action_name": action_name,
        "micro_segment_id": micro.get("micro_segment_id"),
        "parent_segment_id": micro.get("parent_segment_id") or micro.get("segment_id"),
        "segment_id": segment.get("segment_id") or micro.get("parent_segment_id") or micro.get("segment_id"),
        "view": view,
        "frame_type": frame_type,
        "start_sec": _safe_float(micro.get("start_sec", micro.get("session_start_sec"))),
        "end_sec": _safe_float(micro.get("end_sec", micro.get("session_end_sec"))),
        "primary_object": micro.get("primary_object") or (micro.get("interaction") or {}).get("primary_object"),
        "source_file": str(source),
        "stored_file": str(target),
        "stored_filename": target.name,
        "file_name": target.name,
        "exists": bool(target.exists()),
        "generated": bool(generated),
        "dry_run": bool(dry_run),
        "error": error,
        "yolo_box_required": True,
        "box_filter": "hand_and_primary_object_only",
        "time_range_sec": f"{_safe_float(micro.get('start_sec', micro.get('session_start_sec'))):.3f}-{_safe_float(micro.get('end_sec', micro.get('session_end_sec'))):.3f}",
        "frame_role": frame_type,
        "yolo_annotated_required": True,
        "yolo_evidence_count": len([row for row in micro.get("yolo_evidence", []) if isinstance(row, dict)]),
    }


def _ffmpeg_available(ffmpeg_path: str | Path) -> bool:
    try:
        subprocess.run([str(ffmpeg_path), "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False


def _run_ffmpeg(args: list[str]) -> None:
    subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def _cut_video(ffmpeg_path: str | Path, source: Path, offset: float, duration: float, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        [
            str(ffmpeg_path),
            "-y",
            "-ss",
            f"{max(0.0, offset):.3f}",
            "-t",
            f"{max(0.1, duration):.3f}",
            "-i",
            str(source),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "22",
            "-movflags",
            "+faststart",
            str(target),
        ]
    )


def _render_filtered_interaction_clip(
    source: Path,
    offset: float,
    duration: float,
    target: Path,
    evidence_rows: list[dict[str, Any]],
    primary_object: str,
    segment_start_sec: float,
) -> None:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("opencv-python is required for filtered interaction clip rendering") from exc

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source clip for filtered rendering: {source}")
    tmp_dir = target.parent.parent / "_render_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_target = tmp_dir / f"render_{uuid.uuid4().hex}.mp4"
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 15.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if fps <= 0 or width <= 0 or height <= 0:
            raise RuntimeError(f"Invalid clip metadata for filtered rendering: {source}")
        writer = cv2.VideoWriter(str(tmp_target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create filtered interaction clip: {target}")

        start = max(0.0, float(offset))
        clip_duration = max(0.1, float(duration))
        end = start + clip_duration
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
        max_frames = max(1, int(round(clip_duration * fps)) + 2)
        frame_index = 0
        while frame_index < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            source_offset_sec = start + frame_index / fps
            if source_offset_sec > end + (1.0 / max(fps, 1.0)):
                break
            evidence = _nearest_evidence_row(
                evidence_rows,
                segment_start_sec + source_offset_sec,
                hold_sec=_annotation_hold_sec(evidence_rows, clip_duration),
            )
            if evidence is not None:
                frame = _draw_filtered_interaction_boxes(frame, evidence, primary_object)
            writer.write(frame)
            frame_index += 1
        writer.release()
        if not tmp_target.exists() or tmp_target.stat().st_size <= 0:
            raise RuntimeError(f"Filtered interaction clip was not written: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.unlink()
        shutil.move(str(tmp_target), str(target))
        _transcode_rendered_clip_for_browser(target)
    finally:
        cap.release()
        tmp_target.unlink(missing_ok=True)
        try:
            tmp_dir.rmdir()
        except OSError:
            pass


def _transcode_rendered_clip_for_browser(path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not path.exists() or path.stat().st_size <= 0:
        return
    tmp = path.with_name(f"{path.stem}.h264_tmp_{uuid.uuid4().hex}{path.suffix}")
    try:
        _run_ffmpeg(
            [
                ffmpeg,
                "-y",
                "-i",
                str(path),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "22",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-an",
                str(tmp),
            ]
        )
        if tmp.exists() and tmp.stat().st_size > 0:
            path.unlink(missing_ok=True)
            shutil.move(str(tmp), str(path))
    except Exception:
        tmp.unlink(missing_ok=True)


def _extract_filtered_interaction_frame(
    source: Path,
    offset: float,
    target: Path,
    evidence_row: dict[str, Any],
    primary_object: str,
) -> None:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("opencv-python is required for filtered interaction keyframes") from exc

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source clip for filtered keyframe: {source}")
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(offset)) * 1000.0)
        ok, frame = cap.read()
        if not ok:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if frame_count > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count - 1))
                ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Cannot read filtered keyframe at {offset:.3f}s from {source}")
        frame = _draw_filtered_interaction_boxes(frame, evidence_row, primary_object)
        target.parent.mkdir(parents=True, exist_ok=True)
        ok, encoded = cv2.imencode(target.suffix if target.suffix else ".jpg", frame)
        if not ok:
            raise RuntimeError(f"Cannot encode filtered interaction keyframe: {target}")
        encoded.tofile(str(target))
    finally:
        cap.release()


def _annotation_hold_sec(evidence_rows: list[dict[str, Any]], clip_duration: float) -> float:
    times = sorted(
        _safe_float(row.get("local_time_sec"), _safe_float(row.get("time_sec"), 0.0))
        for row in evidence_rows
        if isinstance(row, dict) and (row.get("local_time_sec") is not None or row.get("time_sec") is not None)
    )
    deltas = [b - a for a, b in zip(times, times[1:]) if b > a]
    if deltas:
        return max(0.35, min(1.25, max(deltas) * 0.75))
    return max(0.45, min(1.25, float(clip_duration) * 0.65))


def _nearest_evidence_row(
    evidence_rows: list[dict[str, Any]],
    local_time_sec: float,
    *,
    hold_sec: float = 0.45,
) -> dict[str, Any] | None:
    rows = [
        row
        for row in evidence_rows
        if isinstance(row, dict) and (row.get("local_time_sec") is not None or row.get("time_sec") is not None)
    ]
    if not rows:
        return None
    nearest = min(
        rows,
        key=lambda row: abs(_safe_float(row.get("local_time_sec"), _safe_float(row.get("time_sec"), local_time_sec)) - local_time_sec),
    )
    delta = abs(_safe_float(nearest.get("local_time_sec"), _safe_float(nearest.get("time_sec"), local_time_sec)) - local_time_sec)
    return nearest if delta <= max(0.0, float(hold_sec)) else None


def _draw_filtered_interaction_boxes(frame: Any, evidence_row: dict[str, Any], primary_object: str) -> Any:
    try:
        import cv2
    except Exception:  # pragma: no cover
        return frame

    drawn = False
    for detection, color in _filtered_interaction_detections(evidence_row, primary_object, frame=frame):
        bbox = detection.get("bbox")
        if not isinstance(bbox, list) or len(bbox) < 4:
            continue
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = [int(round(float(value))) for value in bbox[:4]]
        x1 = max(0, min(width - 1, x1))
        x2 = max(0, min(width - 1, x2))
        y1 = max(0, min(height - 1, y1))
        y2 = max(0, min(height - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        label = str(detection.get("label") or "")
        confidence = detection.get("confidence")
        text = f"{label} {float(confidence):.2f}" if confidence is not None else label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        text_top = max(0, y1 - text_h - 8)
        cv2.rectangle(frame, (x1, text_top), (min(width - 1, x1 + text_w + 8), y1), color, -1)
        cv2.putText(frame, text, (x1 + 4, max(text_h + 1, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        drawn = True
    if drawn:
        footer = f"YOLO evidence: hand + {canonical_yolo_label(primary_object) or primary_object}"
        cv2.rectangle(frame, (0, 0), (min(frame.shape[1] - 1, 560), 30), (20, 20, 20), -1)
        cv2.putText(frame, footer, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return frame


def _filtered_interaction_detections(
    evidence_row: dict[str, Any],
    primary_object: str,
    *,
    frame: Any | None = None,
) -> list[tuple[dict[str, Any], tuple[int, int, int]]]:
    primary = canonical_yolo_label(primary_object)
    raw_detections = [item for item in evidence_row.get("detections") or [] if isinstance(item, dict)]
    frame_height = frame_width = None
    if frame is not None:
        try:
            frame_height, frame_width = frame.shape[:2]
        except Exception:
            frame_height = frame_width = None
    source_view = str(evidence_row.get("source_view") or evidence_row.get("view") or "")
    detections, _ignored = filter_implausible_detections(
        raw_detections,
        frame_width=frame_width,
        frame_height=frame_height,
        frame=frame,
        source_view=source_view,
    )
    interactions = [
        item
        for item in find_hand_object_interactions(
            detections,
            frame_width=frame_width,
            frame_height=frame_height,
            frame=frame,
            source_view=source_view,
            min_interaction_score=0.1,
        )
        if isinstance(item, dict) and canonical_yolo_label(item.get("object_label") or item.get("target_label") or item.get("object")) == primary
    ]
    interaction = max(interactions, key=lambda item: _safe_float(item.get("score"), 0.0), default=None)
    if interaction is None:
        return []
    hand = _detection_for_labels(detections, HAND_LABELS)
    obj = _detection_for_labels(detections, {primary})
    hand = _interaction_detection(interaction, "hand", hand)
    obj = _interaction_detection(interaction, "object", obj)
    filtered: list[tuple[dict[str, Any], tuple[int, int, int]]] = []
    if obj is not None:
        filtered.append((obj, (255, 170, 0)))
    if hand is not None:
        filtered.append((hand, (0, 200, 80)))
    return filtered


def _detection_for_labels(detections: list[dict[str, Any]], labels: set[str] | frozenset[str]) -> dict[str, Any] | None:
    normalized = {canonical_yolo_label(label) for label in labels if canonical_yolo_label(label)}
    candidates = [
        item
        for item in detections
        if canonical_yolo_label(item.get("label")) in normalized and _bbox(item.get("bbox")) is not None
    ]
    return max(candidates, key=lambda item: _safe_float(item.get("confidence"), 0.0), default=None)


def _interaction_detection(
    interaction: dict[str, Any],
    role: str,
    fallback: dict[str, Any] | None,
) -> dict[str, Any] | None:
    label_key = "hand_label" if role == "hand" else "object_label"
    bbox_key = "hand_bbox" if role == "hand" else "object_bbox"
    bbox = _bbox(interaction.get(bbox_key))
    if bbox is None:
        return fallback
    item = dict(fallback or {})
    item["label"] = canonical_yolo_label(interaction.get(label_key)) or ("gloved_hand" if role == "hand" else "object")
    item["bbox"] = bbox
    if item.get("confidence") is None and interaction.get("score") is not None:
        item["confidence"] = _safe_float(interaction.get("score"), 0.0)
    return item


def _bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(item) for item in value[:4]]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _extract_frame(ffmpeg_path: str | Path, source: Path, offset: float, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        [
            str(ffmpeg_path),
            "-y",
            "-ss",
            f"{max(0.0, offset):.3f}",
            "-i",
            str(source),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(target),
        ]
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def _asset_count(rows: list[dict[str, Any]], asset_kind: str) -> int:
    return sum(1 for row in rows if row.get("asset_kind") == asset_kind)


def _summary_asset_count(summary: dict[str, Any], key: str, asset_kind: str) -> int:
    if key in summary:
        return int(summary.get(key) or 0)
    records = [row for row in (summary.get("records") or []) if isinstance(row, dict)]
    return _asset_count(records, asset_kind)


def _manifest(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": summary["schema_version"],
        "created_at": summary["created_at"],
        "session_dir": summary["session_dir"],
        "material_references": summary.get("material_references"),
        "formal_material_references": summary.get("formal_material_references"),
        "local_material_references_mirror": summary.get("local_material_references_mirror"),
        "experiment_title": summary.get("experiment_title"),
        "experiment_date": summary.get("experiment_date"),
        "experiment_label": summary.get("experiment_label"),
        "keyframe_folder": summary["keyframe_folder"],
        "key_clip_folder": summary["key_clip_folder"],
        "formal_keyframe_folder": summary.get("formal_keyframe_folder") or summary.get("simplified_keyframe_folder"),
        "formal_key_clip_folder": summary.get("formal_key_clip_folder") or summary.get("simplified_key_clip_folder"),
        "index_json": summary["index_json"],
        "index_jsonl": summary["index_jsonl"],
        "file_count": summary["file_count"],
        "planned_file_count": summary["planned_file_count"],
        "keyframe_count": summary["keyframe_count"],
        "key_clip_count": summary["key_clip_count"],
        "report_count": _summary_asset_count(summary, "report_count", REPORT_DIR_NAME),
        "naming_rule": summary["naming_rule"],
        "policy": summary["policy"],
        "archive_root": summary["archive_root"],
        "excluded_stale_markers": summary["excluded_stale_markers"],
    }


def _write_readme(path: Path, summary: dict[str, Any]) -> None:
    path.write_text(
        f"""# {README_TITLE}

session_dir: {summary["session_dir"]}

- {KEYFRAME_DIR_NAME}: {summary["keyframe_count"]} files
- {KEY_CLIP_DIR_NAME}: {summary["key_clip_count"]} files
- {REPORT_DIR_NAME}: {_summary_asset_count(summary, "report_count", REPORT_DIR_NAME)} files
- planned_files: {summary["planned_file_count"]}
- formal_delivery: {summary.get("formal_material_references") or summary.get("material_references")}

Naming rule: {summary["naming_rule"]}

Policy: {summary["policy"]}
""",
        encoding="utf-8",
    )


def _simplified_delivery_root(ref_root: Path, report_summary: dict[str, Any]) -> Path:
    experiment_title = str(
        report_summary.get("experiment_title")
        or report_summary.get("experiment_name")
        or report_summary.get("title")
        or ref_root.parent.name
    )
    experiment_date = str(report_summary.get("experiment_date") or report_summary.get("date") or datetime.now().strftime("%Y%m%d"))
    label = f"{_safe_name(experiment_title)}_{_date_label(experiment_date)}"
    if ref_root.parent.name == "material_references":
        return ref_root
    experiment_dir = ref_root.parent
    outputs_dir = experiment_dir.parent.parent if experiment_dir.parent.name == "experiments" else experiment_dir.parent
    return outputs_dir / "material_references" / label


def _professional_report_record(*, role: str, source: Path, target: Path) -> dict[str, Any]:
    return {
        "schema_version": "material_reference.item.v1",
        "material_type": REPORT_DIR_NAME,
        "asset_kind": REPORT_DIR_NAME,
        "role": role,
        "source_file": str(source),
        "stored_file": str(target),
        "file_name": target.name,
        "exists": target.exists(),
        "size_bytes": target.stat().st_size if target.exists() else 0,
        "review_status": "accepted",
        "delivery_scope": "professional_report",
        "yolo_annotated_required": False,
    }


def _sync_simplified_metadata(
    ref_root: Path,
    simplified_root: Path,
    rows: list[dict[str, Any]],
    report_summary: dict[str, Any],
) -> None:
    simplified_root.mkdir(parents=True, exist_ok=True)
    simplified_rows: list[dict[str, Any]] = []
    for row in rows:
        updated = dict(row)
        stored = Path(str(row.get("stored_file") or ""))
        if stored.exists() and ref_root in stored.parents:
            rel = stored.relative_to(ref_root)
            target = simplified_root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if stored.is_file() and stored.resolve() != target.resolve():
                shutil.copy2(stored, target)
            updated["stored_file"] = str(target)
            updated["exists"] = target.exists()
        simplified_rows.append(updated)
    _write_jsonl(simplified_root / f"{MATERIAL_INDEX_BASENAME}.jsonl", simplified_rows)
    _write_json(simplified_root / f"{MATERIAL_INDEX_BASENAME}.json", {"records": simplified_rows})
    keyframe_count = _asset_count(simplified_rows, KEYFRAME_DIR_NAME)
    key_clip_count = _asset_count(simplified_rows, KEY_CLIP_DIR_NAME)
    report_count = _asset_count(simplified_rows, REPORT_DIR_NAME)
    _write_json(
        simplified_root / "manifest.json",
        {
            "schema_version": "material_references.simplified_delivery.v1",
            "created_at": datetime.now().isoformat(),
            "session_dir": str(ref_root),
            "material_references": str(simplified_root),
            "formal_material_references": str(simplified_root),
            "local_material_references_mirror": str(ref_root),
            "keyframe_folder": str(simplified_root / KEYFRAME_DIR_NAME),
            "key_clip_folder": str(simplified_root / KEY_CLIP_DIR_NAME),
            "report_folder": str(simplified_root / REPORT_DIR_NAME),
            "file_count": len(simplified_rows),
            "planned_file_count": 0,
            "keyframe_count": keyframe_count,
            "key_clip_count": key_clip_count,
            "report_count": report_count or report_summary.get("report_count", 0),
            "naming_rule": NAMING_RULE,
            "policy": "Only frontend-approved candidates are stored in the formal material reference folders.",
            "archive_root": None,
            "excluded_stale_markers": list(STALE_SPLIT_MARKERS),
        },
    )
    _write_readme(
        simplified_root / "README.md",
        {
            "session_dir": str(ref_root),
            "keyframe_count": keyframe_count,
            "key_clip_count": key_clip_count,
            "report_count": report_count or report_summary.get("report_count", 0),
            "planned_file_count": 0,
            "formal_material_references": str(simplified_root),
            "material_references": str(simplified_root),
            "naming_rule": NAMING_RULE,
            "policy": "Only frontend-approved candidates are stored in the formal material reference folders.",
        },
    )

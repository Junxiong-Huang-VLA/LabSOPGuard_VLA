from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .evidence_package import build_evidence_package, validate_evidence_package
from .experiment_focus import extract_experiment_focus_clips, select_experiment_focus_window
from .schemas import read_jsonl, write_jsonl
from .video_utils import get_video_duration_sec


SCHEMA_VERSION = "key_action_frontend_sync.v1"
MATERIAL_INDEX_JSON = "\u7d20\u6750\u7d22\u5f15.json"
MATERIAL_INDEX_JSONL = "\u7d20\u6750\u7d22\u5f15.jsonl"
KEY_MATERIAL_REFERENCES_JSONL = "key_material_references.jsonl"
FOCUS_VIEWS = ("third_person", "first_person")


def sync_frontend_artifacts(
    *,
    target_experiment_dir: str | Path,
    source_session_dir: str | Path | None = None,
    source_key_action_index_dir: str | Path | None = None,
    source_material_root: str | Path | None = None,
    experiment_id: str | None = None,
    experiment_title: str | None = None,
    third_person_video: str | Path | None = None,
    first_person_video: str | Path | None = None,
    archive_existing: bool = True,
    hardlink_media: bool = True,
    approve_materials: bool = True,
    refresh_focus: bool = True,
    force_refresh_focus: bool = False,
    run_yolo_overlay: bool = False,
    require_yolo_overlay: bool = True,
    min_focus_duration_sec: float = 30.0,
    yolo_model_path: str | Path | None = None,
    yolo_first_person_model_path: str | Path | None = None,
    yolo_third_person_model_path: str | Path | None = None,
    yolo_project_root: str | Path | None = None,
    yolo_device: str = "auto",
    yolo_conf: float = 0.25,
    yolo_iou: float = 0.45,
    yolo_detect_fps: float = 5.0,
    dry_run: bool = False,
    output_summary_path: str | Path | None = None,
) -> dict[str, Any]:
    """Synchronize an offline key-action run into a frontend experiment folder.

    The function is intentionally path-based. It does not import LabSOPGuard backend
    modules, so key_action_indexer remains usable as a standalone package.
    """

    target_exp = Path(target_experiment_dir).resolve()
    resolved_experiment_id = str(experiment_id or target_exp.name)
    source_session = Path(source_session_dir).resolve() if source_session_dir else None
    source_key = Path(source_key_action_index_dir).resolve() if source_key_action_index_dir else None
    source_material = Path(source_material_root).resolve() if source_material_root else None
    if source_session is not None:
        source_key = source_key or _resolve_source_key_action_index(source_session)
        source_material = source_material or source_session / "material_references"

    target_key = target_exp / "key_action_index"
    target_material = target_exp / "material_references"
    now = _now_iso()
    operations: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    if not dry_run:
        target_exp.mkdir(parents=True, exist_ok=True)
    if source_key is not None:
        if not source_key.exists():
            errors.append({"code": "missing_source_key_action_index", "path": str(source_key)})
        elif not dry_run:
            operations.append(
                _sync_tree(
                    source_key,
                    target_key,
                    archive_existing=archive_existing,
                    hardlink_media=hardlink_media,
                )
            )
        else:
            operations.append({"operation": "plan_sync_tree", "source": str(source_key), "target": str(target_key)})
    if source_material is not None:
        if not source_material.exists():
            errors.append({"code": "missing_source_material_root", "path": str(source_material)})
        elif not dry_run:
            operations.append(
                _sync_tree(
                    source_material,
                    target_material,
                    archive_existing=archive_existing,
                    hardlink_media=hardlink_media,
                )
            )
        else:
            operations.append({"operation": "plan_sync_tree", "source": str(source_material), "target": str(target_material)})

    raw_videos = {
        "third_person": Path(third_person_video).resolve() if third_person_video else None,
        "first_person": Path(first_person_video).resolve() if first_person_video else None,
    }
    discovered = _discover_raw_videos(target_exp)
    raw_videos = {view: path or discovered.get(view) for view, path in raw_videos.items()}

    replacement_roots: dict[Path | None, Path] = {
        source_key: target_key,
        source_material: target_material,
    }
    if source_session is not None and source_session != source_key:
        replacement_roots[source_session] = target_exp
    replacements = _path_replacements(replacement_roots)

    if not dry_run and target_key.exists():
        _patch_key_action_manifest(
            target_key,
            experiment_id=resolved_experiment_id,
            experiment_title=experiment_title,
            raw_videos=raw_videos,
        )
        _rewrite_text_payloads(target_key, replacements)

    material_counts = {"reference_count": 0, "approved_count": 0}
    evidence_package_summary: dict[str, Any] | None = None
    if not dry_run and target_material.exists():
        _rewrite_text_payloads(target_material, replacements)
        material_counts = _normalize_material_reference_rows(
            target_material,
            experiment_id=resolved_experiment_id,
            approve_materials=approve_materials,
            replacements=replacements,
        )
        try:
            evidence_package_summary = build_evidence_package(
                target_material,
                source_manifest=target_key / "manifest.json" if (target_key / "manifest.json").exists() else None,
                key_action_index_dir=target_key if target_key.exists() else None,
                package_id=f"{resolved_experiment_id}:material_references",
                experiment_id=resolved_experiment_id,
                include_reports=True,
            )
            operations.append(
                {
                    "operation": "refresh_openclaw_evidence_package",
                    "manifest": evidence_package_summary.get("manifest_path"),
                    "reference_count": evidence_package_summary.get("reference_count"),
                    "physical_change_count": evidence_package_summary.get("physical_change_count"),
                }
            )
        except Exception as exc:
            warnings.append({"code": "openclaw_evidence_package_refresh_failed", "error": str(exc)})

    focus_summary: dict[str, Any] | None = None
    if refresh_focus and target_key.exists() and not dry_run:
        try:
            focus_summary = _ensure_focus_clips(
                target_key,
                force=force_refresh_focus,
                min_duration_sec=min_focus_duration_sec,
            )
            operations.append({"operation": "refresh_focus", "available": bool(focus_summary.get("available", True))})
        except Exception as exc:
            warnings.append({"code": "focus_refresh_failed", "error": str(exc)})
    elif dry_run:
        operations.append({"operation": "plan_refresh_focus", "enabled": bool(refresh_focus)})

    yolo_summary: dict[str, Any] | None = None
    if run_yolo_overlay and target_key.exists() and not dry_run:
        try:
            from .yolo_analysis import run_yolo_on_experiment_focus_clips

            yolo_summary = run_yolo_on_experiment_focus_clips(
                target_key,
                model_path=yolo_model_path,
                project_root=yolo_project_root,
                preferred_view="first_person",
                views=list(FOCUS_VIEWS),
                model_paths_by_view={
                    "first_person": yolo_first_person_model_path,
                    "third_person": yolo_third_person_model_path,
                },
                conf=yolo_conf,
                iou=yolo_iou,
                device=yolo_device,
                detect_fps=yolo_detect_fps,
            )
            operations.append({"operation": "render_yolo_focus_overlay", "summary": _compact_yolo_summary(yolo_summary)})
        except Exception as exc:
            warnings.append({"code": "yolo_overlay_failed", "error": str(exc)})
    elif dry_run:
        operations.append({"operation": "plan_yolo_overlay", "enabled": bool(run_yolo_overlay)})

    validation = validate_frontend_artifact_sync(
        target_exp,
        require_yolo_overlay=require_yolo_overlay,
        min_focus_duration_sec=min_focus_duration_sec,
    )
    errors.extend(validation.get("errors") or [])
    warnings.extend(validation.get("warnings") or [])

    if not dry_run:
        _update_frontend_experiment_json(
            target_exp,
            experiment_id=resolved_experiment_id,
            experiment_title=experiment_title,
            raw_videos=raw_videos,
            evidence_count=material_counts.get("reference_count", 0),
            synced_at=now,
            source_session=source_session,
            source_key_action_index=source_key,
            source_material_root=source_material,
        )
        _update_frontend_task_state(target_exp, experiment_id=resolved_experiment_id, synced_at=now)
        _write_key_action_job_status(
            target_key,
            experiment_id=resolved_experiment_id,
            evidence_package_dir=target_material if target_material.exists() else None,
            synced_at=now,
        )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "failed" if errors else ("needs_attention" if warnings else "passed"),
        "dry_run": dry_run,
        "target_experiment_dir": str(target_exp),
        "target_key_action_index_dir": str(target_key),
        "target_material_root": str(target_material),
        "source_session_dir": str(source_session) if source_session else None,
        "source_key_action_index_dir": str(source_key) if source_key else None,
        "source_material_root": str(source_material) if source_material else None,
        "experiment_id": resolved_experiment_id,
        "material_counts": material_counts,
        "evidence_package_summary": evidence_package_summary,
        "focus_summary": focus_summary,
        "yolo_summary": yolo_summary,
        "validation": validation,
        "operations": operations,
        "warnings": warnings,
        "errors": errors,
        "synced_at": now,
    }
    if output_summary_path and not dry_run:
        _write_json(Path(output_summary_path), summary)
    return summary


def validate_frontend_artifact_sync(
    target_experiment_dir: str | Path,
    *,
    require_yolo_overlay: bool = True,
    min_focus_duration_sec: float = 30.0,
) -> dict[str, Any]:
    target_exp = Path(target_experiment_dir).resolve()
    key_dir = target_exp / "key_action_index"
    material_root = target_exp / "material_references"
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    segment_count = _jsonl_count(key_dir / "metadata" / "key_action_segments.jsonl")
    micro_count = _jsonl_count(key_dir / "metadata" / "micro_segments.jsonl")
    yolo_frame_count = _jsonl_count(key_dir / "cv_outputs" / "yolo_frame_rows.jsonl")
    vector_count = _jsonl_count(key_dir / "metadata" / "vector_metadata.jsonl") + _jsonl_count(
        key_dir / "metadata" / "micro_vector_metadata.jsonl"
    )
    if not (key_dir / "manifest.json").exists():
        errors.append({"code": "missing_key_action_manifest", "path": str(key_dir / "manifest.json")})
    if segment_count <= 0:
        errors.append({"code": "missing_key_action_segments", "path": str(key_dir / "metadata" / "key_action_segments.jsonl")})
    if micro_count <= 0:
        warnings.append({"code": "missing_micro_segments", "path": str(key_dir / "metadata" / "micro_segments.jsonl")})
    if yolo_frame_count <= 0:
        warnings.append({"code": "missing_yolo_frame_rows", "path": str(key_dir / "cv_outputs" / "yolo_frame_rows.jsonl")})
    if vector_count <= 0:
        warnings.append({"code": "missing_vector_metadata"})

    segment_source_ranges_available = _segments_have_dual_view_source_ranges(key_dir)
    focus_window = _read_json(key_dir / "metadata" / "experiment_focus_window.json")
    focus_duration = _safe_float(focus_window.get("duration_sec"), 0.0) if isinstance(focus_window, dict) else 0.0
    focus_source = focus_window.get("source") if isinstance(focus_window, dict) else None
    if not focus_window:
        item = {"code": "missing_experiment_focus_window"}
        if segment_source_ranges_available:
            warnings.append({**item, "fallback": "dual_view_segment_source_ranges"})
        else:
            errors.append(item)
    elif focus_duration < min_focus_duration_sec:
        errors.append(
            {
                "code": "short_experiment_focus_window",
                "duration_sec": focus_duration,
                "min_duration_sec": float(min_focus_duration_sec),
                "source": focus_source,
            }
        )
    elif focus_source != "all_true_experiment_episodes":
        warnings.append({"code": "focus_window_not_all_episode_based", "source": focus_source})

    focus_clips = _read_json(key_dir / "metadata" / "experiment_focus_clips.json")
    focus_clip_status = _validate_focus_clip_files(key_dir, focus_duration, annotated=False)
    if segment_source_ranges_available:
        warnings.extend(
            {**item, "fallback": "dual_view_segment_source_ranges"} for item in focus_clip_status["errors"]
        )
    else:
        errors.extend(focus_clip_status["errors"])
    warnings.extend(focus_clip_status["warnings"])
    annotated_status = _validate_focus_clip_files(key_dir, focus_duration, annotated=True)
    if require_yolo_overlay:
        errors.extend(annotated_status["errors"])
    else:
        warnings.extend(annotated_status["errors"])
    warnings.extend(annotated_status["warnings"])

    refs_path = material_root / KEY_MATERIAL_REFERENCES_JSONL
    if not refs_path.exists():
        refs_path = material_root / MATERIAL_INDEX_JSONL
    references = _safe_read_jsonl(refs_path)
    if not references:
        errors.append({"code": "missing_material_references", "path": str(refs_path)})
    absolute_reference_paths = _absolute_reference_path_issues(references)
    if absolute_reference_paths:
        errors.append(
            {
                "code": "non_portable_material_reference_paths",
                "count": len(absolute_reference_paths),
                "examples": absolute_reference_paths[:10],
            }
        )
    missing_files = _missing_material_files(material_root, references)
    if missing_files:
        errors.append({"code": "missing_material_files", "count": len(missing_files), "examples": missing_files[:10]})

    required_package_files = {
        "evidence_package_manifest": material_root / "evidence_package_manifest.json",
        "time_alignment": material_root / "time_alignment.json",
        "physical_change_log": material_root / "physical_change_log.jsonl",
    }
    for name, path in required_package_files.items():
        if not path.exists():
            errors.append({"code": f"missing_{name}", "path": str(path)})
    physical_change_count = _jsonl_count(material_root / "physical_change_log.jsonl")
    if physical_change_count <= 0:
        warnings.append({"code": "empty_physical_change_log", "path": str(material_root / "physical_change_log.jsonl")})

    evidence_package_validation: dict[str, Any] | None = None
    if (material_root / "evidence_package_manifest.json").exists():
        evidence_package_validation = validate_evidence_package(material_root, strict=True)
        if evidence_package_validation.get("status") == "failed":
            errors.append({"code": "evidence_package_validation_failed", "details": evidence_package_validation.get("errors")})
        elif evidence_package_validation.get("status") == "warning":
            warnings.append({"code": "evidence_package_validation_warning", "details": evidence_package_validation.get("warnings")})

    return {
        "schema_version": "key_action_frontend_sync_validation.v1",
        "status": "failed" if errors else ("needs_attention" if warnings else "passed"),
        "target_experiment_dir": str(target_exp),
        "counts": {
            "segment_count": segment_count,
            "micro_segment_count": micro_count,
            "yolo_frame_count": yolo_frame_count,
            "vector_count": vector_count,
            "material_reference_count": len(references),
            "physical_change_count": physical_change_count,
        },
        "focus": {
            "source": focus_source,
            "duration_sec": focus_duration,
            "clips_manifest_available": bool(focus_clips),
        },
        "require_yolo_overlay": require_yolo_overlay,
        "evidence_package_validation": evidence_package_validation,
        "errors": errors,
        "warnings": warnings,
    }


def _sync_tree(
    source: Path,
    target: Path,
    *,
    archive_existing: bool,
    hardlink_media: bool,
) -> dict[str, Any]:
    backup_path: str | None = None
    if target.exists():
        if archive_existing:
            backup = target.parent / ".sync_backups" / f"{target.name}_{_timestamp_label()}"
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(target), str(backup))
            backup_path = str(backup)
        else:
            shutil.rmtree(target)
    file_count = 0
    byte_count = 0
    for src_file in source.rglob("*"):
        if not src_file.is_file():
            continue
        rel = src_file.relative_to(source)
        dst_file = target / rel
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        if hardlink_media and _is_large_media(src_file):
            try:
                os.link(src_file, dst_file)
            except OSError:
                shutil.copy2(src_file, dst_file)
        else:
            shutil.copy2(src_file, dst_file)
        file_count += 1
        try:
            byte_count += dst_file.stat().st_size
        except OSError:
            pass
    return {
        "operation": "sync_tree",
        "source": str(source),
        "target": str(target),
        "backup_path": backup_path,
        "file_count": file_count,
        "byte_count": byte_count,
        "hardlink_media": hardlink_media,
    }


def _resolve_source_key_action_index(source_session: Path) -> Path:
    nested = source_session / "key_action_index"
    if nested.exists():
        return nested
    flat_markers = (
        source_session / "metadata" / "key_action_segments.jsonl",
        source_session / "session_manifest.json",
        source_session / "manifest.json",
    )
    if any(path.exists() for path in flat_markers):
        return source_session
    return nested


def _key_action_manifest_path(key_dir: Path) -> Path:
    manifest_path = key_dir / "manifest.json"
    if manifest_path.exists():
        return manifest_path
    fallback = key_dir / "session_manifest.json"
    return fallback if fallback.exists() else manifest_path


def _patch_key_action_manifest(
    key_dir: Path,
    *,
    experiment_id: str,
    experiment_title: str | None,
    raw_videos: Mapping[str, Path | None],
) -> None:
    source_manifest_path = _key_action_manifest_path(key_dir)
    manifest = _read_json(source_manifest_path)
    if not manifest:
        return
    manifest_path = key_dir / "manifest.json"
    manifest["session_id"] = manifest.get("session_id") or experiment_id
    if experiment_title:
        manifest["experiment_title"] = experiment_title
    manifest["output_dir"] = str(key_dir)
    raw_root = key_dir.parent / "raw"
    if raw_root.exists():
        manifest["source_dataset"] = str(raw_root)
    videos = manifest.get("videos")
    if not isinstance(videos, dict):
        videos = {}
        manifest["videos"] = videos
    for view, path in raw_videos.items():
        if path is None:
            continue
        ref = videos.get(view)
        if not isinstance(ref, dict):
            ref = {}
            videos[view] = ref
        ref["path"] = str(path)
        ref.setdefault("role", view)
        ref.setdefault("camera_id", _camera_id_from_path(path, view))
    _write_json(manifest_path, manifest)


def _segments_have_dual_view_source_ranges(key_dir: Path) -> bool:
    for row in _safe_read_jsonl(key_dir / "metadata" / "key_action_segments.jsonl"):
        if _segment_view_has_source_range(row.get("third_person")) and _segment_view_has_source_range(row.get("first_person")):
            return True
    return False


def _segment_view_has_source_range(view_ref: Any) -> bool:
    if not isinstance(view_ref, Mapping):
        return False
    has_video = bool(view_ref.get("video_path") or view_ref.get("clip_path"))
    start = view_ref.get("local_start_sec")
    end = view_ref.get("local_end_sec")
    try:
        return has_video and float(end) > float(start)
    except (TypeError, ValueError):
        return False


def _normalize_material_reference_rows(
    material_root: Path,
    *,
    experiment_id: str,
    approve_materials: bool,
    replacements: Mapping[str, str],
) -> dict[str, int]:
    total = 0
    approved = 0
    paths = [material_root / KEY_MATERIAL_REFERENCES_JSONL, material_root / MATERIAL_INDEX_JSONL]
    seen_paths: set[Path] = set()
    for path in paths:
        if path in seen_paths or not path.exists():
            continue
        seen_paths.add(path)
        rows = []
        for row in _safe_read_jsonl(path):
            normalized = _normalize_material_row(
                row,
                material_root=material_root,
                experiment_id=experiment_id,
                approve_materials=approve_materials,
                replacements=replacements,
            )
            rows.append(normalized)
        if rows:
            write_jsonl(path, rows)
            total = max(total, len(rows))
            approved = max(
                approved,
                sum(1 for row in rows if str(row.get("candidate_status") or "").lower() == "approved"),
            )

    index_json_path = material_root / MATERIAL_INDEX_JSON
    payload = _read_json(index_json_path)
    records = payload.get("records") if isinstance(payload.get("records"), list) else []
    if records:
        normalized_records = [
            _normalize_material_row(
                row,
                material_root=material_root,
                experiment_id=experiment_id,
                approve_materials=approve_materials,
                replacements=replacements,
            )
            for row in records
            if isinstance(row, dict)
        ]
        payload["records"] = normalized_records
        payload["total"] = len(normalized_records)
        _write_json(index_json_path, payload)
        total = max(total, len(normalized_records))
        approved = max(
            approved,
            sum(1 for row in normalized_records if str(row.get("candidate_status") or "").lower() == "approved"),
        )
    return {"reference_count": total, "approved_count": approved}


def _normalize_material_row(
    row: Mapping[str, Any],
    *,
    material_root: Path,
    experiment_id: str,
    approve_materials: bool,
    replacements: Mapping[str, str],
) -> dict[str, Any]:
    normalized = _rewrite_value(dict(row), replacements)
    if not isinstance(normalized, dict):
        normalized = dict(row)
    normalized["experiment_id"] = experiment_id
    if approve_materials:
        normalized["formal_material_reference"] = True
        normalized["candidate_status"] = "approved"
        normalized.setdefault("review_status", "accepted")
        normalized.setdefault("approved_by", "frontend_artifact_sync")
        normalized.setdefault("approved_at", _now_iso())
        normalized["frontend_bridge_read_only"] = True
    _normalize_material_path_field(normalized, "stored_file", material_root)
    _normalize_material_path_field(normalized, "formal_clip_path", material_root)
    _normalize_material_path_field(normalized, "clip_path", material_root)
    _normalize_material_path_field(normalized, "formal_preview_path", material_root)
    _normalize_material_path_field(normalized, "preview_path", material_root)
    for field in ("source_file", "source_clip", "source_clip_path"):
        _normalize_informational_source_path_field(normalized, field)
    if normalized.get("stored_file"):
        normalized["path_mode"] = "relative_to_material_root"
        normalized.setdefault("package_uri", f"package://material-root/{str(normalized['stored_file']).replace(os.sep, '/')}")
    payload_json = normalized.get("payload_json")
    if isinstance(payload_json, str) and payload_json.strip():
        try:
            payload = json.loads(payload_json)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            payload = _rewrite_value(payload, replacements)
            if isinstance(payload, dict):
                _normalize_material_path_field(payload, "stored_file", material_root)
                for field in ("source_file", "source_clip", "source_clip_path"):
                    _normalize_informational_source_path_field(payload, field)
                payload["experiment_id"] = experiment_id
                if approve_materials:
                    payload["formal_material_reference"] = True
                    payload["candidate_status"] = "approved"
                    payload.setdefault("review_status", "accepted")
                normalized["payload_json"] = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return normalized


def _normalize_material_path_field(row: dict[str, Any], field: str, material_root: Path) -> None:
    value = row.get(field)
    if not value or str(value).startswith("package://"):
        return
    text = str(value)
    path = Path(text)
    if path.is_absolute():
        try:
            row[field] = path.resolve().relative_to(material_root.resolve()).as_posix()
            return
        except Exception:
            match = _find_material_file_by_name(material_root, path.name)
            row[field] = match.relative_to(material_root).as_posix() if match else path.name
            return
    row[field] = text.replace("\\", "/")


def _find_material_file_by_name(material_root: Path, file_name: str) -> Path | None:
    if not file_name:
        return None
    try:
        matches = [path for path in material_root.rglob(file_name) if path.is_file()]
    except OSError:
        return None
    return matches[0] if matches else None


def _normalize_informational_source_path_field(row: dict[str, Any], field: str) -> None:
    value = row.get(field)
    if not value or str(value).startswith("package://"):
        return
    path = Path(str(value))
    if path.is_absolute():
        row[field] = path.name


def _ensure_focus_clips(key_dir: Path, *, force: bool, min_duration_sec: float) -> dict[str, Any]:
    window = select_experiment_focus_window(key_dir)
    existing = _read_json(key_dir / "metadata" / "experiment_focus_clips.json")
    if not force and _plain_focus_clips_look_current(key_dir, window, min_duration_sec=min_duration_sec):
        return existing or {"available": True, "window": window}
    return extract_experiment_focus_clips(key_dir, dry_run=False)


def _plain_focus_clips_look_current(key_dir: Path, window: Mapping[str, Any], *, min_duration_sec: float) -> bool:
    expected = _safe_float(window.get("duration_sec"), 0.0)
    if expected < min_duration_sec:
        return False
    for view in FOCUS_VIEWS:
        path = key_dir / "clips" / "experiment_focus" / f"{view}.mp4"
        if not path.exists() or path.stat().st_size <= 0:
            return False
        duration = _safe_duration(path)
        if duration is not None and abs(duration - expected) > max(2.0, expected * 0.05):
            return False
    return True


def _validate_focus_clip_files(key_dir: Path, focus_duration: float, *, annotated: bool) -> dict[str, list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    suffix = "_yolo_annotated.mp4" if annotated else ".mp4"
    for view in FOCUS_VIEWS:
        path = key_dir / "clips" / "experiment_focus" / f"{view}{suffix}"
        if not path.exists() or path.stat().st_size <= 0:
            errors.append({"code": "missing_focus_annotated_clip" if annotated else "missing_focus_clip", "view": view, "path": str(path)})
            continue
        if focus_duration > 0:
            duration = _safe_duration(path)
            if duration is not None and abs(duration - focus_duration) > max(2.0, focus_duration * 0.05):
                warnings.append(
                    {
                        "code": "focus_clip_duration_mismatch",
                        "view": view,
                        "annotated": annotated,
                        "path": str(path),
                        "duration_sec": duration,
                        "expected_duration_sec": focus_duration,
                    }
                )
    return {"errors": errors, "warnings": warnings}


def _update_frontend_experiment_json(
    target_exp: Path,
    *,
    experiment_id: str,
    experiment_title: str | None,
    raw_videos: Mapping[str, Path | None],
    evidence_count: int,
    synced_at: str,
    source_session: Path | None,
    source_key_action_index: Path | None,
    source_material_root: Path | None,
) -> None:
    path = target_exp / "experiment.json"
    exp = _read_json(path)
    exp.setdefault("experiment_id", experiment_id)
    if experiment_title:
        exp["title"] = experiment_title
        exp.setdefault("experiment_name", experiment_title)
    exp["status"] = "analyzed"
    exp["processing_stage"] = "key_action_index_completed"
    exp["processing_error"] = None
    exp["analyzed_at"] = synced_at
    exp["completed_at"] = synced_at
    exp["updated_at"] = synced_at
    exp["evidence_count"] = int(evidence_count or exp.get("evidence_count") or 0)
    videos = [str(path) for path in raw_videos.values() if path]
    if videos:
        exp["video_paths"] = videos
        exp["video_asset_id"] = exp.get("video_asset_id") or f"{experiment_id}:key-action-video:0"
    output_paths = exp.get("output_paths") if isinstance(exp.get("output_paths"), dict) else {}
    output_paths.update(
        {
            "key_action_index": str(target_exp / "key_action_index"),
            "key_material_references_jsonl": str(target_exp / "material_references" / KEY_MATERIAL_REFERENCES_JSONL),
            "key_material_reference_manifest_json": str(target_exp / "material_references" / "evidence_package_manifest.json"),
            "physical_change_log_jsonl": str(target_exp / "material_references" / "physical_change_log.jsonl"),
            "time_alignment_json": str(target_exp / "material_references" / "time_alignment.json"),
        }
    )
    exp["output_paths"] = output_paths
    metadata = exp.get("metadata") if isinstance(exp.get("metadata"), dict) else {}
    previous_bridge = metadata.get("frontend_bridge") if isinstance(metadata.get("frontend_bridge"), dict) else {}
    metadata["frontend_bridge"] = {
        "schema_version": SCHEMA_VERSION,
        "synced_at": synced_at,
        "source_session": str(source_session) if source_session else previous_bridge.get("source_session"),
        "source_key_action_index": str(source_key_action_index) if source_key_action_index else previous_bridge.get("source_key_action_index"),
        "source_material_references": str(source_material_root) if source_material_root else previous_bridge.get("source_material_references"),
        "mode": "read_only_artifact_sync",
    }
    exp["metadata"] = metadata
    _write_json(path, exp)


def _update_frontend_task_state(target_exp: Path, *, experiment_id: str, synced_at: str) -> None:
    exp = _read_json(target_exp / "experiment.json")
    task_id = exp.get("analysis_job_id")
    tasks_dir = target_exp.parent / "tasks"
    if not task_id or not tasks_dir.exists():
        return
    task_path = tasks_dir / f"{task_id}.json"
    if not task_path.exists():
        return
    task = _read_json(task_path)
    task.update(
        {
            "task_id": task_id,
            "experiment_id": experiment_id,
            "status": "completed",
            "current_stage": "key_action_index_completed",
            "progress": 1.0,
            "completed_at": synced_at,
            "updated_at": synced_at,
            "error_type": None,
            "error_message": None,
        }
    )
    _write_json(task_path, task)


def _write_key_action_job_status(
    key_dir: Path,
    *,
    experiment_id: str,
    evidence_package_dir: Path | None,
    synced_at: str,
) -> None:
    if not key_dir.exists():
        return
    metadata_dir = key_dir / "metadata"
    cv_dir = key_dir / "cv_outputs"
    summary = {
        "source": "key_action_index_metadata",
        "segment_count": _jsonl_count(metadata_dir / "key_action_segments.jsonl"),
        "micro_segment_count": _jsonl_count(metadata_dir / "micro_segments.jsonl"),
        "interaction_count": _yolo_interaction_count(cv_dir / "yolo_frame_rows.jsonl"),
        "raw_yolo_interaction_count": _yolo_interaction_count(cv_dir / "yolo_frame_rows.jsonl"),
        "vector_count": _jsonl_count(metadata_dir / "vector_metadata.jsonl") + _jsonl_count(metadata_dir / "micro_vector_metadata.jsonl"),
    }
    if evidence_package_dir is not None:
        summary["evidence_package"] = {
            "path": str(evidence_package_dir),
            "manifest": str(evidence_package_dir / "evidence_package_manifest.json"),
            "portable": (evidence_package_dir / "evidence_package_manifest.json").exists(),
        }
    payload = {
        "schema_version": "key_action_job_status.v1",
        "status": "completed",
        "progress": 1.0,
        "message": "Offline key-action evidence package synchronized for frontend review.",
        "experiment_id": experiment_id,
        "output_dir": str(key_dir),
        "completed_at": synced_at,
        "updated_at": synced_at,
        "summary": summary,
    }
    _write_json(key_dir / "job_status.json", payload)


def _rewrite_text_payloads(root: Path, replacements: Mapping[str, str]) -> None:
    if not replacements:
        return
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl", ".md", ".txt", ".csv"}:
            continue
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            continue
        replaced = _rewrite_text(text, replacements)
        if replaced != text:
            path.write_text(replaced, encoding="utf-8")


def _rewrite_value(value: Any, replacements: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        return _rewrite_text(value, replacements)
    if isinstance(value, list):
        return [_rewrite_value(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: _rewrite_value(item, replacements) for key, item in value.items()}
    return value


def _rewrite_text(text: str, replacements: Mapping[str, str]) -> str:
    updated = text
    for source, target in replacements.items():
        updated = updated.replace(source, target)
    return updated


def _path_replacements(paths: Mapping[Path | None, Path]) -> dict[str, str]:
    replacements: dict[str, str] = {}
    for source, target in paths.items():
        if source is None:
            continue
        for source_variant, target_variant in zip(_path_text_variants(source), _path_text_variants(target)):
            replacements[source_variant] = target_variant
    return replacements


def _path_text_variants(path: Path) -> list[str]:
    raw = str(path)
    forward = raw.replace("\\", "/")
    escaped = json.dumps(raw, ensure_ascii=False)[1:-1]
    escaped_forward = json.dumps(forward, ensure_ascii=False)[1:-1]
    variants = [raw, escaped, forward, escaped_forward]
    result: list[str] = []
    for item in variants:
        if item and item not in result:
            result.append(item)
    return result


def _discover_raw_videos(target_exp: Path) -> dict[str, Path | None]:
    raw_dir = target_exp / "raw"
    videos = sorted(path for path in raw_dir.glob("*.mp4") if path.is_file()) if raw_dir.exists() else []
    result: dict[str, Path | None] = {"third_person": None, "first_person": None}
    for path in videos:
        name = path.name.lower()
        if result["first_person"] is None and ("first" in name or "operator" in name or "camera_01" in name):
            result["first_person"] = path
        if result["third_person"] is None and ("third" in name or "top" in name or "camera_00" in name):
            result["third_person"] = path
    if result["third_person"] is None and videos:
        result["third_person"] = videos[0]
    if result["first_person"] is None and len(videos) > 1:
        result["first_person"] = videos[1] if videos[1] != result["third_person"] else videos[0]
    return result


def _compact_yolo_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "available": bool(summary.get("available")),
        "clips": int(summary.get("clips") or 0),
        "detections": int(summary.get("detections") or 0),
        "views": summary.get("views"),
        "errors": summary.get("errors") or [],
    }


def _absolute_reference_path_issues(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    fields = ("stored_file", "formal_clip_path", "clip_path", "formal_preview_path", "preview_path")
    for index, row in enumerate(rows, start=1):
        for field in fields:
            value = row.get(field)
            if value and _is_absolute_path_text(value):
                issues.append({"row_index": index, "field": field, "path": str(value)})
    return issues


def _missing_material_files(material_root: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    missing: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        value = row.get("stored_file") or row.get("formal_clip_path") or row.get("clip_path") or row.get("formal_preview_path") or row.get("preview_path")
        if not value or str(value).startswith("package://") or _is_absolute_path_text(value):
            continue
        path = material_root / str(value)
        if not path.exists():
            missing.append({"row_index": index, "path": str(value), "material_id": row.get("material_id")})
    return missing


def _is_absolute_path_text(value: Any) -> bool:
    text = str(value or "")
    return bool(text and not text.startswith("package://") and (Path(text).is_absolute() or (len(text) > 2 and text[1:3] in {":\\", ":/"})))


def _is_large_media(path: Path) -> bool:
    return path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _camera_id_from_path(path: Path, view: str) -> str:
    name = path.stem
    for token in name.split("_"):
        if token.lower().startswith("camera"):
            return token
    return "camera_01" if view == "first_person" else "camera_00"


def _jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return sum(1 for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip())
    except OSError:
        return 0


def _yolo_interaction_count(path: Path) -> int:
    total = 0
    for row in _safe_read_jsonl(path):
        interactions = row.get("hand_object_interactions")
        if isinstance(interactions, list):
            total += len(interactions)
    return total


def _safe_read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        return [row for row in read_jsonl(path) if isinstance(row, dict)]
    except Exception:
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
        return rows


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_duration(path: Path) -> float | None:
    try:
        return float(get_video_duration_sec(path))
    except Exception:
        return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timestamp_label() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


__all__ = [
    "SCHEMA_VERSION",
    "sync_frontend_artifacts",
    "validate_frontend_artifact_sync",
]

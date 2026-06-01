from __future__ import annotations

import csv
import json
import hashlib
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

from .physical_evidence import (
    PHYSICAL_EVIDENCE_MIN_FRAMES,
    evidence_view,
    physical_evidence_policy_summary,
    valid_yolo_physical_evidence,
    yolo_physical_evidence_diagnostics,
)
from .schemas import read_jsonl
from .semantic_actions import enhance_material_semantics
from .tracklet_annotations import build_tracklet_annotations, detections_for_time, summarize_tracklets
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
OPENCLAW_EVIDENCE_PACKAGE_FILES = (
    "evidence_package_manifest.json",
    "key_material_references.jsonl",
    "key_material_references.sqlite",
    "key_material_references.sqlite-shm",
    "key_material_references.sqlite-wal",
    "physical_change_log.jsonl",
    "time_alignment.json",
)
MATERIAL_TAXONOMY_SCHEMA_VERSION = "material_action_taxonomy.v1"
MATERIAL_REFERENCE_TRACE_SCHEMA_VERSION = "material_reference_trace.v1"
MATERIAL_CANDIDATE_DISPOSITION_SCHEMA_VERSION = "material_candidate_disposition.v1"
MIN_RECOMMENDED_HAND_CONFIDENCE = 0.65
STRICT_PHYSICAL_EVIDENCE_MODE = "strict_yolo_physical_evidence"
SPARSE_PHYSICAL_EVIDENCE_MODE = "sparse_yolo_interaction_review_required"
PAIRED_VIEW_CONTEXT_MODE = "paired_view_time_alignment"
MICRO_MATERIAL_EVIDENCE_WINDOW_PAD_SEC = 0.5
PAIRED_CONTEXT_GATE_VERSION = "paired_view_context_scene_gate.v1"
FORMAL_MATERIAL_PUBLISH_GATE_VERSION = "formal_dual_view_action_material_gate.v1"
DUAL_VIEW_ACTION_EVENT_KEYS = (
    "dual_event_id",
    "dual_view_event_id",
    "dual_view_action_event_id",
    "paired_event_id",
    "paired_action_event_id",
    "aligned_event_id",
    "alignment_event_id",
    "time_alignment_event_id",
    "physical_action_alignment_event_id",
)
YOLO_SUFFIX = "YOLO\u6807\u6ce8"
NAMING_RULE = "\u6b63\u5f0f\u4ea4\u4ed8\u76ee\u5f55=\u5b9e\u9a8c\u7c7b\u578b_\u65e5\u671f\uff1b\u5173\u952e\u5e27/\u5173\u952e\u7247\u6bb5=\u5b9e\u9a8c\u7c7b\u578b_\u52a8\u4f5c\u7c7b\u578b_\u65e5\u671f[_\u5e8f\u53f7].\u6269\u5c55\u540d"
README_TITLE = "\u5173\u952e\u7269\u7406\u52a8\u4f5c\u7d20\u6750\u5f15\u7528"
ANNOTATION_COLOR_BY_LABEL = {
    "gloved_hand": (0, 200, 80),
    "hand": (0, 200, 80),
    "paper": (255, 190, 0),
    "weighing_paper": (255, 190, 0),
    "balance": (0, 165, 255),
    "scale": (0, 165, 255),
    "reagent_bottle": (220, 70, 220),
    "sample_bottle": (230, 80, 120),
    "sample_bottle_blue": (255, 90, 40),
    "spatula": (0, 220, 255),
    "pipette": (255, 80, 180),
    "pipette_tip": (255, 120, 210),
    "beaker": (90, 220, 220),
    "container": (80, 210, 160),
    "tube": (120, 210, 255),
}

STALE_SPLIT_MARKERS = (
    "seg_000001_part02",
    "seg_000001_part03",
    "seg_000001_part04",
    "part02",
    "part03",
    "part04",
)

NON_REAL_ASSET_MARKERS = (
    "placeholder",
    "poster",
    "synthetic",
    "dry_run",
    "dry-run",
    "black_screen",
    "white_screen",
    "blank_screen",
    "黑屏",
    "白屏",
    "黑白屏",
)

MATERIAL_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MATERIAL_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
MATERIAL_SOURCE_PTS_TOLERANCE_SEC = 0.05
MaterialGenerationTask = Callable[[], tuple[dict[str, Any], Path, bool, str | None]]
_MATERIAL_FILE_REAL_CACHE: dict[tuple[str, int, int, bool], bool] = {}
_MATERIAL_FILE_REAL_CACHE_LOCK = threading.Lock()
_MATERIAL_FILE_SHA256_CACHE: dict[tuple[str, int, int], str] = {}
_MATERIAL_FILE_SHA256_CACHE_LOCK = threading.Lock()
_SOURCE_VIDEO_DURATION_CACHE: dict[tuple[str, int, int], float | None] = {}
_SOURCE_VIDEO_DURATION_CACHE_LOCK = threading.Lock()


def _material_file_stat_key(path: Path, *, dry_run: bool = False) -> tuple[str, int, int, bool] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    inode = int(getattr(stat, "st_ino", 0) or 0)
    identity = f"{int(getattr(stat, 'st_dev', 0) or 0)}:{inode}" if inode else str(path.resolve())
    return identity, int(stat.st_size), int(stat.st_mtime_ns), bool(dry_run)


def _material_file_sha256(path: Path) -> str | None:
    key = _material_file_stat_key(path, dry_run=False)
    if key is None:
        return None
    sha_key = (key[0], key[1], key[2])
    with _MATERIAL_FILE_SHA256_CACHE_LOCK:
        cached = _MATERIAL_FILE_SHA256_CACHE.get(sha_key)
        if cached is not None:
            return cached
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    value = digest.hexdigest()
    with _MATERIAL_FILE_SHA256_CACHE_LOCK:
        _MATERIAL_FILE_SHA256_CACHE[sha_key] = value
    return value


def _material_file_is_real(path: Path | None, *, dry_run: bool = False) -> bool:
    if dry_run or path is None:
        return False
    cache_key = _material_file_stat_key(path, dry_run=dry_run)
    if cache_key is not None:
        with _MATERIAL_FILE_REAL_CACHE_LOCK:
            cached = _MATERIAL_FILE_REAL_CACHE.get(cache_key)
            if cached is not None:
                return cached
    try:
        if not path.is_file() or path.stat().st_size <= 0:
            if cache_key is not None:
                with _MATERIAL_FILE_REAL_CACHE_LOCK:
                    _MATERIAL_FILE_REAL_CACHE[cache_key] = False
            return False
        path_text = path.name.lower()
        if any(marker in path_text for marker in NON_REAL_ASSET_MARKERS):
            if cache_key is not None:
                with _MATERIAL_FILE_REAL_CACHE_LOCK:
                    _MATERIAL_FILE_REAL_CACHE[cache_key] = False
            return False
        with path.open("rb") as handle:
            header = handle.read(128).upper()
    except OSError:
        return False
    if header.startswith(b"DRY RUN") or b"PLACEHOLDER" in header or b"SYNTHETIC" in header:
        result = False
    else:
        result = _material_visual_content_is_real(path)
    if cache_key is not None:
        with _MATERIAL_FILE_REAL_CACHE_LOCK:
            _MATERIAL_FILE_REAL_CACHE[cache_key] = result
    return result


def _material_visual_content_is_real(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix in MATERIAL_IMAGE_EXTENSIONS:
        return _image_material_visual_is_real(path)
    if suffix in MATERIAL_VIDEO_EXTENSIONS:
        return _video_material_visual_is_real(path)
    return True


def _image_material_visual_is_real(path: Path) -> bool:
    try:
        from PIL import Image

        image = Image.open(path).convert("RGB")
        image.thumbnail((128, 128))
        pixels = list(image.getdata())
    except Exception:
        return True
    return _rgb_pixels_are_real_material(pixels)


def _video_material_visual_is_real(path: Path) -> bool:
    try:
        import cv2
        import numpy as np

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return True
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        positions = [0]
        if frame_count > 4:
            positions.extend([max(0, frame_count // 2), max(0, frame_count - 2)])
        sampled: list[Any] = []
        for position in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, position)
            ok, frame = cap.read()
            if ok and frame is not None:
                sampled.append(frame)
        cap.release()
        if not sampled:
            return True
        return any(_cv_frame_is_real_material(frame, np_module=np) for frame in sampled)
    except Exception:
        return True


def _cv_frame_is_real_material(frame: Any, *, np_module: Any) -> bool:
    try:
        height, width = frame.shape[:2]
        step_y = max(1, height // 96)
        step_x = max(1, width // 128)
        sample = frame[::step_y, ::step_x, :3].astype("float32")
        if sample.size <= 0:
            return False
        channels = sample.reshape((-1, 3))
        luma = (0.114 * channels[:, 0]) + (0.587 * channels[:, 1]) + (0.299 * channels[:, 2])
        mean_luma = float(luma.mean())
        std_luma = float(luma.std())
        dark_ratio = float((luma < 8.0).mean())
        bright_ratio = float((luma > 247.0).mean())
        channel_range = channels.max(axis=1) - channels.min(axis=1)
        color_std = float(np_module.mean(channel_range))
    except Exception:
        return True
    return _visual_stats_are_real_material(mean_luma, std_luma, dark_ratio, bright_ratio, color_std)


def _rgb_pixels_are_real_material(pixels: list[tuple[int, int, int]]) -> bool:
    if not pixels:
        return False
    count = float(len(pixels))
    luma_values = [(0.299 * r) + (0.587 * g) + (0.114 * b) for r, g, b in pixels]
    mean_luma = sum(luma_values) / count
    variance = sum((value - mean_luma) ** 2 for value in luma_values) / count
    std_luma = variance ** 0.5
    dark_ratio = sum(1 for value in luma_values if value < 8.0) / count
    bright_ratio = sum(1 for value in luma_values if value > 247.0) / count
    color_std = sum(max(pixel) - min(pixel) for pixel in pixels) / count
    return _visual_stats_are_real_material(mean_luma, std_luma, dark_ratio, bright_ratio, color_std)


def _visual_stats_are_real_material(
    mean_luma: float,
    std_luma: float,
    dark_ratio: float,
    bright_ratio: float,
    color_std: float,
) -> bool:
    if dark_ratio >= 0.985 or bright_ratio >= 0.985:
        return False
    if mean_luma <= 4.0 or mean_luma >= 251.0:
        return False
    if std_luma < 1.2 and color_std < 1.2:
        return False
    return True


def _row_has_non_real_marker(row: Mapping[str, Any]) -> bool:
    if bool(row.get("placeholder") or row.get("dry_run") or row.get("dry_run_placeholder")):
        return True
    fields: list[Any] = [
        row.get("candidate_source"),
        row.get("source_type"),
        row.get("asset_source"),
        row.get("role"),
        row.get("reason"),
        row.get("missing_reason"),
        Path(str(row.get("stored_file") or "")).name,
        Path(str(row.get("source_file") or "")).name,
        row.get("file_name"),
        row.get("stored_filename"),
    ]
    for key in ("quality_reasons", "warnings", "review_reason_codes"):
        value = row.get(key)
        if isinstance(value, list):
            fields.extend(value)
        else:
            fields.append(value)
    text = " ".join(str(value or "").lower() for value in fields)
    return any(marker in text for marker in NON_REAL_ASSET_MARKERS)


def _material_row_is_publishable(row: Mapping[str, Any], *, root: Path | None = None) -> tuple[bool, str]:
    asset_kind = str(row.get("asset_kind") or row.get("material_type") or "")
    if asset_kind not in {KEYFRAME_DIR_NAME, KEY_CLIP_DIR_NAME, REPORT_DIR_NAME}:
        return False, "unsupported_material_asset_kind"
    source = _stored_path_from_row(dict(row), root) if root is not None else Path(str(row.get("stored_file") or ""))
    if source is None:
        return False, "missing_stored_file"
    if not _material_file_is_real(source, dry_run=bool(row.get("dry_run"))):
        return False, "stored_file_not_real_video_material"
    if _row_has_non_real_marker(row):
        return False, "non_real_asset_marker"
    if row.get("source_real") is False:
        return False, "source_real_false"
    pts_ok, pts_reason, _pts_details = _material_row_source_pts_gate(row, root=root)
    if not pts_ok:
        return False, pts_reason
    return True, ""


def _material_link_or_copy(source: Path, target: Path) -> None:
    """Materialize a file while avoiding duplicate video copies on the same volume."""

    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        if target.exists():
            try:
                if source.resolve() == target.resolve():
                    return
            except OSError:
                pass
            try:
                source_stat = source.stat()
                target_stat = target.stat()
                if (
                    source_stat.st_size == target_stat.st_size
                    and int(source_stat.st_mtime_ns) == int(target_stat.st_mtime_ns)
                ):
                    return
            except OSError:
                pass
            target.unlink()
    except PermissionError:
        if target.exists():
            return
        raise

    if os.environ.get("KEY_ACTION_MATERIAL_HARDLINK_ENABLED", "1").strip().lower() not in {
        "",
        "0",
        "false",
        "no",
        "off",
    }:
        try:
            os.link(source, target)
            return
        except OSError:
            pass
    try:
        shutil.copy2(source, target)
    except PermissionError:
        if not target.exists():
            raise


def _allow_paired_view_context_material() -> bool:
    return str(os.environ.get("KEY_ACTION_ALLOW_PAIRED_VIEW_CONTEXT_MATERIAL", "1")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _paired_view_context_scene_gate_enabled() -> bool:
    return str(os.environ.get("KEY_ACTION_PAIRED_VIEW_CONTEXT_SCENE_GATE", "1")).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _candidate_is_paired_view_context(row: Mapping[str, Any]) -> bool:
    return (
        str(row.get("physical_evidence_mode") or "") == PAIRED_VIEW_CONTEXT_MODE
        or str(row.get("candidate_source") or "") in {"paired_view_micro_segment_key_asset_reference", PAIRED_VIEW_CONTEXT_MODE}
        or str(row.get("box_filter") or "") == "paired_view_time_alignment_asset_reference"
    )


def _paired_context_keyframe_path(row: Mapping[str, Any]) -> Path | None:
    source_file = Path(str(row.get("source_file") or row.get("source_reference_file") or ""))
    if source_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"} and source_file.is_file():
        return source_file
    micro_segment_id = str(row.get("micro_segment_id") or "").strip()
    view = str(row.get("view") or row.get("camera_view") or "").strip()
    if not micro_segment_id or not view:
        return None
    for key in ("source_file", "source_clip", "source_clip_path", "source_reference_file"):
        value = row.get(key)
        if not value:
            continue
        candidate = Path(str(value))
        parts = list(candidate.parts)
        try:
            root_index = parts.index("clips")
        except ValueError:
            try:
                root_index = parts.index("keyframes")
            except ValueError:
                continue
        root = Path(*parts[:root_index])
        frame_path = root / "keyframes" / "micro" / micro_segment_id / view / "peak.jpg"
        if frame_path.is_file():
            return frame_path
    return None


def paired_view_context_scene_gate_passed(row: Mapping[str, Any]) -> bool:
    """Reject paired-view context frames that are not an actual lab-bench view."""

    if not _candidate_is_paired_view_context(row):
        return True
    if not _paired_view_context_scene_gate_enabled():
        return True
    frame_path = _paired_context_keyframe_path(row)
    if frame_path is None:
        return False
    try:
        from PIL import Image
        import colorsys

        image = Image.open(frame_path).convert("RGB")
        image.thumbnail((160, 90))
        pixels = list(image.getdata())
        if not pixels:
            return False
        hsv = [colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0) for r, g, b in pixels]
        count = float(len(hsv))
        mean_saturation = sum(s for _h, s, _v in hsv) / count
        dark_ratio = sum(1 for _h, _s, v in hsv if v < 0.08) / count
        bright_ratio = sum(1 for _h, _s, v in hsv if v > 0.92) / count
        lab_color_ratio = sum(
            1
            for h, s, v in hsv
            if s > 0.25
            and v > 0.15
            and (
                0.52 <= h <= 0.72  # blue lab mat / gloves
                or h < 0.06
                or h > 0.93  # red caps / bottle labels
                or 0.06 <= h <= 0.16  # orange/yellow racks and labels
                or 0.22 <= h <= 0.46  # green annotations / labels
            )
        ) / count
        passed = (
            mean_saturation >= 0.14
            and lab_color_ratio >= 0.08
            and dark_ratio < 0.65
            and bright_ratio < 0.65
        )
        if isinstance(row, dict):
            row["paired_context_scene_gate"] = {
                "schema_version": PAIRED_CONTEXT_GATE_VERSION,
                "status": "passed" if passed else "failed",
                "frame_path": str(frame_path),
                "mean_saturation": round(mean_saturation, 6),
                "lab_color_ratio": round(lab_color_ratio, 6),
                "dark_ratio": round(dark_ratio, 6),
                "bright_ratio": round(bright_ratio, 6),
            }
        return passed
    except Exception:
        if isinstance(row, dict):
            row["paired_context_scene_gate"] = {
                "schema_version": PAIRED_CONTEXT_GATE_VERSION,
                "status": "failed",
                "frame_path": str(frame_path),
                "reason": "scene_gate_exception",
            }
        return False


def _candidate_formal_promotable(row: Mapping[str, Any]) -> bool:
    if _candidate_is_paired_view_context(row) and not _allow_paired_view_context_material():
        return False
    if _candidate_is_paired_view_context(row) and not paired_view_context_scene_gate_passed(row):
        return False
    return True


def _require_dual_view_complete_material_groups() -> bool:
    return _env_truthy("KEY_ACTION_REQUIRE_DUAL_VIEW_COMPLETE_MATERIAL_GROUPS", True)


def _require_reliable_dual_view_alignment() -> bool:
    return _env_truthy("KEY_ACTION_REQUIRE_RELIABLE_DUAL_VIEW_ALIGNMENT", True)


def _require_formal_dual_view_action_event() -> bool:
    return _env_truthy("KEY_ACTION_REQUIRE_FORMAL_DUAL_VIEW_ACTION_EVENT", True)


def _require_same_view_physical_evidence_for_formal_material() -> bool:
    return _env_truthy("KEY_ACTION_REQUIRE_SAME_VIEW_PHYSICAL_EVIDENCE_FOR_FORMAL_MATERIAL", True)


def _dual_view_action_alignment_artifacts_present(session_dir: str | Path) -> bool:
    metadata_dir = Path(session_dir) / "metadata"
    return any(
        path.exists()
        for path in (
            metadata_dir / "dual_view_action_events.jsonl",
            metadata_dir / "dual_view_action_alignment_summary.json",
        )
    )


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _confirmed_dual_view_action_event_count(session_dir: str | Path) -> int:
    metadata_dir = Path(session_dir) / "metadata"
    summary = _load_json(metadata_dir / "dual_view_action_alignment_summary.json")
    if summary:
        diagnostics = summary.get("view_alignment_diagnostics")
        nested = diagnostics if isinstance(diagnostics, Mapping) else {}
        for value in (
            summary.get("formal_event_count"),
            nested.get("formal_event_count"),
            summary.get("dual_view_action_event_count"),
        ):
            parsed = _optional_int(value)
            if parsed is not None:
                return parsed
    return len(_read_jsonl_if_exists(metadata_dir / "dual_view_action_events.jsonl"))


def formal_dual_view_action_gate_status(session_dir: str | Path) -> dict[str, Any]:
    """Return the formal dual-view action gate status for material publication."""

    metadata_dir = Path(session_dir) / "metadata"
    summary_path = metadata_dir / "dual_view_action_alignment_summary.json"
    events_path = metadata_dir / "dual_view_action_events.jsonl"
    summary = _load_json(summary_path)
    event_index = _dual_view_action_event_index(session_dir)
    diagnostics = summary.get("view_alignment_diagnostics") if isinstance(summary.get("view_alignment_diagnostics"), Mapping) else {}
    formal_event_count = None
    formal_event_count_source = ""
    for source, value in (
        ("summary.formal_event_count", summary.get("formal_event_count")),
        ("summary.view_alignment_diagnostics.formal_event_count", diagnostics.get("formal_event_count")),
        ("summary.dual_view_action_event_count", summary.get("dual_view_action_event_count")),
    ):
        parsed = _optional_int(value)
        if parsed is not None:
            formal_event_count = parsed
            formal_event_count_source = source
            break
    if formal_event_count is None:
        formal_event_count = int(event_index.get("event_count") or 0)
        formal_event_count_source = "dual_view_action_events.jsonl"

    explicit_allowed = summary.get("formal_results_allowed") if "formal_results_allowed" in summary else None
    artifacts_present = bool(summary) or events_path.exists()
    formal_results_allowed = bool(explicit_allowed) if explicit_allowed is not None else (
        True if not artifacts_present else formal_event_count > 0
    )
    blocked_reason = ""
    if explicit_allowed is False:
        blocked_reason = "formal_results_not_allowed"
    elif artifacts_present and formal_event_count <= 0:
        blocked_reason = "no_confirmed_dual_view_action_events"
    elif artifacts_present and not formal_results_allowed:
        blocked_reason = "formal_results_not_allowed"

    return {
        "schema_version": FORMAL_MATERIAL_PUBLISH_GATE_VERSION,
        "status": "blocked" if blocked_reason else "passed",
        "allowed": not bool(blocked_reason),
        "blocked_reason": blocked_reason or None,
        "artifacts_present": artifacts_present,
        "formal_results_allowed": bool(formal_results_allowed),
        "formal_results_allowed_explicit": explicit_allowed if explicit_allowed is not None else None,
        "formal_event_count": int(formal_event_count),
        "formal_event_count_source": formal_event_count_source,
        "dual_view_action_event_count": _optional_int(summary.get("dual_view_action_event_count")) or int(event_index.get("event_count") or 0),
        "indexed_dual_view_action_event_count": int(event_index.get("event_count") or 0),
        "decision": summary.get("decision"),
        "summary_path": str(summary_path) if summary_path.exists() else None,
        "events_path": str(events_path),
    }


def _session_experiment_root(session_root: Path) -> Path:
    return session_root.parent if session_root.name == "key_action_index" else session_root


def _timeline_alignment_payload(session_root: Path) -> dict[str, Any]:
    experiment_root = _session_experiment_root(session_root)
    for path in (
        session_root / "metadata" / "dual_view_alignment_pipeline_summary.json",
        session_root / "metadata" / "dual_view_alignment" / "dual_view_alignment_pipeline_summary.json",
        session_root / "metadata" / "time_axis_health.json",
        experiment_root / "timeline_alignment.json",
        session_root / "metadata" / "pre_coarse_timeline_alignment.json",
        session_root / "metadata" / "view_alignment_from_yolo.json",
    ):
        payload = _load_json(path)
        if payload:
            return payload
    return {}


def _alignment_status_is_reliable(status: Any) -> bool:
    return str(status or "").strip().lower() in {
        "aligned",
        "explicit",
        "shared_recording",
        "calibrated",
        "calibrated_zero_offset",
        "manual_offset",
        "capture_start_common_timeline",
        "frame_time_map",
        "frame_time_map_aligned",
    }


def session_has_reliable_dual_view_alignment(session_dir: str | Path) -> bool:
    if not _require_reliable_dual_view_alignment():
        return True
    payload = _timeline_alignment_payload(Path(session_dir))
    if not payload:
        return False
    payload_status = str(payload.get("status") or "").strip().lower()
    if bool(payload.get("time_axis_unreliable")) or payload_status == "time_axis_unreliable":
        return False
    if payload_status in {"healthy", "warning"}:
        return bool(payload.get("can_publish_formal_materials", payload_status == "healthy"))
    if not _alignment_status_is_reliable(payload.get("alignment_status") or payload.get("status")):
        return False
    streams = payload.get("streams") if isinstance(payload.get("streams"), list) else []
    if not streams:
        return True
    relevant = [
        stream
        for stream in streams
        if isinstance(stream, dict)
        and str(stream.get("role") or stream.get("view_type") or "").strip() in {"first_person", "third_person"}
    ]
    if len(relevant) < 2:
        return False
    return all(_alignment_status_is_reliable(stream.get("alignment_status") or stream.get("status")) for stream in relevant)


def _session_formal_material_publish_allowed(session_dir: str | Path) -> tuple[bool, str]:
    formal_output_gate = _load_json(Path(session_dir) / "metadata" / "formal_output_gate.json")
    if str(formal_output_gate.get("status") or "").strip().lower() == "blocked":
        return False, str(formal_output_gate.get("blocked_reason") or "formal_output_gate_blocked")
    payload = _timeline_alignment_payload(Path(session_dir))
    if not payload:
        pass
    else:
        status = str(payload.get("status") or "").strip().lower()
        if status in {"frame_alignment_unreliable", "cross_view_action_phase_unreliable"}:
            return False, status
        if (
            payload.get("formal_results_allowed") is False
            and str(payload.get("schema_version") or "").startswith("dual_view_alignment_pipeline")
            and status != "alignment_ready_pending_yolo_phase"
        ):
            return False, str(payload.get("blocked_reason") or status or "dual_view_alignment_pipeline_not_formal_publishable")
        if bool(payload.get("time_axis_unreliable")) or status == "time_axis_unreliable":
            return False, "time_axis_unreliable"
        if status in {"healthy", "warning"} and payload.get("can_publish_formal_materials") is False:
            return False, "time_axis_not_formal_publishable"
        if payload.get("can_publish_formal_materials") is False and (
            "capture_span_sec" in payload
            or "mp4_duration_sec" in payload
            or "duration_delta_sec" in payload
            or "largest_gap_sec" in payload
        ):
            return False, "time_axis_not_formal_publishable"
    if _require_formal_dual_view_action_event() and _dual_view_action_alignment_artifacts_present(session_dir):
        gate = formal_dual_view_action_gate_status(session_dir)
        if not gate.get("allowed"):
            return False, str(gate.get("blocked_reason") or "formal_results_not_allowed")
    return True, ""


def _row_nested_mapping(row: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = row.get(key)
    return value if isinstance(value, Mapping) else {}


def _material_row_nested_value(row: Mapping[str, Any], key: str) -> Any:
    value = row.get(key)
    if value is not None and value != "":
        return value
    for nested_key in ("payload", "evidence_chain"):
        nested = _row_nested_mapping(row, nested_key)
        value = nested.get(key)
        if value is not None and value != "":
            return value
    return None


def _material_row_source_video_path(row: Mapping[str, Any], *, root: Path | None = None) -> Path | None:
    for key in (
        "source_video_path",
        "source_video",
        "source_clip_path",
        "source_clip",
        "source_reference_clip",
        "source_reference_file",
        "source_file",
    ):
        value = _material_row_nested_value(row, key)
        if not value:
            continue
        path = Path(str(value))
        if root is not None and not path.is_absolute() and not path.exists():
            path = root / path
        if path.suffix.lower() in MATERIAL_VIDEO_EXTENSIONS:
            return path
    return None


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _material_row_source_pts_span(row: Mapping[str, Any]) -> tuple[float, float] | None:
    start = None
    for key in (
        "source_offset_sec",
        "source_start_sec",
        "local_video_start_sec",
        "local_video_sec",
        "local_pts_sec",
    ):
        start = _optional_float(_material_row_nested_value(row, key))
        if start is not None:
            break
    if start is None:
        return None
    asset_kind = str(_material_row_nested_value(row, "asset_kind") or _material_row_nested_value(row, "material_type") or "")
    if asset_kind == KEYFRAME_DIR_NAME:
        return start, start
    end = None
    for key in ("source_end_sec", "local_video_end_sec", "local_pts_end_sec"):
        end = _optional_float(_material_row_nested_value(row, key))
        if end is not None:
            break
    if end is None:
        duration = None
        for key in ("source_duration_sec", "clip_duration_sec", "duration_sec"):
            duration = _optional_float(_material_row_nested_value(row, key))
            if duration is not None:
                break
        if duration is None:
            row_start = _optional_float(_material_row_nested_value(row, "start_sec"))
            row_end = _optional_float(_material_row_nested_value(row, "end_sec"))
            if row_start is not None and row_end is not None and row_end >= row_start:
                duration = row_end - row_start
        end = start + max(0.0, duration) if duration is not None else start
    return start, end


def _material_row_declared_source_duration(row: Mapping[str, Any]) -> float | None:
    for key in (
        "source_video_duration_sec",
        "source_clip_duration_sec",
        "source_media_duration_sec",
        "source_file_duration_sec",
        "mp4_duration_sec",
    ):
        duration = _optional_float(_material_row_nested_value(row, key))
        if duration is not None and duration > 0:
            return duration
    return None


def _source_video_duration_sec(path: Path) -> float | None:
    if path.suffix.lower() not in MATERIAL_VIDEO_EXTENSIONS or not path.is_file():
        return None
    key = _material_file_stat_key(path, dry_run=False)
    cache_key = (key[0], key[1], key[2]) if key is not None else None
    if cache_key is not None:
        with _SOURCE_VIDEO_DURATION_CACHE_LOCK:
            if cache_key in _SOURCE_VIDEO_DURATION_CACHE:
                return _SOURCE_VIDEO_DURATION_CACHE[cache_key]
    duration: float | None = None
    ffprobe = shutil.which("ffprobe") or "ffprobe"
    try:
        result = subprocess.run(
            [
                str(ffprobe),
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True,
        )
        duration = _optional_float(result.stdout.strip())
    except Exception:
        duration = None
    if duration is None or duration <= 0:
        try:
            import cv2

            cap = cv2.VideoCapture(str(path))
            if cap.isOpened():
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
                if fps > 0 and frame_count > 0:
                    duration = frame_count / fps
            cap.release()
        except Exception:
            duration = None
    if duration is not None and duration <= 0:
        duration = None
    if cache_key is not None:
        with _SOURCE_VIDEO_DURATION_CACHE_LOCK:
            _SOURCE_VIDEO_DURATION_CACHE[cache_key] = duration
    return duration


def _material_row_source_pts_gate(
    row: Mapping[str, Any],
    *,
    root: Path | None = None,
) -> tuple[bool, str, dict[str, Any]]:
    asset_kind = str(_material_row_nested_value(row, "asset_kind") or _material_row_nested_value(row, "material_type") or "")
    if asset_kind not in {KEYFRAME_DIR_NAME, KEY_CLIP_DIR_NAME}:
        return True, "", {}
    span = _material_row_source_pts_span(row)
    if span is None:
        return True, "", {}
    source_video = _material_row_source_video_path(row, root=root)
    if source_video is None:
        return True, "", {}
    source_duration = _material_row_declared_source_duration(row)
    if source_duration is None:
        source_duration = _source_video_duration_sec(source_video)
    if source_duration is None:
        return True, "", {"source_video": str(source_video), "status": "source_video_duration_unavailable"}
    start_sec, end_sec = span
    details = {
        "source_video": str(source_video),
        "source_start_sec": round(start_sec, 6),
        "source_end_sec": round(end_sec, 6),
        "source_duration_sec": round(source_duration, 6),
        "tolerance_sec": MATERIAL_SOURCE_PTS_TOLERANCE_SEC,
    }
    if start_sec < -MATERIAL_SOURCE_PTS_TOLERANCE_SEC:
        details["status"] = "failed"
        details["reason"] = "source_pts_negative"
        return False, "source_pts_out_of_range", details
    if end_sec > source_duration + MATERIAL_SOURCE_PTS_TOLERANCE_SEC:
        details["status"] = "failed"
        details["reason"] = "source_pts_exceeds_source_duration"
        return False, "source_pts_out_of_range", details
    details["status"] = "passed"
    return True, "", details


def _material_row_value(row: Mapping[str, Any], key: str) -> Any:
    value = row.get(key)
    if value not in {None, ""}:
        return value
    for nested_key in ("payload", "evidence_chain", "evidence"):
        payload = row.get(nested_key)
        if isinstance(payload, Mapping):
            value = payload.get(key)
            if value not in {None, ""}:
                return value
    return None


def _material_row_dual_event_id(row: Mapping[str, Any]) -> str:
    for key in DUAL_VIEW_ACTION_EVENT_KEYS:
        value = _material_row_value(row, key)
        if value not in {None, ""}:
            return str(value).strip()
    return ""


def _material_row_group_id(row: Mapping[str, Any]) -> str:
    """Return the physical evidence window id used for dual-view publication.

    `candidate_group_id` can be view-local in older runs, so prefer durable
    segment/window identifiers before falling back to the candidate group.
    """

    return str(
        _material_row_value(row, "evidence_group_id")
        or _material_row_value(row, "material_group_id")
        or _material_row_value(row, "physical_action_material_id")
        or _material_row_value(row, "evidence_window_id")
        or _material_row_value(row, "time_window_id")
        or _material_row_value(row, "window_id")
        or _material_row_value(row, "micro_segment_id")
        or _material_row_value(row, "candidate_group_id")
        or ""
    ).strip()


def _material_evidence_group_id(
    row: Mapping[str, Any],
    *,
    primary_object: Any = "",
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> str:
    for key in ("evidence_group_id", "material_group_id", "physical_action_material_id", "evidence_window_id"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    raw = "|".join(
        str(value or "")
        for value in (
            row.get("micro_segment_id"),
            row.get("parent_segment_id") or row.get("segment_id"),
            canonical_yolo_label(primary_object or row.get("primary_object") or row.get("canonical_object"))
            or primary_object
            or row.get("primary_object")
            or row.get("canonical_object"),
            f"{start_sec if start_sec is not None else _safe_float(row.get('start_sec', row.get('time_start')), 0.0):.3f}",
            f"{end_sec if end_sec is not None else _safe_float(row.get('end_sec', row.get('time_end')), 0.0):.3f}",
        )
    )
    digest = hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"evidence_group_{digest}"


def _material_row_action(row: Mapping[str, Any]) -> str:
    return str(
        _material_row_value(row, "canonical_action_type")
        or _material_row_value(row, "physical_action_type")
        or _material_row_value(row, "action_name")
        or _material_row_value(row, "display_title")
        or ""
    ).strip()


def _material_row_object_family_key(row: Mapping[str, Any]) -> str:
    for key in ("primary_object_family", "object_family"):
        value = str(_material_row_value(row, key) or "").strip()
        if value:
            return value
    for key in ("primary_object", "raw_primary_object", "canonical_object", "manipulated_object"):
        value = _material_row_value(row, key)
        family = _material_object_family_for_label(value)
        if family:
            return family
    return ""


def _material_object_family_for_label(label: Any) -> str:
    value = canonical_yolo_label(label)
    if value in {"reagent_bottle", "reagent_bottle_open", "bottle_cap", "cap"}:
        return "reagent_bottle_family"
    if value in {"sample_bottle", "sample_bottle_blue", "bottle", "vial"}:
        return "sample_bottle_family"
    if value in {"pipette", "pipette_tip", "spearhead"}:
        return "pipette_family"
    if value in {"balance", "scale", "panel"}:
        return "balance_family"
    if value in {"paper", "weighing_paper"}:
        return "paper_family"
    if value == "spatula":
        return "spatula_family"
    if value in {"beaker", "container", "tube", "tube_cap", "tube_rack", "flask"}:
        return "container_family"
    if value in {"magnetic_stirrer", "magnetic_stir_bar"}:
        return "equipment_family"
    return ""


def _material_row_dual_view_group_key(row: Mapping[str, Any]) -> str:
    group_id = _material_row_dual_event_id(row) or _material_row_group_id(row)
    action = _material_row_action(row)
    object_family = _material_row_object_family_key(row)
    parts = [part for part in (group_id, action, object_family) if part]
    return "::".join(parts) if parts else group_id


def _material_row_view(row: Mapping[str, Any]) -> str:
    return str(_material_row_value(row, "view") or _material_row_value(row, "camera_view") or "").strip()


def _material_row_kind(row: Mapping[str, Any]) -> str:
    raw = str(_material_row_value(row, "asset_kind") or _material_row_value(row, "material_type") or "").strip().lower()
    if raw in {KEYFRAME_DIR_NAME.lower(), "keyframe", "key_frame", "frame", "关键帧"}:
        return "keyframe"
    if raw in {KEY_CLIP_DIR_NAME.lower(), "keyclip", "key_clip", "clip", "video", "关键片段"}:
        return "keyclip"
    return raw


def _material_row_primary_object(row: Mapping[str, Any]) -> str:
    for key in ("primary_object", "raw_primary_object", "canonical_object", "manipulated_object"):
        value = _material_row_value(row, key)
        label = canonical_yolo_label(value)
        if label:
            return label
        if str(value or "").strip():
            return str(value).strip()
    return ""


def _material_row_evidence_rows(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    containers: list[Mapping[str, Any]] = [row]
    evidence_chain = row.get("evidence_chain")
    if isinstance(evidence_chain, Mapping):
        containers.append(evidence_chain)
    for container in containers:
        for key in ("source_yolo_evidence", "yolo_evidence", "physical_evidence_rows"):
            value = container.get(key)
            if isinstance(value, list):
                rows.extend(dict(item) for item in value if isinstance(item, dict))
            elif isinstance(value, Mapping):
                nested = value.get("rows") or value.get("evidence_rows")
                if isinstance(nested, list):
                    rows.extend(dict(item) for item in nested if isinstance(item, dict))
    return rows


def _material_view_count(mapping: Mapping[str, Any], view: str) -> int:
    try:
        return int(float(mapping.get(view) or 0))
    except (TypeError, ValueError):
        return 0


def _material_row_same_view_physical_evidence_gate(row: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    view = _material_row_view(row)
    details: dict[str, Any] = {
        "status": "failed",
        "required_view": view or None,
    }
    if view not in {"first_person", "third_person"}:
        details["reason"] = "missing_material_view"
        return False, details

    evidence_chain = row.get("evidence_chain")
    chain = evidence_chain if isinstance(evidence_chain, Mapping) else {}
    chain_view = str(chain.get("camera_view") or chain.get("view") or chain.get("source_view") or "").strip()
    if chain_view and chain_view != view:
        details["reason"] = "evidence_chain_view_mismatch"
        details["evidence_chain_view"] = chain_view
        return False, details

    physical_mode = str(row.get("physical_evidence_mode") or chain.get("physical_evidence_mode") or "").strip()
    candidate_source = str(row.get("candidate_source") or chain.get("candidate_source") or "").strip()
    if physical_mode == PAIRED_VIEW_CONTEXT_MODE or candidate_source in {
        "paired_view_time_alignment",
        "paired_view_micro_segment_key_asset_reference",
    }:
        details["reason"] = "paired_view_context_is_not_same_view_physical_evidence"
        details["physical_evidence_mode"] = physical_mode or None
        details["candidate_source"] = candidate_source or None
        return False, details

    source_rows = _material_row_evidence_rows(row)
    same_view_rows = [item for item in source_rows if evidence_view(item) == view]
    primary_object = _material_row_primary_object(row)
    same_view_valid_count = (
        len(valid_yolo_physical_evidence(same_view_rows, primary_object))
        if primary_object
        else len(same_view_rows)
    )

    diagnostics = row.get("physical_evidence_diagnostics")
    diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
    evidence_by_view = diagnostics.get("evidence_by_view") if isinstance(diagnostics.get("evidence_by_view"), Mapping) else {}
    valid_by_view = (
        diagnostics.get("valid_evidence_by_view")
        if isinstance(diagnostics.get("valid_evidence_by_view"), Mapping)
        else {}
    )
    diagnostic_evidence_count = _material_view_count(evidence_by_view, view)
    diagnostic_valid_count = _material_view_count(valid_by_view, view)

    details.update(
        {
            "primary_object": primary_object or None,
            "source_yolo_evidence_count": len(source_rows),
            "same_view_source_yolo_evidence_count": len(same_view_rows),
            "same_view_valid_yolo_evidence_count": same_view_valid_count,
            "diagnostic_same_view_evidence_count": diagnostic_evidence_count,
            "diagnostic_same_view_valid_evidence_count": diagnostic_valid_count,
        }
    )

    if source_rows and same_view_valid_count <= 0:
        target_labels = _interaction_target_labels(primary_object)
        if _material_row_dual_event_id(row) and target_labels:
            sparse_same_view_rows = [
                item
                for item in same_view_rows
                if _has_plausible_sparse_target_interaction(item, target_labels)
            ]
            if sparse_same_view_rows:
                details["status"] = "passed"
                details["sparse_dual_view_event_evidence_count"] = len(sparse_same_view_rows)
                details["mode"] = "confirmed_dual_view_sparse_same_view_action_evidence"
                details.pop("reason", None)
                return True, details
        details["reason"] = "source_yolo_evidence_missing_same_view_valid_physical_evidence"
        return False, details
    if diagnostics and (diagnostic_evidence_count <= 0 or (valid_by_view and diagnostic_valid_count <= 0)):
        details["reason"] = "physical_evidence_diagnostics_missing_same_view_valid_evidence"
        return False, details
    if not source_rows and not diagnostics:
        details["reason"] = "missing_source_yolo_evidence_and_physical_diagnostics"
        return False, details
    if same_view_valid_count > 0 or diagnostic_valid_count > 0 or (diagnostic_evidence_count > 0 and not valid_by_view):
        details["status"] = "passed"
        details.pop("reason", None)
        return True, details
    details["reason"] = "missing_same_view_valid_physical_evidence"
    return False, details


def _require_dual_view_key_clips() -> bool:
    return _env_truthy("KEY_ACTION_REQUIRE_DUAL_VIEW_KEY_CLIPS", True)


def _required_dual_view_material_assets() -> set[tuple[str, str]]:
    required = {
        ("third_person", "keyframe"),
        ("first_person", "keyframe"),
    }
    if _require_dual_view_key_clips():
        required.update(
            {
                ("third_person", "keyclip"),
                ("first_person", "keyclip"),
            }
        )
    return required


def _material_row_counts_for_dual_view_gate(row: Mapping[str, Any]) -> bool:
    return row.get("exists") is not False or bool(row.get("dry_run"))


def _dual_view_material_group_assets(rows: list[Mapping[str, Any]]) -> dict[str, set[tuple[str, str]]]:
    by_group: dict[str, set[tuple[str, str]]] = {}
    for row in rows:
        if not _material_row_counts_for_dual_view_gate(row):
            continue
        group_id = _material_row_dual_view_group_key(row)
        if not group_id:
            continue
        view = _material_row_view(row)
        kind = _material_row_kind(row)
        if view in {"first_person", "third_person"} and kind in {"keyframe", "keyclip"}:
            by_group.setdefault(group_id, set()).add((view, kind))
    return by_group


def complete_dual_view_material_group_ids(rows: list[Mapping[str, Any]]) -> set[str]:
    """Return material groups that contain enough same-action dual-view assets."""

    required = _required_dual_view_material_assets()
    by_group = _dual_view_material_group_assets(rows)
    return {group_id for group_id, assets in by_group.items() if required.issubset(assets)}


def _event_view_payload_has_evidence(payload: Any, view: str) -> bool:
    if not isinstance(payload, Mapping):
        return False
    payload_view = str(payload.get("view") or payload.get("camera_view") or view).strip()
    if payload_view and payload_view != view:
        return False
    return any(
        str(payload.get(key) or "").strip()
        for key in ("evidence_id", "view_action_evidence_id", "source_evidence_id")
    ) or _safe_float(payload.get("frame_count"), 0.0) > 0


def _formal_dual_view_event_has_required_evidence(row: Mapping[str, Any]) -> bool:
    first_id = str(row.get("first_evidence_id") or row.get("first_person_evidence_id") or "").strip()
    third_id = str(row.get("third_evidence_id") or row.get("third_person_evidence_id") or "").strip()
    views = row.get("views") if isinstance(row.get("views"), Mapping) else {}
    first_payload = views.get("first_person") if isinstance(views, Mapping) else None
    third_payload = views.get("third_person") if isinstance(views, Mapping) else None
    first_ok = bool(first_id) or _event_view_payload_has_evidence(first_payload, "first_person")
    third_ok = bool(third_id) or _event_view_payload_has_evidence(third_payload, "third_person")
    if first_payload is not None:
        first_ok = first_ok and _event_view_payload_has_evidence(first_payload, "first_person")
    if third_payload is not None:
        third_ok = third_ok and _event_view_payload_has_evidence(third_payload, "third_person")
    return first_ok and third_ok


def _dual_view_action_event_index(session_dir: str | Path) -> dict[str, Any]:
    metadata_dir = Path(session_dir) / "metadata"
    rows = [
        row
        for row in _read_jsonl_if_exists(metadata_dir / "dual_view_action_events.jsonl")
        if isinstance(row, dict)
    ]
    event_ids: set[str] = set()
    event_ids_by_micro: dict[str, str] = {}
    event_micro_ids: dict[str, set[str]] = {}
    events_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("formal_event_promoted") is False:
            continue
        event_id = str(
            row.get("dual_event_id")
            or row.get("dual_view_action_event_id")
            or row.get("event_id")
            or ""
        ).strip()
        status = str(row.get("status") or "").strip().lower()
        if not event_id:
            continue
        if status and status not in {"confirmed", "matched_dual_view", "formal"}:
            continue
        if not _formal_dual_view_event_has_required_evidence(row):
            continue
        event_ids.add(event_id)
        events_by_id[event_id] = dict(row)
        micros = {str(micro_id or "").strip() for micro_id in row.get("micro_segment_ids") or [] if str(micro_id or "").strip()}
        event_micro_ids[event_id] = micros
        for micro_id in row.get("micro_segment_ids") or []:
            micro = str(micro_id or "").strip()
            if micro:
                event_ids_by_micro.setdefault(micro, event_id)
    return {
        "event_ids": event_ids,
        "event_ids_by_micro": event_ids_by_micro,
        "event_micro_ids": event_micro_ids,
        "events_by_id": events_by_id,
        "event_count": len(event_ids),
        "path": str(metadata_dir / "dual_view_action_events.jsonl"),
    }


def _material_row_micro_segment_ids(row: Mapping[str, Any]) -> set[str]:
    ids: set[str] = set()
    for key in ("micro_segment_id", "source_micro_segment_id"):
        value = _material_row_value(row, key)
        if value not in {None, ""}:
            ids.add(str(value).strip())
    for key in ("micro_segment_ids", "source_micro_segment_ids"):
        value = _material_row_value(row, key)
        if isinstance(value, (list, tuple, set)):
            ids.update(str(item or "").strip() for item in value if str(item or "").strip())
        elif value not in {None, ""}:
            ids.add(str(value).strip())
    ids.discard("")
    return ids


def _material_action_values_match(left: Any, right: Any) -> bool:
    left_text = str(left or "").strip().lower().replace("_", "-")
    right_text = str(right or "").strip().lower().replace("_", "-")
    return not left_text or not right_text or left_text == right_text


def _material_row_matches_indexed_event(
    row: Mapping[str, Any],
    event_id: str,
    event_index: Mapping[str, Any],
) -> bool:
    event_micro_ids = (
        event_index.get("event_micro_ids")
        if isinstance(event_index.get("event_micro_ids"), Mapping)
        else {}
    )
    row_micro_ids = _material_row_micro_segment_ids(row)
    indexed_micro_ids = event_micro_ids.get(event_id)
    if isinstance(indexed_micro_ids, set) and indexed_micro_ids and row_micro_ids and not row_micro_ids.intersection(indexed_micro_ids):
        binding_source = str(_material_row_value(row, "dual_event_binding_source") or "").strip()
        if binding_source != "explicit_confirmed_dual_view_action_event":
            return False

    events_by_id = event_index.get("events_by_id") if isinstance(event_index.get("events_by_id"), Mapping) else {}
    event = events_by_id.get(event_id) if isinstance(events_by_id.get(event_id), Mapping) else {}
    event_action = event.get("canonical_action_type")
    if not _material_action_values_match(_material_row_value(row, "canonical_action_type"), event_action):
        return False
    event_object_family = str(event.get("primary_object_family") or event.get("object_family") or "").strip()
    row_object_family = _material_row_object_family_key(row)
    if event_object_family and row_object_family and event_object_family != row_object_family:
        return False
    return True


def _material_row_formal_dual_event_id(row: Mapping[str, Any], event_index: Mapping[str, Any]) -> str:
    event_ids = event_index.get("event_ids") if isinstance(event_index.get("event_ids"), set) else set()
    event_ids_by_micro = (
        event_index.get("event_ids_by_micro")
        if isinstance(event_index.get("event_ids_by_micro"), Mapping)
        else {}
    )
    explicit = _material_row_dual_event_id(row)
    if explicit:
        return explicit if explicit in event_ids and _material_row_matches_indexed_event(row, explicit, event_index) else ""
    for key in ("micro_segment_id", "source_micro_segment_id"):
        micro_id = str(_material_row_value(row, key) or "").strip()
        if micro_id and micro_id in event_ids_by_micro:
            event_id = str(event_ids_by_micro[micro_id])
            return event_id if _material_row_matches_indexed_event(row, event_id, event_index) else ""
    group_id = _material_row_group_id(row)
    if group_id and group_id in event_ids_by_micro:
        event_id = str(event_ids_by_micro[group_id])
        return event_id if _material_row_matches_indexed_event(row, event_id, event_index) else ""
    return ""


def _material_row_allows_explicit_formal_binding(row: Mapping[str, Any]) -> bool:
    binding_source = str(_material_row_value(row, "dual_event_binding_source") or "").strip()
    return binding_source != "derived_from_complete_dual_view_material_group"


def _dual_view_material_group_same_view_assets(
    rows: list[Mapping[str, Any]],
    same_view_status: Mapping[int, tuple[bool, dict[str, Any]]],
) -> dict[str, set[tuple[str, str]]]:
    by_group: dict[str, set[tuple[str, str]]] = {}
    for row in rows:
        if not _material_row_counts_for_dual_view_gate(row):
            continue
        ok, _details = same_view_status.get(id(row), (False, {}))
        if not ok:
            continue
        group_id = _material_row_dual_view_group_key(row)
        if not group_id:
            continue
        view = _material_row_view(row)
        kind = _material_row_kind(row)
        if view in {"first_person", "third_person"} and kind in {"keyframe", "keyclip"}:
            by_group.setdefault(group_id, set()).add((view, kind))
    return by_group


def filter_complete_dual_view_material_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not _require_dual_view_complete_material_groups():
        return rows
    complete_group_ids = complete_dual_view_material_group_ids(rows)
    filtered: list[dict[str, Any]] = []
    for row in rows:
        group_key = _material_row_dual_view_group_key(row)
        if group_key not in complete_group_ids:
            continue
        item = dict(row)
        _attach_formal_dual_event_binding(item, group_key)
        filtered.append(item)
    return filtered


def apply_formal_dual_view_material_publish_gate(
    session_dir: str | Path,
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep only rows backed by a complete dual-view action event."""

    if not _require_dual_view_complete_material_groups():
        return rows, []
    if not session_has_reliable_dual_view_alignment(session_dir):
        return [], [
            _formal_material_gate_rejection(row, "missing_reliable_dual_view_alignment", assets=set())
            for row in rows
        ]
    action_gate = formal_dual_view_action_gate_status(session_dir)
    if action_gate.get("summary_path") and not action_gate.get("allowed"):
        blocked_reason = str(action_gate.get("blocked_reason") or "formal_results_not_allowed")
        return [], [
            _formal_material_gate_rejection(
                row,
                blocked_reason,
                assets=set(),
                action_gate=action_gate,
            )
            for row in rows
        ]
    event_index = _dual_view_action_event_index(session_dir)
    formal_event_required = True
    if formal_event_required and not event_index.get("event_count"):
        return [], [
            _formal_material_gate_rejection(row, "missing_explicit_dual_view_action_event", assets=set())
            for row in rows
        ]
    complete_group_ids = complete_dual_view_material_group_ids(rows)
    group_assets = _dual_view_material_group_assets(rows)
    same_view_status = {
        id(row): _material_row_same_view_physical_evidence_gate(row)
        for row in rows
    }
    group_same_view_assets = _dual_view_material_group_same_view_assets(rows, same_view_status)
    same_view_complete_group_ids = {
        group_id
        for group_id, assets in group_same_view_assets.items()
        if _required_dual_view_material_assets().issubset(assets)
    }
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in rows:
        group_key = _material_row_dual_view_group_key(row)
        formal_dual_event_id = (
            _material_row_formal_dual_event_id(row, event_index)
            if formal_event_required
            else _material_row_dual_event_id(row)
        )
        if group_key in complete_group_ids:
            if (
                formal_event_required
                and (not formal_dual_event_id or not _material_row_allows_explicit_formal_binding(row))
            ):
                rejected.append(
                    _formal_material_gate_rejection(
                        row,
                        "missing_explicit_dual_view_action_event",
                        assets=group_assets.get(group_key, set()),
                    )
                )
                continue
            same_view_ok, same_view_details = same_view_status.get(id(row), (False, {}))
            if not same_view_ok or group_key not in same_view_complete_group_ids:
                rejected.append(
                    _formal_material_gate_rejection(
                        row,
                        "missing_same_view_physical_evidence",
                        assets=group_same_view_assets.get(group_key, set()),
                        same_view_details=same_view_details,
                    )
                )
                continue
            item = dict(row)
            _attach_formal_dual_event_binding(item, group_key, dual_event_id=formal_dual_event_id or None)
            kept.append(item)
            continue
        assets = group_assets.get(group_key, set())
        if formal_event_required and (not formal_dual_event_id or not _material_row_allows_explicit_formal_binding(row)):
            reason = "missing_explicit_dual_view_action_event"
        else:
            reason = "incomplete_dual_view_material_group"
        _same_view_ok, same_view_details = same_view_status.get(id(row), (False, {}))
        rejected.append(
            _formal_material_gate_rejection(
                row,
                reason,
                assets=group_same_view_assets.get(group_key, assets),
                same_view_details=same_view_details,
            )
        )
    return kept, rejected


def filter_aligned_dual_view_material_rows(session_dir: str | Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not _require_dual_view_complete_material_groups():
        return rows
    if session_has_reliable_dual_view_alignment(session_dir):
        return rows
    return []


def _dual_view_action_event_id_from_group(group_key: str) -> str:
    digest = hashlib.sha1(str(group_key).encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"dual_event_{digest}"


def _attach_formal_dual_event_binding(row: dict[str, Any], group_key: str, *, dual_event_id: str | None = None) -> None:
    dual_event_id = str(dual_event_id or _material_row_dual_event_id(row) or "").strip()
    binding_source = "explicit_dual_event_id"
    if not dual_event_id:
        row["formal_publish_gate"] = {
            "schema_version": FORMAL_MATERIAL_PUBLISH_GATE_VERSION,
            "status": "rejected",
            "reason": "missing_explicit_dual_view_action_event",
            "requires": "explicit confirmed dual_view_action_event_id",
        }
        return
    row["dual_event_id"] = dual_event_id
    row["dual_view_action_event_id"] = dual_event_id
    row["dual_event_binding_source"] = binding_source
    row["material_group_id"] = group_key
    row["physical_action_material_id"] = group_key
    gate = {
        "schema_version": FORMAL_MATERIAL_PUBLISH_GATE_VERSION,
        "status": "passed",
        "dual_event_id": dual_event_id,
        "binding_source": binding_source,
        "requires": "first_person+third_person keyframe and keyclip",
    }
    row["formal_publish_gate"] = gate
    evidence_chain = dict(row.get("evidence_chain") if isinstance(row.get("evidence_chain"), Mapping) else {})
    evidence_chain["dual_event_id"] = dual_event_id
    evidence_chain["dual_view_action_event_id"] = dual_event_id
    evidence_chain["dual_event_binding_source"] = binding_source
    evidence_chain["formal_publish_gate"] = gate
    row["evidence_chain"] = evidence_chain


def _material_row_has_same_view_physical_evidence(row: Mapping[str, Any]) -> bool:
    if not _require_same_view_physical_evidence_for_formal_material():
        return True
    view = _material_row_view(row)
    if view not in {"first_person", "third_person"}:
        return False
    diagnostics = row.get("physical_evidence_diagnostics")
    if isinstance(diagnostics, Mapping):
        for key in ("valid_evidence_by_view", "evidence_by_view"):
            by_view = diagnostics.get(key)
            if isinstance(by_view, Mapping):
                try:
                    if int(float(by_view.get(view) or 0)) > 0:
                        return True
                except Exception:
                    pass
    for key in ("source_yolo_evidence", "yolo_evidence"):
        evidence_rows = row.get(key)
        if not isinstance(evidence_rows, list):
            continue
        for evidence in evidence_rows:
            if not isinstance(evidence, Mapping):
                continue
            evidence_view = str(
                evidence.get("source_view")
                or evidence.get("view")
                or evidence.get("camera_view")
                or ""
            ).strip()
            if evidence_view == view:
                return True
    evidence_chain = row.get("evidence_chain")
    if isinstance(evidence_chain, Mapping):
        evidence_rows = evidence_chain.get("source_yolo_evidence") or evidence_chain.get("yolo_evidence")
        if isinstance(evidence_rows, list):
            for evidence in evidence_rows:
                if not isinstance(evidence, Mapping):
                    continue
                evidence_view = str(
                    evidence.get("source_view")
                    or evidence.get("view")
                    or evidence.get("camera_view")
                    or ""
                ).strip()
                if evidence_view == view:
                    return True
    return False


def _formal_material_gate_rejection(
    row: Mapping[str, Any],
    reason: str,
    *,
    assets: set[tuple[str, str]],
    same_view_details: Mapping[str, Any] | None = None,
    action_gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    required = _required_dual_view_material_assets()
    missing_assets = sorted(f"{view}:{kind}" for view, kind in required.difference(assets))
    present_assets = sorted(f"{view}:{kind}" for view, kind in assets)
    rejection = {
        "reason": "formal_material_publish_gate",
        "suppression_reason": reason,
        "blocked_reason": reason,
        "gate_schema_version": FORMAL_MATERIAL_PUBLISH_GATE_VERSION,
        "candidate_id": row.get("candidate_id"),
        "candidate_group_id": row.get("candidate_group_id"),
        "dual_event_id": _material_row_dual_event_id(row) or None,
        "dual_view_group_key": _material_row_dual_view_group_key(row) or None,
        "micro_segment_id": row.get("micro_segment_id"),
        "segment_id": row.get("segment_id") or row.get("parent_segment_id"),
        "asset_kind": row.get("asset_kind") or row.get("material_type"),
        "view": row.get("view") or row.get("camera_view"),
        "stored_file": row.get("stored_file"),
        "source_file": row.get("source_file"),
        "present_dual_view_assets": present_assets,
        "missing_dual_view_assets": missing_assets,
    }
    if same_view_details:
        rejection["same_view_physical_evidence_gate"] = dict(same_view_details)
    if action_gate:
        rejection["formal_dual_view_action_gate"] = dict(action_gate)
    return rejection

ACTION_NAME_BY_OBJECT = {
    "balance": "\u5929\u5e73\u8bbe\u5907\u9762\u677f\u64cd\u4f5c",
    "scale": "\u5929\u5e73\u8bbe\u5907\u9762\u677f\u64cd\u4f5c",
    "panel": "\u5929\u5e73\u8bbe\u5907\u9762\u677f\u64cd\u4f5c",
    "spatula": "\u624b\u4e0e\u836f\u5319\u64cd\u4f5c",
    "pipette": "\u624b\u4e0e\u79fb\u6db2\u67aa\u64cd\u4f5c",
    "pipette_tip": "\u624b\u4e0e\u79fb\u6db2\u67aa\u5934\u64cd\u4f5c",
    "paper": "\u624b\u90e8\u4e0e\u79f0\u91cf\u7eb8\u64cd\u4f5c",
    "weighing_paper": "\u624b\u90e8\u4e0e\u79f0\u91cf\u7eb8\u64cd\u4f5c",
    "reagent_bottle": "\u624b\u90e8\u4e0e\u8bd5\u5242\u74f6\u64cd\u4f5c",
    "reagent_bottle_open": "\u624b\u90e8\u4e0e\u8bd5\u5242\u74f6\u64cd\u4f5c",
    "bottle_cap": "\u624b\u90e8\u4e0e\u8bd5\u5242\u74f6\u64cd\u4f5c",
    "sample_bottle": "\u624b\u90e8\u4e0e\u8bd5\u5242\u74f6\u64cd\u4f5c",
    "sample_bottle_blue": "\u624b\u90e8\u4e0e\u8bd5\u5242\u74f6\u64cd\u4f5c",
    "bottle": "\u624b\u90e8\u4e0e\u8bd5\u5242\u74f6\u64cd\u4f5c",
    "vial": "\u624b\u90e8\u4e0e\u8bd5\u5242\u74f6\u64cd\u4f5c",
    "beaker": "\u624b\u4e0e\u70e7\u676f\u64cd\u4f5c",
    "container": "\u624b\u4e0e\u5bb9\u5668\u64cd\u4f5c",
    "magnetic_stir_bar": "\u624b\u4e0e\u78c1\u529b\u6405\u62cc\u5b50\u64cd\u4f5c",
    "magnetic_stirrer": "\u624b\u4e0e\u78c1\u529b\u6405\u62cc\u5668\u64cd\u4f5c",
}

CANONICAL_ACTION_BY_OBJECT = {
    "balance": ("equipment_panel_operation", "panel", "equipment-panel-operation"),
    "scale": ("equipment_panel_operation", "panel", "equipment-panel-operation"),
    "panel": ("equipment_panel_operation", "panel", "equipment-panel-operation"),
    "spatula": ("hand-spatula", "spatula", "solid-transfer"),
    "magnetic_stir_bar": ("hand-equipment", "magnetic_stir_bar", "stirring-prep"),
    "magnetic_stirrer": ("hand-equipment", "magnetic_stirrer", "stirring-prep"),
    "paper": ("hand-paper", "paper", "weighing-paper-prep"),
    "weighing_paper": ("hand-paper", "paper", "weighing-paper-prep"),
    "reagent_bottle": ("hand-bottle", "bottle", "reagent-bottle-handling"),
    "reagent_bottle_open": ("hand-bottle", "bottle", "reagent-bottle-handling"),
    "bottle_cap": ("hand-bottle", "bottle", "reagent-bottle-handling"),
    "sample_bottle": ("hand-bottle", "bottle", "reagent-bottle-handling"),
    "bottle": ("hand-bottle", "bottle", "reagent-bottle-handling"),
    "vial": ("hand-bottle", "bottle", "reagent-bottle-handling"),
    "beaker": ("hand-container", "container", "container-handling"),
    "container": ("hand-container", "container", "container-handling"),
    "tube": ("hand-container", "container", "container-handling"),
    "flask": ("hand-container", "container", "container-handling"),
}

EVENT_BACKED_CANDIDATE_TYPES = {
    "hand_object_contact",
    "object_movement_candidate",
    "object_movement_detected",
    "liquid_transfer_candidate",
    "liquid_flow_detected",
    "liquid_level_change_detected",
    "equipment_panel_operation_candidate",
    "equipment_panel_operation_detected",
    "container_state_change_candidate",
    "container_state_change_detected",
    "equipment_control_change",
    "container_open_close",
    "object_trajectory_movement",
}

CORE_V1_EVENT_BACKED_CANDIDATE_TYPES = {
    "hand_object_contact",
    "object_movement_candidate",
    "object_movement_detected",
    "object_trajectory_movement",
    "equipment_panel_operation_candidate",
    "equipment_panel_operation_detected",
    "equipment_control_change",
}

EVENT_CANONICAL_ACTIONS = {
    "hand_object_contact": ("hand_object_contact", "object", "hand-object-contact", "hand-object"),
    "object_movement_candidate": ("object_movement", "object", "object-movement", "movement"),
    "object_movement_detected": ("object_movement", "object", "object-movement", "movement"),
    "object_trajectory_movement": ("object_movement", "object", "object-movement", "movement"),
    "liquid_transfer_candidate": ("liquid_movement", "liquid", "liquid-movement", "liquid"),
    "liquid_flow_detected": ("liquid_movement", "liquid", "liquid-movement", "liquid"),
    "liquid_level_change_detected": ("liquid_movement", "liquid", "liquid-movement", "liquid"),
    "equipment_panel_operation_candidate": ("equipment_panel_operation", "panel", "equipment-panel-operation", "equipment"),
    "equipment_panel_operation_detected": ("equipment_panel_operation", "panel", "equipment-panel-operation", "equipment"),
    "equipment_control_change": ("equipment_panel_operation", "panel", "equipment-panel-operation", "equipment"),
    "container_state_change_candidate": ("container_state_change", "container", "container-state-change", "container-state"),
    "container_state_change_detected": ("container_state_change", "container", "container-state-change", "container-state"),
    "container_open_close": ("container_state_change", "container", "container-state-change", "container-state"),
}

PHYSICAL_ACTION_TYPES = {
    "hand_object_contact",
    "object_movement",
    "liquid_movement",
    "equipment_panel_operation",
    "container_state_change",
}

CORE_V1_PHYSICAL_ACTION_TYPES = {
    "hand_object_contact",
    "object_movement",
    "equipment_panel_operation",
}

MATERIAL_LIBRARY_ROOT_ENV_NAMES = (
    "LAB_MATERIAL_LIBRARY_ROOT",
    "KEY_ACTION_MATERIAL_LIBRARY_ROOT",
)
DEFAULT_D_DRIVE_MATERIAL_LIBRARY_ROOT = Path("D:/LabMaterialLibrary")

EVENT_BACKED_REVIEW_GROUP_LIMIT_TOTAL = 48
EVENT_BACKED_LOW_QUALITY_GROUP_LIMIT_TOTAL = 20
EVENT_BACKED_REVIEW_GROUP_LIMIT_PER_OBJECT = 5
EVENT_BACKED_LOW_QUALITY_GROUP_LIMIT_PER_OBJECT = 2
EVENT_BACKED_REVIEW_GROUP_LIMIT_BY_ACTION = {
    "hand_object_contact": 16,
    "object_movement": 12,
    "liquid_movement": 8,
    "equipment_panel_operation": 8,
    "container_state_change": 8,
}
EVENT_BACKED_LOW_QUALITY_GROUP_LIMIT_BY_ACTION = {
    "hand_object_contact": 6,
    "object_movement": 5,
    "liquid_movement": 4,
    "equipment_panel_operation": 3,
    "container_state_change": 3,
}
EVENT_BACKED_SUPPRESS_GATE_REASONS = {
    "label_level_pseudotrack_not_physical_movement",
    "object_movement_not_measured",
    "physical_event_gate_rejected_object_movement",
    "unstable_object_track_for_movement",
}
EVENT_BACKED_SPARSE_CONTACT_REVIEW_MIN_SCORE = 0.45
EVENT_BACKED_SMALL_OBJECT_SPARSE_CONTACT_REVIEW_MIN_SCORE = 0.30
EVENT_BACKED_SPARSE_CONTACT_SMALL_OBJECTS = {"beaker", "paper", "weighing_paper", "spatula", "pipette", "pipette_tip"}

OBJECT_DISPLAY_NAMES = {
    "balance": "天平",
    "scale": "天平",
    "paper": "称量纸",
    "weighing_paper": "称量纸",
    "reagent_bottle": "试剂瓶",
    "sample_bottle": "样品瓶",
    "sample_bottle_blue": "蓝盖样品瓶",
    "spatula": "药匙",
    "pipette": "移液枪",
    "pipette_tip": "移液枪头",
    "beaker": "烧杯",
    "container": "容器",
    "tube": "试管",
    "flask": "烧瓶",
    "panel": "设备面板",
    "magnetic_stir_bar": "磁力搅拌子",
    "magnetic_stirrer": "磁力搅拌器",
    "liquid": "液体",
    "object": "物体",
}

CHINESE_OBJECT_NAMES = {
    "balance": "\u5929\u5e73",
    "scale": "\u5929\u5e73",
    "panel": "\u8bbe\u5907\u9762\u677f",
    "paper": "\u79f0\u91cf\u7eb8",
    "weighing_paper": "\u79f0\u91cf\u7eb8",
    "reagent_bottle": "\u8bd5\u5242\u74f6",
    "reagent_bottle_open": "\u8bd5\u5242\u74f6",
    "bottle_cap": "\u8bd5\u5242\u74f6",
    "sample_bottle": "\u6837\u54c1\u74f6",
    "sample_bottle_blue": "\u84dd\u76d6\u6837\u54c1\u74f6",
    "bottle": "\u8bd5\u5242\u74f6",
    "vial": "\u6837\u54c1\u74f6",
    "spatula": "\u836f\u5319",
    "pipette": "\u79fb\u6db2\u67aa",
    "pipette_tip": "\u79fb\u6db2\u67aa\u5934",
    "beaker": "\u70e7\u676f",
    "container": "\u5bb9\u5668",
    "tube": "\u8bd5\u7ba1",
    "flask": "\u70e7\u74f6",
    "magnetic_stir_bar": "\u78c1\u529b\u6405\u62cc\u5b50",
    "magnetic_stirrer": "\u78c1\u529b\u6405\u62cc\u5668",
    "liquid": "\u6db2\u4f53",
    "object": "\u7269\u4f53",
}

MATERIAL_FRAME_DISPLAY_NAMES = {
    "contact": "\u63a5\u89e6\u5e27",
    "peak": "\u5cf0\u503c\u5e27",
    "release": "\u91ca\u653e\u5e27",
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


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, str(default))))
    except Exception:
        return int(default)


def _env_truthy(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() not in {"", "0", "false", "no", "off"}


def _fast_locate_material_mode() -> bool:
    return _env_truthy("KEY_ACTION_FAST_LOCATE_ONLY", False) or _env_truthy("KEY_ACTION_DEFER_SEGMENT_ASSETS", False)


def _material_candidate_rerender_boxes_enabled() -> bool:
    raw = os.environ.get("KEY_ACTION_MATERIAL_CANDIDATE_RERENDER_BOXES")
    if raw is None and _fast_locate_material_mode():
        raw = os.environ.get("KEY_ACTION_FAST_LOCATE_MATERIAL_CANDIDATE_RERENDER_BOXES", "0")
    if raw is None:
        return True
    return str(raw).strip().lower() not in {"", "0", "false", "no", "off"}


def _material_candidate_worker_count() -> int:
    if os.environ.get("KEY_ACTION_MATERIAL_CANDIDATE_WORKERS") is None:
        fast_workers = os.environ.get("KEY_ACTION_FAST_LOCATE_MATERIAL_CANDIDATE_WORKERS")
        if fast_workers is not None:
            return max(1, _env_int("KEY_ACTION_FAST_LOCATE_MATERIAL_CANDIDATE_WORKERS", 4))
    return max(1, _env_int("KEY_ACTION_MATERIAL_CANDIDATE_WORKERS", 4))


def _material_reference_worker_count() -> int:
    if os.environ.get("KEY_ACTION_MATERIAL_REFERENCE_WORKERS") is not None:
        return max(1, _env_int("KEY_ACTION_MATERIAL_REFERENCE_WORKERS", 4))
    return _material_candidate_worker_count()


def _active_event_backed_candidate_types() -> set[str]:
    configured = os.environ.get("KEY_ACTION_EVENT_BACKED_CANDIDATE_TYPES")
    if configured and configured.strip():
        return {item.strip() for item in configured.split(",") if item.strip()}
    scope = os.environ.get("KEY_ACTION_PHYSICAL_ACTION_SCOPE", "").strip().lower()
    if scope in {"core_v1", "v1_core", "preprocess_v1", "material_preprocess_v1"}:
        return set(CORE_V1_EVENT_BACKED_CANDIDATE_TYPES)
    return set(EVENT_BACKED_CANDIDATE_TYPES)


def active_physical_action_types() -> set[str]:
    configured = os.environ.get("KEY_ACTION_PHYSICAL_ACTION_TYPES")
    if configured and configured.strip():
        return {item.strip() for item in configured.split(",") if item.strip()}
    scope = os.environ.get("KEY_ACTION_PHYSICAL_ACTION_SCOPE", "").strip().lower()
    if scope in {"core_v1", "v1_core", "preprocess_v1", "material_preprocess_v1"}:
        return set(CORE_V1_PHYSICAL_ACTION_TYPES)
    return set(PHYSICAL_ACTION_TYPES)


def _read_jsonl_if_exists(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    return read_jsonl(source)


def _material_yolo_frame_rows(session_root: Path) -> list[dict[str, Any]]:
    cv_dir = session_root / "cv_outputs"
    return [
        *_read_jsonl_if_exists(cv_dir / "yolo_frame_rows.jsonl"),
        *_read_jsonl_if_exists(cv_dir / "yolo_micro_frame_rows.jsonl"),
    ]


def _primary_object_from_dual_view_event(event: Mapping[str, Any]) -> str:
    for value in (
        event.get("primary_object"),
        event.get("canonical_object"),
        event.get("canonical_object_family"),
        event.get("object_family"),
        event.get("object_display_name"),
        event.get("canonical_action_type"),
    ):
        text = str(value or "").strip().lower()
        if "balance" in text or "天平" in text:
            return "balance"
        if "paper" in text or "称量纸" in text:
            return "paper"
        if "bottle" in text or "试剂瓶" in text or "瓶" in text:
            return "reagent_bottle"
        if "pipette" in text or "移液" in text:
            return "pipette"
        if "spatula" in text or "药匙" in text:
            return "spatula"
        if "beaker" in text or "烧杯" in text:
            return "beaker"
        if "container" in text or "容器" in text:
            return "container"
        canonical = canonical_yolo_label(value)
        if canonical:
            return canonical
    return "container"


def _segment_covering_time(segment_rows: list[dict[str, Any]], start_sec: float, end_sec: float) -> dict[str, Any]:
    if not segment_rows:
        return {}
    center = (start_sec + end_sec) / 2.0
    best: tuple[float, dict[str, Any]] | None = None
    for segment in segment_rows:
        seg_start, seg_end = _segment_session_window(segment, fallback_start=start_sec, fallback_end=end_sec)
        if seg_start <= center <= seg_end:
            return segment
        distance = min(abs(center - seg_start), abs(center - seg_end))
        if best is None or distance < best[0]:
            best = (distance, segment)
    return best[1] if best is not None else segment_rows[0]


def _segment_session_window(
    segment: Mapping[str, Any],
    *,
    fallback_start: float,
    fallback_end: float,
) -> tuple[float, float]:
    cv_detection = segment.get("cv_detection") if isinstance(segment.get("cv_detection"), Mapping) else {}
    start = _safe_float(
        segment.get(
            "session_start_sec",
            segment.get("start_sec", segment.get("global_start_sec", cv_detection.get("start_sec", fallback_start))),
        ),
        fallback_start,
    )
    end = _safe_float(
        segment.get(
            "session_end_sec",
            segment.get("end_sec", segment.get("global_end_sec", cv_detection.get("end_sec", fallback_end))),
        ),
        fallback_end,
    )
    if end <= start:
        starts: list[float] = []
        ends: list[float] = []
        for view in ("third_person", "first_person"):
            view_data = segment.get(view) if isinstance(segment.get(view), Mapping) else {}
            if view_data:
                starts.append(_safe_float(view_data.get("local_start_sec", view_data.get("start_sec", fallback_start)), fallback_start))
                ends.append(_safe_float(view_data.get("local_end_sec", view_data.get("end_sec", fallback_end)), fallback_end))
        if starts and ends:
            start = min(starts)
            end = max(ends)
    if end <= start:
        end = max(fallback_end, start)
    return start, end


def _evidence_rows_for_view_action(
    view_action: Mapping[str, Any],
    yolo_frame_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    view = str(view_action.get("view") or "").strip()
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for raw_index in view_action.get("source_row_indices") or []:
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            continue
        if index < 0 or index >= len(yolo_frame_rows) or index in seen:
            continue
        seen.add(index)
        row = dict(yolo_frame_rows[index])
        if view and evidence_view(row) != view:
            continue
        rows.append(row)
    if rows:
        return rows

    target_time = _safe_float(view_action.get("session_start_sec", view_action.get("start_sec", view_action.get("local_time_sec", 0.0))))
    for row in yolo_frame_rows:
        if view and evidence_view(row) != view:
            continue
        if abs(_material_evidence_time(row) - target_time) <= 0.75:
            rows.append(dict(row))
    return rows


def _dual_view_action_event_material_micro_rows(
    session_root: Path,
    segment_rows: list[dict[str, Any]],
    yolo_frame_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    metadata_dir = session_root / "metadata"
    event_rows = []
    for row in _read_jsonl_if_exists(metadata_dir / "dual_view_action_events.jsonl"):
        if not isinstance(row, dict):
            continue
        status = str(row.get("status") or "").strip().lower()
        formal_allowed = bool(row.get("formal_material_allowed") or row.get("formal_event_promoted"))
        if status in {"confirmed", "matched_dual_view"} or formal_allowed:
            event_rows.append(row)
    if not event_rows:
        return []
    view_action_rows = {
        str(row.get("evidence_id") or ""): row
        for row in _read_jsonl_if_exists(metadata_dir / "view_action_evidence.jsonl")
        if isinstance(row, dict) and str(row.get("evidence_id") or "")
    }
    micros: list[dict[str, Any]] = []
    for event in event_rows:
        event_id = str(event.get("dual_event_id") or event.get("dual_view_action_event_id") or event.get("event_id") or "").strip()
        if not event_id:
            continue
        first_evidence = view_action_rows.get(str(event.get("first_evidence_id") or ""))
        third_evidence = view_action_rows.get(str(event.get("third_evidence_id") or ""))
        if not isinstance(first_evidence, dict) or not isinstance(third_evidence, dict):
            continue
        first_rows = _evidence_rows_for_view_action(first_evidence, yolo_frame_rows)
        third_rows = _evidence_rows_for_view_action(third_evidence, yolo_frame_rows)
        if not first_rows or not third_rows:
            continue
        start_sec = _safe_float(event.get("session_start_sec", event.get("start_sec", 0.0)))
        end_sec = _safe_float(event.get("session_end_sec", event.get("end_sec", start_sec)), start_sec)
        segment = _segment_covering_time(segment_rows, start_sec, end_sec)
        segment_start, segment_end = _segment_session_window(segment, fallback_start=start_sec, fallback_end=end_sec)
        padded_start = max(segment_start, start_sec - 0.8)
        padded_end = min(segment_end, max(end_sec, start_sec + 0.6) + 0.8)
        if padded_end <= padded_start:
            padded_start, padded_end = start_sec, max(start_sec + 0.6, end_sec)
        primary = canonical_yolo_label(_primary_object_from_dual_view_event(event)) or "container"
        semantic = _canonical_action_fields(primary, event.get("action_display_name") or event.get("action_family"))
        action_name = _approved_material_chinese_action_name(
            {
                "primary_object": primary,
                "action_name": event.get("action_display_name") or _action_name(primary),
                **semantic,
            }
        )
        micro_id = f"{event_id}_micro"
        micros.append(
            {
                "schema_version": "dual_view_action_event_material_micro.v1",
                "micro_segment_id": micro_id,
                "source_micro_segment_id": micro_id,
                "parent_segment_id": str(segment.get("segment_id") or segment.get("episode_id") or "seg_000001"),
                "segment_id": str(segment.get("segment_id") or segment.get("episode_id") or "seg_000001"),
                "session_id": event.get("session_id"),
                "start_sec": padded_start,
                "end_sec": padded_end,
                "session_start_sec": padded_start,
                "session_end_sec": padded_end,
                "event_start_sec": start_sec,
                "event_end_sec": end_sec,
                "duration_sec": max(0.0, padded_end - padded_start),
                "primary_object": primary,
                "primary_object_family": _material_object_family_for_label(primary),
                "object_family": event.get("object_family") or _material_object_family_for_label(primary),
                "canonical_action_type": event.get("canonical_action_type") or semantic.get("canonical_action_type"),
                "canonical_object": semantic.get("canonical_object"),
                "sop_phase": semantic.get("sop_phase"),
                "interaction_family": semantic.get("interaction_family"),
                "action_name": action_name,
                "display_title": action_name,
                "semantic_action": action_name,
                "dual_event_id": event_id,
                "dual_view_action_event_id": event_id,
                "dual_event_binding_source": "explicit_confirmed_dual_view_action_event",
                "dual_view_action_alignment_score": _safe_float(event.get("alignment_score"), 0.0),
                "first_evidence_id": event.get("first_evidence_id"),
                "third_evidence_id": event.get("third_evidence_id"),
                "formal_dual_view_action": True,
                "single_view_candidate": False,
                "yolo_evidence": [*third_rows, *first_rows],
                "interaction": {
                    "primary_object": primary,
                    "first_evidence_id": event.get("first_evidence_id"),
                    "third_evidence_id": event.get("third_evidence_id"),
                    "dual_event_id": event_id,
                    "alignment_score": _safe_float(event.get("alignment_score"), 0.0),
                },
                "quality_reasons": [
                    "confirmed_dual_view_action_event",
                    "first_person_same_action_evidence",
                    "third_person_same_action_evidence",
                ],
            }
        )
    return micros


def material_references_root(session_dir: str | Path) -> Path:
    """Return the run-local material reference mirror for a key-action run."""

    session_root = Path(session_dir)
    if session_root.name == "key_action_index":
        return session_root.parent / "material_references"
    return session_root / "material_references"


def material_library_root(session_dir: str | Path | None = None) -> Path | None:
    """Return the external D-drive material library root when configured.

    The run-local ``material_references`` folder remains the frontend-compatible
    mirror. This root is the formal handoff library for CLI/RAG consumers.
    """

    for env_name in MATERIAL_LIBRARY_ROOT_ENV_NAMES:
        configured = os.environ.get(env_name)
        if configured and configured.strip():
            return Path(configured.strip())
    if session_dir is not None and _looks_like_labembodied_session(Path(session_dir)):
        return DEFAULT_D_DRIVE_MATERIAL_LIBRARY_ROOT
    return None


def formal_material_library_root(session_dir: str | Path | None = None) -> Path | None:
    root = material_library_root(session_dir)
    if root is None:
        return None
    return root if root.name.lower() == "material_references" else root / "material_references"


def _looks_like_labembodied_session(session_root: Path) -> bool:
    try:
        parts = {part.lower() for part in session_root.resolve().parts}
    except Exception:
        parts = {part.lower() for part in session_root.parts}
    return "labembodied" in parts and "outputs" in parts


def formal_material_references_root(session_dir: str | Path) -> Path:
    """Return the formal handoff folder for generated key materials."""

    session_root = Path(session_dir)
    experiment = _experiment_metadata(session_root)
    return _formal_material_root(session_root, experiment["label"])


def frontend_material_references_root(session_dir: str | Path) -> Path:
    """Return the frontend material library folder used by the UI."""

    session_root = Path(session_dir)
    experiment = _experiment_metadata(session_root)
    return _legacy_formal_material_root(session_root, experiment["label"])


def material_reference_delivery_roots(session_dir: str | Path) -> list[Path]:
    """Return formal and frontend delivery roots without duplicates."""

    roots: list[Path] = []
    for root in (formal_material_references_root(session_dir), frontend_material_references_root(session_dir)):
        if root not in roots:
            roots.append(root)
    return roots


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
    index_json = ref_root / f"{MATERIAL_INDEX_BASENAME}.json"
    index_jsonl = ref_root / f"{MATERIAL_INDEX_BASENAME}.jsonl"
    archive_root = session_root / "archive" / f"material_references_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    archived_items = _prepare_reference_root(ref_root, archive_root, archive_existing=archive_existing)

    segment_rows = _read_jsonl_if_exists(metadata_dir / "key_action_segments.jsonl")
    yolo_frame_rows = _material_yolo_frame_rows(session_root)
    micro_rows = _read_jsonl_if_exists(metadata_dir / "micro_segments.jsonl")
    dual_event_micro_rows = _dual_view_action_event_material_micro_rows(session_root, segment_rows, yolo_frame_rows)
    if dual_event_micro_rows:
        if _env_truthy("KEY_ACTION_FORMAL_MATERIAL_REFERENCES_DUAL_EVENTS_ONLY", False):
            micro_rows = list(dual_event_micro_rows)
        else:
            existing_micro_ids = {str(row.get("micro_segment_id") or "") for row in micro_rows if isinstance(row, dict)}
            micro_rows = [
                *micro_rows,
                *[
                    row
                    for row in dual_event_micro_rows
                    if str(row.get("micro_segment_id") or "") not in existing_micro_ids
                ],
            ]
    annotated_lookup = _annotated_clip_lookup(_read_jsonl_if_exists(metadata_dir / "annotated_clips.jsonl"))
    segment_by_id = {str(row.get("segment_id") or ""): row for row in segment_rows}
    experiment = _experiment_metadata(session_root)
    formal_root = formal_material_references_root(session_root)
    ffmpeg_ok = (not dry_run) and _ffmpeg_available(ffmpeg_path)

    records: list[dict[str, Any]] = []
    planned_records: list[dict[str, Any]] = []
    material_tasks: list[MaterialGenerationTask] = []
    skipped: list[dict[str, Any]] = []
    used_names: set[str] = set()
    publish_allowed, blocked_reason = _session_formal_material_publish_allowed(session_root)
    if not publish_allowed and not dry_run:
        return _write_blocked_material_reference_summary(
            session_root=session_root,
            ref_root=ref_root,
            formal_root=formal_root,
            keyframe_dir=keyframe_dir,
            clip_dir=clip_dir,
            index_json=index_json,
            index_jsonl=index_jsonl,
            experiment=experiment,
            archived_items=archived_items,
            archive_root=archive_root,
            blocked_reason=blocked_reason,
            dry_run=dry_run,
            ffmpeg_ok=ffmpeg_ok,
        )

    for micro in micro_rows:
        micro_id = str(micro.get("micro_segment_id") or "")
        parent_id = str(micro.get("parent_segment_id") or micro.get("segment_id") or "")
        if _is_stale_identifier(micro_id) or _is_stale_identifier(parent_id):
            skipped.append({"micro_segment_id": micro_id, "reason": "stale_split_marker"})
            continue
        interaction = micro.get("interaction") if isinstance(micro.get("interaction"), dict) else {}
        primary = canonical_yolo_label(micro.get("primary_object") or interaction.get("primary_object") or "")
        start_sec = _safe_float(micro.get("start_sec", micro.get("session_start_sec")))
        end_sec = _safe_float(micro.get("end_sec", micro.get("session_end_sec")), start_sec)
        if end_sec <= start_sec:
            skipped.append({"micro_segment_id": micro_id, "reason": "invalid_time_range"})
            continue
        raw_evidence = _micro_material_evidence_rows(
            micro,
            yolo_frame_rows,
            primary_object=primary,
            start_sec=start_sec,
            end_sec=end_sec,
        )
        if not raw_evidence:
            skipped.append({"micro_segment_id": micro_id, "reason": "missing_yolo_evidence"})
            continue
        semantic_fields = enhance_material_semantics(
            micro,
            micro=micro,
            evidence_rows=raw_evidence,
            primary_object=primary,
            action_name=_action_name(primary or "object"),
        )
        semantic_display_title = semantic_fields.get("display_title")
        action_context = _material_name_context(
            micro,
            primary_object=primary,
            semantic_fields=semantic_fields,
        )
        action_name = _approved_material_chinese_action_name(action_context)
        semantic_fields = dict(semantic_fields)
        if semantic_display_title and str(semantic_display_title).strip() != action_name:
            semantic_fields["semantic_display_title"] = semantic_display_title
        semantic_fields["display_title"] = action_name
        primary_label = canonical_yolo_label(primary)
        if primary_label not in {"balance", "scale", "panel"}:
            semantic_fields.update(_canonical_action_fields(primary_label or primary, action_name))
        diagnostics = yolo_physical_evidence_diagnostics(raw_evidence, primary)
        valid_evidence = valid_yolo_physical_evidence(raw_evidence, primary)
        material_evidence, material_mode, material_valid_count = _select_material_evidence_rows(
            raw_evidence,
            primary,
        )
        formal_dual_event_id = _material_row_dual_event_id(micro)
        if not material_evidence and formal_dual_event_id:
            target_labels = _interaction_target_labels(primary)
            material_evidence = [
                row
                for row in raw_evidence
                if target_labels and _has_plausible_sparse_target_interaction(row, target_labels)
            ]
            if material_evidence:
                material_mode = SPARSE_PHYSICAL_EVIDENCE_MODE
                material_valid_count = len(valid_yolo_physical_evidence(material_evidence, primary))
        if not material_evidence:
            skipped.append(
                {
                    "micro_segment_id": micro_id,
                    "reason": "no_usable_yolo_hand_object_evidence",
                    "primary_object": primary,
                    "diagnostics": diagnostics,
                    "valid_evidence_count": len(valid_evidence),
                }
            )
            continue

        material_micro = dict(micro)
        material_micro["yolo_evidence"] = material_evidence
        material_micro["physical_evidence_mode"] = material_mode
        material_micro["valid_yolo_evidence_count"] = material_valid_count
        segment = segment_by_id.get(parent_id, {})
        file_date = _micro_date_label(micro, experiment["date"])
        material_name_context = _material_name_context(
            material_micro,
            primary_object=primary,
            semantic_fields=semantic_fields,
        )
        filename_base = _material_target_basename(material_name_context, experiment, date=file_date)
        duration = max(0.1, end_sec - start_sec)

        for view in ("third_person", "first_person"):
            raw_view_evidence = [item for item in raw_evidence if evidence_view(item) == view]
            view_evidence, view_mode, view_valid_count = _select_material_evidence_rows(
                raw_view_evidence,
                primary,
            )
            if not view_evidence and formal_dual_event_id:
                target_labels = _interaction_target_labels(primary)
                view_evidence = [
                    row
                    for row in raw_view_evidence
                    if target_labels and _has_plausible_sparse_target_interaction(row, target_labels)
                ]
                if view_evidence:
                    view_mode = SPARSE_PHYSICAL_EVIDENCE_MODE
                    view_valid_count = len(valid_yolo_physical_evidence(view_evidence, primary))
            required_frames = PHYSICAL_EVIDENCE_MIN_FRAMES if view_mode == STRICT_PHYSICAL_EVIDENCE_MODE else 1
            paired_view_alignment = False
            if len(view_evidence) < required_frames:
                if material_evidence and _allow_paired_view_context_material() and session_has_reliable_dual_view_alignment(session_root):
                    view_evidence = material_evidence
                    view_mode = PAIRED_VIEW_CONTEXT_MODE
                    view_valid_count = material_valid_count
                    required_frames = 0
                    paired_view_alignment = True
                else:
                    skipped.append(
                        {
                            "micro_segment_id": micro_id,
                            "view": view,
                            "reason": "no_usable_yolo_physical_evidence_for_view",
                            "primary_object": primary,
                            "usable_evidence_count": len(view_evidence),
                            "valid_evidence_count": view_valid_count,
                            "required_min_frames": required_frames,
                            "diagnostics": yolo_physical_evidence_diagnostics(raw_view_evidence, primary),
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
            annotation_target = _annotation_target_query(primary, semantic_fields)
            annotation_target_labels = sorted(_interaction_target_labels(annotation_target))
            tracklet_summary = (
                {
                    "mode": "paired_view_time_alignment",
                    "anchor_views": sorted({evidence_view(row) for row in view_evidence if evidence_view(row)}),
                    "target_view": view,
                }
                if paired_view_alignment
                else summarize_tracklets(
                    build_tracklet_annotations(
                        view_evidence,
                        target_labels=annotation_target_labels,
                        include_hands=True,
                    )
                )
            )
            evidence_diagnostics = yolo_physical_evidence_diagnostics(raw_view_evidence or view_evidence, primary)
            _add_clip_material_records(
                micro=material_micro,
                segment=segment,
                source_clip=source_clip,
                clip_dir=clip_dir,
                used_names=used_names,
                action_name=action_name,
                file_date=file_date,
                filename_base=filename_base,
                view=view,
                offset=offset,
                duration=duration,
                frame_rows=frame_rows,
                start_sec=start_sec,
                segment_start=segment_start,
                view_evidence=view_evidence,
                annotation_target=annotation_target,
                annotation_target_labels=annotation_target_labels,
                tracklet_summary=tracklet_summary,
                semantic_fields=semantic_fields,
                physical_evidence_mode=view_mode,
                valid_evidence_count=view_valid_count,
                usable_evidence_count=len(view_evidence),
                evidence_diagnostics=evidence_diagnostics,
                render_yolo_annotations=not paired_view_alignment,
                planned_records=planned_records,
                material_tasks=material_tasks,
                dry_run=dry_run,
                ffmpeg_ok=ffmpeg_ok,
                ffmpeg_path=ffmpeg_path,
            )
            _add_keyframe_material_records(
                micro=material_micro,
                segment=segment,
                source_clip=source_clip,
                keyframe_dir=keyframe_dir,
                used_names=used_names,
                action_name=action_name,
                file_date=file_date,
                filename_base=filename_base,
                view=view,
                frame_rows=frame_rows,
                start_sec=start_sec,
                segment_start=segment_start,
                annotation_target=annotation_target,
                annotation_target_labels=annotation_target_labels,
                tracklet_summary=tracklet_summary,
                semantic_fields=semantic_fields,
                physical_evidence_mode=view_mode,
                valid_evidence_count=view_valid_count,
                usable_evidence_count=len(view_evidence),
                evidence_diagnostics=evidence_diagnostics,
                render_yolo_annotations=not paired_view_alignment,
                planned_records=planned_records,
                material_tasks=material_tasks,
                dry_run=dry_run,
                ffmpeg_ok=ffmpeg_ok,
                ffmpeg_path=ffmpeg_path,
            )

    if material_tasks and not dry_run:
        for row, target, generated, error in _run_material_reference_tasks(material_tasks):
            _append_material_record_result(records, row, target, generated=generated, error=error)

    unfiltered_output_rows = records if not dry_run else planned_records
    output_rows, formal_gate_suppressed = apply_formal_dual_view_material_publish_gate(
        session_root,
        list(unfiltered_output_rows),
    )
    skipped.extend(formal_gate_suppressed)
    if not dry_run:
        output_rows, non_real_suppressed = _filter_publishable_material_rows(output_rows, ref_root, session_root=session_root)
        skipped.extend(non_real_suppressed)
    summary = _generated_material_summary(
        session_root,
        ref_root,
        formal_root,
        keyframe_dir,
        clip_dir,
        index_json,
        index_jsonl,
        experiment,
        records,
        planned_records,
        output_rows,
        skipped,
        archived_items,
        archive_root,
        dry_run=dry_run,
        ffmpeg_ok=ffmpeg_ok,
    )
    summary["parallel_workers"] = _material_reference_worker_count()
    summary["material_generation_task_count"] = len(material_tasks)
    stream_rows = _write_material_stream(ref_root, output_rows, session_root=session_root)
    summary["material_stream"] = str(ref_root / "material_stream.jsonl")
    summary["material_stream_count"] = len(stream_rows)
    _write_material_reference_summary(ref_root, index_json, index_jsonl, summary, output_rows)
    if not dry_run:
        delivery_roots = [
            root
            for root in material_reference_delivery_roots(session_root)
            if root.resolve() != ref_root.resolve()
        ]

        def _sync_delivery_root(target_root: Path) -> str:
            _copy_simplified_materials(ref_root, target_root, summary)
            return str(target_root)

        if len(delivery_roots) == 1:
            summary["delivery_roots"] = [_sync_delivery_root(delivery_roots[0])]
        elif delivery_roots:
            with ThreadPoolExecutor(max_workers=len(delivery_roots), thread_name_prefix="material-delivery") as executor:
                summary["delivery_roots"] = list(executor.map(_sync_delivery_root, delivery_roots))
        index_roots = [ref_root, *delivery_roots]
        if index_roots:
            def _build_reference_index(target_root: Path) -> dict[str, Any]:
                try:
                    from .material_reference_index import build_key_material_reference_index

                    return build_key_material_reference_index(target_root)
                except Exception as exc:  # pragma: no cover - index generation is best-effort delivery metadata.
                    return {"material_root": str(target_root), "error": str(exc)}

            if len(index_roots) == 1:
                summary["reference_indexes"] = [_build_reference_index(index_roots[0])]
            else:
                with ThreadPoolExecutor(max_workers=len(index_roots), thread_name_prefix="material-index") as executor:
                    summary["reference_indexes"] = list(executor.map(_build_reference_index, index_roots))
        formal_library = formal_material_library_root(session_root)
        global_index_roots: list[Path] = []
        if formal_library is not None:
            formal_resolved = formal_library.resolve()
            for target_root in index_roots:
                try:
                    target_root.resolve().relative_to(formal_resolved)
                except ValueError:
                    continue
                global_index_roots.append(target_root)
        if global_index_roots:
            def _sync_global_index(target_root: Path) -> dict[str, Any]:
                try:
                    from .material_library_store import sync_material_library_package

                    return sync_material_library_package(target_root, library_root=formal_library.parent if formal_library.name.lower() == "material_references" else formal_library)
                except Exception as exc:  # pragma: no cover - global catalog sync is best-effort sidecar metadata.
                    return {"material_root": str(target_root), "error": str(exc)}

            if len(global_index_roots) == 1:
                summary["global_material_library_indexes"] = [_sync_global_index(global_index_roots[0])]
            else:
                with ThreadPoolExecutor(max_workers=len(global_index_roots), thread_name_prefix="global-material-index") as executor:
                    summary["global_material_library_indexes"] = list(executor.map(_sync_global_index, global_index_roots))
        if delivery_roots:
            _write_material_reference_summary(ref_root, index_json, index_jsonl, summary, output_rows)
    return summary


def _experiment_metadata(session_root: Path) -> dict[str, str]:
    payloads = _experiment_metadata_payloads(session_root)
    experiment_id = _top_level_metadata_text(payloads, ("experiment_id", "id")) or session_root.parent.name
    title, raw_title, title_is_technical = _select_experiment_title(payloads, session_root.parent.name)
    date = _select_experiment_date(
        payloads,
        experiment_id=experiment_id,
        session_root=session_root,
        raw_title=raw_title,
        raw_title_is_technical=title_is_technical,
    )
    if not date:
        date = datetime.now().strftime("%Y%m%d")
    return {
        "id": experiment_id,
        "title": title,
        "date": date,
        "label": _experiment_label(title, date),
    }


def _experiment_metadata_payloads(session_root: Path) -> list[dict[str, Any]]:
    paths = [
        session_root.parent / "experiment.json",
        session_root / "experiment.json",
        session_root.parent / "stream_manifest.json",
        session_root / "stream_manifest.json",
        session_root.parent / "experiment_run_manifest.json",
        session_root / "experiment_run_manifest.json",
        session_root / "manifest.json",
        session_root / "run_manifest.json",
        session_root.parent / "manifest.json",
    ]
    payloads: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if not path.is_file():
            continue
        payload = _load_json(path)
        if not payload:
            continue
        payload["_metadata_path"] = str(path)
        payloads.append(payload)
    return payloads


def _top_level_metadata_text(payloads: list[dict[str, Any]], keys: tuple[str, ...]) -> str | None:
    key_set = {key.lower() for key in keys}
    for payload in payloads:
        for key, value in payload.items():
            if key.lower() not in key_set:
                continue
            text = _metadata_text(value)
            if text:
                return text
    return None


def _metadata_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _iter_metadata_text_items(value: Any) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, (str, int, float)):
                text = _metadata_text(item)
                if text:
                    items.append((str(key), text))
            items.extend(_iter_metadata_text_items(item))
    elif isinstance(value, list):
        for item in value:
            items.extend(_iter_metadata_text_items(item))
    return items


def _select_experiment_title(payloads: list[dict[str, Any]], fallback: str) -> tuple[str, str, bool]:
    raw_candidates: list[str] = []
    for key in ("experiment_title", "experiment_name", "title", "name"):
        raw = _top_level_metadata_text(payloads, (key,))
        if raw:
            raw_candidates.append(raw)
    raw_candidates.append(fallback)

    for raw in raw_candidates:
        title = _clean_experiment_title(raw)
        if title and not _is_technical_experiment_title(title):
            return title, raw, False

    inferred = _domain_experiment_title(payloads)
    if inferred:
        return inferred, raw_candidates[0] if raw_candidates else fallback, True
    return "\u5b9e\u9a8c", raw_candidates[0] if raw_candidates else fallback, True


def _clean_experiment_title(value: Any) -> str:
    title = str(value or "").strip()
    title = re.sub(r"\s+", " ", title)
    title = re.sub(
        r"(?:[_\-\s]+(?:20\d{6}|20\d{2}[-/.年](?:0?[1-9]|1[0-2])[-/.月](?:0?[1-9]|[12]\d|3[01])日?))"
        r"(?:[_\-\s]+\d{3,6})?$",
        "",
        title,
    ).strip(" _-.")
    return title


def _is_technical_experiment_title(value: str) -> bool:
    text = _safe_name(value).lower()
    if not text:
        return True
    technical_patterns = (
        r"^candidate_disposition(?:_|$)",
        r"^material(?:_|$|20\d{6})",
        r"^material_references(?:_|$)",
        r"^material_candidates(?:_|$)",
        r"^test_long(?:_|$)",
        r"^camera_\d",
        r"vlm_enabled",
        r"key_action_rerun",
        r"key-action_rerun",
        r"rerun_20\d{6}",
        r"^[0-9a-f]{8}[-_][0-9a-f]{4}[-_][0-9a-f]{4}[-_][0-9a-f]{4}[-_][0-9a-f]{12}$",
    )
    if any(re.search(pattern, text) for pattern in technical_patterns):
        return True
    return bool(re.fullmatch(r"(?:20\d{6}|[0-9_]+)", text))


def _domain_experiment_title(payloads: list[dict[str, Any]]) -> str | None:
    texts = " ".join(text for payload in payloads for _key, text in _iter_metadata_text_items(payload)).lower()
    if "\u56fa\u4f53\u79f0\u91cf" in texts or "solid weighing" in texts:
        return "\u56fa\u4f53\u79f0\u91cf\u5b9e\u9a8c"
    if "\u79f0\u91cf" in texts and ("balance" in texts or "weigh" in texts):
        return "\u56fa\u4f53\u79f0\u91cf\u5b9e\u9a8c"
    if "\u6ef4\u5b9a" in texts or "titration" in texts:
        return "\u6ef4\u5b9a\u5b9e\u9a8c"
    if "\u79fb\u6db2" in texts or "pipett" in texts:
        return "\u79fb\u6db2\u5b9e\u9a8c"
    return None


def _select_experiment_date(
    payloads: list[dict[str, Any]],
    *,
    experiment_id: str,
    session_root: Path,
    raw_title: str,
    raw_title_is_technical: bool,
) -> str | None:
    for key in ("created_at",):
        date = _first_date_for_key(payloads, key)
        if date:
            return date

    for key in ("experiment_date", "recorded_date", "source_date", "date"):
        date = _first_date_for_key(payloads, key)
        if date:
            return date

    if not raw_title_is_technical:
        date = _date_from_text(raw_title)
        if date:
            return date

    for value in (experiment_id, session_root.parent.name, session_root.name):
        date = _date_from_text(str(value))
        if date:
            return date

    date = _source_path_date(payloads)
    if date:
        return date

    for key in ("session_start_time", "start_time", "recording_start_time"):
        date = _first_date_for_key(payloads, key)
        if date:
            return date

    for key in ("updated_at",):
        date = _first_date_for_key(payloads, key)
        if date:
            return date

    if raw_title_is_technical:
        date = _date_from_text(raw_title)
        if date:
            return date
    return None


def _first_date_for_key(payloads: list[dict[str, Any]], key: str) -> str | None:
    target = key.lower()
    for payload in payloads:
        for item_key, text in _iter_metadata_text_items(payload):
            if item_key.lower() != target:
                continue
            date = _date_from_text(text)
            if date:
                return date
    return None


def _source_path_date(payloads: list[dict[str, Any]]) -> str | None:
    path_key_markers = ("path", "file", "video", "clip", "source")
    path_value_markers = (".mp4", ".mov", ".avi", ".mkv", ".jsonl", "test_long")
    for payload in payloads:
        for key, text in _iter_metadata_text_items(payload):
            key_l = key.lower()
            text_l = text.lower()
            if not any(marker in key_l for marker in path_key_markers) and not any(
                marker in text_l for marker in path_value_markers
            ):
                continue
            date = _date_from_text(text)
            if date:
                return date
    return None


def _experiment_label(title: str, date: str) -> str:
    clean_title = _clean_experiment_title(title)
    safe_title = _safe_name(clean_title)
    safe_title = re.sub(r"(?:_20\d{6})+$", "", safe_title).strip("_") or "\u5b9e\u9a8c"
    return _safe_name(f"{safe_title}_{date}")


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
    formal_library = formal_material_library_root(session_root)
    if formal_library is not None:
        return formal_library / experiment_label
    return _legacy_formal_material_root(session_root, experiment_label)


def _legacy_formal_material_root(session_root: Path, experiment_label: str) -> Path:
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
    micro_id = str(micro.get("micro_segment_id") or "")
    view_data = segment.get(view) if isinstance(segment.get(view), dict) else {}
    micro_view_data = micro.get(view) if isinstance(micro.get(view), dict) else {}
    binding_candidates: list[Any] = []
    bindings = micro.get("asset_bindings") if isinstance(micro.get("asset_bindings"), list) else []
    for binding in bindings:
        if not isinstance(binding, dict) or str(binding.get("view") or "") != view:
            continue
        binding_candidates.extend(
            [
                binding.get("annotated_clip_path"),
                binding.get("clip_path"),
                binding.get("raw_clip_path"),
                binding.get("video_path"),
            ]
        )
    candidates = [
        micro_view_data.get("annotated_clip_path"),
        micro_view_data.get("clip_path"),
        micro_view_data.get("raw_clip_path"),
        micro_view_data.get("video_path"),
        micro.get(f"{view}_annotated_clip"),
        micro.get(f"{view}_clip_path"),
        micro.get(f"{view}_clip"),
        *binding_candidates,
        view_data.get("clip_path"),
        view_data.get("raw_clip_path"),
        lookup.get((segment_id, view)),
        view_data.get("annotated_clip_path"),
        segment.get(f"{view}_annotated_clip"),
        session_root / "clips" / segment_id / f"{view}.mp4" if segment_id else None,
        session_root / "clips" / segment_id / f"{view}_annotated.mp4" if segment_id else None,
        session_root / "clips" / "micro" / f"{micro_id}_{view}.mp4" if micro_id else None,
    ]
    if view == "third_person":
        # Legacy micro rows use a view-agnostic annotated_clip for the third-person
        # YOLO render. Never reuse it for first-person material, or the library
        # will publish a third-person image under a first-person filename.
        candidates.append(micro.get("annotated_clip"))
    resolved: list[Path] = []
    for candidate in candidates:
        if not candidate:
            continue
        path = candidate if isinstance(candidate, Path) else Path(str(candidate))
        if not path.is_absolute():
            path = session_root / path
        resolved.append(path)
        if path.is_file():
            return path
    return resolved[0] if resolved else None


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
    records = [
        row
        for row in summary.get("records", [])
        if isinstance(row, dict) and _material_row_is_publishable(row, root=ref_root)[0]
    ]
    desired_names: dict[str, set[str]] = {
        KEYFRAME_DIR_NAME: set(),
        KEY_CLIP_DIR_NAME: set(),
        REPORT_DIR_NAME: set(),
    }
    copy_specs: list[tuple[Path, Path]] = []
    for row in records:
        asset_kind = str(row.get("asset_kind") or row.get("material_type") or "")
        if asset_kind not in desired_names:
            continue
        filename = str(row.get("stored_filename") or row.get("file_name") or "").strip()
        source = _stored_path_from_row(row, ref_root)
        if not filename and source is not None:
            filename = source.name
        if not filename:
            continue
        desired_names[asset_kind].add(filename)
        if source is None or not source.is_file():
            continue
        target = simplified_root / asset_kind / filename
        if target.exists() and target.stat().st_size == source.stat().st_size:
            continue
        copy_specs.append((source, target))

    if simplified_root.exists():
        for name in (
            KEYFRAME_DIR_NAME,
            KEY_CLIP_DIR_NAME,
            REPORT_DIR_NAME,
            "manifest.json",
            "README.md",
            "material_stream.jsonl",
            f"{MATERIAL_INDEX_BASENAME}.json",
            f"{MATERIAL_INDEX_BASENAME}.jsonl",
            *OPENCLAW_EVIDENCE_PACKAGE_FILES,
        ):
            target = simplified_root / name
            if target.is_dir():
                for stale_file in target.iterdir():
                    if not stale_file.is_file():
                        continue
                    if name in desired_names and stale_file.name in desired_names[name]:
                        continue
                    try:
                        stale_file.unlink()
                    except PermissionError:
                        # Browser previews can keep MP4 handles open on Windows.
                        # The index below is authoritative, so a locked stale file
                        # should not block publishing newly approved materials.
                        pass
            elif target.exists():
                try:
                    target.unlink()
                except PermissionError:
                    pass
    for folder in (KEYFRAME_DIR_NAME, KEY_CLIP_DIR_NAME, REPORT_DIR_NAME):
        target_dir = simplified_root / folder
        target_dir.mkdir(parents=True, exist_ok=True)

    def _copy_material_file(item: tuple[Path, Path]) -> None:
        source, target = item
        _material_link_or_copy(source, target)

    _run_material_candidate_tasks([lambda item=item: _copy_material_file(item) for item in copy_specs])
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
    _write_material_stream(simplified_root, rebased_rows, session_root=Path(str(summary.get("session_dir") or "")))
    _write_json(simplified_root / "manifest.json", _manifest(simplified_summary))
    _write_readme(simplified_root / "README.md", simplified_summary)


def _write_material_stream(root: Path, rows: list[dict[str, Any]], *, session_root: Path | None = None) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        group_id = str(
            row.get("physical_action_material_id")
            or row.get("evidence_bundle_id")
            or row.get("dual_event_id")
            or row.get("candidate_group_id")
            or row.get("material_id")
            or row.get("reference_id")
            or ""
        ).strip()
        if not group_id:
            group_id = f"material_group_{len(grouped) + 1:04d}"
        grouped.setdefault(group_id, []).append(row)
    window_by_id: dict[str, dict[str, Any]] = {}
    if session_root is not None:
        formal_payload = _load_json(session_root / "metadata" / "formal_experiment_windows.json")
        for window in (formal_payload.get("windows", []) if isinstance(formal_payload, dict) else []):
            if not isinstance(window, dict):
                continue
            window_id = str(window.get("experiment_window_id") or window.get("window_id") or "").strip()
            if window_id:
                window_by_id[window_id] = window
    stream_rows: list[dict[str, Any]] = []
    for group_id, group_rows in grouped.items():
        if not group_rows:
            continue
        representative = group_rows[0]
        assets = _material_stream_assets(group_rows)
        window_id = (
            representative.get("segment_id")
            or representative.get("experiment_window_id")
            or representative.get("parent_segment_id")
        )
        window_id = str(window_id or "").strip() or None
        window = window_by_id.get(str(window_id or ""))
        source_window_sync_index = _material_stream_lineage_value(group_rows, "source_window_sync_index") or _material_stream_lineage_value(group_rows, "window_sync_index")
        orphan_material = bool(not window_id or not source_window_sync_index or (window_by_id and window_id not in window_by_id))
        stream_rows.append(
            {
                "schema_version": "time_anchored_material_stream.v1",
                "material_id": group_id,
                "evidence_bundle_id": representative.get("evidence_bundle_id")
                or representative.get("dual_event_id")
                or group_id,
                "action_event_id": representative.get("action_event_id")
                or representative.get("source_event_id")
                or representative.get("source_evidence_id"),
                "action_type": representative.get("physical_action_type")
                or representative.get("canonical_action_type")
                or representative.get("action_type")
                or representative.get("action_display_name"),
                "official_status": "official"
                if any(bool(row.get("official_material")) for row in group_rows)
                else "needs_review",
                "window_id": window_id,
                "experiment_window_id": window_id,
                "unit_id": _material_stream_lineage_value(group_rows, "unit_id"),
                "global_timestamp_us": _material_stream_lineage_value(group_rows, "global_timestamp_us"),
                "start_global_timestamp_us": _material_stream_lineage_value(group_rows, "start_global_timestamp_us"),
                "end_global_timestamp_us": _material_stream_lineage_value(group_rows, "end_global_timestamp_us"),
                "start_sync_index": _material_stream_lineage_value(group_rows, "start_sync_index"),
                "end_sync_index": _material_stream_lineage_value(group_rows, "end_sync_index"),
                "source_window_sync_index": source_window_sync_index,
                "start_window_sync_index": _material_stream_lineage_value(group_rows, "start_window_sync_index"),
                "end_window_sync_index": _material_stream_lineage_value(group_rows, "end_window_sync_index"),
                "peak_window_sync_index": _material_stream_lineage_value(group_rows, "peak_window_sync_index"),
                "window_preview": (window or {}).get("window_preview"),
                "sample_grid": (window or {}).get("sample_grid"),
                "first_keyframe": assets.get("first_person", {}).get(KEYFRAME_DIR_NAME),
                "third_keyframe": assets.get("third_person", {}).get(KEYFRAME_DIR_NAME),
                "first_keyclip": assets.get("first_person", {}).get(KEY_CLIP_DIR_NAME),
                "third_keyclip": assets.get("third_person", {}).get(KEY_CLIP_DIR_NAME),
                "side_by_side_keyclip": assets.get("side_by_side_keyclip"),
                "first_bbox_trace": _material_stream_lineage_value(group_rows, "first_bbox_trace"),
                "third_bbox_trace": _material_stream_lineage_value(group_rows, "third_bbox_trace"),
                "object_refs": _material_stream_objects(group_rows),
                "instrument_refs": _material_stream_instruments(group_rows),
                "action_phase": representative.get("action_phase")
                or representative.get("formal_publish_gate", {}).get("status")
                if isinstance(representative.get("formal_publish_gate"), dict)
                else representative.get("action_phase"),
                "sync_quality": _material_stream_lineage_value(group_rows, "sync_quality"),
                "cross_view_consistency": _material_stream_lineage_value(group_rows, "cross_view_consistency"),
                "keyframe_quality_score": _best_keyframe_quality_score(group_rows),
                "confidence": _safe_float(
                    representative.get("quality_score")
                    or representative.get("confidence")
                    or representative.get("score"),
                    0.0,
                ),
                "quality_flags": _material_stream_quality_flags(group_rows),
                "review_reason": _material_stream_review_reason(group_rows),
                "review_status": "needs_review"
                if not any(bool(row.get("official_material")) for row in group_rows)
                else "official",
                "orphan_material": orphan_material,
                "lineage": {
                    "session_dir": str(session_root) if session_root else None,
                    "material_root": str(root),
                    "source_rows": [
                        {
                            "reference_id": row.get("reference_id") or row.get("material_id"),
                            "stored_file": row.get("stored_file"),
                            "source_file": row.get("source_file"),
                            "asset_kind": row.get("asset_kind"),
                            "view": row.get("view") or _material_row_view(row),
                            "source_window_sync_index": row.get("source_window_sync_index") or row.get("window_sync_index"),
                            "start_window_sync_index": row.get("start_window_sync_index"),
                            "end_window_sync_index": row.get("end_window_sync_index"),
                            "peak_window_sync_index": row.get("peak_window_sync_index"),
                        }
                        for row in group_rows
                    ],
                },
                "cli_ready_folder": str(root),
                "frontend_item_id": representative.get("frontend_item_id")
                or representative.get("candidate_id")
                or representative.get("reference_id")
                or group_id,
                "memory_eligible": any(bool(row.get("memory_write_allowed")) for row in group_rows),
            }
        )
    _write_jsonl(root / "material_stream.jsonl", stream_rows)
    _write_material_window_dependency_report(root, stream_rows, session_root=session_root)
    return stream_rows


def _write_material_window_dependency_report(root: Path, stream_rows: list[dict[str, Any]], *, session_root: Path | None = None) -> dict[str, Any]:
    items = []
    for row in stream_rows:
        sync_path = row.get("source_window_sync_index")
        window_preview = row.get("window_preview")
        sample_grid = row.get("sample_grid")
        issues = []
        if not row.get("window_id"):
            issues.append("missing_window_id")
        if not sync_path:
            issues.append("missing_source_window_sync_index")
        elif not Path(str(sync_path)).is_file():
            issues.append("source_window_sync_index_missing_on_disk")
        if window_preview and not Path(str(window_preview)).is_file():
            issues.append("window_preview_missing_on_disk")
        if sample_grid and not Path(str(sample_grid)).is_file():
            issues.append("sample_grid_missing_on_disk")
        if row.get("orphan_material"):
            issues.append("orphan_material")
        items.append(
            {
                "material_id": row.get("material_id"),
                "evidence_bundle_id": row.get("evidence_bundle_id"),
                "window_id": row.get("window_id"),
                "source_window_sync_index": sync_path,
                "window_preview": window_preview,
                "sample_grid": sample_grid,
                "official_status": row.get("official_status"),
                "review_status": row.get("review_status"),
                "orphan_material": bool(row.get("orphan_material")),
                "issues": issues,
                "validation_status": "pass" if not issues else "fail",
            }
        )
    report = {
        "schema_version": "material_window_dependency_report.v1",
        "material_count": len(stream_rows),
        "orphan_material_count": sum(1 for row in stream_rows if row.get("orphan_material")),
        "missing_window_id_count": sum(1 for item in items if "missing_window_id" in item["issues"]),
        "missing_source_window_sync_index_count": sum(1 for item in items if "missing_source_window_sync_index" in item["issues"]),
        "fail_count": sum(1 for item in items if item["validation_status"] != "pass"),
        "items": items,
        "policy": "Normal frontend material lists must only expose rows with a valid window_id and source_window_sync_index.",
    }
    reports_dir = root / "reports"
    _write_json(reports_dir / "material_window_dependency_report.json", report)
    if session_root is not None:
        _write_json(session_root / "metadata" / "material_window_dependency_report.json", report)
    return report


def _best_keyframe_quality_score(rows: list[dict[str, Any]]) -> float | None:
    scores = [
        _safe_float(row.get("selected_keyframe_score"), -1.0)
        for row in rows
        if str(row.get("asset_kind") or row.get("material_type") or "") == KEYFRAME_DIR_NAME
    ]
    scores = [score for score in scores if score >= 0]
    return round(max(scores), 6) if scores else None


def _material_stream_review_reason(rows: list[dict[str, Any]]) -> str | None:
    for row in rows:
        for key in ("review_reason", "review_reason_codes", "quality_reasons"):
            value = row.get(key)
            if isinstance(value, list) and value:
                return ", ".join(str(item) for item in value[:4])
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _mirror_review_candidates_to_external_library(
    session_root: Path,
    candidate_root: Path,
    candidate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    root = material_library_root(session_root)
    if root is None:
        return {"enabled": False, "reason": "material_library_root_not_configured"}
    review_rows = [
        dict(row)
        for row in candidate_rows
        if str(row.get("review_status") or "") == "needs_review"
        or str(row.get("candidate_source") or "") == "view_action_evidence_needs_review"
    ]
    experiment_id = session_root.parent.name if session_root.name == "key_action_index" else session_root.name
    target_root = root / experiment_id
    review_root = target_root / "review_candidates"
    copy_specs: list[tuple[Path, Path]] = []
    mirrored_rows: list[dict[str, Any]] = []
    for row in review_rows:
        mirrored = dict(row)
        mirrored["official_material"] = False
        mirrored["memory_write_allowed"] = False
        mirrored["candidate_status"] = mirrored.get("candidate_status") or "needs_review"
        mirrored["review_status"] = "needs_review"
        stored_file = row.get("stored_file")
        if stored_file:
            source = Path(str(stored_file))
            try:
                rel = source.relative_to(candidate_root)
            except ValueError:
                rel = Path(str(row.get("asset_kind") or "assets")) / source.name
            target = review_root / rel
            mirrored["stored_file"] = str(target)
            mirrored["cli_ready_folder"] = str(target_root)
            if source.is_file():
                copy_specs.append((source, target))
        side_by_side = row.get("side_by_side_keyclip")
        if side_by_side:
            side_source = Path(str(side_by_side))
            try:
                side_rel = side_source.relative_to(candidate_root)
            except ValueError:
                side_rel = Path(str(row.get("asset_kind") or KEY_CLIP_DIR_NAME)) / side_source.name
            side_target = review_root / side_rel
            mirrored["side_by_side_keyclip"] = str(side_target)
            if side_source.is_file():
                copy_specs.append((side_source, side_target))
        mirrored_rows.append(mirrored)

    def _copy(item: tuple[Path, Path]) -> None:
        source, target = item
        _material_link_or_copy(source, target)

    _run_material_candidate_tasks([lambda item=item: _copy(item) for item in copy_specs])
    target_root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(target_root / "review_candidate_materials.jsonl", mirrored_rows)
    _write_jsonl(target_root / "official_materials.jsonl", [])
    stream_rows = _write_material_stream(target_root, mirrored_rows, session_root=session_root)
    _write_json(
        target_root / "experiment_manifest.json",
        {
            "schema_version": "lab_material_library_experiment_manifest.v1",
            "experiment_id": experiment_id,
            "session_dir": str(session_root),
            "candidate_source_root": str(candidate_root),
            "review_candidate_root": str(review_root),
            "review_candidate_count": len(mirrored_rows),
            "official_material_count": 0,
            "memory_policy": "needs_review_materials_are_not_memory_eligible",
        },
    )
    _write_json(
        target_root / "cli_ready_report.json",
        {
            "schema_version": "cli_ready_report.v1",
            "status": "ready_with_review_candidates" if mirrored_rows else "no_review_candidates",
            "material_root": str(target_root),
            "material_stream": str(target_root / "material_stream.jsonl"),
            "review_candidate_materials": str(target_root / "review_candidate_materials.jsonl"),
            "official_materials": str(target_root / "official_materials.jsonl"),
            "review_candidate_count": len(mirrored_rows),
            "material_stream_count": len(stream_rows),
        },
    )
    ledger_summary: dict[str, Any] | None = None
    corpus_summary: dict[str, Any] | None = None
    try:
        from .experiment_action_ledger import build_experiment_action_ledger, refresh_labvideo_memory_corpus

        ledger = build_experiment_action_ledger(root, experiment_id)
        corpus = refresh_labvideo_memory_corpus(root)
        ledger_summary = {
            "ledger": str(target_root / "experiment_action_ledger.json"),
            "ledger_status": ledger.get("ledger_status"),
            "memory_eligible": ledger.get("memory_eligible"),
        }
        corpus_summary = {
            "corpus": str(root / "labvideo_memory_corpus.json"),
            "is_real_30_day_memory": corpus.get("is_real_30_day_memory"),
            "ledger_count": corpus.get("ledger_count"),
        }
    except Exception as exc:  # pragma: no cover - mirror should not fail material generation.
        ledger_summary = {"error": str(exc)}
    return {
        "enabled": True,
        "material_root": str(target_root),
        "review_candidate_root": str(review_root),
        "review_candidate_count": len(mirrored_rows),
        "material_stream_count": len(stream_rows),
        "official_material_count": 0,
        "experiment_action_ledger": ledger_summary,
        "labvideo_memory_corpus": corpus_summary,
    }


def _material_stream_assets(group_rows: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    assets: dict[str, dict[str, str]] = {}
    for row in group_rows:
        side_by_side = row.get("side_by_side_keyclip")
        if side_by_side:
            assets["side_by_side_keyclip"] = str(side_by_side)  # type: ignore[assignment]
        view = str(row.get("view") or _material_row_view(row) or "").strip() or "unknown"
        raw_kind = _material_row_kind(row)
        if raw_kind == "keyframe":
            kind = KEYFRAME_DIR_NAME
        elif raw_kind == "keyclip":
            kind = KEY_CLIP_DIR_NAME
        else:
            continue
        path = row.get("stored_file") or row.get("source_file")
        if not path:
            continue
        assets.setdefault(view, {})[kind] = str(path)
    return assets


def _material_stream_lineage_value(group_rows: list[dict[str, Any]], key: str) -> Any:
    for row in group_rows:
        if row.get(key) not in (None, ""):
            return row.get(key)
        for nested_key in ("lineage", "evidence_chain", "formal_publish_gate"):
            nested = row.get(nested_key)
            if isinstance(nested, dict) and nested.get(key) not in (None, ""):
                return nested.get(key)
    return None


def _material_stream_objects(group_rows: list[dict[str, Any]]) -> list[str]:
    values: set[str] = set()
    for row in group_rows:
        for key in ("primary_object", "object_label", "target_label", "raw_yolo_label"):
            value = row.get(key)
            if value:
                values.add(str(value))
    return sorted(values)


def _material_stream_instruments(group_rows: list[dict[str, Any]]) -> list[str]:
    instruments = set()
    for row in group_rows:
        for value in _material_stream_objects([row]):
            if value in {"balance", "device_panel", "panel", "pipette"}:
                instruments.add(value)
    return sorted(instruments)


def _material_stream_quality_flags(group_rows: list[dict[str, Any]]) -> list[str]:
    flags: set[str] = set()
    for row in group_rows:
        raw_flags = row.get("quality_flags")
        if isinstance(raw_flags, list):
            flags.update(str(item) for item in raw_flags if item)
        gate = row.get("formal_publish_gate")
        if isinstance(gate, dict) and gate.get("status"):
            flags.add(f"formal_gate_{gate.get('status')}")
    return sorted(flags)


def _refresh_openclaw_evidence_package(
    material_root: Path,
    session_root: Path,
    experiment: dict[str, str],
    *,
    package_scope: str,
) -> dict[str, Any]:
    try:
        from .evidence_package import build_evidence_package, validate_evidence_package

        source_manifest = session_root / "manifest.json"
        build_summary = build_evidence_package(
            material_root,
            source_manifest=source_manifest if source_manifest.exists() else None,
            key_action_index_dir=session_root if session_root.exists() else None,
            package_id=f"{experiment['id']}:{package_scope}",
            experiment_id=experiment["id"],
            include_reports=True,
        )
        validation = validate_evidence_package(material_root, strict=False)
        return {
            "schema_version": "openclaw_evidence_package_refresh.v1",
            "status": validation.get("status") or "unknown",
            "ok": bool(validation.get("ok")),
            "scope": package_scope,
            "package_root": str(material_root),
            "manifest": str(material_root / "evidence_package_manifest.json"),
            "reference_count": int(build_summary.get("reference_count") or 0),
            "physical_change_count": int(build_summary.get("physical_change_count") or 0),
            "portable": bool(build_summary.get("portable")),
            "validation": validation,
        }
    except Exception as exc:
        return {
            "schema_version": "openclaw_evidence_package_refresh.v1",
            "status": "failed",
            "ok": False,
            "scope": package_scope,
            "package_root": str(material_root),
            "error": str(exc),
        }


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
    ):
        source_value = report_summary.get(key) or (report_summary.get("json_path") if role == "professional_report_json" else None)
        if not source_value:
            continue
        source = Path(str(source_value))
        if not source.exists():
            continue
        candidate_target = report_dir / source.name
        _material_link_or_copy(source, candidate_target)
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
        **_candidate_asset_counts(rows),
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
        _material_link_or_copy(source, formal_target)
        _material_link_or_copy(source, simplified_target)
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


def _physical_candidate_asset_kind(row: dict[str, Any]) -> str:
    return str(row.get("asset_kind") or row.get("material_type") or "")


def _load_existing_candidate_review_rows(candidate_root: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    existing_candidate_rows = _read_jsonl_if_exists(candidate_root / f"{MATERIAL_CANDIDATE_INDEX_BASENAME}.jsonl")
    existing_status_by_id = {
        str(row.get("candidate_id") or ""): row
        for row in existing_candidate_rows
        if str(row.get("candidate_id") or "")
    }
    preserved_candidate_rows = [
        row
        for row in existing_candidate_rows
        if _physical_candidate_asset_kind(row) not in {KEYFRAME_DIR_NAME, KEY_CLIP_DIR_NAME}
    ]
    return existing_candidate_rows, existing_status_by_id, preserved_candidate_rows


def _candidate_source_rows(
    session_root: Path,
    source_index: Path,
    *,
    micro_source_rows: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    source_rows = read_jsonl(source_index) if source_index.exists() else []
    if source_index.exists() and not source_rows and _source_index_blocked_by_formal_publish_gate(source_index):
        return [], False
    if micro_source_rows is None:
        micro_source_rows = _micro_level_material_reference_rows(session_root)
    if micro_source_rows:
        nonphysical_source_rows = [
            row
            for row in source_rows
            if _physical_candidate_asset_kind(row) not in {KEYFRAME_DIR_NAME, KEY_CLIP_DIR_NAME}
        ]
        return _dedupe_preferred_candidate_source_rows([*micro_source_rows, *nonphysical_source_rows]), False

    physical_source_rows = [
        row
        for row in source_rows
        if _physical_candidate_asset_kind(row) in {KEYFRAME_DIR_NAME, KEY_CLIP_DIR_NAME}
    ]
    if physical_source_rows:
        return source_rows, False

    segment_rows = _segment_level_material_reference_rows(session_root)
    if segment_rows:
        return source_rows + segment_rows, True
    return source_rows, False


def _source_index_blocked_by_formal_publish_gate(source_index: Path) -> bool:
    summary = _load_json(source_index.with_suffix(".json"))
    skipped = summary.get("skipped") if isinstance(summary.get("skipped"), list) else []
    return any(
        isinstance(item, dict)
        and item.get("reason") == "formal_material_publish_gate"
        and item.get("suppression_reason") in {
            "single_view_material_rejected",
            "incomplete_dual_view_material_group",
            "missing_dual_view_action_event_binding",
            "missing_reliable_dual_view_alignment",
        }
        for item in skipped
    )


def _dedupe_preferred_candidate_source_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for row in rows:
        identity = _candidate_source_row_identity(row)
        if identity in seen:
            continue
        seen.add(identity)
        deduped.append(row)
    return deduped


def _candidate_source_row_identity(row: dict[str, Any]) -> tuple[str, ...]:
    raw_path = row.get("stored_file") or row.get("source_file") or row.get("source_clip_path") or row.get("source_clip")
    if raw_path:
        path_key = str(Path(str(raw_path)).resolve()).lower()
    else:
        path_key = str(row.get("stored_filename") or row.get("file_name") or "")
    return (
        _physical_candidate_asset_kind(row),
        str(row.get("micro_segment_id") or ""),
        str(row.get("parent_segment_id") or row.get("segment_id") or ""),
        str(row.get("view") or row.get("camera_view") or ""),
        str(row.get("frame_type") or row.get("frame_role") or ""),
        path_key,
    )


def _micro_level_material_reference_rows(session_root: Path) -> list[dict[str, Any]]:
    micro_rows = _read_jsonl_if_exists(session_root / "metadata" / "micro_segments.jsonl")
    yolo_frame_rows = _material_yolo_frame_rows(session_root)
    rows: list[dict[str, Any]] = []
    for micro in micro_rows:
        if not isinstance(micro, dict):
            continue
        micro_id = str(micro.get("micro_segment_id") or "").strip()
        parent_id = str(micro.get("parent_segment_id") or micro.get("segment_id") or "").strip()
        if not micro_id or _is_stale_identifier(micro_id) or _is_stale_identifier(parent_id):
            continue
        rows.extend(_micro_level_material_reference_rows_for_micro(session_root, micro, yolo_frame_rows=yolo_frame_rows))
    return _dedupe_preferred_candidate_source_rows(rows)


def _micro_level_material_reference_rows_for_micro(
    session_root: Path,
    micro: dict[str, Any],
    *,
    yolo_frame_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    primary_object = _micro_primary_object(micro)
    start_sec = _safe_float(micro.get("start_sec", micro.get("session_start_sec")))
    end_sec = _safe_float(micro.get("end_sec", micro.get("session_end_sec")), start_sec)
    raw_evidence = _micro_material_evidence_rows(
        micro,
        yolo_frame_rows or [],
        primary_object=primary_object,
        start_sec=start_sec,
        end_sec=end_sec,
    )
    secondary_objects = _micro_secondary_objects(micro, primary_object)
    action_name = _micro_action_name(micro, primary_object)
    semantic_fields = enhance_material_semantics(
        micro,
        micro=micro,
        evidence_rows=raw_evidence,
        primary_object=primary_object,
        secondary_objects=secondary_objects,
        action_name=action_name,
    )
    semantic_display_title = semantic_fields.get("display_title")
    action_context = _material_name_context(
        micro,
        primary_object=primary_object,
        semantic_fields=semantic_fields,
    )
    action_name = _approved_material_chinese_action_name(action_context)
    semantic_fields = dict(semantic_fields)
    if semantic_display_title and str(semantic_display_title).strip() != action_name:
        semantic_fields["semantic_display_title"] = semantic_display_title
    semantic_fields["display_title"] = action_name
    primary_label = canonical_yolo_label(primary_object)
    if primary_label not in {"balance", "scale", "panel"}:
        semantic_fields.update(_canonical_action_fields(primary_label or primary_object, action_name))
    bindings = _micro_asset_bindings_from_row(micro)
    rows: list[dict[str, Any]] = []
    for binding in bindings:
        view = _micro_binding_view(binding, micro, raw_evidence)
        if not view:
            continue
        rows.extend(
            _micro_binding_reference_rows(
                session_root=session_root,
                micro=micro,
                binding=binding,
                view=view,
                primary_object=primary_object,
                secondary_objects=secondary_objects,
                action_name=action_name,
                semantic_fields=semantic_fields,
                raw_evidence=raw_evidence,
            )
        )
    return rows


def _micro_primary_object(micro: dict[str, Any]) -> str:
    interaction = micro.get("interaction") if isinstance(micro.get("interaction"), dict) else {}
    for value in (
        micro.get("primary_object"),
        micro.get("interaction_object"),
        interaction.get("primary_object"),
        interaction.get("peak_primary_object"),
    ):
        label = canonical_yolo_label(value)
        if label:
            return label
        if str(value or "").strip():
            return str(value).strip()
    for row in micro.get("yolo_evidence") or []:
        if not isinstance(row, dict):
            continue
        label = canonical_yolo_label(row.get("primary_object") or row.get("object_label"))
        if label:
            return label
    return "container"


def _micro_action_name(micro: dict[str, Any], primary_object: str) -> str:
    text_description = micro.get("text_description") if isinstance(micro.get("text_description"), dict) else {}
    for value in (
        micro.get("action_name"),
        micro.get("semantic_action"),
        text_description.get("action_type"),
    ):
        text = str(value or "").strip()
        if text and text != "unknown_operation":
            return text
    return _action_name(primary_object or "container")


def _micro_asset_bindings_from_row(micro: dict[str, Any]) -> list[dict[str, Any]]:
    top_keyframes = _micro_keyframe_map(micro)
    raw_bindings = micro.get("asset_bindings")
    if isinstance(raw_bindings, list):
        bindings = [dict(item) for item in raw_bindings if isinstance(item, dict)]
        if bindings:
            views_present = {str(binding.get("view") or binding.get("camera_view") or "").strip() for binding in bindings}
            for view in ("third_person", "first_person"):
                view_data = micro.get(view) if isinstance(micro.get(view), dict) else {}
                clip_path = view_data.get("clip_path") or view_data.get("annotated_clip_path") or micro.get(f"{view}_clip") or micro.get(f"{view}_clip_path")
                if view not in views_present and (view_data or clip_path):
                    bindings.append(
                        {
                            "level": "micro_segment",
                            "micro_segment_id": micro.get("micro_segment_id"),
                            "parent_segment_id": micro.get("parent_segment_id") or micro.get("segment_id"),
                            "view": view,
                            "local_start_sec": view_data.get("local_start_sec", micro.get("start_sec")),
                            "local_end_sec": view_data.get("local_end_sec", micro.get("end_sec")),
                            "clip_path": clip_path,
                            "keyframe_path": view_data.get("keyframe_path"),
                            "keyframe_paths": view_data.get("keyframe_paths") if isinstance(view_data.get("keyframe_paths"), list) else list(top_keyframes.values()),
                            "keyframes": view_data.get("keyframes") if isinstance(view_data.get("keyframes"), dict) else dict(top_keyframes),
                            "evidence_role": "synchronized_context",
                            "evidence_source": "micro_view_fallback",
                        }
                    )
            for binding in bindings:
                if top_keyframes and not isinstance(binding.get("keyframes"), dict):
                    binding["keyframes"] = dict(top_keyframes)
                if top_keyframes and not isinstance(binding.get("keyframe_paths"), list):
                    binding["keyframe_paths"] = list(top_keyframes.values())
            return bindings

    bindings: list[dict[str, Any]] = []
    for view in ("third_person", "first_person"):
        view_data = micro.get(view) if isinstance(micro.get(view), dict) else {}
        clip_path = (
            view_data.get("clip_path")
            or view_data.get("annotated_clip_path")
            or micro.get(f"{view}_clip")
            or micro.get(f"{view}_clip_path")
        )
        if not view_data and not clip_path:
            continue
        bindings.append(
            {
                "level": "micro_segment",
                "micro_segment_id": micro.get("micro_segment_id"),
                "parent_segment_id": micro.get("parent_segment_id") or micro.get("segment_id"),
                "view": view,
                "local_start_sec": view_data.get("local_start_sec", micro.get("start_sec")),
                "local_end_sec": view_data.get("local_end_sec", micro.get("end_sec")),
                "clip_path": clip_path,
                "keyframe_path": view_data.get("keyframe_path"),
                "keyframe_paths": view_data.get("keyframe_paths") if isinstance(view_data.get("keyframe_paths"), list) else list(top_keyframes.values()),
                "keyframes": view_data.get("keyframes") if isinstance(view_data.get("keyframes"), dict) else dict(top_keyframes),
            }
        )
    if top_keyframes and not bindings:
        bindings.append(
            {
                "level": "micro_segment",
                "micro_segment_id": micro.get("micro_segment_id"),
                "parent_segment_id": micro.get("parent_segment_id") or micro.get("segment_id"),
                "view": micro.get("source_view") or micro.get("view"),
                "local_start_sec": micro.get("start_sec"),
                "local_end_sec": micro.get("end_sec"),
                "keyframe_paths": list(top_keyframes.values()),
                "keyframes": dict(top_keyframes),
            }
        )
    return bindings


def _micro_keyframe_map(micro: dict[str, Any]) -> dict[str, Any]:
    raw_keyframes = micro.get("keyframes")
    if isinstance(raw_keyframes, dict):
        return {str(key): value for key, value in raw_keyframes.items() if value}
    keyframes: dict[str, Any] = {}
    for role, keys in (
        ("contact", ("contact_keyframe", "contact_frame")),
        ("peak", ("peak_keyframe", "peak_frame")),
        ("release", ("release_keyframe", "release_frame")),
        ("middle", ("keyframe_path",)),
    ):
        for key in keys:
            if micro.get(key):
                keyframes[role] = micro.get(key)
                break
    if isinstance(raw_keyframes, list):
        for index, value in enumerate(raw_keyframes, start=1):
            if value:
                keyframes.setdefault(f"frame_{index}", value)
    return keyframes


def _micro_binding_view(binding: dict[str, Any], micro: dict[str, Any], raw_evidence: list[dict[str, Any]]) -> str:
    for value in (
        binding.get("view"),
        binding.get("camera_view"),
        binding.get("source_view"),
        micro.get("source_view"),
        micro.get("view"),
    ):
        text = str(value or "").strip()
        if text:
            return text
    for row in raw_evidence:
        view = evidence_view(row)
        if view:
            return view
    return ""


def _micro_binding_reference_rows(
    *,
    session_root: Path,
    micro: dict[str, Any],
    binding: dict[str, Any],
    view: str,
    primary_object: str,
    secondary_objects: list[str],
    action_name: str,
    semantic_fields: dict[str, Any],
    raw_evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    raw_view_evidence = [row for row in raw_evidence if evidence_view(row) == view]
    view_evidence = raw_view_evidence
    selected_evidence, physical_evidence_mode, valid_count = _select_material_evidence_rows(
        view_evidence,
        primary_object,
    )
    paired_view_alignment = False
    if (
        not selected_evidence
        and raw_evidence
        and _allow_paired_view_context_material()
        and session_has_reliable_dual_view_alignment(session_root)
    ):
        fallback_evidence, fallback_mode, fallback_valid_count = _select_material_evidence_rows(
            raw_evidence,
            primary_object,
        )
        if fallback_evidence:
            view_evidence = raw_evidence
            selected_evidence = fallback_evidence
            physical_evidence_mode = "paired_view_time_alignment"
            valid_count = fallback_valid_count
            paired_view_alignment = True
        elif not view_evidence:
            view_evidence = raw_evidence
            physical_evidence_mode = fallback_mode
            valid_count = fallback_valid_count
    if not selected_evidence:
        return []
    evidence_diagnostics = yolo_physical_evidence_diagnostics(view_evidence, primary_object)
    evidence_fields = _micro_asset_evidence_fields(
        selected_evidence,
        view_evidence,
        physical_evidence_mode=physical_evidence_mode,
        valid_evidence_count=valid_count,
        evidence_diagnostics=evidence_diagnostics,
    )
    if paired_view_alignment:
        evidence_fields["candidate_source"] = "paired_view_micro_segment_key_asset_reference"
        evidence_fields["physical_evidence_mode"] = "paired_view_time_alignment"
        evidence_fields["quality_reasons"] = _ordered_unique_text(
            [
                *(_list_strings(evidence_fields.get("quality_reasons"))),
                "paired_view_time_alignment",
                "target_view_uses_synchronized_micro_asset",
            ]
        )
    start_sec = _safe_float(micro.get("start_sec", micro.get("session_start_sec", binding.get("local_start_sec", 0.0))))
    end_sec = _safe_float(micro.get("end_sec", micro.get("session_end_sec", binding.get("local_end_sec", start_sec))), start_sec)
    taxonomy = _semantic_taxonomy(semantic_fields) or _canonical_action_fields(
        _semantic_primary_for_taxonomy(semantic_fields, primary_object),
        action_name,
    )
    evidence_group_id = _material_evidence_group_id(
        micro,
        primary_object=primary_object,
        start_sec=start_sec,
        end_sec=end_sec,
    )
    clip_source = _micro_binding_clip_source(session_root, micro, binding, view)
    common = {
        "schema_version": "material_reference.item.v1",
        "trace_schema_version": MATERIAL_REFERENCE_TRACE_SCHEMA_VERSION,
        "candidate_source": evidence_fields.get("candidate_source"),
        "fallback_reason": "micro_segment_assets_preferred_over_pdf_or_segment_fallback",
        "action_name": action_name,
        **taxonomy,
        **semantic_fields,
        "micro_segment_id": micro.get("micro_segment_id"),
        "parent_segment_id": micro.get("parent_segment_id") or micro.get("segment_id"),
        "segment_id": micro.get("parent_segment_id") or micro.get("segment_id"),
        "evidence_group_id": evidence_group_id,
        "material_group_id": evidence_group_id,
        "physical_action_material_id": evidence_group_id,
        "evidence_window_id": evidence_group_id,
        "view": view,
        "camera_view": view,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "time_start": start_sec,
        "time_end": end_sec,
        "primary_object": primary_object,
        "secondary_objects": secondary_objects,
        "secondary_actions": _micro_secondary_actions(micro, primary_object, secondary_objects),
        "generated": False,
        "dry_run": False,
        "error": None,
        "yolo_box_required": bool(physical_evidence_mode and physical_evidence_mode != "paired_view_time_alignment"),
        "box_filter": "paired_view_time_alignment_asset_reference" if paired_view_alignment else "micro_segment_asset_reference",
        "time_range_sec": f"{start_sec:.3f}-{end_sec:.3f}",
        "yolo_annotated_required": False,
        "yolo_evidence_count": len(view_evidence),
        "source_clip": str(clip_source) if clip_source else None,
        "source_clip_path": str(clip_source) if clip_source else None,
        "evidence_chain": {
            "schema_version": MATERIAL_REFERENCE_TRACE_SCHEMA_VERSION,
            "camera_view": view,
            "time_start": start_sec,
            "time_end": end_sec,
            "evidence_group_id": evidence_group_id,
            "micro_segment_id": micro.get("micro_segment_id"),
            "segment_id": micro.get("parent_segment_id") or micro.get("segment_id"),
            "candidate_source": evidence_fields.get("candidate_source"),
            "candidate_disposition": None,
        },
        **evidence_fields,
    }
    rows: list[dict[str, Any]] = []
    if clip_source is not None:
        rows.append(
            _micro_reference_row(
                common,
                material_type=KEY_CLIP_DIR_NAME,
                source=clip_source,
                frame_type=None,
            )
        )
    for frame_type, frame_path in _micro_keyframe_sources(session_root, micro, binding):
        rows.append(
            _micro_reference_row(
                common,
                material_type=KEYFRAME_DIR_NAME,
                source=frame_path,
                frame_type=frame_type,
            )
        )
    return rows


def _micro_asset_evidence_fields(
    selected_evidence: list[dict[str, Any]],
    view_evidence: list[dict[str, Any]],
    *,
    physical_evidence_mode: str,
    valid_evidence_count: int,
    evidence_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    if physical_evidence_mode:
        return _material_evidence_record_fields(
            selected_evidence,
            physical_evidence_mode=physical_evidence_mode,
            valid_evidence_count=valid_evidence_count,
            usable_evidence_count=len(selected_evidence),
            evidence_diagnostics=evidence_diagnostics,
        )
    return {
        "candidate_source": "micro_segment_key_asset_reference",
        "physical_evidence_mode": "micro_segment_asset_reference",
        "physical_evidence_required_min_frames": PHYSICAL_EVIDENCE_MIN_FRAMES,
        "valid_yolo_evidence_count": int(valid_evidence_count),
        "usable_yolo_evidence_count": len(view_evidence),
        "physical_evidence_diagnostics": evidence_diagnostics,
        "evidence": {"raw_labels": _raw_yolo_labels_from_evidence(view_evidence)},
        "source_yolo_evidence": _compact_yolo_evidence_rows(view_evidence),
        "quality_reasons": ["micro_segment_asset_reference", "manual_material_review_required"],
    }


def _micro_binding_clip_source(session_root: Path, micro: dict[str, Any], binding: dict[str, Any], view: str) -> Path | None:
    view_data = micro.get(view) if isinstance(micro.get(view), dict) else {}
    for value in (
        binding.get("clip_path"),
        view_data.get("clip_path"),
        binding.get("micro_clip_path"),
        micro.get(f"{view}_clip"),
        micro.get(f"{view}_clip_path"),
        binding.get("annotated_clip_path"),
        view_data.get("annotated_clip_path"),
    ):
        path = _resolve_session_path(session_root, value)
        if path is not None:
            return path
    return None


def _micro_keyframe_sources(session_root: Path, micro: dict[str, Any], binding: dict[str, Any]) -> list[tuple[str, Path]]:
    sources: list[tuple[str, Path]] = []
    keyframes = binding.get("keyframes") if isinstance(binding.get("keyframes"), dict) else _micro_keyframe_map(micro)
    for role in ("peak", "contact", "release", "middle", "start", "end"):
        path = _resolve_session_path(session_root, keyframes.get(role))
        if path is not None:
            sources.append((role, path))
    for role, value in keyframes.items():
        if str(role) in {"peak", "contact", "release", "middle", "start", "end"}:
            continue
        path = _resolve_session_path(session_root, value)
        if path is not None and all(path != existing for _, existing in sources):
            sources.append((str(role), path))
    keyframe_paths = binding.get("keyframe_paths") if isinstance(binding.get("keyframe_paths"), list) else []
    for index, value in enumerate(keyframe_paths, start=1):
        path = _resolve_session_path(session_root, value)
        if path is not None and all(path != existing for _, existing in sources):
            sources.append((f"frame_{index}", path))
    single = _resolve_session_path(session_root, binding.get("keyframe_path") or micro.get("keyframe_path"))
    if single is not None and all(single != existing for _, existing in sources):
        sources.append(("middle", single))
    return sources


def _micro_reference_row(
    common: dict[str, Any],
    *,
    material_type: str,
    source: Path,
    frame_type: str | None,
) -> dict[str, Any]:
    row = dict(common)
    row.update(
        {
            "material_type": material_type,
            "asset_kind": material_type,
            "frame_type": frame_type,
            "frame_role": frame_type,
            "source_file": str(source),
            "stored_file": str(source),
            "stored_filename": source.name,
            "file_name": source.name,
            "exists": source.is_file(),
            "size_bytes": source.stat().st_size if source.is_file() else 0,
        }
    )
    if material_type == KEY_CLIP_DIR_NAME:
        row["source_clip"] = str(source)
        row["source_clip_path"] = str(source)
    evidence_chain = dict(row.get("evidence_chain") or {})
    evidence_chain.update(
        {
            "source_clip": row.get("source_clip") or str(source),
            "source_file": str(source),
            "asset_kind": material_type,
            "frame_type": frame_type,
        }
    )
    row["evidence_chain"] = evidence_chain
    return row


def _prepare_candidate_build_dirs(session_root: Path, candidate_root: Path, *, archive_existing: bool) -> tuple[Path, Path]:
    archive_root = session_root / "archive" / f"material_review_queue_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    _prepare_candidate_root(candidate_root, archive_root, archive_existing=archive_existing)
    keyframe_dir = candidate_root / KEYFRAME_DIR_NAME
    clip_dir = candidate_root / KEY_CLIP_DIR_NAME
    keyframe_dir.mkdir(parents=True, exist_ok=True)
    clip_dir.mkdir(parents=True, exist_ok=True)
    return keyframe_dir, clip_dir


def _run_material_candidate_tasks(tasks: list[Callable[[], None]]) -> None:
    if not tasks:
        return

    def _run_isolated(task: Callable[[], None]) -> None:
        # A single asset-extraction failure (e.g. an unreadable source video or
        # missing codec) must not abort the whole candidate build. The failed
        # output simply stays absent and is reconciled downstream into a
        # placeholder row with a missing_reason, keeping diagnostics isolated
        # from the rest of the queue (spec: failed items go to diagnostics and
        # never block or pollute the review queue).
        try:
            task()
        except Exception:
            return

    worker_count = min(_material_candidate_worker_count(), len(tasks))
    if worker_count <= 1:
        for task in tasks:
            _run_isolated(task)
        return
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="material-candidates") as executor:
        for _result in executor.map(_run_isolated, tasks):
            pass


def _run_material_reference_tasks(tasks: list[MaterialGenerationTask]) -> list[tuple[dict[str, Any], Path, bool, str | None]]:
    if not tasks:
        return []
    worker_count = min(_material_reference_worker_count(), len(tasks))
    if worker_count <= 1:
        return [task() for task in tasks]
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="material-references") as executor:
        return list(executor.map(lambda task: task(), tasks))


def _build_candidate_rows_from_source_rows(
    source_rows: list[dict[str, Any]],
    source_root: Path,
    *,
    keyframe_dir: Path,
    clip_dir: Path,
    existing_status_by_id: dict[str, dict[str, Any]],
    dry_run: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    skipped: list[dict[str, Any]] = []
    prepared: list[tuple[dict[str, Any], Path, Path]] = []
    used_names: set[str] = set()
    for row in source_rows:
        if _physical_candidate_asset_kind(row) not in {KEYFRAME_DIR_NAME, KEY_CLIP_DIR_NAME}:
            continue
        source = _stored_path_from_row(row, source_root)
        if source is None or (not dry_run and not source.is_file()):
            skipped.append({"source": str(source) if source else "", "reason": "source_file_missing"})
            continue
        target_dir = keyframe_dir if str(row.get("asset_kind") or "") == KEYFRAME_DIR_NAME else clip_dir
        filename = str(row.get("stored_filename") or row.get("file_name") or (source.name if source else "candidate"))
        target = target_dir / _unique_name(used_names, Path(filename).stem, Path(filename).suffix or source.suffix)
        prepared.append((row, source, target))

    if not dry_run:
        _run_material_candidate_tasks(
            [lambda source=source, target=target: _material_link_or_copy(source, target) for _row, source, target in prepared]
        )

    candidate_rows: list[dict[str, Any]] = []
    for row, source, target in prepared:
        candidate = _candidate_record_from_reference(row, source, target, exists=target.exists())
        _preserve_candidate_review_state(candidate, existing_status_by_id.get(str(candidate.get("candidate_id") or "")))
        candidate_rows.append(candidate)
    return candidate_rows, skipped


def _build_event_backed_candidate_rows(
    session_root: Path,
    *,
    keyframe_dir: Path,
    clip_dir: Path,
    existing_status_by_id: dict[str, dict[str, Any]],
    dry_run: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Surface non-box physical events as review candidates.

    These rows are deliberately not YOLO box deliverables. They expose the
    physical-event signals already produced by the evidence pipeline so the
    frontend can review liquid transfer, container state, panel operation, and
    basic contact candidates instead of showing an empty material library.
    """

    metadata_dir = session_root / "metadata"
    video_rows = _read_jsonl_if_exists(metadata_dir / "video_understanding.jsonl")
    advanced_rows = _read_jsonl_if_exists(metadata_dir / "advanced_vision_evidence.jsonl")
    micro_rows = _read_jsonl_if_exists(metadata_dir / "micro_segments.jsonl")
    micro_by_id = {
        str(row.get("micro_segment_id") or ""): row
        for row in micro_rows
        if isinstance(row, dict) and str(row.get("micro_segment_id") or "")
    }
    event_rows = _event_backed_source_events(video_rows, advanced_rows)
    candidate_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    used_names = {path.name for path in [*keyframe_dir.glob("*"), *clip_dir.glob("*")] if path.is_file()}
    prepared_events: list[dict[str, Any]] = []
    for event in event_rows:
        selected_assets = _selected_event_assets(session_root, event, micro_by_id=micro_by_id)
        if not selected_assets:
            skipped.append(
                {
                    "source_event_id": event.get("video_event_id") or event.get("evidence_id"),
                    "event_type": event.get("event_type") or event.get("evidence_type"),
                    "reason": "event_real_asset_missing",
                    "asset_skip_reasons": event.get("_asset_skip_reasons") or [],
                }
            )
            continue
        prepared_events.append(
            {
                "event": event,
                "selected_assets": selected_assets,
                **_event_surface_fields(event, micro_by_id),
            }
        )
    surfaced_events, suppressed_events = _select_event_backed_surface_events(prepared_events)
    skipped.extend(_event_backed_suppressed_records(suppressed_events))
    asset_specs: list[tuple[dict[str, Any], str, Path, dict[str, Any], Path]] = []
    for surface_item in surfaced_events:
        event = surface_item["event"]
        selected_assets = surface_item["selected_assets"]
        for asset_kind, source, asset_ref in selected_assets:
            target_dir = keyframe_dir if asset_kind == KEYFRAME_DIR_NAME else clip_dir
            target_suffix = ".jpg" if asset_kind == KEYFRAME_DIR_NAME else (source.suffix or ".mp4")
            target = target_dir / _unique_name(
                used_names,
                _event_asset_basename(event, asset_ref, asset_kind),
                target_suffix,
            )
            asset_specs.append((event, asset_kind, source, asset_ref, target))

    def _materialize_event_asset(event: dict[str, Any], asset_kind: str, source: Path, asset_ref: dict[str, Any], target: Path) -> None:
        if asset_kind == KEYFRAME_DIR_NAME and source.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}:
            _render_event_candidate_keyframe(
                event,
                source=source,
                target=target,
                asset_ref=asset_ref,
                micro_by_id=micro_by_id,
            )
        else:
            _material_link_or_copy(source, target)

    if not dry_run:
        _run_material_candidate_tasks(
            [
                lambda event=event, asset_kind=asset_kind, source=source, asset_ref=asset_ref, target=target: _materialize_event_asset(
                    event,
                    asset_kind,
                    source,
                    asset_ref,
                    target,
                )
                for event, asset_kind, source, asset_ref, target in asset_specs
            ]
        )

    annotation_tasks: list[Callable[[], None]] = []
    for event, asset_kind, source, asset_ref, target in asset_specs:
        candidate = _candidate_record_from_event(
            event,
            asset_ref,
            source,
            target,
            asset_kind,
            exists=target.exists(),
            micro_by_id=micro_by_id,
        )
        if not dry_run:
            annotation_tasks.append(
                lambda event=event, candidate=candidate, source=source, target=target, asset_kind=asset_kind: _render_event_candidate_annotation(
                    event,
                    candidate,
                    source=source,
                    target=target,
                    asset_kind=asset_kind,
                    micro_by_id=micro_by_id,
                )
            )
        _preserve_candidate_review_state(candidate, existing_status_by_id.get(str(candidate.get("candidate_id") or "")))
        candidate_rows.append(candidate)
    if not dry_run:
        _run_material_candidate_tasks(annotation_tasks)
    summary = {
        "enabled": True,
        "source_video_understanding_events": len(video_rows),
        "source_advanced_vision_events": len(advanced_rows),
        "event_groups_considered": len(event_rows),
        "event_groups_with_assets": len(prepared_events),
        "surfaced_event_groups": len(surfaced_events),
        "suppressed_event_groups": len(suppressed_events),
        "candidate_count": len(candidate_rows),
        "skipped_count": len(skipped),
        "event_type_counts": _count_by_field(candidate_rows, "event_type"),
        "review_route_counts": _count_by_field(candidate_rows, "review_route"),
        "surfaced_action_counts": _count_prepared_events_by(surfaced_events, "physical_action_type"),
        "suppressed_action_counts": _count_prepared_events_by(suppressed_events, "physical_action_type"),
        "suppressed_reason_counts": _count_prepared_events_by(suppressed_events, "suppression_reason"),
        "review_group_limit_total": EVENT_BACKED_REVIEW_GROUP_LIMIT_TOTAL,
        "low_quality_group_limit_total": EVENT_BACKED_LOW_QUALITY_GROUP_LIMIT_TOTAL,
    }
    return candidate_rows, skipped, summary


def _build_view_action_review_candidate_rows(
    session_root: Path,
    *,
    keyframe_dir: Path,
    clip_dir: Path,
    existing_status_by_id: dict[str, dict[str, Any]],
    dry_run: bool,
    ffmpeg_path: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Expose YOLO view-action evidence as frontend review candidates.

    This is deliberately a review layer, not a formal material publisher. It
    gives the operator real keyframes/keyclips to inspect when strict dual-view
    action pairing has not yet promoted any official micro material.
    """

    metadata_dir = session_root / "metadata"
    evidence_rows = _read_jsonl_if_exists(metadata_dir / "view_action_evidence.jsonl")
    source_by_view = _view_action_video_sources(session_root)
    windows = _view_action_review_windows(session_root)
    if not evidence_rows:
        return [], [], {
            "enabled": True,
            "source_evidence_rows": 0,
            "review_window_count": len(windows),
            "candidate_count": 0,
            "skipped_count": 0,
            "reason": "view_action_evidence_missing",
        }
    if not source_by_view:
        return [], [{"reason": "video_sources_missing_for_view_action_candidates"}], {
            "enabled": True,
            "source_evidence_rows": len(evidence_rows),
            "review_window_count": len(windows),
            "candidate_count": 0,
            "skipped_count": 1,
            "reason": "video_sources_missing",
        }

    prepared = _prepare_view_action_review_groups(evidence_rows, windows, source_by_view)
    selected_groups = _select_view_action_review_groups(prepared)
    used_names = {path.name for path in [*keyframe_dir.glob("*"), *clip_dir.glob("*")] if path.is_file()}
    candidate_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    tasks: list[Callable[[], None]] = []
    sync_generation_rows: list[dict[str, Any]] = []
    side_clip_targets: dict[str, Path] = {}
    side_clip_task_groups: set[str] = set()
    for group in selected_groups:
        for evidence in group.get("evidence_rows") or []:
            if not isinstance(evidence, dict):
                continue
            view = str(evidence.get("view") or "").strip()
            source_video = source_by_view.get(view)
            if source_video is None:
                skipped.append(
                    {
                        "evidence_id": evidence.get("evidence_id"),
                        "view": view,
                        "reason": "source_video_missing_for_view",
                    }
                )
                continue
            peak_sec = _view_action_peak_sec(evidence)
            if peak_sec is None:
                skipped.append(
                    {
                        "evidence_id": evidence.get("evidence_id"),
                        "view": view,
                        "reason": "evidence_time_missing",
                    }
                )
                continue
            group_id = str(group.get("candidate_group_id") or "")
            for asset_kind in (KEYFRAME_DIR_NAME, KEY_CLIP_DIR_NAME):
                sync_plan = _build_window_sync_asset_plan(
                    session_root, group, evidence, asset_kind, source_by_view
                )
                evidence_for_record = dict(evidence)
                if sync_plan:
                    evidence_for_record["start_sec"] = sync_plan["action_start_sec"]
                    evidence_for_record["end_sec"] = sync_plan["action_end_sec"]
                    evidence_for_record["peak_sec"] = sync_plan["action_peak_sec"]
                    evidence_for_record["start_global_timestamp_us"] = sync_plan.get("action_start_global_timestamp_us")
                    evidence_for_record["end_global_timestamp_us"] = sync_plan.get("action_end_global_timestamp_us")
                    evidence_for_record["global_timestamp_us"] = sync_plan.get("action_peak_global_timestamp_us")
                target_dir = keyframe_dir if asset_kind == KEYFRAME_DIR_NAME else clip_dir
                suffix = ".jpg" if asset_kind == KEYFRAME_DIR_NAME else ".mp4"
                target = target_dir / _unique_name(
                    used_names,
                    f"{group_id}_{view}_{'keyframe' if asset_kind == KEYFRAME_DIR_NAME else 'keyclip'}",
                    suffix,
                )
                if not dry_run:
                    if sync_plan and asset_kind == KEYFRAME_DIR_NAME:
                        tasks.append(
                            lambda plan=sync_plan, view=view, target=target: _extract_window_sync_keyframe(
                                plan,
                                view,
                                target,
                            )
                        )
                    elif sync_plan and asset_kind == KEY_CLIP_DIR_NAME:
                        tasks.append(
                            lambda plan=sync_plan, view=view, target=target: _write_window_sync_keyclip(
                                plan,
                                view,
                                target,
                            )
                        )
                        side_target = side_clip_targets.get(group_id)
                        if side_target is None:
                            side_target = clip_dir / _unique_name(
                                used_names,
                                f"{group_id}_side_by_side_keyclip",
                                ".mp4",
                            )
                            side_clip_targets[group_id] = side_target
                        sync_plan["side_by_side_keyclip"] = str(side_target)
                        if group_id not in side_clip_task_groups:
                            side_clip_task_groups.add(group_id)
                            tasks.append(
                                lambda plan=sync_plan, target=side_target: _write_window_sync_side_by_side_keyclip(
                                    plan,
                                    target,
                                )
                            )
                    elif asset_kind == KEYFRAME_DIR_NAME:
                        tasks.append(
                            lambda source_video=source_video, peak_sec=peak_sec, target=target: _extract_frame(
                                ffmpeg_path,
                                source_video,
                                max(0.0, peak_sec),
                                target,
                            )
                        )
                    else:
                        clip_start = max(0.0, peak_sec - _view_action_material_pre_context_sec())
                        duration = _view_action_material_pre_context_sec() + _view_action_material_post_context_sec()
                        tasks.append(
                            lambda source_video=source_video, clip_start=clip_start, duration=duration, target=target: _cut_video(
                                ffmpeg_path,
                                source_video,
                                clip_start,
                                duration,
                                target,
                            )
                        )
                row = _view_action_candidate_record(
                    session_root,
                    group=group,
                    evidence=evidence_for_record,
                    source_video=source_video,
                    target=target,
                    asset_kind=asset_kind,
                    exists=False,
                )
                if sync_plan:
                    _attach_window_sync_asset_plan(row, sync_plan, asset_kind)
                    sync_generation_rows.append(
                        _window_sync_keyclip_generation_row(
                            row,
                            sync_plan,
                            target,
                            asset_kind,
                        )
                    )
                else:
                    row["window_sync_generation_status"] = "missing_window_sync_plan"
                    row["quality_reasons"] = _ordered_unique_text(
                        [*(row.get("quality_reasons") or []), "missing_window_sync_plan"]
                    )
                    sync_generation_rows.append(
                        {
                            "schema_version": "window_sync_keyclip_generation.item.v1",
                            "material_id": row.get("candidate_id"),
                            "window_id": row.get("experiment_window_id"),
                            "source_window_sync_index": row.get("source_window_sync_index"),
                            "asset_kind": asset_kind,
                            "generation_status": "missing_window_sync_plan",
                            "failure_reason": "window_sync_index_missing_or_action_range_unmapped",
                        }
                    )
                _preserve_candidate_review_state(row, existing_status_by_id.get(str(row.get("candidate_id") or "")))
                candidate_rows.append(row)
    if not dry_run:
        _run_material_candidate_tasks(tasks)
        for row in candidate_rows:
            target = Path(str(row.get("stored_file") or ""))
            exists = target.is_file()
            row["exists"] = exists
            row["size_bytes"] = target.stat().st_size if exists else 0
            row["source_real"] = bool(exists and _material_file_is_real(target))
            row["placeholder"] = not bool(row.get("source_real"))
            row["missing_reason"] = None if row.get("source_real") else "view_action_candidate_file_not_real_video_material"
            if row.get("window_sync_generation_status") == "planned_from_window_sync_index":
                row["window_sync_generation_status"] = "generated_from_window_sync_index" if row.get("source_real") else "window_sync_generation_failed"
    final_sync_generation_rows = _final_window_sync_generation_rows(candidate_rows, sync_generation_rows)
    _write_json(
        metadata_dir / "window_sync_keyclip_generation_report.json",
        {
            "schema_version": "window_sync_keyclip_generation_report.v1",
            "material_count": len(final_sync_generation_rows),
            "generated_from_window_sync_index_count": sum(
                1
                for row in final_sync_generation_rows
                if row.get("generation_status") == "generated_from_window_sync_index"
            ),
            "missing_or_failed_count": sum(
                1
                for row in final_sync_generation_rows
                if str(row.get("generation_status") or "").startswith(("missing", "window_sync_generation_failed"))
            ),
            "rows": final_sync_generation_rows,
            "policy": "Final view-action keyframes/keyclips are selected from source_window_sync_index rows. Raw local seconds may only be used after matched sync rows have selected source frame indices.",
        },
    )

    summary = {
        "enabled": True,
        "source_evidence_rows": len(evidence_rows),
        "review_window_count": len(windows),
        "selected_group_count": len(selected_groups),
        "paired_group_count": len([group for group in selected_groups if group.get("cross_view_consistency") == "paired_needs_review"]),
        "single_view_group_count": len([group for group in selected_groups if group.get("cross_view_consistency") != "paired_needs_review"]),
        "candidate_count": len(candidate_rows),
        "skipped_count": len(skipped),
        "action_type_counts": _count_by_field(candidate_rows, "action_type"),
        "review_status_counts": _count_by_field(candidate_rows, "review_status"),
        "candidate_status": "needs_review",
        "official_material_count": 0,
        "memory_write_allowed": False,
    }
    return candidate_rows, skipped, summary


def _view_action_video_sources(session_root: Path) -> dict[str, Path]:
    source_rows = _read_jsonl_if_exists(session_root / "metadata" / "video_sources.jsonl")
    sources: dict[str, Path] = {}
    for row in source_rows:
        view = str(row.get("view_id") or row.get("role") or row.get("name") or "").strip()
        if view not in {"first_person", "third_person"}:
            continue
        path_value = row.get("absolute_path") or row.get("path")
        if not path_value:
            continue
        path = Path(str(path_value))
        if path.is_file():
            sources[view] = path
    return sources


def _view_action_review_windows(session_root: Path) -> list[dict[str, Any]]:
    payload = _load_json(session_root / "metadata" / "formal_experiment_windows.json")
    windows = payload.get("windows") if isinstance(payload, dict) else None
    if not isinstance(windows, list):
        return []
    reviewable: list[dict[str, Any]] = []
    for window in windows:
        if not isinstance(window, dict):
            continue
        status = str(window.get("status") or window.get("visual_review_status") or "").lower()
        if status == "formal_window_rejected":
            continue
        reviewable.append(window)
    return reviewable


def _prepare_view_action_review_groups(
    evidence_rows: list[dict[str, Any]],
    windows: list[dict[str, Any]],
    source_by_view: dict[str, Path],
) -> list[dict[str, Any]]:
    rows_by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    skipped_windowless: list[dict[str, Any]] = []
    for row in evidence_rows:
        if not isinstance(row, dict):
            continue
        view = str(row.get("view") or "").strip()
        if view not in source_by_view:
            continue
        peak_sec = _view_action_peak_sec(row)
        if peak_sec is None:
            continue
        window = _view_action_window_for_sec(windows, peak_sec)
        if window is None:
            skipped_windowless.append(row)
            continue
        enriched = dict(row)
        enriched["_review_window"] = window
        action_type = str(row.get("action_type") or row.get("canonical_action_type") or row.get("action_family") or "unknown_action")
        primary_object = canonical_yolo_label(row.get("primary_object")) or str(row.get("primary_object") or "object")
        key = (str(window.get("experiment_window_id") or "window"), action_type, primary_object)
        rows_by_key.setdefault(key, []).append(enriched)

    groups: list[dict[str, Any]] = []
    used_evidence_ids: set[str] = set()
    max_delta = _view_action_pair_window_sec()
    for key, rows in rows_by_key.items():
        first_rows = sorted([row for row in rows if row.get("view") == "first_person"], key=_view_action_evidence_rank, reverse=True)
        third_rows = sorted([row for row in rows if row.get("view") == "third_person"], key=_view_action_evidence_rank, reverse=True)
        for first in first_rows:
            first_id = str(first.get("evidence_id") or "")
            if first_id in used_evidence_ids:
                continue
            first_peak = _view_action_peak_sec(first)
            if first_peak is None:
                continue
            best: dict[str, Any] | None = None
            best_delta = max_delta + 1.0
            for third in third_rows:
                third_id = str(third.get("evidence_id") or "")
                if third_id in used_evidence_ids:
                    continue
                third_peak = _view_action_peak_sec(third)
                if third_peak is None:
                    continue
                delta = abs(first_peak - third_peak)
                if delta <= max_delta and delta < best_delta:
                    best = third
                    best_delta = delta
            if best is None:
                continue
            used_evidence_ids.add(first_id)
            used_evidence_ids.add(str(best.get("evidence_id") or ""))
            groups.append(_view_action_group(key, [first, best], "paired_needs_review", best_delta))

    single_limit = _view_action_single_group_limit()
    if single_limit > 0:
        unmatched: list[dict[str, Any]] = []
        for rows in rows_by_key.values():
            for row in rows:
                evidence_id = str(row.get("evidence_id") or "")
                if evidence_id and evidence_id not in used_evidence_ids:
                    unmatched.append(row)
        for row in sorted(unmatched, key=_view_action_evidence_rank, reverse=True)[:single_limit]:
            action_type = str(row.get("action_type") or row.get("canonical_action_type") or row.get("action_family") or "unknown_action")
            primary_object = canonical_yolo_label(row.get("primary_object")) or str(row.get("primary_object") or "object")
            window = row.get("_review_window") if isinstance(row.get("_review_window"), dict) else {}
            key = (str(window.get("experiment_window_id") or "window"), action_type, primary_object)
            groups.append(_view_action_group(key, [row], "single_view_needs_review", None))
    return groups


def _select_view_action_review_groups(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    max_groups = _view_action_group_limit()
    groups = sorted(groups, key=_view_action_group_rank, reverse=True)
    return groups[:max_groups]


def _view_action_group(
    key: tuple[str, str, str],
    rows: list[dict[str, Any]],
    consistency: str,
    delta_sec: float | None,
) -> dict[str, Any]:
    window_id, action_type, primary_object = key
    identity = "|".join([window_id, action_type, primary_object, consistency, *[str(row.get("evidence_id") or "") for row in rows]])
    digest = hashlib.sha1(identity.encode("utf-8", errors="ignore")).hexdigest()[:12]
    window = rows[0].get("_review_window") if rows and isinstance(rows[0].get("_review_window"), dict) else {}
    peak_values = [_view_action_peak_sec(row) for row in rows]
    peak_values = [value for value in peak_values if value is not None]
    return {
        "candidate_group_id": f"view_action_review_group_{digest}",
        "evidence_bundle_id": f"view_action_review_bundle_{digest}",
        "experiment_window_id": window_id,
        "unit_id": window.get("unit_id"),
        "window": window,
        "action_type": action_type,
        "canonical_action_type": rows[0].get("canonical_action_type") if rows else action_type,
        "primary_object": primary_object,
        "cross_view_consistency": consistency,
        "delta_sec": round(float(delta_sec), 3) if delta_sec is not None else None,
        "center_sec": round(sum(peak_values) / len(peak_values), 3) if peak_values else None,
        "evidence_rows": rows,
    }


def _view_action_candidate_record(
    session_root: Path,
    *,
    group: dict[str, Any],
    evidence: dict[str, Any],
    source_video: Path,
    target: Path,
    asset_kind: str,
    exists: bool,
) -> dict[str, Any]:
    evidence_id = str(evidence.get("evidence_id") or uuid.uuid4().hex)
    identity = "|".join([str(group.get("candidate_group_id") or ""), evidence_id, asset_kind])
    digest = hashlib.sha1(identity.encode("utf-8", errors="ignore")).hexdigest()[:12]
    view = str(evidence.get("view") or "").strip()
    action_type = str(group.get("action_type") or evidence.get("action_type") or "unknown_action")
    primary_object = canonical_yolo_label(group.get("primary_object") or evidence.get("primary_object")) or str(
        group.get("primary_object") or evidence.get("primary_object") or "object"
    )
    start_sec = _safe_float(evidence.get("start_sec") or evidence.get("session_start_sec") or evidence.get("peak_sec"), 0.0)
    end_sec = _safe_float(evidence.get("end_sec") or evidence.get("session_end_sec") or evidence.get("peak_sec"), start_sec)
    window = group.get("window") if isinstance(group.get("window"), dict) else {}
    global_timestamp_us = _view_action_global_timestamp_us(evidence, window)
    source_window_sync_index = _source_window_sync_index_for_material(session_root, window)
    source_real = bool(exists and _material_file_is_real(target))
    action_name = _view_action_display_name(action_type, primary_object)
    keyframe_quality = _score_view_action_keyframe(target, evidence, asset_kind=asset_kind, exists=exists)
    review_reasons = _ordered_unique_text(
        [
            "view_action_evidence_needs_human_review",
            "not_official_material",
            "not_written_to_memory",
            str(group.get("cross_view_consistency") or "needs_review"),
            *[str(item) for item in evidence.get("quality_flags") or []],
        ]
    )
    return {
        "schema_version": "material_reference.item.v1",
        "trace_schema_version": MATERIAL_REFERENCE_TRACE_SCHEMA_VERSION,
        "candidate_id": f"view_action_candidate_{digest}",
        "candidate_group_id": group.get("candidate_group_id"),
        "evidence_bundle_id": group.get("evidence_bundle_id"),
        "candidate_source": "view_action_evidence_needs_review",
        "source_event_id": evidence_id,
        "source_evidence_id": evidence_id,
        "event_type": action_type,
        "action_type": action_type,
        "physical_action_type": _view_action_physical_action_type(action_type, primary_object, evidence),
        "material_type": asset_kind,
        "asset_kind": asset_kind,
        "action_name": action_name,
        "display_title": action_name,
        "semantic_action": evidence.get("semantic_action_type") or action_type,
        "taxonomy_schema_version": MATERIAL_TAXONOMY_SCHEMA_VERSION,
        "canonical_action_type": evidence.get("canonical_action_type") or group.get("canonical_action_type") or action_type,
        "primary_object": primary_object,
        "raw_primary_object": evidence.get("primary_object"),
        "manipulated_object": primary_object,
        "objects": _ordered_unique_text([primary_object, *(evidence.get("raw_labels") or evidence.get("raw_yolo_labels") or [])]),
        "actions": _ordered_unique_text([action_type, str(evidence.get("canonical_action_type") or "")]),
        "experiment_window_id": group.get("experiment_window_id"),
        "segment_id": group.get("experiment_window_id"),
        "unit_id": group.get("unit_id"),
        "view": view,
        "camera_view": view,
        "frame_type": "peak" if asset_kind == KEYFRAME_DIR_NAME else "clip",
        "frame_role": "peak" if asset_kind == KEYFRAME_DIR_NAME else "clip",
        "start_sec": start_sec,
        "end_sec": end_sec,
        "time_start": start_sec,
        "time_end": end_sec,
        "global_timestamp_us": global_timestamp_us,
        "start_global_timestamp_us": _view_action_global_timestamp_us({"peak_sec": start_sec}, window),
        "end_global_timestamp_us": _view_action_global_timestamp_us({"peak_sec": end_sec}, window),
        "start_sync_index": window.get("start_sync_index"),
        "end_sync_index": window.get("end_sync_index"),
        "window_sync_index": source_window_sync_index,
        "source_window_sync_index": source_window_sync_index,
        "source_window_sync_index_required": True,
        "source_file": str(source_video),
        "source_clip": str(source_video),
        "source_clip_path": str(source_video),
        "source_time_basis": "window_sync_index" if source_window_sync_index else "session_sec_mapped_from_view_action_evidence",
        "final_time_basis": "window_sync_index_required_for_official",
        "stored_file": str(target),
        "stored_filename": target.name,
        "file_name": target.name,
        "exists": bool(exists),
        "size_bytes": target.stat().st_size if exists and target.is_file() else 0,
        "source_real": source_real,
        "placeholder": not source_real,
        "publishable_material": False,
        "official_material": False,
        "memory_write_allowed": False,
        "missing_reason": None if source_real else "view_action_candidate_file_not_real_video_material",
        "generated": False,
        "dry_run": False,
        "error": None,
        "quality_score": round(max(0.35, min(0.78, _safe_float(evidence.get("confidence"), 0.55))), 3),
        "selected_keyframe_score": keyframe_quality.get("selected_keyframe_score"),
        "selected_keyframe_reason": keyframe_quality.get("selected_keyframe_reason"),
        "rejected_blurry_frames_count": keyframe_quality.get("rejected_blurry_frames_count"),
        "candidate_frame_scores": keyframe_quality.get("candidate_frame_scores"),
        "best_first_keyframe": str(target) if asset_kind == KEYFRAME_DIR_NAME and view == "first_person" and source_real else None,
        "best_third_keyframe": str(target) if asset_kind == KEYFRAME_DIR_NAME and view == "third_person" and source_real else None,
        "keyframe_quality": keyframe_quality if asset_kind == KEYFRAME_DIR_NAME else None,
        "quality_bucket": "review_candidate",
        "quality_status": "needs_review",
        "quality_reasons": _ordered_unique_text(
            [
                "view_action_evidence_review_candidate",
                f"cross_view_consistency:{group.get('cross_view_consistency')}",
                f"interaction_frame_count:{int(_safe_float(evidence.get('interaction_frame_count'), 0.0))}",
                *review_reasons,
            ]
        ),
        "candidate_status": "needs_review",
        "review_status": "needs_review",
        "review_required": True,
        "review_route": "human_review",
        "review_reason_codes": review_reasons,
        "recommended": False,
        "pipeline_schema_version": "view_action_evidence_material_candidates.v1",
        "pipeline_flow": [
            "view_action_evidence",
            "candidate_keyframe_keyclip_generation",
            "frontend_review_gate",
        ],
        "pipeline_stage": "frontend_review_gate",
        "pipeline_status": "view_action_evidence_needs_review",
        "review_gate_policy": "View-action evidence candidates remain visible for review but cannot enter official material or Memory until confirmed.",
        "yolo_box_required": False,
        "yolo_annotated_required": False,
        "yolo_annotation_rendered": None,
        "box_filter": "view_action_evidence_unannotated_review_asset",
        "physical_evidence_mode": "view_action_evidence_review",
        "yolo_evidence_count": int(_safe_float(evidence.get("row_count"), 0.0)),
        "interaction_frame_count": int(_safe_float(evidence.get("interaction_frame_count"), 0.0)),
        "contact_peak_score": round(_safe_float(evidence.get("max_interaction_score"), 0.0), 3),
        "event_confidence": round(_safe_float(evidence.get("confidence"), 0.0), 3),
        "source_view_action_evidence": evidence,
        "cross_view_consistency": group.get("cross_view_consistency"),
        "cross_view_delta_sec": group.get("delta_sec"),
        "evidence_chain": {
            "schema_version": MATERIAL_REFERENCE_TRACE_SCHEMA_VERSION,
            "candidate_source": "view_action_evidence_needs_review",
            "source_evidence_id": evidence_id,
            "source_file": str(source_video),
            "camera_view": view,
            "time_start": start_sec,
            "time_end": end_sec,
            "global_timestamp_us": global_timestamp_us,
            "experiment_window_id": group.get("experiment_window_id"),
            "unit_id": group.get("unit_id"),
            "sync_index_range": [window.get("start_sync_index"), window.get("end_sync_index")],
            "source_window_sync_index": source_window_sync_index,
            "time_basis": "window_sync_index" if source_window_sync_index else "session_sec_mapped_from_view_action_evidence",
            "review_route": "human_review",
            "review_reason_codes": review_reasons,
            "candidate_disposition": "needs_review",
            "session_dir": str(session_root),
        },
    }


def _source_window_sync_index_for_material(session_root: Path, window: Mapping[str, Any]) -> str | None:
    explicit = window.get("source_window_sync_index") or window.get("window_sync_index")
    if explicit:
        path = Path(str(explicit))
        return str(path)
    window_id = str(window.get("experiment_window_id") or window.get("window_id") or "").strip()
    if not window_id:
        return None
    experiment_root = session_root.parent if session_root.name == "key_action_index" else session_root
    candidate = experiment_root / "windows" / window_id / "window_sync_index.csv"
    return str(candidate)


def _build_window_sync_asset_plan(
    session_root: Path,
    group: Mapping[str, Any],
    evidence: Mapping[str, Any],
    asset_kind: str,
    source_by_view: Mapping[str, Path] | None = None,
) -> dict[str, Any] | None:
    window = group.get("window") if isinstance(group.get("window"), Mapping) else {}
    source_window_sync_index = _source_window_sync_index_for_material(session_root, window)
    if not source_window_sync_index:
        return None
    sync_path = Path(str(source_window_sync_index))
    sync_rows = _read_window_sync_index_rows(sync_path)
    if not sync_rows:
        return None
    start_sec, end_sec, peak_sec = _view_action_sync_time_range(group, evidence, window, asset_kind)
    selected_rows = _select_window_sync_rows_for_session_range(sync_rows, window, start_sec, end_sec, peak_sec)
    if not selected_rows:
        return None
    peak_row = min(
        selected_rows,
        key=lambda row: abs(_window_sync_row_session_sec(row, sync_rows, window) - peak_sec),
    )
    start_row = selected_rows[0]
    end_row = selected_rows[-1]
    # Per-window CSVs normally carry first/third_video_path columns, but the
    # video sources resolved from metadata/video_sources.jsonl are the
    # authoritative fallback when those columns are blank or absent. Surfacing
    # them on the plan lets keyframe/keyclip extraction recover the source
    # video without re-reading the CSV.
    view_video_paths = {
        view: str(path)
        for view, path in (source_by_view or {}).items()
        if path is not None
    }
    return {
        "schema_version": "window_sync_asset_plan.v1",
        "window_id": str(window.get("experiment_window_id") or window.get("window_id") or ""),
        "source_window_sync_index": str(sync_path),
        "asset_kind": asset_kind,
        "action_start_sec": round(start_sec, 6),
        "action_end_sec": round(end_sec, 6),
        "action_peak_sec": round(peak_sec, 6),
        "action_start_global_timestamp_us": _sync_row_global_timestamp(start_row),
        "action_end_global_timestamp_us": _sync_row_global_timestamp(end_row),
        "action_peak_global_timestamp_us": _sync_row_global_timestamp(peak_row),
        "start_window_sync_index": _sync_row_index(start_row),
        "end_window_sync_index": _sync_row_index(end_row),
        "peak_window_sync_index": _sync_row_index(peak_row),
        "rows": selected_rows,
        "start_row": start_row,
        "end_row": end_row,
        "peak_row": peak_row,
        "view_video_paths": view_video_paths,
        "output_fps": _window_sync_keyclip_output_fps(sync_rows),
        "reference_camera": _sync_row_value(peak_row, "reference_camera"),
        "row_count": len(selected_rows),
    }


def _attach_window_sync_asset_plan(row: dict[str, Any], plan: Mapping[str, Any], asset_kind: str) -> None:
    row["source_window_sync_index"] = plan.get("source_window_sync_index")
    row["window_sync_index"] = plan.get("source_window_sync_index")
    row["source_time_basis"] = "window_sync_index_row_range"
    row["final_time_basis"] = "window_sync_index_row_range"
    row["window_sync_generation_status"] = "planned_from_window_sync_index"
    row["start_window_sync_index"] = plan.get("start_window_sync_index")
    row["end_window_sync_index"] = plan.get("end_window_sync_index")
    row["peak_window_sync_index"] = plan.get("peak_window_sync_index")
    row["window_sync_pair_count"] = plan.get("row_count")
    row["window_sync_reference_camera"] = plan.get("reference_camera")
    row["keyclip_output_fps"] = plan.get("output_fps") if asset_kind == KEY_CLIP_DIR_NAME else None
    if plan.get("side_by_side_keyclip"):
        row["side_by_side_keyclip"] = plan.get("side_by_side_keyclip")
    row["start_global_timestamp_us"] = plan.get("action_start_global_timestamp_us")
    row["end_global_timestamp_us"] = plan.get("action_end_global_timestamp_us")
    row["global_timestamp_us"] = plan.get("action_peak_global_timestamp_us")
    row["quality_reasons"] = _ordered_unique_text(
        [*(row.get("quality_reasons") or []), "generated_from_window_sync_index_row_range"]
    )
    chain = row.get("evidence_chain")
    if isinstance(chain, dict):
        chain["time_basis"] = "window_sync_index_row_range"
        chain["source_window_sync_index"] = plan.get("source_window_sync_index")
        chain["window_sync_index_range"] = [plan.get("start_window_sync_index"), plan.get("end_window_sync_index")]
        chain["peak_window_sync_index"] = plan.get("peak_window_sync_index")


def _window_sync_keyclip_generation_row(
    row: Mapping[str, Any],
    plan: Mapping[str, Any],
    target: Path,
    asset_kind: str,
) -> dict[str, Any]:
    return {
        "schema_version": "window_sync_keyclip_generation.item.v1",
        "material_id": row.get("candidate_id"),
        "evidence_bundle_id": row.get("evidence_bundle_id"),
        "asset_kind": asset_kind,
        "window_id": plan.get("window_id"),
        "source_window_sync_index": plan.get("source_window_sync_index"),
        "action_start_global_timestamp_us": plan.get("action_start_global_timestamp_us"),
        "action_end_global_timestamp_us": plan.get("action_end_global_timestamp_us"),
        "start_window_sync_index": plan.get("start_window_sync_index"),
        "end_window_sync_index": plan.get("end_window_sync_index"),
        "peak_window_sync_index": plan.get("peak_window_sync_index"),
        "first_keyclip": str(target) if asset_kind == KEY_CLIP_DIR_NAME and row.get("view") == "first_person" else None,
        "third_keyclip": str(target) if asset_kind == KEY_CLIP_DIR_NAME and row.get("view") == "third_person" else None,
        "side_by_side_keyclip": plan.get("side_by_side_keyclip"),
        "generation_status": "planned_from_window_sync_index",
        "failure_reason": None,
    }


def _final_window_sync_generation_rows(
    candidate_rows: list[dict[str, Any]],
    planned_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {str(row.get("candidate_id") or row.get("material_id") or ""): row for row in candidate_rows}
    final_rows: list[dict[str, Any]] = []
    for planned in planned_rows:
        material_id = str(planned.get("material_id") or "")
        candidate = by_id.get(material_id, {})
        target = Path(str(candidate.get("stored_file") or ""))
        status = str(candidate.get("window_sync_generation_status") or planned.get("generation_status") or "")
        if status == "planned_from_window_sync_index":
            status = "generated_from_window_sync_index" if target.is_file() and bool(candidate.get("source_real")) else "window_sync_generation_failed"
        final = dict(planned)
        final["generation_status"] = status
        final["failure_reason"] = None if status == "generated_from_window_sync_index" else (candidate.get("missing_reason") or planned.get("failure_reason"))
        final["output_file"] = str(target) if target else planned.get("output_file")
        final["output_exists"] = target.is_file() if target else False
        final_rows.append(final)
    return final_rows


def _read_window_sync_index_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except Exception:
        return []


def _view_action_sync_time_range(
    group: Mapping[str, Any],
    evidence: Mapping[str, Any],
    window: Mapping[str, Any],
    asset_kind: str,
) -> tuple[float, float, float]:
    peak = _safe_float(group.get("center_sec"), _safe_float(_view_action_peak_sec(evidence), 0.0))
    evidence_rows = [row for row in group.get("evidence_rows") or [] if isinstance(row, Mapping)]
    start_values = [
        _safe_float(row.get("start_sec") or row.get("session_start_sec") or row.get("peak_sec"), peak)
        for row in evidence_rows
    ] or [_safe_float(evidence.get("start_sec") or evidence.get("session_start_sec") or evidence.get("peak_sec"), peak)]
    end_values = [
        _safe_float(row.get("end_sec") or row.get("session_end_sec") or row.get("peak_sec"), peak)
        for row in evidence_rows
    ] or [_safe_float(evidence.get("end_sec") or evidence.get("session_end_sec") or evidence.get("peak_sec"), peak)]
    window_start = _safe_float(window.get("start_sec"), min(start_values + [peak]))
    window_end = _safe_float(window.get("end_sec"), max(end_values + [peak]))
    pre = _view_action_material_pre_context_sec()
    post = _view_action_material_post_context_sec()
    if asset_kind == KEYFRAME_DIR_NAME:
        start = peak
        end = peak
    else:
        start = min(start_values + [peak - pre])
        end = max(end_values + [peak + post])
        if end <= start:
            start = peak - pre
            end = peak + post
    start = max(window_start, float(start))
    end = min(window_end, float(end))
    if asset_kind == KEY_CLIP_DIR_NAME and end - start < 0.5:
        start = max(window_start, peak - 0.5)
        end = min(window_end, peak + 0.5)
    return float(start), float(max(end, start)), float(peak)


def _select_window_sync_rows_for_session_range(
    sync_rows: list[dict[str, Any]],
    window: Mapping[str, Any],
    start_sec: float,
    end_sec: float,
    peak_sec: float,
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in sync_rows
        if start_sec <= _window_sync_row_session_sec(row, sync_rows, window) <= end_sec
        and _sync_row_valid(row)
    ]
    if selected:
        return selected
    valid_rows = [row for row in sync_rows if _sync_row_valid(row)]
    if not valid_rows:
        return []
    nearest_index = min(
        range(len(valid_rows)),
        key=lambda idx: abs(_window_sync_row_session_sec(valid_rows[idx], sync_rows, window) - peak_sec),
    )
    half_window = max(1, int(round(_window_sync_keyclip_output_fps(sync_rows) * 0.5)))
    start = max(0, nearest_index - half_window)
    end = min(len(valid_rows), nearest_index + half_window + 1)
    return valid_rows[start:end]


def _window_sync_row_session_sec(
    row: Mapping[str, Any],
    sync_rows: list[dict[str, Any]],
    window: Mapping[str, Any],
) -> float:
    window_start = _safe_float(window.get("start_sec"), 0.0)
    first_global = _sync_row_global_timestamp(sync_rows[0]) if sync_rows else None
    current = _sync_row_global_timestamp(row)
    if first_global is None or current is None:
        local_pts = _safe_float(row.get("local_pts_first") or row.get("local_pts_third"), 0.0)
        return window_start + local_pts
    return window_start + (float(current) - float(first_global)) / 1_000_000.0


def _sync_row_valid(row: Mapping[str, Any]) -> bool:
    first_valid = str(row.get("first_valid") or "true").lower() not in {"false", "0", "no"}
    third_valid = str(row.get("third_valid") or "true").lower() not in {"false", "0", "no"}
    return first_valid and third_valid and _sync_row_int(row, "first_frame_index") is not None and _sync_row_int(row, "third_frame_index") is not None


def _sync_row_global_timestamp(row: Mapping[str, Any]) -> int | None:
    return _sync_row_int(row, "global_timestamp_us")


def _sync_row_index(row: Mapping[str, Any]) -> int | None:
    return _sync_row_int(row, "window_sync_index") or _sync_row_int(row, "sync_index") or _sync_row_int(row, "source_sync_index")


def _sync_row_int(row: Mapping[str, Any], key: str) -> int | None:
    try:
        value = row.get(key)
        if value is None or value == "":
            return None
        return int(float(str(value)))
    except Exception:
        return None


def _sync_row_value(row: Mapping[str, Any], key: str) -> str | None:
    value = row.get(key)
    return str(value) if value is not None and str(value) else None


def _window_sync_keyclip_output_fps(sync_rows: list[dict[str, Any]]) -> float:
    configured = _safe_float(os.environ.get("KEY_ACTION_KEYCLIP_OUTPUT_FPS"), 15.0)
    minimum = _safe_float(os.environ.get("KEY_ACTION_MIN_KEYCLIP_OUTPUT_FPS"), 15.0)
    return max(1.0, max(minimum, configured))


def _extract_window_sync_keyframe(plan: Mapping[str, Any], view: str, target: Path) -> None:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("opencv-python is required for window_sync keyframes") from exc
    row = plan.get("peak_row") if isinstance(plan.get("peak_row"), Mapping) else {}
    video_path = _window_sync_video_path(row, view, plan)
    frame_index = _window_sync_frame_index(row, view)
    if not video_path or frame_index is None:
        raise RuntimeError(f"window_sync_keyframe_missing_source:{view}")
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"cannot_open_window_sync_video:{video_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"cannot_read_window_sync_frame:{video_path}:{frame_index}")
        target.parent.mkdir(parents=True, exist_ok=True)
        ok, encoded = cv2.imencode(target.suffix if target.suffix else ".jpg", frame)
        if not ok:
            raise RuntimeError(f"cannot_encode_window_sync_keyframe:{target}")
        encoded.tofile(str(target))
    finally:
        cap.release()


def _write_window_sync_keyclip(plan: Mapping[str, Any], view: str, target: Path) -> None:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("opencv-python is required for window_sync keyclips") from exc
    rows = [row for row in plan.get("rows") or [] if isinstance(row, Mapping)]
    if not rows:
        raise RuntimeError("window_sync_keyclip_empty_rows")
    video_path = _window_sync_video_path(rows[0], view, plan)
    if not video_path:
        raise RuntimeError(f"window_sync_keyclip_missing_video:{view}")
    output_fps = _safe_float(plan.get("output_fps"), 15.0)
    cap = cv2.VideoCapture(str(video_path))
    writer = None
    frame_count = 0
    try:
        if not cap.isOpened():
            raise RuntimeError(f"cannot_open_window_sync_video:{video_path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        for idx, row in enumerate(rows):
            frame = _read_window_sync_frame(cap, row, view)
            if frame is None:
                continue
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f"cannot_open_window_sync_keyclip_writer:{target}")
            repeat = _window_sync_repeat_count(rows, idx, output_fps)
            for _ in range(repeat):
                writer.write(frame)
                frame_count += 1
        if writer is not None:
            writer.release()
            writer = None
        if frame_count <= 0:
            raise RuntimeError("window_sync_keyclip_no_frames_written")
        _transcode_rendered_clip_for_browser(target)
    finally:
        if writer is not None:
            writer.release()
        cap.release()


def _write_window_sync_side_by_side_keyclip(plan: Mapping[str, Any], target: Path) -> None:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("opencv-python is required for window_sync side-by-side keyclips") from exc
    rows = [row for row in plan.get("rows") or [] if isinstance(row, Mapping)]
    if not rows:
        raise RuntimeError("window_sync_side_by_side_empty_rows")
    third_video = _window_sync_video_path(rows[0], "third_person", plan)
    first_video = _window_sync_video_path(rows[0], "first_person", plan)
    if not third_video or not first_video:
        raise RuntimeError("window_sync_side_by_side_missing_video")
    output_fps = _safe_float(plan.get("output_fps"), 15.0)
    third_cap = cv2.VideoCapture(str(third_video))
    first_cap = cv2.VideoCapture(str(first_video))
    writer = None
    frame_count = 0
    try:
        if not third_cap.isOpened() or not first_cap.isOpened():
            raise RuntimeError("cannot_open_window_sync_side_by_side_source")
        target.parent.mkdir(parents=True, exist_ok=True)
        for idx, row in enumerate(rows):
            third_frame = _read_window_sync_frame(third_cap, row, "third_person")
            first_frame = _read_window_sync_frame(first_cap, row, "first_person")
            if third_frame is None or first_frame is None:
                continue
            composed = _compose_window_sync_side_by_side(third_frame, first_frame)
            if writer is None:
                height, width = composed.shape[:2]
                writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f"cannot_open_window_sync_side_by_side_writer:{target}")
            repeat = _window_sync_repeat_count(rows, idx, output_fps)
            for _ in range(repeat):
                writer.write(composed)
                frame_count += 1
        if writer is not None:
            writer.release()
            writer = None
        if frame_count <= 0:
            raise RuntimeError("window_sync_side_by_side_no_frames_written")
        _transcode_rendered_clip_for_browser(target)
    finally:
        if writer is not None:
            writer.release()
        third_cap.release()
        first_cap.release()


def _window_sync_video_path(
    row: Mapping[str, Any], view: str, plan: Mapping[str, Any] | None = None
) -> Path | None:
    key = "first_video_path" if view == "first_person" else "third_video_path"
    value = row.get(key)
    if value:
        path = Path(str(value))
        if path.is_file():
            return path
    # Fall back to the view video paths resolved from video_sources.jsonl when
    # the window_sync row has no usable path column (e.g. minimal CSVs or stale
    # absolute paths). This keeps keyframe/keyclip extraction working off the
    # authoritative source list rather than failing with missing_source.
    if plan is not None:
        view_video_paths = plan.get("view_video_paths")
        if isinstance(view_video_paths, Mapping):
            fallback = view_video_paths.get(view)
            if fallback:
                fallback_path = Path(str(fallback))
                if fallback_path.is_file():
                    return fallback_path
    return None


def _window_sync_frame_index(row: Mapping[str, Any], view: str) -> int | None:
    key = "first_frame_index" if view == "first_person" else "third_frame_index"
    return _sync_row_int(row, key)


def _read_window_sync_frame(cap: Any, row: Mapping[str, Any], view: str) -> Any | None:
    frame_index = _window_sync_frame_index(row, view)
    if frame_index is None:
        return None
    cap.set(1, int(frame_index))
    ok, frame = cap.read()
    return frame if ok else None


def _window_sync_repeat_count(rows: list[Mapping[str, Any]], index: int, output_fps: float) -> int:
    current = _sync_row_global_timestamp(rows[index])
    next_ts = _sync_row_global_timestamp(rows[index + 1]) if index + 1 < len(rows) else None
    if current is None or next_ts is None or next_ts <= current:
        return 1
    return max(1, int(round(((next_ts - current) / 1_000_000.0) * max(output_fps, 0.001))))


def _video_duration_metadata(path: Path) -> dict[str, float]:
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
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        duration_s = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
        return {"fps": fps, "frame_count": frame_count, "duration_s": duration_s}
    finally:
        cap.release()


def _compose_window_sync_side_by_side(third_frame: Any, first_frame: Any) -> Any:
    import cv2

    max_height = 720
    target_height = min(max_height, max(int(third_frame.shape[0]), int(first_frame.shape[0])))

    def _resize(frame: Any) -> Any:
        h, w = frame.shape[:2]
        if h == target_height:
            return frame
        target_width = max(1, int(round(w * (target_height / max(1, h)))))
        return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

    return cv2.hconcat([_resize(third_frame), _resize(first_frame)])


def _score_view_action_keyframe(
    path: Path,
    evidence: Mapping[str, Any],
    *,
    asset_kind: str,
    exists: bool,
) -> dict[str, Any]:
    if asset_kind != KEYFRAME_DIR_NAME:
        return {
            "schema_version": "keyframe_quality.v1",
            "selected_keyframe_score": None,
            "selected_keyframe_reason": "not_a_keyframe_asset",
            "candidate_frame_scores": [],
            "rejected_blurry_frames_count": 0,
            "limitations": ["quality_scoring_only_applies_to_keyframe_assets"],
        }
    detection_confidence = max(0.0, min(1.0, _safe_float(evidence.get("confidence"), 0.0)))
    bbox_visibility = max(0.0, min(1.0, _safe_float(evidence.get("interaction_frame_count"), 0.0) / 3.0))
    action_peak = max(0.0, min(1.0, _safe_float(evidence.get("max_interaction_score"), detection_confidence)))
    sharpness = _image_sharpness_score(path) if exists and path.is_file() else 0.0
    bbox_stability = 0.5
    score = (
        0.35 * detection_confidence
        + 0.25 * bbox_visibility
        + 0.20 * sharpness
        + 0.10 * bbox_stability
        + 0.10 * action_peak
    )
    reasons = [
        f"detection_confidence={detection_confidence:.3f}",
        f"bbox_visibility={bbox_visibility:.3f}",
        f"sharpness={sharpness:.3f}",
        f"action_peak={action_peak:.3f}",
    ]
    blurry = 1 if sharpness < _keyframe_blurry_threshold() and exists else 0
    return {
        "schema_version": "keyframe_quality.v1",
        "selected_keyframe_score": round(max(0.0, min(1.0, score)), 6),
        "selected_keyframe_reason": "; ".join(reasons),
        "candidate_frame_scores": [
            {
                "path": str(path),
                "detection_confidence": round(detection_confidence, 6),
                "bbox_visibility_score": round(bbox_visibility, 6),
                "sharpness_score": round(sharpness, 6),
                "bbox_stability_score": bbox_stability,
                "action_peak_score": round(action_peak, 6),
                "frame_score": round(max(0.0, min(1.0, score)), 6),
            }
        ],
        "rejected_blurry_frames_count": blurry,
        "limitations": ["bbox_stability_score_defaulted"] if bbox_stability == 0.5 else [],
    }


def _keyframe_blurry_threshold() -> float:
    return max(0.0, min(1.0, _safe_float(os.environ.get("KEY_ACTION_KEYFRAME_BLURRY_THRESHOLD"), 0.12)))


def _image_sharpness_score(path: Path) -> float:
    try:
        import cv2
        import numpy as np

        # cv2.imread cannot open non-ASCII (e.g. Chinese) paths on Windows and
        # returns None; decode from a byte buffer to support Unicode paths.
        buffer = np.fromfile(str(path), dtype=np.uint8)
        if buffer.size == 0:
            return 0.5
        image = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return 0.5
        variance = float(cv2.Laplacian(image, cv2.CV_64F).var())
        return max(0.0, min(1.0, variance / 800.0))
    except Exception:
        return 0.5


def _refresh_candidate_keyframe_quality(candidate_rows: list[dict[str, Any]], report_path: Path) -> dict[str, Any]:
    keyframe_rows = [
        row
        for row in candidate_rows
        if str(row.get("asset_kind") or row.get("material_type") or "") == KEYFRAME_DIR_NAME
    ]
    quality_rows: list[dict[str, Any]] = []
    for row in keyframe_rows:
        stored = Path(str(row.get("stored_file") or ""))
        evidence = row.get("source_view_action_evidence") if isinstance(row.get("source_view_action_evidence"), Mapping) else row
        quality = _score_view_action_keyframe(
            stored,
            evidence,
            asset_kind=KEYFRAME_DIR_NAME,
            exists=stored.is_file(),
        )
        row["selected_keyframe_score"] = quality.get("selected_keyframe_score")
        row["selected_keyframe_reason"] = quality.get("selected_keyframe_reason")
        row["rejected_blurry_frames_count"] = quality.get("rejected_blurry_frames_count")
        row["candidate_frame_scores"] = quality.get("candidate_frame_scores")
        row["keyframe_quality"] = quality
        if str(row.get("view") or "") == "first_person" and stored.is_file():
            row["best_first_keyframe"] = str(stored)
        if str(row.get("view") or "") == "third_person" and stored.is_file():
            row["best_third_keyframe"] = str(stored)
        quality_rows.append(
            {
                "material_id": row.get("material_id") or row.get("candidate_id"),
                "candidate_id": row.get("candidate_id"),
                "candidate_group_id": row.get("candidate_group_id"),
                "view": row.get("view"),
                "stored_file": row.get("stored_file"),
                "selected_keyframe_score": row.get("selected_keyframe_score"),
                "selected_keyframe_reason": row.get("selected_keyframe_reason"),
                "rejected_blurry_frames_count": row.get("rejected_blurry_frames_count"),
                "candidate_frame_scores": row.get("candidate_frame_scores"),
                "source_window_sync_index": row.get("source_window_sync_index"),
            }
        )
    report = {
        "schema_version": "keyframe_quality_report.v1",
        "keyframe_count": len(keyframe_rows),
        "scored_keyframe_count": len(quality_rows),
        "blurry_frame_count": sum(int(row.get("rejected_blurry_frames_count") or 0) for row in quality_rows),
        "rows": quality_rows,
        "policy": "Keyframes are scored for detection confidence, bbox visibility, sharpness, and action peak before frontend/CLI review.",
    }
    _write_json(report_path, report)
    return {
        "keyframe_count": report["keyframe_count"],
        "scored_keyframe_count": report["scored_keyframe_count"],
        "blurry_frame_count": report["blurry_frame_count"],
        "report_path": str(report_path),
    }


def _view_action_window_for_sec(windows: list[dict[str, Any]], sec: float) -> dict[str, Any] | None:
    for window in windows:
        start = _safe_float(window.get("start_sec"), float("-inf"))
        end = _safe_float(window.get("end_sec"), float("inf"))
        if start <= sec <= end:
            return window
    return None


def _view_action_peak_sec(row: Mapping[str, Any]) -> float | None:
    for key in ("peak_sec", "peak_session_sec", "center_sec", "session_start_sec", "start_sec"):
        value = row.get(key)
        if value not in (None, ""):
            return _safe_float(value)
    return None


def _view_action_global_timestamp_us(evidence: Mapping[str, Any], window: Mapping[str, Any]) -> int | None:
    peak = _view_action_peak_sec(evidence)
    if peak is None:
        return None
    start_sec = _safe_float(window.get("start_sec"), peak)
    start_global = window.get("start_global_timestamp_us")
    if start_global in (None, ""):
        return None
    return int(_safe_float(start_global) + max(0.0, peak - start_sec) * 1_000_000.0)


def _view_action_evidence_rank(row: Mapping[str, Any]) -> tuple[float, float, float, float]:
    return (
        _safe_float(row.get("interaction_frame_count"), 0.0),
        _safe_float(row.get("confidence"), 0.0),
        _safe_float(row.get("row_count"), 0.0),
        _safe_float(row.get("duration_sec"), 0.0),
    )


def _view_action_group_rank(group: Mapping[str, Any]) -> tuple[int, float, float, float]:
    rows = [row for row in group.get("evidence_rows") or [] if isinstance(row, dict)]
    paired = 1 if str(group.get("cross_view_consistency") or "") == "paired_needs_review" else 0
    interaction = sum(_safe_float(row.get("interaction_frame_count"), 0.0) for row in rows)
    confidence = sum(_safe_float(row.get("confidence"), 0.0) for row in rows) / max(1, len(rows))
    delta = _safe_float(group.get("delta_sec"), 999.0)
    return (paired, interaction, confidence, -delta)


def _view_action_group_limit() -> int:
    return max(1, int(_safe_float(os.environ.get("KEY_ACTION_VIEW_ACTION_REVIEW_GROUP_LIMIT"), 12)))


def _view_action_single_group_limit() -> int:
    return max(0, int(_safe_float(os.environ.get("KEY_ACTION_VIEW_ACTION_REVIEW_SINGLE_GROUP_LIMIT"), 6)))


def _view_action_pair_window_sec() -> float:
    return max(0.25, _safe_float(os.environ.get("KEY_ACTION_VIEW_ACTION_REVIEW_PAIR_WINDOW_SEC"), 3.0))


def _view_action_material_pre_context_sec() -> float:
    return max(0.0, _safe_float(os.environ.get("KEY_ACTION_REVIEW_MATERIAL_PRE_CONTEXT_SEC"), 1.0))


def _view_action_material_post_context_sec() -> float:
    return max(0.5, _safe_float(os.environ.get("KEY_ACTION_REVIEW_MATERIAL_POST_CONTEXT_SEC"), 1.5))


def _view_action_display_name(action_type: str, primary_object: str) -> str:
    action = str(action_type or "").lower()
    primary = canonical_yolo_label(primary_object) or str(primary_object or "").lower()
    if "balance" in action or primary in {"balance", "scale"}:
        return "天平设备面板操作"
    if "paper" in action or primary in {"paper", "weighing_paper"}:
        return "手部与称量纸操作"
    if "pipette" in action or primary in {"pipette", "pipette_tip"}:
        return "手部与移液枪操作"
    if "bottle" in action or primary in {"reagent_bottle", "sample_bottle", "sample_bottle_blue", "bottle_cap", "tube_cap"}:
        return "手部与试剂瓶操作"
    if "container" in action or primary in {"container", "beaker", "tube"}:
        return "手部与容器操作"
    if "move" in action or "movement" in action:
        return "物体移动"
    return "手部与物体操作"


def _view_action_physical_action_type(action_type: str, primary_object: str, evidence: Mapping[str, Any]) -> str:
    text = " ".join(
        [
            str(action_type or ""),
            str(primary_object or ""),
            str(evidence.get("interaction_type") or ""),
            str(evidence.get("semantic_action_type") or ""),
        ]
    ).lower()
    if "panel" in text or "balance" in text or "scale" in text:
        return "device_panel_interaction"
    if "move" in text or "movement" in text:
        return "object_move"
    return "hand_object_contact"


def _event_surface_fields(event: dict[str, Any], micro_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    event_type = str(event.get("event_type") or event.get("evidence_type") or "")
    primary_object = canonical_yolo_label(event.get("primary_object") or event.get("object_label")) or str(
        event.get("primary_object") or event.get("object_label") or "object"
    )
    physical_action_type = _event_physical_action_type(event_type)
    contact_valid_count = _event_valid_contact_evidence_count(event, primary_object, micro_by_id)
    contact_usable_count = _event_usable_contact_evidence_count(event, primary_object, micro_by_id)
    contact_peak_score = _event_contact_peak_score(event, primary_object, micro_by_id)
    gate_reasons = _event_action_gate_reasons(
        event,
        event_type=event_type,
        physical_action_type=physical_action_type,
        primary_object=primary_object,
        contact_valid_count=contact_valid_count,
        contact_usable_count=contact_usable_count,
        contact_peak_score=contact_peak_score,
    )
    quality_bucket = "low_quality" if gate_reasons else "review_candidate"
    return {
        "event_type": event_type,
        "source_event_id": event.get("video_event_id") or event.get("evidence_id"),
        "physical_action_type": physical_action_type or "unmapped",
        "primary_object": primary_object,
        "quality_bucket": quality_bucket,
        "gate_reasons": gate_reasons,
        "surface_rank": _event_surface_rank(
            event,
            quality_bucket=quality_bucket,
            contact_valid_count=contact_valid_count,
            contact_usable_count=contact_usable_count,
        ),
    }


def _event_surface_rank(
    event: dict[str, Any],
    *,
    quality_bucket: str,
    contact_valid_count: int,
    contact_usable_count: int,
) -> tuple[int, float, int, int, int]:
    event_type = str(event.get("event_type") or event.get("evidence_type") or "")
    physical_action = _event_physical_action_type(event_type)
    action_priority = {
        "hand_object_contact": 90,
        "liquid_movement": 80,
        "equipment_panel_operation": 70,
        "container_state_change": 60,
        "object_movement": 50,
    }.get(physical_action, 0)
    if quality_bucket == "low_quality":
        action_priority -= 40
    source_priority = 1 if str(event.get("event_candidate_source") or "") == "video_understanding_event" else 0
    if event_type == "object_trajectory_movement":
        source_priority += 1
    confidence = _safe_float(event.get("confidence"), 0.0)
    asset_count = len([item for item in event.get("asset_refs") or [] if isinstance(item, dict)])
    evidence_score = contact_valid_count * 3 + contact_usable_count
    if _event_has_weak_visual_quality(event) or _event_has_negative_visual_confirmation(event):
        evidence_score -= 2
    return (action_priority, confidence, evidence_score, source_priority, asset_count)


def _select_event_backed_surface_events(prepared_events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    surfaced: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    review_total = 0
    low_total = 0
    review_by_action: dict[str, int] = {}
    low_by_action: dict[str, int] = {}
    review_by_object: dict[tuple[str, str], int] = {}
    low_by_object: dict[tuple[str, str], int] = {}

    for item in sorted(prepared_events, key=lambda value: value.get("surface_rank") or (), reverse=True):
        bucket = str(item.get("quality_bucket") or "low_quality")
        action = str(item.get("physical_action_type") or "unmapped")
        primary = str(item.get("primary_object") or "object")
        object_key = (action, primary)
        blocked_reasons = sorted(
            {
                str(reason)
                for reason in item.get("gate_reasons") or []
                if str(reason) in EVENT_BACKED_SUPPRESS_GATE_REASONS
            }
        )
        if blocked_reasons:
            suppressed.append(
                {
                    **item,
                    "suppression_reason": "event_backed_gate_blocked",
                    "suppression_gate_reasons": blocked_reasons,
                }
            )
            continue
        if bucket == "low_quality":
            action_limit = EVENT_BACKED_LOW_QUALITY_GROUP_LIMIT_BY_ACTION.get(action, 2)
            if low_total >= EVENT_BACKED_LOW_QUALITY_GROUP_LIMIT_TOTAL:
                suppressed.append({**item, "suppression_reason": "event_backed_low_quality_total_limit_exceeded"})
                continue
            if low_by_action.get(action, 0) >= action_limit:
                suppressed.append({**item, "suppression_reason": "event_backed_low_quality_action_limit_exceeded"})
                continue
            if low_by_object.get(object_key, 0) >= EVENT_BACKED_LOW_QUALITY_GROUP_LIMIT_PER_OBJECT:
                suppressed.append({**item, "suppression_reason": "event_backed_low_quality_object_limit_exceeded"})
                continue
            low_total += 1
            low_by_action[action] = low_by_action.get(action, 0) + 1
            low_by_object[object_key] = low_by_object.get(object_key, 0) + 1
            surfaced.append(item)
            continue

        action_limit = EVENT_BACKED_REVIEW_GROUP_LIMIT_BY_ACTION.get(action, 4)
        if review_total >= EVENT_BACKED_REVIEW_GROUP_LIMIT_TOTAL:
            suppressed.append({**item, "suppression_reason": "event_backed_review_total_limit_exceeded"})
            continue
        if review_by_action.get(action, 0) >= action_limit:
            suppressed.append({**item, "suppression_reason": "event_backed_review_action_limit_exceeded"})
            continue
        if review_by_object.get(object_key, 0) >= EVENT_BACKED_REVIEW_GROUP_LIMIT_PER_OBJECT:
            suppressed.append({**item, "suppression_reason": "event_backed_review_object_limit_exceeded"})
            continue
        review_total += 1
        review_by_action[action] = review_by_action.get(action, 0) + 1
        review_by_object[object_key] = review_by_object.get(object_key, 0) + 1
        surfaced.append(item)
    return surfaced, suppressed


def _event_backed_suppressed_records(suppressed_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in suppressed_events:
        records.append(
            {
                "source_event_id": item.get("source_event_id"),
                "event_type": item.get("event_type"),
                "physical_action_type": item.get("physical_action_type"),
                "primary_object": item.get("primary_object"),
                "quality_bucket": item.get("quality_bucket"),
                "reason": item.get("suppression_reason") or "event_backed_surface_limit_exceeded",
                "gate_reasons": item.get("gate_reasons") or [],
                "suppression_gate_reasons": item.get("suppression_gate_reasons") or [],
            }
        )
    return records


def _count_prepared_events_by(items: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        value = str(item.get(key) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return counts


def _event_backed_source_events(video_rows: list[dict[str, Any]], advanced_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    seen: set[str] = set()
    enabled_event_types = _active_event_backed_candidate_types()
    for row in video_rows:
        event_type = str(row.get("event_type") or "")
        if event_type not in enabled_event_types:
            continue
        if not _event_physical_action_type(event_type):
            continue
        if event_type == "experiment_action_classification":
            continue
        key = str(row.get("video_event_id") or f"video:{len(events)}")
        if key in seen:
            continue
        seen.add(key)
        events.append({**row, "event_candidate_source": "video_understanding_event"})
    for row in advanced_rows:
        evidence_type = str(row.get("evidence_type") or "")
        if evidence_type not in enabled_event_types:
            continue
        if not _event_physical_action_type(evidence_type):
            continue
        key = str(row.get("evidence_id") or f"advanced:{len(events)}")
        if key in seen:
            continue
        seen.add(key)
        events.append(
            {
                **row,
                "video_event_id": key,
                "event_type": evidence_type,
                "primary_object": row.get("object_label") or row.get("primary_object"),
                "anomaly_flags": _ordered_unique_text(
                    [
                        *(_as_list_for_semantics(row.get("anomaly_flags"))),
                        "requires_human_confirmation" if row.get("requires_human_confirmation") else "",
                        "visual_confirmation_limited"
                        if str(row.get("confirmation_level") or row.get("visual_confirmation_level") or "").lower().startswith("candidate")
                        else "",
                    ]
                ),
                "confidence_reasons": _as_list_for_semantics(row.get("confidence_reasons") or row.get("evidence_reasons")),
                "event_candidate_source": "advanced_vision_evidence",
            }
        )
    return _dedupe_event_backed_source_events(events)


def _dedupe_event_backed_source_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[tuple[str, str, str], dict[str, Any]] = {}
    order: list[tuple[str, str, str]] = []
    for event in events:
        key = _event_dedupe_key(event)
        if key not in selected:
            selected[key] = event
            order.append(key)
            continue
        if _event_selection_rank(event) > _event_selection_rank(selected[key]):
            selected[key] = event
    return [selected[key] for key in order if key in selected]


def _event_dedupe_key(event: dict[str, Any]) -> tuple[str, str, str]:
    event_type = str(event.get("event_type") or event.get("evidence_type") or "")
    micro_id = str(event.get("micro_segment_id") or event.get("segment_id") or event.get("video_event_id") or event.get("evidence_id") or "")
    primary = canonical_yolo_label(event.get("primary_object") or event.get("object_label")) or str(
        event.get("primary_object") or event.get("object_label") or "object"
    )
    physical_action = _event_physical_action_type(event_type) or event_type
    return (micro_id, physical_action, primary)


def _event_selection_rank(event: dict[str, Any]) -> tuple[int, int, float, int]:
    event_type = str(event.get("event_type") or event.get("evidence_type") or "")
    event_priority = {
        "object_trajectory_movement": 90,
        "liquid_flow_detected": 90,
        "liquid_level_change_detected": 90,
        "hand_object_contact": 80,
        "equipment_panel_operation_detected": 80,
        "container_state_change_detected": 80,
        "liquid_transfer_candidate": 70,
        "object_movement_detected": 60,
        "equipment_control_change": 55,
        "container_open_close": 55,
        "object_movement_candidate": 40,
        "equipment_panel_operation_candidate": 40,
        "container_state_change_candidate": 40,
    }.get(event_type, 0)
    source_priority = 2 if str(event.get("event_candidate_source") or "") == "video_understanding_event" else 1
    if event_type == "object_trajectory_movement":
        source_priority = 3
    confidence = _safe_float(event.get("confidence"), 0.0)
    asset_count = len([item for item in event.get("asset_refs") or [] if isinstance(item, dict)])
    return (event_priority, source_priority, confidence, asset_count)


def _selected_event_assets(
    session_root: Path,
    event: dict[str, Any],
    *,
    micro_by_id: dict[str, dict[str, Any]] | None = None,
) -> list[tuple[str, Path, dict[str, Any]]]:
    keyframes: list[tuple[str, Path, dict[str, Any]]] = []
    clips: list[tuple[str, Path, dict[str, Any]]] = []
    seen_paths: set[str] = set()
    asset_refs = [ref for ref in event.get("asset_refs") or [] if isinstance(ref, dict)]
    micro_asset_refs = _event_micro_asset_refs(event, micro_by_id or {})
    if micro_asset_refs:
        asset_refs = [*micro_asset_refs, *asset_refs]
    for raw_ref in asset_refs:
        if not isinstance(raw_ref, dict):
            continue
        if _asset_ref_is_non_real(raw_ref):
            event.setdefault("_asset_skip_reasons", []).append(
                {
                    "reason": "placeholder_or_synthetic_asset_ref_not_publishable",
                    "asset_ref": {key: raw_ref.get(key) for key in ("asset_type", "rel", "path", "source_type")},
                }
            )
            continue
        source = _event_asset_path(session_root, raw_ref)
        if source is None or not source.is_file():
            event.setdefault("_asset_skip_reasons", []).append(
                {
                    "reason": "asset_ref_file_missing",
                    "asset_ref": {key: raw_ref.get(key) for key in ("asset_type", "rel", "path", "source_type")},
                }
            )
            continue
        if not _material_file_is_real(source):
            event.setdefault("_asset_skip_reasons", []).append(
                {
                    "reason": "asset_ref_file_not_real_video_material",
                    "path": str(source),
                    "asset_ref": {key: raw_ref.get(key) for key in ("asset_type", "rel", "path", "source_type")},
                }
            )
            continue
        key = str(source.resolve()).lower()
        if key in seen_paths:
            continue
        seen_paths.add(key)
        asset_type = str(raw_ref.get("asset_type") or raw_ref.get("type") or "").lower()
        suffix = source.suffix.lower()
        if asset_type in {"keyframe", "image", "frame"} or suffix in {".jpg", ".jpeg", ".png", ".webp"}:
            keyframes.append((KEYFRAME_DIR_NAME, source, raw_ref))
        elif asset_type in {"clip", "video_clip", "video"} or suffix in {".mp4", ".mov", ".avi", ".mkv"}:
            clips.append((KEY_CLIP_DIR_NAME, source, raw_ref))
    keyframes.sort(key=_event_asset_rank)
    clips.sort(key=_event_asset_rank)
    event_primary = canonical_yolo_label(event.get("primary_object") or event.get("object_label")) or str(
        event.get("primary_object") or event.get("object_label") or ""
    )
    micro_available = _event_best_matching_micro(event, event_primary, micro_by_id or {}) is not None
    preferred_view = _event_preferred_view(event, micro_by_id or {})
    if preferred_view:
        view_clips = [item for item in clips if _event_asset_view(item[2], item[1]) == preferred_view]
        if view_clips:
            clips = view_clips
        known_view_keyframes = [item for item in keyframes if _event_asset_view(item[2], item[1]) == preferred_view]
        if known_view_keyframes:
            keyframes = known_view_keyframes
        elif clips and micro_available:
            keyframes = []
    selected: list[tuple[str, Path, dict[str, Any]]] = []
    if clips:
        selected_clip = clips[0]
        if keyframes:
            selected.append(keyframes[0])
        else:
            clip_ref = selected_clip[2]
            clip_view = _event_asset_view(clip_ref, selected_clip[1])
            selected.append(
                (
                    KEYFRAME_DIR_NAME,
                    selected_clip[1],
                    {
                        **clip_ref,
                        "asset_type": "generated_keyframe",
                        "rel": f"{clip_view or 'view'}.generated_keyframe",
                        "path": str(selected_clip[1]),
                        "view": clip_view,
                        "source_type": "event_clip_frame",
                    },
                )
            )
        selected.append(selected_clip)
    elif keyframes:
        selected.append(keyframes[0])
    return selected


def _asset_ref_is_non_real(ref: Mapping[str, Any]) -> bool:
    if bool(ref.get("placeholder") or ref.get("dry_run") or ref.get("synthetic")):
        return True
    text = " ".join(
        str(ref.get(key) or "").lower()
        for key in ("asset_type", "type", "rel", "path", "source_type", "role")
    )
    return any(marker in text for marker in NON_REAL_ASSET_MARKERS)


def _event_preferred_view(event: dict[str, Any], micro_by_id: dict[str, dict[str, Any]]) -> str:
    for key in ("view", "camera_view", "source_view", "requested_view"):
        value = str(event.get(key) or "").strip()
        if value in {"first_person", "third_person"}:
            return value
    text = " ".join(
        str(event.get(key) or "")
        for key in ("video_event_id", "evidence_id", "source_event_id", "event_id")
    ).lower()
    if "third_person" in text:
        return "third_person"
    if "first_person" in text:
        return "first_person"
    primary = canonical_yolo_label(event.get("primary_object") or event.get("object_label")) or str(
        event.get("primary_object") or event.get("object_label") or ""
    )
    micro = _event_best_matching_micro(event, primary, micro_by_id)
    if micro:
        rows = [
            row
            for row in micro.get("yolo_evidence") or []
            if isinstance(row, dict)
            and (not primary or canonical_yolo_label(row.get("primary_object")) == primary or _row_has_material_target_evidence(row, {primary}))
        ]
        if rows:
            return evidence_view(rows[0])
    for ref in event.get("asset_refs") or []:
        if not isinstance(ref, dict):
            continue
        ref_text = " ".join(str(ref.get(key) or "") for key in ("rel", "path", "view", "source_view")).lower()
        if "third_person" in ref_text:
            return "third_person"
        if "first_person" in ref_text:
            return "first_person"
    return ""


def _event_micro_asset_refs(event: dict[str, Any], micro_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    primary = canonical_yolo_label(event.get("primary_object") or event.get("object_label")) or str(
        event.get("primary_object") or event.get("object_label") or ""
    )
    micro = _event_best_matching_micro(event, primary, micro_by_id)
    if not micro:
        return []
    refs: list[dict[str, Any]] = []
    keyframes = micro.get("keyframes") if isinstance(micro.get("keyframes"), dict) else {}
    for rel, value in keyframes.items():
        if value:
            refs.append({"asset_type": "keyframe", "rel": str(rel), "path": value, "source_type": "micro_keyframe_reference"})
    for view in ("third_person", "first_person"):
        view_data = micro.get(view) if isinstance(micro.get(view), dict) else {}
        clip_path = view_data.get("clip_path") or view_data.get("annotated_clip_path")
        if clip_path:
            refs.append({"asset_type": "clip", "rel": f"{view}.clip_path", "path": clip_path, "view": view, "source_type": "micro_clip_reference"})
    return refs


def _event_asset_path(session_root: Path, ref: dict[str, Any]) -> Path | None:
    quality = ref.get("quality") if isinstance(ref.get("quality"), dict) else {}
    for value in (quality.get("resolved_path"), ref.get("resolved_path"), ref.get("path"), ref.get("file"), ref.get("source_file")):
        path = _resolve_session_path(session_root, value)
        if path is not None:
            return path
    return None


def _event_asset_rank(item: tuple[str, Path, dict[str, Any]]) -> tuple[int, int, str]:
    _asset_kind, source, ref = item
    text = " ".join(str(ref.get(key) or "") for key in ("rel", "source_type", "asset_type", "path")).lower()
    if "peak" in text or source.stem == "peak":
        role_rank = 0
    elif "contact" in text:
        role_rank = 1
    elif "third_person" in text:
        role_rank = 2
    elif "release" in text:
        role_rank = 3
    else:
        role_rank = 4
    view_rank = 0 if "third_person" in text else 1 if "first_person" in text else 2
    return (role_rank, view_rank, source.name)


def _candidate_record_from_event(
    event: dict[str, Any],
    asset_ref: dict[str, Any],
    source_file: Path,
    target: Path,
    asset_kind: str,
    *,
    exists: bool,
    micro_by_id: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    event_type = str(event.get("event_type") or event.get("evidence_type") or "physical_event_candidate")
    event_id = str(event.get("video_event_id") or event.get("evidence_id") or event_type)
    primary_object = canonical_yolo_label(event.get("primary_object") or event.get("object_label")) or str(
        event.get("primary_object") or event.get("object_label") or "object"
    )
    action_type, canonical_object, sop_phase, interaction_family = _event_canonical_fields(event_type, primary_object)
    group_id = _event_candidate_group_id(event)
    candidate_id = _event_candidate_id(event, asset_kind, source_file, asset_ref)
    start_sec, end_sec = _event_time_window(event)
    micro = _event_best_matching_micro(event, primary_object, micro_by_id or {})
    if micro:
        start_sec = start_sec if start_sec is not None else _optional_float(micro.get("start_sec"))
        end_sec = end_sec if end_sec is not None else _optional_float(micro.get("end_sec"))
    contact_valid_count = _event_valid_contact_evidence_count(event, primary_object, micro_by_id or {})
    contact_usable_count = _event_usable_contact_evidence_count(event, primary_object, micro_by_id or {})
    contact_peak_score = _event_contact_peak_score(event, primary_object, micro_by_id or {})
    physical_action_type = _event_physical_action_type(event_type)
    gate_reasons = _event_action_gate_reasons(
        event,
        event_type=event_type,
        physical_action_type=physical_action_type,
        primary_object=primary_object,
        contact_valid_count=contact_valid_count,
        contact_usable_count=contact_usable_count,
        contact_peak_score=contact_peak_score,
    )
    low_quality_event = bool(gate_reasons)
    sparse_contact_review = (
        physical_action_type == "hand_object_contact"
        and not low_quality_event
        and contact_usable_count > contact_valid_count
    )
    review_route = "human_review" if (low_quality_event or sparse_contact_review) else _event_review_route(event, exists=exists)
    review_reason_codes = _event_review_reason_codes(event, exists=exists)
    if sparse_contact_review:
        review_reason_codes = _ordered_unique_text(
            [
                *review_reason_codes,
                "sparse_contact_evidence_needs_human_review",
                "valid_contact_frame_count_below_auto_ready",
            ]
        )
    if low_quality_event:
        review_reason_codes = _ordered_unique_text(
            [
                *review_reason_codes,
                *gate_reasons,
            ]
        )
    action_name = _event_action_name(event_type, primary_object)
    view = _event_asset_view(asset_ref, source_file)
    confidence = round(max(0.0, min(1.0, _safe_float(event.get("confidence"), 0.55))), 3)
    quality_score = round(max(0.5, min(0.88, confidence + (0.06 if asset_kind == KEY_CLIP_DIR_NAME else 0.03))), 3)
    if low_quality_event:
        quality_score = min(quality_score, 0.42)
    source_real = bool(exists and _material_file_is_real(target) and not _asset_ref_is_non_real(asset_ref))
    secondary_objects = _event_secondary_objects(event_type, primary_object)
    evidence_chain = {
        "schema_version": MATERIAL_REFERENCE_TRACE_SCHEMA_VERSION,
        "candidate_source": event.get("event_candidate_source") or "video_understanding_event",
        "source_event_id": event_id,
        "event_type": event_type,
        "source_file": str(source_file),
        "camera_view": view,
        "time_start": start_sec,
        "time_end": end_sec,
        "confidence": confidence,
        "confidence_reasons": _as_list_for_semantics(event.get("confidence_reasons") or event.get("evidence_reasons")),
        "review_route": review_route,
        "review_reason_codes": review_reason_codes,
        "candidate_disposition": None,
    }
    return {
        "schema_version": "material_reference.item.v1",
        "trace_schema_version": MATERIAL_REFERENCE_TRACE_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "candidate_group_id": group_id,
        "candidate_source": event.get("event_candidate_source") or "video_understanding_event",
        "source_event_id": event_id,
        "event_type": event_type,
        "change_type": action_type,
        "physical_action_type": physical_action_type,
        "material_type": asset_kind,
        "asset_kind": asset_kind,
        "action_name": action_name,
        "display_title": action_name,
        "semantic_action": action_type,
        "taxonomy_schema_version": MATERIAL_TAXONOMY_SCHEMA_VERSION,
        "canonical_action_type": action_type,
        "canonical_object": canonical_object,
        "sop_phase": sop_phase,
        "interaction_family": interaction_family,
        "primary_object": primary_object,
        "raw_primary_object": primary_object,
        "manipulated_object": primary_object,
        "instrument_context": _event_instrument_context(event_type),
        "corrected_primary_object": canonical_object if action_type in {"equipment_panel_operation", "liquid_movement"} else primary_object,
        "secondary_objects": secondary_objects,
        "objects": _ordered_unique_text([canonical_object, primary_object, *secondary_objects, _event_instrument_context(event_type)]),
        "actions": _ordered_unique_text([action_type, event_type]),
        "micro_segment_id": event.get("micro_segment_id"),
        "parent_segment_id": event.get("segment_id"),
        "segment_id": event.get("segment_id"),
        "view": view,
        "camera_view": view,
        "frame_type": _event_frame_type(asset_ref),
        "frame_role": _event_frame_type(asset_ref),
        "start_sec": start_sec,
        "end_sec": end_sec,
        "time_start": start_sec,
        "time_end": end_sec,
        "source_file": str(source_file),
        "source_clip": str(source_file),
        "source_clip_path": str(source_file),
        "stored_file": str(target),
        "stored_filename": target.name,
        "file_name": target.name,
        "exists": bool(exists),
        "size_bytes": target.stat().st_size if exists and target.is_file() else 0,
        "source_real": source_real,
        "placeholder": not source_real,
        "publishable_material": source_real,
        "missing_reason": None if source_real else "event_candidate_file_not_real_video_material",
        "generated": False,
        "dry_run": False,
        "error": None,
        "quality_score": quality_score,
        "quality_bucket": "low_quality" if low_quality_event else "review_candidate",
        "quality_reasons": _ordered_unique_text(
            [
                "event_backed_physical_evidence",
                f"event_type:{event_type}",
                f"physical_action_type:{physical_action_type or 'unmapped'}",
                f"review_route:{review_route}",
                *gate_reasons,
                *_as_list_for_semantics(event.get("confidence_reasons") or event.get("evidence_reasons")),
            ]
        ),
        "candidate_status": "pending",
        "review_status": "pending",
        "review_required": True,
        "review_route": review_route,
        "review_reason_codes": review_reason_codes,
        "recommended": False,
        "pipeline_schema_version": "event_backed_material_candidates.v1",
        "pipeline_flow": [
            "video_understanding_event",
            "event_asset_selection",
            "vlm_or_human_review_route",
            "frontend_review_gate",
        ],
        "pipeline_stage": "frontend_review_gate",
        "pipeline_status": "event_backed_low_quality_evidence" if low_quality_event else "event_backed_review_required",
        "review_gate_policy": "Event-backed candidates must be reviewed before entering material_references.",
        "yolo_box_required": False,
        "yolo_annotated_required": False,
        "yolo_annotation_rendered": None,
        "box_filter": "event_evidence_asset_no_context_boxes",
        "physical_evidence_mode": (
            "event_backed_gate_failed"
            if low_quality_event
            else "event_backed_sparse_contact_review"
            if sparse_contact_review
            else "video_understanding_event_candidate"
        ),
        "yolo_evidence_count": contact_valid_count,
        "valid_contact_yolo_evidence_count": contact_valid_count,
        "usable_contact_yolo_evidence_count": contact_usable_count,
        "contact_peak_score": round(contact_peak_score, 3),
        "event_confidence": confidence,
        "event_anomaly_flags": _as_list_for_semantics(event.get("anomaly_flags")),
        "source_asset_ref": asset_ref,
        "evidence_chain": evidence_chain,
    }


def _render_event_candidate_keyframe(
    event: dict[str, Any],
    *,
    source: Path,
    target: Path,
    asset_ref: dict[str, Any],
    micro_by_id: dict[str, dict[str, Any]],
) -> None:
    primary = canonical_yolo_label(event.get("primary_object") or event.get("object_label")) or str(
        event.get("primary_object") or event.get("object_label") or ""
    )
    micro = _event_best_matching_micro(event, primary, micro_by_id)
    view = _event_asset_view(asset_ref, source)
    evidence_rows = [row for row in (micro or {}).get("yolo_evidence") or [] if isinstance(row, dict)]
    if view:
        evidence_rows = [row for row in evidence_rows if evidence_view(row) == view] or evidence_rows
    start_sec = _safe_float((micro or {}).get("start_sec"), _safe_float(event.get("start_sec"), 0.0))
    evidence_row = _event_nearest_evidence_for_object(evidence_rows, primary)
    event_time = _material_evidence_time(evidence_row) if evidence_row else start_sec
    offset = max(0.0, event_time - start_sec)
    if evidence_row is None:
        evidence_row = {"view": view, "local_time_sec": event_time, "detections": [], "hand_object_interactions": []}
    _extract_filtered_interaction_frame(
        source,
        offset,
        target,
        evidence_row,
        primary,
        allow_frame_filter_fallback=True,
    )


def _event_nearest_evidence_for_object(evidence_rows: list[dict[str, Any]], primary_object: str) -> dict[str, Any] | None:
    target_labels = _interaction_target_labels(primary_object)
    if not evidence_rows:
        return None
    if target_labels:
        matching = [
            row
            for row in evidence_rows
            if _row_has_material_target_evidence(row, target_labels)
            or canonical_yolo_label(row.get("primary_object")) in target_labels
        ]
        if matching:
            return max(matching, key=lambda row: _target_interaction_score(row, target_labels))
    return evidence_rows[0]


def _render_event_candidate_annotation(
    event: dict[str, Any],
    candidate: dict[str, Any],
    *,
    source: Path,
    target: Path,
    asset_kind: str,
    micro_by_id: dict[str, dict[str, Any]],
) -> None:
    if asset_kind != KEY_CLIP_DIR_NAME:
        return
    if str(candidate.get("quality_bucket") or "") == "low_quality":
        candidate["yolo_annotation_rendered"] = None
        return
    physical_action_type = str(candidate.get("physical_action_type") or "")
    if physical_action_type not in {"hand_object_contact", "liquid_movement"}:
        return
    primary = str(candidate.get("primary_object") or "")
    micro = _event_best_matching_micro(event, primary, micro_by_id)
    if not micro:
        return
    yolo_rows = [row for row in micro.get("yolo_evidence") or [] if isinstance(row, dict)]
    view = str(candidate.get("view") or candidate.get("camera_view") or "")
    if view:
        scoped_rows = [row for row in yolo_rows if evidence_view(row) == view] or yolo_rows
    else:
        scoped_rows = yolo_rows
    if not scoped_rows:
        return
    start_sec = _safe_float(micro.get("start_sec"), _safe_float(candidate.get("start_sec"), 0.0))
    end_sec = _safe_float(micro.get("end_sec"), _safe_float(candidate.get("end_sec"), start_sec + 1.5))
    duration = max(0.1, end_sec - start_sec)
    try:
        _render_filtered_interaction_clip(
            source,
            0.0,
            duration,
            target,
            scoped_rows,
            primary,
            start_sec,
            allow_frame_filter_fallback=True,
        )
        candidate["exists"] = target.is_file()
        candidate["size_bytes"] = target.stat().st_size if target.is_file() else 0
        candidate["generated"] = True
        candidate["yolo_annotation_rendered"] = True
        candidate["box_filter"] = "event_active_hand_object_annotation"
    except Exception as exc:
        candidate["yolo_annotation_rendered"] = False
        candidate["rerender_error"] = f"event_annotation_failed:{exc}"


def _event_canonical_fields(event_type: str, primary_object: str) -> tuple[str, str, str, str]:
    fields = EVENT_CANONICAL_ACTIONS.get(event_type)
    if not fields:
        return ("", primary_object or "object", "unmapped-physical-action", "unmapped")
    action_type, canonical_object, sop_phase, family = fields
    if canonical_object == "object":
        canonical_object = _canonical_action_fields(primary_object).get("canonical_object", primary_object or "object")
    return action_type, canonical_object, sop_phase, family


def _event_physical_action_type(event_type: str) -> str:
    action_type = EVENT_CANONICAL_ACTIONS.get(event_type, ("", "", "", ""))[0]
    return action_type if action_type in active_physical_action_types() else ""


def _event_action_gate_reasons(
    event: dict[str, Any],
    *,
    event_type: str,
    physical_action_type: str,
    primary_object: str,
    contact_valid_count: int,
    contact_usable_count: int,
    contact_peak_score: float,
) -> list[str]:
    reasons: list[str] = []
    sparse_contact_review_allowed = _event_sparse_contact_review_allowed(
        event_type=event_type,
        primary_object=primary_object,
        contact_valid_count=contact_valid_count,
        contact_usable_count=contact_usable_count,
        contact_peak_score=contact_peak_score,
    )
    if not physical_action_type:
        reasons.append("unmapped_physical_action_type")
    if _event_requires_contact_yolo(event_type) and contact_valid_count < PHYSICAL_EVIDENCE_MIN_FRAMES:
        if not sparse_contact_review_allowed:
            reasons.extend(["insufficient_stable_yolo_contact_evidence", "contact_event_blocked_from_default_review"])
            if contact_usable_count > contact_valid_count:
                reasons.append("sparse_contact_evidence_not_default_result")
    if _event_small_object_requires_contact(primary_object) and contact_valid_count < PHYSICAL_EVIDENCE_MIN_FRAMES:
        if not sparse_contact_review_allowed:
            reasons.extend(["no_valid_yolo_contact_evidence", "small_object_event_blocked_without_contact"])
    if physical_action_type == "object_movement":
        if _event_gate_rejected(event):
            reasons.append("physical_event_gate_rejected_object_movement")
        if not _event_movement_supported(event):
            reasons.append("object_movement_not_measured")
        if _event_motion_track_unstable(event):
            reasons.append("unstable_object_track_for_movement")
        if _event_is_label_level_pseudotrack(event) and contact_valid_count < PHYSICAL_EVIDENCE_MIN_FRAMES:
            reasons.extend(
                [
                    "label_level_pseudotrack_not_physical_movement",
                    "object_movement_requires_active_hand_object_evidence",
                ]
            )
            if contact_usable_count > contact_valid_count:
                reasons.append("sparse_movement_contact_evidence_not_default_result")
    if physical_action_type == "liquid_movement" and not _event_liquid_movement_supported(event, primary_object):
        reasons.append("liquid_movement_not_visually_supported")
    if physical_action_type == "liquid_movement" and _event_has_weak_visual_quality(event):
        reasons.append("weak_liquid_visual_evidence_not_default_result")
    if physical_action_type == "equipment_panel_operation" and not _event_equipment_panel_supported(event, primary_object):
        reasons.append("equipment_panel_operation_not_confirmed")
    if physical_action_type == "equipment_panel_operation" and _event_gate_rejected(event):
        reasons.append("physical_event_gate_rejected_panel_operation")
    if physical_action_type == "container_state_change" and not _event_container_state_supported(event):
        reasons.append("container_state_change_not_confirmed")
    if physical_action_type == "container_state_change" and _event_gate_rejected(event):
        reasons.append("physical_event_gate_rejected_container_state_change")
    return _ordered_unique_text(reasons)


def _event_sparse_contact_review_allowed(
    *,
    event_type: str,
    primary_object: str,
    contact_valid_count: int,
    contact_usable_count: int,
    contact_peak_score: float,
) -> bool:
    if not _event_requires_contact_yolo(event_type):
        return False
    if contact_usable_count <= 0 and contact_valid_count <= 0:
        return False
    if contact_valid_count > 0:
        return True
    primary = canonical_yolo_label(primary_object)
    threshold = (
        EVENT_BACKED_SMALL_OBJECT_SPARSE_CONTACT_REVIEW_MIN_SCORE
        if primary in EVENT_BACKED_SPARSE_CONTACT_SMALL_OBJECTS
        else EVENT_BACKED_SPARSE_CONTACT_REVIEW_MIN_SCORE
    )
    return contact_peak_score >= threshold


def _event_requires_contact_yolo(event_type: str) -> bool:
    return event_type == "hand_object_contact"


def _event_small_object_requires_contact(primary_object: str) -> bool:
    return canonical_yolo_label(primary_object) in {"paper", "weighing_paper"}


def _event_valid_contact_evidence_count(
    event: dict[str, Any],
    primary_object: str,
    micro_by_id: dict[str, dict[str, Any]],
) -> int:
    yolo_rows = [
        row
        for micro in _event_matching_micros(event, primary_object, micro_by_id)
        for row in micro.get("yolo_evidence") or []
        if isinstance(row, dict)
    ]
    if not yolo_rows:
        return 0
    return len(valid_yolo_physical_evidence(yolo_rows, primary_object))


def _event_usable_contact_evidence_count(
    event: dict[str, Any],
    primary_object: str,
    micro_by_id: dict[str, dict[str, Any]],
) -> int:
    yolo_rows = [
        row
        for micro in _event_matching_micros(event, primary_object, micro_by_id)
        for row in micro.get("yolo_evidence") or []
        if isinstance(row, dict)
    ]
    if not yolo_rows:
        return 0
    target_labels = _interaction_target_labels(primary_object)
    if not target_labels:
        return 0
    return sum(1 for row in yolo_rows if _has_plausible_sparse_target_interaction(row, target_labels))


def _event_contact_peak_score(
    event: dict[str, Any],
    primary_object: str,
    micro_by_id: dict[str, dict[str, Any]],
) -> float:
    target_labels = _interaction_target_labels(primary_object)
    scores: list[float] = []
    for micro in _event_matching_micros(event, primary_object, micro_by_id):
        for row in micro.get("yolo_evidence") or []:
            if isinstance(row, dict):
                scores.append(_target_interaction_score(row, target_labels))
    for text in _as_list_for_semantics(event.get("confidence_reasons") or event.get("evidence_reasons")):
        for match in re.finditer(r"(?:max_)?interaction_score=([0-9]+(?:\.[0-9]+)?)", str(text)):
            scores.append(_safe_float(match.group(1), 0.0))
    return max(scores, default=0.0)


def _event_matching_micros(
    event: dict[str, Any],
    primary_object: str,
    micro_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    micro_id = str(event.get("micro_segment_id") or "")
    direct = micro_by_id.get(micro_id) if micro_id else None
    if direct:
        return [direct]
    segment_id = str(event.get("segment_id") or event.get("parent_segment_id") or "")
    if not segment_id:
        return []
    target_labels = _interaction_target_labels(primary_object)
    matches: list[dict[str, Any]] = []
    for micro in micro_by_id.values():
        if not isinstance(micro, dict):
            continue
        if str(micro.get("parent_segment_id") or "") != segment_id:
            continue
        interaction = micro.get("interaction") if isinstance(micro.get("interaction"), dict) else {}
        micro_primary = canonical_yolo_label(interaction.get("primary_object") or micro.get("primary_object"))
        if target_labels and micro_primary not in target_labels:
            continue
        matches.append(micro)
    return sorted(matches, key=lambda micro: _micro_target_peak_score(micro, primary_object), reverse=True)


def _event_best_matching_micro(
    event: dict[str, Any],
    primary_object: str,
    micro_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    matches = _event_matching_micros(event, primary_object, micro_by_id)
    return matches[0] if matches else None


def _micro_target_peak_score(micro: dict[str, Any], primary_object: str) -> float:
    target_labels = _interaction_target_labels(primary_object)
    scores = [
        _target_interaction_score(row, target_labels)
        for row in micro.get("yolo_evidence") or []
        if isinstance(row, dict)
    ]
    interaction = micro.get("interaction") if isinstance(micro.get("interaction"), dict) else {}
    scores.append(_safe_float(interaction.get("max_interaction_score"), 0.0))
    return max(scores, default=0.0)


def _event_movement_supported(event: dict[str, Any]) -> bool:
    if str(event.get("event_type") or event.get("evidence_type") or "") in {"object_movement_candidate"}:
        return False
    if _event_gate_rejected(event):
        return False
    measurement = _event_measurement(event)
    point_count = _safe_float(measurement.get("point_count"), 0.0)
    displacement = _safe_float(measurement.get("displacement_px"), 0.0)
    path_length = _safe_float(measurement.get("path_length_px"), 0.0)
    identity = _safe_float(measurement.get("identity_confidence"), 0.0)
    if path_length > 0 and displacement > 0 and displacement / max(path_length, 1.0) < 0.18:
        return False
    return bool(point_count >= 4 and identity >= 0.70 and (displacement >= 14.0 or (path_length >= 80.0 and displacement >= 8.0)))


def _event_gate_rejected(event: dict[str, Any]) -> bool:
    gate = event.get("physical_event_gate")
    if isinstance(gate, dict) and str(gate.get("status") or "").lower() in {"rejected", "rejected_by_audit"}:
        return True
    metrics = event.get("metrics") if isinstance(event.get("metrics"), dict) else {}
    nested_gate = metrics.get("physical_event_gate") if isinstance(metrics.get("physical_event_gate"), dict) else {}
    if str(nested_gate.get("status") or "").lower() in {"rejected", "rejected_by_audit"}:
        return True
    text = _event_quality_text(event)
    return "rejected_by_gate" in text or "physical_event_gate_status=rejected" in text


def _event_motion_track_unstable(event: dict[str, Any]) -> bool:
    measurement = _event_measurement(event)
    identity = _safe_float(measurement.get("identity_confidence"), 0.0)
    id_switch_risk = _safe_float(measurement.get("id_switch_risk"), 0.0)
    point_count = _safe_float(measurement.get("point_count"), 0.0)
    if identity and identity < 0.70:
        return True
    if id_switch_risk and id_switch_risk > 0.35:
        return True
    if point_count and point_count < 4:
        return True
    text = _event_quality_text(event)
    return any(
        token in text
        for token in (
            "bbox_jitter_or_static_object",
            "high_id_switch_risk",
            "low_identity_confidence",
            "non_persistent_motion",
            "weak_bbox_continuity",
        )
    )


def _event_is_label_level_pseudotrack(event: dict[str, Any]) -> bool:
    text = _event_quality_text(event)
    if "label-level pseudo-track" in text:
        return True
    if "source_mode=yolo_frame_rows" in text:
        return True
    if "yolo detection rows converted to standard object track observation" in text:
        return True
    measurement = _event_measurement(event)
    source_mode = str(measurement.get("source_mode") or "").lower()
    track_id = str(measurement.get("track_id") or "").lower()
    return source_mode == "yolo_frame_rows" or track_id.startswith("yolo_track:")


def _event_liquid_movement_supported(event: dict[str, Any], primary_object: str) -> bool:
    event_type = str(event.get("event_type") or event.get("evidence_type") or "")
    if event_type in {"liquid_flow_detected", "liquid_level_change_detected"}:
        return True
    if event_type == "liquid_transfer_candidate" and _event_has_negative_visual_confirmation(event):
        return False
    if _confirmed_or_measured_event(event):
        return True
    canonical = canonical_yolo_label(primary_object)
    if canonical not in {"beaker", "container", "tube", "flask", "pipette", "pipette_tip"}:
        return False
    return not _event_has_negative_visual_confirmation(event)


def _event_equipment_panel_supported(event: dict[str, Any], primary_object: str) -> bool:
    event_type = str(event.get("event_type") or event.get("evidence_type") or "")
    if event_type == "equipment_panel_operation_detected" or _confirmed_or_measured_event(event):
        return True
    canonical = canonical_yolo_label(primary_object)
    return canonical in {"balance", "scale", "panel"} and _safe_float(event.get("confidence"), 0.0) >= 0.65


def _event_container_state_supported(event: dict[str, Any]) -> bool:
    event_type = str(event.get("event_type") or event.get("evidence_type") or "")
    if event_type == "container_state_change_detected" or _confirmed_or_measured_event(event):
        return True
    metrics = event.get("metrics") if isinstance(event.get("metrics"), dict) else {}
    indicators = metrics.get("container_state_indicators") if isinstance(metrics.get("container_state_indicators"), dict) else {}
    cap_tokens = metrics.get("cap_lid_tokens") if isinstance(metrics.get("cap_lid_tokens"), list) else []
    state_signal = str(indicators.get("state_signal") or "").lower()
    return bool(cap_tokens or state_signal not in {"", "container_interaction_only"})


def _confirmed_or_measured_event(event: dict[str, Any]) -> bool:
    text = " ".join(
        str(event.get(key) or "")
        for key in ("confirmation_level", "visual_confirmation_level", "conclusion_status")
    ).lower()
    return "confirmed" in text or "measured" in text


def _event_has_negative_visual_confirmation(event: dict[str, Any]) -> bool:
    text = _event_quality_text(event)
    negative_tokens = (
        "not_visual_liquid_flow_confirmed",
        "visual_confirmation_limited",
        "candidate_weak_bundle_rollup",
        "low_confidence_candidate_event",
        "evidence_limitation_missing_visual_or_transcript",
    )
    return any(token in text for token in negative_tokens)


def _event_has_weak_visual_quality(event: dict[str, Any]) -> bool:
    text = _event_quality_text(event)
    weak_tokens = (
        "only_single_frame_evidence",
        "weak_bbox_continuity",
        "low_signal_yolo_candidate",
        "physical_evidence_validation_relaxed",
        "coverage_backfill_candidate",
    )
    return any(token in text for token in weak_tokens)


def _event_quality_text(event: dict[str, Any]) -> str:
    values: list[Any] = [
        event.get("confirmation_level"),
        event.get("visual_confirmation_level"),
        event.get("conclusion_status"),
        *(_as_list_for_semantics(event.get("anomaly_flags"))),
        *(_as_list_for_semantics(event.get("limitations"))),
        *(_as_list_for_semantics(event.get("confidence_reasons"))),
        *(_as_list_for_semantics(event.get("evidence_reasons"))),
    ]
    values.extend(_event_asset_quality_values(event))
    return " ".join(str(value or "").lower() for value in values)


def _event_asset_quality_values(event: dict[str, Any]) -> list[str]:
    values: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if str(key).lower() in {"warnings", "limitations", "status", "confidence"}:
                    visit(item)
                elif str(key).lower() in {"source_quality", "quality"}:
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, (str, int, float)):
            text = str(value).strip()
            if text:
                values.append(text)

    for ref in event.get("asset_refs") or []:
        if isinstance(ref, dict):
            visit(ref.get("quality"))
    return values


def _event_measurement(event: dict[str, Any]) -> dict[str, Any]:
    metrics = event.get("metrics") if isinstance(event.get("metrics"), dict) else {}
    measurement = metrics.get("measurement") if isinstance(metrics.get("measurement"), dict) else {}
    if measurement:
        return measurement
    model_metrics = metrics.get("model_metrics") if isinstance(metrics.get("model_metrics"), dict) else {}
    return model_metrics


def _event_review_route(event: dict[str, Any], *, exists: bool) -> str:
    confidence = _safe_float(event.get("confidence"), 0.0)
    if not exists or confidence < 0.45:
        return "human_review"
    flags = " ".join(_as_list_for_semantics(event.get("anomaly_flags"))).lower()
    if "conflict" in flags or "contradiction" in flags:
        return "human_review"
    return "vlm_review"


def _event_review_reason_codes(event: dict[str, Any], *, exists: bool) -> list[str]:
    values = [
        *(_as_list_for_semantics(event.get("anomaly_flags"))),
        *(_as_list_for_semantics(event.get("limitations"))),
    ]
    if not exists:
        values.append("source_asset_missing")
    if _safe_float(event.get("confidence"), 0.0) < 0.65:
        values.append("event_confidence_below_auto_ready")
    values.append("event_backed_candidate_requires_review")
    return _ordered_unique_text(values)


def _event_time_window(event: dict[str, Any]) -> tuple[float | None, float | None]:
    start = _optional_float(event.get("start_sec", event.get("time_start")))
    end = _optional_float(event.get("end_sec", event.get("time_end")))
    if start is None and event.get("payload") and isinstance(event.get("payload"), dict):
        micro = event["payload"].get("micro_segment") if isinstance(event["payload"].get("micro_segment"), dict) else {}
        start = _optional_float(micro.get("start_sec"))
        end = _optional_float(micro.get("end_sec")) if end is None else end
    if end is None and start is not None:
        end = start
    return start, end


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _event_action_name(event_type: str, primary_object: str) -> str:
    object_name = _object_display_name(primary_object)
    primary_label = canonical_yolo_label(primary_object)
    business_name = ACTION_NAME_BY_OBJECT.get(primary_label)
    if event_type == "hand_object_contact":
        return business_name or f"手部与物体接触-{object_name}"
    if event_type in {"object_movement_candidate", "object_movement_detected", "object_trajectory_movement"}:
        return f"物体移动-{object_name}"
    if event_type in {"liquid_transfer_candidate", "liquid_flow_detected", "liquid_level_change_detected"}:
        return f"液体移动-{object_name}"
    if event_type in {"equipment_panel_operation_candidate", "equipment_panel_operation_detected", "equipment_control_change"}:
        return "\u5929\u5e73\u8bbe\u5907\u9762\u677f\u64cd\u4f5c" if primary_label in {"balance", "scale", "panel"} else f"设备面板操作-{object_name}"
    if event_type in {"container_state_change_candidate", "container_state_change_detected", "container_open_close"}:
        return f"容器状态变化-{object_name}"
    return f"物理动作候选-{object_name}"


def _object_display_name(label: Any) -> str:
    canonical = canonical_yolo_label(label) or str(label or "object")
    return OBJECT_DISPLAY_NAMES.get(canonical, canonical)


def _event_instrument_context(event_type: str) -> str | None:
    if event_type in {"equipment_panel_operation_candidate", "equipment_panel_operation_detected", "equipment_control_change"}:
        return "panel"
    if event_type in {"liquid_transfer_candidate", "liquid_flow_detected", "liquid_level_change_detected"}:
        return "container"
    return None


def _event_secondary_objects(event_type: str, primary_object: str) -> list[str]:
    values = []
    context = _event_instrument_context(event_type)
    if context:
        values.append(context)
    if event_type in {"container_state_change_candidate", "container_state_change_detected", "container_open_close"} and primary_object != "container":
        values.append("container")
    return _ordered_unique_text(values)


def _event_frame_type(ref: dict[str, Any]) -> str | None:
    text = " ".join(str(ref.get(key) or "") for key in ("rel", "path", "source_type")).lower()
    for role in ("contact", "peak", "release", "start", "end", "middle"):
        if role in text:
            return role
    return None


def _event_asset_view(ref: dict[str, Any], source: Path) -> str:
    for key in ("view", "camera_view", "source_view"):
        if ref.get(key):
            return str(ref[key])
    text = f"{ref.get('rel') or ''} {source}".lower()
    if "third_person" in text:
        return "third_person"
    if "first_person" in text:
        return "first_person"
    return ""


def _event_asset_basename(event: dict[str, Any], ref: dict[str, Any], asset_kind: str) -> str:
    event_type = str(event.get("event_type") or event.get("evidence_type") or "event")
    primary = canonical_yolo_label(event.get("primary_object") or event.get("object_label")) or "object"
    role = _event_frame_type(ref) or ("clip" if asset_kind == KEY_CLIP_DIR_NAME else "frame")
    micro_id = str(event.get("micro_segment_id") or event.get("segment_id") or "event")
    return f"{_event_action_name(event_type, primary)}_{micro_id}_{role}"


def _event_candidate_group_id(event: dict[str, Any]) -> str:
    source = "|".join(
        str(event.get(key) or "")
        for key in ("video_event_id", "evidence_id", "event_type", "evidence_type", "micro_segment_id", "segment_id")
    )
    digest = hashlib.sha1(source.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"event_candidate_group_{digest}"


def _event_candidate_id(event: dict[str, Any], asset_kind: str, source_file: Path, ref: dict[str, Any]) -> str:
    source = "|".join(
        [
            str(event.get("video_event_id") or event.get("evidence_id") or ""),
            asset_kind,
            str(source_file),
            str(ref.get("asset_id") or ref.get("rel") or ""),
        ]
    )
    digest = hashlib.sha1(source.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"event_material_candidate_{digest}"


def _count_by_field(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(field) or "")
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return counts


def _candidate_asset_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    group_ids: set[str] = set()
    for index, row in enumerate(rows):
        group_ids.add(str(row.get("candidate_group_id") or row.get("candidate_id") or f"candidate_row_{index}"))
    return {
        "keyframe_count": sum(1 for row in rows if row.get("asset_kind") == KEYFRAME_DIR_NAME),
        "key_clip_count": sum(1 for row in rows if row.get("asset_kind") == KEY_CLIP_DIR_NAME),
        "candidate_group_count": len(group_ids),
    }


def _material_candidate_summary(
    session_root: Path,
    candidate_root: Path,
    keyframe_dir: Path,
    clip_dir: Path,
    index_json: Path,
    index_jsonl: Path,
    candidate_rows: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    pipeline_summary: dict[str, Any],
    *,
    dry_run: bool,
    segment_fallback_used: bool,
    preserved_candidate_count: int,
) -> dict[str, Any]:
    asset_counts = _candidate_asset_counts(candidate_rows)
    return {
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
        **asset_counts,
        "pending_total": sum(1 for row in candidate_rows if row.get("candidate_status") == "pending"),
        "recommended_total": sum(1 for row in candidate_rows if row.get("recommended") is True),
        "skipped": skipped,
        "pipeline_summary": pipeline_summary,
        "segment_level_fallback_used": segment_fallback_used,
        "preserved_candidate_count": preserved_candidate_count,
        "policy": "Candidates require frontend approval before entering material_references.",
        "records": candidate_rows,
    }


def _generated_material_summary(
    session_root: Path,
    ref_root: Path,
    formal_root: Path,
    keyframe_dir: Path,
    clip_dir: Path,
    index_json: Path,
    index_jsonl: Path,
    experiment: dict[str, str],
    records: list[dict[str, Any]],
    planned_records: list[dict[str, Any]],
    output_rows: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    archived_items: list[dict[str, str]],
    archive_root: Path,
    *,
    dry_run: bool,
    ffmpeg_ok: bool,
) -> dict[str, Any]:
    return {
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
        "published_real_file_count": sum(1 for row in output_rows if row.get("source_real") is True and row.get("placeholder") is not True),
        "formal_published_file_count": sum(1 for row in output_rows if row.get("source_real") is True and row.get("placeholder") is not True),
        "planned_file_count": len(planned_records) if dry_run else 0,
        "keyframe_count": sum(1 for row in records if row.get("asset_kind") == KEYFRAME_DIR_NAME and row.get("exists")),
        "key_clip_count": sum(1 for row in records if row.get("asset_kind") == KEY_CLIP_DIR_NAME and row.get("exists")),
        "naming_rule": NAMING_RULE,
        "policy": "Validated YOLO hand-object keyframes and clips are staged here as review sources; only approved candidates are copied to the formal delivery folder.",
        "real_source_policy": "Formal material rows must point to real extracted keyframes or clips from user-uploaded video; placeholder, poster, synthetic, missing, and dry-run rows are suppressed.",
        "formal_publish_gate_policy": "Formal material rows require a dual_event_id or equivalent dual-view action event plus first/third keyframes and key clips.",
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


def _mark_material_summary_blocked(summary: dict[str, Any], session_root: Path, blocked_reason: str) -> dict[str, Any]:
    action_gate = formal_dual_view_action_gate_status(session_root)
    summary["status"] = "blocked"
    summary["formal_publish_blocked"] = True
    summary["blocked_reason"] = blocked_reason
    summary["formal_dual_view_action_gate"] = action_gate
    summary["video_memory_allowed"] = False
    summary["memory_write_allowed"] = False
    return summary


def _write_blocked_material_reference_summary(
    *,
    session_root: Path,
    ref_root: Path,
    formal_root: Path,
    keyframe_dir: Path,
    clip_dir: Path,
    index_json: Path,
    index_jsonl: Path,
    experiment: dict[str, str],
    archived_items: list[dict[str, str]],
    archive_root: Path,
    blocked_reason: str,
    dry_run: bool,
    ffmpeg_ok: bool,
) -> dict[str, Any]:
    gate = formal_dual_view_action_gate_status(session_root)
    skipped = [
        {
            "reason": "formal_material_publish_gate",
            "suppression_reason": blocked_reason,
            "blocked_reason": blocked_reason,
            "formal_dual_view_action_gate": gate,
        }
    ]
    summary = _generated_material_summary(
        session_root,
        ref_root,
        formal_root,
        keyframe_dir,
        clip_dir,
        index_json,
        index_jsonl,
        experiment,
        [],
        [],
        [],
        skipped,
        archived_items,
        archive_root,
        dry_run=dry_run,
        ffmpeg_ok=ffmpeg_ok,
    )
    _mark_material_summary_blocked(summary, session_root, blocked_reason)
    _write_material_reference_summary(ref_root, index_json, index_jsonl, summary, [])
    _write_json(
        session_root / "metadata" / "formal_material_publish_gate.json",
        {
            "schema_version": FORMAL_MATERIAL_PUBLISH_GATE_VERSION,
            "status": "blocked",
            "blocked_reason": blocked_reason,
            "formal_dual_view_action_gate": gate,
            "formal_publish_blocked": True,
            "video_memory_allowed": False,
        },
    )
    if not dry_run:
        delivery_roots = [
            root
            for root in material_reference_delivery_roots(session_root)
            if root.resolve() != ref_root.resolve()
        ]
        for delivery_root in delivery_roots:
            _copy_simplified_materials(ref_root, delivery_root, summary)
        index_roots = _ordered_unique_paths([ref_root, *delivery_roots])
        summary["reference_indexes"] = _build_material_reference_indexes(index_roots)
        summary["global_material_library_indexes"] = _sync_global_material_library_indexes(session_root, index_roots)
        _write_material_reference_summary(ref_root, index_json, index_jsonl, summary, [])
    return summary


def _filter_publishable_material_rows(
    rows: list[dict[str, Any]],
    ref_root: Path,
    *,
    session_root: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if session_root is not None:
        allowed, block_reason = _session_formal_material_publish_allowed(session_root)
        if not allowed:
            action_gate = formal_dual_view_action_gate_status(session_root)
            return [], [
                {
                    "reason": "formal_material_publish_gate",
                    "suppression_reason": block_reason,
                    "blocked_reason": block_reason,
                    "formal_dual_view_action_gate": action_gate if action_gate.get("artifacts_present") else None,
                    "candidate_id": row.get("candidate_id"),
                    "micro_segment_id": row.get("micro_segment_id"),
                    "segment_id": row.get("segment_id") or row.get("parent_segment_id"),
                    "asset_kind": row.get("asset_kind") or row.get("material_type"),
                    "view": row.get("view") or row.get("camera_view"),
                    "stored_file": row.get("stored_file"),
                    "source_file": row.get("source_file"),
                }
                for row in rows
            ]
    kept: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    for row in rows:
        ok, reason = _material_row_is_publishable(row, root=ref_root)
        if ok:
            _pts_ok, _pts_reason, pts_details = _material_row_source_pts_gate(row, root=ref_root)
            kept.append(
                {
                    **row,
                    "exists": True,
                    "source_real": True,
                    "placeholder": False,
                    "publishable_material": True,
                    "missing_reason": None,
                    **({"source_pts_gate": pts_details} if pts_details else {}),
                }
            )
            continue
        _pts_ok, _pts_reason, pts_details = _material_row_source_pts_gate(row, root=ref_root)
        suppressed.append(
            {
                "reason": "non_real_material_suppressed",
                "suppression_reason": reason,
                "candidate_id": row.get("candidate_id"),
                "micro_segment_id": row.get("micro_segment_id"),
                "segment_id": row.get("segment_id") or row.get("parent_segment_id"),
                "asset_kind": row.get("asset_kind") or row.get("material_type"),
                "view": row.get("view") or row.get("camera_view"),
                "stored_file": row.get("stored_file"),
                "source_real": row.get("source_real"),
                "placeholder": row.get("placeholder"),
                **({"source_pts_gate": pts_details} if pts_details else {}),
            }
        )
    return kept, suppressed


def _write_material_reference_summary(ref_root: Path, index_json: Path, index_jsonl: Path, summary: dict[str, Any], output_rows: list[dict[str, Any]]) -> None:
    _write_json(index_json, summary)
    _write_jsonl(index_jsonl, output_rows)
    _write_json(ref_root / "manifest.json", _manifest(summary))
    _write_readme(ref_root / "README.md", summary)


def _append_material_record_result(
    records: list[dict[str, Any]],
    row: dict[str, Any],
    target: Path,
    *,
    generated: bool,
    error: str | None,
) -> None:
    size_bytes = target.stat().st_size if target.exists() else 0
    generated_ok = bool(generated)
    if target.suffix.lower() == ".mp4" and 0 < size_bytes < 1024 and _looks_like_empty_ffmpeg_mp4(target):
        try:
            target.unlink()
        except OSError:
            pass
        return
    if not generated_ok or size_bytes <= 0:
        return
    if not _material_file_is_real(target):
        return
    records.append(
        {
            **row,
            "exists": generated_ok,
            "generated": generated_ok,
            "error": error,
            "size_bytes": size_bytes,
            "source_real": True,
            "placeholder": False,
            "publishable_material": True,
            "missing_reason": None,
            "yolo_annotation_rendered": bool(generated_ok and not error),
        }
    )


def _looks_like_empty_ffmpeg_mp4(target: Path) -> bool:
    try:
        header = target.read_bytes()[:256]
    except OSError:
        return False
    return b"ftyp" in header and b"moov" in header


def _render_material_clip_result(
    *,
    source_clip: Path,
    clip_offset: float,
    clip_duration: float,
    target: Path,
    view_evidence: list[dict[str, Any]],
    annotation_target: str,
    segment_start: float,
    render_yolo_annotations: bool,
    ffmpeg_ok: bool,
    ffmpeg_path: str | Path,
) -> tuple[bool, str | None]:
    if not render_yolo_annotations:
        if not ffmpeg_ok:
            return False, "paired_view_clip_failed_ffmpeg_unavailable"
        try:
            _cut_video(ffmpeg_path, source_clip, clip_offset, clip_duration, target)
            return target.exists(), None
        except Exception as exc:  # pragma: no cover
            return False, str(exc)
    try:
        _render_filtered_interaction_clip(
            source_clip,
            clip_offset,
            clip_duration,
            target,
            view_evidence,
            annotation_target,
            segment_start,
        )
        return target.exists(), None
    except Exception as exc:  # pragma: no cover
        if ffmpeg_ok:
            try:
                _cut_video(ffmpeg_path, source_clip, clip_offset, clip_duration, target)
                return target.exists(), f"annotation_fallback_unboxed:{exc}"
            except Exception as fallback_exc:
                return False, str(fallback_exc)
        return False, f"annotation_failed_ffmpeg_unavailable:{exc}"


def _render_material_keyframe_result(
    *,
    source_clip: Path,
    source_offset_sec: float,
    target: Path,
    evidence_row: dict[str, Any],
    annotation_target: str,
    render_yolo_annotations: bool,
    ffmpeg_ok: bool,
    ffmpeg_path: str | Path,
) -> tuple[bool, str | None]:
    if not render_yolo_annotations:
        if not ffmpeg_ok:
            return False, "paired_view_frame_failed_ffmpeg_unavailable"
        try:
            _extract_frame(ffmpeg_path, source_clip, source_offset_sec, target)
            return target.exists(), None
        except Exception as exc:  # pragma: no cover
            return False, str(exc)
    try:
        _extract_filtered_interaction_frame(
            source_clip,
            source_offset_sec,
            target,
            evidence_row,
            annotation_target,
            require_boxes=True,
        )
        return target.exists(), None
    except Exception as exc:  # pragma: no cover
        if ffmpeg_ok:
            try:
                _extract_frame(ffmpeg_path, source_clip, source_offset_sec, target)
                return target.exists(), f"annotation_fallback_unboxed:{exc}"
            except Exception as fallback_exc:
                return False, str(fallback_exc)
        return False, f"annotation_failed_ffmpeg_unavailable:{exc}"


def _micro_material_evidence_rows(
    micro: dict[str, Any],
    yolo_frame_rows: list[dict[str, Any]],
    *,
    primary_object: str,
    start_sec: float,
    end_sec: float,
) -> list[dict[str, Any]]:
    raw_rows = [
        _enrich_material_evidence_frame_metadata(item, yolo_frame_rows)
        for item in micro.get("yolo_evidence") or []
        if isinstance(item, dict)
    ]
    target_labels = _interaction_target_labels(primary_object)
    supplemental: list[dict[str, Any]] = []
    if target_labels:
        for row in yolo_frame_rows:
            if not isinstance(row, dict):
                continue
            time_sec = _material_evidence_time(row)
            if time_sec < start_sec - MICRO_MATERIAL_EVIDENCE_WINDOW_PAD_SEC:
                continue
            if time_sec > end_sec + MICRO_MATERIAL_EVIDENCE_WINDOW_PAD_SEC:
                continue
            if _row_has_material_target_evidence(row, target_labels):
                supplemental.append(row)
    rows = _dedupe_material_evidence_rows([*raw_rows, *supplemental])
    return sorted(rows, key=lambda item: (evidence_view(item), _material_evidence_time(item), _safe_float(item.get("frame_index"), 0.0)))


def _enrich_material_evidence_frame_metadata(row: dict[str, Any], yolo_frame_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if _safe_float(row.get("frame_width"), 0.0) > 0 and _safe_float(row.get("frame_height"), 0.0) > 0:
        return row
    view = evidence_view(row)
    if not view:
        return row
    target_time = _material_evidence_time(row)
    target_frame = _safe_float(row.get("frame_index"), -1.0)
    best: tuple[float, dict[str, Any]] | None = None
    for candidate in yolo_frame_rows:
        if not isinstance(candidate, dict) or evidence_view(candidate) != view:
            continue
        frame_width = int(_safe_float(candidate.get("frame_width"), 0.0))
        frame_height = int(_safe_float(candidate.get("frame_height"), 0.0))
        if frame_width <= 0 or frame_height <= 0:
            continue
        candidate_time = _material_evidence_time(candidate)
        candidate_frame = _safe_float(candidate.get("frame_index"), -1.0)
        time_delta = abs(candidate_time - target_time) if target_time or candidate_time else 9999.0
        frame_delta = abs(candidate_frame - target_frame) if target_frame >= 0 and candidate_frame >= 0 else 9999.0
        if time_delta > 1.0 and frame_delta > 30.0:
            continue
        score = min(time_delta, frame_delta / 30.0)
        if best is None or score < best[0]:
            best = (score, candidate)
    if best is None:
        return row
    enriched = dict(row)
    source = best[1]
    enriched["frame_width"] = int(_safe_float(source.get("frame_width"), 0.0))
    enriched["frame_height"] = int(_safe_float(source.get("frame_height"), 0.0))
    if enriched.get("time_sec") is None and source.get("time_sec") is not None:
        enriched["time_sec"] = source.get("time_sec")
    return enriched


def _select_material_evidence_rows(
    evidence_rows: list[dict[str, Any]],
    primary_object: str,
) -> tuple[list[dict[str, Any]], str, int]:
    rows = [item for item in evidence_rows if isinstance(item, dict)]
    valid_rows = valid_yolo_physical_evidence(rows, primary_object)
    if len(valid_rows) >= PHYSICAL_EVIDENCE_MIN_FRAMES:
        return valid_rows, STRICT_PHYSICAL_EVIDENCE_MODE, len(valid_rows)
    target_labels = _interaction_target_labels(primary_object)
    if target_labels.intersection({"balance", "scale"}):
        return [], "", len(valid_rows)
    sparse_rows = [
        row
        for row in rows
        if target_labels and _has_plausible_sparse_target_interaction(row, target_labels)
    ]
    if sparse_rows:
        return sparse_rows, SPARSE_PHYSICAL_EVIDENCE_MODE, len(valid_rows)
    return [], "", len(valid_rows)


def _has_plausible_sparse_target_interaction(row: dict[str, Any], target_labels: set[str]) -> bool:
    interactions, _detections = _target_interactions_from_evidence(row, target_labels, frame=None)
    return any(_sparse_interaction_reject_reason(row, interaction) is None for interaction in interactions)


def _sparse_interaction_reject_reason(row: dict[str, Any], interaction: dict[str, Any]) -> str | None:
    object_label = canonical_yolo_label(interaction.get("object_label") or interaction.get("target_label") or interaction.get("object"))
    hand_bbox = _bbox(interaction.get("hand_bbox"))
    object_bbox = _bbox(interaction.get("object_bbox"))
    if hand_bbox is None or object_bbox is None:
        return "missing_sparse_interaction_bbox"
    frame_width = int(_safe_float(row.get("frame_width"), 0.0))
    frame_height = int(_safe_float(row.get("frame_height"), 0.0))
    if frame_width <= 0 or frame_height <= 0:
        frame_width, frame_height = _infer_frame_size_from_bboxes(row, hand_bbox, object_bbox)
    if frame_width <= 0 or frame_height <= 0:
        return None
    view = evidence_view(row).lower()
    hand_width_ratio, hand_height_ratio, hand_area_ratio = _bbox_ratios(hand_bbox, frame_width, frame_height)
    object_width_ratio, object_height_ratio, object_area_ratio = _bbox_ratios(object_bbox, frame_width, frame_height)
    hand_touches_x, hand_touches_y = _bbox_touches_frame(hand_bbox, frame_width, frame_height)
    object_touches_x, object_touches_y = _bbox_touches_frame(object_bbox, frame_width, frame_height)
    score = _safe_float(interaction.get("score"), _safe_float(interaction.get("interaction_score"), 0.0))
    low_res = min(frame_width, frame_height) <= 540
    if view == "first_person" and low_res:
        if (
            object_label == "paper"
            and object_touches_x
            and object_area_ratio > 0.055
            and hand_area_ratio > 0.10
            and hand_width_ratio > 0.30
            and score < 0.45
        ):
            return "low_res_static_edge_paper_with_oversized_sparse_hand"
        if (
            hand_area_ratio > 0.145
            and hand_width_ratio > 0.34
            and hand_height_ratio > 0.34
            and score < 0.42
            and not hand_touches_y
        ):
            return "low_res_oversized_nonbottom_sparse_hand"
        if (
            object_label in {"paper", "spatula", "pipette", "pipette_tip"}
            and (object_touches_x or object_touches_y)
            and score < 0.38
            and hand_area_ratio > 0.08
        ):
            return "low_res_edge_context_object_sparse_interaction"
    return None


def _infer_frame_size_from_bboxes(
    row: dict[str, Any],
    *bboxes: list[float],
) -> tuple[int, int]:
    max_x = 0.0
    max_y = 0.0
    for bbox in bboxes:
        if bbox:
            max_x = max(max_x, float(bbox[2]))
            max_y = max(max_y, float(bbox[3]))
    for detection in row.get("detections") or []:
        if isinstance(detection, dict):
            bbox = _bbox(detection.get("bbox"))
            if bbox:
                max_x = max(max_x, float(bbox[2]))
                max_y = max(max_y, float(bbox[3]))
    if max_x < 320 or max_y < 240:
        return 0, 0
    width = 640 if 560 <= max_x <= 660 else 1280 if 1100 <= max_x <= 1320 else int(max_x)
    height = 480 if 420 <= max_y <= 520 else 720 if 620 <= max_y <= 760 else int(max_y)
    return width, height


def _bbox_ratios(bbox: list[float], frame_width: int, frame_height: int) -> tuple[float, float, float]:
    width = max(0.0, float(bbox[2]) - float(bbox[0]))
    height = max(0.0, float(bbox[3]) - float(bbox[1]))
    frame_area = max(1.0, float(frame_width) * float(frame_height))
    return width / max(1.0, float(frame_width)), height / max(1.0, float(frame_height)), (width * height) / frame_area


def _bbox_touches_frame(bbox: list[float], frame_width: int, frame_height: int) -> tuple[bool, bool]:
    margin_x = max(2.0, float(frame_width) * 0.015)
    margin_y = max(2.0, float(frame_height) * 0.015)
    return (
        float(bbox[0]) <= margin_x or float(bbox[2]) >= float(frame_width) - margin_x,
        float(bbox[1]) <= margin_y or float(bbox[3]) >= float(frame_height) - margin_y,
    )


def _material_evidence_record_fields(
    evidence_rows: list[dict[str, Any]],
    *,
    physical_evidence_mode: str,
    valid_evidence_count: int,
    usable_evidence_count: int,
    evidence_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    evidence_debug = {"raw_labels": _raw_yolo_labels_from_evidence(evidence_rows)}
    if physical_evidence_mode == "paired_view_time_alignment":
        anchor_views = sorted({evidence_view(row) for row in evidence_rows if evidence_view(row)})
        return {
            "candidate_source": "paired_view_time_alignment",
            "physical_evidence_mode": physical_evidence_mode,
            "physical_evidence_required_min_frames": 0,
            "valid_yolo_evidence_count": int(valid_evidence_count),
            "usable_yolo_evidence_count": int(usable_evidence_count),
            "physical_evidence_diagnostics": evidence_diagnostics,
            "evidence": evidence_debug,
            "source_yolo_evidence": _compact_yolo_evidence_rows(evidence_rows),
            "quality_reasons": [
                "paired_first_person_clip",
                "same_time_window_as_yolo_verified_material",
                *[f"anchor_view:{view}" for view in anchor_views],
            ],
        }
    strict = physical_evidence_mode == STRICT_PHYSICAL_EVIDENCE_MODE
    required_frames = PHYSICAL_EVIDENCE_MIN_FRAMES if strict else 1
    reasons = ["yolo_physical_evidence"]
    if not strict:
        reasons.extend(["sparse_yolo_evidence", "manual_material_review_required"])
    return {
        "candidate_source": "micro_segment_yolo_physical_evidence" if strict else "micro_segment_yolo_sparse_evidence",
        "physical_evidence_mode": physical_evidence_mode,
        "physical_evidence_required_min_frames": required_frames,
        "valid_yolo_evidence_count": int(valid_evidence_count),
        "usable_yolo_evidence_count": int(usable_evidence_count),
        "physical_evidence_diagnostics": evidence_diagnostics,
        "evidence": evidence_debug,
        "source_yolo_evidence": _compact_yolo_evidence_rows(evidence_rows),
        "quality_reasons": reasons,
    }


def _raw_yolo_labels_from_evidence(rows: list[dict[str, Any]]) -> list[str]:
    labels: list[Any] = []
    for row in rows:
        labels.extend([row.get("raw_label"), row.get("label"), row.get("object_label"), row.get("primary_object")])
        for detection in row.get("detections") or []:
            if isinstance(detection, dict):
                labels.extend([detection.get("raw_label"), detection.get("label"), detection.get("object_label")])
        for interaction in row.get("hand_object_interactions") or []:
            if isinstance(interaction, dict):
                labels.extend(
                    [
                        interaction.get("raw_label"),
                        interaction.get("hand_label"),
                        interaction.get("object_label"),
                        interaction.get("target_label"),
                        interaction.get("object"),
                    ]
                )
    return _ordered_unique_text(labels)


def _material_evidence_time(row: dict[str, Any]) -> float:
    for key in ("time_sec", "local_time_sec", "timestamp_sec", "source_offset_sec"):
        value = row.get(key)
        if value is not None:
            return _safe_float(value)
    return 0.0


def _row_has_material_target_evidence(row: dict[str, Any], target_labels: set[str]) -> bool:
    if _target_interaction_score(row, target_labels) > 0.0:
        return True
    labels = {
        canonical_yolo_label(item.get("label") or item.get("object_label") or item.get("raw_label"))
        for item in row.get("detections") or []
        if isinstance(item, dict)
    }
    return bool(labels.intersection(HAND_LABELS) and labels.intersection(target_labels))


def _dedupe_material_evidence_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: dict[tuple[Any, ...], int] = {}
    for row in rows:
        key = (
            evidence_view(row),
            round(_material_evidence_time(row), 3),
            str(row.get("frame_index") or ""),
            _material_evidence_label_signature(row),
        )
        if key in seen:
            existing_index = seen[key]
            if _material_evidence_completeness_score(row) > _material_evidence_completeness_score(deduped[existing_index]):
                deduped[existing_index] = row
            continue
        seen[key] = len(deduped)
        deduped.append(row)
    return deduped


def _material_evidence_completeness_score(row: dict[str, Any]) -> int:
    score = 0
    if _safe_float(row.get("frame_width"), 0.0) > 0 and _safe_float(row.get("frame_height"), 0.0) > 0:
        score += 16
    if row.get("time_sec") is not None:
        score += 4
    if row.get("local_time_sec") is not None:
        score += 2
    score += min(4, len([item for item in row.get("detections") or [] if isinstance(item, dict)]))
    score += min(4, len([item for item in row.get("hand_object_interactions") or [] if isinstance(item, dict)]))
    return score


def _material_evidence_label_signature(row: dict[str, Any]) -> tuple[str, ...]:
    labels: list[str] = []
    for item in row.get("detections") or []:
        if isinstance(item, dict):
            label = canonical_yolo_label(item.get("label") or item.get("object_label") or item.get("raw_label"))
            if label:
                labels.append(label)
    return tuple(sorted(labels))


def _compact_yolo_evidence_rows(rows: list[dict[str, Any]], *, limit: int = 8) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for row in rows[: max(0, limit)]:
        compact.append(
            {
                "view": evidence_view(row) or None,
                "time_sec": _material_evidence_time(row),
                "local_time_sec": _safe_float(row.get("local_time_sec"), _material_evidence_time(row)),
                "frame_index": row.get("frame_index"),
                "frame_width": int(_safe_float(row.get("frame_width"), 0.0)) or None,
                "frame_height": int(_safe_float(row.get("frame_height"), 0.0)) or None,
                "detections": [
                    _compact_detection(item)
                    for item in row.get("detections") or []
                    if isinstance(item, dict)
                ],
                "hand_object_interactions": [
                    _compact_interaction(item)
                    for item in row.get("hand_object_interactions") or []
                    if isinstance(item, dict)
                ],
            }
        )
    return compact


def _compact_detection(detection: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": canonical_yolo_label(detection.get("label") or detection.get("object_label") or detection.get("raw_label")),
        "confidence": _safe_float(detection.get("confidence"), _safe_float(detection.get("score"), 0.0)),
        "bbox": detection.get("bbox"),
    }


def _compact_interaction(interaction: dict[str, Any]) -> dict[str, Any]:
    return {
        "object_label": canonical_yolo_label(
            interaction.get("object_label") or interaction.get("target_label") or interaction.get("object")
        ),
        "score": _safe_float(interaction.get("score"), _safe_float(interaction.get("interaction_score"), 0.0)),
        "hand_bbox": interaction.get("hand_bbox"),
        "object_bbox": interaction.get("object_bbox"),
    }


def _add_clip_material_records(
    *,
    micro: dict[str, Any],
    segment: dict[str, Any],
    source_clip: Path,
    clip_dir: Path,
    used_names: set[str],
    action_name: str,
    file_date: str,
    filename_base: str,
    view: str,
    offset: float,
    duration: float,
    frame_rows: list[tuple[str, dict[str, Any]]],
    start_sec: float,
    segment_start: float,
    view_evidence: list[dict[str, Any]],
    annotation_target: str,
    annotation_target_labels: list[str],
    tracklet_summary: dict[str, Any],
    semantic_fields: dict[str, Any],
    physical_evidence_mode: str,
    valid_evidence_count: int,
    usable_evidence_count: int,
    evidence_diagnostics: dict[str, Any],
    render_yolo_annotations: bool,
    planned_records: list[dict[str, Any]],
    material_tasks: list[MaterialGenerationTask],
    dry_run: bool,
    ffmpeg_ok: bool,
    ffmpeg_path: str | Path,
) -> None:
    peak_row = frame_rows[min(1, len(frame_rows) - 1)][1] if frame_rows else {"local_time_sec": start_sec}
    clip_windows = [
        ("micro_clip", offset, duration),
        ("peak_clip", max(0.0, _safe_float(peak_row.get("local_time_sec"), start_sec) - segment_start - 0.8), min(1.6, duration)),
    ]
    for role, clip_offset, clip_duration in clip_windows:
        target = clip_dir / _unique_name(used_names, filename_base or f"{action_name}_{file_date}", ".mp4")
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
            semantic_fields=semantic_fields,
        )
        row.update(
            {
                "role": role,
                "source_offset_sec": clip_offset,
                "source_duration_sec": clip_duration,
                "annotation_target_labels": annotation_target_labels,
                "yolo_annotation_mode": "tracklet_interpolated" if render_yolo_annotations else "paired_view_time_alignment",
                "yolo_tracklet_summary": tracklet_summary,
                "yolo_box_required": bool(render_yolo_annotations),
                "box_filter": "hand_and_primary_object_only" if render_yolo_annotations else "paired_view_no_boxes",
                **_material_evidence_record_fields(
                    view_evidence,
                    physical_evidence_mode=physical_evidence_mode,
                    valid_evidence_count=valid_evidence_count,
                    usable_evidence_count=usable_evidence_count,
                    evidence_diagnostics=evidence_diagnostics,
                ),
            }
        )
        planned_records.append(row)
        if dry_run:
            continue

        def _task(
            *,
            row: dict[str, Any] = row,
            target: Path = target,
            clip_offset: float = clip_offset,
            clip_duration: float = clip_duration,
        ) -> tuple[dict[str, Any], Path, bool, str | None]:
            generated, error = _render_material_clip_result(
                source_clip=source_clip,
                clip_offset=clip_offset,
                clip_duration=clip_duration,
                target=target,
                view_evidence=view_evidence,
                annotation_target=annotation_target,
                segment_start=segment_start,
                render_yolo_annotations=render_yolo_annotations,
                ffmpeg_ok=ffmpeg_ok,
                ffmpeg_path=ffmpeg_path,
            )
            return row, target, generated, error

        material_tasks.append(_task)


def _add_keyframe_material_records(
    *,
    micro: dict[str, Any],
    segment: dict[str, Any],
    source_clip: Path,
    keyframe_dir: Path,
    used_names: set[str],
    action_name: str,
    file_date: str,
    filename_base: str,
    view: str,
    frame_rows: list[tuple[str, dict[str, Any]]],
    start_sec: float,
    segment_start: float,
    annotation_target: str,
    annotation_target_labels: list[str],
    tracklet_summary: dict[str, Any],
    semantic_fields: dict[str, Any],
    physical_evidence_mode: str,
    valid_evidence_count: int,
    usable_evidence_count: int,
    evidence_diagnostics: dict[str, Any],
    render_yolo_annotations: bool,
    planned_records: list[dict[str, Any]],
    material_tasks: list[MaterialGenerationTask],
    dry_run: bool,
    ffmpeg_ok: bool,
    ffmpeg_path: str | Path,
) -> None:
    for frame_type, evidence_row in frame_rows:
        local_time = _safe_float(evidence_row.get("local_time_sec"), start_sec)
        target = keyframe_dir / _unique_name(used_names, filename_base or f"{action_name}_{file_date}", ".jpg")
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
            semantic_fields=semantic_fields,
        )
        row.update(
            {
                "source_offset_sec": max(0.0, local_time - segment_start),
                "annotation_target_labels": annotation_target_labels,
                "yolo_annotation_mode": "evidence_frame" if render_yolo_annotations else "paired_view_time_alignment",
                "yolo_tracklet_summary": tracklet_summary,
                "yolo_box_required": bool(render_yolo_annotations),
                "box_filter": "hand_and_primary_object_only" if render_yolo_annotations else "paired_view_no_boxes",
                **_material_evidence_record_fields(
                    [evidence_row],
                    physical_evidence_mode=physical_evidence_mode,
                    valid_evidence_count=valid_evidence_count,
                    usable_evidence_count=usable_evidence_count,
                    evidence_diagnostics=evidence_diagnostics,
                ),
            }
        )
        planned_records.append(row)
        if dry_run:
            continue

        def _task(
            *,
            row: dict[str, Any] = row,
            target: Path = target,
            evidence_row: dict[str, Any] = evidence_row,
        ) -> tuple[dict[str, Any], Path, bool, str | None]:
            generated, error = _render_material_keyframe_result(
                source_clip=source_clip,
                source_offset_sec=row["source_offset_sec"],
                target=target,
                evidence_row=evidence_row,
                annotation_target=annotation_target,
                render_yolo_annotations=render_yolo_annotations,
                ffmpeg_ok=ffmpeg_ok,
                ffmpeg_path=ffmpeg_path,
            )
            return row, target, generated, error

        material_tasks.append(_task)


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

    total_start = time.perf_counter()
    timing_sec: dict[str, float] = {}

    def _mark_timing(name: str, started_at: float) -> float:
        timing_sec[name] = round(time.perf_counter() - started_at, 3)
        return time.perf_counter()

    session_root = Path(session_dir)
    stage_start = time.perf_counter()
    source_root = material_references_root(session_root)
    source_index = source_root / f"{MATERIAL_INDEX_BASENAME}.jsonl"
    candidate_root = material_candidates_root(session_root)
    _existing_candidate_rows, existing_status_by_id, preserved_candidate_rows = _load_existing_candidate_review_rows(candidate_root)
    micro_source_rows = _micro_level_material_reference_rows(session_root)
    stage_start = _mark_timing("micro_asset_source_rows_sec", stage_start)
    source_rebuild_skipped_reason = ""
    if micro_source_rows:
        source_rebuild_skipped_reason = "micro_keyframes_and_key_clips_available"
    source_index_empty = (not source_index.exists()) or source_index.stat().st_size <= 0
    if (rebuild_source or source_index_empty) and not micro_source_rows:
        build_yolo_material_references(
            session_root,
            dry_run=dry_run,
            ffmpeg_path=ffmpeg_path,
            archive_existing=archive_existing,
        )
    stage_start = _mark_timing("legacy_source_rebuild_sec", stage_start)

    source_rows, segment_fallback_used = _candidate_source_rows(session_root, source_index, micro_source_rows=micro_source_rows)
    keyframe_dir, clip_dir = _prepare_candidate_build_dirs(session_root, candidate_root, archive_existing=archive_existing)
    candidate_rows, skipped = _build_candidate_rows_from_source_rows(
        source_rows,
        source_root,
        keyframe_dir=keyframe_dir,
        clip_dir=clip_dir,
        existing_status_by_id=existing_status_by_id,
        dry_run=dry_run,
    )
    stage_start = _mark_timing("candidate_asset_copy_sec", stage_start)

    _mark_recommended_candidates(candidate_rows)
    micro_path = session_root / "metadata" / "micro_segments.jsonl"
    micro_rows = read_jsonl(micro_path) if micro_path.exists() else []
    micro_by_id = {
        str(row.get("micro_segment_id") or ""): row
        for row in micro_rows
        if isinstance(row, dict) and str(row.get("micro_segment_id") or "")
    }
    pipeline_summary = apply_yolo_vlm_review_pipeline(
        session_root,
        candidate_rows,
        micro_rows,
        vlm_client=vlm_client,
        enable_vlm=enable_vlm,
        max_vlm_groups=max_vlm_groups,
        vlm_model_name=vlm_model_name,
    )
    stage_start = _mark_timing("yolo_vlm_review_pipeline_sec", stage_start)
    _refresh_candidate_semantics(candidate_rows, micro_by_id, yolo_frame_rows=_material_yolo_frame_rows(session_root))
    stage_start = _mark_timing("semantic_refresh_sec", stage_start)
    event_candidate_rows, event_skipped, event_summary = _build_event_backed_candidate_rows(
        session_root,
        keyframe_dir=keyframe_dir,
        clip_dir=clip_dir,
        existing_status_by_id=existing_status_by_id,
        dry_run=dry_run,
    )
    stage_start = _mark_timing("event_backed_candidates_sec", stage_start)
    candidate_rows.extend(event_candidate_rows)
    skipped.extend(event_skipped)
    view_action_rows, view_action_skipped, view_action_summary = _build_view_action_review_candidate_rows(
        session_root,
        keyframe_dir=keyframe_dir,
        clip_dir=clip_dir,
        existing_status_by_id=existing_status_by_id,
        dry_run=dry_run,
        ffmpeg_path=ffmpeg_path,
    )
    stage_start = _mark_timing("view_action_review_candidates_sec", stage_start)
    candidate_rows.extend(view_action_rows)
    skipped.extend(view_action_skipped)
    candidate_rows.extend(preserved_candidate_rows)
    asset_counts = _candidate_asset_counts(candidate_rows)
    pipeline_summary = {
        **pipeline_summary,
        "candidate_count": len(candidate_rows),
        "group_count": asset_counts["candidate_group_count"],
        "candidate_group_count": asset_counts["candidate_group_count"],
        "keyframe_count": asset_counts["keyframe_count"],
        "key_clip_count": asset_counts["key_clip_count"],
        "event_backed_candidates": event_summary,
        "view_action_review_candidates": view_action_summary,
        "parallel_workers": _material_candidate_worker_count(),
        "source_rebuild_skipped_reason": source_rebuild_skipped_reason,
        "micro_asset_source_count": len(micro_source_rows),
    }
    _mark_recommended_candidates(candidate_rows)
    if not dry_run:
        _run_material_candidate_tasks(
            [
                lambda candidate=candidate: _rerender_corrected_candidate_keyframe(session_root, candidate, micro_by_id)
                for candidate in candidate_rows
            ]
        )
        keyframe_quality_report = _refresh_candidate_keyframe_quality(
            candidate_rows,
            session_root / "metadata" / "keyframe_quality_report.json",
        )
        pipeline_summary["keyframe_quality"] = keyframe_quality_report
    stage_start = _mark_timing("candidate_keyframe_refresh_sec", stage_start)

    index_json = candidate_root / f"{MATERIAL_CANDIDATE_INDEX_BASENAME}.json"
    index_jsonl = candidate_root / f"{MATERIAL_CANDIDATE_INDEX_BASENAME}.jsonl"
    timing_sec["total_sec"] = round(time.perf_counter() - total_start, 3)
    pipeline_summary["timing_sec"] = timing_sec
    summary = _material_candidate_summary(
        session_root,
        candidate_root,
        keyframe_dir,
        clip_dir,
        index_json,
        index_jsonl,
        candidate_rows,
        skipped,
        pipeline_summary,
        dry_run=dry_run,
        segment_fallback_used=segment_fallback_used,
        preserved_candidate_count=len(preserved_candidate_rows),
    )
    _write_jsonl(index_jsonl, candidate_rows)
    _write_json(index_json, summary)
    _write_json(candidate_root / "pipeline_summary.json", pipeline_summary)
    _write_json(candidate_root / "manifest.json", _candidate_manifest(summary))
    _write_candidate_readme(candidate_root / "README.md", summary)
    timing_sec["write_index_sec"] = round(time.perf_counter() - stage_start, 3)
    external_candidate_publish = {} if dry_run else _mirror_review_candidates_to_external_library(session_root, candidate_root, candidate_rows)
    pipeline_summary["external_review_candidate_library"] = external_candidate_publish
    timing_sec["total_sec"] = round(time.perf_counter() - total_start, 3)
    summary["pipeline_summary"] = pipeline_summary
    summary["timing_sec"] = timing_sec
    _write_json(index_json, summary)
    _write_json(candidate_root / "pipeline_summary.json", pipeline_summary)
    _write_material_stream(candidate_root, candidate_rows, session_root=session_root)
    _write_jsonl(
        session_root / "metadata" / "review_candidate_materials.jsonl",
        [
            row
            for row in candidate_rows
            if str(row.get("candidate_source") or "") == "view_action_evidence_needs_review"
            or str(row.get("review_status") or "") == "needs_review"
        ],
    )
    _write_jsonl(
        session_root / "metadata" / "high_confidence_materials.jsonl",
        [
            row
            for row in candidate_rows
            if bool(row.get("official_material")) and bool(row.get("memory_write_allowed"))
        ],
    )
    _write_json(
        session_root / "metadata" / "material_candidate_review_manifest.json",
        {
            "schema_version": "material_candidate_review_manifest.v1",
            "session_dir": str(session_root),
            "candidate_root": str(candidate_root),
            "candidate_count": len(candidate_rows),
            "official_material_count": len(
                [row for row in candidate_rows if bool(row.get("official_material")) and bool(row.get("memory_write_allowed"))]
            ),
            "needs_review_material_count": len(
                [
                    row
                    for row in candidate_rows
                    if str(row.get("candidate_source") or "") == "view_action_evidence_needs_review"
                    or str(row.get("review_status") or "") == "needs_review"
                ]
            ),
            "view_action_review_candidates": view_action_summary,
            "material_stream": str(candidate_root / "material_stream.jsonl"),
            "external_review_candidate_library": external_candidate_publish,
            "memory_policy": "official_materials_only",
        },
    )
    _write_json(
        session_root / "metadata" / "key_material_fix_report.json",
        {
            "schema_version": "key_material_fix_report.v1",
            "fix": "surface_view_action_evidence_as_needs_review_candidates",
            "official_material_count": len(
                [row for row in candidate_rows if bool(row.get("official_material")) and bool(row.get("memory_write_allowed"))]
            ),
            "review_candidate_material_count": len(
                [
                    row
                    for row in candidate_rows
                    if str(row.get("candidate_source") or "") == "view_action_evidence_needs_review"
                    or str(row.get("review_status") or "") == "needs_review"
                ]
            ),
            "view_action_review_candidates": view_action_summary,
            "publish_policy": "review_candidates_visible_but_not_published_to_memory",
        },
    )
    _write_p0_material_algorithm_reports(session_root, candidate_rows)
    return summary


def _write_p0_material_algorithm_reports(session_root: Path, candidate_rows: list[dict[str, Any]]) -> None:
    metadata = session_root / "metadata"
    formal_payload = _load_json(metadata / "formal_experiment_windows.json")
    formal_windows = formal_payload.get("windows") if isinstance(formal_payload, dict) else []
    if not isinstance(formal_windows, list):
        formal_windows = []
    window_status_by_id = {
        str(window.get("experiment_window_id") or window.get("window_id")): str(
            window.get("status") or window.get("visual_review_status") or ""
        )
        for window in formal_windows
        if isinstance(window, dict) and (window.get("experiment_window_id") or window.get("window_id"))
    }
    review_rows = [
        row
        for row in candidate_rows
        if str(row.get("candidate_source") or "") == "view_action_evidence_needs_review"
        or str(row.get("review_status") or "") == "needs_review"
    ]
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in review_rows:
        bundle_id = str(row.get("evidence_bundle_id") or row.get("candidate_group_id") or row.get("candidate_id") or "")
        if bundle_id:
            groups.setdefault(bundle_id, []).append(row)

    action_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    pairing_rows: list[dict[str, Any]] = []
    material_validation_rows: list[dict[str, Any]] = []
    keyclip_rows: list[dict[str, Any]] = []
    for bundle_id, rows in sorted(groups.items()):
        first_rows = [row for row in rows if str(row.get("view") or "") == "first_person"]
        third_rows = [row for row in rows if str(row.get("view") or "") == "third_person"]
        window_ids = sorted({str(row.get("experiment_window_id") or "") for row in rows if row.get("experiment_window_id")})
        sync_paths = sorted({str(row.get("source_window_sync_index") or "") for row in rows if row.get("source_window_sync_index")})
        action_type = str(rows[0].get("action_type") or rows[0].get("event_type") or "unknown_action")
        consistency = str(rows[0].get("cross_view_consistency") or "")
        status, reason = _p0_action_phase_status(
            first_rows,
            third_rows,
            window_ids,
            sync_paths,
            consistency,
            window_statuses=[window_status_by_id.get(window_id, "") for window_id in window_ids],
        )
        start_values = [_safe_float(row.get("start_sec"), 0.0) for row in rows]
        end_values = [_safe_float(row.get("end_sec"), 0.0) for row in rows]
        action_rows.append(
            {
                "schema_version": "action_candidate_row.v1",
                "action_candidate_id": bundle_id,
                "evidence_bundle_id": bundle_id,
                "action_type": action_type,
                "experiment_window_id": window_ids[0] if len(window_ids) == 1 else None,
                "window_ids": window_ids,
                "start_sec": round(min(start_values), 6) if start_values else None,
                "end_sec": round(max(end_values), 6) if end_values else None,
                "first_evidence_count": len(first_rows),
                "third_evidence_count": len(third_rows),
                "source_window_sync_index": sync_paths[0] if len(sync_paths) == 1 else None,
                "window_status": window_status_by_id.get(window_ids[0], "") if len(window_ids) == 1 else None,
                "status": status,
                "reason": reason,
                "confidence": round(
                    sum(_safe_float(row.get("event_confidence") or row.get("quality_score"), 0.0) for row in rows)
                    / max(1, len(rows)),
                    6,
                ),
            }
        )
        phase_rows.append(
            {
                "schema_version": "dual_view_action_phase.item.v1",
                "dual_event_id": bundle_id,
                "evidence_bundle_id": bundle_id,
                "action_type": action_type,
                "window_id": window_ids[0] if len(window_ids) == 1 else None,
                "first_evidence_exists": bool(first_rows),
                "third_evidence_exists": bool(third_rows),
                "first_evidence_type": _asset_kinds(first_rows),
                "third_evidence_type": _asset_kinds(third_rows),
                "first_frame_refs": _stored_files(first_rows, KEYFRAME_DIR_NAME),
                "third_frame_refs": _stored_files(third_rows, KEYFRAME_DIR_NAME),
                "first_clip_refs": _stored_files(first_rows, KEY_CLIP_DIR_NAME),
                "third_clip_refs": _stored_files(third_rows, KEY_CLIP_DIR_NAME),
                "source_window_sync_index": sync_paths[0] if len(sync_paths) == 1 else None,
                "window_status": window_status_by_id.get(window_ids[0], "") if len(window_ids) == 1 else None,
                "visual_phase_match": status == "dual_view_valid",
                "spatial_relation_match": status == "dual_view_valid",
                "status": status,
                "reject_reason_if_any": [] if status in {"dual_view_valid", "first_dominant_with_reason"} else [reason],
                "policy": "Timestamp proximity is not enough; official promotion still requires visual/action validation.",
            }
        )
        pairing_rows.append(_dual_view_material_pairing_row(bundle_id, action_type, rows, status, reason))
        trace_rows.append(
            {
                "schema_version": "action_trace_debug.v1",
                "evidence_bundle_id": bundle_id,
                "action_type": action_type,
                "status": status,
                "reason": reason,
                "rows": [
                    {
                        "candidate_id": row.get("candidate_id"),
                        "view": row.get("view"),
                        "asset_kind": row.get("asset_kind"),
                        "stored_file": row.get("stored_file"),
                        "source_window_sync_index": row.get("source_window_sync_index"),
                        "start_sec": row.get("start_sec"),
                        "end_sec": row.get("end_sec"),
                        "selected_keyframe_score": row.get("selected_keyframe_score"),
                    }
                    for row in rows
                ],
            }
        )
        for row in rows:
            validation = _p0_material_validation_row(row)
            material_validation_rows.append(validation)
            if str(row.get("asset_kind") or "") == KEY_CLIP_DIR_NAME:
                keyclip_rows.append(_p0_keyclip_quality_row(row, validation))

    _write_jsonl(metadata / "action_candidate_rows.jsonl", action_rows)
    _write_jsonl(metadata / "detected_actions.jsonl", action_rows)
    _write_jsonl(metadata / "action_trace_debug.jsonl", trace_rows)
    phase_report = {
        "schema_version": "dual_view_action_phase_report.v1",
        "event_count": len(phase_rows),
        "status_counts": _count_values(phase_rows, "status"),
        "events": phase_rows,
        "policy": "dual_view_valid means the event is aligned inside a validated window; official promotion still requires the material publish policy and/or human confirmation.",
    }
    _write_json(metadata / "dual_view_action_phase_report.json", phase_report)
    _write_json(
        metadata / "dual_view_material_alignment_audit.json",
        {
            "schema_version": "dual_view_material_alignment_audit.v1",
            "event_count": len(phase_rows),
            "dual_view_valid_count": len([row for row in phase_rows if row.get("status") == "dual_view_valid"]),
            "first_dominant_count": len([row for row in phase_rows if row.get("status") == "first_dominant_with_reason"]),
            "suspicious_count": len([row for row in phase_rows if row.get("status") == "suspicious_needs_review"]),
            "rejected_count": len([row for row in phase_rows if str(row.get("status") or "").startswith("rejected")]),
            "events": phase_rows,
        },
    )
    _write_json(
        metadata / "dual_view_material_pairing_report.json",
        {
            "schema_version": "dual_view_material_pairing_report.v1",
            "material_group_count": len(pairing_rows),
            "paired_valid_count": len([row for row in pairing_rows if row.get("pairing_status") == "dual_view_valid"]),
            "mismatch_or_suspicious_count": len([row for row in pairing_rows if row.get("pairing_status") != "dual_view_valid"]),
            "rows": pairing_rows,
            "policy": "First/third keyframes and keyclips must be generated from the same or nearby window_sync_index rows; otherwise the material cannot be presented as normal aligned dual-view evidence.",
        },
    )
    extraction_report = {
        "schema_version": "key_material_extraction_algorithm_report.v1",
        "candidate_event_count": len(action_rows),
        "candidate_material_count": len(review_rows),
        "official_material_count": len([row for row in review_rows if bool(row.get("official_material"))]),
        "status_counts": _count_values(action_rows, "status"),
        "focused_actions": ["hand_object_contact", "object_move", "device_panel_interaction"],
        "fix_summary": [
            "paired first/third evidence in validated windows can become dual_view_valid",
            "single-view evidence remains first_dominant_with_reason or suspicious",
            "missing window/sync evidence is rejected",
        ],
    }
    _write_json(metadata / "key_material_extraction_algorithm_report.json", extraction_report)
    _write_json(metadata / "action_extraction_fix_report.json", extraction_report)
    _write_json(
        metadata / "false_positive_events_report.json",
        {
            "schema_version": "false_positive_events_report.v1",
            "status": "requires_expected_actions_or_visual_review",
            "suspicious_event_count": len([row for row in action_rows if str(row.get("status")) == "suspicious_needs_review"]),
            "events": [row for row in action_rows if str(row.get("status")) == "suspicious_needs_review"],
        },
    )
    _write_json(
        metadata / "missed_events_report.json",
        {
            "schema_version": "missed_events_report.v1",
            "status": "not_computed_without_expected_actions",
            "recommendation": "Provide expected_actions.json to compute missed events, precision, recall, and F1.",
        },
    )
    _write_json(
        metadata / "keyclip_quality_report.json",
        {
            "schema_version": "keyclip_quality_report.v1",
            "keyclip_count": len(keyclip_rows),
            "rows": keyclip_rows,
            "policy": "Keyclips must come from the same source_window_sync_index as their material group.",
        },
    )
    keyclip_timing_rows = _keyclip_timing_fix_rows(session_root, keyclip_rows)
    _write_json(
        metadata / "keyclip_timing_fix_report.json",
        {
            "schema_version": "keyclip_timing_fix_report.v1",
            "keyclip_count": len(keyclip_rows),
            "material_count": len(keyclip_timing_rows),
            "matching_duration_count": sum(1 for row in keyclip_timing_rows if row.get("status") == "pass"),
            "needs_review_count": sum(1 for row in keyclip_timing_rows if row.get("status") != "pass"),
            "rows": keyclip_timing_rows,
            "policy": "First, third, and side-by-side keyclips must be generated from the same window_sync_index row range and preserve the real action duration.",
        },
    )
    _write_json(
        metadata / "material_self_validation_report.json",
        {
            "schema_version": "material_self_validation_report.v1",
            "material_count": len(material_validation_rows),
            "pass_count": sum(1 for row in material_validation_rows if row.get("validation_status") == "pass"),
            "needs_review_count": sum(1 for row in material_validation_rows if row.get("validation_status") != "pass"),
            "rows": material_validation_rows,
        },
    )


def _p0_action_phase_status(
    first_rows: list[dict[str, Any]],
    third_rows: list[dict[str, Any]],
    window_ids: list[str],
    sync_paths: list[str],
    consistency: str,
    *,
    window_statuses: list[str] | None = None,
) -> tuple[str, str]:
    if not window_ids:
        return "rejected_missing_window", "missing_experiment_window_id"
    if not sync_paths:
        return "rejected_missing_sync_index", "missing_source_window_sync_index"
    normalized_window_statuses = {str(status or "").lower() for status in (window_statuses or []) if status}
    if normalized_window_statuses and "validated_formal" not in normalized_window_statuses:
        return "suspicious_needs_review", "window_not_validated_formal"
    if first_rows and third_rows:
        first_has_keyframe = bool(_stored_files(first_rows, KEYFRAME_DIR_NAME))
        third_has_keyframe = bool(_stored_files(third_rows, KEYFRAME_DIR_NAME))
        first_keyclip_status = _keyclip_evidence_status(first_rows)
        third_keyclip_status = _keyclip_evidence_status(third_rows)
        first_has_keyclip = first_keyclip_status["valid"]
        third_has_keyclip = third_keyclip_status["valid"]
        if first_has_keyframe and third_has_keyframe and first_has_keyclip and third_has_keyclip:
            return "dual_view_valid", "validated_window_with_paired_keyframe_keyclip_evidence"
        if not first_keyclip_status["valid"] or not third_keyclip_status["valid"]:
            return (
                "suspicious_needs_review",
                "paired_evidence_keyclip_invalid:"
                + ",".join([*first_keyclip_status["issues"], *third_keyclip_status["issues"]]),
            )
        return "suspicious_needs_review", "paired_evidence_missing_keyframe_or_keyclip"
    if first_rows and not third_rows:
        return "first_dominant_with_reason", "first_view_has_evidence_third_missing_or_off_bench"
    if not first_rows and third_rows:
        return "suspicious_needs_review", "third_only_evidence_cannot_be_dual_view"
    return "suspicious_needs_review", "action_phase_not_validated"


def _dual_view_material_pairing_row(
    bundle_id: str,
    action_type: str,
    rows: list[dict[str, Any]],
    status: str,
    reason: str,
) -> dict[str, Any]:
    first_rows = [row for row in rows if str(row.get("view") or "") == "first_person"]
    third_rows = [row for row in rows if str(row.get("view") or "") == "third_person"]
    first_keyframes = [row for row in first_rows if str(row.get("asset_kind") or "") == KEYFRAME_DIR_NAME]
    third_keyframes = [row for row in third_rows if str(row.get("asset_kind") or "") == KEYFRAME_DIR_NAME]
    first_keyclips = [row for row in first_rows if str(row.get("asset_kind") or "") == KEY_CLIP_DIR_NAME]
    third_keyclips = [row for row in third_rows if str(row.get("asset_kind") or "") == KEY_CLIP_DIR_NAME]
    first_keyframe_row = _first_present_sync_row(first_keyframes, "peak_window_sync_index")
    third_keyframe_row = _first_present_sync_row(third_keyframes, "peak_window_sync_index")
    first_clip_range = _sync_range_for_rows(first_keyclips)
    third_clip_range = _sync_range_for_rows(third_keyclips)
    row_range_match = bool(first_clip_range and third_clip_range and first_clip_range == third_clip_range)
    timestamp_delta_ms = None
    first_ts = _first_present_int(first_keyframes, "global_timestamp_us")
    third_ts = _first_present_int(third_keyframes, "global_timestamp_us")
    if first_ts is not None and third_ts is not None:
        timestamp_delta_ms = round(abs(first_ts - third_ts) / 1000.0, 6)
    pairing_status = status
    pairing_reason = reason
    if status == "dual_view_valid" and not row_range_match:
        pairing_status = "suspicious_needs_review"
        pairing_reason = "first_third_keyclip_row_range_mismatch"
    if status == "dual_view_valid" and first_keyframe_row is not None and third_keyframe_row is not None:
        if abs(int(first_keyframe_row) - int(third_keyframe_row)) > 1:
            pairing_status = "suspicious_needs_review"
            pairing_reason = "first_third_keyframe_sync_row_mismatch"
    sync_paths = sorted({str(row.get("source_window_sync_index") or "") for row in rows if row.get("source_window_sync_index")})
    window_ids = sorted({str(row.get("experiment_window_id") or "") for row in rows if row.get("experiment_window_id")})
    return {
        "schema_version": "dual_view_material_pairing.item.v1",
        "material_id": bundle_id,
        "evidence_bundle_id": bundle_id,
        "action_type": action_type,
        "window_id": window_ids[0] if len(window_ids) == 1 else None,
        "source_window_sync_index": sync_paths[0] if len(sync_paths) == 1 else None,
        "first_keyframe_row": first_keyframe_row,
        "third_keyframe_row": third_keyframe_row,
        "first_keyclip_row_range": first_clip_range,
        "third_keyclip_row_range": third_clip_range,
        "row_range_match": row_range_match,
        "timestamp_delta_ms": timestamp_delta_ms,
        "pairing_status": pairing_status,
        "reason": pairing_reason,
    }


def _first_present_sync_row(rows: list[Mapping[str, Any]], key: str) -> int | None:
    return _first_present_int(rows, key)


def _first_present_int(rows: list[Mapping[str, Any]], key: str) -> int | None:
    for row in rows:
        try:
            value = row.get(key)
            if value is None or value == "":
                continue
            return int(float(str(value)))
        except Exception:
            continue
    return None


def _sync_range_for_rows(rows: list[Mapping[str, Any]]) -> list[int] | None:
    if not rows:
        return None
    start = _first_present_int(rows, "start_window_sync_index")
    end = _first_present_int(rows, "end_window_sync_index")
    if start is None or end is None:
        return None
    return [start, end]


def _keyclip_evidence_status(rows: list[dict[str, Any]]) -> dict[str, Any]:
    clip_rows = [row for row in rows if str(row.get("asset_kind") or "") == KEY_CLIP_DIR_NAME]
    issues: list[str] = []
    if not clip_rows:
        return {"valid": False, "issues": ["missing_keyclip"]}
    valid_count = 0
    for row in clip_rows:
        stored = Path(str(row.get("stored_file") or ""))
        if not stored.is_file():
            issues.append("keyclip_file_missing")
            continue
        duration = max(0.0, _safe_float(row.get("end_sec"), 0.0) - _safe_float(row.get("start_sec"), 0.0))
        if duration <= 0:
            issues.append("non_positive_keyclip_duration")
            continue
        if not row.get("source_window_sync_index"):
            issues.append("missing_source_window_sync_index")
            continue
        start_row = _first_present_int([row], "start_window_sync_index")
        end_row = _first_present_int([row], "end_window_sync_index")
        if start_row is None or end_row is None or end_row < start_row:
            issues.append("missing_or_invalid_window_sync_row_range")
            continue
        valid_count += 1
    return {"valid": valid_count > 0, "issues": sorted(set(issues))}


def _asset_kinds(rows: list[dict[str, Any]]) -> list[str]:
    return sorted({str(row.get("asset_kind") or row.get("material_type") or "") for row in rows if row.get("asset_kind") or row.get("material_type")})


def _stored_files(rows: list[dict[str, Any]], asset_kind: str) -> list[str]:
    return [str(row.get("stored_file")) for row in rows if str(row.get("asset_kind") or "") == asset_kind and row.get("stored_file")]


def _p0_material_validation_row(row: Mapping[str, Any]) -> dict[str, Any]:
    missing: list[str] = []
    for key in ("experiment_window_id", "source_window_sync_index", "stored_file", "evidence_bundle_id"):
        if not row.get(key):
            missing.append(f"missing_{key}")
    stored = Path(str(row.get("stored_file") or ""))
    if not stored.is_file():
        missing.append("stored_file_missing")
    if row.get("placeholder"):
        missing.append("placeholder_or_non_real_material")
    return {
        "material_id": row.get("material_id") or row.get("candidate_id"),
        "candidate_id": row.get("candidate_id"),
        "evidence_bundle_id": row.get("evidence_bundle_id"),
        "action_type": row.get("action_type"),
        "asset_kind": row.get("asset_kind"),
        "view": row.get("view"),
        "experiment_window_id": row.get("experiment_window_id"),
        "source_window_sync_index": row.get("source_window_sync_index"),
        "stored_file": row.get("stored_file"),
        "selected_keyframe_score": row.get("selected_keyframe_score"),
        "validation_status": "pass" if not missing else "needs_review",
        "issues": missing,
    }


def _p0_keyclip_quality_row(row: Mapping[str, Any], validation: Mapping[str, Any]) -> dict[str, Any]:
    stored = Path(str(row.get("stored_file") or ""))
    duration = max(0.0, _safe_float(row.get("end_sec"), 0.0) - _safe_float(row.get("start_sec"), 0.0))
    media = _video_duration_metadata(stored)
    actual_duration = _safe_float(media.get("duration_s"), 0.0)
    playback_speed_ratio = round(duration / actual_duration, 6) if actual_duration > 0 else None
    side_path = Path(str(row.get("side_by_side_keyclip") or ""))
    side_media = _video_duration_metadata(side_path) if side_path else {}
    side_duration = _safe_float(side_media.get("duration_s"), 0.0)
    issues = list(validation.get("issues") or [])
    if duration <= 0:
        issues.append("non_positive_clip_duration")
    if playback_speed_ratio is not None and playback_speed_ratio > 1.25:
        issues.append("keyclip_playback_accelerated")
    start_row = _first_present_int([row], "start_window_sync_index")
    end_row = _first_present_int([row], "end_window_sync_index")
    if start_row is None or end_row is None or end_row < start_row:
        issues.append("missing_or_invalid_window_sync_row_range")
    if side_path and side_path.is_file() and actual_duration > 0 and side_duration > 0 and abs(side_duration - actual_duration) > 0.25:
        issues.append("side_by_side_keyclip_duration_mismatch")
    return {
        "material_id": row.get("material_id") or row.get("candidate_id"),
        "candidate_id": row.get("candidate_id"),
        "evidence_bundle_id": row.get("evidence_bundle_id"),
        "view": row.get("view"),
        "stored_file": row.get("stored_file"),
        "exists": stored.is_file(),
        "duration_sec": round(duration, 6),
        "actual_clip_duration_s": round(actual_duration, 6),
        "side_by_side_keyclip": str(side_path) if side_path else "",
        "side_by_side_duration_s": round(side_duration, 6),
        "playback_speed_ratio": playback_speed_ratio,
        "source_window_sync_index": row.get("source_window_sync_index"),
        "start_window_sync_index": start_row,
        "end_window_sync_index": end_row,
        "peak_window_sync_index": _first_present_int([row], "peak_window_sync_index"),
        "generation_basis": row.get("final_time_basis") or row.get("source_time_basis"),
        "keyclip_output_fps": row.get("keyclip_output_fps"),
        "timing_status": "needs_review" if issues else "pass",
        "quality_status": "needs_review" if issues else "candidate_clip_valid",
        "issues": issues,
    }


def _keyclip_timing_fix_rows(session_root: Path, keyclip_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    material_stream_path = session_root.parent / "_material_review_queue" / "material_stream.jsonl"
    stream_rows = read_jsonl(material_stream_path) if material_stream_path.is_file() else []
    rows: list[dict[str, Any]] = []
    if stream_rows:
        for material in stream_rows:
            first_path = Path(str(material.get("first_keyclip") or ""))
            third_path = Path(str(material.get("third_keyclip") or ""))
            side_path = Path(str(material.get("side_by_side_keyclip") or ""))
            first_meta = _video_duration_metadata(first_path) if first_path else {}
            third_meta = _video_duration_metadata(third_path) if third_path else {}
            side_meta = _video_duration_metadata(side_path) if side_path else {}
            durations = [
                _safe_float(first_meta.get("duration_s"), 0.0),
                _safe_float(third_meta.get("duration_s"), 0.0),
                _safe_float(side_meta.get("duration_s"), 0.0),
            ]
            nonzero = [value for value in durations if value > 0]
            max_delta = max(nonzero) - min(nonzero) if len(nonzero) >= 2 else 0.0
            issues: list[str] = []
            if len(nonzero) < 3:
                issues.append("missing_keyclip_duration")
            if max_delta > 0.25:
                issues.append("first_third_side_by_side_duration_mismatch")
            if not material.get("source_window_sync_index"):
                issues.append("missing_source_window_sync_index")
            if material.get("start_window_sync_index") is None or material.get("end_window_sync_index") is None:
                issues.append("missing_window_sync_row_range")
            rows.append(
                {
                    "schema_version": "keyclip_timing_fix.item.v1",
                    "material_id": material.get("material_id"),
                    "window_id": material.get("experiment_window_id") or material.get("window_id"),
                    "source_window_sync_index": material.get("source_window_sync_index"),
                    "real_clip_duration_s": max(nonzero) if nonzero else 0.0,
                    "first_clip_duration_s": round(durations[0], 6),
                    "third_clip_duration_s": round(durations[1], 6),
                    "side_by_side_duration_s": round(durations[2], 6),
                    "output_fps": material.get("keyclip_output_fps"),
                    "playback_speed_ratio": 1.0 if max_delta <= 0.25 and nonzero else None,
                    "row_range": [material.get("start_window_sync_index"), material.get("end_window_sync_index")],
                    "status": "pass" if not issues else "needs_review",
                    "issues": issues,
                }
            )
        return rows
    for row in keyclip_rows:
        rows.append(
            {
                "schema_version": "keyclip_timing_fix.item.v1",
                "material_id": row.get("material_id"),
                "window_id": row.get("experiment_window_id") or row.get("window_id"),
                "source_window_sync_index": row.get("source_window_sync_index"),
                "real_clip_duration_s": row.get("duration_sec"),
                "first_clip_duration_s": row.get("actual_clip_duration_s") if row.get("view") == "first_person" else None,
                "third_clip_duration_s": row.get("actual_clip_duration_s") if row.get("view") == "third_person" else None,
                "side_by_side_duration_s": row.get("side_by_side_duration_s"),
                "output_fps": row.get("keyclip_output_fps"),
                "playback_speed_ratio": row.get("playback_speed_ratio"),
                "row_range": [row.get("start_window_sync_index"), row.get("end_window_sync_index")],
                "status": row.get("timing_status"),
                "issues": row.get("issues") or [],
            }
        )
    return rows


def _count_values(rows: list[Mapping[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "")
        counts[value] = counts.get(value, 0) + 1
    return counts


def _preserve_candidate_review_state(candidate: dict[str, Any], previous: dict[str, Any] | None) -> None:
    if not previous:
        return
    for key in (
        "candidate_disposition_schema_version",
        "candidate_status",
        "review_status",
        "review_required",
        "disposition",
        "approved_at",
        "approved_by",
        "reviewed_at",
        "reviewed_by",
        "restored_at",
        "restored_by",
        "review_notes",
        "approval_reason_code",
        "approval_reason",
        "rejection_reason_code",
        "rejection_reason",
        "previous_rejection_reason_code",
        "previous_rejection_reason",
    ):
        if key in previous:
            candidate[key] = previous[key]
    if previous.get("manual_correction"):
        for key in (
            "manual_correction",
            "action_name",
            "primary_object",
            "canonical_action_type",
            "canonical_object",
            "sop_phase",
            "interaction_family",
            "secondary_actions",
            "secondary_objects",
            "box_filter",
        ):
            if key in previous:
                candidate[key] = previous[key]


def _refresh_candidate_semantics(
    candidate_rows: list[dict[str, Any]],
    micro_by_id: dict[str, dict[str, Any]],
    *,
    yolo_frame_rows: list[dict[str, Any]] | None = None,
) -> None:
    for candidate in candidate_rows:
        if candidate.get("manual_correction"):
            continue
        micro = micro_by_id.get(str(candidate.get("micro_segment_id") or "")) or {}
        view = str(candidate.get("view") or candidate.get("camera_view") or "")
        start_sec = _safe_float(micro.get("start_sec", candidate.get("time_start")))
        end_sec = _safe_float(micro.get("end_sec", candidate.get("time_end")), start_sec)
        merged_evidence = _micro_material_evidence_rows(
            micro,
            yolo_frame_rows or [],
            primary_object=str(candidate.get("primary_object") or micro.get("primary_object") or ""),
            start_sec=start_sec,
            end_sec=end_sec,
        ) if micro else []
        evidence_rows = [
            row
            for row in (merged_evidence or micro.get("yolo_evidence") or [])
            if isinstance(row, dict) and (not view or evidence_view(row) == view)
        ] or [row for row in (merged_evidence or micro.get("yolo_evidence") or []) if isinstance(row, dict)]
        semantic_fields = enhance_material_semantics(
            candidate,
            micro=micro,
            evidence_rows=evidence_rows,
            primary_object=candidate.get("primary_object"),
            secondary_objects=_as_list_for_semantics(candidate.get("secondary_objects")),
            action_name=candidate.get("action_name"),
            vlm_semantics=candidate.get("vlm_semantics") if isinstance(candidate.get("vlm_semantics"), dict) else None,
        )
        if not semantic_fields:
            continue
        candidate.update(semantic_fields)
        if semantic_fields.get("display_title"):
            candidate["action_name"] = semantic_fields["display_title"]
        taxonomy = _semantic_taxonomy(semantic_fields)
        if taxonomy:
            candidate.update(taxonomy)
        primary_label = canonical_yolo_label(candidate.get("primary_object") or candidate.get("manipulated_object") or candidate.get("canonical_object"))
        if primary_label not in {"balance", "scale", "panel"}:
            candidate.update(_canonical_action_fields(primary_label or candidate.get("primary_object"), candidate.get("action_name")))
        semantic_display_title = candidate.get("display_title")
        action_context = _material_name_context(
            micro or candidate,
            primary_object=candidate.get("primary_object") or primary_label,
            semantic_fields=candidate,
        )
        action_name = _approved_material_chinese_action_name(action_context)
        if semantic_display_title and str(semantic_display_title).strip() != action_name:
            candidate["semantic_display_title"] = semantic_display_title
        candidate["action_name"] = action_name
        candidate["display_title"] = action_name
        candidate["box_filter"] = (
            "hand_manipulated_object_and_instrument_context_tracklet"
            if semantic_fields.get("instrument_context")
            else candidate.get("box_filter") or "hand_and_primary_object_only"
        )


def _as_list_for_semantics(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return [value]


def _rerender_corrected_candidate_keyframe(
    session_root: Path,
    candidate: dict[str, Any],
    micro_by_id: dict[str, dict[str, Any]],
) -> None:
    """Refresh corrected keyframe previews so reviewer-visible boxes match the corrected action."""

    if not _material_candidate_rerender_boxes_enabled():
        candidate["yolo_annotation_rendered"] = False
        candidate["annotation_display_mode"] = "clean_keyframe_no_burned_boxes"
        return
    if str(candidate.get("asset_kind") or candidate.get("material_type") or "") != KEYFRAME_DIR_NAME:
        return
    if not candidate.get("manual_correction") and candidate.get("box_filter") != "hand_and_corrected_target_interaction_only":
        return
    target = Path(str(candidate.get("stored_file") or ""))
    if not target.parent.exists():
        return
    source_clip = _candidate_source_clip(candidate, session_root)
    if source_clip is None or not source_clip.is_file():
        return
    micro = micro_by_id.get(str(candidate.get("micro_segment_id") or ""))
    if not micro:
        return
    view = str(candidate.get("view") or candidate.get("camera_view") or "")
    evidence_rows = [
        item
        for item in micro.get("yolo_evidence") or []
        if isinstance(item, dict) and (not view or evidence_view(item) == view)
    ]
    if not evidence_rows:
        return
    start_sec = _safe_float(micro.get("start_sec", candidate.get("time_start")))
    end_sec = _safe_float(micro.get("end_sec", candidate.get("time_end")), start_sec)
    target_labels = _candidate_target_labels(candidate)
    if not target_labels:
        return
    target_object_query = ",".join(sorted(target_labels))
    evidence_row = _select_corrected_candidate_evidence_row(
        candidate,
        evidence_rows,
        start_sec=start_sec,
        end_sec=end_sec,
        target_labels=target_labels,
    )
    if evidence_row is None:
        candidate["yolo_annotation_rendered"] = False
        candidate["rerender_error"] = f"no_hand_target_interaction_evidence:{target_object_query}"
        return
    try:
        _extract_filtered_interaction_frame(
            source_clip,
            _safe_float(candidate.get("source_offset_sec"), 0.0),
            target,
            evidence_row,
            target_object_query,
            require_boxes=True,
        )
        if target.is_file():
            candidate["size_bytes"] = target.stat().st_size
            candidate["exists"] = True
            candidate["yolo_annotation_rendered"] = True
            candidate["rerendered_from_manual_correction"] = True
    except Exception as exc:  # pragma: no cover
        candidate["yolo_annotation_rendered"] = False
        candidate["rerender_error"] = str(exc)


def _candidate_target_labels(candidate: dict[str, Any]) -> set[str]:
    for key in ("manipulated_object", "raw_primary_object", "primary_object", "canonical_object"):
        labels = _interaction_target_labels(candidate.get(key))
        if labels:
            return labels
    return set()


def _candidate_context_labels(candidate: dict[str, Any]) -> set[str]:
    labels: set[str] = set()
    for key in ("instrument_context", "secondary_objects", "corrected_primary_object"):
        labels.update(_interaction_target_labels(candidate.get(key)))
    return labels


def _select_corrected_candidate_evidence_row(
    candidate: dict[str, Any],
    evidence_rows: list[dict[str, Any]],
    *,
    start_sec: float,
    end_sec: float,
    target_labels: set[str],
) -> dict[str, Any] | None:
    if not target_labels:
        return None
    frame_rows = _evidence_frame_rows(evidence_rows, start_sec, end_sec)
    frame_type = str(candidate.get("frame_type") or candidate.get("frame_role") or "")
    frame_row_by_role = {role: row for role, row in frame_rows}
    preferred_row = frame_row_by_role.get(frame_type)
    preferred_time = _evidence_local_time(preferred_row, (start_sec + end_sec) / 2.0)
    candidates = []
    for row in evidence_rows:
        if not isinstance(row, dict):
            continue
        score = _target_interaction_score(row, target_labels)
        if score <= 0.0:
            continue
        local_time = _evidence_local_time(row, preferred_time)
        same_role = row is preferred_row
        candidates.append((same_role, score, -abs(local_time - preferred_time), row))
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1], item[2]))[3]


def _target_interaction_score(evidence_row: dict[str, Any], target_labels: set[str]) -> float:
    interactions, _detections = _target_interactions_from_evidence(evidence_row, target_labels, frame=None)
    return max((_safe_float(item.get("score"), 0.0) for item in interactions), default=0.0)


def _evidence_local_time(evidence_row: dict[str, Any] | None, default: float) -> float:
    if not isinstance(evidence_row, dict):
        return default
    return _safe_float(evidence_row.get("local_time_sec"), _safe_float(evidence_row.get("time_sec"), default))


def _candidate_source_clip(candidate: dict[str, Any], session_root: Path) -> Path | None:
    for key in ("source_clip", "source_clip_path", "source_file"):
        raw = candidate.get(key)
        if not raw:
            continue
        path = Path(str(raw))
        if not path.is_absolute():
            path = session_root / path
        if path.is_file():
            return path
    return None


def _segment_level_material_reference_rows(session_root: Path) -> list[dict[str, Any]]:
    segment_path = session_root / "metadata" / "key_action_segments.jsonl"
    segment_rows = _read_jsonl_if_exists(segment_path)
    rows: list[dict[str, Any]] = []
    for segment in segment_rows:
        if not isinstance(segment, dict):
            continue
        segment_id = str(segment.get("segment_id") or "").strip()
        if not segment_id or _is_stale_identifier(segment_id):
            continue
        primary_object = _segment_primary_object(segment)
        action_name = _segment_action_name(segment, primary_object)
        for binding in _segment_asset_bindings(segment):
            view = str(binding.get("view") or binding.get("camera_view") or "").strip()
            if not view:
                continue
            rows.extend(
                _segment_binding_reference_rows(
                    session_root=session_root,
                    segment=segment,
                    binding=binding,
                    view=view,
                    primary_object=primary_object,
                    action_name=action_name,
                )
            )
    return rows


def _segment_binding_reference_rows(
    *,
    session_root: Path,
    segment: dict[str, Any],
    binding: dict[str, Any],
    view: str,
    primary_object: str,
    action_name: str,
) -> list[dict[str, Any]]:
    segment_id = str(segment.get("segment_id") or binding.get("segment_id") or "")
    view_data = segment.get(view) if isinstance(segment.get(view), dict) else {}
    start_sec = _safe_float(
        binding.get("local_start_sec", view_data.get("local_start_sec", segment.get("start_sec", 0.0)))
    )
    end_sec = _safe_float(
        binding.get("local_end_sec", view_data.get("local_end_sec", segment.get("end_sec", start_sec))),
        start_sec,
    )
    yolo_count = int(
        _safe_float(
            binding.get(
                "yolo_detection_count",
                view_data.get("yolo_detection_count", segment.get("yolo_interaction_count", 0)),
            ),
            0.0,
        )
    )
    common = {
        "schema_version": "material_reference.item.v1",
        "trace_schema_version": MATERIAL_REFERENCE_TRACE_SCHEMA_VERSION,
        "candidate_source": "segment_level_key_action",
        "fallback_reason": "micro_segment_candidates_unavailable",
        "action_name": action_name,
        **_canonical_action_fields(primary_object, action_name),
        "micro_segment_id": None,
        "parent_segment_id": segment_id,
        "segment_id": segment_id,
        "view": view,
        "camera_view": view,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "time_start": start_sec,
        "time_end": end_sec,
        "primary_object": primary_object,
        "generated": True,
        "dry_run": False,
        "error": None,
        "yolo_box_required": False,
        "box_filter": "segment_level_key_action_review",
        "time_range_sec": f"{start_sec:.3f}-{end_sec:.3f}",
        "yolo_annotated_required": False,
        "yolo_evidence_count": yolo_count,
        "yolo_label_counts": _segment_label_counts(segment, view),
        "evidence_chain": {
            "schema_version": MATERIAL_REFERENCE_TRACE_SCHEMA_VERSION,
            "camera_view": view,
            "time_start": start_sec,
            "time_end": end_sec,
            "segment_id": segment_id,
            "candidate_source": "segment_level_key_action",
            "candidate_disposition": None,
        },
    }
    rows: list[dict[str, Any]] = []
    clip_source = _segment_clip_source(session_root, segment, binding, view)
    if clip_source and clip_source.is_file():
        rows.append(
            _segment_reference_row(
                common,
                material_type=KEY_CLIP_DIR_NAME,
                source=clip_source,
                frame_type=None,
            )
        )
    for frame_type, frame_path in _segment_keyframe_sources(session_root, binding):
        if frame_path.is_file():
            rows.append(
                _segment_reference_row(
                    common,
                    material_type=KEYFRAME_DIR_NAME,
                    source=frame_path,
                    frame_type=frame_type,
                )
            )
    return rows


def _segment_reference_row(
    common: dict[str, Any],
    *,
    material_type: str,
    source: Path,
    frame_type: str | None,
) -> dict[str, Any]:
    row = dict(common)
    row.update(
        {
            "material_type": material_type,
            "asset_kind": material_type,
            "frame_type": frame_type,
            "frame_role": frame_type,
            "source_file": str(source),
            "source_clip": str(source) if material_type == KEY_CLIP_DIR_NAME else common.get("source_clip"),
            "source_clip_path": str(source) if material_type == KEY_CLIP_DIR_NAME else common.get("source_clip_path"),
            "stored_file": str(source),
            "stored_filename": source.name,
            "file_name": source.name,
            "exists": source.is_file(),
            "size_bytes": source.stat().st_size if source.is_file() else 0,
        }
    )
    evidence_chain = dict(row.get("evidence_chain") or {})
    evidence_chain.update(
        {
            "source_clip": row.get("source_clip") or str(source),
            "source_file": str(source),
            "asset_kind": material_type,
            "frame_type": frame_type,
        }
    )
    row["evidence_chain"] = evidence_chain
    return row


def _segment_asset_bindings(segment: dict[str, Any]) -> list[dict[str, Any]]:
    raw_bindings = segment.get("asset_bindings")
    bindings = [dict(item) for item in raw_bindings if isinstance(item, dict)] if isinstance(raw_bindings, list) else []
    if bindings:
        return bindings
    fallback: list[dict[str, Any]] = []
    for view in ("third_person", "first_person"):
        view_data = segment.get(view) if isinstance(segment.get(view), dict) else {}
        if not view_data:
            continue
        keyframes = view_data.get("keyframes") if isinstance(view_data.get("keyframes"), dict) else {}
        keyframe_paths = view_data.get("keyframe_paths") if isinstance(view_data.get("keyframe_paths"), list) else []
        fallback.append(
            {
                "level": "segment",
                "segment_id": segment.get("segment_id"),
                "view": view,
                "clip_path": view_data.get("clip_path"),
                "annotated_clip_path": view_data.get("annotated_clip_path"),
                "keyframe_path": view_data.get("keyframe_path"),
                "keyframe_paths": keyframe_paths,
                "keyframes": keyframes,
                "local_start_sec": view_data.get("local_start_sec"),
                "local_end_sec": view_data.get("local_end_sec"),
                "yolo_detection_count": view_data.get("yolo_detection_count"),
            }
        )
    return fallback


def _segment_clip_source(session_root: Path, segment: dict[str, Any], binding: dict[str, Any], view: str) -> Path | None:
    view_data = segment.get(view) if isinstance(segment.get(view), dict) else {}
    for value in (
        view_data.get("annotated_clip_path"),
        binding.get("annotated_clip_path"),
        binding.get("clip_path"),
        view_data.get("clip_path"),
    ):
        path = _resolve_session_path(session_root, value)
        if path is not None:
            return path
    return None


def _segment_keyframe_sources(session_root: Path, binding: dict[str, Any]) -> list[tuple[str, Path]]:
    sources: list[tuple[str, Path]] = []
    keyframes = binding.get("keyframes") if isinstance(binding.get("keyframes"), dict) else {}
    for role in ("middle", "peak", "contact", "start", "release", "end"):
        path = _resolve_session_path(session_root, keyframes.get(role))
        if path is not None:
            sources.append((role, path))
    keyframe_paths = binding.get("keyframe_paths") if isinstance(binding.get("keyframe_paths"), list) else []
    for index, value in enumerate(keyframe_paths, start=1):
        path = _resolve_session_path(session_root, value)
        if path is not None and all(path != existing for _, existing in sources):
            sources.append((f"frame_{index}", path))
    single = _resolve_session_path(session_root, binding.get("keyframe_path"))
    if single is not None and all(single != existing for _, existing in sources):
        sources.append(("middle", single))
    return sources


def _resolve_session_path(session_root: Path, value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else session_root / path


def _segment_primary_object(segment: dict[str, Any]) -> str:
    counts = _segment_label_counts(segment, "")
    labels = sorted(counts, key=lambda label: _safe_float(counts.get(label), 0.0), reverse=True)
    for label in labels:
        canonical = canonical_yolo_label(label)
        if canonical in CANONICAL_ACTION_BY_OBJECT:
            return canonical
    for label in labels:
        canonical = canonical_yolo_label(label)
        if canonical and canonical not in HAND_LABELS and canonical not in {"lab_coat", "ppe_storage"}:
            return canonical
    return "container"


def _segment_action_name(segment: dict[str, Any], primary_object: str) -> str:
    description = segment.get("text_description") if isinstance(segment.get("text_description"), dict) else {}
    action_type = str(description.get("action_type") or "").strip()
    if action_type and action_type != "unknown_operation":
        return action_type
    return _action_name(primary_object or "container")


def _segment_label_counts(segment: dict[str, Any], view: str) -> dict[str, Any]:
    if view:
        view_data = segment.get(view) if isinstance(segment.get(view), dict) else {}
        labels = view_data.get("yolo_label_counts")
        if isinstance(labels, dict):
            return labels
    labels = segment.get("yolo_label_counts")
    if isinstance(labels, dict):
        return labels
    merged: dict[str, Any] = {}
    for view_name in ("third_person", "first_person"):
        view_data = segment.get(view_name) if isinstance(segment.get(view_name), dict) else {}
        view_labels = view_data.get("yolo_label_counts")
        if isinstance(view_labels, dict):
            for label, count in view_labels.items():
                merged[str(label)] = _safe_float(merged.get(str(label)), 0.0) + _safe_float(count, 0.0)
    return merged


def _load_material_candidate_index(session_root: Path) -> tuple[Path, Path, list[dict[str, Any]]]:
    candidate_root = existing_material_candidates_root(session_root)
    index_jsonl = candidate_root / f"{MATERIAL_CANDIDATE_INDEX_BASENAME}.jsonl"
    rows = read_jsonl(index_jsonl)
    if not rows:
        raise FileNotFoundError(f"Material candidate index is not ready: {index_jsonl}")
    return candidate_root, index_jsonl, rows


def _explicit_candidate_ids(candidate_ids: list[str] | None) -> set[str]:
    return {str(item) for item in (candidate_ids or []) if str(item).strip()}


def _candidate_ids_for_review_request(
    rows: list[dict[str, Any]],
    *,
    candidate_group_id: str,
    candidate_ids: list[str] | None,
    empty_error: str,
) -> set[str]:
    explicit_ids = _explicit_candidate_ids(candidate_ids)
    target_ids: set[str] = set()
    for row in rows:
        row_group = str(row.get("candidate_group_id") or "")
        row_id = str(row.get("candidate_id") or "")
        if explicit_ids and row_id in explicit_ids:
            target_ids.add(row_id)
        elif not explicit_ids and row_group == str(candidate_group_id):
            target_ids.add(row_id)
    if not target_ids:
        raise ValueError(empty_error)
    return target_ids


def _approved_candidate_rows_for_sync(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if _candidate_formal_promotable(row)
        and (
            str(row.get("candidate_status") or "").lower() == "approved"
            or str(row.get("review_status") or "").lower() == "accepted"
        )
    ]


def _sync_approved_candidate_rows(session_root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return reset_material_references_to_approved_candidates(
        session_root,
        approved_rows=_approved_candidate_rows_for_sync(rows),
        merge_existing=_env_truthy("KEY_ACTION_MERGE_EXISTING_APPROVED_MATERIALS", False),
    )


def _persist_candidate_review_update(
    session_root: Path,
    candidate_root: Path,
    index_jsonl: Path,
    updated_rows: list[dict[str, Any]],
    review_log_entry: dict[str, Any],
) -> dict[str, Any]:
    _write_jsonl(index_jsonl, updated_rows)
    _refresh_candidate_review_metadata(candidate_root, updated_rows)
    _append_review_log(candidate_root / MATERIAL_CANDIDATE_REVIEW_LOG, review_log_entry)
    sync_summary = _sync_approved_candidate_rows(session_root, updated_rows)
    library_root = material_library_root(session_root)
    if library_root is not None:
        try:
            from .experiment_action_ledger import sync_candidate_review_outputs

            sync_summary["experiment_action_ledger_sync"] = sync_candidate_review_outputs(
                session_root,
                candidate_root,
                updated_rows,
                review_log_entry,
                material_root=library_root,
                sync_summary=sync_summary,
            )
        except Exception as exc:  # pragma: no cover - best-effort audit sync.
            sync_summary["experiment_action_ledger_sync"] = {
                "enabled": False,
                "error": str(exc),
            }
    return sync_summary


def _prepare_approved_material_dirs(ref_root: Path) -> tuple[Path, Path, Path]:
    keyframe_dir = ref_root / KEYFRAME_DIR_NAME
    clip_dir = ref_root / KEY_CLIP_DIR_NAME
    report_dir = ref_root / REPORT_DIR_NAME
    ref_root.mkdir(parents=True, exist_ok=True)
    keyframe_dir.mkdir(parents=True, exist_ok=True)
    clip_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    return keyframe_dir, clip_dir, report_dir


def _clear_files_in_folders(folders: tuple[Path, ...]) -> None:
    for folder in folders:
        for stale_file in folder.iterdir():
            if stale_file.is_file():
                try:
                    stale_file.unlink()
                except PermissionError:
                    pass


def _candidate_approved_target_dir(row: dict[str, Any], keyframe_dir: Path, clip_dir: Path, report_dir: Path) -> Path:
    if row.get("asset_kind") == KEYFRAME_DIR_NAME:
        return keyframe_dir
    if row.get("asset_kind") == REPORT_DIR_NAME:
        return report_dir
    return clip_dir


def _promote_approved_candidate_rows(
    approved_rows: list[dict[str, Any]],
    candidate_root: Path,
    experiment: dict[str, str],
    *,
    keyframe_dir: Path,
    clip_dir: Path,
    report_dir: Path,
) -> list[dict[str, Any]]:
    used_names: dict[str, set[str]] = {
        KEYFRAME_DIR_NAME: {path.name for path in keyframe_dir.iterdir() if path.is_file()},
        KEY_CLIP_DIR_NAME: {path.name for path in clip_dir.iterdir() if path.is_file()},
        REPORT_DIR_NAME: {path.name for path in report_dir.iterdir() if path.is_file()},
    }
    prepared: list[tuple[dict[str, Any], Path, Path]] = []
    for row in approved_rows:
        source = _stored_path_from_row(row, candidate_root)
        if source is None or not source.is_file():
            continue
        if row.get("source_real") is False or row.get("placeholder") is True or not _material_file_is_real(source):
            continue
        target_dir = _candidate_approved_target_dir(row, keyframe_dir, clip_dir, report_dir)
        target_name = _approved_material_target_name(row, source, experiment, used_names.setdefault(target_dir.name, set()))
        target = target_dir / target_name
        prepared.append((row, source, target))
    _run_material_candidate_tasks(
        [lambda source=source, target=target: _material_link_or_copy(source, target) for _row, source, target in prepared]
    )
    promoted: list[dict[str, Any]] = []
    for row, source, target in prepared:
        promoted_row = _approved_reference_record_from_candidate(row, source, target)
        promoted_row["experiment_id"] = str(promoted_row.get("experiment_id") or experiment["id"])
        promoted_row["session_id"] = str(promoted_row.get("session_id") or experiment["id"])
        promoted_row["package_session_id"] = str(promoted_row.get("package_session_id") or experiment["label"])
        promoted_row["experiment_title"] = str(promoted_row.get("experiment_title") or experiment["title"])
        promoted_row["experiment_date"] = str(promoted_row.get("experiment_date") or experiment["date"])
        promoted_row["experiment_label"] = str(promoted_row.get("experiment_label") or experiment["label"])
        promoted.append(promoted_row)
    return promoted


def _material_asset_kind_count(rows: list[dict[str, Any]], asset_kind: str) -> int:
    return sum(1 for row in rows if row.get("asset_kind") == asset_kind)


def _approved_material_local_summary(
    session_root: Path,
    ref_root: Path,
    formal_root: Path,
    experiment: dict[str, str],
    rows: list[dict[str, Any]],
    *,
    keyframe_dir: Path,
    clip_dir: Path,
    report_dir: Path,
    local_index_json: Path,
    local_index_jsonl: Path,
) -> dict[str, Any]:
    return {
        "schema_version": "material_references.approved_candidates.v1",
        "created_at": datetime.now().isoformat(),
        "experiment_id": experiment["id"],
        "session_id": experiment["id"],
        "package_session_id": experiment["label"],
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
        "index_jsonl": str(local_index_jsonl),
        "local_index_json": str(local_index_json),
        "local_index_jsonl": str(local_index_jsonl),
        "file_count": len(rows),
        "planned_file_count": 0,
        "keyframe_count": _material_asset_kind_count(rows, KEYFRAME_DIR_NAME),
        "key_clip_count": _material_asset_kind_count(rows, KEY_CLIP_DIR_NAME),
        "report_count": _material_asset_kind_count(rows, REPORT_DIR_NAME),
        "naming_rule": NAMING_RULE,
        "policy": "Only frontend-approved candidates are stored in the formal material reference folders.",
        "formal_publish_gate_policy": "Formal material rows require a dual_event_id or equivalent dual-view action event plus first/third keyframes and key clips.",
        "archive_root": None,
        "excluded_stale_markers": list(STALE_SPLIT_MARKERS),
        "records": rows,
    }


def _formal_material_summary_from_local(
    local_summary: dict[str, Any],
    formal_root: Path,
    *,
    formal_package: dict[str, Any],
    local_package: dict[str, Any],
    frontend_root: Path | None = None,
    frontend_package: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = dict(local_summary)
    frontend_root = frontend_root or formal_root
    summary.update(
        {
            "material_references": str(formal_root),
            "keyframe_folder": str(formal_root / KEYFRAME_DIR_NAME),
            "key_clip_folder": str(formal_root / KEY_CLIP_DIR_NAME),
            "index_json": str(formal_root / f"{MATERIAL_INDEX_BASENAME}.json"),
            "index_jsonl": str(formal_root / f"{MATERIAL_INDEX_BASENAME}.jsonl"),
            "formal_material_references": str(formal_root),
            "frontend_material_references": str(frontend_root),
            "frontend_keyframe_folder": str(frontend_root / KEYFRAME_DIR_NAME),
            "frontend_key_clip_folder": str(frontend_root / KEY_CLIP_DIR_NAME),
            "external_material_library_enabled": frontend_root.resolve() != formal_root.resolve(),
            "openclaw_evidence_package": formal_package,
            "local_openclaw_evidence_package": local_package,
            "frontend_openclaw_evidence_package": frontend_package or formal_package,
        }
    )
    return summary


def approve_material_candidates(
    session_dir: str | Path,
    *,
    candidate_group_id: str | None = None,
    candidate_ids: list[str] | None = None,
    reviewer: str | None = None,
    notes: str | None = None,
    reason_code: str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    """Promote reviewed candidate files into the formal material reference folder."""

    session_root = Path(session_dir)
    candidate_root, index_jsonl, rows = _load_material_candidate_index(session_root)

    explicit_ids = _explicit_candidate_ids(candidate_ids)
    if explicit_ids:
        selected = [row for row in rows if str(row.get("candidate_id") or "") in explicit_ids]
    elif candidate_group_id:
        group_rows = [row for row in rows if str(row.get("candidate_group_id") or "") == str(candidate_group_id)]
        selected = [row for row in group_rows if row.get("recommended") is True]
        if not selected:
            selected = _best_approvable_candidate_rows(group_rows)
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
                    **_canonical_action_fields(row.get("primary_object") or row.get("canonical_object"), row.get("action_name")),
                    "candidate_disposition_schema_version": MATERIAL_CANDIDATE_DISPOSITION_SCHEMA_VERSION,
                    "candidate_status": "approved",
                    "review_status": "accepted",
                    "approved_at": approved_at,
                    "approved_by": reviewer or "operator",
                    "review_notes": notes,
                    "approval_reason_code": reason_code or "representative_yolo_hand_object_evidence",
                    "approval_reason": reason or notes or "Approved as representative hand-object evidence for the canonical material library.",
                }
            )
        elif str(row.get("candidate_group_id") or "") in selected_groups:
            updated["candidate_disposition_schema_version"] = MATERIAL_CANDIDATE_DISPOSITION_SCHEMA_VERSION
            updated["candidate_status"] = "not_selected"
            updated["review_status"] = "not_selected"
            updated["disposition"] = "not_selected_after_group_approval"
            updated["reviewed_at"] = approved_at
            updated["reviewed_by"] = reviewer or "operator"
            updated["review_notes"] = "Not selected because a stronger representative asset from the same candidate group was approved."
        updated_rows.append(updated)

    sync_summary = _persist_candidate_review_update(
        session_root,
        candidate_root,
        index_jsonl,
        updated_rows,
        {
            "reviewed_at": approved_at,
            "reviewer": reviewer or "operator",
            "decision": "approved",
            "candidate_group_id": candidate_group_id,
            "candidate_ids": sorted(selected_ids),
            "reason_code": reason_code or "representative_yolo_hand_object_evidence",
            "reason": reason or notes,
            "notes": notes,
        },
    )
    return {
        "schema_version": "material_candidate_review.v1",
        "approved_candidate_ids": sorted(selected_ids),
        "approved_count": len(selected_ids),
        "material_references_summary": sync_summary,
        "candidate_index": str(index_jsonl),
    }


VALID_MATERIAL_REJECTION_REASONS = {
    "wrong_object",
    "wrong_action",
    "wrong_time_window",
    "duplicate",
    "bad_visibility",
    "not_experiment_action",
    "low_evidence",
    "evidence_mismatch",
}


def confirm_material_candidates(
    session_dir: str | Path,
    *,
    candidate_group_id: str,
    reviewer: str | None = None,
    notes: str | None = None,
    candidate_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Mark review candidates as human-confirmed without promoting them yet."""

    session_root = Path(session_dir)
    candidate_root, index_jsonl, rows = _load_material_candidate_index(session_root)
    target_ids = _candidate_ids_for_review_request(
        rows,
        candidate_group_id=candidate_group_id,
        candidate_ids=candidate_ids,
        empty_error="No material candidates matched the confirmation request",
    )

    confirmed_at = datetime.now().astimezone().isoformat()
    updated_rows: list[dict[str, Any]] = []
    for row in rows:
        updated = dict(row)
        if str(row.get("candidate_id") or "") in target_ids:
            updated.update(
                {
                    "candidate_disposition_schema_version": MATERIAL_CANDIDATE_DISPOSITION_SCHEMA_VERSION,
                    "candidate_status": "confirmed",
                    "review_status": "confirmed",
                    "confirmed_at": confirmed_at,
                    "confirmed_by": reviewer or "operator",
                    "review_notes": notes,
                    "memory_eligible": False,
                }
            )
        updated_rows.append(updated)

    sync_summary = _persist_candidate_review_update(
        session_root,
        candidate_root,
        index_jsonl,
        updated_rows,
        {
            "reviewed_at": confirmed_at,
            "reviewer": reviewer or "operator",
            "decision": "confirmed",
            "candidate_group_id": candidate_group_id,
            "candidate_ids": sorted(target_ids),
            "notes": notes,
        },
    )
    return {
        "schema_version": "material_candidate_confirmation.v1",
        "decision": "confirmed",
        "candidate_group_id": candidate_group_id,
        "candidate_ids": sorted(target_ids),
        "updated_count": len(target_ids),
        "material_references_summary": sync_summary,
        "candidate_index": str(index_jsonl),
    }


def rename_material_candidates(
    session_dir: str | Path,
    *,
    candidate_group_id: str,
    display_title: str,
    reviewer: str | None = None,
    notes: str | None = None,
    candidate_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Apply a display-only label correction to review candidates."""

    normalized_title = str(display_title or "").strip()
    if not normalized_title:
        raise ValueError("display_title is required")

    session_root = Path(session_dir)
    candidate_root, index_jsonl, rows = _load_material_candidate_index(session_root)
    target_ids = _candidate_ids_for_review_request(
        rows,
        candidate_group_id=candidate_group_id,
        candidate_ids=candidate_ids,
        empty_error="No material candidates matched the rename request",
    )

    renamed_at = datetime.now().astimezone().isoformat()
    updated_rows: list[dict[str, Any]] = []
    for row in rows:
        updated = dict(row)
        if str(row.get("candidate_id") or "") in target_ids:
            original_title = row.get("display_title") or row.get("action_name")
            updated.update(
                {
                    "candidate_disposition_schema_version": MATERIAL_CANDIDATE_DISPOSITION_SCHEMA_VERSION,
                    "display_title": normalized_title,
                    "human_display_title": normalized_title,
                    "original_display_title": original_title,
                    "renamed_at": renamed_at,
                    "renamed_by": reviewer or "operator",
                    "review_notes": notes,
                    "rename_scope": "display_only",
                }
            )
        updated_rows.append(updated)

    sync_summary = _persist_candidate_review_update(
        session_root,
        candidate_root,
        index_jsonl,
        updated_rows,
        {
            "reviewed_at": renamed_at,
            "reviewer": reviewer or "operator",
            "decision": "rename",
            "candidate_group_id": candidate_group_id,
            "candidate_ids": sorted(target_ids),
            "display_title": normalized_title,
            "notes": notes,
        },
    )
    return {
        "schema_version": "material_candidate_rename.v1",
        "decision": "rename",
        "candidate_group_id": candidate_group_id,
        "candidate_ids": sorted(target_ids),
        "display_title": normalized_title,
        "updated_count": len(target_ids),
        "material_references_summary": sync_summary,
        "candidate_index": str(index_jsonl),
    }


def dispose_material_candidates(
    session_dir: str | Path,
    *,
    candidate_group_id: str,
    decision: str,
    reason_code: str | None = None,
    reason: str | None = None,
    reviewer: str | None = None,
    notes: str | None = None,
    candidate_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Close a candidate review group without promoting it to the formal library."""

    normalized_decision = str(decision or "").strip().lower()
    if normalized_decision in {"reject", "rejected", "false_positive", "misclassified"}:
        normalized_decision = "false_positive"
    elif normalized_decision in {"defer", "deferred", "skip", "hold"}:
        normalized_decision = "deferred"
    else:
        raise ValueError("decision must be false_positive or deferred")

    if normalized_decision == "false_positive":
        normalized_reason = str(reason_code or "").strip()
        if normalized_reason not in VALID_MATERIAL_REJECTION_REASONS:
            raise ValueError("reason_code is required for false_positive material candidates")
    else:
        normalized_reason = str(reason_code or "").strip() or None

    session_root = Path(session_dir)
    candidate_root, index_jsonl, rows = _load_material_candidate_index(session_root)
    target_ids = _candidate_ids_for_review_request(
        rows,
        candidate_group_id=candidate_group_id,
        candidate_ids=candidate_ids,
        empty_error="No material candidates matched the disposition request",
    )

    reviewed_at = datetime.now().astimezone().isoformat()
    candidate_status = "rejected" if normalized_decision == "false_positive" else "deferred"
    updated_rows: list[dict[str, Any]] = []
    for row in rows:
        updated = dict(row)
        if str(row.get("candidate_id") or "") in target_ids:
            updated.update(
                {
                    "candidate_disposition_schema_version": MATERIAL_CANDIDATE_DISPOSITION_SCHEMA_VERSION,
                    "candidate_status": candidate_status,
                    "review_status": candidate_status,
                    "disposition": normalized_decision,
                    "reviewed_at": reviewed_at,
                    "reviewed_by": reviewer or "operator",
                    "review_notes": notes,
                }
            )
            if normalized_reason:
                updated["rejection_reason_code"] = normalized_reason
            if reason:
                updated["rejection_reason"] = reason
        updated_rows.append(updated)

    sync_summary = _persist_candidate_review_update(
        session_root,
        candidate_root,
        index_jsonl,
        updated_rows,
        {
            "reviewed_at": reviewed_at,
            "reviewer": reviewer or "operator",
            "decision": normalized_decision,
            "candidate_group_id": candidate_group_id,
            "candidate_ids": sorted(target_ids),
            "reason_code": normalized_reason,
            "reason": reason,
            "notes": notes,
        },
    )
    return {
        "schema_version": "material_candidate_disposition.v1",
        "decision": normalized_decision,
        "candidate_group_id": candidate_group_id,
        "candidate_ids": sorted(target_ids),
        "reason_code": normalized_reason,
        "updated_count": len(target_ids),
        "material_references_summary": sync_summary,
        "candidate_index": str(index_jsonl),
    }


def restore_material_candidates(
    session_dir: str | Path,
    *,
    candidate_group_id: str,
    candidate_ids: list[str] | None = None,
    reviewer: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Return disposed candidates to the default review queue while preserving audit history."""

    session_root = Path(session_dir)
    candidate_root, index_jsonl, rows = _load_material_candidate_index(session_root)
    target_ids = _candidate_ids_for_review_request(
        rows,
        candidate_group_id=candidate_group_id,
        candidate_ids=candidate_ids,
        empty_error="No material candidates matched the restore request",
    )

    restored_at = datetime.now().astimezone().isoformat()
    updated_rows: list[dict[str, Any]] = []
    restored_from: dict[str, int] = {}
    for row in rows:
        updated = dict(row)
        if str(row.get("candidate_id") or "") in target_ids:
            previous_status = str(row.get("candidate_status") or row.get("review_status") or "pending")
            restored_from[previous_status] = restored_from.get(previous_status, 0) + 1
            if row.get("rejection_reason_code"):
                updated["previous_rejection_reason_code"] = row.get("rejection_reason_code")
            if row.get("rejection_reason"):
                updated["previous_rejection_reason"] = row.get("rejection_reason")
            updated.update(
                {
                    "candidate_status": "pending",
                    "review_status": "pending",
                    "disposition": "restored_for_review",
                    "restored_at": restored_at,
                    "restored_by": reviewer or "operator",
                    "reviewed_at": None,
                    "reviewed_by": None,
                    "review_notes": notes or "Restored to the material review queue.",
                    "rejection_reason_code": None,
                    "rejection_reason": None,
                }
            )
        updated_rows.append(updated)

    sync_summary = _persist_candidate_review_update(
        session_root,
        candidate_root,
        index_jsonl,
        updated_rows,
        {
            "reviewed_at": restored_at,
            "reviewer": reviewer or "operator",
            "decision": "restored",
            "candidate_group_id": candidate_group_id,
            "candidate_ids": sorted(target_ids),
            "restored_from": restored_from,
            "notes": notes,
        },
    )
    return {
        "schema_version": "material_candidate_restore.v1",
        "decision": "restored",
        "candidate_group_id": candidate_group_id,
        "candidate_ids": sorted(target_ids),
        "updated_count": len(target_ids),
        "restored_from": restored_from,
        "material_references_summary": sync_summary,
        "candidate_index": str(index_jsonl),
    }


def reset_material_references_to_approved_candidates(
    session_dir: str | Path,
    *,
    approved_rows: list[dict[str, Any]],
    merge_existing: bool = True,
) -> dict[str, Any]:
    total_start = time.perf_counter()
    timing_sec: dict[str, float] = {}

    def _mark_timing(name: str, started_at: float) -> float:
        timing_sec[name] = round(time.perf_counter() - started_at, 3)
        return time.perf_counter()

    stage_start = time.perf_counter()
    session_root = Path(session_dir)
    ref_root = material_references_root(session_root)
    candidate_root = existing_material_candidates_root(session_root)
    keyframe_dir, clip_dir, report_dir = _prepare_approved_material_dirs(ref_root)
    experiment = _experiment_metadata(session_root)
    formal_root = formal_material_references_root(session_root)
    frontend_root = frontend_material_references_root(session_root)

    existing_index = ref_root / f"{MATERIAL_INDEX_BASENAME}.jsonl"
    kept_rows: list[dict[str, Any]] = []
    if merge_existing and existing_index.exists():
        for row in _read_jsonl_if_exists(existing_index):
            if not isinstance(row, dict):
                continue
            stored_path = _stored_path_from_row(row, ref_root)
            if stored_path is not None and stored_path.is_file():
                kept_rows.append(row)
    else:
        _clear_files_in_folders((keyframe_dir, clip_dir, report_dir))
    stage_start = _mark_timing("load_existing_sec", stage_start)
    promotable_rows, formal_gate_suppressed = apply_formal_dual_view_material_publish_gate(
        session_root,
        list(approved_rows),
    )
    stage_start = _mark_timing("dual_view_complete_group_filter_sec", stage_start)
    promoted = _promote_approved_candidate_rows(
        promotable_rows,
        candidate_root,
        experiment,
        keyframe_dir=keyframe_dir,
        clip_dir=clip_dir,
        report_dir=report_dir,
    )
    stage_start = _mark_timing("promote_candidate_files_sec", stage_start)

    rows = _dedupe_formal_material_rows([*kept_rows, *promoted])
    rows, non_real_suppressed = _filter_publishable_material_rows(rows, ref_root, session_root=session_root)
    local_index_json = ref_root / f"{MATERIAL_INDEX_BASENAME}.json"
    local_summary = _approved_material_local_summary(
        session_root,
        ref_root,
        formal_root,
        experiment,
        rows,
        keyframe_dir=keyframe_dir,
        clip_dir=clip_dir,
        report_dir=report_dir,
        local_index_json=local_index_json,
        local_index_jsonl=existing_index,
    )
    suppressed_rows = [*formal_gate_suppressed, *non_real_suppressed]
    if suppressed_rows:
        local_summary["skipped"] = suppressed_rows
        local_summary["skipped_count"] = len(suppressed_rows)
    publish_allowed, blocked_reason = _session_formal_material_publish_allowed(session_root)
    if not publish_allowed:
        _mark_material_summary_blocked(local_summary, session_root, blocked_reason)
    _write_jsonl(existing_index, rows)
    stream_rows = _write_material_stream(ref_root, rows, session_root=session_root)
    local_summary["material_stream"] = str(ref_root / "material_stream.jsonl")
    local_summary["material_stream_count"] = len(stream_rows)
    _write_json(local_index_json, local_summary)
    _write_json(ref_root / "manifest.json", _manifest(local_summary))
    _write_readme(ref_root / "README.md", local_summary)
    stage_start = _mark_timing("write_local_metadata_sec", stage_start)
    if _env_truthy("KEY_ACTION_BUILD_LOCAL_EVIDENCE_PACKAGE", False):
        local_package = _refresh_openclaw_evidence_package(
            ref_root,
            session_root,
            experiment,
            package_scope="local_material_references",
        )
    else:
        local_package = {
            "schema_version": "openclaw_evidence_package_refresh.v1",
            "status": "skipped",
            "ok": True,
            "scope": "local_material_references",
            "package_root": str(ref_root),
            "reason": "formal_material_library_package_is_authoritative",
        }
    stage_start = _mark_timing("local_evidence_package_sec", stage_start)
    local_summary["openclaw_evidence_package"] = local_package
    _write_json(local_index_json, local_summary)
    _write_json(ref_root / "manifest.json", _manifest(local_summary))
    delivery_targets: list[tuple[str, Path]] = [("formal", formal_root)]
    if frontend_root.resolve() != formal_root.resolve():
        delivery_targets.append(("frontend", frontend_root))

    def _sync_target(item: tuple[str, Path]) -> tuple[str, dict[str, Any]]:
        scope, target_root = item
        _copy_simplified_materials(ref_root, target_root, local_summary)
        should_build_package = (
            scope == "formal"
            and _env_truthy("KEY_ACTION_BUILD_FORMAL_EVIDENCE_PACKAGE", True)
        ) or (
            scope == "frontend"
            and _env_truthy("KEY_ACTION_BUILD_FRONTEND_EVIDENCE_PACKAGE", False)
        )
        if not should_build_package:
            return scope, {
                "schema_version": "openclaw_evidence_package_refresh.v1",
                "status": "skipped",
                "ok": True,
                "scope": f"{scope}_material_references",
                "package_root": str(target_root),
                "reason": "material_files_and_json_index_synced_without_rebuilding_evidence_package",
            }
        return scope, _refresh_openclaw_evidence_package(
            target_root,
            session_root,
            experiment,
            package_scope=f"{scope}_material_references",
        )

    if len(delivery_targets) == 1:
        delivery_packages = dict([_sync_target(delivery_targets[0])])
    else:
        with ThreadPoolExecutor(max_workers=len(delivery_targets), thread_name_prefix="material-delivery") as executor:
            delivery_packages = dict(executor.map(_sync_target, delivery_targets))
    stage_start = _mark_timing("delivery_sync_and_package_sec", stage_start)
    index_targets = _ordered_unique_paths([ref_root, *(target for _scope, target in delivery_targets)])
    reference_indexes = _build_material_reference_indexes(index_targets)
    stage_start = _mark_timing("reference_index_sec", stage_start)
    global_indexes = _sync_global_material_library_indexes(session_root, index_targets)
    stage_start = _mark_timing("global_material_library_index_sec", stage_start)
    formal_package = delivery_packages.get("formal") or {}
    frontend_package = delivery_packages.get("frontend")
    summary = _formal_material_summary_from_local(
        local_summary,
        formal_root,
        formal_package=formal_package,
        local_package=local_package,
        frontend_root=frontend_root,
        frontend_package=frontend_package,
    )
    summary["reference_indexes"] = reference_indexes
    summary["global_material_library_indexes"] = global_indexes
    timing_sec["total_sec"] = round(time.perf_counter() - total_start, 3)
    summary["timing_sec"] = timing_sec
    return summary


def _ordered_unique_paths(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        try:
            key = str(path.resolve()).lower()
        except OSError:
            key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _build_material_reference_indexes(target_roots: list[Path]) -> list[dict[str, Any]]:
    def _build(target_root: Path) -> dict[str, Any]:
        try:
            from .material_reference_index import build_key_material_reference_index

            return build_key_material_reference_index(target_root)
        except Exception as exc:  # pragma: no cover - best-effort handoff metadata.
            return {"material_root": str(target_root), "error": str(exc)}

    if not target_roots:
        return []
    if len(target_roots) == 1:
        return [_build(target_roots[0])]
    with ThreadPoolExecutor(max_workers=min(len(target_roots), _material_reference_worker_count()), thread_name_prefix="material-index") as executor:
        return list(executor.map(_build, target_roots))


def _sync_global_material_library_indexes(session_root: Path, target_roots: list[Path]) -> list[dict[str, Any]]:
    formal_library = formal_material_library_root(session_root)
    if formal_library is None:
        return []
    try:
        formal_resolved = formal_library.resolve()
    except OSError:
        formal_resolved = formal_library
    sync_roots: list[Path] = []
    for target_root in target_roots:
        try:
            target_root.resolve().relative_to(formal_resolved)
        except ValueError:
            continue
        sync_roots.append(target_root)
    sync_roots = _ordered_unique_paths(sync_roots)
    if not sync_roots:
        return []
    library_root = formal_library.parent if formal_library.name.lower() == "material_references" else formal_library

    def _sync(target_root: Path) -> dict[str, Any]:
        try:
            from .material_library_store import sync_material_library_package

            return sync_material_library_package(target_root, library_root=library_root)
        except Exception as exc:  # pragma: no cover - global catalog is a sidecar.
            return {"material_root": str(target_root), "error": str(exc)}

    if len(sync_roots) == 1:
        return [_sync(sync_roots[0])]
    with ThreadPoolExecutor(max_workers=min(len(sync_roots), _material_reference_worker_count()), thread_name_prefix="global-material-index") as executor:
        return list(executor.map(_sync, sync_roots))


def _dedupe_formal_material_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped_reversed: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for row in reversed(rows):
        identity = (
            str(row.get("candidate_id") or row.get("source_candidate_file") or ""),
            str(row.get("micro_segment_id") or ""),
            str(row.get("parent_segment_id") or row.get("segment_id") or ""),
            str(row.get("asset_kind") or row.get("material_type") or ""),
            str(row.get("view") or row.get("camera_view") or ""),
            str(row.get("frame_type") or row.get("frame_role") or ""),
            str(row.get("start_sec") or row.get("time_start") or ""),
            str(row.get("end_sec") or row.get("time_end") or ""),
        )
        if identity in seen:
            continue
        seen.add(identity)
        deduped_reversed.append(row)
    return list(reversed(deduped_reversed))


def _prepare_reference_root(ref_root: Path, archive_root: Path, *, archive_existing: bool) -> list[dict[str, str]]:
    ref_root.mkdir(parents=True, exist_ok=True)
    archived: list[dict[str, str]] = []
    for name in (
        KEYFRAME_DIR_NAME,
        KEY_CLIP_DIR_NAME,
        "manifest.json",
        "README.md",
        f"{MATERIAL_INDEX_BASENAME}.json",
        f"{MATERIAL_INDEX_BASENAME}.jsonl",
        *OPENCLAW_EVIDENCE_PACKAGE_FILES,
    ):
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
        if path.is_absolute() or path.exists():
            return path
        return root / path
    filename = row.get("stored_filename") or row.get("file_name")
    asset_kind = row.get("asset_kind") or row.get("material_type")
    if filename and asset_kind:
        return root / str(asset_kind) / str(filename)
    return None


def _approved_material_target_name(
    row: dict[str, Any],
    source: Path,
    experiment: dict[str, str],
    used_names: set[str],
) -> str:
    asset_kind = str(row.get("asset_kind") or row.get("material_type") or "")
    if asset_kind == REPORT_DIR_NAME:
        return _unique_name(used_names, source.stem, source.suffix or ".pdf")
    file_date = str(experiment.get("date") or "").strip()
    basename = _material_target_basename(row, experiment, date=file_date)
    return _unique_name(used_names, basename, source.suffix or ".material")


def _approved_material_experiment_type(row: Mapping[str, Any], experiment: Mapping[str, Any]) -> str:
    for key in ("experiment_type", "experiment_title", "title"):
        text = str(experiment.get(key) or "").strip()
        if text and _contains_cjk(text) and text not in {"\u5b9e\u9a8c", "\u672a\u547d\u540d\u5b9e\u9a8c"}:
            return text
    if _row_has_balance_context(row):
        return "\u56fa\u4f53\u79f0\u91cf\u5b9e\u9a8c"
    texts = " ".join(
        str(value or "").lower()
        for value in [
            row.get("primary_object"),
            row.get("manipulated_object"),
            row.get("canonical_object"),
            row.get("action_name"),
            row.get("display_title"),
            *(_list_strings(row.get("secondary_objects"))),
            *(_list_strings(row.get("objects"))),
        ]
    )
    if "\u79fb\u6db2" in texts or "pipette" in texts or "pipetting" in texts:
        return "\u79fb\u6db2\u5b9e\u9a8c"
    if "\u6ef4\u5b9a" in texts or "titration" in texts:
        return "\u6ef4\u5b9a\u5b9e\u9a8c"
    return "\u5b9e\u9a8c"


def _approved_material_asset_label(row: Mapping[str, Any]) -> str:
    asset_kind = str(row.get("asset_kind") or row.get("material_type") or "")
    if asset_kind == KEYFRAME_DIR_NAME:
        return KEYFRAME_DIR_NAME
    if asset_kind == KEY_CLIP_DIR_NAME:
        return KEY_CLIP_DIR_NAME
    return asset_kind or "\u7d20\u6750"


def _material_name_context(
    micro: Mapping[str, Any],
    *,
    primary_object: Any,
    semantic_fields: Mapping[str, Any],
) -> dict[str, Any]:
    context = dict(micro)
    interaction = micro.get("interaction") if isinstance(micro.get("interaction"), dict) else {}
    context.update(dict(semantic_fields))
    context.setdefault("primary_object", primary_object or interaction.get("primary_object"))
    context.setdefault("raw_primary_object", interaction.get("primary_object") or primary_object)
    context.setdefault("manipulated_object", semantic_fields.get("manipulated_object") or primary_object)
    context.setdefault("canonical_object", semantic_fields.get("canonical_object") or primary_object)
    context.setdefault("action_name", semantic_fields.get("display_title") or semantic_fields.get("semantic_action"))
    return context


def _material_target_basename(row: Mapping[str, Any], experiment: Mapping[str, Any], *, date: str) -> str:
    experiment_type = _approved_material_experiment_type(row, experiment)
    action_label = _approved_material_action_type_label(row)
    return "_".join(part for part in (experiment_type, action_label, date) if str(part or "").strip())


def _approved_material_action_type_label(row: Mapping[str, Any]) -> str:
    taxonomy = _semantic_taxonomy(dict(row)) or _canonical_action_fields(
        _semantic_primary_for_taxonomy(dict(row), row.get("primary_object") or row.get("canonical_object")),
        row.get("action_name"),
    )
    secondary_objects = _list_strings(row.get("secondary_objects"))
    actions = _list_strings(row.get("actions")) or _list_strings(row.get("secondary_actions"))
    physical_action = _core_material_physical_action_type(
        taxonomy,
        row,
        row.get("primary_object") or row.get("canonical_object"),
        secondary_objects,
        actions,
    )
    object_name = _approved_material_object_name(row)
    if physical_action == "equipment_panel_operation":
        return "\u5929\u5e73\u8bbe\u5907\u9762\u677f\u64cd\u4f5c" if _row_has_balance_context(row) else "\u8bbe\u5907\u9762\u677f\u64cd\u4f5c"
    if physical_action == "object_movement":
        return f"{object_name}\u79fb\u52a8"
    business_name = _business_action_name_for_row(row)
    if business_name:
        return business_name
    if physical_action == "hand_object_contact":
        return f"\u624b\u90e8\u4e0e{object_name}\u64cd\u4f5c"
    action_name = str(row.get("action_name") or row.get("display_title") or "").strip()
    return action_name if action_name and _contains_cjk(action_name) else f"\u624b\u90e8\u4e0e{object_name}\u64cd\u4f5c"


def _approved_material_object_token(row: Mapping[str, Any]) -> str:
    labels: list[Any] = [
        row.get("manipulated_object"),
        row.get("primary_object"),
        row.get("raw_primary_object"),
        row.get("corrected_primary_object"),
        row.get("canonical_object"),
    ]
    for value in labels:
        text = str(value or "").strip()
        if not text:
            continue
        canonical = canonical_yolo_label(text)
        if canonical:
            return canonical
        return _safe_name(text)
    return "object"


def _approved_material_view_label(row: Mapping[str, Any]) -> str:
    view = str(row.get("view") or row.get("camera_view") or "").strip()
    return VIEW_LABELS.get(view, "")


def _approved_material_time_label(row: Mapping[str, Any]) -> str:
    start = _safe_float(row.get("start_sec", row.get("time_start")), 0.0)
    end = _safe_float(row.get("end_sec", row.get("time_end")), start)
    if end > start:
        return f"{start:.1f}-{end:.1f}\u79d2"
    if start > 0:
        return f"{start:.1f}\u79d2"
    return ""


def _approved_material_chinese_action_name(row: Mapping[str, Any]) -> str:
    taxonomy = _semantic_taxonomy(dict(row)) or _canonical_action_fields(
        _semantic_primary_for_taxonomy(dict(row), row.get("primary_object") or row.get("canonical_object")),
        row.get("action_name"),
    )
    secondary_objects = _list_strings(row.get("secondary_objects"))
    actions = _list_strings(row.get("actions")) or _list_strings(row.get("secondary_actions"))
    physical_action = _core_material_physical_action_type(
        taxonomy,
        row,
        row.get("primary_object") or row.get("canonical_object"),
        secondary_objects,
        actions,
    )
    object_name = _approved_material_object_name(row)
    if physical_action == "equipment_panel_operation":
        return "\u5929\u5e73\u8bbe\u5907\u9762\u677f\u64cd\u4f5c" if _row_has_balance_context(row) else "\u8bbe\u5907\u9762\u677f\u64cd\u4f5c"
    if physical_action == "object_movement":
        return f"{object_name}\u79fb\u52a8"
    business_name = _business_action_name_for_row(row)
    if business_name:
        return business_name
    if physical_action == "hand_object_contact":
        return f"\u624b\u90e8\u4e0e{object_name}\u64cd\u4f5c"
    action_name = str(row.get("action_name") or row.get("display_title") or "").strip()
    return action_name if _contains_cjk(action_name) else f"\u624b\u90e8\u4e0e{object_name}\u64cd\u4f5c"


def _approved_material_evidence_action_name(row: Mapping[str, Any]) -> str:
    object_name = _approved_material_object_name(row)
    taxonomy = _semantic_taxonomy(dict(row)) or _canonical_action_fields(
        _semantic_primary_for_taxonomy(dict(row), row.get("primary_object") or row.get("canonical_object")),
        row.get("action_name"),
    )
    secondary_objects = _list_strings(row.get("secondary_objects"))
    actions = _list_strings(row.get("actions")) or _list_strings(row.get("secondary_actions"))
    physical_action = _core_material_physical_action_type(
        taxonomy,
        row,
        row.get("primary_object") or row.get("canonical_object"),
        secondary_objects,
        actions,
    )
    if physical_action == "equipment_panel_operation":
        return "\u5929\u5e73\u8bbe\u5907\u9762\u677f\u64cd\u4f5c" if _row_has_balance_context(row) else "\u8bbe\u5907\u9762\u677f\u64cd\u4f5c"
    if physical_action == "object_movement":
        return f"{object_name}\u79fb\u52a8"
    business_name = _business_action_name_for_row(row)
    if business_name:
        return business_name
    return f"\u624b\u90e8\u4e0e{object_name}\u64cd\u4f5c"


def _approved_material_object_name(row: Mapping[str, Any]) -> str:
    labels: list[Any] = [
        row.get("manipulated_object"),
        row.get("primary_object"),
        row.get("raw_primary_object"),
        row.get("corrected_primary_object"),
        row.get("canonical_object"),
    ]
    for key in ("secondary_objects", "objects"):
        value = row.get(key)
        if isinstance(value, (list, tuple, set)):
            labels.extend(value)
    for value in labels:
        text = str(value or "").strip()
        if not text:
            continue
        canonical = canonical_yolo_label(text)
        if canonical in CHINESE_OBJECT_NAMES:
            return CHINESE_OBJECT_NAMES[canonical]
        normalized = text.lower().replace("-", "_").replace(" ", "_")
        if normalized in CHINESE_OBJECT_NAMES:
            return CHINESE_OBJECT_NAMES[normalized]
        display = str(OBJECT_DISPLAY_NAMES.get(canonical) or OBJECT_DISPLAY_NAMES.get(normalized) or "").strip()
        if display and _contains_cjk(display):
            return display
        if _contains_cjk(text):
            return text
    return "\u7269\u4f53"


def _row_has_balance_context(row: Mapping[str, Any]) -> bool:
    values: list[Any] = [
        row.get("canonical_object"),
        row.get("primary_object"),
        row.get("manipulated_object"),
        row.get("instrument_context"),
        row.get("action_name"),
        row.get("display_title"),
        row.get("sop_phase"),
    ]
    for key in ("secondary_objects", "objects", "actions", "secondary_actions"):
        value = row.get(key)
        if isinstance(value, (list, tuple, set)):
            values.extend(value)
        else:
            values.append(value)
    text = " ".join(str(value or "").lower() for value in values)
    return any(token in text for token in ("balance", "scale", "panel", "display", "weigh", "\u5929\u5e73", "\u9762\u677f", "\u79f0\u91cf"))


def _contains_cjk(value: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in value)


def _candidate_record_from_reference(row: dict[str, Any], source_file: Path, target: Path, *, exists: bool) -> dict[str, Any]:
    identity = "|".join(
        str(row.get(key) or "")
        for key in ("micro_segment_id", "parent_segment_id", "asset_kind", "view", "frame_type", "file_name")
    )
    digest = hashlib.sha1(identity.encode("utf-8", errors="ignore")).hexdigest()[:12]
    candidate = dict(row)
    taxonomy_source = _semantic_primary_for_taxonomy(row, row.get("primary_object") or row.get("canonical_object"))
    target_real = _material_file_is_real(target) if exists else False
    candidate.update(
        {
            **(_semantic_taxonomy(row) or _canonical_action_fields(taxonomy_source, row.get("action_name"))),
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
            "source_real": target_real,
            "placeholder": not target_real,
            "publishable_material": target_real,
            "missing_reason": None if target_real else "candidate_file_not_real_video_material",
            "quality_score": _candidate_quality_score(row),
            "quality_reasons": _candidate_quality_reasons(row),
        }
    )
    if _candidate_quality_bucket(candidate) == "low_quality":
        candidate["quality_bucket"] = "low_quality"
    return candidate


def _candidate_group_id(row: dict[str, Any]) -> str:
    group_source = "|".join(
        str(row.get(key) or "")
        for key in ("micro_segment_id", "parent_segment_id", "primary_object", "start_sec", "end_sec")
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
    if _candidate_annotation_failed(row):
        score = min(score, 0.42)
    return round(min(1.0, score), 3)


def _candidate_quality_reasons(row: dict[str, Any]) -> list[str]:
    reasons = _as_list_for_semantics(row.get("quality_reasons"))
    reasons.extend(["yolo_physical_evidence", "frontend_review_required_before_publish"])
    error_text = _candidate_error_text(row)
    if _candidate_annotation_failed(row):
        reasons.append("no_target_box:yolo_annotation_not_rendered")
    if "annotation_fallback_unboxed" in error_text:
        reasons.append("annotation_fallback_unboxed")
    if "no_hand_target_interaction" in error_text or "no_target_box" in error_text:
        reasons.append("no_hand_target_interaction_evidence")
    hand_confidence = _candidate_max_hand_confidence(row)
    if hand_confidence is not None and hand_confidence < MIN_RECOMMENDED_HAND_CONFIDENCE:
        reasons.append("low_confidence_hand_evidence")
    return _ordered_unique_text(reasons)


def _candidate_annotation_failed(row: dict[str, Any]) -> bool:
    asset_kind = str(row.get("asset_kind") or row.get("material_type") or "")
    if asset_kind not in {KEYFRAME_DIR_NAME, KEY_CLIP_DIR_NAME}:
        return False
    if row.get("candidate_source") == "segment_level_key_action" or row.get("yolo_annotated_required") is False:
        return False
    error_text = _candidate_error_text(row)
    return (
        row.get("yolo_annotation_rendered") is False
        or "annotation_fallback_unboxed" in error_text
        or "no_hand_target_interaction" in error_text
        or "no_target_box" in error_text
    )


def _candidate_error_text(row: dict[str, Any]) -> str:
    values = [
        row.get("error"),
        row.get("rerender_error"),
        row.get("missing_reason"),
        row.get("reason"),
        *(row.get("quality_reasons") if isinstance(row.get("quality_reasons"), list) else []),
    ]
    return " ".join(str(value or "").lower() for value in values)


def _candidate_max_hand_confidence(row: dict[str, Any]) -> float | None:
    best: float | None = None

    def consider(detection: Any) -> None:
        nonlocal best
        if not isinstance(detection, dict):
            return
        label = canonical_yolo_label(detection.get("label") or detection.get("object_label") or detection.get("hand_label"))
        if label not in HAND_LABELS:
            return
        if detection.get("confidence") is None:
            return
        confidence = _safe_float(detection.get("confidence"), -1.0)
        if confidence < 0:
            return
        best = confidence if best is None else max(best, confidence)

    def scan_packet(packet: Any) -> None:
        if not isinstance(packet, dict):
            return
        for key in ("top_detections", "detections", "ignored_detections"):
            for detection in packet.get(key) or []:
                consider(detection)

    vlm_semantics = row.get("vlm_semantics") if isinstance(row.get("vlm_semantics"), dict) else {}
    scan_packet(vlm_semantics.get("evidence_packet"))
    scan_packet(row.get("evidence_packet"))
    scan_packet(row.get("evidence_chain"))
    for evidence in row.get("source_yolo_evidence") or row.get("yolo_evidence") or []:
        scan_packet(evidence)
    return best


def _candidate_quality_bucket(row: dict[str, Any]) -> str:
    if str(row.get("physical_evidence_mode") or "") == SPARSE_PHYSICAL_EVIDENCE_MODE:
        return "low_quality"
    if str(row.get("candidate_source") or "") == "micro_segment_yolo_sparse_evidence":
        return "low_quality"
    if _candidate_annotation_failed(row):
        return "low_quality"
    hand_confidence = _candidate_max_hand_confidence(row)
    if hand_confidence is not None and hand_confidence < MIN_RECOMMENDED_HAND_CONFIDENCE:
        return "low_quality"
    if _safe_float(row.get("quality_score"), 0.0) >= 0.72:
        return "priority"
    return "low_quality"


def _candidate_declared_file_is_real(row: Mapping[str, Any]) -> bool:
    stored_file = row.get("stored_file")
    if not stored_file:
        return True
    return _material_file_is_real(Path(str(stored_file)), dry_run=bool(row.get("dry_run")))


def _mark_recommended_candidates(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        row["recommended"] = False
        if row.get("recommendation_reason") == "best_quality_per_asset_kind":
            row.pop("recommendation_reason", None)
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
        asset_rows = [
            row
            for row in rows
            if row.get("asset_kind") == asset_kind and _candidate_recommendation_eligible(row)
        ]
        if not asset_rows:
            continue
        views = _ordered_unique_text([row.get("view") or row.get("camera_view") or "" for row in asset_rows])
        if not views:
            views = [""]
        for view in views:
            subset = [
                row
                for row in asset_rows
                if (not view or str(row.get("view") or row.get("camera_view") or "") == view)
            ]
            if subset:
                selected.append(max(subset, key=lambda item: (_safe_float(item.get("quality_score")), str(item.get("frame_type") or ""))))
    return selected


def _best_approvable_candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for asset_kind in (KEYFRAME_DIR_NAME, KEY_CLIP_DIR_NAME, REPORT_DIR_NAME):
        asset_rows = [
            row
            for row in rows
            if row.get("asset_kind") == asset_kind
            and row.get("exists") is not False
            and row.get("source_real") is not False
            and row.get("placeholder") is not True
            and _candidate_declared_file_is_real(row)
            and _candidate_formal_promotable(row)
        ]
        if not asset_rows:
            continue
        views = _ordered_unique_text([row.get("view") or row.get("camera_view") or "" for row in asset_rows])
        if not views:
            views = [""]
        for view in views:
            subset = [
                row
                for row in asset_rows
                if (not view or str(row.get("view") or row.get("camera_view") or "") == view)
            ]
            if subset:
                selected.append(max(subset, key=lambda item: (_safe_float(item.get("quality_score")), str(item.get("frame_type") or ""))))
    return selected


def _candidate_recommendation_eligible(row: dict[str, Any]) -> bool:
    if row.get("exists") is False:
        return False
    if row.get("source_real") is False or row.get("placeholder") is True:
        return False
    if not _candidate_declared_file_is_real(row):
        return False
    if not _candidate_formal_promotable(row):
        return False
    if str(row.get("review_route") or "") in {"vlm_review", "human_review"}:
        return False
    if _candidate_annotation_failed(row):
        return False
    if str(row.get("quality_bucket") or _candidate_quality_bucket(row)) == "low_quality":
        return False
    if _safe_float(row.get("quality_score"), 0.0) < 0.62:
        return False
    return True


def _approved_reference_record_from_candidate(row: dict[str, Any], source_file: Path, target: Path) -> dict[str, Any]:
    approved = dict(row)
    evidence_chain = dict(approved.get("evidence_chain") if isinstance(approved.get("evidence_chain"), dict) else {})
    action_display_name = _approved_material_chinese_action_name(row)
    evidence_action_name = _approved_material_evidence_action_name(row)
    taxonomy = _semantic_taxonomy(row) or _canonical_action_fields(
        _semantic_primary_for_taxonomy(row, row.get("primary_object") or row.get("canonical_object")),
        row.get("action_name"),
    )
    secondary_objects = _list_strings(row.get("secondary_objects"))
    actions = _list_strings(row.get("actions")) or _list_strings(row.get("secondary_actions"))
    physical_action_type = _core_material_physical_action_type(
        taxonomy,
        row,
        row.get("primary_object") or row.get("canonical_object"),
        secondary_objects,
        actions,
    )
    evidence_group_id = _material_evidence_group_id(
        row,
        primary_object=row.get("primary_object") or row.get("canonical_object"),
        start_sec=_safe_float(row.get("start_sec", row.get("time_start")), 0.0),
        end_sec=_safe_float(row.get("end_sec", row.get("time_end")), 0.0),
    )
    dual_event_id = str(
        row.get("dual_event_id")
        or row.get("dual_view_action_event_id")
        or evidence_chain.get("dual_event_id")
        or evidence_chain.get("dual_view_action_event_id")
        or ""
    ).strip()
    if dual_event_id:
        evidence_group_id = dual_event_id
    evidence_chain.update(
        {
            "schema_version": evidence_chain.get("schema_version") or MATERIAL_REFERENCE_TRACE_SCHEMA_VERSION,
            "candidate_disposition": "approved",
            "source_candidate_file": str(source_file),
            "formal_material_file": str(target),
            "evidence_group_id": evidence_group_id,
            "physical_action_type": physical_action_type,
            "action_display_name": action_display_name,
            "evidence_action_name": evidence_action_name,
        }
    )
    if dual_event_id:
        evidence_chain.update(
            {
                "dual_event_id": dual_event_id,
                "dual_view_action_event_id": dual_event_id,
                "dual_event_binding_source": row.get("dual_event_binding_source")
                or evidence_chain.get("dual_event_binding_source")
                or "explicit_confirmed_dual_view_action_event",
                "formal_dual_view_action": True,
            }
        )
    target_real = _material_file_is_real(target)
    target_sha256 = str(row.get("sha256") or "").strip() or (_material_file_sha256(target) if target_real else None)
    approved.update(
        {
            **taxonomy,
            "physical_action_type": physical_action_type,
            "physical_action_scope": os.environ.get("KEY_ACTION_PHYSICAL_ACTION_SCOPE", "").strip() or "default",
            "action_name": action_display_name,
            "display_title": action_display_name,
            "physical_action_display_name": action_display_name,
            "physical_evidence_action_name": evidence_action_name,
            "material_type": row.get("asset_kind"),
            "evidence_group_id": evidence_group_id,
            "material_group_id": evidence_group_id,
            "physical_action_material_id": evidence_group_id,
            "evidence_window_id": evidence_group_id,
            "dual_event_id": dual_event_id or None,
            "dual_view_action_event_id": dual_event_id or None,
            "dual_event_binding_source": (
                row.get("dual_event_binding_source") if dual_event_id else None
            ),
            "formal_dual_view_action": bool(dual_event_id),
            "primary_object_family": row.get("primary_object_family") or _material_object_family_for_label(row.get("primary_object")),
            "object_family": row.get("object_family") or _material_object_family_for_label(row.get("primary_object")),
            "stored_file": str(target),
            "stored_filename": target.name,
            "file_name": target.name,
            "source_candidate_file": str(source_file),
            "exists": target.is_file(),
            "size_bytes": target.stat().st_size if target.is_file() else 0,
            "source_real": target_real,
            "placeholder": not target_real,
            "publishable_material": target_real,
            "missing_reason": None if target_real else "approved_file_not_real_video_material",
            "sha256": target_sha256,
            "candidate_disposition_schema_version": row.get("candidate_disposition_schema_version") or MATERIAL_CANDIDATE_DISPOSITION_SCHEMA_VERSION,
            "evidence_chain": evidence_chain,
            "candidate_status": "approved",
            "review_status": "accepted",
            "formal_material_reference": True,
        }
    )
    return approved


def _candidate_manifest(summary: dict[str, Any]) -> dict[str, Any]:
    asset_counts = _candidate_asset_counts(summary.get("records") or [])
    return {
        "schema_version": summary["schema_version"],
        "created_at": summary["created_at"],
        "updated_at": summary.get("updated_at"),
        "session_dir": summary["session_dir"],
        "candidate_folder": summary["candidate_folder"],
        "keyframe_folder": summary["keyframe_folder"],
        "key_clip_folder": summary["key_clip_folder"],
        "candidate_count": summary["candidate_count"],
        "keyframe_count": summary.get("keyframe_count", asset_counts["keyframe_count"]),
        "key_clip_count": summary.get("key_clip_count", asset_counts["key_clip_count"]),
        "candidate_group_count": summary.get("candidate_group_count", asset_counts["candidate_group_count"]),
        "pending_total": summary["pending_total"],
        "approved_total": summary.get("approved_total", 0),
        "not_selected_total": summary.get("not_selected_total", 0),
        "rejected_total": summary.get("rejected_total", 0),
        "deferred_total": summary.get("deferred_total", 0),
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
        "deferred_total": 0,
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
        elif status == "deferred":
            counts["deferred_total"] += 1
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
    asset_counts = _candidate_asset_counts(rows)
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
            "candidate_group_count": asset_counts["candidate_group_count"],
            "keyframe_count": asset_counts["keyframe_count"],
            "key_clip_count": asset_counts["key_clip_count"],
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
        **asset_counts,
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
- keyframes: {summary.get("keyframe_count", 0)}
- key_clips: {summary.get("key_clip_count", 0)}
- candidate_groups: {summary.get("candidate_group_count", 0)}
- pending: {summary["pending_total"]}
- approved: {summary.get("approved_total", 0)}
- not_selected: {summary.get("not_selected_total", 0)}
- deferred: {summary.get("deferred_total", 0)}
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
    for fallback_view in ("third_person", "first_person"):
        fallback_data = segment.get(fallback_view) if isinstance(segment.get(fallback_view), dict) else {}
        if not fallback_data:
            continue
        value = fallback_data.get("local_start_sec", fallback_data.get("start_sec"))
        if value is not None:
            return _safe_float(value)
    starts = segment.get("view_start_sec") if isinstance(segment.get("view_start_sec"), dict) else {}
    return _safe_float(starts.get(view, segment.get("start_sec", 0.0)))


def _has_yolo_evidence(micro: dict[str, Any]) -> bool:
    return any(isinstance(row, dict) for row in micro.get("yolo_evidence", []))


def _action_name(primary_object: str) -> str:
    canonical = canonical_yolo_label(primary_object)
    return ACTION_NAME_BY_OBJECT.get(canonical, ACTION_NAME_BY_OBJECT.get(str(primary_object), f"\u624b\u4e0e{primary_object}\u64cd\u4f5c"))


def _business_action_name_for_row(row: Mapping[str, Any]) -> str:
    labels: list[Any] = [
        row.get("manipulated_object"),
        row.get("primary_object"),
        row.get("raw_primary_object"),
        row.get("corrected_primary_object"),
        row.get("canonical_object"),
        row.get("instrument_context"),
    ]
    for key in ("secondary_objects", "objects"):
        value = row.get(key)
        if isinstance(value, (list, tuple, set)):
            labels.extend(value)
    for label in labels:
        canonical = canonical_yolo_label(label)
        if canonical in ACTION_NAME_BY_OBJECT:
            return ACTION_NAME_BY_OBJECT[canonical]
    return ""


def _semantic_taxonomy(semantic_fields: dict[str, Any]) -> dict[str, str]:
    required = ("canonical_action_type", "canonical_object", "sop_phase", "interaction_family")
    if all(str(semantic_fields.get(key) or "").strip() for key in required):
        return {key: str(semantic_fields[key]) for key in required}
    return {}


def _semantic_primary_for_taxonomy(semantic_fields: dict[str, Any], primary_object: Any) -> Any:
    return (
        semantic_fields.get("corrected_primary_object")
        or semantic_fields.get("instrument_context")
        or semantic_fields.get("manipulated_object")
        or primary_object
    )


def _annotation_target_query(primary_object: Any, semantic_fields: dict[str, Any] | None = None) -> str:
    semantic_fields = semantic_fields or {}
    labels = _ordered_unique_text(
        [
            semantic_fields.get("manipulated_object"),
            primary_object,
        ]
    )
    return ",".join(labels)


def _canonical_primary_hint(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text.startswith("hand_"):
        first = text.split("+", 1)[0]
        return first.removeprefix("hand_")
    return text


def _canonical_action_fields(primary_object: Any, action_name: Any = None) -> dict[str, str]:
    text = _canonical_primary_hint(primary_object)
    for key, values in CANONICAL_ACTION_BY_OBJECT.items():
        if text == key or key in text:
            action_type, canonical_object, sop_phase = values
            return {
                "taxonomy_schema_version": MATERIAL_TAXONOMY_SCHEMA_VERSION,
                "canonical_action_type": action_type,
                "canonical_object": canonical_object,
                "sop_phase": sop_phase,
                "interaction_family": "hand-object",
            }
    action_text = str(action_name or "").strip().lower()
    action_primary = _canonical_primary_hint(action_name)
    if action_primary and action_primary != action_text.replace("-", "_").replace(" ", "_"):
        return _canonical_action_fields(action_primary)
    for needle, values in (
        ("balance", CANONICAL_ACTION_BY_OBJECT["balance"]),
        ("天平", CANONICAL_ACTION_BY_OBJECT["balance"]),
        ("panel", CANONICAL_ACTION_BY_OBJECT["panel"]),
        ("面板", CANONICAL_ACTION_BY_OBJECT["panel"]),
        ("spatula", CANONICAL_ACTION_BY_OBJECT["spatula"]),
        ("药匙", CANONICAL_ACTION_BY_OBJECT["spatula"]),
        ("paper", CANONICAL_ACTION_BY_OBJECT["paper"]),
        ("纸", CANONICAL_ACTION_BY_OBJECT["paper"]),
        ("bottle", CANONICAL_ACTION_BY_OBJECT["bottle"]),
        ("瓶", CANONICAL_ACTION_BY_OBJECT["bottle"]),
        ("magnetic_stir_bar", CANONICAL_ACTION_BY_OBJECT["magnetic_stir_bar"]),
        ("搅拌子", CANONICAL_ACTION_BY_OBJECT["magnetic_stir_bar"]),
        ("container", CANONICAL_ACTION_BY_OBJECT["container"]),
        ("容器", CANONICAL_ACTION_BY_OBJECT["container"]),
        ("beaker", CANONICAL_ACTION_BY_OBJECT["beaker"]),
        ("烧杯", CANONICAL_ACTION_BY_OBJECT["beaker"]),
    ):
        if needle in action_text:
            action_type, canonical_object, sop_phase = values
            return {
                "taxonomy_schema_version": MATERIAL_TAXONOMY_SCHEMA_VERSION,
                "canonical_action_type": action_type,
                "canonical_object": canonical_object,
                "sop_phase": sop_phase,
                "interaction_family": "hand-object",
            }
    action_type, canonical_object, sop_phase = CANONICAL_ACTION_BY_OBJECT["container"]
    return {
        "taxonomy_schema_version": MATERIAL_TAXONOMY_SCHEMA_VERSION,
        "canonical_action_type": action_type,
        "canonical_object": canonical_object,
        "sop_phase": sop_phase,
        "interaction_family": "hand-object",
    }


def _core_material_physical_action_type(
    taxonomy: Mapping[str, Any],
    semantic_fields: Mapping[str, Any],
    primary_object: Any,
    secondary_objects: list[str],
    actions: list[str],
) -> str:
    existing = str(semantic_fields.get("physical_action_type") or "").strip()
    canonical = str(taxonomy.get("canonical_action_type") or "").strip().lower().replace("_", "-")
    interaction_family = str(taxonomy.get("interaction_family") or semantic_fields.get("interaction_family") or "").strip().lower()
    main_object_labels = {
        canonical_yolo_label(value)
        for value in (
            taxonomy.get("canonical_object"),
            semantic_fields.get("canonical_object"),
            semantic_fields.get("corrected_primary_object"),
            semantic_fields.get("manipulated_object"),
            semantic_fields.get("instrument_context"),
            primary_object,
        )
        if str(value or "").strip()
    }
    values: list[Any] = [
        taxonomy.get("canonical_action_type"),
        taxonomy.get("canonical_object"),
        taxonomy.get("sop_phase"),
        taxonomy.get("interaction_family"),
        semantic_fields.get("semantic_action"),
        semantic_fields.get("instrument_context"),
        semantic_fields.get("manipulated_object"),
        semantic_fields.get("display_title"),
        primary_object,
        *secondary_objects,
        *actions,
    ]
    texts = [str(value or "").strip().lower().replace("_", "-") for value in values if str(value or "").strip()]
    allowed = active_physical_action_types()
    panel_objects = {"balance", "scale", "panel"}
    primary_label = canonical_yolo_label(primary_object)
    explicit_panel = any(
        any(label in text for label in ("equipment-panel", "equipment-control", "panel", "display", "\u8bbe\u5907\u9762\u677f"))
        for text in texts
    )
    if existing in allowed:
        if existing == "equipment_panel_operation" and primary_label not in panel_objects:
            existing = ""
        else:
            return existing
    if "object_movement" in allowed and any("object-movement" in text or "movement" in text or "move" in text for text in texts):
        return "object_movement"
    if "equipment_panel_operation" in allowed and primary_label in panel_objects and (main_object_labels.intersection(panel_objects) or explicit_panel):
        return "equipment_panel_operation"
    if "hand_object_contact" in allowed and (canonical.startswith("hand-") or interaction_family == "hand-object"):
        return "hand_object_contact"
    return ""


def _list_strings(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if item is not None and str(item).strip()]
    return [str(value).strip()]


def _ordered_unique_text(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        text = str(value or "").strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            ordered.append(text)
    return ordered


def _normalized_object(value: Any) -> str:
    label = canonical_yolo_label(value)
    return str(label or "").strip().lower().replace("-", "_").replace(" ", "_")


def _normalize_action_string(value: Any) -> str:
    text = str(value or "").strip().lower().replace(" ", "_")
    if not text:
        return ""
    parts = []
    for index, part in enumerate(text.split("+")):
        part = part.strip().replace("_", "-")
        if index == 0 and part.startswith("hand-"):
            parts.append(part)
        elif index == 0 and part.startswith("hand_"):
            parts.append("hand-" + part[5:].replace("_", "-"))
        else:
            parts.append(part)
    return "+".join(part for part in parts if part)


def _evidence_objects(micro: dict[str, Any]) -> list[str]:
    values: list[Any] = []
    for row in micro.get("yolo_evidence") or []:
        if not isinstance(row, dict):
            continue
        for detection in row.get("detections") or []:
            if isinstance(detection, dict):
                values.append(detection.get("label") or detection.get("class_name") or detection.get("name"))
        for interaction in row.get("hand_object_interactions") or []:
            if isinstance(interaction, dict):
                values.append(interaction.get("object_label") or interaction.get("target_label") or interaction.get("object"))
    return _ordered_unique_text([_normalized_object(value) for value in values])


def _micro_secondary_objects(micro: dict[str, Any], primary_object: Any) -> list[str]:
    interaction = micro.get("interaction") if isinstance(micro.get("interaction"), dict) else {}
    primary = _normalized_object(primary_object)
    values: list[Any] = []
    values.extend(_list_strings(micro.get("secondary_objects")))
    values.extend(_list_strings(interaction.get("secondary_objects")))
    values.extend(_evidence_objects(micro))
    excluded = {primary, "", *HAND_LABELS, "lab_coat", "ppe_storage"}
    return [
        label
        for label in _ordered_unique_text([_normalized_object(value) for value in values])
        if label not in excluded
    ]


def _canonical_action_aliases(primary_object: Any, secondary_objects: list[str], raw_secondary_actions: list[Any] | None = None) -> list[str]:
    taxonomy = _canonical_action_fields(primary_object)
    base_action = taxonomy["canonical_action_type"]
    values: list[Any] = [*(raw_secondary_actions or [])]
    secondary_action_objects: list[str] = []
    for obj in secondary_objects:
        secondary_taxonomy = _canonical_action_fields(obj)
        values.append(secondary_taxonomy["canonical_action_type"])
        secondary_action_objects.append(secondary_taxonomy["canonical_object"].replace("_", "-"))
    if base_action and secondary_action_objects:
        values.append(f"{base_action}+{'+'.join(secondary_action_objects)}")
    return _ordered_unique_text([_normalize_action_string(value) for value in values])


def _micro_secondary_actions(micro: dict[str, Any], primary_object: Any, secondary_objects: list[str]) -> list[str]:
    interaction = micro.get("interaction") if isinstance(micro.get("interaction"), dict) else {}
    raw_actions = [*_list_strings(micro.get("secondary_actions")), *_list_strings(interaction.get("secondary_actions"))]
    return _canonical_action_aliases(primary_object, secondary_objects, raw_actions)


def _evidence_support_for_object(evidence_rows: list[dict[str, Any]], object_label: str) -> dict[str, Any]:
    target = _normalized_object(object_label)
    target_labels = _interaction_target_labels(target)
    detection_count = 0
    interaction_count = 0
    max_score = 0.0
    frame_indices: list[int] = []
    views: set[str] = set()
    for index, row in enumerate(evidence_rows):
        if not isinstance(row, dict):
            continue
        detections = row.get("detections") or []
        if any(_normalized_object(det.get("label") if isinstance(det, dict) else "") == target for det in detections):
            detection_count += 1
        score = _target_interaction_score(row, target_labels) if target_labels else 0.0
        if score > 0.0:
            interaction_count += 1
            max_score = max(max_score, score)
            frame_indices.append(int(_safe_float(row.get("frame_index"), index)))
            view = evidence_view(row)
            if view:
                views.add(view)
    return {
        "object": target,
        "frame_count": interaction_count,
        "interaction_frame_count": interaction_count,
        "detection_frame_count": detection_count,
        "max_interaction_score": round(max_score, 6),
        "views": sorted(views),
        "frame_indices": frame_indices[:24],
    }


def _micro_window_audit(
    micro: dict[str, Any],
    *,
    primary_object: Any,
    secondary_objects: list[str],
) -> dict[str, Any]:
    existing = micro.get("window_audit")
    if isinstance(existing, dict) and existing:
        return dict(existing)
    evidence_rows = [row for row in micro.get("yolo_evidence") or [] if isinstance(row, dict)]
    target_support = _evidence_support_for_object(evidence_rows, str(primary_object or ""))
    secondary_support = [_evidence_support_for_object(evidence_rows, item) for item in secondary_objects]
    reasons: list[str] = []
    if target_support["interaction_frame_count"] < PHYSICAL_EVIDENCE_MIN_FRAMES:
        reasons.append("target_object_support_below_physical_evidence_min_frames")
    for item in secondary_support:
        if item["detection_frame_count"] > item["interaction_frame_count"]:
            reasons.append(f"secondary_{item['object']}_detection_exceeds_interaction_support")
        if item["interaction_frame_count"] == 0:
            reasons.append(f"secondary_{item['object']}_has_no_hand_interaction_frames")
    reasons = _ordered_unique_text(reasons)
    uncertainty = "none" if not reasons else "low"
    return {
        "schema_version": "key_action_window_audit.v1",
        "source": "material_reference_yolo_evidence",
        "interaction_frame_count": len(evidence_rows),
        "target_object_support": target_support,
        "secondary_object_support": secondary_support,
        "uncertainty": uncertainty,
        "uncertainty_reasons": reasons,
        "reasons": reasons,
    }


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
    semantic_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    primary_object = micro.get("primary_object") or (micro.get("interaction") or {}).get("primary_object")
    start_sec = _safe_float(micro.get("start_sec", micro.get("session_start_sec")))
    end_sec = _safe_float(micro.get("end_sec", micro.get("session_end_sec")))
    yolo_evidence_count = len([row for row in micro.get("yolo_evidence", []) if isinstance(row, dict)])
    secondary_objects = _micro_secondary_objects(micro, primary_object)
    semantic_fields = semantic_fields or enhance_material_semantics(
        micro,
        micro=micro,
        evidence_rows=[row for row in micro.get("yolo_evidence", []) if isinstance(row, dict)],
        primary_object=primary_object,
        secondary_objects=secondary_objects,
        action_name=action_name,
    )
    semantic_fields = dict(semantic_fields)
    semantic_display_title = semantic_fields.get("semantic_display_title") or semantic_fields.get("display_title")
    taxonomy = _semantic_taxonomy(semantic_fields) or _canonical_action_fields(
        _semantic_primary_for_taxonomy(semantic_fields, primary_object),
        action_name,
    )
    secondary_actions = _micro_secondary_actions(micro, primary_object, secondary_objects)
    objects = _ordered_unique_text(
        [
            taxonomy["canonical_object"],
            semantic_fields.get("instrument_context"),
            semantic_fields.get("manipulated_object"),
            primary_object,
            *secondary_objects,
        ]
    )
    actions = _ordered_unique_text([taxonomy["canonical_action_type"], semantic_fields.get("semantic_action"), *secondary_actions])
    physical_action_type = _core_material_physical_action_type(
        taxonomy,
        semantic_fields,
        primary_object,
        secondary_objects,
        actions,
    )
    action_context = {
        **taxonomy,
        **semantic_fields,
        "physical_action_type": physical_action_type,
        "primary_object": primary_object,
        "secondary_objects": secondary_objects,
        "actions": actions,
    }
    action_name = _approved_material_chinese_action_name(action_context)
    if semantic_display_title and str(semantic_display_title).strip() != action_name:
        semantic_fields["semantic_display_title"] = semantic_display_title
    semantic_fields["display_title"] = action_name
    evidence_group_id = _material_evidence_group_id(
        micro,
        primary_object=primary_object,
        start_sec=start_sec,
        end_sec=end_sec,
    )
    dual_event_id = str(
        micro.get("dual_event_id")
        or micro.get("dual_view_action_event_id")
        or micro.get("paired_event_id")
        or ""
    ).strip()
    if dual_event_id:
        evidence_group_id = dual_event_id
    window_audit = _micro_window_audit(
        micro,
        primary_object=primary_object,
        secondary_objects=secondary_objects,
    )
    evidence_chain = {
        "schema_version": MATERIAL_REFERENCE_TRACE_SCHEMA_VERSION,
        "source_clip": str(source),
        "camera_view": view,
        "time_start": start_sec,
        "time_end": end_sec,
        "evidence_group_id": evidence_group_id,
        "yolo_evidence_count": yolo_evidence_count,
        "canonical_action_type": taxonomy["canonical_action_type"],
        "physical_action_type": physical_action_type,
        "semantic_action": semantic_fields.get("semantic_action"),
        "display_title": action_name,
        "semantic_display_title": semantic_display_title,
        "manipulated_object": semantic_fields.get("manipulated_object"),
        "instrument_context": semantic_fields.get("instrument_context"),
        "corrected_primary_object": semantic_fields.get("corrected_primary_object"),
        "raw_primary_object": semantic_fields.get("raw_primary_object"),
        "semantic_evidence_refs": semantic_fields.get("semantic_evidence_refs", []),
        "secondary_actions": secondary_actions,
        "target_object_support": window_audit.get("target_object_support"),
        "secondary_object_support": window_audit.get("secondary_object_support"),
        "uncertainty": window_audit.get("uncertainty"),
        "uncertainty_reasons": window_audit.get("uncertainty_reasons", []),
        "candidate_disposition": None,
    }
    if dual_event_id:
        evidence_chain.update(
            {
                "dual_event_id": dual_event_id,
                "dual_view_action_event_id": dual_event_id,
                "dual_event_binding_source": micro.get("dual_event_binding_source")
                or "explicit_confirmed_dual_view_action_event",
                "formal_dual_view_action": True,
                "first_evidence_id": micro.get("first_evidence_id"),
                "third_evidence_id": micro.get("third_evidence_id"),
                "dual_view_action_alignment_score": micro.get("dual_view_action_alignment_score"),
            }
        )
    return {
        "schema_version": "material_reference.item.v1",
        "trace_schema_version": MATERIAL_REFERENCE_TRACE_SCHEMA_VERSION,
        "material_type": material_type,
        "asset_kind": material_type,
        "action_name": action_name,
        **taxonomy,
        **semantic_fields,
        "display_title": action_name,
        "semantic_display_title": semantic_display_title,
        "physical_action_type": physical_action_type,
        "physical_action_scope": os.environ.get("KEY_ACTION_PHYSICAL_ACTION_SCOPE", "").strip() or "default",
        "micro_segment_id": micro.get("micro_segment_id"),
        "parent_segment_id": micro.get("parent_segment_id") or micro.get("segment_id"),
        "segment_id": segment.get("segment_id") or micro.get("parent_segment_id") or micro.get("segment_id"),
        "evidence_group_id": evidence_group_id,
        "material_group_id": evidence_group_id,
        "physical_action_material_id": evidence_group_id,
        "evidence_window_id": evidence_group_id,
        "dual_event_id": dual_event_id or None,
        "dual_view_action_event_id": dual_event_id or None,
        "dual_event_binding_source": (
            micro.get("dual_event_binding_source") if dual_event_id else None
        ),
        "formal_dual_view_action": bool(dual_event_id),
        "single_view_candidate": False if dual_event_id else micro.get("single_view_candidate"),
        "primary_object_family": micro.get("primary_object_family") or _material_object_family_for_label(primary_object),
        "object_family": micro.get("object_family") or _material_object_family_for_label(primary_object),
        "view": view,
        "camera_view": view,
        "frame_type": frame_type,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "time_start": start_sec,
        "time_end": end_sec,
        "primary_object": primary_object,
        "secondary_objects": secondary_objects,
        "secondary_actions": secondary_actions,
        "objects": objects,
        "actions": actions,
        "window_audit": window_audit,
        "source_file": str(source),
        "source_clip": str(source),
        "source_clip_path": str(source),
        "stored_file": str(target),
        "stored_filename": target.name,
        "file_name": target.name,
        "exists": bool(target.exists()),
        "generated": bool(generated),
        "dry_run": bool(dry_run),
        "source_real": False,
        "placeholder": bool(dry_run),
        "publishable_material": False,
        "missing_reason": "dry_run_planned_material" if dry_run else None,
        "error": error,
        "yolo_box_required": True,
        "box_filter": "hand_and_primary_object_only",
        "time_range_sec": f"{start_sec:.3f}-{end_sec:.3f}",
        "frame_role": frame_type,
        "yolo_annotated_required": True,
        "yolo_evidence_count": yolo_evidence_count,
        "evidence_chain": evidence_chain,
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
    *,
    allow_frame_filter_fallback: bool = False,
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

        scaled_evidence_rows = [
            _scale_evidence_row_to_frame_size(row, width=width, height=height)
            for row in evidence_rows
            if isinstance(row, dict)
        ]
        start = max(0.0, float(offset))
        clip_duration = max(0.1, float(duration))
        end = start + clip_duration
        target_labels = _interaction_target_labels(primary_object)
        render_source_view = str(
            next(
                (row.get("view") or row.get("source_view") for row in scaled_evidence_rows if isinstance(row, dict) and (row.get("view") or row.get("source_view"))),
                "",
            )
            or ""
        )
        tracklet_annotation = build_tracklet_annotations(
            scaled_evidence_rows,
            target_labels=sorted(target_labels),
            include_hands=True,
        )
        hold_sec = _annotation_hold_sec(scaled_evidence_rows, clip_duration)
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
        max_frames = max(1, int(round(clip_duration * fps)) + 2)
        frame_index = 0
        annotated_frame_count = 0
        while frame_index < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            source_offset_sec = start + frame_index / fps
            if source_offset_sec > end + (1.0 / max(fps, 1.0)):
                break
            local_time_sec = segment_start_sec + source_offset_sec
            evidence = _nearest_evidence_row(
                scaled_evidence_rows,
                local_time_sec,
                hold_sec=hold_sec,
            )
            tracklet_detections = detections_for_time(
                tracklet_annotation,
                local_time_sec,
                hold_sec=hold_sec,
            )
            filtered_tracklet_detections = (
                _tracklet_detections_for_active_interaction(
                    tracklet_detections,
                    evidence,
                    primary_object,
                    frame=frame,
                    source_view=render_source_view,
                )
                if tracklet_detections and evidence is not None
                else []
            )
            if filtered_tracklet_detections:
                frame = _draw_tracklet_annotation_boxes(
                    frame,
                    filtered_tracklet_detections,
                    primary_object,
                    source_view=render_source_view,
                    pre_filtered=True,
                )
                annotated_frame_count += 1
            else:
                if evidence is not None:
                    filtered_evidence = _filtered_interaction_detections(
                        evidence,
                        primary_object,
                        frame=frame,
                        allow_frame_filter_fallback=allow_frame_filter_fallback,
                    )
                    if filtered_evidence:
                        frame = _draw_filtered_interaction_boxes(
                            frame,
                            evidence,
                            primary_object,
                            allow_frame_filter_fallback=allow_frame_filter_fallback,
                        )
                        annotated_frame_count += 1
            writer.write(frame)
            frame_index += 1
        writer.release()
        if not tmp_target.exists() or tmp_target.stat().st_size <= 0:
            raise RuntimeError(f"Filtered interaction clip was not written: {target}")
        if annotated_frame_count <= 0:
            target_text = ",".join(sorted(_interaction_target_labels(primary_object)))
            raise RuntimeError(f"no_hand_target_interaction_boxes:{target_text or primary_object}")
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
    *,
    require_boxes: bool = False,
    allow_frame_filter_fallback: bool = False,
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
        if require_boxes and not _filtered_interaction_detections(
            evidence_row,
            primary_object,
            frame=frame,
            allow_frame_filter_fallback=allow_frame_filter_fallback,
        ):
            target_text = ",".join(sorted(_interaction_target_labels(primary_object)))
            raise RuntimeError(f"no_hand_target_interaction_boxes:{target_text or primary_object}")
        frame = _draw_filtered_interaction_boxes(
            frame,
            evidence_row,
            primary_object,
            allow_frame_filter_fallback=allow_frame_filter_fallback,
        )
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


def _draw_filtered_interaction_boxes(
    frame: Any,
    evidence_row: dict[str, Any],
    primary_object: str,
    *,
    allow_frame_filter_fallback: bool = False,
) -> Any:
    try:
        import cv2
    except Exception:  # pragma: no cover
        return frame

    drawn = False
    target_labels = _interaction_target_labels(primary_object)
    for detection, color in _filtered_interaction_detections(
        evidence_row,
        primary_object,
        frame=frame,
        allow_frame_filter_fallback=allow_frame_filter_fallback,
    ):
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
        text_color = _annotation_text_color(color)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        text_top = max(0, y1 - text_h - 8)
        cv2.rectangle(frame, (x1, text_top), (min(width - 1, x1 + text_w + 8), y1), color, -1)
        cv2.putText(frame, text, (x1 + 4, max(text_h + 1, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2)
        drawn = True
    if drawn:
        target_text = "+".join(sorted(target_labels)) if target_labels else (canonical_yolo_label(primary_object) or primary_object)
        footer = f"YOLO evidence: hand + {target_text}"
        cv2.rectangle(frame, (0, 0), (min(frame.shape[1] - 1, 560), 30), (20, 20, 20), -1)
        cv2.putText(frame, footer, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return frame


def _draw_tracklet_annotation_boxes(
    frame: Any,
    detections: list[dict[str, Any]],
    primary_object: str,
    *,
    source_view: str | None = None,
    pre_filtered: bool = False,
) -> Any:
    try:
        import cv2
    except Exception:  # pragma: no cover
        return frame

    target_labels = _interaction_target_labels(primary_object)
    drawn = False
    draw_detections = (
        list(detections)
        if pre_filtered
        else _filtered_tracklet_annotation_detections(
            detections,
            primary_object,
            frame=frame,
            source_view=source_view,
        )
    )
    for detection in draw_detections:
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
        label = canonical_yolo_label(detection.get("label")) or str(detection.get("label") or "")
        color = _annotation_color_for_label(label)
        text_color = _annotation_text_color(color)
        source = str(detection.get("tracklet_source") or "tracklet")
        tracklet_id = str(detection.get("tracklet_id") or detection.get("object_track_id") or "")
        confidence = detection.get("confidence")
        conf_text = f" {float(confidence):.2f}" if confidence is not None else ""
        suffix = " interp" if source == "interpolated" else " trk"
        text = f"{label}{conf_text}{suffix}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2 if source == "interpolated" else 3)
        if source == "interpolated":
            for dash_x in range(x1, x2, 16):
                cv2.line(frame, (dash_x, y1), (min(x2, dash_x + 8), y1), color, 2)
                cv2.line(frame, (dash_x, y2), (min(x2, dash_x + 8), y2), color, 2)
            for dash_y in range(y1, y2, 16):
                cv2.line(frame, (x1, dash_y), (x1, min(y2, dash_y + 8)), color, 2)
                cv2.line(frame, (x2, dash_y), (x2, min(y2, dash_y + 8)), color, 2)
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        text_top = max(0, y1 - text_h - 8)
        cv2.rectangle(frame, (x1, text_top), (min(width - 1, x1 + text_w + 8), y1), color, -1)
        cv2.putText(frame, text, (x1 + 4, max(text_h + 1, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
        if tracklet_id:
            cv2.putText(frame, tracklet_id[-8:], (x1 + 4, min(height - 6, y2 + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        drawn = True
    if drawn:
        target_text = "+".join(sorted(target_labels)) if target_labels else (canonical_yolo_label(primary_object) or primary_object)
        footer = f"YOLO tracklet evidence: hand + {target_text}"
        cv2.rectangle(frame, (0, 0), (min(frame.shape[1] - 1, 680), 30), (20, 20, 20), -1)
        cv2.putText(frame, footer, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return frame


def _annotation_color_for_label(label: Any) -> tuple[int, int, int]:
    canonical = canonical_yolo_label(label) or str(label or "")
    if canonical in HAND_LABELS:
        return ANNOTATION_COLOR_BY_LABEL["gloved_hand"]
    return ANNOTATION_COLOR_BY_LABEL.get(canonical, (255, 170, 0))


def _annotation_text_color(color: tuple[int, int, int]) -> tuple[int, int, int]:
    blue, green, red = color
    luminance = 0.114 * blue + 0.587 * green + 0.299 * red
    return (20, 20, 20) if luminance >= 160 else (255, 255, 255)


def _tracklet_detections_for_active_interaction(
    tracklet_detections: list[dict[str, Any]],
    evidence_row: dict[str, Any] | None,
    primary_object: str,
    *,
    frame: Any | None = None,
    source_view: str | None = None,
) -> list[dict[str, Any]]:
    if evidence_row is None:
        return []
    anchors = [item for item, _color in _filtered_interaction_detections(evidence_row, primary_object, frame=frame)]
    if not anchors:
        return []
    candidates = _filtered_tracklet_annotation_detections(
        tracklet_detections,
        primary_object,
        frame=frame,
        source_view=source_view,
    )
    if not candidates:
        return []
    selected: list[dict[str, Any]] = []
    used_indexes: set[int] = set()
    for anchor in anchors:
        anchor_label = canonical_yolo_label(anchor.get("label") or anchor.get("object_label"))
        best_index: int | None = None
        best_score = 0.0
        for index, candidate in enumerate(candidates):
            if index in used_indexes:
                continue
            if canonical_yolo_label(candidate.get("label") or candidate.get("object_label")) != anchor_label:
                continue
            score = _tracklet_anchor_match_score(candidate, anchor)
            if score > best_score:
                best_score = score
                best_index = index
        if best_index is None or best_score < 0.1:
            continue
        used_indexes.add(best_index)
        selected.append(candidates[best_index])
    labels = {canonical_yolo_label(item.get("label") or item.get("object_label")) for item in selected}
    target_labels = _interaction_target_labels(primary_object)
    if not labels.intersection(HAND_LABELS) or not labels.intersection(target_labels):
        return []
    return selected


def _tracklet_anchor_match_score(candidate: dict[str, Any], anchor: dict[str, Any]) -> float:
    candidate_bbox = _bbox(candidate.get("bbox"))
    anchor_bbox = _bbox(anchor.get("bbox"))
    if candidate_bbox is None or anchor_bbox is None:
        return 0.0
    iou = _bbox_iou(candidate_bbox, anchor_bbox)
    if iou >= 0.08:
        return 1.0 + iou
    distance = _bbox_center_distance(candidate_bbox, anchor_bbox)
    candidate_diag = _bbox_diag(candidate_bbox)
    anchor_diag = _bbox_diag(anchor_bbox)
    limit = max(36.0, min(120.0, 0.42 * max(candidate_diag, anchor_diag)))
    if distance <= limit:
        return max(0.1, 1.0 - (distance / max(limit, 1.0)))
    return 0.0


def _bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * max(0.0, float(box_a[3]) - float(box_a[1]))
    area_b = max(0.0, float(box_b[2]) - float(box_b[0])) * max(0.0, float(box_b[3]) - float(box_b[1]))
    return inter / max(area_a + area_b - inter, 1.0)


def _bbox_center_distance(box_a: list[float], box_b: list[float]) -> float:
    ax = (float(box_a[0]) + float(box_a[2])) / 2.0
    ay = (float(box_a[1]) + float(box_a[3])) / 2.0
    bx = (float(box_b[0]) + float(box_b[2])) / 2.0
    by = (float(box_b[1]) + float(box_b[3])) / 2.0
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _bbox_diag(box: list[float]) -> float:
    return ((float(box[2]) - float(box[0])) ** 2 + (float(box[3]) - float(box[1])) ** 2) ** 0.5


def _filtered_tracklet_annotation_detections(
    detections: list[dict[str, Any]],
    primary_object: str,
    *,
    frame: Any | None = None,
    source_view: str | None = None,
) -> list[dict[str, Any]]:
    target_labels = _interaction_target_labels(primary_object)
    allowed_labels = set(HAND_LABELS) | set(target_labels)
    candidates = [
        item
        for item in detections
        if isinstance(item, dict) and canonical_yolo_label(item.get("label") or item.get("object_label")) in allowed_labels
    ]
    if not candidates:
        return []
    if frame is None:
        filtered = candidates
    else:
        filtered, _ignored = filter_implausible_detections(candidates, frame=frame, source_view=source_view)
    if frame is not None:
        refined: list[dict[str, Any]] = []
        for item in filtered:
            refined_item = _refine_visual_detection_for_frame(item, frame, peer_detections=filtered)
            if refined_item is not None:
                refined.append(refined_item)
        filtered = refined
    labels = {canonical_yolo_label(item.get("label") or item.get("object_label")) for item in filtered}
    if not labels.intersection(HAND_LABELS) or not labels.intersection(target_labels):
        return []
    return filtered


def _interaction_target_labels(primary_object: Any) -> set[str]:
    if isinstance(primary_object, dict):
        labels: set[str] = set()
        for key in ("canonical_object", "primary_object", "object", "object_label", "target_label", "label"):
            labels.update(_interaction_target_labels(primary_object.get(key)))
        return labels
    if isinstance(primary_object, (list, tuple, set)):
        labels: set[str] = set()
        for item in primary_object:
            labels.update(_interaction_target_labels(item))
        return labels
    raw = str(primary_object or "").strip()
    labels: set[str] = set()
    for part in re.split(r"[,;/+|]+", raw):
        label = canonical_yolo_label(part)
        if label:
            labels.add(label)
    if not re.search(r"[,;/+|]+", raw):
        label = canonical_yolo_label(raw)
        if label:
            labels.add(label)
    return labels


def _filtered_interaction_detections(
    evidence_row: dict[str, Any],
    primary_object: str,
    *,
    frame: Any | None = None,
    allow_frame_filter_fallback: bool = False,
) -> list[tuple[dict[str, Any], tuple[int, int, int]]]:
    if frame is not None:
        try:
            frame_height, frame_width = frame.shape[:2]
            evidence_row = _scale_evidence_row_to_frame_size(evidence_row, width=frame_width, height=frame_height)
        except Exception:
            pass
    primary_labels = _interaction_target_labels(primary_object)
    if not primary_labels:
        return []
    source_interactions, detections = _target_interactions_from_evidence(evidence_row, primary_labels, frame=frame)
    if (
        allow_frame_filter_fallback
        and frame is not None
        and not source_interactions
        and not primary_labels.intersection({"paper", "weighing_paper"})
    ):
        source_interactions, detections = _target_interactions_from_evidence(evidence_row, primary_labels, frame=None)
    filtered: list[tuple[dict[str, Any], tuple[int, int, int]]] = []
    seen: set[tuple[str, tuple[int, int, int, int]]] = set()
    hand_fallback = _detection_for_labels(detections, HAND_LABELS)
    for primary in sorted(primary_labels):
        interactions = [
            _interaction_with_detection_support(item, primary, detections)
            for item in source_interactions
            if canonical_yolo_label(item.get("object_label") or item.get("target_label") or item.get("object")) == primary
        ]
        interaction = max(interactions, key=_interaction_selection_rank, default=None)
        if interaction is None:
            continue
        obj_fallback = _detection_for_labels(detections, {primary})
        hand = _interaction_detection(interaction, "hand", hand_fallback)
        obj = _interaction_detection(interaction, "object", obj_fallback)
        for detection in (obj, hand):
            if detection is None:
                continue
            color = _annotation_color_for_label(detection.get("label"))
            bbox = _bbox(detection.get("bbox"))
            if bbox is None:
                continue
            key = (
                canonical_yolo_label(detection.get("label")),
                tuple(int(round(value)) for value in bbox[:4]),
            )
            if key in seen:
                continue
            seen.add(key)
            filtered.append((detection, color))
    if frame is not None and filtered:
        peer_detections = [item for item, _color in filtered] + detections
        refined_pairs: list[tuple[dict[str, Any], tuple[int, int, int]]] = []
        for detection, color in filtered:
            refined_detection = _refine_visual_detection_for_frame(
                detection,
                frame,
                peer_detections=peer_detections,
            )
            if refined_detection is not None:
                refined_pairs.append((refined_detection, color))
        labels = {canonical_yolo_label(item.get("label")) for item, _color in refined_pairs}
        if not labels.intersection(HAND_LABELS) or not labels.intersection(primary_labels):
            original_labels = {canonical_yolo_label(item.get("label")) for item, _color in filtered}
            if (
                not primary_labels.intersection({"paper", "weighing_paper"})
                and original_labels.intersection(HAND_LABELS)
                and original_labels.intersection(primary_labels)
            ):
                return filtered
            return []
        return refined_pairs
    return filtered


def _interaction_with_detection_support(
    interaction: dict[str, Any],
    primary_label: str,
    detections: list[dict[str, Any]],
) -> dict[str, Any]:
    item = dict(interaction)
    item["_hand_detection_confidence"] = _interaction_bbox_matching_detection_confidence(
        item.get("hand_bbox"),
        HAND_LABELS,
        detections,
    )
    item["_object_detection_confidence"] = _interaction_bbox_matching_detection_confidence(
        item.get("object_bbox"),
        {primary_label},
        detections,
    )
    return item


def _interaction_selection_rank(interaction: dict[str, Any]) -> tuple[int, float, float, float]:
    hand_confidence = _safe_float(interaction.get("_hand_detection_confidence"), -1.0)
    object_confidence = _safe_float(interaction.get("_object_detection_confidence"), -1.0)
    score = _safe_float(interaction.get("score"), 0.0)
    strong_hand = int(hand_confidence >= MIN_RECOMMENDED_HAND_CONFIDENCE)
    return (strong_hand, hand_confidence, object_confidence, score)


def _interaction_bbox_matching_detection_confidence(
    bbox_value: Any,
    labels: set[str] | frozenset[str],
    detections: list[dict[str, Any]],
) -> float | None:
    bbox = _bbox(bbox_value)
    if bbox is None:
        return None
    normalized = {canonical_yolo_label(label) for label in labels if canonical_yolo_label(label)}
    rounded = tuple(int(round(value)) for value in bbox[:4])
    best: float | None = None
    for detection in detections:
        if canonical_yolo_label(detection.get("label")) not in normalized:
            continue
        det_bbox = _bbox(detection.get("bbox"))
        if det_bbox is None:
            continue
        if tuple(int(round(value)) for value in det_bbox[:4]) != rounded:
            continue
        confidence = _safe_float(detection.get("confidence"), -1.0)
        if confidence < 0:
            continue
        best = confidence if best is None else max(best, confidence)
    return best


def _scale_evidence_row_to_frame_size(evidence_row: dict[str, Any], *, width: int, height: int) -> dict[str, Any]:
    source_width, source_height = _source_frame_size_for_evidence(evidence_row, width=width, height=height)
    if source_width <= 0 or source_height <= 0 or width <= 0 or height <= 0:
        return evidence_row
    scale_x = float(width) / source_width
    scale_y = float(height) / source_height
    if abs(scale_x - 1.0) < 0.002 and abs(scale_y - 1.0) < 0.002:
        return evidence_row
    scaled = dict(evidence_row)
    scaled["bbox_source_frame"] = {
        "frame_width": int(round(source_width)),
        "frame_height": int(round(source_height)),
        "render_width": int(width),
        "render_height": int(height),
        "scale_x": scale_x,
        "scale_y": scale_y,
    }
    scaled["frame_width"] = int(width)
    scaled["frame_height"] = int(height)
    scaled["detections"] = [
        _scale_detection_bbox(item, scale_x=scale_x, scale_y=scale_y)
        for item in evidence_row.get("detections") or []
        if isinstance(item, dict)
    ]
    if isinstance(evidence_row.get("ignored_detections"), list):
        scaled["ignored_detections"] = [
            _scale_detection_bbox(item, scale_x=scale_x, scale_y=scale_y)
            for item in evidence_row.get("ignored_detections") or []
            if isinstance(item, dict)
        ]
    scaled["hand_object_interactions"] = [
        _scale_interaction_bbox(item, scale_x=scale_x, scale_y=scale_y)
        for item in evidence_row.get("hand_object_interactions") or []
        if isinstance(item, dict)
    ]
    return scaled


def _source_frame_size_for_evidence(evidence_row: dict[str, Any], *, width: int, height: int) -> tuple[float, float]:
    source_width = _safe_float(evidence_row.get("frame_width"), 0.0)
    source_height = _safe_float(evidence_row.get("frame_height"), 0.0)
    if source_width > 0 and source_height > 0:
        return source_width, source_height
    max_x, max_y = _max_bbox_extent(evidence_row)
    if width > 0 and height > 0 and (max_x > float(width) * 1.02 or max_y > float(height) * 1.02):
        overflow = max(max_x / max(float(width), 1.0), max_y / max(float(height), 1.0))
        if overflow <= 2.25:
            scale = 2.0
        elif overflow <= 3.25:
            scale = 3.0
        elif overflow <= 4.25:
            scale = 4.0
        else:
            scale = overflow
        return float(width) * scale, float(height) * scale
    return float(width), float(height)


def _max_bbox_extent(evidence_row: dict[str, Any]) -> tuple[float, float]:
    max_x = 0.0
    max_y = 0.0

    def visit(value: Any) -> None:
        nonlocal max_x, max_y
        bbox = _bbox(value)
        if bbox is None:
            return
        max_x = max(max_x, float(bbox[2]))
        max_y = max(max_y, float(bbox[3]))

    for item in evidence_row.get("detections") or []:
        if isinstance(item, dict):
            visit(item.get("bbox"))
    for item in evidence_row.get("ignored_detections") or []:
        if isinstance(item, dict):
            visit(item.get("bbox"))
    for item in evidence_row.get("hand_object_interactions") or []:
        if isinstance(item, dict):
            visit(item.get("hand_bbox"))
            visit(item.get("object_bbox"))
            visit(item.get("target_bbox"))
    return max_x, max_y


def _scale_detection_bbox(detection: dict[str, Any], *, scale_x: float, scale_y: float) -> dict[str, Any]:
    item = dict(detection)
    bbox = _bbox(item.get("bbox"))
    if bbox is not None:
        item["source_bbox"] = list(bbox)
        item["bbox"] = _scale_bbox_values(bbox, scale_x=scale_x, scale_y=scale_y)
    return item


def _scale_interaction_bbox(interaction: dict[str, Any], *, scale_x: float, scale_y: float) -> dict[str, Any]:
    item = dict(interaction)
    for key in ("hand_bbox", "object_bbox", "target_bbox"):
        bbox = _bbox(item.get(key))
        if bbox is not None:
            item[f"source_{key}"] = list(bbox)
            item[key] = _scale_bbox_values(bbox, scale_x=scale_x, scale_y=scale_y)
    return item


def _scale_bbox_values(bbox: list[float], *, scale_x: float, scale_y: float) -> list[float]:
    return [
        round(float(bbox[0]) * scale_x, 3),
        round(float(bbox[1]) * scale_y, 3),
        round(float(bbox[2]) * scale_x, 3),
        round(float(bbox[3]) * scale_y, 3),
    ]


def _refine_visual_detection_for_frame(
    detection: dict[str, Any],
    frame: Any,
    *,
    peer_detections: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    label = canonical_yolo_label(detection.get("label") or detection.get("object_label"))
    if not label:
        return None
    item = dict(detection)
    item["label"] = label
    if item.get("visual_refinement"):
        return item
    if label in HAND_LABELS:
        return _refine_hand_detection_for_frame(item, frame) or item
    if label == "paper":
        return _refine_paper_detection_for_frame(item, frame, peer_detections=peer_detections or []) or item
    return item


def _refine_hand_detection_for_frame(detection: dict[str, Any], frame: Any) -> dict[str, Any] | None:
    try:
        import cv2
        import numpy as np
    except Exception:  # pragma: no cover
        return dict(detection)

    source_bbox = _bbox(detection.get("bbox"))
    clipped = _clip_bbox_to_frame(source_bbox, frame)
    if clipped is None:
        return None
    x1, y1, x2, y2 = clipped
    crop = frame[y1:y2, x1:x2]
    if crop.size <= 0:
        return None

    height, width = frame.shape[:2]
    frame_area = float(max(1, width * height))
    raw_area = float(max(1, (x2 - x1) * (y2 - y1)))
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    blue, green, red = cv2.split(crop)
    blue_i = blue.astype("int16")
    green_i = green.astype("int16")
    red_i = red.astype("int16")
    label = canonical_yolo_label(detection.get("label") or detection.get("object_label"))
    blue_glove = ((hue >= 86) & (hue <= 136) & (sat >= 55) & (val >= 90)) | (
        (blue_i > green_i + 12) & (blue_i > red_i + 28) & (sat >= 45) & (val >= 90)
    )
    skin_or_wrist = (((hue <= 24) | (hue >= 168)) & (sat >= 28) & (sat <= 190) & (val >= 70) & (red_i >= green_i - 8) & (green_i >= blue_i - 18))
    color_mask = blue_glove if label == "gloved_hand" else (blue_glove | skin_or_wrist)
    hand_mask = color_mask.astype(np.uint8) * 255
    kernel = np.ones((3, 3), dtype=np.uint8)
    hand_mask = cv2.morphologyEx(hand_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    hand_mask = cv2.morphologyEx(hand_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask_ratio = float(np.count_nonzero(hand_mask)) / raw_area
    if mask_ratio < 0.025:
        return None

    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(hand_mask, 8)
    if count <= 1:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.count_nonzero(edges)) / raw_area
    if mask_ratio > 0.58 and edge_density < 0.045:
        return None

    min_area = max(70.0, frame_area * 0.00010, raw_area * 0.018)
    max_area = min(raw_area * 1.01, frame_area * 0.14)
    best: tuple[float, list[float], float] | None = None
    for index in range(1, count):
        comp_x, comp_y, comp_w, comp_h, comp_area = [float(value) for value in stats[index]]
        if comp_area < min_area or comp_area > max_area:
            continue
        if comp_w < 8.0 or comp_h < 8.0:
            continue
        aspect = comp_w / max(comp_h, 1.0)
        if aspect < 0.16 or aspect > 6.0:
            continue
        touches = sum(
            (
                comp_x <= 1.0,
                comp_y <= 1.0,
                comp_x + comp_w >= (x2 - x1) - 1.0,
                comp_y + comp_h >= (y2 - y1) - 1.0,
            )
        )
        if raw_area / frame_area > 0.06 and touches >= 3 and comp_area / raw_area > 0.68:
            continue
        frame_box = [x1 + comp_x, y1 + comp_y, x1 + comp_x + comp_w, y1 + comp_y + comp_h]
        component = _labels == index
        mean_s = float(sat[component].mean()) if bool(component.any()) else 0.0
        mean_v = float(val[component].mean()) if bool(component.any()) else 0.0
        if mean_s < 45.0 or mean_v < 90.0:
            continue
        score = (comp_area / raw_area) + min(0.20, edge_density) - (0.08 if touches >= 2 else 0.0)
        if best is None or score > best[0]:
            best = (score, frame_box, comp_area)
    if best is None:
        return None

    _score, refined_box, comp_area = best
    pad = max(3.0, min(refined_box[2] - refined_box[0], refined_box[3] - refined_box[1]) * 0.08)
    refined_box = [
        max(0.0, refined_box[0] - pad),
        max(0.0, refined_box[1] - pad),
        min(float(width - 1), refined_box[2] + pad),
        min(float(height - 1), refined_box[3] + pad),
    ]
    if _bbox(refined_box) is None:
        return None
    refined = dict(detection)
    refined["raw_yolo_bbox"] = list(source_bbox or [])
    refined["bbox"] = refined_box
    refined["visual_refinement"] = {
        "method": "hand_color_component",
        "source_bbox": list(source_bbox or []),
        "component_area": comp_area,
        "mask_ratio": mask_ratio,
        "edge_density": edge_density,
    }
    return refined


def _refine_paper_detection_for_frame(
    detection: dict[str, Any],
    frame: Any,
    *,
    peer_detections: list[dict[str, Any]],
) -> dict[str, Any] | None:
    try:
        import cv2
        import numpy as np
    except Exception:  # pragma: no cover
        return dict(detection)

    source_bbox = _bbox(detection.get("bbox"))
    clipped = _clip_bbox_to_frame(source_bbox, frame)
    if clipped is None:
        return None
    x1, y1, x2, y2 = clipped
    crop = frame[y1:y2, x1:x2]
    if crop.size <= 0:
        return None

    height, width = frame.shape[:2]
    frame_area = float(max(1, width * height))
    raw_area = float(max(1, (x2 - x1) * (y2 - y1)))
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    channel_range = crop.max(axis=2) - crop.min(axis=2)
    bright_neutral = ((sat <= 68) & (val >= 150)) | ((sat <= 92) & (val >= 188) & (channel_range <= 90))
    mask = bright_neutral.astype(np.uint8) * 255
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return None

    hand_boxes = [
        _bbox(item.get("bbox"))
        for item in peer_detections
        if isinstance(item, dict) and canonical_yolo_label(item.get("label") or item.get("object_label")) in HAND_LABELS
    ]
    hand_boxes = [box for box in hand_boxes if box is not None]
    min_area = max(120.0, frame_area * 0.0010, raw_area * 0.004)
    max_area = min(raw_area * 0.36, frame_area * 0.032)
    best: tuple[float, list[float], dict[str, Any]] | None = None
    for index in range(1, count):
        comp_x, comp_y, comp_w, comp_h, comp_area = [float(value) for value in stats[index]]
        if comp_area < min_area or comp_area > max_area:
            continue
        if comp_w < 8.0 or comp_h < 8.0:
            continue
        raw_area_ratio = raw_area / frame_area
        if raw_area_ratio > 0.045 and max(comp_w / max(width, 1), comp_h / max(height, 1)) > 0.34:
            continue
        aspect = comp_w / max(comp_h, 1.0)
        if aspect < 0.35 or aspect > 2.20:
            continue
        component_ratio = comp_area / raw_area
        touches = sum(
            (
                comp_x <= 1.0,
                comp_y <= 1.0,
                comp_x + comp_w >= (x2 - x1) - 1.0,
                comp_y + comp_h >= (y2 - y1) - 1.0,
            )
        )
        if raw_area_ratio > 0.035 and touches >= 2 and component_ratio > 0.38:
            continue
        frame_box = [x1 + comp_x, y1 + comp_y, x1 + comp_x + comp_w, y1 + comp_y + comp_h]
        hand_distance = min((_bbox_edge_distance(frame_box, hand_box) for hand_box in hand_boxes), default=0.0)
        hand_limit = max(72.0, min(width, height) * 0.085, (comp_w + comp_h) * 0.55)
        if hand_boxes and hand_distance > hand_limit:
            continue
        component_mask = mask[int(comp_y) : int(comp_y + comp_h), int(comp_x) : int(comp_x + comp_w)]
        component_pixels = crop[int(comp_y) : int(comp_y + comp_h), int(comp_x) : int(comp_x + comp_w)]
        if component_mask.size <= 0 or component_pixels.size <= 0:
            continue
        selected = component_mask > 0
        if not bool(selected.any()):
            continue
        mean_v = float(val[int(comp_y) : int(comp_y + comp_h), int(comp_x) : int(comp_x + comp_w)][selected].mean())
        mean_s = float(sat[int(comp_y) : int(comp_y + comp_h), int(comp_x) : int(comp_x + comp_w)][selected].mean())
        if mean_v < 178.0 or mean_s > 78.0:
            continue
        whiteness = (mean_v / 255.0) - min(0.35, mean_s / 510.0)
        hand_score = 1.0 - min(1.0, hand_distance / max(hand_limit, 1.0))
        size_score = 1.0 - min(1.0, comp_area / max(max_area, 1.0))
        border_penalty = 0.16 if touches >= 2 and raw_area_ratio > 0.035 else 0.0
        score = whiteness + (0.42 * hand_score) + (0.24 * size_score) - border_penalty
        if best is None or score > best[0]:
            best = (
                score,
                frame_box,
                {
                    "method": "paper_bright_component",
                    "source_bbox": list(source_bbox or []),
                    "component_area": comp_area,
                    "hand_distance_px": hand_distance,
                    "mean_value": mean_v,
                    "mean_saturation": mean_s,
                },
            )
    if best is None or best[0] < 0.72:
        return None

    _score, refined_box, metadata = best
    pad = max(2.0, min(refined_box[2] - refined_box[0], refined_box[3] - refined_box[1]) * 0.06)
    refined_box = [
        max(0.0, refined_box[0] - pad),
        max(0.0, refined_box[1] - pad),
        min(float(width - 1), refined_box[2] + pad),
        min(float(height - 1), refined_box[3] + pad),
    ]
    if _bbox(refined_box) is None:
        return None
    refined = dict(detection)
    refined["raw_yolo_bbox"] = list(source_bbox or [])
    refined["bbox"] = refined_box
    refined["visual_refinement"] = metadata
    return refined


def _clip_bbox_to_frame(bbox: list[float] | None, frame: Any) -> tuple[int, int, int, int] | None:
    if bbox is None:
        return None
    try:
        height, width = frame.shape[:2]
    except Exception:
        return None
    x1, y1, x2, y2 = bbox
    x1_i = max(0, min(width - 1, int(round(x1))))
    y1_i = max(0, min(height - 1, int(round(y1))))
    x2_i = max(0, min(width, int(round(x2))))
    y2_i = max(0, min(height, int(round(y2))))
    if x2_i <= x1_i or y2_i <= y1_i:
        return None
    return x1_i, y1_i, x2_i, y2_i


def _bbox_edge_distance(box_a: list[float], box_b: list[float]) -> float:
    dx = max(box_b[0] - box_a[2], box_a[0] - box_b[2], 0.0)
    dy = max(box_b[1] - box_a[3], box_a[1] - box_b[3], 0.0)
    return (dx * dx + dy * dy) ** 0.5


def _target_interactions_from_evidence(
    evidence_row: dict[str, Any],
    target_labels: set[str],
    *,
    frame: Any | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_detections = [item for item in evidence_row.get("detections") or [] if isinstance(item, dict)]
    frame_width = int(_safe_float(evidence_row.get("frame_width"), 0.0)) or None
    frame_height = int(_safe_float(evidence_row.get("frame_height"), 0.0)) or None
    if frame is not None:
        try:
            frame_height, frame_width = frame.shape[:2]
        except Exception:
            pass
    source_view = str(evidence_row.get("source_view") or evidence_row.get("view") or "")
    detections, _ignored = filter_implausible_detections(
        raw_detections,
        frame_width=frame_width,
        frame_height=frame_height,
        frame=frame,
        source_view=source_view,
    )
    cached_interactions = [
        item
        for item in evidence_row.get("hand_object_interactions") or []
        if isinstance(item, dict)
    ]
    if cached_interactions and frame is not None:
        cached_interactions = [
            item for item in cached_interactions if _cached_interaction_supported_by_detections(item, detections)
        ]
    matching_cached = _interactions_matching_targets(cached_interactions, target_labels)
    if matching_cached:
        return matching_cached, detections
    recomputed = [
        item
        for item in find_hand_object_interactions(
            detections,
            frame_width=frame_width,
            frame_height=frame_height,
            frame=frame,
            source_view=source_view,
            min_interaction_score=0.1,
        )
        if isinstance(item, dict)
    ]
    return _interactions_matching_targets(recomputed, target_labels), detections


def _interactions_matching_targets(
    interactions: list[dict[str, Any]],
    target_labels: set[str],
) -> list[dict[str, Any]]:
    return [
        item
        for item in interactions
        if canonical_yolo_label(item.get("object_label") or item.get("target_label") or item.get("object")) in target_labels
    ]


def _cached_interaction_supported_by_detections(
    interaction: dict[str, Any],
    detections: list[dict[str, Any]],
) -> bool:
    object_label = canonical_yolo_label(
        interaction.get("object_label") or interaction.get("target_label") or interaction.get("object")
    )
    if not object_label:
        return False
    return _interaction_bbox_has_matching_detection(interaction.get("hand_bbox"), HAND_LABELS, detections) and _interaction_bbox_has_matching_detection(
        interaction.get("object_bbox"),
        {object_label},
        detections,
    )


def _interaction_bbox_has_matching_detection(
    bbox_value: Any,
    labels: set[str] | frozenset[str],
    detections: list[dict[str, Any]],
) -> bool:
    bbox = _bbox(bbox_value)
    if bbox is None:
        return False
    normalized = {canonical_yolo_label(label) for label in labels if canonical_yolo_label(label)}
    rounded = tuple(int(round(value)) for value in bbox[:4])
    for detection in detections:
        if canonical_yolo_label(detection.get("label")) not in normalized:
            continue
        det_bbox = _bbox(detection.get("bbox"))
        if det_bbox is None:
            continue
        if tuple(int(round(value)) for value in det_bbox[:4]) == rounded:
            return True
    return False


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
    confidence_key = "_hand_detection_confidence" if role == "hand" else "_object_detection_confidence"
    bbox = _bbox(interaction.get(bbox_key))
    if bbox is None:
        return fallback
    item = dict(fallback or {})
    item["label"] = canonical_yolo_label(interaction.get(label_key)) or ("gloved_hand" if role == "hand" else "object")
    item["bbox"] = bbox
    support_confidence = interaction.get(confidence_key)
    if support_confidence is not None:
        item["confidence"] = _safe_float(support_confidence, 0.0)
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
    manifest = {
        "schema_version": summary["schema_version"],
        "created_at": summary["created_at"],
        "experiment_id": summary.get("experiment_id"),
        "session_id": summary.get("session_id") or summary.get("experiment_id"),
        "package_session_id": summary.get("package_session_id") or summary.get("experiment_label"),
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
        "openclaw_evidence_package": summary.get("openclaw_evidence_package"),
    }
    for key in (
        "status",
        "formal_publish_blocked",
        "blocked_reason",
        "formal_dual_view_action_gate",
        "video_memory_allowed",
        "memory_write_allowed",
    ):
        if key in summary:
            manifest[key] = summary.get(key)
    return manifest


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
    summary_label = str(report_summary.get("experiment_label") or "").strip()
    if summary_label:
        label = _safe_name(summary_label)
    else:
        synthetic_payloads = [dict(report_summary)]
        experiment_title = str(
            report_summary.get("experiment_title")
            or report_summary.get("experiment_name")
            or report_summary.get("title")
            or ref_root.parent.name
        )
        title = _clean_experiment_title(experiment_title)
        if _is_technical_experiment_title(title):
            title = _domain_experiment_title(synthetic_payloads) or "\u5b9e\u9a8c"
        experiment_date = str(
            report_summary.get("experiment_date")
            or report_summary.get("date")
            or report_summary.get("created_at")
            or datetime.now().strftime("%Y%m%d")
        )
        label = _experiment_label(title, _date_label(experiment_date))
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
        "source_real": _material_file_is_real(target),
        "placeholder": not _material_file_is_real(target),
        "publishable_material": _material_file_is_real(target),
        "missing_reason": None if _material_file_is_real(target) else "report_file_not_real_material",
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
        if not _material_row_is_publishable(row, root=ref_root)[0]:
            continue
        updated = dict(row)
        stored = Path(str(row.get("stored_file") or ""))
        if stored.exists() and ref_root in stored.parents:
            rel = stored.relative_to(ref_root)
            target = simplified_root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if stored.is_file() and stored.resolve() != target.resolve():
                _material_link_or_copy(stored, target)
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

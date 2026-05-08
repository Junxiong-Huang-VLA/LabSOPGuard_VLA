from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import SessionManifest, VideoSource, WorkbenchROI
from .time_alignment import parse_time


def _issue(severity: str, message: str, path: str | None = None) -> dict[str, str]:
    item = {"severity": severity, "message": message}
    if path:
        item["path"] = path
    return item


def validate_video_source(video_path: str | Path) -> dict[str, Any]:
    path = Path(video_path)
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "can_open": False,
        "fps": None,
        "width": None,
        "height": None,
        "frame_count": None,
        "duration_sec": None,
        "issues": [],
    }
    if not path.exists():
        result["issues"].append(_issue("error", "Video path does not exist", str(path)))
        return result
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        result["issues"].append(_issue("error", f"opencv-python is unavailable: {exc}", str(path)))
        return result

    cap = cv2.VideoCapture(str(path))
    try:
        result["can_open"] = bool(cap.isOpened())
        if not cap.isOpened():
            result["issues"].append(_issue("error", "Video cannot be opened by OpenCV", str(path)))
            return result
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_sec = float(frame_count / fps) if fps > 0 else 0.0
        result.update(
            {
                "fps": fps,
                "width": width,
                "height": height,
                "frame_count": frame_count,
                "duration_sec": duration_sec,
            }
        )
        if fps <= 0:
            result["issues"].append(_issue("error", "Video FPS is not positive", str(path)))
        if width <= 0 or height <= 0:
            result["issues"].append(_issue("error", "Video dimensions are invalid", str(path)))
        if frame_count <= 0:
            result["issues"].append(_issue("error", "Video has no frames", str(path)))
    finally:
        cap.release()
    return result


def validate_transcript(transcript_path: str | Path) -> dict[str, Any]:
    path = Path(transcript_path)
    result: dict[str, Any] = {"path": str(path), "exists": path.exists(), "utterance_count": 0, "issues": []}
    if not path.exists():
        result["issues"].append(_issue("error", "Transcript path does not exist", str(path)))
        return result

    previous_start = None
    previous_end = None
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                result["issues"].append(_issue("error", f"Line {line_no} is not valid JSON: {exc}", str(path)))
                continue
            missing = [key for key in ("utterance_id", "start_sec", "end_sec", "text") if key not in row]
            if missing:
                result["issues"].append(_issue("error", f"Line {line_no} missing fields: {', '.join(missing)}", str(path)))
                continue
            try:
                start = float(row["start_sec"])
                end = float(row["end_sec"])
            except (TypeError, ValueError):
                result["issues"].append(_issue("error", f"Line {line_no} start_sec/end_sec must be numeric", str(path)))
                continue
            if start >= end:
                result["issues"].append(_issue("error", f"Line {line_no} start_sec must be smaller than end_sec", str(path)))
            if previous_start is not None and start < previous_start:
                result["issues"].append(_issue("warning", f"Line {line_no} starts before the previous utterance", str(path)))
            if previous_end is not None and start < previous_end:
                result["issues"].append(_issue("warning", f"Line {line_no} overlaps the previous utterance", str(path)))
            previous_start = start
            previous_end = end
            result["utterance_count"] += 1
    return result


def _validate_video_manifest_fields(source: VideoSource, prefix: str) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    if not source.path:
        issues.append(_issue("error", f"{prefix}.path is required"))
    try:
        parse_time(source.start_time)
    except Exception as exc:
        issues.append(_issue("error", f"{prefix}.start_time is not valid ISO8601: {exc}"))
    if source.fps is not None and source.fps <= 0:
        issues.append(_issue("error", f"{prefix}.fps must be positive"))
    return issues


def _validate_roi(roi: WorkbenchROI | None) -> list[dict[str, str]]:
    if roi is None:
        return []
    issues = []
    if roi.w <= 0 or roi.h <= 0:
        issues.append(_issue("error", "workbench_roi width and height must be positive"))
    if roi.x < 0 or roi.y < 0:
        issues.append(_issue("error", "workbench_roi x/y must be non-negative"))
    if roi.w > 10000 or roi.h > 10000:
        issues.append(_issue("warning", "workbench_roi is unusually large"))
    return issues


def validate_manifest(manifest_path: str | Path, validate_video: bool = True) -> dict[str, Any]:
    path = Path(manifest_path)
    result: dict[str, Any] = {
        "manifest_path": str(path),
        "exists": path.exists(),
        "manifest": None,
        "video_sources": {},
        "transcript": None,
        "input_sources": {},
        "issues": [],
        "can_run_real_pipeline": False,
    }
    if not path.exists():
        result["issues"].append(_issue("error", "Manifest file does not exist", str(path)))
        return result
    try:
        manifest = SessionManifest.load(path)
    except Exception as exc:
        result["issues"].append(_issue("error", f"Manifest cannot be parsed: {exc}", str(path)))
        return result

    result["manifest"] = manifest.to_json_dict()
    if not manifest.session_id:
        result["issues"].append(_issue("error", "session_id is required"))
    try:
        parse_time(manifest.session_start_time)
    except Exception as exc:
        result["issues"].append(_issue("error", f"session_start_time is not valid ISO8601: {exc}"))

    for view_id, source in manifest.videos.all_sources().items():
        result["issues"].extend(_validate_video_manifest_fields(source, f"videos.{view_id}"))
    result["issues"].extend(_validate_roi(manifest.workbench_roi))

    output_dir = Path(manifest.output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        result["issues"].append(_issue("error", f"output_dir cannot be created: {exc}", str(output_dir)))

    if validate_video:
        for view_id, source in manifest.videos.all_sources().items():
            result["video_sources"][view_id] = validate_video_source(source.path)
    else:
        for view_id, source in manifest.videos.all_sources().items():
            result["video_sources"][view_id] = {"path": source.path, "exists": Path(source.path).exists()}

    if manifest.transcript is not None:
        try:
            parse_time(manifest.transcript.start_time)
        except Exception as exc:
            result["issues"].append(_issue("error", f"transcript.start_time is not valid ISO8601: {exc}"))
        result["transcript"] = validate_transcript(manifest.transcript.path)

    for name, source in manifest.input_sources.items():
        source_result = {
            "path": source.path,
            "source_type": source.source_type,
            "event_type": source.event_type,
            "exists": Path(source.path).exists(),
            "required": source.required,
        }
        if source.required and not source_result["exists"]:
            issue = _issue("error", f"input_sources.{name}.path does not exist", source.path)
            source_result["issues"] = [issue]
            result["issues"].append(issue)
        else:
            source_result["issues"] = []
        result["input_sources"][name] = source_result

    for video_result in result["video_sources"].values():
        result["issues"].extend(video_result.get("issues", []))
    if result["transcript"]:
        result["issues"].extend(result["transcript"].get("issues", []))

    has_errors = any(item.get("severity") == "error" for item in result["issues"])
    result["can_run_real_pipeline"] = bool(validate_video and not has_errors)
    return result

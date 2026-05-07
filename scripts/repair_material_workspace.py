from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2


EVENT_TYPE_LABELS = {
    "hand_object_interaction": "手部接触与操作",
    "object_move": "物体移动与摆放",
    "liquid_transfer": "液体转移",
    "panel_operation": "设备/称量操作",
    "container_state_change": "容器状态变化",
}

# Keep repaired step names stable across Windows code pages.
EVENT_TYPE_LABELS = {
    "hand_object_interaction": "\u624b\u90e8\u63a5\u89e6\u4e0e\u64cd\u4f5c",
    "object_move": "\u7269\u4f53\u79fb\u52a8\u4e0e\u6446\u653e",
    "liquid_transfer": "\u6db2\u4f53\u8f6c\u79fb",
    "panel_operation": "\u8bbe\u5907/\u79f0\u91cf\u64cd\u4f5c",
    "container_state_change": "\u5bb9\u5668\u72b6\u6001\u53d8\u5316",
}


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def clean_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text or None


def event_time(event: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(event.get(key) if event.get(key) is not None else default)
    except (TypeError, ValueError):
        return default


def confidence_from_events(events: List[Dict[str, Any]]) -> float:
    grade_scores = {"strong": 0.9, "medium": 0.68, "weak": 0.42}
    values = [grade_scores.get(str(event.get("evidence_grade")), 0.58) for event in events]
    return round(sum(values) / len(values), 4) if values else 0.0


def build_aligned_steps(exp_dir: Path, experiment_id: str) -> List[Dict[str, Any]]:
    physical = read_json(exp_dir / "physical_events.json", {})
    events = physical.get("events") if isinstance(physical, dict) else physical
    events = sorted([event for event in (events or []) if isinstance(event, dict)], key=lambda item: event_time(item, "start_time_sec"))
    if not events:
        return []

    groups: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    current_type = ""
    current_start = 0.0
    for event in events:
        start = event_time(event, "start_time_sec")
        event_type = str(event.get("event_type") or "hand_object_interaction")
        should_split = False
        if current:
            elapsed = start - current_start
            should_split = elapsed >= 12.0 or (event_type != current_type and elapsed >= 4.0)
        if should_split:
            groups.append(current)
            current = []
        if not current:
            current_start = start
            current_type = event_type
        current.append(event)
    if current:
        groups.append(current)

    now = datetime.now(timezone.utc).isoformat()
    steps: List[Dict[str, Any]] = []
    for idx, group in enumerate(groups):
        event_type = Counter(str(event.get("event_type") or "hand_object_interaction") for event in group).most_common(1)[0][0]
        start = min(event_time(event, "start_time_sec") for event in group)
        end = max(event_time(event, "end_time_sec", event_time(event, "start_time_sec")) for event in group)
        if end <= start:
            end = start + 1.0
        label = EVENT_TYPE_LABELS.get(event_type, event_type)
        first_display = next((str(event.get("display_name")) for event in group if event.get("display_name")), label)
        confidence = confidence_from_events(group)
        steps.append(
            {
                "step_id": f"aligned_step_{idx:03d}",
                "protocol_step_id": f"aligned_step_{idx:03d}",
                "experiment_id": experiment_id,
                "step_index": idx,
                "step_name": label,
                "protocol_step_name": label,
                "step_description": first_display,
                "status": "candidate",
                "start_time_sec": round(start, 3),
                "end_time_sec": round(end, 3),
                "duration_sec": round(end - start, 3),
                "confidence": confidence,
                "step_confidence": "high" if confidence >= 0.75 else ("medium" if confidence >= 0.5 else "low"),
                "completed_by_inference": False,
                "inference_method": "material_stream_event_alignment",
                "inference_model": "material_stream_fallback",
                "required_event_types": [event_type],
                "optional_event_types": [],
                "critical_fields": [],
                "event_reuse_policy": "allow_reuse",
                "evidence_refs": [
                    {
                        "evidence_id": f"{str(event.get('event_id') or idx)}:evidence",
                        "evidence_type": "physical_event",
                        "source": "material_stream",
                        "timestamp_sec": event_time(event, "start_time_sec"),
                        "confidence": confidence_from_events([event]),
                        "description": event.get("display_name") or event.get("event_type"),
                    }
                    for event in group[:5]
                ],
                "parameters": [],
                "linked_context_events": [],
                "linked_physical_events": [str(event.get("event_id")) for event in group if event.get("event_id")],
                "metadata": {"source": "material_stream_event_alignment", "event_count": len(group)},
                "created_at": now,
                "updated_at": now,
            }
        )

    write_json(exp_dir / "steps.json", steps)
    existing_timeline = read_json(exp_dir / "timeline.json", {})
    timeline = {
        **existing_timeline,
        "timeline_id": existing_timeline.get("timeline_id") or f"timeline_{experiment_id}",
        "experiment_id": experiment_id,
        "title": existing_timeline.get("title") or experiment_id,
        "steps": steps,
        "total_steps": len(steps),
        "confirmed_steps": 0,
        "candidate_steps": len(steps),
        "inferred_steps": 0,
        "skipped_steps": 0,
        "avg_confidence": round(sum(float(step.get("confidence") or 0.0) for step in steps) / len(steps), 4),
        "start_time_sec": steps[0]["start_time_sec"],
        "end_time_sec": steps[-1]["end_time_sec"],
        "total_duration_sec": round(steps[-1]["end_time_sec"] - steps[0]["start_time_sec"], 3),
        "updated_at": now,
    }
    write_json(exp_dir / "timeline.json", timeline)
    return steps


def open_browser_writer(path: Path, fps: float, size: Tuple[int, int]) -> cv2.VideoWriter:
    for codec in ("avc1", "H264"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer
        writer.release()
    return cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)


def _find_ffmpeg() -> Optional[str]:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _strip_browser_suffix(stem: str) -> str:
    suffix = ".browser_h264"
    while stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    return stem


def _browser_h264_target(source: Path) -> Path:
    return source.with_name(f"{_strip_browser_suffix(source.stem)}.browser_h264{source.suffix}")


def _browser_webm_target(source: Path) -> Path:
    stem = _strip_browser_suffix(source.stem)
    if stem.endswith(".browser"):
        stem = stem[: -len(".browser")]
    return source.with_name(f"{stem}.browser.webm")


def _base_video_candidate(path: Path) -> Path:
    if ".browser_h264" not in path.stem:
        return path
    base = path.with_name(f"{_strip_browser_suffix(path.stem)}{path.suffix}")
    return base if base.exists() else path


def transcode_for_browser(source: Path, *, force: bool = True) -> Optional[Path]:
    if not source.exists() or source.suffix.lower() != ".mp4":
        return None
    source = _base_video_candidate(source)
    target = _browser_h264_target(source)
    if source.resolve() == target.resolve() and target.exists() and target.stat().st_size > 0:
        return target
    if target.exists() and target.stat().st_size > 0 and not force:
        return target

    ffmpeg_exe = _find_ffmpeg()
    if ffmpeg_exe:
        tmp = target.with_name(f"{target.stem}.tmp{target.suffix}")
        cmd = [
            ffmpeg_exe,
            "-y",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(tmp),
        ]
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
            if result.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
                shutil.move(str(tmp), str(target))
                return target
        except Exception as exc:
            print(f"[WARN] ffmpeg transcode failed for {source.name}: {exc}")
        tmp.unlink(missing_ok=True)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tmp = target.with_name(f"{target.stem}.opencv_tmp{target.suffix}")
    writer = open_browser_writer(tmp, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        return None
    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        frame_count += 1
        if frame_count % 300 == 0:
            print(f"  transcoded {frame_count} frames from {source.name}")
    cap.release()
    writer.release()
    if tmp.exists() and tmp.stat().st_size > 0:
        shutil.move(str(tmp), str(target))
        return target
    tmp.unlink(missing_ok=True)
    return None


def transcode_for_browser_webm(source: Path, *, force: bool = True) -> Optional[Path]:
    if not source.exists() or not source.is_file():
        return None
    target = _browser_webm_target(source)
    if target.exists() and target.stat().st_size > 0 and not force:
        return target
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        return None
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 15.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        return None
    tmp = target.with_name(f"{target.stem}.tmp{target.suffix}")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"VP80"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        tmp.unlink(missing_ok=True)
        return None
    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        frame_count += 1
        if frame_count % 300 == 0:
            print(f"  webm transcoded {frame_count} frames from {source.name}")
    cap.release()
    writer.release()
    if tmp.exists() and tmp.stat().st_size > 0 and frame_count > 0:
        shutil.move(str(tmp), str(target))
        return target
    tmp.unlink(missing_ok=True)
    return None


def browser_compatible_clip(source: Path) -> Optional[Path]:
    ffmpeg_exe = _find_ffmpeg()
    if ffmpeg_exe:
        converted = transcode_for_browser(source, force=True)
        if converted and converted.exists():
            return converted
    return transcode_for_browser_webm(source, force=True)


OLD_PROJECT_ROOTS = (
    "D:/LabEmbodiedVLA/LabSOPGuard",
    "C:/Users/Xx7/Desktop/LabTest/LabSOPGuard",
)


def rewrite_project_path_value(value: Any, project_root: Path) -> Any:
    if not isinstance(value, str):
        return value
    normalized = value.replace("\\", "/")
    for old_root in OLD_PROJECT_ROOTS:
        old = old_root.replace("\\", "/").rstrip("/")
        if normalized.lower() == old.lower() or normalized.lower().startswith(f"{old.lower()}/"):
            suffix = normalized[len(old):].lstrip("/")
            return str((project_root / suffix).resolve())
    marker = "outputs/experiments/"
    idx = normalized.find(marker)
    if idx >= 0:
        return str((project_root / normalized[idx:]).resolve())
    return value


def rewrite_project_paths(value: Any, project_root: Path) -> Any:
    if isinstance(value, dict):
        return {key: rewrite_project_paths(item, project_root) for key, item in value.items()}
    if isinstance(value, list):
        return [rewrite_project_paths(item, project_root) for item in value]
    return rewrite_project_path_value(value, project_root)


def existing_video_candidates(values: List[Any], project_root: Path, exp_dir: Path, *, include_uploads: bool = True) -> List[Path]:
    candidates: List[Path] = []
    seen = set()
    for value in values:
        if not value:
            continue
        rewritten = rewrite_project_path_value(value, project_root)
        path = Path(str(rewritten))
        if not path.is_absolute():
            path = project_root / path
        base = _base_video_candidate(path)
        for candidate in (base, path):
            if candidate.suffix.lower() != ".mp4" or not candidate.exists() or not candidate.is_file():
                continue
            key = str(candidate.resolve()).lower()
            if key not in seen:
                seen.add(key)
                candidates.append(candidate)
    if include_uploads:
        upload_dir = exp_dir / "uploads"
        upload_videos = sorted(upload_dir.glob("*.mp4")) if upload_dir.exists() else []
        preferred = [path for path in upload_videos if ".browser_h264" not in path.stem]
        preferred += [path for path in upload_videos if path.stem.endswith(".browser_h264") and ".browser_h264.browser_h264" not in path.stem]
        preferred += upload_videos
        for candidate in preferred:
            key = str(candidate.resolve()).lower()
            if key not in seen:
                seen.add(key)
                candidates.append(candidate)
    return candidates


def update_video_paths(project_root: Path, exp_dir: Path, experiment_id: str, title: Optional[str], transcode: bool) -> Dict[str, Any]:
    experiment_path = exp_dir / "experiment.json"
    experiment = rewrite_project_paths(read_json(experiment_path, {}), project_root)
    title = clean_text(title)
    if title:
        experiment["title"] = title
        experiment["experiment_name"] = title

    output_paths = experiment.setdefault("output_paths", {})
    if transcode:
        source_values: List[Any] = []
        source_values.extend(output_paths.get("source_videos") or [])
        if experiment.get("video_paths"):
            source_values.extend(experiment.get("video_paths") or [])
        if output_paths.get("source_video"):
            source_values.append(output_paths.get("source_video"))
        for item in experiment.get("video_inputs") or []:
            if isinstance(item, dict):
                source_values.extend([item.get("video_path"), item.get("source")])
        for item in experiment.get("video_assets") or []:
            if isinstance(item, dict):
                source_values.append(item.get("file_path"))

        source_h264 = None
        source_original = None
        for path in existing_video_candidates(source_values, project_root, exp_dir):
            source_original = _base_video_candidate(path)
            source_h264 = transcode_for_browser(source_original, force=True)
            if source_h264:
                break
        if source_h264:
            experiment["video_paths"] = [str(source_h264)]
            if experiment.get("video_inputs"):
                experiment["video_inputs"][0]["video_path"] = str(source_h264)
                experiment["video_inputs"][0]["source"] = str(source_h264)
            output_paths["source_video"] = str(source_h264)
            if source_original and source_original.exists():
                output_paths["source_videos"] = [str(source_original)]

        annotated_values: List[Any] = [
            exp_dir / "analysis" / "annotated.mp4",
            output_paths.get("annotated_video"),
            exp_dir / "analysis" / "annotated.browser_h264.mp4",
        ]
        annotated_h264 = None
        for annotated in existing_video_candidates(annotated_values, project_root, exp_dir, include_uploads=False):
            annotated_h264 = transcode_for_browser(_base_video_candidate(annotated), force=True)
            if annotated_h264:
                break
        if annotated_h264:
            output_paths["annotated_video"] = str(annotated_h264)

    write_json(experiment_path, experiment)

    registry_path = project_root / "outputs" / "experiments" / "experiments.json"
    registry = read_json(registry_path, [])
    if isinstance(registry, list):
        for item in registry:
            if isinstance(item, dict) and item.get("experiment_id") == experiment_id:
                item.update(rewrite_project_paths(item, project_root))
                if title:
                    item["title"] = title
                    item["experiment_name"] = title
                item["output_paths"] = output_paths
                item["video_paths"] = experiment.get("video_paths") or item.get("video_paths")
                item["video_inputs"] = experiment.get("video_inputs") or item.get("video_inputs")
        write_json(registry_path, registry)
    return experiment


def _as_detection(item: Dict[str, Any], frame_idx: int, timestamp_sec: float):
    from labsopguard.video_analysis import DetectionResult

    bbox = item.get("bbox") or item.get("xyxy") or item.get("box") or [0, 0, 0, 0]
    if not isinstance(bbox, list) or len(bbox) < 4:
        bbox = [0, 0, 0, 0]
    keypoints = item.get("keypoints")
    if keypoints is not None:
        keypoints = [tuple(int(v) for v in pair[:2]) for pair in keypoints if isinstance(pair, list) and len(pair) >= 2]
    return DetectionResult(
        frame_idx=int(item.get("frame_idx") if item.get("frame_idx") is not None else frame_idx),
        timestamp_sec=float(item.get("timestamp_sec") if item.get("timestamp_sec") is not None else timestamp_sec),
        bbox=tuple(int(float(v)) for v in bbox[:4]),
        class_name=str(item.get("class_name") or item.get("label") or item.get("name") or "object"),
        confidence=float(item.get("confidence") if item.get("confidence") is not None else item.get("score") or 0.0),
        keypoints=keypoints,
    )


def _as_frame_analysis(item: Dict[str, Any]):
    from labsopguard.video_analysis import FrameAnalysis

    frame_idx = int(item.get("frame_idx") or 0)
    timestamp_sec = float(item.get("timestamp_sec") or 0.0)
    detections = [
        _as_detection(det, frame_idx, timestamp_sec)
        for det in (item.get("detections") or [])
        if isinstance(det, dict)
    ]
    return FrameAnalysis(
        frame_idx=frame_idx,
        timestamp_sec=timestamp_sec,
        detections=detections,
        scene_description=str(item.get("scene_description") or ""),
        detected_activities=[str(value) for value in (item.get("detected_activities") or [])],
        object_labels=[str(value) for value in (item.get("object_labels") or [])],
        step_indicators=[str(value) for value in (item.get("step_indicators") or [])],
        ppe_status=item.get("ppe_status") if isinstance(item.get("ppe_status"), dict) else {},
        vlm_confidence=float(item.get("vlm_confidence") or 0.0),
        alerts=[str(value) for value in (item.get("alerts") or [])],
        alert_details=[value for value in (item.get("alert_details") or []) if isinstance(value, dict)],
    )


def load_frame_analyses(path: Path) -> List[Any]:
    payload = read_json(path, [])
    if isinstance(payload, dict):
        payload = payload.get("analyses") or payload.get("items") or []
    if not isinstance(payload, list):
        return []
    return [_as_frame_analysis(item) for item in payload if isinstance(item, dict)]


def regenerate_annotated_video(project_root: Path, exp_dir: Path, experiment: Dict[str, Any], transcode: bool) -> Optional[Path]:
    output_paths = experiment.get("output_paths") or {}
    source_value = output_paths.get("source_video") or (experiment.get("video_paths") or [None])[0]
    if not source_value:
        return None
    source_video = Path(str(source_value))
    if not source_video.exists():
        return None

    from labsopguard.config import load_runtime_settings
    from labsopguard.video_analysis import VideoAnalysisPipeline

    settings = load_runtime_settings(project_root)
    try:
        overlay_conf = float(os.environ.get("LABSOPGUARD_MATERIAL_YOLO_CONF", "0.08"))
        settings.confidence_threshold = min(float(settings.confidence_threshold), overlay_conf)
        settings.max_detections = max(int(settings.max_detections), 50)
    except Exception:
        pass
    pipeline = VideoAnalysisPipeline(settings=settings, yolo_model_path=settings.yolo_model_path)
    analyses = load_frame_analyses(exp_dir / "analysis" / "analysis.json")
    output = exp_dir / "analysis" / "annotated.mp4"
    print(f"[INFO] regenerating annotated video with YOLO overlays: {output}")
    pipeline.create_annotated_video(str(source_video), analyses, str(output))
    if transcode:
        return transcode_for_browser(output, force=True) or output
    return output


def transcode_published_clips_for_browser(exp_dir: Path, published: Dict[str, Any]) -> Dict[str, Any]:
    converted_count = 0
    for item in published.get("items") or []:
        if not isinstance(item, dict):
            continue
        paths = item.get("published_paths") or {}
        clip_value = paths.get("clip")
        if not clip_value:
            continue
        clip_path = Path(str(clip_value))
        if not clip_path.exists():
            continue
        browser_clip = browser_compatible_clip(clip_path)
        if not browser_clip or not browser_clip.exists():
            continue
        paths["clip"] = str(browser_clip)
        item["published_paths"] = paths
        extra = item.setdefault("extra", {})
        extra["browser_clip_format"] = browser_clip.suffix.lstrip(".")
        extra["browser_clip_source"] = str(clip_path)
        converted_count += 1

        material_publish = paths.get("material_publish")
        if material_publish:
            material_publish_path = Path(str(material_publish))
            if material_publish_path.exists():
                payload = read_json(material_publish_path, {})
                if isinstance(payload, dict):
                    payload["published_paths"] = paths
                    payload["extra"] = {**(payload.get("extra") or {}), **extra}
                    write_json(material_publish_path, payload)

    write_json(exp_dir / "published_materials.json", published)
    print(f"[OK] browser_compatible_clips={converted_count}")
    return published


def count_published_clips(payload: Dict[str, Any]) -> int:
    count = 0
    for item in payload.get("items") or []:
        paths = item.get("published_paths") or {}
        clip = paths.get("clip")
        if clip and Path(str(clip)).exists():
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair published materials, aligned steps, title, and browser video paths for one experiment.")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--title", default=None)
    parser.add_argument("--skip-video-transcode", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    sys.path.insert(0, str(project_root / "src"))

    from labsopguard.material_publishing import SemanticMaterialPublisher

    exp_dir = project_root / "outputs" / "experiments" / args.experiment_id
    if not exp_dir.exists():
        raise SystemExit(f"Experiment output directory not found: {exp_dir}")

    print(f"[INFO] project_root={project_root}")
    print(f"[INFO] experiment_id={args.experiment_id}")
    title = clean_text(args.title)
    os.environ.setdefault("LABSOPGUARD_FORCE_YOLO_REPAIR", "1")
    os.environ.setdefault("LABSOPGUARD_MATERIAL_YOLO_CONF", "0.08")
    experiment = update_video_paths(project_root, exp_dir, args.experiment_id, title, not args.skip_video_transcode)
    annotated_video = regenerate_annotated_video(project_root, exp_dir, experiment, not args.skip_video_transcode)
    if annotated_video:
        experiment = read_json(exp_dir / "experiment.json", {})
        output_paths = experiment.setdefault("output_paths", {})
        output_paths["annotated_video"] = str(annotated_video)
        write_json(exp_dir / "experiment.json", experiment)
    published = SemanticMaterialPublisher(exp_dir, experiment_id=args.experiment_id).publish()["published_materials"]
    published = transcode_published_clips_for_browser(exp_dir, published)
    steps = build_aligned_steps(exp_dir, args.experiment_id)
    print(f"[OK] published_materials={published['total']}")
    print(f"[OK] published_clips={count_published_clips(published)}")
    print(f"[OK] aligned_steps={len(steps)}")
    print(f"[OK] workspace repaired: {exp_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

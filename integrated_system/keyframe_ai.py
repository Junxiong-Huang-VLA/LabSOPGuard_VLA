from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from openai_wrapper import analyze_keyframes_with_openai


def _safe_write_jpg(path: Path, frame: np.ndarray) -> bool:
    ok = cv2.imwrite(str(path), frame)
    if ok:
        return True
    enc_ok, buf = cv2.imencode(".jpg", frame)
    if not enc_ok:
        return False
    try:
        buf.tofile(str(path))
        return True
    except Exception:
        return False


def extract_keyframes_by_diff(
    video_path: str,
    output_dir: Path,
    diff_threshold: float,
    min_interval_sec: float,
    max_keyframes: int,
    progress_callback: Optional[Callable[[int, int, int, str], None]] = None,
) -> Tuple[List[Dict[str, Any]], List[Path]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 25.0
    total_frames_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_frames = total_frames_raw if total_frames_raw > 0 else 0
    progress_interval = max(8, int(round(fps)))

    output_dir.mkdir(parents=True, exist_ok=True)

    keyframes_meta: List[Dict[str, Any]] = []
    saved_paths: List[Path] = []
    prev_gray = None
    last_saved_t = -1e9

    frame_id = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1
        processed = frame_id + 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if prev_gray is None:
            prev_gray = gray
            t = frame_id / fps
            out = output_dir / f"keyframe_{len(saved_paths)+1:03d}.jpg"
            if _safe_write_jpg(out, frame):
                saved_paths.append(out)
                keyframes_meta.append({"frame_id": frame_id, "timestamp": t, "score": 1.0, "image": out.name})
                last_saved_t = t
            if progress_callback:
                progress_callback(processed, total_frames, len(saved_paths), "关键帧提取中：正在处理视频帧。")
            if len(saved_paths) >= max_keyframes:
                break
            continue

        diff = cv2.absdiff(prev_gray, gray)
        score = float(np.mean(diff))
        t = frame_id / fps

        if score >= diff_threshold and (t - last_saved_t) >= min_interval_sec:
            out = output_dir / f"keyframe_{len(saved_paths)+1:03d}.jpg"
            if _safe_write_jpg(out, frame):
                saved_paths.append(out)
                keyframes_meta.append({"frame_id": frame_id, "timestamp": t, "score": score, "image": out.name})
                last_saved_t = t
                if len(saved_paths) >= max_keyframes:
                    break

        prev_gray = gray
        if progress_callback and (processed % progress_interval == 0):
            progress_callback(processed, total_frames, len(saved_paths), "关键帧提取中：正在扫描视频变化。")

    cap.release()
    if progress_callback:
        final_processed = max(frame_id + 1, 0)
        progress_callback(
            final_processed if final_processed > 0 else total_frames,
            total_frames,
            len(saved_paths),
            f"关键帧提取完成：共提取 {len(saved_paths)} 帧。",
        )
    return keyframes_meta, saved_paths


def run_keyframe_ai_pipeline(
    video_path: str,
    output_dir: Path,
    diff_threshold: float,
    min_interval_sec: float,
    max_keyframes: int,
    enable_ai_analysis: bool,
    ai_model: str,
    ai_base_url: str,
    extraction_progress_callback: Optional[Callable[[int, int, int, str], None]] = None,
    ai_progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    keyframe_dir = output_dir / "keyframes"
    keyframe_meta, keyframe_paths = extract_keyframes_by_diff(
        video_path=video_path,
        output_dir=keyframe_dir,
        diff_threshold=diff_threshold,
        min_interval_sec=min_interval_sec,
        max_keyframes=max_keyframes,
        progress_callback=extraction_progress_callback,
    )

    part1 = {
        "video_path": video_path,
        "keyframe_count": len(keyframe_paths),
        "keyframes": keyframe_meta,
    }
    (output_dir / "part1_keyframes.json").write_text(
        json.dumps(part1, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if enable_ai_analysis:
        ai_result = analyze_keyframes_with_openai(
            image_paths=keyframe_paths,
            model=ai_model,
            base_url=ai_base_url,
            progress_callback=ai_progress_callback,
        )
    else:
        if ai_progress_callback:
            ai_progress_callback(0, len(keyframe_paths), "AI 分析已跳过：开关关闭。")
        ai_result = {
            "enabled": False,
            "reason": "disabled_by_option",
            "analyses": [
                {
                    "image": p.name,
                    "summary": "AI analysis disabled by request.",
                    "risk_level": "unknown",
                }
                for p in keyframe_paths
            ],
            "overall_summary": f"Extracted {len(keyframe_paths)} keyframes. AI analysis disabled.",
        }

    (output_dir / "keyframe_ai_analysis.json").write_text(
        json.dumps(ai_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "overall_summary.txt").write_text(
        ai_result.get("overall_summary", ""),
        encoding="utf-8",
    )

    return {
        "part1_keyframes": (output_dir / "part1_keyframes.json").as_posix(),
        "overall_summary": (output_dir / "overall_summary.txt").as_posix(),
        "keyframe_ai_analysis": (output_dir / "keyframe_ai_analysis.json").as_posix(),
        "keyframes": keyframe_meta,
        "analysis": ai_result,
    }

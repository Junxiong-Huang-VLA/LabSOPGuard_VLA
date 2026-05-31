"""Per-segment experiment naming with optional VLM scene understanding."""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from labsopguard.resilience import CircuitBreaker, RateLimiter, RetryConfig, resilient_call

logger = logging.getLogger(__name__)


_NAMING_BREAKER = CircuitBreaker("segment-vlm-naming", failure_threshold=3, cooldown_sec=60.0)
_NAMING_RATE_LIMITER = RateLimiter(float(os.environ.get("LABSOPGUARD_SEGMENT_NAMING_QPS", "0.5")))


def name_experiment_segments(
    *,
    video_path: str | Path,
    segmentation: Any,
    output_dir: str | Path,
    vlm_client: Optional[Any] = None,
    max_segments: Optional[int] = None,
) -> Dict[str, Any]:
    """Assign a stable display name and scene summary to each experiment segment.

    The function mutates ``segmentation.segments`` so downstream JSON, previews,
    and frontend cards all use the same per-segment identity.
    """
    segments = list(getattr(segmentation, "segments", []) or [])
    output = Path(output_dir)
    frame_dir = output / "segment_naming_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    limit = int(max_segments or os.environ.get("LABSOPGUARD_SEGMENT_NAMING_MAX_SEGMENTS", "10"))
    vlm_enabled = _env_enabled("LABSOPGUARD_SEGMENT_NAMING_VLM_ENABLED", default=True)
    client = vlm_client if vlm_client is not None else (_build_vlm_client() if vlm_enabled else None)

    manifest: Dict[str, Any] = {
        "schema_version": "segment_naming.v1",
        "video_path": str(video_path),
        "vlm_enabled": bool(client),
        "segment_count": len(segments),
        "segments": [],
    }

    for segment in segments:
        fallback = _fallback_metadata(segment, len(segments))
        setattr(segment, "display_name", fallback["display_name"])
        setattr(segment, "scene_summary", fallback["scene_summary"])
        setattr(segment, "naming_confidence", fallback["naming_confidence"])
        setattr(segment, "naming_source", fallback["naming_source"])

        frame_path: Optional[Path] = None
        if int(getattr(segment, "index", len(manifest["segments"]))) < limit:
            frame_path = _extract_segment_contact_sheet(Path(video_path), segment, frame_dir)

        vlm_error = None
        if client is not None and frame_path is not None:
            try:
                vlm_meta = _name_with_vlm(client, frame_path, segment)
                if vlm_meta:
                    setattr(segment, "display_name", vlm_meta["display_name"])
                    setattr(segment, "scene_summary", vlm_meta["scene_summary"])
                    setattr(segment, "naming_confidence", vlm_meta["naming_confidence"])
                    setattr(segment, "naming_source", vlm_meta["naming_source"])
            except Exception as exc:
                vlm_error = str(exc)
                logger.warning(
                    "VLM segment naming failed for %s: %s",
                    getattr(segment, "segment_id", "segment"),
                    exc,
                )

        manifest["segments"].append(
            {
                "segment_id": getattr(segment, "segment_id", ""),
                "index": int(getattr(segment, "index", len(manifest["segments"]))),
                "start_sec": round(float(getattr(segment, "start_sec", 0.0)), 3),
                "end_sec": round(float(getattr(segment, "end_sec", 0.0)), 3),
                "display_name": getattr(segment, "display_name", ""),
                "scene_summary": getattr(segment, "scene_summary", ""),
                "naming_confidence": round(float(getattr(segment, "naming_confidence", 0.0)), 3),
                "naming_source": getattr(segment, "naming_source", "fallback"),
                "representative_frame_path": str(frame_path) if frame_path else None,
                "vlm_error": vlm_error,
            }
        )

    return manifest


def _name_with_vlm(client: Any, frame_path: Path, segment: Any) -> Optional[Dict[str, Any]]:
    prompt = (
        "请基于这张由同一实验片段的起始/中间/结束帧拼成的实验室画面，判断该片段最可能是什么实验或操作。"
        "只返回严格 JSON，不要 Markdown。字段如下："
        '{"experiment_name":"简短中文名称，不超过14个字",'
        '"scene_summary":"一句话说明该片段里看见的关键操作、器材或材料",'
        '"visible_equipment":["器材或材料"],'
        '"likely_experiment_type":"实验类型或未知",'
        '"confidence":0.0}'
    )
    timeout = float(os.environ.get("LABSOPGUARD_SEGMENT_NAMING_VLM_TIMEOUT_SEC", "20"))
    retries = int(os.environ.get("LABSOPGUARD_SEGMENT_NAMING_RETRIES", "1"))
    desc = resilient_call(
        client.describe_scene,
        str(frame_path),
        prompt=prompt,
        temperature=0.0,
        retry_config=RetryConfig(max_retries=retries, backoff_factor=1.5, max_backoff=6.0, timeout=timeout),
        circuit_breaker=_NAMING_BREAKER,
        rate_limiter=_NAMING_RATE_LIMITER,
        fallback=None,
    )
    if desc is None:
        return None

    parsed = _coerce_json(getattr(desc, "raw_response", None), getattr(desc, "description", ""))
    name = _clean_name(parsed.get("experiment_name") or parsed.get("likely_experiment_type") or "")
    summary = str(parsed.get("scene_summary") or parsed.get("description") or getattr(desc, "description", "") or "").strip()
    confidence = _safe_float(parsed.get("confidence"), getattr(desc, "confidence", 0.6))
    if not name:
        name = _fallback_metadata(segment, 1)["display_name"]
    if not summary:
        summary = f"{format_time(getattr(segment, 'start_sec', 0.0))} - {format_time(getattr(segment, 'end_sec', 0.0))} 的实验片段"
    return {
        "display_name": name,
        "scene_summary": summary[:180],
        "naming_confidence": max(0.0, min(1.0, confidence)),
        "naming_source": f"vlm:{getattr(desc, 'model', 'unknown')}",
    }


def _extract_segment_contact_sheet(video_path: Path, segment: Any, frame_dir: Path) -> Optional[Path]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        start = max(0.0, float(getattr(segment, "start_sec", 0.0)))
        end = max(start, float(getattr(segment, "end_sec", start)))
        duration = max(end - start, 0.1)
        timestamps = [start + duration * ratio for ratio in (0.2, 0.5, 0.8)]
        frames: List[np.ndarray] = []
        for timestamp in timestamps:
            frame_idx = int(round(timestamp * fps))
            if total_frames > 0:
                frame_idx = min(max(frame_idx, 0), total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue
            frames.append(_prepare_frame(frame, timestamp))
        if not frames:
            return None
        while len(frames) < 3:
            frames.append(frames[-1].copy())
        sheet = np.hstack(frames[:3])
        out_path = frame_dir / f"segment_{int(getattr(segment, 'index', 0)):02d}_contact.jpg"
        cv2.imwrite(str(out_path), sheet)
        return out_path if out_path.exists() else None
    finally:
        cap.release()


def _prepare_frame(frame: np.ndarray, timestamp: float) -> np.ndarray:
    target_w = 360
    h, w = frame.shape[:2]
    scale = target_w / max(w, 1)
    target_h = max(120, int(h * scale))
    resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
    label_h = 36
    canvas = np.zeros((target_h + label_h, target_w, 3), dtype=np.uint8)
    canvas[:target_h, :] = resized
    cv2.putText(
        canvas,
        format_time(timestamp),
        (12, target_h + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    return canvas


def _fallback_metadata(segment: Any, total_segments: int) -> Dict[str, Any]:
    index = int(getattr(segment, "index", 0))
    start = float(getattr(segment, "start_sec", 0.0))
    end = float(getattr(segment, "end_sec", start))
    name = f"实验 {index + 1}" if total_segments != 1 else "连续实验"
    return {
        "display_name": name,
        "scene_summary": f"{format_time(start)} - {format_time(end)} 的独立实验片段，等待 VLM 场景命名。",
        "naming_confidence": 0.3,
        "naming_source": "fallback",
    }


def _build_vlm_client() -> Optional[Any]:
    if not os.environ.get("DASHSCOPE_API_KEY"):
        return None
    try:
        from experiment.vlm_client import DashScopeVLClient

        return DashScopeVLClient(timeout=int(float(os.environ.get("LABSOPGUARD_SEGMENT_NAMING_VLM_TIMEOUT_SEC", "20"))))
    except Exception as exc:
        logger.warning("Segment naming VLM is unavailable, using fallback names: %s", exc)
        return None


def _coerce_json(raw_response: Any, text: str) -> Dict[str, Any]:
    if isinstance(raw_response, dict) and raw_response:
        return raw_response
    value = str(text or "").strip()
    if not value:
        return {}
    value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*```$", "", value)
    start = value.find("{")
    end = value.rfind("}")
    if start >= 0 and end > start:
        value = value[start : end + 1]
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {"description": text}


def _clean_name(value: str) -> str:
    name = re.sub(r"\s+", "", str(value or "").strip())
    if name in {"未知", "unknown", "Unknown", "N/A", "NA"}:
        return ""
    return name[:24]


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _env_enabled(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def format_time(seconds: Any) -> str:
    total = max(0, int(round(_safe_float(seconds, 0.0))))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2


@dataclass
class BackfilledClip:
    clip_id: str
    camera_id: Optional[str]
    start_time_sec: float
    end_time_sec: float
    file_path: str
    source_segment_count: int
    file_exists: bool
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _load_material_stream(path: str | Path) -> List[Dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        items = payload.get("items") or payload.get("material_stream") or []
        return [item for item in items if isinstance(item, dict)]
    return []


def _item_start(item: Dict[str, Any]) -> float:
    return float(item.get("timestamp_sec") or item.get("start_time_sec") or item.get("local_timestamp_sec") or 0.0)


def _item_end(item: Dict[str, Any]) -> float:
    if item.get("end_time_sec") is not None:
        return float(item.get("end_time_sec"))
    if item.get("duration_sec") is not None:
        return _item_start(item) + float(item.get("duration_sec"))
    return _item_start(item)


def _item_path(item: Dict[str, Any]) -> Optional[Path]:
    for key in ("recorded_file_path", "clip_file_path", "file_path", "source_path"):
        value = item.get(key)
        if value:
            return Path(str(value))
    return None


def select_material_segments(
    material_stream: Sequence[Dict[str, Any]],
    *,
    start_time_sec: float,
    end_time_sec: float,
    camera_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for item in material_stream:
        if camera_id and item.get("camera_id") != camera_id and item.get("stream_id") != camera_id:
            continue
        item_start = _item_start(item)
        item_end = _item_end(item)
        if item_start <= end_time_sec and item_end >= start_time_sec:
            selected.append(item)
    selected.sort(key=_item_start)
    return selected


def backfill_clip_from_material_stream(
    material_stream_path: str | Path,
    *,
    start_time_sec: float,
    end_time_sec: float,
    output_path: str | Path,
    camera_id: Optional[str] = None,
    clip_id: Optional[str] = None,
) -> BackfilledClip:
    if end_time_sec <= start_time_sec:
        raise ValueError("end_time_sec must be greater than start_time_sec")

    material_stream = _load_material_stream(material_stream_path)
    segments = select_material_segments(
        material_stream,
        start_time_sec=float(start_time_sec),
        end_time_sec=float(end_time_sec),
        camera_id=camera_id,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    warnings: List[str] = []
    writer = None
    source_count = 0
    try:
        for segment in segments:
            path = _item_path(segment)
            if path is None or not path.exists():
                warnings.append(f"missing segment file: {path}")
                continue
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                warnings.append(f"cannot open segment file: {path}")
                continue
            fps = float(cap.get(cv2.CAP_PROP_FPS) or segment.get("fps") or 30.0)
            segment_start = _item_start(segment)
            segment_end = _item_end(segment)
            start_frame = max(0, int(max(0.0, start_time_sec - segment_start) * fps))
            end_frame = int(max(0.0, min(end_time_sec, segment_end) - segment_start) * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_number = start_frame
            wrote_from_segment = False
            while frame_number <= end_frame:
                ok, frame = cap.read()
                if not ok:
                    break
                if writer is None:
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                    if not writer.isOpened():
                        raise RuntimeError(f"cannot open clip writer: {output}")
                writer.write(frame)
                wrote_from_segment = True
                frame_number += 1
            cap.release()
            if wrote_from_segment:
                source_count += 1
    finally:
        if writer is not None:
            writer.release()

    if source_count == 0:
        warnings.append("no overlapping materialized segments were written")

    return BackfilledClip(
        clip_id=clip_id or output.stem,
        camera_id=camera_id,
        start_time_sec=round(float(start_time_sec), 3),
        end_time_sec=round(float(end_time_sec), 3),
        file_path=str(output),
        source_segment_count=source_count,
        file_exists=output.exists() and output.stat().st_size > 0,
        warnings=warnings,
    )

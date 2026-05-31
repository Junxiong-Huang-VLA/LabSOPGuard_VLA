from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

from labsopguard.runtime_paths import STREAM_BUFFER_ROOT, ensure_runtime_dirs


@dataclass
class StreamSegment:
    segment_id: str
    camera_id: str
    source_id: str
    start_time_sec: float
    end_time_sec: float
    file_path: str
    fps: float
    frame_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RingSegmentRecorder:
    """Segmented stream recorder that keeps a bounded history for clip backfill."""

    def __init__(
        self,
        camera_id: str,
        source_id: str = "stream",
        output_dir: Optional[str | Path] = None,
        segment_duration_sec: float = 10.0,
        retention_sec: float = 300.0,
        fps: float = 30.0,
    ) -> None:
        ensure_runtime_dirs()
        self.camera_id = camera_id
        self.source_id = source_id
        self.output_dir = Path(output_dir) if output_dir else STREAM_BUFFER_ROOT / camera_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.segment_duration_sec = max(1.0, float(segment_duration_sec))
        self.retention_sec = max(self.segment_duration_sec, float(retention_sec))
        self.fps = max(1.0, float(fps))
        self.manifest_path = self.output_dir / "segments.json"
        self.segments: List[StreamSegment] = self._load_manifest()
        self._writer = None
        self._current_path: Optional[Path] = None
        self._current_start: Optional[float] = None
        self._current_frame_count = 0

    def _load_manifest(self) -> List[StreamSegment]:
        if not self.manifest_path.exists():
            return []
        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            return [StreamSegment(**item) for item in payload.get("segments", [])]
        except Exception:
            return []

    def _save_manifest(self) -> None:
        payload = {"segments": [segment.to_dict() for segment in self.segments]}
        self.manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def append_frame(self, frame_bgr: Any, timestamp_sec: float) -> None:
        timestamp = float(timestamp_sec)
        if self._writer is None or self._current_start is None:
            self._start_segment(timestamp, frame_bgr)
        elif timestamp - self._current_start >= self.segment_duration_sec:
            self._finish_segment(timestamp)
            self._start_segment(timestamp, frame_bgr)
        if self._writer is not None:
            self._writer.write(frame_bgr)
            self._current_frame_count += 1

    def _start_segment(self, timestamp_sec: float, frame_bgr: Any) -> None:
        height, width = frame_bgr.shape[:2]
        segment_id = f"{self.source_id}_{self.camera_id}_{int(timestamp_sec * 1000):012d}"
        self._current_path = self.output_dir / f"{segment_id}.mp4"
        self._current_start = float(timestamp_sec)
        self._current_frame_count = 0
        self._writer = cv2.VideoWriter(
            str(self._current_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (width, height),
        )
        if not self._writer.isOpened():
            self._writer = None

    def _finish_segment(self, end_time_sec: float) -> None:
        if self._writer is not None:
            self._writer.release()
        if self._current_path is not None and self._current_start is not None and self._current_path.exists():
            self.segments.append(
                StreamSegment(
                    segment_id=self._current_path.stem,
                    camera_id=self.camera_id,
                    source_id=self.source_id,
                    start_time_sec=round(self._current_start, 3),
                    end_time_sec=round(float(end_time_sec), 3),
                    file_path=str(self._current_path),
                    fps=self.fps,
                    frame_count=self._current_frame_count,
                )
            )
            self._prune(end_time_sec)
            self._save_manifest()
        self._writer = None
        self._current_path = None
        self._current_start = None
        self._current_frame_count = 0

    def close(self, end_time_sec: Optional[float] = None) -> None:
        if self._writer is not None:
            end_time = float(end_time_sec) if end_time_sec is not None else (
                (self._current_start or 0.0) + self._current_frame_count / self.fps
            )
            self._finish_segment(end_time)

    def _prune(self, newest_time_sec: float) -> None:
        keep_after = float(newest_time_sec) - self.retention_sec
        retained: List[StreamSegment] = []
        for segment in self.segments:
            if segment.end_time_sec >= keep_after:
                retained.append(segment)
                continue
            path = Path(segment.file_path)
            if path.exists():
                path.unlink()
        self.segments = retained

    def segments_for_range(self, start_time_sec: float, end_time_sec: float) -> List[StreamSegment]:
        self.close()
        return [
            segment
            for segment in self.segments
            if segment.start_time_sec <= end_time_sec and segment.end_time_sec >= start_time_sec
        ]

    def cut_clip(self, start_time_sec: float, end_time_sec: float, output_path: str | Path) -> Optional[str]:
        segments = self.segments_for_range(start_time_sec, end_time_sec)
        if not segments:
            return None
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        writer = None
        try:
            for segment in segments:
                cap = cv2.VideoCapture(segment.file_path)
                if not cap.isOpened():
                    continue
                fps = float(cap.get(cv2.CAP_PROP_FPS) or segment.fps or self.fps)
                segment_start_frame = max(0, int(max(0.0, start_time_sec - segment.start_time_sec) * fps))
                segment_end_frame = int(max(0.0, min(end_time_sec, segment.end_time_sec) - segment.start_time_sec) * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, segment_start_frame)
                frame_number = segment_start_frame
                while frame_number <= segment_end_frame:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if writer is None:
                        height, width = frame.shape[:2]
                        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                        if not writer.isOpened():
                            return None
                    writer.write(frame)
                    frame_number += 1
                cap.release()
            return str(output) if output.exists() else None
        finally:
            if writer is not None:
                writer.release()


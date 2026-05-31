"""Layer 0: Lightweight activity pre-segmentation.

Scans video at very low resolution and frame rate to identify temporal
segments where physical activity is happening.  Only these active segments
are later sent to YOLO for expensive per-frame detection.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ActivitySegment:
    start_sec: float
    end_sec: float
    peak_score: float
    avg_score: float
    trigger: str  # "motion" | "histogram" | "combined"
    stream_id: str = ""

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec

    def to_dict(self) -> dict:
        return {
            "start_sec": round(self.start_sec, 3),
            "end_sec": round(self.end_sec, 3),
            "duration_sec": round(self.duration_sec, 3),
            "peak_score": round(self.peak_score, 4),
            "avg_score": round(self.avg_score, 4),
            "trigger": self.trigger,
            "stream_id": self.stream_id,
        }


@dataclass
class PresegmentConfig:
    enabled: bool = True
    scan_fps: float = 2.0
    scan_resolution: tuple = (160, 120)
    motion_threshold_mode: str = "adaptive"  # "adaptive" | "fixed"
    motion_fixed_threshold: float = 0.02
    min_segment_sec: float = 3.0
    merge_gap_sec: float = 5.0
    padding_sec: float = 2.0
    skip_if_video_shorter_than: float = 30.0
    forced_sample_interval_sec: float = 60.0


class ActivityPreSegmenter:
    """Identifies active time segments in a video using lightweight motion analysis."""

    def __init__(self, config: Optional[PresegmentConfig] = None) -> None:
        self.config = config or PresegmentConfig()

    def segment(self, video_path: str | Path, stream_id: str = "") -> List[ActivitySegment]:
        video_path = Path(video_path)
        if not video_path.exists():
            logger.warning("Video not found for presegment: %s", video_path)
            return []

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning("Cannot open video for presegment: %s", video_path)
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = total_frames / fps if fps > 0 else 0.0

        if duration < self.config.skip_if_video_shorter_than:
            cap.release()
            logger.info(
                "Video %.1fs < %.1fs threshold, skipping presegment (full processing)",
                duration,
                self.config.skip_if_video_shorter_than,
            )
            return [ActivitySegment(
                start_sec=0.0,
                end_sec=duration,
                peak_score=1.0,
                avg_score=1.0,
                trigger="short_video_bypass",
                stream_id=stream_id,
            )]

        scores = self._compute_motion_scores(cap, fps, total_frames, duration)
        cap.release()

        if not scores:
            return [ActivitySegment(
                start_sec=0.0, end_sec=duration, peak_score=1.0,
                avg_score=1.0, trigger="no_scores_fallback", stream_id=stream_id,
            )]

        active_mask = self._threshold_scores(scores)
        active_mask = self._inject_forced_samples(active_mask, duration)
        segments = self._mask_to_segments(active_mask, duration, stream_id)

        total_active = sum(seg.duration_sec for seg in segments)
        logger.info(
            "Presegment: %.1fs video -> %d active segments (%.1fs total, %.0f%% reduction)",
            duration,
            len(segments),
            total_active,
            (1 - total_active / duration) * 100 if duration > 0 else 0,
        )
        return segments

    def _compute_motion_scores(
        self, cap: cv2.VideoCapture, fps: float, total_frames: int, duration: float
    ) -> List[tuple]:
        """Returns list of (timestamp_sec, motion_score) tuples.

        Uses sequential read with grab/skip pattern to avoid expensive seek.
        """
        scan_step = max(1, int(round(fps / self.config.scan_fps)))
        w, h = self.config.scan_resolution
        scores = []
        prev_gray = None
        prev_hist = None

        frame_idx = 0
        while frame_idx < total_frames:
            ok = cap.grab()
            if not ok:
                break
            if frame_idx % scan_step != 0:
                frame_idx += 1
                continue

            ok, frame = cap.retrieve()
            if not ok:
                frame_idx += 1
                continue

            ts = frame_idx / fps
            small = cv2.resize(frame, (w, h))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            motion_score = 0.0
            hist_score = 0.0

            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                motion_score = float(diff.mean() / 255.0)

                hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1], None, [18, 8], [0, 180, 0, 256])
                cv2.normalize(hist, hist)
                if prev_hist is not None:
                    hist_score = 1.0 - float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL))
                prev_hist = hist
            else:
                hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
                prev_hist = cv2.calcHist([hsv], [0, 1], None, [18, 8], [0, 180, 0, 256])
                cv2.normalize(prev_hist, prev_hist)

            prev_gray = gray
            combined = max(motion_score, hist_score * 0.7)
            scores.append((ts, combined))
            frame_idx += 1

        return scores

    def _threshold_scores(self, scores: List[tuple]) -> List[tuple]:
        """Returns list of (timestamp_sec, is_active) based on adaptive or fixed threshold."""
        values = np.array([s[1] for s in scores])

        if self.config.motion_threshold_mode == "fixed":
            threshold = self.config.motion_fixed_threshold
        else:
            mean_val = float(np.mean(values))
            std_val = float(np.std(values))
            threshold = max(self.config.motion_fixed_threshold * 0.5, mean_val + 1.0 * std_val)

        # Smoothing: sliding window of 5 to reduce noise
        kernel_size = 5
        if len(values) >= kernel_size:
            kernel = np.ones(kernel_size) / kernel_size
            smoothed = np.convolve(values, kernel, mode="same")
        else:
            smoothed = values

        result = []
        for i, (ts, _) in enumerate(scores):
            is_active = smoothed[i] >= threshold
            result.append((ts, is_active))

        return result

    def _inject_forced_samples(self, active_mask: List[tuple], duration: float) -> List[tuple]:
        """Ensure at least one active point every forced_sample_interval_sec."""
        if self.config.forced_sample_interval_sec <= 0:
            return active_mask

        interval = self.config.forced_sample_interval_sec
        forced_times = set()
        t = 0.0
        while t < duration:
            forced_times.add(t)
            t += interval

        result = list(active_mask)
        for forced_t in forced_times:
            closest_idx = min(range(len(result)), key=lambda i: abs(result[i][0] - forced_t))
            if not result[closest_idx][1]:
                ts = result[closest_idx][0]
                result[closest_idx] = (ts, True)

        return result

    def _mask_to_segments(
        self, active_mask: List[tuple], duration: float, stream_id: str
    ) -> List[ActivitySegment]:
        """Convert boolean time mask to merged ActivitySegment list."""
        if not active_mask:
            return []

        segments: List[ActivitySegment] = []
        in_segment = False
        seg_start = 0.0
        seg_scores: List[float] = []

        for ts, is_active in active_mask:
            if is_active and not in_segment:
                seg_start = ts
                seg_scores = [ts]
                in_segment = True
            elif is_active and in_segment:
                seg_scores.append(ts)
            elif not is_active and in_segment:
                segments.append(self._make_segment(seg_start, ts, seg_scores, stream_id))
                in_segment = False
                seg_scores = []

        if in_segment:
            segments.append(self._make_segment(seg_start, duration, seg_scores, stream_id))

        segments = self._merge_close_segments(segments, stream_id)
        segments = self._apply_padding_and_min_duration(segments, duration, stream_id)
        return segments

    def _make_segment(
        self, start: float, end: float, score_times: List[float], stream_id: str
    ) -> ActivitySegment:
        return ActivitySegment(
            start_sec=start,
            end_sec=end,
            peak_score=1.0,
            avg_score=1.0,
            trigger="combined",
            stream_id=stream_id,
        )

    def _merge_close_segments(
        self, segments: List[ActivitySegment], stream_id: str
    ) -> List[ActivitySegment]:
        if len(segments) <= 1:
            return segments

        merged: List[ActivitySegment] = [segments[0]]
        for seg in segments[1:]:
            gap = seg.start_sec - merged[-1].end_sec
            if gap <= self.config.merge_gap_sec:
                merged[-1] = ActivitySegment(
                    start_sec=merged[-1].start_sec,
                    end_sec=seg.end_sec,
                    peak_score=max(merged[-1].peak_score, seg.peak_score),
                    avg_score=(merged[-1].avg_score + seg.avg_score) / 2,
                    trigger="combined",
                    stream_id=stream_id,
                )
            else:
                merged.append(seg)
        return merged

    def _apply_padding_and_min_duration(
        self, segments: List[ActivitySegment], duration: float, stream_id: str
    ) -> List[ActivitySegment]:
        result: List[ActivitySegment] = []
        for seg in segments:
            start = max(0.0, seg.start_sec - self.config.padding_sec)
            end = min(duration, seg.end_sec + self.config.padding_sec)
            if (end - start) < self.config.min_segment_sec:
                mid = (seg.start_sec + seg.end_sec) / 2
                start = max(0.0, mid - self.config.min_segment_sec / 2)
                end = min(duration, mid + self.config.min_segment_sec / 2)
            result.append(ActivitySegment(
                start_sec=round(start, 3),
                end_sec=round(end, 3),
                peak_score=seg.peak_score,
                avg_score=seg.avg_score,
                trigger=seg.trigger,
                stream_id=stream_id,
            ))

        # Final merge pass after padding expansion
        if len(result) > 1:
            result = self._merge_close_segments(result, stream_id)

        return result

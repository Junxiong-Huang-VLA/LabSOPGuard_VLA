from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import cv2
import numpy as np


@dataclass
class SyncAnchor:
    camera_id: str
    local_time_sec: float
    reference_time_sec: float
    method: str = "manual"
    confidence: float = 1.0
    metadata: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["metadata"] = self.metadata or {}
        return data


@dataclass
class CameraSyncProfile:
    camera_id: str
    reference_camera_id: str = "global"
    offset_sec: float = 0.0
    clock_scale: float = 1.0
    drift_ppm: float = 0.0
    method: str = "manual"
    confidence: float = 1.0
    anchor_count: int = 0
    residual_error_sec: float = 0.0

    def local_to_global(self, timestamp_sec: float) -> float:
        return round(float(timestamp_sec) * self.clock_scale + self.offset_sec, 6)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VisualSyncCandidate:
    camera_id: str
    local_time_sec: float
    method: str
    confidence: float = 1.0
    signal_value: float = 0.0
    signature: str | None = None
    metadata: Dict[str, Any] | None = None


class TimeSyncCalibrator:
    """Fit per-camera timestamp transforms from manual, hardware, sync-board, or audio/flash anchors."""

    METHOD_WEIGHTS = {
        "hardware_timecode": 1.0,
        "ptp": 0.98,
        "ntp": 0.9,
        "sync_board": 0.95,
        "audio_flash": 0.8,
        "flash": 0.75,
        "visual_peak": 0.72,
        "event_time": 0.7,
        "auto_visual": 0.72,
        "multimodal_semantic": 0.78,
        "manual": 0.6,
    }

    def __init__(self, reference_camera_id: str = "global") -> None:
        self.reference_camera_id = reference_camera_id
        self._anchors: List[SyncAnchor] = []

    def add_anchor(self, anchor: SyncAnchor) -> None:
        self._anchors.append(anchor)

    def extend(self, anchors: Iterable[SyncAnchor]) -> None:
        for anchor in anchors:
            self.add_anchor(anchor)

    def fit(self, camera_id: str) -> CameraSyncProfile:
        anchors = [anchor for anchor in self._anchors if anchor.camera_id == camera_id]
        return self.fit_profile_from_anchors(
            camera_id=camera_id,
            anchors=anchors,
            reference_camera_id=self.reference_camera_id,
        )

    @classmethod
    def fit_profile_from_anchors(
        cls,
        camera_id: str,
        anchors: Sequence[SyncAnchor],
        reference_camera_id: str = "global",
    ) -> CameraSyncProfile:
        if not anchors:
            return CameraSyncProfile(camera_id=camera_id, reference_camera_id=reference_camera_id)

        if len(anchors) == 1:
            anchor = anchors[0]
            return CameraSyncProfile(
                camera_id=camera_id,
                reference_camera_id=reference_camera_id,
                offset_sec=round(anchor.reference_time_sec - anchor.local_time_sec, 6),
                method=anchor.method,
                confidence=float(anchor.confidence),
                anchor_count=1,
            )

        x = np.array([float(anchor.local_time_sec) for anchor in anchors], dtype=np.float64)
        y = np.array([float(anchor.reference_time_sec) for anchor in anchors], dtype=np.float64)
        weights = np.array(
            [
                max(1e-6, float(anchor.confidence) * cls.METHOD_WEIGHTS.get(anchor.method, 0.5))
                for anchor in anchors
            ],
            dtype=np.float64,
        )
        design = np.vstack([x, np.ones_like(x)]).T
        weighted_design = design * weights[:, None]
        weighted_y = y * weights
        scale, offset = np.linalg.lstsq(weighted_design, weighted_y, rcond=None)[0]
        predicted = x * scale + offset
        residual = float(np.sqrt(np.mean((predicted - y) ** 2)))
        confidence = max(0.0, min(1.0, float(np.mean(weights)) * (1.0 / (1.0 + residual))))
        method = "calibrated:" + "+".join(sorted({anchor.method for anchor in anchors}))
        return CameraSyncProfile(
            camera_id=camera_id,
            reference_camera_id=reference_camera_id,
            offset_sec=round(float(offset), 6),
            clock_scale=round(float(scale), 9),
            drift_ppm=round((float(scale) - 1.0) * 1_000_000.0, 3),
            method=method,
            confidence=round(confidence, 4),
            anchor_count=len(anchors),
            residual_error_sec=round(residual, 6),
        )

    @staticmethod
    def anchors_from_descriptor(camera_id: str, descriptor: Dict[str, Any]) -> List[SyncAnchor]:
        anchors: List[SyncAnchor] = []
        for item in descriptor.get("sync_anchors", []) or []:
            if not isinstance(item, dict):
                continue
            try:
                anchors.append(
                    SyncAnchor(
                        camera_id=str(item.get("camera_id") or camera_id),
                        local_time_sec=float(item["local_time_sec"]),
                        reference_time_sec=float(item["reference_time_sec"]),
                        method=str(item.get("method", descriptor.get("sync_method", "manual"))),
                        confidence=float(item.get("confidence", 1.0)),
                        metadata={k: v for k, v in item.items() if k not in {"camera_id", "local_time_sec", "reference_time_sec", "method", "confidence"}},
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        return anchors

    @staticmethod
    def _as_float(value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _timestamp_from_item(cls, item: Any) -> Optional[float]:
        if isinstance(item, dict):
            for key in ("timestamp_sec", "local_time_sec", "time_sec", "t"):
                value = cls._as_float(item.get(key))
                if value is not None:
                    return value
            return None
        if isinstance(item, (tuple, list)) and item:
            return cls._as_float(item[0])
        return None

    @classmethod
    def _brightness_from_item(cls, item: Any) -> Optional[float]:
        if isinstance(item, dict):
            for key in (
                "brightness_delta",
                "brightness_change",
                "luma_delta",
                "mean_brightness_delta",
                "brightness",
                "mean_brightness",
                "luma",
                "mean_luma",
            ):
                value = cls._as_float(item.get(key))
                if value is not None:
                    return value
            return None
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            return cls._as_float(item[1])
        return None

    @classmethod
    def _event_signature(cls, item: Dict[str, Any]) -> str:
        for key in ("event_id", "label", "name", "type", "event_type"):
            value = item.get(key)
            if value not in (None, ""):
                return str(value)
        return "event"

    @classmethod
    def visual_candidates_from_stream(
        cls,
        camera_id: str,
        stream: Dict[str, Any],
        *,
        z_threshold: float = 2.5,
        min_separation_sec: float = 0.2,
        max_candidates: int = 20,
    ) -> List[VisualSyncCandidate]:
        """Extract deterministic visual/event sync candidates from precomputed stream summaries."""
        candidates: List[VisualSyncCandidate] = []
        samples = stream.get("frame_summaries") or stream.get("brightness_samples") or []
        parsed_samples: List[tuple[float, float, Any]] = []
        for item in samples:
            timestamp = cls._timestamp_from_item(item)
            brightness = cls._brightness_from_item(item)
            if timestamp is None or brightness is None:
                continue
            parsed_samples.append((timestamp, brightness, item))
        parsed_samples.sort(key=lambda row: row[0])

        if parsed_samples:
            explicit_delta = any(
                isinstance(item, dict)
                and any(
                    key in item
                    for key in (
                        "brightness_delta",
                        "brightness_change",
                        "luma_delta",
                        "mean_brightness_delta",
                    )
                )
                for _, _, item in parsed_samples
            )
            values = [row[1] for row in parsed_samples]
            if explicit_delta:
                signal = values
            else:
                signal = [0.0]
                signal.extend(max(0.0, values[idx] - values[idx - 1]) for idx in range(1, len(values)))
            signal_arr = np.array(signal, dtype=np.float64)
            mean = float(np.mean(signal_arr))
            std = float(np.std(signal_arr)) or 1.0
            peak_rows: List[VisualSyncCandidate] = []
            for idx, (timestamp, brightness, item) in enumerate(parsed_samples):
                score = float(signal_arr[idx])
                z_score = (score - mean) / std
                prev_score = signal_arr[idx - 1] if idx > 0 else -np.inf
                next_score = signal_arr[idx + 1] if idx + 1 < len(signal_arr) else -np.inf
                if z_score < z_threshold or score < prev_score or score < next_score:
                    continue
                peak_rows.append(
                    VisualSyncCandidate(
                        camera_id=camera_id,
                        local_time_sec=float(timestamp),
                        method="visual_peak",
                        confidence=round(min(1.0, z_score / max(z_threshold, 1e-6)), 4),
                        signal_value=round(float(score), 6),
                        metadata={
                            "brightness": float(brightness),
                            "brightness_signal": round(float(score), 6),
                            "z_score": round(float(z_score), 6),
                        },
                    )
                )
            peak_rows.sort(key=lambda row: (-row.confidence, row.local_time_sec))
            selected: List[VisualSyncCandidate] = []
            for row in peak_rows:
                if any(abs(row.local_time_sec - existing.local_time_sec) < min_separation_sec for existing in selected):
                    continue
                selected.append(row)
                if len(selected) >= max_candidates:
                    break
            candidates.extend(sorted(selected, key=lambda row: row.local_time_sec))

        for item in stream.get("events", []) or []:
            if not isinstance(item, dict):
                continue
            timestamp = cls._timestamp_from_item(item)
            if timestamp is None:
                continue
            event_type = str(item.get("type") or item.get("event_type") or item.get("label") or "event")
            if event_type.lower() not in {
                "flash",
                "sync_flash",
                "visual_peak",
                "peak",
                "sync_event",
                "event",
            } and not item.get("sync_anchor"):
                continue
            candidates.append(
                VisualSyncCandidate(
                    camera_id=camera_id,
                    local_time_sec=float(timestamp),
                    method="event_time",
                    confidence=float(item.get("confidence", 0.85)),
                    signature=cls._event_signature(item),
                    metadata={k: v for k, v in item.items() if k not in {"timestamp_sec", "local_time_sec", "time_sec", "t"}},
                )
            )

        candidates.sort(key=lambda row: (row.local_time_sec, row.method, row.signature or ""))
        return candidates

    @classmethod
    def generate_visual_sync_anchors(
        cls,
        streams: Dict[str, Dict[str, Any]] | Sequence[Dict[str, Any]],
        *,
        reference_camera_id: str,
        z_threshold: float = 2.5,
        min_separation_sec: float = 0.2,
        max_match_time_delta_sec: Optional[float] = None,
    ) -> List[SyncAnchor]:
        """Generate sync anchors by matching common visual peaks/events across streams."""
        if isinstance(streams, dict):
            stream_items = [(str(camera_id), stream) for camera_id, stream in streams.items()]
        else:
            stream_items = [
                (str(stream.get("camera_id") or f"camera_{idx:02d}"), stream)
                for idx, stream in enumerate(streams)
                if isinstance(stream, dict)
            ]
        candidate_map: Dict[str, List[VisualSyncCandidate]] = {
            camera_id: cls.visual_candidates_from_stream(
                camera_id,
                stream,
                z_threshold=z_threshold,
                min_separation_sec=min_separation_sec,
            )
            for camera_id, stream in stream_items
        }
        reference_candidates = candidate_map.get(reference_camera_id, [])
        if not reference_candidates:
            return []

        anchors: List[SyncAnchor] = []
        for camera_id, candidates in candidate_map.items():
            if camera_id == reference_camera_id:
                continue
            matched: List[tuple[VisualSyncCandidate, VisualSyncCandidate]] = []
            used_reference_ids: set[int] = set()

            reference_by_signature: Dict[str, List[VisualSyncCandidate]] = {}
            for ref in reference_candidates:
                if ref.signature:
                    reference_by_signature.setdefault(ref.signature, []).append(ref)
            for candidate in candidates:
                if not candidate.signature or candidate.signature not in reference_by_signature:
                    continue
                options = reference_by_signature[candidate.signature]
                ref = options[min(len([m for _, m in matched if m.signature == candidate.signature]), len(options) - 1)]
                matched.append((candidate, ref))
                used_reference_ids.add(id(ref))

            unmatched_candidates = [candidate for candidate in candidates if not candidate.signature]
            unmatched_references = [ref for ref in reference_candidates if not ref.signature and id(ref) not in used_reference_ids]
            for candidate, ref in zip(unmatched_candidates, unmatched_references):
                matched.append((candidate, ref))

            for candidate, ref in sorted(matched, key=lambda pair: (pair[1].local_time_sec, pair[0].local_time_sec)):
                if (
                    max_match_time_delta_sec is not None
                    and abs(candidate.local_time_sec - ref.local_time_sec) > max_match_time_delta_sec
                ):
                    continue
                method = "auto_visual:" + "+".join(sorted({candidate.method, ref.method}))
                anchors.append(
                    SyncAnchor(
                        camera_id=camera_id,
                        local_time_sec=round(candidate.local_time_sec, 6),
                        reference_time_sec=round(ref.local_time_sec, 6),
                        method=method,
                        confidence=round(min(candidate.confidence, ref.confidence), 4),
                        metadata={
                            "reference_camera_id": reference_camera_id,
                            "candidate_method": candidate.method,
                            "reference_method": ref.method,
                            "signature": candidate.signature or ref.signature,
                            "candidate_signal_value": candidate.signal_value,
                            "reference_signal_value": ref.signal_value,
                        },
                    )
                )
        anchors.sort(key=lambda row: (row.camera_id, row.reference_time_sec, row.local_time_sec))
        return anchors

    @staticmethod
    def detect_flash_anchors(
        frames: Sequence[tuple[float, Any]],
        camera_id: str,
        reference_time_sec: Optional[float] = None,
        z_threshold: float = 3.0,
    ) -> List[SyncAnchor]:
        if not frames:
            return []
        brightness: List[float] = []
        for _, frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if getattr(frame, "ndim", 0) == 3 else frame
            brightness.append(float(np.mean(gray)))
        mean = float(np.mean(brightness))
        std = float(np.std(brightness)) or 1.0
        anchors: List[SyncAnchor] = []
        for (timestamp, _), value in zip(frames, brightness):
            z_score = (value - mean) / std
            if z_score >= z_threshold:
                anchors.append(
                    SyncAnchor(
                        camera_id=camera_id,
                        local_time_sec=float(timestamp),
                        reference_time_sec=float(reference_time_sec if reference_time_sec is not None else timestamp),
                        method="audio_flash",
                        confidence=min(1.0, z_score / max(z_threshold, 1e-6)),
                        metadata={"brightness": value, "z_score": z_score},
                    )
                )
        return anchors

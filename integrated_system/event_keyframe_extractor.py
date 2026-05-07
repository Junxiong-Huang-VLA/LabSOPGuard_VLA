"""
Event-Driven Keyframe Extractor
===============================
取代传统的均匀采样/帧差法，采用事件驱动式帧捕获策略：
- 违规触发捕获：检测到违规时保存当前帧及前后各2秒上下文
- 步骤转换捕获：SOP步骤切换时捕获代表帧
- 定时基线捕获：每5分钟保存一张状态帧
- 清晰度评分：拉普拉斯方差法选择最佳帧
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class KeyframeEvent:
    """单个关键帧的元数据"""
    frame_id: int
    timestamp_sec: float
    event_type: str  # "violation", "step_transition", "baseline"
    sharpness_score: float
    event_details: Dict[str, Any] = field(default_factory=dict)
    detection_results: Dict[str, Any] = field(default_factory=dict)
    file_path: str = ""
    is_violation: bool = False
    priority: int = 0  # 0=normal, 1=important, 2=critical

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EventDrivenKeyframeExtractor:
    """事件驱动式关键帧提取器"""

    def __init__(
        self,
        output_dir: Path,
        fps: float = 30.0,
        baseline_interval_sec: float = 300.0,  # 5分钟
        context_seconds: float = 2.0,  # 违规前后上下文秒数
        max_keyframes: int = 50,
        jpeg_quality: int = 90,
        max_dimension: int = 800,  # 常规帧降采样最大尺寸
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.baseline_interval_sec = baseline_interval_sec
        self.context_frames = int(fps * context_seconds)
        self.max_keyframes = max_keyframes
        self.jpeg_quality = jpeg_quality
        self.max_dimension = max_dimension

        # 滚动缓冲区：存储最近 N 帧用于违规回溯
        self._buffer: deque[Tuple[int, np.ndarray, float]] = deque(maxlen=self.context_frames * 2)

        # 状态追踪
        self._frame_counter: int = 0
        self._last_baseline_ts: float = 0.0
        self._last_step_id: str = ""
        self._keyframes: List[KeyframeEvent] = []
        self._metadata_path: Path = self.output_dir / "keyframes_metadata.json"

    @property
    def keyframes(self) -> List[KeyframeEvent]:
        return list(self._keyframes)

    @property
    def count(self) -> int:
        return len(self._keyframes)

    def feed_frame(self, frame: np.ndarray, timestamp_sec: float) -> int:
        """向缓冲区送入一帧，返回帧ID"""
        fid = self._frame_counter
        self._frame_counter += 1
        self._buffer.append((fid, frame.copy(), timestamp_sec))

        # 检查是否需要定时基线捕获
        if timestamp_sec - self._last_baseline_ts >= self.baseline_interval_sec:
            self._last_baseline_ts = timestamp_sec
            if frame is not None:
                self._capture_frame(frame, timestamp_sec, "baseline",
                                    {"description": "定时基线捕获"})

        return fid

    def on_violation(
        self,
        frame: np.ndarray,
        timestamp_sec: float,
        violation_type: str,
        severity: str = "medium",
        details: Optional[Dict[str, Any]] = None,
    ) -> List[KeyframeEvent]:
        """违规触发：保存当前帧及缓冲区中的上下文帧"""
        captured = []

        # 1. 先保存缓冲区中的上下文帧（回溯帧）
        for fid, buf_frame, buf_ts in list(self._buffer):
            if buf_ts < timestamp_sec:  # 只保存时间戳早于当前的
                evt = self._capture_frame(
                    buf_frame, buf_ts, "violation_context",
                    {
                        "violation_type": violation_type,
                        "severity": severity,
                        "context_of": f"frame_{self._frame_counter}",
                        "direction": "before",
                        **(details or {}),
                    },
                    priority=1,
                )
                if evt:
                    captured.append(evt)

        # 2. 保存当前违规帧（最高优先级）
        evt = self._capture_frame(
            frame, timestamp_sec, "violation",
            {
                "violation_type": violation_type,
                "severity": severity,
                **(details or {}),
            },
            priority=2,
            is_violation=True,
        )
        if evt:
            captured.append(evt)

        return captured

    def on_step_transition(
        self,
        frame: np.ndarray,
        timestamp_sec: float,
        step_id: str,
        step_name: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> Optional[KeyframeEvent]:
        """步骤转换触发：捕获新步骤的代表帧"""
        if step_id == self._last_step_id:
            return None  # 同一步骤不重复捕获

        self._last_step_id = step_id
        return self._capture_frame(
            frame, timestamp_sec, "step_transition",
            {
                "step_id": step_id,
                "step_name": step_name,
                **(details or {}),
            },
            priority=1,
        )

    def _capture_frame(
        self,
        frame: np.ndarray,
        timestamp_sec: float,
        event_type: str,
        event_details: Optional[Dict[str, Any]] = None,
        priority: int = 0,
        is_violation: bool = False,
    ) -> Optional[KeyframeEvent]:
        """捕获并保存单帧"""
        if frame is None or frame.size == 0:
            return None
        if len(self._keyframes) >= self.max_keyframes:
            return None

        # 计算清晰度评分
        sharpness = self._compute_sharpness(frame)

        # 生成文件名
        ts_str = f"{timestamp_sec:.1f}".replace(".", "p")
        fname = f"keyframe_{self._frame_counter:04d}_{event_type}_{ts_str}.jpg"
        fpath = self.output_dir / fname

        # 存储策略：违规帧全分辨率，常规帧降采样
        save_frame = frame.copy()
        if not is_violation and not event_type == "violation_context":
            save_frame = self._downsample(save_frame)

        cv2.imwrite(str(fpath), save_frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])

        evt = KeyframeEvent(
            frame_id=self._frame_counter,
            timestamp_sec=timestamp_sec,
            event_type=event_type,
            sharpness_score=sharpness,
            event_details=event_details or {},
            file_path=str(fpath),
            is_violation=is_violation,
            priority=priority,
        )
        self._keyframes.append(evt)
        self._save_metadata()
        return evt

    def select_best_frame(
        self,
        candidates: Optional[List[KeyframeEvent]] = None,
    ) -> Optional[KeyframeEvent]:
        """从候选帧中选择最佳帧（综合评分）"""
        pool = candidates or self._keyframes
        if not pool:
            return None

        def score(evt: KeyframeEvent) -> float:
            sharp_norm = min(evt.sharpness_score / 500.0, 1.0)  # 归一化
            conf = 0.0
            if evt.detection_results:
                confs = [d.get("confidence", 0) for d in evt.detection_results.get("detections", [])]
                conf = max(confs) if confs else 0.0
            person_vis = 1.0 if evt.detection_results.get("person_detected", False) else 0.5
            return sharp_norm * 0.4 + conf * 0.3 + person_vis * 0.3

        return max(pool, key=score)

    def get_violation_frames(self) -> List[KeyframeEvent]:
        """获取所有违规相关帧"""
        return [k for k in self._keyframes if k.is_violation or k.event_type == "violation_context"]

    def get_step_frames(self) -> List[KeyframeEvent]:
        """获取所有步骤转换帧"""
        return [k for k in self._keyframes if k.event_type == "step_transition"]

    def get_baseline_frames(self) -> List[KeyframeEvent]:
        """获取所有基线帧"""
        return [k for k in self._keyframes if k.event_type == "baseline"]

    def export_metadata(self) -> Dict[str, Any]:
        """导出完整元数据"""
        return {
            "total_keyframes": len(self._keyframes),
            "fps": self.fps,
            "violation_frames": len(self.get_violation_frames()),
            "step_frames": len(self.get_step_frames()),
            "baseline_frames": len(self.get_baseline_frames()),
            "keyframes": [k.to_dict() for k in self._keyframes],
        }

    @staticmethod
    def _compute_sharpness(frame: np.ndarray) -> float:
        """拉普拉斯方差法计算帧清晰度"""
        if frame is None or frame.size == 0:
            return 0.0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())

    def _downsample(self, frame: np.ndarray) -> np.ndarray:
        """降采样帧至最大尺寸"""
        h, w = frame.shape[:2]
        if max(h, w) <= self.max_dimension:
            return frame
        scale = self.max_dimension / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _save_metadata(self):
        """保存元数据到JSON"""
        try:
            data = self.export_metadata()
            self._metadata_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass  # 元数据保存失败不影响主流程

    def finalize(self) -> Path:
        """完成提取，保存最终元数据，返回元数据文件路径"""
        self._save_metadata()
        return self._metadata_path

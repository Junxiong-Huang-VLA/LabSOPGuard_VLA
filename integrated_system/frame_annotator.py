"""
Frame Annotator
===============
使用 OpenCV 原生绘制功能对检测帧进行标注：
- 边界框：红色=违规, 绿色=合规, 黄色=警告
- 标签：高对比度背景色块，确保PDF打印可读
- 骨架连线：YOLO pose 17关键点
- 违规高亮：半透明红色覆盖层
- 每帧最多标注5个最重要目标
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class Detection:
    """单个检测目标"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    label: str
    confidence: float
    category: str = "normal"  # "normal", "violation", "warning", "ppe"
    keypoints: Optional[List[Tuple[int, int, float]]] = None  # (x, y, confidence)


@dataclass
class AnnotatedFrame:
    """标注后的帧"""
    frame: np.ndarray
    detection_count: int
    violation_count: int
    annotation_details: List[Dict[str, Any]] = field(default_factory=list)


class FrameAnnotator:
    """帧标注器 — 使用cv2原生绘制"""

    # 颜色定义 (BGR)
    COLOR_VIOLATION = (0, 0, 255)      # 红色
    COLOR_COMPLIANT = (0, 200, 0)      # 绿色
    COLOR_WARNING = (0, 180, 255)      # 黄色/橙色
    COLOR_PPE = (255, 180, 0)          # 蓝色
    COLOR_TEXT_BG = (40, 40, 40)       # 深灰背景
    COLOR_TEXT_FG = (255, 255, 255)    # 白色文字
    COLOR_SKELETON = (0, 255, 255)     # 青色骨架
    COLOR_KEYPOINT = (255, 0, 255)     # 紫色关键点

    # Pose骨架连接定义（COCO 17关键点）
    SKELETON_CONNECTIONS = [
        (0, 1), (0, 2), (1, 3), (2, 4),      # 头部
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 上肢
        (5, 11), (6, 12), (11, 12),           # 躯干
        (11, 13), (13, 15), (12, 14), (14, 16)  # 下肢
    ]

    def __init__(
        self,
        thickness: int = 2,
        font_scale: float = 0.5,
        max_detections: int = 5,
        show_confidence: bool = True,
        show_skeleton: bool = True,
    ):
        self.thickness = thickness
        self.font_scale = font_scale
        self.max_detections = max_detections
        self.show_confidence = show_confidence
        self.show_skeleton = show_skeleton

    def annotate(
        self,
        frame: np.ndarray,
        detections: List[Detection],
    ) -> AnnotatedFrame:
        """对帧进行完整标注"""
        if frame is None or frame.size == 0:
            return AnnotatedFrame(frame=frame, detection_count=0, violation_count=0)

        canvas = frame.copy()
        details = []
        violation_count = 0

        # 按优先级排序：violation > warning > ppe > normal
        priority = {"violation": 0, "warning": 1, "ppe": 2, "normal": 3}
        sorted_dets = sorted(detections, key=lambda d: (priority.get(d.category, 3), -d.confidence))
        top_dets = sorted_dets[:self.max_detections]

        for det in top_dets:
            color = self._get_color(det.category)
            x1, y1, x2, y2 = det.bbox

            # 绘制边界框
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, self.thickness)

            # 绘制标签背景
            label_text = self._build_label(det)
            (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX,
                                                  self.font_scale, 1)
            label_y = max(y1 - 8, th + 4)
            cv2.rectangle(canvas, (x1, label_y - th - 4), (x1 + tw + 4, label_y + 2),
                          self.COLOR_TEXT_BG, -1)
            cv2.putText(canvas, label_text, (x1 + 2, label_y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.COLOR_TEXT_FG, 1)

            # 违规高亮半透明覆盖
            if det.category == "violation":
                overlay = canvas.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.12, canvas, 0.88, 0, canvas)
                violation_count += 1

            # 骨架连线
            if self.show_skeleton and det.keypoints and len(det.keypoints) >= 17:
                self._draw_skeleton(canvas, det.keypoints)

            details.append({
                "label": det.label,
                "confidence": det.confidence,
                "category": det.category,
                "bbox": det.bbox,
            })

        return AnnotatedFrame(
            frame=canvas,
            detection_count=len(top_dets),
            violation_count=violation_count,
            annotation_details=details,
        )

    def annotate_violation_card(
        self,
        frame: np.ndarray,
        detection: Detection,
        violation_text: str = "",
    ) -> np.ndarray:
        """为PDF违规卡片生成带醒目标注的帧"""
        canvas = frame.copy()
        x1, y1, x2, y2 = detection.bbox

        # 粗红色边界框
        cv2.rectangle(canvas, (x1, y1), (x2, y2), self.COLOR_VIOLATION, 3)

        # 半透明红色覆盖
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.2, canvas, 0.8, 0, canvas)

        # 违规标签（大字体）
        label = f"VIOLATION: {detection.label}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        label_y = max(y1 - 12, th + 8)
        cv2.rectangle(canvas, (x1, label_y - th - 6), (x1 + tw + 8, label_y + 4),
                      (0, 0, 180), -1)
        cv2.putText(canvas, label, (x1 + 4, label_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 违规描述文字
        if violation_text:
            cv2.putText(canvas, violation_text[:60], (x1, y2 + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 骨架
        if self.show_skeleton and detection.keypoints and len(detection.keypoints) >= 17:
            self._draw_skeleton(canvas, detection.keypoints)

        return canvas

    def _get_color(self, category: str) -> Tuple[int, int, int]:
        color_map = {
            "violation": self.COLOR_VIOLATION,
            "warning": self.COLOR_WARNING,
            "ppe": self.COLOR_PPE,
            "normal": self.COLOR_COMPLIANT,
        }
        return color_map.get(category, self.COLOR_COMPLIANT)

    def _build_label(self, det: Detection) -> str:
        if self.show_confidence:
            return f"{det.label} {det.confidence:.0%}"
        return det.label

    def _draw_skeleton(
        self,
        canvas: np.ndarray,
        keypoints: List[Tuple[int, int, float]],
    ):
        """绘制骨架连线"""
        for (i, j) in self.SKELETON_CONNECTIONS:
            if i < len(keypoints) and j < len(keypoints):
                kp1, kp2 = keypoints[i], keypoints[j]
                if kp1[2] > 0.3 and kp2[2] > 0.3:  # confidence threshold
                    cv2.line(canvas, (kp1[0], kp1[1]), (kp2[0], kp2[1]),
                             self.COLOR_SKELETON, self.thickness)

        # 绘制关键点
        for (x, y, conf) in keypoints:
            if conf > 0.3:
                cv2.circle(canvas, (x, y), 3, self.COLOR_KEYPOINT, -1)

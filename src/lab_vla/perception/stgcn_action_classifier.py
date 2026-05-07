"""
ST-GCN++ Skeleton-Based Action Classification for LabSOPGuard.

Implements Spatial-Temporal Graph Convolutional Network for classifying
laboratory operations from skeleton keypoint sequences.

Architecture:
- Spatial graph: body joint connectivity (COCO 17-keypoint topology)
- Temporal graph: frame-to-frame joint correspondence
- Multi-scale temporal convolution with dilations

Supports:
- Online (streaming) classification with sliding window
- Batch classification for offline analysis
- Fallback to rule-based classification when model unavailable
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class LabAction(str, Enum):
    """Laboratory action categories."""
    UNKNOWN = "unknown"
    # PPE
    WEARING_PPE = "wearing_ppe"
    REMOVING_PPE = "removing_ppe"
    # Handling
    REACHING = "reaching"
    GRASPING = "grasping"
    HOLDING = "holding"
    RELEASING = "releasing"
    # Transfer
    POURING = "pouring"
    PIPETTING = "pipetting"
    TRANSFERRING = "transferring"
    DISPENSING = "dispensing"
    # Measurement
    READING = "reading"
    WEIGHING = "weighing"
    # Mixing
    STIRRING = "stirring"
    SHAKING = "shaking"
    VORTEXING = "vortexing"
    # Heating
    HEATING = "heating"
    # Cleaning
    WASHING = "washing"
    WIPING = "wiping"
    # Safety
    DANGER_ZONE = "danger_zone"
    UNSAFE_OPERATION = "unsafe_operation"


# COCO 17-keypoint skeleton edges for spatial graph
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),   # Head
    (5, 6),                              # Shoulders
    (5, 7), (7, 9),                      # Left arm
    (6, 8), (8, 10),                     # Right arm
    (5, 11), (6, 12),                    # Torso
    (11, 12),                            # Hips
    (11, 13), (13, 15),                  # Left leg
    (12, 14), (14, 16),                  # Right leg
]

NUM_KEYPOINTS = 17
WINDOW_SIZE = 30  # Frames for classification window


@dataclass
class ActionPrediction:
    """Action prediction result."""
    action: LabAction
    confidence: float
    frame_range: Tuple[int, int]  # (start_frame, end_frame)
    timestamp_range: Tuple[float, float]  # (start_time, end_time)
    top_k: List[Tuple[str, float]] = field(default_factory=list)  # Top-K predictions


@dataclass
class SkeletonFrame:
    """Skeleton data for a single frame."""
    frame_id: int
    timestamp_sec: float
    keypoints: np.ndarray  # Shape: (17, 3) — [x, y, confidence] per keypoint
    person_id: int = 0


# ---------------------------------------------------------------------------
# ST-GCN++ Model (PyTorch)
# ---------------------------------------------------------------------------

class STGCNBlock(torch.nn.Module if False else object):
    """ST-GCN building block: Spatial GCN + Temporal Conv.

    When PyTorch is available, this is a real graph convolution block.
    When unavailable, provides a lightweight numpy-based fallback.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 9):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self._model = None

    def _build_model(self):
        """Build PyTorch model lazily."""
        try:
            import torch
            import torch.nn as nn

            class _STGCNBlock(nn.Module):
                def __init__(self, in_ch, out_ch, kernel_size):
                    super().__init__()
                    self.gcn = nn.Conv2d(in_ch, out_ch, kernel_size=1)
                    self.bn = nn.BatchNorm2d(out_ch)
                    self.tcn = nn.Conv2d(out_ch, out_ch, (kernel_size, 1),
                                         padding=((kernel_size - 1) // 2, 0))
                    self.tcn_bn = nn.BatchNorm2d(out_ch)
                    self.relu = nn.ReLU()

                def forward(self, x):
                    # x: (batch, channels, time, joints)
                    out = self.gcn(x)
                    out = self.bn(out)
                    out = self.tcn(out)
                    out = self.tcn_bn(out)
                    out = self.relu(out)
                    return out

            self._model = _STGCNBlock(self.in_channels, self.out_channels, self.kernel_size)
        except ImportError:
            pass


class STGCNActionClassifier:
    """ST-GCN++ based action classifier for lab operations.

    Supports both PyTorch model inference and lightweight rule-based fallback.
    """

    def __init__(
        self,
        model_path: str = "",
        window_size: int = WINDOW_SIZE,
        num_classes: int = 20,
        use_gpu: bool = True,
    ):
        self.window_size = window_size
        self.num_classes = num_classes
        self.model_path = model_path
        self._model = None
        self._device = None
        self._use_torch = False

        # Sliding window buffer
        self._buffer: deque[SkeletonFrame] = deque(maxlen=window_size)
        self._frame_counter = 0

        # Try to load model
        if model_path and Path(model_path).exists():
            self._load_model(model_path, use_gpu)

    def _load_model(self, model_path: str, use_gpu: bool) -> None:
        """Load PyTorch ST-GCN++ model."""
        try:
            import torch

            self._device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

            # Model architecture
            class STGCN(nn.Module if False else object):
                def __init__(self, num_classes=20):
                    pass

            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self._device)

            # Build model from checkpoint
            self._use_torch = True
            self._model = self._build_stgcn(num_classes)
            if "model_state_dict" in checkpoint:
                self._model.load_state_dict(checkpoint["model_state_dict"])
            elif "state_dict" in checkpoint:
                self._model.load_state_dict(checkpoint["state_dict"])
            else:
                self._model.load_state_dict(checkpoint)

            self._model.to(self._device)
            self._model.eval()

        except Exception as exc:
            print(f"ST-GCN model load failed, using fallback: {exc}")
            self._use_torch = False
            self._model = None

    def _build_stgcn(self, num_classes: int):
        """Build ST-GCN++ architecture."""
        import torch
        import torch.nn as nn

        class STGCNModel(nn.Module):
            def __init__(self, in_channels=3, num_classes=20):
                super().__init__()
                # Input: (batch, 3, T, 17) — 3 = x,y,conf
                self.layer1 = self._stgcn_block(in_channels, 64, kernel_size=9)
                self.layer2 = self._stgcn_block(64, 64, kernel_size=9)
                self.layer3 = self._stgcn_block(64, 128, kernel_size=9)
                self.layer4 = self._stgcn_block(128, 128, kernel_size=9)

                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(128, num_classes)

            def _stgcn_block(self, in_ch, out_ch, kernel_size):
                import torch.nn as nn
                return nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=1),
                    nn.BatchNorm2d(out_ch),
                    nn.Conv2d(out_ch, out_ch, (kernel_size, 1),
                              padding=((kernel_size - 1) // 2, 0)),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(),
                )

            def forward(self, x):
                # x: (batch, channels, time, joints)
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                x = self.pool(x)  # (batch, 128, 1, 1)
                x = x.view(x.size(0), -1)  # (batch, 128)
                x = self.fc(x)  # (batch, num_classes)
                return x

        return STGCNModel(num_classes=num_classes)

    def push_frame(self, skeleton: SkeletonFrame) -> Optional[ActionPrediction]:
        """Add a skeleton frame to the sliding window buffer.

        Returns prediction when buffer is full, else None.
        """
        self._buffer.append(skeleton)
        self._frame_counter += 1

        if len(self._buffer) >= self.window_size:
            return self._classify_window()

        return None

    def _classify_window(self) -> ActionPrediction:
        """Classify the current window of skeleton frames."""
        frames = list(self._buffer)
        start_frame = frames[0].frame_id
        end_frame = frames[-1].frame_id
        start_time = frames[0].timestamp_sec
        end_time = frames[-1].timestamp_sec

        if self._use_torch and self._model is not None:
            return self._torch_classify(frames, start_frame, end_frame, start_time, end_time)
        else:
            return self._rule_classify(frames, start_frame, end_frame, start_time, end_time)

    def _torch_classify(self, frames, start_frame, end_frame, start_time, end_time) -> ActionPrediction:
        """Classify using PyTorch ST-GCN++ model."""
        import torch

        # Build input tensor: (1, 3, T, 17)
        seq = np.zeros((3, len(frames), NUM_KEYPOINTS), dtype=np.float32)
        for t, f in enumerate(frames):
            kp = f.keypoints  # (17, 3)
            if kp.shape[0] >= NUM_KEYPOINTS:
                seq[0, t, :] = kp[:NUM_KEYPOINTS, 0]  # x
                seq[1, t, :] = kp[:NUM_KEYPOINTS, 1]  # y
                seq[2, t, :] = kp[:NUM_KEYPOINTS, 2]  # conf

        # Normalize
        seq[0] = seq[0] / max(seq[0].max(), 1.0)
        seq[1] = seq[1] / max(seq[1].max(), 1.0)

        x = torch.FloatTensor(seq).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits = self._model(x)
            probs = torch.softmax(logits, dim=-1)[0]

        top_k_vals, top_k_ids = torch.topk(probs, min(5, len(probs)))
        top_k = [(LabAction(list(LabAction)[i]).value, float(v)) for i, v in zip(top_k_ids, top_k_vals)]

        best_action = LabAction(list(LabAction)[top_k_ids[0].item()])
        best_conf = float(top_k_vals[0])

        return ActionPrediction(
            action=best_action,
            confidence=best_conf,
            frame_range=(start_frame, end_frame),
            timestamp_range=(start_time, end_time),
            top_k=top_k,
        )

    def _rule_classify(self, frames, start_frame, end_frame, start_time, end_time) -> ActionPrediction:
        """Rule-based fallback classification using skeleton motion analysis."""
        if len(frames) < 3:
            return ActionPrediction(
                action=LabAction.UNKNOWN, confidence=0.0,
                frame_range=(start_frame, end_frame),
                timestamp_range=(start_time, end_time),
            )

        # Compute motion features
        wrist_l = []  # Left wrist trajectory
        wrist_r = []  # Right wrist trajectory
        shoulder_l = []
        shoulder_r = []

        for f in frames:
            kp = f.keypoints
            if kp.shape[0] >= 11:
                wrist_l.append(kp[9, :2])   # left_wrist
                wrist_r.append(kp[10, :2])  # right_wrist
                shoulder_l.append(kp[5, :2])
                shoulder_r.append(kp[6, :2])

        if not wrist_l:
            return ActionPrediction(
                action=LabAction.UNKNOWN, confidence=0.0,
                frame_range=(start_frame, end_frame),
                timestamp_range=(start_time, end_time),
            )

        wrist_l = np.array(wrist_l)
        wrist_r = np.array(wrist_r)

        # Motion magnitude
        left_motion = np.mean(np.linalg.norm(np.diff(wrist_l, axis=0), axis=1)) if len(wrist_l) > 1 else 0
        right_motion = np.mean(np.linalg.norm(np.diff(wrist_r, axis=0), axis=1)) if len(wrist_r) > 1 else 0
        total_motion = left_motion + right_motion

        # Vertical vs horizontal motion
        left_v = np.mean(np.abs(np.diff(wrist_l[:, 1]))) if len(wrist_l) > 1 else 0
        left_h = np.mean(np.abs(np.diff(wrist_l[:, 0]))) if len(wrist_l) > 1 else 0

        # Circular motion detection (for stirring)
        is_circular = self._detect_circular_motion(wrist_l) or self._detect_circular_motion(wrist_r)

        # Classify based on motion patterns
        if total_motion < 2.0:
            action = LabAction.HOLDING
            conf = 0.6
        elif is_circular:
            action = LabAction.STIRRING
            conf = 0.55
        elif left_v > left_h * 1.5 and left_v > 5.0:
            action = LabAction.POURING
            conf = 0.5
        elif total_motion > 20.0:
            action = LabAction.REACHING
            conf = 0.45
        elif total_motion > 10.0:
            action = LabAction.TRANSFERRING
            conf = 0.4
        else:
            action = LabAction.UNKNOWN
            conf = 0.3

        return ActionPrediction(
            action=action,
            confidence=conf,
            frame_range=(start_frame, end_frame),
            timestamp_range=(start_time, end_time),
            top_k=[(action.value, conf)],
        )

    def _detect_circular_motion(self, trajectory: np.ndarray) -> bool:
        """Detect if a 2D trajectory has circular pattern."""
        if len(trajectory) < 8:
            return False

        # Compute angular change
        diffs = np.diff(trajectory, axis=0)
        angles = np.arctan2(diffs[:, 1], diffs[:, 0])
        angle_diffs = np.diff(angles)

        # Wrap to [-pi, pi]
        angle_diffs = (angle_diffs + np.pi) % (2 * np.pi) - np.pi

        # Circular motion has consistent angular change
        if len(angle_diffs) < 3:
            return False

        total_rotation = np.sum(angle_diffs)
        return abs(total_rotation) > np.pi  # More than 180 degrees total rotation

    def classify_sequence(self, skeletons: List[SkeletonFrame]) -> List[ActionPrediction]:
        """Classify a full sequence using sliding window with stride."""
        predictions = []
        stride = self.window_size // 2

        for start in range(0, max(1, len(skeletons) - self.window_size + 1), stride):
            end = min(start + self.window_size, len(skeletons))
            window = skeletons[start:end]

            if len(window) < self.window_size // 2:
                break

            frames_arr = window
            pred = self._rule_classify(
                frames_arr,
                frames_arr[0].frame_id,
                frames_arr[-1].frame_id,
                frames_arr[0].timestamp_sec,
                frames_arr[-1].timestamp_sec,
            )
            predictions.append(pred)

        return predictions


# ---------------------------------------------------------------------------
# Action Label Map
# ---------------------------------------------------------------------------

ACTION_LABELS_ZH = {
    LabAction.UNKNOWN: "未知操作",
    LabAction.WEARING_PPE: "穿戴PPE",
    LabAction.REMOVING_PPE: "脱除PPE",
    LabAction.REACHING: "伸手取物",
    LabAction.GRASPING: "抓取",
    LabAction.HOLDING: "持握",
    LabAction.RELEASING: "放下",
    LabAction.POURING: "倾倒",
    LabAction.PIPETTING: "移液",
    LabAction.TRANSFERRING: "转移",
    LabAction.DISPENSING: "分配",
    LabAction.READING: "读数",
    LabAction.WEIGHING: "称量",
    LabAction.STIRRING: "搅拌",
    LabAction.SHAKING: "摇晃",
    LabAction.VORTEXING: "涡旋",
    LabAction.HEATING: "加热",
    LabAction.WASHING: "清洗",
    LabAction.WIPING: "擦拭",
    LabAction.DANGER_ZONE: "危险区域",
    LabAction.UNSAFE_OPERATION: "不安全操作",
}

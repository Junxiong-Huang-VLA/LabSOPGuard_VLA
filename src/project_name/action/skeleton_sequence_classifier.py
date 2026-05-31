from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional

import numpy as np


@dataclass
class SkeletonActionResult:
    action: str
    confidence: float
    backend: str
    window_size: int


class SkeletonSequenceClassifier:
    """
    Layer-2 action classifier interface.
    - preferred backend: SkateFormer (placeholder interface)
    - fallback backend: lightweight keypoint motion heuristic (runnable)
    """

    def __init__(self, window_size: int = 30, backend: str = "skateformer") -> None:
        self.window_size = max(5, int(window_size))
        self.backend = backend.lower()
        self._history: Deque[np.ndarray] = deque(maxlen=self.window_size)

    def update(self, keypoints17: Optional[List[List[float]]]) -> SkeletonActionResult:
        if keypoints17 is None:
            return SkeletonActionResult(
                action="unknown",
                confidence=0.1,
                backend="no_pose",
                window_size=self.window_size,
            )

        arr = np.asarray(keypoints17, dtype=np.float32)
        if arr.shape != (17, 3):
            return SkeletonActionResult(
                action="unknown",
                confidence=0.1,
                backend="invalid_pose_shape",
                window_size=self.window_size,
            )

        self._history.append(arr)
        if len(self._history) < min(10, self.window_size):
            return SkeletonActionResult(
                action="warmup",
                confidence=0.2,
                backend=f"{self.backend}_fallback",
                window_size=self.window_size,
            )

        return self._classify_with_motion_heuristic()

    def _classify_with_motion_heuristic(self) -> SkeletonActionResult:
        seq = np.stack(list(self._history), axis=0)  # [T,17,3]
        # COCO indices: left_wrist=9, right_wrist=10, left_shoulder=5, right_shoulder=6
        lw = seq[:, 9, :2]
        rw = seq[:, 10, :2]
        ls = seq[:, 5, :2]
        rs = seq[:, 6, :2]

        wrist_speed = np.mean(np.linalg.norm(np.diff((lw + rw) / 2.0, axis=0), axis=-1))
        shoulder_center = (ls + rs) / 2.0
        hand_to_torso = np.mean(np.linalg.norm(((lw + rw) / 2.0) - shoulder_center, axis=-1))

        if wrist_speed > 6.0 and hand_to_torso > 35.0:
            action = "pipette_transfer"
            conf = 0.75
        elif wrist_speed > 2.5:
            action = "reaching_operation"
            conf = 0.62
        else:
            action = "verify_label"
            conf = 0.68

        return SkeletonActionResult(
            action=action,
            confidence=float(conf),
            backend=f"{self.backend}_fallback",
            window_size=self.window_size,
        )

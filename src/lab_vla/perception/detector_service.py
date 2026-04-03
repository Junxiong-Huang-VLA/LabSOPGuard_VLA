from __future__ import annotations

from typing import List, Optional

from project_name.detection.multi_level_detector import MultiLevelDetector
from project_name.video.capture import FramePacket

from lab_vla.core.contracts import DetectionObject, PerceptionPacket, SceneState


def _select_target(objects: List[DetectionObject], target_name: str) -> Optional[DetectionObject]:
    if not objects:
        return None
    for obj in objects:
        if obj.label.lower() == target_name.lower():
            return obj
    return max(objects, key=lambda x: x.score)


class DetectorService:
    def __init__(self, confidence_threshold: float, runtime_config_path: str) -> None:
        self._fallback_only = False
        try:
            self.detector = MultiLevelDetector(
                confidence_threshold=confidence_threshold,
                runtime_config_path=runtime_config_path,
            )
        except Exception:
            self.detector = None
            self._fallback_only = True

    def _fallback_packet(self, frame: FramePacket) -> PerceptionPacket:
        obj = DetectionObject(
            label="sample_container",
            bbox=[120, 100, 240, 260],
            score=0.7,
        )
        return PerceptionPacket(
            frame_id=frame.frame_id,
            timestamp_sec=frame.timestamp_sec,
            ppe={"wear_gloves": True, "wear_goggles": True, "wear_lab_coat": True},
            objects=[obj],
            actions=["verify_label"],
            confidence=0.7,
            layer_outputs={"backend": "fallback"},
        )

    def infer(self, frame: FramePacket) -> PerceptionPacket:
        if self._fallback_only or self.detector is None:
            return self._fallback_packet(frame)

        try:
            det = self.detector.detect(frame)
        except Exception:
            # Runtime fallback for device/model mismatch (for example CUDA requested on CPU-only host).
            self._fallback_only = True
            return self._fallback_packet(frame)

        objects = [
            DetectionObject(
                label=str(x.get("label", "")),
                bbox=[int(v) for v in x.get("bbox", [0, 0, 0, 0])],
                score=float(x.get("score", 0.0)),
            )
            for x in det.objects
        ]
        return PerceptionPacket(
            frame_id=det.frame_id,
            timestamp_sec=det.timestamp_sec,
            ppe=dict(det.ppe),
            objects=objects,
            actions=list(det.actions),
            confidence=float(det.confidence),
            layer_outputs=dict(det.layer_outputs),
        )

    def build_scene_state(
        self,
        sample_id: str,
        target_object: str,
        perception: PerceptionPacket,
    ) -> SceneState:
        target = _select_target(perception.objects, target_object)
        labels = [o.label for o in perception.objects]
        ppe_ok = (
            bool(perception.ppe.get("wear_gloves", False))
            and bool(perception.ppe.get("wear_goggles", False))
        )
        return SceneState(
            sample_id=sample_id,
            frame_id=perception.frame_id,
            timestamp_sec=perception.timestamp_sec,
            target_object=target_object,
            target_bbox=target.bbox if target else None,
            target_xyz_m=[0.22, 0.04, 0.12] if target else None,
            ppe_ok=ppe_ok,
            object_labels=labels,
            action_hints=list(perception.actions),
            confidence=perception.confidence,
            layer_outputs=dict(perception.layer_outputs),
        )

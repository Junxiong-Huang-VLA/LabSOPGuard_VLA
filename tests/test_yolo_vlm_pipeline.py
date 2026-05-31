from __future__ import annotations

from typing import Any

from key_action_indexer.yolo_vlm_pipeline import apply_yolo_vlm_review_pipeline


class _FakeQwenDescription:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.raw_response = payload
        self.model = "qwen-test"


class _FakeQwenClient:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.prompt = ""

    def describe_scene(self, path: str, *, prompt: str, temperature: float = 0.0) -> _FakeQwenDescription:
        self.prompt = prompt
        return _FakeQwenDescription(self.payload)


def test_qwen_correction_without_yolo_refs_stays_proposed(tmp_path) -> None:
    keyframe = tmp_path / "front_peak.jpg"
    keyframe.write_bytes(b"fake image placeholder")
    candidate_rows = [_candidate_row(keyframe)]
    client = _FakeQwenClient(
        {
            "description": "手正在接触移液器。",
            "physical_action": "pipette contact",
            "semantic_action": "aspirating liquid",
            "corrected_primary_object": "tube",
            "confirmed_objects": ["pipette", "gloved_hand"],
            "confidence": 0.82,
            "evidence_alignment": "aligned",
        }
    )

    apply_yolo_vlm_review_pipeline(
        tmp_path,
        candidate_rows,
        [_micro_row()],
        vlm_client=client,
        enable_vlm=True,
    )

    assert candidate_rows[0]["primary_object"] == "pipette"
    semantics = candidate_rows[0]["vlm_semantics"]
    assert semantics["primary_object_override_allowed"] is False
    assert "corrected_primary_object" not in semantics
    assert semantics["proposed_primary_object"] == "tube"
    assert semantics["proposed_semantic_action"] == "aspirating liquid"
    assert "tube" in semantics["uncertain_objects"]
    assert semantics["yolo_evidence_refs"] == []


def test_qwen_correction_with_matching_yolo_refs_is_accepted(tmp_path) -> None:
    keyframe = tmp_path / "front_peak.jpg"
    keyframe.write_bytes(b"fake image placeholder")
    candidate_rows = [_candidate_row(keyframe)]
    client = _FakeQwenClient(
        {
            "description": "手正在把管靠近移液器。",
            "physical_action": "pipette contact",
            "semantic_action": "tube positioning",
            "corrected_primary_object": "tube",
            "confirmed_objects": ["pipette", "gloved_hand"],
            "yolo_evidence_refs": [
                {
                    "frame_id": "frame-001",
                    "time_sec": 1.0,
                    "view": "front",
                    "label": "tube",
                    "confidence": 0.88,
                },
                "candidate-front-peak",
            ],
            "confidence": 0.86,
            "evidence_alignment": "aligned",
        }
    )

    apply_yolo_vlm_review_pipeline(
        tmp_path,
        candidate_rows,
        [_micro_row()],
        vlm_client=client,
        enable_vlm=True,
    )

    semantics = candidate_rows[0]["vlm_semantics"]
    assert semantics["primary_object_override_allowed"] is True
    assert semantics["corrected_primary_object"] == "tube"
    assert semantics["semantic_action"] == "tube positioning"
    assert semantics["yolo_evidence_refs"][0]["label"] == "tube"
    assert semantics["yolo_evidence_refs"][0]["frame_id"] == "frame-001"
    assert semantics["yolo_evidence_refs"][1]["candidate_id"] == "candidate-front-peak"


def _candidate_row(keyframe) -> dict[str, Any]:
    return {
        "candidate_id": "candidate-front-peak",
        "candidate_group_id": "candidate-group-1",
        "action_name": "hand pipette operation",
        "primary_object": "pipette",
        "view": "front",
        "micro_segment_id": "micro-1",
        "asset_kind": "关键帧",
        "stored_file": str(keyframe),
        "recommended": True,
        "frame_role": "peak",
        "time_range_sec": "1.000-2.000",
    }


def _micro_row() -> dict[str, Any]:
    return {
        "micro_segment_id": "micro-1",
        "primary_object": "pipette",
        "yolo_evidence": [
            _evidence_row("frame-001", 1.0),
            _evidence_row("frame-002", 1.2),
        ],
    }


def _evidence_row(frame_id: str, time_sec: float) -> dict[str, Any]:
    hand_bbox = [180, 90, 250, 170]
    pipette_bbox = [238, 112, 340, 126]
    return {
        "frame_id": frame_id,
        "local_time_sec": time_sec,
        "view": "front",
        "frame_width": 960,
        "frame_height": 540,
        "detections": [
            {"label": "gloved_hand", "confidence": 0.91, "bbox": hand_bbox},
            {"label": "pipette", "confidence": 0.93, "bbox": pipette_bbox},
            {"label": "tube", "confidence": 0.88, "bbox": [355, 105, 372, 185]},
        ],
        "hand_object_interactions": [
            {
                "hand_label": "gloved_hand",
                "object_label": "pipette",
                "score": 0.9,
                "hand_bbox": hand_bbox,
                "object_bbox": pipette_bbox,
            }
        ],
    }

from __future__ import annotations

from key_action_indexer import material_references


def test_pipeline_contract_keeps_annotation_target_on_manipulated_object() -> None:
    semantic_fields = {
        "manipulated_object": "paper",
        "instrument_context": "balance",
        "corrected_primary_object": "balance",
    }
    candidate = {
        "manipulated_object": "paper",
        "primary_object": "paper",
        "instrument_context": "balance",
        "secondary_objects": ["balance"],
        "corrected_primary_object": "balance",
    }

    assert material_references._annotation_target_query("paper", semantic_fields) == "paper"
    assert material_references._candidate_target_labels(candidate) == {"paper"}
    assert material_references._candidate_context_labels(candidate) == {"balance"}


def test_pipeline_contract_renders_only_active_hand_object_instance() -> None:
    evidence_row = {
        "view": "third_person",
        "detections": [
            {"label": "gloved_hand", "confidence": 0.93, "bbox": [250, 80, 330, 210]},
            {"label": "paper", "confidence": 0.91, "bbox": [300, 100, 440, 220]},
            {"label": "paper", "confidence": 0.85, "bbox": [20, 150, 150, 260]},
            {"label": "balance", "confidence": 0.96, "bbox": [190, 210, 500, 500]},
        ],
        "hand_object_interactions": [
            {
                "hand_label": "gloved_hand",
                "object_label": "paper",
                "score": 0.94,
                "hand_bbox": [250, 80, 330, 210],
                "object_bbox": [300, 100, 440, 220],
            }
        ],
    }
    tracklet_detections = [
        {"label": "gloved_hand", "confidence": 0.92, "bbox": [252, 82, 332, 212], "tracklet_id": "trk_hand"},
        {"label": "paper", "confidence": 0.88, "bbox": [302, 102, 442, 222], "tracklet_id": "trk_active_paper"},
        {"label": "paper", "confidence": 0.86, "bbox": [20, 150, 150, 260], "tracklet_id": "trk_inactive_paper"},
        {"label": "balance", "confidence": 0.97, "bbox": [190, 210, 500, 500], "tracklet_id": "trk_context_balance"},
    ]

    selected = material_references._tracklet_detections_for_active_interaction(
        tracklet_detections,
        evidence_row,
        "paper",
    )

    assert {item["tracklet_id"] for item in selected} == {"trk_hand", "trk_active_paper"}


def test_pipeline_contract_scales_source_yolo_boxes_to_render_frame() -> None:
    evidence_row = {
        "frame_width": 1920,
        "frame_height": 1080,
        "detections": [{"label": "paper", "confidence": 0.9, "bbox": [960, 540, 1440, 900]}],
        "hand_object_interactions": [
            {
                "hand_label": "gloved_hand",
                "object_label": "paper",
                "hand_bbox": [100, 200, 300, 400],
                "object_bbox": [960, 540, 1440, 900],
            }
        ],
    }

    scaled = material_references._scale_evidence_row_to_frame_size(evidence_row, width=960, height=540)

    assert scaled["detections"][0]["bbox"] == [480.0, 270.0, 720.0, 450.0]
    assert scaled["hand_object_interactions"][0]["object_bbox"] == [480.0, 270.0, 720.0, 450.0]
    assert scaled["bbox_source_frame"]["frame_width"] == 1920
    assert scaled["bbox_source_frame"]["render_width"] == 960


def test_pipeline_contract_low_quality_candidates_never_default_recommended() -> None:
    row = {
        "exists": True,
        "asset_kind": material_references.KEYFRAME_DIR_NAME,
        "quality_score": 0.95,
        "quality_bucket": "low_quality",
        "yolo_annotation_rendered": True,
        "vlm_semantics": {
            "evidence_packet": {
                "top_detections": [
                    {"label": "paper", "confidence": 0.94, "bbox": [100, 100, 180, 160]},
                    {"label": "gloved_hand", "confidence": 0.93, "bbox": [80, 90, 130, 150]},
                ]
            }
        },
    }

    assert material_references._candidate_recommendation_eligible(row) is False

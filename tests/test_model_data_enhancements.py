import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from labsopguard.datasets.action_dataset import load_action_dataset, validate_action_dataset
from labsopguard.evaluation.event_regression import evaluate_dataset_outputs, evaluate_event_predictions
from labsopguard.event_preprocessing.event_proposal import EventProposalBuilder
from labsopguard.event_preprocessing.event_segmentation import EventSegmenter
from labsopguard.event_preprocessing.key_material_extraction import KeyMaterialExtractor
from labsopguard.event_preprocessing.material_index_writer import EventMaterialIndexWriter
from labsopguard.event_preprocessing.schemas import DetectionBox, DetectionFrame, EventProposal, PhysicalEvent
from labsopguard.event_preprocessing.state_resolution import ModelBasedStateResolver, write_default_container_state_model
from labsopguard.event_preprocessing.tracking import TrackRelationGraphBuilder, TrackStreamBuilder
from labsopguard.event_preprocessing.tracking.providers import StrongSortLiteTrackingProvider, build_tracking_provider


def test_strong_sort_lite_recovers_fast_motion_where_iou_breaks():
    frames = [
        DetectionFrame(0, 0.0, [DetectionBox((10, 10, 30, 30), "bottle", 0.9)]),
        DetectionFrame(1, 0.5, []),
        DetectionFrame(2, 1.0, [DetectionBox((58, 10, 78, 30), "bottle", 0.92)]),
    ]
    tracks = StrongSortLiteTrackingProvider(distance_gate_px=120).track(frames)
    assert len(tracks) == 1
    assert tracks[0].recovered_from_fragment is True
    assert tracks[0].fragment_count >= 2
    assert tracks[0].tracking_backend == "strong_sort_lite"
    assert build_tracking_provider("strong_sort_lite").backend_info.available is True


def test_gloved_hand_object_geometry_generates_interaction_event():
    frames = [
        DetectionFrame(
            0,
            0.0,
            [
                DetectionBox((10, 10, 45, 70), "gloved_hand", 0.91),
                DetectionBox((42, 18, 78, 78), "sample_bottle", 0.88),
            ],
        ),
        DetectionFrame(
            1,
            0.5,
            [
                DetectionBox((14, 10, 49, 70), "gloved_hand", 0.92),
                DetectionBox((48, 18, 84, 78), "sample_bottle", 0.89),
            ],
        ),
        DetectionFrame(
            2,
            1.0,
            [
                DetectionBox((18, 10, 53, 70), "gloved_hand", 0.92),
                DetectionBox((56, 18, 92, 78), "sample_bottle", 0.89),
            ],
        ),
    ]
    tracklets = StrongSortLiteTrackingProvider(distance_gate_px=120).track(frames)
    tracked = TrackStreamBuilder().build(experiment_id="exp", source_video_id="video", tracklets=tracklets)
    relations = TrackRelationGraphBuilder().build(tracked)
    assert any(rel.relation_type == "glove_object_interaction" for rel in relations)

    proposals = EventProposalBuilder(min_event_duration=0.5).build(frames, tracklets, tracked, relations)
    interactions = [item for item in proposals if item.event_type == "hand_object_interaction"]
    assert interactions
    assert interactions[0].actor_track_id
    assert "gloved_hand" in interactions[0].related_detection_classes
    assert "sample_bottle" in interactions[0].related_detection_classes


def test_gloved_hand_rules_cover_all_core_event_types():
    frames = [
        DetectionFrame(
            0,
            0.0,
            [
                DetectionBox((10, 10, 45, 70), "gloved_hand", 0.91),
                DetectionBox((42, 18, 78, 78), "sample_bottle", 0.88),
                DetectionBox((90, 18, 126, 78), "beaker", 0.86),
                DetectionBox((52, 0, 72, 30), "pipette", 0.82),
                DetectionBox((120, 5, 180, 55), "balance", 0.8),
                DetectionBox((72, 8, 88, 24), "tube-cap", 0.75),
            ],
        ),
        DetectionFrame(
            1,
            0.5,
            [
                DetectionBox((18, 10, 53, 70), "gloved_hand", 0.92),
                DetectionBox((50, 18, 86, 78), "sample_bottle", 0.89),
                DetectionBox((90, 18, 126, 78), "beaker", 0.86),
                DetectionBox((58, 0, 78, 30), "pipette", 0.83),
                DetectionBox((124, 5, 184, 55), "balance", 0.8),
                DetectionBox((80, 8, 96, 24), "tube-cap", 0.75),
            ],
            semantic_activities=["transfer liquid", "operate balance"],
            change_score=0.08,
        ),
        DetectionFrame(
            2,
            1.0,
            [
                DetectionBox((26, 10, 61, 70), "gloved_hand", 0.92),
                DetectionBox((60, 18, 96, 78), "sample_bottle", 0.89),
                DetectionBox((90, 18, 126, 78), "beaker", 0.86),
                DetectionBox((64, 0, 84, 30), "pipette", 0.83),
                DetectionBox((128, 5, 188, 55), "balance", 0.8),
                DetectionBox((88, 8, 104, 24), "tube-cap", 0.75),
            ],
            semantic_activities=["pour transfer pipette", "open cap"],
            change_score=0.08,
        ),
    ]
    tracklets = StrongSortLiteTrackingProvider(distance_gate_px=140).track(frames)
    tracked = TrackStreamBuilder().build(experiment_id="exp", source_video_id="video", tracklets=tracklets)
    relations = TrackRelationGraphBuilder().build(tracked)
    relation_types = {rel.relation_type for rel in relations}
    assert {"glove_object_interaction", "object_manipulation", "container_state_interaction"} <= relation_types

    proposals = EventProposalBuilder(min_event_duration=0.5).build(frames, tracklets, tracked, relations)
    event_types = {item.event_type for item in proposals}
    assert "hand_object_interaction" in event_types
    assert "object_move" in event_types
    assert "liquid_transfer" in event_types
    assert "panel_operation" in event_types
    assert "container_state_change" in event_types
    liquid = next(item for item in proposals if item.event_type == "liquid_transfer")
    assert liquid.actor_track_id
    assert liquid.source_container is not None


def _event_proposal(
    event_type: str,
    start: float,
    end: float,
    *,
    dominant_object: str,
    contact_target_class: str = "",
) -> EventProposal:
    return EventProposal(
        proposal_id=f"proposal_{event_type}_{start}_{end}_{dominant_object}",
        event_type=event_type,
        start_frame_idx=int(start * 10),
        end_frame_idx=int(end * 10),
        start_time_sec=start,
        end_time_sec=end,
        evidence_frame_indices=[int(start * 10), int(end * 10)],
        involved_objects=["gloved_hand", dominant_object],
        dominant_object=dominant_object,
        involved_track_ids=[],
        primary_track_id=None,
        source_container=None,
        target_container=None,
        track_motion_summary={},
        actor_track_id=None,
        tool_track_id=None,
        related_tracks=[],
        transfer_mode=None,
        action_resolution_source="test",
        action_resolution_notes="test",
        supporting_relation_ids=[],
        direction_confidence=None,
        direction_status=None,
        direction_evidence=[],
        state_before=None,
        state_after=None,
        state_change_type=None,
        state_confidence=None,
        state_evidence=[],
        evidence_grade="medium",
        review_status="candidate_review",
        evidence_summary="test",
        related_detection_classes=["gloved_hand", dominant_object],
        confidence=0.8,
        contact_target_class=contact_target_class,
    )


def test_event_segmenter_keeps_different_contact_targets_separate_and_bounds_duration():
    segmenter = EventSegmenter(pre_roll_sec=0.5, post_roll_sec=0.5, max_gap_merge=2.0, max_event_duration_sec=6.0)
    proposals = [
        _event_proposal("hand_object_interaction", 0.0, 4.0, dominant_object="balance", contact_target_class="balance"),
        _event_proposal("hand_object_interaction", 4.5, 8.0, dominant_object="sample_bottle", contact_target_class="sample_bottle"),
        _event_proposal("object_move", 10.0, 35.0, dominant_object="beaker"),
    ]

    events = segmenter.segment(
        proposals,
        experiment_id="exp",
        source_video_id="video",
        video_duration_sec=40.0,
        experiment_name="test",
    )

    interactions = [item for item in events if item.event_type == "hand_object_interaction"]
    assert len(interactions) == 2
    assert interactions[0].dominant_object == "balance"
    assert interactions[1].dominant_object == "sample_bottle"
    assert all(item.duration_sec <= 6.0 for item in events)


def _physical_event(duration: float = 6.0, event_type: str = "hand_object_interaction") -> PhysicalEvent:
    return PhysicalEvent(
        event_id="evt_quality",
        experiment_id="exp",
        source_video_id="video",
        event_type=event_type,
        stable_name="quality_event",
        display_name="Quality event",
        actor_name="operator",
        start_time_sec=10.0,
        end_time_sec=10.0 + duration,
        duration_sec=duration,
        key_timestamps=[11.0, 13.0],
        involved_objects=["gloved_hand", "sample_bottle"],
        dominant_object="sample_bottle",
        involved_track_ids=[],
        primary_track_id=None,
        source_container=None,
        target_container=None,
        track_motion_summary={},
        actor_track_id=None,
        tool_track_id=None,
        related_tracks=[],
        transfer_mode=None,
        action_resolution_source="test",
        action_resolution_notes="test",
        supporting_relation_ids=[],
        direction_confidence=None,
        direction_status=None,
        direction_evidence=[],
        state_before=None,
        state_after=None,
        state_change_type=None,
        state_confidence=None,
        state_evidence=[],
        evidence_grade="medium",
        review_status="candidate_review",
        evidence_summary="test event",
        confidence=0.8,
        event_status="ready",
        proposal_source="test",
        evidence_frame_indices=[],
        related_detection_classes=["gloved_hand", "sample_bottle"],
    )


def test_key_material_quality_score_uses_asset_and_keyframe_metadata(tmp_path: Path):
    event = _physical_event(duration=5.0)
    preview = tmp_path / "preview.jpg"
    preview.write_bytes(b"preview")
    keyframes = []
    for idx in range(3):
        path = tmp_path / f"keyframe_{idx}.jpg"
        path.write_bytes(b"frame")
        keyframes.append(str(path))
    keyframe_selection = [
        {"detection_count": 3, "quality": {"contact": True, "sharpness": 0.72, "box_area_ratio": 0.18}},
        {"detection_count": 2, "quality": {"contact": False, "sharpness": 0.65, "box_area_ratio": 0.12}},
        {"detection_count": 4, "quality": {"contact": True, "sharpness": 0.8, "box_area_ratio": 0.2}},
    ]

    quality = KeyMaterialExtractor.score_asset_quality(
        event=event,
        asset_status="ready",
        preview_path=preview,
        keyframe_paths=keyframes,
        keyframe_selection=keyframe_selection,
    )

    assert quality["quality_score"] >= 85
    assert quality["quality_grade"] == "excellent"
    assert "contact_evidence_present" in quality["quality_reasons"]
    assert "preview_present" in quality["quality_reasons"]


def test_key_material_quality_penalizes_failed_sparse_assets(tmp_path: Path):
    event = _physical_event(duration=0.4)
    quality = KeyMaterialExtractor.score_asset_quality(
        event=event,
        asset_status="failed:cannot-open-video",
        preview_path=tmp_path / "missing_preview.jpg",
        keyframe_paths=[],
        keyframe_selection=[],
    )

    assert quality["quality_score"] < 50
    assert quality["quality_grade"] == "poor"
    assert {"duration_too_short", "asset_not_ready", "keyframes_missing", "preview_missing"} <= set(quality["quality_reasons"])


def test_material_index_writer_persists_quality_fields(tmp_path: Path):
    event = _physical_event(duration=4.0)
    event.asset_pack = {
        "clip_path": str(tmp_path / "clip.mp4"),
        "preview_path": str(tmp_path / "preview.jpg"),
        "keyframe_paths": [str(tmp_path / "keyframe_01.jpg"), str(tmp_path / "keyframe_02.jpg")],
        "asset_status": "ready",
        "quality_score": 76.5,
        "quality_grade": "good",
        "quality_reasons": ["duration_ok", "preview_present"],
    }
    writer = EventMaterialIndexWriter(tmp_path / "materials.db")
    try:
        writer.write_events([event])
        rows = writer.query_events(experiment_id="exp")
    finally:
        writer.close()

    assert rows[0]["quality_score"] == 76.5
    assert rows[0]["quality_grade"] == "good"
    assert rows[0]["quality_reasons"] == ["duration_ok", "preview_present"]
    assert rows[0]["payload"]["asset_pack"]["quality_grade"] == "good"


def test_action_dataset_validation_and_event_regression(tmp_path: Path):
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            {
                "schema_version": "lab_action_dataset.v1",
                "dataset_id": "regression_smoke",
                "records": [
                    {
                        "video_id": "video_001",
                        "video_path": "videos/video_001.mp4",
                        "annotations": [
                            {"event_id": "ann_1", "event_type": "object_move", "start_time_sec": 81.0, "end_time_sec": 84.0},
                            {
                                "event_id": "ann_2",
                                "event_type": "liquid_transfer",
                                "start_time_sec": 96.0,
                                "end_time_sec": 103.0,
                                "source_container": {"class_name": "bottle"},
                                "target_container": {"class_name": "beaker"},
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    dataset = load_action_dataset(dataset_path)
    validation = validate_action_dataset(dataset)
    assert validation["is_valid"] is True
    metrics = evaluate_event_predictions(
        [
            {"event_id": "evt_1", "event_type": "object_move", "start_time_sec": 80.8, "end_time_sec": 84.2},
            {"event_id": "evt_2", "event_type": "liquid_transfer", "start_time_sec": 96.1, "end_time_sec": 102.9},
        ],
        dataset.records[0].annotations,
    )
    assert metrics["precision"] == 1.0
    outputs = tmp_path / "outputs" / "video_001"
    outputs.mkdir(parents=True)
    (outputs / "physical_events.json").write_text(json.dumps({"events": [{"event_id": "evt_1", "event_type": "object_move", "start_time_sec": 81.0, "end_time_sec": 84.0}]}), encoding="utf-8")
    report = evaluate_dataset_outputs(dataset_path, tmp_path / "outputs")
    assert report["summary"]["true_positive"] == 1
    assert report["summary"]["false_negative"] == 1


def test_prototype_container_state_model(tmp_path: Path):
    import cv2

    model_path = tmp_path / "state_model.json"
    write_default_container_state_model(model_path)
    image_path = tmp_path / "keyframe.jpg"
    image = np.full((64, 64, 3), 180, dtype=np.uint8)
    cv2.rectangle(image, (15, 15), (50, 50), (255, 255, 255), 2)
    cv2.imwrite(str(image_path), image)
    resolver = ModelBasedStateResolver(model_path=model_path)
    result = resolver.resolve(
        tracked_object=None,
        keyframe_paths=[image_path],
        semantic_summary={"summary": "operator opened the bottle cap"},
    )
    assert result.state_change_type == "container_open_candidate"
    assert result.confidence >= 0.8
    assert result.resolution_source == "prototype_container_state_model"

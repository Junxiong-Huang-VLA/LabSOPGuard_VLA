from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from labsopguard.config import RuntimeSettings
from labsopguard.video_analysis import DetectionResult, VideoAnalysisPipeline


def _pipeline():
    return VideoAnalysisPipeline(
        settings=RuntimeSettings(project_root=ROOT, yolo_model_path=None, device="cpu"),
        yolo_model_path=None,
        vlm_api_key=None,
    )


def test_no_actor_scene_does_not_trigger_missing_ppe():
    p = _pipeline()
    alerts = p._build_alerts(
        {"gloves": False, "goggles": False, "lab_coat": False},
        [],
        {"description": "实验台上有电子天平和试剂瓶，未见操作人员，物品静置摆放。"},
    )
    assert "missing_goggles" not in alerts
    assert "missing_lab_coat" not in alerts
    assert alerts == ["ppe_not_applicable:no_actor_detected"]


def test_actor_ppe_relevant_scene_triggers_missing_ppe():
    p = _pipeline()
    detections = [DetectionResult(frame_idx=1, timestamp_sec=1.0, bbox=(0, 0, 10, 10), class_name="hand", confidence=0.9)]
    alerts = p._build_alerts(
        {"gloves": True, "goggles": False, "lab_coat": False},
        detections,
        {"description": "操作人员正在移液 transfer reagent", "detected_activities": ["transfer"]},
    )
    assert "missing_goggles" in alerts
    assert "missing_lab_coat" in alerts
    assert "missing_gloves" not in alerts

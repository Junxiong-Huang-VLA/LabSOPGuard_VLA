from __future__ import annotations

from pathlib import Path

from backend.feishu_step_material_bot import (
    build_step_query_feishu_message,
    send_step_query_result_to_feishu,
)


class FakeFeishuNotifier:
    def __init__(self) -> None:
        self.text = ""
        self.image_bytes = b""
        self.image_filename = ""
        self.image_content_type = ""

    def send_text(self, text: str) -> str:
        self.text = text
        return "om_text"

    def upload_image(self, image_bytes: bytes, *, filename: str, content_type: str = "image/jpeg") -> str:
        self.image_bytes = image_bytes
        self.image_filename = filename
        self.image_content_type = content_type
        return "img_key"

    def send_image(self, image_key: str) -> str:
        assert image_key == "img_key"
        return "om_image"


def _query_result() -> dict:
    return {
        "experiment_id": "exp_001",
        "step_text": "检查试剂瓶归位是否正确",
        "message_video_time_sec": 725.0,
        "search_window": {"start_sec": 635.0, "end_sec": 905.0},
        "judgement": {
            "status": "incorrect",
            "label": "不符合要求",
            "confidence": 0.86,
            "reason": "目标对象从 A 区移动到了 B 区。",
            "evidence_material_id": "mat_evt_001",
        },
        "candidates": [
            {
                "material_id": "mat_evt_001",
                "event_type": "object_move",
                "start_sec": 720.0,
                "end_sec": 735.0,
                "objects": ["gloved_hand", "reagent_bottle"],
                "clip_path": "materials/events/evt_001/clip.mp4",
                "preview_path": "materials/events/evt_001/preview.jpg",
                "confidence": 0.86,
            }
        ],
    }


def test_build_step_query_feishu_message_contains_alignment_and_material_urls(tmp_path: Path) -> None:
    message = build_step_query_feishu_message(
        experiment_id="exp_001",
        experiment_title="固体称量实验",
        query_result=_query_result(),
        experiment_dir=tmp_path,
        public_base_url="https://labsop.example.com",
    )

    assert "实验：固体称量实验" in message
    assert "结论：不符合要求" in message
    assert "视频 12:05" in message
    assert "https://labsop.example.com/api/v1/experiments/exp_001/files/materials/events/evt_001/clip.mp4" in message
    assert "素材ID：mat_evt_001" in message


def test_send_step_query_result_to_feishu_sends_text_and_first_evidence_image(tmp_path: Path) -> None:
    image_path = tmp_path / "materials" / "events" / "evt_001" / "preview.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"fake-jpeg")
    notifier = FakeFeishuNotifier()

    result = send_step_query_result_to_feishu(
        notifier=notifier,
        experiment_dir=tmp_path,
        experiment_id="exp_001",
        experiment_title="固体称量实验",
        query_result=_query_result(),
        public_base_url="https://labsop.example.com",
    ).to_dict()

    assert result["send_result"]["text_sent"] is True
    assert result["send_result"]["image_sent"] is True
    assert result["send_result"]["message_ids"] == ["om_text", "om_image"]
    assert result["image_path"] == str(image_path)
    assert notifier.image_bytes == b"fake-jpeg"
    assert notifier.image_filename == "preview.jpg"
    assert notifier.image_content_type == "image/jpeg"


def test_send_step_query_result_uses_formal_material_stored_file_image(tmp_path: Path) -> None:
    image_path = tmp_path / "material_references" / "keyframes" / "paper.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"formal-image")
    notifier = FakeFeishuNotifier()
    query = _query_result()
    query["candidates"] = [
        {
            "material_id": "mat_clip",
            "asset_type": "video_clip",
            "start_sec": 10.0,
            "end_sec": 12.0,
            "stored_file": "clips/paper.mp4",
        },
        {
            "material_id": "mat_frame",
            "asset_type": "keyframe",
            "start_sec": 10.0,
            "end_sec": 12.0,
            "stored_file": "keyframes/paper.jpg",
        },
    ]

    result = send_step_query_result_to_feishu(
        notifier=notifier,
        experiment_dir=tmp_path,
        experiment_id="exp_001",
        experiment_title="Formal material package",
        query_result=query,
        public_base_url="https://labsop.example.com",
    ).to_dict()

    assert result["send_result"]["image_sent"] is True
    assert result["image_path"] == str(image_path)
    assert notifier.image_bytes == b"formal-image"

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import backend.main as backend_main


class FakeQueryResult:
    def to_dict(self) -> dict:
        return {
            "experiment_id": "feishu_json_exp",
            "step_text": "verify reagent bottle storage",
            "message_video_time_sec": 10.0,
            "search_window": {"start_sec": 0.0, "end_sec": 40.0},
            "judgement": {"status": "correct", "confidence": 0.8, "reason": "matched evidence"},
            "candidates": [],
        }


class FakePushResult:
    def to_dict(self) -> dict:
        return {
            "text": "evidence answer",
            "image_path": None,
            "send_result": {"text_sent": True, "message_ids": ["om_fake"]},
        }


def test_feishu_inbound_json_payload_routes_existing_material_query(tmp_path: Path, monkeypatch) -> None:
    backend_main.PROJECT_ROOT = tmp_path
    backend_main._EXPERIMENTS.clear()
    experiment_id = "feishu_json_exp"
    exp_dir = backend_main._experiment_output_dir(experiment_id)
    exp_dir.mkdir(parents=True)
    (exp_dir / "experiment.json").write_text(
        json.dumps({"experiment_id": experiment_id, "title": "Feishu JSON experiment"}),
        encoding="utf-8",
    )

    captured: dict[str, dict] = {}

    import backend.feishu_notifier as feishu_notifier
    import backend.feishu_step_material_bot as feishu_bot
    import labsopguard.key_material_reference as key_material_reference

    def fake_query_step_materials(**kwargs):
        captured["query"] = kwargs
        return FakeQueryResult()

    def fake_send_step_query_result_to_feishu(**kwargs):
        captured["push"] = kwargs
        return FakePushResult()

    def fake_from_env_for_receive(*, receive_id: str, receive_id_type: str):
        captured["receive"] = {"receive_id": receive_id, "receive_id_type": receive_id_type}
        return object()

    monkeypatch.setattr(key_material_reference, "query_step_materials", fake_query_step_materials)
    monkeypatch.setattr(feishu_bot, "send_step_query_result_to_feishu", fake_send_step_query_result_to_feishu)
    monkeypatch.setattr(feishu_notifier.FeishuNotifier, "from_env_for_receive", staticmethod(fake_from_env_for_receive))
    monkeypatch.setattr(backend_main, "_ensure_key_material_references", lambda *args, **kwargs: None)

    message = SimpleNamespace(
        message_type="text",
        chat_id="oc_json",
        sender_open_id="ou_json",
        create_time_iso="2026-05-12T09:59:00+08:00",
        text=json.dumps(
            {
                "experiment_id": experiment_id,
                "query": "verify reagent bottle storage",
                "message_sent_at": "2026-05-12T10:00:00+08:00",
                "window": {"before_sec": 15, "after_sec": 30},
                "limit": 4,
            }
        ),
    )

    result = backend_main._handle_feishu_inbound_message(message)

    assert result["status"] == "replied"
    assert result["experiment_id"] == experiment_id
    assert captured["query"]["experiment_dir"] == exp_dir
    assert captured["query"]["step_text"] == "verify reagent bottle storage"
    assert captured["query"]["message_sent_at"] == "2026-05-12T10:00:00+08:00"
    assert captured["query"]["window_before_sec"] == 15
    assert captured["query"]["window_after_sec"] == 30
    assert captured["query"]["limit"] == 4
    assert captured["receive"] == {"receive_id": "oc_json", "receive_id_type": "chat_id"}
    assert captured["push"]["query_result"]["judgement"]["status"] == "correct"

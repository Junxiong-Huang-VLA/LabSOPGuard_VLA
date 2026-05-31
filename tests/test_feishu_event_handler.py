from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.feishu_event_handler import (
    FeishuEventError,
    feishu_url_verification_response,
    get_bound_experiment_id,
    load_feishu_chat_bindings,
    parse_feishu_inbound_message,
    parse_step_command,
    save_feishu_chat_binding,
)


def test_feishu_url_verification_checks_token_and_returns_challenge() -> None:
    payload = {"type": "url_verification", "token": "tok", "challenge": "challenge-value"}

    assert feishu_url_verification_response(payload, expected_token="tok") == {"challenge": "challenge-value"}
    with pytest.raises(FeishuEventError):
        feishu_url_verification_response(payload, expected_token="other")


def test_parse_feishu_message_and_step_query_command() -> None:
    payload = {
        "schema": "2.0",
        "header": {"event_id": "evt_1", "event_type": "im.message.receive_v1", "token": "tok"},
        "event": {
            "sender": {"sender_id": {"open_id": "ou_1"}},
            "message": {
                "message_id": "om_1",
                "chat_id": "oc_1",
                "chat_type": "p2p",
                "message_type": "text",
                "create_time": "1778586725000",
                "content": '{"text":"<at user_id=\\"bot\\">LabSOPGuard</at> 实验ID: exp_001\\n检查试剂瓶归位是否正确"}',
            },
        },
    }

    message = parse_feishu_inbound_message(payload, expected_token="tok")
    assert message is not None
    assert message.event_id == "evt_1"
    assert message.chat_id == "oc_1"
    assert message.sender_open_id == "ou_1"
    assert message.text == "实验ID: exp_001\n检查试剂瓶归位是否正确"
    assert message.create_time_iso is not None

    parsed = parse_step_command(message.text)
    assert parsed.command == "query"
    assert parsed.experiment_id == "exp_001"
    assert parsed.step_text == "检查试剂瓶归位是否正确"


def test_parse_json_step_query_command_with_window_options() -> None:
    parsed = parse_step_command(
        json.dumps(
            {
                "experiment_id": "exp_json_001",
                "query": "verify bottle returned to tray",
                "message_sent_at": "2026-05-12T10:00:00+08:00",
                "window": {"before_sec": 30, "after_sec": 45},
                "limit": 2,
            }
        )
    )

    assert parsed.command == "query"
    assert parsed.experiment_id == "exp_json_001"
    assert parsed.step_text == "verify bottle returned to tray"
    assert parsed.message_sent_at == "2026-05-12T10:00:00+08:00"
    assert parsed.window_before_sec == 30
    assert parsed.window_after_sec == 45
    assert parsed.limit == 2


def test_parse_json_like_step_text_command() -> None:
    parsed = parse_step_command(
        "experiment_id: exp_jsonish_001, step_text: verify pipette tip disposal, window: 60, limit: 3"
    )

    assert parsed.command == "query"
    assert parsed.experiment_id == "exp_jsonish_001"
    assert parsed.step_text == "verify pipette tip disposal"
    assert parsed.window_before_sec == 60
    assert parsed.window_after_sec == 60
    assert parsed.limit == 3


def test_parse_bind_command_and_persist_chat_binding(tmp_path: Path) -> None:
    parsed = parse_step_command("绑定实验 solid_weighing_20260512")
    assert parsed.command == "bind"
    assert parsed.experiment_id == "solid_weighing_20260512"

    bindings_path = tmp_path / "chat_bindings.json"
    save_feishu_chat_binding(bindings_path, chat_id="oc_abc", experiment_id="solid_weighing_20260512")
    assert get_bound_experiment_id(bindings_path, chat_id="oc_abc") == "solid_weighing_20260512"

    payload = load_feishu_chat_bindings(bindings_path)
    assert payload["bindings"]["oc_abc"]["experiment_id"] == "solid_weighing_20260512"

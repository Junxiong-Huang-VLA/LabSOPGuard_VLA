from __future__ import annotations

from typing import Any

from key_action_indexer.yolo_vlm_pipeline import _normalize_vlm_payload, apply_yolo_vlm_review_pipeline


class _FakeQwenDescription:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.raw_response = payload
        self.model = "qwen-test"


class _FakeQwenClient:
    def __init__(self, payload: dict[str, Any] | BaseException) -> None:
        self.payload = payload
        self.calls = 0

    def describe_scene(self, path: str, *, prompt: str, temperature: float = 0.0) -> _FakeQwenDescription:
        self.calls += 1
        if isinstance(self.payload, BaseException):
            raise self.payload
        return _FakeQwenDescription(self.payload)


def test_qwen_cannot_upgrade_candidate_when_yolo_recheck_fails(tmp_path) -> None:
    keyframe = _keyframe(tmp_path)
    candidate_rows = [_candidate_row(keyframe)]
    client = _FakeQwenClient(
        {
            "description": "Qwen claims the hand is using a pipette.",
            "semantic_action": "confirmed pipetting",
            "confirmed_objects": ["pipette", "gloved_hand"],
            "confidence": 0.99,
            "evidence_alignment": "aligned",
        }
    )

    summary = apply_yolo_vlm_review_pipeline(
        tmp_path,
        candidate_rows,
        [_micro_row([_invalid_evidence_row()])],
        vlm_client=client,
        enable_vlm=True,
    )

    row = candidate_rows[0]
    assert client.calls == 0
    assert summary["groups_rejected_by_yolo_recheck"] == 1
    assert row["yolo_recheck"]["status"] == "failed"
    assert row["pipeline_stage"] == "blocked_by_yolo_recheck"
    assert row["pipeline_status"] == "blocked_by_yolo_recheck"
    assert row["candidate_status"] == "rejected"
    assert row["review_required"] is False


def test_qwen_cannot_downgrade_yolo_passed_candidate_to_rejected(tmp_path) -> None:
    keyframe = _keyframe(tmp_path)
    candidate_rows = [_candidate_row(keyframe)]
    client = _FakeQwenClient(
        {
            "description": "Qwen is unsure and does not cite the pipette.",
            "confirmed_objects": ["background_monitor"],
            "confidence": 0.01,
            "evidence_alignment": "uncertain",
        }
    )

    apply_yolo_vlm_review_pipeline(
        tmp_path,
        candidate_rows,
        [_micro_row(_valid_evidence_rows())],
        vlm_client=client,
        enable_vlm=True,
    )

    row = candidate_rows[0]
    assert client.calls == 1
    assert row["yolo_recheck"]["status"] == "passed"
    assert row["pipeline_stage"] == "frontend_review_gate"
    assert row["pipeline_status"] == "yolo_recheck_passed_vlm_advisory_uncertain"
    assert row.get("candidate_status", "pending") == "pending"
    assert row.get("review_status", "pending") == "pending"
    assert row["primary_object"] == "pipette"
    assert row["vlm_semantics"]["status"] == "uncertain_vlm_review"
    assert "background_monitor" in row["vlm_semantics"]["unsupported_confirmed_objects"]


def test_qwen_parse_failure_is_audit_only_and_preserves_yolo_pass(tmp_path) -> None:
    keyframe = _keyframe(tmp_path)
    candidate_rows = [_candidate_row(keyframe)]
    client = _FakeQwenClient(RuntimeError("parse failed: invalid JSON"))

    summary = apply_yolo_vlm_review_pipeline(
        tmp_path,
        candidate_rows,
        [_micro_row(_valid_evidence_rows())],
        vlm_client=client,
        enable_vlm=True,
    )

    row = candidate_rows[0]
    assert client.calls == 1
    assert summary["groups_rejected_by_yolo_recheck"] == 0
    assert row["yolo_recheck"]["status"] == "passed"
    assert row["vlm_semantics"]["status"] == "error"
    assert "parse failed" in row["vlm_semantics"]["error"]
    assert row["pipeline_status"] == "yolo_recheck_passed_vlm_advisory_uncertain"
    assert row.get("candidate_status", "pending") == "pending"


def test_qwen_empty_gate_cannot_accept() -> None:
    normalized = _normalize_vlm_payload(
        {"decision": "accept", "should_write_confirmed_event": True},
        {"allowed_confirmed_objects": ["pipette"], "primary_object": "pipette", "physical_event_gates": []},
    )

    assert normalized["decision"] == "uncertain"
    assert normalized["should_write_confirmed_event"] is False
    assert "missing_hard_gate" in normalized["missing_evidence"]


def test_qwen_candidate_gate_cannot_accept() -> None:
    normalized = _normalize_vlm_payload(
        {"decision": "accept", "should_write_confirmed_event": True},
        {"allowed_confirmed_objects": ["pipette"], "primary_object": "pipette", "physical_event_gates": [_gate("candidate", False)]},
    )

    assert normalized["decision"] == "uncertain"
    assert normalized["should_write_confirmed_event"] is False


def test_qwen_rejected_gate_cannot_accept() -> None:
    normalized = _normalize_vlm_payload(
        {"decision": "accept", "should_write_confirmed_event": True},
        {"allowed_confirmed_objects": ["pipette"], "primary_object": "pipette", "physical_event_gates": [_gate("rejected", False)]},
    )

    assert normalized["decision"] == "uncertain"
    assert normalized["should_write_confirmed_event"] is False


def test_qwen_parse_failed_cannot_accept() -> None:
    normalized = _normalize_vlm_payload(
        {},
        {"allowed_confirmed_objects": ["pipette"], "primary_object": "pipette", "physical_event_gates": [_gate("confirmed", True)]},
    )

    assert normalized["status"] == "parse_failed"
    assert normalized["should_write_confirmed_event"] is False


def test_qwen_event_audits_jsonl_is_written(tmp_path) -> None:
    keyframe = _keyframe(tmp_path)
    candidate_rows = [_candidate_row(keyframe)]
    client = _FakeQwenClient(
        {
            "decision": "accept",
            "should_write_confirmed_event": True,
            "confirmed_objects": ["pipette", "gloved_hand"],
            "evidence_alignment": "aligned",
            "confidence": 0.99,
        }
    )

    summary = apply_yolo_vlm_review_pipeline(
        tmp_path,
        candidate_rows,
        [_micro_row(_valid_evidence_rows())],
        vlm_client=client,
        enable_vlm=True,
    )

    audit_path = tmp_path / "metadata" / "qwen_event_audits.jsonl"
    rows = [line for line in audit_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert summary["qwen_audit_count"] == 1
    assert len(rows) == 1
    assert '"missing_hard_gate"' in rows[0]
    assert '"should_write_confirmed_event": false' in rows[0]


def _gate(status: str, passed: bool) -> dict[str, Any]:
    return {
        "status": status,
        "hard_gate": {
            "passed": passed,
            "gate_name": "unit_test_gate",
            "required_evidence": [],
            "passed_evidence": [],
            "failed_evidence": [],
        },
    }


def _keyframe(tmp_path) -> Any:
    keyframe = tmp_path / "front_peak.jpg"
    keyframe.write_bytes(b"fake image placeholder")
    return keyframe


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


def _micro_row(evidence: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "micro_segment_id": "micro-1",
        "primary_object": "pipette",
        "yolo_evidence": evidence,
    }


def _valid_evidence_rows() -> list[dict[str, Any]]:
    return [
        _valid_evidence_row("frame-001", 1.0),
        _valid_evidence_row("frame-002", 1.2),
    ]


def _valid_evidence_row(frame_id: str, time_sec: float) -> dict[str, Any]:
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


def _invalid_evidence_row() -> dict[str, Any]:
    return {
        "frame_id": "frame-001",
        "local_time_sec": 1.0,
        "view": "front",
        "frame_width": 960,
        "frame_height": 540,
        "detections": [
            {"label": "pipette", "confidence": 0.95, "bbox": [700, 380, 760, 440]},
            {"label": "gloved_hand", "confidence": 0.95, "bbox": [40, 40, 100, 120]},
        ],
        "hand_object_interactions": [],
    }

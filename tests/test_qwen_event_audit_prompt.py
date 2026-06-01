from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LABSOPGUARD_SRC = ROOT / "src"
if str(LABSOPGUARD_SRC) not in sys.path:
    sys.path.insert(0, str(LABSOPGUARD_SRC))

from labsopguard.event_preprocessing.qwen_event_audit_prompt import (  # noqa: E402
    FEW_SHOT_NEGATIVES,
    OUTPUT_SCHEMA,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    build_physical_event_audit_prompt,
)


def test_prompt_constants_are_json_serializable_and_hard_gate_scoped() -> None:
    json.dumps(OUTPUT_SCHEMA)
    json.dumps(FEW_SHOT_NEGATIVES)

    assert "hard_gate.status 不是 \"confirmed\"" in SYSTEM_PROMPT
    assert "不能因为物体出现在画面中就判断物体移动" in SYSTEM_PROMPT
    assert "{event_candidate_json}" in USER_PROMPT_TEMPLATE
    assert OUTPUT_SCHEMA["properties"]["decision"]["enum"] == ["accept", "reject", "uncertain"]
    assert "should_write_confirmed_event" in OUTPUT_SCHEMA["required"]


def test_few_shot_negatives_cover_five_forbidden_shortcuts() -> None:
    negatives = json.dumps(FEW_SHOT_NEGATIVES, ensure_ascii=False)

    assert "displacement_below_threshold" in negatives
    assert "near_only" in negatives
    assert "Static liquid presence" in negatives
    assert "Device presence alone" in negatives
    assert "Container detection alone" in negatives


def test_build_prompt_includes_gate_evidence_and_allowed_refs() -> None:
    bundle = build_physical_event_audit_prompt(
        {
            "candidate_id": "evt-001",
            "event_type": "object_move",
            "object_track_ids": ["trk_bottle_2"],
            "object_labels": ["reagent_bottle"],
        },
        hard_gate={
            "status": "rejected",
            "hard_gate": {"passed": False, "gate_name": "gate_object_move"},
        },
        evidence_json={
            "track_id": "trk_bottle_2",
            "stabilized_displacement_px": 3.2,
            "motion_threshold_px": 12.0,
        },
        frame_context=[{"frame_index": 10, "time_sec": 1.2}],
    )

    assert bundle["temperature"] == 0.0
    assert bundle["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert bundle["output_schema"] == OUTPUT_SCHEMA

    prompt = bundle["user_prompt"]
    assert "evt-001" in prompt
    assert "trk_bottle_2" in prompt
    assert "stabilized_displacement_px" in prompt
    assert "不要把 candidate / rejected / uncertain 的 hard_gate 提升成 confirmed" in prompt


def test_build_prompt_extracts_existing_gate_and_evidence_fields() -> None:
    bundle = build_physical_event_audit_prompt(
        event={
            "candidate_id": "evt-002",
            "event_type": "hand_object_interaction",
            "physical_event_gate": {
                "status": "candidate",
                "hard_gate": {"passed": False, "gate_name": "gate_hand_object_contact"},
            },
            "evidence_detail": {"near_only": True, "contact_frames": 0},
            "actor_track_id": "trk_hand_1",
            "object_track_ids": ["trk_paper_1"],
            "involved_objects": ["gloved_hand", "paper"],
        }
    )

    prompt = bundle["user_prompt"]
    assert "gate_hand_object_contact" in prompt
    assert "near_only" in prompt
    assert "trk_hand_1" in prompt
    assert "paper" in prompt

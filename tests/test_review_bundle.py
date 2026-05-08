from __future__ import annotations

import json
from pathlib import Path

from key_action_indexer.review_bundle import apply_review_to_process, export_review_bundle
from key_action_indexer.schemas import write_jsonl


def test_export_review_bundle_resolves_labsopguard_physical_event_refs(tmp_path: Path) -> None:
    session = tmp_path / "session"
    metadata = session / "metadata"
    metadata.mkdir(parents=True)
    bridge_ref = {"evidence_type": "labsopguard_physical_event", "event_id": "evt_bridge"}
    (metadata / "experiment_process.json").write_text(
        json.dumps(
            {
                "session_id": "s1",
                "steps": [
                    {
                        "step_id": "step_1",
                        "name": "Touch container",
                        "evidence_refs": [bridge_ref],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    write_jsonl(
        metadata / "human_confirmation_queue.jsonl",
        [
            {
                "confirmation_id": "s1:step_1",
                "item_id": "step_1",
                "status": "pending",
                "confidence": 0.45,
                "evidence_refs": [bridge_ref],
            }
        ],
    )
    write_jsonl(
        metadata / "model_observation_events.jsonl",
        [
            {
                "event_id": "evt_bridge",
                "observation_id": "evt_bridge",
                "event_type": "hand_object_contact",
                "path": "keyframes/contact.jpg",
                "asset_refs": [{"asset_type": "keyframe", "path": "keyframes/contact.jpg"}],
                "keyframe_refs": [{"path": "keyframes/contact.jpg"}],
            }
        ],
    )

    result = export_review_bundle(session, session / "exports" / "bundle.json")
    payload = json.loads(Path(result["output_path"]).read_text(encoding="utf-8"))
    item = payload["items"][0]

    assert item["step_id"] == "step_1"
    assert item["suggested_decision"] == "review"
    assert item["resolved_evidence_refs"][0]["resolved"] is True
    assert item["keyframe_paths"] == ["keyframes/contact.jpg"]


def test_apply_review_to_process_marks_approved_step_completed(tmp_path: Path) -> None:
    session = tmp_path / "session"
    metadata = session / "metadata"
    metadata.mkdir(parents=True)
    (metadata / "experiment_process.json").write_text(
        json.dumps({"session_id": "s1", "steps": [{"step_id": "step_1", "requires_human_confirmation": True}]}),
        encoding="utf-8",
    )

    result = apply_review_to_process(session, "s1:step_1", "approved", "operator", "looks correct")
    process = json.loads((metadata / "experiment_process.json").read_text(encoding="utf-8"))

    assert result["decision"] == "approved"
    assert process["steps"][0]["status"] == "completed"
    assert process["steps"][0]["requires_human_confirmation"] is False
    assert (metadata / "step_review_history.jsonl").exists()

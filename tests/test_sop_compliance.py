from __future__ import annotations

import json
from pathlib import Path

from key_action_indexer.cli import main
from key_action_indexer.sop_compliance import build_sop_compliance_report


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def test_sop_compliance_maps_steps_to_local_evidence_and_flags_gaps() -> None:
    sop = {
        "steps": [
            {"step_id": "weigh", "name": "Weigh sample", "expected_action": "weighing", "required": True},
            {"step_id": "transfer", "name": "Transfer liquid", "expected_action": "pipetting", "required": True},
            {"step_id": "record", "name": "Record readout", "expected_action": "recording", "required": False},
        ]
    }
    report = build_sop_compliance_report(
        sop,
        key_actions=[
            {
                "segment_id": "seg_weigh",
                "action_type": "weighing",
                "objects": ["balance", "sample"],
                "confidence": 0.92,
            }
        ],
        evidence_refs=[
            {
                "material_id": "mat_transfer",
                "event_type": "liquid_transfer",
                "actions": ["liquid_transfer"],
                "objects": ["pipette", "tube"],
                "confidence": 0.84,
            }
        ],
        physical_changes=[
            {
                "change_id": "chg_unexpected_move",
                "event_type": "object_move",
                "actions": ["object_move"],
                "objects": ["waste_container"],
                "confidence": 0.9,
            }
        ],
    )

    events = report["events"]
    by_step = {event["sop_ref"].get("step_id"): event for event in events if event["sop_ref"].get("step_id")}
    field_names = {"sop_ref", "severity", "description", "recommendation", "evidence_refs", "confidence", "review_status"}

    assert report["backend_required"] is False
    assert report["dry_run_compatible"] is True
    assert field_names <= set(events[0])
    assert by_step["weigh"]["compliance_status"] == "compliant"
    assert by_step["weigh"]["severity"] == "Minor"
    assert by_step["transfer"]["compliance_status"] == "compliant"
    assert by_step["transfer"]["evidence_refs"][0]["material_id"] == "mat_transfer"
    assert by_step["record"]["compliance_status"] == "missing_evidence"
    assert by_step["record"]["severity"] == "Minor"
    assert report["coverage"]["mapped_step_count"] == 2
    assert report["coverage"]["missing_step_count"] == 1
    assert any(event["compliance_status"] == "unexpected_evidence" for event in events)


def test_sop_compliance_preserves_declared_critical_missing_step() -> None:
    report = build_sop_compliance_report(
        {
            "steps": [
                {
                    "step_id": "critical_transfer",
                    "name": "Transfer reagent",
                    "expected_action": "pipetting",
                    "critical": True,
                }
            ]
        },
        include_unmapped_observations=False,
    )

    event = report["events"][0]

    assert event["compliance_status"] == "missing_evidence"
    assert event["severity"] == "Critical"
    assert event["review_status"] == "needs_review"
    assert event["evidence_refs"] == []


def test_sop_compliance_cli_reads_jsonl_and_writes_report(tmp_path: Path) -> None:
    sop_path = tmp_path / "sop.json"
    key_actions_path = tmp_path / "key_actions.jsonl"
    changes_path = tmp_path / "physical_change_log.jsonl"
    report_path = tmp_path / "sop_compliance.json"
    events_path = tmp_path / "sop_compliance_events.jsonl"
    _write_json(
        sop_path,
        {
            "steps": [
                {"step_id": "weigh", "name": "Weigh sample", "expected_action": "weighing"},
                {"step_id": "cleanup", "name": "Clean bench", "expected_action": "cleanup", "required": False},
            ]
        },
    )
    _write_jsonl(
        key_actions_path,
        [
            {
                "segment_id": "seg_001",
                "action_type": "weighing",
                "objects": ["balance", "sample"],
                "confidence": 0.93,
            }
        ],
    )
    _write_jsonl(
        changes_path,
        [
            {
                "change_id": "chg_001",
                "event_type": "object_move",
                "actions": ["object_move"],
                "objects": ["sample"],
                "confidence": 0.7,
            }
        ],
    )

    exit_code = main(
        [
            "sop-compliance",
            "--sop",
            str(sop_path),
            "--key-actions",
            str(key_actions_path),
            "--physical-change-log",
            str(changes_path),
            "--output",
            str(report_path),
            "--events-output",
            str(events_path),
        ]
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    event_rows = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert exit_code == 0
    assert report["schema_version"] == "sop_compliance_report.v1"
    assert report["source_counts"]["key_actions"] == 1
    assert report["source_counts"]["physical_changes"] == 1
    assert event_rows
    assert {event["severity"] for event in event_rows} <= {"Critical", "Major", "Minor"}

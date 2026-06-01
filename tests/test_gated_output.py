from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LABSOPGUARD_SRC = ROOT / "src"
if str(LABSOPGUARD_SRC) not in sys.path:
    sys.path.insert(0, str(LABSOPGUARD_SRC))

from labsopguard.event_preprocessing.gated_output import (  # noqa: E402
    empty_gated_physical_events_payload,
    legacy_physical_event_candidate_rows,
    write_gated_physical_events,
)


def test_legacy_physical_events_are_ungated_candidates_not_final_payload() -> None:
    rows = legacy_physical_event_candidate_rows(
        [
            {
                "event_id": "legacy_evt_1",
                "event_type": "object_move",
                "metadata": {"status": "confirmed"},
            }
        ]
    )

    assert rows[0]["status"] == "ungated_legacy"
    assert rows[0]["hard_gate"]["passed"] is False
    assert "legacy_ungated_event" in rows[0]["reject_reasons"]


def test_writer_rejects_final_confirmed_event_without_passed_hard_gate(tmp_path: Path) -> None:
    ok = write_gated_physical_events(
        tmp_path,
        {
            "schema": "physical_events.v4",
            "events": [
                {
                    "event_id": "evt_1",
                    "event_type": "object_move",
                    "status": "confirmed",
                    "hard_gate": {"passed": False},
                    "evidence_detail": {},
                    "reject_reasons": [],
                    "limitations": [],
                }
            ],
        },
    )

    assert ok is False
    assert not (tmp_path / "physical_events.json").exists()
    assert (tmp_path / "physical_events_write_rejected.json").exists()


def test_source_video_missing_failure_payload_replaces_final_with_empty_gated_payload(tmp_path: Path) -> None:
    payload = empty_gated_physical_events_payload(
        experiment_id="exp_missing",
        status="failed",
        failure_reason="source_video_missing",
    )

    ok = write_gated_physical_events(tmp_path, payload, source="formal_v2_source_video_missing")

    assert ok is True
    final_payload = json.loads((tmp_path / "physical_events.json").read_text(encoding="utf-8"))
    assert final_payload["schema"] == "physical_events.v4"
    assert final_payload["status"] == "failed"
    assert final_payload["failure_reason"] == "source_video_missing"
    assert final_payload["events"] == []
    assert final_payload["gate_required"] is True

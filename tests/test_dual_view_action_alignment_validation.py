from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from key_action_indexer.dual_view_action_validation import (
    canonicalize_action,
    summarize_gpu_config,
    validate_dual_view_action_alignment,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _candidate(candidate_id: str, view: str, action: str, *, group: str = "micro-1", start: float = 10.0) -> dict:
    return {
        "candidate_id": candidate_id,
        "micro_segment_id": group,
        "view": view,
        "action_type": action,
        "start_sec": start,
        "end_sec": start + 1.0,
        "evidence_level": "strong",
        "confidence": 0.91,
    }


def _materials(group: str = "micro-1", action: str = "hand-bottle") -> list[dict]:
    rows = []
    for view in ("first_person", "third_person"):
        rows.append(
            {
                "candidate_id": f"{view}-frame",
                "micro_segment_id": group,
                "canonical_action_type": action,
                "view": view,
                "asset_kind": "keyframe",
                "source_file": f"{view}.jpg",
            }
        )
        rows.append(
            {
                "candidate_id": f"{view}-clip",
                "micro_segment_id": group,
                "canonical_action_type": action,
                "view": view,
                "asset_kind": "keyclip",
                "source_file": f"{view}.mp4",
            }
        )
    return rows


def test_single_view_strong_evidence_does_not_create_formal_dual_view_event() -> None:
    report = validate_dual_view_action_alignment(
        [_candidate("first-only", "first_person", "hand bottle contact")],
        material_rows=_materials(),
    )

    assert report["events"] == []
    assert report["summary"]["formal_event_count"] == 0
    assert report["rejected_candidates"][0]["reason"] == "missing_complementary_view"


def test_dual_view_action_normalization_and_timing_fields() -> None:
    rows = [
        _candidate("first", "first_person", "hand bottle contact", start=10.0),
        _candidate("third", "third_person", "reagent_bottle_interaction", start=10.4),
    ]

    report = validate_dual_view_action_alignment(rows, material_rows=_materials())

    assert canonicalize_action("reagent_bottle_interaction") == "hand-bottle"
    assert report["summary"]["formal_event_count"] == 1
    event = report["events"][0]
    assert event["canonical_action_type"] == "hand-bottle"
    assert event["status"] == "formal"
    assert set(event["material_refs"]) == {"first_person", "third_person"}
    assert event["timing"]["start_sec"] == 10.0
    assert event["timing"]["duration_sec"] == 1.4
    assert event["timing"]["view_delta_sec"] == 0.4


def test_formal_material_requires_first_and_third_keyframe_and_keyclip() -> None:
    rows = [
        _candidate("first", "first_person", "hand-paper", group="micro-paper"),
        _candidate("third", "third_person", "hand-paper", group="micro-paper", start=10.2),
    ]
    materials = [
        row
        for row in _materials(group="micro-paper", action="hand-paper")
        if not (row["view"] == "third_person" and row["asset_kind"] == "keyclip")
    ]

    report = validate_dual_view_action_alignment(rows, material_rows=materials)

    assert report["events"] == []
    assert report["rejected_candidates"][0]["reason"] == "missing_formal_material"
    assert "third_person:keyclip" in report["rejected_candidates"][0]["missing_materials"]


def test_gpu_config_detection_is_data_only_and_timing_rows_are_checked() -> None:
    timing_rows = [
        {
            "stage": "coarse_scan",
            "wall_sec": 1.25,
            "requested_device": "cuda:0",
            "actual_device": "0",
        }
    ]

    gpu = summarize_gpu_config({"yolo_device": "cuda:0"}, timing_rows)
    report = validate_dual_view_action_alignment([], timing_rows=timing_rows, config={"yolo_device": "cuda:0"})

    assert gpu["gpu_requested"] is True
    assert gpu["gpu_observed"] is True
    assert gpu["status"] == "gpu_observed"
    assert report["timing_rows_summary"]["has_required_timing_fields"] is True


def test_validation_script_accepts_small_synthetic_jsonl(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.jsonl"
    materials = tmp_path / "materials.jsonl"
    timing = tmp_path / "timing.jsonl"
    config = tmp_path / "config.json"
    output = tmp_path / "dual_view_report.json"
    _write_jsonl(
        candidates,
        [
            _candidate("first", "first_person", "hand bottle contact", start=20.0),
            _candidate("third", "third_person", "reagent_bottle_interaction", start=20.5),
        ],
    )
    _write_jsonl(materials, _materials())
    _write_jsonl(
        timing,
        [{"stage": "micro_refine", "wall_sec": 0.5, "requested_device": "auto", "actual_device": "cpu"}],
    )
    config.write_text(json.dumps({"yolo_device": "auto"}), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "validate_dual_view_action_alignment.py"),
            "--candidates",
            str(candidates),
            "--materials",
            str(materials),
            "--timing",
            str(timing),
            "--config",
            str(config),
            "--output",
            str(output),
            "--require-formal-event",
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["summary"]["formal_event_count"] == 1
    assert report["events"][0]["timing"]["alignment_source"] == "dual_view_candidate_jsonl"

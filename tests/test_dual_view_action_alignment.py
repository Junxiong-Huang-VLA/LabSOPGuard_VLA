from __future__ import annotations

from pathlib import Path

from key_action_indexer.dual_view_action_alignment import (
    ViewActionEvidence,
    build_dual_view_action_events,
    match_dual_view_action_events,
)
from key_action_indexer.schemas import SessionManifest, read_jsonl, write_jsonl


def _interaction_row(view: str, time_sec: float, primary_object: str, score: float) -> dict:
    return {
        "source_view": view,
        "alignment_time_sec": time_sec,
        "local_time_sec": time_sec,
        "frame_index": int(time_sec * 30),
        "interaction_score": score,
        "active_score": score,
        "detections": [
            {"label": "gloved_hand", "confidence": 0.9, "bbox": [10, 10, 40, 40]},
            {"label": primary_object, "confidence": 0.86, "bbox": [34, 20, 78, 64]},
        ],
        "hand_object_interactions": [
            {
                "hand_label": "gloved_hand",
                "object_label": primary_object,
                "score": score,
                "iou": 0.08,
            }
        ],
    }


def _phase_interaction_row(view: str, time_sec: float, primary_object: str, score: float, phase: str) -> dict:
    row = _interaction_row(view, time_sec, primary_object, score)
    row["action_phase"] = phase
    return row


def _copresence_row(view: str, time_sec: float, primary_object: str, score: float) -> dict:
    return {
        "source_view": view,
        "alignment_time_sec": time_sec,
        "local_time_sec": time_sec,
        "frame_index": int(time_sec * 30),
        "active_score": score,
        "label_counts": {"gloved_hand": 1, primary_object: 1},
        "detections": [
            {"label": "gloved_hand", "confidence": 0.9, "bbox": [10, 10, 40, 40]},
            {"label": primary_object, "confidence": 0.86, "bbox": [34, 20, 78, 64]},
        ],
    }


def _manifest(tmp_path: Path) -> SessionManifest:
    session = tmp_path / "session"
    return SessionManifest.from_dict(
        {
            "session_id": "s1",
            "session_start_time": "1970-01-01T00:00:00+00:00",
            "videos": {
                "first_person": {"path": str(session / "first.mp4"), "start_time": "1970-01-01T00:00:00+00:00", "fps": 30},
                "third_person": {"path": str(session / "third.mp4"), "start_time": "1970-01-01T00:00:00+00:00", "fps": 30},
            },
            "output_dir": str(session),
        }
    )


def _strong_evidence(
    evidence_id: str,
    view: str,
    action_family: str,
    object_family: str,
    *,
    start: float = 10.0,
    end: float = 10.4,
    action_phase: str = "bench_operation",
) -> ViewActionEvidence:
    return ViewActionEvidence(
        evidence_id=evidence_id,
        session_id="s1",
        view=view,
        session_start_sec=start,
        session_end_sec=end,
        duration_sec=max(0.1, end - start),
        peak_session_sec=end,
        action_family=action_family,
        action_display_name=action_family,
        object_family=object_family,
        object_display_name=object_family,
        raw_yolo_labels=[],
        hand_count=1,
        object_count=1,
        row_count=2,
        interaction_row_count=2,
        max_interaction_score=0.9,
        avg_interaction_score=0.85,
        evidence_density=1.0,
        evidence_kind="hand_object_interaction",
        source_row_indices=[1, 2],
        source_frame_indices=[300, 312],
        quality_flags=[],
        action_phase=action_phase,
    )


def test_dual_view_action_events_require_first_and_third_evidence(tmp_path: Path) -> None:
    session = tmp_path / "session"
    metadata = session / "metadata"
    cv_outputs = session / "cv_outputs"
    metadata.mkdir(parents=True)
    cv_outputs.mkdir()
    write_jsonl(
        metadata / "micro_segments.jsonl",
        [
            {
                "session_id": "s1",
                "parent_segment_id": "seg_001",
                "micro_segment_id": "micro_bottle",
                "start_sec": 9.5,
                "end_sec": 11.0,
                "interaction": {"primary_object": "sample_bottle"},
            },
            {
                "session_id": "s1",
                "parent_segment_id": "seg_001",
                "micro_segment_id": "micro_spatula",
                "start_sec": 19.5,
                "end_sec": 21.0,
                "interaction": {"primary_object": "spatula"},
            },
        ],
    )
    write_jsonl(
        cv_outputs / "yolo_frame_rows.jsonl",
        [
            _interaction_row("first_person", 10.0, "reagent_bottle", 0.82),
            _interaction_row("first_person", 10.4, "reagent_bottle", 0.84),
            _interaction_row("first_person", 20.0, "spatula", 0.91),
            _interaction_row("first_person", 20.4, "spatula", 0.9),
        ],
    )
    write_jsonl(
        cv_outputs / "yolo_micro_frame_rows.jsonl",
        [
            _interaction_row("third_person", 10.1, "reagent_bottle", 0.78),
            _interaction_row("third_person", 10.5, "reagent_bottle", 0.8),
        ],
    )

    summary = build_dual_view_action_events(session)
    evidence = read_jsonl(metadata / "view_action_evidence.jsonl")
    events = read_jsonl(metadata / "dual_view_action_events.jsonl")
    unmatched = read_jsonl(metadata / "unmatched_view_evidence.jsonl")

    assert summary["view_action_evidence_count"] == 3
    assert summary["dual_view_action_event_count"] == 1
    assert summary["unmatched_reason_counts"] == {"action_mismatch": 1}
    diagnostics = summary["view_alignment_diagnostics"]
    assert diagnostics["formal_event_count"] == 1
    assert diagnostics["first_person"]["explicit_interaction_count"] == 2
    assert diagnostics["third_person"]["explicit_interaction_count"] == 1
    assert any(
        row["object_family"] == "reagent_bottle_family" and row["explicit_both_views_present"]
        for row in diagnostics["coverage_by_action_family"]
    )
    assert len(evidence) == 3
    assert len(events) == 1
    assert len(unmatched) == 1
    assert set(events[0]["views"]) == {"first_person", "third_person"}
    assert events[0]["required_views"] == ["first_person", "third_person"]
    assert events[0]["status"] == "matched_dual_view"
    assert "micro_bottle" in events[0]["micro_segment_ids"]
    assert unmatched[0]["view"] == "first_person"
    assert unmatched[0]["primary_object"] == "spatula"
    assert unmatched[0]["formal_event_promoted"] is False


def test_micro_segment_yolo_evidence_can_form_dual_view_event(tmp_path: Path) -> None:
    session = tmp_path / "session"
    metadata = session / "metadata"
    metadata.mkdir(parents=True)
    write_jsonl(
        metadata / "micro_segments.jsonl",
        [
            {
                "session_id": "s1",
                "parent_segment_id": "seg_001",
                "micro_segment_id": "micro_001",
                "start_sec": 5.0,
                "end_sec": 6.0,
                "interaction": {
                    "primary_object": "pipette",
                    "interaction_type": "hand_pipette_contact",
                    "peak_interaction_sec": 5.5,
                    "max_interaction_score": 0.86,
                },
                "yolo_evidence": [
                    {
                        "source_view": "first_person",
                        "time_sec": 5.4,
                        "frame_index": 162,
                        "primary_object": "pipette",
                        "interaction_score": 0.86,
                    },
                    {
                        "source_view": "first_person",
                        "time_sec": 5.55,
                        "frame_index": 166,
                        "primary_object": "pipette",
                        "interaction_score": 0.85,
                    },
                        {
                            "source_view": "third_person",
                            "time_sec": 5.5,
                            "frame_index": 168,
                            "primary_object": "pipette_tip",
                            "interaction_score": 0.82,
                        },
                        {
                            "source_view": "third_person",
                            "time_sec": 5.65,
                            "frame_index": 172,
                            "primary_object": "pipette_tip",
                            "interaction_score": 0.81,
                    },
                ],
            }
        ],
    )

    build_dual_view_action_events(session)
    events = read_jsonl(metadata / "dual_view_action_events.jsonl")
    unmatched = read_jsonl(metadata / "unmatched_view_evidence.jsonl")

    assert len(events) == 1
    assert events[0]["primary_object_family"] == "pipette_family"
    assert events[0]["micro_segment_ids"] == ["micro_001"]
    assert unmatched == []


def test_sparse_paired_explicit_interaction_can_form_dual_view_event(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("KEY_ACTION_ALLOW_DUAL_VIEW_SPARSE_PAIRING", "1")
    monkeypatch.setenv("KEY_ACTION_REQUIRE_DUAL_VIEW_TEMPORAL_OVERLAP", "0")
    session = tmp_path / "session"
    cv_outputs = session / "cv_outputs"
    metadata = session / "metadata"
    cv_outputs.mkdir(parents=True)
    metadata.mkdir()
    write_jsonl(
        cv_outputs / "yolo_frame_rows.jsonl",
        [
            _interaction_row("first_person", 10.0, "pipette", 0.9),
            _interaction_row("third_person", 10.1, "pipette", 0.91),
        ],
    )

    summary = build_dual_view_action_events(session)
    events = read_jsonl(metadata / "dual_view_action_events.jsonl")
    unmatched = read_jsonl(metadata / "unmatched_view_evidence.jsonl")

    assert summary["dual_view_action_event_count"] == 1
    assert len(events) == 1
    assert unmatched == []
    assert "dual_view_sparse_pairing" in ",".join(events[0]["decision_trace"])


def test_action_phase_mismatch_blocks_formal_dual_view_event(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    first = [
        _strong_evidence(
            "first_glove_prep",
            "first_person",
            "container_operation",
            "container_family",
            action_phase="prep_glove_or_sleeve",
        )
    ]
    third = [
        _strong_evidence(
            "third_bench_operation",
            "third_person",
            "container_operation",
            "container_family",
            action_phase="bench_operation",
        )
    ]

    events, unmatched = match_dual_view_action_events(manifest, first, third)

    assert events == []
    assert {row["reason"] for row in unmatched} == {"phase_mismatch"}
    assert all(row["formal_material_allowed"] is False for row in unmatched)


def test_sparse_paired_interaction_is_rejected_by_default(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("KEY_ACTION_ALLOW_DUAL_VIEW_SPARSE_PAIRING", raising=False)
    session = tmp_path / "session"
    cv_outputs = session / "cv_outputs"
    metadata = session / "metadata"
    cv_outputs.mkdir(parents=True)
    metadata.mkdir()
    write_jsonl(
        cv_outputs / "yolo_frame_rows.jsonl",
        [
            _interaction_row("first_person", 10.0, "pipette", 0.9),
            _interaction_row("third_person", 10.1, "pipette", 0.91),
        ],
    )

    summary = build_dual_view_action_events(session)
    events = read_jsonl(metadata / "dual_view_action_events.jsonl")
    unmatched = read_jsonl(metadata / "unmatched_view_evidence.jsonl")

    assert summary["dual_view_action_event_count"] == 0
    assert events == []
    assert {row["reason"] for row in unmatched} == {"weak_view_evidence"}

    session_no_explicit = tmp_path / "session_no_explicit"
    cv_outputs_no_explicit = session_no_explicit / "cv_outputs"
    metadata_no_explicit = session_no_explicit / "metadata"
    cv_outputs_no_explicit.mkdir(parents=True)
    metadata_no_explicit.mkdir()
    write_jsonl(
        cv_outputs_no_explicit / "yolo_frame_rows.jsonl",
        [
            _copresence_row("first_person", 20.0, "pipette", 0.9),
            _copresence_row("first_person", 20.4, "pipette", 0.9),
            _copresence_row("third_person", 20.1, "pipette", 0.91),
            _copresence_row("third_person", 20.5, "pipette", 0.91),
        ],
    )

    build_dual_view_action_events(session_no_explicit)
    no_explicit_events = read_jsonl(metadata_no_explicit / "dual_view_action_events.jsonl")
    no_explicit_unmatched = read_jsonl(metadata_no_explicit / "unmatched_view_evidence.jsonl")

    assert no_explicit_events == []
    assert {row["reason"] for row in no_explicit_unmatched} == {"weak_view_evidence"}
    assert all("no_explicit_hand_object_interaction" in ",".join(row["weak_evidence_reasons"]) for row in no_explicit_unmatched)


def test_magnetic_stir_bar_forms_equipment_dual_view_event(tmp_path: Path) -> None:
    session = tmp_path / "session"
    cv_outputs = session / "cv_outputs"
    metadata = session / "metadata"
    cv_outputs.mkdir(parents=True)
    metadata.mkdir()
    write_jsonl(
        cv_outputs / "yolo_frame_rows.jsonl",
        [
            _interaction_row("first_person", 10.0, "magnetic_stir_bar", 0.9),
            _interaction_row("first_person", 10.4, "magnetic_stir_bar", 0.9),
            _interaction_row("third_person", 10.1, "magnetic_stir_bar", 0.91),
            _interaction_row("third_person", 10.5, "magnetic_stir_bar", 0.91),
        ],
    )

    summary = build_dual_view_action_events(session)
    events = read_jsonl(metadata / "dual_view_action_events.jsonl")

    assert summary["dual_view_action_event_count"] == 1
    assert events[0]["action_family"] == "equipment_operation"
    assert events[0]["object_family"] == "equipment_family"
    assert events[0]["action_phase"] == "bench_operation"


def test_third_bench_and_first_glove_prep_do_not_form_formal_event(tmp_path: Path) -> None:
    session = tmp_path / "session"
    cv_outputs = session / "cv_outputs"
    metadata = session / "metadata"
    cv_outputs.mkdir(parents=True)
    metadata.mkdir()
    write_jsonl(
        cv_outputs / "yolo_frame_rows.jsonl",
        [
            _phase_interaction_row("first_person", 10.0, "pipette", 0.9, "prep_glove_or_sleeve"),
            _phase_interaction_row("first_person", 10.4, "pipette", 0.91, "prep_glove_or_sleeve"),
            _phase_interaction_row("third_person", 10.1, "pipette", 0.9, "bench_operation"),
            _phase_interaction_row("third_person", 10.5, "pipette", 0.91, "bench_operation"),
        ],
    )

    summary = build_dual_view_action_events(session)
    evidence = read_jsonl(metadata / "view_action_evidence.jsonl")
    events = read_jsonl(metadata / "dual_view_action_events.jsonl")
    unmatched = read_jsonl(metadata / "unmatched_view_evidence.jsonl")

    assert summary["dual_view_action_event_count"] == 0
    assert summary["phase_mismatch"] == 2
    assert summary["phase_unknown"] == 0
    assert summary["unmatched_reason_counts"] == {"phase_mismatch": 2}
    assert events == []
    assert {row["action_phase"] for row in evidence} == {"prep_glove_or_sleeve", "bench_operation"}
    assert {row["reason"] for row in unmatched} == {"phase_mismatch"}
    assert all(row["formal_event_promoted"] is False for row in unmatched)


def test_unknown_phase_stays_unmatched_and_summarized(tmp_path: Path) -> None:
    session = tmp_path / "session"
    cv_outputs = session / "cv_outputs"
    metadata = session / "metadata"
    cv_outputs.mkdir(parents=True)
    metadata.mkdir()
    write_jsonl(
        cv_outputs / "yolo_frame_rows.jsonl",
        [
            _phase_interaction_row("first_person", 30.0, "reagent_bottle", 0.88, "unknown"),
            _phase_interaction_row("first_person", 30.4, "reagent_bottle", 0.89, "unknown"),
            _phase_interaction_row("third_person", 30.1, "reagent_bottle", 0.9, "bench_operation"),
            _phase_interaction_row("third_person", 30.5, "reagent_bottle", 0.91, "bench_operation"),
        ],
    )

    summary = build_dual_view_action_events(session)
    events = read_jsonl(metadata / "dual_view_action_events.jsonl")
    unmatched = read_jsonl(metadata / "unmatched_view_evidence.jsonl")

    assert events == []
    assert summary["phase_unknown"] == 2
    assert summary["phase_mismatch"] == 0
    assert summary["unmatched_reason_counts"] == {"phase_unknown": 2}
    assert {row["reason"] for row in unmatched} == {"phase_unknown"}


def test_single_view_sparse_interaction_stays_candidate(tmp_path: Path) -> None:
    session = tmp_path / "session"
    cv_outputs = session / "cv_outputs"
    metadata = session / "metadata"
    cv_outputs.mkdir(parents=True)
    metadata.mkdir()
    write_jsonl(
        cv_outputs / "yolo_frame_rows.jsonl",
        [_interaction_row("first_person", 10.0, "pipette", 0.9)],
    )

    summary = build_dual_view_action_events(session)
    events = read_jsonl(metadata / "dual_view_action_events.jsonl")
    unmatched = read_jsonl(metadata / "unmatched_view_evidence.jsonl")

    assert summary["dual_view_action_event_count"] == 0
    assert events == []
    assert unmatched[0]["reason"] == "weak_view_evidence"
    assert unmatched[0]["formal_material_allowed"] is False


def test_strict_pair_gate_reports_action_mismatch_and_no_temporal_overlap(tmp_path: Path) -> None:
    session = tmp_path / "session"
    cv_outputs = session / "cv_outputs"
    metadata = session / "metadata"
    cv_outputs.mkdir(parents=True)
    metadata.mkdir()
    write_jsonl(
        cv_outputs / "yolo_frame_rows.jsonl",
        [
            _interaction_row("first_person", 10.0, "pipette", 0.9),
            _interaction_row("first_person", 10.4, "pipette", 0.91),
            _interaction_row("third_person", 10.1, "spatula", 0.89),
            _interaction_row("third_person", 10.5, "spatula", 0.9),
        ],
    )

    build_dual_view_action_events(session)
    unmatched = read_jsonl(metadata / "unmatched_view_evidence.jsonl")

    assert {row["reason"] for row in unmatched} == {"action_mismatch"}

    session_far = tmp_path / "session_far"
    cv_outputs_far = session_far / "cv_outputs"
    metadata_far = session_far / "metadata"
    cv_outputs_far.mkdir(parents=True)
    metadata_far.mkdir()
    write_jsonl(
        cv_outputs_far / "yolo_frame_rows.jsonl",
        [
            _interaction_row("first_person", 10.0, "pipette", 0.9),
            _interaction_row("first_person", 10.4, "pipette", 0.91),
            _interaction_row("third_person", 13.0, "pipette", 0.89),
            _interaction_row("third_person", 13.4, "pipette", 0.9),
        ],
    )

    build_dual_view_action_events(session_far)
    far_unmatched = read_jsonl(metadata_far / "unmatched_view_evidence.jsonl")

    assert {row["reason"] for row in far_unmatched} == {"no_temporal_overlap"}


def test_long_overlapping_bucket_with_mismatched_peak_stays_unmatched(tmp_path: Path) -> None:
    first = _strong_evidence(
        "first",
        "first_person",
        "weighing_paper_operation",
        "paper_family",
        start=100.0,
        end=100.4,
    )
    third = _strong_evidence(
        "third",
        "third_person",
        "weighing_paper_operation",
        "paper_family",
        start=96.0,
        end=106.0,
    )

    events, unmatched = match_dual_view_action_events(_manifest(tmp_path), [first], [third])

    assert events == []
    assert {row["reason"] for row in unmatched} == {"peak_mismatch"}
    assert all(row["formal_material_allowed"] is False for row in unmatched)


def test_object_mismatch_reason_uses_canonical_action_before_object(tmp_path: Path) -> None:
    first = _strong_evidence("first", "first_person", "hand_object_operation", "pipette_family")
    third = _strong_evidence("third", "third_person", "hand_object_operation", "spatula_family", start=10.1, end=10.5)

    events, unmatched = match_dual_view_action_events(_manifest(tmp_path), [first], [third])

    assert events == []
    assert {row["reason"] for row in unmatched} == {"object_mismatch"}


def test_reagent_bottle_open_and_cap_align_with_raw_labels_preserved(tmp_path: Path) -> None:
    session = tmp_path / "session"
    cv_outputs = session / "cv_outputs"
    metadata = session / "metadata"
    cv_outputs.mkdir(parents=True)
    metadata.mkdir()
    write_jsonl(
        cv_outputs / "yolo_frame_rows.jsonl",
        [
            _interaction_row("first_person", 10.0, "reagent_bottle_open", 0.9),
            _interaction_row("first_person", 10.4, "reagent_bottle_open", 0.91),
            _interaction_row("third_person", 10.1, "bottle_cap", 0.88),
            _interaction_row("third_person", 10.5, "bottle_cap", 0.89),
        ],
    )

    build_dual_view_action_events(session)
    evidence = read_jsonl(metadata / "view_action_evidence.jsonl")
    events = read_jsonl(metadata / "dual_view_action_events.jsonl")

    assert len(events) == 1
    assert events[0]["action_display_name"] == "手部与试剂瓶操作"
    assert events[0]["object_family"] == "reagent_bottle_family"
    assert events[0]["views"]["first_person"]["raw_labels"] == ["gloved_hand", "reagent_bottle_open"]
    assert events[0]["views"]["third_person"]["raw_labels"] == ["bottle_cap", "gloved_hand"]
    assert any("reagent_bottle_open" in row["raw_labels"] for row in evidence)
    assert any("bottle_cap" in row["raw_labels"] for row in evidence)


def test_reagent_and_sample_bottle_same_action_but_different_object_family_do_not_align(tmp_path: Path) -> None:
    session = tmp_path / "session"
    cv_outputs = session / "cv_outputs"
    metadata = session / "metadata"
    cv_outputs.mkdir(parents=True)
    metadata.mkdir()
    write_jsonl(
        cv_outputs / "yolo_frame_rows.jsonl",
        [
            _interaction_row("first_person", 10.0, "reagent_bottle", 0.9),
            _interaction_row("first_person", 10.4, "reagent_bottle", 0.91),
            _interaction_row("third_person", 10.1, "sample_bottle", 0.88),
            _interaction_row("third_person", 10.5, "sample_bottle", 0.89),
        ],
    )

    build_dual_view_action_events(session)
    events = read_jsonl(metadata / "dual_view_action_events.jsonl")
    unmatched = read_jsonl(metadata / "unmatched_view_evidence.jsonl")

    assert events == []
    assert {row["reason"] for row in unmatched} == {"object_mismatch"}
    assert all(row["formal_material_allowed"] is False for row in unmatched)


def test_single_view_candidate_reason_when_no_other_view(tmp_path: Path) -> None:
    first = _strong_evidence("first", "first_person", "pipette_operation", "pipette_family")

    events, unmatched = match_dual_view_action_events(_manifest(tmp_path), [first], [])

    assert events == []
    assert unmatched[0]["reason"] == "single_view_candidate"

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from key_action_indexer.episode_segmenter import _apply_episode_support_quality_to_segment_row, _episode_specs_from_micros
from key_action_indexer.schemas import SessionManifest


def _manifest(tmp_path: Path) -> SessionManifest:
    output_dir = tmp_path / "session"
    return SessionManifest.from_dict(
        {
            "session_id": "episode-test",
            "session_start_time": "2026-04-29T17:25:00+08:00",
            "videos": {
                "third_person": {
                    "path": str(output_dir / "raw" / "third.mp4"),
                    "start_time": "2026-04-29T17:25:00+08:00",
                    "fps": 30,
                    "offset_sec": 0,
                },
                "first_person": {
                    "path": str(output_dir / "raw" / "first.mp4"),
                    "start_time": "2026-04-29T17:25:00+08:00",
                    "fps": 30,
                    "offset_sec": 0,
                },
            },
            "output_dir": str(output_dir),
        }
    )


def _micro(mid: str, start: float, *, obj: str = "beaker", view: str = "third_person") -> dict[str, object]:
    return {
        "micro_segment_id": mid,
        "source_view": view,
        "start_sec": start,
        "end_sec": start + 4.0,
        "interaction": {
            "primary_object": obj,
            "detected_objects": [obj],
            "interaction_type": "hand_object_contact",
        },
        "yolo_evidence": [{"object_label": obj, "interaction": "contact", "source_view": view}],
    }


def _micro_with_evidence(mid: str, start: float, evidence: dict[str, object], *, view: str = "third_person") -> dict[str, object]:
    row = _micro(mid, start, view=view)
    row["evidence"] = evidence
    row["evidence_level"] = evidence.get("evidence_level")
    row["quality"] = {"warnings": evidence.get("limitations", [])}
    return row


def _micro_with_global_time(mid: str, session_start: float, local_start: float, view: str) -> dict[str, object]:
    base_time = datetime.fromisoformat("2026-04-29T17:25:00+08:00")
    row = _micro(mid, local_start, view=view)
    row["global_start_time"] = (base_time + timedelta(seconds=session_start)).isoformat()
    row["global_end_time"] = (base_time + timedelta(seconds=session_start + 4.0)).isoformat()
    return row


def test_episode_specs_merge_dense_interaction_fragments_without_expected_count(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("KEY_ACTION_EPISODE_INTERNAL_MERGE_GAP_SEC", raising=False)
    monkeypatch.delenv("KEY_ACTION_FAST_LOCATE_EXPERIMENT_MACRO_MERGE_GAP_SEC", raising=False)
    monkeypatch.setenv("KEY_ACTION_EPISODE_DENSE_MERGE_MIN_FRAGMENTS", "2")
    manifest = _manifest(tmp_path)
    offsets_by_episode = [
        [0, 24, 90, 160, 205],
        [0, 30, 110, 190, 240, 285],
        [0, 22, 85, 145, 215],
        [0, 28, 105, 180, 245, 292],
        [0, 25, 95, 170, 230],
        [0, 32, 108, 175, 238, 288],
    ]
    rows: list[dict[str, object]] = []
    for episode_index, offsets in enumerate(offsets_by_episode, start=1):
        base = (episode_index - 1) * 520.0
        for offset_index, offset in enumerate(offsets, start=1):
            view = "first_person" if offset_index % 2 else "third_person"
            rows.append(_micro(f"e{episode_index:02d}_m{offset_index:02d}", base + offset, view=view))

    specs = _episode_specs_from_micros(
        manifest,
        rows,
        gap_sec=7.0,
        pre_roll_sec=2.0,
        post_roll_sec=3.0,
        min_episode_duration_sec=6.0,
        min_micro_evidence_count=2,
    )

    assert len(rows) == 33
    assert len(specs) == 6
    assert [len(spec["micro_segment_ids"]) for spec in specs] == [5, 6, 5, 6, 5, 6]
    assert all(spec.get("episode_merge_strategy") == "density_gap_macro_merge" for spec in specs)
    assert specs[0]["true_start_sec"] == 0.0
    assert specs[0]["true_end_sec"] == 209.0
    assert specs[1]["previous_episode_gap_sec"] > specs[1]["macro_merge_gap_sec"]
    assert "small_silence_gap_within_macro_merge_gap" in specs[0]["episode_merge_reasons"]
    assert specs[0]["boundary_evidence"]["start_reason"] == "earliest_micro_physical_evidence_on_session_timeline"
    assert specs[0]["episode_merge_events"][0]["decision"] == "merge"


def test_episode_specs_keep_large_silent_gaps_as_episode_boundaries(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("KEY_ACTION_EPISODE_INTERNAL_MERGE_GAP_SEC", "120")
    monkeypatch.setenv("KEY_ACTION_EPISODE_DENSE_MERGE_MIN_FRAGMENTS", "2")
    manifest = _manifest(tmp_path)
    rows = [
        _micro("e1_m1", 0, view="first_person"),
        _micro("e1_m2", 45, view="third_person"),
        _micro("e1_m3", 110, view="first_person"),
        _micro("e2_m1", 340, view="first_person"),
        _micro("e2_m2", 386, view="third_person"),
    ]

    specs = _episode_specs_from_micros(
        manifest,
        rows,
        gap_sec=7.0,
        pre_roll_sec=2.0,
        post_roll_sec=3.0,
        min_episode_duration_sec=6.0,
        min_micro_evidence_count=2,
    )

    assert len(specs) == 2
    assert specs[0]["micro_segment_ids"] == ["e1_m1", "e1_m2", "e1_m3"]
    assert specs[1]["micro_segment_ids"] == ["e2_m1", "e2_m2"]
    assert specs[1]["previous_episode_gap_sec"] > 120
    assert specs[1]["previous_boundary_decision"]["decision"] == "split"
    assert specs[1]["previous_boundary_decision"]["reason"] == "large_silence_gap_exceeds_macro_merge_gap"


def test_episode_specs_use_session_timeline_before_merging_dual_view_rows(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("KEY_ACTION_EPISODE_DENSE_MERGE_MIN_FRAGMENTS", "2")
    manifest = _manifest(tmp_path)
    rows = [
        _micro_with_global_time("first_m1", session_start=100.0, local_start=5.0, view="first_person"),
        _micro_with_global_time("third_m1", session_start=108.0, local_start=108.0, view="third_person"),
    ]

    specs = _episode_specs_from_micros(
        manifest,
        rows,
        gap_sec=10.0,
        pre_roll_sec=0.0,
        post_roll_sec=0.0,
        min_episode_duration_sec=6.0,
        min_micro_evidence_count=2,
    )

    assert len(specs) == 1
    assert specs[0]["true_start_sec"] == 100.0
    assert specs[0]["true_end_sec"] == 112.0
    assert specs[0]["micro_segment_ids"] == ["first_m1", "third_m1"]
    assert specs[0]["source_view_counts"] == {"first_person": 1, "third_person": 1}
    assert specs[0]["boundary_evidence"]["timeline_source_counts"] == {"global_time": 2}


def test_episode_specs_mark_partial_or_weak_support_as_not_strong_fact(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("KEY_ACTION_EPISODE_DENSE_MERGE_MIN_FRAGMENTS", "2")
    manifest = _manifest(tmp_path)
    rows = [
        _micro_with_evidence(
            "strong_m1",
            0,
            {
                "evidence_level": "visual_confirmed",
                "process_evidence_role": "strong_process_evidence",
                "strong_process_evidence": True,
            },
            view="first_person",
        ),
        _micro_with_evidence(
            "weak_m2",
            8,
            {
                "evidence_level": "weak_visual_evidence",
                "process_evidence_role": "retrieval_candidate",
                "limitations": ["retrieval candidate only; not eligible for strong process claims"],
            },
            view="third_person",
        ),
    ]

    specs = _episode_specs_from_micros(
        manifest,
        rows,
        gap_sec=12.0,
        pre_roll_sec=0.0,
        post_roll_sec=0.0,
        min_episode_duration_sec=6.0,
        min_micro_evidence_count=2,
    )

    assert len(specs) == 1
    support = specs[0]["support_quality"]
    assert support["fact_strength"] == "partial"
    assert support["strong_fact_allowed"] is False
    assert support["strong_micro_segment_ids"] == ["strong_m1"]
    assert support["weak_micro_segment_ids"] == ["weak_m2"]
    assert specs[0]["understanding_fact_strength"] == "partial"
    assert specs[0]["strong_fact_allowed"] is False


def test_four_second_micro_is_candidate_action_window_not_official_episode(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("KEY_ACTION_EPISODE_DENSE_MERGE_MIN_FRAGMENTS", "2")
    manifest = _manifest(tmp_path)

    official_specs = _episode_specs_from_micros(
        manifest,
        [_micro("tiny_m1", 10.0, view="first_person")],
        gap_sec=7.0,
        pre_roll_sec=0.0,
        post_roll_sec=0.0,
        min_episode_duration_sec=6.0,
        min_micro_evidence_count=2,
    )
    all_specs = _episode_specs_from_micros(
        manifest,
        [_micro("tiny_m1", 10.0, view="first_person")],
        gap_sec=7.0,
        pre_roll_sec=0.0,
        post_roll_sec=0.0,
        min_episode_duration_sec=6.0,
        min_micro_evidence_count=2,
        include_candidates=True,
    )

    assert official_specs == []
    assert len(all_specs) == 1
    assert all_specs[0]["candidate_status"] == "candidate_action_window"
    assert all_specs[0]["official_episode"] is False
    assert "too_short_action_window" in all_specs[0]["candidate_reasons"]


def test_adjacent_action_fragments_merge_into_official_episode(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("KEY_ACTION_EPISODE_SHORT_MERGE_GAP_SEC", "10")
    monkeypatch.setenv("KEY_ACTION_EPISODE_DENSE_MERGE_MIN_FRAGMENTS", "6")
    manifest = _manifest(tmp_path)
    rows = [
        _micro("frag1_first", 0.0, view="first_person"),
        _micro("frag1_third", 4.5, view="third_person"),
        _micro("frag2_first", 12.0, view="first_person"),
        _micro("frag2_third", 16.0, view="third_person"),
    ]

    specs = _episode_specs_from_micros(
        manifest,
        rows,
        gap_sec=1.0,
        pre_roll_sec=0.0,
        post_roll_sec=0.0,
        min_episode_duration_sec=18.0,
        min_micro_evidence_count=2,
    )

    assert len(specs) == 1
    assert specs[0]["official_episode"] is True
    assert specs[0]["candidate_status"] == "official_episode"
    assert specs[0]["true_start_sec"] == 0.0
    assert specs[0]["true_end_sec"] == 20.0
    assert specs[0]["episode_merge_strategy"] == "short_gap_merge_without_dense_fragment_count"


def test_official_episode_expands_to_experiment_lifecycle_state(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("KEY_ACTION_EXPERIMENT_LIFECYCLE_GAP_SEC", "120")
    monkeypatch.setenv("KEY_ACTION_EXPERIMENT_WINDOW_MIN_SEC", "30")
    manifest = _manifest(tmp_path)
    rows = [
        _micro("action_first", 100.0, view="first_person", obj="paper"),
        _micro("action_third", 104.0, view="third_person", obj="paper"),
    ]
    activity_rows = [
        {
            "session_time_sec": 40.0,
            "source_view": "third_person",
            "label_counts": {"lab_coat": 1, "hand": 1},
        },
        {
            "session_time_sec": 45.0,
            "source_view": "first_person",
            "label_counts": {"gloved_hand": 1},
        },
        {
            "session_time_sec": 101.0,
            "source_view": "first_person",
            "label_counts": {"gloved_hand": 1, "paper": 1},
            "hand_object_interactions": [{"object_label": "paper", "score": 0.9}],
        },
        {
            "session_time_sec": 150.0,
            "source_view": "first_person",
            "label_counts": {"gloved_hand": 1},
        },
    ]

    specs = _episode_specs_from_micros(
        manifest,
        rows,
        activity_rows=activity_rows,
        gap_sec=12.0,
        pre_roll_sec=0.0,
        post_roll_sec=0.0,
        min_episode_duration_sec=30.0,
        min_micro_evidence_count=2,
    )

    assert len(specs) == 1
    assert specs[0]["official_episode"] is True
    assert specs[0]["start_sec"] <= 40.0
    assert specs[0]["end_sec"] >= 150.0
    expansion = specs[0]["episode_window_expansion"]
    assert expansion["source"] == "experiment_lifecycle_state_and_action_evidence"
    assert expansion["lifecycle_window_count"] == 1
    assert specs[0]["boundary_evidence"]["official_boundary_source"] == "experiment_lifecycle_state_window"


def test_candidate_and_official_episode_status_are_explicit(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("KEY_ACTION_EPISODE_DENSE_MERGE_MIN_FRAGMENTS", "2")
    monkeypatch.setenv("KEY_ACTION_EPISODE_SHORT_MERGE_GAP_SEC", "15")
    monkeypatch.setenv("KEY_ACTION_EPISODE_INTERNAL_MERGE_GAP_SEC", "20")
    manifest = _manifest(tmp_path)
    rows = [
        _micro("single_view_m1", 0.0, view="third_person"),
        _micro("single_view_m2", 8.0, view="third_person"),
        _micro("official_first", 100.0, view="first_person"),
        _micro("official_third", 118.0, view="third_person"),
    ]

    all_specs = _episode_specs_from_micros(
        manifest,
        rows,
        gap_sec=12.0,
        pre_roll_sec=0.0,
        post_roll_sec=0.0,
        min_episode_duration_sec=10.0,
        min_micro_evidence_count=2,
        include_candidates=True,
    )
    official_specs = _episode_specs_from_micros(
        manifest,
        rows,
        gap_sec=12.0,
        pre_roll_sec=0.0,
        post_roll_sec=0.0,
        min_episode_duration_sec=10.0,
        min_micro_evidence_count=2,
    )

    assert [spec["candidate_status"] for spec in all_specs] == ["candidate_action_window", "official_episode"]
    assert all_specs[0]["single_view_candidate"] is True
    assert "single_view_candidate" in all_specs[0]["candidate_reasons"]
    assert [spec["micro_segment_ids"] for spec in official_specs] == [["official_first", "official_third"]]


def test_weak_episode_support_downgrades_parent_segment_evidence() -> None:
    row = {
        "segment_id": "episode_000001",
        "evidence": {
            "evidence_level": "visual_confirmed",
            "evidence_reasons": ["parent clip had visual signal"],
            "limitations": [],
        },
    }
    support_quality = {
        "fact_strength": "weak",
        "strong_fact_allowed": False,
        "recommended_evidence_level": "weak_visual_evidence",
        "reasons": ["weak_child_micro_evidence", "not_eligible_for_strong_fact_claim"],
        "weak_micro_segment_ids": ["weak_m1"],
    }

    _apply_episode_support_quality_to_segment_row(row, support_quality)

    assert row["evidence"]["evidence_level"] == "weak_visual_evidence"
    assert row["evidence"]["original_segment_evidence_level"] == "visual_confirmed"
    assert row["strong_fact_allowed"] is False
    assert row["understanding_fact_strength"] == "weak"
    assert "not_eligible_for_strong_fact_claim" in row["evidence"]["evidence_reasons"]

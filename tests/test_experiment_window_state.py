from __future__ import annotations

import csv
import json
from pathlib import Path

from key_action_indexer.experiment_window_state import (
    SyncIndexLookup,
    build_experiment_state_artifacts,
    write_chunk_manifest,
)
from key_action_indexer.material_references import _write_material_stream
from key_action_indexer.schemas import SessionManifest, SessionVideos, VideoSource


def _manifest(tmp_path: Path) -> SessionManifest:
    third = tmp_path / "aligned_third.mp4"
    first = tmp_path / "aligned_first.mp4"
    third.write_bytes(b"third")
    first.write_bytes(b"first")
    sync = tmp_path / "metadata" / "dual_view_alignment" / "sync_index.csv"
    sync.parent.mkdir(parents=True, exist_ok=True)
    with sync.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sync_index",
                "unit_id",
                "global_timestamp_us",
                "reference_camera",
                "reference_frame_index",
                "reference_timestamp_us",
                "third_frame_index",
                "third_timestamp_us",
                "third_video_path",
                "first_frame_index",
                "first_timestamp_us",
                "first_video_path",
                "local_offset_ms",
                "delta_ms",
                "sync_quality",
                "is_valid_pair",
                "drop_reason",
            ],
        )
        writer.writeheader()
        for idx in range(0, 201):
            writer.writerow(
                {
                    "sync_index": idx,
                    "unit_id": "unit_001",
                    "global_timestamp_us": idx * 1_000_000,
                    "reference_camera": "first_person",
                    "reference_frame_index": idx,
                    "reference_timestamp_us": idx * 1_000_000,
                    "third_frame_index": idx,
                    "third_timestamp_us": idx * 1_000_000,
                    "third_video_path": "third.mp4",
                    "first_frame_index": idx,
                    "first_timestamp_us": idx * 1_000_000,
                    "first_video_path": "first.mp4",
                    "local_offset_ms": 0,
                    "delta_ms": 10,
                    "sync_quality": "good",
                    "is_valid_pair": "true",
                    "drop_reason": "",
                }
            )
    return SessionManifest(
        session_id="state-machine-test",
        session_start_time="2026-05-20T09:52:24+08:00",
        videos=SessionVideos(
            third_person=VideoSource(
                name="third_person",
                path=str(third),
                start_time="2026-05-20T09:52:24+08:00",
                duration_sec=200.0,
            ),
            first_person=VideoSource(
                name="first_person",
                path=str(first),
                start_time="2026-05-20T09:52:24+08:00",
                duration_sec=200.0,
            ),
        ),
        output_dir=str(tmp_path),
        config={
            "aligned_video_analysis": {
                "sync_index_csv": str(sync),
                "raw_sources": {
                    "third_person": {"path": "raw_third.mp4"},
                    "first_person": {"path": "raw_first.mp4"},
                },
            }
        },
    )


def _row(view: str, sec: float, labels: dict[str, int], *, interaction: bool = False) -> dict:
    return {
        "source_view": view,
        "aligned_time_sec": sec,
        "frame_index": int(sec),
        "label_counts": labels,
        "detections": [{"label": label, "confidence": 0.8, "bbox": [0, 0, 10, 10]} for label in labels],
        "hand_object_interactions": [
            {"hand_label": "gloved_hand", "object_label": "paper", "score": 0.8}
        ]
        if interaction
        else [],
        "interaction_score": 0.8 if interaction else 0.0,
    }


def test_state_machine_keeps_window_open_when_third_empty_but_first_active(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("KEY_ACTION_EXPERIMENT_STATE_NO_ACTIVITY_END_TIMEOUT_SEC", "20")
    manifest = _manifest(tmp_path)
    metadata = tmp_path / "metadata"
    rows = [
        _row("third_person", 0.0, {"gloved_hand": 1}),
        _row("first_person", 0.0, {"gloved_hand": 1}),
        _row("third_person", 10.0, {"gloved_hand": 1, "paper": 1}, interaction=True),
        _row("first_person", 10.0, {"gloved_hand": 1, "paper": 1}, interaction=True),
        _row("first_person", 40.0, {"gloved_hand": 1, "bottle": 1}, interaction=True),
        _row("first_person", 50.0, {"gloved_hand": 1, "bottle": 1}, interaction=True),
        _row("first_person", 80.0, {}),
        _row("third_person", 80.0, {}),
    ]

    result = build_experiment_state_artifacts(
        metadata,
        manifest,
        rows,
        [],
        min_duration_sec=5.0,
    )

    assert result["summary"]["formal_window_count"] == 1
    formal = json.loads((metadata / "formal_experiment_windows.json").read_text(encoding="utf-8"))
    window = formal["windows"][0]
    assert window["end_sec"] >= 50.0
    assert window["third_missing_but_first_active_ranges"]
    assert window["should_pass_formal_window"] is False
    assert window["status"] in {"formal_window_suspicious_needs_review", "formal_window_needs_human_review"}
    assert window["source_window_sync_index"].endswith("window_sync_index.csv")
    assert Path(window["source_window_sync_index"]).exists()
    window_rows = list(csv.DictReader(Path(window["source_window_sync_index"]).open("r", encoding="utf-8")))
    assert window_rows
    assert {
        "window_sync_index",
        "global_timestamp_us",
        "reference_camera",
        "first_frame_index",
        "third_frame_index",
        "source_sync_index",
        "local_pts_first",
        "local_pts_third",
    }.issubset(window_rows[0])
    review = json.loads((metadata / "formal_window_human_review_manifest.json").read_text(encoding="utf-8"))
    assert review["passed_visual_review_count"] == 0
    assert (metadata / "window_boundary_diagnosis_report.json").exists()
    assert (metadata / "window_sync_index_enforcement_report.json").exists()
    manifest = json.loads((tmp_path / "windows" / "window_artifact_manifest.json").read_text(encoding="utf-8"))
    assert manifest["window_count"] == 1
    trace = (metadata / "experiment_window_state_trace.jsonl").read_text(encoding="utf-8")
    assert "third_empty_end_blocked" in trace


def test_single_view_only_window_is_rejected_before_formal_phase_pass(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("KEY_ACTION_EXPERIMENT_STATE_NO_ACTIVITY_END_TIMEOUT_SEC", "20")
    manifest = _manifest(tmp_path)
    metadata = tmp_path / "metadata"
    rows = [
        _row("third_person", 0.0, {"gloved_hand": 1, "paper": 1}, interaction=True),
        _row("third_person", 10.0, {"gloved_hand": 1, "paper": 1}, interaction=True),
        _row("first_person", 80.0, {}),
        _row("third_person", 80.0, {}),
    ]

    result = build_experiment_state_artifacts(
        metadata,
        manifest,
        rows,
        [],
        min_duration_sec=5.0,
    )

    assert result["summary"]["formal_window_count"] == 0
    assert result["summary"]["formal_window_rejected_count"] == 1
    formal = json.loads((metadata / "formal_experiment_windows.json").read_text(encoding="utf-8"))
    window = formal["windows"][0]
    assert window["status"] == "formal_window_rejected"
    assert window["first_has_experiment_activity"] is False
    assert "first_view_has_no_experiment_activity_in_window" in window["reject_reason_if_any"]
    review = json.loads((metadata / "formal_window_human_review_manifest.json").read_text(encoding="utf-8"))
    assert review["recommended_reject_window_ids"] == ["formal_window_001"]


def test_formal_window_start_trims_long_third_only_prefix_to_first_anchor(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("KEY_ACTION_EXPERIMENT_STATE_NO_ACTIVITY_END_TIMEOUT_SEC", "30")
    monkeypatch.setenv("KEY_ACTION_EXPERIMENT_WINDOW_MAX_THIRD_ONLY_PREFIX_SEC", "20")
    manifest = _manifest(tmp_path)
    metadata = tmp_path / "metadata"
    rows = [
        _row("third_person", 0.0, {"gloved_hand": 1, "paper": 1}, interaction=True),
        _row("third_person", 10.0, {"gloved_hand": 1, "paper": 1}, interaction=True),
        _row("third_person", 20.0, {"gloved_hand": 1, "paper": 1}, interaction=True),
        _row("third_person", 30.0, {"gloved_hand": 1, "paper": 1}, interaction=True),
        _row("third_person", 40.0, {"gloved_hand": 1, "paper": 1}, interaction=True),
        _row("first_person", 80.0, {"gloved_hand": 1, "paper": 1}, interaction=True),
        _row("third_person", 80.0, {"gloved_hand": 1, "paper": 1}, interaction=True),
        _row("first_person", 90.0, {"gloved_hand": 1, "paper": 1}, interaction=True),
        _row("third_person", 90.0, {"gloved_hand": 1, "paper": 1}, interaction=True),
        _row("first_person", 130.0, {}),
        _row("third_person", 130.0, {}),
    ]

    build_experiment_state_artifacts(
        metadata,
        manifest,
        rows,
        [],
        min_duration_sec=5.0,
    )

    formal = json.loads((metadata / "formal_experiment_windows.json").read_text(encoding="utf-8"))
    window = formal["windows"][0]
    assert window["start_sec"] == 70.0
    assert "trimmed_long_third_only_prefix_to_first_person_anchor" in window["first_person_anchor_trim"]["notes"]
    assert all(item["start_sec"] >= 70.0 for item in window["supporting_third_segments"])


def test_static_ppe_storage_does_not_open_experiment_window(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("KEY_ACTION_EXPERIMENT_STATE_NO_ACTIVITY_END_TIMEOUT_SEC", "20")
    manifest = _manifest(tmp_path)
    metadata = tmp_path / "metadata"
    rows = [
        _row("third_person", 0.0, {"ppe_storage": 1, "pipette": 5, "tube": 4}),
        _row("third_person", 10.0, {"ppe_storage": 1, "pipette": 5, "tube": 4}),
        _row("first_person", 10.0, {}),
        _row("third_person", 20.0, {"ppe_storage": 1, "pipette": 5, "tube": 4}),
    ]

    result = build_experiment_state_artifacts(
        metadata,
        manifest,
        rows,
        [],
        min_duration_sec=5.0,
    )

    assert result["summary"]["candidate_window_count"] == 0
    assert result["summary"]["formal_window_count"] == 0
    signals = [
        json.loads(line)
        for line in (metadata / "state_signal_rows.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert signals
    assert any(row["static_ppe_signal"] for row in signals)
    assert all(row["state_label"] == "unknown" for row in signals)


def test_state_signal_rows_capture_operator_and_interaction_signals(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("KEY_ACTION_EXPERIMENT_STATE_NO_ACTIVITY_END_TIMEOUT_SEC", "20")
    manifest = _manifest(tmp_path)
    metadata = tmp_path / "metadata"
    rows = [
        _row("first_person", 0.0, {"lab_coat": 1}),
        _row("first_person", 10.0, {"gloved_hand": 1, "paper": 1}, interaction=True),
        _row("third_person", 10.0, {"gloved_hand": 1, "paper": 1}, interaction=True),
        _row("first_person", 40.0, {}),
        _row("third_person", 40.0, {}),
    ]

    build_experiment_state_artifacts(
        metadata,
        manifest,
        rows,
        [],
        min_duration_sec=5.0,
    )

    signals = [
        json.loads(line)
        for line in (metadata / "state_signal_rows.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(row["lab_coat_signal"] and row["state_label"] == "preparing" for row in signals)
    assert any(row["object_interaction_signal"] and row["state_label"] == "active_experiment" for row in signals)
    assert {"bin_id", "bin_start_global_timestamp_us", "first_activity_score", "third_activity_score"}.issubset(
        signals[0]
    )
    assert any(row["first_has_object_interaction"] for row in signals)
    assert any(row["third_bench_activity"] for row in signals)
    assert all("missing_detector_capabilities" in row for row in signals)
    assert (metadata / "state_signal_algorithm_report.json").exists()


def test_first_person_off_bench_activity_is_continuity_signal(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("KEY_ACTION_EXPERIMENT_STATE_NO_ACTIVITY_END_TIMEOUT_SEC", "20")
    manifest = _manifest(tmp_path)
    metadata = tmp_path / "metadata"
    rows = [
        _row("first_person", 0.0, {"gloved_hand": 1, "paper": 1}, interaction=True),
        _row("third_person", 0.0, {"gloved_hand": 1, "paper": 1}, interaction=True),
        _row("first_person", 10.0, {"hand": 1}),
        _row("first_person", 20.0, {"hand": 1}),
        _row("first_person", 30.0, {}),
        _row("third_person", 30.0, {}),
    ]

    result = build_experiment_state_artifacts(
        metadata,
        manifest,
        rows,
        [],
        min_duration_sec=5.0,
    )

    assert result["summary"]["candidate_window_count"] == 1
    formal = json.loads((metadata / "formal_experiment_windows.json").read_text(encoding="utf-8"))
    window = formal["windows"][0]
    assert window["end_sec"] >= 20.0
    signals = [
        json.loads(line)
        for line in (metadata / "state_signal_rows.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    off_bench = [row for row in signals if row.get("first_off_bench_experiment_related")]
    assert off_bench
    assert all(row["state_label"] == "off_table_but_experiment_related" for row in off_bench)
    trace = (metadata / "experiment_window_state_trace.jsonl").read_text(encoding="utf-8")
    assert "EXPERIMENT_ACTIVE_OFF_BENCH" in trace


def test_chunk_manifest_uses_aligned_video_and_sync_index(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    summary = write_chunk_manifest(
        tmp_path / "metadata",
        manifest,
        [
            {
                "view": "third_person",
                "chunk_index": 0,
                "chunk_start_sec": 10.0,
                "chunk_end_sec": 40.0,
            }
        ],
        sample_fps=0.2,
    )

    assert summary["chunk_count"] == 1
    rows = (tmp_path / "metadata" / "chunk_manifest.jsonl").read_text(encoding="utf-8").splitlines()
    row = json.loads(rows[0])
    assert row["aligned_video_path"].endswith("aligned_third.mp4")
    assert row["raw_video_path"] == "raw_third.mp4"
    assert row["start_sync_index"] == 10
    assert row["end_sync_index"] == 40


def test_sync_index_lookup_accepts_utf8_sig_header(tmp_path: Path) -> None:
    path = tmp_path / "sync_index.csv"
    path.write_text(
        "\ufeffsync_index,unit_id,global_timestamp_us,is_valid_pair,third_frame_index,first_frame_index,delta_ms,sync_quality\n"
        "7,unit_001,7000000,true,70,71,12.5,good\n",
        encoding="utf-8",
    )

    lookup = SyncIndexLookup.load(path)
    sync = lookup.at_sec(0.0)

    assert sync["sync_index"] == 7
    assert sync["unit_id"] == "unit_001"
    assert sync["third_frame_index"] == 70


def test_material_stream_groups_dual_view_keyframes_and_keyclips(tmp_path: Path) -> None:
    root = tmp_path / "material_references"
    for folder in ("keyframe", "keyclip"):
        (root / folder).mkdir(parents=True)
    rows = []
    for view in ("first_person", "third_person"):
        for kind in ("keyframe", "keyclip"):
            stored = root / kind / f"{view}_{kind}.dat"
            stored.write_bytes(b"x")
            rows.append(
                {
                    "reference_id": f"{view}_{kind}",
                    "physical_action_material_id": "dual_event_001",
                    "dual_event_id": "dual_event_001",
                    "asset_kind": kind,
                    "view": view,
                    "stored_file": str(stored),
                    "primary_object": "paper",
                    "formal_publish_gate": {"status": "passed"},
                }
            )

    stream = _write_material_stream(root, rows, session_root=tmp_path)

    assert len(stream) == 1
    item = stream[0]
    assert item["first_keyframe"].endswith("first_person_keyframe.dat")
    assert item["third_keyclip"].endswith("third_person_keyclip.dat")
    assert item["cli_ready_folder"] == str(root)
    assert (root / "material_stream.jsonl").exists()

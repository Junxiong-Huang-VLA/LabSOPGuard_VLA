from __future__ import annotations

import json
from pathlib import Path

import pytest

from key_action_indexer.pipeline import _apply_aligned_videos_to_manifest, _ensure_pre_coarse_timeline_alignment, _local_scan_bounds_for_common_overlap
from key_action_indexer.schemas import SessionManifest, SessionVideos, VideoSource


def _manifest_with_unreliable_dual_view() -> SessionManifest:
    return SessionManifest(
        session_id="timeline_gate",
        session_start_time="2026-05-20T09:52:24+08:00",
        videos=SessionVideos(
            third_person=VideoSource(
                name="third_person",
                path="third.mp4",
                start_time="2026-05-20T09:52:24+08:00",
                duration_sec=120.0,
            ),
            first_person=VideoSource(
                name="first_person",
                path="first.mp4",
                start_time="2026-05-20T09:52:24+08:00",
                duration_sec=80.0,
            ),
        ),
        config={
            "timeline_alignment": {
                "status": "capture_start_common_timeline",
                "alignment_reliable_for_dual_view_pairing": False,
                "reliability_reasons": ["missing_capture_metadata_duration_mismatch"],
            }
        },
    )


def test_pre_coarse_gate_respects_explicit_unreliable_alignment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("KEY_ACTION_REQUIRE_RELIABLE_DUAL_VIEW_ALIGNMENT", "1")
    monkeypatch.setattr(
        "key_action_indexer.pipeline._coarse_scan_source_duration_sec",
        lambda source, dry_run=False: float(source.duration_sec or 0.0),
    )
    paths = {"metadata": tmp_path / "metadata"}
    paths["metadata"].mkdir()

    with pytest.raises(RuntimeError, match="timeline alignment is not reliable"):
        _ensure_pre_coarse_timeline_alignment(
            _manifest_with_unreliable_dual_view(),
            paths,
            dry_run=False,
        )

    payload = json.loads((paths["metadata"] / "pre_coarse_timeline_alignment.json").read_text(encoding="utf-8"))
    assert payload["alignment_reliable_for_dual_view_pairing"] is False
    assert payload["gate_required"] is True
    assert payload["gate_passed"] is False


def test_pre_coarse_common_overlap_is_strict_view_interval_intersection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("KEY_ACTION_REQUIRE_DUAL_VIEW_FRAME_ALIGNMENT", "0")
    monkeypatch.setattr(
        "key_action_indexer.pipeline._coarse_scan_source_duration_sec",
        lambda source, dry_run=False: float(source.duration_sec or 0.0),
    )
    paths = {"metadata": tmp_path / "metadata"}
    paths["metadata"].mkdir()
    (paths["metadata"] / "timeline_alignment.json").write_text(
        json.dumps(
            {
                "alignment_status": "shared_recording",
                "alignment_reliable_for_dual_view_pairing": True,
                "common_overlap": {
                    "global_start_sec": 0.0,
                    "global_end_sec": 7501.0,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manifest = SessionManifest(
        session_id="strict_overlap",
        session_start_time="2026-05-20T09:52:24+08:00",
        videos=SessionVideos(
            third_person=VideoSource(
                name="third_person",
                path="third.mp4",
                start_time="2026-05-20T09:52:24+08:00",
                duration_sec=6349.512,
                frames_csv_path="third_frames.csv",
            ),
            first_person=VideoSource(
                name="first_person",
                path="first.mp4",
                start_time="2026-05-20T09:52:24+08:00",
                duration_sec=5737.247,
                frames_csv_path="first_frames.csv",
            ),
        ),
        output_dir=str(tmp_path / "session"),
    )

    payload = _ensure_pre_coarse_timeline_alignment(manifest, paths, dry_run=False)

    assert payload["alignment_reliable_for_dual_view_pairing"] is True
    assert payload["common_overlap"]["global_start_sec"] == pytest.approx(0.0)
    assert payload["common_overlap"]["global_end_sec"] == pytest.approx(5737.247)
    assert payload["common_overlap_end_sec"] == pytest.approx(5737.247)
    assert payload["common_overlap"]["requested_common_overlap"]["global_end_sec"] == pytest.approx(7501.0)
    assert payload["common_overlap"]["requested_common_overlap_clamped"] is True
    assert payload["views"]["first_person"]["global_end"] == pytest.approx(5737.247)
    assert payload["views"]["third_person"]["global_end"] == pytest.approx(6349.512)
    assert manifest.config["timeline_alignment"]["views"]["first_person"]["global_end"] == pytest.approx(5737.247)


def test_common_overlap_scan_bounds_use_frames_csv_map_for_video_pts(tmp_path: Path) -> None:
    video = tmp_path / "third.mp4"
    video.write_bytes(b"placeholder")
    frames = tmp_path / "frames.csv"
    frames.write_text(
        "\n".join(
            [
                "local_time_us,stream_type,frame_id",
                "1000000,rgb,0",
                "11000000,rgb,1",
                "51000000,rgb,2",
                "91000000,rgb,3",
                "121000000,rgb,4",
            ]
        ),
        encoding="utf-8",
    )
    source = VideoSource(
        name="third_person",
        path=str(video),
        start_time="2026-05-20T09:52:24+08:00",
        duration_sec=4.0,
        fps=30.0,
        frames_csv_path=str(frames),
    )
    manifest = SessionManifest(
        session_id="mapped_overlap",
        session_start_time="2026-05-20T09:52:24+08:00",
        videos=SessionVideos(third_person=source),
    )

    start, end, applied = _local_scan_bounds_for_common_overlap(
        manifest,
        source,
        4.0,
        {"available": True, "global_start_sec": 50.0, "global_end_sec": 90.0},
    )

    assert applied is True
    assert start == pytest.approx(2.0)
    assert end == pytest.approx(3.0)


def test_common_overlap_scan_bounds_clamp_mapped_pts_to_video_duration(tmp_path: Path) -> None:
    video = tmp_path / "third.mp4"
    video.write_bytes(b"placeholder")
    frames = tmp_path / "frames.csv"
    frames.write_text(
        "\n".join(
            [
                "local_time_us,stream_type,frame_id",
                "1000000,rgb,0",
                "11000000,rgb,1",
                "51000000,rgb,2",
                "91000000,rgb,3",
                "121000000,rgb,4",
            ]
        ),
        encoding="utf-8",
    )
    source = VideoSource(
        name="third_person",
        path=str(video),
        start_time="2026-05-20T09:52:24+08:00",
        duration_sec=4.0,
        fps=30.0,
        frames_csv_path=str(frames),
    )
    manifest = SessionManifest(
        session_id="mapped_overlap_clamped",
        session_start_time="2026-05-20T09:52:24+08:00",
        videos=SessionVideos(third_person=source),
    )

    start, end, applied = _local_scan_bounds_for_common_overlap(
        manifest,
        source,
        4.0,
        {"available": True, "global_start_sec": 90.0, "global_end_sec": 200.0},
    )

    assert applied is True
    assert start == pytest.approx(3.0)
    assert end == pytest.approx(4.0)


def test_apply_aligned_videos_to_manifest_replaces_downstream_sources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw_third = tmp_path / "raw_third.mp4"
    raw_first = tmp_path / "raw_first.mp4"
    aligned_third = tmp_path / "aligned_third.mp4"
    aligned_first = tmp_path / "aligned_first.mp4"
    for path in (raw_third, raw_first, aligned_third, aligned_first):
        path.write_bytes(b"video")
    manifest = SessionManifest(
        session_id="aligned_manifest",
        session_start_time="2026-05-20T09:52:24+08:00",
        videos=SessionVideos(
            third_person=VideoSource(
                name="third_person",
                path=str(raw_third),
                start_time="2026-05-20T09:52:24+08:00",
                frames_csv_path=str(tmp_path / "third_frames.csv"),
            ),
            first_person=VideoSource(
                name="first_person",
                path=str(raw_first),
                start_time="2026-05-20T09:52:24+08:00",
                frames_csv_path=str(tmp_path / "first_frames.csv"),
            ),
        ),
    )
    monkeypatch.setattr("key_action_indexer.pipeline.get_video_duration_sec", lambda path: 12.5)

    applied = _apply_aligned_videos_to_manifest(
        manifest,
        {
            "aligned_video_outputs": {
                "aligned_third_video": str(aligned_third),
                "aligned_first_video": str(aligned_first),
                "aligned_side_by_side_video": str(tmp_path / "side.mp4"),
            },
            "artifacts": {"sync_index_csv": str(tmp_path / "sync_index.csv")},
        },
    )

    assert applied is True
    assert manifest.videos.third_person.path == str(aligned_third)
    assert manifest.videos.first_person.path == str(aligned_first)
    assert manifest.videos.third_person.frames_csv_path is None
    assert manifest.videos.first_person.frames_csv_path is None
    assert manifest.videos.third_person.duration_sec == pytest.approx(12.5)
    assert manifest.config["aligned_video_analysis"]["raw_sources"]["third_person"]["raw_video_path"] == str(raw_third)

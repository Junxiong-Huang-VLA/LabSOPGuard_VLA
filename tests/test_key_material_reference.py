from __future__ import annotations

from pathlib import Path

from labsopguard.key_material_reference import (
    load_key_material_references,
    query_step_materials,
    write_experiment_reference_outputs,
)


def test_reference_outputs_folder_and_step_judgement(tmp_path: Path) -> None:
    exp_dir = tmp_path / "outputs" / "experiments" / "exp_001"
    event_dir = exp_dir / "materials" / "events" / "evt_001"
    event_dir.mkdir(parents=True)
    (event_dir / "clip.mp4").write_bytes(b"fake video bytes")
    (event_dir / "preview.jpg").write_bytes(b"fake jpg bytes")

    experiment_record = {
        "experiment_id": "exp_001",
        "title": "固体称量实验_20260508",
        "session_start_time": "2026-05-12T14:20:00+08:00",
        "timezone": "Asia/Shanghai",
    }
    preprocessing = {
        "video_streams": [
            {
                "video_index": 0,
                "video_asset_id": "video_0",
                "file_path": "source.mp4",
                "duration_sec": 1800.0,
                "start_offset_sec": 0.0,
                "end_offset_sec": 1800.0,
                "clock_scale": 1.0,
                "offset_source": "explicit",
            }
        ],
        "physical_events": [
            {
                "event_id": "evt_001",
                "event_type": "object_move",
                "display_name": "试剂瓶移动",
                "start_time_sec": 720.0,
                "end_time_sec": 735.0,
                "key_timestamps": [720.0, 728.0, 735.0],
                "involved_objects": ["gloved_hand", "reagent_bottle"],
                "state_before": {"zone": "A", "centroid": [120, 220]},
                "state_after": {"zone": "B", "centroid": [420, 220]},
                "asset_pack": {
                    "clip_path": "materials/events/evt_001/clip.mp4",
                    "preview_path": "materials/events/evt_001/preview.jpg",
                    "keyframe_paths": ["materials/events/evt_001/preview.jpg"],
                },
                "confidence": 0.86,
                "evidence_grade": "B",
                "review_status": "candidate",
            }
        ],
        "key_frames": [],
        "key_clips": [],
    }
    steps = [
        {
            "step_id": "solid_weighing.return_bottle",
            "step_name": "试剂瓶归位",
            "start_time_sec": 700.0,
            "end_time_sec": 760.0,
        }
    ]

    outputs = write_experiment_reference_outputs(
        experiment_dir=exp_dir,
        experiment_record=experiment_record,
        material_stream=[],
        preprocessing=preprocessing,
        steps=steps,
        segmentation={"segments": [{"segment_id": "seg_0", "index": 0, "start_sec": 0.0, "end_sec": 1800.0}]},
        formal_library_root=tmp_path / "outputs" / "material_references",
    )

    refs_path = exp_dir / "artifacts" / "key_material_references.jsonl"
    changes_path = exp_dir / "artifacts" / "physical_change_log.jsonl"
    assert refs_path.exists()
    assert (exp_dir / "artifacts" / "key_material_references.sqlite").exists()
    assert changes_path.exists()
    assert outputs["manifest"]["reference_count"] >= 1

    refs = load_key_material_references(exp_dir)
    assert refs[0]["step_name"] == "试剂瓶归位"
    assert refs[0]["clip_path"] == "materials/events/evt_001/clip.mp4"

    package_dir = tmp_path / "outputs" / "material_references" / "固体称量实验_20260508"
    assert (package_dir / "关键片段").exists()
    assert (package_dir / "关键帧").exists()
    assert (package_dir / "素材索引.json").exists()
    assert (package_dir / "key_material_references.jsonl").exists()
    assert (package_dir / "key_material_references.sqlite").exists()

    result = query_step_materials(
        experiment_dir=exp_dir,
        step_text="检查试剂瓶归位是否正确",
        message_sent_at="2026-05-12T14:32:05+08:00",
        limit=5,
    ).to_dict()
    assert result["message_video_time_sec"] == 725.0
    assert result["candidates"]
    assert result["judgement"]["status"] == "incorrect"


def test_step_query_reads_formal_material_reference_package(tmp_path: Path) -> None:
    exp_dir = tmp_path / "outputs" / "experiments" / "exp_formal"
    package_dir = exp_dir / "material_references"
    package_dir.mkdir(parents=True)
    artifacts_dir = exp_dir / "artifacts"
    artifacts_dir.mkdir()
    (artifacts_dir / "key_material_references.jsonl").write_text("", encoding="utf-8")
    (exp_dir / "experiment.json").write_text(
        '{"experiment_id":"exp_formal","title":"Formal material package"}',
        encoding="utf-8",
    )
    (package_dir / "time_alignment.json").write_text(
        '{"session_start_at":"2026-05-12T06:00:00+00:00","message_alignment_policy":{"default_window_before_sec":10,"default_window_after_sec":10}}',
        encoding="utf-8",
    )
    (package_dir / "key_material_references.jsonl").write_text(
        (
            '{"schema_version":"key_material_reference.v1","material_id":"mat_paper",'
            '"asset_type":"event_clip","asset_kind":"key_clip","start_sec":5,"end_sec":8,'
            '"confidence":0.82,"canonical_action_type":"hand-paper","canonical_object":"paper",'
            '"searchable_text":"hand paper contact key evidence","clip_path":"material_references/key_clip/paper.mp4"}\n'
        ),
        encoding="utf-8",
    )

    refs = load_key_material_references(exp_dir)
    assert [row["material_id"] for row in refs] == ["mat_paper"]

    result = query_step_materials(
        experiment_dir=exp_dir,
        step_text="paper contact",
        message_sent_at="2026-05-12T06:00:06+00:00",
        limit=3,
    ).to_dict()
    assert result["message_video_time_sec"] == 6.0
    assert result["candidates"][0]["material_id"] == "mat_paper"
    assert result["judgement"]["status"] == "correct"

    mixed_result = query_step_materials(
        experiment_dir=exp_dir,
        step_text="check hand-paper operation",
        limit=3,
    ).to_dict()
    assert mixed_result["candidates"][0]["material_id"] == "mat_paper"

    chinese_result = query_step_materials(
        experiment_dir=exp_dir,
        step_text="\u68c0\u67e5\u79f0\u91cf\u7eb8\u662f\u5426\u653e\u597d",
        limit=3,
    ).to_dict()
    assert chinese_result["candidates"][0]["material_id"] == "mat_paper"

    missing_target_result = query_step_materials(
        experiment_dir=exp_dir,
        step_text="\u68c0\u67e5\u79f0\u91cf\u7eb8\u662f\u5426\u653e\u5230\u5929\u5e73\u4e0a",
        limit=3,
    ).to_dict()
    assert missing_target_result["judgement"]["status"] == "insufficient"
    assert missing_target_result["judgement"]["missing_objects"] == ["balance"]

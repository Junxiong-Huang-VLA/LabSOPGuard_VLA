from __future__ import annotations

import json
from pathlib import Path

from key_action_indexer import material_references
from key_action_indexer.material_references import (
    MATERIAL_CANDIDATE_INDEX_BASENAME,
    KEYFRAME_DIR_NAME,
    KEY_CLIP_DIR_NAME,
    MATERIAL_INDEX_BASENAME,
    approve_material_candidates,
    build_yolo_material_candidates,
    build_yolo_material_references,
    formal_material_references_root,
    material_candidates_root,
    material_references_root,
    reset_material_references_to_approved_candidates,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _valid_evidence(view: str, *, primary_object: str = "balance", score: float = 0.78) -> list[dict]:
    times = [46.433667, 47.2, 49.133666]
    evidence: list[dict] = []
    for index, local_time in enumerate(times):
        shift = index * 3
        hand_bbox = [110 + shift, 120, 190 + shift, 230]
        object_bbox = [165 + shift, 155, 335 + shift, 315]
        evidence.append(
            {
                "view": view,
                "local_time_sec": local_time,
                "interaction_score": score,
                "detections": [
                    {"label": "gloved_hand", "confidence": 0.74, "bbox": hand_bbox},
                    {"label": primary_object, "confidence": 0.72, "bbox": object_bbox},
                ],
                "hand_object_interactions": [
                    {
                        "hand_label": "gloved_hand",
                        "object_label": primary_object,
                        "score": score,
                        "hand_bbox": hand_bbox,
                        "object_bbox": object_bbox,
                        "iou": 0.08,
                        "distance_px": 24.0,
                    }
                ],
            }
        )
    return evidence


def _false_balance_evidence() -> list[dict]:
    return [
        {
            "view": "first_person",
            "local_time_sec": 61.866667 + index * 0.2,
            "interaction_score": 0.70,
            "detections": [
                {"label": "gloved_hand", "confidence": 0.62, "bbox": [366, 73, 542, 140]},
                {"label": "balance", "confidence": 0.31, "bbox": [342, 82, 562, 369]},
            ],
            "hand_object_interactions": [
                {
                    "hand_label": "gloved_hand",
                    "object_label": "balance",
                    "score": 0.70,
                    "hand_bbox": [366, 73, 542, 140],
                    "object_bbox": [342, 82, 562, 369],
                }
            ],
        }
        for index in range(3)
    ]


def _interaction_proxy_evidence(view: str, *, primary_object: str = "reagent_bottle", score: float = 0.92) -> list[dict]:
    times = [46.433667, 47.2, 49.133666]
    evidence: list[dict] = []
    for index, local_time in enumerate(times):
        shift = index * 3
        hand_bbox = [595 + shift, 178, 682 + shift, 292]
        object_bbox = [603 + shift, 228, 675 + shift, 274]
        evidence.append(
            {
                "view": view,
                "local_time_sec": local_time,
                "interaction_score": score,
                "detections": [
                    {"label": "gloved_hand", "confidence": 0.86, "bbox": hand_bbox},
                    {"label": "pipette", "confidence": 0.99, "bbox": [753, 347, 842, 438]},
                    {"label": "balance", "confidence": 0.98, "bbox": [393, 146, 566, 366]},
                ],
                "hand_object_interactions": [
                    {
                        "hand_label": "gloved_hand",
                        "object_label": primary_object,
                        "score": score,
                        "hand_bbox": hand_bbox,
                        "object_bbox": object_bbox,
                        "iou": 0.39,
                        "distance_px": 22.0,
                    }
                ],
            }
        )
    return evidence


def _session_with_one_yolo_micro(
    tmp_path: Path,
    *,
    create_source: bool = True,
    evidence: list[dict] | None = None,
    primary_object: str = "balance",
) -> Path:
    session = tmp_path / "experiment" / "key_action_index"
    metadata = session / "metadata"
    session.parent.mkdir(parents=True, exist_ok=True)
    (session.parent / "experiment.json").write_text(
        json.dumps(
            {
                "experiment_id": "titration_title_case_20260506_abcd1234",
                "title": "\u6ef4\u5b9a\u5b9e\u9a8cA",
                "created_at": "2026-05-06T12:34:56+08:00",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    annotated_first = session / "clips" / "seg_000001" / "first_person_yolo_annotated.mp4"
    annotated_third = session / "clips" / "seg_000001" / "third_person_yolo_annotated.mp4"
    if create_source:
        annotated_first.parent.mkdir(parents=True, exist_ok=True)
        annotated_first.write_bytes(b"first-yolo")
        annotated_third.write_bytes(b"third-yolo")
    _write_jsonl(
        metadata / "key_action_segments.jsonl",
        [
            {
                "segment_id": "seg_000001",
                "first_person": {
                    "local_start_sec": 16.0,
                    "annotated_clip_path": str(annotated_first),
                },
                "third_person": {
                    "local_start_sec": 16.0,
                    "annotated_clip_path": str(annotated_third),
                },
            }
        ],
    )
    _write_jsonl(
        metadata / "micro_segments.jsonl",
        [
            {
                "micro_segment_id": "seg_000001_micro_004",
                "parent_segment_id": "seg_000001",
                "start_sec": 46.433667,
                "end_sec": 49.133666,
                "global_start_time": "2026-04-24T16:58:04.433667+08:00",
                "global_end_time": "2026-04-24T16:58:07.133666+08:00",
                "interaction": {"primary_object": primary_object},
                "yolo_evidence": evidence
                if evidence is not None
                else _valid_evidence("third_person", primary_object=primary_object)
                + _valid_evidence("first_person", primary_object=primary_object),
            },
            {
                "micro_segment_id": "seg_000001_part02_micro_001",
                "parent_segment_id": "seg_000001_part02",
                "start_sec": 1.0,
                "end_sec": 2.0,
                "interaction": {"primary_object": "balance"},
                "yolo_evidence": _valid_evidence("third_person"),
            },
            {
                "micro_segment_id": "seg_000001_micro_no_yolo",
                "parent_segment_id": "seg_000001",
                "start_sec": 50.0,
                "end_sec": 51.0,
                "interaction": {"primary_object": "spatula"},
                "yolo_evidence": [],
            },
        ],
    )
    return session


def test_build_yolo_material_references_creates_yolo_only_named_files(tmp_path: Path, monkeypatch) -> None:
    session = _session_with_one_yolo_micro(tmp_path)

    monkeypatch.setattr(material_references, "_ffmpeg_available", lambda _path: True)

    def fake_ffmpeg(args: list[str]) -> None:
        target = Path(args[-1])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"generated-yolo-material")

    monkeypatch.setattr(material_references, "_run_ffmpeg", fake_ffmpeg)

    summary = build_yolo_material_references(session)

    ref_root = material_references_root(session)
    keyframes = sorted((ref_root / KEYFRAME_DIR_NAME).glob("*"))
    clips = sorted((ref_root / KEY_CLIP_DIR_NAME).glob("*"))
    assert summary["file_count"] == 10
    assert summary["experiment_id"] == "titration_title_case_20260506_abcd1234"
    assert summary["experiment_title"] == "\u6ef4\u5b9a\u5b9e\u9a8cA"
    assert summary["experiment_date"] == "20260506"
    assert Path(summary["formal_material_references"]).name == "\u6ef4\u5b9a\u5b9e\u9a8cA_20260506"
    assert Path(summary["simplified_material_references"]).name == "\u6ef4\u5b9a\u5b9e\u9a8cA_20260506"
    assert not formal_material_references_root(session).exists()
    assert len(keyframes) == 6
    assert len(clips) == 4
    assert all("\u624b\u4e0e\u5929\u5e73\u64cd\u4f5c" in item.name for item in keyframes + clips)
    assert all("20260424" in item.name for item in keyframes + clips)
    assert not any("YOLO" in item.name for item in keyframes + clips)
    assert not any("46.4-49.1s" in item.name or "\u7b2c\u4e00\u4eba\u79f0" in item.name or "\u7b2c\u4e09\u4eba\u79f0" in item.name for item in keyframes + clips)
    assert not any("\u63a5\u89e6\u5e27" in item.name or "\u5cf0\u503c\u5e27" in item.name or "\u91ca\u653e\u5e27" in item.name for item in keyframes)
    assert not any("part02" in item.name or "\u5b8c\u6574\u5b9e\u9a8c\u7247\u6bb5" in item.name for item in keyframes + clips)

    index_rows = [
        json.loads(line)
        for line in (ref_root / f"{MATERIAL_INDEX_BASENAME}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(index_rows) == 10
    assert all(row["yolo_box_required"] is True for row in index_rows)
    assert all(row["box_filter"] == "hand_and_primary_object_only" for row in index_rows)
    assert {row["frame_role"] for row in index_rows if row["asset_kind"] == KEYFRAME_DIR_NAME} == {"contact", "peak", "release"}


def test_build_yolo_material_references_clears_generated_files_without_archive(tmp_path: Path, monkeypatch) -> None:
    session = _session_with_one_yolo_micro(tmp_path)
    ref_root = material_references_root(session)
    stale_keyframe = ref_root / KEYFRAME_DIR_NAME / "stale.jpg"
    stale_clip = ref_root / KEY_CLIP_DIR_NAME / "stale.mp4"
    stale_keyframe.parent.mkdir(parents=True, exist_ok=True)
    stale_clip.parent.mkdir(parents=True, exist_ok=True)
    stale_keyframe.write_bytes(b"stale-keyframe")
    stale_clip.write_bytes(b"stale-clip")

    monkeypatch.setattr(material_references, "_ffmpeg_available", lambda _path: True)
    monkeypatch.setattr(material_references, "_run_ffmpeg", lambda args: Path(args[-1]).write_bytes(b"material"))

    summary = build_yolo_material_references(session, archive_existing=False)

    assert summary["archived_count"] == 0
    assert not stale_keyframe.exists()
    assert not stale_clip.exists()
    assert not any(path.name.startswith("stale") for path in (ref_root / KEYFRAME_DIR_NAME).glob("*"))
    assert not any(path.name.startswith("stale") for path in (ref_root / KEY_CLIP_DIR_NAME).glob("*"))


def test_build_yolo_material_references_requires_valid_evidence_per_view(tmp_path: Path, monkeypatch) -> None:
    session = _session_with_one_yolo_micro(tmp_path, evidence=_valid_evidence("third_person"))
    monkeypatch.setattr(material_references, "_ffmpeg_available", lambda _path: True)
    monkeypatch.setattr(material_references, "_run_ffmpeg", lambda args: Path(args[-1]).write_bytes(b"material"))

    summary = build_yolo_material_references(session)

    ref_root = material_references_root(session)
    index_rows = [
        json.loads(line)
        for line in (ref_root / f"{MATERIAL_INDEX_BASENAME}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert summary["file_count"] == 5
    assert {row["view"] for row in index_rows} == {"third_person"}
    assert any(item.get("reason") == "no_valid_yolo_physical_evidence_for_view" for item in summary["skipped"])


def test_build_yolo_material_references_rejects_false_physical_evidence(tmp_path: Path, monkeypatch) -> None:
    session = _session_with_one_yolo_micro(tmp_path, evidence=_false_balance_evidence())
    monkeypatch.setattr(material_references, "_ffmpeg_available", lambda _path: True)
    monkeypatch.setattr(material_references, "_run_ffmpeg", lambda args: Path(args[-1]).write_bytes(b"material"))

    summary = build_yolo_material_references(session)

    assert summary["file_count"] == 0
    assert summary["planned_file_count"] == 0
    assert any(item.get("reason") == "no_valid_yolo_physical_evidence" for item in summary["skipped"])
    skipped = next(item for item in summary["skipped"] if item.get("reason") == "no_valid_yolo_physical_evidence")
    assert "primary_object_confidence_below_threshold" in skipped["diagnostics"]["invalid_reason_counts"]


def test_build_yolo_material_references_accepts_stable_interaction_object_proxy(tmp_path: Path, monkeypatch) -> None:
    session = _session_with_one_yolo_micro(
        tmp_path,
        evidence=_interaction_proxy_evidence("third_person"),
        primary_object="reagent_bottle",
    )
    monkeypatch.setattr(material_references, "_ffmpeg_available", lambda _path: True)
    monkeypatch.setattr(material_references, "_run_ffmpeg", lambda args: Path(args[-1]).write_bytes(b"material"))

    summary = build_yolo_material_references(session)

    ref_root = material_references_root(session)
    index_rows = [
        json.loads(line)
        for line in (ref_root / f"{MATERIAL_INDEX_BASENAME}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert summary["file_count"] == 5
    assert {row["primary_object"] for row in index_rows} == {"reagent_bottle"}
    assert all("\u624b\u4e0e\u8bd5\u5242\u74f6\u64cd\u4f5c" in row["stored_filename"] for row in index_rows)


def test_filtered_interaction_detections_keep_only_hand_and_primary_object() -> None:
    evidence = _valid_evidence("third_person")[0]
    evidence["detections"].append({"label": "beaker", "confidence": 0.99, "bbox": [1, 2, 30, 40]})

    filtered = material_references._filtered_interaction_detections(evidence, "balance")

    labels = {item["label"] for item, _color in filtered}
    assert labels == {"gloved_hand", "balance"}


def test_filtered_interaction_detections_recomputes_cached_false_hand_interaction() -> None:
    import numpy as np

    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    evidence = {
        "view": "third_person",
        "local_time_sec": 44.533333,
        "detections": [
            {"label": "gloved_hand", "confidence": 0.51, "bbox": [2, 212, 151, 354]},
            {"label": "container", "confidence": 0.76, "bbox": [0, 154, 90, 285]},
        ],
        "hand_object_interactions": [
            {
                "hand_label": "gloved_hand",
                "object_label": "container",
                "score": 0.94,
                "hand_bbox": [2, 212, 151, 354],
                "object_bbox": [0, 154, 90, 285],
            }
        ],
    }

    filtered = material_references._filtered_interaction_detections(evidence, "container", frame=frame)

    assert filtered == []


def test_source_clip_for_view_prefers_unannotated_segment_clip(tmp_path: Path) -> None:
    session = tmp_path / "experiment" / "key_action_index"
    raw_clip = session / "clips" / "seg_000001" / "third_person.mp4"
    annotated_clip = session / "clips" / "seg_000001" / "third_person_yolo_annotated.mp4"
    raw_clip.parent.mkdir(parents=True, exist_ok=True)
    raw_clip.write_bytes(b"raw")
    annotated_clip.write_bytes(b"annotated")
    segment = {
        "segment_id": "seg_000001",
        "third_person": {
            "clip_path": str(raw_clip),
            "annotated_clip_path": str(annotated_clip),
        },
    }

    source = material_references._source_clip_for_view(
        session,
        {},
        segment,
        {"parent_segment_id": "seg_000001"},
        "third_person",
    )

    assert source == raw_clip


def test_build_yolo_material_references_dry_run_without_video_or_ffmpeg(tmp_path: Path, monkeypatch) -> None:
    session = _session_with_one_yolo_micro(tmp_path, create_source=False)
    monkeypatch.setattr(material_references, "_ffmpeg_available", lambda _path: False)

    summary = build_yolo_material_references(session, dry_run=True)

    ref_root = material_references_root(session)
    assert (ref_root / KEYFRAME_DIR_NAME).exists()
    assert (ref_root / KEY_CLIP_DIR_NAME).exists()
    assert summary["file_count"] == 0
    assert summary["planned_file_count"] == 10
    assert summary["experiment_label"] == "\u6ef4\u5b9a\u5b9e\u9a8cA_20260506"
    assert summary["dry_run"] is True
    assert summary["ffmpeg_available"] is False


def test_build_yolo_material_candidates_marks_best_files_for_review(tmp_path: Path, monkeypatch) -> None:
    session = _session_with_one_yolo_micro(tmp_path)
    monkeypatch.setattr(material_references, "_ffmpeg_available", lambda _path: True)
    monkeypatch.setattr(material_references, "_run_ffmpeg", lambda args: Path(args[-1]).write_bytes(b"material"))
    build_yolo_material_references(session)

    summary = build_yolo_material_candidates(session, archive_existing=False)

    candidate_root = material_candidates_root(session)
    rows = [
        json.loads(line)
        for line in (candidate_root / f"{MATERIAL_CANDIDATE_INDEX_BASENAME}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert summary["candidate_count"] == 10
    assert len(rows) == 10
    assert all(row["candidate_status"] == "pending" for row in rows)
    assert all(row["review_required"] is True for row in rows)
    assert all(row["box_filter"] == "hand_and_primary_object_only" for row in rows)
    assert {row["asset_kind"] for row in rows} == {KEYFRAME_DIR_NAME, KEY_CLIP_DIR_NAME}
    group_ids = {row["candidate_group_id"] for row in rows}
    assert len(group_ids) == 2
    for group_id in group_ids:
        group_rows = [row for row in rows if row["candidate_group_id"] == group_id]
        assert sum(1 for row in group_rows if row["recommended"] and row["asset_kind"] == KEYFRAME_DIR_NAME) == 1
        assert sum(1 for row in group_rows if row["recommended"] and row["asset_kind"] == KEY_CLIP_DIR_NAME) == 1


def test_approve_material_candidates_promotes_recommended_files_only(tmp_path: Path, monkeypatch) -> None:
    session = _session_with_one_yolo_micro(tmp_path)
    monkeypatch.setattr(material_references, "_ffmpeg_available", lambda _path: True)
    monkeypatch.setattr(material_references, "_run_ffmpeg", lambda args: Path(args[-1]).write_bytes(b"material"))
    build_yolo_material_references(session)
    build_yolo_material_candidates(session, archive_existing=False)
    reset_material_references_to_approved_candidates(session, approved_rows=[], merge_existing=False)

    candidate_root = material_candidates_root(session)
    candidate_rows = [
        json.loads(line)
        for line in (candidate_root / f"{MATERIAL_CANDIDATE_INDEX_BASENAME}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    group_id = candidate_rows[0]["candidate_group_id"]

    approval = approve_material_candidates(session, candidate_group_id=group_id, reviewer="tester")

    ref_root = material_references_root(session)
    reference_rows = [
        json.loads(line)
        for line in (ref_root / f"{MATERIAL_INDEX_BASENAME}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert approval["approved_count"] == 2
    assert len(reference_rows) == 2
    assert {row["asset_kind"] for row in reference_rows} == {KEYFRAME_DIR_NAME, KEY_CLIP_DIR_NAME}
    assert all(row["review_status"] == "accepted" for row in reference_rows)
    assert all(Path(row["stored_file"]).exists() for row in reference_rows)
    simplified_root = Path(approval["material_references_summary"]["simplified_material_references"])
    formal_root = formal_material_references_root(session)
    simplified_rows = [
        json.loads(line)
        for line in (simplified_root / f"{MATERIAL_INDEX_BASENAME}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert simplified_root.name == "\u6ef4\u5b9a\u5b9e\u9a8cA_20260506"
    assert Path(approval["material_references_summary"]["material_references"]) == formal_root
    assert Path(approval["material_references_summary"]["local_material_references_mirror"]) == ref_root
    assert {row["asset_kind"] for row in simplified_rows} == {KEYFRAME_DIR_NAME, KEY_CLIP_DIR_NAME}
    assert len(simplified_rows) == 2
    assert all(formal_root in Path(row["stored_file"]).parents for row in simplified_rows)
    assert len(list((simplified_root / KEYFRAME_DIR_NAME).glob("*"))) == 1
    assert len(list((simplified_root / KEY_CLIP_DIR_NAME).glob("*"))) == 1

    updated_candidates = [
        json.loads(line)
        for line in (candidate_root / f"{MATERIAL_CANDIDATE_INDEX_BASENAME}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    approved_ids = set(approval["approved_candidate_ids"])
    assert {row["candidate_status"] for row in updated_candidates if row["candidate_id"] in approved_ids} == {"approved"}
    assert "not_selected" in {row["candidate_status"] for row in updated_candidates if row["candidate_group_id"] == group_id and row["candidate_id"] not in approved_ids}
    candidate_manifest = json.loads((candidate_root / "manifest.json").read_text(encoding="utf-8"))
    candidate_summary = json.loads((candidate_root / f"{MATERIAL_CANDIDATE_INDEX_BASENAME}.json").read_text(encoding="utf-8"))
    expected_pending = sum(1 for row in updated_candidates if row["candidate_status"] == "pending")
    expected_pending_groups = {
        row["candidate_group_id"]
        for row in updated_candidates
        if row["candidate_status"] == "pending"
    }
    assert candidate_manifest["pending_total"] == expected_pending
    assert candidate_manifest["approved_total"] == 2
    assert candidate_manifest["not_selected_total"] >= 1
    assert candidate_manifest["pipeline_summary"]["groups_waiting_frontend_review"] == len(expected_pending_groups)
    assert candidate_summary["pending_total"] == candidate_manifest["pending_total"]

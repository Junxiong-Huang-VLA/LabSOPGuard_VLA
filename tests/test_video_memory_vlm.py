from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from key_action_indexer.material_library_store import sync_material_library
from key_action_indexer.video_memory import VLM_MODE_REAL_QWEN_ASYNC, VLM_MODE_REUSE_EXISTING, build_video_memory


class _FakeQwenClient:
    def __init__(self) -> None:
        self.item_calls = 0
        self.bundle_calls = 0
        self.models: list[str] = []

    async def enhance_video_memory_item(
        self,
        payload: dict[str, Any],
        *,
        model: str,
        prompt_version: str,
        task_type: str,
    ) -> dict[str, Any]:
        self.item_calls += 1
        self.models.append(model)
        return {
            "description": f"Qwen item support for {payload['material_id']}",
            "confirmed_objects": ["hand", "paper"],
            "semantic_action": "hand paper contact",
            "evidence_alignment": "aligned",
            "reason": "The provided evidence item supports hand-paper contact.",
            "confidence": 0.88,
            "model": model,
        }

    async def enhance_video_memory_bundle(
        self,
        payload: dict[str, Any],
        *,
        model: str,
        prompt_version: str,
        task_type: str,
    ) -> dict[str, Any]:
        self.bundle_calls += 1
        self.models.append(model)
        return {
            "merged_scene_understanding": f"Qwen bundle support for {payload['bundle_id']}",
            "merged_action_understanding": "dual-view hand paper contact",
            "view_agreement": "dual_view_supported",
            "strong_facts": ["Both material items support the same hand-paper action."],
            "confidence": 0.91,
            "model": model,
        }


def test_video_memory_reuses_material_vlm_sources_without_calling_qwen(tmp_path: Path) -> None:
    library_root = tmp_path / "LabMaterialLibrary"
    _write_material_package(library_root, include_existing_vlm=True)
    sync_material_library(library_root, rebuild=True)
    client = _FakeQwenClient()

    result = build_video_memory(
        library_root=library_root,
        window_end_date="2026-05-25",
        vlm_mode=VLM_MODE_REAL_QWEN_ASYNC,
        item_vlm_model="qwen3.5-flash",
        bundle_vlm_model="qwen3.5-plus",
        vlm_client=client,
    )

    assert client.item_calls == 0
    assert client.bundle_calls == 0
    assert result["counts"]["materials"] == 2

    evidence_items = _read_jsonl(Path(result["memory_index_root"]) / "evidence_items.jsonl")
    assert evidence_items
    item = evidence_items[0]
    assert item["schema_version"] == "video_memory.evidence_item.v1"
    assert item["time_range"]["start_sec"] == 10
    assert item["action"]["primary_object"] == "paper"
    assert item["views"][0]["clip_uri"].startswith("package://") or item["views"][0]["keyframe_uri"].startswith("package://")
    assert item["micro_segment"]["micro_segment_id"] == "micro1"
    assert item["time_alignment"]["mode"] == "paired_view_time_alignment"
    assert {source["source_type"] for source in item["vlm_existing_sources"]} >= {
        "vlm_semantics",
        "qwen_event_audits",
        "advanced_vision_evidence",
    }

    item_results = _read_jsonl(Path(result["memory_index_root"]) / "vlm_item_results.jsonl")
    assert {row["vlm_source"] for row in item_results} == {"reuse_existing"}


def test_video_memory_qwen_item_and_bundle_cache_avoids_duplicate_fake_calls(tmp_path: Path) -> None:
    library_root = tmp_path / "LabMaterialLibrary"
    _write_material_package(library_root, include_existing_vlm=False)
    sync_material_library(library_root, rebuild=True)

    first_client = _FakeQwenClient()
    first = build_video_memory(
        library_root=library_root,
        window_end_date="2026-05-25",
        vlm_mode=VLM_MODE_REAL_QWEN_ASYNC,
        item_vlm_model="qwen3.5-flash",
        bundle_vlm_model="qwen3.5-plus",
        vlm_client=first_client,
    )
    assert first_client.item_calls == 2
    assert first_client.bundle_calls == 1
    assert set(first_client.models) == {"qwen3.5-flash", "qwen3.5-plus"}

    second_client = _FakeQwenClient()
    second = build_video_memory(
        library_root=library_root,
        window_end_date="2026-05-25",
        vlm_mode=VLM_MODE_REAL_QWEN_ASYNC,
        item_vlm_model="qwen3.5-flash",
        bundle_vlm_model="qwen3.5-plus",
        vlm_client=second_client,
    )
    assert second_client.item_calls == 0
    assert second_client.bundle_calls == 0
    assert second["job"]["vlm_cache_hit_count"] >= first["counts"]["item_vlm_results"] + first["counts"]["bundle_vlm_results"]

    conn = sqlite3.connect(first["sqlite_path"])
    try:
        cache_rows = conn.execute("SELECT COUNT(*) FROM vlm_result_cache").fetchone()[0]
        cache_payloads = [
            json.loads(row[0])
            for row in conn.execute("SELECT payload_json FROM vlm_result_cache").fetchall()
        ]
    finally:
        conn.close()
    assert cache_rows >= 3
    assert {row["cache_scope"] for row in cache_payloads} >= {"item", "bundle"}
    bundle_cache = [row for row in cache_payloads if row["cache_scope"] == "bundle"][0]
    item_cache = [row for row in cache_payloads if row["cache_scope"] == "item"][0]
    assert item_cache["material_id"]
    assert item_cache["asset_sha256"]
    assert bundle_cache["material_id"].startswith("bundle-materials:")
    assert bundle_cache["sha256s"]
    assert bundle_cache["result_json"]["material_ids"]


def test_reuse_existing_vlm_mode_reads_qwen_cache_without_new_calls(tmp_path: Path) -> None:
    library_root = tmp_path / "LabMaterialLibrary"
    _write_material_package(library_root, include_existing_vlm=False)
    sync_material_library(library_root, rebuild=True)

    first_client = _FakeQwenClient()
    first = build_video_memory(
        library_root=library_root,
        window_end_date="2026-05-25",
        vlm_mode=VLM_MODE_REAL_QWEN_ASYNC,
        item_vlm_model="qwen3.5-flash",
        bundle_vlm_model="qwen3.5-plus",
        vlm_client=first_client,
    )
    assert first_client.item_calls == 2
    assert first_client.bundle_calls == 1

    second_client = _FakeQwenClient()
    second = build_video_memory(
        library_root=library_root,
        window_end_date="2026-05-25",
        vlm_mode=VLM_MODE_REUSE_EXISTING,
        item_vlm_model="qwen3.5-flash",
        bundle_vlm_model="qwen3.5-plus",
        vlm_client=second_client,
    )

    assert second_client.item_calls == 0
    assert second_client.bundle_calls == 0
    assert second["job"]["vlm_qwen_cache_reuse_count"] >= first["counts"]["item_vlm_results"] + first["counts"]["bundle_vlm_results"]
    assert second["job"]["vlm_real_call_count"] == 0


def _write_material_package(library_root: Path, *, include_existing_vlm: bool) -> None:
    package_root = library_root / "material_references" / "pkg_20260525"
    keyframes = package_root / "keyframes"
    clips = package_root / "clips"
    keyframes.mkdir(parents=True)
    clips.mkdir(parents=True)
    (keyframes / "paper_third.jpg").write_bytes(b"fake-jpeg-third")
    (clips / "paper_first.mp4").write_bytes(b"fake-mp4-first")
    rows = [
        _material_row(
            asset_type="keyframe",
            stored_file="keyframes/paper_third.jpg",
            file_name="paper_third.jpg",
            view="third_person",
            include_existing_vlm=include_existing_vlm,
        ),
        _material_row(
            asset_type="video_clip",
            stored_file="clips/paper_first.mp4",
            file_name="paper_first.mp4",
            view="first_person",
            include_existing_vlm=include_existing_vlm,
        ),
    ]
    (package_root / "key_material_references.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )


def _material_row(
    *,
    asset_type: str,
    stored_file: str,
    file_name: str,
    view: str,
    include_existing_vlm: bool,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "experiment_id": "exp1",
        "session_id": "session1",
        "asset_type": asset_type,
        "asset_kind": asset_type,
        "action_name": "hand paper contact",
        "canonical_action_type": "hand_object_contact",
        "primary_object": "paper",
        "secondary_objects": ["hand"],
        "objects": ["hand", "paper"],
        "view": view,
        "start_sec": 10,
        "end_sec": 12,
        "stored_file": stored_file,
        "file_name": file_name,
        "micro_segment_id": "micro1",
        "segment_id": "seg1",
        "quality_score": 0.82,
        "yolo_evidence_count": 7,
        "candidate_source": "paired_view_micro_segment_key_asset_reference",
        "physical_evidence_mode": "paired_view_time_alignment",
        "source_yolo_evidence": [{"frame_id": "f001", "label": "paper", "confidence": 0.93}],
        "micro_segment": {
            "micro_segment_id": "micro1",
            "parent_segment_id": "seg1",
            "start_sec": 10,
            "end_sec": 12,
            "action_label": "hand_object_contact",
            "primary_object": "paper",
        },
        "time_alignment": {
            "mode": "paired_view_time_alignment",
            "global_start_sec": 10,
            "global_end_sec": 12,
            "confidence": 0.97,
            "views": {"third_person": {"offset_sec": 0.0}, "first_person": {"offset_sec": 0.05}},
        },
    }
    if include_existing_vlm:
        row.update(
            {
                "vlm_semantics": {
                    "description": "Existing VLM says a hand contacts paper.",
                    "confirmed_objects": ["hand", "paper"],
                    "semantic_action": "hand paper contact",
                    "confidence": 0.84,
                    "model": "existing-qwen",
                },
                "qwen_event_audits": [
                    {
                        "audit_id": "audit1",
                        "decision": "accept",
                        "reason": "Existing Qwen audit supports the same action.",
                        "confidence": 0.86,
                        "model": "existing-qwen-audit",
                    }
                ],
                "advanced_vision_evidence": {
                    "event_id": "advanced1",
                    "summary": "Advanced evidence marks hand-object contact.",
                    "confidence": 0.8,
                },
            }
        )
    return row


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

"""Publish key materials directly from the asset catalog (demo unblock).

The fast-locate experiment left metadata/micro_segments.jsonl empty, so the
normal build_yolo_material_references() path produces 0 published materials even
though 491 real, confirmed assets exist in metadata/material_asset_catalog.jsonl
(349 real video clips + 142 keyframes, all source_real=True / exists=True).

This converter reads the catalog, keeps the real & confirmed assets, and writes
them into material_references/key_material_references.jsonl in the row schema the
backend _material_reference_items() reader expects, so the frontend "关键素材库"
shows them. It does NOT call ffmpeg (clips already exist), does NOT call any VLM,
does NOT touch needs_review->official promotion, and does NOT modify the catalog.

Usage:
    python scripts/publish_materials_from_catalog.py <EXPERIMENT_DIR> [--max-per-kind N]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Keep only assets whose review verdict is trustworthy for a demo.
ACCEPT_CONFIRMATION = {"confirmed", "visual_confirmed"}

# Clean Chinese display labels, reusing the project's canonical action map.
try:
    from key_action_indexer.chinese_index import ACTION_DISPLAY_NAMES
except Exception:  # pragma: no cover - fall back to a small local map
    ACTION_DISPLAY_NAMES = {
        "weighing_paper_operation": "手部与称量纸操作",
        "reagent_bottle_interaction": "手部与试剂瓶操作",
        "spatula_interaction": "刮勺/药匙取样",
        "tube_interaction": "试管操作",
        "pipetting": "移液/加样",
        "weighing": "称量",
    }

OBJECT_DISPLAY_NAMES = {
    "paper": "称量纸",
    "weighing_paper": "称量纸",
    "balance": "天平",
    "scale": "天平",
    "pipette": "移液枪",
    "pipette_tip": "枪头",
    "reagent_bottle": "试剂瓶",
    "bottle": "试剂瓶",
    "bottle_cap": "瓶盖",
    "spatula": "药匙",
    "beaker": "烧杯",
    "tube": "试管",
}


def _clean_title(actions: list, objects: list, asset_kind: str) -> str:
    for action in actions:
        label = ACTION_DISPLAY_NAMES.get(str(action))
        if label:
            return label
    for obj in objects:
        label = OBJECT_DISPLAY_NAMES.get(str(obj))
        if label:
            return f"{label}操作"
    return asset_kind


def _load_catalog(metadata_dir: Path) -> list[dict]:
    path = metadata_dir / "material_asset_catalog.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _is_publishable(row: dict) -> bool:
    if not (row.get("publishable_material") and row.get("source_real") and row.get("exists")):
        return False
    if str(row.get("path") or "").strip() == "":
        return False
    level = row.get("confirmation_level") or row.get("evidence_level")
    return level in ACCEPT_CONFIRMATION


def _reference_row(row: dict, index: int) -> dict:
    asset_type = row.get("asset_type")
    asset_kind = "关键帧" if asset_type == "keyframe" else "关键片段"
    objects = row.get("objects") if isinstance(row.get("objects"), list) else []
    actions = row.get("actions") if isinstance(row.get("actions"), list) else []
    primary_object = str(objects[0]) if objects else ""
    primary_action = str(actions[0]) if actions else ""
    title = _clean_title(actions, objects, asset_kind)
    return {
        "item_id": row.get("asset_id") or f"catalog_material_{index:04d}",
        "asset_kind": asset_kind,
        "material_type": asset_kind,
        "stored_file": row.get("path"),
        "file_path": row.get("path"),
        "display_title": title,
        "display_name": title,
        "action_name": primary_action or title,
        "canonical_action_type": primary_action or None,
        "canonical_object": primary_object or None,
        "objects": objects,
        "time_start": row.get("global_start_time"),
        "time_end": row.get("global_end_time"),
        "segment_id": row.get("segment_id"),
        "micro_segment_id": row.get("micro_segment_id"),
        "event_id": row.get("source_id") or row.get("asset_id"),
        "event_type": row.get("event_type") or primary_action or asset_kind,
        "review_status": "accepted",
        "candidate_status": "approved",
        "formal_material_reference": True,
        "approved_by": "catalog_confirmed_import",
        "approved_at": row.get("created_at") or row.get("global_start_time"),
        "evidence_level": row.get("evidence_level") or "visual_confirmed",
        "search_text": row.get("search_text") or title,
        "source_real": True,
        "published_from": "material_asset_catalog",
    }


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python scripts/publish_materials_from_catalog.py <EXPERIMENT_DIR> [--max-per-kind N]")
        return 2
    exp = Path(sys.argv[1]).resolve()
    if not exp.is_dir():
        print(f"experiment dir not found: {exp}")
        return 2
    max_per_kind = None
    if "--max-per-kind" in sys.argv:
        try:
            max_per_kind = int(sys.argv[sys.argv.index("--max-per-kind") + 1])
        except (ValueError, IndexError):
            max_per_kind = None

    metadata_dir = exp / "metadata"
    catalog = _load_catalog(metadata_dir)
    publishable = [r for r in catalog if _is_publishable(r)]
    print(f"catalog rows={len(catalog)} | publishable(real+confirmed)={len(publishable)}")

    # Stable order: keyframes and clips grouped, newest activity first by time.
    publishable.sort(key=lambda r: (str(r.get("asset_type")), str(r.get("global_start_time") or "")))

    refs: list[dict] = []
    per_kind: dict[str, int] = {}
    for row in publishable:
        kind = "关键帧" if row.get("asset_type") == "keyframe" else "关键片段"
        if max_per_kind is not None and per_kind.get(kind, 0) >= max_per_kind:
            continue
        per_kind[kind] = per_kind.get(kind, 0) + 1
        refs.append(_reference_row(row, len(refs) + 1))

    ref_root = exp / "material_references"
    ref_root.mkdir(parents=True, exist_ok=True)
    # The backend reads material_references/素材索引.jsonl (falling back to
    # 素材索引.json -> records). Write the canonical index file it actually reads,
    # plus key_material_references.jsonl for compatibility with other readers.
    index_jsonl = ref_root / "素材索引.jsonl"
    index_json = ref_root / "素材索引.json"
    compat_jsonl = ref_root / "key_material_references.jsonl"
    for out_path in (index_jsonl, compat_jsonl):
        with out_path.open("w", encoding="utf-8") as fh:
            for row in refs:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    index_json.write_text(
        json.dumps(
            {
                "status": "published",
                "formal_publish_blocked": False,
                "published_real_file_count": len(refs),
                "records": refs,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    kinds = {k: v for k, v in per_kind.items()}
    print(f"wrote {len(refs)} references -> {index_jsonl}")
    print(f"by kind: {kinds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

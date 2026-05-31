from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .evaluation import compute_micro_quality_stats
from .chinese_index import refresh_micro_row_chinese_index
from .family_merge import object_family
from .micro_postprocess import merge_same_object_adjacent_micro_segments
from .micro_segmenter import micro_row_to_vector_metadata
from .schemas import MicroSegmentConfig, read_jsonl, write_jsonl
from .tuning import _micro_segments_from_rows
from .vector_index import VectorIndex


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def apply_micro_tuning(
    session_dir: str | Path,
    sweep_best_path: str | Path,
    output_config_path: str | Path | None = None,
) -> dict[str, Any]:
    session = Path(session_dir)
    payload = _read_json(sweep_best_path)
    best = payload.get("recommended_config") or payload.get("best_by_f1") or payload.get("best_config") or {}
    config = MicroSegmentConfig()
    recommended = {
        **config.__dict__,
        "micro_interaction_threshold": float(best.get("interaction_threshold", config.micro_interaction_threshold)),
        "default_interaction_threshold": float(best.get("interaction_threshold", config.default_interaction_threshold)),
        "micro_merge_gap_sec": float(best.get("merge_gap_sec", config.micro_merge_gap_sec)),
        "default_merge_gap_sec": float(best.get("merge_gap_sec", config.default_merge_gap_sec)),
        "micro_min_duration_sec": float(best.get("min_duration_sec", config.micro_min_duration_sec)),
        "default_min_duration_sec": float(best.get("min_duration_sec", config.default_min_duration_sec)),
        "config_source": "threshold_sweep_recommended",
        "sweep_best_path": str(sweep_best_path),
        "sweep_metric_snapshot": best,
    }
    target = Path(output_config_path) if output_config_path else session / "metadata" / "recommended_micro_segment_config.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(recommended, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "session_dir": str(session),
        "sweep_best": str(sweep_best_path),
        "output_config": str(target),
        "recommended_config": recommended,
    }


def _minimal_micro_rows_from_yolo(session: Path, config: MicroSegmentConfig) -> list[dict[str, Any]]:
    rows_path = session / "cv_outputs" / "yolo_micro_frame_rows.jsonl"
    if not rows_path.exists():
        return []
    rows = read_jsonl(rows_path)
    predicted = _micro_segments_from_rows(
        rows,
        interaction_threshold=config.micro_interaction_threshold,
        merge_gap_sec=config.micro_merge_gap_sec,
        min_duration_sec=config.micro_min_duration_sec,
    )
    for index, row in enumerate(predicted, start=1):
        micro_id = str(row.get("micro_segment_id") or f"micro_{index:03d}")
        row.setdefault("session_id", session.parent.name)
        row.setdefault("display_order", index)
        row.setdefault("display_id", f"micro_{index:03d}")
        row.setdefault("global_start_time", "")
        row.setdefault("global_end_time", "")
        row.setdefault("first_person", {"clip_path": None, "local_start_sec": row.get("start_sec"), "local_end_sec": row.get("end_sec")})
        row.setdefault("third_person", {"clip_path": None, "local_start_sec": row.get("start_sec"), "local_end_sec": row.get("end_sec")})
        row.setdefault("keyframes", {})
        row.setdefault("dialogue_context", [])
        row.setdefault("quality", {"confidence": "unknown", "warnings": []})
        row.setdefault("text_description", {"action_type": "", "summary": "", "index_text": f"index_level: micro_segment\nmicro_segment_id: {micro_id}\nprimary_object: {row.get('interaction', {}).get('primary_object')}"})
        row.setdefault("index", {"index_level": "micro_segment", "embedding_id": f"emb_{micro_id}"})
    return predicted


def _combined_segment_metadata(session: Path) -> list[dict[str, Any]]:
    path = session / "metadata" / "vector_metadata.jsonl"
    if not path.exists():
        return []
    return [row for row in read_jsonl(path) if row.get("index_level", "segment") == "segment"]


def _label(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _micro_row_primary_object(row: dict[str, Any]) -> str:
    interaction = row.get("interaction") if isinstance(row.get("interaction"), dict) else {}
    return _label(interaction.get("primary_object") or row.get("primary_object"))


def _enrich_micro_row_arbitration(row: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(row)
    interaction = dict(enriched.get("interaction") if isinstance(enriched.get("interaction"), dict) else {})
    primary = _label(interaction.get("primary_object") or enriched.get("primary_object"))
    family = interaction.get("primary_object_family") or enriched.get("primary_object_family") or object_family(primary)
    if primary:
        interaction.setdefault("primary_object_family", family)
        interaction.setdefault("primary_object_arbitration", "reused_micro_row_family_enriched")
        interaction.setdefault("primary_object_vote_score", interaction.get("max_interaction_score"))
        interaction.setdefault("primary_object_vote_margin", None)
        interaction.setdefault("primary_object_vote_counts", {primary: len(interaction.get("evidence_frame_indices") or []) or 1})
        interaction.setdefault("primary_object_vote_scores", {primary: interaction.get("primary_object_vote_score") or 0.0})
        interaction.setdefault("peak_primary_object", primary)
        enriched["primary_object_family"] = family
        enriched["primary_object_arbitration"] = interaction.get("primary_object_arbitration")
    enriched["interaction"] = interaction
    return enriched


def _filter_micro_rows_for_config(rows: list[dict[str, Any]], config: MicroSegmentConfig) -> list[dict[str, Any]]:
    disabled = {_label(item) for item in getattr(config, "disabled_primary_objects", []) or []}
    allowed = {_label(item) for item in getattr(config, "allowed_primary_objects", []) or []}
    filtered: list[dict[str, Any]] = []
    for row in rows:
        primary = _micro_row_primary_object(row)
        if disabled and primary in disabled:
            continue
        if allowed and primary not in allowed:
            continue
        filtered.append(_enrich_micro_row_arbitration(row))
    return filtered


def _rebuild_indexes(session: Path, vector_metadata: list[dict[str, Any]], micro_vector_metadata: list[dict[str, Any]]) -> None:
    index_dir = session / "index"
    combined = [*_combined_segment_metadata(session), *micro_vector_metadata]
    if not combined:
        combined = vector_metadata
    write_jsonl(session / "metadata" / "micro_vector_metadata.jsonl", micro_vector_metadata)
    write_jsonl(session / "metadata" / "vector_metadata.jsonl", combined)
    index = VectorIndex()
    index.build([str(item.get("index_text") or "") for item in combined], combined)
    index.save(index_dir)
    write_jsonl(index_dir / "docstore.jsonl", combined)
    micro_index = VectorIndex()
    micro_index.build([str(item.get("index_text") or "") for item in micro_vector_metadata], micro_vector_metadata)
    micro_index.save(index_dir / "micro_segments")


def rerun_micro_with_config(
    session_dir: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    session = Path(session_dir)
    config = MicroSegmentConfig.from_dict(_read_json(config_path))
    raw_path = session / "metadata" / "micro_segments_raw.jsonl"
    if raw_path.exists():
        raw_rows = read_jsonl(raw_path)
    else:
        raw_rows = _minimal_micro_rows_from_yolo(session, config)
        write_jsonl(raw_path, raw_rows)
    active_raw_rows = _filter_micro_rows_for_config(raw_rows, config)
    merged_rows, merge_stats = merge_same_object_adjacent_micro_segments(active_raw_rows, config=config)
    merged_rows = [refresh_micro_row_chinese_index(row) for row in merged_rows]
    write_jsonl(session / "metadata" / "micro_segments.jsonl", merged_rows)
    micro_vector_metadata = [micro_row_to_vector_metadata(row) for row in merged_rows]
    _rebuild_indexes(session, micro_vector_metadata, micro_vector_metadata)
    (session / "evaluation").mkdir(parents=True, exist_ok=True)
    quality = compute_micro_quality_stats(
        session / "metadata" / "micro_segments.jsonl",
        session / "evaluation" / "micro_quality_stats.json",
    )
    parent_duration_sec = float(quality.get("parent_duration_sec") or 0.0)
    if parent_duration_sec > 0:
        merge_stats["micro_per_minute_before"] = len(active_raw_rows) / parent_duration_sec * 60.0
        merge_stats["micro_per_minute_after"] = len(merged_rows) / parent_duration_sec * 60.0
        merge_stats["parent_duration_sec"] = parent_duration_sec
    (session / "evaluation" / "micro_merge_stats.json").write_text(json.dumps(merge_stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "session_dir": str(session),
        "config": str(config_path),
        "raw_micro_segments": str(raw_path),
        "micro_segments": str(session / "metadata" / "micro_segments.jsonl"),
        "micro_vector_metadata": str(session / "metadata" / "micro_vector_metadata.jsonl"),
        "vector_metadata": str(session / "metadata" / "vector_metadata.jsonl"),
        "index_dir": str(session / "index"),
        "raw_micro_count": len(active_raw_rows),
        "unfiltered_raw_micro_count": len(raw_rows),
        "merged_micro_count": len(merged_rows),
        "allowed_primary_objects": getattr(config, "allowed_primary_objects", []),
        "disabled_primary_objects": getattr(config, "disabled_primary_objects", []),
        "merge_stats": merge_stats,
        "quality_stats": quality,
        "reran_yolo": False,
    }

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import MicroSegmentConfig, read_jsonl
from .semantic_alias import chinese_aliases_for_label
from .vector_index import VectorIndex


DEFAULT_QUERY = "\u627e\u4e00\u4e0b\u4f7f\u7528\u79fb\u6db2\u67aa\u52a0\u6837\u7684\u7247\u6bb5"
QUERY_EXAMPLES = [
    "\u624b\u78b0\u74f6\u5b50",
    "\u79f0\u91cf",
    "\u4f7f\u7528\u522e\u52fa",
    "\u52a0\u6837",
]


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return default


def _list_values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _format_list(values: Any) -> str:
    items = [str(item) for item in _list_values(values) if item]
    return ", ".join(items) if items else "none"


def _short_text(text: Any, limit: int = 220) -> str:
    clean = " ".join(str(text or "").split())
    return clean if len(clean) <= limit else clean[: limit - 3] + "..."


def _format_score(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except Exception:
        return "n/a"


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _format_thresholds(thresholds: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for label, value in sorted(thresholds.items()):
        aliases = chinese_aliases_for_label(label)
        alias_text = f" ({', '.join(aliases)})" if aliases else ""
        if isinstance(value, dict):
            fields = ", ".join(f"{key}={item}" for key, item in sorted(value.items()))
        else:
            fields = str(value)
        lines.append(f"- {label}{alias_text}: {fields}")
    return lines


def _primary_object(micro: dict[str, Any]) -> str:
    interaction = micro.get("interaction") if isinstance(micro.get("interaction"), dict) else {}
    return str(micro.get("primary_object") or interaction.get("primary_object") or "unknown")


def _quality(micro: dict[str, Any]) -> str:
    quality = micro.get("quality") if isinstance(micro.get("quality"), dict) else {}
    return str(quality.get("confidence") or "unknown")


def _evidence_profile(item: dict[str, Any]) -> dict[str, Any]:
    evidence = item.get("evidence") if isinstance(item.get("evidence"), dict) else {}
    level = str(item.get("evidence_level") or evidence.get("evidence_level") or "unknown")
    reasons = _list_values(item.get("evidence_reasons") or evidence.get("evidence_reasons"))
    limitations = _list_values(item.get("limitations") or evidence.get("limitations"))
    return {
        "evidence_level": level,
        "evidence_reasons": [str(value) for value in reasons if value],
        "limitations": [str(value) for value in limitations if value],
    }


def _distribution(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = value or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _micro_stats(micro_segments: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    by_object: dict[str, int] = {}
    by_quality: dict[str, int] = {}
    small_counts = {name: 0 for name in ("spatula", "pipette", "pipette_tip", "tube")}
    for micro in micro_segments:
        primary = _primary_object(micro)
        quality = _quality(micro)
        by_object[primary] = by_object.get(primary, 0) + 1
        by_quality[quality] = by_quality.get(quality, 0) + 1
        if primary in small_counts:
            small_counts[primary] += 1
    return by_object, by_quality, small_counts


def _derived_micro_quality_stats(micro_segments: list[dict[str, Any]]) -> dict[str, Any]:
    profiles = [_evidence_profile(item) for item in micro_segments]
    warning_counts: dict[str, int] = {}
    limitation_counts: dict[str, int] = {}
    for micro, profile in zip(micro_segments, profiles):
        quality = micro.get("quality") if isinstance(micro.get("quality"), dict) else {}
        for warning in _list_values(quality.get("warnings")):
            warning_counts[str(warning)] = warning_counts.get(str(warning), 0) + 1
        for limitation in profile["limitations"]:
            limitation_counts[limitation] = limitation_counts.get(limitation, 0) + 1
    return {
        "total": len(micro_segments),
        "quality_distribution": _distribution([_quality(item) for item in micro_segments]),
        "evidence_level_distribution": _distribution([profile["evidence_level"] for profile in profiles]),
        "manual_corrected_count": sum(1 for item in micro_segments if item.get("manual_corrected")),
        "dialogue_context_available_count": sum(
            1 for item in micro_segments if item.get("dialogue_context") or item.get("dialogue_context_available")
        ),
        "warning_counts": warning_counts,
        "limitation_counts": limitation_counts,
    }


def _transcript_coverage(
    key_segments: list[dict[str, Any]],
    micro_segments: list[dict[str, Any]],
    aligned_utterances: list[dict[str, Any]],
) -> dict[str, Any]:
    parent_count = len(key_segments)
    micro_count = len(micro_segments)
    parents_with_dialogue = sum(1 for segment in key_segments if segment.get("dialogue_context"))
    micros_with_dialogue = sum(
        1 for micro in micro_segments if micro.get("dialogue_context") or micro.get("dialogue_context_available")
    )
    return {
        "utterance_count": len(aligned_utterances),
        "parent_count": parent_count,
        "parents_with_dialogue": parents_with_dialogue,
        "parent_coverage": parents_with_dialogue / parent_count if parent_count else 0.0,
        "micro_count": micro_count,
        "micros_with_dialogue": micros_with_dialogue,
        "micro_coverage": micros_with_dialogue / micro_count if micro_count else 0.0,
    }


def _query_index(index_dir: Path, query_text: str, top_k: int) -> list[dict[str, Any]]:
    if not (index_dir / "fallback_index.pkl").exists():
        return []
    try:
        return VectorIndex.load(index_dir).query(query_text, top_k=top_k)
    except Exception as exc:
        return [{"error": str(exc)}]


def _best_tuning_config(root: Path) -> dict[str, Any] | None:
    for rel in ("evaluation/micro_threshold_sweep_best.json", "evaluation/tuning_results.json"):
        payload = _read_json(root / rel, None)
        if isinstance(payload, dict):
            best = payload.get("recommended_config") or payload.get("best_by_f1") or payload.get("best_config")
            if isinstance(best, dict):
                return best
    return None


def generate_report(
    session_dir: str | Path,
    query_text: str = DEFAULT_QUERY,
    top_k: int = 3,
    output_path: str | Path | None = None,
) -> Path:
    root = Path(session_dir)
    target = Path(output_path) if output_path else root / "reports" / "mvp_validation_report.md"
    target.parent.mkdir(parents=True, exist_ok=True)

    manifest = _read_json(root / "manifest.json", {})
    summary = _read_json(root / "pipeline_summary.json", {})
    video_info = _read_json(root / "video_info.json", _read_json(root / "metadata" / "video_info.json", {}))
    detector_config = _read_json(root / "metadata" / "detector_config.json", manifest.get("detection_config", {}))
    micro_refine_summary = _read_json(root / "metadata" / "yolo_micro_scan_summary.json", summary.get("micro_refine_summary", {}))
    segment_eval = _read_json(root / "evaluation" / "segment_eval.json", None)
    micro_eval = _read_json(root / "evaluation" / "micro_segment_eval.json", None)
    stored_quality_stats = _read_json(root / "evaluation" / "micro_quality_stats.json", None)
    micro_merge_stats = _read_json(root / "evaluation" / "micro_merge_stats.json", {})
    recommended_config_path = root / "metadata" / "recommended_micro_segment_config.json"
    recommended_config_applied = recommended_config_path.exists()

    key_segments_path = root / "metadata" / "key_action_segments.jsonl"
    micro_segments_path = root / "metadata" / "micro_segments_corrected.jsonl"
    if not micro_segments_path.exists():
        micro_segments_path = root / "metadata" / "micro_segments.jsonl"
    key_segments = read_jsonl(key_segments_path) if key_segments_path.exists() else []
    micro_segments = read_jsonl(micro_segments_path) if micro_segments_path.exists() else []
    aligned_utterances = read_jsonl(root / "transcript" / "aligned_transcript.jsonl") if (root / "transcript" / "aligned_transcript.jsonl").exists() else []

    micro_config = MicroSegmentConfig.from_dict(
        manifest.get("micro_segment_config") if isinstance(manifest.get("micro_segment_config"), dict) else None
    )
    transcript_coverage = _transcript_coverage(key_segments, micro_segments, aligned_utterances)
    by_object, by_quality, small_counts = _micro_stats(micro_segments)
    derived_quality_stats = _derived_micro_quality_stats(micro_segments)
    micro_quality_stats = {**derived_quality_stats, **stored_quality_stats} if isinstance(stored_quality_stats, dict) else derived_quality_stats
    micro_quality_stats.setdefault("evidence_level_distribution", derived_quality_stats["evidence_level_distribution"])
    micro_quality_stats.setdefault("limitation_counts", derived_quality_stats["limitation_counts"])
    best_tuning_config = _best_tuning_config(root)

    lines: list[str] = [
        "# MVP Validation Report",
        "",
        "## Session",
        "",
        f"- session_id: {manifest.get('session_id', summary.get('session_id', 'unknown'))}",
        f"- output_dir: {root}",
        f"- dry_run: {summary.get('dry_run', 'unknown')}",
        "",
        "## Input Videos",
        "",
    ]
    videos = video_info.get("video_sources", video_info if isinstance(video_info, dict) else {})
    if isinstance(videos, dict) and videos:
        for name, info in videos.items():
            if isinstance(info, dict):
                lines.append(
                    f"- {name}: path={info.get('path')} can_open={info.get('can_open')} "
                    f"fps={info.get('fps')} size={info.get('width')}x{info.get('height')} "
                    f"duration_sec={info.get('duration_sec')}"
                )
    else:
        lines.append("- no video_info.json found")

    lines.extend(["", "## Transcript", ""])
    lines.append(f"- path: {manifest['transcript'].get('path')}" if manifest.get("transcript") else "- transcript not provided")
    lines.append(f"- aligned_utterance_count: {transcript_coverage['utterance_count']}")
    lines.append(
        f"- parent_dialogue_coverage: {transcript_coverage['parents_with_dialogue']}/"
        f"{transcript_coverage['parent_count']} ({transcript_coverage['parent_coverage']:.1%})"
    )
    lines.append(
        f"- micro_dialogue_coverage: {transcript_coverage['micros_with_dialogue']}/"
        f"{transcript_coverage['micro_count']} ({transcript_coverage['micro_coverage']:.1%})"
    )
    if transcript_coverage["utterance_count"] == 0:
        lines.append(
            "- \u5f53\u524d\u6837\u4f8b\u7f3a\u5c11 transcript\uff0c"
            "\u52a0\u6837\u3001\u8bfb\u6570\u3001\u8bb0\u5f55\u3001\u79f0\u91cf\u7b49\u8bed\u4e49\u52a8\u4f5c"
            "\u4e3b\u8981\u4f9d\u8d56\u89c6\u89c9\u8bc1\u636e\uff0c\u8bed\u4e49\u68c0\u7d22\u7a33\u5b9a\u6027\u4f1a\u4e0b\u964d\u3002"
        )

    lines.extend(["", "## ASR Coverage", ""])
    lines.append(f"- asr_utterance_count: {transcript_coverage['utterance_count']}")
    lines.append(
        f"- parent_asr_coverage: {transcript_coverage['parents_with_dialogue']}/"
        f"{transcript_coverage['parent_count']} ({transcript_coverage['parent_coverage']:.1%})"
    )
    lines.append(
        f"- micro_asr_coverage: {transcript_coverage['micros_with_dialogue']}/"
        f"{transcript_coverage['micro_count']} ({transcript_coverage['micro_coverage']:.1%})"
    )

    lines.extend(["", "## Detection Config", "", "```json", json.dumps(detector_config, ensure_ascii=False, indent=2), "```", ""])
    lines.extend(["## Micro Refinement", ""])
    used_refine = bool(summary.get("artifacts", {}).get("yolo_micro_frame_rows") or micro_refine_summary.get("output_path"))
    lines.append(f"- used_micro_refine_rescan: {used_refine}")
    lines.append(f"- summary: `{json.dumps(micro_refine_summary, ensure_ascii=False)}`")

    lines.extend(["", "## Class Thresholds", ""])
    yolo_thresholds = detector_config.get("yolo_class_thresholds") if isinstance(detector_config, dict) else {}
    lines.append("YOLO class thresholds:")
    lines.extend(_format_thresholds(yolo_thresholds if isinstance(yolo_thresholds, dict) else {}) or ["- none configured"])
    lines.append("")
    lines.append("Micro interaction class thresholds:")
    lines.extend(_format_thresholds(micro_config.class_thresholds) or ["- none configured"])

    lines.extend(["", "## Threshold Sweep Best Config", ""])
    if best_tuning_config:
        lines.append(f"- threshold_sweep_best_config: `{json.dumps(best_tuning_config, ensure_ascii=False)}`")
    else:
        lines.append("- threshold_sweep_best_config: not available")
    lines.append(f"- recommended_config_path: {recommended_config_path if recommended_config_applied else 'not available'}")
    lines.append(f"- recommended_config_applied: {recommended_config_applied}")

    lines.extend(["", "## GT Coverage", ""])
    if isinstance(micro_eval, dict):
        lines.append(f"- gt_completeness: {micro_eval.get('gt_completeness', 'unknown')}")
        lines.append(f"- labeled_window_count: {micro_eval.get('labeled_window_count', 0)}")
        lines.append(f"- labeled_duration_sec: {micro_eval.get('labeled_duration_sec', 0.0)}")
        lines.append(f"- predictions_inside_labeled_windows: {micro_eval.get('predictions_inside_labeled_windows', micro_eval.get('num_predicted'))}")
        lines.append(f"- predictions_outside_labeled_windows: {micro_eval.get('predictions_outside_labeled_windows', 0)}")
        lines.append(f"- precision_is_formal: {micro_eval.get('precision_is_formal', False)}")
        if not micro_eval.get("precision_is_formal", False):
            lines.append("- GT coverage unknown or partial; precision is for debugging only.")
        else:
            lines.append("- GT coverage complete; precision is formal within labeled windows.")
    else:
        lines.append("- GT coverage unknown; precision is for debugging only.")

    total_duration = sum(float(item.get("duration_sec", 0.0) or 0.0) for item in key_segments)
    lines.extend(
        [
            "",
            "## Detected Segments",
            "",
            f"- parent_segment_count: {len(key_segments)}",
            f"- total_action_duration_sec: {total_duration:.3f}",
            f"- micro_segment_count: {len(micro_segments)}",
            "",
        ]
    )
    for segment in key_segments:
        third = segment.get("third_person") or {}
        first = segment.get("first_person") or {}
        text_desc = segment.get("text_description") or {}
        lines.extend(
            [
                f"### {segment.get('segment_id')}",
                "",
                f"- global_start_time: {segment.get('global_start_time')}",
                f"- global_end_time: {segment.get('global_end_time')}",
                f"- duration_sec: {segment.get('duration_sec')}",
                f"- third_person_clip: {third.get('clip_path')}",
                f"- first_person_clip: {first.get('clip_path') if first else None}",
                f"- related_dialogue: {segment.get('dialogue_context', [])}",
                f"- action_type: {text_desc.get('action_type')}",
                f"- index_text: {_short_text((segment.get('index') or {}).get('index_text', ''))}",
                "",
            ]
        )
        children = [item for item in micro_segments if item.get("parent_segment_id") == segment.get("segment_id")]
        if children:
            lines.extend(["Micro-segments:", ""])
            for micro in sorted(children, key=lambda item: float(item.get("start_sec", 0.0) or 0.0)):
                interaction = micro.get("interaction") or {}
                quality = micro.get("quality") or {}
                keyframes = micro.get("keyframes") or {}
                evidence = _evidence_profile(micro)
                evidence_marker = " insufficient_evidence" if str(evidence["evidence_level"]).startswith("insufficient") else ""
                lines.append(
                    f"- {micro.get('display_id') or micro.get('micro_segment_id')} "
                    f"({micro.get('micro_segment_id')}): object={interaction.get('primary_object')} "
                    f"type={interaction.get('interaction_type')} duration={micro.get('duration_sec')} "
                    f"max_score={interaction.get('max_interaction_score')} quality={quality.get('confidence')} "
                    f"manual_corrected={micro.get('manual_corrected', False)} peak={keyframes.get('peak_frame')} "
                    f"evidence_level={evidence['evidence_level']}{evidence_marker} "
                    f"evidence_reasons={_format_list(evidence['evidence_reasons'])} "
                    f"limitations={_format_list(evidence['limitations'])}"
                )
            lines.append("")

    lines.extend(["## Micro-segment Statistics", ""])
    lines.append(f"- total: {len(micro_segments)}")
    if micro_merge_stats:
        lines.append(f"- raw_micro_count: {micro_merge_stats.get('raw_micro_count')}")
        lines.append(f"- merged_micro_count: {micro_merge_stats.get('merged_micro_count')}")
        lines.append(f"- merge_count: {micro_merge_stats.get('merge_count')}")
        lines.append(f"- micro_per_minute_before: {_format_score(micro_merge_stats.get('micro_per_minute_before'))}")
        lines.append(f"- micro_per_minute_after: {_format_score(micro_merge_stats.get('micro_per_minute_after'))}")
    lines.append(f"- by_primary_object: {json.dumps(by_object, ensure_ascii=False)}")
    lines.append(f"- quality_distribution: {json.dumps(by_quality, ensure_ascii=False)}")
    lines.append(f"- small_object_counts: {json.dumps(small_counts, ensure_ascii=False)}")
    lines.append(f"- evidence_level_distribution: {json.dumps(micro_quality_stats.get('evidence_level_distribution', {}), ensure_ascii=False)}")
    lines.append(f"- micro_quality_stats: `{json.dumps(micro_quality_stats, ensure_ascii=False)}`")
    has_pipette_evidence = any(_primary_object(item) in {"pipette", "pipette_tip", "tube"} for item in micro_segments)
    if not has_pipette_evidence and transcript_coverage["utterance_count"] == 0:
        lines.append(
            "- \u5f53\u524d\u6837\u4f8b\u7f3a\u5c11\u53ef\u4fe1\u52a0\u6837\u8bc1\u636e\uff0c"
            "\u7cfb\u7edf\u672a\u5f3a\u884c\u751f\u6210\u52a0\u6837 micro-segment\u3002"
        )
    lines.append("")

    lines.extend(["## ASR Effect", ""])
    lines.append(f"- transcript_coverage_parent: {transcript_coverage['parent_coverage']:.1%}")
    lines.append(f"- transcript_coverage_micro: {transcript_coverage['micro_coverage']:.1%}")
    semantic_keywords = {
        "weighing": ["称量", "天平"],
        "spatula": ["刮勺", "药匙"],
        "sample_adding": ["加样", "移液", "微升"],
        "reading": ["读数", "记录"],
    }
    dialogue_text = " ".join(str(row.get("text", "")) for row in aligned_utterances)
    found = {name: any(keyword in dialogue_text for keyword in keywords) for name, keywords in semantic_keywords.items()}
    lines.append(f"- semantic_actions_found: {json.dumps(found, ensure_ascii=False)}")
    evidence_levels = [(_evidence_profile(row).get("evidence_level") or "unknown") for row in micro_segments]
    lines.append(f"- evidence_level_changes_possible: {transcript_coverage['utterance_count'] > 0}")
    lines.append(f"- evidence_level_distribution: {json.dumps(_distribution(evidence_levels), ensure_ascii=False)}")
    lines.append("")

    lines.extend(
        [
            "## Debug Files",
            "",
            f"- roi_preview: {root / 'debug' / 'roi_preview.jpg'}",
            f"- frame_scores: {root / 'debug' / 'frame_scores.png'}",
            f"- segments_contact_sheet: {root / 'debug' / 'segments_contact_sheet.jpg'}",
            "",
            "## Evaluation",
            "",
        ]
    )
    if segment_eval:
        lines.extend(
            [
                f"- segment_precision: {segment_eval.get('precision')}",
                f"- segment_recall: {segment_eval.get('recall')}",
                f"- segment_f1: {segment_eval.get('f1')}",
                f"- segment_mean_iou: {segment_eval.get('mean_iou')}",
            ]
        )
    else:
        lines.append("- manual segment evaluation not available")
    if micro_eval:
        lines.extend(
            [
                f"- micro_precision: {micro_eval.get('precision')}",
                f"- micro_recall: {micro_eval.get('recall')}",
                f"- micro_f1: {micro_eval.get('f1')}",
                f"- micro_mean_iou: {micro_eval.get('mean_iou')}",
                f"- primary_object_accuracy: {micro_eval.get('primary_object_accuracy')}",
                f"- object_family_accuracy: {micro_eval.get('object_family_accuracy', 'not_available')}",
                f"- object_presence_accuracy: {micro_eval.get('object_presence_accuracy', 'not_available')}",
                f"- interaction_type_accuracy: {micro_eval.get('interaction_type_accuracy')}",
                f"- action_type_accuracy: {micro_eval.get('action_type_accuracy')}",
            ]
        )
    else:
        lines.append("- manual micro-segment evaluation not available")

    lines.extend(["", "## Query Rerank Smoke Tests", ""])
    first_query_explanation: dict[str, Any] | None = None
    for query in _ordered_unique([query_text, *QUERY_EXAMPLES]):
        results = _query_index(root / "index", query_text=query, top_k=top_k)
        lines.append(f"### Query: {query}")
        if not results:
            lines.append("- no query results")
            lines.append("")
            continue
        for rank, item in enumerate(results, start=1):
            if "error" in item:
                lines.append(f"- error: {item['error']}")
                continue
            evidence = _evidence_profile(item)
            if first_query_explanation is None:
                first_query_explanation = {"query": query, "rank": rank, "item": item, "evidence": evidence}
            lines.append(
                f"- rank={rank} result_id={item.get('micro_segment_id') or item.get('segment_id')} "
                f"index_level={item.get('index_level')} primary_object={item.get('primary_object')} "
                f"vector_score={_format_score(item.get('vector_score'))} "
                f"rerank_score={_format_score(item.get('rerank_score'))} "
                f"score={_format_score(item.get('score'))} "
                f"rerank_reasons={item.get('rerank_reasons', [])} "
                f"evidence_level={evidence['evidence_level']} "
                f"evidence_reasons={_format_list(evidence['evidence_reasons'])} "
                f"limitations={_format_list(evidence['limitations'])} "
                f"transcript_contributed={bool(item.get('dialogue_context_available') or item.get('related_dialogue') or item.get('dialogue_keywords'))}"
            )
        lines.append("")

    lines.extend(["## Query Evidence Explanation", ""])
    if first_query_explanation:
        item = first_query_explanation["item"]
        evidence = first_query_explanation["evidence"]
        lines.append(f"- query: {first_query_explanation['query']}")
        lines.append(f"- rank: {first_query_explanation['rank']}")
        lines.append(f"- result_id: {item.get('micro_segment_id') or item.get('segment_id')}")
        lines.append(f"- vector_score: {_format_score(item.get('vector_score'))}")
        lines.append(f"- rerank_score: {_format_score(item.get('rerank_score'))}")
        lines.append(f"- rerank_reasons: {_format_list(item.get('rerank_reasons'))}")
        lines.append(f"- evidence_level: {evidence['evidence_level']}")
        lines.append(f"- evidence_reasons: {_format_list(evidence['evidence_reasons'])}")
        lines.append(f"- limitations: {_format_list(evidence['limitations'])}")
        if str(evidence["evidence_level"]).startswith("insufficient"):
            lines.append("- insufficient_evidence: visual/transcript support is not enough for a trusted hit.")
    else:
        lines.append("- query evidence explanation not available")
    lines.append("")

    lines.extend(
        [
            "## Current Limitations",
            "",
            "- \u5fae\u52a8\u4f5c\u8fb9\u754c\u53ef\u80fd\u53d7 YOLO \u6f0f\u68c0\u548c\u91c7\u6837\u5bc6\u5ea6\u5f71\u54cd\u3002",
            "- \u7269\u4f53\u7c7b\u522b\u9519\u8bef\u4f1a\u5f71\u54cd primary_object \u548c action_type \u63a8\u65ad\u3002",
            "- \u53cc\u89c6\u89d2\u540c\u6b65\u4f9d\u8d56 manifest \u65f6\u95f4\u6233\u548c offset\u3002",
            "- \u5982\u679c\u771f\u5b9e\u5bf9\u8bdd\u7f3a\u5931\uff0c\u52a0\u6837\u3001\u8bfb\u6570\u3001\u79f0\u91cf\u7b49\u8bed\u4e49\u89e3\u91ca\u4f1a\u5f31\u4e00\u4e9b\u3002",
            "- \u6ca1\u6709 pipette\u3001pipette_tip\u3001tube \u6216 ASR \u8bc1\u636e\u65f6\uff0c\u4e0d\u628a\u52a0\u6837\u5f3a\u884c\u6807\u6210\u53ef\u4fe1 micro\u3002",
            "",
            "## Next Steps",
            "",
            "- \u8865\u5145 ASR transcript \u540e\u91cd\u65b0\u8fd0\u884c pipeline\u3002",
            "- \u5bf9 spatula\u3001pipette_tip\u3001tube \u7b49\u5c0f\u7269\u4f53\u7528\u66f4\u9ad8\u91c7\u6837\u7387\u6216\u5c40\u90e8 crop \u590d\u626b\u3002",
            "- \u901a\u8fc7 micro_segments_overrides.jsonl \u505a\u5c11\u91cf\u4eba\u5de5\u6821\u6b63\u5e76\u91cd\u5efa\u7d22\u5f15\u3002",
            "- \u7528 manual_micro_segments.example.jsonl \u683c\u5f0f\u8865 micro-level \u6807\u6ce8\uff0c\u8ba1\u7b97 IoU \u548c\u7c7b\u522b\u51c6\u786e\u7387\u3002",
        ]
    )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def generate_formal_validation_report(
    session_dir: str | Path,
    output_path: str | Path | None = None,
    top_k: int = 3,
) -> Path:
    root = Path(session_dir)
    target = Path(output_path) if output_path else root / "reports" / "formal_validation_report.md"
    target.parent.mkdir(parents=True, exist_ok=True)

    manifest = _read_json(root / "manifest.json", {})
    summary = _read_json(root / "pipeline_summary.json", {})
    video_info = _read_json(root / "video_info.json", _read_json(root / "metadata" / "video_info.json", {}))
    micro_eval = _read_json(root / "evaluation" / "micro_segment_eval.json", {})
    micro_gt_validation = _read_json(root / "evaluation" / "micro_gt_validation.json", {})
    query_validation = _read_json(root / "evaluation" / "query_validation.json", {})
    family_analysis = _read_json(root / "evaluation" / "object_family_merge_analysis.json", {})
    micro_merge_stats = _read_json(root / "evaluation" / "micro_merge_stats.json", {})
    micro_quality_stats = _read_json(root / "evaluation" / "micro_quality_stats.json", {})
    key_segments_path = root / "metadata" / "key_action_segments.jsonl"
    micro_segments_path = root / "metadata" / "micro_segments_corrected.jsonl"
    if not micro_segments_path.exists():
        micro_segments_path = root / "metadata" / "micro_segments.jsonl"
    key_segments = read_jsonl(key_segments_path) if key_segments_path.exists() else []
    micro_segments = read_jsonl(micro_segments_path) if micro_segments_path.exists() else []
    aligned_utterances = read_jsonl(root / "transcript" / "aligned_transcript.jsonl") if (root / "transcript" / "aligned_transcript.jsonl").exists() else []
    transcript_coverage = _transcript_coverage(key_segments, micro_segments, aligned_utterances)
    metric_mode = str(micro_eval.get("metric_mode") or ("formal" if micro_eval.get("precision_is_formal") else "debugging"))
    precision_is_formal = bool(micro_eval.get("precision_is_formal"))
    by_object, by_quality, small_counts = _micro_stats(micro_segments)

    lines = [
        "# Formal Validation Report",
        "",
        "## Session Summary",
        "",
        f"- session_id: {manifest.get('session_id', summary.get('session_id', root.name))}",
        f"- output_dir: {root}",
        f"- parent_segment_count: {len(key_segments)}",
        f"- parent_duration_sec: {sum(float(item.get('duration_sec', 0.0) or 0.0) for item in key_segments):.3f}",
        f"- raw_micro_count: {micro_merge_stats.get('raw_micro_count', summary.get('raw_micro_segment_count', len(micro_segments)))}",
        f"- merged_micro_count: {micro_merge_stats.get('merged_micro_count', len(micro_segments))}",
        "",
        "## Input Video Summary",
        "",
    ]
    videos = video_info.get("video_sources", video_info if isinstance(video_info, dict) else {})
    if isinstance(videos, dict) and videos:
        for name, info in videos.items():
            if isinstance(info, dict):
                lines.append(
                    f"- {name}: can_open={info.get('can_open')} fps={info.get('fps')} "
                    f"duration_sec={info.get('duration_sec')} path={info.get('path')}"
                )
    else:
        lines.append("- video metadata not available")

    lines.extend(
        [
            "",
            "## ASR Coverage",
            "",
            f"- utterance_count: {transcript_coverage['utterance_count']}",
            f"- parent_coverage: {transcript_coverage['parents_with_dialogue']}/{transcript_coverage['parent_count']} ({transcript_coverage['parent_coverage']:.1%})",
            f"- micro_coverage: {transcript_coverage['micros_with_dialogue']}/{transcript_coverage['micro_count']} ({transcript_coverage['micro_coverage']:.1%})",
            "",
            "## GT Coverage And Metric Mode",
            "",
            f"- gt_completeness: {micro_eval.get('gt_completeness', 'unknown')}",
            f"- labeled_window_count: {micro_eval.get('labeled_window_count', 0)}",
            f"- labeled_duration_sec: {micro_eval.get('labeled_duration_sec', 0.0)}",
            f"- metric_mode: {metric_mode}",
            f"- precision_is_formal: {precision_is_formal}",
        ]
    )
    lines.extend(
        [
            f"- gt_validation_valid: {micro_gt_validation.get('valid', 'not_available')}",
            f"- gt_validation_warning_count: {len(micro_gt_validation.get('warnings', []) if isinstance(micro_gt_validation.get('warnings'), list) else [])}",
        ]
    )
    if not precision_is_formal:
        lines.append("")
        lines.append("**Current precision is not a final quality metric because GT coverage is incomplete or unknown.**")
        lines.append("")

    metric_payload = micro_eval.get("formal_metrics") if precision_is_formal else micro_eval.get("debugging_metrics")
    if not isinstance(metric_payload, dict):
        metric_payload = micro_eval
    lines.extend(
        [
            "## Micro Evaluation Metrics",
            "",
            f"- precision: {metric_payload.get('precision')}",
            f"- recall: {metric_payload.get('recall')}",
            f"- f1: {metric_payload.get('f1')}",
            f"- mean_iou: {metric_payload.get('mean_iou')}",
            f"- primary_object_accuracy: {metric_payload.get('primary_object_accuracy')}",
            f"- object_family_accuracy: {metric_payload.get('object_family_accuracy', 'not_available')}",
            f"- object_presence_accuracy: {metric_payload.get('object_presence_accuracy', 'not_available')}",
            f"- interaction_type_accuracy: {metric_payload.get('interaction_type_accuracy')}",
            f"- action_type_accuracy: {metric_payload.get('action_type_accuracy')}",
            "",
            "## Micro Summary",
            "",
            f"- micro_per_minute_before: {_format_score(micro_merge_stats.get('micro_per_minute_before'))}",
            f"- micro_per_minute_after: {_format_score(micro_merge_stats.get('micro_per_minute_after'))}",
            f"- possible_over_segmentation: {micro_quality_stats.get('possible_over_segmentation')}",
            f"- quality_counts: {json.dumps(micro_quality_stats.get('confidence_counts', by_quality), ensure_ascii=False)}",
            f"- primary_object_counts: {json.dumps(micro_quality_stats.get('primary_object_counts', by_object), ensure_ascii=False)}",
            f"- primary_object_family_counts: {json.dumps(micro_quality_stats.get('primary_object_family_counts', {}), ensure_ascii=False)}",
            f"- small_object_counts: {json.dumps(small_counts, ensure_ascii=False)}",
            "",
            "## Object-family Merge Analysis",
            "",
            f"- adjacent_same_family_micro_count: {family_analysis.get('adjacent_same_family_micro_count', 'not_available')}",
            f"- estimated_micro_count_after_family_merge: {family_analysis.get('estimated_micro_count_after_family_merge', 'not_available')}",
            f"- risky_merge_pairs: {len(family_analysis.get('risky_merge_pairs', []) if isinstance(family_analysis.get('risky_merge_pairs'), list) else [])}",
            f"- recommendation: {family_analysis.get('recommendation', 'not_available')}",
            "",
            "## Query Validation",
            "",
            f"- query_hit_rate: {query_validation.get('query_hit_rate', 'not_available')}",
            f"- top1_hit_rate: {query_validation.get('top1_hit_rate', 'not_available')}",
            f"- topk_hit_rate: {query_validation.get('topk_hit_rate', 'not_available')}",
            f"- expected_object_hit_rate: {query_validation.get('expected_object_hit_rate', 'not_available')}",
            f"- valid_fallback_count: {query_validation.get('valid_fallback_count', 'not_available')}",
            "",
        ]
    )
    query_rows = query_validation.get("queries") if isinstance(query_validation, dict) else None
    if isinstance(query_rows, list) and query_rows:
        for row in query_rows:
            top = row.get("top_result") if isinstance(row.get("top_result"), dict) else {}
            lines.append(
                f"- {row.get('query')}: top1_hit={row.get('top1_hit')} topk_hit={row.get('topk_hit')} "
                f"object_hit={row.get('expected_object_hit')} fallback_valid={row.get('fallback_reason_valid')} "
                f"top={top.get('index_level')}:{top.get('micro_segment_id') or top.get('segment_id')} "
                f"object={top.get('primary_object')} evidence={top.get('evidence_level')} "
                f"limitations={_format_list(top.get('limitations'))}"
            )
    else:
        for query in QUERY_EXAMPLES:
            results = _query_index(root / "index", query, top_k)
            top = results[0] if results else {}
            lines.append(
                f"- {query}: top={top.get('index_level')}:{top.get('micro_segment_id') or top.get('segment_id')} "
                f"object={top.get('primary_object')} evidence={top.get('evidence_level')} "
                f"limitations={_format_list(top.get('limitations'))}"
            )

    lines.extend(
        [
            "",
            "## Evidence Explanation",
            "",
            "- visual_confirmed: YOLO hand-object evidence is sufficient, transcript may be absent.",
            "- visual_and_transcript_confirmed: visual evidence and ASR keywords agree.",
            "- transcript_supported: ASR supports the action, but visual object evidence is insufficient.",
            "- weak_visual_evidence: visual signal exists but is too weak or sparse.",
            "- insufficient_evidence: neither visual nor transcript evidence is enough.",
            "",
            "## Limitations",
            "",
            "- Formal precision requires complete GT over the labeled window.",
            "- Object-family merge is analysis-only by default and does not alter micro_segments.jsonl.",
            "- Sample adding remains untrusted without pipette/tube visual evidence or ASR support.",
            "- Micro boundaries still depend on YOLO stability and sample density.",
            "",
            "## Recommended Next Actions",
            "",
            "- Fill the full-window manual micro GT template and validate it.",
            "- Re-run evaluate-micro with the complete eval config to obtain formal metrics.",
            "- Review object-family merge candidates before enabling any family-level merge.",
            "- Add real ASR transcript to improve semantic actions such as 加样 and 读数.",
        ]
    )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from .micro_segmenter import micro_row_to_vector_metadata
from .schemas import read_jsonl, write_jsonl
from .vector_index import VectorIndex


CURRENT_SCOPE_OBJECTS = {
    "balance",
    "beaker",
    "container",
    "paper",
    "pipette",
    "pipette_tip",
    "reagent_bottle",
    "sample_bottle",
    "sample_bottle_blue",
    "spatula",
}


def enrich_micro_quality(
    session_dir: str | Path,
    *,
    output_report: str | Path | None = None,
    target_objects: list[str] | None = None,
) -> dict[str, Any]:
    session = Path(session_dir)
    metadata = session / "metadata"
    micro_path = metadata / "micro_segments.jsonl"
    vector_path = metadata / "vector_metadata.jsonl"
    micro_vector_path = metadata / "micro_vector_metadata.jsonl"
    rows = read_jsonl(micro_path)
    targets = {_norm(item) for item in (target_objects or sorted(CURRENT_SCOPE_OBJECTS))}
    enriched_rows: list[dict[str, Any]] = []
    changed = 0
    grade_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    priority_counts: Counter[str] = Counter()
    object_counts: Counter[str] = Counter()
    low_signal_count = 0
    strong_process_micro_count = 0
    retrieval_candidate_micro_count = 0
    for row in rows:
        enriched, did_change = enrich_micro_row_quality(row, target_objects=targets)
        enriched_rows.append(enriched)
        changed += int(did_change)
        interaction = _as_dict(enriched.get("interaction"))
        primary = _norm(interaction.get("primary_object") or enriched.get("primary_object"))
        if primary:
            object_counts[primary] += 1
        evidence = _as_dict(enriched.get("evidence"))
        grade = str(evidence.get("coverage_signal_grade") or "")
        if grade:
            grade_counts[grade] += 1
        role = str(evidence.get("process_evidence_role") or "")
        if role:
            role_counts[role] += 1
        priority_bucket = str(evidence.get("retrieval_priority_bucket") or "")
        if priority_bucket:
            priority_counts[priority_bucket] += 1
        strong_process_micro_count += int(bool(evidence.get("strong_process_evidence")))
        retrieval_candidate_micro_count += int(role == "retrieval_candidate")
        warnings = _warnings(enriched)
        if "low_signal_yolo_candidate" in warnings or "very_low_signal_yolo_candidate" in warnings:
            low_signal_count += 1

    write_jsonl(micro_path, enriched_rows)
    micro_vector_metadata = [micro_row_to_vector_metadata(row) for row in enriched_rows]
    write_jsonl(micro_vector_path, micro_vector_metadata)
    if vector_path.exists():
        existing_vectors = read_jsonl(vector_path)
        segment_vectors = [row for row in existing_vectors if str(row.get("index_level") or "") != "micro_segment"]
        combined = [*segment_vectors, *micro_vector_metadata]
    else:
        combined = micro_vector_metadata
    write_jsonl(vector_path, combined)
    _rebuild_indexes(session, combined, micro_vector_metadata)

    report = {
        "schema_version": "key_action_micro_quality_enrichment.v1",
        "session_dir": str(session),
        "micro_segment_count": len(enriched_rows),
        "updated_micro_segment_count": changed,
        "target_objects": sorted(targets),
        "object_counts": dict(sorted(object_counts.items())),
        "coverage_signal_grade_counts": dict(sorted(grade_counts.items())),
        "process_evidence_role_counts": dict(sorted(role_counts.items())),
        "retrieval_priority_bucket_counts": dict(sorted(priority_counts.items())),
        "strong_process_micro_count": strong_process_micro_count,
        "retrieval_candidate_micro_count": retrieval_candidate_micro_count,
        "low_signal_micro_count": low_signal_count,
        "artifacts": {
            "micro_segments": str(micro_path),
            "micro_vector_metadata": str(micro_vector_path),
            "vector_metadata": str(vector_path),
            "index_dir": str(session / "index"),
        },
    }
    target = Path(output_report) if output_report else session / "evaluation" / "micro_quality_enrichment_report.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def enrich_micro_row_quality(row: Mapping[str, Any], *, target_objects: set[str] | None = None) -> tuple[dict[str, Any], bool]:
    output = dict(row)
    interaction = dict(_as_dict(output.get("interaction")))
    primary = _norm(interaction.get("primary_object") or output.get("primary_object"))
    targets = target_objects or CURRENT_SCOPE_OBJECTS
    if primary and primary not in targets:
        return output, False

    yolo_evidence = [dict(item) for item in _as_list(output.get("yolo_evidence")) if isinstance(item, Mapping)]
    frame_indices = _as_list(interaction.get("evidence_frame_indices"))
    frame_count = len(yolo_evidence) or len(frame_indices)
    bbox_frame_count = sum(1 for frame in yolo_evidence if _frame_has_bbox_interaction(frame, primary))
    max_score = _max_interaction_score(output, yolo_evidence)
    avg_score = _avg_interaction_score(output, yolo_evidence)
    grade = _coverage_signal_grade(frame_count, bbox_frame_count, max_score)
    warnings = set(_warnings(output))
    coverage_candidate = "coverage_backfill_candidate" in warnings or _has_coverage_limitation(output)

    if coverage_candidate:
        warnings.update({"coverage_backfill_candidate", "physical_evidence_validation_relaxed"})
    if frame_count <= 1:
        warnings.update({"single_frame_coverage_candidate", "low_signal_yolo_candidate"})
    if bbox_frame_count <= 1:
        warnings.add("weak_bbox_continuity")
    if max_score < 0.35:
        warnings.add("very_low_signal_yolo_candidate")
    elif max_score < 0.5:
        warnings.add("low_signal_yolo_candidate")
    elif frame_count >= 2 and bbox_frame_count >= 2:
        warnings.discard("low_signal_yolo_candidate")

    quality = dict(_as_dict(output.get("quality")))
    quality["warnings"] = sorted(warnings)
    output["quality"] = quality

    evidence = dict(_as_dict(output.get("evidence")))
    limitations = [str(item) for item in _as_list(evidence.get("limitations"))]
    if coverage_candidate:
        evidence["coverage_backfill"] = True
        evidence["coverage_backfill_reason"] = "parent segment had YOLO hand-object candidates but no validated micro-segment"
        limitations.append("coverage backfill candidate; requires human/model confirmation before strong process claims")
    evidence["coverage_signal_grade"] = grade
    evidence["coverage_evidence_frame_count"] = int(frame_count)
    evidence["coverage_bbox_frame_count"] = int(bbox_frame_count)
    evidence["coverage_max_interaction_score"] = round(max_score, 6)
    evidence["coverage_avg_interaction_score"] = round(avg_score, 6)
    evidence["keyframe_selection_basis"] = _keyframe_selection_basis(output, bbox_frame_count)
    forced_retrieval = bool(evidence.get("force_retrieval_candidate") or evidence.get("segment_level_coverage_backfill"))
    evidence["process_evidence_role"] = "retrieval_candidate" if forced_retrieval else _process_evidence_role(grade)
    evidence["process_eligible"] = evidence["process_evidence_role"] == "strong_process_evidence"
    evidence["strong_process_evidence"] = bool(evidence["process_eligible"])
    priority, bucket = _retrieval_priority(grade, max_score)
    if forced_retrieval:
        priority = min(priority, 0.35)
        bucket = "segment_level_backfill"
        warnings.add("segment_level_retrieval_backfill")
        limitations.append("segment-level backfill is retrieval-only and cannot establish strong process evidence")
    evidence["retrieval_priority"] = priority
    evidence["retrieval_priority_bucket"] = bucket
    if evidence["process_evidence_role"] == "retrieval_candidate":
        warnings.add("retrieval_candidate_only")
        limitations.append("retrieval candidate only; not eligible for strong process claims")
    elif evidence["process_evidence_role"] == "supporting_process_candidate":
        warnings.add("supporting_process_candidate")
        limitations.append("supporting process candidate; requires continuity confirmation for strong claims")
    evidence["limitations"] = _ordered_unique(limitations)
    quality["warnings"] = sorted(warnings)
    output["quality"] = quality
    output["evidence"] = evidence
    output["evidence_level"] = evidence.get("evidence_level") or output.get("evidence_level")
    output["limitations"] = evidence.get("limitations", [])
    return output, True


def _rebuild_indexes(
    session: Path,
    combined_vector_metadata: list[dict[str, Any]],
    micro_vector_metadata: list[dict[str, Any]],
) -> None:
    index_dir = session / "index"
    index = VectorIndex()
    index.build([str(item.get("index_text") or "") for item in combined_vector_metadata], combined_vector_metadata)
    index.save(index_dir)
    micro_index = VectorIndex()
    micro_index.build([str(item.get("index_text") or "") for item in micro_vector_metadata], micro_vector_metadata)
    micro_index.save(index_dir / "micro_segments")
    segment_metadata = [item for item in combined_vector_metadata if str(item.get("index_level") or "") == "segment"]
    segment_index = VectorIndex()
    segment_index.build([str(item.get("index_text") or "") for item in segment_metadata], segment_metadata)
    segment_index.save(index_dir / "segments")


def _frame_has_bbox_interaction(frame: Mapping[str, Any], primary: str) -> bool:
    interactions = [item for item in _as_list(frame.get("hand_object_interactions")) if isinstance(item, Mapping)]
    if not interactions:
        return False
    candidates = [item for item in interactions if not primary or _norm(item.get("object_label")) == primary] or interactions
    for item in candidates:
        hand_bbox = item.get("hand_bbox")
        object_bbox = item.get("object_bbox")
        if isinstance(hand_bbox, list) and len(hand_bbox) >= 4 and isinstance(object_bbox, list) and len(object_bbox) >= 4:
            return True
    return False


def _max_interaction_score(row: Mapping[str, Any], yolo_evidence: list[dict[str, Any]]) -> float:
    scores = [_float(item.get("interaction_score")) for item in yolo_evidence]
    for frame in yolo_evidence:
        for interaction in _as_list(frame.get("hand_object_interactions")):
            if isinstance(interaction, Mapping):
                scores.append(_float(interaction.get("score")))
    interaction = _as_dict(row.get("interaction"))
    scores.append(_float(interaction.get("max_interaction_score")))
    return max(scores or [0.0])


def _avg_interaction_score(row: Mapping[str, Any], yolo_evidence: list[dict[str, Any]]) -> float:
    scores = [_float(item.get("interaction_score")) for item in yolo_evidence if item.get("interaction_score") is not None]
    if scores:
        return sum(scores) / len(scores)
    interaction = _as_dict(row.get("interaction"))
    return _float(interaction.get("avg_interaction_score"))


def _coverage_signal_grade(frame_count: int, bbox_frame_count: int, max_score: float) -> str:
    if max_score < 0.35:
        return "very_low_signal_yolo_candidate"
    if frame_count <= 1 or bbox_frame_count <= 1:
        return "single_frame_yolo_candidate"
    if bbox_frame_count < frame_count:
        return "continuous_yolo_candidate"
    return "physical_continuity_candidate"


def _process_evidence_role(grade: str) -> str:
    normalized = _norm(grade)
    if normalized == "physical_continuity_candidate":
        return "strong_process_evidence"
    if normalized == "continuous_yolo_candidate":
        return "supporting_process_candidate"
    return "retrieval_candidate"


def _retrieval_priority(grade: str, max_score: float) -> tuple[int, str]:
    normalized = _norm(grade)
    if normalized == "physical_continuity_candidate":
        return 100, "high_physical_continuity"
    if normalized == "continuous_yolo_candidate":
        return 75, "medium_continuous"
    if normalized == "single_frame_yolo_candidate":
        return 45 if max_score >= 0.5 else 35, "low_single_frame"
    return 15, "very_low_signal"


def _keyframe_selection_basis(row: Mapping[str, Any], bbox_frame_count: int) -> str:
    keyframes = _as_dict(row.get("keyframes"))
    present = [key for key in ("contact_frame", "peak_frame", "release_frame") if keyframes.get(key)]
    if bbox_frame_count > 0 and len(present) >= 3:
        return "contact_peak_release_from_physical_yolo_frames"
    if bbox_frame_count > 0 and present:
        return "available_keyframes_ranked_by_physical_yolo_bbox"
    if present:
        return "available_keyframes_without_bbox_continuity"
    return "missing_keyframes"


def _has_coverage_limitation(row: Mapping[str, Any]) -> bool:
    evidence = _as_dict(row.get("evidence"))
    text = " ".join(str(item) for item in _as_list(evidence.get("limitations")) + _as_list(row.get("limitations"))).casefold()
    return "coverage backfill" in text


def _warnings(row: Mapping[str, Any]) -> list[str]:
    quality = _as_dict(row.get("quality"))
    return [str(item) for item in _as_list(quality.get("warnings")) if str(item)]


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _norm(value: Any) -> str:
    return str(value or "").strip().casefold().replace("-", "_").replace(" ", "_")


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from .reviewed_dataset import reviewed_index_dir
from .vector_index import VectorIndex

QUERY_VALIDATION_SCHEMA_VERSION = "key_action_query_validation.v3"

DEFAULT_ACCEPTANCE_THRESHOLDS: dict[str, float] = {
    "min_acceptance_hit_rate": 1.0,
}

THRESHOLD_RATE_FIELDS = {
    "min_acceptance_hit_rate": "acceptance_hit_rate",
    "min_query_hit_rate": "query_hit_rate",
    "min_top1_hit_rate": "top1_hit_rate",
    "min_topk_hit_rate": "topk_hit_rate",
    "min_expected_object_hit_rate": "expected_object_hit_rate",
    "min_expected_index_level_hit_rate": "expected_index_level_hit_rate",
    "min_expected_action_hit_rate": "expected_action_hit_rate",
    "min_expected_id_hit_rate": "expected_id_hit_rate",
    "min_expected_time_window_hit_rate": "expected_time_window_hit_rate",
    "min_traceability_hit_rate": "traceability_hit_rate",
    "min_quality_hit_rate": "quality_hit_rate",
}

COVERAGE_SIGNAL_GRADE_RANK = {
    "very_low_signal_yolo_candidate": 0,
    "single_frame_yolo_candidate": 1,
    "continuous_yolo_candidate": 2,
    "physical_continuity_candidate": 3,
}

EVIDENCE_LEVEL_RANK = {
    "insufficient": 0,
    "insufficient_evidence": 0,
    "weak_visual_evidence": 1,
    "transcript_supported": 2,
    "visual_confirmed": 3,
    "visual_and_transcript_confirmed": 4,
    "trusted": 5,
}


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _primary(result: dict[str, Any]) -> str:
    return str(result.get("primary_object") or "")


def _norm(value: Any) -> str:
    return str(value or "").strip().casefold().replace("-", "_").replace(" ", "_")


def _short_text(value: Any, limit: int = 360) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _query_result_row(result: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "index_level",
        "segment_id",
        "micro_segment_id",
        "parent_segment_id",
        "primary_object",
        "primary_object_family",
        "detected_objects",
        "action_type",
        "interaction_type",
        "global_start_time",
        "global_end_time",
        "third_person_clip",
        "first_person_clip",
        "evidence_level",
        "evidence_reasons",
        "limitations",
        "rerank_reasons",
        "visual_keywords",
        "keyframes",
        "interaction_keyframes",
        "asset_bindings",
    )
    row = {field: result.get(field) for field in fields if result.get(field) not in (None, "", [])}
    row["score"] = result.get("score")
    row["vector_score"] = result.get("vector_score")
    row["rerank_score"] = result.get("rerank_score")
    if result.get("index_text"):
        row["index_text_preview"] = _short_text(result.get("index_text"))
    return row


def query_index(
    index_dir: str | Path,
    queries: list[str],
    *,
    top_k: int = 5,
    filters: dict[str, Any] | None = None,
    output_path: str | Path | None = None,
    session_dir: str | Path | None = None,
) -> dict[str, Any]:
    index_source = Path(index_dir)
    index = VectorIndex.load(index_source)
    normalized_queries = [str(query) for query in queries if str(query).strip()]
    query_rows = []
    for query in normalized_queries:
        results = index.query(query, top_k=top_k, filters=filters or {})
        query_rows.append(
            {
                "query": query,
                "top_k": int(top_k),
                "result_count": len(results),
                "results": [_query_result_row(result) for result in results],
            }
        )
    result = {
        "session_dir": str(session_dir) if session_dir is not None else str(index_source.parent),
        "index_dir": str(index_source),
        "query_count": len(query_rows),
        "filters": filters or {},
        "queries": query_rows,
    }
    if output_path:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def query_session_index(
    session_dir: str | Path,
    queries: list[str],
    *,
    top_k: int = 5,
    filters: dict[str, Any] | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    session = Path(session_dir)
    return query_index(
        reviewed_index_dir(session),
        queries,
        top_k=top_k,
        filters=filters,
        output_path=output_path,
        session_dir=session,
    )


def _fallback_valid(result: dict[str, Any], rule: dict[str, Any]) -> bool:
    if not rule.get("allow_parent_fallback_when_insufficient_evidence"):
        return False
    if str(result.get("index_level") or "") != "segment":
        return False
    limitations = " ".join(str(item) for item in _list(result.get("limitations")))
    reasons = " ".join(str(item) for item in _list(result.get("rerank_reasons")))
    text = f"{limitations} {reasons}".lower()
    if "missing pipette" in text and "missing transcript" in text:
        return True
    if "fallback_parent" in text:
        return True
    return "insufficient" in text and ("missing transcript" in text or "sample_adding" in text)


def validate_queries(
    session_dir: str | Path,
    config_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    session = Path(session_dir)
    config = _read_json(config_path)
    index_dir = reviewed_index_dir(session)
    index = VectorIndex.load(index_dir)
    thresholds = _validation_thresholds(config)
    base_quality_policy = _quality_policy(config)
    rows: list[dict[str, Any]] = []
    acceptance_hits = 0
    top1_hits = 0
    topk_hits = 0
    quality_hits = 0
    expected_object_hits = 0
    expected_index_level_hits = 0
    expected_action_hits = 0
    expected_id_hits = 0
    expected_time_window_hits = 0
    traceability_hits = 0
    valid_fallbacks = 0
    for rule in config.get("queries", []):
        query = str(rule.get("query") or "")
        top_k = int(rule.get("top_k") or 3)
        expected_objects = {_norm(item) for item in rule.get("expected_objects") or []}
        expected_actions = {_norm(item) for item in rule.get("expected_actions") or []}
        expected_segment_ids = {str(item) for item in rule.get("expected_segment_ids") or [] if str(item)}
        expected_micro_segment_ids = {str(item) for item in rule.get("expected_micro_segment_ids") or [] if str(item)}
        expected_level = str(rule.get("expected_index_level") or "")
        expected_time_window = rule.get("expected_time_window") if isinstance(rule.get("expected_time_window"), dict) else None
        require_traceability = bool(rule.get("require_traceability"))
        quality_policy = _quality_policy(rule, base=base_quality_policy)
        results = index.query(query, top_k=top_k)
        top = results[0] if results else {}
        object_hit_top1 = bool(_object_values(top) & expected_objects) if expected_objects else True
        object_hit_topk = any(_object_values(item) & expected_objects for item in results) if expected_objects else True
        level_hit_top1 = not expected_level or str(top.get("index_level") or "") == expected_level
        level_hit_topk = not expected_level or any(str(item.get("index_level") or "") == expected_level for item in results)
        action_hit_top1 = bool(_action_values(top) & expected_actions) if expected_actions else True
        action_hit_topk = any(_action_values(item) & expected_actions for item in results) if expected_actions else True
        id_hit_top1 = _id_hit(top, expected_segment_ids, expected_micro_segment_ids) if expected_segment_ids or expected_micro_segment_ids else True
        id_hit_topk = any(_id_hit(item, expected_segment_ids, expected_micro_segment_ids) for item in results) if expected_segment_ids or expected_micro_segment_ids else True
        time_hit_top1 = _overlaps_expected_time_window(top, expected_time_window) if expected_time_window else True
        time_hit_topk = any(_overlaps_expected_time_window(item, expected_time_window) for item in results) if expected_time_window else True
        traceability_hit_top1 = _has_traceability(top) if require_traceability else True
        traceability_hit_topk = any(_has_traceability(item) for item in results) if require_traceability else True
        fallback_top1_valid = bool(top and _fallback_valid(top, rule))
        fallback_valid = any(_fallback_valid(item, rule) for item in results)
        top1_expected_hit = _result_matches_expectations(
            top,
            expected_objects=expected_objects,
            expected_actions=expected_actions,
            expected_segment_ids=expected_segment_ids,
            expected_micro_segment_ids=expected_micro_segment_ids,
            expected_level=expected_level,
            expected_time_window=expected_time_window,
            require_traceability=require_traceability,
        )
        expected_matches = [
            item
            for item in results
            if _result_matches_expectations(
                item,
                expected_objects=expected_objects,
                expected_actions=expected_actions,
                expected_segment_ids=expected_segment_ids,
                expected_micro_segment_ids=expected_micro_segment_ids,
                expected_level=expected_level,
                expected_time_window=expected_time_window,
                require_traceability=require_traceability,
            )
        ]
        accepted_matches = [item for item in expected_matches if not _quality_failures(item, quality_policy)]
        top1_quality_failures = _quality_failures(top, quality_policy) if top1_expected_hit else []
        top1_quality_hit = bool(top1_expected_hit and not top1_quality_failures)
        topk_quality_hit = bool(accepted_matches) if expected_matches else False
        top1_hit = (
            object_hit_top1
            and level_hit_top1
            and action_hit_top1
            and id_hit_top1
            and time_hit_top1
            and traceability_hit_top1
        ) or fallback_top1_valid
        topk_hit = (
            bool(expected_matches)
        ) or fallback_valid
        acceptance_hit = bool(accepted_matches) or fallback_valid
        acceptance_hits += int(acceptance_hit)
        top1_hits += int(top1_hit)
        topk_hits += int(topk_hit)
        quality_hits += int(topk_quality_hit or fallback_valid)
        expected_object_hits += int(object_hit_topk)
        expected_index_level_hits += int(level_hit_topk)
        expected_action_hits += int(action_hit_topk)
        expected_id_hits += int(id_hit_topk)
        expected_time_window_hits += int(time_hit_topk)
        traceability_hits += int(traceability_hit_topk)
        valid_fallbacks += int(fallback_valid)
        failure_explanation = _failure_explanation(
            object_hit=object_hit_topk,
            level_hit=level_hit_topk,
            action_hit=action_hit_topk,
            id_hit=id_hit_topk,
            time_hit=time_hit_topk,
            traceability_hit=traceability_hit_topk,
            quality_hit=topk_quality_hit or fallback_valid,
            acceptance_hit=acceptance_hit,
            top1_quality_failures=top1_quality_failures,
            fallback_valid=fallback_valid,
        )
        rows.append(
            {
                "query_id": rule.get("query_id"),
                "query": query,
                "benchmark_category": rule.get("benchmark_category"),
                "binding_source": rule.get("binding_source"),
                "human_verified": bool(rule.get("human_verified")),
                "manual_review_status": rule.get("manual_review_status"),
                "top_k": top_k,
                "top1_hit": top1_hit,
                "topk_hit": topk_hit,
                "expected_object_hit": object_hit_topk,
                "expected_index_level_hit": level_hit_topk,
                "expected_action_hit": action_hit_topk,
                "expected_id_hit": id_hit_topk,
                "expected_time_window_hit": time_hit_topk,
                "traceability_hit": traceability_hit_topk,
                "quality_hit": topk_quality_hit or fallback_valid,
                "acceptance_hit": acceptance_hit,
                "fallback_reason_valid": fallback_valid,
                "matching_result_count": len(expected_matches),
                "accepted_result_count": len(accepted_matches),
                "top1_quality_failures": top1_quality_failures,
                "failure_explanation": failure_explanation,
                "quality_policy": quality_policy,
                "top_result": {
                    "index_level": top.get("index_level"),
                    "segment_id": top.get("segment_id"),
                    "micro_segment_id": top.get("micro_segment_id"),
                    "expected_segment_ids": sorted(expected_segment_ids),
                    "expected_micro_segment_ids": sorted(expected_micro_segment_ids),
                    "primary_object": top.get("primary_object"),
                    "detected_objects": top.get("detected_objects"),
                    "action_type": top.get("action_type"),
                    "interaction_type": top.get("interaction_type"),
                    "global_start_time": top.get("global_start_time"),
                    "global_end_time": top.get("global_end_time"),
                    "score": top.get("score"),
                    "evidence_level": top.get("evidence_level"),
                    "quality": top.get("quality"),
                    "quality_warnings": _quality_warnings(top),
                    "coverage_signal_grade": _coverage_signal_grade(top),
                    "coverage_backfill": _coverage_backfill(top),
                    "limitations": top.get("limitations"),
                    "rerank_reasons": top.get("rerank_reasons"),
                    "keyframes": top.get("keyframes"),
                    "interaction_keyframes": top.get("interaction_keyframes"),
                    "asset_bindings": top.get("asset_bindings"),
                }
                if top
                else None,
                "accepted_result": _accepted_result_summary(accepted_matches[0]) if accepted_matches else None,
            }
        )
    total = len(rows)
    failed_queries = [
        {
            "query": row["query"],
            "query_id": row.get("query_id"),
            "benchmark_category": row.get("benchmark_category"),
            "top1_hit": row["top1_hit"],
            "topk_hit": row["topk_hit"],
            "expected_id_hit": row.get("expected_id_hit"),
            "quality_hit": row["quality_hit"],
            "acceptance_hit": row["acceptance_hit"],
            "top1_quality_failures": row["top1_quality_failures"],
            "failure_explanation": row.get("failure_explanation"),
            "top_result": row.get("top_result"),
        }
        for row in rows
        if not row["acceptance_hit"]
    ]
    result: dict[str, Any] = {
        "schema_version": QUERY_VALIDATION_SCHEMA_VERSION,
        "generated_at": datetime.now().isoformat(),
        "acceptance_name": config.get("acceptance_name"),
        "benchmark_version": config.get("benchmark_version"),
        "benchmark_binding_mode": config.get("benchmark_binding_mode"),
        "human_verified_query_count": config.get("human_verified_query_count"),
        "gold_benchmark_path": config.get("gold_benchmark_path"),
        "session_dir": str(session),
        "index_dir": str(index_dir),
        "config": str(config_path),
        "query_count": total,
        "acceptance_hit_rate": acceptance_hits / total if total else 0.0,
        "query_hit_rate": topk_hits / total if total else 0.0,
        "top1_hit_rate": top1_hits / total if total else 0.0,
        "topk_hit_rate": topk_hits / total if total else 0.0,
        "quality_hit_rate": quality_hits / total if total else 0.0,
        "expected_object_hit_rate": expected_object_hits / total if total else 0.0,
        "expected_index_level_hit_rate": expected_index_level_hits / total if total else 0.0,
        "expected_action_hit_rate": expected_action_hits / total if total else 0.0,
        "expected_id_hit_rate": expected_id_hits / total if total else 0.0,
        "expected_time_window_hit_rate": expected_time_window_hits / total if total else 0.0,
        "traceability_hit_rate": traceability_hits / total if total else 0.0,
        "valid_fallback_count": valid_fallbacks,
        "thresholds": thresholds,
        "quality_policy": base_quality_policy,
        "failed_query_count": len(failed_queries),
        "failed_queries": failed_queries,
        "queries": rows,
    }
    threshold_failures = _threshold_failures(result, thresholds)
    result["threshold_failures"] = threshold_failures
    result["status"] = "pass" if not threshold_failures else "fail"
    target = Path(output_path) if output_path else session / "evaluation" / "query_validation.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _validation_thresholds(config: Mapping[str, Any]) -> dict[str, float]:
    thresholds = dict(DEFAULT_ACCEPTANCE_THRESHOLDS)
    raw = config.get("thresholds") if isinstance(config.get("thresholds"), Mapping) else {}
    for key, value in raw.items():
        if key in THRESHOLD_RATE_FIELDS:
            thresholds[str(key)] = _float(value)
    return thresholds


def _threshold_failures(result: Mapping[str, Any], thresholds: Mapping[str, float]) -> list[dict[str, Any]]:
    failures = []
    for threshold_key, metric_key in THRESHOLD_RATE_FIELDS.items():
        if threshold_key not in thresholds:
            continue
        actual = _float(result.get(metric_key))
        minimum = _float(thresholds.get(threshold_key))
        if actual < minimum:
            failures.append(
                {
                    "threshold": threshold_key,
                    "metric": metric_key,
                    "actual": round(actual, 6),
                    "minimum": round(minimum, 6),
                }
            )
    return failures


def _quality_policy(source: Mapping[str, Any], *, base: Mapping[str, Any] | None = None) -> dict[str, Any]:
    policy = dict(base or {})
    raw = source.get("quality_policy")
    if not isinstance(raw, Mapping):
        raw = source.get("result_quality")
    if isinstance(raw, Mapping):
        policy.update(dict(raw))
    return policy


def _result_matches_expectations(
    result: dict[str, Any],
    *,
    expected_objects: set[str],
    expected_actions: set[str],
    expected_segment_ids: set[str],
    expected_micro_segment_ids: set[str],
    expected_level: str,
    expected_time_window: dict[str, Any] | None,
    require_traceability: bool,
) -> bool:
    if not result:
        return False
    if expected_objects and not (_object_values(result) & expected_objects):
        return False
    if expected_actions and not (_action_values(result) & expected_actions):
        return False
    if (expected_segment_ids or expected_micro_segment_ids) and not _id_hit(result, expected_segment_ids, expected_micro_segment_ids):
        return False
    if expected_level and str(result.get("index_level") or "") != expected_level:
        return False
    if expected_time_window and not _overlaps_expected_time_window(result, expected_time_window):
        return False
    if require_traceability and not _has_traceability(result):
        return False
    return True


def _id_hit(result: Mapping[str, Any], expected_segment_ids: set[str], expected_micro_segment_ids: set[str]) -> bool:
    if not result:
        return False
    segment_values = {
        str(result.get("segment_id") or ""),
        str(result.get("parent_segment_id") or ""),
    }
    micro_values = {str(result.get("micro_segment_id") or "")}
    metadata = result.get("metadata") if isinstance(result.get("metadata"), Mapping) else {}
    segment_values.update({str(metadata.get("segment_id") or ""), str(metadata.get("parent_segment_id") or "")})
    micro_values.add(str(metadata.get("micro_segment_id") or ""))
    if expected_micro_segment_ids and micro_values & expected_micro_segment_ids:
        return True
    if expected_segment_ids and segment_values & expected_segment_ids:
        return True
    return False


def _quality_failures(result: Mapping[str, Any], policy: Mapping[str, Any]) -> list[str]:
    if not result or not policy:
        return []
    failures: list[str] = []
    score = _float(result.get("score"))
    min_score = policy.get("min_score")
    if min_score is not None and score < _float(min_score):
        failures.append(f"score_below_min:{score:.6f}<{_float(min_score):.6f}")

    evidence_level = _norm(result.get("evidence_level"))
    min_evidence_level = _norm(policy.get("min_evidence_level"))
    if min_evidence_level and EVIDENCE_LEVEL_RANK.get(evidence_level, -1) < EVIDENCE_LEVEL_RANK.get(min_evidence_level, 0):
        failures.append(f"evidence_level_below_min:{evidence_level or 'missing'}<{min_evidence_level}")

    coverage_grade = _coverage_signal_grade(result)
    min_grade = _norm(policy.get("min_coverage_signal_grade"))
    if min_grade and COVERAGE_SIGNAL_GRADE_RANK.get(_norm(coverage_grade), -1) < COVERAGE_SIGNAL_GRADE_RANK.get(min_grade, 0):
        failures.append(f"coverage_signal_grade_below_min:{coverage_grade or 'missing'}<{min_grade}")

    disallowed_grades = {_norm(item) for item in _list(policy.get("disallowed_coverage_signal_grades"))}
    if coverage_grade and _norm(coverage_grade) in disallowed_grades:
        failures.append(f"disallowed_coverage_signal_grade:{coverage_grade}")

    if policy.get("allow_coverage_backfill") is False and _coverage_backfill(result):
        failures.append("coverage_backfill_not_allowed")

    warnings = {_norm(item) for item in _quality_warnings(result)}
    disallowed_warnings = {_norm(item) for item in _list(policy.get("disallowed_quality_warnings"))}
    for warning in sorted(warnings & disallowed_warnings):
        failures.append(f"disallowed_quality_warning:{warning}")

    if policy.get("allow_very_low_signal") is False:
        if _norm(coverage_grade) == "very_low_signal_yolo_candidate" or "very_low_signal_yolo_candidate" in warnings:
            failures.append("very_low_signal_not_allowed")
    if policy.get("allow_single_frame_evidence") is False:
        if _norm(coverage_grade) == "single_frame_yolo_candidate" or "single_frame_coverage_candidate" in warnings:
            failures.append("single_frame_evidence_not_allowed")

    limitation_text = " ".join(str(item) for item in _list(result.get("limitations"))).casefold()
    for token in _list(policy.get("disallowed_limitations")):
        needle = str(token or "").strip().casefold()
        if needle and needle in limitation_text:
            failures.append(f"disallowed_limitation:{needle}")
    return _ordered_unique(failures)


def _failure_explanation(
    *,
    object_hit: bool,
    level_hit: bool,
    action_hit: bool,
    id_hit: bool,
    time_hit: bool,
    traceability_hit: bool,
    quality_hit: bool,
    acceptance_hit: bool,
    top1_quality_failures: list[str],
    fallback_valid: bool,
) -> list[str]:
    if acceptance_hit:
        return ["accepted by expected match"] if not fallback_valid else ["accepted by configured fallback rule"]
    reasons: list[str] = []
    if not object_hit:
        reasons.append("no top-k result matched expected object terms")
    if not action_hit:
        reasons.append("no top-k result matched expected action terms")
    if not level_hit:
        reasons.append("no top-k result matched expected index level")
    if not id_hit:
        reasons.append("no top-k result matched expected segment or micro id")
    if not time_hit:
        reasons.append("no top-k result overlapped expected time window")
    if not traceability_hit:
        reasons.append("matching candidates lacked clip/keyframe traceability")
    if not quality_hit:
        reasons.append("matching candidates failed result quality policy")
    reasons.extend(top1_quality_failures)
    return _ordered_unique(reasons or ["no accepted retrieval result"])


def _accepted_result_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "index_level": result.get("index_level"),
        "segment_id": result.get("segment_id"),
        "micro_segment_id": result.get("micro_segment_id"),
        "primary_object": result.get("primary_object"),
        "action_type": result.get("action_type"),
        "interaction_type": result.get("interaction_type"),
        "global_start_time": result.get("global_start_time"),
        "global_end_time": result.get("global_end_time"),
        "score": result.get("score"),
        "evidence_level": result.get("evidence_level"),
        "quality": result.get("quality"),
        "quality_warnings": _quality_warnings(result),
        "coverage_signal_grade": _coverage_signal_grade(result),
        "coverage_backfill": _coverage_backfill(result),
    }


def _object_values(result: dict[str, Any]) -> set[str]:
    values = {
        _norm(result.get("primary_object")),
        _norm(result.get("primary_object_family")),
        *{_norm(item) for item in _list(result.get("detected_objects"))},
        *{_norm(item) for item in _list(result.get("visual_keywords"))},
    }
    for key in ("interaction_events", "yolo_interactions", "interaction_keyframes", "yolo_evidence"):
        for item in _list(result.get(key)):
            if isinstance(item, dict):
                values.add(_norm(item.get("object_label")))
                values.add(_norm(item.get("object_name")))
                values.add(_norm(item.get("primary_object")))
                values.add(_norm(item.get("hand_label")))
                values.update(_norm(label) for label in _list(item.get("labels")))
                for detection in _list(item.get("detections")):
                    if isinstance(detection, dict):
                        values.add(_norm(detection.get("label")))
    values.update(_norm(part) for part in str(result.get("index_text") or "").replace(",", " ").replace(":", " ").split())
    values.discard("")
    return values


def _action_values(result: dict[str, Any]) -> set[str]:
    values = {
        _norm(result.get("action_type")),
        _norm(result.get("interaction_type")),
        *{_norm(item) for item in _list(result.get("visual_keywords"))},
    }
    for key in ("interaction_events", "yolo_interactions", "interaction_keyframes", "yolo_evidence"):
        for item in _list(result.get(key)):
            if isinstance(item, dict):
                values.add(_norm(item.get("interaction")))
                values.add(_norm(item.get("interaction_type")))
    text = str(result.get("index_text") or "").casefold()
    if "weigh" in text or "称量" in text:
        values.add("weighing")
    if "pipet" in text or "移液" in text or "加样" in text:
        values.add("pipetting")
    if "sample" in text or "样品" in text:
        values.add("sample_handling")
    if "hand" in text and "object" in text:
        values.add("hand_object_interaction")
    values.discard("")
    return values


def _overlaps_expected_time_window(result: dict[str, Any], expected_time_window: dict[str, Any] | None) -> bool:
    if not expected_time_window or not result:
        return False
    expected_start = _parse_time(expected_time_window.get("start"))
    expected_end = _parse_time(expected_time_window.get("end"))
    result_start = _parse_time(result.get("global_start_time") or result.get("start_time"))
    result_end = _parse_time(result.get("global_end_time") or result.get("end_time")) or result_start
    if result_start is None:
        return False
    if expected_start and result_end and result_end < expected_start:
        return False
    if expected_end and result_start > expected_end:
        return False
    return True


def _has_traceability(result: dict[str, Any]) -> bool:
    if not result:
        return False
    has_anchor = bool(result.get("micro_segment_id") or result.get("segment_id"))
    has_keyframe = bool(
        _list(result.get("keyframes"))
        or _list(result.get("interaction_keyframes"))
        or _asset_binding_has("keyframe", result.get("asset_bindings"))
    )
    has_clip = bool(
        result.get("third_person_clip")
        or result.get("first_person_clip")
        or _asset_binding_has("clip", result.get("asset_bindings"))
        or _asset_binding_has("video", result.get("asset_bindings"))
    )
    return has_anchor and (has_keyframe or has_clip)


def _asset_binding_has(token: str, bindings: Any) -> bool:
    needle = _norm(token)
    for binding in _list(bindings):
        text = json.dumps(binding, ensure_ascii=False).casefold() if isinstance(binding, dict) else str(binding).casefold()
        if needle in _norm(text):
            return True
    return False


def _coverage_signal_grade(result: Mapping[str, Any]) -> str:
    evidence = result.get("evidence") if isinstance(result.get("evidence"), Mapping) else {}
    return str(result.get("coverage_signal_grade") or evidence.get("coverage_signal_grade") or "")


def _coverage_backfill(result: Mapping[str, Any]) -> bool:
    evidence = result.get("evidence") if isinstance(result.get("evidence"), Mapping) else {}
    if bool(result.get("coverage_backfill") or evidence.get("coverage_backfill")):
        return True
    text = " ".join(str(item) for item in _list(result.get("limitations")) + _list(evidence.get("limitations"))).casefold()
    return "coverage backfill" in text


def _quality_warnings(result: Mapping[str, Any]) -> list[str]:
    quality = result.get("quality") if isinstance(result.get("quality"), Mapping) else {}
    warnings = [
        *[str(item) for item in _list(result.get("quality_warnings"))],
        *[str(item) for item in _list(quality.get("warnings"))],
    ]
    return _ordered_unique(warnings)


def _float(value: Any) -> float:
    try:
        if value in (None, ""):
            return 0.0
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


def _parse_time(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None

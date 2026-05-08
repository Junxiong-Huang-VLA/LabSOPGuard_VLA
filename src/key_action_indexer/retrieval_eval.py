from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .query_validation import validate_queries
from .reviewed_dataset import reviewed_metadata_path
from .schemas import read_jsonl


DEFAULT_QUERY_COUNT = 50
GOLD_QUERY_BENCHMARK_FILENAME = "gold_query_benchmark.json"
TREND_JSONL_FILENAME = "retrieval_eval_trend.jsonl"
TREND_MD_FILENAME = "retrieval_eval_trend.md"

FIXED_CHINESE_QUERY_BENCHMARK: tuple[tuple[str, list[str], list[str], str], ...] = (
    ("称量样品", ["balance", "sample", "weighing_paper"], ["weighing"], "micro_segment"),
    ("打开离心管盖", ["tube", "centrifuge_tube", "cap", "lid"], ["open", "container_state_change"], "micro_segment"),
    ("移液到孔板", ["pipette", "plate", "well_plate", "liquid"], ["pipetting", "sample_adding"], "micro_segment"),
    ("记录天平读数", ["balance", "display", "readout"], ["recording", "weighing"], "segment"),
    ("拿起样品瓶", ["sample_bottle", "bottle"], ["hand_object_interaction", "pick_up"], "micro_segment"),
    ("放下移液枪", ["pipette"], ["release", "hand_object_interaction"], "micro_segment"),
    ("手接触容器", ["container", "tube", "bottle"], ["hand_object_interaction"], "micro_segment"),
    ("试剂转移", ["reagent", "pipette", "liquid"], ["liquid_transfer", "pipetting"], "micro_segment"),
    ("观察液面变化", ["liquid", "meniscus", "container"], ["liquid_state_change", "observe"], "micro_segment"),
    ("关闭管盖", ["tube", "cap", "lid"], ["close", "container_state_change"], "micro_segment"),
    ("混匀离心管", ["tube", "centrifuge_tube"], ["mixing", "shake"], "micro_segment"),
    ("取出枪头", ["pipette_tip", "tip_box"], ["pick_up", "hand_object_interaction"], "micro_segment"),
    ("安装移液枪枪头", ["pipette", "pipette_tip"], ["attach_tip", "pipetting"], "micro_segment"),
    ("吸取液体", ["pipette", "liquid", "tube"], ["aspirate", "pipetting"], "micro_segment"),
    ("排出液体", ["pipette", "liquid", "plate"], ["dispense", "pipetting"], "micro_segment"),
    ("移动孔板", ["well_plate", "plate"], ["hand_object_interaction", "move"], "micro_segment"),
    ("刮取样品", ["spatula", "sample"], ["spatula_interaction", "sample_handling"], "micro_segment"),
    ("把样品加入称量纸", ["sample", "weighing_paper", "balance"], ["sample_adding", "weighing"], "micro_segment"),
    ("调整容器位置", ["container", "bottle", "tube"], ["move", "hand_object_interaction"], "micro_segment"),
    ("查看仪器面板", ["display", "panel", "equipment"], ["panel_reading", "recording"], "segment"),
    ("按下设备按钮", ["button", "equipment_panel"], ["equipment_control_state", "press"], "micro_segment"),
    ("旋转旋钮", ["knob", "equipment_panel"], ["equipment_control_state", "rotate"], "micro_segment"),
    ("打开试剂瓶", ["reagent_bottle", "bottle", "cap"], ["open", "bottle_interaction"], "micro_segment"),
    ("盖回试剂瓶盖", ["reagent_bottle", "bottle", "cap"], ["close", "bottle_interaction"], "micro_segment"),
    ("从瓶中取液", ["reagent_bottle", "pipette", "liquid"], ["aspirate", "pipetting"], "micro_segment"),
    ("向离心管加样", ["tube", "pipette", "liquid"], ["sample_adding", "pipetting"], "micro_segment"),
    ("向孔板加样", ["well_plate", "pipette", "liquid"], ["sample_adding", "pipetting"], "micro_segment"),
    ("确认样品标签", ["label", "sample_bottle", "tube"], ["label_check", "observe"], "segment"),
    ("擦拭工作台", ["bench", "wipe"], ["cleaning"], "segment"),
    ("等待反应", ["tube", "plate", "sample"], ["waiting", "incubation"], "segment"),
    ("拿起离心管架", ["tube_rack", "tube"], ["hand_object_interaction", "pick_up"], "micro_segment"),
    ("把离心管放入管架", ["tube_rack", "tube"], ["place", "hand_object_interaction"], "micro_segment"),
    ("检查颜色变化", ["liquid", "container", "color"], ["container_color_change", "observe"], "micro_segment"),
    ("读取体积刻度", ["graduated_container", "liquid", "scale"], ["liquid_level_measured", "observe"], "micro_segment"),
    ("打开电子天平门", ["balance", "door"], ["open", "weighing"], "micro_segment"),
    ("关闭电子天平门", ["balance", "door"], ["close", "weighing"], "micro_segment"),
    ("把容器放到天平上", ["balance", "container"], ["weighing", "place"], "micro_segment"),
    ("从天平上移走容器", ["balance", "container"], ["weighing", "move"], "micro_segment"),
    ("更换移液枪量程", ["pipette", "dial"], ["adjust", "pipetting"], "micro_segment"),
    ("丢弃枪头", ["pipette_tip", "waste"], ["discard", "pipetting"], "micro_segment"),
    ("抓取试管", ["tube", "hand"], ["hand_object_interaction", "pick_up"], "micro_segment"),
    ("松开试管", ["tube", "hand"], ["release", "hand_object_interaction"], "micro_segment"),
    ("观察沉淀", ["precipitate", "tube", "liquid"], ["observe", "liquid_state_change"], "micro_segment"),
    ("记录实验现象", ["notebook", "display", "sample"], ["recording", "observe"], "segment"),
    ("扫描关键帧证据", ["keyframe", "hand", "object"], ["hand_object_interaction"], "micro_segment"),
    ("查找手和移液枪接触", ["hand", "pipette"], ["hand_object_interaction", "pipetting"], "micro_segment"),
    ("查找手和瓶子接触", ["hand", "bottle"], ["hand_object_interaction", "bottle_interaction"], "micro_segment"),
    ("查找手和天平接触", ["hand", "balance"], ["hand_object_interaction", "weighing"], "micro_segment"),
    ("查找液体流动证据", ["liquid", "pipette", "container"], ["liquid_transfer", "liquid_flow_observed"], "micro_segment"),
    ("导出已审核片段", ["segment", "reviewed", "evidence"], ["reviewed_export"], "segment"),
)


def build_default_chinese_query_eval_config(
    session_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    query_count: int = DEFAULT_QUERY_COUNT,
) -> dict[str, Any]:
    session = Path(session_dir)
    metadata = session / "metadata"
    target = Path(output_path) if output_path is not None else metadata / "default_chinese_query_validation.json"
    selected_count = max(1, min(query_count, len(FIXED_CHINESE_QUERY_BENCHMARK)))
    gold = build_gold_query_benchmark(session, query_count=selected_count)
    rules = list(gold.get("queries") or [])[:selected_count]
    config = {
        "schema_version": "key_action_query_validation.v3",
        "acceptance_name": "fixed_50_chinese_key_action_retrieval_eval",
        "benchmark_version": "2026-05-08.fixed50",
        "gold_benchmark_path": str(metadata / GOLD_QUERY_BENCHMARK_FILENAME),
        "benchmark_binding_mode": gold.get("binding_mode"),
        "human_verified_query_count": gold.get("human_verified_query_count"),
        "id_authoritative": bool(gold.get("id_authoritative")),
        "thresholds": {
            "min_acceptance_hit_rate": 0.4,
            "min_topk_hit_rate": 0.4,
            "min_traceability_hit_rate": 0.75,
            "min_quality_hit_rate": 0.4,
            "min_expected_id_hit_rate": 0.75,
        },
        "quality_policy": {
            "allow_very_low_signal": False,
            "allow_single_frame_evidence": True,
            "min_evidence_level": "weak_visual_evidence",
        },
        "queries": rules,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return config


def build_gold_query_benchmark(
    session_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    query_count: int = DEFAULT_QUERY_COUNT,
    overwrite: bool = False,
) -> dict[str, Any]:
    session = Path(session_dir)
    metadata = session / "metadata"
    target = Path(output_path) if output_path is not None else metadata / GOLD_QUERY_BENCHMARK_FILENAME
    if target.exists() and not overwrite:
        try:
            data = json.loads(target.read_text(encoding="utf-8-sig"))
            if isinstance(data, dict) and isinstance(data.get("queries"), list) and len(data["queries"]) >= query_count:
                data["benchmark_path"] = str(target)
                return data
        except (OSError, json.JSONDecodeError):
            pass
    micros = _read_jsonl(reviewed_metadata_path(session, "micro_segments.jsonl"))
    micro_vectors = _read_jsonl(reviewed_metadata_path(session, "micro_vector_metadata.jsonl"))
    segments = _read_jsonl(reviewed_metadata_path(session, "key_action_segments.jsonl"))
    segment_vectors = _read_jsonl(reviewed_metadata_path(session, "vector_metadata.jsonl"))
    micro_pool = micro_vectors or micros
    segment_pool = segment_vectors or segments
    rules = []
    selected = FIXED_CHINESE_QUERY_BENCHMARK[: max(1, min(query_count, len(FIXED_CHINESE_QUERY_BENCHMARK)))]
    for index, (query, objects, actions, level) in enumerate(selected, start=1):
        rule = _bind_rule(query, objects, actions, level, micro_pool, segment_pool)
        rule.update(
            {
                "query_id": f"gold_cn_{index:03d}",
                "binding_source": "bootstrap_auto",
                "human_verified": False,
                "manual_review_status": "needs_human_verification",
                "benchmark_category": _category_for_actions(actions, query),
            }
        )
        rules.append(rule)
    categories = Counter(str(rule.get("benchmark_category") or "uncategorized") for rule in rules)
    payload = {
        "schema_version": "key_action_gold_query_benchmark.v1",
        "benchmark_version": "2026-05-08.fixed50.human_gt",
        "session_dir": str(session),
        "query_count": len(rules),
        "human_verified_query_count": sum(1 for rule in rules if rule.get("human_verified")),
        "binding_mode": "bootstrap_pending_human_verification",
        "id_authoritative": False,
        "note": "Expected segment/micro ids are human GT slots. bootstrap_auto bindings are only initial anchors until human_verified=true.",
        "categories": dict(sorted(categories.items())),
        "queries": rules,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["benchmark_path"] = str(target)
    return payload


def confirm_gold_query_benchmark(
    session_dir: str | Path,
    *,
    reviewer: str = "manual_reviewer",
    note: str = "Human-verified against the current reviewed release.",
    decisions_path: str | Path | None = None,
    output_path: str | Path | None = None,
    query_count: int = DEFAULT_QUERY_COUNT,
) -> dict[str, Any]:
    if decisions_path is None:
        raise ValueError("confirm-gold-query-benchmark requires a human decision file; refusing to auto-mark human_verified=true")
    session = Path(session_dir)
    target = Path(output_path) if output_path is not None else session / "metadata" / GOLD_QUERY_BENCHMARK_FILENAME
    gold = build_gold_query_benchmark(session, output_path=target, query_count=query_count)
    queries = [dict(row) for row in gold.get("queries") or []][: max(1, min(query_count, DEFAULT_QUERY_COUNT))]
    decisions = _gold_query_decisions_by_id(Path(decisions_path))
    now = _now()
    confirmed = []
    unresolved = []
    for row in queries:
        query_id = str(row.get("query_id") or "")
        decision = decisions.get(query_id) or decisions.get(str(row.get("query") or ""))
        if not decision:
            row["human_verified"] = False
            row["manual_review_status"] = "needs_human_verification"
            row["verification_note"] = "No human decision row was provided for this query."
            unresolved.append(query_id or row.get("query"))
            confirmed.append(row)
            continue
        status = _normalize_gold_decision(decision.get("decision") or decision.get("status"))
        if status != "approved":
            row["human_verified"] = False
            row["manual_review_status"] = "rejected" if status == "rejected" else "needs_more_review"
            row["verification_note"] = decision.get("note") or note
            row["verified_by"] = decision.get("reviewer") or reviewer
            row["verified_at"] = now
            unresolved.append(query_id or row.get("query"))
            confirmed.append(row)
            continue
        segment_ids = _string_list(decision.get("expected_segment_ids") or decision.get("segment_ids") or decision.get("segment_id"))
        micro_ids = _string_list(decision.get("expected_micro_segment_ids") or decision.get("micro_segment_ids") or decision.get("micro_segment_id"))
        if not segment_ids and not micro_ids:
            row["human_verified"] = False
            row["manual_review_status"] = "needs_more_review"
            row["verification_note"] = "Approved decision is missing expected segment or micro ids."
            row["verified_by"] = decision.get("reviewer") or reviewer
            row["verified_at"] = now
            unresolved.append(query_id or row.get("query"))
            confirmed.append(row)
            continue
        if segment_ids:
            row["expected_segment_ids"] = segment_ids
        if micro_ids:
            row["expected_micro_segment_ids"] = micro_ids
        elif "expected_micro_segment_ids" in row:
            row.pop("expected_micro_segment_ids", None)
        if decision.get("expected_index_level"):
            row["expected_index_level"] = str(decision["expected_index_level"])
        if isinstance(decision.get("expected_time_window"), Mapping):
            row["expected_time_window"] = dict(decision["expected_time_window"])
        row["human_verified"] = True
        row["manual_review_status"] = "approved"
        row["binding_source"] = "human_confirmed_decision_file"
        row["id_authoritative"] = True
        row["verified_by"] = decision.get("reviewer") or reviewer
        row["verified_at"] = now
        row["verification_note"] = decision.get("note") or note
        confirmed.append(row)
    categories = Counter(str(rule.get("benchmark_category") or "uncategorized") for rule in confirmed)
    human_verified_count = sum(1 for rule in confirmed if rule.get("human_verified"))
    fully_verified = human_verified_count == len(confirmed)
    payload = {
        **{key: value for key, value in gold.items() if key not in {"queries", "categories", "human_verified_query_count", "binding_mode", "id_authoritative"}},
        "schema_version": "key_action_gold_query_benchmark.v1",
        "benchmark_version": "2026-05-08.fixed50.human_review_file",
        "session_dir": str(session),
        "query_count": len(confirmed),
        "human_verified_query_count": human_verified_count,
        "binding_mode": "human_verified_review_file" if fully_verified else "partial_human_verified_review_file",
        "id_authoritative": fully_verified,
        "verified_by": reviewer,
        "verified_at": now,
        "manual_review_status": "approved" if fully_verified else "partial",
        "unresolved_query_ids": unresolved,
        "source_decisions_path": str(decisions_path),
        "categories": dict(sorted(categories.items())),
        "queries": confirmed,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["benchmark_path"] = str(target)
    return payload


def run_default_chinese_query_eval(
    session_dir: str | Path,
    *,
    config_path: str | Path | None = None,
    output_path: str | Path | None = None,
    query_count: int = DEFAULT_QUERY_COUNT,
) -> dict[str, Any]:
    session = Path(session_dir)
    config_target = Path(config_path) if config_path is not None else session / "metadata" / "default_chinese_query_validation.json"
    if config_path is None or not config_target.exists():
        build_default_chinese_query_eval_config(session, output_path=config_target, query_count=query_count)
    output = Path(output_path) if output_path is not None else session / "evaluation" / "default_chinese_query_validation.json"
    result = validate_queries(session, config_target, output)
    result["category_summary"] = _category_summary(result.get("queries") or [])
    result["gold_benchmark_path"] = str(session / "metadata" / GOLD_QUERY_BENCHMARK_FILENAME)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _append_trend(session, result)
    return result


def _bind_rule(
    query: str,
    objects: list[str],
    actions: list[str],
    expected_level: str,
    micros: list[Mapping[str, Any]],
    segments: list[Mapping[str, Any]],
) -> dict[str, Any]:
    micro = _best_match(micros, objects, actions)
    segment = _best_match(segments, objects, actions)
    fallback_micro = micros[0] if micros else None
    fallback_segment = segments[0] if segments else None
    selected_micro = micro or fallback_micro
    selected_segment = segment or fallback_segment
    rule: dict[str, Any] = {
        "query": query,
        "top_k": 3,
        "expected_objects": objects,
        "expected_actions": actions,
        "expected_index_level": expected_level,
        "require_traceability": True,
        "allow_parent_fallback_when_insufficient_evidence": True,
        "benchmark_category": _category_for_actions(actions, query),
        "binding_status": "semantic_match" if (micro or segment) else "fallback_session_anchor",
    }
    if selected_micro:
        micro_id = _first_text(selected_micro, "micro_segment_id", "id")
        parent_id = _first_text(selected_micro, "parent_segment_id", "segment_id")
        if micro_id:
            rule["expected_micro_segment_ids"] = [micro_id]
        if parent_id:
            rule["expected_segment_ids"] = [parent_id]
    elif selected_segment:
        segment_id = _first_text(selected_segment, "segment_id", "id")
        if segment_id:
            rule["expected_segment_ids"] = [segment_id]
    return rule


def _best_match(rows: list[Mapping[str, Any]], objects: list[str], actions: list[str]) -> Mapping[str, Any] | None:
    expected_objects = {_norm(item) for item in objects}
    expected_actions = {_norm(item) for item in actions}
    best: tuple[int, Mapping[str, Any]] | None = None
    for row in rows:
        text = _cached_row_text(row)
        score = sum(1 for item in expected_objects if item and item in text) + sum(2 for item in expected_actions if item and item in text)
        if score <= 0:
            continue
        if best is None or score > best[0]:
            best = (score, row)
    return best[1] if best else None


def _append_trend(session: Path, result: Mapping[str, Any]) -> None:
    evaluation = session / "evaluation"
    evaluation.mkdir(parents=True, exist_ok=True)
    row = {
        "generated_at": result.get("generated_at"),
        "status": result.get("status"),
        "query_count": result.get("query_count"),
        "top1_hit_rate": result.get("top1_hit_rate"),
        "top3_hit_rate": result.get("topk_hit_rate"),
        "expected_id_hit_rate": result.get("expected_id_hit_rate"),
        "quality_hit_rate": result.get("quality_hit_rate"),
        "acceptance_hit_rate": result.get("acceptance_hit_rate"),
        "failed_query_count": result.get("failed_query_count"),
        "threshold_failure_count": len(result.get("threshold_failures") or []),
        "category_summary": result.get("category_summary") or {},
    }
    trend_path = evaluation / TREND_JSONL_FILENAME
    with trend_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    rows = _read_jsonl(trend_path)
    lines = [
        "| generated_at | status | queries | top1 | top3 | expected_id | quality | failed |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in rows[-20:]:
        lines.append(
            "| {generated_at} | {status} | {query_count} | {top1:.3f} | {top3:.3f} | {ids:.3f} | {quality:.3f} | {failed} |".format(
                generated_at=item.get("generated_at") or "",
                status=item.get("status") or "",
                query_count=item.get("query_count") or 0,
                top1=float(item.get("top1_hit_rate") or 0.0),
                top3=float(item.get("top3_hit_rate") or 0.0),
                ids=float(item.get("expected_id_hit_rate") or 0.0),
                quality=float(item.get("quality_hit_rate") or 0.0),
                failed=item.get("failed_query_count") or 0,
            )
        )
    (evaluation / TREND_MD_FILENAME).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _category_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    summary: dict[str, dict[str, Any]] = {}
    for row in rows:
        category = str(row.get("benchmark_category") or "uncategorized")
        bucket = summary.setdefault(
            category,
            {"query_count": 0, "top1_hits": 0, "top3_hits": 0, "expected_id_hits": 0, "failed": 0, "failure_reasons": Counter()},
        )
        bucket["query_count"] += 1
        bucket["top1_hits"] += int(bool(row.get("top1_hit")))
        bucket["top3_hits"] += int(bool(row.get("topk_hit")))
        bucket["expected_id_hits"] += int(bool(row.get("expected_id_hit")))
        bucket["failed"] += int(not bool(row.get("acceptance_hit")))
        for reason in row.get("failure_explanation") or []:
            bucket["failure_reasons"][str(reason)] += 1
    output = {}
    for category, bucket in sorted(summary.items()):
        total = max(1, int(bucket["query_count"]))
        output[category] = {
            "query_count": bucket["query_count"],
            "top1_hit_rate": bucket["top1_hits"] / total,
            "top3_hit_rate": bucket["top3_hits"] / total,
            "expected_id_hit_rate": bucket["expected_id_hits"] / total,
            "failed_query_count": bucket["failed"],
            "failure_reasons": dict(bucket["failure_reasons"].most_common(5)),
        }
    return output


def _category_for_actions(actions: list[str], query: str = "") -> str:
    values = {_norm(item) for item in actions}
    if values & {"weighing"}:
        return "weighing"
    if values & {"open", "close", "container_state_change", "bottle_interaction"}:
        return "open_close_container"
    if values & {"pipetting", "aspirate", "dispense", "sample_adding", "liquid_transfer", "attach_tip", "adjust", "discard"}:
        return "pipetting"
    if values & {"recording", "panel_reading", "label_check"}:
        return "reading_recording"
    if values & {"mixing", "shake"}:
        return "mixing"
    if values & {"liquid_state_change", "liquid_level_measured", "liquid_flow_observed", "container_color_change"}:
        return "liquid_level"
    if values & {"equipment_control_state"}:
        return "container_state"
    return _category_for_query(query)


def _category_for_query(query: str) -> str:
    if any(token in query for token in ("称量", "天平")):
        return "weighing"
    if any(token in query for token in ("移液", "加样", "吸取", "排出")):
        return "pipetting"
    if any(token in query for token in ("打开", "关闭", "盖")):
        return "container_state"
    if any(token in query for token in ("液面", "颜色", "沉淀", "液体")):
        return "liquid_or_state"
    if any(token in query for token in ("记录", "读取", "面板")):
        return "recording_or_panel"
    return "hand_object_interaction"


def _row_text(row: Mapping[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True).casefold().replace("-", "_").replace(" ", "_")


def _cached_row_text(row: Mapping[str, Any]) -> str:
    cache_key = "_retrieval_eval_search_text"
    if isinstance(row, dict):
        cached = row.get(cache_key)
        if isinstance(cached, str):
            return cached
        text = _row_text(row)
        row[cache_key] = text
        return text
    return _row_text(row)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.exists() else []


def _gold_query_decisions_by_id(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"gold query decision file not found: {path}")
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        rows = read_jsonl(path)
    else:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        rows = payload.get("decisions") if isinstance(payload, Mapping) else payload
    if not isinstance(rows, list):
        raise ValueError("gold query decision file must be a list or contain a decisions list")
    decisions: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        key = str(row.get("query_id") or row.get("query") or "").strip()
        if key:
            decisions[key] = dict(row)
    return decisions


def _normalize_gold_decision(value: Any) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "approve": "approved",
        "approved": "approved",
        "reject": "rejected",
        "rejected": "rejected",
        "needs_more_review": "needs_review",
        "needs-review": "needs_review",
        "needs_review": "needs_review",
    }
    if text not in aliases:
        raise ValueError("gold query decision must be one of: approve, reject, needs_more_review")
    return aliases[text]


def _string_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    values = value if isinstance(value, list) else [value]
    return [str(item).strip() for item in values if str(item).strip()]


def _first_text(row: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _norm(value: Any) -> str:
    return str(value or "").strip().casefold().replace("-", "_").replace(" ", "_")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "DEFAULT_QUERY_COUNT",
    "FIXED_CHINESE_QUERY_BENCHMARK",
    "GOLD_QUERY_BENCHMARK_FILENAME",
    "TREND_JSONL_FILENAME",
    "TREND_MD_FILENAME",
    "build_default_chinese_query_eval_config",
    "confirm_gold_query_benchmark",
    "build_gold_query_benchmark",
    "run_default_chinese_query_eval",
]

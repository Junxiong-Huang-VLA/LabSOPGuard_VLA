from __future__ import annotations

import json
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping

from .schemas import read_jsonl


HISTORY_MODEL_SCHEMA_VERSION = "key_action_history_model.v2"
HISTORY_RECORD_SCHEMA_VERSION = "key_action_history_record.v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _audit_entry(action: str, *, actor: str = "key_action_indexer", source_session_id: str = "", details: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "timestamp": _now(),
        "actor": actor,
        "action": action,
        "source_session_id": source_session_id,
        "details": dict(details or {}),
    }


def build_history_model(
    sources: Iterable[str | Path],
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    source_list = [Path(source) for source in sources]
    source_quality = [_source_quality(source) for source in source_list]
    events = _load_history_events(source_list)
    history_records = _load_history_process_records(source_list)
    sessions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        session_id = str(event.get("session_id") or event.get("session") or "unknown")
        sessions[session_id].append(event)
    for rows in sessions.values():
        rows.sort(key=lambda row: str(row.get("global_start_time") or row.get("global_time") or ""))

    action_counts: Counter[str] = Counter()
    transition_counts: Counter[tuple[str, str]] = Counter()
    durations: dict[str, list[float]] = defaultdict(list)
    material_counts: Counter[str] = Counter()
    evidence_levels: Counter[str] = Counter()
    for rows in sessions.values():
        actions = [_action(row) for row in rows if _action(row)]
        action_counts.update(actions)
        transition_counts.update(zip(actions, actions[1:]))
        for row in rows:
            action = _action(row)
            if not action:
                continue
            duration = _duration(row)
            if duration is not None:
                durations[action].append(duration)
            for material in _materials(row):
                material_counts[material] += 1
            if row.get("evidence_level"):
                evidence_levels[str(row["evidence_level"])] += 1

    transition_probabilities = _transition_probabilities(transition_counts)
    action_probabilities = _action_probabilities(action_counts)
    model = {
        "schema_version": HISTORY_MODEL_SCHEMA_VERSION,
        "version": 2,
        "created_at": _now(),
        "source_session_ids": sorted(sessions),
        "source_quality": source_quality,
        "source_quality_counts": dict(sorted(Counter(str(row.get("source_kind") or "unknown") for row in source_quality).items())),
        "key_action_index_session_count": sum(1 for row in source_quality if row.get("is_complete_key_action_index")),
        "legacy_source_count": sum(1 for row in source_quality if not row.get("is_complete_key_action_index")),
        "step_reasoning_support": {
            "history_record_count": len(history_records),
            "sources_with_process_records": sum(1 for row in source_quality if row.get("has_experiment_process")),
            "has_transition_priors": bool(transition_counts),
        },
        "audit_trail": [
            _audit_entry(
                "history_model_built",
                details={"source_count": len(source_list), "event_count": len(events), "record_count": len(history_records)},
            )
        ],
        "session_count": len(sessions),
        "event_count": len(events),
        "history_record_count": len(history_records),
        "history_records": history_records,
        "action_counts": dict(sorted(action_counts.items())),
        "action_probabilities": action_probabilities,
        "transition_counts": {f"{src}->{dst}": count for (src, dst), count in sorted(transition_counts.items())},
        "transition_probabilities": transition_probabilities,
        "duration_stats": _duration_stats(durations),
        "material_counts": dict(sorted(material_counts.items())),
        "evidence_level_counts": dict(sorted(evidence_levels.items())),
        "recommended_sop": _recommended_sop(action_counts, transition_probabilities),
    }
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    return model


def load_history_model(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        return {}
    return json.loads(source.read_text(encoding="utf-8-sig"))


def load_history_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    if source.suffix.lower() == ".jsonl":
        return read_jsonl(source)
    data = json.loads(source.read_text(encoding="utf-8-sig"))
    if isinstance(data, Mapping) and isinstance(data.get("history_records"), list):
        return [dict(row) for row in data["history_records"] if isinstance(row, Mapping)]
    if isinstance(data, Mapping) and isinstance(data.get("records"), list):
        return [dict(row) for row in data["records"] if isinstance(row, Mapping)]
    if isinstance(data, list):
        return [dict(row) for row in data if isinstance(row, Mapping)]
    if isinstance(data, Mapping) and data.get("schema_version") == HISTORY_RECORD_SCHEMA_VERSION:
        return [dict(data)]
    return []


def build_history_process_record(
    source: str | Path | Mapping[str, Any],
    *,
    version: int = 1,
    source_session_id: str | None = None,
    actor: str = "key_action_indexer",
    audit_trail: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    process, timeline, source_path = _load_process_payload(source)
    session_id = source_session_id or str(process.get("session_id") or (Path(source_path).stem if source_path else "unknown"))
    steps = [dict(step) for step in process.get("steps", []) if isinstance(step, Mapping)]
    actions = [str(step.get("expected_action") or "") for step in steps if step.get("expected_action")]
    audit = list(audit_trail or [])
    audit.append(
        _audit_entry(
            "history_process_record_created",
            actor=actor,
            source_session_id=session_id,
            details={"source_path": source_path, "version": version, "step_count": len(steps)},
        )
    )
    return {
        "schema_version": HISTORY_RECORD_SCHEMA_VERSION,
        "record_id": f"{session_id}:v{int(version)}",
        "source_session_id": session_id,
        "version": int(version),
        "created_at": _now(),
        "updated_at": _now(),
        "source_path": source_path,
        "audit_trail": audit,
        "summary": {
            "step_count": len(steps),
            "actions": actions,
            "status_counts": process.get("status_counts") or _status_counts(steps),
            "process_status": process.get("process_status"),
            "timeline_event_count": len(timeline),
        },
        "process": _versioned_process(process, session_id, int(version), audit),
        "steps": [_versioned_process(step, session_id, int(version), audit) for step in steps],
        "timeline": [_versioned_process(row, session_id, int(version), audit) for row in timeline],
    }


def write_back_process_data(
    session_dir: str | Path,
    history_store_path: str | Path,
    *,
    updates: Mapping[str, Any] | None = None,
    actor: str = "key_action_indexer",
    note: str = "process_data_writeback",
) -> dict[str, Any]:
    session = Path(session_dir)
    process_path = session / "metadata" / "experiment_process.json"
    process = json.loads(process_path.read_text(encoding="utf-8-sig")) if process_path.exists() else {}
    if updates:
        process = _deep_merge(process, updates)
    source_session_id = str(process.get("session_id") or session.name)
    store = Path(history_store_path)
    records = load_history_records(store)
    next_version = _next_record_version(records, source_session_id)
    audit = [
        _audit_entry(
            note,
            actor=actor,
            source_session_id=source_session_id,
            details={"session_dir": str(session), "history_store_path": str(store), "version": next_version},
        )
    ]
    record = build_history_process_record(
        process,
        version=next_version,
        source_session_id=source_session_id,
        actor=actor,
        audit_trail=audit,
    )
    records.append(record)
    _write_history_records(store, records)
    return {
        "schema_version": HISTORY_RECORD_SCHEMA_VERSION,
        "source_session_id": source_session_id,
        "version": next_version,
        "history_store_path": str(store),
        "record": record,
        "record_count": len(records),
    }


def search_similar_history(
    process: Mapping[str, Any],
    history: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    *,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    records = _records_from_history(history)
    query_actions = _process_actions(process)
    query_materials = _process_materials(process)
    query_transitions = set(zip(query_actions, query_actions[1:]))
    results: list[dict[str, Any]] = []
    for record in records:
        record_actions = [str(action) for action in (record.get("summary") or {}).get("actions", []) if action]
        if not record_actions:
            record_actions = _process_actions(record.get("process") if isinstance(record.get("process"), Mapping) else record)
        record_materials = _process_materials(record.get("process") if isinstance(record.get("process"), Mapping) else record)
        record_transitions = set(zip(record_actions, record_actions[1:]))
        action_score = _jaccard(query_actions, record_actions)
        transition_score = _jaccard(query_transitions, record_transitions)
        material_score = _jaccard(query_materials, record_materials)
        order_score = _ordered_subsequence_score(query_actions, record_actions)
        score = round(0.45 * action_score + 0.25 * transition_score + 0.2 * order_score + 0.1 * material_score, 6)
        results.append(
            {
                "source_session_id": record.get("source_session_id"),
                "version": record.get("version"),
                "score": score,
                "action_similarity": round(action_score, 6),
                "transition_similarity": round(transition_score, 6),
                "order_similarity": round(order_score, 6),
                "material_similarity": round(material_score, 6),
                "shared_actions": sorted(set(query_actions) & set(record_actions)),
                "audit_trail": record.get("audit_trail", []),
                "record_id": record.get("record_id"),
            }
        )
    results.sort(key=lambda row: (float(row["score"]), int(row.get("version") or 0)), reverse=True)
    return results[: max(1, int(top_k))]


def score_process_with_history(process: Mapping[str, Any], history_model: Mapping[str, Any]) -> dict[str, Any]:
    steps = [step for step in process.get("steps", []) if isinstance(step, Mapping)]
    transitions = history_model.get("transition_probabilities") if isinstance(history_model.get("transition_probabilities"), Mapping) else {}
    action_probabilities = _history_action_probabilities(history_model)
    action_counts = _as_mapping(history_model.get("action_counts"))
    transition_scores = []
    action_scores = []
    flags = []
    step_priors = []
    actions = [str(step.get("expected_action") or "") for step in steps if step.get("expected_action")]
    for step in steps:
        action = str(step.get("expected_action") or "")
        if not action:
            continue
        action_probability = float(action_probabilities.get(action, 0.0) or 0.0)
        support_count = int(float(action_counts.get(action, 0) or 0))
        action_scores.append(action_probability)
        step_prior = {
            "step_id": step.get("step_id"),
            "action": action,
            "action_probability": round(action_probability, 6),
            "support_count": support_count,
            "history_session_count": history_model.get("session_count", 0),
        }
        step_priors.append(step_prior)
        if action_probabilities and action_probability <= 0.0:
            flags.append({"action": action, "probability": 0.0, "flag": "unseen_action"})
        elif action_probabilities and action_probability < 0.05:
            flags.append({"action": action, "probability": round(action_probability, 6), "flag": "rare_action"})
    rare_transition_flags = []
    for src, dst in zip(actions, actions[1:]):
        probability = float(_as_mapping(transitions.get(src)).get(dst, 0.0) or 0.0)
        transition_scores.append(probability)
        if probability < 0.15:
            flag = {"transition": f"{src}->{dst}", "probability": probability, "flag": "rare_transition"}
            rare_transition_flags.append(flag)
            flags.append(flag)
    action_score = round(mean(action_scores), 4) if action_scores else 0.0
    transition_score = round(mean(transition_scores), 4) if transition_scores else 0.0
    prior_score_parts = [score for score in (action_score, transition_score) if score > 0.0]
    prior_score = round(mean(prior_score_parts), 4) if prior_score_parts else 0.0
    return {
        "transition_score": transition_score,
        "rare_transition_flags": rare_transition_flags,
        "history_session_count": history_model.get("session_count", 0),
        "action_prior_score": action_score,
        "history_prior": {
            "score": prior_score,
            "action_score": action_score,
            "transition_score": transition_score,
            "step_priors": step_priors,
            "history_session_count": history_model.get("session_count", 0),
        },
        "history_deviation": {
            "score": round(max(0.0, min(1.0, 1.0 - prior_score)), 4) if prior_score else 1.0,
            "flags": flags,
            "rare_transition_count": len(rare_transition_flags),
            "unseen_action_count": sum(1 for flag in flags if flag.get("flag") == "unseen_action"),
        },
    }


def retrieve_related_history(
    query_context: Mapping[str, Any],
    sources: Iterable[str | Path],
    output_path: str | Path | None = None,
    *,
    limit: int = 10,
) -> dict[str, Any]:
    events = _load_history_events(sources)
    query_actions = _query_actions(query_context)
    query_materials = _query_materials(query_context)
    sessions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        sessions[str(event.get("session_id") or "unknown")].append(event)
    related_records: list[dict[str, Any]] = []
    for session_id, rows in sessions.items():
        rows.sort(key=lambda row: str(row.get("global_start_time") or row.get("global_time") or ""))
        actions = [_action(row) for row in rows if _action(row)]
        materials = sorted({material for row in rows for material in _materials(row)})
        matched_actions = sorted(set(actions) & query_actions)
        matched_materials = sorted(set(materials) & query_materials)
        score = len(matched_actions) * 2.0 + len(matched_materials) * 1.5
        if score <= 0 and (query_actions or query_materials):
            continue
        related_records.append(
            {
                "record_id": session_id,
                "session_id": session_id,
                "score": round(score, 4),
                "event_count": len(rows),
                "matched_actions": matched_actions,
                "matched_materials": matched_materials,
                "transition_sequence": actions,
                "global_start_time": rows[0].get("global_start_time") or rows[0].get("global_time") if rows else None,
                "global_end_time": rows[-1].get("global_end_time") or rows[-1].get("global_time") if rows else None,
            }
        )
    related_records.sort(key=lambda row: (-float(row.get("score") or 0.0), str(row.get("record_id") or "")))
    related_records = related_records[:limit]
    result = {
        "related_records": related_records,
        "transition_priors": _transition_priors_from_records(related_records),
        "query": {
            "actions": sorted(query_actions),
            "materials": sorted(query_materials),
        },
        "source_event_count": len(events),
    }
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _load_history_events(sources: Iterable[str | Path]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for source_value in sources:
        source = Path(source_value)
        if not source.exists():
            continue
        if source.is_dir():
            for candidate, reader in (
                (source / "metadata" / "video_understanding.jsonl", _read_history_jsonl_rows),
                (source / "metadata" / "experiment_process_timeline.jsonl", _read_history_jsonl_rows),
                (source / "metadata" / "micro_segments.jsonl", _read_history_jsonl_rows),
                (source / "physical_events.json", _read_history_json_rows),
                (source / "steps.json", _read_history_json_rows),
                (source / "official_steps.json", _read_history_json_rows),
                (source / "material_stream.json", _read_history_json_rows),
            ):
                if candidate.exists():
                    events.extend(_normalize_rows(reader(candidate), source.name))
                    break
        elif source.suffix.lower() == ".jsonl":
            events.extend(_normalize_rows(_read_history_jsonl_rows(source), source.stem))
        else:
            events.extend(_normalize_rows(_read_history_json_rows(source), source.stem))
    return events


def _source_quality(source: Path) -> dict[str, Any]:
    metadata = source / "metadata" if source.is_dir() else source.parent
    if source.is_dir() and not metadata.exists() and (source / "key_action_index" / "metadata").exists():
        metadata = source / "key_action_index" / "metadata"
    has_key_action_index = metadata.name == "metadata" and metadata.parent.name == "key_action_index"
    source_session_id = _source_quality_session_id(source, metadata)
    has_process = (metadata / "experiment_process.json").exists()
    has_timeline = (metadata / "experiment_process_timeline.jsonl").exists()
    has_micro = (metadata / "micro_segments.jsonl").exists()
    has_video = (metadata / "video_understanding.jsonl").exists()
    complete = bool(has_key_action_index and has_process and has_timeline and has_micro and has_video)
    if complete:
        source_kind = "complete_key_action_index"
    elif has_key_action_index:
        source_kind = "partial_key_action_index"
    else:
        source_kind = "legacy_or_external"
    return {
        "source": str(source),
        "source_session_id": source_session_id,
        "source_kind": source_kind,
        "is_complete_key_action_index": complete,
        "has_key_action_index": has_key_action_index,
        "has_experiment_process": has_process,
        "has_process_timeline": has_timeline,
        "has_micro_segments": has_micro,
        "has_video_understanding": has_video,
    }


def _source_quality_session_id(source: Path, metadata: Path) -> str:
    if metadata.name == "metadata" and metadata.parent.name == "key_action_index":
        return metadata.parent.parent.name
    if source.name == "metadata" and source.parent.name == "key_action_index":
        return source.parent.parent.name
    if source.name == "key_action_index":
        return source.parent.name
    return source.stem if source.is_file() else source.name


def _read_history_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def _read_history_json_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    rows: Any = data
    if isinstance(data, Mapping):
        for key in ("events", "physical_events", "steps", "items", "records"):
            value = data.get(key)
            if isinstance(value, list):
                rows = value
                break
        else:
            timeline = data.get("timeline") if isinstance(data.get("timeline"), Mapping) else {}
            rows = timeline.get("steps") if isinstance(timeline.get("steps"), list) else data
    if isinstance(rows, list):
        return [dict(row) for row in rows if isinstance(row, Mapping)]
    if isinstance(rows, Mapping):
        return [dict(rows)]
    return []


def _load_history_process_records(sources: Iterable[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source in sources:
        if source.is_dir():
            process_path = source / "metadata" / "experiment_process.json"
            if process_path.exists():
                records.append(build_history_process_record(source))
        elif source.suffix.lower() in {".json", ".jsonl"}:
            loaded = load_history_records(source)
            if loaded:
                records.extend(loaded)
            elif source.name == "experiment_process.json":
                records.append(build_history_process_record(source))
    return records


def _normalize_rows(rows: list[dict[str, Any]], fallback_session_id: str) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        item = dict(row)
        item.setdefault("session_id", fallback_session_id)
        normalized.append(item)
    return normalized


def _action(row: Mapping[str, Any]) -> str:
    return str(row.get("action_type") or row.get("expected_action") or row.get("event_type") or "").strip()


def _duration(row: Mapping[str, Any]) -> float | None:
    if row.get("duration_sec") is not None:
        try:
            return float(row["duration_sec"])
        except (TypeError, ValueError):
            return None
    return None


def _materials(row: Mapping[str, Any]) -> list[str]:
    values = []
    for key in ("primary_object", "object_label", "materials", "objects"):
        value = row.get(key)
        if isinstance(value, list):
            values.extend(str(item) for item in value if item)
        elif value:
            values.append(str(value))
    return values


def _transition_probabilities(counts: Counter[tuple[str, str]]) -> dict[str, dict[str, float]]:
    totals: Counter[str] = Counter()
    for (src, _dst), count in counts.items():
        totals[src] += count
    result: dict[str, dict[str, float]] = defaultdict(dict)
    for (src, dst), count in counts.items():
        result[src][dst] = round(count / totals[src], 6) if totals[src] else 0.0
    return {src: dict(sorted(dsts.items())) for src, dsts in sorted(result.items())}


def _action_probabilities(counts: Counter[str]) -> dict[str, float]:
    total = sum(counts.values())
    if not total:
        return {}
    return {action: round(count / total, 6) for action, count in sorted(counts.items())}


def _duration_stats(durations: Mapping[str, list[float]]) -> dict[str, dict[str, float]]:
    return {
        action: {
            "count": len(values),
            "mean_sec": round(mean(values), 4),
            "median_sec": round(median(values), 4),
            "min_sec": round(min(values), 4),
            "max_sec": round(max(values), 4),
        }
        for action, values in sorted(durations.items())
        if values
    }


def _recommended_sop(action_counts: Counter[str], transition_probabilities: Mapping[str, Mapping[str, float]]) -> list[dict[str, Any]]:
    if not action_counts:
        return []
    start = action_counts.most_common(1)[0][0]
    order = [start]
    seen = {start}
    current = start
    while current in transition_probabilities:
        next_items = sorted(transition_probabilities[current].items(), key=lambda item: (-item[1], item[0]))
        next_action = next((action for action, _prob in next_items if action not in seen), "")
        if not next_action:
            break
        order.append(next_action)
        seen.add(next_action)
        current = next_action
        if len(order) >= 20:
            break
    return [{"step_id": f"hist_{index:03d}", "expected_action": action, "name": action.replace("_", " ").title()} for index, action in enumerate(order, start=1)]


def _as_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _history_action_probabilities(history_model: Mapping[str, Any]) -> dict[str, float]:
    probabilities = history_model.get("action_probabilities")
    if isinstance(probabilities, Mapping):
        parsed: dict[str, float] = {}
        for action, probability in probabilities.items():
            try:
                parsed[str(action)] = float(probability or 0.0)
            except (TypeError, ValueError):
                parsed[str(action)] = 0.0
        return parsed
    counts: Counter[str] = Counter()
    for action, count in _as_mapping(history_model.get("action_counts")).items():
        try:
            counts[str(action)] = int(float(count or 0))
        except (TypeError, ValueError):
            counts[str(action)] = 0
    return _action_probabilities(counts)


def _load_process_payload(source: str | Path | Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], str | None]:
    if isinstance(source, Mapping):
        return dict(source), [], None
    path = Path(source)
    if path.is_dir():
        process_path = path / "metadata" / "experiment_process.json"
        timeline_path = path / "metadata" / "experiment_process_timeline.jsonl"
    else:
        process_path = path
        timeline_path = path.with_name("experiment_process_timeline.jsonl")
    process = json.loads(process_path.read_text(encoding="utf-8-sig")) if process_path.exists() else {}
    timeline = read_jsonl(timeline_path) if timeline_path.exists() else []
    return process, timeline, str(process_path) if process_path.exists() else str(path)


def _versioned_process(row: Mapping[str, Any], source_session_id: str, version: int, audit_trail: list[dict[str, Any]]) -> dict[str, Any]:
    item = dict(row)
    item.setdefault("source_session_id", source_session_id)
    item.setdefault("version", version)
    item.setdefault("audit_trail", audit_trail)
    return item


def _status_counts(steps: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(step.get("status") or "unknown") for step in steps).items()))


def _next_record_version(records: Iterable[Mapping[str, Any]], source_session_id: str) -> int:
    versions = []
    for record in records:
        if str(record.get("source_session_id") or "") != source_session_id:
            continue
        try:
            versions.append(int(record.get("version") or 0))
        except (TypeError, ValueError):
            continue
    return max(versions, default=0) + 1


def _write_history_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".jsonl":
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    else:
        payload = {
            "schema_version": HISTORY_MODEL_SCHEMA_VERSION,
            "updated_at": _now(),
            "record_count": len(records),
            "history_records": records,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _deep_merge(base: Mapping[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _records_from_history(history: Mapping[str, Any] | Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(history, Mapping):
        for key in ("history_records", "records"):
            value = history.get(key)
            if isinstance(value, list):
                return [dict(item) for item in value if isinstance(item, Mapping)]
        if history.get("schema_version") == HISTORY_RECORD_SCHEMA_VERSION:
            return [dict(history)]
        return []
    return [dict(item) for item in history if isinstance(item, Mapping)]


def _process_actions(process: Mapping[str, Any]) -> list[str]:
    steps = process.get("steps") if isinstance(process.get("steps"), list) else []
    actions = [str(step.get("expected_action") or "") for step in steps if isinstance(step, Mapping) and step.get("expected_action")]
    if actions:
        return actions
    return [str(row.get("action_type") or row.get("event_type") or "") for row in _list(process.get("events")) if isinstance(row, Mapping)]


def _process_materials(process: Mapping[str, Any]) -> list[str]:
    materials: list[str] = []
    context = process.get("context") if isinstance(process.get("context"), Mapping) else {}
    for item in _list(context.get("materials")):
        if isinstance(item, Mapping) and item.get("name"):
            materials.append(str(item["name"]))
        elif item:
            materials.append(str(item))
    for step in _list(process.get("steps")):
        if isinstance(step, Mapping):
            for value in _list(step.get("required_material") or step.get("required_materials")):
                if value:
                    materials.append(str(value))
    return materials


def _list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _jaccard(left: Iterable[Any], right: Iterable[Any]) -> float:
    left_set = {str(item) for item in left if item}
    right_set = {str(item) for item in right if item}
    if not left_set and not right_set:
        return 1.0
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def _ordered_subsequence_score(query_actions: list[str], record_actions: list[str]) -> float:
    if not query_actions or not record_actions:
        return 0.0
    cursor = 0
    matched = 0
    for action in query_actions:
        try:
            index = record_actions.index(action, cursor)
        except ValueError:
            continue
        matched += 1
        cursor = index + 1
    return matched / max(len(query_actions), 1)


__all__ = [
    "HISTORY_MODEL_SCHEMA_VERSION",
    "HISTORY_RECORD_SCHEMA_VERSION",
    "build_history_model",
    "build_history_process_record",
    "load_history_model",
    "load_history_records",
    "score_process_with_history",
    "search_similar_history",
    "write_back_process_data",
]

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from .schemas import read_jsonl


KEYWORD_MAP = {
    "weighing": ("称量", "天平", "balance", "weigh", "mass"),
    "pipetting": ("移液", "移液枪", "加样", "微升", "pipette", "pipetting", "transfer"),
    "recording": ("记录", "读数", "readout", "record"),
    "sample_handling": ("样品", "样品瓶", "试管", "vial", "tube", "bottle", "sample"),
}
MATERIAL_KEYWORDS = {
    "balance": ("天平", "balance"),
    "pipette": ("移液枪", "pipette"),
    "sample_bottle": ("样品瓶", "bottle", "vial"),
    "tube": ("试管", "tube"),
    "sample": ("样品", "sample"),
}
REAGENT_KEYWORDS = {
    "reagent": ("reagent", "solution", "buffer", "试剂", "溶液"),
    "sample": ("sample", "specimen", "样品"),
    "water": ("water", "h2o", "distilled_water", "蒸馏水"),
}
EQUIPMENT_KEYWORDS = {
    "balance": ("balance", "scale", "澶╁钩"),
    "pipette": ("pipette", "pipettor", "绉绘恫"),
    "equipment_panel": ("panel", "display", "readout", "button", "knob", "switch"),
    "centrifuge": ("centrifuge",),
    "vortex": ("vortex", "mixer"),
    "incubator": ("incubator", "heater"),
}
PARAM_PATTERN = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>微升|毫升|秒|分钟|小时|°C|℃|C|ul|uL|µL|mL|ml|min|s|sec|g|mg)",
    re.IGNORECASE,
)


def build_experiment_context(
    session_dir: str | Path,
    output_path: str | Path | None = None,
    database_paths: list[str | Path] | None = None,
) -> dict[str, Any]:
    session = Path(session_dir)
    metadata = session / "metadata"
    timeline_rows = _read_jsonl_if_exists(metadata / "unified_multimodal_timeline.jsonl")
    transcript_rows = _read_jsonl_if_exists(session / "transcript" / "aligned_transcript.jsonl")
    asset_rows = _read_jsonl_if_exists(metadata / "material_asset_catalog.jsonl")
    video_rows = _read_jsonl_if_exists(metadata / "video_understanding.jsonl")
    database_rows = _read_jsonl_if_exists(metadata / "database_records.jsonl")
    database_rows.extend(_read_database_rows(database_paths or []))
    sop_rows = _read_jsonl_if_exists(metadata / "sop_records.jsonl")
    user_text_rows = _read_jsonl_if_exists(metadata / "user_text_events.jsonl")

    evidence = {
        "text_evidence": [
            *_timeline_evidence(timeline_rows, {"session_context", "user_text", "user", "manual_note"}),
            *_row_text_evidence(user_text_rows, "user_text"),
        ],
        "upload_evidence": _timeline_evidence(timeline_rows, {"upload"}),
        "ai_evidence": _timeline_evidence(timeline_rows, {"ai_reply", "ai", "assistant"}),
        "transcript_evidence": _row_text_evidence(transcript_rows, "transcript"),
        "database_evidence": _row_text_evidence(database_rows, "database"),
        "sop_evidence": _row_text_evidence(sop_rows, "sop"),
        "video_evidence": _video_evidence(video_rows),
    }
    all_texts = _all_texts(evidence, asset_rows)
    procedure_candidates = _procedure_candidates(all_texts, video_rows)
    materials = _materials(all_texts, asset_rows, video_rows)
    reagents = _reagents(all_texts, asset_rows, video_rows)
    equipment = _equipment(all_texts, asset_rows, video_rows)
    parameters = _merge_parameters([*_parameters(all_texts), *_video_parameters(video_rows)])
    related_records = _related_database_records(database_rows, procedure_candidates, materials)
    transition_priors = _transition_priors_from_related_records(related_records)
    purpose = _purpose(procedure_candidates, materials, all_texts)
    gaps = _gaps(procedure_candidates, materials, parameters, video_rows)
    source_counts = {
        "timeline_events": len(timeline_rows),
        "transcript_rows": len(transcript_rows),
        "material_assets": len(asset_rows),
        "video_events": len(video_rows),
        "database_rows": len(database_rows),
        "sop_rows": len(sop_rows),
        "user_text_events": len(user_text_rows),
    }
    confidence = _confidence(source_counts, procedure_candidates, materials, parameters)
    result = {
        "session_id": _session_id(timeline_rows, transcript_rows, asset_rows, video_rows, database_rows, sop_rows, user_text_rows, session),
        "purpose": purpose,
        "procedure_candidates": procedure_candidates,
        "materials": materials,
        "reagents": reagents,
        "equipment": equipment,
        "parameters": parameters,
        "related_records": related_records,
        "transition_priors": transition_priors,
        "source_counts": source_counts,
        **evidence,
        "fused_context": {
            "purpose": purpose,
            "procedure": [item["action_type"] for item in procedure_candidates],
            "materials": [item["name"] for item in materials],
            "reagents": [item["name"] for item in reagents],
            "equipment": [item["name"] for item in equipment],
            "parameters": parameters,
            "related_record_ids": [item["record_id"] for item in related_records],
            "transition_priors": transition_priors,
            "evidence_sources": [key for key, rows in evidence.items() if rows],
        },
        "confidence": confidence,
        "gaps": gaps,
    }
    target = Path(output_path) if output_path is not None else metadata / "experiment_context.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def load_experiment_context(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        return {}
    return json.loads(source.read_text(encoding="utf-8-sig"))


def _timeline_evidence(rows: list[Mapping[str, Any]], event_types: set[str]) -> list[dict[str, Any]]:
    evidence = []
    for row in rows:
        event_type = str(row.get("event_type") or row.get("source") or "").lower()
        if event_type in event_types:
            evidence.append(_evidence_row(row, event_type))
    return evidence


def _row_text_evidence(rows: list[Mapping[str, Any]], source: str) -> list[dict[str, Any]]:
    return [_evidence_row(row, source) for row in rows if _row_text(row)]


def _video_evidence(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "source": "video_understanding",
            "id": row.get("video_event_id"),
            "text": _join_text(row.get("semantic_description"), row.get("text")),
            "global_time": row.get("global_start_time"),
            "action_type": row.get("action_type"),
            "primary_object": row.get("primary_object"),
            "confidence": row.get("confidence"),
            "conclusion_status": row.get("conclusion_status"),
            "object_category": row.get("object_category"),
        }
        for row in rows
    ]


def _evidence_row(row: Mapping[str, Any], source: str) -> dict[str, Any]:
    return {
        "source": source,
        "id": row.get("timeline_event_id") or row.get("utterance_id") or row.get("id") or row.get("record_id"),
        "text": _row_text(row),
        "global_time": row.get("global_time") or row.get("global_start_time"),
    }


def _all_texts(evidence: Mapping[str, list[Mapping[str, Any]]], asset_rows: list[Mapping[str, Any]]) -> list[str]:
    texts = [str(row.get("text") or "") for rows in evidence.values() for row in rows if row.get("text")]
    texts.extend(str(row.get("search_text") or "") for row in asset_rows if row.get("search_text"))
    texts.extend(" ".join(_strings(row.get("objects")) + _strings(row.get("actions"))) for row in asset_rows)
    return [text for text in texts if text]


def _procedure_candidates(texts: list[str], video_rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    sources: dict[str, set[str]] = {key: set() for key in KEYWORD_MAP}
    for text in texts:
        lowered = text.lower()
        for action, keywords in KEYWORD_MAP.items():
            if any(keyword.lower() in lowered for keyword in keywords):
                counts[action] += 1
                sources[action].add("text")
    for row in video_rows:
        action = str(row.get("action_type") or "")
        event_type = str(row.get("event_type") or "")
        for candidate, keywords in KEYWORD_MAP.items():
            haystack = f"{action} {event_type}".lower()
            if candidate in haystack or any(keyword.lower() in haystack for keyword in keywords):
                counts[candidate] += 2
                sources[candidate].add("video")
    order = ["weighing", "pipetting", "sample_handling", "recording"]
    return [
        {
            "action_type": action,
            "score": counts[action],
            "source_types": sorted(sources[action]),
        }
        for action in order
        if counts[action] > 0
    ]


def _materials(texts: list[str], asset_rows: list[Mapping[str, Any]], video_rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    sources: dict[str, set[str]] = {key: set() for key in MATERIAL_KEYWORDS}
    joined = "\n".join(texts).lower()
    for material, keywords in MATERIAL_KEYWORDS.items():
        if any(keyword.lower() in joined for keyword in keywords):
            counts[material] += 1
            sources[material].add("text")
    for row in asset_rows:
        for obj in _strings(row.get("objects")):
            normalized = _normalize_material(obj)
            if normalized:
                counts[normalized] += 1
                sources.setdefault(normalized, set()).add("asset")
    for row in video_rows:
        normalized = _normalize_material(row.get("primary_object"))
        if normalized:
            counts[normalized] += 2
            sources.setdefault(normalized, set()).add("video")
    return [
        {"name": name, "score": count, "source_types": sorted(sources.get(name, set()))}
        for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _reagents(texts: list[str], asset_rows: list[Mapping[str, Any]], video_rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return _keyword_entities(
        texts,
        asset_rows,
        video_rows,
        REAGENT_KEYWORDS,
        entity_key="reagents",
        category_names={"liquid_or_reagent"},
    )


def _equipment(texts: list[str], asset_rows: list[Mapping[str, Any]], video_rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return _keyword_entities(
        texts,
        asset_rows,
        video_rows,
        EQUIPMENT_KEYWORDS,
        entity_key="equipment",
        category_names={"equipment", "equipment_control"},
    )


def _keyword_entities(
    texts: list[str],
    asset_rows: list[Mapping[str, Any]],
    video_rows: list[Mapping[str, Any]],
    keywords_by_name: Mapping[str, Iterable[str]],
    *,
    entity_key: str,
    category_names: set[str],
) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    sources: dict[str, set[str]] = {key: set() for key in keywords_by_name}
    joined = "\n".join(texts).lower()
    for name, keywords in keywords_by_name.items():
        if name.lower() in joined or any(str(keyword).lower() in joined for keyword in keywords):
            counts[name] += 1
            sources[name].add("text")
    for row in asset_rows:
        for obj in _strings(row.get("objects")):
            normalized = _normalize_keyword_entity(obj, keywords_by_name)
            if normalized:
                counts[normalized] += 1
                sources.setdefault(normalized, set()).add("asset")
    for row in video_rows:
        extracted = row.get("extracted_entities")
        if isinstance(extracted, Mapping):
            for value in _strings(extracted.get(entity_key)):
                normalized = _normalize_keyword_entity(value, keywords_by_name) or str(value).lower()
                counts[normalized] += 2
                sources.setdefault(normalized, set()).add("video")
        category = str(row.get("object_category") or "")
        if category in category_names:
            normalized = _normalize_keyword_entity(row.get("primary_object"), keywords_by_name)
            normalized = normalized or _normalize_keyword_entity(_as_dict(row.get("normalized_object")).get("canonical_label"), keywords_by_name)
            if normalized:
                counts[normalized] += 2
                sources.setdefault(normalized, set()).add("video")
    return [
        {"name": name, "score": count, "source_types": sorted(sources.get(name, set()))}
        for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _parameters(texts: list[str]) -> list[dict[str, Any]]:
    found: dict[tuple[str, str], dict[str, Any]] = {}
    for text in texts:
        for match in PARAM_PATTERN.finditer(text):
            value = match.group("value")
            unit = match.group("unit")
            key = (value, unit.lower())
            found.setdefault(key, {"value": float(value), "unit": unit, "text": match.group(0), "source_text": text[:240]})
    return list(found.values())


def _video_parameters(video_rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    params: list[dict[str, Any]] = []
    for row in video_rows:
        extracted = row.get("extracted_entities")
        if isinstance(extracted, Mapping):
            for item in extracted.get("parameters") or []:
                if isinstance(item, Mapping):
                    params.append({**dict(item), "source_event_id": row.get("video_event_id")})
        measurement = _video_measurement(row)
        for key in ("value", "reading", "readout", "display_text", "volume_ml", "volume_ul", "knob_angle_deg", "liquid_level_y_norm", "liquid_level_before", "liquid_level_after"):
            if key not in measurement or measurement.get(key) is None:
                continue
            unit = measurement.get("unit")
            if key == "volume_ml":
                unit = "ml"
            elif key == "volume_ul":
                unit = "ul"
            elif key == "knob_angle_deg":
                unit = "deg"
            elif key == "liquid_level_y_norm":
                unit = "normalized_y"
            elif key in {"liquid_level_before", "liquid_level_after"}:
                unit = "normalized_y"
            params.append(
                {
                    "name": key,
                    "value": measurement.get(key),
                    "unit": unit,
                    "source_event_id": row.get("video_event_id"),
                    "source_text": row.get("semantic_description") or row.get("text"),
                }
            )
    return params


def _merge_parameters(parameters: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    found: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in parameters:
        value = item.get("value")
        unit = str(item.get("unit") or "")
        name = str(item.get("name") or item.get("text") or "")
        key = (str(value), unit.lower(), name.lower())
        found.setdefault(key, dict(item))
    return list(found.values())


def _purpose(procedure: list[Mapping[str, Any]], materials: list[Mapping[str, Any]], texts: list[str]) -> str:
    action_names = {str(item.get("action_type")) for item in procedure}
    material_names = {str(item.get("name")) for item in materials}
    joined = " ".join(texts)
    if "pipetting" in action_names and "weighing" in action_names:
        return "sample preparation workflow with weighing and liquid transfer steps"
    if "pipetting" in action_names:
        return "liquid transfer or sample addition workflow"
    if "weighing" in action_names:
        return "sample weighing workflow"
    if "sample" in material_names or "样品" in joined:
        return "sample handling workflow"
    return "unknown_experiment_context"


def _gaps(procedure: list[Mapping[str, Any]], materials: list[Mapping[str, Any]], parameters: list[Mapping[str, Any]], video_rows: list[Mapping[str, Any]]) -> list[str]:
    gaps = []
    if not procedure:
        gaps.append("missing_procedure_candidates")
    if not materials:
        gaps.append("missing_materials")
    if not parameters:
        gaps.append("missing_explicit_parameters")
    if not video_rows:
        gaps.append("missing_video_understanding")
    return gaps


def _confidence(source_counts: Mapping[str, int], procedure: list[Mapping[str, Any]], materials: list[Mapping[str, Any]], parameters: list[Mapping[str, Any]]) -> float:
    score = 0.2
    if source_counts.get("video_events", 0) > 0:
        score += 0.25
    if source_counts.get("transcript_rows", 0) > 0 or source_counts.get("timeline_events", 0) > 0:
        score += 0.2
    if procedure:
        score += 0.15
    if materials:
        score += 0.1
    if parameters:
        score += 0.1
    return round(min(score, 1.0), 4)


def _read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.exists() else []


def _read_database_rows(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path_value in paths:
        path = Path(path_value)
        if not path.exists():
            continue
        if path.suffix.lower() == ".jsonl":
            rows.extend(read_jsonl(path))
        else:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            if isinstance(data, list):
                rows.extend(item for item in data if isinstance(item, dict))
            elif isinstance(data, dict):
                rows.append(data)
    return rows


def _related_database_records(
    database_rows: list[Mapping[str, Any]],
    procedure_candidates: list[Mapping[str, Any]],
    materials: list[Mapping[str, Any]],
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    query_actions = {str(item.get("action_type") or "") for item in procedure_candidates if item.get("action_type")}
    query_materials = {str(item.get("name") or "") for item in materials if item.get("name")}
    scored: list[dict[str, Any]] = []
    for index, row in enumerate(database_rows):
        actions = set(_record_actions(row))
        material_values = set(_record_materials(row))
        text = _row_text(row)
        matched_actions = sorted(action for action in actions if action in query_actions)
        matched_materials = sorted(material for material in material_values if material in query_materials)
        text_hits = [
            token
            for token in sorted(query_actions | query_materials)
            if token and token.lower() in text.lower()
        ]
        score = len(matched_actions) * 2.0 + len(matched_materials) * 1.5 + len(text_hits) * 0.5
        if score <= 0 and not text:
            continue
        record_id = str(row.get("record_id") or row.get("id") or row.get("session_id") or row.get("session") or f"database_record_{index + 1:03d}")
        scored.append(
            {
                "record_id": record_id,
                "session_id": row.get("session_id") or row.get("session"),
                "score": round(score, 4),
                "matched_actions": matched_actions,
                "matched_materials": matched_materials,
                "text_hits": sorted(set(text_hits)),
                "transition_sequence": _record_actions(row),
                "source_text": text[:240],
            }
        )
    scored.sort(key=lambda item: (-float(item["score"]), str(item["record_id"])))
    return scored[:limit]


def _transition_priors_from_related_records(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    transition_counts: Counter[tuple[str, str]] = Counter()
    for record in records:
        sequence = [str(item) for item in record.get("transition_sequence") or [] if item]
        transition_counts.update(zip(sequence, sequence[1:]))
    totals: Counter[str] = Counter()
    for (source, _destination), count in transition_counts.items():
        totals[source] += count
    probabilities: dict[str, dict[str, float]] = {}
    for (source, destination), count in sorted(transition_counts.items()):
        probabilities.setdefault(source, {})[destination] = round(count / totals[source], 6) if totals[source] else 0.0
    return {
        "record_count": len(records),
        "transition_counts": {f"{source}->{destination}": count for (source, destination), count in sorted(transition_counts.items())},
        "transition_probabilities": probabilities,
    }


def _record_actions(row: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("action_type", "expected_action", "event_type"):
        if row.get(key):
            values.append(_normalize_action(row[key]))
    for key in ("procedure", "steps", "events", "actions"):
        for item in _strings(row.get(key)):
            if isinstance(item, str):
                values.append(_normalize_action(item))
    for item in row.get("steps", []) if isinstance(row.get("steps"), list) else []:
        if isinstance(item, Mapping):
            values.extend(_record_actions(item))
    return [value for value in _dedupe(values) if value]


def _record_materials(row: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("primary_object", "object_label", "materials", "objects"):
        for item in _strings(row.get(key)):
            normalized = _normalize_material(item)
            if normalized:
                values.append(normalized)
    for item in row.get("steps", []) if isinstance(row.get("steps"), list) else []:
        if isinstance(item, Mapping):
            values.extend(_record_materials(item))
    return [value for value in _dedupe(values) if value]


def _normalize_action(value: Any) -> str:
    text = str(value or "").lower()
    for action, keywords in KEYWORD_MAP.items():
        if action in text or any(keyword.lower() in text for keyword in keywords):
            return action
    return str(value or "").strip()


def _row_text(row: Mapping[str, Any]) -> str:
    return str(row.get("text") or row.get("content") or row.get("message") or row.get("summary") or row.get("description") or "")


def _join_text(*values: Any) -> str:
    return " ".join(str(value) for value in values if value)


def _strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        values: list[str] = []
        for nested in value.values():
            values.extend(_strings(nested))
        return values
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _normalize_material(value: Any) -> str:
    text = str(value or "").lower()
    for material, keywords in MATERIAL_KEYWORDS.items():
        if material in text or any(keyword.lower() in text for keyword in keywords):
            return material
    return ""


def _normalize_keyword_entity(value: Any, keywords_by_name: Mapping[str, Iterable[str]]) -> str:
    text = str(value or "").lower()
    for name, keywords in keywords_by_name.items():
        if name.lower() in text or any(str(keyword).lower() in text for keyword in keywords):
            return name
    return ""


def _video_measurement(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = _as_dict(row.get("payload"))
    observation = _as_dict(payload.get("model_observation"))
    measurement = _as_dict(observation.get("measurement"))
    if measurement:
        return measurement
    advanced = _as_dict(payload.get("advanced_evidence"))
    return _as_dict(advanced.get("metrics"))


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dedupe(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def _session_id(*sources: Any) -> str:
    for source in sources[:-1]:
        for row in source:
            if isinstance(row, Mapping) and row.get("session_id"):
                return str(row["session_id"])
    session = sources[-1]
    manifest = Path(session) / "manifest.json"
    if manifest.exists():
        try:
            return str(json.loads(manifest.read_text(encoding="utf-8")).get("session_id") or "")
        except (OSError, json.JSONDecodeError):
            pass
    return Path(session).name


__all__ = ["build_experiment_context", "load_experiment_context"]

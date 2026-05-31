from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CATALOG_FILENAME = "material_asset_catalog.jsonl"

_SEARCH_FIELD_WEIGHTS = {
    "search_text": 4.0,
    "objects": 3.0,
    "actions": 3.0,
    "state_tags": 2.5,
    "event_type": 2.5,
    "confirmation_level": 2.0,
    "path": 1.5,
    "evidence_level": 1.0,
    "evidence_refs": 1.0,
}


def load_material_catalog(path_or_session_dir: str | Path) -> list[dict[str, Any]]:
    """Load a material asset catalog from a JSONL path or a session directory."""
    catalog_path = _resolve_catalog_path(path_or_session_dir)
    rows: list[dict[str, Any]] = []
    with catalog_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {catalog_path} at line {line_number}: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object rows in {catalog_path} at line {line_number}")
            rows.append(dict(row))
    return rows


def search_material_assets(
    path_or_session_dir: str | Path,
    query: str = "",
    asset_type: str | None = None,
    objects: str | list[str] | tuple[str, ...] | set[str] | None = None,
    actions: str | list[str] | tuple[str, ...] | set[str] | None = None,
    state_tags: str | list[str] | tuple[str, ...] | set[str] | None = None,
    start_time: str | datetime | int | float | None = None,
    end_time: str | datetime | int | float | None = None,
    limit: int | None = 20,
) -> list[dict[str, Any]]:
    """Search material assets with offline keyword, metadata, and time filters."""
    catalog = load_material_catalog(path_or_session_dir)
    query_terms = _query_terms(query)
    requested_objects = _normalize_requested_values(objects)
    requested_actions = _normalize_requested_values(actions)
    requested_state_tags = _normalize_requested_values(state_tags)
    requested_asset_type = _normalize_text(asset_type) if asset_type is not None else ""
    time_window = _build_time_window(start_time, end_time)

    results: list[dict[str, Any]] = []
    for asset in catalog:
        match_reasons: list[str] = []
        score = 0.0

        if requested_asset_type:
            if _normalize_text(asset.get("asset_type")) != requested_asset_type:
                continue
            match_reasons.append(f"asset_type={asset_type}")
            score += 0.5

        filter_score, filter_reasons = _score_list_filters(
            asset,
            {
                "objects": requested_objects,
                "actions": requested_actions,
                "state_tags": requested_state_tags,
            },
        )
        if filter_score is None:
            continue
        score += filter_score
        match_reasons.extend(filter_reasons)

        if time_window is not None:
            if not _asset_overlaps_time_window(asset, time_window):
                continue
            match_reasons.append("time_range_overlap")
            score += 0.5

        if query_terms:
            keyword_score, keyword_reasons = _score_keyword_match(asset, query_terms)
            if keyword_score <= 0.0:
                continue
            score += keyword_score
            match_reasons.extend(keyword_reasons)

        if not match_reasons:
            match_reasons.append("catalog_entry")

        result = dict(asset)
        result["score"] = round(score, 6)
        result["match_reasons"] = match_reasons
        results.append(result)

    results.sort(key=lambda item: (-float(item["score"]), _time_sort_key(item), str(item.get("asset_id") or item.get("path") or "")))

    if limit is None:
        return results
    limit_count = max(0, int(limit))
    return results[:limit_count]


def search_material_index(
    session_dir: str | Path,
    query: str = "",
    asset_type: str | None = None,
    objects: str | list[str] | tuple[str, ...] | set[str] | None = None,
    actions: str | list[str] | tuple[str, ...] | set[str] | None = None,
    state_tags: str | list[str] | tuple[str, ...] | set[str] | None = None,
    start_time: str | datetime | int | float | None = None,
    end_time: str | datetime | int | float | None = None,
    index_level: str | None = None,
    limit: int | None = 20,
) -> dict[str, Any]:
    """Search both material assets and the local vector index with shared filters."""
    session = Path(session_dir)
    assets = search_material_assets(
        session,
        query=query,
        asset_type=asset_type,
        objects=objects,
        actions=actions,
        state_tags=state_tags,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
    )
    vector_hits: list[dict[str, Any]] = []
    index_dir = session / "index"
    if index_dir.exists() and (index_dir / "fallback_index.pkl").exists():
        from .vector_index import VectorIndex

        filters: dict[str, Any] = {}
        if index_level and index_level != "all":
            filters["index_level"] = index_level
        if objects:
            filters["objects"] = objects
        if actions:
            filters["actions"] = actions
        if asset_type:
            filters["asset_type"] = asset_type
        if start_time is not None:
            filters["start_time"] = start_time
        if end_time is not None:
            filters["end_time"] = end_time
        vector_hits = VectorIndex.load(index_dir).query(query or " ".join(_normalize_requested_values(objects) + _normalize_requested_values(actions)), top_k=limit or 20, filters=filters)
    return {
        "query": query,
        "filters": {
            "asset_type": asset_type,
            "objects": _normalize_requested_values(objects),
            "actions": _normalize_requested_values(actions),
            "state_tags": _normalize_requested_values(state_tags),
            "start_time": str(start_time) if start_time is not None else None,
            "end_time": str(end_time) if end_time is not None else None,
            "index_level": index_level,
        },
        "assets": assets,
        "segments": vector_hits,
        "asset_count": len(assets),
        "segment_count": len(vector_hits),
    }


def _resolve_catalog_path(path_or_session_dir: str | Path) -> Path:
    source = Path(path_or_session_dir)
    candidates = [source]
    if source.is_dir():
        candidates = [
            source / "metadata" / CATALOG_FILENAME,
            source / CATALOG_FILENAME,
        ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    expected = source / "metadata" / CATALOG_FILENAME if source.is_dir() else source
    raise FileNotFoundError(f"Material asset catalog not found: {expected}")


def _query_terms(query: str | None) -> list[str]:
    text = _normalize_text(query)
    if not text:
        return []
    terms: list[str] = []
    split_terms = [part for part in re.split(r"[\s,;，；]+", text) if part]
    if len(split_terms) > 1:
        terms.append(text)
    terms.extend(split_terms or [text])
    return _dedupe(terms)


def _score_keyword_match(asset: dict[str, Any], query_terms: list[str]) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    for field, weight in _SEARCH_FIELD_WEIGHTS.items():
        field_text = _normalize_text(" ".join(_search_field_values(asset, field)))
        if not field_text:
            continue
        matched_terms = [term for term in query_terms if term in field_text]
        if matched_terms:
            score += weight * len(matched_terms)
            reasons.append(f"query:{field}:{','.join(matched_terms)}")
    return score, reasons


def _score_list_filters(asset: dict[str, Any], filters: dict[str, list[str]]) -> tuple[float | None, list[str]]:
    score = 0.0
    reasons: list[str] = []
    for field, requested_values in filters.items():
        if not requested_values:
            continue
        asset_values = {_normalize_text(value) for value in _search_field_values(asset, field)}
        if not all(value in asset_values for value in requested_values):
            return None, []
        score += 0.75 * len(requested_values)
        reasons.extend(f"{field}={value}" for value in requested_values)
    return score, reasons


def _search_field_values(asset: dict[str, Any], field: str) -> list[str]:
    if field == "evidence_level":
        values = _flatten_to_strings(asset.get("evidence_level"))
        quality = asset.get("quality")
        if isinstance(quality, dict):
            values.extend(_flatten_to_strings(quality.get("evidence_level")))
        evidence = asset.get("evidence")
        if isinstance(evidence, dict):
            values.extend(_flatten_to_strings(evidence.get("evidence_level")))
        return values
    return _flatten_to_strings(asset.get(field))


def _flatten_to_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        values: list[str] = []
        for item in value.values():
            values.extend(_flatten_to_strings(item))
        return values
    if isinstance(value, (list, tuple, set)):
        values = []
        for item in value:
            values.extend(_flatten_to_strings(item))
        return values
    return [str(value)]


def _normalize_requested_values(value: str | list[str] | tuple[str, ...] | set[str] | None) -> list[str]:
    values: list[str] = []
    for item in _flatten_to_strings(value):
        values.extend(part for part in re.split(r"[,;，；]+", item) if part.strip())
    return _dedupe([_normalize_text(item) for item in values if _normalize_text(item)])


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).casefold().strip()


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def _build_time_window(
    start_time: str | datetime | int | float | None,
    end_time: str | datetime | int | float | None,
) -> tuple[float | None, float | None] | None:
    if start_time is None and end_time is None:
        return None
    start = _parse_filter_time(start_time, "start_time") if start_time is not None else None
    end = _parse_filter_time(end_time, "end_time") if end_time is not None else None
    if start is not None and end is not None and end < start:
        start, end = end, start
    return start, end


def _asset_overlaps_time_window(asset: dict[str, Any], time_window: tuple[float | None, float | None]) -> bool:
    query_start, query_end = time_window
    asset_start = _parse_time_value(asset.get("global_start_time"))
    asset_end = _parse_time_value(asset.get("global_end_time"))
    if asset_start is None and asset_end is None:
        return False
    if asset_start is None:
        asset_start = asset_end
    if asset_end is None:
        asset_end = asset_start
    if asset_start is not None and asset_end is not None and asset_end < asset_start:
        asset_start, asset_end = asset_end, asset_start
    if query_start is not None and asset_end is not None and asset_end < query_start:
        return False
    if query_end is not None and asset_start is not None and asset_start > query_end:
        return False
    return True


def _time_sort_key(asset: dict[str, Any]) -> tuple[int, float, str]:
    parsed = _parse_time_value(asset.get("global_start_time"))
    if parsed is None:
        parsed = _parse_time_value(asset.get("global_end_time"))
    if parsed is None:
        return (1, 0.0, str(asset.get("global_start_time") or asset.get("global_end_time") or ""))
    return (0, parsed, "")


def _parse_filter_time(value: str | datetime | int | float, field_name: str) -> float:
    parsed = _parse_time_value(value)
    if parsed is None:
        raise ValueError(f"Invalid {field_name}: {value}")
    return parsed


def _parse_time_value(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        timestamp_value = value
        if timestamp_value.tzinfo is None:
            timestamp_value = timestamp_value.replace(tzinfo=timezone.utc)
        return timestamp_value.timestamp()
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    iso_text = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(iso_text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


__all__ = ["load_material_catalog", "search_material_assets", "search_material_index"]

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, Iterable, Optional

DISPLAY_LABELS = {
    "hand_object_interaction": "\u624b\u90e8\u4e0e\u7269\u4f53\u4ea4\u4e92",
    "object_move": "\u7269\u4f53\u79fb\u52a8",
    "liquid_transfer": "\u6db2\u4f53\u8f6c\u79fb",
    "panel_operation": "\u8bbe\u5907\u9762\u677f\u64cd\u4f5c",
    "container_state_change": "\u5bb9\u5668\u72b6\u6001\u53d8\u5316",
}


def slugify(value: Any, *, fallback: str = "item", max_length: int = 80) -> str:
    """Return a stable ASCII path-safe slug."""
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return (text or fallback)[:max_length].strip("_") or fallback


def timecode(start: float, end: float) -> str:
    return f"t{int(max(0, start)):03d}_{int(max(0, end)):03d}"


def container_label(container: Optional[Dict[str, Any]]) -> Optional[str]:
    if not container:
        return None
    return str(
        container.get("class_name")
        or container.get("object_name")
        or container.get("display_name")
        or container.get("track_id")
        or ""
    ).strip() or None


def object_descriptor(event: Dict[str, Any]) -> str:
    event_type = event.get("event_type")
    if event_type == "liquid_transfer":
        source = slugify(container_label(event.get("source_container")), fallback="source")
        target = slugify(container_label(event.get("target_container")), fallback="target")
        return f"{source}_to_{target}"
    dominant = event.get("dominant_object") or next(iter(event.get("involved_objects") or []), None)
    if dominant:
        return slugify(dominant, fallback="object")
    return "object"


def stable_name(experiment_name: str, event: Dict[str, Any]) -> str:
    start = float(event.get("start_time_sec") or event.get("time_start") or 0.0)
    end = float(event.get("end_time_sec") or event.get("time_end") or start)
    return "__".join(
        [
            slugify(experiment_name, fallback="experiment", max_length=40),
            slugify(event.get("event_type"), fallback="event", max_length=40),
            object_descriptor(event),
            timecode(start, end),
        ]
    )


def display_name(experiment_name: str, event: Dict[str, Any]) -> str:
    if event.get("display_name"):
        return str(event["display_name"])
    prefix = str(experiment_name or "\u5b9e\u9a8c").strip() or "\u5b9e\u9a8c"
    label = DISPLAY_LABELS.get(str(event.get("event_type")), str(event.get("event_type") or "\u5173\u952e\u4e8b\u4ef6"))
    return f"{prefix}-{label}"


def searchable_text(parts: Iterable[Any]) -> str:
    return " ".join(str(part) for part in parts if part not in (None, "", [], {}))

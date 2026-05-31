from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable, Set

try:
    import yaml
except Exception:  # pragma: no cover - yaml is part of project deps
    yaml = None

HAND_TERMS = {"hand", "glove", "gloved_hand", "gloved-hand", "left_hand", "right_hand", "arm"}
CONTAINER_TERMS = {
    "bottle",
    "beaker",
    "vial",
    "tube",
    "cup",
    "jar",
    "container",
    "flask",
    "reagent_bottle",
    "sample_bottle",
    "sample_bottle_blue",
}
TOOL_TERMS = {"pipette", "dropper", "tool", "spoon", "spatula", "spearhead"}
PANEL_TERMS = {"panel", "button", "screen", "display", "balance", "scale", "device"}
LID_TERMS = {"lid", "cap", "cover", "tube-cap", "tube_cap"}
IGNORE_INTERACTION_TERMS = {"lab_coat", "paper", "reagent_label", "label", "bottle_label"}


def norm_label(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def has_any_label(value: str, terms: Iterable[str]) -> bool:
    text = norm_label(value)
    return any(term in text for term in terms)


@lru_cache(maxsize=1)
def configured_class_names() -> Set[str]:
    path = Path("configs") / "data" / "class_schema.yaml"
    if not path.exists() or yaml is None:
        return set()
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return set()
    return {norm_label(item.get("name")) for item in payload.get("classes") or [] if item.get("name")}


def is_hand_label(value: str) -> bool:
    return has_any_label(value, HAND_TERMS)


def is_container_label(value: str) -> bool:
    return has_any_label(value, CONTAINER_TERMS)


def is_tool_label(value: str) -> bool:
    return has_any_label(value, TOOL_TERMS)


def is_panel_label(value: str) -> bool:
    return has_any_label(value, PANEL_TERMS)


def is_lid_label(value: str) -> bool:
    return has_any_label(value, LID_TERMS)


def is_interaction_object_label(value: str) -> bool:
    label = norm_label(value)
    if not label or is_hand_label(label):
        return False
    if label in IGNORE_INTERACTION_TERMS:
        return False
    known = configured_class_names()
    return not known or label in known or is_container_label(label) or is_tool_label(label) or is_panel_label(label) or is_lid_label(label)

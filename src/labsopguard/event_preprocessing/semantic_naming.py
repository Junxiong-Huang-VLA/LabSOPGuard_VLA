from __future__ import annotations

import re
from typing import List

TYPE_LABELS = {
    "hand_object_interaction": "hand object interaction",
    "object_move": "object move",
    "liquid_transfer": "liquid transfer",
    "panel_operation": "panel operation",
    "container_state_change": "container state change",
}

DISPLAY_LABELS = {
    "hand_object_interaction": "手部接触操作",
    "object_move": "物体移动",
    "liquid_transfer": "液体转移",
    "panel_operation": "面板/设备操作",
    "container_state_change": "容器状态变化",
}


def _slug(value: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "event"


class SemanticNamer:
    def stable_name(self, experiment_name: str, event_type: str, involved_objects: List[str], start: float, end: float) -> str:
        exp = _slug(experiment_name)[:32] if experiment_name else "experiment"
        objects = "_to_".join(_slug(item) for item in involved_objects[:2] if item) or "object"
        return f"{exp}__{_slug(event_type)}__{objects}__t{int(start):03d}_{int(end):03d}"

    def display_name(self, experiment_name: str, event_type: str, involved_objects: List[str]) -> str:
        prefix = str(experiment_name or "实验").strip() or "实验"
        label = DISPLAY_LABELS.get(event_type, TYPE_LABELS.get(event_type, event_type))
        objects = "、".join(str(item) for item in involved_objects[:3] if item)
        if objects:
            return f"{prefix}_{label}_{objects}"
        return f"{prefix}_{label}"

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


STAGE_SCOPE_SCHEMA_VERSION = "key_action_stage_scope.v1"

DEFAULT_STAGE_SCOPE: dict[str, Any] = {
    "schema_version": STAGE_SCOPE_SCHEMA_VERSION,
    "scope_name": "p2_hand_object_current_scope",
    "stage": "P2",
    "status": "active",
    "in_scope_objects": [
        "balance",
        "pipette",
        "pipette_tip",
        "container",
        "paper",
        "spatula",
        "beaker",
        "reagent_bottle",
        "sample_bottle",
        "sample_bottle_blue",
    ],
    "in_scope_actions": [
        "weighing",
        "pipetting",
        "sample_handling",
        "hand_object_interaction",
        "bottle_interaction",
        "spatula_interaction",
    ],
    "out_of_scope_capabilities": [
        "container_open_closed_detection",
        "equipment_control_state_detection",
        "equipment_display_readout",
        "liquid_level_detection",
        "liquid_stream_segmentation",
        "meniscus_detection",
        "panel_button_knob_detection",
    ],
    "out_of_scope_labels": [
        "button",
        "cap_closed",
        "cap_open",
        "closed",
        "container_closed",
        "container_open",
        "display",
        "display_readout",
        "knob",
        "lid_closed",
        "lid_open",
        "liquid",
        "liquid_level",
        "liquid_region",
        "liquid_stream",
        "meniscus",
        "meniscus_line",
        "open",
        "stream",
    ],
    "qa_policy": {
        "out_of_scope_gap_status": "informational",
        "out_of_scope_gaps_block_quality_gate": False,
    },
    "rationale": (
        "Current P2 scope focuses on existing YOLO-backed hand-object evidence, "
        "micro-segments, retrieval, and multi-session history. Deferred labels "
        "should be surfaced by QA but must not block this stage."
    ),
}


def build_stage_scope(
    session_dir: str | Path,
    output_path: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    session = Path(session_dir)
    scope = deepcopy(DEFAULT_STAGE_SCOPE)
    if overrides:
        scope.update(dict(overrides))
    scope["created_at"] = datetime.now(timezone.utc).isoformat()
    scope["session_dir"] = str(session)
    target = Path(output_path) if output_path else session / "metadata" / "stage_scope.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(scope, ensure_ascii=False, indent=2), encoding="utf-8")
    return scope


def load_stage_scope(path_or_session_dir: str | Path) -> dict[str, Any]:
    source = Path(path_or_session_dir)
    candidates = []
    if source.is_file():
        candidates.append(source)
    else:
        candidates.extend([source / "metadata" / "stage_scope.json", source / "stage_scope.json"])
    for candidate in candidates:
        if candidate.exists():
            data = json.loads(candidate.read_text(encoding="utf-8-sig"))
            return dict(data) if isinstance(data, Mapping) else {}
    return {}


def is_out_of_scope(value: Any, scope: Mapping[str, Any] | None) -> bool:
    if not scope:
        return False
    token = _norm(value)
    if not token:
        return False
    out_values = _scope_tokens(scope, "out_of_scope_capabilities") | _scope_tokens(scope, "out_of_scope_labels")
    if token in out_values:
        return True
    return any(part and part in out_values for part in token.replace("-", "_").split("_"))


def split_scope_values(values: Any, scope: Mapping[str, Any] | None) -> dict[str, list[str]]:
    in_scope: list[str] = []
    out_of_scope: list[str] = []
    for value in _as_list(values):
        text = str(value or "").strip()
        if not text:
            continue
        if is_out_of_scope(text, scope):
            out_of_scope.append(text)
        else:
            in_scope.append(text)
    return {
        "in_scope": _ordered_unique(in_scope),
        "out_of_scope": _ordered_unique(out_of_scope),
    }


def _scope_tokens(scope: Mapping[str, Any], key: str) -> set[str]:
    return {_norm(item) for item in _as_list(scope.get(key)) if _norm(item)}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return sorted(value)
    return [value]


def _norm(value: Any) -> str:
    return str(value or "").strip().casefold().replace("-", "_").replace(" ", "_")


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

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .naming import slugify, timecode

EVENT_TYPE_DIRS = {
    "hand_object_interaction": "hand_object_interaction",
    "object_move": "object_move",
    "liquid_transfer": "liquid_transfer",
    "panel_operation": "panel_operation",
    "container_state_change": "container_state_change",
}


def _safe_dir(name: str) -> str:
    name = re.sub(r'[\\/:*?"<>|]', "_", name)
    name = re.sub(r"\s+", "_", name.strip())
    return name[:80] or "unnamed"


@dataclass
class ArchivePlan:
    event_id: str
    publish_dir: Path
    clip_path: Optional[Path]
    preview_path: Optional[Path]
    keyframe_paths: List[Path]
    event_json_path: Path
    material_publish_path: Path
    relative_publish_dir: str


class ArchivePlanner:
    """Build stable archive paths for published materials.

    Structure:
      published_materials/
        {actor}/
          {event_type}/
            {timecode}_{semantic_name}/
              clip.mp4, preview.jpg, keyframe_*.jpg, event.json
    """

    def __init__(self, experiment_dir: str | Path) -> None:
        self.experiment_dir = Path(experiment_dir)
        self.root = self.experiment_dir / "published_materials"

    def plan(self, *, event: Dict, stable_name: str, actor_name: str) -> ArchivePlan:
        start = float(event.get("start_time_sec") or 0.0)
        end = float(event.get("end_time_sec") or start)

        actor = _safe_dir(actor_name or "operator")
        event_type_raw = str(event.get("event_type") or "event")
        event_type_dir = EVENT_TYPE_DIRS.get(event_type_raw, event_type_raw)

        display = str(event.get("display_name") or "").strip()
        if display:
            folder = f"{timecode(start, end)}_{_safe_dir(display)}"
        else:
            folder = f"{timecode(start, end)}_{slugify(stable_name, fallback=event.get('event_id', 'event'), max_length=80)}"

        publish_dir = self.root / actor / event_type_dir / folder
        rel = publish_dir.relative_to(self.experiment_dir).as_posix()
        return ArchivePlan(
            event_id=str(event.get("event_id")),
            publish_dir=publish_dir,
            clip_path=publish_dir / "clip.mp4",
            preview_path=publish_dir / "preview.jpg",
            keyframe_paths=[publish_dir / f"keyframe_{idx:02d}.jpg" for idx in range(1, 4)],
            event_json_path=publish_dir / "event.json",
            material_publish_path=publish_dir / "material_publish.json",
            relative_publish_dir=rel,
        )

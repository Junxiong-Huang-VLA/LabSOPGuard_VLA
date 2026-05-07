from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class FileBackedTaskStore:
    base_dir: Path

    def __post_init__(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _task_file(self, task_key: str) -> Path:
        return self.base_dir / f"{task_key}.json"

    def create(self, task_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        state = {
            "task_id": task_id,
            "status": "pending",
            "progress": 0.0,
            "message": "",
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
        }
        state.update(payload)
        self._task_file(task_id).write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return state

    def update(self, task_key: str, **fields: Any) -> Dict[str, Any]:
        state = self.get(task_key)
        state.update(fields)
        state["updated_at"] = _now_iso()
        self._task_file(task_key).write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return state

    def get(self, task_id: str) -> Dict[str, Any]:
        task_file = self._task_file(task_id)
        if not task_file.exists():
            return {}
        return json.loads(task_file.read_text(encoding="utf-8"))

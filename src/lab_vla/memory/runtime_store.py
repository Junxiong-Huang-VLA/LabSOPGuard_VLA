from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict


class RuntimeStore:
    def __init__(self, trace_jsonl: str, summary_json: str) -> None:
        self.trace_path = Path(trace_jsonl)
        self.summary_path = Path(summary_json)
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, row: Dict[str, Any]) -> None:
        with self.trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def write_summary(self, payload: Dict[str, Any]) -> None:
        self.summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def to_dict(obj: Any) -> Dict[str, Any]:
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, dict):
            return obj
        return {"value": obj}


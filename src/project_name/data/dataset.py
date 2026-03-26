from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from project_name.common.io_utils import read_jsonl


@dataclass
class VLASample:
    sample_id: str
    rgb_path: str
    depth_path: str | None
    instruction: str
    action_history: List[str]
    action_sequence: List[str]


class VLADataset:
    def __init__(self, annotation_file: str):
        self.records: List[Dict[str, Any]] = read_jsonl(annotation_file)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> VLASample:
        item = self.records[idx]
        return VLASample(
            sample_id=str(item.get("sample_id", idx)),
            rgb_path=item.get("rgb_path", ""),
            depth_path=item.get("depth_path"),
            instruction=item.get("instruction", ""),
            action_history=item.get("action_history", []),
            action_sequence=item.get("action_sequence", []),
        )

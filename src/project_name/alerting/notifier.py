from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List

from project_name.monitoring.sop_engine import ViolationEvent


class AlertNotifier:
    def __init__(self, alert_file: str = "outputs/predictions/alerts.jsonl") -> None:
        self.alert_file = Path(alert_file)
        self.alert_file.parent.mkdir(parents=True, exist_ok=True)

    def notify(self, violations: Iterable[ViolationEvent]) -> List[Dict]:
        rows: List[Dict] = []
        with self.alert_file.open("a", encoding="utf-8") as f:
            for event in violations:
                row = asdict(event)
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return rows

    def send_console(self, violations: Iterable[ViolationEvent]) -> None:
        for event in violations:
            print(f"[ALERT][{event.severity}] {event.rule_id} @ {event.timestamp_sec:.2f}s: {event.message}")

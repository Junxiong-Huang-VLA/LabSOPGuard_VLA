from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .schemas import IndexedMaterialRecord, PhysicalEvent


class EventMaterialIndexWriter:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self.conn.row_factory = sqlite3.Row
        self.init_schema()

    def close(self) -> None:
        self.conn.close()

    def init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS event_materials (
                material_id TEXT PRIMARY KEY,
                experiment_id TEXT,
                event_id TEXT UNIQUE,
                event_type TEXT,
                display_name TEXT,
                stable_name TEXT,
                actor_name TEXT,
                source_container_json TEXT,
                target_container_json TEXT,
                source_container_class TEXT,
                source_container_track_id TEXT,
                target_container_class TEXT,
                target_container_track_id TEXT,
                actor_track_id TEXT,
                tool_track_id TEXT,
                transfer_mode TEXT,
                direction_confidence REAL,
                direction_status TEXT,
                evidence_grade TEXT,
                review_status TEXT,
                time_start REAL,
                time_end REAL,
                duration_sec REAL,
                semantic_tags TEXT,
                involved_objects_json TEXT,
                clip_path TEXT,
                preview_path TEXT,
                keyframe_count INTEGER,
                quality_score REAL,
                quality_grade TEXT,
                quality_reasons_json TEXT,
                qwen_summary TEXT,
                linked_step_id TEXT,
                searchable_text TEXT,
                created_at TEXT,
                metadata_version TEXT,
                payload_json TEXT
            )
            """
        )
        for statement in (
            "ALTER TABLE event_materials ADD COLUMN source_container_json TEXT",
            "ALTER TABLE event_materials ADD COLUMN target_container_json TEXT",
            "ALTER TABLE event_materials ADD COLUMN source_container_class TEXT",
            "ALTER TABLE event_materials ADD COLUMN source_container_track_id TEXT",
            "ALTER TABLE event_materials ADD COLUMN target_container_class TEXT",
            "ALTER TABLE event_materials ADD COLUMN target_container_track_id TEXT",
            "ALTER TABLE event_materials ADD COLUMN actor_track_id TEXT",
            "ALTER TABLE event_materials ADD COLUMN tool_track_id TEXT",
            "ALTER TABLE event_materials ADD COLUMN transfer_mode TEXT",
            "ALTER TABLE event_materials ADD COLUMN direction_confidence REAL",
            "ALTER TABLE event_materials ADD COLUMN direction_status TEXT",
            "ALTER TABLE event_materials ADD COLUMN evidence_grade TEXT",
            "ALTER TABLE event_materials ADD COLUMN review_status TEXT",
            "ALTER TABLE event_materials ADD COLUMN published_path TEXT",
            "ALTER TABLE event_materials ADD COLUMN material_publish_path TEXT",
            "ALTER TABLE event_materials ADD COLUMN quality_score REAL",
            "ALTER TABLE event_materials ADD COLUMN quality_grade TEXT",
            "ALTER TABLE event_materials ADD COLUMN quality_reasons_json TEXT",
        ):
            try:
                self.conn.execute(statement)
            except sqlite3.OperationalError:
                pass
        for statement in (
            "CREATE INDEX IF NOT EXISTS idx_event_materials_experiment ON event_materials(experiment_id)",
            "CREATE INDEX IF NOT EXISTS idx_event_materials_type ON event_materials(event_type)",
            "CREATE INDEX IF NOT EXISTS idx_event_materials_time ON event_materials(time_start, time_end)",
            "CREATE INDEX IF NOT EXISTS idx_event_materials_actor ON event_materials(actor_name)",
            "CREATE INDEX IF NOT EXISTS idx_event_materials_actor_track ON event_materials(actor_track_id)",
            "CREATE INDEX IF NOT EXISTS idx_event_materials_source_container ON event_materials(source_container_class, source_container_track_id)",
            "CREATE INDEX IF NOT EXISTS idx_event_materials_target_container ON event_materials(target_container_class, target_container_track_id)",
            "CREATE INDEX IF NOT EXISTS idx_event_materials_evidence_grade ON event_materials(evidence_grade, review_status)",
            "CREATE INDEX IF NOT EXISTS idx_event_materials_display ON event_materials(display_name)",
            "CREATE INDEX IF NOT EXISTS idx_event_materials_published ON event_materials(published_path)",
            "CREATE INDEX IF NOT EXISTS idx_event_materials_quality ON event_materials(quality_grade, quality_score)",
        ):
            self.conn.execute(statement)
        self.conn.commit()

    def reset_experiment(self, experiment_id: str) -> None:
        self.conn.execute("DELETE FROM event_materials WHERE experiment_id = ?", (experiment_id,))
        self.conn.commit()

    def write_events(self, events: List[PhysicalEvent]) -> List[IndexedMaterialRecord]:
        records = [self._record_from_event(event) for event in events]
        for record, event in zip(records, events):
            row = record.to_dict()
            row["semantic_tags"] = json.dumps(record.semantic_tags, ensure_ascii=False)
            row["payload_json"] = json.dumps(event.to_dict(), ensure_ascii=False)
            self.conn.execute(
                """
                INSERT OR REPLACE INTO event_materials
                (material_id, experiment_id, event_id, event_type, display_name, stable_name, actor_name,
                 source_container_json, target_container_json, source_container_class, source_container_track_id,
                 target_container_class, target_container_track_id, actor_track_id, tool_track_id, transfer_mode,
                 direction_confidence, direction_status, evidence_grade, review_status,
                 time_start, time_end, duration_sec, semantic_tags, involved_objects_json, clip_path,
                 preview_path, keyframe_count, quality_score, quality_grade, quality_reasons_json,
                 qwen_summary, linked_step_id, searchable_text, created_at, metadata_version, payload_json)
                VALUES (:material_id, :experiment_id, :event_id, :event_type, :display_name, :stable_name,
                 :actor_name, :source_container_json, :target_container_json, :source_container_class, :source_container_track_id,
                 :target_container_class, :target_container_track_id, :actor_track_id, :tool_track_id, :transfer_mode,
                 :direction_confidence, :direction_status, :evidence_grade, :review_status,
                 :time_start, :time_end, :duration_sec, :semantic_tags, :involved_objects_json,
                 :clip_path, :preview_path, :keyframe_count, :quality_score, :quality_grade,
                 :quality_reasons_json, :qwen_summary, :linked_step_id, :searchable_text, :created_at,
                 :metadata_version, :payload_json)
                """,
                row,
            )
        self.conn.commit()
        return records

    def query_events(
        self,
        *,
        experiment_id: Optional[str] = None,
        event_type: Optional[str] = None,
        actor_name: Optional[str] = None,
        display_name: Optional[str] = None,
        source_container_class: Optional[str] = None,
        target_container_class: Optional[str] = None,
        start_time_sec: Optional[float] = None,
        end_time_sec: Optional[float] = None,
        text: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM event_materials"
        clauses: List[str] = []
        params: List[Any] = []
        if experiment_id:
            clauses.append("experiment_id = ?")
            params.append(experiment_id)
        if event_type:
            clauses.append("event_type = ?")
            params.append(event_type)
        if actor_name:
            clauses.append("actor_name = ?")
            params.append(actor_name)
        if display_name:
            clauses.append("display_name LIKE ?")
            params.append(f"%{display_name}%")
        if source_container_class:
            clauses.append("source_container_class LIKE ?")
            params.append(f"%{source_container_class}%")
        if target_container_class:
            clauses.append("target_container_class LIKE ?")
            params.append(f"%{target_container_class}%")
        if start_time_sec is not None:
            clauses.append("time_end >= ?")
            params.append(float(start_time_sec))
        if end_time_sec is not None:
            clauses.append("time_start <= ?")
            params.append(float(end_time_sec))
        if text:
            clauses.append("searchable_text LIKE ?")
            params.append(f"%{text}%")
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY time_start LIMIT ?"
        params.append(max(1, int(limit)))
        rows = []
        for row in self.conn.execute(sql, params).fetchall():
            item = dict(row)
            for key in ("semantic_tags", "involved_objects_json", "source_container_json", "target_container_json", "quality_reasons_json", "payload_json"):
                try:
                    parsed = json.loads(item.get(key) or ("{}" if key in {"source_container_json", "target_container_json", "payload_json"} else "[]"))
                except Exception:
                    parsed = [] if key not in {"source_container_json", "target_container_json", "payload_json"} else {}
                item[key.replace("_json", "")] = parsed
            rows.append(item)
        return rows

    @staticmethod
    def _record_from_event(event: PhysicalEvent) -> IndexedMaterialRecord:
        asset = event.asset_pack or {}
        tags = list(dict.fromkeys([event.event_type, *event.involved_objects, event.actor_name]))
        qwen_summary = None
        searchable = " ".join(
            str(part)
            for part in [
                event.display_name,
                event.stable_name,
                event.event_type,
                event.actor_name,
                event.source_container,
                event.target_container,
                event.primary_track_id,
                " ".join(event.involved_track_ids),
                event.direction_status,
                event.evidence_grade,
                event.review_status,
                event.evidence_summary,
                asset.get("quality_grade"),
                " ".join(asset.get("quality_reasons") or []),
                " ".join(event.involved_objects),
                " ".join(event.related_detection_classes),
                event.notes,
            ]
            if part
        )
        return IndexedMaterialRecord(
            material_id=f"mat_{event.event_id}",
            experiment_id=event.experiment_id,
            event_id=event.event_id,
            event_type=event.event_type,
            display_name=event.display_name,
            stable_name=event.stable_name,
            actor_name=event.actor_name,
            source_container_json=json.dumps(event.source_container or {}, ensure_ascii=False),
            target_container_json=json.dumps(event.target_container or {}, ensure_ascii=False),
            source_container_class=(event.source_container or {}).get("class_name") or (event.source_container or {}).get("object_name"),
            source_container_track_id=(event.source_container or {}).get("track_id") if event.source_container else None,
            target_container_class=(event.target_container or {}).get("class_name") or (event.target_container or {}).get("object_name"),
            target_container_track_id=(event.target_container or {}).get("track_id") if event.target_container else None,
            actor_track_id=event.actor_track_id,
            tool_track_id=event.tool_track_id,
            transfer_mode=event.transfer_mode,
            direction_confidence=event.direction_confidence,
            direction_status=event.direction_status,
            evidence_grade=event.evidence_grade,
            review_status=event.review_status,
            time_start=event.start_time_sec,
            time_end=event.end_time_sec,
            duration_sec=event.duration_sec,
            semantic_tags=tags,
            involved_objects_json=json.dumps(event.involved_objects, ensure_ascii=False),
            clip_path=str(asset.get("clip_path") or ""),
            preview_path=str(asset.get("preview_path") or ""),
            keyframe_count=len(asset.get("keyframe_paths") or []),
            quality_score=float(asset.get("quality_score") or 0.0),
            quality_grade=str(asset.get("quality_grade") or "unknown"),
            quality_reasons_json=json.dumps(asset.get("quality_reasons") or [], ensure_ascii=False),
            qwen_summary=qwen_summary,
            linked_step_id=None,
            searchable_text=searchable,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

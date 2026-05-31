"""Global cross-experiment material index.

Aggregates material records from all experiments into a single searchable index,
enabling queries like "find all liquid transfer events across all experiments."
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from labsopguard.embeddings import cosine_similarity, get_text_embedding_provider

logger = logging.getLogger(__name__)

GLOBAL_INDEX_SCHEMA_VERSION = "global_material_index.v1"


class GlobalMaterialIndex:
    """Cross-experiment material retrieval index."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self.conn.row_factory = sqlite3.Row
        self.embedding_provider = get_text_embedding_provider()
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS global_materials (
                material_id TEXT PRIMARY KEY,
                experiment_id TEXT NOT NULL,
                event_id TEXT,
                event_type TEXT,
                display_name TEXT,
                stable_name TEXT,
                actor_name TEXT,
                involved_objects_json TEXT,
                source_container_class TEXT,
                target_container_class TEXT,
                time_start REAL,
                time_end REAL,
                duration_sec REAL,
                confidence REAL,
                evidence_grade TEXT,
                clip_path TEXT,
                preview_path TEXT,
                keyframe_count INTEGER,
                searchable_text TEXT,
                embedding_json TEXT,
                indexed_at TEXT,
                schema_version TEXT DEFAULT 'global_material_index.v1'
            )
        """)
        for stmt in (
            "CREATE INDEX IF NOT EXISTS idx_global_experiment ON global_materials(experiment_id)",
            "CREATE INDEX IF NOT EXISTS idx_global_event_type ON global_materials(event_type)",
            "CREATE INDEX IF NOT EXISTS idx_global_time ON global_materials(time_start, time_end)",
            "CREATE INDEX IF NOT EXISTS idx_global_objects ON global_materials(involved_objects_json)",
            "CREATE INDEX IF NOT EXISTS idx_global_display ON global_materials(display_name)",
            "CREATE INDEX IF NOT EXISTS idx_global_evidence ON global_materials(evidence_grade)",
        ):
            self.conn.execute(stmt)
        try:
            self.conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS global_materials_fts USING fts5(material_id UNINDEXED, searchable_text)"
            )
            self._has_fts = True
        except sqlite3.OperationalError:
            self._has_fts = False
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def sync_experiment(self, experiment_id: str, experiment_dir: str | Path) -> int:
        """Sync materials from one experiment into the global index."""
        experiment_dir = Path(experiment_dir)
        material_index = experiment_dir / "material_index.sqlite"
        if not material_index.exists():
            logger.warning("No material_index.sqlite for %s", experiment_id)
            return 0

        exp_conn = sqlite3.connect(str(material_index))
        exp_conn.row_factory = sqlite3.Row
        try:
            rows = exp_conn.execute("SELECT * FROM event_materials WHERE experiment_id = ?", (experiment_id,)).fetchall()
        except sqlite3.OperationalError:
            exp_conn.close()
            return 0
        exp_conn.close()

        self.conn.execute("DELETE FROM global_materials WHERE experiment_id = ?", (experiment_id,))
        if self._has_fts:
            self.conn.execute(
                "DELETE FROM global_materials_fts WHERE material_id IN (SELECT material_id FROM global_materials WHERE experiment_id = ?)",
                (experiment_id,),
            )

        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        count = 0

        for row in rows:
            row_dict = dict(row)
            searchable = row_dict.get("searchable_text") or ""
            embedding = self.embedding_provider.embed(searchable) if searchable else []

            self.conn.execute("""
                INSERT OR REPLACE INTO global_materials
                (material_id, experiment_id, event_id, event_type, display_name, stable_name,
                 actor_name, involved_objects_json, source_container_class, target_container_class,
                 time_start, time_end, duration_sec, confidence, evidence_grade,
                 clip_path, preview_path, keyframe_count, searchable_text, embedding_json, indexed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row_dict.get("material_id"),
                experiment_id,
                row_dict.get("event_id"),
                row_dict.get("event_type"),
                row_dict.get("display_name"),
                row_dict.get("stable_name"),
                row_dict.get("actor_name"),
                row_dict.get("involved_objects_json"),
                row_dict.get("source_container_class"),
                row_dict.get("target_container_class"),
                row_dict.get("time_start"),
                row_dict.get("time_end"),
                row_dict.get("duration_sec"),
                None,
                row_dict.get("evidence_grade"),
                row_dict.get("clip_path"),
                row_dict.get("preview_path"),
                row_dict.get("keyframe_count"),
                searchable,
                json.dumps(embedding),
                now,
            ))

            if self._has_fts and searchable:
                self.conn.execute(
                    "INSERT INTO global_materials_fts(material_id, searchable_text) VALUES (?, ?)",
                    (row_dict.get("material_id"), searchable),
                )
            count += 1

        self.conn.commit()
        logger.info("Synced %d materials from experiment %s to global index", count, experiment_id)
        return count

    def sync_all_experiments(self, experiments_dir: str | Path) -> Dict[str, int]:
        """Scan all experiment directories and sync into global index."""
        experiments_dir = Path(experiments_dir)
        results = {}
        if not experiments_dir.exists():
            return results
        for exp_dir in experiments_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            experiment_id = exp_dir.name
            count = self.sync_experiment(experiment_id, exp_dir)
            if count > 0:
                results[experiment_id] = count
        return results

    def search(
        self,
        *,
        text: Optional[str] = None,
        event_type: Optional[str] = None,
        experiment_id: Optional[str] = None,
        objects: Optional[List[str]] = None,
        semantic_query: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Search across all experiments."""
        sql = "SELECT * FROM global_materials"
        clauses: List[str] = []
        params: List[Any] = []

        if experiment_id:
            clauses.append("experiment_id = ?")
            params.append(experiment_id)
        if event_type:
            clauses.append("event_type = ?")
            params.append(event_type)
        if text:
            clauses.append("searchable_text LIKE ?")
            params.append(f"%{text}%")
        if objects:
            for obj in objects:
                clauses.append("(involved_objects_json LIKE ? OR searchable_text LIKE ?)")
                params.extend([f"%{obj}%", f"%{obj}%"])

        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY time_start LIMIT ?"
        params.append(max(1, limit))

        rows = [dict(row) for row in self.conn.execute(sql, params).fetchall()]

        if semantic_query:
            target_embedding = self.embedding_provider.embed(semantic_query)
            for row in rows:
                emb = json.loads(row.get("embedding_json") or "[]")
                row["semantic_score"] = cosine_similarity(emb, target_embedding) if emb else 0.0
            rows.sort(key=lambda r: r.get("semantic_score", 0.0), reverse=True)

        for row in rows:
            row.pop("embedding_json", None)
            try:
                row["involved_objects"] = json.loads(row.get("involved_objects_json") or "[]")
            except (json.JSONDecodeError, TypeError):
                row["involved_objects"] = []

        return rows

    def stats(self) -> Dict[str, Any]:
        """Return index statistics."""
        total = self.conn.execute("SELECT COUNT(*) FROM global_materials").fetchone()[0]
        experiments = self.conn.execute("SELECT COUNT(DISTINCT experiment_id) FROM global_materials").fetchone()[0]
        event_types = self.conn.execute(
            "SELECT event_type, COUNT(*) as cnt FROM global_materials GROUP BY event_type ORDER BY cnt DESC"
        ).fetchall()
        return {
            "total_materials": total,
            "total_experiments": experiments,
            "event_type_distribution": {row[0]: row[1] for row in event_types},
            "fts_enabled": self._has_fts,
            "embedding_mode": self.embedding_provider.mode,
        }

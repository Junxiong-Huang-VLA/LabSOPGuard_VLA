from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from labsopguard.embeddings import cosine_similarity, get_text_embedding_provider


def _json_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _text_blob(*parts: Any) -> str:
    values: List[str] = []
    for part in parts:
        if isinstance(part, (list, tuple, set)):
            values.extend(str(item) for item in part)
        elif isinstance(part, dict):
            for value in part.values():
                if isinstance(value, (dict, list, tuple, set)):
                    values.append(_text_blob(value))
                elif value is not None:
                    values.append(str(value))
        elif part is not None:
            values.append(str(part))
    return " ".join(values)


def _safe_fts_query(value: str) -> bool:
    text = (value or "").strip()
    if not text:
        return False
    # SQLite FTS5 query syntax is brittle for CJK text and punctuation. Use FTS
    # only for simple ASCII tokens; other inputs fall back to LIKE.
    return all(ch.isascii() and (ch.isalnum() or ch.isspace() or ch in {"_", "-", '"'}) for ch in text)


@dataclass
class MaterialQuery:
    objects: Optional[List[str]] = None
    actions: Optional[List[str]] = None
    start_time_sec: Optional[float] = None
    end_time_sec: Optional[float] = None
    camera_id: Optional[str] = None
    stream_id: Optional[str] = None
    has_clip: Optional[bool] = None
    clip_exists: Optional[bool] = None
    text: Optional[str] = None
    embedding_text: Optional[str] = None
    limit: int = 50


class MaterialRetrievalIndex:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._has_fts = False
        self.embedding_provider = get_text_embedding_provider()
        self._init_schema()

    def close(self) -> None:
        self.conn.close()

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS material_items (
                item_id TEXT PRIMARY KEY,
                experiment_id TEXT,
                timestamp_sec REAL,
                local_timestamp_sec REAL,
                camera_id TEXT,
                stream_id TEXT,
                video_asset_id TEXT,
                frame_id TEXT,
                frame_path TEXT,
                clip_id TEXT,
                clip_file_path TEXT,
                clip_exists INTEGER DEFAULT 0,
                object_labels_json TEXT,
                actions_json TEXT,
                event_types_json TEXT,
                text_blob TEXT,
                embedding_json TEXT,
                payload_json TEXT
            )
            """
        )
        for statement in (
            "CREATE INDEX IF NOT EXISTS idx_material_time ON material_items(timestamp_sec)",
            "CREATE INDEX IF NOT EXISTS idx_material_camera ON material_items(camera_id)",
            "CREATE INDEX IF NOT EXISTS idx_material_stream ON material_items(stream_id)",
            "CREATE INDEX IF NOT EXISTS idx_material_clip ON material_items(clip_id, clip_exists)",
        ):
            self.conn.execute(statement)
        try:
            self.conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS material_items_fts USING fts5(item_id UNINDEXED, text_blob)"
            )
            self._has_fts = True
        except sqlite3.OperationalError:
            self._has_fts = False
        self.conn.commit()

    def reset(self) -> None:
        self.conn.execute("DELETE FROM material_items")
        if self._has_fts:
            self.conn.execute("DELETE FROM material_items_fts")
        self.conn.commit()

    def index_payloads(
        self,
        material_stream: Sequence[Dict[str, Any]],
        preprocessing: Optional[Dict[str, Any]] = None,
        experiment_id: Optional[str] = None,
    ) -> None:
        preprocessing = preprocessing or {}
        clip_by_id = {
            str(clip.get("clip_id")): clip
            for clip in preprocessing.get("key_clips", []) or []
            if clip.get("clip_id")
        }
        events_by_item: Dict[str, List[str]] = {}
        for event in preprocessing.get("detected_changes", []) or []:
            metadata = event.get("metadata") or {}
            item_id = str(metadata.get("material_item_id") or "")
            if item_id:
                events_by_item.setdefault(item_id, []).append(str(event.get("event_type", "")))

        video_index_by_item = {
            str(entry.get("item_id")): entry
            for entry in preprocessing.get("video_index", []) or preprocessing.get("time_anchored_material_stream", []) or []
            if entry.get("item_id")
        }
        for item in material_stream:
            item_id = str(item.get("item_id"))
            enriched = dict(video_index_by_item.get(item_id, {}))
            enriched.update(item)
            self.index_item(enriched, clip_by_id=clip_by_id, event_types=events_by_item.get(item_id, []), experiment_id=experiment_id)
        self.conn.commit()

    def index_item(
        self,
        item: Dict[str, Any],
        clip_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
        event_types: Optional[List[str]] = None,
        experiment_id: Optional[str] = None,
    ) -> None:
        clip_by_id = clip_by_id or {}
        clip_id = item.get("clip_id")
        clip = clip_by_id.get(str(clip_id), {}) if clip_id else {}
        clip_file_path = clip.get("file_path")
        clip_exists_on_disk = bool(clip_file_path) and Path(str(clip_file_path)).exists()
        objects = _json_list(item.get("object_labels") or item.get("detected_objects"))
        actions = _json_list(item.get("detected_activities"))
        analysis = item.get("analysis") or {}
        qwen_flash = analysis.get("qwen3_6_flash_frame") or analysis.get("qwen_flash_frame") or {}
        qwen_plus_review = analysis.get("qwen3_6_plus_keyframe_review") or analysis.get("qwen_plus_keyframe_review") or {}
        qwen_objects = _json_list(qwen_flash.get("objects"))
        qwen_actions = _json_list(qwen_flash.get("actions"))
        objects = list(dict.fromkeys([*objects, *[str(item) for item in qwen_objects if item]]))
        actions = list(dict.fromkeys([*actions, *[str(item) for item in qwen_actions if item]]))
        event_types = event_types or []
        blob = _text_blob(
            item.get("scene_description"),
            item.get("transcript_segment"),
            item.get("conversation_context"),
            objects,
            actions,
            qwen_objects,
            qwen_actions,
            qwen_flash.get("scene_summary"),
            qwen_flash.get("risk_flags"),
            qwen_flash.get("state_changes"),
            qwen_plus_review.get("review_summary"),
            qwen_plus_review.get("risk_flags"),
            qwen_plus_review.get("recommended_review_points"),
            event_types,
            clip.get("reason"),
        )
        embedding = self.embedding_provider.embed(blob)
        row = {
            "item_id": str(item.get("item_id") or f"item_{item.get('frame_id', '')}_{item.get('timestamp_sec', 0)}"),
            "experiment_id": str(experiment_id or item.get("experiment_id") or ""),
            "timestamp_sec": float(item.get("timestamp_sec") or 0.0),
            "local_timestamp_sec": float(item.get("local_timestamp_sec") or 0.0),
            "camera_id": item.get("camera_id") or (item.get("metadata") or {}).get("camera_id"),
            "stream_id": item.get("stream_id"),
            "video_asset_id": item.get("video_asset_id") or item.get("media_asset_id"),
            "frame_id": str(item.get("frame_id")),
            "frame_path": item.get("frame_bgr_path"),
            "clip_id": clip_id,
            "clip_file_path": clip_file_path,
            "clip_exists": 1 if clip.get("file_exists") and clip_exists_on_disk else 0,
            "object_labels_json": json.dumps(objects, ensure_ascii=False),
            "actions_json": json.dumps(actions, ensure_ascii=False),
            "event_types_json": json.dumps(event_types, ensure_ascii=False),
            "text_blob": blob,
            "embedding_json": json.dumps(embedding),
            "payload_json": json.dumps(item, ensure_ascii=False),
        }
        self.conn.execute(
            """
            INSERT OR REPLACE INTO material_items
            (item_id, experiment_id, timestamp_sec, local_timestamp_sec, camera_id, stream_id, video_asset_id,
             frame_id, frame_path, clip_id, clip_file_path, clip_exists, object_labels_json, actions_json,
             event_types_json, text_blob, embedding_json, payload_json)
            VALUES (:item_id, :experiment_id, :timestamp_sec, :local_timestamp_sec, :camera_id, :stream_id, :video_asset_id,
             :frame_id, :frame_path, :clip_id, :clip_file_path, :clip_exists, :object_labels_json, :actions_json,
             :event_types_json, :text_blob, :embedding_json, :payload_json)
            """,
            row,
        )
        if self._has_fts:
            self.conn.execute("DELETE FROM material_items_fts WHERE item_id = ?", (row["item_id"],))
            self.conn.execute(
                "INSERT INTO material_items_fts(item_id, text_blob) VALUES (?, ?)",
                (row["item_id"], row["text_blob"]),
            )

    def query(self, query: MaterialQuery) -> List[Dict[str, Any]]:
        use_fts = bool(query.text and self._has_fts and _safe_fts_query(query.text))
        if use_fts:
            sql = "SELECT mi.* FROM material_items mi JOIN material_items_fts ON mi.item_id = material_items_fts.item_id"
            col = "mi."
        else:
            sql = "SELECT * FROM material_items"
            col = ""
        params: List[Any] = []
        clauses: List[str] = []
        if query.start_time_sec is not None:
            clauses.append(f"{col}timestamp_sec >= ?")
            params.append(float(query.start_time_sec))
        if query.end_time_sec is not None:
            clauses.append(f"{col}timestamp_sec <= ?")
            params.append(float(query.end_time_sec))
        if query.camera_id:
            clauses.append(f"{col}camera_id = ?")
            params.append(query.camera_id)
        if query.stream_id:
            clauses.append(f"{col}stream_id = ?")
            params.append(query.stream_id)
        if query.has_clip is not None:
            clauses.append(f"{col}clip_id IS NOT NULL" if query.has_clip else f"{col}clip_id IS NULL")
        if query.clip_exists is not None:
            clauses.append(f"{col}clip_exists = ?")
            params.append(1 if query.clip_exists else 0)
        for obj in query.objects or []:
            clauses.append(f"({col}object_labels_json LIKE ? OR {col}text_blob LIKE ?)")
            token = f"%{obj}%"
            params.extend([token, token])
        for action in query.actions or []:
            clauses.append(f"({col}actions_json LIKE ? OR {col}text_blob LIKE ?)")
            token = f"%{action}%"
            params.extend([token, token])
        if query.text:
            if use_fts:
                clauses.append("material_items_fts MATCH ?")
                params.append(query.text)
            else:
                clauses.append(f"{col}text_blob LIKE ?")
                params.append(f"%{query.text}%")
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += f" ORDER BY {col}timestamp_sec LIMIT ?"
        params.append(max(1, int(query.limit)))
        rows = [self._row_to_dict(row) for row in self.conn.execute(sql, params).fetchall()]
        if query.embedding_text:
            target = self.embedding_provider.embed(query.embedding_text)
            for row in rows:
                row["embedding_score"] = cosine_similarity(json.loads(row.pop("embedding_json") or "[]"), target)
            rows.sort(key=lambda row: row.get("embedding_score", 0.0), reverse=True)
        return rows

    def health_check(self) -> Dict[str, Any]:
        rows = self.conn.execute(
            """
            SELECT item_id, clip_id, clip_file_path, clip_exists
            FROM material_items
            """
        ).fetchall()
        total_items = len(rows)
        clip_refs = 0
        materialized_clips = 0
        broken_clip_refs: List[Dict[str, Any]] = []
        for row in rows:
            data = dict(row)
            if not data.get("clip_id"):
                continue
            clip_refs += 1
            clip_path = data.get("clip_file_path")
            exists_on_disk = bool(clip_path) and Path(str(clip_path)).exists()
            if data.get("clip_exists") and exists_on_disk:
                materialized_clips += 1
            elif clip_path:
                broken_clip_refs.append(
                    {
                        "item_id": data.get("item_id"),
                        "clip_id": data.get("clip_id"),
                        "clip_file_path": clip_path,
                    }
                )
        return {
            "total_items": total_items,
            "clip_reference_count": clip_refs,
            "materialized_clip_count": materialized_clips,
            "broken_clip_reference_count": len(broken_clip_refs),
            "broken_clip_references": broken_clip_refs[:50],
            "embedding_mode": self.embedding_provider.mode,
            "fts_enabled": self._has_fts,
        }

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        data = dict(row)
        for key in ("object_labels_json", "actions_json", "event_types_json", "payload_json"):
            try:
                data[key.replace("_json", "")] = json.loads(data.pop(key) or "[]")
            except json.JSONDecodeError:
                data[key.replace("_json", "")] = []
        return data

    @classmethod
    def build_from_files(
        cls,
        db_path: str | Path,
        material_stream_path: str | Path,
        preprocessing_path: str | Path,
        experiment_id: Optional[str] = None,
    ) -> "MaterialRetrievalIndex":
        material_stream = json.loads(Path(material_stream_path).read_text(encoding="utf-8"))
        preprocessing = json.loads(Path(preprocessing_path).read_text(encoding="utf-8"))
        index = cls(db_path)
        index.reset()
        index.index_payloads(material_stream, preprocessing=preprocessing, experiment_id=experiment_id)
        return index

from __future__ import annotations

import hashlib
import json
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .evidence import attach_evidence, explain_query_evidence
from .semantic_alias import expand_query, score_query_metadata_match
from .schemas import VectorMetadata, write_jsonl


class EmbeddingBackend:
    def __init__(self, kind: str = "auto", dim: int = 512):
        self.kind = kind
        self.dim = dim
        self.vectorizer: Any = None
        self.model: Any = None

    def fit_transform(self, texts: list[str]):
        if self.kind in {"auto", "sentence-transformers"} and os.environ.get("KEY_ACTION_USE_SENTENCE_TRANSFORMERS") == "1":
            try:
                from sentence_transformers import SentenceTransformer

                self.kind = "sentence-transformers"
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
                return self._normalize_dense(np.asarray(self.model.encode(texts), dtype=np.float32))
            except Exception:
                if self.kind == "sentence-transformers":
                    raise

        if self.kind in {"auto", "tfidf"}:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.preprocessing import normalize

                self.kind = "tfidf"
                self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), lowercase=False)
                return normalize(self.vectorizer.fit_transform(texts))
            except Exception:
                if self.kind == "tfidf":
                    raise

        self.kind = "hashing"
        return self._hash_texts(texts)

    def transform(self, texts: list[str]):
        if self.kind == "sentence-transformers":
            return self._normalize_dense(np.asarray(self.model.encode(texts), dtype=np.float32))
        if self.kind == "tfidf":
            from sklearn.preprocessing import normalize

            return normalize(self.vectorizer.transform(texts))
        return self._hash_texts(texts)

    def _hash_texts(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            grams = self._char_ngrams(text)
            if not grams:
                grams = [text]
            for gram in grams:
                digest = hashlib.md5(gram.encode("utf-8")).hexdigest()
                index = int(digest[:8], 16) % self.dim
                vectors[row, index] += 1.0
        return self._normalize_dense(vectors)

    @staticmethod
    def _char_ngrams(text: str) -> list[str]:
        compact = "".join(str(text).split())
        grams: list[str] = []
        for n in (2, 3, 4):
            grams.extend(compact[i : i + n] for i in range(max(0, len(compact) - n + 1)))
        return grams

    @staticmethod
    def _normalize_dense(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms


class VectorIndex:
    def __init__(self, embedding_backend: EmbeddingBackend | None = None):
        self.embedding_backend = embedding_backend or EmbeddingBackend()
        self.vectors: Any = None
        self.metadata: list[dict[str, Any]] = []
        self.faiss_index: Any = None
        self.uses_faiss = False

    def build(self, texts: list[str], metadata: list[VectorMetadata | dict[str, Any]]) -> "VectorIndex":
        normalized_metadata = [
            attach_evidence(item if isinstance(item, dict) else item.__dict__)
            for item in metadata
        ]
        if not texts:
            self.vectors = np.zeros((0, self.embedding_backend.dim), dtype=np.float32)
            self.metadata = normalized_metadata
            self.uses_faiss = False
            self.faiss_index = None
            return self
        self.vectors = self.embedding_backend.fit_transform(texts)
        self.metadata = normalized_metadata
        self._maybe_build_faiss()
        return self

    def _maybe_build_faiss(self) -> None:
        self.uses_faiss = False
        self.faiss_index = None
        if not isinstance(self.vectors, np.ndarray):
            return
        try:
            import faiss

            matrix = np.asarray(self.vectors, dtype=np.float32)
            index = faiss.IndexFlatIP(matrix.shape[1])
            index.add(matrix)
            self.faiss_index = index
            self.uses_faiss = True
        except Exception:
            self.faiss_index = None
            self.uses_faiss = False

    def save(self, index_dir: str | Path) -> None:
        target = Path(index_dir)
        target.mkdir(parents=True, exist_ok=True)
        write_jsonl(target / "vector_metadata.jsonl", self.metadata)
        config = {
            "backend": self.embedding_backend.kind,
            "uses_faiss": self.uses_faiss,
            "metadata_count": len(self.metadata),
        }
        (target / "index_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        if self.uses_faiss:
            import faiss

            faiss.write_index(self.faiss_index, str(target / "faiss.index"))
        with (target / "fallback_index.pkl").open("wb") as handle:
            pickle.dump(
                {
                    "embedding_backend": self.embedding_backend,
                    "vectors": self.vectors,
                    "metadata": self.metadata,
                    "uses_faiss": self.uses_faiss,
                },
                handle,
            )

    @classmethod
    def load(cls, index_dir: str | Path) -> "VectorIndex":
        source = Path(index_dir) / "fallback_index.pkl"
        with source.open("rb") as handle:
            payload = pickle.load(handle)
        index = cls(payload["embedding_backend"])
        index.vectors = payload["vectors"]
        index.metadata = payload["metadata"]
        index.uses_faiss = bool(payload.get("uses_faiss"))
        if index.uses_faiss and (Path(index_dir) / "faiss.index").exists():
            try:
                import faiss

                index.faiss_index = faiss.read_index(str(Path(index_dir) / "faiss.index"))
            except Exception:
                index.faiss_index = None
                index.uses_faiss = False
        return index

    def query(self, query_text: str, top_k: int = 5, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if self.vectors is None:
            raise RuntimeError("Vector index is empty; call build() or load() first")
        if not self.metadata:
            return []
        expanded = expand_query(query_text)
        query_expansion = " ".join(
            [
                str(query_text),
                str(expanded.get("canonical_action", "")),
                " ".join(str(item) for item in expanded.get("target_objects", [])),
                " ".join(str(item) for item in expanded.get("target_interaction_types", [])),
                " ".join(str(item) for item in expanded.get("keywords", [])),
            ]
        )
        query_vector = self.embedding_backend.transform([query_expansion])
        filter_items = {
            str(key): value
            for key, value in (filters or {}).items()
            if value is not None and value != ""
        }
        candidate_indices = [
            idx
            for idx, item in enumerate(self.metadata)
            if _metadata_matches_filters(item, filter_items)
        ]
        if not candidate_indices:
            return []
        top_k = max(1, min(int(top_k), len(candidate_indices)))

        if self.uses_faiss and self.faiss_index is not None and isinstance(query_vector, np.ndarray):
            search_k = len(self.metadata)
            scores, indices = self.faiss_index.search(np.asarray(query_vector, dtype=np.float32), search_k)
            allowed = set(candidate_indices)
            pairs = [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0]) if idx >= 0 and int(idx) in allowed]
        elif isinstance(self.vectors, np.ndarray):
            q = np.asarray(query_vector, dtype=np.float32)[0]
            scores = np.asarray(self.vectors @ q, dtype=np.float32)
            order = sorted(candidate_indices, key=lambda idx: float(scores[idx]), reverse=True)
            pairs = [(int(idx), float(scores[idx])) for idx in order]
        else:
            scores = (self.vectors @ query_vector.T).toarray().ravel()
            order = sorted(candidate_indices, key=lambda idx: float(scores[idx]), reverse=True)
            pairs = [(int(idx), float(scores[idx])) for idx in order]

        results: list[dict[str, Any]] = []
        for idx, vector_score in pairs:
            item = attach_evidence(dict(self.metadata[idx]), query_text=query_text)
            match = score_query_metadata_match(query_text, item)
            rerank_delta = float(match.get("rerank_score", 0.0))
            rerank_reasons = list(match.get("rerank_reasons", []))
            final_score = float(vector_score) + rerank_delta
            if item.get("index_level") != "segment" and "insufficient_sample_adding_evidence" in rerank_reasons:
                final_score -= 0.35
            item["vector_score"] = float(vector_score)
            item["rerank_score"] = rerank_delta
            item["score"] = final_score
            item["class_specific_query_boost"] = float(match.get("class_specific_query_boost", 0.0))
            item["rerank_reasons"] = rerank_reasons
            evidence = explain_query_evidence(query_text, item, item["rerank_reasons"])
            item["evidence_level"] = evidence["evidence_level"]
            item["evidence_reasons"] = evidence["evidence_reasons"]
            item["limitations"] = evidence["limitations"]
            results.append(item)
        results.sort(key=lambda item: (float(item["score"]), float(item["vector_score"])), reverse=True)
        return results[:top_k]


def _metadata_matches_filters(item: dict[str, Any], filters: dict[str, Any]) -> bool:
    time_window = _filter_time_window(filters)
    if time_window is not None and not _metadata_overlaps_time_window(item, time_window):
        return False
    for key, expected in filters.items():
        if key in {"start_time", "end_time", "time_start", "time_end"}:
            continue
        if key in {"object", "objects", "primary_object", "detected_object"}:
            requested = _normalize_filter_values(expected)
            values = _metadata_object_values(item)
            if requested and not all(value in values for value in requested):
                return False
            continue
        if key in {"action", "actions", "action_type"}:
            requested = _normalize_filter_values(expected)
            values = _metadata_action_values(item)
            if requested and not all(value in values for value in requested):
                return False
            continue
        if key in {"asset_type", "material_type"}:
            requested = _normalize_filter_values(expected)
            values = _metadata_asset_types(item)
            if requested and not any(value in values for value in requested):
                return False
            continue
        value = item.get(key)
        if isinstance(value, list):
            if str(expected) not in {str(part) for part in value}:
                return False
        elif str(value) != str(expected):
            return False
    return True


def _metadata_object_values(item: dict[str, Any]) -> set[str]:
    values = {
        _norm(item.get("primary_object")),
        _norm(item.get("primary_object_family")),
        *{_norm(value) for value in _as_list(item.get("detected_objects"))},
        *{_norm(value) for value in _as_list(item.get("visual_keywords"))},
    }
    for key in ("interaction_events", "yolo_interactions", "interaction_keyframes"):
        for event in _as_list(item.get(key)):
            if isinstance(event, dict):
                values.add(_norm(event.get("object_label")))
                values.add(_norm(event.get("object_name")))
                values.update(_norm(label) for label in _as_list(event.get("labels")))
    text = str(item.get("index_text") or "").casefold()
    values.update(value for value in _normalize_filter_values(text.replace(",", " ").replace(":", " ")) if value)
    values.discard("")
    return values


def _metadata_action_values(item: dict[str, Any]) -> set[str]:
    values = {
        _norm(item.get("action_type")),
        _norm(item.get("interaction_type")),
        *{_norm(value) for value in _as_list(item.get("visual_keywords"))},
    }
    for key in ("interaction_events", "yolo_interactions"):
        for event in _as_list(item.get(key)):
            if isinstance(event, dict):
                values.add(_norm(event.get("interaction")))
                values.add(_norm(event.get("interaction_type")))
    values.discard("")
    return values


def _metadata_asset_types(item: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    if item.get("third_person_clip") or item.get("first_person_clip"):
        values.add("video_clip")
        values.add("clip")
    if _as_list(item.get("keyframes")) or _as_list(item.get("interaction_keyframes")):
        values.add("keyframe")
        values.add("image")
    if item.get("index_level"):
        values.add(_norm(item.get("index_level")))
    return values


def _filter_time_window(filters: dict[str, Any]) -> tuple[float | None, float | None] | None:
    start = filters.get("start_time", filters.get("time_start"))
    end = filters.get("end_time", filters.get("time_end"))
    if start is None and end is None:
        return None
    parsed_start = _parse_time_value(start) if start is not None else None
    parsed_end = _parse_time_value(end) if end is not None else None
    if parsed_start is not None and parsed_end is not None and parsed_end < parsed_start:
        parsed_start, parsed_end = parsed_end, parsed_start
    return parsed_start, parsed_end


def _metadata_overlaps_time_window(item: dict[str, Any], time_window: tuple[float | None, float | None]) -> bool:
    query_start, query_end = time_window
    item_start = _parse_time_value(item.get("global_start_time"))
    item_end = _parse_time_value(item.get("global_end_time"))
    if item_start is None and item_end is None:
        return False
    if item_start is None:
        item_start = item_end
    if item_end is None:
        item_end = item_start
    if item_start is not None and item_end is not None and item_end < item_start:
        item_start, item_end = item_end, item_start
    if query_start is not None and item_end is not None and item_end < query_start:
        return False
    if query_end is not None and item_start is not None and item_start > query_end:
        return False
    return True


def _parse_time_value(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _normalize_filter_values(value: Any) -> list[str]:
    values: list[str] = []
    for item in _as_list(value):
        if isinstance(item, str):
            parts = [part for token in item.split(",") for part in token.split(";")]
            if len(parts) == 1:
                parts = item.split()
            values.extend(_norm(part) for part in parts if _norm(part))
        else:
            values.append(_norm(item))
    return [value for value in values if value]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return [value]


def _norm(value: Any) -> str:
    return str(value or "").strip().casefold().replace("-", "_").replace(" ", "_")

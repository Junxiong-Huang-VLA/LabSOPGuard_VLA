from __future__ import annotations

import numpy as np

from key_action_indexer.evidence import LIMIT_MISSING_PIPETTE_TUBE, LIMIT_MISSING_TRANSCRIPT
from key_action_indexer.vector_index import EmbeddingBackend, VectorIndex


class FixedScoreBackend(EmbeddingBackend):
    def __init__(self) -> None:
        super().__init__(kind="fixed", dim=2)

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            [
                [1.00, 0.0],
                [0.80, 0.0],
            ],
            dtype=np.float32,
        )

    def transform(self, texts: list[str]) -> np.ndarray:
        return np.asarray([[1.0, 0.0]], dtype=np.float32)


def test_query_returns_evidence_explanation_and_limitations() -> None:
    metadata = [
        {
            "segment_id": "weak_sample_adding",
            "index_text": "sample_adding candidate from generic bench motion",
            "action_type": "sample_adding",
            "primary_object": "bottle",
            "detected_objects": ["hand", "bottle"],
            "related_dialogue": [],
        },
        {
            "segment_id": "trusted_pipetting",
            "index_text": "使用移液枪向试管加样 200 微升",
            "action_type": "pipetting",
            "primary_object": "pipette",
            "detected_objects": ["hand", "pipette", "tube"],
            "related_dialogue": ["使用移液枪向试管加样 200 微升"],
            "keyframes": ["peak.jpg"],
        },
    ]
    index = VectorIndex(FixedScoreBackend()).build([item["index_text"] for item in metadata], metadata)

    results = index.query("加样", top_k=2)
    by_id = {item["segment_id"]: item for item in results}

    assert results[0]["segment_id"] == "trusted_pipetting"
    assert by_id["trusted_pipetting"]["evidence_level"] == "trusted"
    assert by_id["weak_sample_adding"]["evidence_level"] != "trusted"
    assert by_id["weak_sample_adding"]["evidence_level"] == "insufficient"
    assert LIMIT_MISSING_PIPETTE_TUBE in by_id["weak_sample_adding"]["limitations"]
    assert LIMIT_MISSING_TRANSCRIPT in by_id["weak_sample_adding"]["limitations"]
    assert "insufficient_sample_adding_evidence" in by_id["weak_sample_adding"]["rerank_reasons"]


def test_query_preserves_existing_evidence_field() -> None:
    metadata = [
        {
            "segment_id": "manual_evidence",
            "index_text": "天平称量读数",
            "action_type": "weighing",
            "detected_objects": ["balance"],
            "evidence": {
                "evidence_level": "trusted",
                "evidence_reasons": ["manual_reviewed"],
                "limitations": [],
            },
        }
    ]
    index = VectorIndex(EmbeddingBackend(kind="hashing")).build([metadata[0]["index_text"]], metadata)

    result = index.query("称量", top_k=1)[0]

    assert result["evidence_level"] == "trusted"
    assert "manual_reviewed" in result["evidence_reasons"]
    assert {"evidence_level", "evidence_reasons", "limitations"} <= set(result)

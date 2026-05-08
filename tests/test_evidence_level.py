from __future__ import annotations

from key_action_indexer.evidence import (
    LIMIT_MISSING_PIPETTE_TUBE,
    LIMIT_MISSING_TRANSCRIPT,
    attach_evidence,
    evaluate_metadata_evidence,
)


def test_sample_adding_without_pipette_tube_or_asr_is_insufficient() -> None:
    evidence = evaluate_metadata_evidence(
        {
            "action_type": "sample_adding",
            "primary_object": "bottle",
            "detected_objects": ["hand", "bottle"],
            "related_dialogue": [],
            "index_text": "sample_adding candidate from motion only",
        },
        query_text="加样",
    )

    assert evidence["evidence_level"] == "insufficient"
    assert LIMIT_MISSING_PIPETTE_TUBE in evidence["limitations"]
    assert LIMIT_MISSING_TRANSCRIPT in evidence["limitations"]


def test_sample_adding_with_visual_and_transcript_evidence_is_trusted() -> None:
    evidence = evaluate_metadata_evidence(
        {
            "action_type": "pipetting",
            "primary_object": "pipette",
            "detected_objects": ["hand", "pipette", "tube"],
            "related_dialogue": ["使用移液枪向试管加样 200 微升"],
            "keyframes": ["peak.jpg"],
        },
        query_text="加样",
    )

    assert evidence["evidence_level"] == "trusted"
    assert evidence["limitations"] == []
    assert "sample_adding_visual_evidence:pipette_or_tube" in evidence["evidence_reasons"]
    assert "transcript_evidence" in evidence["evidence_reasons"]


def test_attach_evidence_adds_nested_and_flat_fields() -> None:
    metadata = attach_evidence({"action_type": "weighing", "detected_objects": ["balance"]})

    assert metadata["evidence"]["evidence_level"] == metadata["evidence_level"]
    assert "limitations" in metadata["evidence"]
    assert "evidence_reasons" in metadata

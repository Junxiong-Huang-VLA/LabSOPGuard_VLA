from __future__ import annotations

from key_action_indexer.evidence import evaluate_metadata_evidence


def test_transcript_keywords_upgrade_visual_evidence() -> None:
    metadata = {
        "primary_object": "balance",
        "interaction_type": "hand_balance_contact",
        "interaction": {"max_interaction_score": 0.82},
        "detected_objects": ["hand", "balance"],
        "keyframes": {"peak_frame": "peak.jpg"},
        "dialogue_context": [{"utterance_id": "utt_1", "text": "\u73b0\u5728\u5f00\u59cb\u79f0\u91cf\u3002"}],
    }

    evidence = evaluate_metadata_evidence(metadata)

    assert evidence["evidence_level"] == "visual_and_transcript_confirmed"
    assert "\u79f0\u91cf" in evidence["dialogue_keywords"]


def test_only_sample_adding_asr_without_pipette_is_transcript_supported() -> None:
    metadata = {
        "action_type": "sample_adding",
        "dialogue_context": [{"utterance_id": "utt_1", "text": "\u8fd9\u91cc\u52a0 200 \u5fae\u5347\u3002"}],
        "detected_objects": ["hand", "sample_bottle"],
        "interaction": {"max_interaction_score": 0.0},
    }

    evidence = evaluate_metadata_evidence(metadata, query_text="\u52a0\u6837")

    assert evidence["evidence_level"] == "transcript_supported"
    assert "missing pipette or tube visual evidence" in evidence["limitations"]


def test_missing_asr_keeps_sample_adding_limitations() -> None:
    metadata = {
        "primary_object": "sample_bottle",
        "interaction_type": "hand_sample_bottle_contact",
        "detected_objects": ["hand", "sample_bottle"],
        "interaction": {"max_interaction_score": 0.4},
        "keyframes": {"peak_frame": "peak.jpg"},
    }

    evidence = evaluate_metadata_evidence(metadata, query_text="\u52a0\u6837")

    assert evidence["evidence_level"] in {"weak_visual_evidence", "visual_confirmed"}
    assert "missing pipette or tube visual evidence" in evidence["limitations"]
    assert "missing transcript evidence" in evidence["limitations"]

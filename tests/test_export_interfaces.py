from __future__ import annotations

from pathlib import Path

from key_action_indexer.export_interfaces import EXPORT_MANIFEST_FILENAME, EXPORT_SUMMARY_FILENAME, export_artifact_bundle, summarize_artifact_file
from key_action_indexer.schemas import write_jsonl


def test_export_artifact_bundle_writes_manifest_and_summary(tmp_path: Path) -> None:
    session = tmp_path / "session"
    metadata = session / "metadata"
    metadata.mkdir(parents=True)
    write_jsonl(
        metadata / "video_understanding.jsonl",
        [
            {
                "video_event_id": "evt_1",
                "event_type": "hand_object_contact",
                "confidence": 0.82,
                "confidence_reasons": ["visual evidence"],
                "anomaly_flags": [],
                "asset_refs": [],
                "payload": {},
            }
        ],
    )

    output = tmp_path / "export"
    manifest = export_artifact_bundle(session, output, artifact_types=["video_understanding"], include_interfaces=False, include_reusable_index=False)
    summary = summarize_artifact_file(metadata / "video_understanding.jsonl", "video_understanding")

    assert manifest["valid"] is True
    assert manifest["artifact_count"] == 1
    assert (output / EXPORT_MANIFEST_FILENAME).exists()
    assert (output / EXPORT_SUMMARY_FILENAME).exists()
    assert (output / "artifacts" / "video_understanding.jsonl").exists()
    assert summary["record_count"] == 1
    assert summary["event_type_counts"] == {"hand_object_contact": 1}

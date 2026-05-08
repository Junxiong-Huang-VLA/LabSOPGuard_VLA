from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from key_action_indexer.cli import main


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _make_manifest(tmp_path: Path) -> Path:
    session_dir = tmp_path / "session"
    manifest = {
        "session_id": "timeline_cli",
        "session_start_time": "2026-04-29T17:25:00+08:00",
        "videos": {
            "third_person": {
                "path": str(session_dir / "raw" / "third_person.mp4"),
                "start_time": "2026-04-29T17:25:00+08:00",
                "fps": 30,
                "offset_sec": 0,
            }
        },
        "output_dir": str(session_dir),
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_timeline_cli_invokes_unified_timeline_module_without_video(monkeypatch, tmp_path: Path) -> None:
    manifest_path = _make_manifest(tmp_path)
    user_events_path = tmp_path / "user_events.jsonl"
    ai_events_path = tmp_path / "ai_events.jsonl"
    uploads_path = tmp_path / "uploads.jsonl"
    calibration_path = tmp_path / "calibration.json"
    output_dir = tmp_path / "timeline"

    _write_jsonl(
        user_events_path,
        [{"event_id": "user_001", "start_sec": 1.0, "end_sec": 2.0, "text": "operator starts weighing"}],
    )
    _write_jsonl(ai_events_path, [{"event_id": "ai_001", "time_sec": 3.0, "label": "pipette contact", "confidence": 0.91}])
    _write_jsonl(uploads_path, [{"event_id": "upload_001", "time_sec": 4.0, "filename": "first_person.mp4"}])
    calibration_path.write_text(json.dumps({"clock_offset_sec": 0.25}), encoding="utf-8")

    calls = []
    module = types.ModuleType("key_action_indexer.unified_timeline")

    def generate_unified_timeline(
        manifest_path,
        output_dir,
        user_events_path=None,
        ai_events_path=None,
        uploads_path=None,
        calibration_path=None,
        dry_run=False,
    ):
        calls.append(
            {
                "manifest_path": manifest_path,
                "output_dir": output_dir,
                "user_events_path": user_events_path,
                "ai_events_path": ai_events_path,
                "uploads_path": uploads_path,
                "calibration_path": calibration_path,
                "dry_run": dry_run,
            }
        )
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        rows = []
        for source, path_value in (
            ("user_text", user_events_path),
            ("ai", ai_events_path),
            ("upload", uploads_path),
        ):
            for row in _read_jsonl(Path(path_value)):
                start_sec = row.get("start_sec", row.get("time_sec", 0.0))
                rows.append(
                    {
                        "timeline_id": f"{source}:{row['event_id']}",
                        "source": source,
                        "start_sec": start_sec,
                        "end_sec": row.get("end_sec", start_sec),
                        "text": row.get("text") or row.get("label") or row.get("filename"),
                    }
                )
        rows.sort(key=lambda row: row["start_sec"])
        timeline_path = target / "unified_multimodal_timeline.jsonl"
        report_path = target / "time_calibration_report.json"
        _write_jsonl(timeline_path, rows)
        report_path.write_text(
            json.dumps(
                {
                    "manifest_path": str(manifest_path),
                    "calibration_path": str(calibration_path),
                    "dry_run": dry_run,
                    "event_count": len(rows),
                }
            ),
            encoding="utf-8",
        )
        return {
            "timeline_path": str(timeline_path),
            "calibration_report": str(report_path),
            "event_count": len(rows),
        }

    module.generate_unified_timeline = generate_unified_timeline
    monkeypatch.setitem(sys.modules, "key_action_indexer.unified_timeline", module)

    exit_code = main(
        [
            "timeline",
            "--manifest",
            str(manifest_path),
            "--output",
            str(output_dir),
            "--user-events",
            str(user_events_path),
            "--ai-events",
            str(ai_events_path),
            "--uploads",
            str(uploads_path),
            "--calibration",
            str(calibration_path),
            "--dry-run",
        ]
    )

    timeline_path = output_dir / "unified_multimodal_timeline.jsonl"
    report_path = output_dir / "time_calibration_report.json"
    rows = _read_jsonl(timeline_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert calls == [
        {
            "manifest_path": str(manifest_path),
            "output_dir": str(output_dir),
            "user_events_path": str(user_events_path),
            "ai_events_path": str(ai_events_path),
            "uploads_path": str(uploads_path),
            "calibration_path": str(calibration_path),
            "dry_run": True,
        }
    ]
    assert [row["source"] for row in rows] == ["user_text", "ai", "upload"]
    assert [row["text"] for row in rows] == ["operator starts weighing", "pipette contact", "first_person.mp4"]
    assert report["event_count"] == 3
    assert report["dry_run"] is True


def test_existing_cli_command_still_runs(tmp_path: Path) -> None:
    srt_path = tmp_path / "dialogue.srt"
    output_path = tmp_path / "dialogue.jsonl"
    srt_path.write_text(
        "1\n00:00:01,000 --> 00:00:02,500\nStart weighing sample.\n",
        encoding="utf-8",
    )

    exit_code = main(["transcript-convert", "--input", str(srt_path), "--output", str(output_path)])

    rows = _read_jsonl(output_path)
    assert exit_code == 0
    assert rows == [{"utterance_id": "utt_001", "start_sec": 1.0, "end_sec": 2.5, "text": "Start weighing sample."}]

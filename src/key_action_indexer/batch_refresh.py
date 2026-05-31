from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .derived_refresh import refresh_derived_artifacts
from .micro_coverage_backfill import backfill_micro_coverage
from .micro_quality_enrichment import enrich_micro_quality
from .scope_config import build_stage_scope, load_stage_scope
from .video_understanding import build_video_understanding


BATCH_REFRESH_SCHEMA_VERSION = "key_action_batch_refresh.v1"


def batch_refresh_sessions(
    sources: Iterable[str | Path],
    *,
    query_texts: list[str] | None = None,
    output_summary_path: str | Path | None = None,
    stop_on_error: bool = False,
) -> dict[str, Any]:
    session_dirs = [_resolve_session_dir(source) for source in sources]
    rows = []
    for session in session_dirs:
        try:
            rows.append(_refresh_one(session, query_texts=query_texts or []))
        except Exception as exc:
            error_row = {
                "session_dir": str(session),
                "status": "error",
                "error": str(exc),
                "steps": {},
            }
            rows.append(error_row)
            if stop_on_error:
                break
    summary = {
        "schema_version": BATCH_REFRESH_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_count": len(session_dirs),
        "refreshed_count": sum(1 for row in rows if row.get("status") == "refreshed"),
        "error_count": sum(1 for row in rows if row.get("status") == "error"),
        "query_texts": query_texts or [],
        "sessions": rows,
    }
    if output_summary_path:
        target = Path(output_summary_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _refresh_one(session: Path, *, query_texts: list[str]) -> dict[str, Any]:
    if not session.exists():
        raise FileNotFoundError(f"Session directory does not exist: {session}")
    micro_coverage = backfill_micro_coverage(session)
    micro_quality = enrich_micro_quality(session)
    video_understanding = build_video_understanding(session)
    stage_scope = load_stage_scope(session) or build_stage_scope(session)
    derived = refresh_derived_artifacts(session, query_texts=query_texts)
    return {
        "session_dir": str(session),
        "session_id": video_understanding.get("session_id") or _session_id(session),
        "status": "refreshed",
        "steps": {
            "micro_coverage": {
                "added_micro_count": micro_coverage.get("added_micro_count"),
                "output_micro_count": micro_coverage.get("output_micro_count"),
                "skipped_counts": micro_coverage.get("skipped_counts") or {},
            },
            "micro_quality": {
                "micro_segment_count": micro_quality.get("micro_segment_count"),
                "strong_process_micro_count": micro_quality.get("strong_process_micro_count"),
                "retrieval_candidate_micro_count": micro_quality.get("retrieval_candidate_micro_count"),
                "process_evidence_role_counts": micro_quality.get("process_evidence_role_counts") or {},
            },
            "video_understanding": {
                "video_event_count": video_understanding.get("video_event_count"),
                "conclusion_status_counts": video_understanding.get("conclusion_status_counts") or {},
                "candidate_rollup": video_understanding.get("candidate_rollup") or {},
            },
            "stage_scope": {
                "scope_name": stage_scope.get("scope_name"),
                "stage": stage_scope.get("stage"),
                "status": stage_scope.get("status"),
                "out_of_scope_capabilities": stage_scope.get("out_of_scope_capabilities") or [],
            },
            "derived_refresh": {
                "health": (derived.get("steps") or {}).get("health") or {},
                "quality": (derived.get("steps") or {}).get("quality") or {},
                "artifact_validation": (derived.get("steps") or {}).get("artifact_validation") or {},
                "summary": (derived.get("paths") or {}).get("summary"),
            },
        },
    }


def _resolve_session_dir(source: str | Path) -> Path:
    path = Path(source)
    if (path / "metadata").exists():
        return path
    if (path / "key_action_index" / "metadata").exists():
        return path / "key_action_index"
    return path


def _session_id(session: Path) -> str:
    manifest = session / "manifest.json"
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8-sig"))
            if data.get("session_id"):
                return str(data["session_id"])
        except (OSError, json.JSONDecodeError):
            pass
    return session.name


__all__ = ["BATCH_REFRESH_SCHEMA_VERSION", "batch_refresh_sessions"]

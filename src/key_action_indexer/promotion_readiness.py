from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


PROMOTION_READINESS_SCHEMA_VERSION = "key_action_promotion_readiness.v1"
DEFAULT_QUERY_COUNT = 50
LATEST_REVIEWED_RELEASE_FILENAME = "latest_reviewed_release.json"
PROMOTED_REVIEWED_RELEASE_FILENAME = "promoted_release.json"
REVIEWED_RELEASE_MANIFEST_FILENAME = "reviewed_release_manifest.json"
REVIEWED_RELEASES_DIRNAME = "reviewed_releases"
PROMOTION_AUDIT_EXCLUSION_FILENAME = "promotion_audit_exclusion.json"


def build_promotion_readiness_report(
    sources: Iterable[str | Path],
    *,
    query_count: int = DEFAULT_QUERY_COUNT,
    output_json: str | Path | None = None,
    output_md: str | Path | None = None,
) -> dict[str, Any]:
    sessions = _discover_session_dirs(sources)
    rows = []
    excluded_rows = []
    for session in sessions:
        exclusion = _audit_exclusion(session)
        if exclusion:
            excluded_rows.append(exclusion)
            continue
        rows.append(_audit_session(session, query_count=query_count))
    report = {
        "schema_version": PROMOTION_READINESS_SCHEMA_VERSION,
        "generated_at": _now(),
        "query_count_required": query_count,
        "session_count": len(rows),
        "excluded_session_count": len(excluded_rows),
        "summary": _summary(rows),
        "excluded_sessions": excluded_rows,
        "sessions": rows,
    }
    if output_json:
        target = Path(output_json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if output_md:
        target = Path(output_md)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(render_promotion_readiness_markdown(report), encoding="utf-8")
    return report


def render_promotion_readiness_markdown(report: Mapping[str, Any]) -> str:
    summary = _as_dict(report.get("summary"))
    lines = [
        "# Key Action Promotion Readiness",
        "",
        f"- Generated: `{report.get('generated_at')}`",
        f"- Sessions audited: `{report.get('session_count')}`",
        f"- Sessions excluded: `{report.get('excluded_session_count', 0)}`",
        f"- Active promoted: `{summary.get('active_promoted_count', 0)}`",
        f"- Latest already promoted: `{summary.get('promoted_count', 0)}`",
        f"- Candidate validation failed: `{summary.get('candidate_validation_failed_count', 0)}`",
        f"- Needs candidate validation: `{summary.get('needs_candidate_validation_count', 0)}`",
        f"- Ready to promote: `{summary.get('ready_to_promote_count', 0)}`",
        f"- Blocked: `{summary.get('blocked_count', 0)}`",
        f"- Required gold queries: `{report.get('query_count_required')}`",
        "",
        "## Sessions",
        "",
        "| Session | Status | Latest | Promoted | Release state | Gate | Gold | Eval | Candidate eval | Adapter semantic | Queue pending | Blockers |",
        "|---|---|---|---|---|---|---:|---|---|---:|---:|---:|",
    ]
    for row in report.get("sessions") or []:
        if not isinstance(row, Mapping):
            continue
        releases = _as_dict(row.get("releases"))
        gate = _as_dict(row.get("quality_gate"))
        gold = _as_dict(row.get("gold_benchmark"))
        evaluation = _as_dict(row.get("retrieval_eval"))
        adapters = _as_dict(row.get("adapter_validation"))
        queue = _as_dict(row.get("review_queue"))
        candidate = _as_dict(releases.get("latest_candidate_eval"))
        candidate_label = _candidate_eval_label(candidate)
        applicable_gold_count = _as_int(
            gold.get("applicable_query_count"),
            _as_int(gold.get("query_count"), _as_int(report.get("query_count_required"), 0)),
        )
        lines.append(
            "| "
            f"`{row.get('session_id')}` | "
            f"`{row.get('readiness_status')}` | "
            f"`{releases.get('latest_version') or ''}` | "
            f"`{releases.get('promoted_version') or ''}` | "
            f"`{releases.get('state') or ''}` | "
            f"`{gate.get('status') or 'missing'}` | "
            f"{_as_int(gold.get('human_verified_query_count'), 0)}/{applicable_gold_count}"
            f" (+{_as_int(gold.get('excluded_query_count'), 0)} excluded) | "
            f"`{evaluation.get('status') or 'missing'}` | "
            f"`{candidate_label}` | "
            f"{adapters.get('semantic_issue_count') or 0} | "
            f"{queue.get('pending_count') or 0} | "
            f"{len(row.get('blockers') or [])} |"
        )
    lines.extend(["", "## Blockers", ""])
    for row in report.get("sessions") or []:
        if not isinstance(row, Mapping):
            continue
        lines.append(f"### {row.get('session_id')}")
        blockers = [item for item in row.get("blockers") or [] if isinstance(item, Mapping)]
        if not blockers:
            lines.append("- None")
        else:
            for blocker in blockers:
                command = blocker.get("suggested_command")
                suffix = f" Suggested: `{command}`" if command else ""
                lines.append(f"- `{blocker.get('code')}`: {blocker.get('message')}{suffix}")
        next_actions = [str(item) for item in row.get("next_actions") or [] if item]
        if next_actions:
            lines.append("")
            lines.append("Next actions:")
            for action in next_actions:
                lines.append(f"- `{action}`")
        lines.append("")
    excluded = [item for item in report.get("excluded_sessions") or [] if isinstance(item, Mapping)]
    if excluded:
        lines.extend(["", "## Excluded Sessions", ""])
        for item in excluded:
            lines.append(f"- `{item.get('session_id')}`: {item.get('reason') or 'excluded from promotion audit'}")
    return "\n".join(lines).rstrip() + "\n"


def _audit_session(session: Path, *, query_count: int) -> dict[str, Any]:
    metadata = session / "metadata"
    gate = _read_json(metadata / "quality_gate.json")
    adapters = _adapter_summary(session, gate)
    gold = _gold_summary(session, query_count=query_count)
    evaluation = _eval_summary(session, query_count=query_count)
    queue = _review_queue_summary(session, gate)
    releases = _release_summary(session)
    releases["latest_candidate_eval"] = _candidate_eval_summary(session, releases.get("latest_version"))
    releases["latest_candidate_gate"] = _candidate_gate_summary(session, releases.get("latest_version"))
    releases.update(_candidate_validation_summary(gate, evaluation, releases))
    blockers = _blockers(session, gate=gate, adapters=adapters, gold=gold, evaluation=evaluation, releases=releases, query_count=query_count)
    readiness = _readiness_status(blockers=blockers, releases=releases)
    row = {
        "session_dir": str(session),
        "session_id": _session_id(session),
        "readiness_status": readiness,
        "releases": releases,
        "quality_gate": _gate_summary(gate),
        "adapter_validation": adapters,
        "gold_benchmark": gold,
        "retrieval_eval": evaluation,
        "review_queue": queue,
        "blockers": blockers,
        "next_actions": _next_actions(session, readiness=readiness, blockers=blockers, releases=releases, query_count=query_count),
    }
    return row


def _discover_session_dirs(sources: Iterable[str | Path]) -> list[Path]:
    discovered: list[Path] = []
    seen: set[str] = set()
    for source in sources:
        path = Path(source)
        candidates = _session_candidates(path)
        for candidate in candidates:
            key = str(candidate.resolve()) if candidate.exists() else str(candidate)
            if key in seen:
                continue
            seen.add(key)
            discovered.append(candidate)
    return discovered


def _audit_exclusion(session: Path) -> dict[str, Any] | None:
    data = _read_json(session / "metadata" / PROMOTION_AUDIT_EXCLUSION_FILENAME)
    if not data:
        data = _read_json(session / PROMOTION_AUDIT_EXCLUSION_FILENAME)
    if not data:
        return None
    if not bool(data.get("exclude_from_promotion_audit") or data.get("excluded_from_promotion_audit")):
        return None
    return {
        "session_dir": str(session),
        "session_id": _session_id(session),
        "reason": data.get("reason") or "excluded from promotion audit",
        "reviewer": data.get("reviewer"),
        "excluded_at": data.get("excluded_at"),
        "source_path": str(session / "metadata" / PROMOTION_AUDIT_EXCLUSION_FILENAME)
        if (session / "metadata" / PROMOTION_AUDIT_EXCLUSION_FILENAME).exists()
        else str(session / PROMOTION_AUDIT_EXCLUSION_FILENAME),
    }


def _session_candidates(path: Path) -> list[Path]:
    if (path / "metadata").exists():
        return [path]
    if (path / "key_action_index" / "metadata").exists():
        return [path / "key_action_index"]
    if not path.exists() or not path.is_dir():
        return [path]
    matches = sorted(parent for parent in path.rglob("key_action_index") if (parent / "metadata").exists())
    if matches:
        return matches
    metadata_matches = sorted(parent.parent for parent in path.rglob("metadata") if (parent.parent / "metadata").exists())
    return metadata_matches


def _gate_summary(gate: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(gate.get("summary"))
    blocking = [item for item in gate.get("blocking_checks") or [] if isinstance(item, Mapping)]
    return {
        "available": bool(gate),
        "status": gate.get("status"),
        "can_mark_complete": bool(gate.get("can_mark_complete")),
        "generated_at": gate.get("generated_at"),
        "blocking_count": summary.get("blocking_count", len(blocking)),
        "warning_count": summary.get("warning_count"),
        "metric_source": summary.get("metric_source"),
        "blocking_checks": [
            {
                "name": item.get("name"),
                "actual": item.get("actual"),
                "message": item.get("message"),
            }
            for item in blocking
        ],
        "reviewed_release": summary.get("reviewed_release"),
        "promoted_release": summary.get("promoted_release"),
    }


def _adapter_summary(session: Path, gate: Mapping[str, Any]) -> dict[str, Any]:
    data = _read_json(session / "metadata" / "evidence_adapter_validation.json")
    if not data:
        data = _as_dict(gate.get("adapter_validation"))
    summary = _as_dict(data.get("summary"))
    adapters = _as_dict(data.get("adapters"))
    semantic_by_adapter = {
        name: int(_as_dict(adapter).get("semantic_issue_count") or 0)
        for name, adapter in adapters.items()
        if int(_as_dict(adapter).get("semantic_issue_count") or 0) > 0
    }
    return {
        "available": bool(data),
        "status": data.get("status"),
        "generated_at": data.get("generated_at"),
        "present_adapter_count": int(summary.get("present_adapter_count") or 0),
        "missing_adapter_count": int(summary.get("missing_adapter_count") or 0),
        "error_count": int(summary.get("error_count") or 0),
        "warning_count": int(summary.get("warning_count") or 0),
        "semantic_issue_count": int(summary.get("semantic_issue_count") or 0),
        "semantic_issue_by_adapter": semantic_by_adapter,
    }


def _gold_summary(session: Path, *, query_count: int) -> dict[str, Any]:
    path = session / "metadata" / "gold_query_benchmark.json"
    data = _read_json(path)
    decision_files = sorted(
        str(item)
        for pattern in ("gold_query_decisions*.json", "gold_query_decisions*.jsonl", "gold_query_decisions*.ndjson")
        for item in (session / "metadata").glob(pattern)
    )
    queries = [item for item in data.get("queries") or [] if isinstance(item, Mapping)]
    unresolved = data.get("unresolved_query_ids") if isinstance(data.get("unresolved_query_ids"), list) else []
    reviewed_statuses = {"not_applicable", "out_of_scope", "rejected", "needs_more_review", "needs_review"}
    human_verified_count = _as_int(data.get("human_verified_query_count"), sum(1 for item in queries if item.get("human_verified")))
    human_reviewed_count = _as_int(
        data.get("human_reviewed_query_count"),
        sum(1 for item in queries if item.get("human_verified") or str(item.get("manual_review_status") or "") in reviewed_statuses),
    )
    actual_query_count = _as_int(data.get("query_count"), len(queries))
    total_query_count = _as_int(data.get("total_query_count"), actual_query_count)
    applicable_query_count = _as_int(data.get("applicable_query_count"), actual_query_count)
    excluded_query_count = _as_int(data.get("excluded_query_count"), 0)
    return {
        "available": bool(data),
        "path": str(path) if path.exists() else None,
        "query_count": actual_query_count,
        "total_query_count": total_query_count,
        "applicable_query_count": applicable_query_count,
        "excluded_query_count": excluded_query_count,
        "required_query_count": query_count,
        "human_verified_query_count": human_verified_count,
        "human_reviewed_query_count": human_reviewed_count,
        "binding_mode": data.get("binding_mode"),
        "id_authoritative": bool(data.get("id_authoritative")),
        "reviewed_release": data.get("reviewed_release"),
        "reviewed_release_dir": data.get("reviewed_release_dir"),
        "manual_review_status": data.get("manual_review_status"),
        "source_decisions_path": data.get("source_decisions_path"),
        "decision_files": decision_files,
        "unresolved_query_count": len(unresolved),
        "fully_human_verified": bool(
            data
            and human_verified_count >= applicable_query_count
            and human_reviewed_count >= total_query_count
            and total_query_count >= query_count
            and data.get("binding_mode") == "human_verified_review_file"
            and data.get("id_authoritative")
        ),
    }


def _eval_summary(session: Path, *, query_count: int) -> dict[str, Any]:
    path = session / "evaluation" / "default_chinese_query_validation.json"
    data = _read_json(path)
    if not data:
        metadata_candidate = _read_json(session / "metadata" / "default_chinese_query_validation.json")
        if metadata_candidate.get("status"):
            data = metadata_candidate
            path = session / "metadata" / "default_chinese_query_validation.json"
    thresholds = data.get("threshold_failures") if isinstance(data.get("threshold_failures"), list) else []
    actual_query_count = _as_int(data.get("query_count"), 0)
    total_query_count = _as_int(data.get("total_query_count"), actual_query_count)
    applicable_query_count = _as_int(data.get("applicable_query_count"), actual_query_count)
    excluded_query_count = _as_int(data.get("excluded_query_count"), 0)
    return {
        "available": bool(data),
        "path": str(path) if data else None,
        "status": data.get("status"),
        "generated_at": data.get("generated_at"),
        "index_dir": data.get("index_dir"),
        "query_count": actual_query_count,
        "total_query_count": total_query_count,
        "applicable_query_count": applicable_query_count,
        "excluded_query_count": excluded_query_count,
        "required_query_count": query_count,
        "benchmark_binding_mode": data.get("benchmark_binding_mode"),
        "reviewed_release": data.get("reviewed_release"),
        "reviewed_release_dir": data.get("reviewed_release_dir"),
        "human_verified_query_count": _as_int(data.get("human_verified_query_count"), 0),
        "human_reviewed_query_count": _as_int(data.get("human_reviewed_query_count"), 0),
        "top1_hit_rate": data.get("top1_hit_rate"),
        "top3_hit_rate": data.get("topk_hit_rate"),
        "expected_id_hit_rate": data.get("expected_id_hit_rate"),
        "traceability_hit_rate": data.get("traceability_hit_rate"),
        "failed_query_count": int(data.get("failed_query_count") or 0),
        "threshold_failure_count": len(thresholds),
        "passes_required_eval": bool(data and data.get("status") == "pass" and total_query_count >= query_count and actual_query_count >= applicable_query_count),
    }


def _review_queue_summary(session: Path, gate: Mapping[str, Any]) -> dict[str, Any]:
    path = session / "metadata" / "review_queue.json"
    data = _read_json(path)
    summary = _as_dict(data.get("summary"))
    items = [item for item in data.get("items") or [] if isinstance(item, Mapping)]
    type_counts = Counter(str(item.get("item_type") or "unknown") for item in items)
    status_counts = Counter(str(item.get("review_status") or "pending") for item in items)
    semantic_pending = sum(
        1
        for item in items
        if item.get("item_type") == "evidence_semantic" and str(item.get("review_status") or "pending") in {"pending", "needs_review"}
    )
    conflict_count = sum(1 for item in items if _item_mentions_conflict(item))
    return {
        "available": bool(data),
        "path": str(path) if path.exists() else None,
        "generated_at": data.get("generated_at"),
        "artifact_state": "stale" if _is_older(data.get("generated_at"), gate.get("generated_at")) else "current",
        "total": int(summary.get("total") or len(items)),
        "pending_count": int(summary.get("pending") or status_counts.get("pending") or 0),
        "needs_review_count": int(summary.get("needs_review") or status_counts.get("needs_review") or 0),
        "approved_count": int(summary.get("approved") or status_counts.get("approved") or 0),
        "rejected_count": int(summary.get("rejected") or status_counts.get("rejected") or 0),
        "evidence_semantic_pending_count": semantic_pending,
        "conflict_count": conflict_count,
        "item_type_counts": dict(sorted(type_counts.items())),
        "review_status_counts": dict(sorted(status_counts.items())),
    }


def _release_summary(session: Path) -> dict[str, Any]:
    latest = _latest_release(session)
    promoted = _promoted_release(session)
    latest_version = latest.get("version") or latest.get("active_version")
    promoted_version = promoted.get("active_version") or promoted.get("version")
    return {
        "latest_available": bool(latest),
        "latest_version": latest_version,
        "latest_release_dir": latest.get("release_dir"),
        "promoted_available": bool(promoted),
        "promoted_version": promoted_version,
        "promoted_release_dir": promoted.get("release_dir"),
        "latest_is_promoted": bool(latest_version and promoted_version and latest_version == promoted_version),
        "state": _release_state(latest_version, promoted_version),
    }


def _candidate_eval_summary(session: Path, version: Any) -> dict[str, Any]:
    if not version:
        return {}
    safe_version = _safe_path_token(version)
    candidates = [
        session / "evaluation" / f"default_chinese_query_validation.{safe_version}.candidate.json",
        session / "evaluation" / "candidates" / f"candidate_{safe_version}_default_chinese_query_validation.failed.json",
        session / "evaluation" / f"candidate_{safe_version}_default_chinese_query_validation.failed.json",
    ]
    for path in candidates:
        data = _read_json(path)
        if not data:
            continue
        thresholds = data.get("threshold_failures") if isinstance(data.get("threshold_failures"), list) else []
        return {
            "available": True,
            "path": str(path),
            "status": data.get("status"),
            "query_count": int(data.get("query_count") or 0),
            "top1_hit_rate": data.get("top1_hit_rate"),
            "top3_hit_rate": data.get("topk_hit_rate"),
            "expected_id_hit_rate": data.get("expected_id_hit_rate"),
            "expected_time_window_hit_rate": data.get("expected_time_window_hit_rate"),
            "traceability_hit_rate": data.get("traceability_hit_rate"),
            "failed_query_count": int(data.get("failed_query_count") or 0),
            "threshold_failure_count": len(thresholds),
            "threshold_failures": thresholds,
            "failure_profile": _candidate_failure_profile(data),
            "category_failures": _category_failure_summary(data),
        }
    return {}


def _candidate_gate_summary(session: Path, version: Any) -> dict[str, Any]:
    if not version:
        return {}
    path = session / "metadata" / f"quality_gate.{_safe_path_token(version)}.candidate.json"
    data = _read_json(path)
    if not data:
        return {}
    summary = _as_dict(data.get("summary"))
    return {
        "available": True,
        "path": str(path),
        "status": data.get("status"),
        "can_mark_complete": bool(data.get("can_mark_complete")),
        "blocking_count": summary.get("blocking_count"),
        "reviewed_release": summary.get("reviewed_release"),
    }


def _candidate_validation_summary(gate: Mapping[str, Any], evaluation: Mapping[str, Any], releases: Mapping[str, Any]) -> dict[str, Any]:
    latest_version = releases.get("latest_version")
    promoted_version = releases.get("promoted_version")
    gate_release = _as_dict(gate.get("summary")).get("reviewed_release")
    eval_index_dir = str(evaluation.get("index_dir") or "")
    candidate_eval = _as_dict(releases.get("latest_candidate_eval"))
    candidate_gate = _as_dict(releases.get("latest_candidate_gate"))
    eval_looks_latest = bool(
        latest_version
        and (
            f"{REVIEWED_RELEASES_DIRNAME}\\{latest_version}" in eval_index_dir
            or f"{REVIEWED_RELEASES_DIRNAME}/{latest_version}" in eval_index_dir
        )
    )
    if not latest_version:
        return {
            "candidate_validation_current": False,
            "candidate_validation_release": gate_release,
            "candidate_validation_note": "No latest reviewed release is available.",
        }
    if latest_version == promoted_version:
        return {
            "candidate_validation_current": True,
            "candidate_validation_release": latest_version,
            "candidate_validation_note": "Latest reviewed release is already the promoted release.",
        }
    if candidate_eval.get("available") and candidate_eval.get("status") == "fail":
        return {
            "candidate_validation_current": False,
            "candidate_validation_release": latest_version,
            "candidate_validation_status": "failed",
            "candidate_validation_note": f"Latest release {latest_version} has a failed candidate retrieval eval.",
        }
    if candidate_eval.get("available") and candidate_eval.get("status") == "pass" and (
        not candidate_gate.get("available") or candidate_gate.get("status") == "pass"
    ):
        return {
            "candidate_validation_current": True,
            "candidate_validation_release": latest_version,
            "candidate_validation_status": "pass",
            "candidate_validation_note": f"Latest release {latest_version} has passing candidate validation artifacts.",
        }
    current = bool(gate_release == latest_version and (not evaluation.get("available") or eval_looks_latest))
    return {
        "candidate_validation_current": current,
        "candidate_validation_release": gate_release,
        "candidate_validation_status": "pass" if current else "missing",
        "candidate_validation_note": (
            "Quality gate/eval artifacts reflect the latest reviewed release."
            if current
            else f"Latest release {latest_version} is not the release reflected by current gate/eval artifacts."
        ),
    }


def _latest_release(session: Path) -> dict[str, Any]:
    releases_dir = session / REVIEWED_RELEASES_DIRNAME
    pointer = _read_json(releases_dir / LATEST_REVIEWED_RELEASE_FILENAME)
    release_dir = Path(str(pointer.get("release_dir") or "")) if pointer.get("release_dir") else None
    if release_dir is None:
        versions = sorted(path for path in releases_dir.glob("v*") if path.is_dir()) if releases_dir.exists() else []
        release_dir = versions[-1] if versions else None
    if release_dir is None:
        return {}
    manifest = _read_json(release_dir / REVIEWED_RELEASE_MANIFEST_FILENAME)
    if manifest:
        manifest.setdefault("release_dir", str(release_dir))
        manifest.setdefault("active_version", release_dir.name)
        return manifest
    if pointer:
        return pointer
    return {"active_version": release_dir.name, "release_dir": str(release_dir)}


def _promoted_release(session: Path) -> dict[str, Any]:
    for path in (
        session / REVIEWED_RELEASES_DIRNAME / PROMOTED_REVIEWED_RELEASE_FILENAME,
        session / "metadata" / PROMOTED_REVIEWED_RELEASE_FILENAME,
    ):
        data = _read_json(path)
        if data:
            return data
    return {}


def _release_state(latest_version: Any, promoted_version: Any) -> str:
    if latest_version and promoted_version and latest_version == promoted_version:
        return "promoted_current"
    if latest_version and promoted_version:
        return "latest_unpromoted"
    if latest_version:
        return "unpromoted"
    if promoted_version:
        return "promoted_pointer_without_latest"
    return "missing_release"


def _blockers(
    session: Path,
    *,
    gate: Mapping[str, Any],
    adapters: Mapping[str, Any],
    gold: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    releases: Mapping[str, Any],
    query_count: int,
) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    if not (session / "metadata").exists():
        blockers.append(_blocker("session_metadata_missing", "Session metadata directory is missing.", None))
        return blockers
    if not releases.get("latest_available"):
        blockers.append(
            _blocker(
                "reviewed_release_missing",
                "No frozen reviewed release is available.",
                f"python -m key_action_indexer.cli freeze-reviewed-dataset --session-dir {session}",
            )
        )
    candidate_eval = _as_dict(releases.get("latest_candidate_eval"))
    if candidate_eval.get("available") and candidate_eval.get("status") == "fail":
        expected_id = candidate_eval.get("expected_id_hit_rate")
        time_window = candidate_eval.get("expected_time_window_hit_rate")
        profile = candidate_eval.get("failure_profile")
        profile_suffix = f" ({profile})" if profile else ""
        time_suffix = f", expected_time_window_hit_rate={time_window}" if time_window is not None else ""
        blockers.append(
            _blocker(
                "candidate_retrieval_eval_failed",
                f"Latest release candidate retrieval eval failed with expected_id_hit_rate={expected_id}{time_suffix}{profile_suffix}.",
                f"python -m key_action_indexer.cli promote-reviewed-release --session-dir {session} --version {releases.get('latest_version')} --reviewer <reviewer> --query-count {query_count}",
                details={
                    "candidate_eval_path": candidate_eval.get("path"),
                    "top3_hit_rate": candidate_eval.get("top3_hit_rate"),
                    "expected_id_hit_rate": expected_id,
                    "expected_time_window_hit_rate": time_window,
                    "traceability_hit_rate": candidate_eval.get("traceability_hit_rate"),
                    "failure_profile": profile,
                    "category_failures": candidate_eval.get("category_failures") or [],
                    "threshold_failures": candidate_eval.get("threshold_failures") or [],
                },
            )
        )
    if not gate:
        blockers.append(
            _blocker(
                "quality_gate_missing",
                "Quality gate artifact is missing.",
                f"python -m key_action_indexer.cli quality-gate --session-dir {session} --strict",
            )
        )
    elif gate.get("status") != "pass" or not gate.get("can_mark_complete"):
        for item in gate.get("blocking_checks") or []:
            if isinstance(item, Mapping):
                name = str(item.get("name") or "quality_gate")
                blockers.append(
                    _blocker(
                        f"quality_gate_blocking:{name}",
                        str(item.get("message") or f"Quality gate check failed: {name}"),
                        f"python -m key_action_indexer.cli quality-gate --session-dir {session} --strict",
                        details={"actual": item.get("actual"), "maximum": item.get("maximum"), "minimum": item.get("minimum")},
                    )
                )
    if adapters.get("error_count"):
        blockers.append(
            _blocker(
                "adapter_validation_errors",
                f"Adapter validation has {adapters.get('error_count')} malformed or misaligned rows.",
                f"python -m key_action_indexer.cli validate-evidence-adapters --session-dir {session} --strict",
            )
        )
    if adapters.get("semantic_issue_count"):
        blockers.append(
            _blocker(
                "adapter_semantic_issues",
                f"Adapter validation has {adapters.get('semantic_issue_count')} semantic issue(s).",
                f"python -m key_action_indexer.cli validate-evidence-adapters --session-dir {session} --strict",
                details={"by_adapter": adapters.get("semantic_issue_by_adapter") or {}},
            )
        )
    if not gold.get("available"):
        blockers.append(
            _blocker(
                "gold_benchmark_missing",
                "Gold query benchmark is missing.",
                f"python -m key_action_indexer.cli gold-query-benchmark --session-dir {session} --query-count {query_count}",
            )
        )
    elif not gold.get("fully_human_verified"):
        blockers.append(
            _blocker(
                "gold_benchmark_not_human_verified",
                (
                    "Gold query benchmark is not fully human verified "
                    f"({_as_int(gold.get('human_verified_query_count'), 0)}/{_as_int(gold.get('applicable_query_count'), query_count)} applicable, "
                    f"{_as_int(gold.get('human_reviewed_query_count'), 0)}/{_as_int(gold.get('total_query_count'), query_count)} total reviewed)."
                ),
                f"python -m key_action_indexer.cli confirm-gold-query-benchmark --session-dir {session} --decisions <decision_file> --query-count {query_count} --reviewer <reviewer>",
                details={
                    "binding_mode": gold.get("binding_mode"),
                    "id_authoritative": gold.get("id_authoritative"),
                    "applicable_query_count": gold.get("applicable_query_count"),
                    "excluded_query_count": gold.get("excluded_query_count"),
                    "decision_files": gold.get("decision_files") or [],
                },
            )
        )
        if not gold.get("decision_files"):
            blockers.append(
                _blocker(
                    "gold_decision_file_missing",
                    "No gold_query_decisions*.json/jsonl file is present in metadata.",
                    None,
                )
            )
    enforce_latest_release_binding = _requires_latest_release_binding(releases)
    if enforce_latest_release_binding and gold.get("available") and releases.get("latest_version"):
        gold_release = str(gold.get("reviewed_release") or "").strip()
        latest_version = str(releases.get("latest_version") or "").strip()
        if not gold_release:
            blockers.append(
                _blocker(
                    "gold_benchmark_release_missing",
                    "Gold query benchmark is not bound to a reviewed release.",
                    f"python -m key_action_indexer.cli confirm-gold-query-benchmark --session-dir {session} --decisions <decision_file> --query-count {query_count} --reviewer <reviewer>",
                )
            )
        elif gold_release != latest_version:
            blockers.append(
                _blocker(
                    "gold_benchmark_release_mismatch",
                    f"Gold query benchmark is bound to {gold_release}, but latest reviewed release is {latest_version}.",
                    f"python -m key_action_indexer.cli confirm-gold-query-benchmark --session-dir {session} --decisions <decision_file> --query-count {query_count} --reviewer <reviewer>",
                    details={"reviewed_release": gold_release, "latest_version": latest_version},
                )
            )
    if not evaluation.get("available"):
        blockers.append(
            _blocker(
                "retrieval_eval_missing",
                "Default Chinese query retrieval evaluation is missing.",
                f"python -m key_action_indexer.cli default-query-eval --session-dir {session} --query-count {query_count}",
            )
        )
    elif not evaluation.get("passes_required_eval"):
        blockers.append(
            _blocker(
                "retrieval_eval_not_pass",
                f"Default Chinese query retrieval evaluation status is {evaluation.get('status')!r}.",
                f"python -m key_action_indexer.cli default-query-eval --session-dir {session} --query-count {query_count}",
                details={
                    "failed_query_count": evaluation.get("failed_query_count"),
                    "threshold_failure_count": evaluation.get("threshold_failure_count"),
                    "query_count": evaluation.get("query_count"),
                },
            )
        )
    elif enforce_latest_release_binding and releases.get("latest_version"):
        eval_release = str(evaluation.get("reviewed_release") or "").strip()
        latest_version = str(releases.get("latest_version") or "").strip()
        if not eval_release:
            blockers.append(
                _blocker(
                    "retrieval_eval_release_missing",
                    "Default Chinese query retrieval evaluation is not bound to a reviewed release.",
                    f"python -m key_action_indexer.cli default-query-eval --session-dir {session} --query-count {query_count}",
                )
            )
        elif eval_release != latest_version:
            blockers.append(
                _blocker(
                    "retrieval_eval_release_mismatch",
                    f"Default Chinese query retrieval evaluation is bound to {eval_release}, but latest reviewed release is {latest_version}.",
                    f"python -m key_action_indexer.cli default-query-eval --session-dir {session} --query-count {query_count}",
                    details={"reviewed_release": eval_release, "latest_version": latest_version},
                )
            )
    return _dedupe_blockers(blockers)


def _blocker(code: str, message: str, suggested_command: str | None, *, details: Mapping[str, Any] | None = None) -> dict[str, Any]:
    output = {"code": code, "message": message}
    if suggested_command:
        output["suggested_command"] = suggested_command
    if details:
        output["details"] = dict(details)
    return output


def _readiness_status(*, blockers: list[Mapping[str, Any]], releases: Mapping[str, Any]) -> str:
    if any(str(blocker.get("code") or "") == "candidate_retrieval_eval_failed" for blocker in blockers):
        return "candidate_validation_failed"
    if blockers:
        return "blocked"
    if releases.get("latest_is_promoted"):
        return "promoted"
    if releases.get("latest_available") and not releases.get("candidate_validation_current"):
        return "needs_candidate_validation"
    if releases.get("latest_available"):
        return "ready_to_promote"
    return "blocked"


def _requires_latest_release_binding(releases: Mapping[str, Any]) -> bool:
    if releases.get("latest_is_promoted"):
        return True
    return bool(releases.get("candidate_validation_current"))


def _next_actions(
    session: Path,
    *,
    readiness: str,
    blockers: list[Mapping[str, Any]],
    releases: Mapping[str, Any],
    query_count: int,
) -> list[str]:
    commands = [str(item.get("suggested_command")) for item in blockers if item.get("suggested_command")]
    if readiness == "ready_to_promote":
        version = releases.get("latest_version") or ""
        commands.append(
            f"python -m key_action_indexer.cli promote-reviewed-release --session-dir {session} --version {version} --reviewer <reviewer> --query-count {query_count}"
        )
    if readiness == "needs_candidate_validation":
        version = releases.get("latest_version") or ""
        commands.append(
            f"python -m key_action_indexer.cli promote-reviewed-release --session-dir {session} --version {version} --reviewer <reviewer> --query-count {query_count}"
        )
    return _dedupe(commands)


def _summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    statuses = Counter(str(row.get("readiness_status") or "unknown") for row in rows)
    blocker_codes = Counter(
        str(blocker.get("code") or "unknown")
        for row in rows
        for blocker in row.get("blockers") or []
        if isinstance(blocker, Mapping)
    )
    return {
        "promoted_count": statuses.get("promoted", 0),
        "active_promoted_count": sum(1 for row in rows if _as_dict(row.get("releases")).get("promoted_available")),
        "ready_to_promote_count": statuses.get("ready_to_promote", 0),
        "candidate_validation_failed_count": statuses.get("candidate_validation_failed", 0),
        "needs_candidate_validation_count": statuses.get("needs_candidate_validation", 0),
        "blocked_count": statuses.get("blocked", 0),
        "status_counts": dict(sorted(statuses.items())),
        "blocker_code_counts": dict(sorted(blocker_codes.items())),
        "adapter_semantic_issue_count": sum(int(_as_dict(row.get("adapter_validation")).get("semantic_issue_count") or 0) for row in rows),
        "retrieval_eval_pass_count": sum(1 for row in rows if _as_dict(row.get("retrieval_eval")).get("status") == "pass"),
        "human_verified_gold_ready_count": sum(1 for row in rows if _as_dict(row.get("gold_benchmark")).get("fully_human_verified")),
    }


def _session_id(session: Path) -> str:
    for path in (session / "manifest.json", session.parent / "manifest.json"):
        data = _read_json(path)
        if data.get("session_id"):
            return str(data["session_id"])
    for path in (
        session / "metadata" / "quality_gate.json",
        session / "metadata" / "evidence_adapter_validation.json",
        session / "metadata" / "review_queue.json",
    ):
        data = _read_json(path)
        if data.get("manifest_session_id"):
            return str(data["manifest_session_id"])
        if data.get("session_id"):
            return str(data["session_id"])
    return session.parent.name if session.name == "key_action_index" else session.name


def _item_mentions_conflict(item: Mapping[str, Any]) -> bool:
    text = json.dumps(
        {
            "title": item.get("title"),
            "summary": item.get("summary"),
            "reasons": item.get("reasons"),
            "severity": item.get("severity"),
        },
        ensure_ascii=False,
        default=str,
    ).casefold()
    return "conflict" in text or "冲突" in text


def _is_older(left: Any, right: Any) -> bool:
    left_dt = _parse_time(left)
    right_dt = _parse_time(right)
    return bool(left_dt and right_dt and left_dt < right_dt)


def _parse_time(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _candidate_eval_label(candidate: Mapping[str, Any]) -> str:
    if not candidate:
        return "missing"
    status = str(candidate.get("status") or "unknown")
    expected_id = _format_rate(candidate.get("expected_id_hit_rate"))
    time_window = _format_rate(candidate.get("expected_time_window_hit_rate"))
    if expected_id is None and time_window is None:
        return status
    if time_window is None:
        return f"{status} id={expected_id}"
    if expected_id is None:
        return f"{status} time={time_window}"
    return f"{status} id={expected_id} time={time_window}"


def _candidate_failure_profile(data: Mapping[str, Any]) -> str | None:
    if data.get("status") != "fail":
        return None
    expected_id = _as_float(data.get("expected_id_hit_rate"))
    time_window = _as_float(data.get("expected_time_window_hit_rate"))
    traceability = _as_float(data.get("traceability_hit_rate"))
    if expected_id is not None and expected_id < 0.75 and time_window is not None and time_window >= 0.95:
        return "time_window_pass_id_fail"
    if traceability is not None and traceability < 0.75:
        return "traceability_fail"
    return "retrieval_threshold_fail"


def _category_failure_summary(data: Mapping[str, Any], *, limit: int = 5) -> list[dict[str, Any]]:
    categories = _as_dict(data.get("category_summary"))
    rows: list[dict[str, Any]] = []
    for category, raw in categories.items():
        item = _as_dict(raw)
        failed = int(item.get("failed_query_count") or 0)
        if failed <= 0:
            continue
        rows.append(
            {
                "category": str(category),
                "query_count": int(item.get("query_count") or 0),
                "failed_query_count": failed,
                "top3_hit_rate": item.get("top3_hit_rate"),
                "expected_id_hit_rate": item.get("expected_id_hit_rate"),
            }
        )
    rows.sort(key=lambda item: (-int(item["failed_query_count"]), str(item["category"])))
    return rows[:limit]


def _format_rate(value: Any) -> str | None:
    number = _as_float(value)
    if number is None:
        return None
    return f"{number:.2f}"


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any, default: int) -> int:
    try:
        if value in (None, ""):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_path_token(value: Any) -> str:
    text = str(value or "candidate")
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in text)


def _dedupe(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _dedupe_blockers(blockers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    seen = set()
    for blocker in blockers:
        key = str(blocker.get("code") or "")
        if key in seen:
            continue
        seen.add(key)
        output.append(blocker)
    return output


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "PROMOTION_READINESS_SCHEMA_VERSION",
    "build_promotion_readiness_report",
    "render_promotion_readiness_markdown",
]

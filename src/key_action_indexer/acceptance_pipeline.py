"""One-command acceptance pipeline for key action evidence sessions.

The pipeline is intentionally conservative: it refreshes reports and review
templates from existing session artifacts, but it never upgrades model evidence
to confirmed human evidence unless an explicit decisions file is supplied.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_QUERIES = (
    "\u624b\u78b0\u74f6\u5b50",
    "\u79f0\u91cf",
    "\u4f7f\u7528\u522e\u52fa",
    "\u52a0\u6837",
    "\u6253\u5f00\u5bb9\u5668",
    "\u5173\u95ed\u5bb9\u5668",
    "\u8bbe\u5907\u9762\u677f\u64cd\u4f5c",
)

KNOWN_JSONL_NAMES = {
    "model_observation_events.jsonl",
    "video_understanding.jsonl",
    "state_change_index.jsonl",
    "material_asset_catalog.jsonl",
    "micro_segments.jsonl",
    "segment_index.jsonl",
    "segments.jsonl",
    "retrieval_index.jsonl",
}

DEFAULT_MIN_RETRIEVAL_TOP3_HITS = 1
SOURCE_EVIDENCE_NAMES = {
    "model_observation_events.jsonl",
    "video_understanding.jsonl",
    "state_change_index.jsonl",
    "material_asset_catalog.jsonl",
    "micro_segments.jsonl",
    "segment_index.jsonl",
    "segments.jsonl",
    "retrieval_index.jsonl",
}


@dataclass
class PipelineOptions:
    session_dir: Path
    output_dir: Path | None = None
    decisions_file: Path | None = None
    apply_decisions: bool = False
    strict: bool = False
    dry_run: bool = False
    queries: tuple[str, ...] = DEFAULT_QUERIES


@dataclass
class PipelineResult:
    session_dir: Path
    output_dir: Path
    status: str
    generated_files: list[Path] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    step_results: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status in {"pass", "needs_review"}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return path


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
            if isinstance(payload, dict):
                yield payload
            else:
                yield {"value": payload}


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")
    return path


def relative(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def safe_id(record: dict[str, Any], fallback: str) -> str:
    for key in ("event_id", "segment_id", "micro_segment_id", "id", "asset_id", "record_id"):
        value = record.get(key)
        if value:
            return str(value)
    return fallback


def record_text(record: dict[str, Any]) -> str:
    parts: list[str] = []
    for key, value in record.items():
        if key.startswith("_"):
            continue
        if isinstance(value, (str, int, float, bool)):
            parts.append(str(value))
        elif isinstance(value, list):
            parts.extend(str(item) for item in value if isinstance(item, (str, int, float, bool)))
        elif isinstance(value, dict):
            parts.append(json.dumps(value, ensure_ascii=False, sort_keys=True))
    return " ".join(parts)


def extract_seconds(record: dict[str, Any]) -> tuple[float | None, float | None]:
    starts = ("start_s", "start_sec", "start_time_s", "t_start", "begin_s")
    ends = ("end_s", "end_sec", "end_time_s", "t_end", "finish_s")
    start = next((record.get(key) for key in starts if isinstance(record.get(key), (int, float))), None)
    end = next((record.get(key) for key in ends if isinstance(record.get(key), (int, float))), None)
    return start, end


def discover_artifacts(session_dir: Path) -> dict[str, Any]:
    json_files = sorted(session_dir.rglob("*.json"))
    jsonl_files = sorted(session_dir.rglob("*.jsonl"))
    media_files = sorted(
        p
        for pattern in ("*.mp4", "*.mov", "*.mkv", "*.avi", "*.webm")
        for p in session_dir.rglob(pattern)
    )
    image_files = sorted(
        p for pattern in ("*.jpg", "*.jpeg", "*.png", "*.webp") for p in session_dir.rglob(pattern)
    )
    transcript_files = sorted(
        p
        for pattern in ("*.srt", "*.vtt", "*.txt", "*.json", "*.jsonl")
        for p in (session_dir / "transcript").rglob(pattern)
        if (session_dir / "transcript").exists()
    )

    important_jsonl = {
        path.name: path
        for path in jsonl_files
        if path.name in KNOWN_JSONL_NAMES or path.parent.name in {"metadata", "index", "evaluation"}
    }
    important_json = {
        path.name: path
        for path in json_files
        if path.name
        in {
            "process_quality_report.json",
            "capability_gap_report.json",
            "human_confirmation_review_summary.json",
            "process_record.json",
            "experiment_process.json",
            "evaluation_manifest.json",
        }
    }
    return {
        "json_files": json_files,
        "jsonl_files": jsonl_files,
        "media_files": media_files,
        "image_files": image_files,
        "transcript_files": transcript_files,
        "important_jsonl": important_jsonl,
        "important_json": important_json,
    }


def validate_artifacts(session_dir: Path, artifacts: dict[str, Any]) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    json_record_count = 0
    jsonl_record_count = 0

    for path in artifacts["json_files"]:
        try:
            read_json(path)
            json_record_count += 1
        except Exception as exc:  # noqa: BLE001 - surfaced in report
            issues.append({"path": relative(path, session_dir), "severity": "error", "message": str(exc)})

    for path in artifacts["jsonl_files"]:
        try:
            count = sum(1 for _ in iter_jsonl(path))
            jsonl_record_count += count
        except Exception as exc:  # noqa: BLE001 - surfaced in report
            issues.append({"path": relative(path, session_dir), "severity": "error", "message": str(exc)})

    return {
        "valid": not any(issue["severity"] == "error" for issue in issues),
        "artifact_count": len(artifacts["json_files"]) + len(artifacts["jsonl_files"]),
        "source_evidence_artifact_count": len(
            [path for name, path in artifacts["important_jsonl"].items() if name in SOURCE_EVIDENCE_NAMES]
        ),
        "json_record_count": json_record_count,
        "jsonl_record_count": jsonl_record_count,
        "media_file_count": len(artifacts["media_files"]),
        "image_file_count": len(artifacts["image_files"]),
        "transcript_file_count": len(artifacts["transcript_files"]),
        "issues": issues,
    }


def load_records(paths: Iterable[Path], limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for row in iter_jsonl(path):
            rows.append(row | {"_source_path": str(path)})
            if limit is not None and len(rows) >= limit:
                return rows
    return rows


def build_review_candidates(artifacts: dict[str, Any], session_dir: Path) -> list[dict[str, Any]]:
    source_names = (
        "model_observation_events.jsonl",
        "video_understanding.jsonl",
        "state_change_index.jsonl",
        "micro_segments.jsonl",
        "segment_index.jsonl",
        "segments.jsonl",
    )
    paths = [artifacts["important_jsonl"][name] for name in source_names if name in artifacts["important_jsonl"]]
    candidates: list[dict[str, Any]] = []
    for idx, record in enumerate(load_records(paths), start=1):
        text = record_text(record)
        status = str(record.get("human_confirmation_status") or record.get("confirmation_status") or "").lower()
        decision = str(record.get("decision") or record.get("review_status") or "").lower()
        needs_review = (
            not status
            or status in {"needs_review", "pending", "unconfirmed", "candidate"}
            or decision in {"needs_review", "pending", "candidate"}
        )
        if not needs_review:
            continue
        start_s, end_s = extract_seconds(record)
        source = Path(str(record.get("_source_path", "")))
        candidates.append(
            {
                "candidate_id": f"review-{idx:05d}",
                "evidence_id": safe_id(record, f"evidence-{idx:05d}"),
                "source": relative(source, session_dir) if str(source) else None,
                "start_s": start_s,
                "end_s": end_s,
                "summary": text[:500],
                "model_status": status or decision or "candidate",
                "required_human_decision": True,
                "suggested_decision": "needs_review",
            }
        )
    return candidates


def write_review_bundle(session_dir: Path, candidates: list[dict[str, Any]]) -> list[Path]:
    bundle = {
        "schema_version": "2026-05-07",
        "generated_at": utc_now(),
        "session_dir": str(session_dir),
        "candidate_count": len(candidates),
        "policy": {
            "auto_confirm_model_candidates": False,
            "labels_and_training_required": False,
            "human_decisions_file": "metadata/confirmation_decisions.template.jsonl",
        },
        "candidates": candidates,
    }
    template_rows = [
        {
            "candidate_id": row["candidate_id"],
            "evidence_id": row["evidence_id"],
            "decision": "needs_review",
            "reviewer": "",
            "notes": "",
            "reviewed_at": "",
        }
        for row in candidates
    ]
    if not template_rows:
        template_rows = [
            {
                "candidate_id": "example",
                "evidence_id": "",
                "decision": "needs_review",
                "reviewer": "",
                "notes": "",
                "reviewed_at": "",
            }
        ]
    return [
        write_json(session_dir / "exports" / "review_bundle.json", bundle),
        write_jsonl(session_dir / "metadata" / "confirmation_decisions.template.jsonl", template_rows),
    ]


def apply_confirmation_decisions(session_dir: Path, decisions_file: Path) -> Path:
    allowed = {"approved", "rejected", "needs_review"}
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(iter_jsonl(decisions_file), start=1):
        decision = str(row.get("decision") or "needs_review").strip().lower()
        if decision not in allowed:
            raise ValueError(f"{decisions_file}:{idx}: decision must be one of {sorted(allowed)}")
        rows.append(
            row
            | {
                "decision": decision,
                "applied_at": utc_now(),
                "source_decisions_file": str(decisions_file),
            }
        )
    applied_path = session_dir / "metadata" / "confirmation_decisions.applied.jsonl"
    history_path = session_dir / "metadata" / "step_review_history.jsonl"
    write_jsonl(applied_path, rows)
    write_jsonl(history_path, rows)
    return applied_path


def build_quality_report(
    session_dir: Path,
    artifacts: dict[str, Any],
    validation: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> Path:
    event_sources = [
        name
        for name in (
            "model_observation_events.jsonl",
            "video_understanding.jsonl",
            "state_change_index.jsonl",
            "micro_segments.jsonl",
            "segment_index.jsonl",
            "segments.jsonl",
        )
        if name in artifacts["important_jsonl"]
    ]
    transcript_count = validation["transcript_file_count"]
    checks = [
        {
            "id": "artifact_integrity",
            "status": "pass" if validation["valid"] else "fail",
            "score": 1.0 if validation["valid"] else 0.0,
            "summary": f"{validation['artifact_count']} structured artifacts scanned",
        },
        {
            "id": "physical_evidence_sources",
            "status": "pass" if event_sources else "needs_review",
            "score": 1.0 if event_sources else 0.0,
            "summary": ", ".join(event_sources) if event_sources else "no key action evidence JSONL found",
        },
        {
            "id": "human_confirmation",
            "status": "pass" if not candidates else "needs_review",
            "score": 1.0 if not candidates else 0.55,
            "summary": f"{len(candidates)} candidates still require human decision",
        },
        {
            "id": "asr_or_transcript_coverage",
            "status": "pass" if transcript_count else "needs_review",
            "score": 1.0 if transcript_count else 0.0,
            "summary": f"{transcript_count} transcript artifact(s) detected",
        },
        {
            "id": "labels_and_training",
            "status": "deferred",
            "score": None,
            "summary": "annotation and model training are explicitly non-blocking for this pipeline",
        },
    ]
    status = "fail" if any(check["status"] == "fail" for check in checks) else "needs_review"
    if all(check["status"] in {"pass", "deferred"} for check in checks):
        status = "pass"
    report = {
        "schema_version": "2026-05-07",
        "generated_at": utc_now(),
        "session_dir": str(session_dir),
        "overall_status": status,
        "summary": {
            "structured_artifact_count": validation["artifact_count"],
            "source_evidence_artifact_count": validation["source_evidence_artifact_count"],
            "jsonl_record_count": validation["jsonl_record_count"],
            "human_review_candidate_count": len(candidates),
            "labels_and_training_blocking": False,
        },
        "checks": checks,
        "issues": validation["issues"],
    }
    return write_json(session_dir / "metadata" / "process_quality_report.json", report)


def build_capability_gap_report(session_dir: Path, artifacts: dict[str, Any]) -> Path:
    existing = artifacts["important_json"].get("capability_gap_report.json")
    payload = read_json(existing, {}) if existing else {}
    required_missing = payload.get("summary", {}).get("capabilities_missing_label_foundation", [])
    unavailable = payload.get("summary", {}).get("unavailable_inventory_capabilities", [])
    report = {
        "schema_version": "2026-05-07",
        "generated_at": utc_now(),
        "session_dir": str(session_dir),
        "status": "deferred" if (required_missing or unavailable) else "pass",
        "labels_and_training_blocking": False,
        "summary": {
            "capabilities_missing_label_foundation": required_missing,
            "unavailable_inventory_capabilities": unavailable,
            "source_report": relative(existing, session_dir) if existing else None,
        },
        "annotation_plan": payload.get("annotation_plan", []),
        "next_entrypoints": [
            "capability-gap-report",
            "export-yolo-relabel-pack",
            "micro-gt-template",
            "validate-micro-gt",
        ],
    }
    return write_json(session_dir / "metadata" / "capability_gap_report.normalized.json", report)


def build_evaluation_manifest(session_dir: Path, artifacts: dict[str, Any]) -> list[Path]:
    gt_candidates = sorted((session_dir / "evaluation").glob("*micro*gt*.jsonl"))
    labeled_count = 0
    template_count = 0
    for path in gt_candidates:
        for row in iter_jsonl(path):
            if str(row.get("template_status") or "").lower() in {"template", "unlabeled"}:
                template_count += 1
            elif row.get("label") or row.get("expected_action") or row.get("manual_decision"):
                labeled_count += 1
    manifest = {
        "schema_version": "2026-05-07",
        "generated_at": utc_now(),
        "session_dir": str(session_dir),
        "metric_mode": "formal" if labeled_count else "readiness",
        "gt_completeness": "complete" if labeled_count else "not_started",
        "labeled_window_count": labeled_count,
        "template_window_count": template_count,
        "precision_recall_are_formal": bool(labeled_count),
        "labels_and_training_blocking": False,
    }
    formal_report = {
        "schema_version": "2026-05-07",
        "generated_at": utc_now(),
        "status": "pass" if labeled_count else "needs_gt_for_formal_metrics",
        "summary": manifest,
        "note": "Formal precision/recall requires manual micro-GT; this is not blocking acceptance refresh.",
    }
    return [
        write_json(session_dir / "evaluation" / "evaluation_manifest.json", manifest),
        write_json(session_dir / "reports" / "formal_evaluation_report.json", formal_report),
    ]


def tokenize_query(query: str) -> list[str]:
    tokens = re.findall(r"[\w\u4e00-\u9fff]+", query.lower())
    if len(tokens) == 1 and len(tokens[0]) > 2:
        token = tokens[0]
        tokens.extend(token[idx : idx + 2] for idx in range(0, len(token) - 1))
    return tokens


def score_record(query: str, record: dict[str, Any]) -> float:
    text = record_text(record).lower()
    if not text:
        return 0.0
    score = 0.0
    for token in tokenize_query(query):
        if token and token in text:
            score += max(1.0, len(token) / 2)
    return score


def build_query_acceptance(session_dir: Path, artifacts: dict[str, Any], queries: tuple[str, ...]) -> list[Path]:
    retrieval_names = (
        "retrieval_index.jsonl",
        "segment_index.jsonl",
        "segments.jsonl",
        "micro_segments.jsonl",
        "model_observation_events.jsonl",
        "video_understanding.jsonl",
    )
    paths = [artifacts["important_jsonl"][name] for name in retrieval_names if name in artifacts["important_jsonl"]]
    records = load_records(paths)
    query_results: list[dict[str, Any]] = []
    for query in queries:
        ranked = sorted(
            (
                {
                    "score": score_record(query, record),
                    "evidence_id": safe_id(record, "unknown"),
                    "source": relative(Path(str(record.get("_source_path", ""))), session_dir),
                    "start_s": extract_seconds(record)[0],
                    "end_s": extract_seconds(record)[1],
                    "summary": record_text(record)[:400],
                }
                for record in records
            ),
            key=lambda row: row["score"],
            reverse=True,
        )
        hits = [row for row in ranked if row["score"] > 0][:3]
        query_results.append(
            {
                "query": query,
                "status": "pass" if len(hits) >= DEFAULT_MIN_RETRIEVAL_TOP3_HITS else "needs_review",
                "top_k": hits,
            }
        )
    overall = "skipped" if not records else ("pass" if all(row["status"] == "pass" for row in query_results) else "needs_review")
    report = {
        "schema_version": "2026-05-07",
        "generated_at": utc_now(),
        "session_dir": str(session_dir),
        "overall_status": overall,
        "record_count": len(records),
        "queries": query_results,
    }
    markdown_lines = [
        "# Query Acceptance Report",
        "",
        f"- Status: {overall}",
        f"- Records scanned: {len(records)}",
        f"- Generated at: {report['generated_at']}",
        "",
    ]
    for result in query_results:
        markdown_lines.append(f"## {result['query']}")
        markdown_lines.append(f"- Status: {result['status']}")
        if result["top_k"]:
            for rank, hit in enumerate(result["top_k"], start=1):
                markdown_lines.append(
                    f"- Top {rank}: {hit['evidence_id']} score={hit['score']:.2f} source={hit['source']}"
                )
        else:
            markdown_lines.append("- Top hits: none")
        markdown_lines.append("")
    json_path = write_json(session_dir / "reports" / "query_acceptance_report.json", report)
    md_path = session_dir / "reports" / "query_acceptance_report.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(markdown_lines), encoding="utf-8", newline="\n")
    return [json_path, md_path]


def build_boss_report(
    session_dir: Path,
    quality_path: Path,
    validation: dict[str, Any],
    generated_files: list[Path],
    applied_decisions: Path | None,
) -> Path:
    quality = read_json(quality_path, {})
    checks = quality.get("checks", [])
    lines = [
        "# Key Action Acceptance Boss Report",
        "",
        f"- Session: {session_dir}",
        f"- Generated at: {utc_now()}",
        f"- Overall status: {quality.get('overall_status', 'unknown')}",
        f"- Structured artifacts: {validation['artifact_count']}",
        f"- JSONL records: {validation['jsonl_record_count']}",
        f"- Labels/training blocking: false",
        "",
        "## Checks",
    ]
    for check in checks:
        lines.append(f"- {check.get('id')}: {check.get('status')} - {check.get('summary')}")
    lines.extend(["", "## Human Confirmation"])
    if applied_decisions:
        lines.append(f"- Applied decisions: {relative(applied_decisions, session_dir)}")
    else:
        lines.append("- Decision template: metadata/confirmation_decisions.template.jsonl")
        lines.append("- Run again with --decisions-file and --apply-decisions after human review.")
    lines.extend(["", "## Generated Files"])
    for path in generated_files:
        lines.append(f"- {relative(path, session_dir)}")
    path = session_dir / "reports" / "boss_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def build_acceptance_snapshot(
    options: PipelineOptions,
    validation: dict[str, Any],
    generated_files: list[Path],
    step_results: list[dict[str, Any]],
    started_at: float,
) -> Path:
    session_base = options.session_dir.resolve()
    status = "fail" if any(step.get("status") == "fail" for step in step_results) else "needs_review"
    if all(step.get("status") in {"pass", "skipped", "deferred"} for step in step_results):
        status = "pass"
    payload = {
        "schema_version": "2026-05-07",
        "generated_at": utc_now(),
        "session_dir": str(options.session_dir),
        "status": status,
        "strict": options.strict,
        "dry_run": options.dry_run,
        "duration_s": round(time.time() - started_at, 3),
        "labels_and_training_blocking": False,
        "validation": validation,
        "steps": step_results,
        "generated_files": [relative(path, session_base) for path in generated_files],
        "next_actions": [
            "Fill metadata/confirmation_decisions.template.jsonl and rerun with --apply-decisions.",
            "Review reports/query_acceptance_report.md for retrieval misses.",
            "Add manual micro-GT later when formal precision/recall is required.",
        ],
    }
    return write_json(session_base / "reports" / "acceptance_snapshot.json", payload)


def step(name: str, status: str, **extra: Any) -> dict[str, Any]:
    return {"name": name, "status": status, **extra}


def run_acceptance_pipeline(options: PipelineOptions) -> PipelineResult:
    started_at = time.time()
    session_dir = options.session_dir.resolve()
    output_dir = (options.output_dir or session_dir).resolve()
    if output_dir != session_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    session_dir.mkdir(parents=True, exist_ok=True)

    result = PipelineResult(session_dir=session_dir, output_dir=output_dir, status="needs_review")
    steps: list[dict[str, Any]] = []
    generated: list[Path] = []

    artifacts = discover_artifacts(session_dir)
    validation = validate_artifacts(session_dir, artifacts)
    generated.append(write_json(session_dir / "metadata" / "artifact_validation_report.json", validation))
    steps.append(step("validate_artifacts", "pass" if validation["valid"] else "fail", issues=len(validation["issues"])))

    candidates: list[dict[str, Any]] = []
    if validation["valid"]:
        candidates = build_review_candidates(artifacts, session_dir)
        review_files = write_review_bundle(session_dir, candidates)
        generated.extend(review_files)
        steps.append(step("export_review_bundle", "pass", candidate_count=len(candidates)))
    else:
        steps.append(step("export_review_bundle", "skipped", reason="artifact validation failed"))

    applied_decisions: Path | None = None
    if options.decisions_file and options.apply_decisions:
        applied_decisions = apply_confirmation_decisions(session_dir, options.decisions_file.resolve())
        generated.append(applied_decisions)
        steps.append(step("apply_human_decisions", "pass", decisions_file=str(options.decisions_file)))
    elif options.decisions_file:
        steps.append(step("apply_human_decisions", "skipped", reason="--apply-decisions was not set"))
    else:
        steps.append(step("apply_human_decisions", "needs_review", reason="no decisions file supplied"))

    quality_path = build_quality_report(session_dir, artifacts, validation, candidates)
    generated.append(quality_path)
    quality = read_json(quality_path, {})
    steps.append(step("quality_report", quality.get("overall_status", "needs_review")))

    generated.append(build_capability_gap_report(session_dir, artifacts))
    steps.append(step("capability_gap_normalization", "deferred"))

    generated.extend(build_evaluation_manifest(session_dir, artifacts))
    steps.append(step("formal_evaluation_readiness", "deferred"))

    if validation["valid"]:
        generated.extend(build_query_acceptance(session_dir, artifacts, options.queries))
        query_status = read_json(session_dir / "reports" / "query_acceptance_report.json", {}).get("overall_status", "unknown")
    else:
        query_report = {
            "schema_version": "2026-05-07",
            "generated_at": utc_now(),
            "session_dir": str(session_dir),
            "overall_status": "skipped",
            "reason": "artifact validation failed",
            "queries": [{"query": query, "status": "skipped", "top_k": []} for query in options.queries],
        }
        generated.append(write_json(session_dir / "reports" / "query_acceptance_report.json", query_report))
        md_path = session_dir / "reports" / "query_acceptance_report.md"
        md_path.write_text(
            "# Query Acceptance Report\n\n- Status: skipped\n- Reason: artifact validation failed\n",
            encoding="utf-8",
            newline="\n",
        )
        generated.append(md_path)
        query_status = "skipped"
    steps.append(step("query_acceptance", query_status))

    boss_path = build_boss_report(session_dir, quality_path, validation, generated, applied_decisions)
    generated.append(boss_path)

    snapshot_path = build_acceptance_snapshot(options, validation, generated, steps, started_at)
    generated.append(snapshot_path)

    result.generated_files = generated
    result.step_results = steps
    result.errors = [issue["message"] for issue in validation["issues"] if issue["severity"] == "error"]
    result.warnings = [
        step_result.get("reason", step_result["name"])
        for step_result in steps
        if step_result["status"] in {"needs_review", "deferred", "skipped"}
    ]
    result.status = read_json(snapshot_path, {}).get("status", "needs_review")

    if options.strict and result.errors:
        result.status = "fail"
    if options.strict and not options.dry_run and validation["source_evidence_artifact_count"] == 0:
        result.status = "fail"
        result.errors.append("strict mode requires at least one source evidence JSONL artifact")
    return result


def add_acceptance_pipeline_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("acceptance-pipeline", help="Run the key action acceptance pipeline.")
    parser.add_argument("--session-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--decisions-file", type=Path)
    parser.add_argument("--apply-decisions", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--query", action="append", dest="queries")
    parser.set_defaults(func=acceptance_pipeline_command)


def add_validate_artifacts_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("validate-artifacts", help="Validate JSON/JSONL artifacts in a session.")
    parser.add_argument("--session-dir", required=True, type=Path)
    parser.add_argument("--strict", action="store_true")
    parser.set_defaults(func=validate_artifacts_command)


def acceptance_pipeline_command(args: argparse.Namespace) -> int:
    result = run_acceptance_pipeline(
        PipelineOptions(
            session_dir=args.session_dir,
            output_dir=args.output_dir,
            decisions_file=args.decisions_file,
            apply_decisions=args.apply_decisions,
            strict=args.strict,
            dry_run=args.dry_run,
            queries=tuple(args.queries) if args.queries else DEFAULT_QUERIES,
        )
    )
    print(json.dumps(summarize_result(result), ensure_ascii=False, indent=2))
    return 0 if result.ok else 1


def validate_artifacts_command(args: argparse.Namespace) -> int:
    session_dir = args.session_dir.resolve()
    artifacts = discover_artifacts(session_dir)
    validation = validate_artifacts(session_dir, artifacts)
    if args.strict and validation["source_evidence_artifact_count"] == 0:
        validation["valid"] = False
        validation["issues"].append(
            {
                "path": str(session_dir),
                "severity": "error",
                "message": "strict mode requires at least one source evidence JSONL artifact",
            }
        )
    path = write_json(session_dir / "metadata" / "artifact_validation_report.json", validation)
    print(json.dumps({"report": str(path), **validation}, ensure_ascii=False, indent=2))
    if args.strict and not validation["valid"]:
        return 1
    return 0


def summarize_result(result: PipelineResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "session_dir": str(result.session_dir),
        "output_dir": str(result.output_dir),
        "generated_file_count": len(result.generated_files),
        "generated_files": [relative(path, result.session_dir) for path in result.generated_files],
        "errors": result.errors,
        "warnings": result.warnings,
        "steps": result.step_results,
    }

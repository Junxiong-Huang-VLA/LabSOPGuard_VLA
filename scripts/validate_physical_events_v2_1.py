#!/usr/bin/env python
"""Validate v2.1 physical event gate artifacts.

Usage:
    python scripts/validate_physical_events_v2_1.py <output_dir> [<output_dir> ...]

The checker is intentionally conservative: invariant violations are FAIL,
missing optional diagnostics are WARN, and clean directories are PASS.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


VALID_SCHEMAS = {"physical_events.v4", "physical_events.v2", "physical_events.v2.1"}
EVENT_TYPES = {
    "hand_object_interaction",
    "hand_object_contact",
    "object_move",
    "liquid_transfer",
    "panel_operation",
    "container_state_change",
}


@dataclass
class DirectoryReport:
    output_dir: Path
    status: str = "PASS"
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    event_counts: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    qwen_counts: Counter = field(default_factory=Counter)
    reject_reasons: Counter = field(default_factory=Counter)

    def fail(self, message: str) -> None:
        self.errors.append(message)
        self.status = "FAIL"

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        if self.status == "PASS":
            self.status = "PASS_WITH_WARNINGS"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dirs", nargs="+", help="Experiment output directories to validate")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    args = parser.parse_args(argv)

    reports = [validate_output_dir(Path(value)) for value in args.output_dirs]
    overall = _overall_status(reports)
    summary = _summary(reports, overall)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        _print_text(summary)
    return 1 if overall == "FAIL" else 0


def validate_output_dir(output_dir: Path) -> DirectoryReport:
    report = DirectoryReport(output_dir=output_dir)
    physical_path = output_dir / "physical_events.json"
    summary_path = output_dir / "physical_event_gate_summary.json"
    rejected_path = output_dir / "rejected_physical_event_candidates.jsonl"
    gate_path = output_dir / "physical_event_gate_decisions.jsonl"
    qwen_path = output_dir / "qwen_event_audits.jsonl"

    payload = _read_json(physical_path, report)
    if payload is None:
        return report
    if not isinstance(payload, Mapping):
        report.fail("physical_events.json must be a JSON object")
        return report

    summary = _read_json(summary_path, report, required=False)
    if summary_path.exists() and not isinstance(summary, Mapping):
        report.fail("physical_event_gate_summary.json must be a JSON object")
        summary = {}
    elif not summary_path.exists():
        report.warn("missing physical_event_gate_summary.json")
        summary = {}

    status = str(payload.get("status") or "ok")
    events = payload.get("events")
    if status == "failed":
        if events not in ([], None):
            report.fail("failed physical_events payload must have empty events")
        if payload.get("gate_required") is not True:
            report.fail("failed physical_events payload must set gate_required=true")
        if not summary or not summary.get("failure_reason"):
            report.fail("failed payload requires summary.failure_reason")
        _validate_jsonl_artifacts(report, rejected_path, gate_path, qwen_path, summary)
        return report

    schema = payload.get("schema") or payload.get("schema_version")
    if schema not in VALID_SCHEMAS:
        report.fail(f"unsupported physical_events schema: {schema!r}")
    if not isinstance(events, list):
        report.fail("physical_events.events must be a list")
        events = []

    for index, event in enumerate(events):
        _validate_event(report, index, event)

    _validate_jsonl_artifacts(report, rejected_path, gate_path, qwen_path, summary)
    return report


def _validate_event(report: DirectoryReport, index: int, event: Any) -> None:
    if not isinstance(event, Mapping):
        report.fail(f"event[{index}] must be an object")
        return
    event_type = str(event.get("event_type") or "")
    status = str(event.get("status") or "")
    report.event_counts[event_type or "unknown"][status or "missing"] += 1

    for field_name in ("event_type", "status", "hard_gate", "reject_reasons", "limitations"):
        if field_name not in event:
            report.fail(f"event[{index}] missing {field_name}")
    if "evidence_detail" not in event and "evidence" not in event:
        report.fail(f"event[{index}] missing evidence_detail/evidence")

    hard_gate = event.get("hard_gate")
    if not isinstance(hard_gate, Mapping):
        report.fail(f"event[{index}] hard_gate must be an object")
        hard_gate = {}

    if status == "confirmed":
        if hard_gate.get("passed") is not True:
            report.fail(f"event[{index}] confirmed without hard_gate.passed=true")
        if str(hard_gate.get("status") or event.get("status") or "") != "confirmed":
            report.fail(f"event[{index}] confirmed without confirmed hard_gate/event status")
        if _is_legacy_source(event):
            report.fail(f"event[{index}] confirmed event comes from legacy/ungated source")
        if event_type.endswith("_candidate"):
            report.fail(f"event[{index}] confirmed candidate event_type is invalid")
        qwen = event.get("qwen_audit")
        if isinstance(qwen, Mapping) and _qwen_upgrade_attempt(qwen):
            report.fail(f"event[{index}] qwen_audit indicates forbidden upgrade")
        if event_type == "object_move":
            _validate_confirmed_object_move(report, index, event)
        if event_type == "container_state_change":
            _validate_confirmed_container_state(report, index, event)


def _validate_confirmed_object_move(report: DirectoryReport, index: int, event: Mapping[str, Any]) -> None:
    evidence = _evidence(event)
    required = [
        "raw_displacement_px",
        "motion_threshold_px",
        "track_type",
        "can_confirm_motion",
        "identity_confidence",
        "id_switch_risk",
    ]
    for field_name in required:
        if field_name not in evidence:
            report.fail(f"event[{index}] confirmed object_move missing {field_name}")
    if "stabilized_displacement_px" not in evidence:
        limitations = " ".join(str(item) for item in event.get("limitations") or evidence.get("limitations") or [])
        if "scene" not in limitations.lower():
            report.fail(f"event[{index}] confirmed object_move missing stabilized_displacement_px or scene limitation")
    if evidence.get("track_type") == "label_level_pseudo_track":
        report.fail(f"event[{index}] confirmed object_move uses label_level_pseudo_track")
    if evidence.get("can_confirm_motion") is False:
        report.fail(f"event[{index}] confirmed object_move has can_confirm_motion=false")


def _validate_confirmed_container_state(report: DirectoryReport, index: int, event: Mapping[str, Any]) -> None:
    evidence = _evidence(event)
    same_instance = bool(evidence.get("same_container_instance"))
    if not evidence.get("container_track_id") and not same_instance:
        report.fail(f"event[{index}] confirmed container_state_change missing same-container evidence")
    changed = evidence.get("changed_fields")
    if not isinstance(changed, list) or not changed:
        report.fail(f"event[{index}] confirmed container_state_change missing changed_fields")


def _validate_jsonl_artifacts(
    report: DirectoryReport,
    rejected_path: Path,
    gate_path: Path,
    qwen_path: Path,
    summary: Mapping[str, Any],
) -> None:
    rejected_rows = _read_jsonl(rejected_path, report, required=False)
    gate_rows = _read_jsonl(gate_path, report, required=False)
    qwen_rows = _read_jsonl(qwen_path, report, required=False)

    if not rejected_path.exists():
        report.warn("missing rejected_physical_event_candidates.jsonl")
    for index, row in enumerate(rejected_rows):
        for field_name in ("event_type", "status", "reject_reasons", "evidence_detail"):
            if field_name not in row:
                report.fail(f"rejected[{index}] missing {field_name}")
        for reason in row.get("reject_reasons") or []:
            report.reject_reasons[str(reason)] += 1
        if row.get("event_type") == "object_move":
            evidence = row.get("evidence_detail") or {}
            for field_name in (
                "raw_displacement_px",
                "background_shift_px",
                "stabilized_displacement_px",
                "motion_threshold_px",
                "jitter_sigma_px",
                "track_type",
                "can_confirm_motion",
                "identity_confidence",
                "id_switch_risk",
            ):
                if field_name not in evidence:
                    report.warn(f"rejected object_move missing {field_name}: {row.get('candidate_id')}")

    if not gate_path.exists():
        report.warn("missing physical_event_gate_decisions.jsonl")
    gate_counts = Counter(str(row.get("status") or "missing") for row in gate_rows if isinstance(row, Mapping))
    for key in ("confirmed", "candidate", "rejected", "uncertain"):
        if key in summary and gate_rows and int(summary.get(key) or 0) != gate_counts.get(key, 0):
            report.warn(f"summary {key}={summary.get(key)} differs from gate_decisions={gate_counts.get(key, 0)}")

    if not qwen_path.exists():
        report.warn("missing qwen_event_audits.jsonl")
    for index, row in enumerate(qwen_rows):
        decision = str(row.get("qwen_decision") or row.get("decision") or row.get("status") or "unknown")
        report.qwen_counts[decision] += 1
        missing_gate = _audit_missing_gate(row)
        status_ok = _audit_status_confirmed(row)
        should_write = row.get("should_write_confirmed_event") is True
        if missing_gate and should_write:
            report.fail(f"qwen[{index}] missing hard gate but should_write_confirmed_event=true")
        if not status_ok and should_write:
            report.fail(f"qwen[{index}] non-confirmed hard gate but should_write_confirmed_event=true")
        if str(row.get("qwen_decision") or "") == "parse_failed" and should_write:
            report.fail(f"qwen[{index}] parse_failed but should_write_confirmed_event=true")
        if missing_gate:
            report.qwen_counts["missing_hard_gate"] += 1


def _read_json(path: Path, report: DirectoryReport, *, required: bool = True) -> Any:
    if not path.exists():
        if required:
            report.fail(f"missing {path.name}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        report.fail(f"invalid JSON {path.name}: {exc}")
        return None


def _read_jsonl(path: Path, report: DirectoryReport, *, required: bool = True) -> list[dict[str, Any]]:
    if not path.exists():
        if required:
            report.fail(f"missing {path.name}")
        return []
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception as exc:
            report.fail(f"invalid JSONL {path.name}:{line_no}: {exc}")
            continue
        if not isinstance(row, dict):
            report.fail(f"JSONL row must be object {path.name}:{line_no}")
            continue
        rows.append(row)
    return rows


def _evidence(event: Mapping[str, Any]) -> Mapping[str, Any]:
    detail = event.get("evidence_detail")
    if isinstance(detail, Mapping):
        return detail
    evidence = event.get("evidence")
    if isinstance(evidence, Mapping):
        return evidence
    return {}


def _is_legacy_source(event: Mapping[str, Any]) -> bool:
    haystack = " ".join(
        str(event.get(key) or "")
        for key in ("source", "proposal_source", "candidate_source", "provenance", "action_resolution_source")
    ).lower()
    return "legacy" in haystack or "ungated" in haystack or bool(event.get("legacy_ungated"))


def _qwen_upgrade_attempt(qwen: Mapping[str, Any]) -> bool:
    gate_status = str(qwen.get("hard_gate_status") or qwen.get("gate_status") or "")
    decision = str(qwen.get("qwen_decision") or qwen.get("decision") or "")
    should_write = qwen.get("should_write_confirmed_event") is True
    return should_write and decision == "accept" and gate_status not in {"confirmed", "['confirmed']", '["confirmed"]'}


def _audit_missing_gate(row: Mapping[str, Any]) -> bool:
    status = row.get("hard_gate_status")
    if status in (None, "", [], {}):
        return True
    if isinstance(status, list):
        return not status
    return "missing_hard_gate" in set(str(item) for item in row.get("missing_evidence") or [])


def _audit_status_confirmed(row: Mapping[str, Any]) -> bool:
    status = row.get("hard_gate_status")
    if isinstance(status, list):
        return bool(status) and all(str(item) == "confirmed" for item in status)
    if status in (None, "", [], {}):
        return False
    return str(status) == "confirmed"


def _overall_status(reports: Iterable[DirectoryReport]) -> str:
    values = [report.status for report in reports]
    if any(value == "FAIL" for value in values):
        return "FAIL"
    if any(value == "PASS_WITH_WARNINGS" for value in values):
        return "PASS_WITH_WARNINGS"
    return "PASS"


def _summary(reports: list[DirectoryReport], overall: str) -> dict[str, Any]:
    event_totals: dict[str, Counter] = defaultdict(Counter)
    qwen_totals: Counter = Counter()
    reject_totals: Counter = Counter()
    for report in reports:
        for event_type, counts in report.event_counts.items():
            event_totals[event_type].update(counts)
        qwen_totals.update(report.qwen_counts)
        reject_totals.update(report.reject_reasons)
    return {
        "status": overall,
        "directories": [
            {
                "output_dir": str(report.output_dir),
                "status": report.status,
                "errors": report.errors,
                "warnings": report.warnings,
                "event_counts": {key: dict(value) for key, value in sorted(report.event_counts.items())},
                "qwen_counts": dict(report.qwen_counts),
                "top_reject_reasons": report.reject_reasons.most_common(10),
            }
            for report in reports
        ],
        "event_totals": {key: dict(value) for key, value in sorted(event_totals.items())},
        "qwen_totals": dict(qwen_totals),
        "top_reject_reasons": reject_totals.most_common(10),
    }


def _print_text(summary: Mapping[str, Any]) -> None:
    print(f"STATUS: {summary['status']}")
    print("DIRECTORIES:")
    for item in summary["directories"]:
        print(f"- {item['status']} {item['output_dir']}")
        for error in item["errors"]:
            print(f"  ERROR: {error}")
        for warning in item["warnings"]:
            print(f"  WARN: {warning}")
    print("EVENT_TOTALS:")
    print(json.dumps(summary["event_totals"], ensure_ascii=False, sort_keys=True))
    print("QWEN_TOTALS:")
    print(json.dumps(summary["qwen_totals"], ensure_ascii=False, sort_keys=True))
    print("TOP_REJECT_REASONS:")
    print(json.dumps(summary["top_reject_reasons"], ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())

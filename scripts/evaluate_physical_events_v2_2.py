#!/usr/bin/env python
"""Evaluate v2.2 physical-event candidate recall against JSONL annotations.

The evaluator intentionally separates candidate-layer recall from confirmed
precision.  Raw proposals and gate decisions may count as candidate hits, but
only final gated physical_events.json entries count as confirmed hits.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


POSITIVE_STATUSES = {"should_confirm", "should_candidate"}
HIT_STATUSES = {"confirmed", "candidate", "uncertain"}
DEFAULT_IOU = 0.3


@dataclass
class SystemItem:
    output_dir: Path
    source: str
    event_type: str
    status: str
    start: float
    end: float
    event_id: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class OutputBundle:
    output_dir: Path
    summary: dict[str, Any]
    confirmed_events: list[SystemItem]
    candidate_items: list[SystemItem]
    trace_summary: dict[str, Any]
    qwen_rows: list[dict[str, Any]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", help="benchmarks/physical_events_v2_2/annotations.jsonl")
    parser.add_argument("output_dirs", nargs="+", help="One or more v2.2 output directories")
    parser.add_argument("--output-dir", default=None, help="Directory for benchmark_eval_summary.json/details.jsonl")
    parser.add_argument("--temporal-iou", type=float, default=DEFAULT_IOU)
    parser.add_argument("--json", action="store_true", help="Print JSON summary")
    args = parser.parse_args(argv)

    annotations = _read_jsonl(Path(args.annotations))
    bundles = [_load_output_bundle(Path(value)) for value in args.output_dirs]
    result = evaluate(annotations, bundles, temporal_iou_threshold=args.temporal_iou)

    out_dir = Path(args.output_dir) if args.output_dir else Path(args.annotations).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "benchmark_eval_summary.json").write_text(json.dumps(result["summary"], ensure_ascii=False, indent=2), encoding="utf-8")
    _write_jsonl(out_dir / "benchmark_eval_details.jsonl", result["details"])
    if args.json:
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    else:
        _print_summary(result["summary"])
    return 1 if result["summary"]["overall_status"] == "FAIL" else 0


def evaluate(
    annotations: list[dict[str, Any]],
    bundles: list[OutputBundle],
    *,
    temporal_iou_threshold: float = DEFAULT_IOU,
) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    positives = [row for row in annotations if str(row.get("expected_status")) in POSITIVE_STATUSES and not bool(row.get("negative"))]
    negatives = [row for row in annotations if bool(row.get("negative")) or str(row.get("expected_status")) == "should_reject"]
    positive_hits = 0
    positive_confirmed_hits = 0
    type_hits = 0
    temporal_scores: list[float] = []
    false_confirmed = 0
    confirmed_total = 0
    confirmed_true_positive = 0
    by_type = defaultdict(lambda: Counter({"positive": 0, "negative": 0, "candidate_hit": 0, "confirmed_hit": 0}))
    qwen_counts = Counter()

    for bundle in bundles:
        confirmed_total += len(bundle.confirmed_events)
        for row in bundle.qwen_rows:
            qwen_counts[str(row.get("qwen_decision") or row.get("decision") or row.get("status") or "unknown")] += 1
            if _audit_missing_gate(row):
                qwen_counts["missing_hard_gate"] += 1

    for ann in annotations:
        event_type = str(ann.get("event_type") or "")
        is_positive = ann in positives
        by_type[event_type]["positive" if is_positive else "negative"] += 1
        bundle = _match_bundle(ann, bundles)
        if bundle is None:
            detail = _detail(ann, None, None, "artifact_missing")
            details.append(detail)
            continue
        candidate_match = _best_match(ann, bundle.candidate_items + bundle.confirmed_events, temporal_iou_threshold)
        confirmed_match = _best_match(ann, bundle.confirmed_events, temporal_iou_threshold)
        if is_positive:
            if candidate_match:
                positive_hits += 1
                by_type[event_type]["candidate_hit"] += 1
                if candidate_match.event_type == event_type:
                    type_hits += 1
                temporal_scores.append(_temporal_iou(_span(ann), (candidate_match.start, candidate_match.end)))
            if confirmed_match:
                positive_confirmed_hits += 1
                by_type[event_type]["confirmed_hit"] += 1
            missing_reason = "" if candidate_match else _missing_reason(ann, bundle)
            if candidate_match and str(ann.get("expected_status")) == "should_confirm" and not confirmed_match:
                missing_reason = "gate_candidate_not_confirmed"
            details.append(_detail(ann, bundle, candidate_match or confirmed_match, missing_reason or "hit"))
        else:
            if confirmed_match:
                false_confirmed += 1
                details.append(_detail(ann, bundle, confirmed_match, "false_confirmed_on_negative"))
            else:
                details.append(_detail(ann, bundle, candidate_match, "negative_no_confirmed"))

    positive_ann = [row for row in positives]
    for bundle in bundles:
        for event in bundle.confirmed_events:
            if _best_annotation_match(event, positive_ann, temporal_iou_threshold):
                confirmed_true_positive += 1

    candidate_recall = _ratio(positive_hits, len(positives))
    confirmed_recall = _ratio(positive_confirmed_hits, sum(1 for row in positives if str(row.get("expected_status")) == "should_confirm"))
    confirmed_precision = _ratio(confirmed_true_positive, confirmed_total) if confirmed_total else None
    event_type_match = _ratio(type_hits, positive_hits)
    mean_temporal_iou = round(sum(temporal_scores) / len(temporal_scores), 4) if temporal_scores else None
    false_confirmed_on_negatives = false_confirmed
    insufficient_confirmed = confirmed_total < 3

    overall = "PASS"
    warnings: list[str] = []
    if false_confirmed_on_negatives:
        overall = "FAIL"
    if candidate_recall is not None and candidate_recall < 0.5:
        overall = "FAIL"
    elif candidate_recall is not None and candidate_recall < 0.7:
        warnings.append("candidate_recall_below_0.70")
        if overall == "PASS":
            overall = "PASS_WITH_WARNINGS"
    if confirmed_precision is not None and confirmed_precision < 0.9:
        overall = "FAIL"
    if insufficient_confirmed:
        warnings.append("insufficient_confirmed_samples")
        if overall == "PASS":
            overall = "PASS_WITH_WARNINGS"

    summary = {
        "schema": "benchmark_eval_summary.v2.2",
        "overall_status": overall,
        "warnings": warnings,
        "annotation_count": len(annotations),
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "candidate_recall": candidate_recall,
        "confirmed_precision": confirmed_precision,
        "confirmed_recall": confirmed_recall,
        "false_confirmed_on_negatives": false_confirmed_on_negatives,
        "event_type_match": event_type_match,
        "mean_temporal_iou": mean_temporal_iou,
        "confirmed_total": confirmed_total,
        "confirmed_true_positive": confirmed_true_positive,
        "qwen_audit_counts": dict(qwen_counts),
        "by_event_type": {key: dict(value) for key, value in sorted(by_type.items())},
        "missing_reasons": dict(Counter(row.get("missing_reason") for row in details if row.get("missing_reason"))),
    }
    return {"summary": summary, "details": details}


def _load_output_bundle(output_dir: Path) -> OutputBundle:
    physical = _read_json(output_dir / "physical_events.json")
    trace_summary = _read_json(output_dir / "event_candidate_trace_summary.json")
    confirmed: list[SystemItem] = []
    candidate: list[SystemItem] = []
    if physical:
        for event in physical.get("events") or []:
            if not isinstance(event, Mapping):
                continue
            item = SystemItem(
                output_dir=output_dir,
                source="physical_events",
                event_type=str(event.get("event_type") or ""),
                status=str(event.get("status") or ""),
                start=_float(event.get("start_time_sec") or event.get("time_start")),
                end=_float(event.get("end_time_sec") or event.get("time_end")),
                event_id=str(event.get("event_id") or ""),
                payload=dict(event),
            )
            if item.status == "confirmed":
                confirmed.append(item)
            if item.status in HIT_STATUSES:
                candidate.append(item)
    else:
        physical = {"schema": "key_action_indexer.v2.2", "events": []}
    for row in _read_jsonl(output_dir / "raw_event_proposals.jsonl"):
        item = SystemItem(
            output_dir=output_dir,
            source="raw_event_proposals",
            event_type=str(row.get("event_type") or ""),
            status="candidate",
            start=_float(row.get("time_start")),
            end=_float(row.get("time_end")),
            event_id=str(row.get("proposal_id") or ""),
            payload=dict(row),
        )
        candidate.append(item)
    for row in _read_jsonl(output_dir / "physical_event_gate_decisions.jsonl"):
        item = SystemItem(
            output_dir=output_dir,
            source="physical_event_gate_decisions",
            event_type=str(row.get("event_type") or ""),
            status=str(row.get("status") or "uncertain"),
            start=_float(row.get("time_start")),
            end=_float(row.get("time_end")),
            event_id=str(row.get("candidate_id") or ""),
            payload=dict(row),
        )
        if item.status in HIT_STATUSES:
            candidate.append(item)
    return OutputBundle(
        output_dir=output_dir,
        summary=physical,
        confirmed_events=confirmed,
        candidate_items=candidate,
        trace_summary=trace_summary,
        qwen_rows=_read_jsonl(output_dir / "qwen_event_audits.jsonl"),
    )


def _match_bundle(annotation: Mapping[str, Any], bundles: list[OutputBundle]) -> OutputBundle | None:
    ann_video = _norm_path(annotation.get("video_path"))
    ann_name = Path(str(annotation.get("video_path") or "")).name.lower()
    candidates: list[OutputBundle] = []
    for bundle in bundles:
        summary_video = _norm_path(bundle.trace_summary.get("source_video_path"))
        if ann_video and summary_video and ann_video == summary_video:
            candidates.append(bundle)
        elif ann_name and ann_name == Path(str(bundle.trace_summary.get("source_video_path") or "")).name.lower():
            candidates.append(bundle)
    if not candidates:
        return None
    ann_span = _span(annotation)
    candidates.sort(key=lambda bundle: _range_distance(ann_span, bundle.trace_summary.get("time_range")))
    return candidates[0]


def _best_match(annotation: Mapping[str, Any], items: Iterable[SystemItem], threshold: float) -> SystemItem | None:
    ann_span = _span(annotation)
    ann_type = str(annotation.get("event_type") or "")
    scored: list[tuple[float, int, SystemItem]] = []
    for item in items:
        if item.event_type and ann_type and item.event_type != ann_type:
            continue
        item_span = (item.start, item.end)
        iou = _temporal_iou(ann_span, item_span)
        center = (item.start + item.end) / 2.0
        center_hit = ann_span[0] <= center <= ann_span[1]
        if iou >= threshold or center_hit:
            scored.append((iou, 1 if item.status == "confirmed" else 0, item))
    if not scored:
        return None
    scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return scored[0][2]


def _best_annotation_match(event: SystemItem, annotations: Iterable[Mapping[str, Any]], threshold: float) -> Mapping[str, Any] | None:
    for ann in annotations:
        if str(ann.get("event_type") or "") != event.event_type:
            continue
        if _norm_path(ann.get("video_path")) != _norm_path(event.payload.get("source_video_path") or event.output_dir):
            pass
        iou = _temporal_iou(_span(ann), (event.start, event.end))
        center = (event.start + event.end) / 2.0
        if iou >= threshold or _span(ann)[0] <= center <= _span(ann)[1]:
            return ann
    return None


def _missing_reason(annotation: Mapping[str, Any], bundle: OutputBundle) -> str:
    summary = bundle.trace_summary or {}
    diagnosis = set(summary.get("zero_candidate_diagnosis") or [])
    if not summary:
        return "artifact_missing"
    if "no_yolo_detections" in diagnosis:
        return "no_detection"
    if "tracklets_empty" in diagnosis:
        return "no_tracklet"
    if "relations_empty" in diagnosis:
        return "no_relation"
    event_type = str(annotation.get("event_type") or "")
    proposals = summary.get("raw_proposals_by_type") or {}
    if not proposals or int(proposals.get(event_type, 0) or 0) <= 0:
        return "no_raw_proposal"
    gate = (summary.get("gate_decisions_by_type") or {}).get(event_type) or {}
    if int(gate.get("rejected", 0) or 0) > 0:
        return "gate_rejected"
    if int(gate.get("candidate", 0) or 0) > 0 or int(gate.get("uncertain", 0) or 0) > 0:
        return "gate_candidate_not_confirmed"
    return "unknown"


def _detail(annotation: Mapping[str, Any], bundle: OutputBundle | None, item: SystemItem | None, missing_reason: str) -> dict[str, Any]:
    return {
        "annotation_id": annotation.get("annotation_id"),
        "video_path": annotation.get("video_path"),
        "event_type": annotation.get("event_type"),
        "expected_status": annotation.get("expected_status"),
        "negative": bool(annotation.get("negative")),
        "output_dir": str(bundle.output_dir) if bundle else None,
        "system_result": {
            "source": item.source,
            "event_id": item.event_id,
            "status": item.status,
            "event_type": item.event_type,
            "time_start": item.start,
            "time_end": item.end,
            "temporal_iou": _temporal_iou(_span(annotation), (item.start, item.end)),
        } if item else None,
        "missing_reason": missing_reason,
    }


def _span(row: Mapping[str, Any]) -> tuple[float, float]:
    start = _float(row.get("start_time") or row.get("time_start") or row.get("start_time_sec"))
    end = _float(row.get("end_time") or row.get("time_end") or row.get("end_time_sec"))
    if end < start:
        end = start
    return start, end


def _temporal_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    start = max(a[0], b[0])
    end = min(a[1], b[1])
    inter = max(0.0, end - start)
    union = max(a[1], b[1]) - min(a[0], b[0])
    return round(inter / union, 4) if union > 0 else 0.0


def _range_distance(span: tuple[float, float], range_value: Any) -> float:
    if not isinstance(range_value, list) or len(range_value) != 2:
        return 0.0
    start, end = _float(range_value[0]), _float(range_value[1])
    center = (span[0] + span[1]) / 2.0
    if start <= center <= end:
        return 0.0
    return min(abs(center - start), abs(center - end))


def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _norm_path(value: Any) -> str:
    if not value:
        return ""
    return str(value).replace("\\", "/").lower()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _audit_missing_gate(row: Mapping[str, Any]) -> bool:
    status = row.get("hard_gate_status")
    return status in (None, "", "missing") or bool(row.get("missing_hard_gate"))


def _print_summary(summary: Mapping[str, Any]) -> None:
    print(f"overall_status: {summary.get('overall_status')}")
    for key in (
        "candidate_recall",
        "confirmed_precision",
        "confirmed_recall",
        "false_confirmed_on_negatives",
        "event_type_match",
        "mean_temporal_iou",
    ):
        print(f"{key}: {summary.get(key)}")


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .evidence_adapter_validation import validate_evidence_adapters
from .health_report import build_run_health_report
from .reviewed_dataset import REVIEWED_MANIFEST_FILENAME, REVIEWED_VECTOR_METADATA_FILENAME, active_reviewed_release, reviewed_metadata_path
from .schemas import read_jsonl


QUALITY_GATE_SCHEMA_VERSION = "key_action_quality_gate.v1"
QUALITY_GATE_FILENAME = "quality_gate.json"
LOW_CONFIDENCE_THRESHOLD = 0.55
REVIEWED_VISUAL_CONFIDENCE_FLOOR = 0.5

DEFAULT_GATE_POLICY = {
    "min_health_score": 82,
    "max_total_action_coverage_ratio": 0.65,
    "max_longest_segment_ratio": 0.5,
    "max_unreviewed_count": 0,
    "max_low_confidence_segment_count": 0,
    "min_vector_count": 1,
    "require_adapter_inputs": True,
    "block_on_adapter_validation_error": True,
    "block_on_adapter_semantic_issue": False,
    "block_on_review_required": False,
    "review_required_checks": [
        "unreviewed_count",
        "low_confidence_segment_count",
        "adapter_semantic_issue_count",
    ],
}


def build_quality_gate(
    session_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    policy: Mapping[str, Any] | None = None,
    quality_convergence: Mapping[str, Any] | None = None,
    adapter_validation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    session = Path(session_dir)
    metadata = session / "metadata"
    active_policy = {**DEFAULT_GATE_POLICY, **dict(policy or {})}
    health = (
        quality_convergence.get("health")
        if isinstance(quality_convergence, Mapping) and isinstance(quality_convergence.get("health"), Mapping)
        else build_run_health_report(session)
    )
    metrics = health.get("metrics") if isinstance(health.get("metrics"), Mapping) else {}
    reviewed_manifest = _active_reviewed_manifest(session)
    reviewed_metrics = reviewed_manifest.get("reviewed_metrics") if isinstance(reviewed_manifest.get("reviewed_metrics"), Mapping) else {}
    convergence_metrics = (
        quality_convergence.get("core_metrics")
        if isinstance(quality_convergence, Mapping) and isinstance(quality_convergence.get("core_metrics"), Mapping)
        else {}
    )
    gate_metrics = dict(metrics)
    metric_source = "raw_health_report"
    if reviewed_metrics:
        gate_metrics.update(dict(reviewed_metrics))
        metric_source = "reviewed_dataset"
    adapters = dict(adapter_validation or validate_evidence_adapters(session))
    gate_health_errors = _gate_health_errors(health, metric_source=metric_source)

    vector_count = _vector_count(session)
    unreviewed_count = int(
        _float(
            reviewed_metrics.get("unreviewed_count") if reviewed_metrics else convergence_metrics.get("unreviewed_count"),
            0.0,
        )
    )
    low_confidence_segment_count = _low_confidence_segment_count(session)
    health_score = (
        _reviewed_health_score(reviewed_metrics, error_count=len(gate_health_errors), policy=active_policy)
        if reviewed_metrics
        else _float(quality_convergence.get("health_score") if isinstance(quality_convergence, Mapping) else None, 100.0)
    )
    checks = [
        _check_min(
            "health_score",
            health_score,
            _float(active_policy["min_health_score"]),
            "health score is below completion threshold",
        ),
        _check_max(
            "total_action_coverage_ratio",
            _float(gate_metrics.get("total_action_coverage_ratio")),
            _float(active_policy["max_total_action_coverage_ratio"]),
            "total action coverage is too broad",
        ),
        _check_max(
            "longest_segment_ratio",
            _float(gate_metrics.get("longest_segment_ratio")),
            _float(active_policy["max_longest_segment_ratio"]),
            "longest segment is too coarse",
        ),
        _check_max(
            "unreviewed_count",
            float(unreviewed_count),
            _float(active_policy["max_unreviewed_count"]),
            "review queue still has unresolved evidence",
        ),
        _check_max(
            "low_confidence_segment_count",
            float(low_confidence_segment_count),
            _float(active_policy["max_low_confidence_segment_count"]),
            "low-confidence segments need review or rejection",
        ),
        _check_min(
            "vector_count",
            float(vector_count),
            _float(active_policy["min_vector_count"]),
            "retrieval vector metadata is missing",
        ),
    ]

    adapter_summary = adapters.get("summary") if isinstance(adapters.get("summary"), Mapping) else {}
    missing_adapter_count = int(adapter_summary.get("missing_adapter_count") or 0)
    adapter_error_count = int(adapter_summary.get("error_count") or 0)
    adapter_semantic_issue_count = int(adapter_summary.get("semantic_issue_count") or 0)
    if active_policy.get("require_adapter_inputs"):
        checks.append(
            _check_max(
                "missing_adapter_count",
                float(missing_adapter_count),
                0.0,
                "advanced evidence adapter inputs are missing",
            )
        )
    if active_policy.get("block_on_adapter_validation_error"):
        checks.append(
            _check_max(
                "adapter_validation_error_count",
                float(adapter_error_count),
                0.0,
                "advanced evidence adapter rows are malformed or misaligned",
            )
        )
    checks.append(
        _check_max(
            "adapter_semantic_issue_count",
            float(adapter_semantic_issue_count),
            0.0,
            "advanced evidence adapter rows do not support their expected action semantics",
        )
    )

    error_count = len(gate_health_errors)
    checks.append(_check_max("health_error_count", float(error_count), 0.0, "health report contains errors"))

    review_required_names = {
        str(item)
        for item in active_policy.get("review_required_checks", [])
        if str(item).strip()
    }
    if active_policy.get("block_on_review_required"):
        review_required_names = set()
    if active_policy.get("block_on_adapter_semantic_issue"):
        review_required_names.discard("adapter_semantic_issue_count")
    blocking: list[dict[str, Any]] = []
    review_required: list[dict[str, Any]] = []
    for item in checks:
        if item["status"] != "fail":
            continue
        item["severity"] = "review_required" if item.get("name") in review_required_names else "blocking"
        if item["severity"] == "review_required":
            review_required.append(item)
        else:
            blocking.append(item)
    warning = [item for item in checks if item["status"] == "warning"]
    requires_review = bool(review_required)
    ready_for_delivery = not blocking and not review_required
    gate = {
        "schema_version": QUALITY_GATE_SCHEMA_VERSION,
        "generated_at": _now(),
        "session_dir": str(session),
        "status": "fail" if blocking else ("needs_review" if requires_review else "pass"),
        "can_mark_complete": not blocking,
        "requires_review": requires_review,
        "ready_for_delivery": ready_for_delivery,
        "policy": active_policy,
        "metrics": gate_metrics,
        "summary": {
            "blocking_count": len(blocking),
            "review_required_count": len(review_required),
            "warning_count": len(warning),
            "metric_source": metric_source,
            "vector_count": vector_count,
            "unreviewed_count": unreviewed_count,
            "low_confidence_segment_count": low_confidence_segment_count,
            "missing_adapter_count": missing_adapter_count,
            "adapter_validation_error_count": adapter_error_count,
            "adapter_semantic_issue_count": adapter_semantic_issue_count,
            "reviewed_release": reviewed_manifest.get("release", {}).get("version") if isinstance(reviewed_manifest.get("release"), Mapping) else None,
            "promoted_release": reviewed_manifest.get("promotion", {}).get("active_version") if isinstance(reviewed_manifest.get("promotion"), Mapping) else None,
        },
        "checks": checks,
        "blocking_checks": blocking,
        "review_required_checks": review_required,
        "adapter_validation": adapters,
        "gate_health_errors": gate_health_errors,
    }
    target = Path(output_path) if output_path is not None else metadata / QUALITY_GATE_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    gate["gate_path"] = str(target)
    return gate


def _vector_count(session: Path) -> int:
    reviewed = reviewed_metadata_path(session, "vector_metadata.jsonl")
    if reviewed.name == REVIEWED_VECTOR_METADATA_FILENAME and reviewed.exists():
        return len(_read_jsonl(reviewed))
    metadata = session / "metadata"
    return len(_read_jsonl(metadata / "vector_metadata.jsonl")) + len(_read_jsonl(metadata / "micro_vector_metadata.jsonl"))


def _low_confidence_segment_count(session: Path) -> int:
    count = 0
    for row in _read_jsonl(reviewed_metadata_path(session, "key_action_segments.jsonl")):
        confidence = _segment_confidence(row)
        if confidence is None:
            count += 1
            continue
        if confidence < LOW_CONFIDENCE_THRESHOLD and not _has_approved_visual_support(row, confidence):
            count += 1
    return count


def _active_reviewed_manifest(session: Path) -> dict[str, Any]:
    active = active_reviewed_release(session)
    if isinstance(active, Mapping):
        release_dir = Path(str(active.get("release_dir") or ""))
        candidate = release_dir / "artifacts" / REVIEWED_MANIFEST_FILENAME
        data = _read_json(candidate)
        if data:
            if isinstance(active.get("promotion"), Mapping):
                data["promotion"] = dict(active["promotion"])
            return data
    return _read_json(session / "metadata" / REVIEWED_MANIFEST_FILENAME)


def _reviewed_health_score(metrics: Mapping[str, Any], *, error_count: int, policy: Mapping[str, Any]) -> float:
    score = 100.0 - error_count * 22
    coverage = _float(metrics.get("total_action_coverage_ratio"), 0.0)
    longest = _float(metrics.get("longest_segment_ratio"), 0.0)
    unreviewed = _float(metrics.get("unreviewed_count"), 0.0)
    max_coverage = _float(policy.get("max_total_action_coverage_ratio"), 0.65)
    max_longest = _float(policy.get("max_longest_segment_ratio"), 0.5)
    if coverage > max_coverage:
        score -= min(24.0, (coverage - max_coverage) * 60)
    if longest > max_longest:
        score -= min(22.0, (longest - max_longest) * 60)
    score -= min(18.0, unreviewed * 3.0)
    return max(0.0, min(100.0, round(score, 2)))


def _gate_health_errors(health: Mapping[str, Any], *, metric_source: str) -> list[dict[str, Any]]:
    errors = [dict(item) for item in health.get("errors") or [] if isinstance(item, Mapping)]
    if metric_source != "reviewed_dataset":
        return errors
    reviewed_replaced_codes = {
        "query_validation_failed",
    }
    return [item for item in errors if str(item.get("code") or "") not in reviewed_replaced_codes]


def _segment_confidence(row: Mapping[str, Any]) -> float | None:
    for key in ("boundary_confidence", "confidence", "score"):
        value = _float(row.get(key), None)
        if value is not None:
            return value
    cv = row.get("cv_detection") if isinstance(row.get("cv_detection"), Mapping) else {}
    values = [_float(cv.get("avg_active_score"), None), _float(cv.get("avg_motion_score"), None), _float(cv.get("confidence"), None)]
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _has_approved_visual_support(row: Mapping[str, Any], confidence: float) -> bool:
    review = row.get("review") if isinstance(row.get("review"), Mapping) else {}
    if str(review.get("decision") or row.get("review_status") or "").casefold() != "approved":
        return False
    has_visual_artifact = bool(row.get("keyframes") or row.get("asset_bindings") or row.get("visual_keywords") or row.get("yolo_interactions"))
    evidence_level = str(row.get("evidence_level") or "").casefold()
    if evidence_level in {"visual_confirmed", "multi_view_confirmed", "manual_confirmed"} and has_visual_artifact:
        return True
    if confidence < REVIEWED_VISUAL_CONFIDENCE_FLOOR:
        return False
    quality = row.get("quality") if isinstance(row.get("quality"), Mapping) else {}
    quality_confidence = str(quality.get("confidence") or "").casefold()
    return quality_confidence in {"medium", "high"} and has_visual_artifact


def _check_max(name: str, actual: float | None, maximum: float, message: str) -> dict[str, Any]:
    if actual is None:
        return {"name": name, "status": "warning", "actual": None, "maximum": maximum, "message": f"{message}; metric missing"}
    passed = actual <= maximum
    return {"name": name, "status": "pass" if passed else "fail", "actual": actual, "maximum": maximum, "message": message}


def _check_min(name: str, actual: float | None, minimum: float, message: str) -> dict[str, Any]:
    if actual is None:
        return {"name": name, "status": "warning", "actual": None, "minimum": minimum, "message": f"{message}; metric missing"}
    passed = actual >= minimum
    return {"name": name, "status": "pass" if passed else "fail", "actual": actual, "minimum": minimum, "message": message}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.exists() else []


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "DEFAULT_GATE_POLICY",
    "QUALITY_GATE_FILENAME",
    "QUALITY_GATE_SCHEMA_VERSION",
    "build_quality_gate",
]

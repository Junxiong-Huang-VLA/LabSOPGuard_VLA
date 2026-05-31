"""Benchmark / calibration accuracy harness (AGENTS.md §1.2, §20).

This module computes expected-vs-detected accuracy metrics for experiment
windows and key actions. It is a *measurement* layer that runs strictly AFTER
detection.

Hard contract (do not violate):
  * Expected files are read only to score results already produced by the
    pipeline. Nothing in this module feeds back into detection.
  * Expected files must never force output counts, suppress false positives,
    or create missing detections.
  * When no human-filled expected files exist, only templates are written and
    ``accuracy_validated`` is False.

The harness reads the real on-disk schemas:
  * formal_experiment_windows.json -> {"windows": [{experiment_window_id,
    start_global_timestamp_us, end_global_timestamp_us, ...}]}
  * material_stream.jsonl rows -> {action_type, object_refs:[...],
    start/peak/end_global_timestamp_us, experiment_window_id, ...}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_NS = "benchmark_accuracy"

VALIDATION_MODE_BENCHMARK = "benchmark"
VALIDATION_MODE_PRODUCTION_SELF_CHECK = "production_self_check"

# Default matching tolerances (configurable per call, never hardcoded counts).
DEFAULT_WINDOW_MIN_IOU = 0.30
DEFAULT_ACTION_TIME_TOLERANCE_US = 5_000_000  # 5s overlap window for action match


def _us_to_s(value: Any) -> float | None:
    try:
        return float(value) / 1_000_000.0
    except (TypeError, ValueError):
        return None


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


# --------------------------------------------------------------------------
# Templates (never filled with fake values)
# --------------------------------------------------------------------------

EXPECTED_WINDOWS_TEMPLATE: dict[str, Any] = {
    "schema_version": f"{SCHEMA_NS}.expected_windows.v1",
    "_comment": (
        "Human-filled ground truth for benchmark evaluation ONLY. "
        "Must not be used by the production pipeline or to force detection."
    ),
    "experiment_id": "",
    "expected_windows": [
        # {
        #   "expected_window_id": "exp_win_001",
        #   "label": "weighing",
        #   "start_global_timestamp_us": 0,
        #   "end_global_timestamp_us": 0
        # }
    ],
}

EXPECTED_ACTIONS_TEMPLATE: dict[str, Any] = {
    "schema_version": f"{SCHEMA_NS}.expected_actions.v1",
    "_comment": (
        "Human-filled ground truth for benchmark evaluation ONLY. "
        "Counts here are for post-run scoring; never hardcode into detection."
    ),
    "experiment_id": "",
    "expected_actions": [
        # {
        #   "expected_action_id": "exp_act_001",
        #   "expected_window_id": "exp_win_001",
        #   "action_type": "hand_object_contact",
        #   "object_type": "reagent_bottle",
        #   "expected_count": 0,
        #   "rough_start_global_timestamp_us": 0,
        #   "rough_end_global_timestamp_us": 0
        # }
    ],
}


def write_template(path: Path, template: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def ensure_expected_templates(metadata_dir: Path) -> dict[str, str]:
    """Write template files if absent. Never overwrites human-filled files."""
    metadata_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    win_tmpl = metadata_dir / "expected_windows.template.json"
    act_tmpl = metadata_dir / "expected_actions.template.json"
    if not win_tmpl.exists():
        write_template(win_tmpl, EXPECTED_WINDOWS_TEMPLATE)
    if not act_tmpl.exists():
        write_template(act_tmpl, EXPECTED_ACTIONS_TEMPLATE)
    out["expected_windows_template"] = str(win_tmpl)
    out["expected_actions_template"] = str(act_tmpl)
    return out


@dataclass
class ValidationContext:
    """Resolves whether human expected files exist and sets the mode contract."""

    metadata_dir: Path
    expected_windows_path: Path | None = None
    expected_actions_path: Path | None = None
    has_expected_windows: bool = False
    has_expected_actions: bool = False

    @property
    def validation_mode(self) -> str:
        if self.has_expected_windows or self.has_expected_actions:
            return VALIDATION_MODE_BENCHMARK
        return VALIDATION_MODE_PRODUCTION_SELF_CHECK

    @property
    def accuracy_validated(self) -> bool:
        # Accuracy is only "validated" when real human ground truth is present.
        return self.has_expected_windows or self.has_expected_actions

    def stamp(self) -> dict[str, Any]:
        return {
            "validation_mode": self.validation_mode,
            "accuracy_validated": self.accuracy_validated,
            "expected_windows_path": str(self.expected_windows_path)
            if self.has_expected_windows
            else None,
            "expected_actions_path": str(self.expected_actions_path)
            if self.has_expected_actions
            else None,
        }


def resolve_validation_context(metadata_dir: Path) -> ValidationContext:
    """Detect human-filled expected files; otherwise ensure templates exist."""
    metadata_dir = Path(metadata_dir)
    win = metadata_dir / "expected_windows.json"
    act = metadata_dir / "expected_actions.json"
    ctx = ValidationContext(
        metadata_dir=metadata_dir,
        expected_windows_path=win if win.exists() else None,
        expected_actions_path=act if act.exists() else None,
        has_expected_windows=win.exists(),
        has_expected_actions=act.exists(),
    )
    if not ctx.has_expected_windows or not ctx.has_expected_actions:
        ensure_expected_templates(metadata_dir)
    return ctx


# --------------------------------------------------------------------------
# Detected-artifact loaders (read real on-disk schemas)
# --------------------------------------------------------------------------


def load_detected_windows(formal_windows_path: Path) -> list[dict[str, Any]]:
    """Load formal_experiment_windows.json -> normalized window dicts."""
    data = _load_json(Path(formal_windows_path))
    windows = data.get("windows", []) if isinstance(data, dict) else list(data)
    out: list[dict[str, Any]] = []
    for w in windows:
        out.append(
            {
                "window_id": w.get("experiment_window_id") or w.get("window_id"),
                "start_us": w.get("start_global_timestamp_us"),
                "end_us": w.get("end_global_timestamp_us"),
                "status": w.get("status"),
            }
        )
    return out


def load_detected_actions(material_stream_path: Path) -> list[dict[str, Any]]:
    """Load material_stream.jsonl -> normalized detected-action dicts.

    Orphan materials (no window) are still counted as detections; they simply
    cannot match a windowed expectation. This keeps false positives honest.
    """
    rows = _load_jsonl(Path(material_stream_path))
    out: list[dict[str, Any]] = []
    for r in rows:
        objs = r.get("object_refs") or []
        if isinstance(objs, str):
            objs = [objs]
        out.append(
            {
                "material_id": r.get("material_id"),
                "action_type": r.get("action_type"),
                "object_refs": [str(o).lower() for o in objs],
                "start_us": r.get("start_global_timestamp_us"),
                "end_us": r.get("end_global_timestamp_us"),
                "window_id": r.get("experiment_window_id") or r.get("window_id"),
                "official_status": r.get("official_status"),
            }
        )
    return out


# --------------------------------------------------------------------------
# Matching helpers
# --------------------------------------------------------------------------


def _interval_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    if None in (a_start, a_end, b_start, b_end):
        return 0.0
    lo = max(a_start, b_start)
    hi = min(a_end, b_end)
    inter = max(0.0, hi - lo)
    union = (a_end - a_start) + (b_end - b_start) - inter
    if union <= 0:
        return 0.0
    return inter / union


def _overlaps(a_start: float, a_end: float, b_start: float, b_end: float,
              tol_us: float) -> bool:
    if None in (a_start, a_end):
        return False
    if b_start is None and b_end is None:
        return True  # expectation without a time hint matches on type/object only
    bs = b_start if b_start is not None else a_start
    be = b_end if b_end is not None else a_end
    return (a_start - tol_us) <= be and (a_end + tol_us) >= bs


def _prf(matched: int, detected: int, expected: int) -> dict[str, float]:
    precision = matched / detected if detected else 0.0
    recall = matched / expected if expected else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


# --------------------------------------------------------------------------
# Window accuracy
# --------------------------------------------------------------------------


def score_windows(
    detected: Sequence[Mapping[str, Any]],
    expected: Sequence[Mapping[str, Any]],
    *,
    min_iou: float = DEFAULT_WINDOW_MIN_IOU,
) -> dict[str, Any]:
    """Greedy 1:1 window matching by interval IoU. Pure measurement."""
    used_detected: set[int] = set()
    matches: list[dict[str, Any]] = []
    start_errors: list[float] = []
    end_errors: list[float] = []

    for exp in expected:
        es = _us_to_s(exp.get("start_global_timestamp_us"))
        ee = _us_to_s(exp.get("end_global_timestamp_us"))
        best_idx = -1
        best_iou = min_iou
        for idx, det in enumerate(detected):
            if idx in used_detected:
                continue
            ds = _us_to_s(det.get("start_us"))
            de = _us_to_s(det.get("end_us"))
            if None in (es, ee, ds, de):
                continue
            iou = _interval_iou(ds, de, es, ee)
            if iou >= best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx >= 0:
            used_detected.add(best_idx)
            det = detected[best_idx]
            ds = _us_to_s(det.get("start_us"))
            de = _us_to_s(det.get("end_us"))
            if None not in (es, ds):
                start_errors.append(abs(ds - es))
            if None not in (ee, de):
                end_errors.append(abs(de - ee))
            matches.append(
                {
                    "expected_window_id": exp.get("expected_window_id"),
                    "detected_window_id": det.get("window_id"),
                    "iou": round(best_iou, 4),
                }
            )

    matched = len(matches)
    missed = [
        exp.get("expected_window_id")
        for i, exp in enumerate(expected)
        if exp.get("expected_window_id")
        not in {m["expected_window_id"] for m in matches}
    ]
    false_pos = [
        detected[i].get("window_id")
        for i in range(len(detected))
        if i not in used_detected
    ]

    def _avg(xs: list[float]) -> float | None:
        return round(sum(xs) / len(xs), 3) if xs else None

    report = {
        "expected_window_count": len(expected),
        "detected_window_count": len(detected),
        "matched_window_count": matched,
        "missed_window_count": len(missed),
        "false_positive_window_count": len(false_pos),
        "start_boundary_error_s_avg": _avg(start_errors),
        "end_boundary_error_s_avg": _avg(end_errors),
        "matches": matches,
        "missed_window_ids": missed,
        "false_positive_window_ids": false_pos,
    }
    report.update(_prf(matched, len(detected), len(expected)))
    return report


# --------------------------------------------------------------------------
# Action accuracy
# --------------------------------------------------------------------------


def score_actions(
    detected: Sequence[Mapping[str, Any]],
    expected: Sequence[Mapping[str, Any]],
    *,
    time_tolerance_us: float = DEFAULT_ACTION_TIME_TOLERANCE_US,
) -> dict[str, Any]:
    """Match detected actions to expected (action_type, object_type) groups.

    Expected entries carry an ``expected_count``; detections are matched
    greedily within the rough time range. Surplus detections become false
    positives, shortfalls become missed events. No count is ever forced.
    """
    used: set[int] = set()
    per_group: list[dict[str, Any]] = []
    missed_events: list[dict[str, Any]] = []
    classification_errors: list[dict[str, Any]] = []

    total_matched = total_detected_in_groups = total_expected = 0

    for exp in expected:
        a_type = str(exp.get("action_type") or "").strip()
        o_type = str(exp.get("object_type") or "").strip().lower()
        want = int(exp.get("expected_count") or 0)
        total_expected += want
        bs = exp.get("rough_start_global_timestamp_us")
        be = exp.get("rough_end_global_timestamp_us")

        candidates = []
        for idx, det in enumerate(detected):
            if idx in used:
                continue
            if det.get("action_type") != a_type:
                continue
            if o_type and o_type not in (det.get("object_refs") or []):
                continue
            ds = _us_to_s(det.get("start_us"))
            de = _us_to_s(det.get("end_us")) or ds
            bss = _us_to_s(bs)
            bee = _us_to_s(be)
            tol_s = time_tolerance_us / 1_000_000.0
            if _overlaps(ds, de, bss, bee, tol_s):
                candidates.append(idx)

        matched_here = candidates[:want] if want else candidates
        for idx in matched_here:
            used.add(idx)
        total_matched += len(matched_here)

        if len(matched_here) < want:
            missed_events.append(
                {
                    "expected_action_id": exp.get("expected_action_id"),
                    "action_type": a_type,
                    "object_type": o_type,
                    "expected_count": want,
                    "matched_count": len(matched_here),
                    "missed_count": want - len(matched_here),
                }
            )

        grp = {
            "action_type": a_type,
            "object_type": o_type,
            "expected_count": want,
            "matched_count": len(matched_here),
        }
        grp.update(_prf(len(matched_here), max(len(matched_here), want), want))
        per_group.append(grp)

    # Unmatched detections = false positives. A detection whose action_type was
    # never expected (but object matched some expectation) is a classification
    # error candidate; we report both views without altering detection.
    false_positive_events = []
    for idx, det in enumerate(detected):
        if idx in used:
            continue
        fp = {
            "material_id": det.get("material_id"),
            "action_type": det.get("action_type"),
            "object_refs": det.get("object_refs"),
            "start_global_timestamp_us": det.get("start_us"),
            "window_id": det.get("window_id"),
        }
        false_positive_events.append(fp)
        for exp in expected:
            if (
                str(exp.get("object_type") or "").lower()
                in (det.get("object_refs") or [])
                and exp.get("action_type") != det.get("action_type")
            ):
                classification_errors.append(
                    {
                        "material_id": det.get("material_id"),
                        "detected_action_type": det.get("action_type"),
                        "expected_action_type": exp.get("action_type"),
                        "object_type": exp.get("object_type"),
                    }
                )
                break

    total_detected_in_groups = len(detected)
    summary = {
        "expected_action_total": total_expected,
        "detected_action_total": len(detected),
        "matched_action_total": total_matched,
        "missed_action_total": sum(m["missed_count"] for m in missed_events),
        "false_positive_action_total": len(false_positive_events),
        "classification_error_total": len(classification_errors),
        "per_group": per_group,
    }
    summary.update(_prf(total_matched, total_detected_in_groups, total_expected))
    return {
        "summary": summary,
        "missed_events": missed_events,
        "false_positive_events": false_positive_events,
        "classification_errors": classification_errors,
    }


# --------------------------------------------------------------------------
# Orchestrator — write all reports with the validation_mode contract
# --------------------------------------------------------------------------


def _write_report(path: Path, payload: Mapping[str, Any]) -> Path:
    from .report_io import write_json_report

    return write_json_report(path, payload)


def run_benchmark_evaluation(
    *,
    formal_windows_path: Path,
    material_stream_path: Path,
    metadata_dir: Path,
    reports_dir: Path,
    window_min_iou: float = DEFAULT_WINDOW_MIN_IOU,
    action_time_tolerance_us: float = DEFAULT_ACTION_TIME_TOLERANCE_US,
) -> dict[str, Any]:
    """Score detection output against (optional) human ground truth.

    Always writes the five reports. Detection artifacts are read-only here; this
    function never mutates windows or materials. When no expected files exist it
    writes templates and marks accuracy_validated=False.
    """
    metadata_dir = Path(metadata_dir)
    reports_dir = Path(reports_dir)
    ctx = resolve_validation_context(metadata_dir)
    stamp = ctx.stamp()

    detected_windows = (
        load_detected_windows(Path(formal_windows_path))
        if Path(formal_windows_path).exists()
        else []
    )
    detected_actions = (
        load_detected_actions(Path(material_stream_path))
        if Path(material_stream_path).exists()
        else []
    )

    expected_windows: list[dict[str, Any]] = []
    expected_actions: list[dict[str, Any]] = []
    if ctx.has_expected_windows:
        expected_windows = list(
            _load_json(ctx.expected_windows_path).get("expected_windows", [])
        )
    if ctx.has_expected_actions:
        expected_actions = list(
            _load_json(ctx.expected_actions_path).get("expected_actions", [])
        )

    written: dict[str, str] = {}

    # Window accuracy
    if ctx.has_expected_windows:
        win = score_windows(detected_windows, expected_windows, min_iou=window_min_iou)
    else:
        win = {
            "status": "not_validated_without_expected_windows",
            "detected_window_count": len(detected_windows),
        }
    win.update(stamp)
    written["window_accuracy_report"] = str(
        _write_report(reports_dir / "window_accuracy_report.json", win)
    )

    # Action accuracy + derived reports
    if ctx.has_expected_actions:
        act = score_actions(
            detected_actions, expected_actions,
            time_tolerance_us=action_time_tolerance_us,
        )
    else:
        act = {
            "summary": {
                "status": "not_validated_without_expected_actions",
                "detected_action_total": len(detected_actions),
            },
            "missed_events": [],
            "false_positive_events": [],
            "classification_errors": [],
        }

    action_report = dict(act["summary"])
    action_report.update(stamp)
    written["action_accuracy_report"] = str(
        _write_report(reports_dir / "action_accuracy_report.json", action_report)
    )
    written["missed_events_report"] = str(
        _write_report(
            reports_dir / "missed_events_report.json",
            {**stamp, "missed_events": act["missed_events"]},
        )
    )
    written["false_positive_events_report"] = str(
        _write_report(
            reports_dir / "false_positive_events_report.json",
            {**stamp, "false_positive_events": act["false_positive_events"]},
        )
    )
    written["classification_error_report"] = str(
        _write_report(
            reports_dir / "classification_error_report.json",
            {**stamp, "classification_errors": act["classification_errors"]},
        )
    )

    return {
        **stamp,
        "reports": written,
        "templates": ensure_expected_templates(metadata_dir),
    }

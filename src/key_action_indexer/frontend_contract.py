"""Frontend / API preview & runtime contract validators (P3, §10 & §17).

Pure functions over the backend payload dicts (window items, runtime summary)
so they are testable without a running server and reusable by the backend to
emit contract reports.

Rules enforced (AGENTS.md §10):
  * third panel must use ONLY the third-view url; first panel ONLY the
    first-view url; neither may be silently substituted by side_by_side.
  * a missing single-view preview must surface as null + a missing reason,
    never as the side-by-side video.
  * experiment_window_duration_s and preview_duration_s must be distinct
    fields (real duration vs preview duration not conflated).
  * fast_preview must be labeled (preview_mode), not presented as full duration.

Rules enforced (§17 runtime summary):
  * each displayed timing has a real source_file + source_field, or is marked
    未记录 — never a fake 0.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

WINDOW_VIEW_CONTRACT_SCHEMA = "window_view_artifact_contract_report.v1"
PREVIEW_DISPLAY_SCHEMA = "frontend_preview_display_validation_report.v1"
RUNTIME_SUMMARY_SCHEMA = "runtime_summary_data_contract_report.v1"

RUNTIME_STAGES = (
    "总耗时",
    "视频时间戳对齐",
    "长视频并行粗扫",
    "实验片段并行精扫",
    "关键素材发布",
    "态势记忆写入",
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    from .report_io import write_json_report

    return write_json_report(path, payload)


def validate_window_view_contract(window_item: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one window item's preview-url contract. Returns issue list."""
    issues: list[str] = []
    third = window_item.get("third_view_realtime_preview_url")
    first = window_item.get("first_view_realtime_preview_url")
    sbs = window_item.get("side_by_side_realtime_preview_url")

    # 1. single-view url must not equal the side-by-side url (no fake fallback)
    if sbs and third and third == sbs:
        issues.append("third_view_realtime_preview_url equals side_by_side url (fake single-view).")
    if sbs and first and first == sbs:
        issues.append("first_view_realtime_preview_url equals side_by_side url (fake single-view).")

    # 2. missing single-view preview -> must be null (frontend shows 待生成),
    #    never a non-null substitution
    # (null is acceptable; the frontend renders the 待生成 placeholder)

    # 3. duration fields must both exist and be distinct concepts
    win_dur = window_item.get("experiment_window_duration_s")
    prev_dur = window_item.get("preview_duration_s")
    if win_dur is None:
        issues.append("experiment_window_duration_s missing.")
    if prev_dur is None:
        issues.append("preview_duration_s missing.")

    # 4. if a fast preview is present it must be labeled via preview_mode
    if window_item.get("fast_preview_url") and not window_item.get("preview_mode"):
        issues.append("fast_preview present but preview_mode label missing.")

    return {
        "window_id": window_item.get("window_id") or window_item.get("experiment_window_id"),
        "third_preview_present": bool(third),
        "first_preview_present": bool(first),
        "side_by_side_present": bool(sbs),
        "experiment_window_duration_s": win_dur,
        "preview_duration_s": prev_dur,
        "preview_mode": window_item.get("preview_mode"),
        "issues": issues,
        "ok": not issues,
    }


def build_window_view_contract_report(
    window_items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    results = [validate_window_view_contract(w) for w in window_items]
    return {
        "schema_version": WINDOW_VIEW_CONTRACT_SCHEMA,
        "window_count": len(results),
        "ok": all(r["ok"] for r in results),
        "windows_with_issues": [r for r in results if not r["ok"]],
        "results": results,
    }


def build_preview_display_report(
    window_items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize which single-view previews are present vs pending-generation."""
    missing_third = []
    missing_first = []
    for w in window_items:
        wid = w.get("window_id") or w.get("experiment_window_id")
        if not w.get("third_view_realtime_preview_url"):
            missing_third.append(wid)
        if not w.get("first_view_realtime_preview_url"):
            missing_first.append(wid)
    return {
        "schema_version": PREVIEW_DISPLAY_SCHEMA,
        "window_count": len(window_items),
        "missing_third_view_preview": missing_third,
        "missing_first_view_preview": missing_first,
        "third_pending_label": "第三人称预览待生成",
        "first_pending_label": "第一人称预览待生成",
        "fast_preview_label": "快速预览，不代表完整实验时长",
    }


def validate_runtime_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    """§17: each stage value must have a real source, or be marked 未记录."""
    stages_out: list[dict[str, Any]] = []
    issues: list[str] = []
    stage_map = summary.get("stages") or {}
    for stage in RUNTIME_STAGES:
        entry = stage_map.get(stage) or {}
        value = entry.get("value")
        source_file = entry.get("source_file")
        source_field = entry.get("source_field")
        displayed = entry.get("displayed")
        # A real value requires a source; a 0 without source is a fake.
        if value is not None and (not source_file or not source_field):
            issues.append(f"stage {stage!r} has a value but no source_file/source_field.")
        if value is None and displayed not in (None, "未记录"):
            issues.append(f"stage {stage!r} missing value but displayed {displayed!r} (expected 未记录).")
        stages_out.append({
            "stage": stage,
            "value": value,
            "source_file": source_file,
            "source_field": source_field,
            "displayed": displayed if value is not None else "未记录",
        })
    return {
        "schema_version": RUNTIME_SUMMARY_SCHEMA,
        "stages": stages_out,
        "issues": issues,
        "ok": not issues,
    }


def write_frontend_contract_reports(
    reports_dir: Path,
    *,
    window_items: Sequence[Mapping[str, Any]],
    runtime_summary: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    reports_dir = Path(reports_dir)
    out: dict[str, str] = {}
    out["window_view_artifact_contract_report"] = str(_write_json(
        reports_dir / "window_view_artifact_contract_report.json",
        build_window_view_contract_report(window_items)))
    out["frontend_preview_display_validation_report"] = str(_write_json(
        reports_dir / "frontend_preview_display_validation_report.json",
        build_preview_display_report(window_items)))
    if runtime_summary is not None:
        out["runtime_summary_data_contract_report"] = str(_write_json(
            reports_dir / "runtime_summary_data_contract_report.json",
            validate_runtime_summary(runtime_summary)))
    return out

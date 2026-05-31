from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import SyncConfig
from .frames import CameraFrames


@dataclass(frozen=True)
class SyncReport:
    payload: dict[str, Any]

    @classmethod
    def from_result(
        cls,
        *,
        config: SyncConfig,
        camera_frames: list[CameraFrames],
        long_pairs: pd.DataFrame,
        wide_pairs: pd.DataFrame,
        warnings: list[str],
    ) -> "SyncReport":
        by_camera = {item.camera_id: item for item in camera_frames}
        reference_count = by_camera[config.reference_camera].frame_count
        matched_count: dict[str, int] = {}
        unmatched_count: dict[str, int] = {}
        match_rate: dict[str, float] = {}
        median_abs: dict[str, float | None] = {}
        p95_abs: dict[str, float | None] = {}
        max_abs: dict[str, float | None] = {}

        for camera_id, camera in by_camera.items():
            if camera_id == config.reference_camera:
                matched_count[camera_id] = reference_count
                unmatched_count[camera_id] = 0
                match_rate[camera_id] = 1.0
                median_abs[camera_id] = 0.0
                p95_abs[camera_id] = 0.0
                max_abs[camera_id] = 0.0
                continue
            group = long_pairs[long_pairs["camera_id"] == camera_id]
            matched = group[group["matched_ok"]]
            matched_count[camera_id] = int(len(matched))
            unmatched_count[camera_id] = int(reference_count - len(matched))
            match_rate[camera_id] = round(float(len(matched) / reference_count), 6) if reference_count else 0.0
            abs_diff = matched["abs_time_diff_us"].dropna()
            median_abs[camera_id] = _metric(abs_diff, "median")
            p95_abs[camera_id] = _metric(abs_diff, "p95")
            max_abs[camera_id] = _metric(abs_diff, "max")

        output_files = _output_file_metadata(config, long_pairs, wide_pairs)
        quality_gate = _build_quality_gate(
            config=config,
            camera_frames=camera_frames,
            reference_count=reference_count,
            unmatched_count=unmatched_count,
            max_abs=max_abs,
            warnings=warnings,
            output_files=output_files,
        )
        return cls(
            {
                "schema_version": "realityloop_frame_sync_report.v1",
                "run_id": config.run_id,
                "experiment_name": config.experiment_name,
                "batch_id": config.batch_id,
                "reference_camera": config.reference_camera,
                "tolerance_us": config.tolerance_us,
                "report_status": quality_gate["status"],
                "quality_gate_status": quality_gate["status"],
                "timestamp_col_used": {camera.camera_id: camera.timestamp_col for camera in camera_frames},
                "frame_count": {camera.camera_id: camera.frame_count for camera in camera_frames},
                "matched_count": matched_count,
                "unmatched_count": unmatched_count,
                "match_rate": match_rate,
                "median_abs_time_diff_us": median_abs,
                "p95_abs_time_diff_us": p95_abs,
                "max_abs_time_diff_us": max_abs,
                "detected_median_frame_interval_us": {
                    camera.camera_id: camera.median_frame_interval_us for camera in camera_frames
                },
                "quality_gate": quality_gate,
                "output_files": output_files,
                "warnings": warnings,
            }
        )


def write_outputs(config: SyncConfig, long_pairs: pd.DataFrame, wide_pairs: pd.DataFrame, report: SyncReport) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    long_pairs.to_csv(config.output_dir / "sync_pairs_long.csv", index=False, encoding="utf-8-sig")
    wide_pairs.to_csv(config.output_dir / "sync_pairs_wide.csv", index=False, encoding="utf-8-sig")
    (config.output_dir / "sync_report.json").write_text(
        json.dumps(report.payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (config.output_dir / "run_summary.md").write_text(_summary_markdown(report.payload), encoding="utf-8")


def _metric(series: pd.Series, kind: str) -> float | None:
    if series.empty:
        return None
    if kind == "median":
        return float(series.median())
    if kind == "p95":
        return float(series.quantile(0.95))
    if kind == "max":
        return float(series.max())
    raise ValueError(kind)


def _build_quality_gate(
    *,
    config: SyncConfig,
    camera_frames: list[CameraFrames],
    reference_count: int,
    unmatched_count: dict[str, int],
    max_abs: dict[str, float | None],
    warnings: list[str],
    output_files: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    issue_list: list[dict[str, Any]] = []
    gate_config = config.quality_gate
    max_abs_threshold = gate_config.max_abs_time_diff_us_warning
    max_abs_threshold_source = "quality_gate.max_abs_time_diff_us_warning"
    if max_abs_threshold is None and config.tolerance_us is not None:
        max_abs_threshold = config.tolerance_us
        max_abs_threshold_source = "tolerance_us"

    unmatched_threshold = gate_config.unmatched_count_warning
    unmatched_threshold_source = "quality_gate.unmatched_count_warning"
    if unmatched_threshold is None and config.tolerance_us is not None:
        unmatched_threshold = 0
        unmatched_threshold_source = "tolerance_us"

    unmatched_rate_threshold = gate_config.unmatched_rate_warning
    camera_ids = [camera.camera_id for camera in camera_frames]
    for camera_id in sorted(camera_ids):
        if camera_id == config.reference_camera:
            continue
        camera_max_abs = max_abs.get(camera_id)
        if max_abs_threshold is not None and camera_max_abs is not None and camera_max_abs > max_abs_threshold:
            issue_list.append(
                _quality_issue(
                    code="max_abs_time_diff_us_over_warning_threshold",
                    camera_id=camera_id,
                    metric="max_abs_time_diff_us",
                    value=camera_max_abs,
                    threshold=max_abs_threshold,
                    threshold_source=max_abs_threshold_source,
                    message=(
                        f"{camera_id}: max_abs_time_diff_us {camera_max_abs:.1f} exceeds "
                        f"warning threshold {max_abs_threshold}"
                    ),
                )
            )
        camera_unmatched = unmatched_count.get(camera_id, 0)
        if unmatched_threshold is not None and camera_unmatched > unmatched_threshold:
            issue_list.append(
                _quality_issue(
                    code="unmatched_frames_over_warning_threshold",
                    camera_id=camera_id,
                    metric="unmatched_count",
                    value=camera_unmatched,
                    threshold=unmatched_threshold,
                    threshold_source=unmatched_threshold_source,
                    message=(
                        f"{camera_id}: unmatched_count {camera_unmatched} exceeds "
                        f"warning threshold {unmatched_threshold}"
                    ),
                )
            )
        unmatched_rate = float(camera_unmatched / reference_count) if reference_count else 0.0
        if unmatched_rate_threshold is not None and unmatched_rate > unmatched_rate_threshold:
            issue_list.append(
                _quality_issue(
                    code="unmatched_rate_over_warning_threshold",
                    camera_id=camera_id,
                    metric="unmatched_rate",
                    value=round(unmatched_rate, 6),
                    threshold=unmatched_rate_threshold,
                    threshold_source="quality_gate.unmatched_rate_warning",
                    message=(
                        f"{camera_id}: unmatched_rate {unmatched_rate:.6f} exceeds "
                        f"warning threshold {unmatched_rate_threshold}"
                    ),
                )
            )

    for warning in warnings:
        issue_list.append(
            {
                "severity": "warning",
                "code": "input_metadata_warning",
                "message": warning,
            }
        )

    severity_counts = _severity_counts(issue_list)
    status = "warning" if severity_counts.get("warning", 0) else "pass"
    return {
        "schema_version": "realityloop_sync_quality_gate.v1",
        "status": status,
        "passed": status == "pass",
        "tolerance_us": config.tolerance_us,
        "thresholds": {
            "max_abs_time_diff_us_warning": max_abs_threshold,
            "max_abs_time_diff_us_warning_source": max_abs_threshold_source if max_abs_threshold is not None else None,
            "unmatched_count_warning": unmatched_threshold,
            "unmatched_count_warning_source": unmatched_threshold_source if unmatched_threshold is not None else None,
            "unmatched_rate_warning": unmatched_rate_threshold,
            "unmatched_rate_warning_source": (
                "quality_gate.unmatched_rate_warning" if unmatched_rate_threshold is not None else None
            ),
        },
        "issue_count": len(issue_list),
        "severity_counts": severity_counts,
        "issues": issue_list,
        "manifest_fields": {
            "record_type": "realityloop_frame_sync",
            "run_id": config.run_id,
            "reference_camera": config.reference_camera,
            "status": status,
            "output_dir": str(config.output_dir),
            "output_files": output_files,
        },
    }


def _quality_issue(
    *,
    code: str,
    camera_id: str,
    metric: str,
    value: Any,
    threshold: Any,
    threshold_source: str,
    message: str,
) -> dict[str, Any]:
    return {
        "severity": "warning",
        "code": code,
        "camera_id": camera_id,
        "metric": metric,
        "value": value,
        "threshold": threshold,
        "threshold_source": threshold_source,
        "message": message,
    }


def _severity_counts(issues: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {"warning": 0}
    for issue in issues:
        severity = str(issue.get("severity") or "warning")
        counts[severity] = counts.get(severity, 0) + 1
    return counts


def _output_file_metadata(
    config: SyncConfig,
    long_pairs: pd.DataFrame,
    wide_pairs: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    return {
        "sync_pairs_long": {
            "path": str(config.output_dir / "sync_pairs_long.csv"),
            "relative_path": "sync_pairs_long.csv",
            "format": "csv",
            "row_count": int(len(long_pairs)),
        },
        "sync_pairs_wide": {
            "path": str(config.output_dir / "sync_pairs_wide.csv"),
            "relative_path": "sync_pairs_wide.csv",
            "format": "csv",
            "row_count": int(len(wide_pairs)),
        },
        "sync_report": {
            "path": str(config.output_dir / "sync_report.json"),
            "relative_path": "sync_report.json",
            "format": "json",
        },
        "run_summary": {
            "path": str(config.output_dir / "run_summary.md"),
            "relative_path": "run_summary.md",
            "format": "markdown",
        },
    }


def _summary_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# RealityLoop Frame Sync Report",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Experiment: `{payload.get('experiment_name') or ''}`",
        f"- Batch: `{payload.get('batch_id') or ''}`",
        f"- Reference camera: `{payload.get('reference_camera')}`",
        f"- Tolerance us: `{payload.get('tolerance_us')}`",
        f"- Quality gate: `{payload.get('quality_gate_status')}`",
        "",
        "## Cameras",
        "",
        "| camera | frames | timestamp column | matched | unmatched | match rate | median abs diff us | p95 abs diff us | max abs diff us |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    frame_count = payload.get("frame_count") or {}
    timestamp_cols = payload.get("timestamp_col_used") or {}
    matched = payload.get("matched_count") or {}
    unmatched = payload.get("unmatched_count") or {}
    rates = payload.get("match_rate") or {}
    median = payload.get("median_abs_time_diff_us") or {}
    p95 = payload.get("p95_abs_time_diff_us") or {}
    max_abs = payload.get("max_abs_time_diff_us") or {}
    for camera_id in sorted(frame_count):
        lines.append(
            "| {camera} | {frames} | `{col}` | {matched} | {unmatched} | {rate:.3f} | {median} | {p95} | {max_abs} |".format(
                camera=camera_id,
                frames=frame_count.get(camera_id),
                col=timestamp_cols.get(camera_id),
                matched=matched.get(camera_id),
                unmatched=unmatched.get(camera_id),
                rate=float(rates.get(camera_id) or 0.0),
                median=_format_optional(median.get(camera_id)),
                p95=_format_optional(p95.get(camera_id)),
                max_abs=_format_optional(max_abs.get(camera_id)),
            )
        )
    quality_gate = payload.get("quality_gate") or {}
    issues = quality_gate.get("issues") or []
    lines.extend(["", "## Quality Gate", ""])
    lines.append(f"- Status: `{quality_gate.get('status') or payload.get('quality_gate_status')}`")
    lines.append(f"- Issues: {len(issues)}")
    if issues:
        for issue in issues:
            lines.append(f"- {issue.get('code')}: {issue.get('message')}")
    warnings = payload.get("warnings") or []
    lines.extend(["", "## Warnings", ""])
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def _format_optional(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return str(value)
